#!/usr/bin/env python3
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from training import (
    barrier,
    build_scannet_loader,
    cleanup_distributed,
    load_config,
    precision_context,
    resolve_path,
    seed_everything,
    setup_distributed,
)
from utils import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate DINOv2 3DRAE latent normalization statistics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/dino2_stat_scannet.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/stage1/stats.safetensors")
    )
    parser.add_argument("--neck-weights", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-tokens-per-gpu", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def resolve_model_config(config, neck_weights):
    config = deepcopy(config)
    parameters = config["params"]
    parameters["encoder_params"]["dinov2_path"] = str(
        resolve_path(parameters["encoder_params"]["dinov2_path"], RELEASE_ROOT)
    )
    weights = neck_weights or parameters["pretrained_neck_path"]
    parameters["pretrained_neck_path"] = str(resolve_path(weights, RELEASE_ROOT))
    return config


def main():
    args = parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max_batches must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, device = setup_distributed()
    config_path, config = load_config(args.config)
    output = resolve_path(args.output, RELEASE_ROOT).resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {temporary}")
    stat_config = config["statistics"]
    seed = int(args.seed if args.seed is not None else stat_config.get("seed", 42))
    seed_everything(seed * world_size + rank)

    dataset_config = dict(config["dataset"])
    dataset_config["path"] = str(
        resolve_path(args.data_path or dataset_config["path"], RELEASE_ROOT).resolve()
    )
    if args.max_tokens_per_gpu is not None:
        dataset_config["max_token_per_gpu"] = args.max_tokens_per_gpu
    if args.max_batch_size is not None:
        dataset_config["max_batch_size"] = args.max_batch_size
    workers = int(
        args.num_workers
        if args.num_workers is not None
        else stat_config["num_workers"]
    )
    if workers < 0:
        raise ValueError("num_workers must be non-negative")

    model = instantiate_from_config(
        resolve_model_config(config["model"], args.neck_weights)
    ).to(device)
    if model.decoder_img is not None or model.decoder_pmap is not None:
        raise ValueError("Statistic calculation expects an encode-only 3DRAE")
    if model.do_normalization:
        raise ValueError("Statistic calculation requires raw, unnormalized latents")
    model.eval().requires_grad_(False)

    loader, batch_sampler = build_scannet_loader(
        dataset_config, rank, world_size, seed, workers
    )
    batch_sampler.set_epoch(0)
    latent_sum = latent_square_sum = None
    sample_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = batch["images"].to(device, non_blocking=True)
            visible_ids = batch["visible_ids"].to(device, non_blocking=True)
            rays = batch["rays"].to(device, non_blocking=True)
            with precision_context(args.precision):
                latent = model.encode(images, visible_ids, ray_maps=rays)
            expected_shape = (model.neck.latent_token_num, model.neck.hidden_size)
            if latent.shape[1:] != expected_shape:
                raise RuntimeError(
                    f"Unexpected latent shape {tuple(latent.shape)}, "
                    f"expected [B,{expected_shape[0]},{expected_shape[1]}]"
                )
            latent = latent.double()
            if latent_sum is None:
                latent_sum = torch.zeros_like(latent[0])
                latent_square_sum = torch.zeros_like(latent[0])
            latent_sum += latent.sum(dim=0)
            latent_square_sum += latent.square().sum(dim=0)
            sample_count += latent.shape[0]
            if rank == 0:
                print(
                    f"batch={batch_index + 1} local_samples={sample_count} "
                    f"latent_shape={tuple(latent.shape)}",
                    flush=True,
                )

    if latent_sum is None or sample_count == 0:
        raise RuntimeError("No latent samples were collected")
    count = torch.tensor(float(sample_count), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(latent_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(latent_square_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    mean = latent_sum / count
    variance = torch.clamp(latent_square_sum / count - mean.square(), min=0)
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise RuntimeError("Calculated statistics contain non-finite values")

    barrier()
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            "mean": mean.float().cpu().contiguous(),
            "var": variance.float().cpu().contiguous(),
        }
        metadata = {
            "config": config_path.name,
            "samples": str(int(count.item())),
            "precision": args.precision,
        }
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, output)
        print(f"Saved {output}")
        print(
            f"samples={int(count.item())} shape={tuple(tensors['mean'].shape)} "
            f"mean_finite={bool(torch.isfinite(tensors['mean']).all())} "
            f"var_finite={bool(torch.isfinite(tensors['var']).all())}"
        )
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
