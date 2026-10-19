#!/usr/bin/env python3
import argparse
import math
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from stage2.transport import Sampler, create_transport
from training import (
    barrier,
    build_scannet_loader,
    cleanup_distributed,
    configure_logging,
    load_checkpoint,
    load_config,
    make_grad_scaler,
    mask_invisible_views,
    model_state_dict,
    precision_context,
    reduce_mean,
    resolve_path,
    save_checkpoint,
    save_training_visualization,
    seed_everything,
    setup_distributed,
    unwrap_model,
    update_ema,
    wrap_ddp,
)
from training.optim import build_optimizer, build_scheduler
from utils import instantiate_from_config, load_safetensors


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv2 3DDiT on ScanNet.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/dino2_stage2_scannet.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--neck-weights", type=Path)
    parser.add_argument("--decoder-weights", type=Path)
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-tokens-per-gpu", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--vis-interval", type=int)
    parser.add_argument("--vis-num-steps", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def resolve_autoencoder_paths(
    config,
    include_decoder,
    neck_weights=None,
    decoder_weights=None,
    normalization_stats=None,
):
    config = deepcopy(config)
    parameters = config["params"]
    if neck_weights is not None:
        parameters["pretrained_neck_path"] = str(neck_weights)
    if decoder_weights is not None:
        parameters["pretrained_decoder_img_path"] = str(decoder_weights)
    if normalization_stats is not None:
        parameters["normalization_stat_path"] = str(normalization_stats)
    parameters["encoder_params"]["dinov2_path"] = str(
        resolve_path(parameters["encoder_params"]["dinov2_path"], RELEASE_ROOT)
    )
    if not include_decoder:
        for key in (
            "decoder_img_cls",
            "decoder_img_params",
            "pretrained_decoder_img_path",
        ):
            parameters.pop(key, None)
    for key in (
        "pretrained_neck_path",
        "pretrained_decoder_img_path",
        "normalization_stat_path",
    ):
        if key in parameters:
            parameters[key] = str(resolve_path(parameters[key], RELEASE_ROOT))
    return config


def next_batch(iterator, loader, sampler, epoch):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


@torch.inference_mode()
def save_visualization(
    autoencoder,
    ema,
    sampler,
    noise,
    batch,
    condition,
    path,
    precision,
):
    with precision_context(precision):
        generated = sampler(noise.clone(), ema.forward, ci=condition[:1])[-1]
        prediction, _, _ = autoencoder.decode(generated.float(), batch["rays"][:1])
    target = batch["images"][:1]
    prediction = prediction.view(1, target.shape[1], *target.shape[2:])
    visible_ids = batch["visible_ids"][0]
    visible = mask_invisible_views(target[0], visible_ids)
    save_training_visualization(path, visible, target[0], prediction[0])


def main():
    args = parse_args()
    if args.resume is not None and args.init_weights is not None:
        raise ValueError("--resume and --init-weights are mutually exclusive")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, device = setup_distributed()
    config_path, config = load_config(args.config)
    output_dir = resolve_path(args.output_dir, Path.cwd()).resolve()
    logger = configure_logging(output_dir, rank)
    if rank == 0:
        (output_dir / "config.yaml").write_text(config_path.read_text())

    training_config = config["training"]
    dataset_config = dict(config["dataset"])
    dataset_config["path"] = str(
        resolve_path(args.data_path or dataset_config["path"], RELEASE_ROOT).resolve()
    )
    if args.max_tokens_per_gpu is not None:
        dataset_config["max_token_per_gpu"] = args.max_tokens_per_gpu
    if args.max_batch_size is not None:
        dataset_config["max_batch_size"] = args.max_batch_size
    maximum_steps = int(args.max_steps or training_config["max_steps"])
    workers = int(
        args.num_workers
        if args.num_workers is not None
        else training_config["num_workers"]
    )
    seed = int(training_config.get("seed", 42))
    vis_interval = int(
        args.vis_interval
        if args.vis_interval is not None
        else training_config.get("vis_interval", 0)
    )
    vis_num_steps = int(
        args.vis_num_steps
        if args.vis_num_steps is not None
        else training_config.get("vis_num_steps", 50)
    )
    if vis_interval < 0:
        raise ValueError("vis_interval must be non-negative")
    if vis_interval > 0 and vis_num_steps < 2:
        raise ValueError("vis_num_steps must be at least 2")
    seed_everything(seed * world_size + rank)

    autoencoder = instantiate_from_config(
        resolve_autoencoder_paths(
            config["autoencoder"],
            include_decoder=rank == 0 and vis_interval > 0,
            neck_weights=args.neck_weights,
            decoder_weights=args.decoder_weights,
            normalization_stats=args.normalization_stats,
        )
    ).to(device)
    if autoencoder.decoder_pmap is not None:
        raise ValueError("Stage 2 training does not use the point decoder")
    if rank == 0 and vis_interval > 0 and autoencoder.decoder_img is None:
        raise ValueError("Stage 2 visualization requires an RGB decoder")
    autoencoder.eval().requires_grad_(False)
    model = instantiate_from_config(config["model"]).to(device)
    if args.init_weights is not None:
        load_safetensors(
            model, resolve_path(args.init_weights, Path.cwd()), strict=True
        )
    ema = deepcopy(model).to(device).eval().requires_grad_(False)
    wrapped_model = wrap_ddp(model, device)
    wrapped_model.train()

    optimizer, optimizer_description = build_optimizer(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        training_config,
    )
    scheduler, scheduler_description = build_scheduler(
        optimizer, training_config
    )
    scaler = make_grad_scaler(args.precision)
    loader, batch_sampler = build_scannet_loader(
        dataset_config, rank, world_size, seed, workers
    )

    transport_config = dict(config["transport"])
    latent_shape = tuple(config["latent_shape"])
    shift = math.sqrt(
        int(config["time_distribution_shift_dim"])
        / int(config["time_distribution_shift_base"])
    )
    transport = create_transport(**transport_config, time_dist_shift=shift)
    visualization_sampler = visualization_noise = None
    if rank == 0 and vis_interval > 0:
        visualization_sampler = Sampler(transport).sample_ode(
            num_steps=vis_num_steps,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        visualization_noise = torch.randn(
            1,
            *latent_shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    gradient_accumulation = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else training_config.get("gradient_accumulation_steps", 1)
    )
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    ema_decay = float(training_config["ema_decay"])
    clip_gradient = float(training_config.get("clip_gradient", 0))
    checkpoint_interval = int(training_config["checkpoint_interval"])
    log_interval = int(training_config["log_interval"])
    checkpoint_dir = output_dir / "checkpoints"

    epoch = step = 0
    if args.resume is not None:
        state = load_checkpoint(resolve_path(args.resume, Path.cwd()))
        if state.get("stage") != "stage2":
            raise ValueError("Resume checkpoint is not a Stage 2 checkpoint")
        unwrap_model(wrapped_model).load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, step = int(state["epoch"]), int(state["step"])
    if step >= maximum_steps:
        raise ValueError(f"Checkpoint step {step} already reached max_steps={maximum_steps}")

    if rank == 0:
        logger.info(optimizer_description)
        logger.info(scheduler_description)
        logger.info(
            "Training steps=%d, accumulation=%d, world_size=%d, latent_shape=%s",
            maximum_steps,
            gradient_accumulation,
            world_size,
            latent_shape,
        )
        logger.info(
            "Visualization interval=%d, sampling_steps=%d",
            vis_interval,
            vis_num_steps,
        )

    batch_sampler.set_epoch(epoch)
    iterator = iter(loader)
    while step < maximum_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        for micro_step in range(gradient_accumulation):
            batch, iterator, epoch = next_batch(
                iterator, loader, batch_sampler, epoch
            )
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            with torch.no_grad(), precision_context(args.precision):
                latent = autoencoder.encode(
                    batch["images"], None, ray_maps=batch["rays"]
                )
                condition = autoencoder.encode(
                    batch["images"],
                    batch["visible_ids"],
                    ray_maps=batch["rays"],
                )
            if latent.shape[1:] != latent_shape or condition.shape[1:] != latent_shape:
                raise RuntimeError(
                    f"Unexpected latent shapes {tuple(latent.shape)}, {tuple(condition.shape)}"
                )
            synchronize = micro_step + 1 == gradient_accumulation
            synchronization_context = (
                nullcontext()
                if synchronize or not hasattr(wrapped_model, "no_sync")
                else wrapped_model.no_sync()
            )
            with synchronization_context, precision_context(args.precision):
                loss = transport.training_losses(
                    wrapped_model, latent, {"ci": condition}
                )["loss"].mean()
                scaled_loss = loss / gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.detach()

        scaler.unscale_(optimizer)
        if clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema, unwrap_model(wrapped_model), ema_decay)
        step += 1

        if log_interval > 0 and step % log_interval == 0:
            loss_value = reduce_mean(accumulated_loss / gradient_accumulation)
            if rank == 0:
                logger.info(
                    "step %d/%d | loss=%.6f | lr=%.8f",
                    step,
                    maximum_steps,
                    loss_value,
                    optimizer.param_groups[0]["lr"],
                )

        if vis_interval > 0 and step % vis_interval == 0:
            barrier()
            if rank == 0:
                path = output_dir / "visualizations" / f"step-{step:08d}.png"
                save_visualization(
                    autoencoder,
                    ema,
                    visualization_sampler,
                    visualization_noise,
                    batch,
                    condition,
                    path,
                    args.precision,
                )
                logger.info("Saved visualization %s", path)
            barrier()

        should_save = step == maximum_steps or (
            checkpoint_interval > 0 and step % checkpoint_interval == 0
        )
        if should_save and rank == 0:
            path = checkpoint_dir / f"step-{step:08d}.pt"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
            save_checkpoint(
                path,
                {
                    "stage": "stage2",
                    "step": step,
                    "epoch": epoch,
                    "model": model_state_dict(wrapped_model),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
            )
            logger.info("Saved %s", path)

    barrier()
    if rank == 0:
        logger.info("Stage 2 training complete at step %d", step)
    cleanup_distributed()


if __name__ == "__main__":
    main()
