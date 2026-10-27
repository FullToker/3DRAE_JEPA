#!/usr/bin/env python3
"""Train FusionGVJEPA with the 3DRAE X-encoder on posed SPAR.

Same loop as fusion_gv/train_gvjepa.py (GVJEPATrainer, open-vljepa warm
start, cosine schedule, MLflow), with jepa3d's model and loaders.

Usage (from the repo root):
    accelerate launch [--multi_gpu --num_processes N] jepa3d/train.py \
        --config jepa3d/configs/sft_3drae_spar.yaml [--overfit --overfit-samples 32]
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs

from jepa3d.paths import expand_paths
from fusion_gv.gvjepa_trainer import GVJEPATrainer, build_optimizer_and_scheduler_from_config
from fusion_gv.mlflow_utils import start_mlflow_run
from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _repeat_first_batch(loader):
    """Overfit check: freeze the first batch and repeat it indefinitely."""
    fixed = next(iter(loader))

    class _RepeatLoader:
        def __iter__(self):
            while True:
                yield fixed

    return _RepeatLoader()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return expand_paths(yaml.safe_load(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit", action="store_true",
                        help="repeat one batch (train.batch_size samples) for --overfit-steps steps")
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    if args.output_dir:
        tcfg["output_dir"] = args.output_dir

    precision = tcfg.get("precision", "bf16")
    accelerator = Accelerator(
        mixed_precision=precision if precision in {"bf16", "fp16"} else "no",
        # frozen 3DRAE + pad-view token unused in some steps
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    _set_seed(tcfg.get("seed", 42) + accelerator.process_index)

    model = build_model_from_config(cfg)
    if cfg.get("init_vljepa_ckpt"):
        from fusion_gv.load_vljepa_init import load_predictor_and_y_encoder_from_vljepa
        load_predictor_and_y_encoder_from_vljepa(model, cfg["init_vljepa_ckpt"], load_vis_proj=False)

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"[params] trainable={trainable / 1e6:.1f}M frozen={frozen / 1e6:.1f}M")

    loader = build_loader(cfg["data"]["train_manifests"], cfg, tcfg["batch_size"], train=True)
    val_manifests = cfg["data"].get("val_manifests")
    val_loader = (
        build_loader(val_manifests, cfg, tcfg.get("val_batch_size", tcfg["batch_size"]), train=False)
        if val_manifests else None
    )
    optimizer, scheduler = build_optimizer_and_scheduler_from_config(model, cfg)
    # Scheduler deliberately not prepared: it steps once per optimizer update on every rank.
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if args.overfit:
        loader = _repeat_first_batch(loader)
        tcfg.update(max_steps=args.overfit_steps, log_every=10, save_every=0,
                    gradient_accumulation_steps=1)
        scheduler = None
        val_loader = None

    output_dir = Path(tcfg["output_dir"])
    config_copy = None
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_copy = output_dir / "config.yaml"
        config_copy.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with start_mlflow_run(cfg if accelerator.is_main_process else {}) as mlflow_logger:
        if config_copy and getattr(mlflow_logger, "enabled", False):
            mlflow_logger.log_artifact(config_copy)
        GVJEPATrainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            output_dir=tcfg["output_dir"],
            max_steps=tcfg["max_steps"],
            scheduler=scheduler,
            val_loader=val_loader,
            val_every=tcfg.get("val_every", 500),
            val_max_batches=tcfg.get("val_max_batches", 20),
            grad_accum_steps=tcfg.get("gradient_accumulation_steps", 1),
            clip_grad_norm=tcfg.get("clip_grad_norm", 1.0),
            temperature=tcfg.get("temperature", 0.07),
            log_every=tcfg.get("log_every", 20),
            save_every=tcfg.get("save_every", 0),
            mlflow_logger=mlflow_logger,
            accelerator=accelerator,
        ).fit()


if __name__ == "__main__":
    main()
