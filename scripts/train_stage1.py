#!/usr/bin/env python3
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from disc import (
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from training import (
    barrier,
    build_scannet_loader,
    cleanup_distributed,
    configure_logging,
    load_checkpoint,
    load_config,
    make_grad_scaler,
    mask_invisible_views,
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
from training.optim import build_scheduler, get_muon_optimizer
from utils import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv2 Stage 1 on ScanNet.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/dino2_stage1_scannet.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-tokens-per-gpu", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--discriminator-start-step", type=int)
    parser.add_argument("--gan-start-step", type=int)
    parser.add_argument("--vis-interval", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def adaptive_weight(reconstruction, adversarial, layer, maximum):
    reconstruction_gradient = torch.autograd.grad(
        reconstruction, layer, retain_graph=True
    )[0]
    adversarial_gradient = torch.autograd.grad(
        adversarial, layer, retain_graph=True
    )[0]
    weight = reconstruction_gradient.norm() / (adversarial_gradient.norm() + 1e-6)
    return weight.clamp(0, maximum).detach()


def checkpoint_state(
    step,
    epoch,
    trainable,
    ema,
    optimizer,
    scheduler,
    discriminator,
    discriminator_optimizer,
    discriminator_scheduler,
):
    return {
        "stage": "stage1",
        "step": step,
        "epoch": epoch,
        "model": trainable.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "discriminator": unwrap_model(discriminator).state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "discriminator_scheduler": (
            discriminator_scheduler.state_dict()
            if discriminator_scheduler is not None
            else None
        ),
    }


@torch.inference_mode()
def save_visualization(model, ema, batch, path, precision):
    images = batch["images"][:1]
    visible_ids = batch["visible_ids"][:1]
    rays = batch["rays"][:1]
    target_rays = batch["target_rays"][:1]
    target = batch["target_images"][:1]
    with precision_context(precision):
        normalized = (
            images - model.image_mean.unsqueeze(1)
        ) / model.image_std.unsqueeze(1)
        features = model._encode_visible(normalized, visible_ids)
        latent = ema["neck"](features, visible_ids, rays)
        output = ema["decoder"](latent, target_rays)
        prediction = ema["decoder"].unpatchify(
            output.logits, original_image_size=target.shape[-2:]
        )
    prediction = prediction.view(1, target.shape[1], *target.shape[2:])
    condition = mask_invisible_views(images[0], visible_ids[0])
    save_training_visualization(path, condition, target[0], prediction[0])


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, device = setup_distributed()
    config_path, config = load_config(args.config)
    output_dir = resolve_path(args.output_dir, Path.cwd()).resolve()
    logger = configure_logging(output_dir, rank)
    if rank == 0:
        (output_dir / "config.yaml").write_text(config_path.read_text())

    training_config = config["training"]
    loss_config = config["loss"]
    dataset_config = dict(config["dataset"])
    dataset_config["path"] = str(
        resolve_path(args.data_path or dataset_config["path"], RELEASE_ROOT).resolve()
    )
    if args.max_tokens_per_gpu is not None:
        dataset_config["max_token_per_gpu"] = args.max_tokens_per_gpu
        dataset_config["gan_max_token_per_gpu"] = args.max_tokens_per_gpu
    if args.max_batch_size is not None:
        dataset_config["max_batch_size"] = args.max_batch_size
        dataset_config["gan_max_batch_size"] = args.max_batch_size
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
    if vis_interval < 0:
        raise ValueError("vis_interval must be non-negative")
    seed_everything(seed * world_size + rank)

    model_config = deepcopy(config["model"])
    encoder_path = model_config["params"]["encoder_params"]["dinov2_path"]
    model_config["params"]["encoder_params"]["dinov2_path"] = str(
        resolve_path(encoder_path, RELEASE_ROOT)
    )
    model = instantiate_from_config(model_config).to(device)
    if model.decoder_img is None or model.decoder_pmap is not None:
        raise ValueError("Stage 1 training expects an RGB decoder and no point decoder")
    model.encoder.requires_grad_(False)
    model.neck.requires_grad_(True)
    model.decoder_img.requires_grad_(True)
    model.train()
    model.encoder.eval()
    trainable = nn.ModuleDict({"neck": model.neck, "decoder": model.decoder_img})
    ema = deepcopy(trainable).to(device).eval().requires_grad_(False)
    wrapped_model = wrap_ddp(model, device)

    discriminator_config = deepcopy(config["discriminator"])
    discriminator_config["architecture"]["weights"] = str(
        resolve_path(
            discriminator_config["architecture"]["weights"], RELEASE_ROOT
        )
    )
    discriminator, augmentation = build_discriminator(discriminator_config, device)
    discriminator.train()
    wrapped_discriminator = wrap_ddp(discriminator, device)

    perceptual_config = config["perceptual"]
    perceptual = LPIPS(
        resolve_path(perceptual_config["vgg_weights"], RELEASE_ROOT),
        resolve_path(perceptual_config["lpips_weights"], RELEASE_ROOT),
    ).to(device)

    optimizer, optimizer_description = get_muon_optimizer(
        trainable, training_config
    )
    discriminator_optimizer, discriminator_optimizer_description = get_muon_optimizer(
        unwrap_model(wrapped_discriminator), discriminator_config
    )
    scheduler = discriminator_scheduler = None
    scheduler_description = discriminator_scheduler_description = None
    if training_config.get("scheduler"):
        scheduler, scheduler_description = build_scheduler(
            optimizer, training_config
        )
    if discriminator_config.get("scheduler"):
        discriminator_scheduler, discriminator_scheduler_description = build_scheduler(
            discriminator_optimizer, discriminator_config
        )
    scaler = make_grad_scaler(args.precision)

    loader, batch_sampler = build_scannet_loader(
        dataset_config, rank, world_size, seed, workers
    )
    epoch = step = 0
    if args.resume is not None:
        state = load_checkpoint(resolve_path(args.resume, Path.cwd()))
        if state.get("stage") != "stage1":
            raise ValueError("Resume checkpoint is not a Stage 1 checkpoint")
        trainable.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        unwrap_model(wrapped_discriminator).load_state_dict(
            state["discriminator"], strict=True
        )
        discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        if (
            discriminator_scheduler is not None
            and state["discriminator_scheduler"] is not None
        ):
            discriminator_scheduler.load_state_dict(
                state["discriminator_scheduler"]
            )
        epoch, step = int(state["epoch"]), int(state["step"])
    if step >= maximum_steps:
        raise ValueError(f"Checkpoint step {step} already reached max_steps={maximum_steps}")

    discriminator_loss = {
        "hinge": hinge_d_loss,
        "vanilla": vanilla_d_loss,
    }[loss_config["discriminator"]]
    generator_loss = {"vanilla": vanilla_g_loss}[loss_config["generator"]]
    gan_start = int(
        args.gan_start_step
        if args.gan_start_step is not None
        else loss_config["gan_start_step"]
    )
    discriminator_start = int(
        args.discriminator_start_step
        if args.discriminator_start_step is not None
        else loss_config["discriminator_start_step"]
    )
    gan_weight = float(loss_config["gan_weight"])
    perceptual_weight = float(loss_config["perceptual_weight"])
    maximum_adaptive_weight = float(loss_config["maximum_adaptive_weight"])
    ema_decay = float(training_config["ema_decay"])
    clip_gradient = float(training_config.get("clip_gradient", 0))
    checkpoint_interval = int(training_config["checkpoint_interval"])
    log_interval = int(training_config["log_interval"])
    checkpoint_dir = output_dir / "checkpoints"
    reconstruction_token_budget = int(dataset_config["max_token_per_gpu"])
    reconstruction_batch_limit = int(dataset_config["max_batch_size"])
    gan_token_budget = int(
        dataset_config.get("gan_max_token_per_gpu", reconstruction_token_budget)
    )
    gan_batch_limit = int(
        dataset_config.get("gan_max_batch_size", reconstruction_batch_limit)
    )

    if rank == 0:
        logger.info(optimizer_description)
        logger.info(discriminator_optimizer_description)
        logger.info(scheduler_description or "No generator scheduler")
        logger.info(discriminator_scheduler_description or "No discriminator scheduler")
        logger.info(
            "Training steps=%d, discriminator_start=%d, gan_start=%d, world_size=%d",
            maximum_steps,
            discriminator_start,
            gan_start,
            world_size,
        )
        logger.info("Visualization interval=%d", vis_interval)

    batch_sampler.set_epoch(epoch)
    iterator = iter(loader)
    gan_batch_budget_enabled = None
    while step < maximum_steps:
        current_gan_budget = step >= gan_start
        if current_gan_budget != gan_batch_budget_enabled:
            batch_sampler.config["max_token_per_gpu"] = (
                gan_token_budget
                if current_gan_budget
                else reconstruction_token_budget
            )
            batch_sampler.config["max_batch_size"] = (
                gan_batch_limit
                if current_gan_budget
                else reconstruction_batch_limit
            )
            gan_batch_budget_enabled = current_gan_budget
            if rank == 0:
                logger.info(
                    "Batch budget switched to max_tokens=%d, max_batch=%d",
                    batch_sampler.config["max_token_per_gpu"],
                    batch_sampler.config["max_batch_size"],
                )
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            batch_sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)

        use_gan = step >= gan_start and gan_weight > 0
        train_discriminator = step >= discriminator_start and gan_weight > 0
        target = batch["target_images"]
        batch_size, views, _, height, width = target.shape
        target = target.reshape(batch_size * views, 3, height, width)
        optimizer.zero_grad(set_to_none=True)
        unwrap_model(wrapped_discriminator).eval().requires_grad_(False)
        with precision_context(args.precision):
            reconstruction, _, _ = wrapped_model(
                batch["images"],
                batch["visible_ids"],
                ray_maps=batch["rays"],
                ray_maps_nv=batch["target_rays"],
            )
            reconstruction_loss = (reconstruction - target).abs().mean()
            perceptual_loss = perceptual(
                target * 2 - 1, reconstruction * 2 - 1
            )
            reconstruction_total = (
                reconstruction_loss + perceptual_weight * perceptual_loss
            )
            if use_gan:
                fake = augmentation.aug(reconstruction * 2 - 1)
                logits_fake, _ = unwrap_model(wrapped_discriminator)(fake, None)
                adversarial_loss = generator_loss(logits_fake)
            else:
                adversarial_loss = reconstruction_total.new_zeros(())
        if use_gan:
            weight = adaptive_weight(
                reconstruction_total,
                adversarial_loss,
                unwrap_model(wrapped_model).decoder_img.decoder_pred.weight,
                maximum_adaptive_weight,
            )
            total_loss = reconstruction_total + gan_weight * weight * adversarial_loss
        else:
            weight = reconstruction_total.new_zeros(())
            total_loss = reconstruction_total
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), clip_gradient)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        update_ema(ema, trainable, ema_decay)

        discriminator_metrics = None
        if train_discriminator:
            wrapped_model.eval()
            unwrap_model(wrapped_model).encoder.eval()
            unwrap_model(wrapped_discriminator).train().requires_grad_(True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), precision_context(args.precision):
                fresh_reconstruction, _, _ = wrapped_model(
                    batch["images"],
                    batch["visible_ids"],
                    ray_maps=batch["rays"],
                    ray_maps_nv=batch["target_rays"],
                )
            fake = (fresh_reconstruction * 2 - 1).clamp(-1, 1)
            fake = torch.round((fake + 1) * 127.5) / 127.5 - 1
            with precision_context(args.precision):
                logits_fake, logits_real = wrapped_discriminator(
                    augmentation.aug(fake.detach()),
                    augmentation.aug(target * 2 - 1),
                )
                discriminator_total = discriminator_loss(logits_real, logits_fake)
            scaler.scale(discriminator_total).backward()
            scaler.unscale_(discriminator_optimizer)
            scaler.step(discriminator_optimizer)
            scaler.update()
            if discriminator_scheduler is not None:
                discriminator_scheduler.step()
            discriminator_metrics = (
                discriminator_total,
                logits_real.mean(),
                logits_fake.mean(),
            )
            wrapped_model.train()
            unwrap_model(wrapped_model).encoder.eval()

        step += 1
        if log_interval > 0 and step % log_interval == 0:
            metrics = {
                "total": reduce_mean(total_loss),
                "reconstruction": reduce_mean(reconstruction_loss),
                "perceptual": reduce_mean(perceptual_loss),
                "generator_gan": reduce_mean(adversarial_loss),
                "adaptive_weight": reduce_mean(weight),
            }
            if discriminator_metrics is not None:
                metrics.update(
                    {
                        "discriminator": reduce_mean(discriminator_metrics[0]),
                        "logits_real": reduce_mean(discriminator_metrics[1]),
                        "logits_fake": reduce_mean(discriminator_metrics[2]),
                    }
                )
            if rank == 0:
                logger.info(
                    "step %d/%d | %s",
                    step,
                    maximum_steps,
                    " | ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                )

        if vis_interval > 0 and step % vis_interval == 0:
            barrier()
            if rank == 0:
                path = output_dir / "visualizations" / f"step-{step:08d}.png"
                save_visualization(
                    unwrap_model(wrapped_model), ema, batch, path, args.precision
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
                checkpoint_state(
                    step,
                    epoch,
                    trainable,
                    ema,
                    optimizer,
                    scheduler,
                    wrapped_discriminator,
                    discriminator_optimizer,
                    discriminator_scheduler,
                ),
            )
            logger.info("Saved %s", path)

    barrier()
    if rank == 0:
        logger.info("Stage 1 training complete at step %d", step)
    cleanup_distributed()


if __name__ == "__main__":
    main()
