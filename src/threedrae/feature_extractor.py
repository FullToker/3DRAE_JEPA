from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from utils import instantiate_from_config, plucker_rays_from_K_w2c
from utils.camera import patch_aligned_size, resize_intrinsics

RELEASE_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIGS = {
    "dino2": RELEASE_ROOT / "configs/models/dino2_feature.yaml",
    "dino2-256": RELEASE_ROOT / "configs/models/dino2_256_feature.yaml",
    "siglip2": RELEASE_ROOT / "configs/models/siglip2_feature.yaml",
}


def _resolve_config_paths(config: dict):
    params = config.get("params", {})
    for key in (
        "pretrained_neck_path",
        "pretrained_decoder_img_path",
        "pretrained_decoder_pmap_path",
        "normalization_stat_path",
    ):
        value = params.get(key)
        if value is not None and not Path(value).is_absolute():
            params[key] = str(RELEASE_ROOT / value)

    encoder_params = params.get("encoder_params", {})
    for key in ("dinov2_path", "model_name"):
        value = encoder_params.get(key)
        if value is None or Path(value).is_absolute():
            continue
        release_path = RELEASE_ROOT / value
        if release_path.exists():
            encoder_params[key] = str(release_path)
    return config


class FeatureExtractor3D(nn.Module):
    def __init__(
        self,
        model_name: str = "dino2",
        device: str | torch.device = "cuda",
        precision: str = "bf16",
        config_path: Optional[str | Path] = None,
    ):
        super().__init__()
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model_name={model_name!r}; choose from {sorted(MODEL_CONFIGS)}"
            )
        if precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be 'bf16' or 'fp32'")

        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if precision == "bf16" and device.type != "cuda":
            raise ValueError("bf16 inference requires a CUDA device")

        if config_path is None:
            config_path = MODEL_CONFIGS[model_name]
        else:
            config_path = Path(config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        config = yaml.safe_load(config_path.read_text())
        self.model = instantiate_from_config(_resolve_config_paths(config))
        self.model.to(device)
        self.precision = precision
        self.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _validate_and_batch(self, images, intrinsics, extrinsics):
        if not torch.is_floating_point(images):
            raise TypeError("images must be a floating-point tensor in [0, 1]")
        if images.ndim == 4:
            images = images.unsqueeze(0)
        elif images.ndim != 5:
            raise ValueError(
                f"images must have shape [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}"
            )
        if intrinsics.ndim == 3:
            intrinsics = intrinsics.unsqueeze(0)
        if extrinsics.ndim == 3:
            extrinsics = extrinsics.unsqueeze(0)

        batch_size, views, channels, _, _ = images.shape
        if batch_size == 0 or views == 0:
            raise ValueError("images must contain at least one batch item and one view")
        if channels != 3:
            raise ValueError(f"images must have 3 channels, got {channels}")
        if intrinsics.shape != (batch_size, views, 3, 3):
            raise ValueError(
                "intrinsics must have shape [S,3,3] or [B,S,3,3] matching images"
            )
        if extrinsics.shape != (batch_size, views, 4, 4):
            raise ValueError(
                "extrinsics must have shape [S,4,4] or [B,S,4,4] matching images"
            )
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (images, intrinsics, extrinsics)
        ):
            raise ValueError("Inputs contain non-finite values")
        if images.min() < 0 or images.max() > 1:
            raise ValueError("images must be in [0, 1]")
        return images, intrinsics, extrinsics

    def _resize_inputs(self, images, intrinsics):
        source_size = images.shape[-2:]
        target_size = patch_aligned_size(source_size, self.model.neck.patch_size)
        if target_size == source_size:
            return images, intrinsics
        batch_size, views, channels, _, _ = images.shape
        images = F.interpolate(
            images.reshape(batch_size * views, channels, *source_size),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        ).reshape(batch_size, views, channels, *target_size)
        return images, resize_intrinsics(intrinsics, source_size, target_size)

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        visible_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        images, intrinsics, extrinsics = self._validate_and_batch(
            images, intrinsics, extrinsics
        )
        images = images.to(self.device, dtype=torch.float32)
        intrinsics = intrinsics.to(dtype=torch.float32)
        extrinsics = extrinsics.to(intrinsics.device, dtype=torch.float32)
        images, intrinsics = self._resize_inputs(images, intrinsics)
        height, width = images.shape[-2:]
        ray_maps = torch.stack(
            [
                plucker_rays_from_K_w2c(K, w2c, height, width)
                for K, w2c in zip(intrinsics, extrinsics)
            ]
        ).to(self.device)
        if visible_ids is not None:
            if visible_ids.ndim == 1:
                visible_ids = visible_ids.unsqueeze(0)
            if visible_ids.ndim != 2 or visible_ids.shape[0] != images.shape[0]:
                raise ValueError("visible_ids must have shape [V] or [B,V]")
            if visible_ids.numel() and (
                visible_ids.min() < 0 or visible_ids.max() >= images.shape[1]
            ):
                raise IndexError("visible_ids contains an out-of-range view index")
            if any(len(row) != len(torch.unique(row)) for row in visible_ids):
                raise ValueError("visible_ids must not contain duplicates")
            visible_ids = visible_ids.to(self.device, dtype=torch.long)

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        with autocast:
            latent = self.model.encode(
                images, visible_ids=visible_ids, ray_maps=ray_maps
            )
        if not torch.isfinite(latent).all():
            raise RuntimeError("Feature extraction produced non-finite latent values")
        return latent
