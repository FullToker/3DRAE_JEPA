from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from stage1.decoders import build_decoder
from utils import load_safetensors, load_tensor_dict, plucker_rays_from_K_w2c
from utils.camera import patch_aligned_size, resize_intrinsics

from .feature_extractor import RELEASE_ROOT, _resolve_config_paths

MODEL_CONFIGS = {
    "dino2": RELEASE_ROOT / "configs/models/dino2_rae.yaml",
    "dino2-256": RELEASE_ROOT / "configs/models/dino2_256_rae.yaml",
    "siglip2": RELEASE_ROOT / "configs/models/siglip2_rae.yaml",
}


class Stage1Decoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dino2",
        device: str | torch.device = "cuda",
        precision: str = "fp32",
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

        config = _resolve_config_paths(yaml.safe_load(config_path.read_text()))
        params = config["params"]
        self.image_decoder = build_decoder(
            params["decoder_img_cls"], **params["decoder_img_params"]
        )
        load_safetensors(
            self.image_decoder, params["pretrained_decoder_img_path"], strict=True
        )

        point_cls = params.get("decoder_pmap_cls")
        if point_cls is not None:
            self.point_decoder = build_decoder(
                point_cls, **params["decoder_pmap_params"]
            )
            load_safetensors(
                self.point_decoder,
                params["pretrained_decoder_pmap_path"],
                strict=True,
            )
        else:
            self.point_decoder = None

        stats = load_tensor_dict(params["normalization_stat_path"])
        if set(stats) != {"mean", "var"}:
            raise KeyError(
                "Normalization stats must contain exactly the 'mean' and 'var' tensors"
            )
        self.register_buffer("latent_mean", stats["mean"], persistent=False)
        self.register_buffer("latent_var", stats["var"], persistent=False)
        self.eps = params.get("eps", 1e-5)
        self.precision = precision
        self.to(device)
        self.eval()

    @property
    def device(self):
        return next(self.image_decoder.parameters()).device

    def _prepare_inputs(self, latent, intrinsics, extrinsics, image_size):
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        elif latent.ndim != 3:
            raise ValueError(
                f"latent must have shape [N,C] or [B,N,C], got {tuple(latent.shape)}"
            )
        if intrinsics.ndim == 3:
            intrinsics = intrinsics.unsqueeze(0)
        if extrinsics.ndim == 3:
            extrinsics = extrinsics.unsqueeze(0)

        batch_size = latent.shape[0]
        if intrinsics.ndim != 4 or intrinsics.shape[0] != batch_size:
            raise ValueError("intrinsics batch dimension must match latent")
        views = intrinsics.shape[1]
        if views == 0:
            raise ValueError("At least one target view is required")
        if intrinsics.shape[2:] != (3, 3):
            raise ValueError("intrinsics must have shape [S,3,3] or [B,S,3,3]")
        if extrinsics.shape != (batch_size, views, 4, 4):
            raise ValueError(
                "extrinsics must have shape [S,4,4] or [B,S,4,4] matching intrinsics"
            )

        if len(image_size) != 2:
            raise ValueError("image_size must be (height, width)")
        height, width = (int(value) for value in image_size)
        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive")
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (latent, intrinsics, extrinsics)
        ):
            raise ValueError("Inputs contain non-finite values")
        return latent, intrinsics, extrinsics, height, width

    @torch.inference_mode()
    def forward(
        self,
        latent: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        latent, intrinsics, extrinsics, height, width = self._prepare_inputs(
            latent, intrinsics, extrinsics, image_size
        )
        network_size = patch_aligned_size(
            (height, width), self.image_decoder.patch_size
        )
        intrinsics = intrinsics.to(dtype=torch.float32)
        extrinsics = extrinsics.to(intrinsics.device, dtype=torch.float32)
        network_intrinsics = resize_intrinsics(
            intrinsics, (height, width), network_size
        )
        network_height, network_width = network_size
        ray_maps = torch.stack(
            [
                plucker_rays_from_K_w2c(
                    K, w2c, network_height, network_width
                )
                for K, w2c in zip(network_intrinsics, extrinsics)
            ]
        ).to(self.device)
        latent = latent.to(self.device, dtype=torch.float32)
        latent = latent * torch.sqrt(self.latent_var + self.eps) + self.latent_mean

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        with autocast:
            image_output = self.image_decoder(latent, ray_maps)
            images = self.image_decoder.unpatchify(
                image_output.logits, original_image_size=network_size
            )
            images = images.view(
                latent.shape[0], -1, 3, network_height, network_width
            )
            if self.point_decoder is not None:
                point_output = self.point_decoder(latent, ray_maps)
                points = point_output.pts3d
                confidence = point_output.pts3d_conf
            else:
                points = confidence = None

        if network_size != (height, width):
            batch_size, views = images.shape[:2]
            images = F.interpolate(
                images.reshape(batch_size * views, 3, *network_size),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).reshape(batch_size, views, 3, height, width)
            if points is not None:
                points = F.interpolate(
                    points.reshape(batch_size * views, *network_size, 3).permute(
                        0, 3, 1, 2
                    ),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).reshape(
                    batch_size, views, height, width, 3
                )
                confidence = F.interpolate(
                    confidence.reshape(batch_size * views, 1, *network_size),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(batch_size, views, height, width)

        if not torch.isfinite(images).all():
            raise RuntimeError("Image decoder produced non-finite values")
        if points is not None and not all(
            torch.isfinite(tensor).all() for tensor in (points, confidence)
        ):
            raise RuntimeError("Point decoder produced non-finite values")
        return images, points, confidence
