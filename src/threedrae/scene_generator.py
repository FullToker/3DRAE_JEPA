import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import nn

from stage2.transport import Sampler, create_transport
from utils import instantiate_from_config, load_safetensors

from .feature_extractor import FeatureExtractor3D, RELEASE_ROOT

DEFAULT_CONFIG = RELEASE_ROOT / "configs/models/dino2_3ddit.yaml"


class SceneGenerator3D(nn.Module):
    def __init__(
        self,
        device: str | torch.device = "cuda",
        precision: str = "bf16",
        config_path: Optional[str | Path] = None,
    ):
        super().__init__()
        if precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be 'bf16' or 'fp32'")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if precision == "bf16" and device.type != "cuda":
            raise ValueError("bf16 inference requires a CUDA device")

        config_path = (
            DEFAULT_CONFIG
            if config_path is None
            else Path(config_path).expanduser().resolve()
        )
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = yaml.safe_load(config_path.read_text())
        inference = config["inference"]

        self.condition_encoder = FeatureExtractor3D(
            model_name="dino2", device=device, precision=precision
        )
        self.model = instantiate_from_config(config)
        weights = Path(config["weights"])
        if not weights.is_absolute():
            weights = RELEASE_ROOT / weights
        load_safetensors(self.model, weights, strict=True)
        self.model.to(device)

        transport_config = inference["transport"]
        latent_size = tuple(inference["latent_size"])
        shift = math.sqrt(
            inference["time_dist_shift_dim"] / inference["time_dist_shift_base"]
        )
        self.transport = create_transport(
            **transport_config,
            time_dist_shift=shift,
        )
        self.latent_size = latent_size
        self.default_num_steps = int(inference["sampler"]["num_steps"])
        self.guidance_scale = float(inference["guidance_scale"])
        self.cfg_interval = tuple(inference["cfg_interval"])
        self.precision = precision
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _sample_ode(self, num_steps: int):
        return Sampler(self.transport).sample_ode(num_steps=num_steps)

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        visible_ids: torch.Tensor,
        seed: int = 0,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = 1 if images.ndim == 4 else images.shape[0]
        if visible_ids.ndim == 1:
            visible_ids = visible_ids.unsqueeze(0)
        if visible_ids.shape[0] != batch_size or visible_ids.shape[1] == 0:
            raise ValueError("visible_ids must select at least one view per batch item")

        guidance_scale = (
            self.guidance_scale if guidance_scale is None else float(guidance_scale)
        )
        if guidance_scale < 1:
            raise ValueError("guidance_scale must be at least 1")
        num_steps = self.default_num_steps if num_steps is None else num_steps

        condition = self.condition_encoder(
            images, intrinsics, extrinsics, visible_ids=visible_ids
        )
        use_guidance = guidance_scale > 1
        if use_guidance:
            empty_ids = torch.empty(
                batch_size, 0, dtype=torch.long, device=visible_ids.device
            )
            unconditional = self.condition_encoder(
                images, intrinsics, extrinsics, visible_ids=empty_ids
            )
            condition = torch.cat([condition, unconditional], dim=0)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            batch_size,
            *self.latent_size,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        if use_guidance:
            noise = torch.cat([noise, noise], dim=0)
            model_fn = self.model.forward_with_cfg
            model_kwargs = {
                "ci": condition,
                "cfg_scale": guidance_scale,
                "cfg_interval": self.cfg_interval,
            }
        else:
            model_fn = self.model.forward
            model_kwargs = {"ci": condition}

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        with autocast:
            generated = self._sample_ode(int(num_steps))(
                noise, model_fn, **model_kwargs
            )[-1]
        if use_guidance:
            generated = generated.chunk(2, dim=0)[0]
        generated = generated.float()
        if generated.shape != (batch_size, *self.latent_size):
            raise RuntimeError(f"Unexpected generated latent shape {tuple(generated.shape)}")
        if not torch.isfinite(generated).all():
            raise RuntimeError("3DDiT produced non-finite latent values")
        return generated
