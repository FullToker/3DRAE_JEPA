"""FusionGVJEPA with the 3DRAE scene encoder as X-encoder.

Everything after the X-encoder (vis_proj, Llama predictor, y-encoder,
InfoNCE) is fusion_gv's FusionGVJEPA, unchanged. Only two things differ:
  - the X-encoder is jepa3d.x_encoder.ThreeDRAEXEncoder;
  - visual pooling is skipped: 3DRAE already compresses all views into a
    fixed set of scene tokens, which go to the predictor as-is.

Config (YAML) layout matches fusion_gv's, with
    fusion:  x_encoder_type: 3drae, x_encoder_output_dim: 768
    rae:     feature_config, max_views
"""

from functools import partial
from unittest import mock

import torch

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
import fusion_gv.gvjepa as gvjepa
from fusion_gv.gvjepa import FusionGVJEPA
from fusion_gv.gvjepa_trainer import build_model_from_config as _build_fusion_gv_model
from jepa3d.x_encoder import ThreeDRAEXEncoder


class ThreeDRAEJEPA(FusionGVJEPA):
    def __init__(self, config, rae_cfg: dict | None = None):
        rae_cfg = rae_cfg or {}
        x_encoder = ThreeDRAEXEncoder(
            feature_config=rae_cfg.get("feature_config", "configs/models/dino2_feature.yaml"),
            max_views=rae_cfg.get("max_views", 32),
        )
        if config.fusion.x_encoder_output_dim != x_encoder.dim:
            raise ValueError(
                f"fusion.x_encoder_output_dim must be {x_encoder.dim} for 3DRAE, "
                f"got {config.fusion.x_encoder_output_dim}"
            )
        if config.visual_pool_k != 1 or config.grounding_enabled:
            raise ValueError("3DRAE X-encoder supports neither visual_pool_k > 1 nor grounding")
        # FusionGVJEPA.__init__ builds its X-encoder by name; hand it ours instead.
        with mock.patch.object(gvjepa, "build_x_encoder", lambda _cfg: x_encoder):
            super().__init__(config)

    def _pool_visual(self, images, cams, batch_size):
        """images (B, V, 3, H, W), cams (B, V, 25) -> tokens (B, T, D), (B, 1, T, D).

        The second value stands in for per-frame patches: a single "frame"
        means FusionGVJEPA tags every token with the same frame_embed row and
        frame shuffling is a no-op.
        """
        with torch.profiler.record_function("x_encoder"):
            tokens = self.x_encoder(images, cams)
        return tokens, tokens.unsqueeze(1)


def build_model_from_config(cfg: dict) -> ThreeDRAEJEPA:
    """fusion_gv's config -> GVJEPAConfig mapping, instantiating ThreeDRAEJEPA."""
    cfg = {**cfg, "fusion": {"jepa_ckpt": None, **cfg["fusion"]}}
    factory = partial(ThreeDRAEJEPA, rae_cfg=cfg.get("rae"))
    with mock.patch.object(gvjepa, "FusionGVJEPA", factory):
        return _build_fusion_gv_model(cfg)
