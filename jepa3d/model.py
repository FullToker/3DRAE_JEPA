"""FusionGVJEPA with the 3DRAE scene encoder as X-encoder.

Everything after the X-encoder (vis_proj, Llama predictor, y-encoder,
InfoNCE) is fusion_gv's FusionGVJEPA, unchanged. Only two things differ:
  - the X-encoder is jepa3d.x_encoder.ThreeDRAEXEncoder;
  - visual pooling is skipped: 3DRAE already compresses all views into a
    fixed set of scene tokens, which go to the predictor as-is.

Optional object slots (jepa3d/slots.py; config "slots: {enabled: true}"): each
object name of the question carries candidate 3D positions, found without
training by dino.txt and the point probe. A SlotEncoder turns them into one
vector per name, which is added to the name's query tokens right after
query_in_proj. Nothing else changes; a model without slots has no slot
parameters.

Config (YAML) layout matches fusion_gv's, with
    fusion:  x_encoder_type: 3drae, x_encoder_output_dim: 768
    rae:     feature_config, max_views
    slots:   enabled, hidden
"""

from functools import partial
from unittest import mock

import torch

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
import fusion_gv.gvjepa as gvjepa
from fusion_gv.gvjepa import FusionGVJEPA
from fusion_gv.gvjepa_trainer import build_model_from_config as _build_fusion_gv_model
from jepa3d.slots import SlotEncoder
from jepa3d.x_encoder import ThreeDRAEXEncoder


class ThreeDRAEJEPA(FusionGVJEPA):
    def __init__(self, config, rae_cfg: dict | None = None, slot_cfg: dict | None = None):
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
        self._pending = None   # (queries, slots) of the forward in progress
        self.slot_encoder = None
        if (slot_cfg or {}).get("enabled"):
            self.slot_encoder = SlotEncoder(self.query_in_proj.out_features, slot_cfg.get("hidden", 512))
            self.query_in_proj.register_forward_hook(self._add_slots)

    def _add_slots(self, _module, _inputs, q_emb):
        """Forward hook on query_in_proj: q_emb (B, L, h) plus each slot's vector on its name's tokens."""
        if self._pending is None or self._pending[1] is None:
            return None
        queries, slots = self._pending
        offsets = self.query_tokenizer(queries, max_length=self.config.max_query_tokens, padding=True,
                                       truncation=True, return_offsets_mapping=True,
                                       return_tensors="pt")["offset_mapping"]
        assert offsets.shape[:2] == q_emb.shape[:2]
        start, end = offsets[slots["row"].cpu()].unbind(-1)                 # (N, L) characters of each token
        on_name = torch.zeros_like(start, dtype=torch.bool)
        for n, spans in enumerate(slots["spans"]):
            for a, b in spans:
                on_name[n] |= (start[n] < b) & (end[n] > a)
        vectors = self.slot_encoder(slots["pos"].to(q_emb.device), slots["conf"].to(q_emb.device))   # (N, h)
        add = on_name.to(q_emb.device)[..., None] * vectors[:, None].to(q_emb.dtype)
        return q_emb.index_add(0, slots["row"].to(q_emb.device), add)

    def forward(self, images, cams, queries, targets=None, mode: str = "infonce", slots=None):
        self._pending = (queries, slots)
        try:
            return super().forward(images, cams, queries, targets, mode)
        finally:
            self._pending = None

    def predict_embedding(self, images, cams, queries, slots=None):
        self._pending = (queries, slots)
        try:
            return super().predict_embedding(images, cams, queries)
        finally:
            self._pending = None

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
    factory = partial(ThreeDRAEJEPA, rae_cfg=cfg.get("rae"), slot_cfg=cfg.get("slots"))
    with mock.patch.object(gvjepa, "FusionGVJEPA", factory):
        return _build_fusion_gv_model(cfg)
