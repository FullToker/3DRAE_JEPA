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

With "slots: {enabled: true, mode: tokens}" (slots_v2) every name gets a token
of its own instead (jepa3d.slots.ObjectTokens): built from the frozen patch
features and pixel rays of the name's candidate patches, appended to the
predictor's input with the position id of the name's last token (so it sits
next to the name for the rotary embedding), and removed from the output
before pooling. forward() returns the loss on the output at these tokens
("slot_loss", "slot_regression") with "slot_acc" and "slot_error_m";
predict_embedding() leaves them in self.slot_metrics.

With "numeric_head: {enabled: true}" a small head on the pooled predictor
output (the input of pred_proj) gives one number per kind of numeric answer
(jepa3d.balance.KINDS), as log values: forward() returns it as "numeric"
(B, kinds), predict_embedding() leaves it in self.numeric_output. Which rows
and which kind it is read for is the caller's choice (the row's question
type); a model without the head has no such parameters.

With "ground: {enabled: true, layer: 4}" (jepa3d/ground.py) the hidden states
after predictor layer `layer` at the tokens of a labelled object name must
tell where the object is (GroundHead): forward(..., ground=) returns
"ground_loss", "ground_regression", "ground_acc" and "ground_error_m",
predict_embedding() leaves the terms in self.ground_metrics. Nothing is added
to the predictor's input.

With "slots: {enabled: true, mode: pointer, layers: [2, 4]}" (jepa3d/pointer.py, dino.txt encoder only)
every name gets a placeholder token that starts as the average of the scene tokens dino.txt matches
to the name, plus the name's text vector. It is appended like slots_v2's tokens, but its outputs are
part of the mean the answer is read from. After each predictor layer in `layers` the token must point
at a scene token that shows the object (rows with "ground" labels and their "views"): forward(...,
ground=) returns "ptr_loss" and the per-name terms, predict_embedding() leaves them in
self.pointer_metrics.

Config (YAML) layout matches fusion_gv's, with
    fusion:  x_encoder_type: 3drae, x_encoder_output_dim: 768
    rae:     feature_config, max_views; or encoder: dinotxt, pool, max_views (dino.txt patches instead of 3DRAE)
    slots:   enabled, hidden, mode (add | tokens | text: dino.txt name vectors, jepa3d/names.py | pointer
             | oracle: true positions, jepa3d/oracle.py), layers
    rae:     blind: true gives the predictor one all-zero scene token and never runs the encoder (E1, arm B)
    name_target: enabled, hidden (jepa3d/names.py)
    numeric_head: enabled, hidden
    ground:  enabled, layer, hidden
"""

from functools import partial
from unittest import mock

import torch
import torch.nn.functional as F

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
import fusion_gv.gvjepa as gvjepa
from fusion_gv.gvjepa import FusionGVJEPA
from fusion_gv.gvjepa_trainer import build_model_from_config as _build_fusion_gv_model
from jepa3d.balance import KINDS
from jepa3d.ground import GroundHead
from jepa3d.oracle import OracleTokens
from jepa3d.pointer import ObjectPointers, label_cells
from jepa3d.slots import ObjectTokens, SlotEncoder
from jepa3d.x_encoder import ThreeDRAEXEncoder, encode_patches, unpack_cams, view_counts


class ThreeDRAEJEPA(FusionGVJEPA):
    def __init__(self, config, rae_cfg: dict | None = None, slot_cfg: dict | None = None,
                 numeric_cfg: dict | None = None, ground_cfg: dict | None = None, name_cfg: dict | None = None):
        rae_cfg = rae_cfg or {}
        if rae_cfg.get("encoder", "3drae") == "dinotxt":   # baseline without 3DRAE (jepa3d/x_encoder_dinotxt.py)
            from jepa3d.x_encoder_dinotxt import DinoTxtXEncoder
            x_encoder = DinoTxtXEncoder(dim=config.fusion.x_encoder_output_dim, pool=rae_cfg.get("pool", 2),
                                        max_views=rae_cfg.get("max_views", 32))
        else:
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
        self.object_tokens = None
        self.slot_metrics = None
        self._layout = None    # (row, column, count) of the object tokens of the forward in progress
        self.pointers = None
        self.pointer_metrics = None
        self._ptr = None       # what the pointer tokens of the forward in progress were built from
        self._taps = {}        # {layer: hidden states at the pointer tokens}
        self.oracle_tokens = None
        self.blind = bool(rae_cfg.get("blind"))
        slot_cfg = slot_cfg or {}
        if slot_cfg.get("enabled") and slot_cfg.get("mode") == "oracle":   # jepa3d/oracle.py
            self.oracle_tokens = OracleTokens(self.vis_proj.out_features, slot_cfg.get("hidden", 512))
            self.predictor.register_forward_pre_hook(self._append_tokens, with_kwargs=True)
            self.predictor.register_forward_hook(self._remove_tokens, with_kwargs=True)
        elif slot_cfg.get("enabled") and slot_cfg.get("mode") == "pointer":   # jepa3d/pointer.py
            if rae_cfg.get("encoder") != "dinotxt":
                raise ValueError("slots.mode: pointer needs rae.encoder: dinotxt (scene tokens laid out by cell)")
            self.pointers = ObjectPointers(self.vis_proj.out_features, slot_cfg.get("layers", (2, 4)),
                                           slot_cfg.get("embed", 256))
            self.predictor.register_forward_pre_hook(self._append_tokens, with_kwargs=True)
            self.predictor.register_forward_hook(self._remove_tokens, with_kwargs=True)
            for k in self.pointers.layers:
                self.predictor.layers[k - 1].register_forward_hook(partial(self._tap_pointer, k))
            self.pred_proj.register_forward_pre_hook(self._pool_pointers)
        elif slot_cfg.get("enabled") and slot_cfg.get("mode", "add") == "tokens":
            self.object_tokens = ObjectTokens(self.vis_proj.out_features, x_encoder.rae.encoder.hidden_size + 6,
                                              slot_cfg.get("hidden", 512))
            self.predictor.register_forward_pre_hook(self._append_tokens, with_kwargs=True)
            self.predictor.register_forward_hook(self._remove_tokens, with_kwargs=True)
        elif slot_cfg.get("enabled") and slot_cfg.get("mode") == "text":   # jepa3d/names.py
            self.name_gain = torch.nn.Parameter(torch.ones(()))
            self.query_in_proj.register_forward_hook(self._add_names)
        elif slot_cfg.get("enabled"):
            self.slot_encoder = SlotEncoder(self.vis_proj.out_features, slot_cfg.get("hidden", 512))
            self.query_in_proj.register_forward_hook(self._add_slots)

        self.numeric_head = None
        self.numeric_output = None
        self._pooled = None
        if (numeric_cfg or {}).get("enabled"):
            hidden = numeric_cfg.get("hidden", 512)
            self.numeric_head = torch.nn.Sequential(
                torch.nn.LayerNorm(self.pred_proj.in_features), torch.nn.Linear(self.pred_proj.in_features, hidden),
                torch.nn.GELU(), torch.nn.Linear(hidden, len(KINDS)))
            self.pred_proj.register_forward_pre_hook(lambda _m, args: setattr(self, "_pooled", args[0]))

        self.name_head = None
        self.name_output = None
        if (name_cfg or {}).get("enabled"):   # second target for object-name answers (jepa3d/names.py)
            hidden, width = name_cfg.get("hidden", 512), x_encoder.feat_proj[0].normalized_shape[0]
            self.name_head = torch.nn.Sequential(
                torch.nn.LayerNorm(self.pred_proj.in_features), torch.nn.Linear(self.pred_proj.in_features, hidden),
                torch.nn.GELU(), torch.nn.Linear(hidden, width))
            if self.numeric_head is None:
                self.pred_proj.register_forward_pre_hook(lambda _m, args: setattr(self, "_pooled", args[0]))

        self.ground_head = None
        self.ground_metrics = None
        self._ground_hidden = None
        if (ground_cfg or {}).get("enabled"):
            if self.object_tokens is not None or self.pointers is not None:
                raise ValueError("ground and slots.mode: tokens / pointer cannot be combined")
            self.ground_head = GroundHead(self.pred_proj.in_features, ground_cfg.get("hidden", 512))
            layer = self.predictor.layers[ground_cfg.get("layer", 4) - 1]
            layer.register_forward_hook(self._tap_ground)

    def _tap_ground(self, _module, _args, out):
        self._ground_hidden = out[0] if isinstance(out, tuple) else out

    def _ground_loss(self, queries, ground) -> dict:
        """The GroundHead's terms for the names of `ground` (jepa3d.ground.collate_ground), read from the
        hidden states the tap kept: (B, S + L, h), the L question tokens last."""
        hidden, self._ground_hidden = self._ground_hidden, None
        if ground is None or hidden is None:
            return self.ground_head.zero()
        device = hidden.device
        on_name = self.name_tokens(queries, ground).to(device)                    # (N, L)
        rows = ground["row"].to(device)
        tokens = hidden[rows, hidden.shape[1] - on_name.shape[1]:].float()        # (N, L, h)
        has = on_name.any(1)                                                      # a truncated question may lose a name
        if not has.any():
            return self.ground_head.zero()
        mean = (tokens * on_name[..., None]).sum(1) / on_name.sum(1, keepdim=True).clamp_min(1)
        moved = {k: v.to(device)[has] for k, v in ground.items() if isinstance(v, torch.Tensor)}
        # "kept": which of the batch's labelled names the terms are for
        return {**self.ground_head.loss(mean[has], moved), "kept": has.nonzero().flatten()}

    def _numeric(self):
        return self.numeric_head(self._pooled.float())

    @property
    def uses_slots(self) -> bool:
        return (self.slot_encoder is not None or self.object_tokens is not None or self.pointers is not None
                or self.oracle_tokens is not None or hasattr(self, "name_gain"))

    def name_tokens(self, queries, slots) -> torch.Tensor:
        """(N, L) bool: the query tokens (as the predictor tokenizes them) that spell each slot's name."""
        offsets = self.query_tokenizer(queries, max_length=self.config.max_query_tokens, padding=True,
                                       truncation=True, return_offsets_mapping=True,
                                       return_tensors="pt")["offset_mapping"]
        start, end = offsets[slots["row"].cpu()].unbind(-1)                 # (N, L) characters of each token
        on_name = torch.zeros_like(start, dtype=torch.bool)
        for n, spans in enumerate(slots["spans"]):
            for a, b in spans:
                on_name[n] |= (start[n] < b) & (end[n] > a)
        return on_name

    def _add_slots(self, _module, _inputs, q_emb):
        """Forward hook on query_in_proj: q_emb (B, L, h) plus each slot's vector on its name's tokens."""
        if self._pending is None or self._pending[1] is None:
            return None
        queries, slots = self._pending[:2]
        on_name = self.name_tokens(queries, slots).to(q_emb.device)
        assert on_name.shape[1] == q_emb.shape[1]
        vectors = self.slot_encoder(slots["pos"].to(q_emb.device), slots["conf"].to(q_emb.device))   # (N, h)
        add = on_name[..., None] * vectors[:, None].to(q_emb.dtype)
        return q_emb.index_add(0, slots["row"].to(q_emb.device), add)

    def _add_names(self, _module, _inputs, q_emb):
        """Forward hook on query_in_proj: each name's dino.txt text vector, through the projections of the
        patch features, added on the name's tokens."""
        if self._pending is None or self._pending[1] is None:
            return None
        queries, slots = self._pending[:2]
        on_name = self.name_tokens(queries, slots).to(q_emb.device)
        assert on_name.shape[1] == q_emb.shape[1]
        text = self.x_encoder.text_vectors(slots["names"]).to(q_emb.device)
        vectors = self.vis_proj(self.x_encoder.feat_proj(text)) * self.name_gain                # (N, h)
        add = on_name[..., None] * vectors[:, None].to(q_emb.dtype)
        return q_emb.index_add(0, slots["row"].to(q_emb.device), add)

    def _token_inputs(self, images, cams, slots) -> torch.Tensor:
        """The object tokens (N, h) of a batch's slots: for each candidate patch its frozen patch feature
        and the Plucker ray of its centre in the normalized first-camera frame (the point prompt of
        jepa3d.geo_probe.train.point_queries)."""
        device = images.device
        patch, rows = slots["patch"].to(device), slots["row"].to(device)      # (N, K) over (view, row, column)
        size = self.x_encoder.rae.neck.patch_size
        h, w = images.shape[-2] // size, images.shape[-1] // size
        view, rest = patch // (h * w), patch % (h * w)
        # slots taken from another row (a control) may name a view this row does not have
        view = view % view_counts(cams)[rows][:, None]
        with torch.no_grad():
            feats = encode_patches(self.x_encoder.rae, images).float()        # (B, V, P, C)
            feature = F.layer_norm(feats[rows[:, None], view, rest], feats.shape[-1:])
            K, w2c = (x.float()[rows[:, None], view] for x in unpack_cams(cams))
            pixel = torch.stack([(rest % w + 0.5) * size, (rest // w + 0.5) * size, torch.ones_like(rest)], dim=-1)
            direction = F.normalize(torch.linalg.solve(K, pixel.float().unsqueeze(-1)).squeeze(-1), dim=-1)
            rotation_t = w2c[..., :3, :3].transpose(-1, -2)
            direction = (rotation_t @ direction.unsqueeze(-1)).squeeze(-1)
            centre = -(rotation_t @ w2c[..., :3, 3:]).squeeze(-1)
            queries = torch.cat([feature, direction, torch.cross(centre, direction, dim=-1)], dim=-1)
        return self.object_tokens(queries, slots["conf"].to(device))

    def _append_tokens(self, _module, args, kwargs):
        """Forward pre-hook on the predictor: x (B, S + L, h) gets the object tokens appended."""
        if self._pending is None or self._pending[1] is None:
            return None
        queries, slots = self._pending[:2]
        x, mask, position = args[0], kwargs["attention_mask"], kwargs["position_ids"]
        on_name = self.name_tokens(queries, slots).to(x.device)
        B, L = x.shape[0], on_name.shape[1]
        rows = slots["row"].to(x.device)
        if self.pointers is not None:
            tokens = self._pointer_tokens(x, mask[:, -L:].sum(1), slots, rows)
        elif self.oracle_tokens is not None:
            tokens = self.oracle_tokens(slots["pos"].to(x.device))
        else:
            tokens = self._pending[2].to(x.dtype) + self.modality_embed.weight[1].to(x.dtype)
        column = torch.zeros_like(rows)                       # index of each token among its row's tokens
        for n in range(1, len(rows)):
            column[n] = column[n - 1] + 1 if rows[n] == rows[n - 1] else 0
        count = int(column.max()) + 1
        extra = x.new_zeros(B, count, x.shape[-1])
        extra[rows, column] = tokens.to(x.dtype)
        extra_mask = mask.new_zeros(B, count)
        extra_mask[rows, column] = 1
        # query token l has position id l + 1; the object token takes the id of its name's last token
        last = (on_name * torch.arange(1, L + 1, device=x.device)).amax(1).clamp_min(1)
        extra_position = position.new_ones(B, count)
        extra_position[rows, column] = last.to(position.dtype)
        if position.dim() == 1:
            position = position.unsqueeze(0).expand(B, -1)
        self._layout = (rows, column, count)
        return (torch.cat([x, extra], dim=1),), {**kwargs, "attention_mask": torch.cat([mask, extra_mask], dim=1),
                                                 "position_ids": torch.cat([position, extra_position], dim=1)}

    def _remove_tokens(self, _module, _args, _kwargs, out):
        if self._layout is None:
            return None
        rows, column, count = self._layout
        self._layout = None
        self._token_outputs = out[:, -count:][rows, column]
        return out[:, :-count]

    def _pointer_tokens(self, x, query_tokens, slots, rows) -> torch.Tensor:
        """The placeholder tokens (N, h) of a batch's names, from the predictor's input x (B, S + L, h):
        the encoder's patch tokens come first in x, view by view, cell by cell."""
        last = self.x_encoder.last
        cells = last["features"].shape[1] * last["features"].shape[2]
        scene = x[:, :cells]
        text = self.x_encoder.text_vectors(slots["names"]).to(x.device)
        similarity = self.pointers.similarity(last["features"], last["valid"], text, rows)
        tokens, spread = self.pointers.tokens(scene, similarity[0], rows, self.vis_proj(self.x_encoder.feat_proj(text)))
        self._ptr = {"scene": scene, "similarity": similarity, "rows": rows, "spread": spread,
                     "query_tokens": query_tokens}
        return tokens

    def _tap_pointer(self, layer, _module, _args, out):
        if self._layout is None:
            return
        rows, column, count = self._layout
        hidden = out[0] if isinstance(out, tuple) else out
        self._taps[layer] = hidden[:, -count:][rows, column]

    def _pool_pointers(self, _module, args):
        """Forward pre-hook on pred_proj: the mean over the question tokens becomes the mean over the
        question tokens and the row's placeholder tokens."""
        if self._ptr is None or self._token_outputs is None:
            return None
        pooled, rows = args[0], self._ptr["rows"]
        total = torch.zeros_like(pooled).index_add(0, rows, self._token_outputs.to(pooled.dtype))
        placeholders = pooled.new_zeros(len(pooled)).index_add(0, rows, pooled.new_ones(len(rows)))
        questions = self._ptr["query_tokens"].to(pooled.dtype)
        return ((pooled * questions[:, None] + total) / (questions + placeholders).clamp_min(1)[:, None],)

    def _pointer_loss(self, slots, ground) -> dict:
        ptr, taps, self._ptr, self._taps = self._ptr, self._taps, None, {}
        if ptr is None or slots is None or ground is None or not any(ground.get("views") or []):
            return self.pointers.zero()
        # the tokens are in the order of the batch's slots: a labelled name is its row's first token + its slot index
        token = torch.searchsorted(slots["row"], ground["row"]) + ground["slot"]
        last = self.x_encoder.last
        labels = label_cells(ground, token, last["valid"], last["grid"], ptr["similarity"], last["rays"])
        if labels is None:
            return self.pointers.zero()
        hidden = {k: taps[k][labels["token"]] for k in self.pointers.layers}
        return {**self.pointers.terms(hidden, ptr["scene"], labels), "spread": ptr["spread"].mean()}

    def _begin(self, images, cams, queries, slots):
        self._token_outputs = None
        self._ptr, self._taps = None, {}
        if self.object_tokens is not None and slots is not None:
            self._pending = (queries, slots, self._token_inputs(images, cams, slots))
        else:
            self._pending = (queries, slots)

    def _slot_loss(self, slots) -> dict:
        if self._token_outputs is None:   # no names in this batch: keep the parameters in the graph
            zero = sum(p.sum() for p in self.object_tokens.parameters()) * 0.0
            return {"loss": zero, "regression": zero, "acc": zero.detach(), "error_m": zero.detach()}
        device = self._token_outputs.device
        return self.object_tokens.loss(self._token_outputs, slots["pos"][:, 0].to(device), slots["scale"].to(device))

    def forward(self, images, cams, queries, targets=None, mode: str = "infonce", slots=None, ground=None):
        self._begin(images, cams, queries, slots)
        try:
            out = super().forward(images, cams, queries, targets, mode)
            if self.ground_head is not None:
                out.update({f"ground_{k}": v for k, v in self._ground_loss(queries, ground).items()})
            if self.object_tokens is not None:
                out.update({f"slot_{k}": v for k, v in self._slot_loss(slots).items()})
            if self.pointers is not None:
                out.update({f"ptr_{k}": v for k, v in self._pointer_loss(slots, ground).items()})
            if self.numeric_head is not None:
                out["numeric"] = self._numeric()
            if self.name_head is not None:
                out["name_pred"] = self.name_head(self._pooled.float())
            return out
        finally:
            self._pending = None

    def predict_embedding(self, images, cams, queries, slots=None, ground=None):
        self._begin(images, cams, queries, slots)
        try:
            pred = super().predict_embedding(images, cams, queries)
            if self.ground_head is not None:
                self.ground_metrics = None if ground is None else self._ground_loss(queries, ground)
            if self.object_tokens is not None:
                self.slot_metrics = None if self._token_outputs is None else self._slot_loss(slots)
            if self.pointers is not None:
                self.pointer_metrics = self._pointer_loss(slots, ground)
            if self.numeric_head is not None:
                self.numeric_output = self._numeric()
            if self.name_head is not None:
                self.name_output = self.name_head(self._pooled.float())
            return pred
        finally:
            self._pending = None

    def _pool_visual(self, images, cams, batch_size):
        """images (B, V, 3, H, W), cams (B, V, 25) -> tokens (B, T, D), (B, 1, T, D).

        The second value stands in for per-frame patches: a single "frame"
        means FusionGVJEPA tags every token with the same frame_embed row and
        frame shuffling is a no-op.
        """
        if self.blind:
            tokens = images.new_zeros(images.shape[0], 1, self.x_encoder.dim)
            return tokens, tokens.unsqueeze(1)
        with torch.profiler.record_function("x_encoder"):
            tokens = self.x_encoder(images, cams)
        return tokens, tokens.unsqueeze(1)


def build_model_from_config(cfg: dict) -> ThreeDRAEJEPA:
    """fusion_gv's config -> GVJEPAConfig mapping, instantiating ThreeDRAEJEPA."""
    cfg = {**cfg, "fusion": {"jepa_ckpt": None, **cfg["fusion"]}}
    factory = partial(ThreeDRAEJEPA, rae_cfg=cfg.get("rae"), slot_cfg=cfg.get("slots"),
                      numeric_cfg=cfg.get("numeric_head"), ground_cfg=cfg.get("ground"),
                      name_cfg=cfg.get("name_target"))
    with mock.patch.object(gvjepa, "FusionGVJEPA", factory):
        return _build_fusion_gv_model(cfg)
