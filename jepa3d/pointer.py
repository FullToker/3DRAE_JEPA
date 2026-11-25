"""Object pointers: one placeholder token per object name of the question, which must point at the object.

Needs the dino.txt X-encoder (jepa3d/x_encoder_dinotxt.py): its scene tokens are
laid out by view and grid cell, so "where is the sofa" has an answer among them.

Input. A name's token is built from what is already in the predictor's input:

    token = sum_cells w * scene token of the cell          the related patch tokens
          + vis_proj(feat_proj(text vector of the name))   what the object is, also when it is not in view
          + type embedding                                  marks the token as a placeholder
    w = softmax(z / tau),  z = cos(text vector, dino.txt patch feature - mean patch feature of the row),
                           standardized over the row's cells; tau is learned, 0.5 at the start
    (the raw cosines differ so little that a softmax over them averaged 215 of the 704 cells)

so before any training the token already sits on the cells dino.txt matches to
the name. It is appended to the predictor's input with the position id of the
name's last token (jepa3d.model.ThreeDRAEJEPA).

Loss. After each predictor layer in `layers` the hidden state at the token is
compared with all scene tokens of its row (the predictor's input, through a
linear layer each) and must choose a cell that shows the object:

    positives  the cell holding the projection of the object's true centre, in every view
               where the centre is in the image and not hidden (jepa3d/tools/add_visibility.py)
    ignored    the 8 cells around a positive, and the cells around a hidden or unknown projection
    loss       -log( sum over positives of softmax over the cells that are not ignored )

Names without a visible view take no part. Reported per name: whether the best
cell is a positive or next to one ("hit"), the distance from the centre to the
ray through the best cell's middle (metres), and the same for the untrained
choice (argmax of w; "raw": the same without subtracting the mean feature).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def per_row(rows: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """left (N, a), right (B, a, b), rows (N,) -> (N, b): left[n] @ right[rows[n]], one row of the batch at a time."""
    out = left.new_zeros(len(left), right.shape[-1])
    for b in rows.unique().tolist():
        mine = rows == b
        out[mine] = (left[mine] @ right[b]).to(out.dtype)   # under autocast the product is bfloat16
    return out


class ObjectPointers(nn.Module):
    def __init__(self, dim: int, layers=(2, 4), embed: int = 256, temperature: float = 0.07,
                 init_temperature: float = 0.5):
        super().__init__()
        self.layers, self.temperature = tuple(layers), temperature
        self.type_embed = nn.Parameter(torch.randn(dim) * 0.02)
        self.name_gain = nn.Parameter(torch.ones(()))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self.read = nn.ModuleDict({str(k): nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed)) for k in layers})
        self.key = nn.ModuleDict({str(k): nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed)) for k in layers})

    @torch.no_grad()
    def similarity(self, feats: torch.Tensor, valid: torch.Tensor, text: torch.Tensor, rows: torch.Tensor):
        """feats (B, V, P, C) frozen patch features, valid (B, V), text (N, C), rows (N,) ->
        cosine of each name with each cell of its row, (N, V * P) twice: with the row's mean patch feature
        subtracted, and without. Cells of padded views are -inf."""
        feats, text = feats.float(), F.normalize(text.float(), dim=-1)
        keep = valid[:, :, None, None].float()
        mean = (feats * keep).sum((1, 2), keepdim=True) / (keep.sum((1, 2), keepdim=True) * feats.shape[2]).clamp_min(1)
        pad = ~valid[:, :, None].expand(-1, -1, feats.shape[2]).flatten(1)[rows]
        out = []
        with torch.autocast(device_type=feats.device.type, enabled=False):   # cosines differ in the third digit
            for cells in (F.normalize(feats - mean, dim=-1), F.normalize(feats, dim=-1)):
                out.append(per_row(rows, text, cells.flatten(1, 2).transpose(1, 2)).masked_fill(pad, float("-inf")))
        return out

    def tokens(self, scene: torch.Tensor, similarity: torch.Tensor, rows: torch.Tensor, names: torch.Tensor):
        """scene (B, V * P, h) the predictor's patch tokens, similarity (N, V * P), names (N, h) projected text
        vectors -> tokens (N, h), and the effective number of cells each one averages (N,)."""
        real = ~torch.isinf(similarity)
        count = real.sum(-1, keepdim=True)
        mean = similarity.masked_fill(~real, 0.0).sum(-1, keepdim=True) / count
        std = ((similarity - mean).masked_fill(~real, 0.0).square().sum(-1, keepdim=True) / count).sqrt()
        weights = ((similarity - mean) / std.clamp_min(1e-6) / self.log_tau.exp()).softmax(-1)
        related = per_row(rows, weights.to(scene.dtype), scene)
        spread = torch.special.entr(weights.detach()).sum(-1).exp()
        return related + (names * self.name_gain + self.type_embed).to(scene.dtype), spread

    def zero(self) -> dict:
        """No labelled, visible name in the batch: keep the parameters in the graph."""
        return {"loss": sum(p.sum() for p in self.parameters()) * 0.0}

    def terms(self, hidden: dict, scene: torch.Tensor, labels: dict) -> dict:
        """hidden {layer: (G, h)} states at the tokens of the G labelled names, scene (B, V * P, h),
        labels: label_cells' output. -> "loss" (sum over the layers) and, per name, "l<k>_hits",
        "l<k>_errors" (metres), "l<k>_loss", "init_hits", "init_errors", "raw_hits", "chance", "rows"."""
        with torch.autocast(device_type=scene.device.type, enabled=False):
            rows, positive, ignored, near, pad = (labels[k] for k in ("row", "positive", "ignored", "near", "pad"))
            out = {"rows": rows, "chance": near.float().sum(1) / (~pad).float().sum(1),
                   "init_hits": self._hit(labels["init"], near), "raw_hits": self._hit(labels["raw"], near),
                   "init_errors": self._error(labels["init"], labels), "loss": 0.0}
            scene = scene.float()
            for k in self.layers:
                keys = F.normalize(self.key[str(k)](scene), dim=-1)                                # (B, V * P, e)
                query = F.normalize(self.read[str(k)](hidden[k].float()), dim=-1)                  # (G, e)
                logits = per_row(rows, query, keys.transpose(1, 2)) / self.temperature
                logits = logits.masked_fill(pad, float("-inf"))
                loss = (logits.masked_fill(ignored, float("-inf")).logsumexp(-1)
                        - logits.masked_fill(~positive, float("-inf")).logsumexp(-1))
                out["loss"] = out["loss"] + loss.mean()
                out[f"l{k}_loss"] = loss.detach()
                out[f"l{k}_hits"] = self._hit(logits.detach(), near)
                out[f"l{k}_errors"] = self._error(logits.detach(), labels)
            return out

    @staticmethod
    def _hit(scores: torch.Tensor, near: torch.Tensor) -> torch.Tensor:
        return near.gather(1, scores.argmax(1, keepdim=True)).squeeze(1).float()

    @staticmethod
    def _error(scores: torch.Tensor, labels: dict) -> torch.Tensor:
        """Metres from each name's true centre to the ray through the middle of its best cell."""
        ray = labels["rays"][labels["row"], scores.argmax(1)]                       # (G, 6) direction, moment
        return (torch.cross(labels["target"], ray[:, :3], dim=-1) - ray[:, 3:]).norm(dim=-1) * labels["scale"]


def label_cells(ground: dict, token: torch.Tensor, valid: torch.Tensor, grid: tuple, similarity, rays) -> dict | None:
    """The labelled names that some view shows -> what ObjectPointers.terms needs.

    ground: jepa3d.ground.collate_ground's batch ("views" per name: [[view, u, v, state], ...]);
    token (N,): index of each labelled name's token among the batch's tokens; valid (B, V) real views;
    grid (h, w) cells per view; similarity: ObjectPointers.similarity's pair for all tokens;
    rays (B, V, P, 6) Plucker rays of the cell middles. None if no name has a visible view."""
    h, w = grid
    cells = h * w
    device = valid.device
    counts = valid.sum(1).tolist()
    kept, positive, ignored, near = [], [], [], []
    for n, (row, views) in enumerate(zip(ground["row"].tolist(), ground["views"])):
        pos = torch.zeros(valid.shape[1] * cells, dtype=torch.bool)
        around = torch.zeros_like(pos)
        skip = torch.zeros_like(pos)
        for view, u, v, state in views or []:
            if view >= counts[row]:
                continue
            r, c = min(int(v * h), h - 1), min(int(u * w), w - 1)
            block = [view * cells + rr * w + cc for rr in range(max(0, r - 1), min(h, r + 2))
                     for cc in range(max(0, c - 1), min(w, c + 2))]
            if state == 1:
                pos[view * cells + r * w + c] = True
                around[block] = True
            else:
                skip[block] = True
        if pos.any():
            kept.append(n)
            positive.append(pos)
            near.append(around)
            ignored.append((around | skip) & ~pos)
    if not kept:
        return None
    kept = torch.tensor(kept)
    rows = ground["row"][kept].to(device)
    token = token[kept].to(device)
    return {"row": rows, "token": token, "kept": kept,
            "positive": torch.stack(positive).to(device), "ignored": torch.stack(ignored).to(device),
            "near": torch.stack(near).to(device),
            "pad": ~valid[rows][:, :, None].expand(-1, -1, cells).flatten(1),
            "init": similarity[0][token], "raw": similarity[1][token],
            "rays": rays.flatten(1, 2).float(), "target": ground["target"][kept].to(device).float(),
            "scale": ground["scale"][kept].to(device).float()}
