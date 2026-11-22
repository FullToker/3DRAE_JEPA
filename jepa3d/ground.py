"""Grounding supervision inside the predictor: where is the object a question names?

A row's "ground" (jepa3d/tools/build_ground_labels.py) says which of its object
names ("slots": name + character spans in the question) is which object of the
scene's table (scene_objects.json: true centres, metres, scene frame). Nothing
is added to the predictor's input. After predictor layer `layer` the hidden
states of a name's tokens are averaged and a GroundHead must tell where the
object is, in two ways:

    choice      pick the object's centre among the centres of the scene's objects
                (cross-entropy; the centres are embedded by a small position encoder)
    regression  the centre itself (smooth L1)

Centres are in the frame of the row's first view, divided by the scene scale,
as the cameras the scene tokens were built with (jepa3d.data.normalize_cameras).
The answer loss stays as it is; the layers after `layer` are left to use what
the name tokens now hold.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa3d.slots import fourier, scene_scale

MAX_CANDIDATES = 64
SAME_OBJECT = 0.2   # metres: another centre this close to the target is not a wrong choice


class GroundHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, embed: int = 256, bands: int = 8, temperature: float = 0.07):
        super().__init__()
        self.bands, self.temperature = bands, temperature
        self.norm = nn.LayerNorm(dim)
        self.read = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, embed))
        self.metres = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.position = nn.Sequential(nn.Linear(3 * 2 * bands + 3, hidden), nn.GELU(), nn.Linear(hidden, embed))

    def zero(self) -> dict:
        """No labelled name in the batch: keep the parameters in the graph."""
        zero = sum(p.sum() for p in self.parameters()) * 0.0
        empty = zero.detach().new_zeros(0)
        return {"loss": zero, "regression": zero, "acc": zero.detach(), "error_m": zero.detach(),
                "errors": empty, "hits": empty, "chance": empty, "rows": empty.long()}

    def loss(self, hidden: torch.Tensor, ground: dict) -> dict:
        """hidden (N, dim): mean hidden state of each name's tokens. ground: collate_ground's tensors on its device.
        -> loss (choice), regression, acc, error_m (median), and per name: errors (metres), hits, chance, rows."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):   # centimetres need float32
            return self._loss(hidden.float(), ground)

    def _loss(self, hidden: torch.Tensor, ground: dict) -> dict:
        x = self.norm(hidden)
        target, candidates, valid = ground["target"], ground["candidates"], ground["valid"]
        scale, label = ground["scale"], ground["label"]
        embedded = F.normalize(self.position(torch.cat([fourier(4 * candidates, self.bands), candidates], dim=-1)), dim=-1)
        logits = (F.normalize(self.read(x), dim=-1)[:, None] * embedded).sum(-1) / self.temperature      # (N, C)
        near = (candidates - target[:, None]).norm(dim=-1) * scale[:, None] < SAME_OBJECT
        near[torch.arange(len(label)), label] = False
        usable = valid & ~near
        logits = logits.masked_fill(~usable, float("-inf"))
        guess = self.metres(x)
        errors = ((guess - target).norm(dim=-1) * scale).detach()
        hits = (logits.argmax(1) == label).float()
        return {"loss": F.cross_entropy(logits, label), "regression": F.smooth_l1_loss(guess, target),
                "acc": hits.mean(), "error_m": errors.median(), "errors": errors, "hits": hits,
                "chance": 1.0 / usable.sum(1).float(), "rows": ground["row"]}


def load_objects(path: str) -> dict:
    """scene_objects.json -> {"<subset>/<scene>": (M, 3) float64 centres in the scene's frame}"""
    import json
    with open(path) as f:
        return {key: np.array(scene["pos"], dtype=np.float64).reshape(-1, 3) for key, scene in json.load(f).items()}


def ground_of(row: dict, objects: dict, c2w: list, fixed: bool):
    """The labelled names of a row -> {"spans", "target" (n, 3), "candidates" (C, 3), "label" (n,), "scale"},
    positions in the first view's frame in metres; None without labels. c2w: the views the row is shown with."""
    entries = row.get("ground") or []
    table = objects.get(f"{row.get('subset')}/{row.get('scene')}")
    if not entries or table is None:
        return None
    w2c = np.linalg.inv(np.array(c2w[0], dtype=np.float64))
    positions = table @ w2c[:3, :3].T + w2c[:3, 3]
    wanted = list(dict.fromkeys(e["object"] for e in entries))
    others = [j for j in range(len(table)) if j not in set(wanted)]
    if not fixed:
        random.shuffle(others)
    keep = wanted + others[:max(0, MAX_CANDIDATES - len(wanted))]
    index = {j: i for i, j in enumerate(keep)}
    slots = row["slots"]   # after delexicalization the spans are those of the labels that replaced the names
    return {"spans": [slots[e["slot"]]["spans"] for e in entries],
            "target": positions[[e["object"] for e in entries]], "candidates": positions[keep],
            "label": [index[e["object"]] for e in entries], "scale": scene_scale(c2w)}


def collate_ground(items: list):
    """Per-row ground_of results (or None) -> None if the batch has no labelled name, else
    {"row" (N,), "spans" [N], "target" (N, 3), "candidates" (N, C, 3), "valid" (N, C), "label" (N,), "scale" (N,)},
    target and candidates divided by the scene scale."""
    rows, spans, target, label, scale, candidates = [], [], [], [], [], []
    for i, item in enumerate(items):
        if item is None:
            continue
        for span, position, index in zip(item["spans"], item["target"], item["label"]):
            rows.append(i)
            spans.append(span)
            target.append(position / item["scale"])
            label.append(index)
            scale.append(item["scale"])
            candidates.append(item["candidates"] / item["scale"])
    if not rows:
        return None
    width = max(len(c) for c in candidates)
    padded = torch.zeros(len(rows), width, 3)
    valid = torch.zeros(len(rows), width, dtype=torch.bool)
    for n, c in enumerate(candidates):
        padded[n, :len(c)] = torch.from_numpy(c).float()
        valid[n, :len(c)] = True
    return {"row": torch.tensor(rows), "spans": spans, "target": torch.tensor(np.array(target), dtype=torch.float32),
            "candidates": padded, "valid": valid, "label": torch.tensor(label),
            "scale": torch.tensor(scale, dtype=torch.float32)}
