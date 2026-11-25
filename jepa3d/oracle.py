"""E1, the upper-bound check: can the predictor answer a spatial question when it is handed the true positions?

Every object the question names gets a token that holds the object's true centre and nothing else
(no patch feature, no dino.txt, no confidence). It is appended to the predictor's input like slots_v2's
object tokens (position id of the name's last token) and removed again before the answer is pooled.
No loss is attached to it: the only training signal is the answer.

The positions come from the rows' "ground" labels (jepa3d/ground.py): first view's frame, divided by the
scene scale, as the cameras the scene tokens were built with. `oracle_slots` turns a batch's labels into
the "slots" the model and the controls of jepa3d/tools/blind_check.py already understand.

The direction rule of VSI-Bench (stand at A, face B, where is C) is here too, for
jepa3d/tools/make_oracle_manifest.py: the same rule as jepa3d/tools/slots_answer_vsibench.py, without
that script's imports.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa3d.slots import fourier


class OracleTokens(nn.Module):
    """Positions (N, 3), divided by the scene scale -> tokens (N, dim) of the size of a query token."""

    def __init__(self, dim: int, hidden: int = 512, bands: int = 8):
        super().__init__()
        self.bands = bands
        self.embed = nn.Sequential(nn.Linear(3 * 2 * bands + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                   nn.GELU(), nn.Linear(hidden, dim), nn.LayerNorm(dim))
        self.gain = nn.Parameter(torch.ones(()))
        self.type_embed = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=pos.device.type, enabled=False):   # centimetres need float32
            pos = pos.float()
            x = self.embed(torch.cat([fourier(4 * pos, self.bands), pos], dim=-1))
            return x * self.gain / x.shape[-1] ** 0.5 + self.type_embed


def oracle_slots(ground: dict | None) -> dict | None:
    """jepa3d.ground.collate_ground's batch -> slots: one per labelled name, "pos" (N, 3) its true centre."""
    if ground is None:
        return None
    return {"row": ground["row"], "spans": ground["spans"], "pos": ground["target"], "scale": ground["scale"]}


def permute_within_rows(slots: dict) -> dict:
    """Control: every name gets the position of the next name of its own question. The set of positions
    a question sees is unchanged; only which name owns which position is wrong."""
    rows, pos = slots["row"], slots["pos"].clone()
    for b in rows.unique().tolist():
        mine = (rows == b).nonzero().flatten()
        pos[mine] = slots["pos"][mine.roll(1)]
    return {**slots, "pos": pos}


def flip_within_rows(slots: dict) -> dict:
    """Control: the first two names of every question (where I stand, what I face) trade positions.
    Unlike the cyclic permute_within_rows, this turns left into right: a model that reads the positions
    must fall below the answer prior, not stay on it."""
    rows, pos = slots["row"], slots["pos"].clone()
    for b in rows.unique().tolist():
        mine = (rows == b).nonzero().flatten()
        if len(mine) >= 2:
            pos[mine[:2]] = slots["pos"][mine[:2].flip(0)]
    return {**slots, "pos": pos}


ANSWERS = ("left", "right", "back", "front-left", "front-right", "back-left", "back-right")


class DirectionHead(nn.Module):
    """Arm D: the direction as a class, read from the pooled predictor output (the input of pred_proj).
    Tells the predictor apart from the sentence-embedding target: same input, same predictor, but a
    target in which left and right are different classes instead of two sentences one word apart."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, len(ANSWERS)))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled.float())


def direction_terms(logits: torch.Tensor, rows: list) -> dict:
    """logits (B, 7), the batch's manifest rows -> {"loss", "hits" [n]} over the rows with a direction answer;
    the classes that are not among a row's options are masked."""
    keep, labels, allowed = [], [], []
    for i, row in enumerate(rows):
        options = list(dict.fromkeys(row.get("options") or []))
        if row.get("target_short") in ANSWERS and options and all(o in ANSWERS for o in options):
            keep.append(i)
            labels.append(ANSWERS.index(row["target_short"]))
            allowed.append([a in options for a in ANSWERS])
    if not keep:
        return {"loss": logits.sum() * 0.0, "hits": []}
    labels = torch.tensor(labels, device=logits.device)
    masked = logits[keep].float().masked_fill(~torch.tensor(allowed, device=logits.device), float("-inf"))
    return {"loss": F.cross_entropy(masked, labels), "hits": (masked.argmax(1) == labels).tolist()}


def azimuth(stand, face, target, up) -> float:
    """Clockwise angle (deg, seen from above) of target for "stand, face"; positive = right."""
    flat = lambda x: x - (x @ up) * up
    forward, delta = flat(face - stand), flat(target - stand)
    right = np.cross(forward, up)
    return float(np.degrees(np.arctan2(delta @ right / max(np.linalg.norm(right), 1e-9),
                                       delta @ forward / max(np.linalg.norm(forward), 1e-9))))


def direction_answer(options: int, angle: float) -> str:
    """The answer among 2 (left / right), 3 (+ back: 135 degrees or more) or 4 (quadrants) options."""
    side = "right" if angle > 0 else "left"
    if options == 2:
        return side
    if options == 3:
        return "back" if abs(angle) >= 135 else side
    return ("front-" if abs(angle) < 90 else "back-") + side


def boundary_distance(options: int, angle: float) -> float:
    """Degrees between the angle and the nearest border between two answers."""
    borders = {2: (0, 180, -180), 3: (0, 135, -135), 4: (0, 90, -90, 180, -180)}[options]
    return min(abs(angle - b) for b in borders)
