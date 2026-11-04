"""Cross-attention position probe, its loss and "stand at A, face C" metrics."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MIN_PAIR_DISTANCE = 0.1   # metres; direction is unstable below this


class ProbeLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_query = nn.LayerNorm(dim)
        self.norm_memory = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, query, memory):
        attended, _ = self.attention(self.norm_query(query), self.norm_memory(memory), self.norm_memory(memory),
                                     need_weights=False)
        query = query + attended
        return query + self.ffn(self.norm_ffn(query))


class GeoProbe(nn.Module):
    """Each query reads the tokens on its own: no attention between objects, so
    a position can only come from the tokens and the object's own category."""

    def __init__(self, text_dim: int, token_dim: int, dim: int = 512, layers: int = 2, heads: int = 8):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, dim)
        self.memory_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, dim))
        self.layers = nn.ModuleList(ProbeLayer(dim, heads) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))

    def forward(self, text, tokens, scale):
        """text (B, O, T), tokens (B, N, D), scale (B) -> positions (B, O, 3), metres."""
        query = self.query_proj(text)
        memory = self.memory_proj(tokens)
        for layer in self.layers:
            query = layer(query, memory)
        return self.head(query) * scale[:, None, None]

    @torch.no_grad()
    def attention_maps(self, text, tokens) -> list:
        """Head-averaged cross-attention of each layer: [(B, O, N)]."""
        query = self.query_proj(text)
        memory = self.memory_proj(tokens)
        maps = []
        for layer in self.layers:
            normed = layer.norm_memory(memory)
            attended, weights = layer.attention(layer.norm_query(query), normed, normed, need_weights=True,
                                                average_attn_weights=True)
            maps.append(weights)
            query = query + attended
            query = query + layer.ffn(layer.norm_ffn(query))
        return maps


def pair_mask(target, mask):
    """Ordered pairs of masked objects far enough apart: (B, O, O)."""
    pairs = mask[:, None, :] & mask[:, :, None]
    pairs = pairs & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
    delta = target[:, None, :, :] - target[:, :, None, :]
    return pairs & (delta.norm(dim=-1) >= MIN_PAIR_DISTANCE), delta


def probe_loss(pred, target, mask) -> dict:
    position = F.smooth_l1_loss(pred[mask], target[mask])
    pairs, delta = pair_mask(target, mask)
    if not pairs.any():
        zero = pred.sum() * 0.0
        return {"loss": position + zero, "position": position, "direction": zero, "distance": zero}
    delta_hat = (pred[:, None, :, :] - pred[:, :, None, :])[pairs]
    delta = delta[pairs]
    direction = (1 - F.cosine_similarity(delta_hat, delta, dim=-1)).mean()
    # sqrt(eps) keeps the gradient finite when two predictions coincide.
    distance = F.smooth_l1_loss(torch.log1p((delta_hat.pow(2).sum(-1) + 1e-8).sqrt()), torch.log1p(delta.norm(dim=-1)))
    return {"loss": position + direction + distance, "position": position, "direction": direction,
            "distance": distance}


def azimuths(points: np.ndarray, up: np.ndarray):
    """Clockwise angle (deg, seen from above) of B for "stand at A, face C".

    points (O, 3). Returns (A, C, B) index triplets and angles. Positive is to
    the right: right = forward x up holds in any right-handed frame.
    """
    horizontal = points - (points @ up)[:, None] * up
    delta = horizontal[None, :, :] - horizontal[:, None, :]     # delta[a, b] = b - a
    triplets, angles = [], []
    count = len(points)
    for a in range(count):
        for c in range(count):
            if c == a:
                continue
            forward = delta[a, c]
            for b in range(count):
                if b in (a, c):
                    continue
                triplets.append((a, c, b))
                right = np.cross(forward, up)
                angles.append(np.degrees(np.arctan2(delta[a, b] @ right / max(np.linalg.norm(right), 1e-9),
                                                    delta[a, b] @ forward / max(np.linalg.norm(forward), 1e-9))))
    return triplets, np.array(angles)


def angular_error(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def quadrant(angle):
    return (np.abs(angle) < 90).astype(int) * 2 + (angle > 0).astype(int)


def left_right_back(angle):
    return np.where(np.abs(angle) >= 135, 2, (angle > 0).astype(int))


class SceneMetrics:
    """Values grouped by scene, so the intervals resample scenes, not objects."""

    def __init__(self):
        self.values = defaultdict(lambda: defaultdict(list))

    def add(self, scene, name, values):
        self.values[name][scene].extend(np.asarray(values, dtype=np.float64).ravel().tolist())

    def update(self, scene, pred, target, up, visible):
        """One sample's numpy arrays, for all single-instance objects and visible ones."""
        for group, keep in (("all", np.ones(len(pred), dtype=bool)), ("visible", visible)):
            p, t = pred[keep], target[keep]
            if len(t) == 0:
                continue
            self.add(scene, f"{group}/position_m", np.linalg.norm(p - t, axis=1))
            if len(t) < 2:
                continue
            distance = np.linalg.norm(t[None] - t[:, None], axis=-1)
            valid = ~np.eye(len(t), dtype=bool) & (distance >= MIN_PAIR_DISTANCE)
            predicted = np.linalg.norm(p[None] - p[:, None], axis=-1)
            self.add(scene, f"{group}/distance_m", np.abs(predicted - distance)[valid])
            if len(t) < 3:
                continue
            triplets, truth = azimuths(t, up)
            _, estimate = azimuths(p, up)
            flat = t - (t @ up)[:, None] * up
            # Standing on, or facing an object at, nearly the same spot has no direction.
            ok = np.array([min(np.linalg.norm(flat[c] - flat[a]), np.linalg.norm(flat[b] - flat[a])) >= MIN_PAIR_DISTANCE
                           for a, c, b in triplets])
            if not ok.any():
                continue
            truth, estimate = truth[ok], estimate[ok]
            self.add(scene, f"{group}/azimuth_deg", angular_error(estimate, truth))
            self.add(scene, f"{group}/quadrant_acc", quadrant(estimate) == quadrant(truth))
            self.add(scene, f"{group}/lrb_acc", left_right_back(estimate) == left_right_back(truth))

    def summary(self, resamples: int = 200, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        out = {}
        for name, per_scene in sorted(self.values.items()):
            arrays = [np.array(v) for v in per_scene.values()]
            statistic = np.mean if name.endswith("_acc") else np.median
            draws = [statistic(np.concatenate([arrays[i] for i in rng.integers(len(arrays), size=len(arrays))]))
                     for _ in range(resamples)]
            out[name] = {"value": float(statistic(np.concatenate(arrays))),
                         "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                         "scenes": len(arrays), "n": int(sum(map(len, arrays)))}
        return out
