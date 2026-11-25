#!/usr/bin/env python3
"""E1 check: are the positions the model is handed enough to answer, without the predictor?

Arm B (jepa3d/configs/e1_oracle_b.yaml) stayed at the answer prior. Before blaming the predictor or the
sentence-embedding objective, this script takes the very batches arm B trains on (same loader, same
"slots": three positions per question, first view's frame, divided by the scene scale) and trains a small
MLP on them: positions of (stand, face, target) + the number of options -> the answer, cross-entropy.

    MLP reaches ~95%   the positions arrive intact and suffice; arm B's failure is the predictor's or the
                       objective's
    MLP stays at prior the positions are wrong or not enough (e.g. the up axis is unknown in the first
                       view's frame); arm B could not have worked

"permuted" is the same control as slots_permuted: every role gets the next role's position.

Usage (from the repo root):
    python jepa3d/tools/oracle_probe.py --config jepa3d/configs/e1_oracle_b.yaml --output <json>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa3d.data import build_loader
from jepa3d.slots import fourier
from jepa3d.train import load_config

ANSWERS = ("left", "right", "back", "front-left", "front-right", "back-left", "back-right")
OPTION_COUNTS = (2, 3, 4)


def collect(loader, rows_wanted: int):
    """-> positions (N, 3, 3) in the order of the rows' "ground" entries (stand, face, target),
    labels (N,), allowed (N, 7) bool, source list."""
    positions, labels, allowed, sources = [], [], [], []
    for batch in loader:
        slots = batch["slots"]
        if slots is None:
            continue
        for i, row in enumerate(batch["rows"]):
            mine = (slots["row"] == i).nonzero().flatten()
            options = list(dict.fromkeys(row.get("options") or []))
            if len(mine) != 3 or row["target_short"] not in ANSWERS or any(o not in ANSWERS for o in options):
                continue
            positions.append(slots["pos"][mine])
            labels.append(ANSWERS.index(row["target_short"]))
            allowed.append(torch.tensor([a in options for a in ANSWERS]))
            sources.append(row.get("source", "?"))
        if len(labels) >= rows_wanted:
            break
    return torch.stack(positions), torch.tensor(labels), torch.stack(allowed), sources


def features(positions: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    flat = positions.flatten(1)
    count = F.one_hot(allowed.sum(1) - 2, len(OPTION_COUNTS)).float()
    return torch.cat([flat, fourier(4 * flat, 8), count], dim=-1)


def logits_of(model, positions, allowed):
    return model(features(positions, allowed)).masked_fill(~allowed, float("-inf"))


@torch.no_grad()
def accuracy(model, positions, labels, allowed, sources) -> dict:
    hit = (logits_of(model, positions, allowed).argmax(1) == labels).float().cpu()
    out = {"all": round(hit.mean().item(), 4), "rows": len(hit)}
    for source in sorted(set(sources)):
        keep = torch.tensor([s == source for s in sources])
        out[source] = round(hit[keep].mean().item(), 4)
    for count in OPTION_COUNTS:
        keep = (allowed.sum(1) == count).cpu()
        if keep.any():
            out[f"options_{count}"] = round(hit[keep].mean().item(), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-rows", type=int, default=120000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["data"]["num_workers"], cfg["data"]["persistent_workers"] = 8, False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = collect(build_loader(cfg["data"]["train_manifests"], cfg, 256, train=True), args.train_rows)
    held = collect(build_loader(cfg["data"]["val_manifests"], cfg, 256, train=False), 10 ** 9)
    x, y, ok = (t.to(device) for t in train[:3])
    hx, hy, hok = (t.to(device) for t in held[:3])
    prior = {str(c): torch.bincount(y[ok.sum(1) == c], minlength=len(ANSWERS)).float().div(
        max(int((ok.sum(1) == c).sum()), 1)).max().item() for c in OPTION_COUNTS}
    stats = {"train_rows": len(y), "holdout_rows": len(hy), "pos_mean_abs": x.abs().mean().item(),
             "pos_std": x.std().item(), "pos_max_abs": x.abs().max().item(), "majority_share_by_options": prior}
    print(json.dumps(stats), flush=True)

    torch.manual_seed(0)
    width = features(x[:2], ok[:2]).shape[1]
    model = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                          nn.Linear(512, 512), nn.GELU(), nn.Linear(512, len(ANSWERS))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    curve = []
    for step in range(1, args.steps + 1):
        pick = torch.randint(len(y), (args.batch_size,), device=device)
        loss = F.cross_entropy(logits_of(model, x[pick], ok[pick]), y[pick])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 2000 == 0 or step == args.steps:
            point = {"step": step, "loss": round(loss.item(), 4),
                     "holdout": accuracy(model, hx, hy, hok, held[3])["all"]}
            curve.append(point)
            print(json.dumps(point), flush=True)

    result = {**stats, "curve": curve,
              "train": accuracy(model, x[:20000], y[:20000], ok[:20000], train[3][:20000]),
              "holdout": accuracy(model, hx, hy, hok, held[3]),
              "holdout_permuted": accuracy(model, hx.roll(1, dims=1), hy, hok, held[3])}
    print(json.dumps(result, indent=1), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
