#!/usr/bin/env python3
"""Reverse probe: given where an object is, can its category be read from frozen tokens?

The position probe found objects badly from their names. This asks the opposite:
is the category there at all once the place is given? Target: the category (one of
the train categories, cross-entropy) of each single-instance object whose box
centre projects into a view.

  --readout attention   same cross-attention probe as the position probe. The query
                        is only the Plucker ray of the centre's pixel (Fourier
                        features), with no image feature, so the category must come
                        from the tokens. Works for every memory, including 3DRAE's
                        tokens, which have no image position.
  --readout index       DINOv2 / VGGT only: an MLP on the one patch token that holds
                        the centre. No search, so it measures what the token holds.

Usage (from the repo root), see README.md:
    python jepa3d/geo_probe/train_reverse.py --train-labels data/geo_probe/labels_train.jsonl \
        --val-labels data/geo_probe/labels_val.jsonl --stats data/frame_stats/scannet.jsonl \
        --output outputs/geo_probe/reverse_3drae_real --memory 3drae
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from jepa3d.geo_probe.data import GeoProbeDataset, collate
from jepa3d.geo_probe.dino_memory import dino_patches
from jepa3d.geo_probe.probe import GeoProbe
from jepa3d.geo_probe.train import build_patch_memory, point_queries, scene_tokens
from jepa3d.geo_probe.vggt_memory import VGGT_DIM, VGGT_PATCH, load_vggt
from jepa3d.x_encoder import ThreeDRAEXEncoder

RAY_BANDS = 8
RAY_DIM = 6 * (1 + 2 * RAY_BANDS)


def ray_queries(encoder, batch, device):
    """Fourier features of the centre pixel's Plucker ray: (B, O, RAY_DIM), valid (B, O)."""
    queries, valid = point_queries(encoder.rae, batch, device)
    ray = queries[..., -6:]
    angles = ray[..., None] * (2.0 ** torch.arange(RAY_BANDS, device=device)) * math.pi
    out = torch.cat([ray, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)
    return out * valid[..., None], valid


class IndexHead(nn.Module):
    """MLP on the patch token that holds the object's projected centre."""

    def __init__(self, feat_dim: int, classes: int, dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim), nn.GELU(), nn.Linear(dim, classes))

    def forward(self, feats, prompt, patch: int):
        """feats (B, V, h, w, C), prompt (B, O, 3) [view, u, v] -> logits (B, O, classes)."""
        view = prompt[..., 0].clamp(min=0).long()
        h, w = feats.shape[2:4]
        row = (prompt[..., 2].clamp(min=0) // patch).long().clamp(max=h - 1)
        col = (prompt[..., 1].clamp(min=0) // patch).long().clamp(max=w - 1)
        b = torch.arange(len(view), device=view.device)[:, None].expand_as(view)
        return self.net(feats[b, view, row, col])


class Reader:
    """Logits for a batch, by attention over a memory or from the indexed patch."""

    def __init__(self, args, classes, device):
        self.args, self.device = args, device
        self.encoder = ThreeDRAEXEncoder().to(device)
        self.memory = self.backbone = None
        if args.readout == "index":
            if args.memory == "vggt":
                self.backbone, feat_dim = load_vggt(device), VGGT_DIM
            else:
                feat_dim = self.encoder.rae.encoder.hidden_size
            self.probe = IndexHead(feat_dim, classes).to(device)
            self.modules = [self.probe]
            self.params = list(self.probe.parameters())
        else:
            self.memory = build_patch_memory(args.memory, self.encoder, device)
            token_dim = self.memory.module.feat_dim if self.memory else self.encoder.dim
            self.probe = GeoProbe(RAY_DIM, token_dim, out_dim=classes).to(device)
            self.modules = [self.probe, self.memory.module if self.memory else self.encoder]
            memory_params = (list(self.memory.module.parameters()) if self.memory
                             else [p for p in self.encoder.parameters() if p.requires_grad])
            self.params = memory_params + list(self.probe.parameters())

    def train(self, mode=True):
        for module in self.modules:
            module.train(mode)

    def state(self):
        trainable = (self.memory.module.state_dict() if self.memory else
                     {k: v for k, v in self.encoder.state_dict().items() if not k.startswith("rae.")})
        return {"probe": self.probe.state_dict(), "encoder_trainable": {} if self.args.readout == "index" else trainable}

    def __call__(self, batch, shuffled=False, previous=None):
        """-> logits (B, O, classes), valid (B, O), tokens (for the next call's previous)."""
        device = self.device
        if self.args.readout == "index":
            images = batch["images"].to(device, non_blocking=True)
            prompt = batch["prompt"].to(device)
            if self.backbone is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = self.backbone(images)[-1].float()
                feats = feats.unflatten(2, (images.shape[-2] // VGGT_PATCH, images.shape[-1] // VGGT_PATCH))
                patch = VGGT_PATCH
            else:
                feats, patch = dino_patches(self.encoder.rae, images), self.encoder.rae.encoder.patch_size
            return self.probe(feats, prompt, patch), prompt[..., 0] >= 0, None
        queries, valid = ray_queries(self.encoder, batch, device)
        tokens = scene_tokens(self.encoder, batch, device, previous, shuffled, self.memory)
        return self.probe(queries, tokens), valid, tokens.detach()


def label_batch(categories, index, device):
    """(B, O) class index, -1 for padding and categories absent from train."""
    labels = torch.full((len(categories), max(map(len, categories))), -1, dtype=torch.long)
    for i, names in enumerate(categories):
        for j, name in enumerate(names):
            labels[i, j] = index.get(name, -1)
    return labels.to(device)


@torch.no_grad()
def evaluate(reader, loader, classes, index, freq, shuffled=False):
    """Summary and one record per queried object."""
    reader.train(False)
    records, previous = [], None
    for batch in loader:
        logits, valid, previous = reader(batch, shuffled, previous)
        top5 = logits.float().topk(5, dim=-1).indices.cpu().numpy()
        valid = (valid.cpu() & batch["mask"]).numpy()
        for i, scene in enumerate(batch["scene"]):
            for j, name in enumerate(batch["categories"][i]):
                if valid[i, j]:
                    records.append({"scene": scene, "category": name, "train_freq": freq[name],
                                    "visible": bool(batch["visible"][i, j]),
                                    "top5": [classes[k] for k in top5[i, j]]})
    known = [r for r in records if r["category"] in index]
    per_scene = defaultdict(list)
    for r in known:
        per_scene[r["scene"]].append((r["top5"][0] == r["category"], r["category"] in r["top5"]))
    arrays = [np.array(v, dtype=np.float64) for v in per_scene.values()]
    stacked = np.concatenate(arrays)
    rng = np.random.default_rng(0)
    draws = np.array([np.concatenate([arrays[k] for k in rng.integers(len(arrays), size=len(arrays))]).mean(0)
                      for _ in range(200)])
    per_class = defaultdict(list)
    for r in known:
        per_class[r["category"]].append(r["top5"][0] == r["category"])
    summary = {"n": len(known), "n_unseen_category": len(records) - len(known), "scenes": len(arrays),
               "macro_top1": float(np.mean([np.mean(v) for v in per_class.values()])),
               "val_classes": len(per_class)}
    for k, name in enumerate(("top1", "top5")):
        summary[name] = {"value": float(stacked[:, k].mean()),
                         "ci95": [float(np.percentile(draws[:, k], 2.5)), float(np.percentile(draws[:, k], 97.5))]}
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory", choices=("3drae", "dino", "vggt"), default="3drae")
    parser.add_argument("--readout", choices=("attention", "index"), default="attention")
    parser.add_argument("--tokens", choices=("real", "shuffled"), default="real",
                        help="shuffled: tokens of another scene, the prior from the ray alone")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="use only this many scenes of each split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.readout == "index" and (args.memory == "3drae" or args.tokens == "shuffled"):
        parser.error("--readout index needs --memory dino or vggt and real tokens")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_set = GeoProbeDataset(args.train_labels, args.stats, limit=args.limit)
    val_set = GeoProbeDataset(args.val_labels, args.stats, limit=args.limit, seeds=list(range(args.val_seeds)))
    train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True, drop_last=True,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)
    freq = Counter(o["category"] for s in train_set.scenes for o in s["objects"])
    classes = sorted(freq)
    index = {name: k for k, name in enumerate(classes)}

    reader = Reader(args, len(classes), device)
    optimizer = torch.optim.AdamW(reader.params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    print(f"{len(classes)} classes, {sum(p.numel() for p in reader.params):,} trainable parameters, {total} steps",
          flush=True)

    shuffled = args.tokens == "shuffled"
    log = open(output / "metrics.jsonl", "a")
    step = 0
    for epoch in range(1, args.epochs + 1):
        reader.train(True)
        losses, hits = [], []
        for batch in train_loader:
            labels = label_batch(batch["categories"], index, device)
            logits, valid, _ = reader(batch, shuffled)
            mask = valid & batch["mask"].to(device) & (labels >= 0)
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask].float(), labels[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(loss.item())
            hits.append((logits[mask].argmax(-1) == labels[mask]).float().mean().item())
        record = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "train_top1": float(np.mean(hits))}
        if epoch % args.eval_every == 0 and epoch != args.epochs:
            record["val"] = evaluate(reader, val_loader, classes, index, freq, shuffled)[0]
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    summary, records = evaluate(reader, val_loader, classes, index, freq)
    known = [r for r in records if r["category"] in index]
    majority = max(freq, key=freq.get)
    final = {"memory": args.memory, "readout": args.readout, "tokens_trained_on": args.tokens, "classes": len(classes),
             "majority_class": majority, "majority_top1": float(np.mean([r["category"] == majority for r in known])),
             "eval_real": summary}
    if args.readout == "attention":
        final["eval_shuffled"] = evaluate(reader, val_loader, classes, index, freq, shuffled=True)[0]
    (output / "final_metrics.json").write_text(json.dumps(final, indent=2))
    with open(output / "objects.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    torch.save({**reader.state(), "classes": classes, "args": vars(args)}, output / "final.pt")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
