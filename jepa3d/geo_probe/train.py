#!/usr/bin/env python3
"""Train the geometry probe on frozen 3DRAE tokens and evaluate on unseen scenes.

Usage (from the repo root), see README.md:
    python jepa3d/geo_probe/train.py --train-labels data/geo_probe/labels_train.jsonl \
        --val-labels data/geo_probe/labels_val.jsonl --stats data/frame_stats/scannet.jsonl \
        --output outputs/geo_probe/real --tokens real
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from jepa3d.paths import VJEPA_ROOT
from jepa3d.geo_probe.data import GeoProbeDataset, collate
from jepa3d.geo_probe.dino_memory import DinoPatchMemory
from jepa3d.geo_probe.probe import GeoProbe, SceneMetrics, probe_loss
from jepa3d.x_encoder import ThreeDRAEXEncoder

TEXT_MODEL = "google/embeddinggemma-300m"


@torch.no_grad()
def embed_categories(categories, device) -> dict:
    """Mean-pooled, frozen text embedding per category name."""
    from transformers import AutoModel, AutoTokenizer

    cache = str(VJEPA_ROOT / "ckpts")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, cache_dir=cache)
    model = AutoModel.from_pretrained(TEXT_MODEL, cache_dir=cache).to(device).eval()
    names = sorted(categories)
    out = {}
    for start in range(0, len(names), 64):
        chunk = names[start:start + 64]
        tok = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        hidden = model(**tok).last_hidden_state
        weight = tok["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * weight).sum(1) / weight.sum(1)
        out.update(zip(chunk, pooled.float()))
    return out


def text_batch(categories, embeddings, device):
    dim = next(iter(embeddings.values())).shape[-1]
    text = torch.zeros(len(categories), max(map(len, categories)), dim, device=device)
    for i, names in enumerate(categories):
        for j, name in enumerate(names):
            text[i, j] = embeddings[name]
    return text


def scene_tokens(encoder, batch, device, previous=None, shuffled=False, dino=None):
    """Tokens of the batch; shuffled = each sample gets another sample's tokens.
    previous covers a final batch of one, which has nobody to swap with.
    dino: a DinoPatchMemory, to read DINOv2 patches instead of 3DRAE's tokens."""
    images = batch["images"].to(device, non_blocking=True)
    cams = batch["cams"].to(device, non_blocking=True)
    if dino is not None:
        tokens = dino(encoder.rae, images, cams)
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder(images, cams)
    tokens = tokens.float()
    if not shuffled:
        return tokens
    if tokens.shape[0] == 1 and previous is not None:
        return previous[:1]
    return tokens.roll(1, 0)


@torch.no_grad()
def evaluate(encoder, probe, loader, embeddings, device, shuffled: bool, dino=None) -> dict:
    encoder.eval()
    probe.eval()
    if dino is not None:
        dino.eval()
    metrics = SceneMetrics()
    previous = None
    for batch in loader:
        tokens = scene_tokens(encoder, batch, device, previous, shuffled, dino)
        previous = tokens.detach()
        text = text_batch(batch["categories"], embeddings, device)
        pred = probe(text, tokens, batch["scale"].to(device)).cpu().numpy()
        target = batch["positions"].numpy()
        for i, scene in enumerate(batch["scene"]):
            keep = batch["mask"][i].numpy()
            metrics.update(scene, pred[i][keep], target[i][keep], batch["up"][i].numpy(),
                           batch["visible"][i].numpy()[keep])
    return metrics.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", choices=("real", "shuffled"), default="real")
    parser.add_argument("--memory", choices=("3drae", "dino"), default="3drae",
                        help="3drae: neck tokens + view tokens; dino: DINOv2 patches + rays, no neck")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="use only this many scenes of each split")
    parser.add_argument("--fixed-views", action="store_true", help="one fixed view subset per scene (overfit test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_set = GeoProbeDataset(args.train_labels, args.stats, limit=args.limit,
                                seeds=[0] if args.fixed_views else None)
    val_set = GeoProbeDataset(args.val_labels, args.stats, limit=args.limit, seeds=list(range(args.val_seeds)))
    train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True, drop_last=True,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)

    categories = {o["category"] for s in train_set.scenes + val_set.scenes for o in s["objects"]}
    embeddings = embed_categories(categories, device)
    encoder = ThreeDRAEXEncoder().to(device)
    text_dim = next(iter(embeddings.values())).shape[-1]
    probe = GeoProbe(text_dim, encoder.dim).to(device)

    dino = DinoPatchMemory(encoder.rae.encoder.hidden_size).to(device) if args.memory == "dino" else None
    memory_params = list(dino.parameters()) if dino else [p for p in encoder.parameters() if p.requires_grad]
    params = memory_params + list(probe.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    print(f"{sum(p.numel() for p in params):,} trainable parameters, {total} steps", flush=True)

    log = open(output / "metrics.jsonl", "a")
    step = 0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        probe.train()
        if dino is not None:
            dino.train()
        running = {}
        for batch in train_loader:
            mask = (batch["mask"] & batch["visible"]).to(device)
            target = batch["positions"].to(device, non_blocking=True)
            if mask.sum() == 0 or mask.shape[0] < 2:
                continue
            tokens = scene_tokens(encoder, batch, device, shuffled=args.tokens == "shuffled", dino=dino)
            pred = probe(text_batch(batch["categories"], embeddings, device), tokens,
                         batch["scale"].to(device, non_blocking=True))
            losses = probe_loss(pred, target, mask)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            for key, value in losses.items():
                running.setdefault(key, []).append(value.item())
        record = {"epoch": epoch, "step": step, **{k: sum(v) / len(v) for k, v in running.items()}}
        if epoch % args.eval_every == 0 and epoch != args.epochs:
            record["val"] = evaluate(encoder, probe, val_loader, embeddings, device, args.tokens == "shuffled", dino)
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    final = {
        "tokens_trained_on": args.tokens,
        "memory": args.memory,
        "eval_real": evaluate(encoder, probe, val_loader, embeddings, device, shuffled=False, dino=dino),
        "eval_shuffled": evaluate(encoder, probe, val_loader, embeddings, device, shuffled=True, dino=dino),
    }
    (output / "final_metrics.json").write_text(json.dumps(final, indent=2))
    trainable = dino.state_dict() if dino else {k: v for k, v in encoder.state_dict().items()
                                                 if not k.startswith("rae.")}
    torch.save({"probe": probe.state_dict(), "encoder_trainable": trainable, "args": vars(args)}, output / "final.pt")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
