#!/usr/bin/env python3
"""How much of a checkpoint's score comes from the images?

Same checkpoint, three visual inputs: "real", "shuffled" (scene tokens rolled
by one row in the batch, so each query sees another row's scene) and "zero"
(all-zero tokens). Reports held-out InfoNCE, overall and per question type,
on a random sample of the val manifest (masked by answer_key as in training),
and candidate accuracy on the eval manifests. If "shuffled" scores match
"real", the score is coming from the query text alone.

Usage (from the repo root):
    python jepa3d/tools/blind_check.py --config <yaml> --checkpoint <pt> \
        --val data/spar_video_val_f8.jsonl \
        --eval data/vsibench_test_posed_f8.jsonl data/vsibench_debiased_posed_f8.jsonl \
        --output <metrics.json>
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from fusion_gv.gvjepa_trainer import group_ids_tensor
from fusion_gv.infer_gvjepa import _candidates, _load_checkpoint, _score_candidate_batch, _target
from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config
from jepa3d.train import load_config

MODES = ("real", "shuffled", "zero")


def set_mode(model, mode: str) -> None:
    """Replace the X-encoder's tokens on this model instance."""
    original = type(model)._pool_visual

    def pool(images, cams, batch_size):
        tokens, frames = original(model, images, cams, batch_size)
        if mode == "shuffled":
            tokens = tokens.roll(1, 0)
        elif mode == "zero":
            tokens = torch.zeros_like(tokens)
        return tokens, tokens.unsqueeze(1)

    model._pool_visual = pool


def val_loss(model, loader, device, temperature: float) -> dict:
    """Per-row InfoNCE (query -> target direction and back), by question type."""
    stats = defaultdict(list)
    for batch in loader:
        if len(batch["query"]) < 2:
            continue
        images = batch["images_vggt"].to(device, non_blocking=True)
        cams = batch["images_jepa"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(images, cams, batch["query"], batch["target"])
        pred = F.normalize(out["pred"].float(), dim=-1)
        target = F.normalize(out["target"].float(), dim=-1)
        logits = pred @ target.T / temperature
        ids = group_ids_tensor(batch.get("group_id"), device)
        if ids is not None:
            same = ids[:, None] == ids[None, :]
            same = same & ~torch.eye(same.shape[0], dtype=torch.bool, device=device)
            logits = logits.masked_fill(same, float("-inf"))
        labels = torch.arange(pred.shape[0], device=device)
        per_row = 0.5 * (F.cross_entropy(logits, labels, reduction="none")
                         + F.cross_entropy(logits.T, labels, reduction="none"))
        for row, loss in zip(batch["rows"], per_row.tolist()):
            stats["all"].append(loss)
            stats[f"type/{row.get('question_type')}"].append(loss)
    return {k: sum(v) / len(v) for k, v in sorted(stats.items())}


def accuracy(model, loader, device) -> dict:
    stats = defaultdict(lambda: [0, 0])
    for batch in loader:
        images = batch["images_vggt"].to(device, non_blocking=True)
        cams = batch["images_jepa"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.predict_embedding(images, cams, batch["query"])
        candidates = [_candidates(row) for row in batch["rows"]]
        scores = _score_candidate_batch(model, pred, candidates, device, include_scores=False)
        for row, cands, score in zip(batch["rows"], candidates, scores):
            target_text, _ = _target(row, cands)
            if score is None or target_text is None:
                continue
            correct = cands[score[0]] == target_text
            for key in ("all", f"type/{row.get('type')}"):
                stats[key][0] += int(correct)
                stats[key][1] += 1
    return {k: c / n for k, (c, n) in sorted(stats.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--eval", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-rows", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = args.num_workers
    device = torch.device("cuda")
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.val) as f:
        rows = f.readlines()
    random.Random(0).shuffle(rows)   # the file is ordered by scene; rows from all scenes in each batch
    sample = output.with_suffix(".val_sample.jsonl")
    sample.write_text("".join(rows[:args.val_rows]))

    results = {"checkpoint": args.checkpoint, "step": step, "val_rows": args.val_rows}
    temperature = cfg["train"].get("temperature", 0.07)
    for mode in MODES:
        set_mode(model, mode)
        with torch.no_grad():
            res = {"val_infonce": val_loss(model, build_loader([str(sample)], cfg, args.batch_size, train=False),
                                           device, temperature)}
            for path in args.eval:
                loader = build_loader([path], cfg, args.batch_size, train=False)
                res[Path(path).stem] = accuracy(model, loader, device)
        del model._pool_visual
        results[mode] = res
        print(mode, json.dumps(res), flush=True)
        output.write_text(json.dumps(results, indent=2))
    sample.unlink()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
