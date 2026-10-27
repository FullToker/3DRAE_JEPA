#!/usr/bin/env python3
"""Multiple-choice SPAR evaluation for the 3DRAE X-encoder model.

Scores each row's candidates by cosine similarity between the predicted
embedding and the y-encoder embedding of every candidate (same rule as
fusion_gv/infer_gvjepa.py --mode select), then reports accuracy overall,
per question type and per subset.

Usage (from the repo root):
    python jepa3d/infer_spar.py --config <yaml> --input <posed eval jsonl> \
        --output-dir <dir> [--checkpoint <pt>]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from jepa3d.train import load_config
from fusion_gv.infer_gvjepa import (
    _candidates,
    _latest_checkpoint,
    _load_checkpoint,
    _score_candidate_batch,
    _target,
)
from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", help="default: latest checkpoint in train.output_dir")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="debug: evaluate only the first N rows")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = args.num_workers
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(cfg["train"]["output_dir"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, checkpoint)
    model.to(device).eval()
    loader = build_loader([args.input], cfg, args.batch_size, train=False, limit=args.limit)

    stats = defaultdict(lambda: [0, 0])   # key -> [correct, evaluated]
    with torch.no_grad(), open(out_dir / "predictions.jsonl", "w") as out_f:
        for batch in tqdm(loader, desc="spar3d-infer"):
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
                pred_text = cands[score[0]]
                correct = pred_text == target_text
                for key in ("all", f"type/{row.get('type')}", f"subset/{row.get('subset')}"):
                    stats[key][0] += int(correct)
                    stats[key][1] += 1
                out_f.write(json.dumps({
                    "id": row.get("id"), "type": row.get("type"), "subset": row.get("subset"),
                    "pred": pred_text, "target": target_text, "correct": correct,
                }) + "\n")

    metrics = {
        "checkpoint": str(checkpoint),
        "step": step,
        "input": args.input,
        "accuracy": {k: c / n for k, (c, n) in sorted(stats.items())},
        "evaluated": {k: n for k, (_, n) in sorted(stats.items())},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"evaluated={stats['all'][1]} accuracy={metrics['accuracy'].get('all', float('nan')):.4f}")
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
