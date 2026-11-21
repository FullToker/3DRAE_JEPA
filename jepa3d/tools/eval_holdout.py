#!/usr/bin/env python3
"""Multiple-choice accuracy on the scene holdout, with real / shuffled / zero scene tokens.

Holdout rows come from jepa3d/tools/add_hard_negatives.py. Each row is scored
in two answer formats:

    long   target sentence vs its "negatives" (same question, wrong answer)
    short  "target_short" vs the row's "options" (VSI-590K multiple choice only)

and with three visual inputs (jepa3d/tools/blind_check.py): "real", "shuffled"
(each query sees another row's scene) and "zero". The gap real - shuffled is
what the images contribute; "chance" is the mean of 1 / candidates.
--eval manifests (VSI-Bench) are scored as in blind_check.py. A model with
object slots is also scored with swapped slots, without slots and with
nothing of the row's own images (blind_check.SLOT_MODES).

Usage (from the repo root):
    python jepa3d/tools/eval_holdout.py --config <yaml> --checkpoint <pt> \
        --holdout experiments/hardneg_v1/data/holdout_f8.jsonl \
        --eval data/vsibench_test_posed_f8.jsonl --output <metrics.json>
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
from fusion_gv.infer_gvjepa import _load_checkpoint, _score_candidate_batch
from jepa3d.balance import KINDS
from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config
from jepa3d.tools.blind_check import accuracy, modes_of, set_mode, slots_of
from jepa3d.train import load_config
from jepa3d.trainer import mean_relative_accuracy


def candidates_of(row: dict) -> dict:
    """{format: (candidates in a fixed random order, answer)}."""
    out = {"long": ([row["target"], *row["negatives"]], row["target"])}
    if row.get("options") and row.get("target_short") in row["options"]:
        out["short"] = (list(dict.fromkeys(row["options"])), row["target_short"])
    rng = random.Random(f"{row['scene']}:{row['query']}")
    for cands, _ in out.values():   # the answer must not sit at a fixed index: argmax ties go to index 0
        rng.shuffle(cands)
    return out


def numeric_accuracy(model, loader, device, medians: dict) -> dict:
    """Rows with a numeric answer (jepa3d.balance.numeric_target), read from the model's numeric head:
    {type: {"mra", "median_mra" (always answering the training median of the kind), "rows"}}."""
    stats = defaultdict(lambda: [0.0, 0.0, 0])
    for batch in loader:
        images = batch["images_vggt"].to(device, non_blocking=True)
        cams = batch["images_jepa"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.predict_embedding(images, cams, batch["query"], slots=slots_of(model, batch))
        kind, value = batch["numeric"]["kind"].to(device), batch["numeric"]["value"].to(device)
        has = kind >= 0
        if not has.any():
            continue
        guess = model.numeric_output.float()[has, kind[has]].exp()
        base = torch.tensor([medians[KINDS[k]] for k in kind[has].tolist()], device=device)
        scores = mean_relative_accuracy(guess, value[has]).tolist()
        base_scores = mean_relative_accuracy(base, value[has]).tolist()
        for row, score, base_score in zip([r for r, h in zip(batch["rows"], has.tolist()) if h], scores, base_scores):
            for key in ("all", f"type/{row.get('type') or row.get('question_type')}"):
                stats[key][0] += score
                stats[key][1] += base_score
                stats[key][2] += 1
    return {k: {"mra": a / n, "median_mra": b / n, "rows": n} for k, (a, b, n) in sorted(stats.items())}


def holdout_accuracy(model, loader, device, slot_log=None) -> dict:
    """slot_log: a list that receives (acc, error_m) of the object tokens of every batch that has them."""
    stats = defaultdict(lambda: [0, 0, 0.0])   # correct, rows, sum of 1 / candidates
    for batch in loader:
        images = batch["images_vggt"].to(device, non_blocking=True)
        cams = batch["images_jepa"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.predict_embedding(images, cams, batch["query"], slots=slots_of(model, batch))
        if slot_log is not None and getattr(model, "slot_metrics", None):
            slot_log.append((float(model.slot_metrics["acc"]), float(model.slot_metrics["error_m"])))
        per_row = [candidates_of(row) for row in batch["rows"]]
        for fmt in ("long", "short"):
            cands = [c[fmt][0] if fmt in c else [] for c in per_row]
            scores = _score_candidate_batch(model, pred, cands, device, include_scores=False)
            for row, c, score in zip(batch["rows"], per_row, scores):
                if score is None:
                    continue
                options, answer = c[fmt]
                for key in (f"{fmt}/all", f"{fmt}/{row['source']}/{row['question_type']}"):
                    stats[key][0] += int(options[score[0]] == answer)
                    stats[key][1] += 1
                    stats[key][2] += 1 / len(options)
    return {k: {"acc": c / n, "chance": p / n, "rows": n} for k, (c, n, p) in sorted(stats.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout", help="omit to score only the --eval manifests")
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--numeric", nargs="*", default=[],
                        help="manifests of numeric questions, for models with a numeric head")
    parser.add_argument("--answer-weights", help="table of jepa3d/balance.py: its medians are the numeric baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N rows of each manifest")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = args.num_workers
    device = torch.device("cuda")
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    medians = {}
    if args.answer_weights:
        with open(args.answer_weights) as f:
            medians = json.load(f)["numeric_median"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {"checkpoint": args.checkpoint, "step": step, "holdout": args.holdout}
    for mode in modes_of(model):
        set_mode(model, mode)
        with torch.no_grad():
            slot_log, res = [], {}
            if args.holdout:
                loader = build_loader([args.holdout], cfg, args.batch_size, train=False, limit=args.limit)
                res["holdout"] = holdout_accuracy(model, loader, device, slot_log)
            if slot_log:   # object tokens: does the output at the token find the position (mean over batches)
                res["object_tokens"] = {"acc": sum(a for a, _ in slot_log) / len(slot_log),
                                        "error_m": sum(e for _, e in slot_log) / len(slot_log)}
            for path in args.eval:
                loader = build_loader([path], cfg, args.batch_size, train=False, limit=args.limit)
                res[Path(path).stem] = accuracy(model, loader, device)
            for path in args.numeric if model.numeric_head is not None else []:
                loader = build_loader([path], cfg, args.batch_size, train=False, limit=args.limit)
                res[Path(path).stem] = numeric_accuracy(model, loader, device, medians)
        del model._pool_visual
        results[mode] = res
        print(mode, json.dumps(res), flush=True)
        output.write_text(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
