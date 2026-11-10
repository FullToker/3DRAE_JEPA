#!/usr/bin/env python3
"""Add same-question wrong answers to a mix manifest and hold out scenes.

Every row gets "negatives" (jepa3d/hard_negatives.py; [] where none can be
built). Scenes with crc32("<subset>/<scene>") % --holdout-mod == 0 leave the
training set entirely (VSI and SPAR rows alike); their rows that have
negatives become the holdout, shuffled and capped per question type.

Writes <output-train>, <output-holdout> and <output-train>.stats.json.

Usage:
    python jepa3d/tools/add_hard_negatives.py --input data/mix_vsi_spar_train_f8.jsonl \
        --output-train experiments/hardneg_v1/data/train_f8.jsonl \
        --output-holdout experiments/hardneg_v1/data/holdout_f8.jsonl
"""

import argparse
import json
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.hard_negatives import negatives_for
from jepa3d.vsi_text import Renderer, build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-holdout", required=True)
    parser.add_argument("--holdout-mod", type=int, default=20, help="1 scene in N is held out")
    parser.add_argument("--holdout-max-per-type", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The object vocabulary must cover the whole file, as when the targets were rendered.
    with open(args.input) as f:
        vocab_rows = [{k: r.get(k) for k in ("question_type", "query", "options")}
                      for r in map(json.loads, f) if r["source"] == "vsi590k"]
    renderer = Renderer(build_vocab(vocab_rows))
    del vocab_rows

    types = defaultdict(Counter)
    scenes = {"train": set(), "holdout": set()}
    holdout = defaultdict(list)
    train_rows = 0
    for path in (args.output_train, args.output_holdout):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as f, open(args.output_train, "w") as out:
        for line in f:
            row = json.loads(line)
            row["negatives"], status = negatives_for(row, renderer)
            scene = f"{row['subset']}/{row['scene']}"
            split = "holdout" if zlib.crc32(scene.encode()) % args.holdout_mod == 0 else "train"
            scenes[split].add(scene)
            stats = types[f"{row['source']}/{row['question_type']}"]
            stats[f"{split}_rows"] += 1
            stats[f"{split}_{status}"] += 1
            stats[f"{split}_negatives"] += len(row["negatives"])
            if split == "train":
                out.write(json.dumps(row) + "\n")
                train_rows += 1
            elif row["negatives"]:
                holdout[(row["source"], row["question_type"])].append(row)

    rng = random.Random(args.seed)
    kept = []
    for key in sorted(holdout):
        rows = holdout[key]
        kept += rng.sample(rows, args.holdout_max_per_type) if len(rows) > args.holdout_max_per_type else rows
    rng.shuffle(kept)   # eval rolls scene tokens within a batch: neighbours must be other scenes
    with open(args.output_holdout, "w") as out:
        for row in kept:
            out.write(json.dumps(row) + "\n")

    summary = {
        "input": args.input,
        "train_rows": train_rows,
        "train_rows_with_negatives": sum(t["train_ok"] for t in types.values()),
        "holdout_rows": len(kept),
        "holdout_rows_by_type": Counter(f"{r['source']}/{r['question_type']}" for r in kept),
        "train_scenes": len(scenes["train"]),
        "holdout_scenes": len(scenes["holdout"]),
        "by_type": {k: dict(sorted(v.items())) for k, v in sorted(types.items())},
    }
    with open(args.output_train + ".stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
