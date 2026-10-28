#!/usr/bin/env python3
"""Concatenate posed manifests, optionally cap rows per question type, shuffle.

Writes <output> and <output>.stats.json (rows by source / subset / type,
scenes, answer_key repetition per type).

Usage:
    python jepa3d/tools/mix_manifests.py --inputs data/vsi590k_sn_f8.jsonl \
        data/spar_video_train_f8.jsonl --output data/mix_vsi_spar_train_f8.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-type", type=int, default=0, help="0 = keep all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    for path in args.inputs:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                by_type[(row["source"], row["question_type"])].append(
                    (line, row["subset"], row["scene"], row["answer_key"]))

    lines, stats = [], {"by_type": {}, "by_source": Counter(), "by_subset": Counter(), "scenes": set()}
    for key, rows in sorted(by_type.items()):
        if args.max_per_type and len(rows) > args.max_per_type:
            rows = rng.sample(rows, args.max_per_type)
        keys = Counter(r[3] for r in rows)
        stats["by_type"]["/".join(key)] = {
            "rows": len(rows),
            "scenes": len({(r[1], r[2]) for r in rows}),
            "unique_answer_keys": len(keys),
            "top_answer_key_share": round(keys.most_common(1)[0][1] / len(rows), 4),
        }
        stats["by_source"][key[0]] += len(rows)
        for line, subset, scene, _ in rows:
            stats["by_subset"][subset] += 1
            stats["scenes"].add((subset, scene))
            lines.append(line)
    rng.shuffle(lines)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.writelines(lines)
    stats["rows"] = len(lines)
    stats["scenes"] = len(stats["scenes"])
    stats["inputs"] = args.inputs
    stats["max_per_type"] = args.max_per_type
    stats["seed"] = args.seed
    with open(args.output + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: stats[k] for k in ("rows", "scenes", "by_source", "by_subset")}))


if __name__ == "__main__":
    main()
