#!/usr/bin/env python3
"""Re-score an old fusion_gv SPAR predictions.jsonl on the posed eval subset.

fusion_gv/infer_gvjepa.py writes one row per input line with a 1-based
"line" and "correct"; the posed manifest keeps each row's 0-based
"src_line". Joining the two gives the baseline's accuracy on exactly the
rows the 3DRAE model is evaluated on, overall / per type / per subset.

Usage:
    python jepa3d/tools/rescore_baseline.py --posed <posed eval jsonl> \
        --predictions <old predictions.jsonl> [--output metrics.json]
"""

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posed", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    correct_by_line = {}
    with open(args.predictions) as f:
        for line in f:
            row = json.loads(line)
            if "correct" in row:
                correct_by_line[row["line"]] = bool(row["correct"])

    stats = defaultdict(lambda: [0, 0])
    missing = 0
    with open(args.posed) as f:
        for line in f:
            row = json.loads(line)
            correct = correct_by_line.get(row["src_line"] + 1)
            if correct is None:
                missing += 1
                continue
            for key in ("all", f"type/{row.get('type')}", f"subset/{row.get('subset')}"):
                stats[key][0] += int(correct)
                stats[key][1] += 1

    metrics = {
        "predictions": args.predictions,
        "missing": missing,
        "accuracy": {k: c / n for k, (c, n) in sorted(stats.items())},
        "evaluated": {k: n for k, (_, n) in sorted(stats.items())},
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
    print(f"evaluated={stats['all'][1]} missing={missing} accuracy={metrics['accuracy']['all']:.4f}")


if __name__ == "__main__":
    main()
