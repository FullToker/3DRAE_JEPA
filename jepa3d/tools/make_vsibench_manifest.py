#!/usr/bin/env python3
"""VSI-Bench (raw) -> posed N-view eval manifest for the 3DRAE X-encoder.

Questions and options are the benchmark's own; only the "A. " letter prefix
is removed from the options (target = the ground-truth option text). Only
multiple-choice rows are kept (numeric questions cannot be scored by
candidate selection), and only ScanNet / ScanNet++ scenes, whose posed
frames are on the server (ARKitScenes has no poses there).

Views: N frames of the scene's posed pool by the same quality filter +
pose FPS as training (jepa3d/frame_pool.py), seeded by the question id.

Input is test.jsonl or one of the parquet variants (test_debiased.parquet,
test_pruned.parquet); rows keep "id" for joining.

Usage:
    python jepa3d/tools/make_vsibench_manifest.py --input <test.jsonl|.parquet> \
        --num-views 8 --stats data/frame_stats/*.jsonl --output data/vsibench_posed_f8.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.frame_pool import load_stats, prepare_scene, sample_views, view_fields
from jepa3d.paths import VJEPA_ROOT

SOURCE = VJEPA_ROOT / "source_data"
PREFIX = re.compile(r"^\s*([A-H])\s*[.:)]\s*")


def read_rows(path: str) -> list:
    if path.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
        for r in rows:
            if r.get("options") is not None:
                r["options"] = list(r["options"])
        return rows
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--min-valid-frac", type=float, default=0.5)
    parser.add_argument("--min-median-depth", type=float, default=0.6)
    parser.add_argument("--min-rel-sharpness", type=float, default=0.3)
    args = parser.parse_args()

    stats = load_stats(args.stats)
    thresholds = {"min_valid_frac": args.min_valid_frac, "min_median_depth": args.min_median_depth,
                  "min_rel_sharpness": args.min_rel_sharpness}
    counts, types, scenes = Counter(), Counter(), {}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for r in read_rows(args.input):
            if r["dataset"] not in ("scannet", "scannetpp"):
                counts[f"skip_{r['dataset']}"] += 1
                continue
            if not r.get("options"):
                counts["skip_numeric"] += 1
                continue
            options = {}
            for opt in r["options"]:
                m = PREFIX.match(opt)
                options[m.group(1) if m else opt] = PREFIX.sub("", opt).strip()
            if r["ground_truth"] not in options:
                counts["bad_ground_truth"] += 1
                continue
            key = (r["dataset"], r["scene_name"])
            if key not in scenes:
                try:
                    scenes[key] = prepare_scene(*key, stats, SOURCE / "spar", SOURCE / "scannet" / "posed_images",
                                                thresholds)
                except OSError:
                    scenes[key] = None
            if scenes[key] is None:
                counts["no_pool"] += 1
                continue
            pool, keep, dist = scenes[key]
            frames = sample_views(pool, keep, dist, args.num_views, f"vsibench:{r['id']}")
            if frames is None:
                counts["too_few_frames"] += 1
                continue
            out.write(json.dumps({
                **view_fields(pool, frames),
                "id": int(r["id"]),
                "query": r["question"],
                "candidates": list(options.values()),
                "target": options[r["ground_truth"]],
                "answer_raw": r["ground_truth"],
                "type": r["question_type"],
                "subset": r["dataset"],
                "scene": r["scene_name"],
                "group_id": f"{r['dataset']}/{r['scene_name']}",
            }) + "\n")
            counts[f"ok_{r['dataset']}"] += 1
            types[r["question_type"]] += 1
    print(json.dumps({"input": args.input, "counts": counts, "types": types,
                      "scenes_ok": sum(v is not None for v in scenes.values())}))


if __name__ == "__main__":
    main()
