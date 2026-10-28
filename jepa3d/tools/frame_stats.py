#!/usr/bin/env python3
"""Per-frame quality stats for every scene pool (see jepa3d/frame_pool.py).

One line per frame: {"subset", "scene", "frame_id", "valid_frac",
"median_depth", "sharpness", "width", "height"}. The samplers read this file
to drop bad frames before FPS, so quality is computed once, not per row.
The summary at the end prints the distributions used to set the thresholds.

Usage:
    python jepa3d/tools/frame_stats.py --subset scannet --output data/frame_stats/scannet.jsonl
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.frame_pool import frame_quality, keep_mask, load_pool
from jepa3d.paths import VJEPA_ROOT

SPAR_ROOT = VJEPA_ROOT / "source_data" / "spar"
POSED_ROOT = VJEPA_ROOT / "source_data" / "scannet" / "posed_images"


def scenes_of(subset):
    root = POSED_ROOT if subset == "scannet" else SPAR_ROOT / subset / "images"
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _scene_stats(item):
    subset, scene = item
    pool = load_pool(subset, scene, SPAR_ROOT, POSED_ROOT)
    out = []
    for frame in pool.frames:
        try:
            stats = frame_quality(frame)
        except OSError:
            continue
        out.append({"subset": subset, "scene": scene, "frame_id": frame.frame_id, **stats})
    return pool.K.tolist(), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["scannet", "scannetpp"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N scenes")
    args = parser.parse_args()

    scenes = scenes_of(args.subset)[: args.limit or None]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows, cx_ratio = [], []
    with open(args.output, "w") as f, Pool(args.workers) as pool:
        for K, stats in pool.imap_unordered(_scene_stats, [(args.subset, s) for s in scenes], chunksize=4):
            for s in stats:
                f.write(json.dumps(s) + "\n")
            rows.extend(stats)
            if stats:
                cx_ratio.append(2 * K[0][2] / stats[0]["width"])

    by_scene = {}
    for s in rows:
        by_scene.setdefault(s["scene"], []).append(s)
    kept = {sc: int(keep_mask(v).sum()) for sc, v in by_scene.items()}
    q = lambda key: np.percentile([s[key] for s in rows], [1, 5, 25, 50, 75]).round(3).tolist()
    print(json.dumps({
        "subset": args.subset,
        "scenes": len(by_scene),
        "frames": len(rows),
        "frames_per_scene_p5_50_95": np.percentile([len(v) for v in by_scene.values()], [5, 50, 95]).tolist(),
        "valid_frac_p1_5_25_50_75": q("valid_frac"),
        "median_depth_p1_5_25_50_75": q("median_depth"),
        "kept_frac_default_thresholds": sum(kept.values()) / max(len(rows), 1),
        "scenes_with_kept_lt": {n: sum(k < n for k in kept.values()) for n in (8, 16, 24)},
        "image_sizes": sorted({(s["width"], s["height"]) for s in rows}),
        "2cx_over_width_min_max": [min(cx_ratio), max(cx_ratio)] if cx_ratio else None,
    }))


if __name__ == "__main__":
    main()
