#!/usr/bin/env python3
"""Build a posed SPAR manifest for the 3DRAE X-encoder.

Reads a fusion_gv SPAR jsonl (rows with "images", "query", "target", ...),
keeps rows whose images all come from a pose-verified subset, removes
repeated image paths (the uni8 rows tile 2-3 images up to 8 slots), and
attaches each view's intrinsics and camera-to-world pose:

    {"images": [...], "intrinsics": [[3x3], ...], "c2w": [[4x4], ...],
     "subset": "scannet", "scene": "scene0261_00", "src_line": 17, ...original fields}

With --val-out, rows from a hash-selected set of scenes go to a separate
held-out file instead (scene-level split, no scene shared with train).

Usage:
    python jepa3d/tools/make_spar_manifest.py --input <jsonl> --output <jsonl> \
        [--val-out <jsonl> --val-scene-mod 200] [--workers 16]
"""

import argparse
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.spar_cameras import SUPPORTED_SUBSETS, load_camera, subset_of


def _convert(item):
    line_no, line = item
    row = json.loads(line)
    images = list(dict.fromkeys(row.get("images") or [row["image"]]))
    subsets = {subset_of(p) for p in images}
    if len(subsets) != 1 or not subsets <= set(SUPPORTED_SUBSETS):
        return "unsupported", None
    try:
        cams = [load_camera(p) for p in images]
    except (OSError, ValueError):
        return "missing_pose", None
    row.pop("image", None)
    row["images"] = images
    row["intrinsics"] = [K.round(6).tolist() for K, _ in cams]
    row["c2w"] = [c2w.round(6).tolist() for _, c2w in cams]
    row["subset"] = subsets.pop()
    row["scene"] = Path(images[0]).parents[1].name
    row["src_line"] = line_no   # 0-based line in --input, for joining with old predictions
    return "ok", row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-out")
    parser.add_argument("--val-scene-mod", type=int, default=200,
                        help="scenes with crc32(scene) %% mod == 0 go to --val-out")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="debug: stop after N input rows")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "val": 0, "unsupported": 0, "missing_pose": 0}
    views = {}
    out = open(args.output, "w")
    val = open(args.val_out, "w") if args.val_out else None
    with open(args.input) as f, Pool(args.workers) as pool:
        lines = (
            (i, l) for i, l in enumerate(f) if l.strip() and not (args.limit and i >= args.limit)
        )
        for status, row in pool.imap(_convert, lines, chunksize=256):
            if row is None:
                counts[status] += 1
                continue
            n = len(row["images"])
            views[n] = views.get(n, 0) + 1
            is_val = val is not None and zlib.crc32(row["scene"].encode()) % args.val_scene_mod == 0
            (val if is_val else out).write(json.dumps(row) + "\n")
            counts["val" if is_val else "ok"] += 1
    out.close()
    if val is not None:
        val.close()
    print(json.dumps({"counts": counts, "unique_views": dict(sorted(views.items()))}))


if __name__ == "__main__":
    main()
