#!/usr/bin/env python3
"""MMSI-Bench rows in the posed-manifest format.

MMSI-Bench (1,000 multiple-choice questions over 2-10 unposed images from several
datasets) has no camera poses, and the 3DRAE encoder needs intrinsics and
extrinsics per view. Cameras are either
    --poses <poses.jsonl>   estimated with VGGT (jepa3d/tools/estimate_vggt_poses.py), or
    placeholders            identity c2w for every view, focal length = the longer image
                            side, principal point at the centre: wrong geometry input,
                            a lower reference only.
Rows keep at most --max-views images.

Input: vjepa201/data/mmsibench_manifest.jsonl (images, query with "Options: ...",
candidates "A. ...", target). Output rows: query without the options line,
candidates and target without the letter prefix, "type" = question type.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

PREFIX = re.compile(r"^\s*([A-D])[.:]\s*")
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--poses", help="poses.jsonl of estimate_vggt_poses.py")
    args = parser.parse_args()
    counts = Counter()
    poses = {}
    if args.poses:
        with open(args.poses) as f:
            poses = {p["id"]: p for p in map(json.loads, f)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as src, open(args.output, "w") as out:
        for line in src:
            row = json.loads(line)
            images = row["images"][:args.max_views]
            intrinsics, c2w = [], [IDENTITY] * len(images)
            if poses:
                pose = poses[row["id"]]
                assert pose["images"] == images, row["id"]
                intrinsics, c2w = pose["intrinsics"], pose["c2w"]
            for path in [] if poses else images:
                with Image.open(path) as image:
                    width, height = image.size
                focal = float(max(width, height))
                intrinsics.append([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
            candidates = [PREFIX.sub("", c).strip() for c in row["candidates"]]
            target = PREFIX.sub("", row["target"]).strip()
            if target not in candidates:
                counts["bad_target"] += 1
                continue
            out.write(json.dumps({
                "images": images, "intrinsics": intrinsics, "c2w": c2w,
                "id": row["id"], "query": row["query"].split("\nOptions:")[0].strip(),
                "candidates": candidates, "target": target,
                "type": row["question_type"], "views": len(row["images"]), "cameras": "vggt" if poses else "placeholder",
                "group_id": f"mmsibench/{row['id']}",
            }) + "\n")
            counts["ok"] += 1
            counts[f"type/{row['question_type']}"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
