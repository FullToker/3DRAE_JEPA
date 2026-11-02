#!/usr/bin/env python3
"""Project labelled box centres onto frames, to check the coordinate transform.

Prints how often a frame's visible objects land inside the image, with the
axis-aligned transform and with the raw pose, and writes one contact sheet per
scene with the object names drawn at their centres.

Usage (from the repo root):
    python jepa3d/geo_probe/check_labels.py --labels data/geo_probe/labels_val.jsonl \
        --output logs/geo_probe/label_check --scenes 20
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw

from jepa3d.frame_pool import load_pool
from jepa3d.paths import VJEPA_ROOT

SOURCE = VJEPA_ROOT / "source_data"
TILE = (648, 484)


def project(points, c2w, K):
    w2c = np.linalg.inv(c2w)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[:, 2]
    uv = camera @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:], 1e-6), depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--views", type=int, default=6)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with open(args.labels) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in sorted(rows, key=lambda r: r["scene"]) if len(r["objects"]) >= 3]
    rows = rows[:: max(1, len(rows) // args.scenes)][:args.scenes]
    hits = {"aligned": [0, 0], "raw": [0, 0]}
    for row in rows:
        pool = load_pool("scannet", row["scene"], SOURCE / "spar", SOURCE / "scannet" / "posed_images")
        align = np.array(row["axis_align"])
        by_id = {o["id"]: o for o in row["objects"]}
        visible = {int(k): v for k, v in row["visible"].items()}
        frames = [f for f in pool.frames if visible.get(f.frame_id)]
        for frame in frames:
            points = np.array([by_id[i]["center"] for i in visible[frame.frame_id]])
            width, height = Image.open(frame.image).size
            for name, c2w in (("aligned", align @ frame.c2w), ("raw", frame.c2w)):
                uv, depth = project(points, c2w, pool.K)
                inside = (depth > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
                hits[name][0] += int(inside.sum())
                hits[name][1] += len(points)

        picked = sorted(frames, key=lambda f: -len(visible[f.frame_id]))[:args.views]
        sheet = Image.new("RGB", (TILE[0] * 3, TILE[1] * ((len(picked) + 2) // 3)), "black")
        for i, frame in enumerate(picked):
            image = Image.open(frame.image).convert("RGB")
            sx, sy = TILE[0] / image.width, TILE[1] / image.height
            image = image.resize(TILE)
            draw = ImageDraw.Draw(image)
            ids = visible[frame.frame_id]
            uv, depth = project(np.array([by_id[j]["center"] for j in ids]), align @ frame.c2w, pool.K)
            for j, (u, v), z in zip(ids, uv, depth):
                if z <= 0.1:
                    continue
                x, y = u * sx, v * sy
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="red", width=2)
                draw.text((x + 7, y - 6), by_id[j]["category"], fill="yellow", stroke_width=1, stroke_fill="black")
            sheet.paste(image, (TILE[0] * (i % 3), TILE[1] * (i // 3)))
        sheet.save(output / f"{row['scene']}.jpg", quality=85)
    print(json.dumps({"scenes": len(rows), **{k: {"inside": a / max(b, 1), "n": b} for k, (a, b) in hits.items()}}))


if __name__ == "__main__":
    main()
