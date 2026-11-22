#!/usr/bin/env python3
"""Camera poses and intrinsics from VGGT for the images of an unposed manifest.

For benchmarks without poses (MMSI-Bench). Poses of our own training and
evaluation scenes come from the datasets, not from here.

Each row's images (at most --max-views) go through VGGT together. Every image is
resized so its longer side is 518 px (sides rounded to multiples of 14) and
padded with white to 518 x 518; VGGT's intrinsics are mapped back to the pixels
of the original image. Poses are OpenCV camera-to-world with the first view as
the world frame, in VGGT's own scale (not metres).

Output: <output>/poses.jsonl, one row per input row:
    {"id", "images", "sizes" [[width, height]], "intrinsics" [3x3], "c2w" [4x4]}
and <output>/README.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SIDE, PATCH = 518, 14


def load_padded(path: str):
    """-> (3, SIDE, SIDE) tensor in [0, 1], (width, height), scale (x, y), offset (x, y)."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    factor = SIDE / max(width, height)
    new_w = min(SIDE, max(PATCH, round(width * factor / PATCH) * PATCH))
    new_h = min(SIDE, max(PATCH, round(height * factor / PATCH) * PATCH))
    pixels = torch.from_numpy(np.asarray(image.resize((new_w, new_h), Image.BICUBIC))).permute(2, 0, 1).float() / 255
    canvas = torch.ones(3, SIDE, SIDE)
    off_x, off_y = (SIDE - new_w) // 2, (SIDE - new_h) // 2
    canvas[:, off_y:off_y + new_h, off_x:off_x + new_w] = pixels
    return canvas, (width, height), (new_w / width, new_h / height), (off_x, off_y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help='jsonl with "id" and "images"')
    parser.add_argument("--output", required=True, help="folder for poses.jsonl")
    parser.add_argument("--vggt-root", default=os.path.join(os.environ.get("VJEPA_ROOT", "."), "vggt"))
    parser.add_argument("--checkpoint", default=os.path.join(os.environ.get("VJEPA_ROOT", "."), "ckpts", "vggt.pt"))
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.vggt_root)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda"
    model = VGGT()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval().to(device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.manifest)]
    rows = rows[:args.limit] if args.limit else rows
    with open(output / "poses.jsonl", "w") as out:
        for n, row in enumerate(rows):
            images = row["images"][:args.max_views]
            loaded = [load_padded(path) for path in images]
            batch = torch.stack([item[0] for item in loaded]).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pose_enc = model(batch)["pose_enc"]
            w2c, K = pose_encoding_to_extri_intri(pose_enc.float(), (SIDE, SIDE))
            w2c, K = w2c[0].double().cpu(), K[0].double().cpu()
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(len(images), 1, 4)
            c2w = torch.linalg.inv(torch.cat([w2c, bottom], dim=1))
            c2w = torch.linalg.inv(c2w[0]) @ c2w   # first view = world frame
            intrinsics = []
            for k, (_, _, (sx, sy), (ox, oy)) in zip(K, loaded):
                intrinsics.append([[float(k[0, 0]) / sx, 0.0, (float(k[0, 2]) - ox) / sx],
                                   [0.0, float(k[1, 1]) / sy, (float(k[1, 2]) - oy) / sy],
                                   [0.0, 0.0, 1.0]])
            out.write(json.dumps({"id": row["id"], "images": images, "sizes": [list(item[1]) for item in loaded],
                                  "intrinsics": intrinsics, "c2w": c2w.tolist()}) + "\n")
            if n % 100 == 0:
                print(f"[{n}/{len(rows)}] views {len(images)} focal {intrinsics[0][0][0]:.0f} of {loaded[0][1]} "
                      f"baseline {float(c2w[:, :3, 3].norm(dim=-1).max()):.3f}", flush=True)
    (output / "README.md").write_text(
        "# Camera poses estimated with VGGT\n\n"
        f"Source rows: `{args.manifest}` ({len(rows)} rows, at most {args.max_views} images per row).\n"
        f"Model: VGGT, weights `{args.checkpoint}`. Tool: `3DRAE_JEPA/jepa3d/tools/estimate_vggt_poses.py`.\n\n"
        "`poses.jsonl`, one row per source row: `id`, `images`, `sizes` ([width, height] of each original image),\n"
        "`intrinsics` (3x3, in pixels of the original image), `c2w` (4x4, OpenCV camera-to-world, first view = world frame).\n"
        "Scale is VGGT's own, not metres. These are estimates, not ground truth; the original dataset is untouched.\n")
    print("wrote", output / "poses.jsonl")


if __name__ == "__main__":
    main()
