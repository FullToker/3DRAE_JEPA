#!/usr/bin/env python3
"""Where do a trained probe's position and azimuth errors come from?

Runs a final.pt on the val scenes and breaks the errors down by
  - scene scale s and object distance from the first camera,
  - how many of the 8 views see the object's centre (projected inside the image,
    in front of the camera); "visible" from EmbodiedScan is much looser,
  - error axis in the first camera's frame (x right, y down, z forward),
  - pair distance, for the azimuth error.
Reference points: the ground-truth statistics, and an oracle that puts every
object at the mean of the sample's true positions.

Usage (from the repo root):
    python jepa3d/geo_probe/analyze.py --run outputs/geo_probe/real \
        --val-labels data/geo_probe/labels_val.jsonl --stats data/frame_stats/scannet.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from jepa3d.paths import VJEPA_ROOT  # noqa: F401  (sets sys.path)
from jepa3d.geo_probe.data import GeoProbeDataset, collate
from jepa3d.geo_probe.dino_memory import DinoPatchMemory
from jepa3d.geo_probe.probe import MIN_PAIR_DISTANCE, GeoProbe, angular_error, azimuths, quadrant
from jepa3d.geo_probe.train import embed_categories, scene_tokens, text_batch
from jepa3d.x_encoder import ThreeDRAEXEncoder, unpack_cams


def views_seeing(points, cams, scale, size):
    """Number of views whose image contains each point's projection. points (O, 3) metres."""
    K, w2c = unpack_cams(cams)
    p = torch.cat([points / scale, torch.ones(len(points), 1)], dim=1)       # normalized first-camera frame
    cam = torch.einsum("vij,oj->voi", w2c, p)[..., :3]
    uv = torch.einsum("vij,voj->voi", K, cam)
    z = cam[..., 2]
    u, v = uv[..., 0] / z.clamp(min=1e-6), uv[..., 1] / z.clamp(min=1e-6)
    inside = (z > 0.1) & (u >= 0) & (u < size[1]) & (v >= 0) & (v < size[0]) & (K[:, None, 0, 0] > 0)
    return inside.sum(0).numpy()


def describe(values) -> str:
    v = np.asarray(values)
    if len(v) == 0:
        return "n=0"
    return f"n={len(v)} median={np.median(v):.3f} p25={np.percentile(v, 25):.3f} p75={np.percentile(v, 75):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run = Path(args.run)
    ckpt = torch.load(run / "final.pt", map_location="cpu")
    train_args = ckpt["args"]
    device = torch.device("cuda")
    val_set = GeoProbeDataset(args.val_labels, args.stats, seeds=list(range(train_args["val_seeds"])))
    loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    categories = {o["category"] for s in val_set.scenes for o in s["objects"]}
    embeddings = embed_categories(categories, device)

    encoder = ThreeDRAEXEncoder().to(device).eval()
    probe = GeoProbe(next(iter(embeddings.values())).shape[-1], encoder.dim).to(device).eval()
    probe.load_state_dict(ckpt["probe"])
    dino = None
    if train_args.get("memory", "3drae") == "dino":
        dino = DinoPatchMemory(encoder.rae.encoder.hidden_size).to(device).eval()
        dino.load_state_dict(ckpt["encoder_trainable"])
    else:
        missing, unexpected = encoder.load_state_dict(ckpt["encoder_trainable"], strict=False)
        assert not unexpected and all(k.startswith("rae.") for k in missing)

    objects, triplets = [], []
    stats = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            tokens = scene_tokens(encoder, batch, device, dino=dino)
            pred = probe(text_batch(batch["categories"], embeddings, device), tokens,
                         batch["scale"].to(device)).cpu()
            size = batch["images"].shape[-2:]
            for i in range(len(batch["scene"])):
                n = int(batch["mask"][i].sum())
                p, t = pred[i, :n].numpy(), batch["positions"][i, :n].numpy()
                s = float(batch["scale"][i])
                seen = views_seeing(batch["positions"][i, :n], batch["cams"][i], s, size)
                centroid = t.mean(0, keepdims=True)
                stats["scale_m"].append(s)
                for j in range(n):
                    objects.append({
                        "scene": batch["scene"][i], "category": batch["categories"][i][j], "scale": s,
                        "visible": bool(batch["visible"][i, j]), "views_seeing_centre": int(seen[j]),
                        "gt_dist_from_cam0": float(np.linalg.norm(t[j])),
                        "error": float(np.linalg.norm(p[j] - t[j])),
                        "error_xyz": (p[j] - t[j]).tolist(),
                        "oracle_centroid_error": float(np.linalg.norm(centroid[0] - t[j])),
                    })
                keep = batch["visible"][i, :n].numpy()
                if keep.sum() < 3:
                    continue
                up = batch["up"][i].numpy()
                trip, truth = azimuths(t[keep], up)
                _, est = azimuths(p[keep], up)
                flat = t[keep] - (t[keep] @ up)[:, None] * up
                for (a, c, b), gt, pr in zip(trip, truth, est):
                    ac, ab = np.linalg.norm(flat[c] - flat[a]), np.linalg.norm(flat[b] - flat[a])
                    if min(ac, ab) >= MIN_PAIR_DISTANCE:
                        triplets.append((min(ac, ab), float(angular_error(pr, gt)), quadrant(pr) == quadrant(gt)))

    out = {}
    err = np.array([o["error"] for o in objects])
    print("== ground truth")
    print("scene scale s (farthest camera from first, m):", describe(stats["scale_m"]))
    print("object distance from first camera (m):", describe([o["gt_dist_from_cam0"] for o in objects]))
    print("oracle: every object at the sample's true centroid, error (m):",
          describe([o["oracle_centroid_error"] for o in objects]))
    print("== probe position error (m), all objects:", describe(err))
    vis = [o for o in objects if o["visible"]]
    print("  EmbodiedScan-visible:", describe([o["error"] for o in vis]))
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 8)):
        sel = [o["error"] for o in objects if lo <= o["views_seeing_centre"] <= hi]
        print(f"  centre inside {lo}-{hi} of the views:", describe(sel))
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        sel = [o["error"] for o in objects if lo <= o["gt_dist_from_cam0"] < hi]
        print(f"  object {lo}-{hi} m from first camera:", describe(sel))
    xyz = np.abs(np.array([o["error_xyz"] for o in objects]))
    print("  |error| per axis of the first camera, median x (right) %.3f  y (down) %.3f  z (forward) %.3f"
          % tuple(np.median(xyz, axis=0)))
    rel = err / np.array([max(o["gt_dist_from_cam0"], 0.1) for o in objects])
    print("  error / distance from first camera:", describe(rel))
    print("== azimuth (visible objects), by the shorter of |AC|, |AB| (horizontal, m)")
    tri = np.array(triplets, dtype=np.float64)
    for lo, hi in ((0.1, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)):
        sel = tri[(tri[:, 0] >= lo) & (tri[:, 0] < hi)]
        if len(sel):
            print(f"  {lo}-{hi} m: n={len(sel)} azimuth error median {np.median(sel[:, 1]):.1f} deg, "
                  f"quadrant acc {sel[:, 2].mean():.3f}")
    out["objects"] = objects
    path = run / "analysis_objects.jsonl"
    with open(path, "w") as f:
        for o in objects:
            f.write(json.dumps(o) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
