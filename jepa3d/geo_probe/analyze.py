#!/usr/bin/env python3
"""Where do a trained probe's position and azimuth errors come from?

Runs a final.pt on the val scenes and breaks the errors down by
  - scene scale s and object distance from the first camera,
  - how many of the 8 views see the object's centre (projected inside the image,
    in front of the camera); "visible" from EmbodiedScan is much looser,
  - how often the object's category occurs in the training labels,
  - error axis in the first camera's frame (x right, y down, z forward),
  - pair distance, for the azimuth error.
Reference: an oracle that puts every object at the mean of the sample's true positions.

--attention (DINOv2 / VGGT memory runs only, whose tokens are image patches): share of
each layer's attention on the patches around the object's projected centre, in
every view that sees the centre, against the share a uniform attention would give.

Usage (from the repo root):
    python jepa3d/geo_probe/analyze.py --run outputs/geo_probe/real \
        --train-labels data/geo_probe/labels_train.jsonl \
        --val-labels data/geo_probe/labels_val.jsonl --stats data/frame_stats/scannet.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from jepa3d.paths import VJEPA_ROOT  # noqa: F401  (sets sys.path)
from jepa3d.geo_probe.data import GeoProbeDataset, collate
from jepa3d.geo_probe.probe import MIN_PAIR_DISTANCE, GeoProbe, angular_error, azimuths, quadrant
from jepa3d.geo_probe.train import build_patch_memory, build_queries, embed_categories, scene_tokens
from jepa3d.x_encoder import ThreeDRAEXEncoder, unpack_cams

FREQ_BINS = ((0, 0), (1, 9), (10, 49), (50, 199), (200, 10**9))
WINDOWS = (1, 3)   # half-widths in patches: 3x3 and 7x7


def project(points, cams, scale, size):
    """(O, 3) metres -> u, v (V, O) input pixels and inside (V, O)."""
    K, w2c = unpack_cams(cams)
    p = torch.cat([points / scale, torch.ones(len(points), 1)], dim=1)       # normalized first-camera frame
    cam = torch.einsum("vij,oj->voi", w2c, p)[..., :3]
    uv = torch.einsum("vij,voj->voi", K, cam)
    z = cam[..., 2]
    u, v = uv[..., 0] / z.clamp(min=1e-6), uv[..., 1] / z.clamp(min=1e-6)
    inside = (z > 0.1) & (u >= 0) & (u < size[1]) & (v >= 0) & (v < size[0]) & (K[:, None, 0, 0] > 0)
    return u.numpy(), v.numpy(), inside.numpy()


def window_masks(u, v, inside, h, w, patch):
    """Token masks (O, V*h*w) of the patches within each half-width of the centre, in every view seeing it."""
    views, count = inside.shape
    out = {r: np.zeros((count, views * h * w), dtype=bool) for r in WINDOWS}
    for o in range(count):
        for view in np.flatnonzero(inside[:, o]):
            row, col = min(int(v[view, o] // patch), h - 1), min(int(u[view, o] // patch), w - 1)
            for r in WINDOWS:
                grid = np.zeros((h, w), dtype=bool)
                grid[max(0, row - r):row + r + 1, max(0, col - r):col + r + 1] = True
                out[r][o, view * h * w:(view + 1) * h * w] = grid.ravel()
    return out


def describe(values) -> str:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return "n=0"
    return f"n={len(v)} median={np.median(v):.3f} p25={np.percentile(v, 25):.3f} p75={np.percentile(v, 75):.3f}"


def freq_bin(n):
    for lo, hi in FREQ_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    freq = Counter()
    with open(args.train_labels) as f:
        for line in f:
            freq.update(o["category"] for o in json.loads(line)["objects"])

    run = Path(args.run)
    ckpt = torch.load(run / "final.pt", map_location="cpu")
    train_args = ckpt["args"]
    query = train_args.get("query", "category")
    memory = train_args.get("memory", "3drae")
    if args.attention and memory == "3drae":
        raise ValueError("--attention needs a DINOv2 / VGGT memory run: 3DRAE tokens have no image position")
    device = torch.device("cuda")
    val_set = GeoProbeDataset(args.val_labels, args.stats, seeds=list(range(train_args["val_seeds"])),
                              memory_size=train_args.get("memory_size"))
    loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)

    encoder = ThreeDRAEXEncoder().to(device).eval()
    if query == "point":
        embeddings, text_dim = None, encoder.rae.encoder.hidden_size + 6
    else:
        embeddings = embed_categories({o["category"] for s in val_set.scenes for o in s["objects"]}, device)
        text_dim = next(iter(embeddings.values())).shape[-1]
    dino = build_patch_memory(memory, encoder, device)
    probe = GeoProbe(text_dim, dino.module.feat_dim if dino else encoder.dim).to(device).eval()
    probe.load_state_dict(ckpt["probe"])
    if dino is not None:
        dino.module.eval()
        dino.module.load_state_dict(ckpt["encoder_trainable"])
    else:
        missing, unexpected = encoder.load_state_dict(ckpt["encoder_trainable"], strict=False)
        assert not unexpected and all(k.startswith("rae.") for k in missing)
    shuffled = train_args["tokens"] == "shuffled"
    patch = encoder.rae.encoder.patch_size

    objects, triplets = [], []
    scales = []
    previous = None
    with torch.no_grad():
        for batch in loader:
            tokens = scene_tokens(encoder, batch, device, previous, shuffled, dino)
            previous = tokens
            queries, valid = build_queries(query, batch, embeddings, encoder, device)
            pred = probe(queries, tokens, batch["scale"].to(device)).cpu()
            maps = [m.float().cpu().numpy() for m in probe.attention_maps(queries, tokens)] if args.attention else None
            valid = (valid.cpu() & batch["mask"]).numpy()
            # the memory's own resolution, so attention windows index its patch grid
            size = batch.get("memory_images", batch["images"]).shape[-2:]
            cams = batch.get("memory_cams", batch["cams"])
            h, w = size[0] // patch, size[1] // patch
            for i in range(len(batch["scene"])):
                n = int(batch["mask"][i].sum())
                p, t = pred[i, :n].numpy(), batch["positions"][i, :n].numpy()
                s = float(batch["scale"][i])
                scales.append(s)
                u, v, inside = project(batch["positions"][i, :n], cams[i], s, size)
                windows = window_masks(u, v, inside, h, w, patch) if args.attention else None
                centroid = t.mean(0)
                for j in range(n):
                    if not valid[i, j]:
                        continue
                    record = {
                        "scene": batch["scene"][i], "category": batch["categories"][i][j], "scale": s,
                        "train_freq": freq[batch["categories"][i][j]],
                        "visible": bool(batch["visible"][i, j]), "views_seeing_centre": int(inside[:, j].sum()),
                        "gt_dist_from_cam0": float(np.linalg.norm(t[j])),
                        "error": float(np.linalg.norm(p[j] - t[j])),
                        "error_xyz": (p[j] - t[j]).tolist(),
                        "oracle_centroid_error": float(np.linalg.norm(centroid - t[j])),
                    }
                    if windows is not None and inside[:, j].any():
                        for layer, m in enumerate(maps):
                            weights = m[i, j]
                            for r in WINDOWS:
                                record[f"attn_l{layer + 1}_r{r}"] = float(weights[windows[r][j]].sum())
                                record[f"uniform_r{r}"] = float(windows[r][j].mean())
                            record[f"top1_in_r3_l{layer + 1}"] = bool(windows[3][j][int(weights.argmax())])
                    objects.append(record)
                keep = valid[i, :n] & batch["visible"][i, :n].numpy()
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

    print(f"run {run.name}: query={query} memory={memory} tokens={train_args['tokens']}")
    print("== ground truth")
    print("scene scale s (farthest camera from first, m):", describe(scales))
    print("object distance from first camera (m):", describe([o["gt_dist_from_cam0"] for o in objects]))
    print("oracle: every object at the sample's true centroid, error (m):",
          describe([o["oracle_centroid_error"] for o in objects]))
    print("== probe position error (m), all queried objects:", describe([o["error"] for o in objects]))
    print("  EmbodiedScan-visible:", describe([o["error"] for o in objects if o["visible"]]))
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 8)):
        print(f"  centre inside {lo}-{hi} of the views:",
              describe([o["error"] for o in objects if lo <= o["views_seeing_centre"] <= hi]))
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        print(f"  object {lo}-{hi} m from first camera:",
              describe([o["error"] for o in objects if lo <= o["gt_dist_from_cam0"] < hi]))
    seen = [o for o in objects if o["views_seeing_centre"] > 0]
    print("  by category frequency in train (centre in a view):")
    for lo, hi in FREQ_BINS:
        sel = [o for o in seen if lo <= o["train_freq"] <= hi]
        print(f"    {freq_bin(lo)}: error {describe([o['error'] for o in sel])}; oracle centroid median "
              f"{np.median([o['oracle_centroid_error'] for o in sel]) if sel else float('nan'):.3f}")
    xyz = np.abs(np.array([o["error_xyz"] for o in objects]))
    print("  |error| per axis of the first camera, median x (right) %.3f  y (down) %.3f  z (forward) %.3f"
          % tuple(np.median(xyz, axis=0)))
    rel = np.array([o["error"] / max(o["gt_dist_from_cam0"], 0.1) for o in objects])
    print("  error / distance from first camera:", describe(rel))
    print("== azimuth (visible queried objects), by the shorter of |AC|, |AB| (horizontal, m)")
    tri = np.array(triplets, dtype=np.float64).reshape(-1, 3)
    for lo, hi in ((0.1, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)):
        sel = tri[(tri[:, 0] >= lo) & (tri[:, 0] < hi)]
        if len(sel):
            print(f"  {lo}-{hi} m: n={len(sel)} azimuth error median {np.median(sel[:, 1]):.1f} deg, "
                  f"quadrant acc {sel[:, 2].mean():.3f}")
    print(f"  all: n={len(tri)} azimuth error median {np.median(tri[:, 1]):.1f} deg, quadrant acc {tri[:, 2].mean():.3f}")

    if args.attention:
        att = [o for o in objects if "attn_l1_r1" in o]
        print(f"== attention on the patches around the projected centre (n={len(att)} objects)")
        groups = [("all", att)] + [(f"train freq {freq_bin(lo)}", [o for o in att if lo <= o["train_freq"] <= hi])
                                   for lo, hi in FREQ_BINS]
        for name, sel in groups:
            if not sel:
                continue
            parts = []
            for layer in (1, 2):
                for r in WINDOWS:
                    mass = np.mean([o[f"attn_l{layer}_r{r}"] for o in sel])
                    uniform = np.mean([o[f"uniform_r{r}"] for o in sel])
                    parts.append(f"L{layer} {2 * r + 1}x{2 * r + 1} {mass:.3f} ({mass / uniform:.1f}x uniform)")
                parts.append(f"L{layer} top1 in 7x7 {np.mean([o[f'top1_in_r3_l{layer}'] for o in sel]):.3f}")
            print(f"  {name} (n={len(sel)}): " + "; ".join(parts))
        for layer in (1, 2):
            key = f"attn_l{layer}_r3"
            cuts = np.percentile([o[key] for o in att], [33.3, 66.7])
            labels = ("low", "mid", "high")
            sel = [[o["error"] for o in att if np.searchsorted(cuts, o[key]) == k] for k in range(3)]
            print(f"  error by L{layer} 7x7 attention tertile: " +
                  ", ".join(f"{labels[k]} {np.median(sel[k]):.3f} m" for k in range(3)))

    path = run / "analysis_objects.jsonl"
    with open(path, "w") as f:
        for o in objects:
            f.write(json.dumps(o) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
