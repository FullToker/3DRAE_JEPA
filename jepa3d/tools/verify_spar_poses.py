#!/usr/bin/env python3
"""Check SPAR camera conventions by depth reprojection between paired views.

For image pairs taken from a SPAR QA manifest, back-project view A's depth,
move it into view B under each pose hypothesis, and compare against B's depth.
The hypothesis/scale with the lowest error is the stored convention.

Usage:
    python jepa3d/tools/verify_spar_poses.py --manifest <jsonl> --pairs-per-subset 20
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.spar_cameras import depth_path, load_camera

DEPTH_UNITS = (1e-3, 1e-4, 1.0, 1e3)


def _pairs(manifest, per_subset):
    found = defaultdict(list)
    with open(manifest) as f:
        for line in f:
            images = list(dict.fromkeys(json.loads(line)["images"]))
            if len(images) < 2:
                continue
            subset = Path(images[0]).parents[3].name
            if len(found[subset]) < per_subset:
                found[subset].append((images[0], images[1]))
            if len(found) >= 3 and all(len(v) >= per_subset for v in found.values()):
                break
    return found


# Camera-axis sign conventions applied on the camera side of c2w
# ("+++" = OpenCV, "+--" = OpenGL).
AXES = {
    "".join("+" if v > 0 else "-" for v in signs): np.diag([*signs, 1.0])
    for signs in itertools.product((1.0, -1.0), repeat=3)
}


def _as_c2w(P, stored):
    if stored == "w2c":
        return np.linalg.inv(P)
    if stored == "mixed":   # w2c rotation with camera-center translation
        out = P.copy()
        out[:3, :3] = P[:3, :3].T
        return out
    return P


def _reproject_error(a, b, stored, unit, axes="+++", stride=4):
    Ka, Pa = load_camera(a)
    Kb, Pb = load_camera(b)
    Pa, Pb = _as_c2w(Pa, stored), _as_c2w(Pb, stored)
    Pa, Pb = Pa @ AXES[axes], Pb @ AXES[axes]
    color_w, color_h = Image.open(a).size
    da = np.asarray(Image.open(depth_path(a)), dtype=np.float64) * unit
    db = np.asarray(Image.open(depth_path(b)), dtype=np.float64) * unit
    h, w = da.shape
    scale = np.array([[w / color_w, 0, 0], [0, h / color_h, 0], [0, 0, 1]])
    Ka, Kb = scale @ Ka, scale @ Kb

    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = da[v, u]
    keep = z > 0
    pix = np.stack([u[keep] + 0.5, v[keep] + 0.5, np.ones(keep.sum())])
    pts = np.linalg.inv(Ka) @ pix * z[keep]
    world = Pa[:3, :3] @ pts + Pa[:3, 3:]
    cam_b = np.linalg.inv(Pb)[:3, :3] @ world + np.linalg.inv(Pb)[:3, 3:]
    front = cam_b[2] > 1e-6
    proj = Kb @ cam_b[:, front]
    ub, vb = proj[0] / proj[2], proj[1] / proj[2]
    inside = (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h)
    if inside.sum() < 50:
        return np.inf, 0.0
    zb = db[vb[inside].astype(int), ub[inside].astype(int)]
    zp = cam_b[2, front][inside]
    valid = zb > 0
    if valid.sum() < 50:
        return np.inf, 0.0
    rel = np.abs(zp[valid] - zb[valid]) / zb[valid]
    return float(np.median(rel)), float((rel < 0.05).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs-per-subset", type=int, default=20)
    parser.add_argument("--top", type=int, default=4, help="hypotheses printed per subset")
    args = parser.parse_args()

    for subset, pairs in sorted(_pairs(args.manifest, args.pairs_per_subset).items()):
        a0 = pairs[0][0]
        K, _ = load_camera(a0)
        cw, ch = Image.open(a0).size
        dh, dw = np.asarray(Image.open(depth_path(a0))).shape
        print(f"== {subset}: color {cw}x{ch}, depth {dw}x{dh}, "
              f"cx/W={K[0, 2] / cw:.3f} cy/H={K[1, 2] / ch:.3f} fx={K[0, 0]:.1f}")
        results = []
        for stored, unit, axes in itertools.product(("c2w", "w2c", "mixed"), DEPTH_UNITS, AXES):
            errs = [_reproject_error(a, b, stored, unit, axes) for a, b in pairs]
            med = np.median([e for e, _ in errs])
            inl = np.mean([i for _, i in errs])
            results.append((med, inl, stored, axes, unit))
        for med, inl, name, axes, unit in sorted(results)[: args.top]:
            print(f"   stored={name} axes={axes} depth_unit={unit:g}: "
                  f"median rel err={med:.3f} inlier<5%={inl:.2f}")


if __name__ == "__main__":
    main()
