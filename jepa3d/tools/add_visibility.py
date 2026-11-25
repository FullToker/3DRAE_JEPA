#!/usr/bin/env python3
"""Where does each labelled object show up in the views of its row, and is it hidden there?

Input: manifests whose rows carry "ground" labels (jepa3d/tools/build_ground_labels.py:
which object of scene_objects.json a name of the question is). For every label
the object's true centre is projected into each view of the row; views where it
falls inside the image are kept as

    "views": [[view, u, v, state], ...]      u, v in 0..1 of the image width / height
        state 1  visible:  the depth map's surface at that pixel is not clearly in front of the centre
        state 0  hidden:   a surface more than --margin metres in front of the centre
        state 2  unknown:  no valid depth around the pixel

A centre lies inside its object, so the surface seen at its pixel is always a
little in front of it; --margin (0.6 m) is how much is still called visible.
Rows without labels are copied as they are. The source manifests are not touched.

Usage (from the repo root):
    python jepa3d/tools/add_visibility.py --objects experiments/ground_v1/data/scene_objects.json \
        --manifests experiments/ground_v1/data/train_f8.jsonl experiments/ground_v1/data/holdout_f8.jsonl \
        --output-dir experiments/objptr_v1/data
"""

import argparse
import json
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_SCALE = 1e-3   # depth PNGs are millimetres
MIN_DEPTH = 0.1
BINS = (-2.0, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 2.0)   # surface depth minus centre depth, metres

OBJECTS = {}
MARGIN = 0.6


def depth_path(image: str) -> Path:
    path = Path(image)
    if path.parent.name == "image_color":      # SPAR ScanNet++: image_color/<id>.jpg -> image_depth/<id>.png
        return path.parent.parent / "image_depth" / f"{path.stem}.png"
    if path.parent.name == "video_color":      # video_color/frame<k>_<id>.jpg -> video_depth/frame_<id>.png
        return path.parent.parent / "video_depth" / f"frame_{path.stem.split('_', 1)[1]}.png"
    return path.with_suffix(".png")            # ScanNet posed_images/<scene>/<id>.jpg -> <id>.png


@lru_cache(maxsize=4096)
def image_size(image: str) -> tuple:
    with Image.open(image) as picture:
        return picture.size   # (width, height)


@lru_cache(maxsize=192)
def depth_map(image: str):
    path = depth_path(image)
    if not path.is_file():
        return None
    with Image.open(path) as picture:
        return np.asarray(picture, dtype=np.float32) * DEPTH_SCALE


@lru_cache(maxsize=512)
def depth_intrinsics(folder: str):
    """ScanNet's depth camera has its own intrinsics; None where depth and colour share the field of view."""
    path = Path(folder) / "depth_intrinsic.txt"
    return np.loadtxt(path).reshape(4, 4)[:3, :3] if path.is_file() else None


def surface_depth(image: str, cam: np.ndarray, u: float, v: float):
    """Median valid depth in a small window around the pixel that shows camera point `cam`; None without one."""
    depth = depth_map(image)
    if depth is None:
        return None
    height, width = depth.shape
    K = depth_intrinsics(str(Path(image).parent))
    if K is None:
        x, y = u * width, v * height
    else:
        x, y = K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]
    r = max(1, round(width / 256))   # 3 x 3 pixels at 256 wide, 5 x 5 at 640
    x, y = int(x), int(y)
    window = depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    window = window[window > MIN_DEPTH]
    return float(np.median(window)) if window.size else None


def views_of(position: np.ndarray, row: dict, tally: Counter) -> list:
    views = []
    for index, (image, K, c2w) in enumerate(zip(row["images"], row["intrinsics"], row["c2w"])):
        K = np.array(K, dtype=np.float64)
        cam = np.linalg.inv(np.array(c2w, dtype=np.float64)) @ np.append(position, 1.0)
        if cam[2] <= MIN_DEPTH:
            continue
        width, height = image_size(image)
        u = (K[0, 0] * cam[0] / cam[2] + K[0, 2]) / width
        v = (K[1, 1] * cam[1] / cam[2] + K[1, 2]) / height
        if not (0 <= u < 1 and 0 <= v < 1):
            continue
        surface = surface_depth(image, cam, u, v)
        if surface is None:
            state = 2
        else:
            state = int(surface > cam[2] - MARGIN)
            tally[f"gap_below_{next((b for b in BINS if surface - cam[2] < b), 'more')}"] += 1
        tally[("hidden", "visible", "unknown")[state]] += 1
        views.append([index, round(float(u), 4), round(float(v), 4), state])
    return views


def process(line: str):
    tally = Counter(rows=1)
    if '"ground": [{' not in line:
        return line, tally
    row = json.loads(line)
    table = OBJECTS.get(f"{row.get('subset')}/{row.get('scene')}")
    for entry in row["ground"] if table is not None else []:
        entry["views"] = views_of(table[entry["object"]], row, tally)
        tally["names"] += 1
        tally["names_visible"] += any(view[3] == 1 for view in entry["views"])
        tally["names_in_frame"] += bool(entry["views"])
    tally["rows_labelled"] += 1
    tally["rows_visible"] += any(view[3] == 1 for entry in row["ground"] for view in entry.get("views", []))
    return json.dumps(row) + "\n", tally


def setup(objects: str, margin: float) -> None:
    global MARGIN
    MARGIN = margin
    with open(objects) as f:
        OBJECTS.update({key: np.array(scene["pos"], dtype=np.float64).reshape(-1, 3)
                        for key, scene in json.load(f).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--objects", required=True, help="scene_objects.json")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin", type=float, default=0.6, help="metres a surface may be in front of the centre")
    parser.add_argument("--workers", type=int, default=26)
    parser.add_argument("--limit", type=int, default=0, help="debug: the first N rows, prints the tally, writes nothing")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"margin_m": args.margin}
    with Pool(args.workers, initializer=setup, initargs=(args.objects, args.margin)) as pool:
        for manifest in args.manifests:
            tally = Counter()
            target = out_dir / Path(manifest).name
            tmp = target.with_suffix(".jsonl.tmp")
            with open(manifest) as source, open("/dev/null" if args.limit else tmp, "w") as sink:
                lines = (line for n, line in enumerate(source) if not args.limit or n < args.limit)
                for line, counts in pool.imap(process, lines, chunksize=256):
                    sink.write(line)
                    tally.update(counts)
            if not args.limit:
                tmp.rename(target)
            stats[Path(manifest).name] = dict(sorted(tally.items()))
            print(manifest, json.dumps(stats[Path(manifest).name], indent=1), flush=True)
    if not args.limit:
        (out_dir / "visibility.stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
