"""Posed frame pools per scene, frame quality, and pose-space FPS.

A pool is every posed color frame available for one scene, ordered by raw
frame id (= capture time):

    scannet     source_data/scannet/posed_images/<scene>/NNNNN.{jpg,txt,png}
                + intrinsic.txt (color, 1296x968). Every 10th raw frame. The
                same raw c2w as SPAR's scannet pose/ files (checked on
                scene0000_00), so the verified SPAR convention holds.
    scannetpp   source_data/spar/scannetpp/images/<scene>/
                image_color/<id>.jpg  + pose/<id>.txt       + image_depth/<id>.png
                video_color/frame<k>_<id>.jpg + video_pose/<id>.txt
                                                     + video_depth/frame_<id>.png
                + intrinsic/intrinsic_color.txt. Ids shared by both folders
                are taken once (image_color wins).

Poses are camera-to-world, OpenCV axes, metres. Depth pngs are uint16 mm.
"""

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEPTH_SCALE = 1e-3


@dataclass
class Frame:
    frame_id: int
    image: str
    depth: str
    c2w: np.ndarray   # (4, 4)


@dataclass
class ScenePool:
    subset: str
    scene: str
    K: np.ndarray     # (3, 3), color resolution
    frames: list      # [Frame], sorted by frame_id


def _load_matrix(path: Path) -> np.ndarray | None:
    try:
        m = np.loadtxt(path).reshape(4, 4)
    except (OSError, ValueError):
        return None
    return m if np.isfinite(m).all() else None


def _scannet_pool(posed_root: Path, scene: str) -> ScenePool:
    scene_dir = posed_root / scene
    K = np.loadtxt(scene_dir / "intrinsic.txt").reshape(4, 4)[:3, :3]
    frames = []
    for image in scene_dir.glob("[0-9][0-9][0-9][0-9][0-9].jpg"):   # skips -annotated.jpg
        c2w = _load_matrix(image.with_suffix(".txt"))
        depth = image.with_suffix(".png")
        if c2w is not None and depth.is_file():
            frames.append(Frame(int(image.stem), str(image), str(depth), c2w))
    return ScenePool("scannet", scene, K, sorted(frames, key=lambda f: f.frame_id))


def _scannetpp_pool(spar_root: Path, scene: str) -> ScenePool:
    scene_dir = spar_root / "scannetpp" / "images" / scene
    K = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_color.txt")
    K = K.reshape(K.shape[0], -1)[:3, :3]
    frames = {}
    for image in (scene_dir / "image_color").glob("*.jpg"):
        c2w = _load_matrix(scene_dir / "pose" / f"{image.stem}.txt")
        depth = scene_dir / "image_depth" / f"{image.stem}.png"
        if c2w is not None and depth.is_file():
            frames[int(image.stem)] = Frame(int(image.stem), str(image), str(depth), c2w)
    for image in (scene_dir / "video_color").glob("frame*_*.jpg"):
        frame_id = image.stem.split("_", 1)[1]
        if int(frame_id) in frames:
            continue
        c2w = _load_matrix(scene_dir / "video_pose" / f"{frame_id}.txt")
        depth = scene_dir / "video_depth" / f"frame_{frame_id}.png"
        if c2w is not None and depth.is_file():
            frames[int(frame_id)] = Frame(int(frame_id), str(image), str(depth), c2w)
    return ScenePool("scannetpp", scene, K, [frames[k] for k in sorted(frames)])


def load_pool(subset: str, scene: str, spar_root, posed_root) -> ScenePool:
    if subset == "scannet":
        return _scannet_pool(Path(posed_root), scene)
    if subset == "scannetpp":
        return _scannetpp_pool(Path(spar_root), scene)
    raise ValueError(f"unsupported subset {subset!r}")


def frame_quality(frame: Frame, min_depth=0.3, max_depth=8.0) -> dict:
    """Depth coverage and sharpness of one frame (all floats)."""
    from PIL import Image

    depth = np.asarray(Image.open(frame.depth), dtype=np.float32) * DEPTH_SCALE
    valid = (depth > min_depth) & (depth < max_depth)
    with Image.open(frame.image) as image:
        width, height = image.size
        gray = np.asarray(image.convert("L").resize((320, round(320 * height / width))), dtype=np.float32)
    lap = gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4 * gray[1:-1, 1:-1]
    return {
        "valid_frac": float(valid.mean()),
        "median_depth": float(np.median(depth[valid])) if valid.any() else 0.0,
        "sharpness": float(lap.var()),
        "width": width,
        "height": height,
    }


def keep_mask(stats: list, min_valid_frac=0.5, min_median_depth=0.6, min_rel_sharpness=0.3) -> np.ndarray:
    """Per-frame keep flags; sharpness is relative to the scene median."""
    valid = np.array([s["valid_frac"] for s in stats])
    median = np.array([s["median_depth"] for s in stats])
    sharp = np.array([s["sharpness"] for s in stats])
    rel = sharp / max(float(np.median(sharp)), 1e-6)
    return (valid >= min_valid_frac) & (median >= min_median_depth) & (rel >= min_rel_sharpness)


def pose_distance(c2w: np.ndarray, rot_weight: float = 1.0) -> np.ndarray:
    """(F, 4, 4) -> (F, F): centre distance (m) + rot_weight * rotation angle / 180 deg."""
    centers = c2w[:, :3, 3]
    trans = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    R = c2w[:, :3, :3]
    cos = (np.einsum("aij,bij->ab", R, R) - 1.0) / 2.0   # trace(Ra^T Rb) = cos of relative angle
    angle = np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi
    return trans + rot_weight * angle


def pose_fps(dist: np.ndarray, n: int, start: int) -> list:
    """Greedy farthest-point sampling on a precomputed distance matrix."""
    chosen = [start]
    nearest = dist[start].copy()
    for _ in range(n - 1):
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, dist[nxt])
    return chosen


def row_seed(key: str) -> int:
    return zlib.crc32(key.encode())


def sample_views(pool: ScenePool, keep: np.ndarray, dist: np.ndarray, n: int, key: str):
    """n frames of the pool via pose FPS over kept frames; None if too few.

    The FPS start frame comes from crc32(key), so different rows of one scene
    get different subsets and reruns give the same ones. Result is in time order.
    """
    idx = np.flatnonzero(keep)
    if len(idx) < n:
        return None
    sub = dist[np.ix_(idx, idx)]
    picked = pose_fps(sub, n, row_seed(key) % len(idx))
    return [pool.frames[i] for i in sorted(idx[picked])]


def load_stats(paths) -> dict:
    """frame_stats.py output files -> {(subset, scene): {frame_id: stats}}."""
    out = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                s = json.loads(line)
                out.setdefault((s["subset"], s["scene"]), {})[s["frame_id"]] = s
    return out


def prepare_scene(subset, scene, stats: dict, spar_root, posed_root, thresholds: dict, rot_weight=1.0):
    """Pool restricted to frames with stats, its keep mask and pose distances."""
    pool = load_pool(subset, scene, spar_root, posed_root)
    scene_stats = stats.get((subset, scene), {})
    pool.frames = [f for f in pool.frames if f.frame_id in scene_stats]
    if not pool.frames:
        return pool, np.zeros(0, dtype=bool), np.zeros((0, 0))
    keep = keep_mask([scene_stats[f.frame_id] for f in pool.frames], **thresholds)
    dist = pose_distance(np.stack([f.c2w for f in pool.frames]), rot_weight)
    return pool, keep, dist


def view_fields(pool: ScenePool, frames: list) -> dict:
    return {
        "images": [f.image for f in frames],
        "frame_ids": [f.frame_id for f in frames],
        "intrinsics": [pool.K.round(6).tolist()] * len(frames),
        "c2w": [f.c2w.round(6).tolist() for f in frames],
    }
