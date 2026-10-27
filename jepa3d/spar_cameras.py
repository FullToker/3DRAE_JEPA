"""Per-image camera lookup for SPAR-7M frames.

SPAR stores each scene as
    <scene>/image_color/<id>.jpg
    <scene>/pose/<id>.txt              4x4 camera-to-world
    <scene>/intrinsic/intrinsic_color.txt   4x4, shared   (scannet, scannetpp)
    <scene>/intrinsic/<id>.txt              3x3, per view (structured3d)

Intrinsics refer to the stored color image resolution.
"""

from pathlib import Path

import numpy as np


def _subset(scene_dir: Path) -> str:
    # .../spar/<subset>/images/<scene>
    return scene_dir.parents[1].name


def load_camera(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (K [3,3], c2w [4,4]) as float64 for one SPAR color image."""
    image_path = Path(image_path)
    scene_dir = image_path.parents[1]
    stem = image_path.stem

    c2w = np.loadtxt(scene_dir / "pose" / f"{stem}.txt").reshape(4, 4)
    per_view = scene_dir / "intrinsic" / f"{stem}.txt"
    if per_view.is_file():
        K = np.loadtxt(per_view)
    else:
        K = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_color.txt")
    K = K.reshape(K.shape[0], -1)[:3, :3]
    if not (np.isfinite(c2w).all() and np.isfinite(K).all()):
        raise ValueError(f"Non-finite camera for {image_path}")
    return K, c2w


def depth_path(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    return image_path.parents[1] / "image_depth" / f"{image_path.stem}.png"
