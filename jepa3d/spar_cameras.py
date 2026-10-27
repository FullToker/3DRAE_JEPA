"""Per-image camera lookup for SPAR-7M frames.

SPAR stores each scene as
    <scene>/image_color/<id>.jpg            + <scene>/pose/<id>.txt
    <scene>/video_color/frame<k>_<id>.jpg   + <scene>/video_pose/<id>.txt
    <scene>/intrinsic/intrinsic_color.txt   4x4, shared   (scannet, scannetpp)
    <scene>/intrinsic/<id>.txt              3x3, per view (structured3d)

Poses are camera-to-world, OpenCV axes, metres; intrinsics refer to the
stored color image resolution. Verified for scannet and scannetpp with
jepa3d/tools/verify_spar_poses.py (depth reprojection error ~1% / ~5%).
structured3d matched no tested convention, so it is not supported.
"""

from pathlib import Path

import numpy as np

SUPPORTED_SUBSETS = ("scannet", "scannetpp")


def subset_of(image_path: str | Path) -> str:
    # .../spar/<subset>/images/<scene>/<dir>/<file>
    return Path(image_path).parents[3].name


def _frame_files(image_path: Path) -> tuple[Path, str, str]:
    scene_dir = image_path.parents[1]
    if image_path.parent.name == "video_color":
        return scene_dir, "video_pose", image_path.stem.split("_", 1)[1]
    return scene_dir, "pose", image_path.stem


def load_camera(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (K [3,3], c2w [4,4]) as float64 for one SPAR color image."""
    image_path = Path(image_path)
    scene_dir, pose_dir, frame_id = _frame_files(image_path)

    c2w = np.loadtxt(scene_dir / pose_dir / f"{frame_id}.txt").reshape(4, 4)
    per_view = scene_dir / "intrinsic" / f"{frame_id}.txt"
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
    scene_dir, _, frame_id = _frame_files(image_path)
    depth_dir = "video_depth" if image_path.parent.name == "video_color" else "image_depth"
    name = image_path.stem if depth_dir == "video_depth" else frame_id
    return scene_dir / depth_dir / f"{name}.png"
