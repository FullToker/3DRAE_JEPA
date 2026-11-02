"""Scene samples for the geometry probe: 8 posed views + object positions."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jepa3d.data import load_views
from jepa3d.frame_pool import load_stats, prepare_scene, sample_views
from jepa3d.paths import VJEPA_ROOT
from jepa3d.x_encoder import CAM_DIM

SOURCE = VJEPA_ROOT / "source_data"
# Same frame filter as make_vsibench_manifest.py and the training manifests.
THRESHOLDS = {"min_valid_frac": 0.5, "min_median_depth": 0.6, "min_rel_sharpness": 0.3}
IMAGE_SIZE = (224, 308)


def to_first_camera(centers: np.ndarray, axis_align: np.ndarray, pose_0: np.ndarray) -> np.ndarray:
    """Box centres (N, 3) in the axis-aligned frame -> first camera's frame, metres."""
    transform = np.linalg.inv(axis_align @ pose_0)
    return centers @ transform[:3, :3].T + transform[:3, 3]


class GeoProbeDataset(Dataset):
    """seeds=None draws a new view subset per sample. Otherwise one sample per
    (scene, seed), with the views fixed by that seed."""

    def __init__(self, labels_path, stats_paths, num_views=8, limit=0, seeds=None, min_objects=2):
        stats = load_stats(stats_paths)
        with open(labels_path) as f:
            rows = [json.loads(line) for line in f]
        rows.sort(key=lambda r: r["scene"])
        self.scenes = []
        self.num_views = num_views
        dropped = 0
        for row in rows:
            if limit and len(self.scenes) >= limit:
                break
            prepared = self._prepare(row, stats, num_views, min_objects)
            if prepared is None:
                dropped += 1
            else:
                self.scenes.append(prepared)
        self.samples = [(i, None) for i in range(len(self.scenes))] if seeds is None else [
            (i, seed) for seed in seeds for i in range(len(self.scenes))]
        print(f"{Path(labels_path).name}: {len(self.scenes)} scenes, {dropped} dropped, "
              f"{len(self.samples)} samples", flush=True)

    @staticmethod
    def _prepare(row, stats, num_views, min_objects):
        if len(row["objects"]) < min_objects:
            return None
        try:
            pool, keep, dist = prepare_scene("scannet", row["scene"], stats, SOURCE / "spar",
                                             SOURCE / "scannet" / "posed_images", THRESHOLDS)
        except OSError:
            return None
        if not len(pool.frames):
            return None
        # Visibility is only known for the frames EmbodiedScan lists.
        visible = {int(k): v for k, v in row["visible"].items()}
        keep = keep & np.array([f.frame_id in visible for f in pool.frames])
        if keep.sum() < num_views:
            return None
        return {
            "scene": row["scene"], "axis_align": np.array(row["axis_align"]), "objects": row["objects"],
            "visible": visible, "pool": pool, "keep": keep, "dist": dist,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        index, seed = self.samples[idx]
        scene = self.scenes[index]
        key = f"{scene['scene']}:{random.getrandbits(32) if seed is None else seed}"
        frames = sample_views(scene["pool"], scene["keep"], scene["dist"], self.num_views, key)
        seen = {i for f in frames for i in scene["visible"][f.frame_id]}
        visible = np.array([o["id"] in seen for o in scene["objects"]])

        pose_0 = frames[0].c2w
        c2w = np.stack([f.c2w for f in frames])
        images, K, w2c = load_views({"image_paths": [f.image for f in frames],
                                     "intrinsics": [scene["pool"].K.tolist()] * len(frames),
                                     "c2w": c2w.tolist()}, IMAGE_SIZE)
        centers = np.array([o["center"] for o in scene["objects"]])
        relative = np.linalg.inv(pose_0) @ c2w
        return {
            "scene": scene["scene"],
            "images": images.clamp(0, 1),
            "cams": torch.cat([K.flatten(1), w2c.flatten(1)], dim=1),
            "categories": [o["category"] for o in scene["objects"]],
            "positions": torch.from_numpy(to_first_camera(centers, scene["axis_align"], pose_0)).float(),
            "visible": torch.from_numpy(visible),
            # Same scale as jepa3d.data.normalize_cameras divides the poses by.
            "scale": float(np.linalg.norm(relative[:, :3, 3], axis=-1).max()),
            # Raw ScanNet world z (up) in the first camera's frame.
            "up": torch.from_numpy(np.linalg.inv(pose_0)[:3, 2].copy()).float(),
        }


def collate(batch):
    objects = max(len(s["categories"]) for s in batch)
    positions = torch.zeros(len(batch), objects, 3)
    visible = torch.zeros(len(batch), objects, dtype=torch.bool)
    mask = torch.zeros(len(batch), objects, dtype=torch.bool)
    for i, s in enumerate(batch):
        n = len(s["categories"])
        positions[i, :n] = s["positions"]
        visible[i, :n] = s["visible"]
        mask[i, :n] = True
    images = torch.stack([s["images"] for s in batch])
    cams = torch.stack([s["cams"] for s in batch])
    assert cams.shape[-1] == CAM_DIM
    return {
        "scene": [s["scene"] for s in batch],
        "images": images,
        "cams": cams,
        "categories": [s["categories"] for s in batch],
        "positions": positions,
        "visible": visible,
        "mask": mask,
        "scale": torch.tensor([s["scale"] for s in batch], dtype=torch.float32),
        "up": torch.stack([s["up"] for s in batch]),
    }
