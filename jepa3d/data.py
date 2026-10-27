"""Posed SPAR dataset + collate for the 3DRAE X-encoder.

Rows come from jepa3d/tools/make_spar_manifest.py (unique images with
"intrinsics" and "c2w"). Camera preprocessing follows 3DRAE's training
pipeline: world frame moved to the first view, translations rescaled so the
farthest camera centre sits at distance 1.

The batch keeps fusion_gv's key names so GVJEPATrainer runs unchanged:
    images_vggt -> images (B, V, 3, H, W) in [0, 1]
    images_jepa -> cams   (B, V, 25), see jepa3d.x_encoder
"""

import functools
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
from fusion_gv.gvjepa_trainer import GVJEPADataset
from jepa3d.x_encoder import CAM_DIM
from utils.camera import resize_intrinsics


class PosedSparDataset(GVJEPADataset):
    def __init__(self, manifest_path, max_views: int | None = None):
        super().__init__(manifest_path, num_frames=None)
        self.max_views = max_views

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._read_row(idx)
        n = len(row["images"]) if self.max_views is None else self.max_views
        return {
            "image_paths": row["images"][:n],
            "intrinsics": row["intrinsics"][:n],
            "c2w": row["c2w"][:n],
            "query": row.get("query", ""),
            "target": row["target"],
            "group_id": row.get("group_id"),
            "row": row,
        }


def normalize_cameras(c2w: torch.Tensor) -> torch.Tensor:
    """c2w (S, 4, 4) -> w2c (S, 4, 4), first view = identity, max centre distance = 1."""
    w2c = torch.linalg.inv(c2w)
    w2c = w2c @ c2w[0]
    centers = torch.linalg.inv(w2c)[:, :3, 3]
    scale = centers.norm(dim=-1).max()
    if scale > 1e-6:
        w2c[:, :3, 3] /= scale
    return w2c


def load_views(sample, image_size):
    height, width = image_size
    images, Ks = [], []
    for path, K in zip(sample["image_paths"], sample["intrinsics"]):
        with Image.open(path) as image:
            image = image.convert("RGB")
            source = (image.height, image.width)
            image = image.resize((width, height), Image.BICUBIC)
        images.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
        Ks.append(resize_intrinsics(torch.tensor(K, dtype=torch.float64), source, image_size))
    w2c = normalize_cameras(torch.tensor(sample["c2w"], dtype=torch.float64))
    return torch.stack(images), torch.stack(Ks).float(), w2c.float()


def spar3d_collate(batch: List[Dict[str, Any]], image_size=(224, 308)) -> Dict[str, Any]:
    V = max(len(s["image_paths"]) for s in batch)
    images = torch.zeros(len(batch), V, 3, *image_size)
    cams = torch.zeros(len(batch), V, CAM_DIM)
    cams[..., 9:] = torch.eye(4).flatten()   # padded slots: K = 0, w2c = I
    for i, sample in enumerate(batch):
        imgs, K, w2c = load_views(sample, image_size)
        n = len(imgs)
        images[i, :n] = imgs.clamp(0, 1)
        cams[i, :n, :9] = K.flatten(1)
        cams[i, :n, 9:] = w2c.flatten(1)
    return {
        "images_vggt": images,
        "images_jepa": cams,
        "query": [s["query"] for s in batch],
        "target": [s["target"] for s in batch],
        "boxes": [None] * len(batch),
        "group_id": [s["group_id"] for s in batch],
        "rows": [s["row"] for s in batch],
    }


def build_loader(manifests, cfg: dict, batch_size: int, train: bool = True, limit: int = 0) -> DataLoader:
    dcfg = cfg["data"]
    datasets = [PosedSparDataset(p, dcfg.get("max_views")) for p in manifests]
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if limit:
        ds = Subset(ds, range(min(limit, len(ds))))
    num_workers = dcfg.get("num_workers" if train else "val_num_workers", 4 if train else 2)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "drop_last": train,
        "num_workers": num_workers,
        "pin_memory": dcfg.get("pin_memory", True),
        "collate_fn": functools.partial(spar3d_collate, image_size=tuple(dcfg["image_size"])),
    }
    if num_workers > 0 and train:
        kwargs["persistent_workers"] = dcfg.get("persistent_workers", True)
        kwargs["prefetch_factor"] = dcfg.get("prefetch_factor", 2)
    return DataLoader(ds, **kwargs)
