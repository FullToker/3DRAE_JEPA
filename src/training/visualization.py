from pathlib import Path

import numpy as np
import torch
from PIL import Image


def mask_invisible_views(images, visible_ids):
    visible = torch.zeros_like(images)
    if visible_ids.numel() > 0:
        visible[visible_ids] = images[visible_ids]
    return visible


def _image_row(images):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [S,3,H,W] images, got {tuple(images.shape)}")
    images = (
        images.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        * 255
    ).astype(np.uint8)
    return np.concatenate(list(images), axis=1)


def save_training_visualization(path, condition, target, prediction):
    rows = [_image_row(images) for images in (condition, target, prediction)]
    maximum_width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] < maximum_width:
            row = np.pad(
                row,
                ((0, 0), (0, maximum_width - row.shape[1]), (0, 0)),
                constant_values=255,
            )
        padded.append(row)
    separator = np.full((4, maximum_width, 3), 255, dtype=np.uint8)
    visualization = np.concatenate(
        [padded[0], separator, padded[1], separator, padded[2]], axis=0
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(path)
