import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class SceneInput:
    name: str
    images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    frame_indices: torch.Tensor
    image_names: list[str]


def _image_paths(scene_dir: Path):
    image_dir = scene_dir / "images"
    paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in image_dir.glob(pattern)
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return paths


def _select_frames(paths, frame_ids: Optional[list[int]]):
    if frame_ids is None:
        frame_ids = list(range(len(paths)))
    if not frame_ids:
        raise ValueError("At least one frame is required")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("frame_ids must not contain duplicates")
    invalid = [index for index in frame_ids if index < 0 or index >= len(paths)]
    if invalid:
        raise IndexError(f"Frame indices out of range: {invalid}")
    return [paths[index] for index in frame_ids], frame_ids


def load_scene(
    scene_dir: str | Path,
    frame_ids: Optional[list[int]] = None,
) -> SceneInput:
    scene_dir = Path(scene_dir)
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("pad_pixel", 0) != 0:
        raise ValueError("Release inference expects scenes with pad_pixel=0")

    selected_paths, frame_ids = _select_frames(_image_paths(scene_dir), frame_ids)
    images, intrinsics, extrinsics = [], [], []
    expected_size = None
    for path in selected_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if expected_size is None:
                expected_size = image.size
            elif image.size != expected_size:
                raise ValueError("All scene images must have the same resolution")
            array = np.array(image, dtype=np.float32, copy=True) / 255.0
        try:
            intrinsics.append(metadata["intrinsics"][path.name])
            extrinsics.append(metadata["extrinsics"][path.name])
        except KeyError as error:
            raise KeyError(f"Missing camera metadata for {path.name}") from error
        images.append(torch.from_numpy(array).permute(2, 0, 1))

    images = torch.stack(images)
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
    height, width = images.shape[-2:]
    if metadata.get("height", height) != height or metadata.get("width", width) != width:
        raise ValueError("metadata.json resolution does not match the image files")
    if intrinsics.shape != (len(images), 3, 3):
        raise ValueError(f"Invalid intrinsics shape: {tuple(intrinsics.shape)}")
    if extrinsics.shape != (len(images), 4, 4):
        raise ValueError(f"Invalid extrinsics shape: {tuple(extrinsics.shape)}")
    if not all(torch.isfinite(tensor).all() for tensor in (images, intrinsics, extrinsics)):
        raise ValueError("Scene input contains non-finite values")

    return SceneInput(
        name=metadata.get("sequence_name", scene_dir.name),
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        frame_indices=torch.tensor(frame_ids, dtype=torch.long),
        image_names=[path.name for path in selected_paths],
    )
