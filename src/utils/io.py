from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement


def save_numpy_array_as_image(array: np.ndarray, output_path: str | Path) -> None:
    if array.ndim == 2:
        image = Image.fromarray(array.astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        image = Image.fromarray(array.astype(np.uint8))
    elif array.ndim == 3 and array.shape[0] in (3, 4):
        image = Image.fromarray(array.transpose(1, 2, 0).astype(np.uint8))
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")
    image.save(output_path)


def points_to_colors(points: torch.Tensor, concat: str = "width") -> np.ndarray:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected point maps [N,H,W,3], got {tuple(points.shape)}")
    rgb = points.detach().clone()
    rgb_min = rgb.amin(dim=(1, 2), keepdim=True)
    rgb_max = rgb.amax(dim=(1, 2), keepdim=True)
    rgb = ((rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)).clamp(0, 1)
    rgb = (rgb.float().cpu().numpy() * 255).astype(np.uint8)
    axis = 1 if concat == "width" else 0
    return np.concatenate(list(rgb), axis=axis)


def save_point_map_to_ply(
    point_map: np.ndarray,
    output_path: str | Path,
    color: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    conf: np.ndarray | None = None,
) -> None:
    if point_map.ndim != 4 or point_map.shape[-1] != 3:
        raise ValueError(f"Expected point maps [N,H,W,3], got {point_map.shape}")
    points = point_map.reshape(-1, 3)
    colors = color.reshape(-1, 3) if color is not None else None
    confidence = conf.reshape(-1) if conf is not None else None
    if mask is not None:
        valid = mask.reshape(-1) > 0
        points = points[valid]
        if colors is not None:
            colors = colors[valid]
        if confidence is not None:
            confidence = confidence[valid]

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        fields.extend((("red", "u1"), ("green", "u1"), ("blue", "u1")))
    if confidence is not None:
        fields.append(("confidence", "f4"))
    vertices = np.empty(len(points), dtype=fields)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    if colors is not None:
        colors = colors.astype(np.uint8)
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
    if confidence is not None:
        vertices["confidence"] = confidence
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(output_path))


def save_point_map_tensor_to_ply(
    point_map: torch.Tensor,
    output_path: str | Path,
    conf: torch.Tensor | None = None,
    color: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    percentile: float = 0.0,
) -> None:
    if point_map.ndim != 4 or point_map.shape[-1] != 3:
        raise ValueError(f"Expected point maps [N,H,W,3], got {tuple(point_map.shape)}")
    if conf is not None and conf.ndim != 3:
        raise ValueError(f"Expected confidence [N,H,W], got {tuple(conf.shape)}")
    if color is not None and color.ndim != 4:
        raise ValueError(
            f"Expected colors [N,3,H,W] or [N,H,W,3], got {tuple(color.shape)}"
        )
    if not 0 <= percentile < 1:
        raise ValueError("percentile must be in [0, 1)")

    finite_mask = torch.isfinite(point_map).all(dim=-1)
    if conf is not None:
        finite_mask = finite_mask & torch.isfinite(conf) & (conf > 0)
        if percentile > 0 and finite_mask.any():
            threshold = torch.quantile(conf[finite_mask].float(), percentile)
            finite_mask = finite_mask & (conf >= threshold)
    mask = finite_mask if mask is None else mask.bool() & finite_mask

    if color is not None and color.shape[-1] != 3:
        color = color.permute(0, 2, 3, 1)
    save_point_map_to_ply(
        point_map.detach().float().cpu().numpy(),
        output_path,
        color=(color.detach().float().clamp(0, 1).cpu().numpy() * 255).astype(
            np.uint8
        )
        if color is not None
        else None,
        mask=mask.detach().cpu().numpy(),
        conf=conf.detach().float().cpu().numpy() if conf is not None else None,
    )


def save_decoded_views(
    output_dir: str | Path,
    images: torch.Tensor,
    points: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    confidence_percentile: float = 0.2,
) -> None:
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError("images must have shape [1,S,3,H,W]")
    output_dir = Path(output_dir)
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images[0]):
        rgb = (
            image.detach().float().clamp(0, 1).cpu().numpy() * 255
        ).astype(np.uint8)
        save_numpy_array_as_image(rgb, rgb_dir / f"{index:03d}.png")

    if points is None:
        if confidence is not None:
            raise ValueError("confidence was provided without points")
        return
    if confidence is None:
        raise ValueError("points require confidence")
    if points.ndim != 5 or points.shape[0] != 1:
        raise ValueError("points must have shape [1,S,H,W,3]")

    point_map_dir = output_dir / "point_map"
    point_map_dir.mkdir(parents=True, exist_ok=True)
    for index, point_map in enumerate(points[0]):
        visualization = points_to_colors(point_map.unsqueeze(0))
        save_numpy_array_as_image(
            visualization, point_map_dir / f"{index:03d}.png"
        )
    save_point_map_tensor_to_ply(
        points[0],
        output_dir / "point_cloud.ply",
        color=images[0],
        conf=confidence[0],
        percentile=confidence_percentile,
    )
