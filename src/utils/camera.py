import math

import torch
import torch.nn.functional as F


def patch_aligned_size(image_size, patch_size):
    height, width = (int(value) for value in image_size)
    patch_size = int(patch_size)
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    return (
        max(patch_size, height // patch_size * patch_size),
        max(patch_size, width // patch_size * patch_size),
    )


def resize_intrinsics(intrinsics, source_size, target_size):
    source_height, source_width = source_size
    target_height, target_width = target_size
    if (source_height, source_width) == (target_height, target_width):
        return intrinsics
    scaled = intrinsics.clone()
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled[..., 0, :2] *= scale_x
    scaled[..., 1, :2] *= scale_y
    scaled[..., 0, 2] = (intrinsics[..., 0, 2] + 0.5) * scale_x - 0.5
    scaled[..., 1, 2] = (intrinsics[..., 1, 2] + 0.5) * scale_y - 0.5
    return scaled


def canonicalize_w2c(extrinsics, reference=None):
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"Expected extrinsics [S,4,4], got {tuple(extrinsics.shape)}")
    reference = extrinsics[0] if reference is None else reference
    if reference.shape != (4, 4):
        raise ValueError(f"Expected reference [4,4], got {tuple(reference.shape)}")
    if not torch.isfinite(extrinsics).all() or not torch.isfinite(reference).all():
        raise ValueError("Camera extrinsics contain non-finite values")
    return extrinsics @ torch.linalg.inv(reference)


def orbit_w2c_from_first(
    first_w2c,
    view_count=6,
    maximum_baseline=1.5,
    arc_degrees=60.0,
):
    if first_w2c.shape != (4, 4):
        raise ValueError(f"Expected first_w2c [4,4], got {tuple(first_w2c.shape)}")
    if not torch.isfinite(first_w2c).all():
        raise ValueError("first_w2c contains non-finite values")
    if view_count < 2:
        raise ValueError("view_count must be at least 2")
    if not math.isfinite(maximum_baseline) or maximum_baseline <= 0:
        raise ValueError("maximum_baseline must be positive")
    if not math.isfinite(arc_degrees) or not 0 < abs(arc_degrees) < 180:
        raise ValueError("abs(arc_degrees) must be in (0, 180)")

    dtype = first_w2c.dtype
    device = first_w2c.device
    arc = torch.tensor(math.radians(arc_degrees), dtype=dtype, device=device)
    radius = maximum_baseline / (2 * torch.sin(arc.abs() / 2))
    angles = torch.linspace(0, 1, view_count, dtype=dtype, device=device) * arc
    centers = torch.stack(
        [
            radius * torch.sin(angles),
            torch.zeros_like(angles),
            radius * (1 - torch.cos(angles)),
        ],
        dim=-1,
    )
    focus = torch.stack(
        [
            torch.zeros_like(radius),
            torch.zeros_like(radius),
            radius,
        ]
    )
    forward = F.normalize(focus.unsqueeze(0) - centers, dim=-1)
    down = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand_as(
        forward
    )
    right = F.normalize(torch.cross(down, forward, dim=-1), dim=-1)
    down = torch.cross(forward, right, dim=-1)
    camera_to_world_rotation = torch.stack([right, down, forward], dim=-1)
    world_to_camera_rotation = camera_to_world_rotation.transpose(-1, -2)

    canonical = torch.eye(4, dtype=dtype, device=device).repeat(view_count, 1, 1)
    canonical[:, :3, :3] = world_to_camera_rotation
    canonical[:, :3, 3] = -torch.bmm(
        world_to_camera_rotation, centers.unsqueeze(-1)
    ).squeeze(-1)
    return canonical @ first_w2c.unsqueeze(0)


def plucker_rays_from_K_w2c(
    K, w2c, height, width, pixel_center=True, normalize_dir=True, flatten=False
):
    device = K.device
    views = K.shape[0]
    u = torch.arange(width, device=device, dtype=K.dtype)
    v = torch.arange(height, device=device, dtype=K.dtype)
    if pixel_center:
        u, v = u + 0.5, v + 0.5
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    # Preserve the pixel ordering used to train the released checkpoints.
    uu = uu.T
    vv = vv.T
    uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)

    K_inv = torch.linalg.inv(K).view(views, 1, 1, 3, 3)
    pixels = uv1.view(1, height, width, 3, 1).expand(
        views, height, width, 3, 1
    )
    directions_camera = (K_inv @ pixels).squeeze(-1)
    if normalize_dir:
        directions_camera = F.normalize(directions_camera, dim=-1, eps=1e-8)

    rotation = w2c[:, :3, :3]
    translation = w2c[:, :3, 3]
    rotation_t = rotation.transpose(-1, -2)
    directions_world = (
        rotation_t.view(views, 1, 1, 3, 3)
        @ directions_camera.unsqueeze(-1)
    ).squeeze(-1)
    centers_world = -(rotation_t @ translation.unsqueeze(-1)).squeeze(-1)
    centers_world = centers_world.view(views, 1, 1, 3).expand(
        views, height, width, 3
    )
    moments = torch.cross(centers_world, directions_world, dim=-1)
    rays = torch.cat([directions_world, moments], dim=-1)
    return rays.reshape(views, height * width, 6) if flatten else rays
