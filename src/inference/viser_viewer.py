from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class CameraGroup:
    name: str
    intrinsics: torch.Tensor | np.ndarray
    extrinsics: torch.Tensor | np.ndarray
    image_size: tuple[int, int]
    color: tuple[int, int, int]
    images: torch.Tensor | np.ndarray | None = None


@dataclass
class TrajectoryGroup:
    name: str
    extrinsics: torch.Tensor | np.ndarray
    color: tuple[int, int, int]


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _camera_to_world(extrinsics):
    extrinsics = _to_numpy(extrinsics).astype(np.float64, copy=False)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"Expected extrinsics [S,4,4], got {extrinsics.shape}")
    if not np.isfinite(extrinsics).all():
        raise ValueError("Camera extrinsics contain non-finite values")
    return np.linalg.inv(extrinsics)


def _as_images(images, count):
    if images is None:
        return None
    images = _to_numpy(images)
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or images.shape[0] != count:
        raise ValueError(f"Expected {count} images, got {images.shape}")
    if images.shape[1] in (3, 4):
        images = images.transpose(0, 2, 3, 1)
    if images.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA images, got {images.shape}")
    if np.issubdtype(images.dtype, np.floating):
        images = np.clip(images, 0, 1) * 255
    return images.astype(np.uint8)


def _prepare_point_cloud(points, colors, confidence):
    if points is None:
        if colors is not None or confidence is not None:
            raise ValueError("Point colors/confidence were provided without points")
        return None
    if colors is None or confidence is None:
        raise ValueError("Points require colors and confidence")
    points = _to_numpy(points)
    colors = _to_numpy(colors)
    confidence = _to_numpy(confidence)
    if points.ndim == 5 and points.shape[0] == 1:
        points = points[0]
    if colors.ndim == 5 and colors.shape[0] == 1:
        colors = colors[0]
    if confidence.ndim == 4 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected points [S,H,W,3], got {points.shape}")
    if colors.ndim != 4:
        raise ValueError(f"Expected colors [S,3,H,W] or [S,H,W,3], got {colors.shape}")
    if colors.shape[1] in (3, 4):
        colors = colors.transpose(0, 2, 3, 1)
    if colors.shape != points.shape:
        raise ValueError(f"Expected colors [S,H,W,3], got {colors.shape}")
    if confidence.shape != points.shape[:-1]:
        raise ValueError(
            f"Expected confidence {points.shape[:-1]}, got {confidence.shape}"
        )
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(colors, 0, 1) * 255
    return (
        points.astype(np.float32, copy=False),
        colors.astype(np.uint8),
        confidence.astype(np.float32, copy=False),
    )


def _safe_name(name):
    return "_".join(name.lower().split())


def launch_3d_viewer(
    *,
    points=None,
    colors=None,
    confidence=None,
    point_labels: Sequence[str] | None = None,
    camera_groups: Sequence[CameraGroup] = (),
    trajectory_groups: Sequence[TrajectoryGroup] = (),
    confidence_percentile: float = 20.0,
    max_points: int = 500_000,
    port: int = 8080,
    block: bool = True,
):
    import viser
    import viser.transforms as viser_tf

    if not 0 <= confidence_percentile < 100:
        raise ValueError("confidence_percentile must be in [0, 100)")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if not 0 <= port <= 65535:
        raise ValueError("port must be in [0, 65535]")

    point_data = _prepare_point_cloud(points, colors, confidence)
    camera_data = []
    all_centers = []
    for group in camera_groups:
        c2w = _camera_to_world(group.extrinsics)
        intrinsics = _to_numpy(group.intrinsics).astype(np.float64, copy=False)
        if intrinsics.shape != (len(c2w), 3, 3):
            raise ValueError(
                f"Camera group {group.name} has invalid intrinsics {intrinsics.shape}"
            )
        images = _as_images(group.images, len(c2w))
        camera_data.append((group, c2w, intrinsics, images))
        all_centers.append(c2w[:, :3, 3])

    trajectory_data = []
    for group in trajectory_groups:
        c2w = _camera_to_world(group.extrinsics)
        trajectory_data.append((group, c2w))
        all_centers.append(c2w[:, :3, 3])

    if point_data is not None:
        point_maps, point_colors, point_confidence = point_data
        finite_points = np.isfinite(point_maps).all(axis=-1)
        if finite_points.any():
            scene_center = np.median(point_maps[finite_points], axis=0)
        elif all_centers:
            scene_center = np.concatenate(all_centers).mean(axis=0)
        else:
            raise ValueError("No finite points or cameras to visualize")
    elif all_centers:
        scene_center = np.concatenate(all_centers).mean(axis=0)
    else:
        raise ValueError("At least one point cloud or camera is required")

    scale_samples = []
    if point_data is not None and finite_points.any():
        centered_valid = point_maps[finite_points] - scene_center
        scale_samples.append(np.linalg.norm(centered_valid, axis=-1))
    if all_centers:
        camera_distances = np.linalg.norm(
            np.concatenate(all_centers) - scene_center, axis=-1
        )
        scale_samples.append(camera_distances)
    scene_scale = max(
        float(np.percentile(np.concatenate(scale_samples), 90)), 1e-3
    )
    frustum_scale = scene_scale * 0.04

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(
        titlebar_content=None,
        control_layout="collapsible",
        control_width="medium",
    )
    server.gui.add_markdown(
        "**Camera colors**  \n"
        "🔴 Appearance / Stage 1 condition  \n"
        "🔵 Trajectory-only condition  \n"
        "🟢 Decoder-only target"
    )

    camera_handles = []
    up_directions = []
    for group, c2w, intrinsics, images in camera_data:
        height, width = group.image_size
        if height <= 0 or width <= 0:
            raise ValueError(f"Camera group {group.name} has invalid image size")
        for index in range(len(c2w)):
            pose = c2w[index].copy()
            pose[:3, 3] -= scene_center
            transform = viser_tf.SE3.from_matrix(pose)
            fy = intrinsics[index, 1, 1]
            if not np.isfinite(fy) or fy <= 0:
                raise ValueError(f"Camera group {group.name} has invalid focal length")
            frustum = server.scene.add_camera_frustum(
                f"/cameras/{_safe_name(group.name)}/{index:03d}",
                fov=float(2 * np.arctan2(height / 2, fy)),
                aspect=float(width / height),
                scale=frustum_scale,
                color=group.color,
                image=None if images is None else images[index],
                wxyz=transform.rotation().wxyz,
                position=transform.translation(),
            )

            @frustum.on_click
            def _(_, handle=frustum):
                for client in server.get_clients().values():
                    with client.atomic():
                        client.camera.wxyz = handle.wxyz
                        client.camera.position = handle.position

            camera_handles.append(frustum)
            up_directions.append(pose[:3, :3] @ np.array([0.0, -1.0, 0.0]))

    trajectory_handles = []
    for group, c2w in trajectory_data:
        centers = c2w[:, :3, 3] - scene_center
        if len(centers) < 2:
            continue
        segments = np.stack([centers[:-1], centers[1:]], axis=1)
        trajectory_handles.append(
            server.scene.add_line_segments(
                f"/trajectories/{_safe_name(group.name)}",
                points=segments.astype(np.float32),
                colors=group.color,
                thickness=scene_scale * 0.003,
            )
        )

    if up_directions:
        average_up = np.mean(up_directions, axis=0)
        norm = np.linalg.norm(average_up)
        if norm > 1e-6:
            server.scene.set_up_direction(tuple(average_up / norm))

    show_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
    show_trajectories = server.gui.add_checkbox(
        "Show trajectories", initial_value=True
    )
    camera_size = server.gui.add_slider(
        "Camera size",
        min=scene_scale * 0.01,
        max=scene_scale * 0.15,
        step=scene_scale * 0.005,
        initial_value=frustum_scale,
    )

    @show_cameras.on_update
    def _(_):
        for handle in camera_handles:
            handle.visible = show_cameras.value

    @show_trajectories.on_update
    def _(_):
        for handle in trajectory_handles:
            handle.visible = show_trajectories.value

    @camera_size.on_update
    def _(_):
        for handle in camera_handles:
            handle.scale = camera_size.value

    if point_data is not None:
        view_count, height, width = point_maps.shape[:3]
        if point_labels is None:
            point_labels = [str(index) for index in range(view_count)]
        if len(point_labels) != view_count or len(set(point_labels)) != view_count:
            raise ValueError("point_labels must be unique and match point-map views")

        flat_points = (point_maps - scene_center).reshape(-1, 3)
        flat_colors = point_colors.reshape(-1, 3)
        flat_confidence = point_confidence.reshape(-1)
        frame_indices = np.repeat(np.arange(view_count), height * width)
        valid = np.isfinite(flat_points).all(axis=-1)
        valid &= np.isfinite(flat_confidence) & (flat_confidence > 0)
        if not valid.any():
            raise ValueError("Point cloud has no finite positive-confidence points")

        confidence_slider = server.gui.add_slider(
            "Filter lowest confidence (%)",
            min=0.0,
            max=99.0,
            step=0.1,
            initial_value=float(confidence_percentile),
        )
        frame_selector = server.gui.add_dropdown(
            "Show points from view",
            options=("All", *point_labels),
            initial_value="All",
        )
        point_size = server.gui.add_slider(
            "Point size",
            min=scene_scale * 0.0002,
            max=scene_scale * 0.01,
            step=scene_scale * 0.0002,
            initial_value=scene_scale * 0.001,
        )
        show_points = server.gui.add_checkbox("Show point cloud", initial_value=True)
        label_to_index = {label: index for index, label in enumerate(point_labels)}

        def selected_indices():
            threshold = np.percentile(
                flat_confidence[valid], confidence_slider.value
            )
            mask = valid & (flat_confidence >= threshold)
            if frame_selector.value != "All":
                mask &= frame_indices == label_to_index[frame_selector.value]
            indices = np.flatnonzero(mask)
            if len(indices) > max_points:
                keep = np.linspace(0, len(indices) - 1, max_points).astype(np.int64)
                indices = indices[keep]
            return indices

        indices = selected_indices()
        point_cloud = server.scene.add_point_cloud(
            "/point_cloud",
            points=flat_points[indices],
            colors=flat_colors[indices],
            point_size=point_size.value,
            point_shape="circle",
        )

        def update_points():
            indices = selected_indices()
            with server.atomic():
                point_cloud.points = flat_points[indices]
                point_cloud.colors = flat_colors[indices]

        @confidence_slider.on_update
        def _(_):
            update_points()

        @frame_selector.on_update
        def _(_):
            update_points()

        @point_size.on_update
        def _(_):
            point_cloud.point_size = point_size.value

        @show_points.on_update
        def _(_):
            point_cloud.visible = show_points.value

    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.position = (
        scene_scale * 1.5,
        -scene_scale * 1.5,
        scene_scale * 1.5,
    )
    server.flush()
    print(f"Viser viewer: http://localhost:{server.get_port()}")
    if block:
        try:
            server.sleep_forever()
        except KeyboardInterrupt:
            server.stop()
    return server
