#!/usr/bin/env python3
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from threedrae import (
    CameraGroup,
    SceneGenerator3D,
    Stage1Decoder,
    TrajectoryGroup,
    launch_3d_viewer,
    load_scene,
    save_decoded_views,
    save_numpy_array_as_image,
)
from utils import canonicalize_w2c, orbit_w2c_from_first


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 3D scene latent with DINOv2-conditioned 3DDiT."
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument(
        "--appearance-condition-frame-ids", type=int, nargs="+", required=True
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("frames", "synthetic-orbit"),
        default="frames",
    )
    parser.add_argument("--trajectory-condition-frame-ids", type=int, nargs="+")
    parser.add_argument("--synthetic-trajectory-views", type=int, default=6)
    parser.add_argument("--synthetic-trajectory-baseline", type=float, default=1.5)
    parser.add_argument(
        "--synthetic-trajectory-arc-degrees", type=float, default=60.0
    )
    parser.add_argument("--target-frame-ids", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--decoder-precision", choices=("bf16", "fp32"), default="fp32"
    )
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence-percentile", type=float, default=0.2)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-max-points", type=int, default=500_000)
    return parser.parse_args()


def resolve_from(base: Path, path: Path):
    return path if path.is_absolute() else base / path


def main():
    args = parse_args()
    if len(args.appearance_condition_frame_ids) not in (1, 2, 6):
        raise ValueError(
            "3DDiT inference supports exactly 1, 2, or 6 appearance condition views"
        )
    if len(args.appearance_condition_frame_ids) != len(
        set(args.appearance_condition_frame_ids)
    ):
        raise ValueError("appearance condition frame IDs must not contain duplicates")
    if not 0 <= args.confidence_percentile < 1:
        raise ValueError("--confidence-percentile must be in [0, 1)")
    if not 0 <= args.viser_port <= 65535:
        raise ValueError("--viser-port must be in [0, 65535]")
    if args.viser_max_points <= 0:
        raise ValueError("--viser-max-points must be positive")
    if args.target_frame_ids is not None and len(args.target_frame_ids) != len(
        set(args.target_frame_ids)
    ):
        raise ValueError("target frame IDs must not contain duplicates")

    if args.trajectory_mode == "frames":
        if args.trajectory_condition_frame_ids is None:
            raise ValueError(
                "--trajectory-condition-frame-ids is required in frames mode"
            )
        if len(args.trajectory_condition_frame_ids) != len(
            set(args.trajectory_condition_frame_ids)
        ):
            raise ValueError("trajectory condition frame IDs must not contain duplicates")
        if args.target_frame_ids is None:
            raise ValueError("--target-frame-ids is required in frames mode")
        trajectory_frame_ids = list(args.trajectory_condition_frame_ids)
        missing_appearance_ids = [
            index
            for index in args.appearance_condition_frame_ids
            if index not in trajectory_frame_ids
        ]
        if missing_appearance_ids:
            warnings.warn(
                "appearance condition frame IDs must have camera poses in the trajectory; "
                f"automatically appending missing IDs {missing_appearance_ids}",
                stacklevel=2,
            )
            trajectory_frame_ids.extend(missing_appearance_ids)
    else:
        if args.trajectory_condition_frame_ids is not None:
            raise ValueError(
                "--trajectory-condition-frame-ids cannot be used in synthetic-orbit mode"
            )
        if args.target_frame_ids is not None:
            raise ValueError("--target-frame-ids cannot be used in synthetic-orbit mode")
        if len(args.appearance_condition_frame_ids) != 1:
            raise ValueError("synthetic-orbit mode supports exactly one appearance view")
        trajectory_frame_ids = None
        missing_appearance_ids = []

    invocation_dir = Path.cwd()
    scene_dir = resolve_from(invocation_dir, args.scene).resolve()
    output_dir = resolve_from(invocation_dir, args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty directory {output_dir}")

    appearance = load_scene(scene_dir, args.appearance_condition_frame_ids)
    source_trajectory_extrinsics = None
    if args.trajectory_mode == "frames":
        trajectory = load_scene(scene_dir, trajectory_frame_ids)
        target = load_scene(scene_dir, args.target_frame_ids)
        trajectory_images = trajectory.images
        trajectory_intrinsics = trajectory.intrinsics
        trajectory_extrinsics = trajectory.extrinsics
        target_intrinsics = target.intrinsics
        target_extrinsics = target.extrinsics
        source_target_extrinsics = target.extrinsics
        target_view_indices = None
        image_size = tuple(target.images.shape[-2:])
        visible_ids = [
            trajectory_frame_ids.index(index)
            for index in args.appearance_condition_frame_ids
        ]
        scene_name = trajectory.name
    else:
        first_w2c = appearance.extrinsics[0]
        source_trajectory_extrinsics = orbit_w2c_from_first(
            first_w2c,
            view_count=args.synthetic_trajectory_views,
            maximum_baseline=args.synthetic_trajectory_baseline,
            arc_degrees=args.synthetic_trajectory_arc_degrees,
        )
        trajectory_extrinsics = canonicalize_w2c(
            source_trajectory_extrinsics, first_w2c
        )
        trajectory_images = appearance.images[:1].repeat(
            args.synthetic_trajectory_views, 1, 1, 1
        )
        trajectory_intrinsics = appearance.intrinsics[:1].repeat(
            args.synthetic_trajectory_views, 1, 1
        )
        target_intrinsics = trajectory_intrinsics
        target_extrinsics = trajectory_extrinsics
        source_target_extrinsics = source_trajectory_extrinsics
        target_view_indices = list(range(args.synthetic_trajectory_views))
        image_size = tuple(appearance.images.shape[-2:])
        visible_ids = [0]
        scene_name = appearance.name

    generator = SceneGenerator3D(args.device, args.precision)
    decoder = Stage1Decoder("dino2", args.device, args.decoder_precision)
    latent = generator(
        trajectory_images,
        trajectory_intrinsics,
        trajectory_extrinsics,
        visible_ids=torch.tensor(visible_ids, dtype=torch.long),
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
    )
    images, points, confidence = decoder(
        latent,
        target_intrinsics,
        target_extrinsics,
        image_size=image_size,
    )
    save_decoded_views(
        output_dir,
        images,
        points,
        confidence,
        confidence_percentile=args.confidence_percentile,
    )

    condition_dir = output_dir / "condition_rgb"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(appearance.images):
        rgb = (image.numpy() * 255).astype(np.uint8)
        save_numpy_array_as_image(rgb, condition_dir / f"{index:03d}.png")

    metadata = {
        "scene": scene_name,
        "trajectory_mode": args.trajectory_mode,
        "appearance_condition_frame_ids": args.appearance_condition_frame_ids,
        "trajectory_condition_frame_ids": trajectory_frame_ids,
        "auto_added_appearance_frame_ids": missing_appearance_ids,
        "synthetic_trajectory": (
            {
                "views": args.synthetic_trajectory_views,
                "maximum_baseline": args.synthetic_trajectory_baseline,
                "arc_degrees": args.synthetic_trajectory_arc_degrees,
                "source_first_extrinsics": appearance.extrinsics[0].tolist(),
                "source_extrinsics": source_trajectory_extrinsics.tolist(),
            }
            if source_trajectory_extrinsics is not None
            else None
        ),
        "trajectory_intrinsics": trajectory_intrinsics.tolist(),
        "trajectory_extrinsics": trajectory_extrinsics.tolist(),
        "visible_trajectory_indices": visible_ids,
        "target_frame_ids": args.target_frame_ids,
        "synthetic_target_view_indices": target_view_indices,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "image_size": list(image_size),
        "camera_convention": "OpenCV world-to-camera",
        "target_intrinsics": target_intrinsics.tolist(),
        "target_extrinsics": target_extrinsics.tolist(),
        "source_target_extrinsics": source_target_extrinsics.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Generated {images.shape[1]} target views from "
        f"{len(args.appearance_condition_frame_ids)} appearance views over "
        f"{len(trajectory_extrinsics)} {args.trajectory_mode} trajectory views "
        f"at {output_dir}"
    )

    if args.viser:
        viewer_images = images.detach().cpu()
        viewer_points = points.detach().cpu()
        viewer_confidence = confidence.detach().cpu()
        camera_groups = [
            CameraGroup(
                name="Appearance condition",
                intrinsics=trajectory_intrinsics[visible_ids],
                extrinsics=trajectory_extrinsics[visible_ids],
                image_size=image_size,
                color=(255, 70, 55),
                images=appearance.images,
            )
        ]
        trajectory_only = [
            index
            for index in range(len(trajectory_extrinsics))
            if index not in visible_ids
        ]
        if trajectory_only:
            camera_groups.append(
                CameraGroup(
                    name="Trajectory condition",
                    intrinsics=trajectory_intrinsics[trajectory_only],
                    extrinsics=trajectory_extrinsics[trajectory_only],
                    image_size=image_size,
                    color=(60, 130, 255),
                )
            )
        if args.trajectory_mode == "frames":
            target_only = [
                index
                for index, frame_id in enumerate(args.target_frame_ids)
                if frame_id not in trajectory_frame_ids
            ]
            if target_only:
                camera_groups.append(
                    CameraGroup(
                        name="Decoder target",
                        intrinsics=target_intrinsics[target_only],
                        extrinsics=target_extrinsics[target_only],
                        image_size=image_size,
                        color=(55, 210, 95),
                        images=viewer_images[0, target_only],
                    )
                )
        trajectory_groups = [
            TrajectoryGroup(
                name="Trajectory condition",
                extrinsics=trajectory_extrinsics,
                color=(60, 130, 255),
            )
        ]
        if args.trajectory_mode == "frames" and len(target_extrinsics) > 1:
            trajectory_groups.append(
                TrajectoryGroup(
                    name="Decoder target",
                    extrinsics=target_extrinsics,
                    color=(55, 210, 95),
                )
            )
        point_labels = (
            [f"target {frame_id}" for frame_id in args.target_frame_ids]
            if args.trajectory_mode == "frames"
            else [f"synthetic {index}" for index in target_view_indices]
        )
        del generator, decoder, latent, images, points, confidence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        launch_3d_viewer(
            points=viewer_points,
            colors=viewer_images,
            confidence=viewer_confidence,
            point_labels=point_labels,
            camera_groups=camera_groups,
            trajectory_groups=trajectory_groups,
            confidence_percentile=args.confidence_percentile * 100,
            max_points=args.viser_max_points,
            port=args.viser_port,
        )


if __name__ == "__main__":
    main()
