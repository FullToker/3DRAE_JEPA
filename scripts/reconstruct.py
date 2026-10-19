#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from threedrae import (
    CameraGroup,
    FeatureExtractor3D,
    Stage1Decoder,
    TrajectoryGroup,
    launch_3d_viewer,
    load_scene,
    save_decoded_views,
    save_numpy_array_as_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 reconstruction or novel-view synthesis."
    )
    parser.add_argument(
        "--model", choices=("dino2", "dino2-256", "siglip2"), default="dino2"
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--condition-frame-ids", type=int, nargs="+", required=True)
    parser.add_argument("--target-frame-ids", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encoder-precision", choices=("bf16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--decoder-precision", choices=("bf16", "fp32"), default="fp32"
    )
    parser.add_argument("--confidence-percentile", type=float, default=0.2)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-max-points", type=int, default=500_000)
    return parser.parse_args()


def resolve_from(base: Path, path: Path):
    return path if path.is_absolute() else base / path


def main():
    args = parse_args()
    invocation_dir = Path.cwd()
    scene_dir = resolve_from(invocation_dir, args.scene).resolve()
    output_dir = resolve_from(invocation_dir, args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty directory {output_dir}")
    if not 0 <= args.confidence_percentile < 1:
        raise ValueError("--confidence-percentile must be in [0, 1)")
    if not 0 <= args.viser_port <= 65535:
        raise ValueError("--viser-port must be in [0, 65535]")
    if args.viser_max_points <= 0:
        raise ValueError("--viser-max-points must be positive")

    condition = load_scene(scene_dir, args.condition_frame_ids)
    target_ids = args.target_frame_ids or args.condition_frame_ids
    target = load_scene(scene_dir, target_ids)
    encoder = FeatureExtractor3D(args.model, args.device, args.encoder_precision)
    decoder = Stage1Decoder(args.model, args.device, args.decoder_precision)
    latent = encoder(condition.images, condition.intrinsics, condition.extrinsics)
    images, points, confidence = decoder(
        latent,
        target.intrinsics,
        target.extrinsics,
        image_size=tuple(target.images.shape[-2:]),
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
    for index, image in enumerate(condition.images):
        rgb = (image.numpy() * 255).astype(np.uint8)
        save_numpy_array_as_image(rgb, condition_dir / f"{index:03d}.png")

    metadata = {
        "scene": target.name,
        "model": args.model,
        "condition_frame_ids": args.condition_frame_ids,
        "condition_image_names": condition.image_names,
        "target_frame_ids": target_ids,
        "target_image_names": target.image_names,
        "image_size": list(target.images.shape[-2:]),
        "camera_convention": "OpenCV world-to-camera",
        "intrinsics": target.intrinsics.tolist(),
        "extrinsics": target.extrinsics.tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {images.shape[1]} decoded views to {output_dir}")

    if args.viser:
        viewer_images = images.detach().cpu()
        viewer_points = None if points is None else points.detach().cpu()
        viewer_confidence = (
            None if confidence is None else confidence.detach().cpu()
        )
        camera_groups = [
            CameraGroup(
                name="Stage 1 condition",
                intrinsics=condition.intrinsics,
                extrinsics=condition.extrinsics,
                image_size=tuple(condition.images.shape[-2:]),
                color=(255, 70, 55),
                images=condition.images,
            )
        ]
        target_only = [
            index
            for index, frame_id in enumerate(target_ids)
            if frame_id not in args.condition_frame_ids
        ]
        if target_only:
            camera_groups.append(
                CameraGroup(
                    name="Decoder target",
                    intrinsics=target.intrinsics[target_only],
                    extrinsics=target.extrinsics[target_only],
                    image_size=tuple(target.images.shape[-2:]),
                    color=(55, 210, 95),
                    images=viewer_images[0, target_only],
                )
            )
        trajectory_groups = [
            TrajectoryGroup(
                name="Stage 1 condition",
                extrinsics=condition.extrinsics,
                color=(255, 70, 55),
            )
        ]
        if args.target_frame_ids is not None and target_ids != args.condition_frame_ids:
            trajectory_groups.append(
                TrajectoryGroup(
                    name="Decoder target",
                    extrinsics=target.extrinsics,
                    color=(55, 210, 95),
                )
            )
        del encoder, decoder, latent, images, points, confidence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        launch_3d_viewer(
            points=viewer_points,
            colors=viewer_images if viewer_points is not None else None,
            confidence=viewer_confidence,
            point_labels=[f"target {frame_id}" for frame_id in target_ids],
            camera_groups=camera_groups,
            trajectory_groups=trajectory_groups,
            confidence_percentile=args.confidence_percentile * 100,
            max_points=args.viser_max_points,
            port=args.viser_port,
        )


if __name__ == "__main__":
    main()
