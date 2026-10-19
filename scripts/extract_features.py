#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from threedrae import FeatureExtractor3D, load_scene


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the FeatureExtractor3D API on a calibrated multi-view scene."
    )
    parser.add_argument(
        "--model", choices=("dino2", "dino2-256", "siglip2"), default="dino2"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--frame-ids", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def resolve_from(base: Path, path: Path):
    return path if path.is_absolute() else base / path


def main():
    args = parse_args()
    invocation_dir = Path.cwd()
    scene_dir = resolve_from(invocation_dir, args.scene).resolve()
    config_path = (
        resolve_from(invocation_dir, args.config).resolve()
        if args.config is not None
        else None
    )
    model = FeatureExtractor3D(
        model_name=args.model,
        device=args.device,
        precision=args.precision,
        config_path=config_path,
    )
    scene = load_scene(scene_dir, args.frame_ids)
    latent_feature_3d = model(
        images=scene.images,
        intrinsics=scene.intrinsics,
        extrinsics=scene.extrinsics,
    )
    print(
        f"Extracted {tuple(latent_feature_3d.shape)} latent tokens "
        f"for scene {scene.name!r} on {latent_feature_3d.device}"
    )


if __name__ == "__main__":
    main()
