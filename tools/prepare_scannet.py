#!/usr/bin/env python3
import argparse
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert exported ScanNet scenes to frame-based training data."
    )
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/scannet"))
    parser.add_argument("--max-image-side", type=int, default=560)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-frames", type=int, default=18)
    parser.add_argument("--png-compression", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def read_scene_list(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes:
        raise ValueError(f"No scenes found in {path}")
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"Duplicate scene names in {path}")
    return scenes


def indexed_images(path):
    images = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for image_path in path.glob(pattern):
            try:
                frame_id = int(image_path.stem)
            except ValueError:
                continue
            if frame_id in images:
                raise ValueError(f"Duplicate frame ID {frame_id} under {path}")
            images[frame_id] = image_path
    return images


def indexed_poses(path):
    poses = {}
    for pose_path in path.glob("*.txt"):
        try:
            frame_id = int(pose_path.stem)
        except ValueError:
            continue
        poses[frame_id] = pose_path
    return poses


def load_world_to_camera(path):
    camera_to_world = np.loadtxt(path, dtype=np.float32)
    if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
        return None
    try:
        world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    except np.linalg.LinAlgError:
        return None
    return world_to_camera if np.isfinite(world_to_camera).all() else None


def output_size(width, height, maximum_side):
    if maximum_side <= 0 or max(width, height) <= maximum_side:
        return width, height
    scale = maximum_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def scaled_intrinsics(path, input_size, resized_size):
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid color intrinsics: {path}")
    input_width, input_height = input_size
    output_width, output_height = resized_size
    matrix = matrix.copy()
    matrix[0] *= output_width / input_width
    matrix[1] *= output_height / input_height
    return matrix


def scene_is_complete(scene_dir):
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    frame_count = int(metadata["sequence_length"])
    digits = int(metadata.get("frame_digits", 6))
    extension = metadata.get("image_extension", ".png")
    images_complete = all(
        (scene_dir / "images" / f"{index:0{digits}d}{extension}").is_file()
        for index in range(frame_count)
    )
    return (
        images_complete
        and (scene_dir / "intrinsics.npy").is_file()
        and (scene_dir / "extrinsics.npy").is_file()
        and (scene_dir / "frame_ids.npy").is_file()
    )


def write_image(source_path, output_path, input_size, resized_size, compression):
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if image.size != input_size:
            raise ValueError(f"Inconsistent image size: {source_path}")
        if image.size != resized_size:
            image = image.resize(resized_size, Image.Resampling.LANCZOS)
        image.save(output_path, compress_level=compression)


def prepare_scene(
    scans_root,
    output_root,
    split,
    scene_name,
    maximum_side,
    frame_stride,
    minimum_frames,
    compression,
    workers,
):
    scene_dir = scans_root / scene_name
    image_map = indexed_images(scene_dir / "color")
    pose_map = indexed_poses(scene_dir / "pose")
    frame_ids = sorted(set(image_map) & set(pose_map))[::frame_stride]
    records = []
    for frame_id in frame_ids:
        extrinsic = load_world_to_camera(pose_map[frame_id])
        if extrinsic is not None:
            records.append((frame_id, image_map[frame_id], extrinsic))
    if len(records) < minimum_frames:
        raise ValueError(
            f"{scene_name} has {len(records)} valid frames, "
            f"expected at least {minimum_frames}"
        )

    output_scene = output_root / split / scene_name
    partial_scene = output_root / split / f".{scene_name}.partial"
    if output_scene.exists():
        if scene_is_complete(output_scene):
            return "skipped"
        raise FileExistsError(f"Incomplete output scene already exists: {output_scene}")
    if partial_scene.exists():
        raise FileExistsError(
            f"Partial output already exists: {partial_scene}. Remove it before retrying."
        )

    with Image.open(records[0][1]) as image:
        input_size = image.size
    resized_size = output_size(*input_size, maximum_side)
    width, height = resized_size
    intrinsic = scaled_intrinsics(
        scene_dir / "intrinsic" / "intrinsic_color.txt",
        input_size,
        resized_size,
    )

    image_dir = partial_scene / "images"
    image_dir.mkdir(parents=True)
    pending = deque()
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
    try:
        for index, (_, image_path, _) in enumerate(records):
            output_path = image_dir / f"{index:06d}.png"
            if executor is None:
                write_image(
                    image_path, output_path, input_size, resized_size, compression
                )
                continue
            pending.append(
                executor.submit(
                    write_image,
                    image_path,
                    output_path,
                    input_size,
                    resized_size,
                    compression,
                )
            )
            if len(pending) >= workers * 2:
                pending.popleft().result()
        while pending:
            pending.popleft().result()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    frame_count = len(records)
    np.save(
        partial_scene / "intrinsics.npy",
        np.broadcast_to(intrinsic, (frame_count, 3, 3)).copy(),
    )
    np.save(
        partial_scene / "extrinsics.npy",
        np.stack([record[2] for record in records]),
    )
    np.save(
        partial_scene / "frame_ids.npy",
        np.asarray([record[0] for record in records], dtype=np.int64),
    )
    metadata = {
        "format_version": 1,
        "split": split,
        "scene_name": scene_name,
        "sequence_length": frame_count,
        "width": width,
        "height": height,
        "pad_pixel": 0,
        "frame_digits": 6,
        "image_extension": ".png",
        "has_depth": False,
    }
    (partial_scene / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    os.replace(partial_scene, output_scene)
    return "prepared"


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    if args.min_frames < 1:
        raise ValueError("--min-frames must be positive")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be in [0, 9]")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes must be positive")

    scans_root = args.scans_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not scans_root.is_dir():
        raise FileNotFoundError(scans_root)
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits = {"train": read_scene_list(args.train_list)}
    if args.test_list is not None:
        splits["test"] = read_scene_list(args.test_list)

    prepared = skipped = 0
    for split, scene_names in splits.items():
        if args.max_scenes is not None:
            scene_names = scene_names[: args.max_scenes]
        (output_root / split).mkdir(exist_ok=True)
        for scene_name in tqdm(scene_names, desc=split, unit="scene"):
            status = prepare_scene(
                scans_root,
                output_root,
                split,
                scene_name,
                args.max_image_side,
                args.frame_stride,
                args.min_frames,
                args.png_compression,
                args.workers,
            )
            prepared += status == "prepared"
            skipped += status == "skipped"
    print(f"Finished: prepared={prepared}, skipped={skipped}, output={output_root}")


if __name__ == "__main__":
    main()
