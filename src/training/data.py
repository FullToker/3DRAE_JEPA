import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms

from utils.camera import plucker_rays_from_K_w2c


def valid_resolutions(config):
    patch_size = int(config["patch_size"])
    minimum_aspect = float(config["min_aspect_ratio"])
    maximum_aspect = float(config["max_aspect_ratio"])
    minimum_pixels = int(config["min_pixel_count"])
    maximum_pixels = int(config["max_pixel_count"])
    resolutions = []
    maximum_side = math.ceil(math.sqrt(maximum_pixels * maximum_aspect))
    for width in range(patch_size, maximum_side + patch_size, patch_size):
        minimum_height = math.ceil(
            max(width / maximum_aspect, minimum_pixels / width) / patch_size
        ) * patch_size
        maximum_height = math.floor(
            min(width / minimum_aspect, maximum_pixels / width) / patch_size
        ) * patch_size
        for height in range(minimum_height, maximum_height + 1, patch_size):
            pixels = width * height
            aspect = width / height
            valid_pixels = minimum_pixels <= pixels <= maximum_pixels
            valid_aspect = minimum_aspect <= aspect <= maximum_aspect
            if valid_pixels and valid_aspect:
                resolutions.append((width, height))
    if not resolutions:
        raise ValueError("Dataset configuration produced no valid resolutions")
    return sorted(set(resolutions))


def _sample_frames(
    frame_count,
    view_count,
    minimum_interval,
    maximum_interval,
    fixed_interval_probability,
    rng,
):
    if frame_count < 1:
        raise ValueError("A scene must contain at least one frame")
    if frame_count < view_count:
        return sorted(rng.choice(frame_count, view_count, replace=True).tolist())
    if view_count == 1:
        return [int(rng.integers(0, frame_count))]
    reference = int(rng.integers(0, frame_count - view_count + 1))
    remaining = frame_count - 1 - reference
    maximum_interval = min(
        maximum_interval, max(1, 2 * remaining // (view_count - 1))
    )
    minimum_interval = min(minimum_interval, maximum_interval)
    if rng.random() < 0.5 and rng.random() < fixed_interval_probability:
        maximum_stride = min(maximum_interval, remaining // (view_count - 1))
        stride = int(rng.integers(1, max(maximum_stride + 1, 2)))
        return [reference + stride * index for index in range(view_count)]

    intervals = rng.integers(
        minimum_interval, maximum_interval + 1, size=view_count - 1
    )
    positions = [reference]
    for interval in intervals:
        candidate = positions[-1] + int(interval)
        if candidate < frame_count:
            positions.append(candidate)
    candidates = [
        index for index in range(reference, frame_count) if index not in positions
    ]
    missing = view_count - len(positions)
    if missing:
        positions.extend(
            rng.choice(
                candidates, missing, replace=len(candidates) < missing
            ).tolist()
        )
    return sorted(positions)


def _novel_positions(view_count):
    inner = list(range(1, view_count - 1))
    count = (len(inner) + 2) // 2
    if count <= 1:
        return inner[:count]
    indices = [round(index * (len(inner) - 1) / (count - 1)) for index in range(count)]
    return [inner[index] for index in indices]


def _canonicalize(extrinsics):
    return extrinsics @ torch.linalg.inv(extrinsics[0])


def _resize_and_crop(images, intrinsics, resolution, focal_crop, crop_jitter, rng):
    if focal_crop > 0:
        crop_scale = focal_crop + (1 - focal_crop) * rng.beta(0.5, 0.5)
        input_width, input_height = images.shape[-1], images.shape[-2]
        output_width = int(np.floor(input_width * crop_scale))
        output_height = int(np.floor(input_height * crop_scale))
        left = (input_width - output_width) // 2
        top = (input_height - output_height) // 2
        images = images[..., top : top + output_height, left : left + output_width]
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 2] -= left
        intrinsics[:, 1, 2] -= top

    target_width, target_height = resolution
    expanded = int(rng.integers(0, crop_jitter)) if crop_jitter > 1 else 0
    requested_width = target_width + expanded
    requested_height = target_height + expanded
    input_width, input_height = images.shape[-1], images.shape[-2]
    scale = max(requested_width / input_width, requested_height / input_height) + 1e-8
    scaled_width = int(np.floor(input_width * scale))
    scaled_height = int(np.floor(input_height * scale))
    images = F.interpolate(
        images,
        size=(scaled_height, scaled_width),
        mode="bilinear",
        align_corners=False,
    )
    intrinsics = intrinsics.clone()
    fractional_x = (input_width * scale - scaled_width) / 2
    fractional_y = (input_height * scale - scaled_height) / 2
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] + 0.5) * scale - fractional_x - 0.5
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] + 0.5) * scale - fractional_y - 0.5
    intrinsics[:, 0, :2] *= scale
    intrinsics[:, 1, :2] *= scale

    left = int(round((scaled_width - target_width) / 2))
    top = int(round((scaled_height - target_height) / 2))
    images = images[..., top : top + target_height, left : left + target_width]
    intrinsics[:, 0, 2] -= left
    intrinsics[:, 1, 2] -= top
    if images.shape[-2:] != (target_height, target_width):
        raise RuntimeError("Image crop produced an unexpected resolution")
    return images, intrinsics


class ScanNetDirectoryDataset(Dataset):
    def __init__(self, config, resolutions, seed):
        self.path = Path(config["path"])
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        self.split = config.get("split", "train")
        self.split_path = self.path / self.split
        if not self.split_path.is_dir():
            raise KeyError(f"Split '{self.split}' is missing from {self.path}")
        self.scenes = sorted(
            path.name
            for path in self.split_path.iterdir()
            if path.is_dir() and (path / "metadata.json").is_file()
        )
        if not self.scenes:
            raise ValueError(f"No scenes found in {self.split_path}")
        self.metadata = {}
        for scene in self.scenes:
            metadata = json.loads(
                (self.split_path / scene / "metadata.json").read_text()
            )
            if int(metadata["sequence_length"]) < 1:
                raise ValueError(f"Invalid sequence_length for {self.split}/{scene}")
            self.metadata[scene] = metadata
        self.repeat = int(config.get("repeat", 1))
        self.resolutions = resolutions
        self.minimum_interval = int(config["min_interval"])
        self.maximum_interval = int(config["max_interval"])
        self.fixed_interval_probability = float(
            config.get("fix_interval_prob", 0.5)
        )
        self.novel_view_ratio = float(config.get("novel_view_ratio", 0))
        self.focal_crop = float(config.get("aug_focal", 0))
        self.crop_jitter = int(config.get("aug_crop", 0))
        self.seed = int(seed)
        augmentations = []
        color_probability = float(config.get("aug_color_jitter_prob", 0))
        if color_probability > 0:
            augmentations.append(
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=float(config.get("aug_color_jitter_brightness", 0)),
                            contrast=float(config.get("aug_color_jitter_contrast", 0)),
                            saturation=float(config.get("aug_color_jitter_saturation", 0)),
                            hue=float(config.get("aug_color_jitter_hue", 0)),
                        )
                    ],
                    p=color_probability,
                )
            )
        grayscale_probability = float(config.get("aug_grayscale_prob", 0))
        if grayscale_probability > 0:
            augmentations.append(transforms.RandomGrayscale(grayscale_probability))
        blur_probability = float(config.get("aug_gaussian_blur_prob", 0))
        if blur_probability > 0:
            augmentations.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(5, sigma=(0.1, 1.0))],
                    p=blur_probability,
                )
            )
        self.augmentation = transforms.Compose(augmentations) if augmentations else None

    def __len__(self):
        return len(self.scenes) * self.repeat

    def _load(self, scene, view_count, visible_count, resolution_id, rng):
        scene_path = self.split_path / scene
        metadata = self.metadata[scene]
        frame_count = int(metadata["sequence_length"])
        novel_count = (
            max(int(self.novel_view_ratio * view_count), 1)
            if self.novel_view_ratio > 0
            else 0
        )
        total_count = view_count + novel_count
        minimum_interval = max(1, round(self.minimum_interval / (1 + self.novel_view_ratio)))
        maximum_interval = max(1, round(self.maximum_interval / (1 + self.novel_view_ratio)))
        frame_ids = _sample_frames(
            frame_count,
            total_count,
            minimum_interval,
            maximum_interval,
            self.fixed_interval_probability,
            rng,
        )
        unique_ids = sorted(set(frame_ids))
        recovery = [unique_ids.index(index) for index in frame_ids]
        digits = int(metadata.get("frame_digits", 6))
        extension = metadata.get("image_extension", ".png")
        loaded_images = []
        for index in unique_ids:
            image_path = scene_path / "images" / f"{index:0{digits}d}{extension}"
            with Image.open(image_path) as image:
                loaded_images.append(np.array(image.convert("RGB"), dtype=np.uint8))
        images = torch.from_numpy(np.stack(loaded_images)[recovery]).float()
        images = images.permute(0, 3, 1, 2) / 255
        stored_intrinsics = np.load(scene_path / "intrinsics.npy", mmap_mode="r")
        stored_extrinsics = np.load(scene_path / "extrinsics.npy", mmap_mode="r")
        intrinsics = torch.from_numpy(
            np.array(stored_intrinsics[unique_ids][recovery], copy=True)
        ).float()
        extrinsics = torch.from_numpy(
            np.array(stored_extrinsics[unique_ids][recovery], copy=True)
        ).float()
        pad = int(metadata.get("pad_pixel", 0))
        if pad > 0:
            images = images[:, :, pad:-pad]
            intrinsics[:, 1, 2] -= pad
        extrinsics = _canonicalize(extrinsics)
        images, intrinsics = _resize_and_crop(
            images,
            intrinsics,
            self.resolutions[resolution_id],
            self.focal_crop,
            self.crop_jitter,
            rng,
        )
        if self.augmentation is not None:
            images = self.augmentation(images)

        if novel_count:
            novel_positions = _novel_positions(total_count)
            context_positions = [
                index for index in range(total_count) if index not in novel_positions
            ]
            target_images = images[novel_positions]
            target_intrinsics = intrinsics[novel_positions]
            target_extrinsics = extrinsics[novel_positions]
            target_ids = [frame_ids[index] for index in novel_positions]
            images = images[context_positions]
            intrinsics = intrinsics[context_positions]
            extrinsics = extrinsics[context_positions]
            frame_ids = [frame_ids[index] for index in context_positions]
        else:
            target_images = target_intrinsics = target_extrinsics = target_ids = None

        camera_to_world = torch.linalg.inv(extrinsics)
        camera_distance = camera_to_world[:, :3, 3].norm(dim=-1).max()
        scale_augmentation = torch.rand(()) * 1.5 + 0.5
        scene_scale = torch.clamp(camera_distance / scale_augmentation, min=0.1)
        extrinsics[:, :3, 3] /= scene_scale
        if target_extrinsics is not None:
            target_extrinsics[:, :3, 3] /= scene_scale

        height, width = images.shape[-2:]
        rays = plucker_rays_from_K_w2c(intrinsics, extrinsics, height, width)
        if visible_count == 0:
            visible_ids = torch.empty(0, dtype=torch.long)
        elif visible_count == view_count:
            visible_ids = torch.arange(view_count)
        else:
            visible_ids = torch.linspace(0, view_count - 1, visible_count).round().long()
        result = {
            "scene_name": scene,
            "frame_ids": frame_ids,
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "rays": rays,
            "visible_ids": visible_ids,
        }
        if target_images is not None:
            result.update(
                {
                    "target_frame_ids": target_ids,
                    "target_images": target_images,
                    "target_intrinsics": target_intrinsics,
                    "target_extrinsics": target_extrinsics,
                    "target_rays": plucker_rays_from_K_w2c(
                        target_intrinsics, target_extrinsics, height, width
                    ),
                }
            )
        return result

    def __getitem__(self, item):
        index, view_count, visible_count, resolution_id = item
        worker = torch.utils.data.get_worker_info()
        worker_seed = worker.seed if worker is not None else self.seed
        for attempt in range(min(len(self.scenes), 8)):
            scene_index = (index + attempt) % len(self.scenes)
            scene = self.scenes[scene_index]
            rng = np.random.default_rng(worker_seed + index * 997 + attempt)
            try:
                return self._load(
                    scene, view_count, visible_count, resolution_id, rng
                )
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError):
                if attempt + 1 == min(len(self.scenes), 8):
                    raise
        raise RuntimeError("Unable to load a valid ScanNet sample")


class DynamicBatchSampler(Sampler):
    def __init__(self, sampler, config, resolutions, seed):
        self.sampler = sampler
        self.config = config
        self.resolutions = resolutions
        self.seed = int(seed)
        self.epoch = 0
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.sampler.set_epoch(epoch)
        self.rng = random.Random(self.seed + epoch)

    def __iter__(self):
        indices = iter(self.sampler)
        while True:
            resolution_id = self.rng.randrange(len(self.resolutions))
            width, height = self.resolutions[resolution_id]
            tokens_per_image = width * height // int(self.config["patch_size"]) ** 2
            maximum_views = int(self.config["max_token_per_gpu"]) // tokens_per_image
            minimum_views = int(self.config["min_sample_len"])
            maximum_views = min(int(self.config["max_sample_len"]), maximum_views)
            if maximum_views < minimum_views:
                raise ValueError("max_token_per_gpu cannot fit the minimum view count")
            view_count = self.rng.randint(minimum_views, maximum_views)
            visible_count = view_count
            if self.rng.random() < float(self.config.get("mask_prob", 0)):
                mask = self.rng.uniform(
                    float(self.config.get("min_mask_ratio", 0)),
                    float(self.config.get("max_mask_ratio", 0)),
                )
                visible_count = round(view_count * (1 - mask))
            if self.rng.random() < float(self.config.get("all_mask_prob", 0)):
                visible_count = 0
            tokens_per_sample = view_count * tokens_per_image
            batch_size = int(self.config["max_token_per_gpu"]) // tokens_per_sample
            maximum_batch = int(self.config.get("max_batch_size", -1))
            if maximum_batch > 0:
                batch_size = min(batch_size, maximum_batch)
            batch = []
            for _ in range(max(batch_size, 1)):
                try:
                    index = next(indices)
                except StopIteration:
                    break
                batch.append((index, view_count, visible_count, resolution_id))
            if not batch:
                return
            yield batch

    def __len__(self):
        return len(self.sampler)


def _collate(batch):
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return result


def _seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_scannet_loader(config, rank, world_size, seed, num_workers):
    resolutions = valid_resolutions(config)
    dataset = ScanNetDirectoryDataset(config, resolutions, seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    batch_sampler = DynamicBatchSampler(sampler, config, resolutions, seed)
    kwargs = {}
    if num_workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        worker_init_fn=_seed_worker,
        **kwargs,
    )
    return loader, batch_sampler
