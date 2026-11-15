#!/usr/bin/env python3
"""Can dino.txt, on its own 2D patches, find a named object in our views?

Gate before distilling dino.txt into a head on 3DRAE tokens (langfeat_v1): if
the teacher cannot ground category names in these frames, the student cannot
either. Nothing is trained here.

Per scene sample (8 posed views, as the geometry probe's validation set) and
per single-instance object:
    text  = dino.txt text embedding of "a photo of a <category>", patch half
    patch = the one patch, over all views, most similar to the text
    error = distance from the object's box centre to the ray through that
            patch's centre, in metres

The ray distance is the position error left with a perfect depth along the
chosen ray, so it measures grounding only. Anchors on the same objects:
    random  mean ray distance over all patches (a patch picked at random)
    oracle  the smallest ray distance of any patch (patch grid + visibility limit)

Usage (from the repo root):
    python jepa3d/tools/check_teacher_grounding.py --stats data/frame_stats/scannet.jsonl \
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl \
        --output experiments/langfeat_teacher_check/results/dinotxt_224x308.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.geo_probe.data import IMAGE_SIZE, GeoProbeDataset, collate
from jepa3d.geo_probe.probe import SceneMetrics
from jepa3d.paths import RAE_ROOT
from jepa3d.x_encoder import unpack_cams

DINOTXT = RAE_ROOT / "checkpoints" / "dinotxt"   # dinov2 checkout + hub/checkpoints/*.pth
PATCH = 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TEMPLATE = "a photo of a {}"
HIT_RADIUS = 0.5   # metres
FREQUENCY_BINS = ((0, 0, "unseen"), (1, 19, "1-19"), (20, 199, "20-199"), (200, 10 ** 9, "200+"))


def load_dinotxt(device):
    """Weights are read from DINOTXT/hub/checkpoints, nothing is downloaded."""
    sys.path.insert(0, str(DINOTXT / "dinov2"))
    torch.hub.set_dir(str(DINOTXT / "hub"))
    from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l
    from dinov2.hub.text.tokenizer import Tokenizer

    model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(device).eval()
    tokenizer = Tokenizer(vocab_path=str(DINOTXT / "hub" / "checkpoints" / "bpe_simple_vocab_16e6.txt.gz"))
    return model, tokenizer


@torch.no_grad()
def embed_names(model, tokenizer, names, device) -> dict:
    """The half of the text embedding that is aligned with patch tokens, unit norm."""
    out = {}
    names = sorted(names)
    for start in range(0, len(names), 64):
        chunk = names[start:start + 64]
        tokens = tokenizer.tokenize([TEMPLATE.format(n) for n in chunk]).to(device)
        features = model.encode_text(tokens).float()
        half = features.shape[-1] // 2
        out.update(zip(chunk, F.normalize(features[:, half:], dim=-1)))
    return out


@torch.no_grad()
def patch_features(model, images) -> torch.Tensor:
    """images (V, 3, H, W) in [0, 1] -> text-aligned patch features (V, h, w, C), unit norm."""
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, patches = model.get_visual_class_and_patch_tokens((images - mean) / std)
    h, w = images.shape[-2] // PATCH, images.shape[-1] // PATCH
    return F.normalize(patches.float(), dim=-1).unflatten(1, (h, w))


def ray_distances(points, K, w2c, size) -> torch.Tensor:
    """Distance from each point to the ray through each patch centre.
    points (O, 3), K (V, 3, 3) at input size, w2c (V, 4, 4), all in one frame
    -> (O, V, h, w). Points behind a camera measure to the camera centre."""
    height, width = size
    h, w = height // PATCH, width // PATCH
    v, u = torch.meshgrid((torch.arange(h, device=K.device) + 0.5) * PATCH,
                          (torch.arange(w, device=K.device) + 0.5) * PATCH, indexing="ij")
    pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(1, -1, 3, 1)            # (1, P, 3, 1)
    rotation_t = w2c[:, :3, :3].transpose(-1, -2)                                            # (V, 3, 3)
    directions = rotation_t[:, None] @ torch.linalg.solve(K[:, None], pixels)                # (V, P, 3, 1)
    directions = F.normalize(directions.squeeze(-1), dim=-1)                                 # (V, P, 3)
    centres = -(rotation_t @ w2c[:, :3, 3:]).squeeze(-1)                                     # (V, 3)
    offset = points[:, None, None, :] - centres[None, :, None, :]                            # (O, V, 1, 3)
    along = (offset * directions[None]).sum(-1).clamp(min=0)                                 # (O, V, P)
    return (offset - along[..., None] * directions[None]).norm(dim=-1).unflatten(-1, (h, w))


def frequency_bin(count: int) -> str:
    return next(name for low, high, name in FREQUENCY_BINS if low <= count <= high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--train-labels", required=True, help="for the category frequency breakdown")
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"),
                        help="teacher input size, multiples of 14")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    size = tuple(args.size)
    if size[0] % PATCH or size[1] % PATCH:
        parser.error("--size must be multiples of 14")

    device = torch.device("cuda")
    with open(args.train_labels) as f:
        frequency = Counter(o["category"] for line in f for o in json.loads(line)["objects"])
    dataset = GeoProbeDataset(args.labels, args.stats, limit=args.limit, seeds=list(range(args.seeds)),
                              memory_size=size)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=collate)
    model, tokenizer = load_dinotxt(device)
    names = {o["category"] for s in dataset.scenes for o in s["objects"]}
    text = embed_names(model, tokenizer, names, device)

    metrics = SceneMetrics()
    for batch in loader:
        images = batch.get("memory_images", batch["images"])[0].to(device)
        K, w2c = unpack_cams(batch.get("memory_cams", batch["cams"])[0].to(device))
        categories = batch["categories"][0]
        scale = batch["scale"][0].item()
        feats = patch_features(model, images)                                               # (V, h, w, C)
        queries = torch.stack([text[c] for c in categories])                                # (O, C)
        similarity = torch.einsum("vhwc,oc->ovhw", feats, queries).flatten(1)               # (O, V*h*w)
        # Positions are metres in the first camera's frame; the cameras are normalized by scale.
        points = batch["positions"][0, :len(categories)].to(device) / scale
        distance = ray_distances(points, K.float(), w2c.float(), size).flatten(1) * scale    # (O, V*h*w), metres
        chosen = distance.gather(1, similarity.argmax(1, keepdim=True)).squeeze(1).cpu().numpy()
        random = distance.mean(1).cpu().numpy()
        random_hit = (distance <= HIT_RADIUS).float().mean(1).cpu().numpy()
        oracle = distance.min(1).values.cpu().numpy()
        in_view = (batch["prompt"][0, :len(categories), 0] >= 0).numpy()
        bins = np.array([frequency_bin(frequency[c]) for c in categories])
        groups = {"all": np.ones(len(categories), dtype=bool), "in_view": in_view,
                  **{f"in_view_freq_{name}": in_view & (bins == name) for _, _, name in FREQUENCY_BINS}}
        for group, keep in groups.items():
            if not keep.any():
                continue
            scene = batch["scene"][0]
            metrics.add(scene, f"{group}/teacher_m", chosen[keep])
            metrics.add(scene, f"{group}/random_m", random[keep])
            metrics.add(scene, f"{group}/oracle_m", oracle[keep])
            metrics.add(scene, f"{group}/teacher_hit_acc", chosen[keep] <= HIT_RADIUS)
            metrics.add(scene, f"{group}/random_hit_acc", random_hit[keep])
            metrics.add(scene, f"{group}/oracle_hit_acc", oracle[keep] <= HIT_RADIUS)

    result = {"teacher": "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l", "size": list(size), "template": TEMPLATE,
              "hit_radius_m": HIT_RADIUS, "samples": len(dataset), "metrics": metrics.summary()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
