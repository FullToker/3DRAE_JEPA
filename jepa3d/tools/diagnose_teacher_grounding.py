#!/usr/bin/env python3
"""Why does dino.txt miss objects in check_teacher_grounding.py: the model or the measurement?

Same samples as the check (EmbodiedScan val ScanNet scenes, 8 posed views,
single-instance objects), nothing is trained. Each object is scored under every
combination of
    text half   patch (the half aligned with patch tokens) | cls (the other half, a control)
    template    single ("a photo of a {}") | ensemble (TEMPLATES averaged)
    score       raw       cosine similarity of patch and text
                centered  raw minus the patch's mean similarity over the vocabulary
                margin    raw minus the patch's best other vocabulary entry
The vocabulary is every category of the train and val labels plus the room
structure classes, so "centered" and "margin" ask which name fits a patch best,
as in the dino.txt segmentation recipe, instead of which patch fits a name best.

Measures per object and variant:
    centre_m   distance from the box centre to the ray of the top patch (as in the check)
    in_box     the ray of the top patch passes through the object's 3D box
    in_box5    any of the 5 top patches does
    auc        probability that a patch whose ray passes through the box outscores one that does not
A ray through the box may be blocked by something in front, so in_box is lenient
where centre_m is strict. Anchors: box_frac (a random patch) and box_any (an
oracle patch).

Writes <output>.json (grouped summary), <output>.objects.jsonl (one row per
object) and, with --dump N, <output>.examples.npz (images and score maps of N
samples).

Usage (from the repo root):
    python jepa3d/tools/diagnose_teacher_grounding.py --stats data/frame_stats/scannet.jsonl \
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl \
        --output experiments/langfeat_teacher_check/results/diagnose_224x308
"""

import argparse
import itertools
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.geo_probe.data import IMAGE_SIZE, GeoProbeDataset
from jepa3d.geo_probe.labels import EXCLUDED, INFOS
from jepa3d.geo_probe.probe import SceneMetrics
from jepa3d.tools.check_teacher_grounding import (FREQUENCY_BINS, HIT_RADIUS, PATCH, TEMPLATE, frequency_bin,
                                                  load_dinotxt, patch_features, ray_distances)
from jepa3d.x_encoder import unpack_cams

TEMPLATES = (TEMPLATE, "a photo of the {}", "a cropped photo of a {}", "a close-up photo of a {}",
             "a blurry photo of a {}", "a low resolution photo of a {}", "a photo of a {} in a room", "{}")
HALVES = ("patch", "cls")
PROMPTS = ("single", "ensemble")
SCORES = ("raw", "centered", "margin")
SIZE_BINS = ((0.0, 0.5, "lt0.5m"), (0.5, 1.0, "0.5-1m"), (1.0, 1e9, "gt1m"))   # longest box edge
TOP_K = 5
DUMP_VARIANTS = ("patch/single/raw", "patch/ensemble/margin")


def euler_zxy(angles: np.ndarray) -> np.ndarray:
    """EmbodiedScan box angles (alpha, beta, gamma) -> rotation Rz(alpha) Rx(beta) Ry(gamma)."""
    (ca, cb, cg), (sa, sb, sg) = np.cos(angles), np.sin(angles)
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    ry = np.array([[cg, 0, sg], [0, 1, 0], [-sg, 0, cg]])
    return rz @ rx @ ry


def load_boxes(split: str) -> dict:
    """{scene: {bbox_id: 9 numbers (centre, size, angles)}} in the axis-aligned frame."""
    with open(str(INFOS).format(split), "rb") as f:
        infos = pickle.load(f)
    return {x["sample_idx"].split("/")[1]: {int(i["bbox_id"]): np.asarray(i["bbox_3d"], dtype=np.float64)
                                            for i in x["instances"]}
            for x in infos["data_list"] if x["sample_idx"].startswith("scannet/")}


class BoxDataset(GeoProbeDataset):
    """Adds each object's box in the first camera's frame, metres: centre (O, 3),
    half size (O, 3), rotation box -> camera (O, 3, 3)."""

    def __init__(self, *args, boxes, **kwargs):
        super().__init__(*args, **kwargs)
        self.boxes = boxes

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        scene = self.scenes[self.samples[idx][0]]
        to_camera = np.linalg.inv(scene["axis_align"] @ sample["pose_0"])              # aligned -> first camera
        boxes = np.stack([self.boxes[scene["scene"]][o["id"]] for o in scene["objects"]])
        rotation = np.stack([to_camera[:3, :3] @ euler_zxy(b[6:9]) for b in boxes])
        sample["box_center"] = torch.from_numpy(boxes[:, :3] @ to_camera[:3, :3].T + to_camera[:3, 3]).float()
        sample["box_half"] = torch.from_numpy(boxes[:, 3:6] / 2).float()
        sample["box_rotation"] = torch.from_numpy(rotation).float()
        return sample


def patch_rays(K, w2c, size):
    """Camera centres (V, 3) and unit directions (V, P, 3) of the patch-centre rays,
    in the frame the cameras are given in. As in ray_distances."""
    height, width = size
    h, w = height // PATCH, width // PATCH
    v, u = torch.meshgrid((torch.arange(h, device=K.device) + 0.5) * PATCH,
                          (torch.arange(w, device=K.device) + 0.5) * PATCH, indexing="ij")
    pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(1, -1, 3, 1)
    rotation_t = w2c[:, :3, :3].transpose(-1, -2)
    directions = F.normalize((rotation_t[:, None] @ torch.linalg.solve(K[:, None], pixels)).squeeze(-1), dim=-1)
    centres = -(rotation_t @ w2c[:, :3, 3:]).squeeze(-1)
    return centres, directions


def rays_hit_boxes(origins, directions, center, half, rotation) -> torch.Tensor:
    """origins (V, 3), directions (V, P, 3), boxes center (O, 3), half (O, 3),
    rotation (O, 3, 3) box -> world, all in one frame -> (O, V, P) bool: the ray
    passes through the box in front of the camera (slab test)."""
    o = torch.einsum("oji,ovj->ovi", rotation, origins[None] - center[:, None])           # (O, V, 3)
    d = torch.einsum("oji,vpj->ovpi", rotation, directions)                               # (O, V, P, 3)
    d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t1 = (-half[:, None, None] - o[:, :, None]) / d
    t2 = (half[:, None, None] - o[:, :, None]) / d
    near = torch.minimum(t1, t2).amax(-1)
    far = torch.maximum(t1, t2).amin(-1)
    return (near <= far) & (far > 0)


@torch.no_grad()
def embed_vocabulary(model, tokenizer, names, device) -> dict:
    """{(half, prompt): (N, C) unit text embeddings}, rows ordered as names."""
    half_dim = None
    per_template = []
    for template in TEMPLATES:
        chunks = []
        for start in range(0, len(names), 64):
            tokens = tokenizer.tokenize([template.format(n) for n in names[start:start + 64]]).to(device)
            chunks.append(model.encode_text(tokens).float())
        per_template.append(torch.cat(chunks))
        half_dim = per_template[-1].shape[-1] // 2
    stacked = torch.stack(per_template)                                                   # (T, N, 2C)
    out = {}
    for half, part in (("cls", stacked[..., :half_dim]), ("patch", stacked[..., half_dim:])):
        part = F.normalize(part, dim=-1)
        out[half, "single"] = part[0]
        out[half, "ensemble"] = F.normalize(part.mean(0), dim=-1)
    return out


def score_maps(similarity: torch.Tensor, rows: torch.Tensor) -> dict:
    """similarity (N, P) of every vocabulary entry with every patch, rows (O,) the
    objects' entries -> {score name: (O, P)}."""
    raw = similarity[rows]
    top2 = similarity.topk(2, dim=0).values                                               # (2, P)
    best_other = torch.where(raw >= top2[0], top2[1], top2[0])
    return {"raw": raw, "centered": raw - similarity.mean(0), "margin": raw - best_other}


def auc(scores: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """(O, P) scores, (O, P) bool -> (O,) rank AUC, nan without both classes."""
    ranks = scores.argsort(1).argsort(1).float() + 1
    n_pos = positive.sum(1).float()
    n_neg = positive.shape[1] - n_pos
    value = ((ranks * positive).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return torch.where((n_pos > 0) & (n_neg > 0), value, torch.full_like(value, float("nan")))


def first(batch):
    return batch[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--train-labels", required=True, help="vocabulary and category frequency")
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="path prefix of the result files")
    parser.add_argument("--split", default="val", help="EmbodiedScan split the labels come from")
    parser.add_argument("--size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dump", type=int, default=12, help="samples whose images and score maps are saved")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    size = tuple(args.size)
    if size[0] % PATCH or size[1] % PATCH:
        parser.error("--size must be multiples of 14")

    device = torch.device("cuda")
    with open(args.train_labels) as f:
        frequency = Counter(o["category"] for line in f for o in json.loads(line)["objects"])
    dataset = BoxDataset(args.labels, args.stats, limit=args.limit, seeds=list(range(args.seeds)),
                         memory_size=size, boxes=load_boxes(args.split))
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=first)
    model, tokenizer = load_dinotxt(device)
    names = sorted(set(frequency) | {o["category"] for s in dataset.scenes for o in s["objects"]} | EXCLUDED)
    index = {name: i for i, name in enumerate(names)}
    text = embed_vocabulary(model, tokenizer, names, device)
    variants = ["/".join(v) for v in itertools.product(HALVES, PROMPTS, SCORES)]

    metrics = SceneMetrics()
    rows_out, dump, dumped = [], {}, 0
    dump_every = max(1, len(dataset) // max(1, args.dump))
    for step, sample in enumerate(loader):
        images = sample.get("memory_images", sample["images"]).to(device)
        K, w2c = unpack_cams(sample.get("memory_cams", sample["cams"]).to(device))
        K, w2c = K.float(), w2c.float()
        categories, scale = sample["categories"], sample["scale"]
        feats = patch_features(model, images).flatten(0, 2)                                # (P, C), P = V*h*w
        points = sample["positions"].to(device)
        distance = ray_distances(points / scale, K, w2c, size).flatten(1) * scale          # (O, P), metres
        origins, directions = patch_rays(K, w2c, size)
        in_box = rays_hit_boxes(origins * scale, directions, sample["box_center"].to(device),
                                sample["box_half"].to(device), sample["box_rotation"].to(device)).flatten(1)
        object_rows = torch.tensor([index[c] for c in categories], device=device)

        in_view = (sample["prompt"][:, 0] >= 0).numpy()
        box_any = in_box.any(1).cpu().numpy()
        longest = (2 * sample["box_half"].max(1).values).numpy()
        per_object = [{"scene": sample["scene"], "category": c, "train_count": frequency[c],
                       "in_view": bool(in_view[i]), "box_any": bool(box_any[i]), "longest_m": float(longest[i]),
                       "box_frac": float(in_box[i].float().mean()),
                       "centre_in_box": bool(in_box[i, distance[i].argmin()])} for i, c in enumerate(categories)]
        keep_dump = step % dump_every == 0 and dumped < args.dump
        for half, prompt in itertools.product(HALVES, PROMPTS):
            maps = score_maps(text[half, prompt] @ feats.T, object_rows)
            for score, values in maps.items():
                variant = f"{half}/{prompt}/{score}"
                top = values.topk(TOP_K, dim=1).indices                                    # (O, K)
                centre = distance.gather(1, top[:, :1]).squeeze(1).cpu().numpy()
                hits = in_box.gather(1, top).cpu().numpy()
                areas = auc(values, in_box).cpu().numpy()
                for i, row in enumerate(per_object):
                    row[variant] = {"centre_m": float(centre[i]), "in_box": bool(hits[i, 0]),
                                    "in_box5": bool(hits[i].any()), "auc": float(areas[i])}
                if keep_dump and variant in DUMP_VARIANTS:
                    dump[f"{step}:{variant}"] = values.float().cpu().numpy().astype(np.float16)
        if keep_dump:
            dumped += 1
            dump[f"{step}:images"] = (images.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy()
            dump[f"{step}:in_box"] = in_box.cpu().numpy()
            dump[f"{step}:categories"] = np.array(categories)
        rows_out.extend(per_object)

        bins = np.array([frequency_bin(frequency[c]) for c in categories])
        sizes = np.array([next(name for low, high, name in SIZE_BINS if low <= x < high) for x in longest])
        groups = {"all": np.ones(len(categories), dtype=bool), "in_view": in_view, "box_visible": box_any,
                  **{f"in_view_freq_{name}": in_view & (bins == name) for _, _, name in FREQUENCY_BINS},
                  **{f"in_view_size_{name}": in_view & (sizes == name) for _, _, name in SIZE_BINS}}
        for group, keep in groups.items():
            if not keep.any():
                continue
            picked = [row for row, k in zip(per_object, keep) if k]
            metrics.add(sample["scene"], f"{group}/anchor/box_frac_acc", [r["box_frac"] for r in picked])
            metrics.add(sample["scene"], f"{group}/anchor/box_any_acc", [r["box_any"] for r in picked])
            metrics.add(sample["scene"], f"{group}/anchor/centre_in_box_acc", [r["centre_in_box"] for r in picked])
            for variant in variants:
                metrics.add(sample["scene"], f"{group}/{variant}/centre_m", [r[variant]["centre_m"] for r in picked])
                metrics.add(sample["scene"], f"{group}/{variant}/centre_hit_acc",
                            [r[variant]["centre_m"] <= HIT_RADIUS for r in picked])
                metrics.add(sample["scene"], f"{group}/{variant}/in_box_acc", [r[variant]["in_box"] for r in picked])
                metrics.add(sample["scene"], f"{group}/{variant}/in_box5_acc", [r[variant]["in_box5"] for r in picked])
                areas = [r[variant]["auc"] for r in picked if not np.isnan(r[variant]["auc"])]
                if areas:
                    metrics.add(sample["scene"], f"{group}/{variant}/auc_acc", areas)   # _acc: averaged, not median

    result = {"teacher": "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l", "size": list(size), "templates": list(TEMPLATES),
              "vocabulary": len(names), "hit_radius_m": HIT_RADIUS, "top_k": TOP_K, "samples": len(dataset),
              "variants": variants, "metrics": metrics.summary()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.name + ".json").write_text(json.dumps(result, indent=1))
    with open(output.with_name(output.name + ".objects.jsonl"), "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    if dump:
        np.savez_compressed(output.with_name(output.name + ".examples.npz"), **dump)
    summary = result["metrics"]
    for variant in variants:
        print(variant, {k: round(summary[f"in_view/{variant}/{k}"]["value"], 3)
                        for k in ("centre_m", "centre_hit_acc", "in_box_acc", "in_box5_acc", "auc_acc")})
    print("anchors", {k: round(summary[f"in_view/anchor/{k}"]["value"], 3)
                      for k in ("box_frac_acc", "box_any_acc", "centre_in_box_acc")})


if __name__ == "__main__":
    main()
