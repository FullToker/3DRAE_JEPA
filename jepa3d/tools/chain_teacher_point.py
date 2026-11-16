#!/usr/bin/env python3
"""Name -> 3D position without training: dino.txt picks the patch, the point probe locates it.

Chains two frozen parts that were only tested apart:
    dino.txt   category name -> the patch, over 8 views, that fits it best
               (diagnose_teacher_grounding.py: patch half, template ensemble, centered score)
    point probe (geo_probe, --query point) that patch + its ray -> 3D position from 3DRAE tokens
The point probe was trained and evaluated with the object's projected centre as
the prompt (0.27 m). Here its prompt comes from one of
    gt        the projected centre (repeats the probe's own evaluation; in-view objects only)
    box       a random patch whose ray passes through the object's 3D box (right object, off centre)
    teacher   dino.txt, centered score
    raw       dino.txt, raw cosine similarity
    random    a random patch
Same validation samples and metrics as the category -> position probe
(outputs/geo_probe/real: 0.95 m visible), which this is to be compared with.

Usage (from the repo root):
    python jepa3d/tools/chain_teacher_point.py --stats data/frame_stats/scannet.jsonl \
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl \
        --probe outputs/geo_probe/point_3drae_real/final.pt \
        --output experiments/slot_chain_check/results/chain
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.geo_probe.data import IMAGE_SIZE, collate
from jepa3d.geo_probe.labels import EXCLUDED
from jepa3d.geo_probe.probe import GeoProbe, SceneMetrics
from jepa3d.geo_probe.train import point_queries
from jepa3d.tools.check_teacher_grounding import PATCH, load_dinotxt, patch_features
from jepa3d.tools.diagnose_teacher_grounding import (SIZE_BINS, BoxDataset, embed_vocabulary, load_boxes, patch_rays,
                                                     rays_hit_boxes, score_maps)
from jepa3d.x_encoder import ThreeDRAEXEncoder, unpack_cams

SOURCES = ("gt", "box", "teacher", "raw", "random")
BOX_KEYS = ("box_center", "box_half", "box_rotation")


def collate_boxes(batch):
    """Batches of one: the probe's collate plus the sample's boxes."""
    out = collate(batch)
    out.update({key: batch[0][key] for key in BOX_KEYS})
    return out


def patch_prompts(index: torch.Tensor, grid) -> torch.Tensor:
    """Flat patch indices (O,) over (V, h, w) -> prompts (O, 3) [view, u, v] at the patch centres."""
    h, w = grid
    view, rest = index // (h * w), index % (h * w)
    return torch.stack([view.float(), (rest % w + 0.5) * PATCH, (rest // w + 0.5) * PATCH], dim=-1)


def random_true(mask: torch.Tensor, generator) -> torch.Tensor:
    """mask (O, P) bool -> (O,) index of a random True per row (any index for an empty row)."""
    noise = torch.rand(mask.shape, generator=generator, device=mask.device)
    return (noise * mask + mask.float()).argmax(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--train-labels", required=True, help="vocabulary of the centered score")
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--probe", required=True, help="final.pt of a geo_probe run with --query point --memory 3drae")
    parser.add_argument("--output", required=True, help="path prefix of the result files")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    ckpt = torch.load(args.probe, map_location="cpu")
    train_args = ckpt["args"]
    if train_args.get("query") != "point" or train_args.get("memory", "3drae") != "3drae":
        parser.error("--probe must be a point-prompt run on 3DRAE tokens")
    with open(args.train_labels) as f:
        frequency = Counter(o["category"] for line in f for o in json.loads(line)["objects"])
    dataset = BoxDataset(args.labels, args.stats, limit=args.limit, seeds=list(range(args.seeds)),
                         boxes=load_boxes(args.split))
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=collate_boxes)

    encoder = ThreeDRAEXEncoder(train_args.get("feature_config", "configs/models/dino2_feature.yaml")).to(device).eval()
    probe = GeoProbe(encoder.rae.encoder.hidden_size + 6, encoder.dim).to(device).eval()
    probe.load_state_dict(ckpt["probe"])
    missing, unexpected = encoder.load_state_dict(ckpt["encoder_trainable"], strict=False)
    assert not unexpected and all(k.startswith("rae.") for k in missing)

    teacher, tokenizer = load_dinotxt(device)
    names = sorted(set(frequency) | {o["category"] for s in dataset.scenes for o in s["objects"]} | EXCLUDED)
    index = {name: i for i, name in enumerate(names)}
    text = embed_vocabulary(teacher, tokenizer, names, device)
    del tokenizer

    grid = (IMAGE_SIZE[0] // PATCH, IMAGE_SIZE[1] // PATCH)
    generator = torch.Generator(device=device).manual_seed(0)
    metrics = SceneMetrics()
    rows = []
    with torch.no_grad():
        for batch in loader:
            categories = batch["categories"][0]
            count = len(categories)
            images = batch["images"].to(device)
            cams = batch["cams"].to(device)
            scale = batch["scale"].to(device)
            K, w2c = unpack_cams(cams[0])
            K, w2c = K.float(), w2c.float()

            feats = patch_features(teacher, images[0]).flatten(0, 2)                        # (P, C)
            object_rows = torch.tensor([index[c] for c in categories], device=device)
            raw = score_maps(text["patch", "single"] @ feats.T, object_rows)["raw"]
            centered = score_maps(text["patch", "ensemble"] @ feats.T, object_rows)["centered"]
            origins, directions = patch_rays(K, w2c, IMAGE_SIZE)
            in_box = rays_hit_boxes(origins * scale, directions, batch["box_center"].to(device),
                                    batch["box_half"].to(device), batch["box_rotation"].to(device)).flatten(1)
            chosen = {"teacher": centered.argmax(1), "raw": raw.argmax(1), "box": random_true(in_box, generator),
                      "random": torch.randint(in_box.shape[1], (count,), generator=generator, device=device)}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = encoder(images, cams).float()
            gt_prompt = batch["prompt"][0, :count]
            in_view = (gt_prompt[:, 0] >= 0).numpy()
            box_any = in_box.any(1).cpu().numpy()
            target = batch["positions"][0, :count].numpy()
            up = batch["up"][0].numpy()
            visible = batch["visible"][0, :count].numpy()
            longest = (2 * batch["box_half"].max(1).values).numpy()
            sizes = np.array([next(name for low, high, name in SIZE_BINS if low <= x < high) for x in longest])
            per_object = [{"scene": batch["scene"][0], "category": c, "train_count": frequency[c],
                           "in_view": bool(in_view[i]), "visible": bool(visible[i]), "box_any": bool(box_any[i]),
                           "longest_m": float(longest[i])} for i, c in enumerate(categories)]

            for source in SOURCES:
                prompt = gt_prompt.to(device) if source == "gt" else patch_prompts(chosen[source], grid)
                # gt has no prompt outside the views, box none for an object no ray passes through.
                usable = {"gt": in_view, "box": box_any}.get(source, np.ones(count, dtype=bool))
                queries, _ = point_queries(encoder.rae, {"images": batch["images"], "cams": batch["cams"],
                                                         "prompt": prompt[None]}, device)
                pred = probe(queries, tokens, scale)[0].cpu().numpy()
                error = np.linalg.norm(pred - target, axis=1)
                hit = (in_box.gather(1, chosen[source][:, None]).squeeze(1).cpu().numpy() if source in chosen
                       else np.ones(count, dtype=bool))
                for i, row in enumerate(per_object):
                    if usable[i]:
                        row[source] = {"error_m": float(error[i]), "in_box": bool(hit[i])}
                scene = batch["scene"][0]
                # all/* and visible/*: every usable object, as the category probe reports them.
                if usable.any():
                    sub = SceneMetrics()
                    sub.update(scene, pred[usable], target[usable], up, visible[usable])
                    for name, per_scene in sub.values.items():
                        metrics.add(scene, f"{source}/{name}", per_scene[scene])
                groups = {"in_view": in_view, "in_view_hit": in_view & hit, "in_view_miss": in_view & ~hit,
                          **{f"in_view_size_{name}": in_view & (sizes == name) for _, _, name in SIZE_BINS}}
                for group, keep in groups.items():
                    keep = keep & usable
                    if keep.any():
                        metrics.add(scene, f"{source}/{group}/position_m", error[keep])
                        metrics.add(scene, f"{source}/{group}/in_box_acc", hit[keep])
            rows.extend(per_object)

    result = {"probe": args.probe, "teacher": "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l", "samples": len(dataset),
              "sources": list(SOURCES), "metrics": metrics.summary()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.name + ".json").write_text(json.dumps(result, indent=1))
    with open(output.with_name(output.name + ".objects.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = result["metrics"]
    for source in SOURCES:
        print(source, {key: round(summary[f"{source}/{key}"]["value"], 3)
                       for key in ("all/position_m", "visible/position_m", "visible/quadrant_acc", "in_view/position_m",
                                   "in_view/in_box_acc", "in_view_hit/position_m", "in_view_miss/position_m")
                       if f"{source}/{key}" in summary})


if __name__ == "__main__":
    main()
