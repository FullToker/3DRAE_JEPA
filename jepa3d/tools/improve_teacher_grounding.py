#!/usr/bin/env python3
"""Can dino.txt name the right patch more often? Scores and patch selection compared, no training.

chain_teacher_point.py: the patch with the top centered score passes through
the object's box for 62% of in-view objects, and the position error is 0.41 m
when it does, 1.78 m when it does not. Here the same samples are scored under
every combination of
    scale   224      similarity at 224x308 (as before)
            ms       mean of the 224x308 map and the 448x616 map pooled to the same grid
    score   centered similarity minus the patch's mean over the vocabulary (as before)
            zscore   centered, divided by the patch's standard deviation over the vocabulary
            softmax  probability of the name among the vocabulary at that patch
    select  argmax     the top patch over all views (as before)
            smooth3/5  the top patch after a 3x3 / 5x5 mean filter inside each view
            epipolar   two candidates per view (top of the raw and of the smooth3 map); each is
                       rescored by the best score other views hold near its ray
            consensus  the same candidates are lifted to 3D by the point probe; the candidate
                       with the most score-weighted candidates around it wins
"consensus+" is the weighted mean position of the candidates around the winner.
Anchors: gt (projected centre), box (random patch in the box), any_cand (some
per-view candidate is in the box: the ceiling of epipolar and consensus).

Every chosen patch is also lifted by the point probe, so each variant has the
position and direction metrics of the category -> position probe.

Usage (from the repo root):
    python jepa3d/tools/improve_teacher_grounding.py --stats data/frame_stats/scannet.jsonl \
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl \
        --probe outputs/geo_probe/point_3drae_real/final.pt \
        --output experiments/teacher_hit_rate/results/improve
"""

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.geo_probe.data import IMAGE_SIZE
from jepa3d.geo_probe.labels import EXCLUDED
from jepa3d.geo_probe.probe import GeoProbe, SceneMetrics
from jepa3d.geo_probe.train import point_queries
from jepa3d.tools.chain_teacher_point import collate_boxes, patch_prompts, random_true
from jepa3d.tools.check_teacher_grounding import PATCH, load_dinotxt, patch_features
from jepa3d.tools.diagnose_teacher_grounding import (SIZE_BINS, BoxDataset, auc, embed_vocabulary, load_boxes,
                                                     patch_rays, rays_hit_boxes)
from jepa3d.x_encoder import ThreeDRAEXEncoder, unpack_cams

BIG_SIZE = (2 * IMAGE_SIZE[0], 2 * IMAGE_SIZE[1])
SCALES = ("224", "ms")
SCORES = ("centered", "zscore", "softmax")
SELECTS = ("argmax", "smooth3", "smooth5", "epipolar", "consensus")
TEMPERATURE = 0.02       # softmax over the vocabulary
RAY_SIGMA = 0.2          # metres, epipolar support
POINT_SIGMA = 0.3        # metres, consensus support


def load_point_probe(path, device):
    """A geo_probe run with --query point --memory 3drae -> (encoder, probe), frozen."""
    ckpt = torch.load(path, map_location="cpu")
    train_args = ckpt["args"]
    if train_args.get("query") != "point" or train_args.get("memory", "3drae") != "3drae":
        raise ValueError("--probe must be a point-prompt run on 3DRAE tokens")
    encoder = ThreeDRAEXEncoder(train_args.get("feature_config", "configs/models/dino2_feature.yaml")).to(device).eval()
    probe = GeoProbe(encoder.rae.encoder.hidden_size + 6, encoder.dim).to(device).eval()
    probe.load_state_dict(ckpt["probe"])
    missing, unexpected = encoder.load_state_dict(ckpt["encoder_trainable"], strict=False)
    assert not unexpected and all(k.startswith("rae.") for k in missing)
    return encoder, probe


def similarities(text, small, big) -> dict:
    """text (N, C), patch features small (V, h, w, C) and big (V, 2h, 2w, C) -> {scale: (N, V*h*w)}."""
    views, h, w, _ = small.shape
    sim = text @ small.flatten(0, 2).T
    sim_big = (text @ big.flatten(0, 2).T).view(-1, 1, 2 * h, 2 * w)
    pooled = F.avg_pool2d(sim_big, 2).view(len(text), -1)
    return {"224": sim, "ms": (sim + pooled) / 2}


def name_scores(similarity: torch.Tensor, rows: torch.Tensor) -> dict:
    """similarity (N, P) of every vocabulary entry with every patch, rows (O,) -> {score: (O, P)}."""
    centered = similarity[rows] - similarity.mean(0)
    return {"centered": centered, "zscore": centered / similarity.std(0).clamp(min=1e-6),
            "softmax": torch.softmax(similarity / TEMPERATURE, dim=0)[rows]}


def smooth(maps: torch.Tensor, grid, kernel: int) -> torch.Tensor:
    """(O, V*h*w) -> the same, mean-filtered inside each view (borders average what exists)."""
    x = maps.reshape(-1, 1, *grid)
    return F.avg_pool2d(x, kernel, 1, kernel // 2, count_include_pad=False).reshape(maps.shape)


def standardize(maps: torch.Tensor) -> torch.Tensor:
    return (maps - maps.mean(1, keepdim=True)) / maps.std(1, keepdim=True).clamp(min=1e-6)


def view_candidates(maps: torch.Tensor, views: int) -> torch.Tensor:
    """(O, V*hw) -> (O, V) flat index of the top patch of each view."""
    per_view = maps.view(len(maps), views, -1)
    return per_view.argmax(-1) + torch.arange(views, device=maps.device) * per_view.shape[-1]


def epipolar_scores(candidates, z, origins, directions, sigma=RAY_SIGMA) -> torch.Tensor:
    """Candidates (O, K) flat patch indices, rescored with the views they are not in.
    z (O, V*hw) standardized scores, origins (V, 3) metres, directions (V, hw, 3).
    Support of view b for a candidate: the best max(z, 0) among b's patches,
    weighted by exp(-gap^2 / 2 sigma^2), gap = distance between the two rays
    (both in front of their cameras). Returns (O, K): own z + mean support."""
    objects = len(candidates)
    views, per_view, _ = directions.shape
    own = candidates // per_view                                                          # (O, K)
    d1 = directions.reshape(-1, 3)[candidates][:, :, None, None, :]                       # (O, K, 1, 1, 3)
    d2 = directions[None, None]                                                           # (1, 1, V, hw, 3)
    w0 = (origins[own][:, :, None] - origins[None, None])[:, :, :, None, :]               # (O, K, V, 1, 3)
    b = (d1 * d2).sum(-1)                                                                 # (O, K, V, hw)
    d = (d1 * w0).sum(-1)
    e = (d2 * w0).sum(-1)
    denom = (1 - b * b).clamp(min=1e-6)
    t1, t2 = (b * e - d) / denom, (e - b * d) / denom
    gap = (w0 + t1[..., None] * d1 - t2[..., None] * d2).norm(dim=-1)
    weight = torch.exp(-gap.pow(2) / (2 * sigma ** 2)) * (t1 > 0) * (t2 > 0)
    support = (weight * z.view(objects, 1, views, -1).clamp(min=0)).amax(-1)              # (O, K, V)
    support = support.masked_fill(own[..., None] == torch.arange(views, device=z.device), 0)
    return z.gather(1, candidates) + support.sum(-1) / (views - 1)


def consensus(positions, weight, sigma=POINT_SIGMA):
    """positions (O, K, 3) of K candidates, weight (O, K) >= 0 -> (winner (O,), refined (O, 3)):
    the candidate with the largest weighted Gaussian support and the weighted
    mean of the candidates within 2 sigma of it."""
    gap = torch.cdist(positions, positions)
    support = (torch.exp(-gap.pow(2) / (2 * sigma ** 2)) * weight[:, None, :]).sum(-1)
    winner = support.argmax(1)
    near = (gap[torch.arange(len(winner)), winner] <= 2 * sigma) * weight
    refined = (near[..., None] * positions).sum(1) / near.sum(1, keepdim=True).clamp(min=1e-9)
    return winner, refined


def select_patches(maps, grid, views, origins, directions, lift) -> dict:
    """Score maps (O, V*hw) -> {select: flat patch index (O,)} plus "candidates" (O, 2V)
    and "consensus+" (O, 3) position. lift: flat indices (O, K) -> positions (O, K, 3).
    Candidates: the top patch of each view in the raw map and in the smooth3 map
    (a small object is one patch, which the filter flattens)."""
    smooth3 = smooth(maps, grid, 3)
    z = torch.maximum(standardize(maps), standardize(smooth3))
    candidates = torch.cat([view_candidates(maps, views), view_candidates(smooth3, views)], dim=1)
    rescored = epipolar_scores(candidates, z, origins, directions)
    # A patch that tops both maps is listed twice; it votes once.
    repeat = (candidates[:, views:, None] == candidates[:, None, :views]).any(-1)
    weight = z.gather(1, candidates).clamp(min=0) + 1e-3
    weight[:, views:] = weight[:, views:] * ~repeat
    winner, refined = consensus(lift(candidates), weight)
    pick = lambda column: candidates.gather(1, column[:, None]).squeeze(1)
    return {"argmax": maps.argmax(1), "smooth3": smooth3.argmax(1), "smooth5": smooth(maps, grid, 5).argmax(1),
            "epipolar": pick(rescored.argmax(1)), "consensus": pick(winner), "candidates": candidates,
            "consensus+": refined}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--train-labels", required=True, help="vocabulary of the scores")
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
    with open(args.train_labels) as f:
        frequency = Counter(o["category"] for line in f for o in json.loads(line)["objects"])
    dataset = BoxDataset(args.labels, args.stats, limit=args.limit, seeds=list(range(args.seeds)),
                         memory_size=BIG_SIZE, boxes=load_boxes(args.split))
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=collate_boxes)
    encoder, probe = load_point_probe(args.probe, device)
    teacher, tokenizer = load_dinotxt(device)
    names = sorted(set(frequency) | {o["category"] for s in dataset.scenes for o in s["objects"]} | EXCLUDED)
    index = {name: i for i, name in enumerate(names)}
    text = embed_vocabulary(teacher, tokenizer, names, device)["patch", "ensemble"]
    del tokenizer

    grid = (IMAGE_SIZE[0] // PATCH, IMAGE_SIZE[1] // PATCH)
    generator = torch.Generator(device=device).manual_seed(0)
    metrics = SceneMetrics()
    rows = []
    with torch.no_grad():
        for batch in loader:
            scene = batch["scene"][0]
            categories = batch["categories"][0]
            count = len(categories)
            images = batch["images"].to(device)
            cams = batch["cams"].to(device)
            scale = batch["scale"].to(device)
            views = images.shape[1]
            K, w2c = unpack_cams(cams[0])
            origins, directions = patch_rays(K.float(), w2c.float(), IMAGE_SIZE)
            origins = origins * scale
            in_box = rays_hit_boxes(origins, directions, batch["box_center"].to(device),
                                    batch["box_half"].to(device), batch["box_rotation"].to(device)).flatten(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = encoder(images, cams).float()

            def lift(flat):
                """Flat patch indices (O, K) -> point-probe positions (O, K, 3), metres."""
                prompt = patch_prompts(flat.reshape(-1), grid)             # queries do not see each other
                queries, _ = point_queries(encoder.rae, {"images": batch["images"], "cams": batch["cams"],
                                                         "prompt": prompt[None]}, device)
                return probe(queries, tokens, scale)[0].view(*flat.shape, 3)

            gt_prompt = batch["prompt"][0, :count]
            in_view = (gt_prompt[:, 0] >= 0).numpy()
            box_any = in_box.any(1).cpu().numpy()
            target = batch["positions"][0, :count].numpy()
            up = batch["up"][0].numpy()
            visible = batch["visible"][0, :count].numpy()
            longest = (2 * batch["box_half"].max(1).values).numpy()
            sizes = np.array([next(name for low, high, name in SIZE_BINS if low <= x < high) for x in longest])
            per_object = [{"scene": scene, "category": c, "train_count": frequency[c], "in_view": bool(in_view[i]),
                           "visible": bool(visible[i]), "box_any": bool(box_any[i]), "longest_m": float(longest[i])}
                          for i, c in enumerate(categories)]
            groups = {"in_view": in_view,
                      **{f"in_view_size_{name}": in_view & (sizes == name) for _, _, name in SIZE_BINS}}

            def record(variant, pred, hit=None, usable=None):
                """pred (O, 3) numpy; hit (O,) bool or None when the variant is not one patch."""
                usable = np.ones(count, dtype=bool) if usable is None else usable
                error = np.linalg.norm(pred - target, axis=1)
                for i, row in enumerate(per_object):
                    if usable[i]:
                        row[variant] = [round(float(error[i]), 3)] + ([] if hit is None else [bool(hit[i])])
                if usable.any():
                    sub = SceneMetrics()
                    sub.update(scene, pred[usable], target[usable], up, visible[usable])
                    for name, per_scene in sub.values.items():
                        metrics.add(scene, f"{variant}/{name}", per_scene[scene])
                for group, keep in groups.items():
                    keep = keep & usable
                    if keep.any():
                        metrics.add(scene, f"{variant}/{group}/position_m", error[keep])
                        if hit is not None:
                            metrics.add(scene, f"{variant}/{group}/in_box_acc", hit[keep])

            hits = lambda flat: in_box.gather(1, flat[:, None]).squeeze(1).cpu().numpy()
            queries, _ = point_queries(encoder.rae, {"images": batch["images"], "cams": batch["cams"],
                                                     "prompt": gt_prompt[None].to(device)}, device)
            record("gt", probe(queries, tokens, scale)[0].cpu().numpy(), np.ones(count, dtype=bool), in_view)
            anchor = random_true(in_box, generator)
            record("box", lift(anchor[:, None])[:, 0].cpu().numpy(), hits(anchor), box_any)

            small = patch_features(teacher, images[0])
            big = patch_features(teacher, batch["memory_images"][0].to(device))
            object_rows = torch.tensor([index[c] for c in categories], device=device)
            for scale_name, similarity in similarities(text, small, big).items():
                for score, maps in name_scores(similarity, object_rows).items():
                    chosen = select_patches(maps, grid, views, origins, directions, lift)
                    prefix = f"{scale_name}/{score}"
                    areas = auc(maps, in_box).cpu().numpy()
                    ok = in_view & ~np.isnan(areas)
                    if ok.any():
                        metrics.add(scene, f"{prefix}/in_view/auc_acc", areas[ok])
                    any_candidate = in_box.gather(1, chosen["candidates"]).any(1).cpu().numpy()
                    metrics.add(scene, f"{prefix}/any_cand/in_view/in_box_acc", any_candidate[in_view])
                    flat = torch.stack([chosen[s] for s in SELECTS], dim=1)                # (O, S)
                    positions = lift(flat).cpu().numpy()
                    for column, select in enumerate(SELECTS):
                        record(f"{prefix}/{select}", positions[:, column], hits(flat[:, column]))
                    record(f"{prefix}/consensus+", chosen["consensus+"].cpu().numpy())
            rows.extend(per_object)

    variants = ["/".join(v) for v in itertools.product(SCALES, SCORES, SELECTS + ("consensus+",))]
    result = {"probe": args.probe, "teacher": "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l", "samples": len(dataset),
              "variants": variants, "anchors": ["gt", "box"], "temperature": TEMPERATURE, "ray_sigma_m": RAY_SIGMA,
              "point_sigma_m": POINT_SIGMA, "metrics": metrics.summary()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.name + ".json").write_text(json.dumps(result, indent=1))
    with open(output.with_name(output.name + ".objects.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = result["metrics"]
    for variant in ["gt", "box"] + variants:
        print(variant, {key: round(summary[f"{variant}/{key}"]["value"], 3)
                        for key in ("in_view/in_box_acc", "in_view/position_m", "visible/position_m",
                                    "visible/quadrant_acc") if f"{variant}/{key}" in summary})


if __name__ == "__main__":
    main()
