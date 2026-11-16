#!/usr/bin/env python3
"""How far do object slots alone get on VSI-Bench? Answers computed from slot positions, no training.

For every question that names its objects, each name becomes a slot:
    dino.txt   name -> a patch in the question's 8 posed frames (centered score)
    point probe that patch + its ray -> 3D position from the frozen 3DRAE tokens
and the answer is computed from the positions with the benchmark's own rule:
    object_rel_direction_*  stand at A, face B, where is C (left / right / back, quadrants)
    object_rel_distance     which candidate's position is nearest to the reference object
    obj_appearance_order    dino.txt only: the first frame in which each name scores
                            at least a share of its best frame; the listed order that agrees best
This is what a predictor would get from the slots if it read them perfectly,
measured against the benchmark's answers (the scenes have no boxes we can
use, so the slots themselves are not checked here). The patch is chosen as in
improve_teacher_grounding.py (argmax, smooth3, epipolar, consensus,
consensus+); "random" takes a random patch and marks chance.

Usage (from the repo root):
    python jepa3d/tools/slots_answer_vsibench.py --manifest data/vsibench_test_posed_f8.jsonl \
        --train-labels data/geo_probe/labels_train.jsonl --probe outputs/geo_probe/point_3drae_real/final.pt \
        --output experiments/slot_vsibench_check/results/test
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.data import load_views
from jepa3d.geo_probe.data import IMAGE_SIZE
from jepa3d.geo_probe.labels import EXCLUDED
from jepa3d.geo_probe.probe import SceneMetrics
from jepa3d.geo_probe.train import point_queries
from jepa3d.tools.chain_teacher_point import patch_prompts
from jepa3d.tools.check_teacher_grounding import PATCH, load_dinotxt, patch_features
from jepa3d.tools.diagnose_teacher_grounding import embed_vocabulary, first, patch_rays
from jepa3d.tools.improve_teacher_grounding import (load_point_probe, name_scores, select_patches, smooth,
                                                    standardize)

DIRECTION = re.compile(r"standing by the (.+?) and facing the (.+?), is the (.+?) to ")
DISTANCE = re.compile(r"\((.+?)\) is the closest to the (.+?)\?")
ORDER = re.compile(r": (.+?)\?$")
SOURCES = ("argmax", "smooth3", "epipolar", "consensus", "consensus+", "random")
ORDER_SHARES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def parse(row):
    """(kind, object names) of a question whose answer follows from object positions
    or first appearances, else None. direction: [stand, face, target]; distance:
    candidates..., reference; order: the listed categories."""
    question = row["query"].split("\n")[0].lower()
    kind = row["type"]
    if kind.startswith("object_rel_direction"):
        match = DIRECTION.search(question)
        return ("direction", list(match.groups())) if match else None
    if kind == "object_rel_distance":
        match = DISTANCE.search(question)
        return ("distance", match.group(1).split(", ") + [match.group(2)]) if match else None
    if kind == "obj_appearance_order":
        match = ORDER.search(question)
        return ("order", match.group(1).split(", ")) if match else None
    return None


def azimuth(stand, face, target, up) -> float:
    """Clockwise angle (deg, seen from above) of target for "stand, face"; positive = right."""
    flat = lambda x: x - (x @ up) * up
    forward, delta = flat(face - stand), flat(target - stand)
    right = np.cross(forward, up)
    return float(np.degrees(np.arctan2(delta @ right / max(np.linalg.norm(right), 1e-9),
                                       delta @ forward / max(np.linalg.norm(forward), 1e-9))))


def direction_answer(kind: str, angle: float) -> str:
    side = "right" if angle > 0 else "left"
    if kind.endswith("easy"):
        return side
    if kind.endswith("medium"):
        return "back" if abs(angle) >= 135 else side
    return ("front-" if abs(angle) < 90 else "back-") + side


def order_answer(candidates, names, first) -> str:
    """The candidate order with the most name pairs in the order of their first frames."""
    position = dict(zip(names, first))

    def agreement(candidate):
        listed = [n.strip().lower() for n in candidate.split(",")]
        return sum(1.0 if position[a] < position[b] else 0.5 if position[a] == position[b] else 0.0
                   for a, b in itertools.combinations(listed, 2))
    return max(candidates, key=agreement)


class Questions(Dataset):
    def __init__(self, manifest, limit=0):
        self.rows = []
        with open(manifest) as f:
            for line in f:
                row = json.loads(line)
                parsed = parse(row)
                if parsed:
                    self.rows.append((row, *parsed))
        if limit:   # spread over types and subsets
            self.rows = self.rows[::max(1, len(self.rows) // limit)][:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row, kind, names = self.rows[idx]
        images, K, w2c = load_views({"image_paths": row["images"], "intrinsics": row["intrinsics"],
                                     "c2w": row["c2w"]}, IMAGE_SIZE)
        c2w = np.array(row["c2w"])
        relative = np.linalg.inv(c2w[0]) @ c2w
        scale = float(np.linalg.norm(relative[:, :3, 3], axis=-1).max())
        return {"row": {k: row[k] for k in ("id", "type", "subset", "scene", "candidates", "target")},
                "kind": kind, "names": names, "images": images.clamp(0, 1),
                "cams": torch.cat([K.flatten(1), w2c.flatten(1)], dim=1), "scale": scale if scale > 1e-6 else 1.0,
                "up": np.linalg.inv(c2w[0])[:3, 2].copy()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-labels", required=True, help="its categories widen the vocabulary of the score")
    parser.add_argument("--probe", required=True, help="final.pt of a geo_probe run with --query point --memory 3drae")
    parser.add_argument("--output", required=True, help="path prefix of the result files")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dataset = Questions(args.manifest, args.limit)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=first)
    encoder, probe = load_point_probe(args.probe, device)
    teacher, tokenizer = load_dinotxt(device)
    with open(args.train_labels) as f:
        vocabulary = {o["category"] for line in f for o in json.loads(line)["objects"]}
    vocabulary |= EXCLUDED | {name for _, _, names in dataset.rows for name in names}
    vocabulary = sorted(vocabulary)
    index = {name: i for i, name in enumerate(vocabulary)}
    text = embed_vocabulary(teacher, tokenizer, vocabulary, device)["patch", "ensemble"]
    del tokenizer

    grid = (IMAGE_SIZE[0] // PATCH, IMAGE_SIZE[1] // PATCH)
    generator = torch.Generator(device=device).manual_seed(0)
    metrics = SceneMetrics()
    rows = []
    with torch.no_grad():
        for sample in loader:
            row, kind, names = sample["row"], sample["kind"], sample["names"]
            images = sample["images"].to(device)
            cams = sample["cams"].to(device)
            scale = torch.tensor([sample["scale"]], device=device)
            views = len(images)
            feats = patch_features(teacher, images).flatten(0, 2)
            object_rows = torch.tensor([index[n] for n in names], device=device)
            maps = name_scores(text @ feats.T, object_rows)["centered"]
            z = standardize(smooth(maps, grid, 3)).view(len(names), views, -1).amax(-1)       # (O, V)
            out = {**row, "names": names, "peak_z": [round(float(x), 2) for x in z.amax(1)]}
            target = row["target"].lower()
            keys = [f"{row['subset']}/{row['type']}", f"all/{row['type']}"]

            if kind == "order":
                presence = z.cpu().numpy()
                for share in ORDER_SHARES:
                    first = (presence >= share * presence.max(1, keepdims=True)).argmax(1)
                    answer = order_answer(row["candidates"], names, first)
                    out[f"order{share}"] = answer
                    for key in keys:
                        metrics.add(row["scene"], f"{key}/order{share}_acc", [answer.lower() == target])
                rows.append(out)
                continue

            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = encoder(images[None], cams[None]).float()
            batch = {"images": images[None], "cams": cams[None]}

            def lift(flat):
                queries, _ = point_queries(encoder.rae, {**batch, "prompt": patch_prompts(flat.reshape(-1), grid)[None]},
                                           device)
                return probe(queries, tokens, scale)[0].view(*flat.shape, 3)

            K, w2c = (x.float() for x in (cams[:, :9].unflatten(-1, (3, 3)), cams[:, 9:].unflatten(-1, (4, 4))))
            origins, directions = patch_rays(K, w2c, IMAGE_SIZE)
            chosen = select_patches(maps, grid, views, origins * scale, directions, lift)
            chosen["random"] = torch.randint(maps.shape[1], (len(names),), generator=generator, device=device)
            up = sample["up"]
            for source in SOURCES:
                position = (chosen[source] if source == "consensus+" else lift(chosen[source][:, None])[:, 0])
                position = position.cpu().numpy().astype(np.float64)
                if kind == "direction":
                    angle = azimuth(*position, up)
                    answer = direction_answer(row["type"], angle)
                    out[source] = [answer, round(angle, 1)]
                else:
                    gaps = np.linalg.norm(position[:-1] - position[-1], axis=1)
                    answer = names[int(gaps.argmin())]
                    out[source] = [answer, [round(float(g), 2) for g in gaps]]
                for key in keys:
                    metrics.add(row["scene"], f"{key}/{source}_acc", [answer == target])
            rows.append(out)

    result = {"manifest": args.manifest, "probe": args.probe, "questions": len(dataset), "sources": list(SOURCES),
              "order_shares": list(ORDER_SHARES), "metrics": metrics.summary()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.name + ".json").write_text(json.dumps(result, indent=1))
    with open(output.with_name(output.name + ".questions.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for key, value in result["metrics"].items():
        if key.startswith("all/"):
            print(key, round(value["value"], 3), value["n"])


if __name__ == "__main__":
    main()
