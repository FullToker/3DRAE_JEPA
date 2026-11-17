#!/usr/bin/env python3
"""Object slots for the questions of a manifest: names -> candidate 3D positions, nothing trained.

For every row the object names of the question are found (jepa3d/slots.py).
Each name gets NUM_CANDIDATES candidates from the row's posed frames:
    dino.txt     the best patch of each view (centered score), the views with the highest scores
    point probe  each patch + its ray -> 3D position from the frozen 3DRAE tokens
Rows that share their frames are computed together. The output is the manifest
with a "slots" field per row (empty when the question names no object):
    {"name", "spans" (characters in the query), "pos" (k, 3) metres in the first
     view's frame, "conf" (k,) standardized dino.txt score}, best candidate first
plus <output>.stats.json.

Usage (from the repo root):
    python jepa3d/tools/precompute_slots.py --manifest experiments/hardneg_v1/data/train_f8.jsonl \
        --vocab-manifests experiments/hardneg_v1/data/train_f8.jsonl data/vsibench_test_posed_f8.jsonl \
        --train-labels data/geo_probe/labels_train.jsonl \
        --probe experiments/point_probe_offcentre/ckpt/box05/final.pt \
        --output experiments/slots_v1/data/train_f8.jsonl [--shard 0/4]
A sharded run writes <output>.part<i>; concatenate the parts.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.data import load_views
from jepa3d.geo_probe.data import IMAGE_SIZE
from jepa3d.geo_probe.labels import EXCLUDED
from jepa3d.geo_probe.train import point_queries
from jepa3d.slots import NUM_CANDIDATES, NameFinder, build_vocabulary
from jepa3d.tools.chain_teacher_point import patch_prompts
from jepa3d.tools.check_teacher_grounding import PATCH, load_dinotxt, patch_features
from jepa3d.tools.diagnose_teacher_grounding import embed_vocabulary, first
from jepa3d.tools.improve_teacher_grounding import load_point_probe, name_scores, standardize, view_candidates


def read_rows(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class FrameGroups(Dataset):
    """Rows of a manifest grouped by their frames; one item = one set of frames and the names asked about it."""

    def __init__(self, manifest, finder, shard=(0, 1), limit=0):
        self.manifest = manifest
        self.found = []        # per row: [(name, spans)]
        self.group_of = []     # per row: group index
        self.offsets, self.names = [], []   # per group: offset of its first row, its names
        index = {}
        with open(manifest, "rb") as f:
            offset = 0
            for raw in f:
                if raw.strip():
                    row = json.loads(raw)
                    found = finder(row["query"])
                    group = index.setdefault(tuple(row["images"]), len(index))
                    if group == len(self.offsets):
                        self.offsets.append(offset)
                        self.names.append(set())
                    self.names[group].update(name for name, _ in found)
                    self.found.append(found)
                    self.group_of.append(group)
                offset = f.tell()
        part, parts = shard
        self.groups = [g for g in range(len(self.offsets)) if g % parts == part]
        if limit:
            self.groups = self.groups[:limit]
        self.todo = [g for g in self.groups if self.names[g]]
        self._file = None

    def __len__(self):
        return len(self.todo)

    def __getitem__(self, idx):
        group = self.todo[idx]
        if self._file is None:
            self._file = open(self.manifest, "rb")
        self._file.seek(self.offsets[group])
        row = json.loads(self._file.readline())
        images, K, w2c = load_views({"image_paths": row["images"], "intrinsics": row["intrinsics"],
                                     "c2w": row["c2w"]}, IMAGE_SIZE)
        c2w = np.array(row["c2w"])
        scale = float(np.linalg.norm((np.linalg.inv(c2w[0]) @ c2w)[:, :3, 3], axis=-1).max())
        return {"group": group, "names": sorted(self.names[group]), "images": images.clamp(0, 1),
                "cams": torch.cat([K.flatten(1), w2c.flatten(1)], dim=1), "scale": scale if scale > 1e-6 else 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocab-manifests", nargs="+", required=True,
                        help="object names are collected from these; give every run of one experiment the same list")
    parser.add_argument("--train-labels", required=True, help="its categories widen the vocabulary of the score")
    parser.add_argument("--probe", required=True, help="final.pt of a geo_probe run with --query point --memory 3drae")
    parser.add_argument("--output", required=True, help="manifest with slots")
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th frame group, starting at i")
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N frame groups")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # cpu: smoke runs
    shard = tuple(int(x) for x in args.shard.split("/"))
    names = build_vocabulary(row for path in args.vocab_manifests for row in read_rows(path))
    print(f"[vocabulary] {len(names)} object names", flush=True)
    dataset = FrameGroups(args.manifest, NameFinder(names), shard, args.limit)
    print(f"[rows] {len(dataset.found)} rows, {len(dataset.offsets)} frame groups, "
          f"{len(dataset.groups)} in this shard, {len(dataset)} with names", flush=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=first)

    encoder, probe = load_point_probe(args.probe, device)
    teacher, tokenizer = load_dinotxt(device)
    with open(args.train_labels) as f:
        vocabulary = {o["category"] for line in f for o in json.loads(line)["objects"]}
    # Tagged names outside the vocabulary (NameFinder's last-word fallback) need a text embedding too.
    vocabulary = sorted(vocabulary | EXCLUDED | set(names) | set().union(*dataset.names))
    index = {name: i for i, name in enumerate(vocabulary)}
    text = embed_vocabulary(teacher, tokenizer, vocabulary, device)["patch", "ensemble"]
    del tokenizer

    grid = (IMAGE_SIZE[0] // PATCH, IMAGE_SIZE[1] // PATCH)
    slots = {}   # group -> {name: (positions, confidences)}
    with torch.no_grad():
        for step, sample in enumerate(loader):
            images, cams = sample["images"].to(device), sample["cams"].to(device)
            views = len(images)
            k = min(NUM_CANDIDATES, views)
            feats = patch_features(teacher, images).flatten(0, 2)
            rows = torch.tensor([index[n] for n in sample["names"]], device=device)
            maps = name_scores(text @ feats.T, rows)["centered"]                      # (O, P)
            z = standardize(maps)
            best = view_candidates(maps, views)                                       # (O, V) top patch of each view
            order = z.gather(1, best).argsort(1, descending=True)[:, :k]
            chosen = best.gather(1, order)                                            # (O, k) best view first
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = encoder(images[None], cams[None]).float()
            queries, _ = point_queries(encoder.rae, {"images": images[None], "cams": cams[None],
                                                     "prompt": patch_prompts(chosen.reshape(-1), grid)[None]}, device)
            scale = torch.tensor([sample["scale"]], device=device)
            position = probe(queries, tokens, scale)[0].view(len(rows), k, 3).cpu().numpy()
            confidence = z.gather(1, chosen).cpu().numpy()
            patches = chosen.cpu().tolist()
            slots[sample["group"]] = {name: ([[round(float(x), 3) for x in p] for p in position[i]],
                                             [round(float(c), 2) for c in confidence[i]], patches[i])
                                      for i, name in enumerate(sample["names"])}
            if step % 2000 == 0:
                print(f"[groups] {step}/{len(dataset)}", flush=True)

    output = Path(args.output if shard[1] == 1 else f"{args.output}.part{shard[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    mine = set(dataset.groups)
    counts, top, with_slots, written = Counter(), [], 0, 0
    with open(output, "w") as f:
        for i, row in enumerate(read_rows(args.manifest)):
            group = dataset.group_of[i]
            if group not in mine:
                continue
            row["slots"] = [{"name": name, "spans": spans, "pos": slots[group][name][0], "conf": slots[group][name][1],
                             "patch": slots[group][name][2]} for name, spans in dataset.found[i]]
            f.write(json.dumps(row) + "\n")
            written += 1
            with_slots += bool(row["slots"])
            kind = f'{row.get("source")}/{row.get("question_type") or row.get("type")}'
            counts[f"{kind}/rows"] += 1
            counts[f"{kind}/names"] += len(row["slots"])
            top.extend(s["conf"][0] for s in row["slots"])
    stats = {"manifest": args.manifest, "probe": args.probe, "shard": args.shard, "rows": written,
             "rows_with_slots": with_slots, "object_names": len(names), "candidates": NUM_CANDIDATES,
             "top_conf_quantiles": dict(zip(("5%", "25%", "50%", "75%", "95%"),
                                            np.quantile(top, [0.05, 0.25, 0.5, 0.75, 0.95]).round(2).tolist())) if top else {},
             "by_type": dict(sorted(counts.items()))}
    Path(f"{output}.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "by_type"}), flush=True)


if __name__ == "__main__":
    main()
