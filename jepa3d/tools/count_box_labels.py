#!/usr/bin/env python3
"""How many training rows can get a true 3D position for the objects their question names?

For every object name of a row (the row's "slots", found by jepa3d.slots.NameFinder) the
EmbodiedScan boxes of the row's scene are searched for the name's category. A name is
"labelled" if exactly one box of the scene has that category and the box centre projects
into at least one of the row's views (no occlusion test). Counted per manifest, EmbodiedScan
version, source and question type; nothing is trained or changed.

Usage (from the repo root):
    python jepa3d/tools/count_box_labels.py --manifests experiments/slots_v2/data/train_f8.jsonl \
        experiments/slots_v2/data/holdout_f8.jsonl --output experiments/grounding_label_count/results/summary.json
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.geo_probe.labels import EXCLUDED
from jepa3d.tools.check_objtext_data import INFOS, norm

TOP = 40   # names listed per reason they got no label


def load_boxes(version: str) -> dict:
    """{"<dataset>/<scene>": {"split", "boxes": {category: (n, 3) centres in the scan's own frame},
    "visible": {frame id: categories EmbodiedScan lists as visible}}}."""
    scenes = {}
    for split in ("train", "val"):
        with open(INFOS[version] / f"embodiedscan_infos_{split}.pkl", "rb") as f:
            infos = pickle.load(f)
        id2cat = {v: norm(k) for k, v in infos["metainfo"]["categories"].items()}
        for info in infos["data_list"]:
            align = np.array(info.get("axis_align_matrix", np.eye(4)), dtype=np.float64)
            unalign = np.linalg.inv(align)
            boxes, category_of = defaultdict(list), {}
            for inst in info.get("instances", []):
                category = id2cat[inst["bbox_label_3d"]]
                if category in EXCLUDED:
                    continue
                centre = np.array(inst["bbox_3d"][:3], dtype=np.float64)
                boxes[category].append(unalign[:3, :3] @ centre + unalign[:3, 3])
                category_of[int(inst["bbox_id"])] = category
            visible = {}
            for image in info["images"]:
                stem = Path(image["img_path"]).stem
                if stem.isdigit():
                    visible[int(stem)] = {category_of[i] for i in map(int, image.get("visible_instance_ids", []))
                                          if i in category_of}
            scenes[info["sample_idx"]] = {"split": split, "boxes": {c: np.stack(v) for c, v in boxes.items()},
                                          "visible": visible}
    return scenes


def category_of(name: str, boxes: dict):
    """The scene's category a question name stands for: the name itself or its singular."""
    name = norm(name)
    for candidate in (name, name[:-1] if name.endswith("s") else None, name[:-2] if name.endswith("es") else None):
        if candidate and candidate in boxes:
            return candidate
    return None


def in_view(centre: np.ndarray, row: dict) -> bool:
    """Does the point (scan frame) project into one of the row's views, at least 0.1 m in front of it?
    The image is taken to be twice the principal point wide and high."""
    for K, c2w in zip(row["intrinsics"], row["c2w"]):
        K = np.array(K, dtype=np.float64)
        cam = np.linalg.inv(np.array(c2w, dtype=np.float64)) @ np.append(centre, 1.0)
        if cam[2] <= 0.1:
            continue
        u, v = K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]
        if 0 <= u < 2 * K[0, 2] and 0 <= v < 2 * K[1, 2]:
            return True
    return False


def count_manifest(path: str, versions: dict) -> dict:
    counts = {v: defaultdict(Counter) for v in versions}       # version -> group -> counter
    misses = {v: defaultdict(Counter) for v in versions}       # version -> reason -> names
    scenes = {v: defaultdict(set) for v in versions}           # version -> kind -> scene keys
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            key = f"{row.get('subset')}/{row['scene']}"
            names = [slot["name"] for slot in row.get("slots") or []]
            groups = ("all", f"subset:{row.get('subset')}", f"{row.get('source')}/{row.get('question_type')}")
            for version, boxes_of in versions.items():
                scene = boxes_of.get(key)
                tally = Counter(rows=1, rows_with_names=bool(names), names=len(names))
                scenes[version]["all"].add(key)
                if scene is not None:
                    tally["rows_in_box_scene"] = 1
                    tally["rows_with_names_in_box_scene"] = bool(names)
                    scenes[version]["with_boxes"].add(key)
                labelled = 0
                listed_frames = scene is not None and any(f in scene["visible"] for f in row.get("frame_ids", []))
                for name in names:
                    if scene is None:
                        tally["names_scene_without_boxes"] += 1
                        continue
                    category = category_of(name, scene["boxes"])
                    if category is None:
                        tally["names_no_such_category"] += 1
                        misses[version]["no_such_category"][norm(name)] += 1
                        continue
                    centres = scene["boxes"][category]
                    if len(centres) > 1:
                        tally["names_several_boxes"] += 1
                        misses[version]["several_boxes"][category] += 1
                        continue
                    tally["names_one_box"] += 1
                    if listed_frames:
                        tally["names_one_box_frames_listed"] += 1
                        tally["names_one_box_listed_visible"] += any(
                            category in scene["visible"].get(f, ()) for f in row["frame_ids"])
                    if in_view(centres[0], row):
                        tally["names_labelled"] += 1
                        labelled += 1
                    else:
                        tally["names_one_box_out_of_view"] += 1
                if names:
                    tally["rows_some_name_labelled"] = labelled > 0
                    tally["rows_two_names_labelled"] = labelled >= 2
                    tally["rows_all_names_labelled"] = labelled == len(names)
                    if labelled:
                        scenes[version][f"labelled_{scene['split']}"].add(key)
                for group in groups:
                    counts[version][group].update(tally)
    return {version: {"groups": {g: dict(c) for g, c in sorted(counts[version].items())},
                      "scenes": {k: len(v) for k, v in sorted(scenes[version].items())},
                      "names_without_label": {reason: names.most_common(TOP)
                                              for reason, names in misses[version].items()}}
            for version in versions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    versions = {version: load_boxes(version) for version in INFOS}
    for version, scenes in versions.items():
        print(f"[boxes] {version}: {dict(Counter(k.split('/')[0] for k in scenes))}", flush=True)
    summary = {}
    for path in args.manifests:
        summary[path] = count_manifest(path, versions)
        for version, result in summary[path].items():
            print(f"\n== {path} | EmbodiedScan {version} ==\nscenes: {result['scenes']}")
            for group, c in result["groups"].items():
                print(f"{group:55s} rows {c['rows']:7d}  with names {c.get('rows_with_names', 0):7d}  "
                      f">=1 labelled {c.get('rows_some_name_labelled', 0):7d}  "
                      f"all labelled {c.get('rows_all_names_labelled', 0):7d}  "
                      f"names {c.get('names', 0):7d}  labelled {c.get('names_labelled', 0):7d}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=1))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
