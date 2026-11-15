#!/usr/bin/env python3
"""What object-level text is on disk, and how much of it can supervise
"region <-> text" on scenes we can actually load?

A text row is usable only if (1) its object has a 3D box (EmbodiedScan),
(2) the scene's posed frames are on disk, (3) the scene is in no evaluation
split. This script counts each condition per dataset and text source and
writes one JSON; it trains nothing and changes nothing.

Object ids: ReferIt3D / ScanRefer / 3RScan captions use the original
datasets' instance ids, EmbodiedScan has its own bbox_id. Whether the two
agree is tested, not assumed: for offsets 0 and +1 the share of text rows
whose id exists as a box of the same category is reported.

Usage (from the repo root):
    python jepa3d/tools/check_objtext_data.py --output experiments/objtext_data_check/results/summary.json
"""

import argparse
import json
import pickle
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.geo_probe.labels import EXCLUDED
from jepa3d.paths import RAE_ROOT, VJEPA_ROOT

SOURCE = VJEPA_ROOT / "source_data"
LEO = SOURCE / "leo_annotations" / "annotations"
MMSCAN = SOURCE / "mmscan_data" / "MMScan-beta"
INFOS = {
    "v1": SOURCE / "embodiedscan" / "embodiedscan",
    "v2": SOURCE / "mmscan_data" / "embodiedscan_split" / "embodiedscan-v2",
}
HOLDOUT_MOD = 20  # as jepa3d/tools/add_hard_negatives.py
FRAMES_CHECKED = 5  # listed frames looked up on disk per scene


def norm(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def load_boxes(version: str):
    """-> scenes {"<dataset>/<scene>": {split, frames, objects {bbox_id: category}, visible ids,
    frame paths}} and the per dataset x split summary."""
    scenes, summary = {}, {}
    for split in ("train", "val", "test"):
        with open(INFOS[version] / f"embodiedscan_infos_{split}.pkl", "rb") as f:
            infos = pickle.load(f)
        id2cat = {v: k for k, v in infos["metainfo"]["categories"].items()}
        for info in infos["data_list"]:
            objects = {int(i["bbox_id"]): norm(id2cat[i["bbox_label_3d"]]) for i in info.get("instances", [])}
            visible = set()
            for image in info["images"]:
                visible.update(int(i) for i in image.get("visible_instance_ids", []))
            scenes[info["sample_idx"]] = {
                "split": split, "frames": [im["img_path"] for im in info["images"]],
                "objects": objects, "visible": visible & set(objects),
            }
    for key, scene in scenes.items():
        dataset = key.split("/")[0]
        s = summary.setdefault(f"{dataset}/{scene['split']}", Counter())
        named = {i: c for i, c in scene["objects"].items() if c not in EXCLUDED}
        per_category = Counter(named.values())
        s["scenes"] += 1
        s["frames_listed"] += len(scene["frames"])
        s["boxes"] += len(scene["objects"])
        s["boxes_named"] += len(named)
        s["boxes_named_visible"] += len(set(named) & scene["visible"])
        # The geometry probes only used categories that occur once in the scene.
        s["boxes_single_instance"] += sum(per_category[c] == 1 for c in named.values())
    return scenes, {k: dict(v) for k, v in sorted(summary.items())}


def frames_on_disk(scenes: dict) -> dict:
    """Per dataset: scenes whose sampled listed frames all exist under source_data."""
    rng = random.Random(0)
    out = defaultdict(Counter)
    for key, scene in scenes.items():
        dataset = key.split("/")[0]
        sample = rng.sample(scene["frames"], min(FRAMES_CHECKED, len(scene["frames"])))
        found = sum((SOURCE / p).exists() for p in sample)
        out[dataset]["scenes"] += 1
        out[dataset]["scenes_all_found"] += found == len(sample) and found > 0
        out[dataset]["scenes_none_found"] += found == 0
        scene["on_disk"] = found == len(sample) and found > 0
    out = {k: dict(v) for k, v in out.items()}
    for dataset, example in (("matterport3d", "matterport3d/v1/scans"), ("3rscan", "3rscan"),
                             ("scannet", "scannet/posed_images"), ("arkitscenes", "arkitscenes")):
        root = SOURCE / example
        out.setdefault(dataset, {})["folder"] = example
        out[dataset]["folder_entries"] = len(list(root.iterdir())) if root.is_dir() else 0
    zips = list((SOURCE / "matterport3d" / "v1" / "scans").glob("*/undistorted_color_images.zip"))
    out["matterport3d"]["scans_still_zipped"] = len(zips)
    return out


def eval_scenes() -> dict:
    """Scenes no training text may come from, as "<dataset>/<scene>" sets."""
    out = {}
    for name in ("vsibench_test_posed_f8", "vsibench_debiased_posed_f8"):
        scenes = set()
        with open(RAE_ROOT / "data" / f"{name}.jsonl") as f:
            for line in f:
                row = json.loads(line)
                scenes.add(f"{row['subset']}/{row['scene']}")
        out[name] = scenes
    return out


def text_sources():
    """Yields (source, dataset, scene, object id, category or None, text). Scene keys as
    EmbodiedScan's sample_idx without the dataset prefix."""
    def load(path):
        with open(path) as f:
            return json.load(f)

    for name in ("scannet_referit3d_sr3d+_train", "scannet_referit3d_nr3d_train"):
        for r in load(LEO / "alignment" / "obj_scene_caption" / f"{name}.json"):
            yield name.split("_")[2], "scannet", r["scan_id"], int(r["target_id"]), norm(r["instance_type"]), r["utterance"]
    for split in ("train", "val"):
        for r in load(LEO / "instruction" / "scan2cap" / f"scanrefer_{split}.json"):
            yield f"scanrefer_{split}", "scannet", r["scan_id"], int(r["target_id"]), norm(r["instance_type"]), r["utterance"]
    for r in load(LEO / "alignment" / "obj_scene_caption" / "3rscan_scanscribe.json"):
        yield "3rscan_scanscribe", "3rscan", r["scan_id"], int(r["object_id"]), None, r["sentence"]
    for scan, objects in load(LEO / "alignment" / "obj_scene_caption" / "3rscan_prompted.json").items():
        for key, items in objects.items():
            category, _, object_id = key.rpartition("-")
            for item in items:
                yield "3rscan_prompted", "3rscan", scan, int(object_id), norm(category), item["response"]
    for version in ("embodiedscan", "mmscan_data/embodiedscan_split/embodiedscan-v2"):
        tag = "es_vg_v1" if version == "embodiedscan" else "es_vg_v2"
        root = SOURCE / version / ("embodiedscan" if version == "embodiedscan" else "")
        for split in ("train", "val"):
            for r in load(root / f"embodiedscan_{split}_vg.json"):
                dataset, _, scene = r["scan_id"].partition("/")
                yield f"{tag}_{split}", dataset, scene, int(r["target_id"]), norm(r["target"]), r["text"]
    for split, rows in load(MMSCAN / "MMScan_samples" / "MMScan_Caption_object.json").items():
        for r in rows:
            dataset, _, scene = r["scan_id"].partition("/")
            yield f"mmscan_caption_{split}", dataset, scene, int(r["object_id"]), norm(r["object_type"]), r["caption"]
    for split, rows in load(MMSCAN / "MMScan_samples" / "MMScan_VG.json").items():
        for r in rows:
            targets = r.get("target_id") or []  # the test split carries no targets
            if len(targets) != 1:  # multi-target and zero-target rows name no single object
                yield f"mmscan_vg_{split}", r["scan_id"].partition("/")[0], None, None, None, r["text"]
                continue
            dataset, _, scene = r["scan_id"].partition("/")
            yield f"mmscan_vg_{split}", dataset, scene, int(targets[0]), norm(r["target"][0]), r["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    boxes_v1, summary_v1 = load_boxes("v1")
    boxes, summary_v2 = load_boxes("v2")
    disk = frames_on_disk(boxes)
    evals = eval_scenes()
    eval_all = set().union(*evals.values())

    def blocked(key: str) -> str:
        """Why a scene may not be used for training, or ""."""
        scene = boxes.get(key)
        if scene is None:
            return "no_boxes"
        if scene["split"] != "train":
            return "embodiedscan_val_test"
        if key in eval_all:
            return "vsibench"
        if zlib.crc32(key.encode()) % HOLDOUT_MOD == 0:
            return "hardneg_holdout"
        if not scene["on_disk"]:
            return "frames_missing"
        return ""

    stats = defaultdict(Counter)
    words = defaultdict(list)
    objects = defaultdict(set)
    scene_sets = defaultdict(set)
    usable_objects = defaultdict(set)
    examples = defaultdict(list)
    for source, dataset, scene, object_id, category, text in text_sources():
        key = f"{source}|{dataset}"
        s = stats[key]
        s["rows"] += 1
        if len(words[key]) < 20000:
            words[key].append(len(text.split()))
        if len(examples[key]) < 3:
            examples[key].append(text[:240])
        if scene is None:
            s["rows_no_single_target"] += 1
            continue
        scene_key = f"{dataset}/{scene}"
        scene_sets[key].add(scene_key)
        objects[key].add((scene_key, object_id))
        box_scene = boxes.get(scene_key)
        if box_scene is None:
            s["rows_scene_without_boxes"] += 1
            continue
        for offset in (0, 1):  # text id + offset == bbox_id ?
            found = box_scene["objects"].get(object_id + offset)
            s[f"id_found_offset{offset}"] += found is not None
            s[f"id_and_category_offset{offset}"] += found is not None and category is not None and (
                found == category or found in category or category in found)
        if object_id not in box_scene["objects"]:
            continue
        s["rows_with_box"] += 1
        s["rows_with_box_visible"] += object_id in box_scene["visible"]
        reason = blocked(scene_key)
        s[f"blocked_{reason}" if reason else "rows_usable"] += 1
        if not reason and object_id in box_scene["visible"]:
            s["rows_usable_visible"] += 1
            usable_objects[key].add((scene_key, object_id))

    text_summary = {}
    for key in sorted(stats):
        text_summary[key] = {**dict(stats[key]), "scenes": len(scene_sets[key]), "objects": len(objects[key]),
                             "usable_objects": len(usable_objects[key]),
                             "words_median": median(words[key]) if words[key] else 0,
                             "examples": examples[key]}

    # Category names alone (no extra text source): every visible named box of a usable scene.
    names = defaultdict(Counter)
    category_counts = Counter()
    for key, scene in boxes.items():
        dataset = key.split("/")[0]
        reason = blocked(key)
        names[dataset][f"scenes_{reason or 'usable'}"] += 1
        if reason:
            continue
        for object_id in scene["visible"]:
            category = scene["objects"][object_id]
            if category not in EXCLUDED:
                names[dataset]["boxes_usable"] += 1
                if dataset in ("scannet", "3rscan"):
                    category_counts[category] += 1
    counts = sorted(category_counts.values(), reverse=True)

    with open(SOURCE / "sceneverse" / "scannet_scene_cap.json") as f:
        scene_caps = json.load(f)

    result = {
        "boxes_v1": summary_v1, "boxes_v2": summary_v2,
        "v1_equals_v2_on_shared_scenes": all(
            boxes_v1[k]["objects"] == boxes[k]["objects"] for k in boxes_v1),
        "frames_on_disk": disk,
        "eval_scenes": {k: len(v) for k, v in evals.items()},
        "eval_scenes_in_embodiedscan_train": sorted(
            k for k in eval_all if k in boxes and boxes[k]["split"] == "train"),
        "text": text_summary,
        "category_names": {k: dict(v) for k, v in names.items()},
        "categories_scannet_3rscan": {
            "categories": len(counts), "boxes": sum(counts),
            "categories_ge_100": sum(c >= 100 for c in counts),
            "categories_ge_20": sum(c >= 20 for c in counts),
            "boxes_in_top20_categories": sum(counts[:20]),
            "top20": category_counts.most_common(20),
        },
        "scene_captions": {"sceneverse_scannet_scenes": len(scene_caps),
                           "captions": sum(len(v["captions"]) for v in scene_caps.values())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "text"}, indent=1))
    for key, value in text_summary.items():
        print(key, {k: v for k, v in value.items() if k != "examples"})


if __name__ == "__main__":
    main()
