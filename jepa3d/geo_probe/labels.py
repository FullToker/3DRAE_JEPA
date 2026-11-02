#!/usr/bin/env python3
"""EmbodiedScan ScanNet boxes -> per-scene object labels for the geometry probe.

Writes data/geo_probe/labels_{train,val}.jsonl, one scene per line:
    scene, axis_align (4x4),
    objects: [{id, category, center}]   single-instance categories, aligned frame
    visible: {frame_id: [object ids]}   frames EmbodiedScan lists for the scene

Usage (from the repo root):
    python jepa3d/geo_probe/labels.py --output-dir data/geo_probe
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.paths import VJEPA_ROOT

INFOS = VJEPA_ROOT / "source_data" / "embodiedscan" / "embodiedscan" / "embodiedscan_infos_{}.pkl"
# Room structure and catch-all classes: no single object to locate.
EXCLUDED = {"wall", "floor", "ceiling", "object", "structure", "furniture", "support", "decoration"}


def scene_labels(info: dict, id2cat: dict) -> dict:
    per_category = Counter(id2cat[i["bbox_label_3d"]] for i in info["instances"])
    objects = []
    for inst in info["instances"]:
        category = id2cat[inst["bbox_label_3d"]]
        if per_category[category] == 1 and category not in EXCLUDED:
            objects.append({"id": int(inst["bbox_id"]), "category": category,
                            "center": [float(x) for x in inst["bbox_3d"][:3]]})
    kept = {o["id"] for o in objects}
    visible = {}
    for image in info["images"]:
        frame_id = int(Path(image["img_path"]).stem)
        visible[frame_id] = sorted(kept & {int(i) for i in image["visible_instance_ids"]})
    return {
        "scene": info["sample_idx"].split("/")[1],
        "axis_align": [[float(x) for x in row] for row in info["axis_align_matrix"]],
        "objects": objects,
        "visible": visible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    seen = {}
    for split in ("train", "val"):
        with open(str(INFOS).format(split), "rb") as f:
            infos = pickle.load(f)
        id2cat = {v: k for k, v in infos["metainfo"]["categories"].items()}
        rows = [scene_labels(x, id2cat) for x in infos["data_list"] if x["sample_idx"].startswith("scannet/")]
        seen[split] = {r["scene"] for r in rows}
        with open(output / f"labels_{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        counts = sorted(len(r["objects"]) for r in rows)
        print(json.dumps({"split": split, "scenes": len(rows), "objects_median": counts[len(counts) // 2],
                          "scenes_lt2_objects": sum(c < 2 for c in counts),
                          "scenes_lt3_objects": sum(c < 3 for c in counts),
                          "frames_per_scene": sum(len(r["visible"]) for r in rows) / len(rows)}))
    overlap = seen["train"] & seen["val"]
    if overlap:
        raise RuntimeError(f"train and val share scenes: {sorted(overlap)[:5]}")


if __name__ == "__main__":
    main()
