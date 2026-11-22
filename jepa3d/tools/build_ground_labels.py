#!/usr/bin/env python3
"""True 3D positions for the objects a question names, from what is already on disk.

Nothing in source_data or in the input manifests is changed; everything is written to --output-dir:
    scene_objects.json   {"<subset>/<scene>": {"pos": [[x, y, z]], "names": [{name: votes}], "source": [...]}}
                         the objects of a scene, in the scene's own (pose) frame, metres
    <manifest name>      the input rows with "ground": [{"slot": i, "object": j, "how": ...}] added:
                         the row's slot i (its name and spans) is object j of the scene's table
    labels.stats.json    counts and the checks below

Where the positions come from:
  boxes   ScanNet: the EmbodiedScan box centres.
  depth   scenes without boxes (ScanNet++): SPAR's depth_prediction_oc questions give, for two objects of
          one image, the pixel of the 3D centre and its depth in the text ("the shelf (red point) lies at
          1.2 meters depth", answer "2.3"). Pixel + depth + pose = the centre; centres closer than
          CLUSTER metres are one object.
  rays    an object no depth question mentions: the centre pixels that SPAR's spatial_imagination
          questions give for it in several frames are rays through the centre; their intersection is
          taken if all rays of that name pass within RAY_FIT metres of it.

Which object a name means:
  row     SPAR spatial_imagination_*_video_hard rows: the raw row (the manifest's "src") holds the centre
          pixel of each "(ObjectK)" in one frame (green = Object0, blue = Object1, red = Object2,
          yellow = Object3; pixels in 0..1000 of width and height). The object whose centre projects
          onto that pixel is meant; this also tells apart several objects of one name.
  name    every other row (VSI-590K, appearance_order, obj_count): the objects of the scene are named by
          the SPAR rows above (and by their EmbodiedScan category); a name that exactly one object of
          the scene carries is that object.
A label is kept only if the centre projects into one of the row's views (no occlusion test).

Checks, on ScanNet where the boxes are the truth: how far SPAR's pixel is from the projected box centre,
and how far the "depth" / "rays" position of an object is from its box centre.

Usage (from the repo root):
    python jepa3d/tools/build_ground_labels.py --manifests experiments/slots_v2/data/train_f8.jsonl \
        experiments/slots_v2/data/holdout_f8.jsonl --output-dir experiments/ground_v1/data
"""

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.geo_probe.labels import EXCLUDED
from jepa3d.paths import VJEPA_ROOT
from jepa3d.spar_cameras import load_camera
from jepa3d.tools.check_objtext_data import INFOS, norm
from jepa3d.tools.sample_spar_video import objects_of

SPAR = VJEPA_ROOT / "source_data" / "spar"
SUBSETS = ("scannet", "scannetpp")
VIDEO_TYPES = ("spatial_imagination_oc_video_hard", "spatial_imagination_oo_video_hard")
COLOURS = ("green", "blue", "red", "yellow")   # the colour of Object0, 1, 2, 3
METRES = re.compile(r"(\d+(?:\.\d+)?)\s*meters?")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
PIXEL = 0.02       # a box centre "is" a pixel if it projects within this share of the image size
PIXEL_LOOSE = 0.04 # the same for a depth / ray centre, which is a few centimetres off
CLUSTER = 0.12     # metres: depth centres closer than this are one object
RAY_FIT = 0.15     # metres: largest distance of a ray to the intersection it is part of
MAX_SAMPLES = 20000


def load_boxes() -> dict:
    """{"scannet/<scene>": [(category, centre in the scan's own frame)]}"""
    scenes = {}
    for split in ("train", "val"):
        with open(INFOS["v1"] / f"embodiedscan_infos_{split}.pkl", "rb") as f:
            infos = pickle.load(f)
        id2cat = {v: norm(k) for k, v in infos["metainfo"]["categories"].items()}
        for info in infos["data_list"]:
            if not info["sample_idx"].startswith("scannet/"):
                continue
            unalign = np.linalg.inv(np.array(info["axis_align_matrix"], dtype=np.float64))
            scenes[info["sample_idx"]] = [
                (id2cat[i["bbox_label_3d"]], unalign[:3, :3] @ np.array(i["bbox_3d"][:3], dtype=np.float64)
                 + unalign[:3, 3])
                for i in info["instances"] if id2cat[i["bbox_label_3d"]] not in EXCLUDED]
    return scenes


def read_spar(limit_scenes: int = 0) -> dict:
    """{"<subset>/<scene>": {"video": [(src, [(name, image, (x, y))])], "depth": [(image, (x, y), z)]}}"""
    scenes = defaultdict(lambda: {"video": [], "depth": []})
    for subset in SUBSETS:
        root = SPAR / subset / "qa_jsonl" / "train"
        for qtype in VIDEO_TYPES:
            for path in sorted((root / qtype / "sentence").glob("*.jsonl")):
                with open(path) as f:
                    for line_no, line in enumerate(f):
                        raw = json.loads(line)
                        names = objects_of(raw["conversations"][0]["value"], raw["conversations"][1]["value"])
                        frames = raw["point_img_idx"][0]
                        mentions = []
                        for k, colour in enumerate(COLOURS[:len(frames)]):
                            if str(k) in names and f"{colour}_point" in raw:
                                mentions.append((k, names[str(k)], raw["image"][frames[k]],
                                                 tuple(raw[f"{colour}_point"][0])))
                        scene = raw["image"][0].split("/")[0]
                        scenes[f"{subset}/{scene}"]["video"].append((f"{path.relative_to(SPAR)}:{line_no}", mentions))
        for path in sorted((root / "depth_prediction_oc").glob("*/*.jsonl")):
            kind = path.parent.name
            with open(path) as f:
                for line in f:
                    raw = json.loads(line)
                    question, answer = raw["conversations"][0]["value"], raw["conversations"][1]["value"]
                    reference = METRES.search(question.split("\n")[0])
                    if kind == "select":
                        option = re.search(rf"\n{re.escape(answer.strip()[:1])}\.\s*(\d+(?:\.\d+)?)", question)
                        asked = option.group(1) if option else None
                    else:
                        numbers = NUMBER.findall(answer)
                        asked = numbers[-1] if numbers else None
                    scene = raw["image"][0].split("/")[0]
                    for colour, depth in (("red", reference.group(1) if reference else None), ("blue", asked)):
                        if depth is not None and float(depth) > 0 and f"{colour}_point" in raw:
                            scenes[f"{subset}/{scene}"]["depth"].append(
                                (raw["image"][0], tuple(raw[f"{colour}_point"][0]), float(depth)))
    keys = sorted(scenes)[:limit_scenes or None]
    return {k: scenes[k] for k in keys}


class Cameras:
    """K, c2w, w2c and image size of a scene's SPAR images."""

    def __init__(self, subset: str):
        self.root = SPAR / subset / "images"
        self.cams, self.sizes = {}, {}

    def __call__(self, image: str):
        if image not in self.cams:
            path = self.root / image
            if path.parent not in self.sizes:
                with Image.open(path) as im:
                    self.sizes[path.parent] = im.size
            K, c2w = load_camera(path)
            self.cams[image] = (K, c2w, np.linalg.inv(c2w), self.sizes[path.parent])
        return self.cams[image]

    def centre(self, image: str, point, depth: float) -> np.ndarray:
        """The world point at this depth (camera z) behind the pixel (0..1000 of width, height)."""
        K, c2w, _, (width, height) = self(image)
        u, v = point[0] / 1000 * width, point[1] / 1000 * height
        cam = np.array([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth, 1.0])
        return (c2w @ cam)[:3]

    def ray(self, image: str, point):
        K, c2w, _, (width, height) = self(image)
        u, v = point[0] / 1000 * width, point[1] / 1000 * height
        direction = c2w[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        return c2w[:3, 3], direction / np.linalg.norm(direction)

    def pixel_distance(self, image: str, point, positions: np.ndarray) -> np.ndarray:
        """Distance (share of the image size) of the pixel to each position's projection; inf behind the camera."""
        K, _, w2c, (width, height) = self(image)
        cam = positions @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.where(cam[:, 2] > 0.05, cam[:, 2], np.nan)
        u = (K[0, 0] * cam[:, 0] / z + K[0, 2]) / width
        v = (K[1, 1] * cam[:, 1] / z + K[1, 2]) / height
        return np.nan_to_num(np.hypot(u - point[0] / 1000, v - point[1] / 1000), nan=np.inf)


def cluster(points: list) -> np.ndarray:
    """Greedy: a point joins the nearest cluster within CLUSTER metres, else starts one. -> cluster means."""
    sums, counts = [], []
    for p in points:
        if sums:
            means = np.array(sums) / np.array(counts)[:, None]
            dist = np.linalg.norm(means - p, axis=1)
            i = int(dist.argmin())
            if dist[i] < CLUSTER:
                sums[i] = sums[i] + p
                counts[i] += 1
                continue
        sums.append(p.copy())
        counts.append(1)
    return np.array(sums) / np.array(counts)[:, None] if sums else np.zeros((0, 3))


def pick(distance: np.ndarray, limit: float) -> int:
    """The object a pixel means: the nearest projection, unless a second one is nearly as near. -1: none."""
    if not len(distance):
        return -1
    order = np.argsort(distance)
    best = distance[order[0]]
    if best > limit or (len(order) > 1 and distance[order[1]] <= limit and distance[order[1]] < 2 * best + 1e-4):
        return -1
    return int(order[0])


def intersect(rays: list):
    """Least-squares intersection of rays [(origin, direction)], or None if they do not meet in one point."""
    if len(rays) < 2:
        return None
    A, b = np.zeros((3, 3)), np.zeros(3)
    for origin, direction in rays:
        P = np.eye(3) - np.outer(direction, direction)
        A += P
        b += P @ origin
    if np.linalg.eigvalsh(A)[0] / len(rays) < 0.02:   # nearly parallel rays
        return None
    point = np.linalg.solve(A, b)
    for origin, direction in rays:
        offset = point - origin
        if offset @ direction <= 0.05 or np.linalg.norm(offset - (offset @ direction) * direction) > RAY_FIT:
            return None
    return point


def locate(cams: Cameras, video: list, positions: np.ndarray, with_rays: bool, limit: float):
    """Assign every mention of the video rows to an object. positions (M, 3) are the known objects;
    with_rays adds objects found by intersecting the rays of unassigned mentions of one name.
    -> positions (M', 3), source of the added ones, {src: [(k, name, object index or -1)]}, pixel distances."""
    assigned, loose, nearest = {}, defaultdict(list), []
    for src, mentions in video:
        row = []
        for k, name, image, point in mentions:
            distance = cams.pixel_distance(image, point, positions) if len(positions) else np.zeros(0)
            j = pick(distance, limit)
            if len(distance):
                nearest.append(float(distance.min()))
            if j < 0:
                loose[name].append((src, k, image, point))
            row.append([k, name, j])
        assigned[src] = row
    added = 0
    if with_rays:
        extra = []
        for name, items in sorted(loose.items()):
            frames = {image: point for _, _, image, point in items}
            point = intersect([cams.ray(image, pixel) for image, pixel in frames.items()])
            if point is None:
                continue
            index = len(positions) + len(extra)
            extra.append(point)
            for src, k, _, _ in items:
                for entry in assigned[src]:
                    if entry[0] == k:
                        entry[2] = index
        if extra:
            positions = np.concatenate([positions, np.array(extra)])
            added = len(extra)
    return positions, added, assigned, nearest


def process_scene(item):
    key, spar, boxes = item
    subset = key.split("/")[0]
    cams = Cameras(subset)
    out = {"key": key, "check": {}}
    try:
        centres = [cams.centre(image, point, depth) for image, point, depth in spar["depth"]]
    except OSError as error:
        return {"key": key, "error": str(error)}
    clusters = cluster(centres)
    try:
        if boxes:
            positions = np.array([c for _, c in boxes])
            positions, _, assigned, nearest = locate(cams, spar["video"], positions, with_rays=False, limit=PIXEL)
            source = ["box"] * len(positions)
            # the check: the same mentions located without the boxes, as in a scene that has none
            other, added, guess, _ = locate(cams, spar["video"], clusters, with_rays=True, limit=PIXEL_LOOSE)
            errors = {"depth": [], "rays": []}
            mentions = found = 0
            for src, row in assigned.items():
                for (_, _, j), (_, _, g) in zip(row, guess[src]):
                    if j < 0:
                        continue
                    mentions += 1
                    if g >= 0:
                        found += 1
                        errors["depth" if g < len(clusters) else "rays"].append(
                            float(np.linalg.norm(other[g] - positions[j])))
            out["check"] = {"pixel": nearest, "errors": errors, "mentions": mentions, "found": found}
        else:
            positions, added, assigned, _ = locate(cams, spar["video"], clusters, with_rays=True, limit=PIXEL_LOOSE)
            source = ["depth"] * len(clusters) + ["rays"] * added
    except OSError as error:
        return {"key": key, "error": str(error)}
    names = [Counter() for _ in positions]
    for row in assigned.values():
        for _, name, j in row:
            if j >= 0:
                names[j][norm(name)] += 1
    if boxes:
        for j, (category, _) in enumerate(boxes):
            names[j][category] += 0   # a name with no vote: the EmbodiedScan category
    out.update(pos=[[round(float(x), 4) for x in p] for p in positions], names=[dict(n) for n in names],
               source=source, rows={src: row for src, row in assigned.items()},
               counts={"depth_mentions": len(centres), "video_rows": len(spar["video"])})
    return out


def in_view(position: np.ndarray, row: dict) -> bool:
    for K, c2w in zip(row["intrinsics"], row["c2w"]):
        K = np.array(K, dtype=np.float64)
        cam = np.linalg.inv(np.array(c2w, dtype=np.float64)) @ np.append(position, 1.0)
        if cam[2] <= 0.1:
            continue
        u, v = K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]
        if 0 <= u < 2 * K[0, 2] and 0 <= v < 2 * K[1, 2]:
            return True
    return False


def objects_named(name: str, by_name: dict) -> list:
    name = norm(name)
    for candidate in (name, name[:-1] if name.endswith("s") else None, name[:-2] if name.endswith("es") else None):
        if candidate and candidate in by_name:
            return by_name[candidate]
    return []


def label_row(row: dict, scene: dict, tally: Counter) -> list:
    slots = row.get("slots") or []
    tally["names"] += len(slots)
    if scene is None or not slots:
        tally["names_scene_without_objects"] += len(slots) if scene is None else 0
        return []
    positions = scene["positions"]
    ground = []
    raw = scene["rows"].get(row.get("src")) if row.get("source") == "spar" else None
    if raw is not None:   # the raw row says which object each (ObjectK) is
        for _, name, j in raw:
            last = norm(name).split()[-1]
            match = [i for i, slot in enumerate(slots) if norm(slot["name"]).split()[-1] == last]
            if len(match) != 1:
                tally["names_row_not_matched"] += 1
            elif j < 0:
                tally["names_row_pixel_without_object"] += 1
            else:
                ground.append({"slot": match[0], "object": j, "how": "row"})
    else:
        for i, slot in enumerate(slots):
            found = objects_named(slot["name"], scene["by_name"])
            if not found:
                tally["names_no_such_object"] += 1
            elif len(found) > 1:
                tally["names_several_objects"] += 1
            else:
                ground.append({"slot": i, "object": found[0], "how": "name"})
    kept = []
    for entry in ground:
        if in_view(positions[entry["object"]], row):
            kept.append(entry)
            tally[f"labelled_{entry['how']}_{scene['source'][entry['object']]}"] += 1
        else:
            tally["names_out_of_view"] += 1
    tally["names_labelled"] += len(kept)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-scenes", type=int, default=0, help="debug: first N scenes, no manifests written")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    boxes = load_boxes()
    spar = read_spar(args.limit_scenes)
    keys = sorted(spar) if args.limit_scenes else sorted(set(spar) | set(boxes))
    print(f"[scenes] {len(keys)} ({len(boxes)} with boxes, {len(spar)} in SPAR)", flush=True)
    items = [(k, spar.get(k, {"video": [], "depth": []}), boxes.get(k)) for k in keys]

    scenes, table, failed = {}, {}, []
    check = {"pixel": [], "depth": [], "rays": [], "mentions": 0, "found": 0}
    sources, per_scene = Counter(), []
    with Pool(args.workers) as pool:
        for n, result in enumerate(pool.imap_unordered(process_scene, items, chunksize=4)):
            if "error" in result:
                failed.append((result["key"], result["error"]))
                continue
            key = result["key"]
            by_name = defaultdict(list)
            for j, names in enumerate(result["names"]):
                for name in names:
                    by_name[name].append(j)
            scenes[key] = {"positions": np.array(result["pos"]).reshape(-1, 3), "by_name": dict(by_name),
                           "rows": result["rows"], "source": result["source"]}
            table[key] = {"pos": result["pos"], "names": result["names"], "source": result["source"]}
            sources.update(result["source"])
            if key.startswith("scannetpp/"):
                per_scene.append(len(result["pos"]))
            if result["check"]:
                check["pixel"] += result["check"]["pixel"]
                check["depth"] += result["check"]["errors"]["depth"]
                check["rays"] += result["check"]["errors"]["rays"]
                check["mentions"] += result["check"]["mentions"]
                check["found"] += result["check"]["found"]
            if n % 200 == 0:
                print(f"[scene] {n}/{len(items)}", flush=True)

    def summary(values):
        values = np.array(values)
        if not len(values):
            return {"n": 0}
        rng = np.random.default_rng(0)
        sample = values if len(values) <= MAX_SAMPLES else rng.choice(values, MAX_SAMPLES, replace=False)
        return {"n": len(values), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
                "mean": float(values.mean()), "sample": [round(float(x), 4) for x in sample]}

    stats = {
        "scenes": len(scenes), "failed_scenes": failed[:20], "objects_by_source": dict(sources),
        "scannetpp_objects_per_scene": summary(per_scene),
        "check_scannet": {
            "pixel_distance_to_nearest_box": summary(check["pixel"]),
            "pixel_within_threshold": float((np.array(check["pixel"]) <= PIXEL).mean()) if check["pixel"] else None,
            "mentions_with_box": check["mentions"], "found_without_boxes": check["found"],
            "error_m_depth": summary(check["depth"]), "error_m_rays": summary(check["rays"]),
        },
        "manifests": {},
    }
    print(json.dumps({k: (v if k != "check_scannet" else {a: ({x: y for x, y in b.items() if x != "sample"}
                                                              if isinstance(b, dict) else b) for a, b in v.items()})
                      for k, v in stats.items() if k not in ("manifests", "scannetpp_objects_per_scene")}, indent=1),
          flush=True)
    if args.limit_scenes:
        return
    (output / "scene_objects.json").write_text(json.dumps(table))

    for path in args.manifests:
        groups = defaultdict(Counter)
        with open(path) as src, open(output / Path(path).name, "w") as dst:
            for line in src:
                row = json.loads(line)
                tally = Counter(rows=1)
                ground = label_row(row, scenes.get(f"{row.get('subset')}/{row['scene']}"), tally)
                tally["rows_with_names"] = bool(row.get("slots"))
                tally["rows_some_name_labelled"] = len(ground) > 0
                tally["rows_two_names_labelled"] = len(ground) >= 2
                tally["rows_all_names_labelled"] = bool(row.get("slots")) and len(ground) == len(row["slots"])
                for group in ("all", f"subset:{row.get('subset')}", f"{row.get('source')}/{row.get('question_type')}"):
                    groups[group].update(tally)
                row["ground"] = ground
                dst.write(json.dumps(row) + "\n")
        stats["manifests"][Path(path).name] = {g: dict(c) for g, c in sorted(groups.items())}
        print(f"\n== {path}")
        for group, c in sorted(groups.items()):
            print(f"{group:55s} rows {c['rows']:7d}  >=1 labelled {c['rows_some_name_labelled']:7d}  "
                  f"all labelled {c['rows_all_names_labelled']:7d}  names {c['names']:7d}  "
                  f"labelled {c['names_labelled']:7d}", flush=True)
    (output / "labels.stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
