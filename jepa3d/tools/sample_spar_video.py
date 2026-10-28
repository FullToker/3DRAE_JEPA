#!/usr/bin/env python3
"""SPAR-7M scene-level video QA -> posed N-view manifest (strict filter).

Kept: the five 32-frame types whose question needs no drawn marker and
no frame index, sentence kind only, ScanNet + ScanNet++:

    obj_count, room_size, appearance_order,
    spatial_imagination_oc_video_hard, spatial_imagination_oo_video_hard

All five ask about the whole scene, so views come from the scene's full
posed pool (jepa3d/frame_pool.py, same sampler as VSI-590K), not only
SPAR's fixed 32 frames.

Row filters / cleanup:
    empty_slot     answer has a template slot left blank ("to the left and  sink")
    shared_noun    two referenced objects end in the same noun (ambiguous without markers)
    duplicate      same scene + query + target seen before
    vsibench_scene scene is in VSI-Bench (train split only)
    spacing        " ." / " ," and repeated blanks fixed (kept)
appearance_order: the target keeps only the final order; the "X at 14"
frame numbers refer to SPAR's 32 frames and mean nothing for other views.

answer_key: a canonical form of the answer (count + category, area, order,
objects + spatial terms) so paraphrases of one answer mask each other in InfoNCE.

Usage:
    python jepa3d/tools/sample_spar_video.py --split train --num-views 8 \
        --stats data/frame_stats/*.jsonl --output data/spar_video_train_f8.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.frame_pool import load_stats, prepare_scene, sample_views, view_fields
from jepa3d.paths import VJEPA_ROOT

SOURCE = VJEPA_ROOT / "source_data"
SUBSETS = ("scannet", "scannetpp")
TYPES = ("obj_count", "room_size", "appearance_order",
         "spatial_imagination_oc_video_hard", "spatial_imagination_oo_video_hard")
OBJECT = re.compile(r"((?:[A-Za-z]+ ){0,3}[A-Za-z]+) \(Object(\d)\)")
SPATIAL = re.compile(r"\b(left|right|above|below|front|behind|closer|farther)\b")
COUNT_CATEGORY = re.compile(
    r"how many (.+?) (?:can|could|would|are|is|do|does|occupy|occupies|exist|exists|appear|appears"
    r"|might|will|have|has|there|in|inside|you)\b", re.I)
EMPTY_SLOT = re.compile(r"[^\s.] {2,}\S")
LEAD_WORDS = {"a", "an", "the", "to", "at", "on", "of", "in", "by", "from", "with", "and", "toward",
              "towards", "near", "facing", "face", "faces", "about", "does", "do", "is", "are", "where",
              "how", "what", "perceive", "moving", "moved", "move", "focus", "focused", "focusing",
              "looking", "look", "turn", "turned", "turning", "settle", "aligning", "aligned"}

_ctx = {}


def clean(text: str) -> str:
    text = text.replace("’", "'")
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"\s+'s\b", "'s", text)
    return re.sub(r"\s+", " ", text).strip()


def objects_of(*texts: str) -> dict:
    """{"0": "spill mat", ...}: the regex also catches words before the name
    ("observer perceive sink"), so keep the word suffix shared by every mention."""
    mentions = defaultdict(list)
    for text in texts:
        for name, idx in OBJECT.findall(text.replace("’", "'")):
            mentions[idx].append(name.lower().split())
    out = {}
    for idx, names in mentions.items():
        suffix = names[0]
        for words in names[1:]:
            n = 0
            while n < min(len(suffix), len(words)) and suffix[-1 - n] == words[-1 - n]:
                n += 1
            suffix = suffix[len(suffix) - n:]
        suffix = suffix or names[0][-1:]
        while len(suffix) > 1 and suffix[0] in LEAD_WORDS:
            suffix = suffix[1:]
        out[idx] = " ".join(suffix)
    return out


def convert(qtype, query_raw, answer_raw):
    """-> (query, target, answer_key) or a drop reason string."""
    if EMPTY_SLOT.search(answer_raw.replace("’", "'")):
        return "empty_slot"
    query, answer = clean(query_raw), clean(answer_raw)
    if qtype.startswith("spatial_imagination"):
        objs = objects_of(query_raw, answer_raw)
        nouns = [n.split()[-1] for n in objs.values()]
        if len(set(nouns)) < len(nouns):
            return "shared_noun"
        names = "|".join(objs[k] for k in sorted(objs))
        return query, answer, f"{qtype}:{names}:{'|'.join(SPATIAL.findall(answer.lower()))}"
    if qtype == "appearance_order":
        order = answer.rsplit(":", 1)[-1].strip().rstrip(".")
        items = [x.strip().lower() for x in order.split(",")]
        if len(items) < 2 or not all(items):
            return "unparsed"
        return query, f"The order of first appearance is: {', '.join(items)}.", "order:" + "|".join(items)
    numbers = re.findall(r"\d+(?:\.\d+)?", answer)
    if not numbers:
        return "unparsed"
    if qtype == "room_size":
        return query, answer, f"area:{round(float(numbers[0]))}"
    m = COUNT_CATEGORY.search(query)
    key = f"count:{numbers[0]}:{m.group(1).lower()}" if m else answer.lower()
    return query, answer, key


def _scene_rows(item):
    (subset, scene), rows = item
    try:
        pool, keep, dist = prepare_scene(subset, scene, _ctx["stats"], SOURCE / "spar",
                                         SOURCE / "scannet" / "posed_images", _ctx["thresholds"])
    except OSError:
        return [], Counter({"no_pool": len(rows)})
    out, drops, seen = [], Counter(), set()
    for src, qtype, raw in rows:
        conv = raw["conversations"]
        result = convert(qtype, conv[0]["value"], conv[1]["value"])
        if isinstance(result, str):
            drops[result] += 1
            continue
        query, target, key = result
        if (query, target) in seen:
            drops["duplicate"] += 1
            continue
        seen.add((query, target))
        frames = sample_views(pool, keep, dist, _ctx["num_views"], f"spar:{src}")
        if frames is None:
            drops["too_few_frames"] += 1
            continue
        out.append({
            **view_fields(pool, frames),
            "query": query,
            "target": target,
            "answer_raw": conv[1]["value"],
            "answer_key": key,
            "group_id": f"{subset}/{scene}",
            "source": "spar",
            "subset": subset,
            "scene": scene,
            "question_type": qtype,
            "src": src,
        })
    return out, drops


def _init(stats_paths, thresholds, num_views):
    _ctx["stats"] = load_stats(stats_paths)
    _ctx["thresholds"] = thresholds
    _ctx["num_views"] = num_views


def vsibench_scenes():
    path = SOURCE / "vsibench" / "test.jsonl"
    with open(path) as f:
        return {json.loads(l)["scene_name"] for l in f if l.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS,
                        help="SPAR's scannetpp val files are byte copies of train: use scannet only for val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--min-valid-frac", type=float, default=0.5)
    parser.add_argument("--min-median-depth", type=float, default=0.6)
    parser.add_argument("--min-rel-sharpness", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N scenes")
    args = parser.parse_args()

    banned = vsibench_scenes() if args.split == "train" else set()
    by_scene, counts = defaultdict(list), Counter()
    for subset in args.subsets:
        for qtype in TYPES:
            for path in sorted((SOURCE / "spar" / subset / "qa_jsonl" / args.split / qtype / "sentence").glob("*.jsonl")):
                with open(path) as f:
                    for line_no, line in enumerate(f):
                        raw = json.loads(line)
                        scene = raw["image"][0].split("/")[0]
                        if scene in banned:
                            counts["vsibench_scene"] += 1
                            continue
                        src = f"{path.relative_to(SOURCE / 'spar')}:{line_no}"
                        by_scene[(subset, scene)].append((src, qtype, raw))
    scenes = sorted(by_scene.items())[: args.limit or None]
    thresholds = {"min_valid_frac": args.min_valid_frac, "min_median_depth": args.min_median_depth,
                  "min_rel_sharpness": args.min_rel_sharpness}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    types = Counter()
    with open(args.output, "w") as out, Pool(args.workers, _init, (args.stats, thresholds, args.num_views)) as pool:
        for rows, drops in pool.imap_unordered(_scene_rows, scenes, chunksize=2):
            for reason, n in drops.items():
                counts[reason] += n
            for r in rows:
                out.write(json.dumps(r) + "\n")
                counts[f"ok_{r['subset']}"] += 1
                types[r["question_type"]] += 1
    print(json.dumps({"split": args.split, "counts": counts, "question_types": types}))


if __name__ == "__main__":
    main()
