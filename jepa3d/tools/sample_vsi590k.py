#!/usr/bin/env python3
"""VSI-590K (ScanNet + ScanNet++ part) -> posed N-view manifest.

Each QA row references a whole scene video; its views are drawn from the
scene's posed frame pool (jepa3d/frame_pool.py): bad frames dropped by the
frame_stats.py thresholds, then N frames by pose-space FPS, in time order.

Text:
    query   question only: no "<image>", "These are frames of a video.",
            option block or answer-format hint (options kept in "options")
    target  option text for multiple choice (no letter), number + unit
            parsed from the question for numeric answers, bare number for counts
    answer_key  normalized target; rows with equal keys are masked as
            negatives in InfoNCE

Usage:
    python jepa3d/tools/sample_vsi590k.py --num-views 8 --stats data/frame_stats/*.jsonl \
        --output data/vsi590k_sn_f8.jsonl
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
SUBSETS = {"scannet": "scannet", "scannetppv2": "scannetpp"}
OPTION = re.compile(r"^([A-H])\.\s*(.+)$")
HINTS = ("Answer with the option's letter", "Please answer the question using a single word or phrase")
# longest first: "square feet" before "feet", "centimeters" before "meters"
UNITS = ("square feet", "square meters", "centimeters", "inches", "meters", "feet")

_ctx = {}


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip()


def answer_key(target: str) -> str:
    return norm_text(target).lower().rstrip(".")


def parse_text(row):
    """-> (query, target, options) or None for an unparseable row."""
    human = row["conversations"][0]["value"]
    answer = row["conversations"][1]["value"].strip()
    lines = [l.strip() for l in human.replace("<image>", "").splitlines() if l.strip()]
    lines = [l for l in lines if l != "These are frames of a video." and not l.startswith(HINTS)]
    options = {}
    question = []
    for line in lines:
        m = OPTION.match(line)
        if m:
            options[m.group(1)] = m.group(2).strip()
        elif line != "Options:":
            question.append(line)
    query = norm_text(" ".join(question))
    if options:
        if answer.rstrip(".") not in options:
            return None
        return query, options[answer.rstrip(".")], list(options.values())
    qtype = row["question_type"]
    if qtype.endswith("count"):
        return query, answer, []
    if qtype == "absolute_direction_object":
        return query, f"{answer} degrees", []
    low = query.lower()
    unit = next((u for u in UNITS if u in low), None)
    return query, (f"{answer} {unit}" if unit else answer), []


def _scene_rows(item):
    (subset, scene), rows = item
    try:
        pool, keep, dist = prepare_scene(subset, scene, _ctx["stats"], SOURCE / "spar",
                                         SOURCE / "scannet" / "posed_images", _ctx["thresholds"])
    except OSError:
        return [], Counter({"no_pool": len(rows)})
    out, drops = [], Counter()
    for line_no, row in rows:
        parsed = parse_text(row)
        if parsed is None:
            drops["unparsed"] += 1
            continue
        frames = sample_views(pool, keep, dist, _ctx["num_views"], f"vsi590k:{line_no}")
        if frames is None:
            drops["too_few_frames"] += 1
            continue
        query, target, options = parsed
        out.append({
            **view_fields(pool, frames),
            "query": query,
            "target": target,
            "options": options,
            "answer_raw": row["conversations"][1]["value"],
            "answer_key": answer_key(target),
            "group_id": f"{subset}/{scene}",
            "source": "vsi590k",
            "subset": subset,
            "scene": scene,
            "question_type": row["question_type"],
            "src_line": line_no,
        })
    return out, drops


def _init(stats_paths, thresholds, num_views):
    _ctx["stats"] = load_stats(stats_paths)
    _ctx["thresholds"] = thresholds
    _ctx["num_views"] = num_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SOURCE / "vsi590k" / "vsi_590k.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--min-valid-frac", type=float, default=0.5)
    parser.add_argument("--min-median-depth", type=float, default=0.6)
    parser.add_argument("--min-rel-sharpness", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N scenes")
    args = parser.parse_args()

    by_scene = defaultdict(list)
    with open(args.input) as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            video = row.get("video")
            if not isinstance(video, str) or video.split("/")[0] not in SUBSETS:
                continue
            by_scene[(SUBSETS[video.split("/")[0]], Path(video).stem)].append((line_no, row))
    scenes = sorted(by_scene.items())[: args.limit or None]
    thresholds = {"min_valid_frac": args.min_valid_frac, "min_median_depth": args.min_median_depth,
                  "min_rel_sharpness": args.min_rel_sharpness}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    counts, types = Counter(), Counter()
    with open(args.output, "w") as out, Pool(args.workers, _init, (args.stats, thresholds, args.num_views)) as pool:
        for rows, drops in pool.imap_unordered(_scene_rows, scenes, chunksize=2):
            counts.update(drops)
            for r in rows:
                out.write(json.dumps(r) + "\n")
                counts[f"ok_{r['subset']}"] += 1
                types[r["question_type"]] += 1
    print(json.dumps({"counts": counts, "question_types": types}))


if __name__ == "__main__":
    main()
