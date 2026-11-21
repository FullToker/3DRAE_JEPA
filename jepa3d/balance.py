#!/usr/bin/env python3
"""Answer classes, their inverse-frequency weights, and numeric answers of manifest rows.

Weights. Within a question type some answers are much more frequent ("2" is
half of all counts, "left" / "right" are 3/4 of the directions), so a model
scores without the images by answering the frequent one. Every row gets
    weight = 1 / (number of classes * share of its answer class in its type)
clipped to [1 / MAX_WEIGHT, MAX_WEIGHT] and rescaled to mean 1 per type, so a
type's total weight does not change. Types without a class (free numbers,
orders) keep weight 1. answer_class(row):
    relative_direction_object      the options offered + the answer (left / right / back / quadrants)
    relative_count                 more / fewer / equal number
    relative_distance / _size      the object named as the answer
    spatial_imagination_*          the answer's relation words (right|above|...)
    obj_count, absolute_count      the number

Numeric answers. numeric_target(row) -> (kind, value) in one unit per kind:
    count  objects        distance  metres        size  metres        area  square metres
for the training rows (SPAR answer_key, VSI-590K target_short with its unit)
and for VSI-Bench rows (the benchmark's unit per question type). Angles are
left to the answer embedding.

This file as a script writes the weight table of a training manifest:
    python jepa3d/balance.py --manifest <train.jsonl> --output <weights.json>
-> {"weights": {type: {class: weight}}, "shares": {type: {class: share}},
    "numeric_median": {kind: value}, "rows": N}
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MAX_WEIGHT = 5.0
KINDS = ("count", "distance", "size", "area")
KIND_OF = {"obj_count": "count", "absolute_count": "count", "object_counting": "count",
           "absolute_distance_object": "distance", "object_abs_distance": "distance",
           "absolute_size_object": "size", "object_size_estimation": "size",
           "room_size": "area", "absolute_size_room": "area", "room_size_estimation": "area"}
UNITS = {"meters": 1.0, "meter": 1.0, "feet": 0.3048, "foot": 0.3048, "inches": 0.0254, "inch": 0.0254,
         "centimeters": 0.01, "centimeter": 0.01, "square meters": 1.0, "square feet": 0.09290304}
BENCH_UNIT = {"object_counting": 1.0, "object_abs_distance": 1.0, "object_size_estimation": 0.01,
              "room_size_estimation": 1.0}   # VSI-Bench answers: count, metres, centimetres, square metres
NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?)\s*([a-z ]*)$")


def type_of(row: dict) -> str:
    return str(row.get("question_type") or row.get("type"))


def type_key(row: dict) -> str:
    return f"{row.get('source', '-')}/{type_of(row)}"


def answer_class(row: dict):
    """The class of the row's answer within its question type, or None."""
    kind, short = type_of(row), row.get("target_short")
    if kind == "relative_direction_object" and short:
        return "/".join(sorted(set(row.get("options") or []))) + " -> " + short
    if kind in ("relative_count", "relative_distance_object", "relative_size_object") and short:
        return short
    if kind.startswith("spatial_imagination"):
        return str(row.get("answer_key", "")).rsplit(":", 1)[-1] or None
    if KIND_OF.get(kind) == "count":
        target = numeric_target(row)
        return None if target is None else str(int(round(target[1])))
    return None


def numeric_target(row: dict):
    """(kind, value) of a numeric answer in the kind's unit, or None."""
    kind = KIND_OF.get(type_of(row))
    if kind is None:
        return None
    qtype = type_of(row)
    if qtype in BENCH_UNIT:                                   # VSI-Bench: plain number
        text, factor = str(row.get("target", "")), BENCH_UNIT[qtype]
    elif row.get("target_short") is not None:                 # VSI-590K: "1.1 meters", "2"
        text, factor = str(row["target_short"]), None
    else:                                                     # SPAR: "count:8:paper", "area:9"
        parts = str(row.get("answer_key", "")).split(":")
        text, factor = (parts[1] if len(parts) > 1 else ""), 1.0
    match = NUMBER.match(text.strip().lower())
    if not match:
        return None
    if factor is None:
        unit = match.group(2).strip()
        factor = 1.0 if not unit and kind == "count" else UNITS.get(unit)
        if factor is None:
            return None
    value = float(match.group(1)) * factor
    return (kind, value) if value > 0 else None


def weight_table(counts: dict) -> dict:
    """{type: Counter(class)} -> {type: {class: weight}}."""
    table = {}
    for key, classes in counts.items():
        total = sum(classes.values())
        raw = {c: min(max(total / (len(classes) * n), 1 / MAX_WEIGHT), MAX_WEIGHT) for c, n in classes.items()}
        mean = sum(raw[c] * n for c, n in classes.items()) / total
        table[key] = {c: round(w / mean, 4) for c, w in raw.items()}
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    counts, values, rows = defaultdict(Counter), defaultdict(list), 0
    with open(args.manifest) as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            cls = answer_class(row)
            if cls is not None:
                counts[type_key(row)][cls] += 1
            target = numeric_target(row)
            if target:
                values[target[0]].append(target[1])
    result = {"manifest": args.manifest, "rows": rows, "max_weight": MAX_WEIGHT,
              "weights": weight_table(counts),
              "shares": {k: {c: round(n / sum(v.values()), 4) for c, n in v.most_common()} for k, v in counts.items()},
              "numeric_rows": {k: len(v) for k, v in values.items()},
              "numeric_median": {k: statistics.median(v) for k, v in values.items()}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))
    for key in sorted(result["weights"]):
        top = list(result["shares"][key].items())[:4]
        print(key, len(result["weights"][key]), "classes;", [(c, s, result["weights"][key][c]) for c, s in top])
    print("numeric", result["numeric_rows"], result["numeric_median"])


if __name__ == "__main__":
    main()
