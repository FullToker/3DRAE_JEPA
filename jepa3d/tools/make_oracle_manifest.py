#!/usr/bin/env python3
"""Manifests for E1 (jepa3d/oracle.py): direction questions whose every object has a true position.

Source: the relative-direction rows of ground_v1's manifests (VSI-590K: stand at A, face B, where is C)
and the scenes' object tables (scene_objects.json). Only 1 row in 9 has a true position for all three of
its names (a name is labelled only when it is unique in the scene), too few to train on. So two kinds of
row are written:

    real    the rows with all three names labelled, unchanged: the dataset's own question and answer
    synth   any direction row, with its three names re-pointed at three random objects of the same scene.
            The wording, the views and the answer options are the row's own; the answer is recomputed
            from the three centres with VSI-Bench's rule (jepa3d.oracle.azimuth). The names in the text
            no longer say which objects are meant, so these rows only make sense delexicalized
            (jepa3d/delex.py: "object A", ...), which is how E1 trains and evaluates.

The rule is checked on the real rows first: "rule_agreement" in the stats is how often the centres give
the dataset's answer. That number is also the ceiling of any model that is given centres.

All rows get the same "answer_key", so the in-batch InfoNCE term masks every pair and only the choice
among the question's own answers trains the model (delexicalized answers of different rows read alike
and would be false negatives).

Usage (from the repo root):
    python jepa3d/tools/make_oracle_manifest.py --data experiments/ground_v1/data --out experiments/e1_oracle/data
"""

import argparse
import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa3d.oracle import azimuth, boundary_distance, direction_answer

ROLES = (re.compile(r"^Standing at the (.+) and facing the (.+), the (.+) is "),
         re.compile(r"^From the (.+), looking toward the (.+), the (.+) is "),
         re.compile(r"^If I stand at the (.+) facing the (.+), the (.+) is "))
UP = np.array([0.0, 0.0, 1.0])   # ScanNet and ScanNet++ scene frames; checked by rule_agreement
MIN_DISTANCE = 0.5               # metres on the floor plane between the standing point and the other two
MIN_MARGIN = 5.0                 # degrees to the nearest border between two answers
KEY = "e1"


def roles_of(row: dict):
    """Slot indices of (stand, face, target), read from the answer sentence; None if it cannot be read."""
    index = {slot["name"]: i for i, slot in enumerate(row.get("slots") or [])}
    for pattern in ROLES:
        match = pattern.match(row["target"])
        if match and all(name in index for name in match.groups()) and len(set(match.groups())) == 3:
            return [index[name] for name in match.groups()]
    return None


def sentences_of(row: dict):
    """{option: its answer sentence in the row's wording}; None unless every option has one."""
    options, short = list(dict.fromkeys(row.get("options") or [])), row.get("target_short")
    wrong = [o for o in options if o != short]
    if short not in options or len(wrong) != len(row.get("negatives") or []) or len(options) not in (2, 3, 4):
        return None
    return {short: row["target"], **dict(zip(wrong, row["negatives"]))}


def flat_distance(a, b) -> float:
    return float(np.linalg.norm((a - b)[:2]))


def synth(row, roles, sentences, table, rng):
    """The row with its three names on three random objects of the scene, or None."""
    for _ in range(30):
        objects = rng.sample(range(len(table)), 3)
        stand, face, target = table[objects]
        if min(flat_distance(stand, face), flat_distance(stand, target)) < MIN_DISTANCE:
            continue
        angle = azimuth(stand, face, target, UP)
        if boundary_distance(len(sentences), angle) < MIN_MARGIN:
            continue
        answer = direction_answer(len(sentences), angle)
        if answer not in sentences:
            return None
        return {**row, "source": "synth", "target": sentences[answer], "target_short": answer,
                "negatives": [s for o, s in sentences.items() if o != answer], "answer_key": KEY,
                "answer_raw": None, "angle": round(angle, 1),
                "ground": [{"slot": slot, "object": obj, "how": "synth"} for slot, obj in zip(roles, objects)]}
    return None


def convert(source: Path, out: Path, tables: dict, variants: int, stats: Counter) -> None:
    with open(source) as f, open(out, "w") as g:
        for line_number, line in enumerate(f):
            row = json.loads(line)
            if row.get("question_type") != "relative_direction_object":
                continue
            stats["direction_rows"] += 1
            roles, sentences = roles_of(row), sentences_of(row)
            table = tables.get(f"{row.get('subset')}/{row.get('scene')}")
            if roles is None or sentences is None or table is None or len(table) < 3:
                stats["skipped"] += 1
                continue
            labelled = {e["slot"]: e["object"] for e in row.get("ground") or []}
            if all(slot in labelled for slot in roles):
                stand, face, target = table[[labelled[slot] for slot in roles]]
                rule = direction_answer(len(sentences), azimuth(stand, face, target, UP))
                stats["real"] += 1
                stats["real_rule_agrees"] += rule == row["target_short"]
                stats[f"real_options_{len(sentences)}"] += 1
                stats[f"real_rule_agrees_options_{len(sentences)}"] += rule == row["target_short"]
                g.write(json.dumps({**row, "answer_key": KEY,
                                    "ground": [{"slot": slot, "object": labelled[slot], "how": "name"}
                                               for slot in roles]}) + "\n")
            rng = random.Random(zlib.crc32(f"{source.name}:{line_number}".encode()))
            for _ in range(variants):
                new = synth(row, roles, sentences, table, rng)
                if new is None:
                    stats["synth_failed"] += 1
                    continue
                stats["synth"] += 1
                stats[f"synth_answer_{len(sentences)}_{new['target_short']}"] += 1
                g.write(json.dumps(new) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="ground_v1's data folder (read only)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-variants", type=int, default=3)
    parser.add_argument("--holdout-variants", type=int, default=5)
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(data / "scene_objects.json") as f:
        tables = {key: np.array(scene["pos"], dtype=np.float64).reshape(-1, 3) for key, scene in json.load(f).items()}
    summary = {}
    for split, variants in (("train", args.train_variants), ("holdout", args.holdout_variants)):
        stats = Counter()
        convert(data / f"{split}_f8.jsonl", out / f"{split}_f8.jsonl", tables, variants, stats)
        stats = dict(sorted(stats.items()))
        stats["rule_agreement"] = stats.get("real_rule_agrees", 0) / max(stats.get("real", 0), 1)
        summary[split] = stats
        print(split, json.dumps(stats, indent=1), flush=True)
    (out / "manifest.stats.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
