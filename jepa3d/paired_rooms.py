"""Lock the hardneg_v1 holdout, native candidates and different-room donors before inference."""

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from jepa3d.hard_negatives import candidates_of


RELATIONS = (
    "spar/spatial_imagination_oc_video_hard", "spar/spatial_imagination_oo_video_hard",
    "vsi590k/relative_direction_object", "vsi590k/relative_distance_object",
)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def publish(path, content):
    """Publish a complete artifact once, including when two jobs target the same path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as out:
        try:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
            os.link(out.name, path)
        finally:
            os.unlink(out.name)


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def write_json(path, value):
    publish(path, json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def native_id(row):
    source = row["source"]
    if source == "vsi590k" and type(row.get("src_line")) is int and row["src_line"] >= 0:
        return f"vsi590k:{row['src_line']}"
    if source == "spar" and isinstance(row.get("src"), str) and re.fullmatch(r".+:[0-9]+", row["src"]):
        return "spar:" + row["src"]
    raise ValueError("missing or invalid native source ID")


def physical_room(subset, scene, aliases):
    # Alias values are global physical-room IDs. Distinct ScanNet++ scan
    # names alone do not certify distinct rooms.
    if subset == "scannet" and re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", scene):
        key = "scannet/" + scene.rsplit("_", 1)[0]
        return aliases.get(key, key)
    return aliases.get(f"{subset}/{scene}")


def _matrix(value, n):
    return (isinstance(value, list) and len(value) == n and
            all(isinstance(line, list) and len(line) == n and
                all(type(v) in (int, float) and math.isfinite(v) for v in line) for line in value))


def lock_row(row, aliases, hashes):
    qid = native_id(row)
    if not all(isinstance(row.get(k), str) and row[k].strip()
               for k in ("query", "scene", "subset", "question_type", "target", "answer_key")):
        raise ValueError("missing question, scene, type, target or answer key")
    if not isinstance(row.get("negatives"), list):
        raise ValueError("missing native negatives")
    options, gold = candidates_of(row)["long"]
    if (len(options) < 2 or not all(isinstance(x, str) and x.strip() for x in options)
            or len(set(options)) != len(options)):
        raise ValueError("invalid or duplicate long-form candidates")
    views = row["images"]
    if (not isinstance(views, list) or len(views) != 8 or len(row["intrinsics"]) != 8
            or len(row["c2w"]) != 8 or not all(_matrix(m, 3) for m in row["intrinsics"])
            or not all(_matrix(m, 4) for m in row["c2w"])):
        raise ValueError("expected eight aligned RGB frames and finite camera matrices")
    if "frame_ids" in row and len(row["frame_ids"]) != 8:
        raise ValueError("unaligned frame IDs")
    for path in views:
        if path not in hashes:
            hashes[path] = file_hash(path)
    bundle = {k: row[k] for k in ("images", "intrinsics", "c2w", "frame_ids") if k in row}
    image_hashes = [hashes[p] for p in views]
    labels = {}
    if row["source"] == "vsi590k" and row.get("target_short") in (row.get("options") or []):
        # vsi_negatives renders in native option order. A failed rendering can
        # shorten that list, in which case only the known gold is identifiable.
        short = row["target_short"]
        labels[gold] = short
        wrong = [s for s in dict.fromkeys(row["options"]) if s != short]
        if len(wrong) == len(row["negatives"]):
            labels.update(zip(row["negatives"], wrong))
    elif row["question_type"].startswith("spatial_imagination"):
        for option in options:
            terms = re.findall(r"\b(left|right|above|below|front|behind|closer|farther)\b", option.lower())
            if terms:
                labels[option] = "|".join(terms)
    return {"id": qid, "query": row["query"], "options": options, "gold": options.index(gold),
            "labels": [labels.get(x) for x in options],
            "family": f"{row['source']}/{row['question_type']}", "subset": row["subset"], "scene": row["scene"],
            "room": physical_room(row["subset"], row["scene"], aliases), "bundle": bundle,
            "bundle_sha256": digest([image_hashes, bundle]), "image_sha256": image_hashes,
            "stratum": [row["source"], row["subset"], len(views), row.get("pose_source", row.get("cameras", "known")),
                        row.get("selection_protocol", row.get("frame_selection", "hardneg_f8_depth_pose_fps"))]}


def assign_donors(rows, seed):
    """Each physical room belongs to one joint bootstrap block across all sources."""
    banks = defaultdict(lambda: defaultdict(list))
    for row in rows:
        row.update(donor=None, block=None, unpaired="unresolved room" if not row["room"] else "no compatible room")
        if row["room"]:
            banks[row["room"]][tuple(row["stratum"])].append(row)
    rng = random.Random(seed)
    order = sorted(banks)
    rng.shuffle(order)
    signatures = {room: set(banks[room]) for room in order}
    while len(order) >= 2:
        room = order.pop()
        # Prefer matching all observation protocols, then the largest overlap.
        partner = max(order, key=lambda other: (signatures[room] == signatures[other],
                                                len(signatures[room] & signatures[other])))
        if not signatures[room] & signatures[partner]:
            continue
        order.remove(partner)
        group = [room, partner]
        # Absorb an odd room if it would otherwise lack any compatible partner.
        for other in order:
            compatible = signatures[other] & (signatures[room] | signatures[partner])
            if compatible and not any(signatures[other] & signatures[r] for r in order if r != other):
                group.append(other)
                order.remove(other)
                break
        block = digest(sorted(group))
        for recipient in group:
            for stratum in sorted(banks[recipient]):
                recipients = banks[recipient][stratum]
                donors = sorted((r for peer in group if peer != recipient for r in banks[peer].get(stratum, ())),
                                key=lambda r: r["id"])
                rng.shuffle(donors)
                for i, row in enumerate(sorted(recipients, key=lambda r: r["id"])):
                    for offset in range(len(donors)):
                        donor = donors[(i + offset) % len(donors)]
                        if set(row["image_sha256"]).isdisjoint(donor["image_sha256"]):
                            row.update(donor=donor["id"], block=block, unpaired=None)
                            break
                    if donors and row["donor"] is None:
                        row["unpaired"] = "shared frame content"


def make_plan(train, holdout, aliases, seed=12610):
    if not isinstance(aliases, dict) or not all(isinstance(v, str) and v.strip() for v in aliases.values()):
        raise ValueError("room alias table must map scan keys to nonempty physical-room IDs")
    canonical = {}
    for key, value in aliases.items():
        subset, scene = key.split("/", 1)
        key = physical_room(subset, scene, {}) or key
        if canonical.setdefault(key, value) != value:
            raise ValueError(f"conflicting room aliases for {key}")
    aliases = canonical
    rows, rejected, hashes, ids = [], [], {}, Counter()
    hold_digest = hashlib.sha256()
    with open(holdout, "rb") as stream:
        for line, raw in enumerate(stream, 1):
            hold_digest.update(raw)
            if not raw.strip():
                continue
            qid = None
            try:
                row = json.loads(raw)
                qid = native_id(row)
                ids[qid] += 1
                rows.append({**lock_row(row, aliases, hashes), "line": line})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                rejected.append({"line": line, "id": qid, "reason": str(exc)})
    unique = []
    for row in rows:
        if ids[row["id"]] == 1:
            unique.append(row)
        else:
            rejected.append({"line": row["line"], "id": row["id"], "reason": "duplicate native ID"})
    train_rooms, unresolved, train_overlap = set(), set(), set()
    train_digest, count = hashlib.sha256(), 0
    with open(train, "rb") as stream:
        for raw in stream:
            train_digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            count += 1
            room = physical_room(row["subset"], row["scene"], aliases)
            if room:
                train_rooms.add(room)
            else:
                unresolved.add(row["subset"])
            qid = native_id(row)
            if qid in ids:
                train_overlap.add(qid)
    if not count or not unique:
        raise ValueError("empty training manifest or no eligible holdout rows")
    for row in unique:
        if row["id"] in train_overlap or row["room"] in train_rooms:
            row["split"] = "overlap"
        elif not row["room"] or row["subset"] in unresolved:
            row["split"] = "unresolved"
        else:
            row["split"] = "unseen"
    assign_donors(unique, seed)
    return {"version": 1, "seed": seed, "train": str(Path(train).resolve()), "holdout": str(Path(holdout).resolve()),
            "train_sha256": train_digest.hexdigest(), "holdout_sha256": hold_digest.hexdigest(),
            "aliases_sha256": digest(aliases), "train_rows": count, "attempted": len(unique) + len(rejected),
            "unresolved_train_subsets": sorted(unresolved), "rejected": rejected, "rows": unique}
