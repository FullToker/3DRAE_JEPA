"""Inference on locked native rows, using the pinned fusion_gv candidate scorer."""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def sample(row, by_id, mode):
    source = by_id[row["donor"]] if mode == "donor" else row
    if mode == "donor" and (not row["room"] or not source["room"] or row["room"] == source["room"]
                            or row["stratum"] != source["stratum"] or row["block"] != source["block"]
                            or not set(row["image_sha256"]).isdisjoint(source["image_sha256"])):
        raise ValueError(f"invalid different-room donor for {row['id']}")
    bundle = {} if mode == "question" else source["bundle"]
    return {"image_paths": bundle.get("images", []), "intrinsics": bundle.get("intrinsics", []),
            "c2w": bundle.get("c2w", []), "query": row["query"], "target": row["options"][row["gold"]],
            "group_id": None, "negatives": [], "row": {"id": row["id"]}}


def record_prediction(row, entry, mode):
    if entry is None:
        raise ValueError(f"scorer skipped {row['id']}")
    winner, ranked = entry
    ranked = sorted(ranked, key=lambda obj: obj["index"])
    options = row["options"]
    if ([s["index"] for s in ranked] != list(range(len(options))) or
            [s["candidate"] for s in ranked] != options):
        raise ValueError(f"scorer changed candidates on {row['id']}")
    scores = [float(s["score"]) for s in ranked]
    if not all(math.isfinite(s) for s in scores) or not 0 <= winner < len(scores) or scores[winner] != max(scores):
        raise ValueError(f"invalid cosine scores on {row['id']}")
    gold = row["gold"]
    return {"gold": gold, "scores": scores, "winner": winner, "answer": options[winner], "correct": winner == gold,
            "margin": scores[gold] - max(s for i, s in enumerate(scores) if i != gold),
            "visual_id": row["donor"] if mode == "donor" else row["id"] if mode == "real" else None}


@torch.inference_mode()
def score_condition(model, rows, mode, device, batch_size, workers, image_size, scorer, collate, calibration=False):
    """Score real and question-only on every eligible row, donor on every paired row."""
    by_id = {r["id"]: r for r in rows}
    selected = [r for r in rows if mode != "donor" or r["donor"]]
    collate = partial(collate, image_size=image_size, question_only=mode == "question")
    samples = [sample(r, by_id, mode) for r in selected]
    loader = DataLoader(samples, batch_size=batch_size, num_workers=workers, collate_fn=collate)
    records, checks = {}, {}

    def predict(batch):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model.predict_embedding(batch["images_vggt"].to(device), batch["images_jepa"].to(device),
                                           batch["query"])
        if pred.shape != (len(batch["query"]), 1536) or not pred.isfinite().all():
            raise ValueError("bad predictor embedding shape or nonfinite values")
        return pred

    for batch in loader:
        ids = [r["id"] for r in batch["rows"]]
        pred = predict(batch)
        options = [by_id[qid]["options"] for qid in ids]
        ranked = scorer(model, pred, options, device, include_scores=True)
        if len(ranked) != len(ids):
            raise ValueError("candidate scorer dropped rows")
        for qid, entry in zip(ids, ranked):
            records[qid] = record_prediction(by_id[qid], entry, mode)
    if calibration and selected:
        # Keep the same small panel independent of the inference batch size.
        # Even a batch-size-one run tests the rows with other companions here.
        indices = []
        seen_rooms = set()
        for i, row in enumerate(selected):
            if row["room"] not in seen_rooms:
                indices.append(i)
                seen_rooms.add(row["room"])
                if len(indices) == 3:
                    break
        mixed = predict(collate([samples[i] for i in indices]))
        probes = [selected[i] for i in indices]
        companions = scorer(model, mixed, [r["options"] for r in probes], device, include_scores=True)
        observations = []
        for i, row, entry in zip(indices, probes, companions):
            qid, opts = row["id"], row["options"]
            singleton = predict(collate([samples[i]]))
            cold = model.encode_target(opts, device).float()
            if cold.shape != (len(opts), 1536) or not cold.isfinite().all() or (cold.norm(dim=1) == 0).any():
                raise ValueError("bad candidate embedding shape, norm or finite values")
            cached = (F.normalize(singleton.float(), dim=-1) @ F.normalize(cold, dim=-1).T)[0]
            solo = scorer(model, singleton, [opts], device, include_scores=True)[0]
            perm = scorer(model, singleton, [opts[::-1]], device, include_scores=True)[0]
            solo_scores = record_prediction(by_id[qid], solo, mode)["scores"]
            tie = max(solo_scores) - sorted(solo_scores)[-2] <= 1e-6
            batch_scores = record_prediction(row, entry, mode)["scores"]
            observation = {"id": qid, "solo_winner": solo[0], "batch_winner": entry[0],
                           "production_winner": records[qid]["winner"], "permuted_winner": opts[::-1][perm[0]],
                           "near_tie": tie, "max_batch_score_change": max(abs(a - b) for a, b in zip(solo_scores, batch_scores)),
                           "cached_score_error": (cached - cached.new_tensor(solo_scores)).abs().max().item()}
            if (observation["cached_score_error"] > 1e-5 or (not tie and
                    (observation["permuted_winner"] != opts[solo[0]] or int(cached.argmax()) != solo[0]))):
                raise ValueError(f"native scorer compatibility check failed: {observation}")
            observations.append(observation)
        checks = {"samples": observations, "cached_score_error": max(r["cached_score_error"] for r in observations)}
    return records, checks


@torch.inference_mode()
def check_prefix(model, rows, question_only, collate, device, image_size):
    by_id = {r["id"]: r for r in rows}
    row = next((r for r in rows if r["donor"]), None)
    if row is None:
        raise ValueError("no verified different-room pair for prefix validation")
    prefixes = []
    for mode in ("real", "donor"):
        batch = collate([sample(row, by_id, mode)], image_size=image_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens, _ = model._pool_visual(batch["images_vggt"].to(device), batch["images_jepa"].to(device), 1)
        if tokens.shape != (1, 1032, 768) or not tokens.isfinite().all():
            raise ValueError("unexpected visual prefix")
        prefixes.append(tokens)
    identical = torch.equal(*prefixes)
    if identical != question_only or (question_only and torch.count_nonzero(prefixes[0]).item()):
        raise ValueError("prefix intervention failed")
    return {"id": row["id"], "donor": row["donor"], "identical": identical,
            "max_abs_change": (prefixes[0] - prefixes[1]).abs().max().item()}
