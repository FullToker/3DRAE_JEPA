"""Paired answer changes and room-block uncertainty for the locked hardneg_v1 panel."""

from collections import Counter, defaultdict

import numpy as np

from jepa3d.paired_rooms import RELATIONS


CONDITIONS = ("real", "donor", "question")


def validate_predictions(rows, records, condition):
    eligible = {r["id"]: r for r in rows if condition != "donor" or r["donor"] is not None}
    if set(records) != set(eligible):
        raise ValueError(f"{condition}: missing or unexpected prediction IDs")
    for qid, rec in records.items():
        row = eligible[qid]
        scores = np.asarray(rec["scores"], dtype=float)
        winner = rec["winner"]
        if (scores.ndim != 1 or len(row["options"]) != len(scores) or not np.isfinite(scores).all()
                or type(winner) is not int or not 0 <= winner < len(scores) or scores[winner] != max(scores)
                or rec["answer"] != row["options"][winner] or rec["gold"] != row["gold"]
                or rec["correct"] != (winner == row["gold"])):
            raise ValueError(f"{condition}: invalid scores, answer or gold for {qid}")
        margin = scores[row["gold"]] - np.delete(scores, row["gold"]).max()
        if not np.isclose(rec["margin"], margin, rtol=0, atol=1e-10):
            raise ValueError(f"{condition}: invalid gold margin for {qid}")
        source = row["donor"] if condition == "donor" else qid if condition == "real" else None
        if rec["visual_id"] != source:
            raise ValueError(f"{condition}: unexpected visual donor for {qid}")


def compare_reference(rows, records, reference):
    expected = {k: v for k, v in reference["real"]["holdout"].items() if k.startswith("long/")}
    observed = defaultdict(list)
    for row in rows:
        for key in ("long/all", "long/" + row["family"]):
            observed[key].append(records[row["id"]]["correct"])
    mismatches = {}
    for key in sorted(set(expected) | set(observed)):
        ref = expected.get(key, {})
        n, actual = len(observed[key]), sum(observed[key])
        # eval_final.json stores unrounded fractions. Counts identify the run.
        if n != ref.get("rows") or n == 0 or abs(actual - ref["acc"] * n) >= 0.5:
            mismatches[key] = {"rows": n, "correct": actual, "reference": ref}
    return mismatches


def panel_metrics(rows, predictions):
    groups = defaultdict(list)
    for r in rows:
        groups[r["family"]].append(r)
    result = {}
    for name, group in [("all", rows), *sorted(groups.items())]:
        if not group:
            continue
        ids = [r["id"] for r in group]
        record = {"n": len(group), "rooms": len({r["room"] for r in group}),
                  "chance": float(np.mean([1 / len(r["options"]) for r in group]))}
        for mode in CONDITIONS:
            values = [predictions[mode][qid] for qid in ids]
            hits = sum(p["correct"] for p in values)
            summary = {"correct": hits, "accuracy": hits / len(group)}
            if name != "all":
                confusion = defaultdict(Counter)
                for row, p in zip(group, values):
                    gold = row["labels"][row["gold"]]
                    if gold is not None:
                        confusion[gold][row["labels"][p["winner"]] or "<unmapped>"] += 1
                summary["confusion"] = dict(confusion)
                summary["recall"] = {k: counts[k] / sum(counts.values()) for k, counts in confusion.items()}
            record[mode] = summary
            if mode == "real":
                continue
            real = [predictions["real"][qid] for qid in ids]
            record[mode]["paired"] = {
                "delta_pp": 100 * (record["real"]["accuracy"] - summary["accuracy"]),
                "flip_rate": sum(x["winner"] != y["winner"] for x, y in zip(real, values)) / len(group),
                "correct_to_wrong": sum(x["correct"] and not y["correct"] for x, y in zip(real, values)),
                "wrong_to_correct": sum(not x["correct"] and y["correct"] for x, y in zip(real, values)),
                "mean_margin_change": float(np.mean([x["margin"] - y["margin"] for x, y in zip(real, values)]))}
        result[name] = record
    return result


def room_intervals(rows, predictions, draws=10000, seed=12610):
    blocks = sorted({r["block"] for r in rows})
    if len(blocks) < 2 or {r["family"] for r in rows} != set(RELATIONS):
        return {"decision": "inconclusive", "reason": "missing family or insufficient donor blocks"}
    block_index = {key: i for i, key in enumerate(blocks)}
    counts = np.zeros((len(blocks), 4), dtype=int)
    differences = np.zeros((len(blocks), 4, 2), dtype=int)
    for r in rows:
        b, f, qid = block_index[r["block"]], RELATIONS.index(r["family"]), r["id"]
        counts[b, f] += 1
        for m, mode in enumerate(("donor", "question")):
            differences[b, f, m] += int(predictions["real"][qid]["correct"]) - int(predictions[mode][qid]["correct"])
    by_family = 100 * differences.sum(0) / counts.sum(0)[:, None]
    effect = by_family.mean(0)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=draws)
    denominators = weights @ counts
    complete = (denominators > 0).all(axis=1)
    totals = np.einsum("db,bfm->dfm", weights[complete], differences)
    samples = (100 * totals / denominators[complete, :, None]).mean(1)
    # Missing families in bootstrap draws leave the contrast unidentified.
    # Bound such draws conservatively instead of silently redrawing them.
    missing = draws - len(samples)
    lower = np.quantile(np.vstack((samples, np.full((missing, 2), -100))), .0125, axis=0)
    upper = np.quantile(np.vstack((samples, np.full((missing, 2), 100))), .9875, axis=0)
    decision = "inconclusive"
    if ((counts > 0).sum(0) >= 2).all():
        if (effect >= 5 - 1e-12).all() and (lower > 0).all() and (by_family[:, 0] > 0).sum() >= 3:
            decision = "useful scene dependence"
        elif (upper < 5 - 1e-12).any():
            decision = "excludes stipulated advantage"
    return {"decision": decision, "blocks": len(blocks), "draws": draws, "incomplete_draws": missing, "seed": seed,
            "family_order": RELATIONS, "blocks_per_family": (counts > 0).sum(0).tolist(),
            "contrasts": {m: {"macro_delta_pp": effect[i], "ci_97_5_pp": [lower[i], upper[i]],
                              "by_family_pp": by_family[:, i].tolist()} for i, m in enumerate(("donor", "question"))}}


def summarize(plan, predictions):
    rows = plan["rows"]
    for mode in CONDITIONS:
        validate_predictions(rows, predictions[mode], mode)
    common = [r for r in rows if r["donor"] is not None]
    relations = [r for r in common if r["family"] in RELATIONS]
    primary = [r for r in relations if r["split"] == "unseen"]
    primary_metrics = panel_metrics(primary, predictions)
    macro = {m: float(np.mean([primary_metrics[f][m]["accuracy"] for f in RELATIONS])) for m in CONDITIONS
             if all(f in primary_metrics for f in RELATIONS)}
    by_id = {r["id"]: r for r in rows}
    return {"coverage": {"attempted": plan["attempted"], "eligible": len(rows),
                         "scored": {k: len(v) for k, v in predictions.items()}, "common": len(common),
                         "relation_attempted": sum(r["family"] in RELATIONS for r in rows),
                         "primary": len(primary), "splits": Counter(r["split"] for r in rows),
                         "primary_donor_split": Counter(by_id[r["donor"]]["split"] for r in primary),
                         "unpaired": Counter(r["unpaired"] for r in rows if r["unpaired"])},
            "accuracy_on_scored": {m: sum(r["correct"] for r in scores.values()) / len(scores) if scores else None
                                   for m, scores in predictions.items()},
            "all_common": panel_metrics(common, predictions), "relations_common": panel_metrics(relations, predictions),
            "primary": primary_metrics, "primary_macro_accuracy": macro, "inference": room_intervals(primary, predictions)}
