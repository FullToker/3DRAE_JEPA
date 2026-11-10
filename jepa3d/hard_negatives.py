"""Same-question wrong answers ("hard negatives") for mix manifest rows.

In-batch InfoNCE negatives are answers to other questions; they name other
objects, so the query text alone tells them apart. A hard negative keeps the
row's question and wording and changes only the answer, so nothing but the
scene can separate it from the target.

    VSI-590K multiple choice   every wrong option, rendered with the row's
                               own sentence wording (jepa3d/vsi_text.py)
    SPAR spatial_imagination   left/right, above/below, closer/farther swapped
                               (every non-empty combination of the pairs the
                               answer uses)
    SPAR appearance_order      other orders of the same objects

Numeric answers (counts, sizes, distances, angles) get none here.
"front" / "behind" are left alone: SPAR words them differently ("to the
front", "in front of", "behind"), so a word swap would leave a grammar cue.
"""

import itertools
import random
import re
import zlib

SWAP_PAIRS = (("left", "right"), ("above", "below"), ("closer", "farther"))
MAX_NEGATIVES = 7


def vsi_negatives(row: dict, renderer) -> tuple[list, str]:
    """-> (negatives, status). renderer: jepa3d.vsi_text.Renderer."""
    options, short = row.get("options") or [], row.get("target_short")
    wrong = [o for o in dict.fromkeys(options) if o != short]
    if not wrong or short not in options:
        return [], "no_options"
    if row.get("target_style") != "sentence":
        return wrong, "ok"
    qtype, query, seed = row["question_type"], row["query"], f"vsi590k:{row['src_line']}"
    if renderer.render(qtype, query, short, seed) != row["target"]:
        return [], "render_mismatch"
    negatives = [renderer.render(qtype, query, o, seed) for o in wrong]
    negatives = [n for n in negatives if n and n != row["target"]]
    return negatives, "ok" if negatives else "unrendered"


def spatial_negatives(target: str) -> list:
    present = [pair for pair in SWAP_PAIRS if re.search(rf"\b({pair[0]}|{pair[1]})\b", target)]
    out = []
    for r in range(1, len(present) + 1):
        for pairs in itertools.combinations(present, r):
            swap = {a: b for pair in pairs for a, b in (pair, pair[::-1])}
            out.append(re.sub(r"\b(" + "|".join(swap) + r")\b", lambda m: swap[m.group(1)], target))
    return out


def order_negatives(row: dict) -> list:
    items = row["answer_key"].removeprefix("order:").split("|")
    wording = "The order of first appearance is: {}."
    if wording.format(", ".join(items)) != row["target"]:
        return []
    others = [p for p in itertools.permutations(items) if list(p) != items]
    rng = random.Random(zlib.crc32(f"{row['scene']}:{row['query']}".encode()))
    return [wording.format(", ".join(p)) for p in rng.sample(others, min(3, len(others)))]


def negatives_for(row: dict, renderer) -> tuple[list, str]:
    """-> (wrong answers in the wording of row["target"], status)."""
    qtype = row["question_type"]
    if row["source"] == "vsi590k":
        negatives, status = vsi_negatives(row, renderer)
    elif qtype.startswith("spatial_imagination"):
        negatives = spatial_negatives(row["target"])
        status = "ok" if negatives else "no_swappable_term"
    elif qtype == "appearance_order":
        negatives = order_negatives(row)
        status = "ok" if negatives else "unparsed_order"
    else:
        return [], "numeric"
    return negatives[:MAX_NEGATIVES], status
