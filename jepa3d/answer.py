"""Short answers for the answer token (config "answer_token: {enabled: true}", jepa3d/model.py).

The model's usual prediction is the mean over the question tokens and is compared with the answer
written as a sentence. The few tokens that tell a right sentence from a wrong one are a small part
of that sentence's mean (experiments/flip_similarity_check). The answer token is one more token
after the question; its output alone is compared with the answer in its short form:

    VSI-590K                   "target_short" ("front-left", "radiator", "2.01 meters"); wrong
                               answers: the row's other options
    SPAR appearance_order      the order itself ("picture, backpack, desk, toilet"); wrong answers:
                               the orders of the row's wrong sentences
    SPAR spatial_imagination   the relation words of "answer_key" ("right, front"); wrong answers:
                               left/right, above/below, closer/farther swapped (as in
                               jepa3d/hard_negatives.py)
    SPAR obj_count, room_size  the number ("3", "9 square meters"); no wrong answers
"""

import re

from jepa3d.hard_negatives import spatial_negatives

ORDER = re.compile(r"^The order of first appearance is: (.*)\.$")


def short_answer(row: dict) -> tuple[str, list] | None:
    """-> (short answer, wrong short answers to the same question), or None for a row without one."""
    short = row.get("target_short")
    if short is not None:
        options = list(dict.fromkeys(row.get("options") or []))
        return str(short), [o for o in options if o != short] if short in options else []
    key, kind = row.get("answer_key") or "", row.get("question_type") or ""
    if kind == "appearance_order":
        match = ORDER.match(row["target"])   # from the sentence: delexicalized rows keep their letters
        if not match:
            return None
        wrong = [m.group(1) for m in map(ORDER.match, row.get("negatives") or []) if m]
        return match.group(1), wrong
    if kind.startswith("spatial_imagination") and ":" in key:
        short = ", ".join(key.rsplit(":", 1)[1].split("|"))
        return short, spatial_negatives(short)
    if key.startswith("count:"):
        return key.split(":")[1], []
    if key.startswith("area:"):
        return f"{key.split(':')[1]} square meters", []
    return None
