"""Delexicalized questions: object names replaced by "object A", "object B", ...

    "Standing by the sofa and facing the tv, is the trash bin to my left or right?"
 -> "Standing by the object C and facing the object A, is the object B to my left or right?"

Without the name the question cannot be answered from what such rooms usually
look like ("the tv faces the sofa", "the nightstand is nearest the bed"): which
object is meant is only told by the name's object token (jepa3d.slots.ObjectTokens,
the patches dino.txt chose for the name), so this is for models with object
tokens. The same letters replace the names in the answer, the wrong answers
and the options. Letters are assigned at random per row, so they carry nothing.

Names and their places in the question come from the row's "slots"
(jepa3d/tools/precompute_slots.py); a row without slots is returned unchanged.
"""

import random
import re
import string

TEXT_FIELDS = ("target", "target_short")
LIST_FIELDS = ("negatives", "candidates", "options")


def delexicalize(row: dict, rng: random.Random) -> dict:
    """A copy of the row with its object names replaced in "query", the answer fields and the slots' spans."""
    slots = row.get("slots") or []
    if not slots or len(slots) > len(string.ascii_uppercase):
        return row
    letters = rng.sample(string.ascii_uppercase[:max(len(slots), 6)], len(slots))
    label = {slot["name"]: f"object {letter}" for slot, letter in zip(slots, letters)}

    # the question: replace at the stored spans (first line), keeping track of where the labels land
    query = row["query"]
    places = sorted((start, end, slot["name"]) for slot in slots for start, end in slot["spans"])
    pieces, cursor, new_spans = [], 0, {name: [] for name in label}
    length = 0
    for start, end, name in places:
        if start < cursor:   # overlapping spans: keep the first
            continue
        pieces.append(query[cursor:start])
        length += start - cursor
        pieces.append(label[name])
        new_spans[name].append([length, length + len(label[name])])
        length += len(label[name])
        cursor = end
    pieces.append(query[cursor:])
    head = "".join(pieces)

    # every other text: replace the names as words (plural endings stay), longest name first
    names = sorted(label, key=lambda n: (-len(n), n))
    pattern = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?=(?:s|es)?(?![A-Za-z0-9]))",
                         re.IGNORECASE)
    replace = lambda text: pattern.sub(lambda m: label[m.group(1).lower()], text) if isinstance(text, str) else text
    first, _, rest = head.partition("\n")
    out = {**row, "query": first + ("\n" + replace(rest) if rest else "")}
    for field in TEXT_FIELDS:
        if row.get(field) is not None:
            out[field] = replace(row[field])
    for field in LIST_FIELDS:
        if row.get(field):
            out[field] = [replace(x) for x in row[field]]
    # "lookup": the real name, only for finding the object's patches (jepa3d/pointer.py)
    out["slots"] = [{**slot, "name": label[slot["name"]], "lookup": slot["name"], "spans": new_spans[slot["name"]]}
                    for slot in slots]
    out["delexicalized"] = True
    return out
