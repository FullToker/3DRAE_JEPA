"""Answer sentences for VSI-590K rows.

VSI-590K answers are bare option texts or numbers ("right", "2.7"). From the
question template we pull the objects and their roles, then state the answer
as a sentence:

    "Standing at the laptop and facing the bed, the cushion is to my right."

Rows whose slots cannot be read keep the short answer (render() returns
None); nothing is dropped. Object names come from a vocabulary built from
the option lists and count questions of the same file (build_vocab), so a
name is only found when it appears verbatim as "the <name>".
"""

import re
import zlib

NON_OBJECTS = {"left", "right", "back", "front", "behind", "more", "fewer", "equal number", "room", "video"}
FACING_BEFORE = re.compile(r"(facing|toward|towards|viewing|(looking|gazing|staring)( straight| directly)? at"
                           r"|fixed on|face|oriented to)$")
POSITION_BEFORE = re.compile(r"\b(at|by|near|beside|alongside|next to|close to|adjacent to|from|of|with)$")


def build_vocab(rows) -> list:
    names = set()
    for r in rows:
        if r["question_type"] in ("relative_distance_object", "relative_size_object", "appearance_order"):
            for opt in r.get("options") or []:
                names.update(x.strip().lower() for x in opt.split(","))
        names.update(count_categories(r["query"]))
    return sorted((n for n in names if n and n not in NON_OBJECTS and not re.search(r"\d", n)),
                  key=len, reverse=True)


COUNT_STOP = {"of", "many", "to", "as", "than", "with", "are", "do", "more", "fewer", "numberof", "the", "and",
              "or", "compared", "count", "number", "quantity", "there", "outnumber"}


def count_categories(query: str) -> list:
    """Categories written as "<name>(s)", in question order: the words before
    each "(s)" back to the first function word ("how many trash bin(s)")."""
    out = []
    for m in re.finditer(r"\(s\)", query.lower()):
        words = re.findall(r"[a-z'-]+|[^a-z'\s-]", query.lower()[:m.start()])
        name = []
        while words and re.fullmatch(r"[a-z'-]+", words[-1]) and words[-1] not in COUNT_STOP and len(name) < 4:
            name.insert(0, words.pop())
        if name:
            out.append(" ".join(name))
    return out


def plural(name: str, n: float = 2) -> str:
    if n == 1 or re.search(r"[^s]s$", name):   # "slippers", "headphones" are already plural
        return name
    if name.endswith("shelf"):
        return name[:-1] + "ves"
    if re.search(r"(s|x|z|ch|sh)$", name):
        return name + "es"
    if re.search(r"[^aeiou]y$", name):
        return name[:-1] + "ies"
    return name + "s"


class Renderer:
    def __init__(self, vocab):
        self.mention = re.compile(r"\bthe (" + "|".join(map(re.escape, vocab)) + r")(?![a-z])")

    def mentions(self, text: str) -> list:
        """[(start, name)] of every "the <name>"."""
        return [(m.start(), m.group(1)) for m in self.mention.finditer(text)]

    def render(self, qtype: str, query: str, short: str, seed: str | None):
        """Answer sentence, or None if the slots cannot be read. seed=None gives
        the first wording (canonical form, used for answer_key)."""
        q = query.lower()
        pick = lambda options: options[0 if seed is None else zlib.crc32(seed.encode()) % len(options)]
        fn = getattr(self, "_" + qtype, None)
        try:
            return fn(q, short, pick) if fn else None
        except (ValueError, IndexError):
            return None

    # --- per type -------------------------------------------------------------

    def _roles(self, q):
        """(position, facing, target) objects of a direction question."""
        found = self.mentions(q)
        if len(found) < 3:
            return None
        pos = face = None
        rest = []
        for start, name in found:
            before = q[max(0, start - 24):start].strip()
            ahead = q.startswith(f"the {name} ahead", start)
            if face is None and (FACING_BEFORE.search(before) or ahead):
                face = name
            elif pos is None and POSITION_BEFORE.search(before):
                pos = name
            else:
                rest.append(name)
        rest = [n for n in rest if n not in (pos, face)]   # "which side of the <face> is the <target>"
        if not (pos and face and rest) or pos == face:
            return None
        return pos, face, rest[0]

    def _relative_direction_object(self, q, short, pick):
        roles = self._roles(q)
        if roles is None or short not in ("left", "right", "back", "front-left", "front-right", "back-left",
                                          "back-right"):
            return None
        pos, face, target = roles
        where = {"left": "to my left", "right": "to my right", "back": "behind me"}.get(short, f"at my {short}")
        return pick([
            f"Standing at the {pos} and facing the {face}, the {target} is {where}.",
            f"From the {pos}, looking toward the {face}, the {target} is {where}.",
            f"If I stand at the {pos} facing the {face}, the {target} is {where}.",
        ])

    def _absolute_direction_object(self, q, short, pick):
        found = [n for _, n in self.mentions(q)]
        if len(found) != 3:
            return None
        pos, face, target = found
        turn = "counterclockwise" if "counterclockwise" in q else "clockwise"
        deg = short.split()[0]
        return pick([
            f"Standing at the {pos} and facing the {face}, I turn {deg} degrees {turn} to face the {target}.",
            f"From the {pos} looking at the {face}, the {target} is {deg} degrees {turn}.",
        ])

    def _relative_distance_object(self, q, short, pick):
        outside = re.sub(r"\([^)]*\)", "", q)
        refs = [n for _, n in self.mentions(outside)]
        if len(refs) != 1 or refs[0] == short:
            return None
        return pick([
            f"The {short} is the closest to the {refs[0]}.",
            f"Of the listed objects, the {short} is nearest to the {refs[0]}.",
        ])

    def _relative_size_object(self, q, short, pick):
        found = [n for _, n in self.mentions(q)]
        if len(found) != 2 or short not in found or found[0] == found[1]:
            return None
        other = found[1] if found[0] == short else found[0]
        cmp = "shorter" if "shorter" in q else "longer"
        return pick([
            f"The {short}'s longest side is {cmp} than the {other}'s.",
            f"The longest side of the {short} is {cmp} than that of the {other}.",
        ])

    def _absolute_distance_object(self, q, short, pick):
        found = [n for _, n in self.mentions(q)]
        if len(found) != 2:
            return None
        a, b = found
        return pick([
            f"The {a} and the {b} are {short} apart at their closest points.",
            f"At their closest points, the {a} is {short} from the {b}.",
        ])

    def _absolute_size_object(self, q, short, pick):
        found = [n for _, n in self.mentions(q)]
        if len(found) != 1:
            return None
        return pick([
            f"The longest side of the {found[0]} is {short}.",
            f"The {found[0]}'s longest side measures {short}.",
        ])

    def _absolute_size_room(self, q, short, pick):
        return pick([f"The room covers about {short}.", f"The room is about {short} in size."])

    def _absolute_count(self, q, short, pick):
        cats = count_categories(q)
        if len(cats) != 1 or not short.isdigit():
            return None
        n = int(short)
        verb = "is" if n == 1 else "are"
        return pick([f"There {verb} {n} {plural(cats[0], n)} in this room.",
                     f"This room has {n} {plural(cats[0], n)}."])

    def _relative_count(self, q, short, pick):
        cats = list(dict.fromkeys(count_categories(q)))
        if len(cats) != 2:
            return None
        a, b = cats
        if re.match(r"(compared to|in comparison with)\b", q):
            a, b = b, a
        if short == "more":
            return f"There are more {plural(a)} than {plural(b)} in this room."
        if short == "fewer":
            return f"There are fewer {plural(a)} than {plural(b)} in this room."
        if short == "equal number":
            return f"There are as many {plural(a)} as {plural(b)} in this room."
        return None

    def _appearance_order(self, q, short, pick):
        return pick([f"The objects first appear in this order: {short}.",
                     f"In order of first appearance: {short}."])
