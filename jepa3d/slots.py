"""Object slots: the objects a question names, each with candidate 3D positions.

A slot belongs to one object name of the question:
    {"name": "trash bin", "spans": [[start, end], ...],      characters of the name in the query
     "pos": [[x, y, z], ...], "conf": [c, ...],              k candidates, best first
     "patch": [p, ...]}                                      their patches, flat index over (view, row, column)
pos is in the frame of the first view, in metres; conf is dino.txt's
standardized score of the candidate's patch. jepa3d/tools/precompute_slots.py
writes them into the manifest rows ("slots"). ThreeDRAEJEPA uses them in one of two ways:
  slots_v1  SlotEncoder turns pos and conf into one vector per name, added to the name's query tokens;
  slots_v2  ObjectTokens gives every name a token of its own, built from what the candidates' patches
            show (frozen patch feature + pixel ray, no position), and asks the predictor's output at that
            token for the position.

Names are found with a vocabulary (build_vocabulary) because the questions are
paraphrased: SPAR tags its objects as "<name> (Object0)", every other question
is searched for the longest vocabulary names.
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa3d.vsi_text import NON_OBJECTS, count_categories

NUM_CANDIDATES = 4
# Words of the question templates that also occur as object names.
NOT_NAMES = NON_OBJECTS | {"object", "objects", "category", "categories", "order", "side", "point", "points", "center",
                           "centre", "edge", "instance", "space", "area", "size", "view", "clip", "scene", "observer",
                           "place", "spot", "position", "world", "system", "distance", "sequence", "end", "top", "case"}
# Verbs of the question templates that are also object names ("can you ...", "stand out").
NOT_NAMES |= {"can", "stand", "cover", "note", "show", "watch", "turn", "face", "count", "number", "present", "sit",
              "look", "total", "step", "rest", "set", "match", "light up"}
TAGGED = re.compile(r"([a-z0-9][a-z0-9 ,'/&\-]*?) \(object\d+\)")
LISTED = re.compile(r"\(([a-z][a-z0-9 '/&\-]*(?:, [a-z][a-z0-9 '/&\-]*)+)\)")
ORDER_ANSWER = re.compile(r": ([^:]+?)\.?$")


def build_vocabulary(rows) -> list:
    """Object names of manifest rows, longest first: appearance-order answers, option lists in brackets and
    "<name>(s)" of count questions. Names that also turn up in room-size questions are template words."""
    names, template = set(), []
    for row in rows:
        query = row["query"].split("\n")[0].lower()
        kind = str(row.get("question_type") or row.get("type"))
        if "appearance_order" in kind:
            answer = ORDER_ANSWER.search(row["target"].lower()) or re.match(r"(.+)", row["target"].lower())
            names.update(x.strip() for x in answer.group(1).split(","))
        elif "room" in kind:
            template.append(query)
        for match in LISTED.finditer(query):
            names.update(x.strip() for x in match.group(1).split(","))
        names.update(count_categories(row["query"]))
    names = {n for n in names if len(n) > 1 and n not in NOT_NAMES and not n.isdigit() and len(n.split()) <= 5}
    if template:
        finder = NameFinder(names)
        names -= {name for query in template for name, _ in finder(query)}
    return sorted(names, key=lambda n: (-len(n), n))


class NameFinder:
    """Object names of a question. SPAR tags them as "<name> (ObjectN)": there the name is the longest
    vocabulary name that ends at the tag (else the last word). Other questions are searched for the longest
    vocabulary names, plural endings allowed."""

    def __init__(self, vocabulary):
        names = "|".join(re.escape(n) for n in sorted(set(vocabulary) - NOT_NAMES, key=lambda n: (-len(n), n)))
        self.anywhere = re.compile(rf"(?<![a-z0-9])({names})(?=(?:s|es)?(?![a-z0-9]))")
        self.at_end = re.compile(rf"(?:.*[^a-z0-9])??({names}|[a-z0-9'\-]+)$")

    def __call__(self, query: str) -> list:
        """[(name, [[start, end], ...])] in order of first mention; only the question's first line is searched."""
        line = query.split("\n")[0].lower()
        found = []
        for tag in TAGGED.finditer(line):
            match = self.at_end.match(tag.group(1))
            if match:
                found.append((match.group(1), tag.start(1) + match.start(1), tag.start(1) + match.end(1)))
        if not found:
            found = [(m.group(1), m.start(1), m.end(1)) for m in self.anywhere.finditer(line)]
        spans = {}
        for name, start, end in found:
            spans.setdefault(name, []).append([start, end])
        return list(spans.items())


def fourier(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (..., C) -> (..., C * 2 * bands): sin / cos at wavelengths 32 m, 16 m, ... (x in metres)."""
    freqs = 2 * math.pi / 32 * 2 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


class SlotEncoder(nn.Module):
    """Candidates (..., K, 3) + confidences (..., K) -> one vector (..., dim) per name.

    Each candidate is embedded from its position and confidence, the K
    embeddings are averaged with learned weights. The output layer starts at
    zero, so an untrained encoder leaves the query tokens as they are."""

    def __init__(self, dim: int, hidden: int = 512, bands: int = 8):
        super().__init__()
        self.bands = bands
        self.embed = nn.Sequential(nn.Linear(3 * 2 * bands + 3 + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                   nn.GELU())
        self.weight = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pos: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        x = self.embed(torch.cat([fourier(pos, self.bands), pos / 4, conf[..., None] / 4], dim=-1))
        return self.out((self.weight(x).softmax(-2) * x).sum(-2))


class ObjectTokens(nn.Module):
    """One token per object name, and the loss on the predictor's output at that token.

    Input: for each of a name's K candidates the frozen patch feature and the
    Plucker ray of the patch centre (jepa3d.geo_probe.train.point_queries) with
    dino.txt's confidence; the K embeddings are averaged with learned weights.
    The token has the size of an ordinary query token (LayerNorm, then
    gain / sqrt(dim)) and carries a type embedding. It does not contain the
    position.

    Loss: the output at the token must pick the position of the name's best
    candidate among the positions of all names of the batch (InfoNCE; positions
    divided by the scene scale, as the cameras the scene tokens were built
    with). The position is not in the input: it has to be read from the scene
    tokens along the token's rays. A linear head on the detached output reports
    the error in metres.
    """

    def __init__(self, dim: int, query_dim: int, hidden: int = 512, embed: int = 256, bands: int = 8,
                 temperature: float = 0.07):
        super().__init__()
        self.bands, self.temperature = bands, temperature
        self.embed = nn.Sequential(nn.LayerNorm(query_dim + 1), nn.Linear(query_dim + 1, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.weight = nn.Linear(hidden, 1)
        self.out = nn.Sequential(nn.Linear(hidden, dim), nn.LayerNorm(dim))
        self.gain = nn.Parameter(torch.ones(()))
        self.type_embed = nn.Parameter(torch.randn(dim) * 0.02)
        self.read = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, embed))
        self.position = nn.Sequential(nn.Linear(3 * 2 * bands + 3, hidden), nn.GELU(), nn.Linear(hidden, embed))
        self.metres = nn.Linear(dim, 3)

    def forward(self, queries: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """queries (N, K, C + 6), conf (N, K) -> tokens (N, dim)."""
        x = self.embed(torch.cat([queries, conf[..., None] / 4], dim=-1))
        x = self.out((self.weight(x).softmax(-2) * x).sum(-2))
        return x * self.gain / x.shape[-1] ** 0.5 + self.type_embed

    def loss(self, hidden: torch.Tensor, pos: torch.Tensor, scale: torch.Tensor) -> dict:
        """hidden (N, dim) predictor output at the tokens, pos (N, 3) metres, scale (N,) ->
        {"loss", "acc" (share of names whose own position scores highest), "error_m" (median), "regression"}."""
        hidden = hidden.float()
        target = pos / scale[:, None]
        t = F.normalize(self.position(torch.cat([fourier(4 * target, self.bands), target], dim=-1)), dim=-1)
        logits = F.normalize(self.read(hidden), dim=-1) @ t.T / self.temperature
        same = (torch.cdist(target, target) < 0.02) & ~torch.eye(len(target), dtype=torch.bool, device=pos.device)
        logits = logits.masked_fill(same, float("-inf"))   # the same object in another row is not a negative
        labels = torch.arange(len(target), device=pos.device)
        guess = self.metres(hidden.detach())
        return {"loss": F.cross_entropy(logits, labels), "acc": (logits.argmax(1) == labels).float().mean(),
                "regression": F.smooth_l1_loss(guess, target),
                "error_m": ((guess - target).norm(dim=-1) * scale).median().detach()}


def scene_scale(c2w) -> float:
    """Largest camera distance from the first view (metres): jepa3d.data.normalize_cameras divides by it."""
    c2w = torch.tensor(c2w, dtype=torch.float64)
    scale = float((torch.linalg.inv(c2w[0]) @ c2w)[:, :3, 3].norm(dim=-1).max())
    return scale if scale > 1e-6 else 1.0


def collate_slots(rows, drop: bool = False):
    """Rows' "slots" -> None if the batch has none, else
    {"pos" (N, K, 3), "conf" (N, K), "row" (N,) batch index, "spans" list of N span lists,
     "scale" (N,) scene scale, "patch" (N, K) if the manifest has it}."""
    pos, conf, owner, spans, scale, patch, names = [], [], [], [], [], [], []
    for i, row in enumerate(rows):
        for slot in ([] if drop else row.get("slots") or []):
            k = min(len(slot["pos"]), NUM_CANDIDATES)
            pad = NUM_CANDIDATES - k   # fewer candidates than K: repeat the best one
            pos.append(slot["pos"][:k] + [slot["pos"][0]] * pad)
            conf.append(slot["conf"][:k] + [slot["conf"][0]] * pad)
            owner.append(i)
            spans.append(slot["spans"])
            names.append(slot.get("name", ""))
            scale.append(scene_scale(row["c2w"]) if "c2w" in row else 1.0)
            if "patch" in slot:
                patch.append(slot["patch"][:k] + [slot["patch"][0]] * pad)
    if not pos:
        return None
    out = {"pos": torch.tensor(pos, dtype=torch.float32), "conf": torch.tensor(conf, dtype=torch.float32),
           "row": torch.tensor(owner), "spans": spans, "names": names,
           "scale": torch.tensor(scale, dtype=torch.float32)}
    if len(patch) == len(pos):
        out["patch"] = torch.tensor(patch)
    return out
