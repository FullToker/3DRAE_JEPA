"""How far apart does the Y-encoder put two sentences that differ only in a spatial relation?

No training. Sentences are encoded the way the model encodes its targets (mean of the last
hidden states of embeddinggemma; with a checkpoint also its fine-tuned weights and y_proj),
then compared by cosine similarity.

Part 1, built sentences. For a base sentence "The A is to the left of the B.":
  same meaning      paraphrase   "The A is located on the left side of the B."
                    converse     "The B is to the right of the A."
  opposite meaning  flip         "The A is to the right of the B."
                    swap         "The B is to the left of the A."
  other fact        other axis   "The A is above the B."
                    other object "The C is to the left of the B."
                    unrelated    another pair of objects, another relation
A target space that can carry the relation ranks same meaning above opposite meaning.

Part 2, the training data: each holdout target against its own wrong answers (the hard
negatives) and against the target of another row of the same question type.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

OBJECTS = ["chair", "table", "sofa", "bed", "lamp", "door", "window", "desk", "monitor", "bookshelf",
           "refrigerator", "sink", "toilet", "bathtub", "pillow", "backpack", "trash can", "microwave",
           "television", "plant", "cabinet", "nightstand", "radiator", "mirror"]

# (base phrase, paraphrase phrase, opposite base phrase, axis)
RELATIONS = [
    ("is to the left of", "is located on the left side of", "is to the right of", "x"),
    ("is to the right of", "is located on the right side of", "is to the left of", "x"),
    ("is above", "is positioned higher than", "is below", "y"),
    ("is below", "is positioned lower than", "is above", "y"),
    ("is in front of", "stands in front of", "is behind", "z"),
    ("is behind", "stands behind", "is in front of", "z"),
]
KINDS = ["paraphrase", "converse", "flip", "swap", "other axis", "other object", "unrelated"]
SAME, OPPOSITE = ["paraphrase", "converse"], ["flip", "swap"]


def built_sentences(count, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        a, b, c, d, e = rng.sample(OBJECTS, 5)
        base, para, opposite, axis = rng.choice(RELATIONS)
        other = rng.choice([r for r in RELATIONS if r[3] != axis])[0]
        rows.append({
            "base": f"The {a} {base} the {b}.",
            "paraphrase": f"The {a} {para} the {b}.",
            "converse": f"The {b} {opposite} the {a}.",
            "flip": f"The {a} {opposite} the {b}.",
            "swap": f"The {b} {base} the {a}.",
            "other axis": f"The {a} {other} the {b}.",
            "other object": f"The {c} {base} the {b}.",
            "unrelated": f"The {d} {other} the {e}.",
        })
    return rows


class Encoder:
    """Targets as the model encodes them (fusion_gv.gvjepa.FusionGVJEPA.encode_target)."""

    def __init__(self, name, cache_dir, device, checkpoint=None, max_tokens=256):
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir).to(device).eval()
        self.proj, self.device, self.max_tokens = None, device, max_tokens
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(state, dict) and key in state:
                    state = state[key]
                    break
            state = {k.removeprefix("module."): v for k, v in state.items()}
            encoder = {k.removeprefix("y_encoder."): v for k, v in state.items() if k.startswith("y_encoder.")}
            if not encoder or "y_proj.weight" not in state:
                raise KeyError(f"no y_encoder / y_proj in {checkpoint}; keys start with {sorted(state)[:5]}")
            print(f"[checkpoint] {checkpoint}: {self.model.load_state_dict(encoder, strict=True)}")
            weight = state["y_proj.weight"]
            self.proj = torch.nn.Linear(weight.shape[1], weight.shape[0])
            self.proj.load_state_dict({"weight": weight, "bias": state["y_proj.bias"]})
            self.proj.to(device).float().eval()

    @torch.no_grad()
    def __call__(self, texts, batch=128):
        """Unit vectors: (pooled, projected or None)."""
        pooled = []
        for start in range(0, len(texts), batch):
            tok = self.tokenizer(texts[start:start + batch], max_length=self.max_tokens, padding=True,
                                 truncation=True, return_tensors="pt").to(self.device)
            hidden = self.model(**tok).last_hidden_state.float()
            mask = tok["attention_mask"].unsqueeze(-1).float()
            pooled.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
        pooled = torch.cat(pooled)
        projected = F.normalize(self.proj(pooled), dim=-1) if self.proj is not None else None
        return F.normalize(pooled, dim=-1), projected


def built_part(encoder, rows):
    """Mean cosine of each variant to its base sentence, and how often meaning wins."""
    columns = ["base"] + KINDS
    texts = [row[c] for row in rows for c in columns]
    out = {}
    for space, vectors in zip(("pooled", "projected"), encoder(texts)):
        if vectors is None:
            continue
        vectors = vectors.view(len(rows), len(columns), -1)
        cos = (vectors[:, :1] * vectors[:, 1:]).sum(-1).cpu()        # (rows, kinds)
        sim = {kind: cos[:, i] for i, kind in enumerate(KINDS)}
        best_same = torch.stack([sim[k] for k in SAME]).max(0).values
        worst_same = torch.stack([sim[k] for k in SAME]).min(0).values
        best_opposite = torch.stack([sim[k] for k in OPPOSITE]).max(0).values
        out[space] = {
            "cosine": {k: {"mean": round(v.mean().item(), 4), "std": round(v.std().item(), 4)}
                       for k, v in sim.items()},
            "paraphrase_above_flip": round((sim["paraphrase"] > sim["flip"]).float().mean().item(), 4),
            "converse_above_flip": round((sim["converse"] > sim["flip"]).float().mean().item(), 4),
            "converse_above_swap": round((sim["converse"] > sim["swap"]).float().mean().item(), 4),
            "every_same_above_every_opposite": round((worst_same > best_opposite).float().mean().item(), 4),
            "some_same_above_every_opposite": round((best_same > best_opposite).float().mean().item(), 4),
            "flip_above_other_object": round((sim["flip"] > sim["other object"]).float().mean().item(), 4),
        }
    return out


def data_part(encoder, manifest, per_type, seed):
    """Each target against its own wrong answers and against another row's target."""
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for line in open(manifest):
        row = json.loads(line)
        negatives = row.get("negatives") or []
        if isinstance(negatives, str):
            continue
        if negatives:
            by_type[row["question_type"]].append((row["target"], negatives))
    out, examples = {}, {}
    for kind, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        rows = rows[:per_type]
        targets = [t for t, _ in rows]
        negatives = [n for _, ns in rows for n in ns]
        owner = torch.tensor([i for i, (_, ns) in enumerate(rows) for _ in ns])
        other = torch.tensor([(i + 1 + rng.randrange(len(rows) - 1)) % len(rows) for i in range(len(rows))])
        examples[kind] = {"target": rows[0][0], "wrong": rows[0][1][0]}
        out[kind] = {"rows": len(rows)}
        for space, vectors in zip(("pooled", "projected"), encoder(targets + negatives)):
            if vectors is None:
                continue
            vectors = vectors.cpu()
            t, n = vectors[:len(targets)], vectors[len(targets):]
            wrong = (t[owner] * n).sum(-1)
            hardest = torch.full((len(rows),), -1.0).scatter_reduce(0, owner, wrong, "amax")
            another = (t * t[other]).sum(-1)
            out[kind][space] = {
                "wrong_answer": round(wrong.mean().item(), 4),
                "hardest_wrong_answer": round(hardest.mean().item(), 4),
                "another_row": round(another.mean().item(), 4),
                "wrong_closer_than_another_row": round((hardest > another).float().mean().item(), 4),
            }
    return out, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="google/embeddinggemma-300m")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="label=path, may repeat")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--built", type=int, default=2000)
    parser.add_argument("--per-type", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = built_sentences(args.built, args.seed)
    result = {"built_example": rows[0], "encoders": {}}
    runs = [("pretrained", None)] + [tuple(item.split("=", 1)) for item in args.checkpoint]
    for label, checkpoint in runs:
        encoder = Encoder(args.name, args.cache_dir, device, checkpoint)
        built = built_part(encoder, rows)
        data, examples = data_part(encoder, args.manifest, args.per_type, args.seed)
        result["encoders"][label] = {"built": built, "data": data}
        result["data_examples"] = examples
        print(f"\n===== {label} =====")
        print(json.dumps(built, indent=1))
        print(json.dumps(data, indent=1))
        del encoder
        torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
