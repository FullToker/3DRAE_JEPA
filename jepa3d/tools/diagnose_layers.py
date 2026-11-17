#!/usr/bin/env python3
"""Layer by layer: do the question and the scene tokens interact, and where does a name's position live?

For a checkpoint, on the first rows of a manifest with slots (the hidden states
are those after each predictor layer; "input" is what enters the first layer):

  attention   share of the attention of the question tokens that lands on the scene
              tokens and of the scene tokens that lands on the question, next to the
              share an even spread would give (only if the layers return weights).
  knockout    attention from question to scene tokens (q>s), from scene to question
              tokens (s>q), or from question to object tokens (q>o, models with object
              tokens) is blocked in one layer or in all: how far the prediction moves
              (cosine distance) and the multiple-choice accuracy.
  scene       the scene tokens of another row: how much the hidden states of the
              question tokens change in each layer (relative L2).
  slots       (models with slots) the slots taken from other names, and with a slot
              encoder also shifted by 2 m: change of the name tokens, the other
              question tokens and the scene tokens in each layer.
  readback    the position of each name's best slot candidate, read from the mean
              hidden state of the name's tokens in each layer and from the mean of
              the scene tokens, with object tokens also from the name's object token
              (ridge regression on half of the names, median error on the other half;
              "mean" always answers the mean position).
              Without slot injection this asks whether the model located the
              object on its own.

Usage (from the repo root):
    python jepa3d/tools/diagnose_layers.py --config jepa3d/configs/slots_v1.yaml \
        --checkpoint experiments/slots_v1/ckpt/final.pt --manifest experiments/slots_v1/data/holdout_f8.jsonl \
        --output experiments/layer_diagnosis/results/slots_v1.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
from fusion_gv.infer_gvjepa import _load_checkpoint
from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config
from jepa3d.tools.eval_holdout import candidates_of
from jepa3d.train import load_config


class Tap:
    """Hooks on the predictor's layers: records hidden states and attention shares, blocks attention."""

    def __init__(self, model):
        self.model, self.layers = model, model.predictor.layers
        self.hidden, self.shares = [], []
        self.block = {}          # layer index -> "q>s" | "s>q"
        self.record = False
        self.attention = False   # also record attention shares
        self.scene = 0           # number of scene tokens of the forward in progress
        self.query = 0           # number of question tokens (object tokens follow them)
        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(lambda _m, args, kwargs, i=i: self._enter(i, args, kwargs), with_kwargs=True)
            layer.register_forward_hook(lambda _m, _a, out: self._leave(out))
            layer.self_attn.register_forward_hook(lambda _m, _a, out: self._attention(out))

    def eager(self, on: bool) -> None:
        """Ask the layers for attention weights (slower, more memory)."""
        self.attention = on
        for config in {id(c): c for c in (getattr(layer.self_attn, "config", None) for layer in self.layers)
                       if c is not None}.values():   # the layers share one config
            if not hasattr(self, "_implementation"):
                self._implementation = config._attn_implementation
            config._attn_implementation = "eager" if on else self._implementation

    def _enter(self, i, args, kwargs):
        mask = kwargs["attention_mask"]                       # (B, 1, T, T) additive
        self.valid = mask[:, 0, 0] == 0                       # (B, T)
        if self.record and i == 0:
            self.hidden.append(args[0].float())
        if i in self.block:
            mask = mask.clone()
            s, q, low = self.scene, self.scene + self.query, torch.finfo(mask.dtype).min
            if self.block[i] == "q>s":
                mask[:, :, s:, :s] = low
            elif self.block[i] == "q>o":
                mask[:, :, s:q, q:] = low
            else:
                mask[:, :, :s, s:] = low
            return args, {**kwargs, "attention_mask": mask}
        return None

    def _leave(self, out):
        if self.record:
            self.hidden.append((out[0] if isinstance(out, tuple) else out).float())

    def _attention(self, out):
        weights = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if not (self.record and self.attention) or weights is None:
            return
        s, valid = self.scene, self.valid.float()
        to_scene = weights.float().mean(1)[..., :s].sum(-1)   # (B, T) mass each token puts on the scene tokens
        question = valid[:, s:]
        self.shares.append({
            "question_on_scene": float((to_scene[:, s:] * question).sum() / question.sum()),
            "scene_on_question": float((1 - to_scene[:, :s]).mean()),
            "even_question_on_scene": float((s / valid.sum(1)).mean()),
            "even_scene_on_question": float((question.sum(1) / valid.sum(1)).mean())})


def readback(features: torch.Tensor, target: torch.Tensor) -> dict:
    """features (N, D), target (N, 3) -> median error (m) on the second half of a fixed random split."""
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(0)).to(target.device)
    fit, test = order[: len(order) // 2], order[len(order) // 2:]
    centre, scale = features[fit].mean(0), features[fit].std(0) + 1e-6
    x = lambda idx: torch.cat([(features[idx] - centre) / scale, torch.ones(len(idx), 1, device=target.device)], 1).double()
    errors = {}
    for ridge in (10.0, 100.0, 1000.0, 10000.0):
        a = x(fit)
        w = torch.linalg.solve(a.T @ a + ridge * torch.eye(a.shape[1], device=a.device, dtype=a.dtype),
                               a.T @ target[fit].double())
        errors[ridge] = float((x(test) @ w - target[test]).norm(dim=-1).median())
    best = min(errors, key=errors.get)
    return {"error_m": errors[best], "ridge": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="a manifest with slots (also for models without injection)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--attention-batches", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = 4
    device = torch.device("cuda")
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    tap = Tap(model)
    depth = len(tap.layers)
    injects = model.uses_slots
    directions = ("q>s", "s>q", "q>o") if model.object_tokens is not None else ("q>s", "s>q")

    cached = {}
    original = type(model)._pool_visual

    def pool(images, cams, batch_size):
        if "tokens" not in cached:
            cached["tokens"] = original(model, images, cams, batch_size)[0]
        tokens = cached["tokens"].roll(1, 0) if cached["shuffle"] else cached["tokens"]
        tap.scene = tokens.shape[1]
        return tokens, tokens.unsqueeze(1)
    model._pool_visual = pool

    def run(batch, slots, shuffle=False, block=None, record=False):
        """-> normalized prediction (B, D); with record also the hidden states [(B, T, h)] * (depth + 1)."""
        cached["shuffle"], tap.block, tap.record, tap.hidden = shuffle, block or {}, record, []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.predict_embedding(batch["images_vggt"].to(device), batch["images_jepa"].to(device),
                                           batch["query"], slots=slots if injects else None)
        tap.record = False
        return F.normalize(pred.float(), dim=-1), tap.hidden

    knockouts = {"none": {}}
    for direction in directions:
        knockouts[f"{direction} all"] = {i: direction for i in range(depth)}
        for i in range(depth):
            knockouts[f"{direction} layer {i + 1}"] = {i: direction}
    moved, correct = defaultdict(list), defaultdict(list)
    change = defaultdict(lambda: [[] for _ in range(depth + 1)])
    name_states, scene_states, object_states, positions = ([[] for _ in range(depth + 1)] for _ in range(3)), []
    name_states, scene_states, object_states = name_states
    relative = lambda a, b, keep: float(((a - b).norm(dim=-1)[keep] / b.norm(dim=-1)[keep].clamp_min(1e-6)).mean())

    loader = build_loader([args.manifest], cfg, args.batch_size, train=False)
    for number, batch in zip(range(args.batches), loader):
        slots = batch["slots"]
        if slots is None:
            continue
        cached.pop("tokens", None)
        on_name = model.name_tokens(batch["query"], slots).to(device)       # (N, L)
        tap.query = L = on_name.shape[1]
        if number < args.attention_batches:
            tap.eager(True)
        base, hidden = run(batch, slots, record=True)
        if tap.attention:
            tap.eager(False)
        s = tap.scene
        q = slice(s, s + L)
        valid = tap.valid[:, q]                                              # (B, L) real question tokens
        rows = slots["row"].to(device)
        column = torch.zeros_like(rows)                                      # as ThreeDRAEJEPA._append_tokens
        for n in range(1, len(rows)):
            column[n] = column[n - 1] + 1 if rows[n] == rows[n - 1] else 0
        in_name = torch.zeros(valid.shape, device=device).index_add(0, rows, on_name.float()).bool() & valid
        other = valid & ~in_name
        weight = on_name.float() / on_name.sum(1, keepdim=True).clamp_min(1)
        for depth_index, h in enumerate(hidden):
            name_states[depth_index].append(torch.einsum("nl,nlh->nh", weight, h[rows, q]).cpu())
            if model.object_tokens is not None:
                object_states[depth_index].append(h[rows, s + L + column].cpu())
            scene_states[depth_index].append(h[rows, :s].mean(1).cpu())
        positions.append(slots["pos"][:, 0])

        # answers: candidates embedded once, scored for every knockout
        per_row = [candidates_of(row)["long"] for row in batch["rows"]]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            options = F.normalize(model.encode_target([c for cands, _ in per_row for c in cands], device).float(), dim=-1)
        bounds = torch.tensor([0] + [len(cands) for cands, _ in per_row]).cumsum(0).tolist()

        def score(pred, key):
            for i, (cands, answer) in enumerate(per_row):
                best = int((options[bounds[i]:bounds[i + 1]] @ pred[i]).argmax())
                correct[key].append(cands[best] == answer)

        for key, block in knockouts.items():
            pred = base if key == "none" else run(batch, slots, block=block)[0]
            moved[key] += (1 - (pred * base).sum(-1)).tolist()
            score(pred, key)
        moved["two rows"] += (1 - (base * base.roll(1, 0)).sum(-1)).tolist()

        pred, states = run(batch, slots, shuffle=True, record=True)
        moved["scene of another row"] += (1 - (pred * base).sum(-1)).tolist()
        score(pred, "scene of another row")
        for depth_index, (a, b) in enumerate(zip(states, hidden)):
            change["scene of another row: question tokens"][depth_index].append(relative(a[:, q], b[:, q], valid))
        if injects:
            shift = F.normalize(torch.randn(len(rows), 1, 3, generator=torch.Generator().manual_seed(number)), dim=-1)
            half = max(1, len(rows) // 2)
            variants = {"slots of other names": {**slots, **{k: slots[k].roll(half, 0)
                                                             for k in ("pos", "conf", "patch") if k in slots}}}
            if model.slot_encoder is not None:
                variants["slots shifted 2 m"] = {**slots, "pos": slots["pos"] + 2 * shift}
            for key, variant in variants.items():
                pred, states = run(batch, variant, record=True)
                moved[key] += (1 - (pred * base).sum(-1)).tolist()
                score(pred, key)
                for depth_index, (a, b) in enumerate(zip(states, hidden)):
                    for part, keep in (("name tokens", in_name), ("other question tokens", other)):
                        change[f"{key}: {part}"][depth_index].append(relative(a[:, q], b[:, q], keep))
                    change[f"{key}: scene tokens"][depth_index].append(
                        relative(a[:, :s], b[:, :s], torch.ones_like(a[:, :s, 0], dtype=torch.bool)))

    labels = ["input"] + [f"layer {i + 1}" for i in range(depth)]
    mean = lambda values: sum(values) / len(values)
    result = {"checkpoint": args.checkpoint, "step": step, "manifest": args.manifest, "rows": len(correct["none"]),
              "slot_injection": injects, "layers": labels,
              "attention": [{k: mean([x[k] for x in tap.shares[i::depth]]) for k in tap.shares[0]}
                            for i in range(depth)] if tap.shares else None,
              "knockout": {key: {"prediction_moved": float(torch.tensor(moved[key]).median()),
                                 "accuracy": mean(correct[key]) if key in correct else None} for key in moved},
              "hidden_change": {key: [mean(v) for v in values] for key, values in change.items()}}
    target = torch.cat(positions).to(device)
    result["readback"] = {"names": len(target), "mean": float((target - target.mean(0)).norm(dim=-1).median()),
                          "name_tokens": [readback(torch.cat(x).to(device), target) for x in name_states],
                          "scene_tokens": [readback(torch.cat(x).to(device), target) for x in scene_states]}
    if model.object_tokens is not None:
        result["readback"]["object_tokens"] = [readback(torch.cat(x).to(device), target) for x in object_states]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
