#!/usr/bin/env python3
"""Does a trained slot encoder say where the object is, and does the predictor listen?

For a checkpoint trained with slots (jepa3d/configs/slots_v1.yaml):
  vectors     the slot vectors of all slots of a manifest: their norm, the share
              of it that differs between slots, how far a vector moves when its
              candidates are shifted or its confidence changed, and how well the
              best candidate's position can be read back from the vector
              (ridge regression on half of the slots, error on the other half;
              "mean" is the error of always answering the mean position).
  prediction  on a few batches: how far the predicted embedding moves (cosine
              distance) without slots, with the slots of other rows, with all
              candidates shifted, next to the distance between two rows.

Usage (from the repo root):
    python jepa3d/tools/diagnose_slot_encoder.py --config jepa3d/configs/slots_v1.yaml \
        --checkpoint experiments/slots_v1/ckpt/final.pt --manifest experiments/slots_v1/data/holdout_f8.jsonl \
        --output experiments/slots_v1/results/diagnose_slot_encoder.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from fusion_gv.infer_gvjepa import _load_checkpoint
from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config
from jepa3d.slots import collate_slots
from jepa3d.train import load_config

SHIFTS = (0.25, 0.5, 1.0, 2.0)   # metres


def vector_stats(encoder, slots, device) -> dict:
    pos, conf = slots["pos"].to(device), slots["conf"].to(device)
    vectors = encoder(pos, conf)
    mean = vectors.mean(0)
    norm = vectors.norm(dim=-1)
    stats = {"slots": len(pos), "norm_mean": float(norm.mean()), "mean_vector_norm": float(mean.norm()),
             "varying_part_norm_mean": float((vectors - mean).norm(dim=-1).mean()),
             "varying_share_of_squared_norm": float((vectors - mean).pow(2).sum() / vectors.pow(2).sum())}
    generator = torch.Generator(device=device).manual_seed(0)
    direction = F.normalize(torch.randn(len(pos), 1, 3, device=device, generator=generator), dim=-1)
    change = lambda other: float(((other - vectors).norm(dim=-1) / norm).mean())
    stats["relative_change"] = {
        **{f"shift_{s}m": change(encoder(pos + s * direction, conf)) for s in SHIFTS},
        "other_slot": change(encoder(pos.roll(len(pos) // 2, 0), conf.roll(len(pos) // 2, 0))),
        "conf_plus_2": change(encoder(pos, conf + 2)), "conf_minus_2": change(encoder(pos, conf - 2))}

    # Read the best candidate's position back from the vector.
    order = torch.randperm(len(pos), device=device, generator=generator)
    fit, test = order[: len(pos) // 2], order[len(pos) // 2:]
    target = pos[:, 0]
    centre, scale = vectors[fit].mean(0), vectors[fit].std(0) + 1e-8
    x = lambda idx: torch.cat([(vectors[idx] - centre) / scale, torch.ones(len(idx), 1, device=device)], dim=1).double()
    stats["position_readout_m"] = {"mean": float((target[test] - target[fit].mean(0)).norm(dim=-1).median())}
    for ridge in (1e-2, 1.0, 100.0):
        a = x(fit)
        w = torch.linalg.solve(a.T @ a + ridge * torch.eye(a.shape[1], device=device, dtype=a.dtype),
                               a.T @ target[fit].double())
        stats["position_readout_m"][f"ridge_{ridge}"] = float((x(test) @ w - target[test]).norm(dim=-1).median())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = 4
    device = torch.device("cuda")
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    result = {"checkpoint": args.checkpoint, "step": step, "manifest": args.manifest}

    with open(args.manifest) as f:
        rows = [json.loads(line) for line in f]
    with torch.no_grad():
        result["vectors"] = vector_stats(model.slot_encoder.float(), collate_slots(rows), device)
    print(json.dumps(result["vectors"], indent=1))

    seen = []
    model.query_in_proj.register_forward_hook(lambda _m, inputs, out: seen.append(
        (float(inputs[0].float().norm(dim=-1).mean()), float((out - inputs[0]).float().norm(dim=-1).max()))))

    def predict(batch, slots):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.predict_embedding(batch["images_vggt"].to(device), batch["images_jepa"].to(device),
                                          batch["query"], slots=slots)
        return F.normalize(out.float(), dim=-1)

    names = ["none", "other_rows", *(f"shift_{s}m" for s in SHIFTS), "between_rows"]
    distances = {name: [] for name in names}
    loader = build_loader([args.manifest], cfg, args.batch_size, train=False)
    for _, batch in zip(range(args.batches), loader):
        slots = batch["slots"]
        if slots is None:
            continue
        has = torch.zeros(len(batch["query"]), dtype=torch.bool)
        has[slots["row"]] = True
        real = predict(batch, slots)
        variants = {"none": None, "other_rows": {**slots, "pos": slots["pos"].roll(len(slots["row"]) // 2, 0),
                                                 "conf": slots["conf"].roll(len(slots["row"]) // 2, 0)}}
        for s in SHIFTS:
            variants[f"shift_{s}m"] = {**slots, "pos": slots["pos"] + s * F.normalize(torch.randn(len(slots["row"]), 1, 3), dim=-1)}
        for name, variant in variants.items():
            distances[name] += (1 - (predict(batch, variant) * real).sum(-1))[has].tolist()
        distances["between_rows"] += (1 - (real * real.roll(1, 0)).sum(-1)).tolist()
    median = lambda values: float(torch.tensor(values).median())
    result["prediction_cosine_distance_median"] = {name: median(values) for name, values in distances.items()}
    result["query_token_norm_mean"] = seen[0][0]
    result["largest_added_vector_norm"] = max(added for _, added in seen)
    print(json.dumps({k: v for k, v in result.items() if k != "vectors"}, indent=1))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
