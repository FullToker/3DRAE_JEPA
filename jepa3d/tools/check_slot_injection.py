#!/usr/bin/env python3
"""Does the slot injection reach the prediction, and on the right tokens?

On a few rows of a manifest with slots (untrained model, as built from the config):
  - prints each slot's name next to the query tokens its vector is added to;
  - gives the slot encoder's zero output layer random weights and reports how
    far the predicted embedding moves with slots, with swapped slots and with
    the candidates of one name changed (cosine distance to the prediction
    without slots; all three must be > 0, and 0 before the weights are set).

Usage (from the repo root):
    python jepa3d/tools/check_slot_injection.py --config jepa3d/configs/slots_v1.yaml \
        --manifest experiments/slots_v1/data/holdout_f8.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config
from jepa3d.train import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = 2
    device = torch.device("cuda")
    model = build_model_from_config(cfg).to(device).eval()
    batch = next(iter(build_loader([args.manifest], cfg, args.rows, train=False)))
    slots, queries = batch["slots"], batch["query"]
    images, cams = batch["images_vggt"].to(device), batch["images_jepa"].to(device)

    ids = model.query_tokenizer(queries, max_length=model.config.max_query_tokens, padding=True, truncation=True,
                                return_tensors="pt")["input_ids"]
    on_name = model.name_tokens(queries, slots)
    for n, row in enumerate(slots["row"].tolist()):
        name = queries[row][slots["spans"][n][0][0]:slots["spans"][n][0][1]]
        print(f"row {row}  {name!r:28} -> {model.query_tokenizer.convert_ids_to_tokens(ids[row][on_name[n]])}")

    def predict(s):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return F.normalize(model.predict_embedding(images, cams, queries, slots=s).float(), dim=-1)

    def moved(s):
        return [round(x, 4) for x in (1 - (predict(s) * base).sum(-1)).tolist()]

    base = predict(None)
    print("distance between the predictions of row 0 and the other rows:",
          [round(x, 4) for x in (1 - base @ base[0]).tolist()])
    seen = []
    handle = model.query_in_proj.register_forward_hook(
        lambda _m, inputs, out: seen.append((float(inputs[0].float().norm(dim=-1).mean()),
                                             float((out - inputs[0]).float().norm(dim=-1).max()))))
    print("zero output layer, with slots:", moved(slots))
    torch.manual_seed(0)
    torch.nn.init.normal_(model.slot_encoder.out.weight, std=0.02)
    swapped = {**slots, "pos": slots["pos"].roll(len(slots["row"]) // 2, 0)}
    one = {**slots, "pos": slots["pos"].clone()}
    one["pos"][0] += 2.0   # the first name of row 0, two metres away
    seen.clear()
    print("with slots:                   ", moved(slots))
    print("query token norm (mean), largest added slot vector norm:", seen)
    handle.remove()
    print("swapped slots:                ", moved(swapped))
    print("one name of row 0 moved 2 m, distance to the prediction with the real slots:",
          [round(x, 4) for x in (1 - (predict(one) * predict(slots)).sum(-1)).tolist()])


if __name__ == "__main__":
    main()
