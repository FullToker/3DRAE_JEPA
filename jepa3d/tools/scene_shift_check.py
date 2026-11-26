#!/usr/bin/env python3
"""What does another scene change in the prediction: its distance to the answer, or the choice among the options?

For every row the model predicts twice, with its own scene and with the previous row's (as scneg_v1's
training term and the "shuffled" control do). Against the row's options (answer first, then the wrong
answers) it reports

    real_closer      share of rows whose own-scene prediction is closer to the answer (what scneg_v1 trains)
    cos_answer       mean cosine to the answer, own scene / other scene
    cos_wrong        mean cosine to the wrong options, own scene / other scene
    margin           mean of (cosine to the answer - best cosine to a wrong option), own / other
    same_choice      share of rows whose chosen option is the same with both scenes
    acc              accuracy, own / other

Usage (from the repo root):
    python jepa3d/tools/scene_shift_check.py --config <yaml> --checkpoint <pt> \
        --manifests <holdout jsonl> <vsibench jsonl> --output <json>
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
from fusion_gv.infer_gvjepa import _candidates, _load_checkpoint, _target
from jepa3d.data import build_loader
from jepa3d.model import build_model_from_config
from jepa3d.tools.blind_check import set_mode, slots_of
from jepa3d.train import load_config


def options_of(row: dict):
    """-> [answer, wrong answers...] or None."""
    if row.get("negatives"):
        return [row["target"], *row["negatives"]]
    cands = _candidates(row)
    answer, _ = _target(row, cands)
    if answer is None or len(cands) < 2:
        return None
    return [answer, *[c for c in cands if c != answer]]


@torch.no_grad()
def cosines(model, loader, device, mode: str) -> list:
    """Per row: cosines of the prediction to [answer, wrong answers...] (None for rows without options)."""
    set_mode(model, mode)
    out = []
    for batch in loader:
        images = batch["images_vggt"].to(device, non_blocking=True)
        cams = batch["images_jepa"].to(device, non_blocking=True)
        options = [options_of(row) for row in batch["rows"]]
        flat = [o for opts in options if opts for o in opts]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = F.normalize(model.predict_embedding(images, cams, batch["query"],
                                                       slots=slots_of(model, batch)).float(), dim=-1)
            target = F.normalize(model.encode_target(flat, device).float(), dim=-1) if flat else None
        start = 0
        for i, opts in enumerate(options):
            if not opts:
                out.append(None)
                continue
            out.append((target[start:start + len(opts)] @ pred[i]).cpu())
            start += len(opts)
    del model._pool_visual
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = 8
    device = torch.device("cuda")
    model = build_model_from_config(cfg)
    _load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    results = {"checkpoint": args.checkpoint}
    for path in args.manifests:
        rows = []
        with open(path) as f:
            for n, line in enumerate(f):
                if args.limit and n >= args.limit:
                    break
                row = json.loads(line)
                rows.append(f"{row.get('source', 'vsibench')}/{row.get('question_type') or row.get('type')}")
        per_mode = {mode: cosines(model, build_loader([path], cfg, args.batch_size, train=False, limit=args.limit),
                                  device, mode) for mode in ("real", "shuffled")}
        stats = defaultdict(lambda: defaultdict(list))
        for kind, real, other in zip(rows, per_mode["real"], per_mode["shuffled"]):
            if real is None:
                continue
            for key in ("all", kind):
                s = stats[key]
                s["real_closer"].append(float(real[0] > other[0]))
                s["cos_answer_real"].append(float(real[0]))
                s["cos_answer_other"].append(float(other[0]))
                s["cos_wrong_real"].append(float(real[1:].mean()))
                s["cos_wrong_other"].append(float(other[1:].mean()))
                s["margin_real"].append(float(real[0] - real[1:].max()))
                s["margin_other"].append(float(other[0] - other[1:].max()))
                s["same_choice"].append(float(real.argmax() == other.argmax()))
                s["acc_real"].append(float(real.argmax() == 0))
                s["acc_other"].append(float(other.argmax() == 0))
        results[Path(path).stem] = {key: {**{k: sum(v) / len(v) for k, v in s.items()}, "rows": len(s["same_choice"])}
                                    for key, s in sorted(stats.items())}
        print(Path(path).stem, json.dumps({k: round(v, 4) for k, v in results[Path(path).stem]["all"].items()}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
