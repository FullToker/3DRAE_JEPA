#!/usr/bin/env python3
"""Pictures and per-name predictions for the grounding labels and the GroundHead.

Without --checkpoint: the first holdout rows with at least two labelled names per subset, each drawn
as its views with the true centres projected into them (a look at the labels).
With --checkpoint: the same rows with the model's regressed centres added (real scene tokens, and the
scene tokens of another row), and for every labelled name of the holdout the true and the regressed
centre -> <output-dir>/predictions.json
    {"real" | "shuffled": [{"type", "subset", "name", "true": [x, y, z], "pred": [x, y, z], "hit": 0 | 1}]}
(metres, frame of the row's first view: x right, y down, z forward).

Usage (from the repo root):
    python jepa3d/tools/show_ground.py --config jepa3d/configs/ground_v1.yaml \
        --manifest experiments/ground_v1/data/holdout_f8.jsonl --output-dir experiments/ground_v1/results/figs \
        [--checkpoint experiments/ground_v1/ckpt/final.pt]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
from jepa3d.data import build_loader
from jepa3d.train import load_config
from jepa3d.x_encoder import unpack_cams, view_counts

COLOURS = ("tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:pink")


def names_of(row: dict) -> list:
    return [row["slots"][e["slot"]]["name"] for e in row["ground"]]


def project(points, K, w2c, scale: float):
    """points (n, 3) metres in the first view's frame -> pixels (V, n, 2) and whether each is inside the view."""
    hom = torch.cat([points / scale, torch.ones(len(points), 1)], dim=1)
    cam = torch.einsum("vij,nj->vni", w2c, hom)[..., :3]
    uv = torch.einsum("vij,vnj->vni", K, cam)
    uv = uv[..., :2] / uv[..., 2:].clamp_min(1e-6)
    return uv, cam[..., 2] > 0.05 / scale


def draw(path: Path, batch: dict, r: int, guesses: dict) -> None:
    """One row: its views with every labelled name's true centre (ring) and the guesses (markers)."""
    ground, row = batch["ground"], batch["rows"][r]
    mine = (ground["row"] == r).nonzero().flatten()
    scale = float(ground["scale"][mine[0]])
    true = ground["target"][mine] * scale
    V = int(view_counts(batch["images_jepa"])[r])
    K, w2c = (x[r, :V].float() for x in unpack_cams(batch["images_jepa"]))
    images = batch["images_vggt"][r, :V]
    height, width = images.shape[-2:]
    names = names_of(row)
    fig, axes = plt.subplots(2, (V + 1) // 2, figsize=(4.2 * ((V + 1) // 2), 6.6))
    for v, ax in enumerate(axes.flat):
        ax.axis("off")
        if v >= V:
            continue
        ax.imshow(images[v].permute(1, 2, 0).numpy())
        for label, points, marker in [("true", true, "o")] + [(k, g, m) for k, (g, m) in guesses.items()]:
            uv, front = project(points, K, w2c, scale)
            for n in range(len(names)):
                u, w = uv[v, n].tolist()
                if not front[v, n] or not (0 <= u < width and 0 <= w < height):
                    continue
                colour = COLOURS[n % len(COLOURS)]
                if label == "true":
                    ax.scatter([u], [w], s=140, facecolors="none", edgecolors=colour, linewidths=2.5)
                    ax.text(u + 6, w - 6, names[n], color="white", fontsize=8,
                            bbox={"facecolor": colour, "alpha": 0.8, "pad": 1, "edgecolor": "none"})
                else:
                    ax.scatter([u], [w], s=90, marker=marker, color=colour, edgecolors="black", linewidths=0.8)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    legend = "ring: true centre" + "".join(f"   {m}: {k}" for k, (_, m) in guesses.items())
    fig.suptitle(f"{row['subset']}/{row['scene']}  [{row['question_type']}]  {row['query'].splitlines()[0][:150]}\n{legend}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--per-subset", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["val_num_workers"] = args.num_workers
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loader = build_loader([args.manifest], cfg, args.batch_size, train=False)

    model = None
    if args.checkpoint:
        from fusion_gv.infer_gvjepa import _load_checkpoint
        from jepa3d.model import build_model_from_config
        from jepa3d.tools.blind_check import set_mode
        model = build_model_from_config(cfg)
        _load_checkpoint(model, args.checkpoint)
        model.to("cuda").eval()

    def guess(batch, mode):
        """-> (N, 3) regressed centres for the batch's labelled names (nan for a name the model dropped), hits (N,)"""
        set_mode(model, mode)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.predict_embedding(batch["images_vggt"].cuda(), batch["images_jepa"].cuda(), batch["query"],
                                    ground=batch["ground"])
        del model._pool_visual
        m = model.ground_metrics
        out = torch.full((len(batch["ground"]["row"]), 3), float("nan"))
        hits = torch.zeros(len(out))
        out[m["kept"].cpu()] = m["guess"].float().cpu()
        hits[m["kept"].cpu()] = m["hits"].float().cpu()
        return out, hits

    drawn, predictions = Counter(), {"real": [], "shuffled": []}
    for batch in loader:
        ground = batch["ground"]
        if ground is None:
            continue
        guesses = {mode: guess(batch, mode) for mode in predictions} if model else {}
        owner = ground["row"].tolist()
        index = Counter()
        for n, r in enumerate(owner):
            row = batch["rows"][r]
            name = names_of(row)[index[r]]
            index[r] += 1
            for mode, (points, hits) in guesses.items():
                if torch.isnan(points[n]).any():
                    continue
                predictions[mode].append({
                    "type": f"{row['source']}/{row['question_type']}", "subset": row["subset"], "name": name,
                    "true": [round(x, 3) for x in (ground["target"][n] * ground["scale"][n]).tolist()],
                    "pred": [round(x, 3) for x in points[n].tolist()], "hit": int(hits[n])})
        for r, row in enumerate(batch["rows"]):
            if owner.count(r) >= 2 and drawn[row["subset"]] < args.per_subset:
                mine = (ground["row"] == r).nonzero().flatten()
                shown = {"real tokens": (guesses["real"][0][mine].nan_to_num(), "X"),
                         "another scene's tokens": (guesses["shuffled"][0][mine].nan_to_num(), "P")} if model else {}
                kind = "pred" if model else "labels"
                draw(output / f"{kind}_{row['subset']}_{drawn[row['subset']]}.png", batch, r, shown)
                drawn[row["subset"]] += 1
        if not model and all(drawn[s] >= args.per_subset for s in ("scannet", "scannetpp")):
            break
    if model:
        (output / "predictions.json").write_text(json.dumps(predictions))
        print({mode: len(v) for mode, v in predictions.items()})
    print(f"[done] {dict(drawn)} -> {output}")


if __name__ == "__main__":
    main()
