"""GVJEPATrainer plus a hard-negative term.

    loss = InfoNCE (unchanged) + hard_weight * hard

hard: for each row that carries "negatives" (same question, wrong answer; see
jepa3d/hard_negatives.py), cross-entropy of the prediction against
[target, its own negatives] -- the multiple-choice objective. The negatives
go through the Y-encoder in the same forward as the targets.

Per log_every steps this trainer appends to <output_dir>/hardneg_log.jsonl:
hard_loss, hard_acc (share of rows whose target beats all its negatives) and
hard_rows (share of rows that have negatives); each validation adds the same
on the val manifest, by question type.
"""

import json
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from jepa3d.paths import expand_paths  # noqa: F401  (sets sys.path for fusion_gv)
from fusion_gv.gvjepa_trainer import (
    GVJEPATrainer,
    _world_size,
    bidirectional_infonce,
    dist_gather_ids,
    gather_with_grad,
    group_ids_tensor,
)


def hard_logits(pred, target, neg, owner, slot, temperature: float):
    """pred, target (B, D); neg (M, D) with neg[m] belonging to row owner[m] at
    column slot[m]. -> logits (R, 1 + K) for the R rows that have negatives,
    target in column 0, missing negatives at -inf; and those rows' indices."""
    pred = F.normalize(pred.float(), dim=-1)
    pos = (pred * F.normalize(target.float(), dim=-1)).sum(-1) / temperature
    table = pred.new_full((pred.shape[0], int(slot.max()) + 1), float("-inf"))
    table[owner, slot] = (pred[owner] * F.normalize(neg.float(), dim=-1)).sum(-1) / temperature
    rows = torch.unique(owner)
    return torch.cat([pos[:, None], table], dim=1)[rows], rows


class HardNegTrainer(GVJEPATrainer):
    def __init__(self, *args, hard_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hard_weight = hard_weight
        self._steps = 0
        self._sums = defaultdict(float)

    def _forward(self, model, batch):
        """-> pred (B, D), target (B, D), hard logits (R, 1 + K) or None, row indices."""
        device = self.accelerator.device
        images = batch["images_vggt"].to(device, non_blocking=True)
        cams = batch["images_jepa"].to(device, non_blocking=True)
        negatives = batch["negatives"]
        flat = [n for ns in negatives for n in ns]
        out = model(images, cams, batch["query"], batch["target"] + flat)
        B = len(negatives)
        pred, target = out["pred"], out["target"][:B]
        if not flat:
            return pred, target, None, None
        owner = torch.tensor([i for i, ns in enumerate(negatives) for _ in ns], device=device)
        slot = torch.tensor([k for ns in negatives for k in range(len(ns))], device=device)
        logits, rows = hard_logits(pred, target, out["target"][B:], owner, slot, self.temperature)
        return pred, target, logits, rows

    def _spar_step(self, batch: dict) -> tuple[torch.Tensor, float]:
        device = self.accelerator.device
        with self.accelerator.autocast():
            pred, target, logits, _ = self._forward(self.model, batch)
            B = pred.shape[0]
            if logits is None:
                hard, acc = pred.sum() * 0.0, 0.0
            else:
                hard = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=device))
                acc = (logits.argmax(1) == 0).float().mean().item()
            # InfoNCE across ranks as in GVJEPATrainer._spar_step. The hard term
            # only uses this rank's rows, so it needs no `* world` correction.
            world = _world_size()
            if world == 1:
                group_ids = group_ids_tensor(batch.get("group_id"), device)
            else:
                group_ids = dist_gather_ids(group_ids_tensor(
                    batch.get("group_id"), device, batch_size=B, rank=dist.get_rank(), force=True))
                pred, target = gather_with_grad(pred.float()), gather_with_grad(target.float())
            infonce = bidirectional_infonce(pred, target, temperature=self.temperature, group_ids=group_ids)
            loss = (infonce * world + self.hard_weight * hard) / self.grad_accum_steps

        rows = 0 if logits is None else len(logits)
        self._sums["hard_loss"] += hard.item() * rows
        self._sums["hard_acc"] += acc * rows
        self._sums["hard_rows"] += rows
        self._sums["rows"] += B
        self._steps += 1
        if self._steps % self.log_every == 0:
            n = max(self._sums["hard_rows"], 1)
            self._log({"step": self._steps, "hard_loss": self._sums["hard_loss"] / n,
                       "hard_acc": self._sums["hard_acc"] / n,
                       "hard_rows": self._sums["hard_rows"] / self._sums["rows"]}, "train")
            self._sums.clear()
        return loss, infonce.item()

    def _log(self, entry: dict, prefix: str) -> None:
        """This rank's numbers only (rank 0), like the base trainer's train log.
        No key may be a path prefix of another: mlflow's file store keeps one file per metric."""
        if not self.accelerator.is_main_process:
            return
        with open(self.output_dir / "hardneg_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics(
                {f"{prefix}/{k}": v for k, v in entry.items() if k != "step"}, step=entry["step"])

    @torch.no_grad()
    def _run_val(self) -> float:
        """Val InfoNCE (returned, as the base trainer) and hard accuracy by question type."""
        device = self.accelerator.device
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        losses, hits = [], defaultdict(list)
        for i, batch in enumerate(self.val_loader):
            if i >= self.val_max_batches:
                break
            if len(batch["query"]) < 2:
                continue
            with self.accelerator.autocast():
                pred, target, logits, rows = self._forward(model, batch)
                losses.append(bidirectional_infonce(
                    pred, target, temperature=self.temperature,
                    group_ids=group_ids_tensor(batch.get("group_id"), device)).item())
            if logits is None:
                continue
            for r, hit in zip(rows.tolist(), (logits.argmax(1) == 0).tolist()):
                hits["val_hard_acc"].append(hit)
                hits[f"val_hard_acc_type/{batch['rows'][r].get('question_type')}"].append(hit)
        model.train()
        self._log({"step": self._steps, **{k: sum(v) / len(v) for k, v in sorted(hits.items())}}, "val")
        return sum(losses) / len(losses) if losses else float("nan")
