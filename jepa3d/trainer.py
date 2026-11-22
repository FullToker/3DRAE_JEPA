"""GVJEPATrainer plus a hard-negative term.

    loss = InfoNCE (unchanged) + hard_weight * hard [+ slot_weight * slot + slot regression]
           [+ ground_weight * (ground choice + ground regression)]

hard: for each row that carries "negatives" (same question, wrong answer; see
jepa3d/hard_negatives.py), cross-entropy of the prediction against
[target, its own negatives] -- the multiple-choice objective. The negatives
go through the Y-encoder in the same forward as the targets.

Per log_every steps this trainer appends to <output_dir>/hardneg_log.jsonl:
hard_loss, hard_acc (share of rows whose target beats all its negatives) and
hard_rows (share of rows that have negatives); each validation adds the same
on the val manifest, by question type.

Row weights (batch["weight"], inverse answer frequency, jepa3d/balance.py):
both the InfoNCE rows and the hard rows are averaged with them; all 1 without
a weight table, which gives the unweighted losses.

numeric (models with a numeric head): smooth L1 between the head's output
for the row's kind and log(answer), on the rows with a numeric answer, weight
numeric_weight; numeric_loss and numeric_mra (VSI-Bench's mean relative
accuracy) go to the log.

slot (models with object tokens, jepa3d.slots.ObjectTokens): the model's own
loss on the output at the object tokens; slot_loss, slot_acc and slot_error_m
go to the same log. The regression term only trains the head that reports
metres (it sees a detached output).

ground (models with a GroundHead, jepa3d/ground.py): the head's choice and
regression terms on the names with a true position; ground_loss, ground_acc
and ground_error_m (median, metres) go to the same log.
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


MRA_THRESHOLDS = tuple(0.5 + 0.05 * i for i in range(10))


def mean_relative_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """VSI-Bench's metric per row: the share of thresholds c in 0.5 ... 0.95 with |pred - target| / target < 1 - c."""
    error = (pred - target).abs() / target
    return torch.stack([(error < 1 - c).float() for c in MRA_THRESHOLDS]).mean(0)


def weighted_infonce(pred, target, temperature: float, group_ids, weights) -> torch.Tensor:
    """fusion_gv's bidirectional_infonce with a weighted mean over the rows."""
    pred, target = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
    logits = (pred @ target.T) / temperature
    if group_ids is not None:
        same = group_ids[:, None] == group_ids[None, :]
        logits = logits.masked_fill(same & ~torch.eye(len(same), dtype=torch.bool, device=same.device), float("-inf"))
    labels = torch.arange(len(pred), device=pred.device)
    rows = 0.5 * (F.cross_entropy(logits, labels, reduction="none") + F.cross_entropy(logits.T, labels, reduction="none"))
    return (rows * weights).sum() / weights.sum()


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
    def __init__(self, *args, hard_weight: float = 1.0, slot_weight: float = 1.0, numeric_weight: float = 1.0,
                 ground_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hard_weight = hard_weight
        self.slot_weight = slot_weight
        self.numeric_weight = numeric_weight
        self.ground_weight = ground_weight
        self._ground = None   # the model's ground terms of the last forward
        self._numeric = None   # the model's numeric head output of the last forward
        self._slot = None   # the model's slot terms of the last forward
        self._steps = 0
        self._sums = defaultdict(float)

    def _forward(self, model, batch):
        """-> pred (B, D), target (B, D), hard logits (R, 1 + K) or None, row indices."""
        device = self.accelerator.device
        images = batch["images_vggt"].to(device, non_blocking=True)
        cams = batch["images_jepa"].to(device, non_blocking=True)
        negatives = batch["negatives"]
        flat = [n for ns in negatives for n in ns]
        out = model(images, cams, batch["query"], batch["target"] + flat, slots=batch.get("slots"),
                    ground=batch.get("ground"))
        self._ground = {k[7:]: v for k, v in out.items() if k.startswith("ground_")} or None
        B = len(negatives)
        self._slot = {k[5:]: v for k, v in out.items() if k.startswith("slot_")} or None
        self._numeric = out.get("numeric")
        pred, target = out["pred"], out["target"][:B]
        if not flat:
            return pred, target, None, None
        owner = torch.tensor([i for i, ns in enumerate(negatives) for _ in ns], device=device)
        slot = torch.tensor([k for ns in negatives for k in range(len(ns))], device=device)
        logits, rows = hard_logits(pred, target, out["target"][B:], owner, slot, self.temperature)
        return pred, target, logits, rows

    def _numeric_terms(self, batch, weights):
        """-> (loss, mean relative accuracy, rows with a numeric answer)."""
        device = self._numeric.device
        kind, value = batch["numeric"]["kind"].to(device), batch["numeric"]["value"].to(device)
        has = kind >= 0
        if not has.any():
            return self._numeric.sum() * 0.0, 0.0, 0
        guess = self._numeric.float()[has, kind[has]]
        rows = F.smooth_l1_loss(guess, value[has].log(), reduction="none")
        loss = (rows * weights[has]).sum() / weights[has].sum()
        return loss, mean_relative_accuracy(guess.detach().exp(), value[has]).mean().item(), int(has.sum())

    def _spar_step(self, batch: dict) -> tuple[torch.Tensor, float]:
        device = self.accelerator.device
        with self.accelerator.autocast():
            pred, target, logits, hard_rows = self._forward(self.model, batch)
            B = pred.shape[0]
            weights = batch["weight"].to(device) if "weight" in batch else torch.ones(B, device=device)
            if logits is None:
                hard, acc = pred.sum() * 0.0, 0.0
            else:
                per_row = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=device),
                                          reduction="none")
                hard = (per_row * weights[hard_rows]).sum() / weights[hard_rows].sum()
                acc = (logits.argmax(1) == 0).float().mean().item()
            # InfoNCE across ranks as in GVJEPATrainer._spar_step. The hard term
            # only uses this rank's rows, so it needs no `* world` correction.
            world = _world_size()
            all_weights = weights
            if world == 1:
                group_ids = group_ids_tensor(batch.get("group_id"), device)
            else:
                group_ids = dist_gather_ids(group_ids_tensor(
                    batch.get("group_id"), device, batch_size=B, rank=dist.get_rank(), force=True))
                pred, target = gather_with_grad(pred.float()), gather_with_grad(target.float())
                all_weights = gather_with_grad(weights)
            infonce = weighted_infonce(pred.float(), target.float(), self.temperature, group_ids, all_weights)
            loss = infonce * world + self.hard_weight * hard
            if self._numeric is not None:
                numeric, mra, count = self._numeric_terms(batch, weights)
                loss = loss + self.numeric_weight * numeric
                self._sums["numeric_loss"] += numeric.item() * count
                self._sums["numeric_mra"] += mra * count
                self._sums["numeric_rows"] += count
            if self._slot is not None:
                loss = loss + self.slot_weight * self._slot["loss"] + self._slot["regression"]
                for key in ("loss", "acc", "error_m"):
                    self._sums[f"slot_{key}"] += float(self._slot[key])
            if self._ground is not None:
                loss = loss + self.ground_weight * (self._ground["loss"] + self._ground["regression"])
                names = len(self._ground["errors"])
                self._sums["ground_names"] += names
                for key in ("loss", "acc", "error_m"):
                    self._sums[f"ground_{key}"] += float(self._ground[key]) * names
            loss = loss / self.grad_accum_steps

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
                       "hard_rows": self._sums["hard_rows"] / self._sums["rows"],
                       **{k: v / self.log_every for k, v in self._sums.items() if k.startswith("slot_")},
                       **({k: self._sums[k] / max(self._sums["ground_names"], 1)
                           for k in ("ground_loss", "ground_acc", "ground_error_m")} if self._ground is not None else {}),
                       **({k: self._sums[k] / max(self._sums["numeric_rows"], 1) for k in ("numeric_loss", "numeric_mra")}
                          if self._numeric is not None else {})}, "train")
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
            if self._slot is not None:
                for key in ("acc", "error_m"):
                    hits[f"val_slot_{key}"].append(float(self._slot[key]))
            if self._ground is not None and len(self._ground["errors"]):
                hits["val_ground_acc"] += self._ground["hits"].tolist()
                hits["val_ground_error_m"] += self._ground["errors"].tolist()
            if self._numeric is not None:
                _, mra, count = self._numeric_terms(batch, torch.ones(len(batch["query"]), device=device))
                hits["val_numeric_mra"] += [mra] * count
            if logits is None:
                continue
            for r, hit in zip(rows.tolist(), (logits.argmax(1) == 0).tolist()):
                hits["val_hard_acc"].append(hit)
                hits[f"val_hard_acc_type/{batch['rows'][r].get('question_type')}"].append(hit)
        model.train()
        self._log({"step": self._steps, **{k: sum(v) / len(v) for k, v in sorted(hits.items()) if v}}, "val")
        return sum(losses) / len(losses) if losses else float("nan")
