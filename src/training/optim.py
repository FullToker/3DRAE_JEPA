import math
from typing import Any, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _as_tuple(values: Any, length: int = 2) -> tuple:
    if isinstance(values, (list, tuple)):
        if len(values) != length:
            raise ValueError(
                f"Expected sequence of length {length}, got {len(values)}."
            )
        return tuple(float(v) for v in values)
    return tuple(float(values) for _ in range(length))


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    training_cfg: dict[str, Any],
) -> tuple[Optimizer, str]:
    opt_cfg = training_cfg["optimizer"]
    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type != "adamw":
        raise ValueError(
            f"Unsupported optimizer '{opt_type}'. Only AdamW is available."
        )

    base_lr = float(opt_cfg["lr"])
    betas = _as_tuple(opt_cfg.get("betas", (0.9, 0.95)))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    eps = float(opt_cfg.get("eps", 1e-8))

    optimizer = torch.optim.AdamW(
        parameters,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        fused=False,
        foreach=True,
    )

    optim_msg = (
        f"Optimizer: AdamW with lr={base_lr}, betas={betas}, "
        f"weight_decay={weight_decay}, eps={eps}"
    )
    return optimizer, optim_msg


def build_scheduler(
    optimizer: Optimizer,
    training_cfg: dict[str, Any],
) -> tuple[LambdaLR, str]:
    sched_cfg = training_cfg["scheduler"]
    schedule_type = sched_cfg.get("type", "linear").lower()

    base_lr = float(sched_cfg["base_lr"])
    final_lr = float(sched_cfg["final_lr"])
    if base_lr <= 0 or final_lr < 0:
        raise ValueError("Scheduler learning rates must be non-negative")
    final_ratio = final_lr / base_lr

    if "warmup_steps" not in sched_cfg or "decay_end_steps" not in sched_cfg:
        raise KeyError("Scheduler requires warmup_steps and decay_end_steps")
    warmup_steps = int(sched_cfg["warmup_steps"])
    decay_end_steps = int(sched_cfg["decay_end_steps"])
    warmup_from_zero = bool(sched_cfg.get("warmup_from_zero", False))

    if warmup_steps < 0 or decay_end_steps < warmup_steps:
        raise ValueError("Invalid scheduler step range")
    total_decay_steps = max(decay_end_steps - warmup_steps, 1)

    for group in optimizer.param_groups:
        group["lr"] = base_lr

    if schedule_type == "linear":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_from_zero:
                    return (step + 1) / warmup_steps
                return 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay_steps
            return 1.0 - (1.0 - final_ratio) * progress

    elif schedule_type == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_from_zero:
                    return (step + 1) / warmup_steps
                return 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine
    else:
        raise ValueError(
            f"Unsupported scheduler '{schedule_type}'. "
            "Choose from ['linear', 'cosine']."
        )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    sched_msg = (
        f"Scheduler: {schedule_type} with warmup_steps={warmup_steps}, "
        f"decay_end_steps={decay_end_steps}, final_lr={final_lr}, "
        f"base_lr={base_lr}, warmup_from_zero={warmup_from_zero}"
    )
    return scheduler, sched_msg


# This code is modified from the GitHub repository of KellerJordan:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5):
    """Approximate a matrix's orthogonal factor with Newton-Schulz iterations."""
    if matrix.ndim < 2:
        raise ValueError("Newton-Schulz orthogonalization expects a matrix.")

    a, b, c = (3.4445, -4.7750, 2.0315)
    result = matrix
    if matrix.size(-2) > matrix.size(-1):
        result = result.mT
    result = result / (result.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        product = result @ result.mT
        update = b * product + c * product @ product
        result = a * result + update @ result

    if matrix.size(-2) > matrix.size(-1):
        result = result.mT
    return result


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Arguments:
        muon_params: Parameters optimized with Muon.
        lr: Learning rate.
        momentum: Momentum for Muon updates.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        adamw_params: Parameters optimized with the AdamW fallback.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params or [])
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            if p.ndim < 2:
                raise ValueError("Muon parameters must have at least two dimensions")
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @staticmethod
    def adjust_lr_for_muon(lr, param_shape):
        rows, columns = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(rows, columns))
        return lr * adjusted_ratio

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = g.bfloat16()
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u.view(p.shape), alpha=-adjusted_lr)

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def get_muon_optimizer(
    model,
    training_cfg,
):
    opt_cfg = training_cfg["optimizer"]
    if opt_cfg.get("type", "muon").lower() != "muon":
        raise ValueError("Stage 1 optimizer must be Muon")
    base_lr = float(opt_cfg["lr"])
    adamw_betas = _as_tuple(opt_cfg.get("betas", (0.9, 0.95)))
    adamw_eps = float(opt_cfg.get("eps", 1e-8))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    adamw_param_names = opt_cfg.get("adamw_param_names", [])
    momentum = opt_cfg.get("momentum", 0.95)

    def is_adamw_param(name):
        return any(key in name for key in adamw_param_names)

    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_muon = p.ndim >= 2 and not is_adamw_param(name)
        if is_muon:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizer = Muon(
        lr=base_lr,
        wd=weight_decay,
        muon_params=muon_params,
        momentum=momentum,
        adamw_params=adamw_params,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
    )
    optim_msg = (
        f"Optimizer: Muon with lr={base_lr}, adamw_betas={adamw_betas}, "
        f"weight_decay={weight_decay}, adamw_eps={adamw_eps}"
    )

    return optimizer, optim_msg
