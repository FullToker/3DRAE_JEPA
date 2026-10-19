import logging
import os
import random
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def load_config(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return path, config


def resolve_path(path: str | Path, root: str | Path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(root) / path


def setup_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank = local_rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def barrier():
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(output_dir: str | Path, rank: int):
    output_dir = Path(output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    logger = logging.getLogger("3drae.training")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    if rank == 0:
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(output_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def precision_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(precision: str):
    return torch.amp.GradScaler("cuda", enabled=precision == "fp16")


def wrap_ddp(model: nn.Module, device: torch.device, find_unused=False):
    if not dist.is_initialized():
        return model
    return DistributedDataParallel(
        model,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=find_unused,
    )


def unwrap_model(model: nn.Module):
    return model.module if isinstance(model, DistributedDataParallel) else model


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    ema_parameters = OrderedDict(ema_model.named_parameters())
    model_parameters = OrderedDict(model.named_parameters())
    if ema_parameters.keys() != model_parameters.keys():
        raise RuntimeError("EMA and model parameter names differ")
    for name, parameter in model_parameters.items():
        ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1 - decay)
    ema_buffers = OrderedDict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(buffer)


def reduce_mean(value: torch.Tensor):
    value = value.detach().float()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value.item()


def save_checkpoint(path: str | Path, state: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"step", "epoch", "model", "ema", "optimizer", "scheduler"}
    missing = required - state.keys()
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {sorted(missing)}")
    return state


def model_state_dict(model: nn.Module):
    return unwrap_model(model).state_dict()
