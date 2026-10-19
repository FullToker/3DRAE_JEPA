#!/usr/bin/env python3
import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export tensor-only release weights from a trusted training checkpoint."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--state-key",
        help="Dot-separated mapping key to select, for example 'ema'.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Keep only keys with this prefix and strip it from exported names.",
    )
    parser.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="Drop keys with this prefix. May be specified more than once.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def select_state(checkpoint, state_key: str | None):
    state = checkpoint
    if state_key:
        for key in state_key.split("."):
            if not isinstance(state, Mapping) or key not in state:
                raise KeyError(f"Missing state key: {state_key}")
            state = state[key]
    if not isinstance(state, Mapping):
        raise TypeError("Selected checkpoint state is not a mapping")
    return state


def collect_tensors(state, prefix: str, drop_prefixes: list[str]):
    tensors = {}
    skipped = []
    for key, value in state.items():
        if prefix and not key.startswith(prefix):
            continue
        output_key = key[len(prefix) :] if prefix else key
        if any(output_key.startswith(item) for item in drop_prefixes):
            continue
        if isinstance(value, torch.Tensor):
            tensors[output_key] = value.detach().cpu().contiguous()
        else:
            skipped.append(key)
    if not tensors:
        raise ValueError("No tensors matched the requested checkpoint selection")
    if skipped:
        print(f"Skipped {len(skipped)} non-tensor entries")
    return dict(sorted(tensors.items()))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.output.suffix != ".safetensors":
        raise ValueError("Output path must end in .safetensors")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    state = select_state(load_checkpoint(args.source), args.state_key)
    tensors = collect_tensors(state, args.prefix, args.drop_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_file": args.source.name,
        "state_key": args.state_key or "",
        "stripped_prefix": args.prefix,
        "dropped_prefixes": ",".join(args.drop_prefix),
    }
    save_file(tensors, args.output, metadata=metadata)

    with safe_open(args.output, framework="pt", device="cpu") as exported:
        if list(exported.keys()) != list(tensors.keys()):
            raise RuntimeError("Exported safetensors keys do not match source tensors")
        for key, tensor in tensors.items():
            if tuple(exported.get_slice(key).get_shape()) != tuple(tensor.shape):
                raise RuntimeError(f"Shape mismatch after export: {key}")
    parameter_count = sum(tensor.numel() for tensor in tensors.values())
    print(f"Exported {len(tensors)} tensors / {parameter_count:,} values")
    print(f"Output: {args.output}")
    print(f"SHA256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
