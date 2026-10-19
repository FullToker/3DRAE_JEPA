#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify release checkpoint files against checkpoints/manifest.json."
    )
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_safetensors(path, entry):
    if "tensors" not in entry and "values" not in entry:
        return
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        values = sum(math.prod(handle.get_slice(key).get_shape()) for key in keys)
    if len(keys) != entry["tensors"]:
        raise ValueError(f"Tensor count mismatch for {path}: {len(keys)}")
    if values != entry["values"]:
        raise ValueError(f"Value count mismatch for {path}: {values}")


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else root / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"File size mismatch for {path}")
        for companion in entry.get("required_companions", []):
            companion_path = path.parent / companion
            if not companion_path.is_file():
                raise FileNotFoundError(companion_path)
        verify_safetensors(path, entry)
        if not args.skip_sha256 and sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")
        print(f"OK {entry['path']}")
    print(f"Verified {len(manifest['files'])} checkpoint files")


if __name__ == "__main__":
    main()
