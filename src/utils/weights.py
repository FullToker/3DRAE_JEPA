from pathlib import Path

from safetensors.torch import load_file


def _require_safetensors(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"Release models only load .safetensors weights: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_tensor_dict(path: str | Path):
    return load_file(_require_safetensors(path), device="cpu")


def load_safetensors(module, path: str | Path, strict: bool = True):
    return module.load_state_dict(load_tensor_dict(path), strict=strict)
