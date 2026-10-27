"""Repo roots and sys.path setup.

jepa3d glues two codebases together:
    3DRAE_ROOT   this repo; 3DRAE's own modules live under src/
    VJEPA_ROOT   the sibling vjepa2 checkout (fusion_gv trainer, predictor,
                 checkpoints). Named vjepa201 on the server, vjepa2 locally;
                 override with the VJEPA_ROOT environment variable.

Config strings may reference ${VJEPA_ROOT} / ${RAE_ROOT}; expand_paths()
resolves them.
"""

import os
import sys
from pathlib import Path

RAE_ROOT = Path(__file__).resolve().parents[1]


def _find_vjepa_root() -> Path:
    env = os.environ.get("VJEPA_ROOT")
    if env:
        return Path(env).resolve()
    for name in ("vjepa201", "vjepa2"):
        candidate = RAE_ROOT.parent / name
        if (candidate / "fusion_gv").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find the vjepa2 checkout next to this repo; set VJEPA_ROOT."
    )


VJEPA_ROOT = _find_vjepa_root()
os.environ.setdefault("VJEPA_ROOT", str(VJEPA_ROOT))
os.environ.setdefault("RAE_ROOT", str(RAE_ROOT))

# Both roots contain a top-level `src` package: vjepa code imports `src.*`,
# 3DRAE code imports the modules inside its src/ directly (`utils`,
# `stage1`, ...). So VJEPA_ROOT must come before RAE_ROOT, and RAE_ROOT/src
# provides 3DRAE's modules.
for _path in (RAE_ROOT, RAE_ROOT / "src", VJEPA_ROOT):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))


def expand_paths(value):
    """Recursively expand ${VAR} in every string of a parsed config."""
    if isinstance(value, dict):
        return {k: expand_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_paths(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
