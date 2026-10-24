"""Filesystem layout shared by the surfaces, runners, and benchmarking scripts."""

import os
from pathlib import Path

IOS_KERNELS = Path(__file__).resolve().parents[1]          # .../ios_kernels
REPO_ROOT = IOS_KERNELS.parent
KERNELS_DIR = IOS_KERNELS / "kernels"
# Large artifacts (weights, llama.cpp checkout). Gitignored; override with $DATA_DIR.
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / ".context"))
MODELS_DIR = DATA_DIR / "models"


def list_kernels() -> list[Path]:
    """Kernel folders, oldest first. Folders are named kernel_<name>_<YYYYMMDD-HHMM>."""
    dirs = [d for d in KERNELS_DIR.glob("kernel_*") if (d / "kernel.json").exists()]
    return sorted(dirs, key=lambda d: d.name.rsplit("_", 1)[-1])


def resolve_kernel(name: str | None = None) -> Path:
    """None -> newest kernel. Otherwise a path, a folder name, or a unique substring of one."""
    kernels = list_kernels()
    if name is None:
        if not kernels:
            raise FileNotFoundError(f"no kernels found in {KERNELS_DIR}")
        return kernels[-1]
    if Path(name).is_dir():
        return Path(name).resolve()
    matches = [k for k in kernels if name in k.name]
    if len(matches) != 1:
        raise ValueError(f"kernel {name!r} matched {[m.name for m in matches]}; available: {[k.name for k in kernels]}")
    return matches[0]
