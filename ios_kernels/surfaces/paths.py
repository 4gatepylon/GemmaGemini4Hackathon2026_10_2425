"""Filesystem layout shared by the surfaces, runners, and benchmarking scripts."""

import json
import os
from pathlib import Path

IOS_KERNELS = Path(__file__).resolve().parents[1]  # .../ios_kernels
REPO_ROOT = IOS_KERNELS.parent
KERNELS_ROOT = IOS_KERNELS / "kernels"  # kernels/<model>/kernel_*, <model> = HF repo id with "/" -> "__"
# The model every command uses by default. Kernels of other models are reachable with --kernel.
MODEL = "unsloth__Llama-3.2-1B-Instruct"
KERNELS_DIR = KERNELS_ROOT / MODEL
# Large artifacts (weights, llama.cpp checkout). Gitignored; override with $DATA_DIR.
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / ".context"))
MODELS_DIR = DATA_DIR / "models"
LLAMA_DIR = DATA_DIR / "deps" / "llama.cpp"  # llama.cpp checkout + build (benchmark baseline)
# Per model: baseline files in MODELS_DIR, made by benchmarking/setup.py from the same HF snapshot.
BASELINES = {
    "unsloth__Llama-3.2-1B-Instruct": {"gguf": "llama1b_f16.gguf", "mlx": "llama1b_mlx_f16"},
}


def list_kernels(model: str = MODEL) -> list[Path]:
    """Kernel folders of one model, oldest first. Folders are named kernel_<name>_<YYYYMMDD-HHMM>."""
    dirs = [d for d in (KERNELS_ROOT / model).glob("kernel_*") if (d / "kernel.json").exists()]
    return sorted(dirs, key=lambda d: d.name.rsplit("_", 1)[-1])


def resolve_kernel(name: str | None = None) -> Path:
    """None -> newest kernel of MODEL. Otherwise a path, a folder name, or a unique substring of one
    (searched in MODEL first, then in every model folder)."""
    kernels = list_kernels()
    if name is None:
        if not kernels:
            raise FileNotFoundError(f"no kernels found in {KERNELS_DIR}")
        return kernels[-1]
    if Path(name).is_dir():
        return Path(name).resolve()
    matches = [k for k in kernels if name in k.name]
    if not matches:
        kernels = [k for m in sorted(KERNELS_ROOT.glob("*__*")) for k in list_kernels(m.name)]
        matches = [k for k in kernels if name in k.name]
    if len(matches) != 1:
        raise ValueError(f"kernel {name!r} matched {[m.name for m in matches]}; available: {[k.name for k in kernels]}")
    return matches[0]


def load_manifest(kernel: Path) -> dict:
    """A kernel folder's kernel.json."""
    return json.loads((kernel / "kernel.json").read_text())


def baseline(kind: str, kernel: Path | None = None) -> Path:
    """A baseline file ("gguf" or "mlx") for the model of `kernel` (default: MODEL)."""
    model = kernel.parent.name if kernel else MODEL
    return MODELS_DIR / BASELINES[model][kind]
