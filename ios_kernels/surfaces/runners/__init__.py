"""Runner registry. To add a backend: implement `Runner` (see base.py) and add it here."""

import importlib

from .base import Chunk, Message, Runner

# name -> (module, class). Imported lazily so optional backends (e.g. MLX) aren't required.
RUNNERS = {
    "kernel": ("surfaces.runners.kernel_runner", "KernelRunner"),  # our Metal kernels (kernels/)
    "fake": ("surfaces.runners.fake_runner", "FakeRunner"),        # canned echo replies, for tests
    "mlx": ("surfaces.runners.mlx_runner", "MLXRunner"),           # mlx-lm, in-process reference
}


def make_runner(name: str, **kwargs) -> Runner:
    if name not in RUNNERS:
        raise ValueError(f"unknown runner {name!r}; choose from {sorted(RUNNERS)}")
    module, cls = RUNNERS[name]
    return getattr(importlib.import_module(module), cls)(**kwargs)


__all__ = ["Chunk", "Message", "Runner", "RUNNERS", "make_runner"]
