"""The mlx-lm runner gives the same greedy text (cross-checks the Runner seam with a second
backend). Skipped unless `uv sync --extra mlx` was run and setup.py created the MLX model."""

import pytest

from surfaces.paths import MODELS_DIR

pytest.importorskip("mlx_lm")
MLX_MODEL = MODELS_DIR / "llama1b_mlx_f16"
pytestmark = pytest.mark.skipif(not MLX_MODEL.exists(), reason=f"needs {MLX_MODEL}")

from .test_kernel_e2e import SKY, SKY_REF  # noqa: E402


@pytest.fixture(scope="module")
def runner():
    from surfaces.runners import make_runner

    return make_runner("mlx")


def test_mlx_greedy_matches_hf_reference(runner):
    chunks = list(runner.stream(SKY, max_tokens=32))
    assert "".join(c.text for c in chunks) == SKY_REF
    assert (chunks[-1].finish_reason, chunks[-1].completion_tokens) == ("length", 32)


def test_mlx_stop_usage_matches_kernel_convention(runner):
    chunks = list(runner.stream([{"role": "user", "content": "What is 2+2? Answer with a number only."}], 16))
    assert "".join(c.text for c in chunks).strip() == "4"
    assert (chunks[-1].finish_reason, chunks[-1].completion_tokens) == ("stop", 1)
