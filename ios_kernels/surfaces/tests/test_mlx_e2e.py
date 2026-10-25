"""The mlx-lm runner gives the same greedy text (cross-checks the Runner seam with a second
backend). Skipped unless `uv sync --extra mlx` was run and setup.py created the MLX model."""

import importlib.util
import threading

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from surfaces.openai_server.server import create_app
from surfaces.paths import MODELS_DIR

MLX_MODEL = MODELS_DIR / "llama1b_mlx_f16"
pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx_lm") is None or not MLX_MODEL.exists(),
                                reason=f"needs the mlx extra and {MLX_MODEL}")

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


@pytest.fixture
def fresh_runner():
    # Loaded on this (main) thread but never used here yet, exactly like the server: it loads
    # at startup, then the first request (which evaluates the lazily loaded weights) runs on a
    # worker thread.
    from surfaces.runners import make_runner

    r = make_runner("mlx")
    yield r
    r.close()


def test_mlx_from_other_threads(fresh_runner):
    # MLX GPU streams are per-thread; the server calls runners from worker threads.
    runner = fresh_runner
    result = {}
    def work():
        try:
            result["text"] = "".join(c.text for c in runner.stream(SKY, 32))
        except Exception as e:
            result["error"] = repr(e)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert result == {"text": SKY_REF}


def test_mlx_through_server(fresh_runner):
    runner = fresh_runner
    client = OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=TestClient(create_app(runner)))
    assert client.chat.completions.create(model="x", messages=SKY, max_tokens=32).choices[0].message.content == SKY_REF
    streamed = "".join(c.choices[0].delta.content or "" for c in
                       client.chat.completions.create(model="x", messages=SKY, max_tokens=32, stream=True) if c.choices)
    assert streamed == SKY_REF


def test_mlx_early_close_then_next_request(runner):
    g = runner.stream(SKY, max_tokens=64)
    next(g)
    g.close()
    assert "".join(c.text for c in runner.stream(SKY, max_tokens=32)) == SKY_REF
