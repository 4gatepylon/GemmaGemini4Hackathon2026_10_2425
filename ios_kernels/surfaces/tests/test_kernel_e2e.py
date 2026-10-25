"""End-to-end: the real Metal kernel behind the real server. Skipped if the weights aren't set up
(run ios_kernels/benchmarking/setup.py). Expected texts come from HF transformers greedy output."""

import json
import platform

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from surfaces.cli_client import chat
from surfaces.openai_server.server import create_app
from surfaces.paths import MODELS_DIR, resolve_kernel

kernel_dir = resolve_kernel()
weights = MODELS_DIR / json.loads((kernel_dir / "kernel.json").read_text())["weights"]
pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or not weights.exists(),
                                reason=f"needs macOS + {weights}")

SKY = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
# HF transformers fp32 greedy continuation (see benchmarking/reference.py).
SKY_REF = ("The sky appears blue because of a phenomenon called Rayleigh scattering, in which shorter "
           "wavelengths of light, such as blue and violet, are scattered more than longer wavelengths")


@pytest.fixture(scope="module")
def runner():
    from surfaces.runners.kernel_runner import KernelRunner

    r = KernelRunner()
    yield r
    r.close()


@pytest.fixture(scope="module")
def client(runner):
    return OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=TestClient(create_app(runner)))


def test_greedy_matches_hf_reference(runner):
    chunks = list(runner.stream(SKY, max_tokens=32))
    assert "".join(c.text for c in chunks) == SKY_REF
    assert (chunks[-1].finish_reason, chunks[-1].completion_tokens) == ("length", 32)


def test_stops_on_eos(runner):
    chunks = list(runner.stream([{"role": "user", "content": "What is 2+2? Answer with a number only."}], 16))
    assert "".join(c.text for c in chunks).strip() == "4"
    assert chunks[-1].finish_reason == "stop"


def test_early_close_keeps_protocol_in_sync(runner):
    g = runner.stream(SKY, max_tokens=64)
    next(g)
    g.close()  # abandon mid-stream; next request must still be correct
    assert "".join(c.text for c in runner.stream(SKY, max_tokens=32)) == SKY_REF


def test_multibyte_tokens_decode_cleanly(runner):
    text = "".join(c.text for c in runner.stream(
        [{"role": "user", "content": "Reply with a single emoji of a cat and nothing else."}], 16))
    assert "�" not in text and text.strip()


def test_server_stream_equals_non_stream(client):
    full = client.chat.completions.create(model="x", messages=SKY, max_tokens=32)
    assert full.choices[0].message.content == SKY_REF
    streamed = "".join(c.choices[0].delta.content or "" for c in
                       client.chat.completions.create(model="x", messages=SKY, max_tokens=32, stream=True) if c.choices)
    assert streamed == SKY_REF


def test_prompt_too_long_is_400(client):
    from openai import BadRequestError

    with pytest.raises(BadRequestError, match="context"):
        client.chat.completions.create(model="x", messages=[{"role": "user", "content": "hello " * 5000}])


def test_cli_against_kernel(runner, live_server, capsys):
    chat.main(["--base-url", live_server(runner), "--prompt", "What is 2+2? Answer with a number only."])
    out = capsys.readouterr().out
    assert out.startswith("4\n[") and "finish=stop" in out


def test_client_disconnect_mid_stream(runner, live_server):
    import httpx

    url = live_server(runner)
    story = [{"role": "user", "content": "Write a long story about a dragon."}]
    with httpx.stream("POST", f"{url}/chat/completions",
                      json={"messages": story, "stream": True, "max_tokens": 200}) as r:
        next(r.iter_lines())
    # The runner drains the abandoned response (<= 200 tokens, ~4 s) and then serves this one.
    r = httpx.post(f"{url}/chat/completions", timeout=30,
                   json={"messages": [{"role": "user", "content": "What is 2+2? Answer with a number only."}]})
    assert r.json()["choices"][0]["message"]["content"].strip() == "4"
