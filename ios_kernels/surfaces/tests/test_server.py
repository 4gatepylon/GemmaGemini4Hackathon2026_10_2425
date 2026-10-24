"""Fast tests: OpenAI API surface + CLI client against the deterministic FakeRunner (no GPU)."""

import json

import pytest
from fastapi.testclient import TestClient
from openai import BadRequestError, OpenAI

from surfaces.cli_client import chat
from surfaces.openai_server.server import create_app
from surfaces.runners import make_runner
from surfaces.runners.fake_runner import FakeRunner

USER = [{"role": "user", "content": "the quick brown fox"}]


@pytest.fixture
def http():
    return TestClient(create_app(FakeRunner()))


@pytest.fixture
def client(http):
    # The real OpenAI SDK, routed into the in-process app.
    return OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=http)


def test_fake_runner_algorithm():
    chunks = list(FakeRunner().stream(USER, max_tokens=100))
    assert "".join(c.text for c in chunks) == "Echo: the quick brown fox"
    assert [c.text for c in chunks[:-1]] == ["Echo: ", "the ", "quick ", "brown ", "fox"]
    assert chunks[-1].finish_reason == "stop" and chunks[-1].completion_tokens == 5
    assert chunks[-1].prompt_tokens == 4


def test_make_runner_registry():
    assert isinstance(make_runner("fake"), FakeRunner)
    with pytest.raises(ValueError):
        make_runner("nope")


def test_health_and_models(http, client):
    assert http.get("/health").json() == {"status": "ok", "model": "fake-echo"}
    assert [m.id for m in client.models.list().data] == ["fake-echo"]


def test_non_streaming(client):
    r = client.chat.completions.create(model="fake-echo", messages=USER, temperature=0.7)
    assert r.object == "chat.completion"
    assert r.choices[0].message.role == "assistant"
    assert r.choices[0].message.content == "Echo: the quick brown fox"
    assert r.choices[0].finish_reason == "stop"
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens) == (4, 5, 9)


def test_max_tokens_truncates(client):
    r = client.chat.completions.create(model="x", messages=USER, max_tokens=2)
    assert r.choices[0].message.content == "Echo: the "
    assert r.choices[0].finish_reason == "length"
    r = client.chat.completions.create(model="x", messages=USER, max_completion_tokens=3)
    assert r.choices[0].message.content == "Echo: the quick "


def test_streaming_matches_non_streaming(client):
    chunks = list(client.chat.completions.create(model="x", messages=USER, stream=True,
                                                 stream_options={"include_usage": True}))
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Echo: the quick brown fox"
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 5
    assert all(c.id == chunks[0].id for c in chunks)


def test_sse_wire_format(http):
    r = http.post("/v1/chat/completions", json={"messages": USER, "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-sampling"].startswith("greedy")
    events = [line for line in r.text.split("\n\n") if line]
    assert all(e.startswith("data: ") for e in events)
    assert events[-1] == "data: [DONE]"
    assert json.loads(events[0][6:])["object"] == "chat.completion.chunk"


def test_multi_turn_uses_last_user_message(client):
    msgs = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Echo: first"}, {"role": "user", "content": "second"}]
    assert client.chat.completions.create(model="x", messages=msgs).choices[0].message.content == "Echo: second"


def test_content_parts_are_flattened(client):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "parts"}]}]
    assert client.chat.completions.create(model="x", messages=msgs).choices[0].message.content == "Echo: hello parts"


def test_bad_requests_are_400(http, client):
    with pytest.raises(BadRequestError):
        client.chat.completions.create(model="x", messages=[])
    r = http.post("/v1/chat/completions", json={"messages": USER, "max_tokens": 0})
    assert r.status_code == 400 and "error" in r.json()


def test_runner_value_error_is_400():
    class Failing(FakeRunner):
        def stream(self, messages, max_tokens):
            raise ValueError("prompt too long")
            yield

    http = TestClient(create_app(Failing()))
    for stream in (False, True):
        r = http.post("/v1/chat/completions", json={"messages": USER, "stream": stream})
        assert r.status_code == 400
        assert r.json()["error"]["message"] == "prompt too long"


def test_cli_one_shot(live_server, capsys):
    url = live_server(FakeRunner())
    chat.main(["--base-url", url, "--prompt", "hello there"])
    out = capsys.readouterr().out
    assert out.startswith("Echo: hello there\n[")
    assert "finish=stop" in out


def test_cli_repl(live_server, capsys, monkeypatch):
    url = live_server(FakeRunner())
    lines = iter(["hi", "/reset", "again", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    chat.main(["--base-url", url])
    out = capsys.readouterr().out
    assert "model: fake-echo" in out
    assert "Echo: hi" in out and "(conversation cleared)" in out and "Echo: again" in out


def test_client_disconnect_releases_runner(live_server):
    import httpx

    url = live_server(FakeRunner(delay=0.05))
    long = [{"role": "user", "content": "word " * 200}]  # ~10 s of streaming
    with httpx.stream("POST", f"{url}/chat/completions", json={"messages": long, "stream": True}) as r:
        next(r.iter_lines())  # read one event, then hang up
    # A new request must not wait on the abandoned stream (runner lock must be released).
    r = httpx.post(f"{url}/chat/completions", json={"messages": USER}, timeout=3)
    assert r.json()["choices"][0]["message"]["content"] == "Echo: the quick brown fox"
