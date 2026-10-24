# Surfaces: try the kernels by hand

An OpenAI-compatible server plus a terminal chat client on top of our kernels (or any
other backend). The design, extension points and limitations are in
[DESIGN_AND_LIMITATIONS.md](DESIGN_AND_LIMITATIONS.md).

All commands run from `ios_kernels/` (a uv project). Run `benchmarking/setup.sh` once first,
to get the weights.

```sh
uv sync                                   # first time only
uv run openai-server                      # newest kernel on http://127.0.0.1:8000/v1
uv run cli-chat                           # in another terminal: chat (/reset, /exit)
uv run cli-chat --prompt "Why is the sky blue?"   # one-shot
```

Other options:

```sh
uv run openai-server --kernel 20261024-1620   # a specific kernel folder (substring match)
uv run openai-server --runner fake            # canned echo replies: no GPU, no weights
uv sync --extra mlx && uv run openai-server --runner mlx   # mlx-lm backend, to compare by hand
uv run cli-chat --base-url http://127.0.0.1:8080/v1        # e.g. llama.cpp's llama-server

curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64}'

uv run pytest                              # API + CLI tests, and e2e kernel tests if weights exist
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (`stream`,
`stream_options.include_usage`, `max_tokens` / `max_completion_tokens`). Decoding is greedy.

| path | what |
|---|---|
| `openai_server/server.py` | FastAPI app (`create_app(runner)`) + `openai-server` CLI |
| `cli_client/chat.py` | `cli-chat`: streaming REPL / one-shot, prints ttft and tok/s |
| `runners/base.py` | the `Runner` interface (the seam for new backends) |
| `runners/kernel_runner.py` | drives a `kernels/*/` binary over the `--serve` protocol |
| `runners/hf_chat.py` | HF tokenizer, chat template, stop ids, incremental detokenizer |
| `runners/fake_runner.py`, `runners/mlx_runner.py` | test backend / mlx-lm backend |
| `paths.py` | repo layout + kernel discovery (newest by timestamp suffix) |
| `tests/` | see DESIGN_AND_LIMITATIONS.md |
