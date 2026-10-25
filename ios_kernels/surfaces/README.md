# Surfaces: try the kernels by hand

An OpenAI-compatible server plus a terminal chat client on top of our kernels (or any
other backend). The design, extension points and limitations are in
[DESIGN_AND_LIMITATIONS.md](DESIGN_AND_LIMITATIONS.md).

All commands run from `ios_kernels/`. Run `make setup` once first, to get the weights. The
`make` commands are shorthand for `ios_kernels_launcher.py` (see `make help`). The
underlying `uv run openai-server` / `uv run cli-chat` entry points work too.

```sh
make serve                                 # newest kernel on http://127.0.0.1:8000/v1
make chat                                  # in another terminal: chat (/reset, /exit)
make ask args="why is the sky blue"        # one-shot
make smoke                                 # start, ask one question, stop
```

Other options:

```sh
make serve args="--kernel 20261024-1620"   # a specific kernel folder (substring match)
make serve-fake                            # canned echo replies: no GPU, no weights
make serve-mlx                             # mlx-lm backend, to compare by hand
make serve-llamacpp                        # llama.cpp's llama-server on :8080 ...
make chat args="--port 8080"               # ... and chat with it

curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64}'

make test                                  # API + CLI tests, and e2e kernel/MLX tests if weights exist
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
