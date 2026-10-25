# Surfaces: design and limitations

> As of: main @ 028399a, 2026-10-25 13:25 PDT

**Purpose:** let a human try out LLM generation from our kernels by hand: chat in a
terminal, or point any OpenAI-compatible tool at a local endpoint. This is a *manual
testing* surface, not a production server. Every choice below favors simplicity and
being able to swap kernels or runtimes over throughput or features.

## Architecture

```
 cli_client/chat.py ──┐   (openai SDK)               any OpenAI client
 curl / Open WebUI  ──┼──── HTTP, OpenAI chat API ───────┐
                      ▼                                  │
          openai_server/server.py   (FastAPI, knows nothing about models)
                      │  Runner.stream(messages, max_tokens) -> Iterator[Chunk]
                      ▼
 runners/  ┌──────────────────────┬────────────────┬────────────────┐
           │ kernel_runner.py     │ mlx_runner.py  │ fake_runner.py │  <- add new runtimes here
           │ + hf_chat.py         │ (mlx-lm,       │ (canned echo,  │
           │ (tokenize/template)  │  in-process)   │  for tests)    │
           └──────────┬───────────┴────────────────┴────────────────┘
                      │  JSON lines over stdin/stdout ("--serve protocol")
                      ▼
 kernels/<model>/kernel_<name>_<YYYYMMDD-HHMM>/.build/llm   (Swift + Metal, token ids only)
```

There are two seams. They are the only places that need to stay stable:

1. **The `Runner` interface** (`runners/base.py`) sits between the HTTP/CLI surfaces and
   any inference backend. It works on messages in and text chunks out.
2. **The `--serve` protocol plus the `kernel.json` manifest** sits between Python and a
   kernel binary. It works on token ids in and token ids out.

## Design decisions

| Decision | Why |
|---|---|
| **OpenAI chat-completions API** | Works with the `openai` SDK, curl, Open WebUI, etc. The same CLI client also talks to llama.cpp's `llama-server`, so you can compare engines side by side by hand. |
| **FastAPI + uvicorn**, one `create_app(runner)` factory | Small and standard. The factory makes tests trivial: `TestClient(create_app(FakeRunner()))`. |
| **Runner = messages in, text chunks out** (not token ids) | Different backends tokenize differently (MLX and llama.cpp bring their own). Putting the seam at the message level means a new backend never touches the server. |
| **Tokenization in Python (HF `transformers`), not in Swift** | The kernel stays pure compute. The chat template and stop tokens come straight from the HF repo, so they match the reference exactly. A Swift BPE tokenizer is only needed on iOS later. |
| **Kernel = long-lived subprocess speaking JSON lines** | Weights and shaders load once (about 2–5 s), real token-by-token streaming, language-agnostic (any binary that speaks the protocol works), and a kernel crash can't take down Python. A new process per request would pay startup each time and couldn't stream. |
| **`kernel.json` manifest per kernel folder** | The folder describes itself (weights file, HF repo, build script, binary), so `KernelRunner` and the benchmarking scripts work on any kernel version with no code changes. |
| **Kernel folders are immutable snapshots, newest is the default** | Old versions stay runnable and benchmarkable. "Newest" is decided by the `_YYYYMMDD-HHMM` suffix. |
| **Greedy decoding only** | Deterministic output, so tests can compare against HF exact strings. Sampling happens on the GPU (argmax), which lets the engine pipeline command buffers. `temperature`/`top_p` are accepted and ignored (response header `x-sampling: greedy; ...`). |
| **One request at a time** (a lock in the server) | There's one KV cache and one decode stream. Batching isn't needed for manual testing. |
| **Every request re-prefills the whole conversation** | No prefix/KV reuse logic. Prefill runs at about 2000 tok/s, so a 2k-token chat costs about 1 s. |
| **Stream window of 3 command buffers** in `--serve` | Keeps the GPU busy (CPU encoding of token *t+1* overlaps GPU work on *t*) while bounding wasted work after a stop token to 2 steps. |
| **`ClosingStreamingResponse`** | Starlette doesn't close a sync generator when the client disconnects, so it would sit there until garbage collection, holding the runner lock and leaving the kernel mid-response. Every later request would then hang (found in manual testing, covered by `test_client_disconnect_*`). We close it explicitly. |
| **The kernel drains abandoned responses** | If a stream stops early, `KernelRunner` reads and discards the rest of that response so the protocol stays in sync. This is simpler than a cancel message. |
| **Kernel exits on stdin EOF** | If the Python server dies, even from `kill -9`, the kernel process exits on its own, so no orphaned GPU processes. |
| **MLX work runs on one thread owned by `MLXRunner`** | MLX GPU streams are per-thread, but the server calls runners from worker threads. The first request would fail with `There is no Stream(gpu, 1) in current thread` (found via `make serve-mlx`, covered by `test_mlx_from_other_threads` / `test_mlx_through_server`). Loading and generation are queued to a dedicated thread. |
| **`FakeRunner` echoes deterministically** | Reply = `"Echo: " + last user message`, one word per "token", `max_tokens` truncates. The API and CLI can be tested in about 1 s with no GPU or weights. |

## How to swap things

### A new kernel version (most common)
1. `make new-kernel args=<name>` copies the newest folder to
   `kernels/<model>/kernel_<name>_<YYYYMMDD-HHMM>/` and fills in its README lineage header
   (see `../CLAUDE.md`). Change the copy.
2. Keep the contract (below). If the weight format changes, point `kernel.json`
   `weights` at a new file name and update its `convert_weights.py`.
3. `make setup` converts the weights and builds the binary. `make serve` now uses it
   automatically (newest). Select an older one with `args="--kernel <substring>"`.
4. Check it: `make test` (the e2e tests run against the newest kernel), `make reference`,
   `make bench` (or `make bench-ours` for quick numbers).

**Kernel folder contract:**
- `kernel.json`: `{"hf_repo", "weights" (file name in $DATA_DIR/models), "build", "binary", "convert"}`
- `build` script: an executable Python script (shebang, `chmod +x`) that produces `binary`. `KernelRunner` rebuilds when `*.swift` is newer than the binary.
- `convert` script: `--repo <hf_repo> --out <weights path>`
- `binary` accepts `--weights PATH --serve --max-ctx N` and speaks the protocol below.
  For `benchmarking/` it also needs `--bench-pp N --bench-tg N --reps R` and
  `--prompt-ids a,b,c --gen N --dump-logits PATH`.

### `--serve` protocol (stdin/stdout, one JSON object per line; stderr is free-form logs)
```
kernel → {"ready": true, "max_ctx": 4096, "vocab": 128256, "device": "Apple M3 Pro"}   (once, after warmup)
python → {"ids": [128000, ...], "max_tokens": 256, "stop_ids": [128001, 128008, 128009]}
kernel → {"token": 791}                                    (one per generated token; stop token not sent)
kernel → {"done": true, "finish_reason": "stop" | "length", "prompt_tokens": 46,
          "completion_tokens": 32, "prefill_s": 0.03, "decode_s": 0.61}
kernel → {"error": "..."}                                  (bad request; the kernel keeps running)
```
Requests are strictly sequential. Every request starts from an empty KV cache
(positions 0..n). EOF on stdin makes the kernel exit.

### A different runtime (MLX, llama.cpp, Core ML, a remote API, ...)
1. Add `runners/<name>_runner.py` with a class implementing `Runner.stream()` (see
   `mlx_runner.py`, about 30 lines). It must yield text `Chunk`s and end with one
   chunk that has `finish_reason` and token counts set. Raise `ValueError` for bad
   requests (these become HTTP 400).
2. Register it in `runners/__init__.py` `RUNNERS`, and add its CLI kwargs in
   `server.py` `main()`.
3. Imports inside the runner are lazy, so optional dependencies (e.g. `mlx-lm`, extra `mlx`)
   are only needed when that runner is used.

Alternatively, for runtimes that already serve the OpenAI API (llama.cpp `llama-server`,
Ollama, mlx-lm's server), there's nothing to write: `make chat args="--base-url <their /v1>"` (`make serve-llamacpp`
starts llama.cpp's on port 8080).

## Limitations (known, accepted for a manual-testing tool)

- **Greedy only.** No temperature/top-p/top-k, repetition penalty, seeds, `n > 1`,
  logprobs, `stop` strings, tools/function calling, JSON mode, or images. Unknown request
  fields are ignored without error.
- **One model, one request at a time.** Concurrent requests queue on a lock. `model` in the
  request is ignored (whatever the server was started with answers).
- **No KV/prefix cache reuse across turns.** Each request re-prefills from scratch. Context is
  capped at `max_ctx` (default 4096, which the attention kernel's threadgroup memory limits).
  Longer prompts get HTTP 400.
- **Abandoned streams still finish on the GPU.** After a client disconnects, the kernel
  generates the rest of that response (up to `max_tokens`) before the next request starts, so
  the next request can wait up to `max_tokens` × ~20 ms. The fix would be a cancel message in
  the protocol.
- **Up to 2 wasted decode steps** after a stop token (the stream window). Invisible to clients.
- **Tokenizer and chat template come from HF `transformers`** (the Llama-3.2 template injects
  a "Cutting Knowledge Date / Today Date" system header; Gemma 4's thinking mode stays off, and
  the MLX runner turns it off too so both backends see the same prompt). The first start downloads nothing
  if the HF cache is warm, otherwise it needs network access.
- **Model-specific bits** live in the kernel (head dims, RoPE variant, Gemma 4's per-layer
  embeddings and KV sharing, ...) and the manifest. The surfaces themselves are model-agnostic.
- **Localhost only, no auth** (binds `127.0.0.1` by default, API key ignored). Don't expose it.
- **End-to-end HTTP streaming is about 48 tok/s vs 52–53 raw**, because of per-token Python
  detokenization and JSON/SSE overhead. Fine for humans, so use `benchmarking/bench.py` for
  performance numbers, never the server.
- **macOS + Apple GPU only** for the kernel runner. The fake runner and API tests run anywhere.

## Tests (`make test`, about 20 s; `make test-fast`, about 2 s, needs no GPU)

| file | needs | what |
|---|---|---|
| `tests/test_server.py` | nothing | FakeRunner algorithm, registry, `/health`, `/v1/models`, non-stream + stream through the real `openai` SDK, SSE wire format, `max_tokens`/`max_completion_tokens`, multi-turn, content parts, 400s, client-disconnect regression, CLI one-shot + REPL against a live uvicorn |
| `tests/test_kernel_e2e.py` | macOS + weights (skipped otherwise) | newest kernel: greedy output == HF transformers string, EOS stop, early close keeps the protocol in sync, multi-byte (emoji) detokenization, stream == non-stream via the server, too-long prompt → 400, CLI against the kernel, client disconnect mid-stream |
| `tests/test_mlx_e2e.py` | mlx extra + MLX model (skipped otherwise) | MLX greedy output == HF string, stop-token usage convention, use from other threads and through the server, early close |
| `../tests/test_launcher.py` | make (some tests) | Makefile and launcher list the same commands, `make help` (overall and per command), `args=` forwarded verbatim, unknown flags rejected, `new-kernel`, `--data-dir` reaches the tools, `smoke` with the fake runner |
