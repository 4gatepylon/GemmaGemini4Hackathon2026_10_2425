# iOS Kernels
This is a sub-project of the broader effort. The goal is to make a more performant small-model inference runtime for iOS and/or mac. It is part of my effort learning how to develop performance software/kernels for ML inference (as my broader inference engineering learning effort). 

## Layout

```
ios_kernels/                 a uv project (pyproject.toml)
  Makefile                   shorthand for ios_kernels_launcher.py (`make help`)
  ios_kernels_launcher.py    one entry point for every command below
  PLAN.md                    design, decisions, roadmap, checklist
  CLAUDE.md                  how to add a kernel (naming, lineage header), rules, checks
  kernels/                   one folder per model, then one immutable folder per engine version; newest is the default
    google__gemma-4-E2B-it/            the default model (MODEL in surfaces/paths.py)
      kernel_metal_f16_20261025-1225/   hand-written Metal + Swift, Gemma 4 E2B (text) FP16
    unsloth__Llama-3.2-1B-Instruct/
      kernel_metal_f16_20261024-1620/   hand-written Metal + Swift, Llama-3.2-1B FP16
  benchmarking/              setup.py (llama.cpp + weights), bench.py, reference.py (correctness)
  utils/                     helper scripts (`uv run utils/kernel_lineage.py`: kernel lineage tree per model)
  surfaces/                  try it by hand: OpenAI-compatible server + CLI chat (Python)
    DESIGN_AND_LIMITATIONS.md   how surfaces, runners and kernels plug together, and what they don't do
```

Large artifacts (weights, GGUF, MLX model, llama.cpp checkout) live in the gitignored
`<repo>/.context/` (override with `--data-dir`).

## Quickstart

Needs macOS on Apple silicon, Xcode (Swift), cmake and [uv](https://docs.astral.sh/uv/).

```sh
cd ios_kernels                    # or run make -C ios_kernels <command> from the repo root
make help                         # every command; `make help args=<command>` for its flags
make setup                        # one-time: builds llama.cpp, converts weights (~19 GB), builds the kernel
make serve                        # serves the newest kernel at http://127.0.0.1:8000/v1
make chat                         # chat with it (in another terminal)
make test                         # all tests (API/CLI + end-to-end kernel and MLX)
make bench                        # ours vs llama.cpp vs MLX
make reference                    # correctness vs HF transformers
```

## Make targets

The `Makefile` is only shorthand: `make <command> args="--flags"` runs
`uv run ios_kernels_launcher.py <command> --flags`, so the two are interchangeable and all
logic (and `--help`) lives in the launcher. For flags with quoting, calling the launcher
directly is easiest: `uv run ios_kernels_launcher.py ask "Why is the sky blue?"`.

| command | what | useful flags |
|---|---|---|
| `help` | list commands (default target) | `args=<command>` for one command's flags |
| `setup` | one-time: llama.cpp, weights (our .bin, GGUF, MLX), kernel build (`benchmarking/setup.py`) | `--kernel` |
| `build` | rebuild the kernel binary | `--kernel` |
| `serve` | OpenAI-compatible server, newest kernel by default | `--kernel` `--runner {kernel,fake,mlx}` `--port` `--host` |
| `serve-fake` / `serve-mlx` | the same server with the echo / mlx-lm backend | `--port` |
| `serve-llamacpp` | llama.cpp's `llama-server` on the same model (port 8080), for side-by-side chats | `--port` |
| `chat` | interactive terminal chat (`/reset`, `/exit`) | `--port` / `--base-url`, `--max-tokens`, `--system` |
| `ask` | one-shot question: `make ask args="why is the sky blue"` | same as `chat` |
| `smoke` | start a server, ask one question, shut down (~5 s) | `--runner` `--kernel` `--prompt` |
| `test` / `test-fast` | all tests / only the no-GPU ones (~2 s) | extra args go to pytest |
| `bench` | ours vs llama.cpp vs MLX (`benchmarking/bench.py`) | `--kernel` `--pp` `--tg` `--reps` |
| `bench-ours` | our engine only, for quick before/after numbers | same, plus `--define MV_NR=2` |
| `reference` | logits + greedy tokens vs HF transformers | `--kernel` |
| `kernels` | list kernel versions, mark the default (newest) | |
| `new-kernel` | copy the newest kernel to `kernels/<model>/kernel_<name>_<YYYYMMDD-HHMM>/`, fill in its lineage header | `<name>` `--from` |
| `clean` | remove build outputs and caches (never the weights) | |

Every command also takes `--data-dir` (default `<repo>/.context`).

```sh
make serve args="--kernel 20261024 --port 9000"
make chat args="--port 9000"
make serve-llamacpp                        # then: make chat args="--port 8080"
make bench-ours args="--define MV_NR=2 --reps 3"
make new-kernel args=metal_llama_q8        # start the next kernel version
```

## Results: Gemma 4 E2B-it (text), FP16, Apple M3 Pro (18-core GPU, 36 GB)

| engine | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| **ours** (`kernel_metal_f16_20261025-1225`) | 1122–1125 | **27.7–28.4** |
| llama.cpp `6050b46` (Metal, `llama-bench -ngl 99`) | 1115–1127 | 25.2–26.3 |
| MLX / mlx-lm (`mlx_lm benchmark`, HF bf16 weights) | **2581–2708** | 26.7* |

Two separate runs of 5 reps each. \*MLX's generation number is measured after the 512-token
prompt, the others from an empty cache.

- **Decode**: ahead of llama.cpp and MLX. 28 tok/s × 4.56 GB/token ≈ 128 GB/s, about 85% of
  the M3 Pro's ~150 GB/s (the per-layer-embedding table is only gathered, so it doesn't count).
- **Prefill**: on par with llama.cpp, well behind MLX (see the kernel's `PLAN.md`).
- **Correctness**: vs HF transformers fp32, last-token logits max |diff| 0.013 (logit
  range ±30), same top-5, 32/32 greedy tokens identical. Also checked end to end through
  the server (`surfaces/tests/test_kernel_e2e.py`), and mlx-lm gives the same text.

## Results: Llama-3.2-1B-Instruct, FP16, Apple M3 Pro (18-core GPU, 36 GB)

| engine | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| **ours** (`kernel_metal_f16_20261024-1620`) | 2044–2062 | **52.1–53.4** |
| llama.cpp `d327bf1` (Metal, `llama-bench -ngl 99`) | **2639–2655** | 49.5–52.2 |
| MLX / mlx-lm (`mlx_lm benchmark`) | 2365–2471 | 51.0–52.0* |

Ranges cover several separate runs of 5 reps each. \*MLX's generation number is measured
after the 512-token prompt, the others from an empty cache.

- **Decode**: on par with llama.cpp. 52 tok/s × 2.47 GB/token ≈ 129 GB/s, about 86% of
  the M3 Pro's ~150 GB/s. FP16 decode is bandwidth-bound, so the next big gain needs fewer
  bytes per token (quantization), not better FP16 kernels.
- **Prefill**: about 77% of llama.cpp. Roughly 35 ms of the 250 ms pp512 is naive
  attention; the rest is the `simdgroup_matrix` GEMM at ~4.6 TFLOP/s (no double-buffering yet).
- **Correctness**: vs HF transformers fp32, last-token logits max |diff| 0.011 (logit
  range ±27), same top-5, 32/32 greedy tokens identical. Also checked end to end through
  the server (`surfaces/tests/test_kernel_e2e.py`).
- **Through the server** (manual testing), streaming runs at about 48 tok/s with ~50 ms
  time-to-first-token. Use `bench.py` for performance numbers, not the server.
