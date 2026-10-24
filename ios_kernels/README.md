# iOS Kernels
This is a sub-project of the broader effort. The goal is to make a more performant small-model inference runtime for iOS and/or mac. It is part of my effort learning how to develop performance software/kernels for ML inference (as my broader inference engineering learning effort). 

## Layout

```
ios_kernels/                 a uv project (pyproject.toml): `uv sync` once
  PLAN.md                    design, decisions, roadmap, checklist
  kernels/                   one folder per model, then one immutable folder per engine version; newest is the default
    unsloth__Llama-3.2-1B-Instruct/
      kernel_metal_f16_20261024-1620/   hand-written Metal + Swift, Llama-3.2-1B FP16
  benchmarking/              setup.sh (llama.cpp + weights), bench.sh, reference.py (correctness)
  surfaces/                  try it by hand: OpenAI-compatible server + CLI chat (Python)
    DESIGN_AND_LIMITATIONS.md   how surfaces, runners and kernels plug together, and what they don't do
```

Large artifacts (weights, GGUF, MLX model, llama.cpp checkout) live in the gitignored
`<repo>/.context/` (override with `$DATA_DIR`).

## Quickstart

```sh
cd ios_kernels
uv sync
benchmarking/setup.sh             # one-time: builds llama.cpp, converts weights (~7.5 GB), builds the kernel
uv run openai-server              # serves the newest kernel at http://127.0.0.1:8000/v1
uv run cli-chat                   # chat with it (in another terminal)
uv run pytest                     # API/CLI tests + end-to-end kernel tests
benchmarking/bench.sh             # ours vs llama.cpp vs MLX
uv run --extra bench python benchmarking/reference.py   # correctness vs HF transformers
```

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
  time-to-first-token. Use `bench.sh` for performance numbers, not the server.
