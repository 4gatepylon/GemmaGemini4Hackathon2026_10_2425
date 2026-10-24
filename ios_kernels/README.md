# iOS Kernels
This is a sub-project of the broader effort. The goal is to make a more performant small-model inference runtime for iOS and/or mac. It is part of my effort learning how to develop performance software/kernels for ML inference (as my broader inference engineering learning effort). 

## `forward/`: hand-written Metal forward pass (MVP)

A from-scratch Llama forward pass. Custom Metal compute kernels
(`kernels.metal`) are driven by a small Swift host (`main.swift`), with no ML
framework dependencies, so it can move to iOS. See [PLAN.md](PLAN.md) for the
design, decisions, and roadmap.

### Results: Llama-3.2-1B-Instruct, FP16, Apple M3 Pro (18-core GPU, 36 GB)

| engine | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| **ours** (hand-written Metal) | 2044–2057 | **52.1–52.2** |
| llama.cpp `d327bf1` (Metal, `llama-bench -ngl 99`) | **2639–2655** | 49.5–52.2 |
| MLX / mlx-lm 0.29.1 (`mlx_lm benchmark`) | 2365–2471 | 51.0–52.0* |

Ranges cover 3 separate runs of 5 reps each. \*MLX's generation number is
measured after the 512-token prompt, the others from an empty cache.

- **Decode**: on par with llama.cpp. 52 tok/s × 2.47 GB/token ≈ 129 GB/s,
  about 86% of the M3 Pro's ~150 GB/s. FP16 decode is bandwidth-bound, so the
  next big gain needs fewer bytes per token (quantization), not better FP16 kernels.
- **Prefill**: about 77% of llama.cpp. Roughly 35 ms of the 250 ms pp512 is
  naive attention (one threadgroup per head × query); the rest is the
  `simdgroup_matrix` GEMM at ~4.6 TFLOP/s (no double-buffering yet).
- **Correctness** (`reference.py`, vs HF transformers fp32): last-token logits
  max |diff| 0.011 (logit range ±27), same top-5, and 32/32 greedy tokens
  identical.

### Reproduce

```sh
ios_kernels/forward/setup.sh    # builds llama.cpp, converts weights into .context/ (about 7.5 GB)
ios_kernels/forward/bench.sh    # ours vs llama.cpp vs MLX
python3 ios_kernels/forward/reference.py --weights .context/models/llama1b_f16.bin  # correctness
ios_kernels/forward/.build/llm --weights .context/models/llama1b_f16.bin \
    --bench-tg 128 --define MV_NSG=4 --define MV_NR=2   # sweep kernel tunables without rebuilding
```

### Files

| file | what |
|---|---|
| `convert_weights.py` | HF safetensors -> flat mmap-able FP16 file (fused QKV, JSON header) |
| `kernels.metal` | all kernels: embed, rmsnorm, matvec, fused SwiGLU matvec, RoPE + KV write, attention, argmax, GEMM, fused SwiGLU GEMM |
| `main.swift` | engine: mmaps weights into one MTLBuffer, encodes the forward pass, GPU-side greedy sampling with pipelined command buffers, benchmark mode |
| `reference.py` | correctness check against HF transformers |
| `build.sh` / `setup.sh` / `bench.sh` | build, one-time setup, benchmark |
