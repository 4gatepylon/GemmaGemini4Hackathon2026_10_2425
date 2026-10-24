# PLAN: hand-written Metal forward pass vs llama.cpp

**Goal (MVP):** get a working, from-scratch LLM inference runtime prototype on
Apple GPUs (custom Metal compute kernels, Swift host so it ports to iOS) as
fast as possible, verify it's numerically correct, and benchmark it against
llama.cpp (Metal backend) on this Mac.

**Long-term goal:** improve this runtime step by step until it's *faster than
llama.cpp* (and other standard runtimes) for our workload. The MVP is
structured with that in mind (see "Structured for iteration" below).

Machine: Apple M3 Pro (18-core GPU, ~150 GB/s memory bandwidth), 36 GB,
macOS 15 (Darwin 24.6), Xcode 26.3, Swift 6.2.

All of our code lives in `ios_kernels/`. Large artifacts (weights, GGUFs,
llama.cpp checkout and build) live in the gitignored `.context/` directory
and are never committed.

---

## Key decisions (please push back on any of these)

| Decision | Choice | Why |
|---|---|---|
| Model | **Llama-3.2-1B-Instruct** (`unsloth/Llama-3.2-1B-Instruct`, already in the HF cache) | Plain Llama architecture (no sliding window, soft-capping, or QK-norm), first-class llama.cpp support, small enough to iterate quickly. **Gemma 3 1B is the natural next model** for the Gemma hackathon (needs: sliding-window layers, QK-norm, pre+post norms, GELU, 262k vocab). |
| Precision | **FP16 weights, FP32 accumulation and residual stream**, FP16 KV cache, for both engines | Apples-to-apples: both engines move the same bytes per token. Quantized formats (Q8_0 / Q4) are a later step, and that's where most of the speedup on a bandwidth-bound device will come from. |
| Host language | **Swift + Metal.framework**, one CLI built with `swiftc` (no Xcode project) | Same APIs as iOS, so the engine can drop into an iOS app later. No extra dependencies. |
| Shader compilation | **Runtime compile** of MSL source (`device.makeLibrary(source:)`), with tunables injected as `#define`s | The offline `metal` compiler isn't installed here. Runtime compile also makes parameter sweeps trivial (no rebuild). |
| Weight format | Custom flat `.bin`: JSON header plus FP16 tensors on 256-byte boundaries, written by `convert_weights.py` | No GGUF/safetensors parser needed in Swift. The file is `mmap`ed and wrapped in one zero-copy `MTLBuffer` (unified memory). Layout is ours to change (fusing, interleaving, quantizing) without touching a parser. |
| Tokenizer | Python (HF `tokenizers`) turns text into ids, the Swift engine works on ids only | Keeps the Swift side focused on compute. A Swift BPE tokenizer is a separate, later task. |
| Baseline | llama.cpp at HEAD (`d327bf1`, 2026-10-24), built from source with Metal, `llama-bench` on an F16 GGUF converted from **the same safetensors** | Standard tool, standard metrics (`pp512`, `tg128`). |
| Extra reference | MLX / `mlx-lm` (already installed) | A third data point, cheap to add. |

---

## Dependencies

Already on this machine (nothing new installed system-wide):
- Xcode 26.3 / Swift 6.2 / Metal.framework (runtime shader compiler)
- cmake (Homebrew), used only to build llama.cpp
- Python 3.13 (miniconda) with `torch`, `safetensors`, `numpy`, `transformers`,
  `tokenizers`, `huggingface_hub`, `gguf`, `mlx`, `mlx-lm`
- HF cache: `unsloth/Llama-3.2-1B-Instruct` `model.safetensors` + tokenizer

Downloaded for this task:
- `config.json`, `generation_config.json`, `tokenizer_config.json` for
  `unsloth/Llama-3.2-1B-Instruct` (public repo, not gated)
- llama.cpp source: `git clone --depth 1 https://github.com/ggml-org/llama.cpp`
  into `.context/deps/llama.cpp`, built with
  `cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF && cmake --build build --target llama-bench llama-cli`

Generated artifacts (in `.context/models/`, ~2.5 GB each):
- `llama1b_f16.bin` (our format, from `ios_kernels/forward/convert_weights.py`)
- `llama1b_f16.gguf` (llama.cpp format, from `convert_hf_to_gguf.py --outtype f16`)

Optional downloads that would help (not blocking, tell me if you want them):
- **Metal Toolchain** (`xcodebuild -downloadComponent MetalToolchain`, about 1 GB):
  enables offline `.metallib` builds (what you'd ship on iOS) and better
  shader debugging/profiling in Xcode's GPU capture.
- **Gemma 3 1B** (`google/gemma-3-1b-it`, gated, needs HF license acceptance +
  token): the next model to port.

---

## Kernel design (decode)

Decode (1 token at a time) is **memory-bandwidth bound**. Per token we read
about 2.47 GB of FP16 weights (16 layers × 60.8M params + the 0.53 GB tied LM
head). At about 150 GB/s that gives a ceiling of roughly **60 tok/s** on this
M3 Pro, so the kernels are built to stream weights at full bandwidth.

Each token is one command buffer, one compute encoder, with serial dispatches:

```
embed(tokens[step]) -> x (fp32 residual)
for each of 16 layers:
  rmsnorm_f32         x -> xn
  matvec_f16          Wqkv·xn -> qkv          (Wq|Wk|Wv fused into [3072,2048] at convert time)
  rope_kv             rotate q,k (Llama-3 scaled RoPE); write k,v to the fp16 KV cache
  attention           per Q head: scores, softmax, ·V  (GQA 32 Q heads -> 8 KV heads)
  matvec_f16_add      x += Wo·attn
  rmsnorm_f32         x -> xn
  matvec_gate_up_silu h = silu(Wg·xn) * (Wu·xn)   (fused, gate/up never hit memory)
  matvec_f16_add      x += Wdown·h
rmsnorm_f32 -> matvec_f16 (LM head = tied embedding, [128256, 2048]) -> argmax -> tokens[step+1]
```

That's about 7 dispatches per layer, roughly 115 per token.

**matvec_f16** (the hot kernel): each SIMD-group owns `MV_NR` rows; each lane
streams 16 bytes (8 halfs) per row per iteration, so 512 contiguous bytes per
row per SIMD-group step, accumulated in fp32 and reduced with `simd_sum`.
`MV_NR` and `MV_NSG` (SIMD-groups per threadgroup) are `#define` tunables.

**GPU-resident sampling / pipelining:** `argmax` writes the next token id into a
GPU buffer that the next step's `embed` reads. So the CPU never has to wait
for token *t* before encoding token *t+1*: command buffers are queued
back-to-back and CPU encode overhead overlaps GPU work. EOS is checked after
the fact.

**Attention:** one threadgroup per (Q head, query row). Scores live in
threadgroup memory, which caps the MVP at `max_ctx` = 4096. That's fine for
benchmarks. Flash-decoding (split-K over the sequence) is the fix for long
contexts.

### Prefill (planned, second milestone)
- MVP fallback: run prompt tokens through the decode path one at a time.
  Correct but slow (about the decode speed, ~60 tok/s).
- Real path: tiled GEMM with `simdgroup_matrix` (8×8
  `simdgroup_multiply_accumulate`, fp16 inputs, fp32 accumulators) for all
  projections, plus `attention` launched over a (heads × T) grid (it already
  supports T > 1 with causal masking). That gives a meaningful `pp512` number.

---

## Correctness plan
- Reference: HF `transformers` (torch, fp32 on CPU) on the same token ids.
- Compare last-position logits (max abs diff, top-5 agreement) and check that
  greedy generation of 32+ tokens matches token for token.
- If something is off: a debug flag dumps intermediate buffers per layer for
  diffing against the reference.

## Benchmark plan
- Metrics, matching `llama-bench`:
  - `tg128`: 128 decode steps from an empty cache (tokens/s)
  - `pp512`: 512-token prompt processing (tokens/s), once the GEMM path lands
- 5 repetitions after warmup, mean ± stddev, nothing else heavy running.
- llama.cpp: `llama-bench -m llama1b_f16.gguf -p 512 -n 128 -r 5 -ngl 99`
  (Metal, flash-attn default).
- Ours: wall clock over the pipelined command buffers, plus GPU time from
  `MTLCommandBuffer.gpuStartTime/gpuEndTime`.
- MLX: `mlx_lm` with the fp16 model, as a third reference.
- Also report **effective bandwidth (GB/s)** = bytes/token × tok/s, against
  the ~150 GB/s peak, to show how much headroom each engine leaves.
- Results and reproduction commands go in `ios_kernels/README.md`.

---

## Structured for iteration (how we beat llama.cpp later)

llama.cpp is general-purpose: a graph builder over ggml ops, generic kernels
across many shapes and types, and per-op dispatch. Our edge comes from
specializing to *one* model on *one* device class. The prototype is set up
for that:

1. **Kernels in one `.metal` file with `#define` tunables**: sweep
   `MV_NR`/`MV_NSG`/etc. from the CLI (`--define MV_NR=8`) with no rebuild.
2. **Weight layout owned by our converter**: we can fuse matrices (done for
   QKV), pre-interleave or tile weights for the kernel's access pattern, or
   quantize to custom block formats without writing a parser.
3. **Explicit, hand-scheduled forward pass** (no graph runtime): a flat list
   of dispatches in `main.swift` makes cross-op fusions easy, e.g.
   rmsnorm→matvec, rope→attention, a fused residual+norm epilogue. Fewer
   dispatches and fewer round trips through memory.
4. **Built-in benchmark and correctness harness**: every optimization is
   checked against the HF reference and timed with the same harness as
   llama.cpp.

Likely optimization roadmap after the MVP, roughly in order of expected payoff:
- Quantized weights (Q8_0 then 4-bit) with dequant fused into the matvec.
  Fewer bytes per token is the only way past the FP16 bandwidth ceiling.
- Fuse rmsnorm into the next matvec; fuse rope+KV write into the QKV matvec epilogue.
- Indirect command buffers / `MTLComputeCommandEncoder` reuse to cut CPU encode cost.
- Flash-decoding attention for long contexts.
- `simdgroup_matrix` GEMM tuning for prefill; speculative decoding (batched verify).
- Port to Gemma 3 1B / 270M; build an iOS app target.

---

## Layout

```
ios_kernels/
  PLAN.md                 <- this file
  README.md               <- results + how to run
  forward/
    convert_weights.py    <- safetensors -> flat fp16 .bin + JSON header
    kernels.metal         <- all Metal kernels (compiled at runtime)
    main.swift            <- engine: load, encode forward pass, generate, bench
    reference.py          <- HF transformers reference / correctness check
    build.sh              <- swiftc build (-> forward/.build/llm)
    bench.sh              <- runs ours, llama.cpp, MLX
    setup.sh              <- clones/builds llama.cpp, converts weights (into .context/)
```

---

## Checklist

- [x] Survey machine and tools (M3 Pro, Xcode 26.3, Swift 6.2, cmake, MLX, torch; HF cache has Llama-3.2-1B safetensors)
- [x] Build llama.cpp (Metal) from source -> `.context/deps/llama.cpp/build/bin/llama-bench`
- [x] Convert Llama-3.2-1B -> F16 GGUF for llama.cpp
- [x] `convert_weights.py`: safetensors -> our flat fp16 format (fused QKV), max|w| checked against fp16 range
- [x] Metal kernels (first draft): embed, rmsnorm, matvec(+residual), fused gate/up SwiGLU, rope+KV write, attention, argmax
- [x] Swift host: mmap weights, pipelines, encode one token per command buffer, pipelined greedy loop
- [x] `reference.py` (HF logits / greedy tokens). `run.py` (text in/out) dropped for now: reference.py covers tokenization
- [x] Correctness vs HF transformers: logits max|diff| 0.011, top-5 identical, 32/32 greedy tokens match (decode and GEMM prefill paths)
- [x] Benchmark decode vs llama.cpp `tg128` (+ MLX): **52.1 vs 49.5–52.2 tok/s (parity)**
- [x] Quick tuning sweep of matvec tunables: all configs land at 52–53 tok/s, so we're at the bandwidth wall for FP16
- [x] Prefill GEMM (`simdgroup_matrix`) + benchmark vs `pp512`: **2050 vs 2650 tok/s (77% of llama.cpp)**
- [x] `setup.sh` / `bench.sh` / `build.sh` for one-command reproduction
- [x] Write results + reproduction steps in README.md
- [ ] (next) Prefill: flash-attention style kernel (GQA-shared K/V tiles), GEMM double-buffering / larger tiles
- [ ] (next) Q8_0 weights + kernel, benchmark vs llama.cpp Q8_0

## Status (2026-10-24)

MVP done. On Llama-3.2-1B FP16, M3 Pro:

| engine | pp512 tok/s | tg128 tok/s |
|---|---:|---:|
| ours | ~2050 | ~52.1 |
| llama.cpp d327bf1 | ~2650 | 49.5–52.2 |
| MLX 0.29 | ~2400 | ~51.5 |

Details are in README.md. Prefill gap breakdown: ~35 ms of 250 ms is naive
attention, the rest is GEMM (~4.6 TFLOP/s).

## Roadblocks / risks (I'll update this as I go)

- **No offline Metal compiler.** Worked around with runtime compilation. Not blocking.
- **No ready-made Llama GGUF locally.** Converted it ourselves with llama.cpp's script. Resolved.
- **unsloth snapshot was missing `config.json`.** Fetched from the public HF repo. Resolved.
- BF16 -> FP16 overflow: checked during conversion, no tensor exceeds the fp16 range. Resolved.
- Nothing is currently blocking.
