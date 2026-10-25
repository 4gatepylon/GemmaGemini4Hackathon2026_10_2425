# PLAN: Gemma 4 E2B (text) on hand-written Metal, FP16

> NOTE: this was written by: Claude Opus 5.5 (`claude-opus-5-5[1m]`) in Claude Code, running
> inside Conductor (workspace `hanoi`), against commit `6880596` (`main`), 2026-10-24 18:15 PDT.
> Status section added by the same agent after implementing it, on top of `a81d5d8`
> (PR #28, multi-model support), 2026-10-25 12:55 PDT.

**Goal:** a from-scratch forward pass for **`google/gemma-4-E2B-it`**, the smallest Gemma 4
model, in the same style as `kernels/unsloth__Llama-3.2-1B-Instruct/kernel_metal_f16_20261024-1620`
(hand-written Metal kernels, Swift host, FP16 weights, GPU-side greedy sampling), served by the
existing surfaces and benchmarked against llama.cpp and MLX. Once it lands, the surfaces,
benchmarks and tests default to this kernel.

Scope: **text only**. The checkpoint (`Gemma4ForConditionalGeneration`) also has vision and
audio towers; the converter drops them.

Machine: Apple M3 Pro (18-core GPU, ~150 GB/s), 36 GB, macOS 15.

---

## The model (from `config.json` `text_config` + HF `modeling_gemma4.py`)

| | value |
|---|---|
| layers | 35: pattern `[sliding ×4, full]` ×7 (full = layers 4, 9, ..., 34) |
| hidden | 1536, RMSNorm eps 1e-6, embeddings scaled by `sqrt(1536)` |
| attention | 8 Q heads, **1 KV head** (MQA), `scaling = 1.0` (QK-norm instead) |
| sliding layers | head_dim 256, window 512 (`kv > q - 512`), RoPE θ=1e4 over all 256 dims |
| full layers | head_dim **512**, "proportional" RoPE θ=1e6: only the first 64 of 256 rotation pairs rotate (`inv_freq[i] = θ^(-2i/512)` for i < 64, 0 after) |
| QK/V norm | `q_norm`, `k_norm` (RMSNorm with weight), `v_norm` (RMSNorm, **no** weight), per head |
| KV sharing | only layers 0-14 compute K/V. Layers 15-34 reuse the K/V of the **last** earlier layer of the same type: layer 13 (sliding) or layer 14 (full) |
| MLP | GeGLU (`gelu_tanh(gate) * up`), intermediate 6144 in layers 0-14, **12288** ("double wide") in 15-34 |
| norms per layer | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `post_per_layer_input_norm` (sandwich norms) |
| per-layer embeddings (PLE) | `embed_tokens_per_layer` [262144, 35×256] (×`sqrt(256)`), plus `per_layer_model_projection` [35×256, 1536] of the scaled embedding (×`1536^-0.5`, then RMSNorm over each 256-slice); the two are summed and ×`2^-0.5` |
| per layer, after the MLP | `h += norm(W_proj · (gelu_tanh(W_gate · h) * ple[layer]))` with W_gate [256, 1536], W_proj [1536, 256]; then `h *= layer_scalar` (a per-layer scalar from the checkpoint) |
| head | tied embedding [262144, 1536], then `30 * tanh(logits / 30)` (softcap, does not change argmax) |
| chat | Gemma chat template from the HF tokenizer; stop ids from `generation_config.json` (1, 106, 50) |

Per decoded token the engine streams about **2.28 B params = 4.56 GB** in FP16 (MLPs 1.56 B,
LM head 0.40 B, attention 0.28 B, PLE projections 0.04 B). At ~150 GB/s that is a ceiling of
about **33 tok/s**. The 2.35 B-param PLE table (4.7 GB) is only gathered (35×256 values per
token), so it costs disk and memory, not bandwidth.

---

## Kernel design (decode)

Same structure as the Llama kernel: one command buffer per token, serial dispatches in one
encoder, fp32 residual stream, fp16 weights and KV cache, argmax on the GPU feeding the next
token's embed, command buffers pipelined back to back.

```
embed(tok) * sqrt(D) -> x
ple_embed(tok) (gather 35×256, ×16)                     -> ple_tok
matvec W_ple_proj · x, ×D^-0.5, RMSNorm per 256-slice   -> ple = (that + ple_tok) * 2^-0.5
for each of 35 layers:
  rmsnorm                     x -> xn
  matvec_f16                  Wq|Wk|Wv · xn (fused at convert time; layers 15-34: Wq only)
  qk_norm_rope_kv             per-head RMSNorm of q, k (weights), v (no weight); RoPE on q, k
                              (sliding: full rotation; full: first 64 pairs); write k, v to cache
  attention                   8 Q heads over 1 KV head; sliding layers only see the last 512
                              positions; layers 15-34 read the cache of layer 13 / 14
  matvec_f16                  Wo · attn -> a
  rmsnorm_add                 x += post_attention_norm(a)
  rmsnorm                     x -> xn  (pre_feedforward)
  matvec_gate_up_gelu         h = gelu_tanh(Wg·xn) * (Wu·xn)
  matvec_f16                  Wdown · h -> m
  rmsnorm_add                 x += post_feedforward_norm(m)
  matvec_gelu_mul             g = gelu_tanh(W_ple_gate · x) * ple[layer]
  matvec_f16                  W_ple_proj_l · g -> p
  rmsnorm_add_scale           x = (x + post_per_layer_input_norm(p)) * layer_scalar
rmsnorm -> matvec (LM head, tied) -> argmax   (softcap only applied to dumped logits)
```

New vs the Llama kernel:
- **head_dim 256 and 512** (Llama kernel: 64 only) in the RoPE/norm and attention kernels.
- **Sandwich norms**: a fused "RMSNorm then add to residual" kernel.
- **GeGLU** (tanh GELU) instead of SwiGLU, in both the MLP and the PLE gate.
- **KV-shared layers**: only 15 KV caches; shared layers skip Wk/Wv and the cache write.
- **Sliding window**: the KV cache stays full length (`max_ctx`), and sliding layers simply
  start attention at `max(0, pos - 511)`. A ring buffer is a later optimization.
- **PLE**: one gather + one [8960, 1536] matvec per token, then a tiny gate/project per layer.
- **Attention scores in threadgroup memory** as in the Llama kernel, so `max_ctx` stays 4096
  for now.

Prefill reuses the Llama kernel's `simdgroup_matrix` GEMM for all projections (including the
PLE projection) and the (heads × T) attention grid with causal + sliding masks.

## Weight format

Same flat `.bin` as the Llama kernel (u64 header length, JSON header with config + tensor
offsets, 256-byte aligned tensors, padded to 16 KiB, mmapped into one `MTLBuffer`), written by
this folder's `convert_weights.py` from the HF safetensors:
- fp16: embed, PLE table, PLE model projection, per layer: fused Wqkv (Wq only for 15-34), Wo,
  Wgate, Wup, Wdown, PLE gate, PLE projection. Asserts every tensor fits fp16.
- fp32: all norm weights and `layer_scalar`.
- Vision/audio tensors, and the unused K/V weights of shared layers, are skipped.
- Gemma's RMSNorm here multiplies by `weight` directly (no `1 + weight` as in Gemma 1-3), so
  weights are stored as-is.

Size: about 9.3 GB (of which 4.7 GB is the PLE table).

## Correctness

- `benchmarking/reference.py`: HF transformers in fp32 on CPU (text model only) on the same
  chat-templated ids. Compare last-position logits (after softcap) and 32 greedy tokens.
- e2e tests (`surfaces/tests/test_kernel_e2e.py`) get Gemma reference strings from HF greedy.
- If they diverge, a debug flag dumps per-layer hidden states to diff against HF hooks.

## Benchmarks

- Ours: `--bench-pp 512 --bench-tg 128` as today.
- llama.cpp: GGUF from `convert_hf_to_gguf.py --outtype f16` on the same snapshot (the pinned
  llama.cpp has `gemma4` support), `llama-bench -p 512 -n 128 -ngl 99`.
- MLX: `mlx_lm benchmark` (mlx-lm has `gemma4_text`).
- Report effective bandwidth against the ~33 tok/s ceiling above.

## Disk budget

HF snapshot 10.2 GB + our `.bin` 9.3 GB + GGUF 9.3 GB is ~29 GB, and free disk on this machine
was ~33 GB when this started. So the MLX baseline loads the HF bf16 snapshot directly (same
bytes per weight as fp16) instead of writing a converted copy (`BASELINES` in
`surfaces/paths.py` has no `mlx` entry for this model).

---

## PRs (each based on `main`)

1. **Multi-model plumbing, no new kernel** (#28): `surfaces/paths.py` knows more than one model
   folder (`resolve_kernel` searches all of them, the default model is one constant), and the
   per-model baseline artifacts (GGUF, MLX model) and e2e reference strings are looked up per
   model instead of hard-coded to Llama, in `setup.py`, `bench.py`, the launcher, the MLX
   runner and the tests. Behavior on `main` stays the same (default model is still Llama).
2. **The kernel + this plan:** `kernel.json`, `convert_weights.py`, `kernels.metal`,
   `main.swift`, `build.py` in this folder; switch the default model to
   `google__gemma-4-E2B-it`; Gemma reference strings for the e2e tests; results in the READMEs.

## Status (2026-10-25)

Implemented as designed above; nothing in the design had to change.
- **Correct:** `benchmarking/reference.py` vs HF fp32: logits max |diff| 0.013 (range ±30),
  same top-5, 32/32 greedy tokens. GEMM prefill and token-by-token prefill give the same
  tokens. All kernel and MLX e2e tests pass on this model (the MLX runner needed
  `enable_thinking=False`: mlx-lm turns Gemma 4's thinking mode on by default, the HF
  template the kernel runner uses doesn't).
- **No fp16 overflow** in the converted weights or in the fp16 prefill activations (prefill
  and decode agree).
- **Speed (M3 Pro, 2 runs × 5 reps):** pp512 1122-1125 tok/s (llama.cpp `6050b46`
  1115-1127, MLX 2581-2708), tg128 27.7-28.4 tok/s (llama.cpp 25.2-26.3, MLX 26.7). Decode
  is ~85% of peak bandwidth at 4.56 GB/token.

Next, in rough order of payoff:
- **Prefill is 2.4x behind MLX.** Unprofiled guess: the naive attention (one threadgroup per
  head × query, re-reading K/V with head_dim 256/512, 8 Q heads on one KV head) and the many
  small PLE dispatches. Profile first (`ios_kernels/profiling/`).
- Q8 / Q4 weights for decode (bandwidth-bound).
- Ring-buffer KV cache for sliding layers; flash-decoding to lift `max_ctx` 4096.
- Fuse the PLE gate + projection, and the norm + residual epilogues, to cut ~12
  dispatches per layer.

## Risks / open questions

- **Numerics:** Gemma activations are large; the residual stream and all norms stay fp32, the
  KV cache is fp16 after QK/V-norm (values are O(1)), and the converter asserts weights fit fp16.
- **HF fp32 reference memory:** ~18.5 GB for the text model; fits in 36 GB but nothing else
  heavy can run at the same time.
- **`max_ctx` 4096** (threadgroup score buffer) is far below the model's 128k; fine for chat
  testing and benchmarks, flash-decoding is the follow-up.
- Follow-ups after the MVP: ring-buffer sliding KV cache, Q8/Q4 weights (decode is
  bandwidth-bound at ~4.6 GB/token), fusing the PLE gate/projection into neighbouring kernels.
