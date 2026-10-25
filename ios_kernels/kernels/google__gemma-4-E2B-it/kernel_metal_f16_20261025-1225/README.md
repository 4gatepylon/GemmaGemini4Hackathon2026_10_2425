# kernel_metal_f16_20261025-1225

```yaml
kernel: kernel_metal_f16_20261025-1225
model: google__gemma-4-E2B-it
origin: root
parent: none
change: first Gemma 4 E2B (text) engine, hand-written Metal kernels + Swift host, FP16 weights (adapted from the Llama kernel_metal_f16_20261024-1620)
```

A from-scratch Gemma 4 E2B text forward pass with hand-written Metal compute kernels, a Swift
host (no ML framework, so it ports to iOS), and FP16 weights. It started as a copy of
`unsloth__Llama-3.2-1B-Instruct/kernel_metal_f16_20261024-1620` (same file format, matvec,
GEMM, sampling, `--serve` protocol). Design, model details and status: [PLAN.md](PLAN.md).

| file | what |
|---|---|
| `kernel.json` | manifest: HF repo, weights file name, build script, binary |
| `convert_weights.py` | HF safetensors (text model only) -> flat mmap-able FP16 file; resolves per-layer facts (head dim, FFN width, sliding, KV source layer, RoPE) into the header |
| `kernels.metal` | all kernels: scaled embedding, RMSNorm, sandwich-norm residual add, PLE combine, matvec (plain / GELU-gated), fused GeGLU matvec, QK/V-norm + RoPE + KV write, sliding-window attention (head dim 256 and 512), argmax, `simdgroup_matrix` GEMM, fused GeGLU GEMM. Compiled at runtime |
| `main.swift` | engine: mmaps weights into one MTLBuffer, encodes the forward pass, GPU-side greedy sampling with pipelined command buffers, benchmark mode, `--serve` mode |
| `build.py` | `swiftc -O` -> `.build/llm` |
| `PLAN.md` | design notes and status |

**Results**: see `ios_kernels/README.md` (`benchmarking/bench.py`, `benchmarking/reference.py`).

**Tunables** (no rebuild): `--define MV_NSG=4 --define MV_NR=2 --define ATT_NSG=4` etc.

**Limitations:** text only (no vision/audio). One KV head (MQA), as in E2B. Max context 4096
(attention scores live in threadgroup memory), and the sliding-window layers keep a full-length
cache. Greedy only. Prefill attention is naive (one threadgroup per head × query).
`--dump-logits` applies the final logit softcap; sampling skips it (it doesn't change argmax).
