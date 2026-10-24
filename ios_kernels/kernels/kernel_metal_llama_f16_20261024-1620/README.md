# kernel_metal_llama_f16_20261024-1620

The first MVP: a from-scratch Llama-3.2-1B forward pass with hand-written Metal compute
kernels, a Swift host (no ML framework, so it ports to iOS), and FP16 weights. Design
and rationale are in [../../PLAN.md](../../PLAN.md).

| file | what |
|---|---|
| `kernel.json` | manifest: HF repo, weights file name, build script, binary |
| `convert_weights.py` | HF safetensors -> flat mmap-able FP16 file (fused QKV, JSON header) |
| `kernels.metal` | all kernels: embed, rmsnorm, matvec, fused SwiGLU matvec, RoPE + KV write, attention, argmax, `simdgroup_matrix` GEMM, fused SwiGLU GEMM. Compiled at runtime |
| `main.swift` | engine: mmaps weights into one MTLBuffer, encodes the forward pass, GPU-side greedy sampling with pipelined command buffers, benchmark mode, `--serve` mode |
| `build.sh` | `swiftc -O` -> `.build/llm` |

**Results** (Apple M3 Pro, `benchmarking/bench.sh`): pp512 ~2050 tok/s (llama.cpp ~2650,
77%), tg128 ~52–53 tok/s (llama.cpp ~50–52, parity, ~86% of peak memory bandwidth).
Matches HF transformers: 32/32 greedy tokens identical, logits max |diff| 0.011.

**Tunables** (no rebuild): `--define MV_NSG=4 --define MV_NR=2` etc. FP16 decode is
bandwidth-bound and every matvec config lands at 52–53 tok/s.

**Limitations:** head_dim 64 only. Max context 4096 (attention scores live in threadgroup
memory). Greedy only. Prefill attention is naive (one threadgroup per head × query).
