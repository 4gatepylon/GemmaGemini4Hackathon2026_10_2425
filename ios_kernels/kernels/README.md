# Kernels

Kernels are grouped by model: `kernels/<model>/kernel_<name>_<YYYYMMDD-HHMM>/`, where
`<model>` is the HF repo id with `/` replaced by `__` (e.g. `unsloth__Llama-3.2-1B-Instruct`).
The default model is `MODEL` in `surfaces/paths.py`; `--kernel <substring>` also finds kernels of other models. How to add a kernel, and the lineage header each kernel's
README starts with: [../CLAUDE.md](../CLAUDE.md).

Each folder `kernel_<name>_<YYYYMMDD-HHMM>/` is a self-contained, **immutable snapshot**
of one inference engine version (kernels, host code, weight converter, build script).
To iterate, `make new-kernel args=<name>` copies the newest folder under a new
timestamp; change the copy. Old versions stay runnable, so `make bench args="--kernel <old>"`
vs `<new>` is always possible. `make kernels` lists them.

The newest folder (by timestamp suffix) of the default model is the default everywhere: `openai-server`,
`benchmarking/*.py`, and the e2e tests.

Each folder must satisfy the contract in
[../surfaces/DESIGN_AND_LIMITATIONS.md](../surfaces/DESIGN_AND_LIMITATIONS.md#how-to-swap-things):
a `kernel.json` manifest, a build script, a weight converter, and a binary that speaks the
`--serve` JSON-lines protocol plus the benchmark flags. Build output goes to the folder's
gitignored `.build/`.

| folder | what | pp512 / tg128 (tok/s, M3 Pro) |
|---|---|---|
| `google__gemma-4-E2B-it/kernel_metal_f16_20261025-1225` | first Gemma 4 E2B (text) kernel: hand-written Metal + Swift, FP16 | ~1120 / ~28 (llama.cpp: ~1120 / ~25–26) |
| `unsloth__Llama-3.2-1B-Instruct/kernel_metal_f16_20261024-1620` | first MVP: hand-written Metal + Swift, Llama-3.2-1B FP16 | ~2050 / ~52–53 (llama.cpp: ~2650 / ~50–52) |
