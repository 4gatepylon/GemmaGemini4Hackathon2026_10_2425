# Kernels

Kernels are grouped by model: `kernels/<model>/kernel_<name>_<YYYYMMDD-HHMM>/`, where
`<model>` is the HF repo id with `/` replaced by `__` (e.g. `unsloth__Llama-3.2-1B-Instruct`).
For now only `unsloth__Llama-3.2-1B-Instruct` is used (hard-coded in `surfaces/paths.py`
and `benchmarking/common.sh`).

Each folder `kernel_<name>_<YYYYMMDD-HHMM>/` is a self-contained, **immutable snapshot**
of one inference engine version (kernels, host code, weight converter, build script).
To iterate, copy the newest folder under a new timestamp and change the copy. Old
versions stay runnable, so `benchmarking/bench.sh <old>` vs `<new>` is always possible.

The newest folder (by timestamp suffix) is the default everywhere: `openai-server`,
`benchmarking/*.sh`, `benchmarking/reference.py`, and the e2e tests.

Each folder must satisfy the contract in
[../surfaces/DESIGN_AND_LIMITATIONS.md](../surfaces/DESIGN_AND_LIMITATIONS.md#how-to-swap-things):
a `kernel.json` manifest, a build script, a weight converter, and a binary that speaks the
`--serve` JSON-lines protocol plus the benchmark flags. Build output goes to the folder's
gitignored `.build/`.

| folder | what | pp512 / tg128 (tok/s, M3 Pro) |
|---|---|---|
| `kernel_metal_f16_20261024-1620` | first MVP: hand-written Metal + Swift, Llama-3.2-1B FP16 | ~2050 / ~52–53 (llama.cpp: ~2650 / ~50–52) |
