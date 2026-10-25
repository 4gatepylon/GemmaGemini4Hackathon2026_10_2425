# PLAN: profiling our Metal kernels vs llama.cpp

> As of: main @ 028399a, 2026-10-25 13:11 PDT

> NOTE: this was written by: Claude Opus 5.5 (`claude-opus-5-5[1m]`, 1M-token context), an AI
> coding agent in Claude Code inside Conductor (workspace `honolulu`), working with @4gatepylon.
> The base commit and time of the latest edit are in the "As of" line above. The first version
> landed in #3, and #14 (another session) added authorship and staleness notes. This version
> replaces those notes: its paths follow the `kernels/` layout.

Working notes for agents in the loop: plan, status, measured numbers and history. Usage lives in
[README.md](README.md). Everything for profiling lives in this folder, and it follows
`../CLAUDE.md`: Python scripts run with `uv run`, never bash; every folder has a README;
docs are present tense; PRs are based on `main`.

Machine: Apple M3 Pro (18-core GPU, `AGXAcceleratorG15X`), 36 GB, macOS 15.7.3, Xcode 26.3,
`xctrace` 26.0. Same box as `../PLAN.md`.

---

## Objectives

"**Be able to X**" means a clean, reusable interface: a library, a CLI, and/or documentation
that other agents and humans can follow. With it, anyone can get reliable, reproducible
performance numbers for X in the future. A one-off measurement doesn't count.

| # | Objective | Interface | Done when |
|---|---|---|---|
| 1 | **Profile a simple, single kernel** (e.g. a matmul) | `kbench` (JSON spec -> timing stats, GB/s, GFLOP/s, % of the *measured* roof). `llama_ops.py` + `compare.py` (llama.cpp's number for the same op, side by side) | matvec and GEMM examples run with one command each; repeatable within a few %; llama.cpp's numbers next to ours |
| 2 | **Profile a combination of kernels in one system** | `kbench` pipelines (`loop`/`group`/`whole`). `RegionProfiler` for any Swift/Metal program. `xctrace_record.py` + `xctrace_export.py` for any process, including `llama-bench` | A decoder-layer pipeline gives a per-step breakdown plus the split overhead; xctrace recordings export to the common format and cross-check our timestamps |
| 3 | **Show Gantt and/or flame charts** | `kviz.py`: Perfetto JSON (interactive), folded stacks (speedscope), static PNG Gantt/flame/share charts | Charts from the outputs of objectives 1 and 2 |
| 4 | **Profile the components of a non-trivial kernel, plus hardware metrics** (cache hits/misses, occupancy, limiters) | `kbench` sweeps/ablations (`#define` variants -> component cost by difference), pipeline stats, working-set sweeps (cache levels), `--capture` (`.gputrace` for Xcode), counter export from a saved Instruments template | A GEMM split into load / MMA / store costs; a working-set sweep that shows the cache cliffs; documented, reproducible steps to get hardware counters for any kernel |
| 5 | **Profile the existing kernels** (`../kernels/<model>/kernel_*/`), *last* | Specs for each engine kernel at real shapes, a whole-engine timeline, a comparison against llama.cpp's exact graph ops | Per-kernel tables and charts for decode and prefill, ours vs llama.cpp, with noise recorded |

Objectives 1-4 land first, as small PRs. Objective 5 comes in a separate PR afterwards.

## Questions profiling has to answer

1. **Per-kernel breakdown**: which kernels dominate decode (`tg`) and prefill (`pp`), in ms and %?
2. **Distance to the roof**: achieved GB/s (bandwidth bound) or FLOP/s (compute bound) against
   the *measured* peak of this GPU.
3. **Why it's slow**: memory, ALU, occupancy, threadgroup-memory/barrier stalls, register
   pressure, cache misses?
4. **Idle GPU time**: gaps between kernels, and whether the CPU encodes fast enough.
5. **Same questions for llama.cpp**, with the same tools where possible.

## Noise policy: record it, don't require a quiet machine

Realistic workloads run with other things going on, so we don't require a quiesced machine.
**Every measurement records the noise present while it ran**, so numbers can be read in context
and outliers explained:
- device-wide GPU utilization (IOAccelerator `PerformanceStatistics`, no sudo), before the run
  (other processes' baseline) and during it;
- power source, Low Power Mode, thermal state, load average, busiest processes;
- spread between repetitions;
- in xctrace recordings, other processes' GPU intervals inside our window, GPU performance
  states, and preempted (split) encoders.

Optional, documented but not required: AC power, closing GPU-heavy apps,
`sudo powermetrics --samplers gpu_power`.

---

## Key decisions (please push back on any of these)

| Decision | Choice | Why |
|---|---|---|
| Timing source | `MTLCounterSampleBuffer` timestamps at encoder boundaries, one encoder per timed region, regions chained with an `MTLFence` | Exact GPU time without a GUI. Apple GPUs can't sample per dispatch, and they run independent encoders concurrently (see roadblocks) |
| One output format | `kprof/1` JSON. Every tool writes it (`kbench`, `knoise`, `RegionProfiler` users, `llama_ops.py`, `xctrace_export.py`) and every consumer reads it (`compare.py`, `kviz.py`) | Producers and viewers stay independent; ours vs llama.cpp is just two files |
| Languages | Swift for anything that touches Metal (plain `swiftc`, built by `build.py`). Python for export and charts, run with `uv run` (matplotlib in the `profiling` extra) | Same toolchain as the kernels; repo rule: scripts are Python |
| Kernel benches are data, not code | JSON specs with arithmetic expressions | New kernels get profiled by writing a spec |
| llama.cpp per-op baseline | `test-export-graph-ops` (the exact ops of our GGUF) -> `test-backend-ops perf --output sql` -> `kprof` | Stock llama.cpp, no patches |
| Cross-process / llama.cpp timelines | `xctrace` + exporter. Per-kernel detail needs a template saved once from the Instruments GUI | The only way to see llama.cpp's kernels without patching it |
| Hardware counters | Instruments counter template (headless after a one-time GUI save) and Xcode GPU capture (`.gputrace`, GUI analysis) | No public API exposes them (`MTLCounterSet` on this GPU is `timestamp` only) |
| Visualization | Perfetto UI and speedscope via exported files, plus our own static PNGs for embedding | Free, local, scale to big traces |
| Data | Outputs in `$DATA_DIR/prof/` (gitignored). Only code, specs, templates and example images are committed | Traces are large |
| Objective 5 vs immutable kernel folders | **Decided: replay, don't instrument.** `kbench` specs replay the engine's own `kernels.metal` (read-only, `--source`) at its dispatch geometry, and the real binary is timed whole (`--measure`) to validate the replay. A derived kernel folder with `RegionProfiler` hooks stays an option if per-op timing inside the real binary is ever needed | Touches no kernel folder. The replay matches the real engine within ~1% (decode) and 0.1% (prefill) |

---

## Tool survey

| Tool | Gives | Headless? | Both engines? | Status |
|---|---|---|---|---|
| `MTLCounterSampleBuffer` timestamps | GPU time per encoder | yes | ours | *verified* (stage boundary only) |
| `xctrace record` with the stock Metal System Trace | GPU intervals per encoder, GPU perf/thermal states, all processes | yes | yes | *verified*, per encoder only |
| The same with a GUI-saved template (shader timeline + counter set) | per-kernel intervals, GPU counters | yes, after a one-time GUI step | yes | *to verify* |
| Xcode GPU capture (`.gputrace`) | per-dispatch counters, limiters, occupancy, per-line shader cost | capture yes (`kbench --capture`), analysis GUI only | yes (`MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=1`) | capture *verified*; Xcode analysis *to verify* |
| `test-export-graph-ops` / `test-backend-ops perf` | a model's unique graph ops / us per op | yes | llama.cpp | *verified* |
| IOAccelerator `PerformanceStatistics` | device-wide GPU busy % | yes, no sudo | n/a (noise) | *verified* |
| Perfetto UI / speedscope | interactive timeline / flame graph | browser | yes | Perfetto *verified* (headless Chrome loads our JSON) |

llama.cpp hooks (*verified* in source, `d327bf1`): `GGML_METAL_CAPTURE_COMPUTE=N` captures the
N-th graph compute to `/tmp/perf-metal-<pid>.gputrace`, with each op in a debug group (needs
`MTL_CAPTURE_ENABLED=1`, *verified*); `GGML_METAL_CONCURRENCY_DISABLE`,
`GGML_METAL_FUSION_DISABLE`, `GGML_METAL_GRAPH_DEBUG`; it encodes with
`MTLDispatchTypeConcurrent` plus memory barriers, so its kernels can overlap, while ours run serially.

---

## Design (usage is in README.md)

- **`kbench`** (`tools/kbench/`, `lib/`): spec -> one timed encoder per step (each step's
  dispatch repeated `repeat` times), fence-chained, pipelined reps; `rotate` buffers against
  cache reuse; `loop`/`group`/`whole` for pipelines; `--set` overrides; `peaks.json` keeps the
  best measured roof.
- **`RegionProfiler`** (`lib/`): the same timestamped, fence-chained encoders for any Swift/Metal
  program. Event names become encoder labels, so xctrace shows them too.
- **`knoise`**: noise around any command.
- **`llama_ops.py`** / **`compare.py`**: llama.cpp per-op `kprof`, side-by-side tables.
- **`xctrace_record.py`** / **`xctrace_export.py`**: headless Instruments recordings of any
  process -> `kprof`, plus `--validate` against our own timestamps.
- **`kviz.py`**: Perfetto JSON, folded stacks, PNG gantt/flame/share, table.
- **Objective 4 (next)**: `sweep` (cartesian over defines/vars, one compile per variant) for
  ablations; pipeline stats (`maxTotalThreadsPerThreadgroup`, `staticThreadgroupMemoryLength`);
  a GEMM with switches that drop loads / MMA / stores (results kept observable so nothing is
  dead-code eliminated); working-set sweeps for cache levels; `kbench --capture`; the
  Instruments template plus counter export.

---

## PR plan

Every PR is based on `main` (no stacking, per `../CLAUDE.md`). The feature PRs depend on each
other, so they merge **in order**. Until its predecessor merges, a PR's diff also shows the
predecessor's commit; after each merge, the next one gets rebased onto `main`.

| PR | Objective | Contents | State |
|---|---|---|---|
| #3 | plan | PLAN v1 (survey + design) | merged |
| #7 | plan | this file: objectives, noise policy, PR plan, status, findings, roadblocks | open |
| #12 | 1 | `lib/`, `kbench`, `build.py`, matmul + roofline examples, README | open, merge 1st |
| #16 | 1 | `knoise`, `kprof.py`, `llama_ops.py`, `compare.py` | open, merge 2nd |
| #20 | 2 | pipeline example, `xctrace_record.py`, `xctrace_export.py`, RegionProfiler docs | open, merge 3rd |
| #27 | 3 | `kviz.py`, `docs/` example images, `profiling` extra (matplotlib) | open, merge 4th |
| #29 | 4 | sweeps/variants, pipeline stats, GEMM ablation, working-set sweep, `--capture`, `kviz sweep`, template instructions | open, merge 5th |
| #32 | 5 | `profile_engine.py`, `engine/` replay specs + categories, `llama_ops.py graph`, `bench --runs` | open, merge 6th |

---

## Findings (M3 Pro, measured 2026-10-24, mostly on battery with background load)

**Measured roof** (`examples/roofline`, best values): streaming read **131.6 GB/s** (copy
126.4, write 118.9); FP16->FP32 `simdgroup_matrix` MMA **6.21 TFLOP/s**; FP32 FMA 5.85
TFLOP/s. With another workspace benchmarking at the same time, the same specs measured
102-124 GB/s and 5.5-6.1 TFLOP/s, which is why `peaks.json` keeps the best value.

**Single kernels, ours vs llama.cpp `d327bf1`** (`examples/matmul` vs `llama_ops.py mul-mat`):

| shape | ours | llama.cpp | ours vs llama |
|---|---:|---:|---:|
| matvec f16 4096x4096 | 263 us, 127.5 GB/s | 249 us, 125.4 GB/s | 0.95x |
| GEMM f16 T=512, 2048x2048 | 856 us, 5.02 TFLOP/s (81% of roof) | 795 us, 5.40 TFLOP/s | 0.93x |
| naive GEMM, same shape | 22.6 ms, 0.19 TFLOP/s | - | - |

Cache effect (8 MiB matvec, `--set M=2048 --set K=2048`): same buffer every dispatch
(`ROT=1`) 175 GB/s, rotated over 8 copies 123 GB/s; llama.cpp's cache-warm number is 187 GB/s.
At 32 MiB the difference is gone (125 vs 127 GB/s), so the GPU's caches hold between 8 and 32 MiB.

**llama.cpp graph ops** (Llama-3.2-1B, 34 unique ops): decode-sized elementwise ops (ADD, MUL,
RMS_NORM, ROPE, DUP on 2048 floats) all cost about 3 us/run, the launch floor that fusion removes.

**Pipeline** (`examples/pipeline`, 16 layers, POS=511): one decode token takes 15.43 ms as one
encoder vs 15.41 ms over 128 split encoders (−0.2%); matvecs run at 128-137 GB/s, gate_up is
51% of the token, down 26%, qkv 10%, wo 7%, attention 4%, and norms/rope sit at 3-7 us each.

**Validation**: across 1293 encoders, `RegionProfiler` and xctrace medians agree exactly on every
label. Means differ only for the encoders WindowServer preempted (≤1.2%), because our span
includes preempted time and xctrace sums only the active pieces.

**GEMM components** (`examples/gemm_components`, 512x2048x2048, reproduced 3 times):

| variant | median us | vs full |
|---|---:|---:|
| full | 867 | - |
| no global loads | 833 | -4% |
| no stores | 855 | -1% |
| MMA only (no loads, no stores) | 832 | -4% |
| no MMA | 128-154 | -82 to -85% |

The inner loop (simdgroup loads from threadgroup memory + MMA) is ~96% of the time, at 83% of
the MMA roof; global loads and stores hide behind it. Room to grow is in the inner loop: fewer
simdgroup loads per MMA (larger register tiles), not in memory traffic.

**Working set** (`examples/roofline/working_set.json`): ~1.0-1.9 TB/s up to 16 MiB, 525 GB/s at
24 MiB, 230 GB/s at 32 MiB, DRAM (~125-140 GB/s) from 48 MiB. The GPU's last-level cache gives out
between 16 and 32 MiB.

**The existing engine vs llama.cpp** (`profile_engine.py --measure`, AC power; the
Llama-3.2-1B artifacts were later removed from this machine, see roadblocks):

| decode, per token (POS=64) | ours | llama.cpp |
|---|---:|---:|
| replay, one encoder / sum of isolated ops | 18.6 ms | 18.1 ms (cache-warm, wall time) |
| real binary (`--bench-tg 128`) / `llama-bench` tg128 | **18.8 ms** | **19.3 ms** |
| dispatches per token | 132 | 276 |
| gate/up + SwiGLU, down, LM head | 8.1 / 4.1 / 4.0 ms | 8.0 / 3.8 / 3.8 ms |
| QKV, output projection (small matrices: llama.cpp's op times are cache-warm) | 1.6 / 1.1 ms | 0.9 / 0.7 ms |

| prefill, 512 tokens | ours | llama.cpp |
|---|---:|---:|
| replay / real binary | 248.0 / **248.3 ms** | sum of ops 194.8 / **llama-bench 192.6 ms** |
| attention | **38.8 ms** (naive) | **9.0 ms** (flash attention) |
| gate/up + SwiGLU | 115.2 ms | 96.6 ms |
| down / output / QKV projection | 54.9 / 13.7 / 19.7 ms | 48.0 / 12.6 / 20.0 ms |

Decode is at parity or slightly ahead (FP16 decode is bandwidth bound; quantization is the
lever). Prefill's gap is mostly attention (~30 ms: a flash-attention kernel) and the SwiGLU/down
GEMMs (~25 ms: GEMM tuning, see the GEMM components finding).

**Power source matters:** on AC, the naive GEMM example runs at 0.28 TFLOP/s vs 0.19 on battery
(same code), which is why every result records the power source.

**Engine prototype** (scratch copy of the engine with one timed encoder per dispatch, not
committed): decode 19.27 ms/token unprofiled vs 20.36 ms split (+5.7%); gate_up 41%, down
21%, lm_head 20%, qkv 8%, wo 6%, everything else 4%. Prefill 512: 253.9 ms vs 257.8 ms
(+1.5%); GEMMs 81% (4.7-5.2 TFLOP/s), naive attention 16% (40 ms).

---

## Dependencies

Xcode 26.3 (Instruments, `xctrace`, Metal debugger), Swift 6.2, uv (the project's Python;
matplotlib comes from the `profiling` extra), and the llama.cpp checkout from
`benchmarking/setup.py`. `llama_ops.py build` builds `test-backend-ops`,
`test-export-graph-ops` and `llama-bench` into `<llama.cpp>/build-tests`.

Optional: the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`, about 1 GB), for
offline `.metallib` builds with source/line info, if per-line shader profiling doesn't work on
runtime-compiled source.

---

## Checklist

- [x] Survey tools; probe counter sampling; headless xctrace record/export; llama.cpp hooks
- [x] Engine prototype with per-dispatch timestamps; measure the perturbation
- [x] Objective 1: `kbench` + lib + examples (#12); llama.cpp baseline + `compare` + `knoise` (#16)
- [x] Objective 2: pipelines, `RegionProfiler` docs, xctrace tools with `--validate` (#20)
- [x] Objective 3: `kviz.py` (#27)
- [x] Rework for `../CLAUDE.md`: bash -> Python, folder READMEs, present-tense docs (numbers live here), ruff-clean, PRs based on `main`
- [ ] Save the Instruments counter/shader-timeline template from the GUI (needs a human, once)
- [x] Objective 4 (#29), except the Instruments counter template, which needs the GUI step above
- [x] Decide how objective 5 handles immutable kernel folders: replay (see key decisions)
- [x] Objective 5 (#32)

## Status

All five objectives are implemented and in review (#12, #16, #20, #27, #29, #32; merge in that
order). Open: the Instruments counter template, which needs a one-time GUI step
(`templates/README.md`). Next candidates: a per-threadgroup "logical clock" timeline (see below),
then profiling the Gemma kernels once they land (new specs; the tools are model-agnostic).

## Roadblocks / risks log

- **Apple GPUs run independent encoders concurrently**, even across command buffers: two
  roofline steps with no shared buffers overlapped completely (std 23-44%). `RegionProfiler`
  chains timed regions with an `MTLFence`. After that: 0 overlaps, 0.3-1 us gaps, 1-3% spread
  on a quiet GPU. *Fundamental, fixed.*
- **No per-dispatch GPU timestamps** (stage boundary only): one encoder per timed region, with
  `repeat` for tiny kernels. Cost on the full engine: +5.7% decode, +1.5% prefill. *Fundamental,
  worked around.*
- **Counter sample buffer limits**: 32 KB per buffer (4096 timestamps), about 32 alive at once.
  *Constraint.*
- **GPU-utilization metric quirks**: the first read after a pause is stale, and polling faster
  than ~100 ms reads 100% even when idle. So: 250 ms polling, first sample dropped,
  device-wide only. *Constraint.*
- **Noisy roofline runs lowered the roof**: `peaks.json` now keeps the best value
  (`--reset-peaks`). *Fixed.*
- **The stock Metal System Trace is too coarse** (per-encoder intervals, shader timeline off,
  only an "RT Unit Active" counter). Patching the template plist fails with *"Selected counter
  profile is not supported on target device"*, even with this GPU's device ID and profiles 1-3.
  Needs a template saved once from the Instruments GUI. *Blocks per-kernel llama.cpp
  timelines and hardware counters.*
- **No headless `.gputrace` analysis**: Xcode ships only GUI frameworks. *Accepted.*
- **Serial vs concurrent dispatch**: llama.cpp overlaps kernels, we don't. Per-kernel sums are
  comparable; overlap and gaps only on unsplit timelines.
- **`test-backend-ops` method**: cache-warm reruns on the same buffers, and GB/s computed from
  whole tensor sizes (absurd for `GET_ROWS`/`SET_ROWS`, e.g. 70,727 GB/s); `--output csv` drops
  the timing columns (use `sql`); `test-export-graph-ops` exports at the model's full context
  (131072) unless `--ctx` is given. *Documented in the README.*
- **`ggml.h` enum parsing**: `GGML_OP_NONE = 0` has an initializer, and missing it shifted
  every op ID by one. The parser now asserts that NONE comes first. *Fixed.*
- **Instruments splits preempted encoders** into several intervals. The exporter sums them and
  counts splits as noise. *Documented.*
- **A relative `xctrace record --output`** once produced an unreadable trace. The recorder now
  makes the path absolute and checks the result. *Fixed.*
- **llama.cpp's encoders are unlabeled and coarse** (about 2 per decode token), so per-kernel
  llama.cpp timelines need the shader-timeline template. *Open, objective 4.*
- **Perfetto drops partially overlapping slices on one thread** (another process's vertex and
  fragment work overlap). `kviz.py` splits such tracks into lanes. *Fixed.*
- **Chrome blocks ui.perfetto.dev from fetching localhost traces** unless local network access
  is allowed (a permission prompt in a normal browser). *Documented.*
- **Repo rules changed mid-way** (`../CLAUDE.md`: Python not bash, a README per folder,
  present-tense docs, no stacked PRs). The tools were ported, the docs rewritten, and the PRs
  rebased onto `main`. *Done.*
- **Ablation switches can change codegen**: an early `SKIP_STORE` folded the 16 accumulators
  with `thread_elements()`, and the kernel got 3x *slower*. The switch now keeps all stores live
  behind a runtime-false branch (`T == 0`). Rule in the README: an ablated variant that gets
  slower means the switch changed more than its component. *Fixed.*
- **GPU clock ramp-up after idle** made the first variant of a sweep noisy (p90 +48%). Warmup is
  now time-based (`--warmup-ms`, default 200). *Fixed.*
- **Small working sets measured per dispatch** were dominated by launch overhead and GPU
  downclocking (0.5-2 MiB read at 15-31 GB/s). `read_passes` re-reads the buffer many times in one
  dispatch (~256 MiB per dispatch). *Fixed.*
- **llama.cpp capture needs Apple's `MTL_CAPTURE_ENABLED=1`.** Its error message names
  `METAL_CAPTURE_ENABLED`, which does nothing. A whole-model capture is 2.3 GB, because it holds
  the weights. *Verified, documented.*
- **`test-backend-ops` reports host wall time**: it times `ggml_backend_graph_compute` on the
  CPU, so CPU load elsewhere (e.g. a busy browser) inflated op times 2x while `kbench`'s GPU
  timestamps stayed steady. `llama_ops.py bench --runs N` keeps each op's fastest run. *Mitigated.*
- **Workspaces are archived mid-run**: another workspace's `.context/` (model files and the
  llama.cpp checkout this workspace's test tools were built from) disappeared during a run, and
  no Llama-3.2-1B artifacts remain on the machine (the project moved to Gemma 4). Results here
  were recorded before that. Rebuilding needs `benchmarking/setup.py`. A shared `$DATA_DIR` (the
  worktree-setup PR) would avoid it. *Noted.*
- **No per-core GPU timeline on Apple GPUs**: Metal has no shader clock (a `clock()` intrinsic is
  an open feature request) and no core/SM id, and Instruments shows GPU work per channel
  (compute/vertex/fragment), not per core. The closest is a DIY "logical clock": each threadgroup
  records an atomic ticket when it starts and ends, giving a Gantt chart of threadgroups in
  ticket order (waves, tail effects, how many are in flight), not in real time. *Not built yet.*
- **Required CI landed mid-way** (#22: ruff lint + format, kernel tree, no shell scripts,
  forbidden markers, pytest). Branches created before it lacked the check scripts, so the
  markers job failed with "No such file". They were rebased onto `main`, formatted with
  `ruff format`, and every check was run locally before pushing. *Done.*
- **Kernel folders are immutable once merged**, so objective 5 can't add hooks to the engine in
  place. *Open; see key decisions.*
- **Runtime-compiled shaders**: per-line Xcode profiling may need offline compilation with source
  info (the optional Metal Toolchain). *Not yet tested.*
