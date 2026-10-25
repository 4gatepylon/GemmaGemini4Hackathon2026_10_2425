# PLAN: profiling our Metal kernels vs llama.cpp

> NOTE: this was written by: an earlier agent session (model and harness not recorded), landed in
> commit 4df248c (#3). These two notes were added by Claude Opus 5.5 (`claude-opus-5-5`) in
> Claude Code, Conductor workspace, 2026-10-24.

> NOTE (stale): this plan predates the `kernels/` layout (#4, #5, #8). `../forward/` (and
> `forward/main.swift`, `../forward/bench.sh`) no longer exist; the engine now lives in
> `../kernels/<model>/kernel_<name>_<YYYYMMDD-HHMM>/` (`main.swift`, `kernels.metal`) and
> benchmarking in `../benchmarking/bench.sh`. Paths below have not been updated yet.

**Goal:** a repeatable way to answer "where does the time go, and why?" for our
hand-written Metal forward pass (`../forward/`), and the same question for
llama.cpp's Metal backend, so the two can be compared kernel by kernel.
That means two kinds of output:
- **numbers**: time per kernel, share of the token/prompt, achieved GB/s and
  FLOP/s against the measured peak (roofline), gaps where the GPU is idle;
- **visuals**: a timeline/Gantt view (token > layer > kernel) and an
  aggregate flame-graph-style view, for both engines.

**Long-term goal:** every future kernel (Q8_0/Q4 matvec, flash attention,
Gemma 3, ...) comes with a one-command profile next to the llama.cpp
baseline, so optimization work is driven by measurements, not guesses.

**This document is the plan only.** The tooling it describes lands in later
PRs. Everything below marked *verified* was run on this machine.
Everything marked *to verify* is still an assumption.

Machine: Apple M3 Pro (18-core GPU), 36 GB, macOS 15.7.3, Xcode 26.3,
`xctrace` 26.0. Same box as `../PLAN.md`.

---

## Questions profiling has to answer

1. **Per-kernel breakdown**: which kernels dominate decode (`tg`) and prefill
   (`pp`), in ms and in % of the total?
2. **Distance to the roof**: for each kernel, achieved GB/s (decode matvecs
   are bandwidth bound) or FLOP/s (prefill GEMMs are compute bound) against
   the *measured* peak of this GPU.
3. **Why it's slow**: what limits a given kernel? Memory, ALU, occupancy,
   threadgroup-memory/barrier stalls, register spills?
4. **Idle GPU time**: gaps between kernels (dispatch overhead, serial
   dependencies), and whether the CPU encodes fast enough.
5. **Same questions for llama.cpp**, with the same tool where possible, so
   that "ours vs theirs" is apples to apples.

---

## Key decisions (please push back on any of these)

| Decision | Choice | Why |
|---|---|---|
| Per-kernel numbers for **our** engine | `--profile` mode: a `Profiler` (in `profiling/`) that puts GPU timestamps around every dispatch of the engine's encode path | Headless, scriptable, exact attribution to our own op names, no GUI. Prototype already works (see "Findings so far"). |
| Where profiling code lives | Everything in `ios_kernels/profiling/`. The one exception is the per-op hook calls inside the engine's encode functions (see Layer 1) | Keeps profiling self-contained while the rest of the code gets reorganized. |
| Per-kernel numbers for **llama.cpp** | `xctrace` (Instruments CLI) with the Metal shader timeline, plus llama.cpp's own `test-backend-ops perf` | We don't want to patch the baseline. xctrace sees any process's GPU work. `test-backend-ops` is llama.cpp's per-op microbenchmark. |
| Same view for both engines | One exporter: xctrace XML -> Chrome Trace Event JSON + per-kernel table, run on both traces | One tool and one format, so differences come from the engines, not the tooling. |
| Deep dive on one kernel ("why") | Xcode GPU frame capture (`.gputrace`) | Only place with per-dispatch hardware counters, limiters, occupancy and per-line shader cost. GUI only, so use it for investigation, not automation. |
| Visualization | **Don't build a viewer.** Emit Chrome Trace Event JSON and open it in [Perfetto UI](https://ui.perfetto.dev) (Gantt/timeline, nested slices). Emit folded stacks for [speedscope](https://www.speedscope.app) (flame graph). Instruments/Xcode for their own views | Free, local (Perfetto processes the trace in the browser), handles nesting and millions of events. Only the small exporters are ours. |
| Roofline peaks | Measure them with our own microbenchmarks (streaming read, `simdgroup_matrix` FMA loop) | Spec-sheet numbers (~150 GB/s) are only approximate. A measured roof makes "% of peak" meaningful. |
| Where data goes | Traces/exports in `.context/prof/` (gitignored). Only scripts, templates and small result tables are committed | `.trace` bundles and XML exports are large (113 MB of counter XML for one ~10 s run). |
| llama.cpp version | Same build as the benchmarks (`d327bf1`, stock) | Profile the exact baseline we compare against. |

---

## Tool survey (what exists, what it gives us)

| Tool | Gives | Headless? | Both engines? | Status |
|---|---|---|---|---|
| **Our `--profile` mode** (`MTLCounterSampleBuffer` timestamps) | Exact per-dispatch GPU time, named by us; bytes/FLOPs per op -> GB/s, FLOP/s; Perfetto JSON | yes | ours only | *verified* (prototype) |
| **`xctrace record --template 'Metal System Trace'`** | GPU timeline per command buffer/encoder, driver events, GPU performance state, thermal state, all processes | yes | yes | *verified*, but default template is too coarse (see roadblocks) |
| **Same, with a custom template (shader timeline + counter set)** | Per-shader (kernel) intervals and GPU counters (limiters, occupancy, ...) | yes, once the template exists | yes | *to verify*, template must be saved once from the Instruments GUI |
| **`xctrace export --toc / --xpath`** | Any table of a trace as XML (schemas: `metal-gpu-intervals`, `metal-shader-profiler-intervals`, `metal-gpu-counter-intervals`, `gpu-counter-info`, `gpu-performance-state-intervals`, `os-signpost`, ...) | yes | yes | *verified* |
| **Xcode GPU capture** (`MTLCaptureManager` -> `.gputrace`) | Per-dispatch counters, limiters, occupancy, shader cost / per-line profile (M3 has hardware support for this) | capture yes, analysis needs the Xcode GUI | yes: llama.cpp has `GGML_METAL_CAPTURE_COMPUTE` | *to verify* on our runtime-compiled shaders |
| **`test-backend-ops perf`** (llama.cpp) | us/run and GB/s or GFLOP/s per op and shape, on `MTL0` | yes | llama.cpp only | built (`.context/llama-build/bin/`), *no numbers yet* |
| **`test-export-graph-ops`** + `test-backend-ops --test-file` (llama.cpp) | Exports the ops of a real model graph, so `test-backend-ops` benchmarks llama.cpp's *exact* Llama-3.2-1B shapes | yes | llama.cpp only | exists in source, *to verify* |
| **`os_signpost`** (Points of Interest) | CPU-side intervals (encode time per token) in Instruments | yes | ours only | *to verify* (planned) |
| **Perfetto UI / speedscope** | Gantt timeline / flame graph from JSON / folded stacks | n/a (browser) | yes | Perfetto JSON *verified* to be produced. Opening it in the UI not yet checked |

Useful facts found along the way (*verified*):
- On Apple silicon, `device.sampleTimestamps()` returns identical CPU and GPU
  values in **nanoseconds** (the `CLOCK_UPTIME_RAW` domain, not 24 MHz Mach
  ticks). So GPU kernel spans and CPU encode spans can share one timeline.
- llama.cpp's Metal backend env vars: `GGML_METAL_CAPTURE_COMPUTE`,
  `GGML_METAL_CONCURRENCY_DISABLE`, `GGML_METAL_FUSION_DISABLE`,
  `GGML_METAL_FUSION_DEBUG`, `GGML_METAL_GRAPH_DEBUG`,
  `GGML_METAL_GRAPH_OPTIMIZE_DISABLE` (and residency/shared-buffer/tensor toggles).
- `GGML_METAL_CAPTURE_COMPUTE=N` captures the N-th graph compute to
  `/tmp/perf-metal-<pid>.gputrace`, and while capturing it wraps every op in a
  debug group named after the ggml op. Capture needs Metal capture enabled
  for a non-Xcode process (Apple's env var is `MTL_CAPTURE_ENABLED=1`.
  llama.cpp's error message says `METAL_CAPTURE_ENABLED`, *to verify* which one works).
- llama.cpp encodes with `MTLDispatchTypeConcurrent` plus explicit
  `memoryBarrier` calls, so independent kernels can overlap. We use the
  default serial encoder. This matters for timeline comparisons (below).

---

## Design

### Layer 1: built-in `--profile` mode (our engine)

The M3 Pro supports counter sampling **only at stage (encoder) boundaries**,
not at dispatch boundaries (*verified*:
`supportsCounterSampling(.atStageBoundary) == true`,
`.atDispatchBoundary == false`). The only counter set is `timestamp`. So in
profile mode:

- Every dispatch gets **its own compute encoder**, created from an
  `MTLComputePassDescriptor` whose `sampleBufferAttachments[0]` points at an
  `MTLCounterSampleBuffer` (start/end sample index per op).
- Each op is registered with a **name** (`L3.qkv`, `L3.attention`, ...) and
  its **bytes moved and FLOPs** (known statically from the shapes), so the
  report can print GB/s and FLOP/s per op.
- The same encode path is used for normal and profiled runs: each op
  boundary is one `op("name", bytes:, flops:)` call, which is a no-op when
  not profiling.
- Code placement: `Profiler` (sample buffer, encoder splitting, timestamp
  resolution, JSON/table output) lives in `profiling/Profiler.swift`, and
  `profiling/build.sh` compiles it together with the engine sources. The
  engine only needs the `op(...)` hook calls in its encode functions (one per
  dispatch, about 11 call sites). Those are the only lines outside
  `profiling/`, to be coordinated with the code cleanup.
- **Perturbation check**: the report always prints unprofiled vs profiled GPU
  time per token (from `gpuStartTime/gpuEndTime`), so we know how much
  splitting into encoders distorts the result.
- Outputs:
  1. a per-op-kind table on stdout (n per token, us, %, GB/s, GFLOP/s);
  2. `trace.json` in Chrome Trace Event format: one track per engine,
     nested slices token > layer > op, with bytes/GB/s/FLOP/s in `args`;
  3. (planned) folded stacks (`tg;layer3;qkv 1640`) for speedscope;
  4. (planned) CPU encode spans on a second track, using the shared ns clock.
- CLI sketch: `llm --weights W --profile tg|pp [--profile-steps N] --trace-out .context/prof/ours_tg.json`.

Optional add-on: **ablation mode** (skip or replace one op kind in the
normal, unsplit pipeline and measure the delta). This gives the marginal
cost of an op including overlap effects that split encoders hide.

### Layer 2: xctrace for both engines (cross-engine comparison)

- A **custom Instruments template**, saved once from the Instruments GUI and
  committed to `profiling/templates/`. It is Metal System Trace with the
  shader timeline on and a useful GPU counter set (e.g. performance limiters,
  occupancy; exact names *to verify* in the GUI).
- `xctrace_record.sh ours|llama [args]`: runs
  `xcrun xctrace record --template <ours.tracetemplate> --no-prompt --output .context/prof/<name>.trace --launch -- <binary> ...`
  for our `llm` or `llama-bench`, with short runs (`--time-limit` or
  `--window`) to keep exports small.
- `xctrace_to_perfetto.py`: `xctrace export --xpath` the shader-interval,
  encoder-interval, counter and performance-state tables. xctrace interns
  repeated values (`id=` on first use, `ref=` afterwards), so it needs a
  small resolver. A prototype parser exists. Filter to the target process.
  Write (a) Chrome trace JSON and (b) a per-kernel table.
- `compare.py`: joins the two per-kernel tables through a small hand-written
  mapping from llama.cpp kernel names to our op kinds (e.g. their
  `mul_mv` f16 kernels <-> our `matvec_*`). The mapping is filled in after the
  first llama.cpp trace shows their kernel names.

### Layer 3: Xcode GPU capture (deep dive, "why is this kernel slow")

- Our engine: add `--capture PATH.gputrace` (capture one decode token or one
  prefill chunk with `MTLCaptureManager`, `destination = .gpuTraceDocument`),
  plus encoder/command-buffer labels and debug groups per op so the capture
  is readable.
- llama.cpp: `MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=N llama-bench ...`.
- Open in Xcode: per-dispatch counters, limiter breakdown, occupancy, shader
  cost / per-line attribution. *To verify*: per-line attribution for
  shaders compiled at runtime from source. If it doesn't work, the fallback is
  the optional Metal Toolchain download (offline compile with source/line info).

### Per-op baseline for llama.cpp: `test-backend-ops perf`

- `test-backend-ops perf -b MTL0 -o MUL_MAT -p '<params regex>' --output csv`
  gives us/run and GB/s or GFLOP/s per op and shape.
- The built-in perf case list may not contain our exact shapes. A first
  guessed filter matched no cases. Plan: use `test-export-graph-ops` to dump
  the real Llama-3.2-1B graph ops and feed them with `--test-file`, or
  benchmark the closest built-in shapes.

### Roofline microbenchmarks

- `peak_bw`: streaming read (and copy) kernel over a buffer much larger
  than the caches -> measured GB/s roof for decode.
- `peak_mma`: `simdgroup_multiply_accumulate` FMA loop in registers ->
  measured FP16-in/FP32-acc FLOP/s roof for prefill GEMMs.
- Every per-kernel table reports "% of measured roof".

### Measurement hygiene (applies to every layer)

- **AC power**, nothing GPU-heavy open. A trace recorded during this
  investigation had Chrome and WindowServer doing far more GPU work than our
  process (about 2.4k Chrome GPU intervals vs our 63 command buffers).
- **Warm page cache** before timing: the first run after the weights were
  cold gave 34 tok/s instead of 52.
- Record **GPU performance state and thermal state** (both are xctrace
  tables) to catch clock ramp-up or throttling.
- Several runs, with mean ± std, as in `../forward/bench.sh`.

---

## Findings so far (prototype `--profile`, rough: on battery, Chrome open)

Prototype = a scratch copy of `forward/main.swift` with the Layer 1 design
(kept in `.context/prof/proto/`, not committed).

Decode, Llama-3.2-1B FP16, per token: unprofiled **19.27 ms** GPU,
profiled 20.36 ms (**+5.7%** from splitting into ~115 encoders). Gaps
between encoders total 0.46 ms.

| op | ms/token | % | GB/s |
|---|---:|---:|---:|
| gate_up (fused SwiGLU matvec) | 8.16 | 41.0 | 131.6 |
| down | 4.27 | 21.4 | 125.8 |
| lm_head + argmax | 4.01 | 20.1 | 131.1 |
| qkv | 1.64 | 8.2 | 122.7 |
| wo | 1.11 | 5.6 | 120.9 |
| norms + rope + attention + embed | 0.72 | 3.6 | n/a |

Prefill, 512 tokens: unprofiled **253.9 ms**, profiled 257.8 ms (+1.5%).

| op | ms | % | TFLOP/s |
|---|---:|---:|---:|
| gate_up GEMM | 116.0 | 45.1 | 4.74 |
| down GEMM | 57.5 | 22.3 | 4.78 |
| attention (naive) | 40.4 | 15.7 | 0.42 |
| qkv GEMM | 19.9 | 7.7 | 5.18 |
| wo GEMM | 14.4 | 5.6 | 4.76 |
| everything else | ~9 | ~3.6 | n/a |

This agrees with `../README.md`: decode matvecs run at ~121-132 GB/s, near
the bandwidth wall. In prefill, GEMMs are ~81% of the time and naive
attention ~16%. Small decode kernels look slower than they are in this mode
(each pays the per-encoder overhead).

---

## Dependencies

Already on this machine: Xcode 26.3 (Instruments, `xctrace`, Metal debugger),
Swift 6.2, Python 3 (stdlib `xml.etree` is enough for the exporter), a
browser for Perfetto/speedscope, llama.cpp `d327bf1` source and build (see
`../PLAN.md`).

Built for this task: `test-backend-ops` in `.context/llama-build/` (from
the same llama.cpp source, `-DLLAMA_BUILD_TESTS=ON`).

Optional:
- **Metal Toolchain** (`xcodebuild -downloadComponent MetalToolchain`, about
  1 GB): offline `.metallib` with source/line info, if per-line shader
  profiling doesn't work on runtime-compiled source.
- `speedscope` is also available as an npm CLI, but the web app is enough.

---

## Layout (planned)

```
ios_kernels/profiling/
  PLAN.md                    <- this file
  README.md                  <- how to profile, results (later)
  Profiler.swift             <- Layer 1: per-dispatch timestamps, table + Perfetto JSON, --capture
  build.sh                   <- builds the engine with Profiler.swift compiled in
  templates/
    metal-shader-timeline.tracetemplate   <- saved once from the Instruments GUI
  xctrace_record.sh          <- record ours / llama-bench with the template
  xctrace_to_perfetto.py     <- xctrace XML -> Chrome trace JSON + per-kernel table
  compare.py                 <- ours vs llama.cpp per-kernel table
  peak.swift + peak.metal    <- roofline microbenchmarks
```

Outside this folder: only the `op(...)` hook calls in the engine's encode
functions (see Layer 1).

---

## Checklist

- [x] Survey profiling tools on this machine (Xcode 26.3, `xctrace` 26.0, Metal System Trace and GPU counter instruments present; no offline Metal compiler)
- [x] Probe GPU counter sampling: stage boundary only, timestamp counter set only; CPU/GPU timestamps share a ns clock
- [x] Record Metal System Trace of our engine from the CLI, export with `--toc`/`--xpath`, write a small XML row parser
- [x] Try to enable shader timeline / counters by patching the stock template: rejected (see roadblocks)
- [x] Read llama.cpp Metal profiling hooks (capture env var, concurrency/fusion toggles, debug groups, `test-backend-ops`, `test-export-graph-ops`)
- [x] Prototype Layer 1 `--profile` (split encoders + timestamps -> table + Perfetto JSON), measure perturbation
- [x] Build llama.cpp `test-backend-ops`
- [ ] Save the custom Instruments template (needs one GUI session), commit it
- [ ] Confirm the shader timeline gives per-kernel compute intervals on M3, for our engine and for `llama-bench`
- [ ] `xctrace_to_perfetto.py` + `xctrace_record.sh`; open both engines' traces in Perfetto
- [ ] llama.cpp per-op numbers with `test-backend-ops perf` on the exact model shapes
- [ ] Roofline microbenchmarks (`peak_bw`, `peak_mma`)
- [ ] Land Layer 1 (`profiling/Profiler.swift` + `op(...)` hooks in the engine, with op labels/debug groups), add folded-stack output and CPU encode spans
- [ ] Xcode GPU capture of one decode token, both engines; check per-line shader profiling on runtime-compiled source
- [ ] `compare.py` and a results section in a profiling README

## Status (2026-10-24)

Investigation done, plan written, nothing in the repo implemented yet. The
Layer 1 prototype works and reproduces the README's breakdown. Layer 2 is
blocked on the one-time GUI template. llama.cpp per-op numbers are not
collected yet.

## Roadblocks / risks log

- **No per-dispatch GPU timestamps on Apple silicon** (stage boundary only).
  Workaround: one encoder per dispatch in profile mode. Cost: +5.7% decode /
  +1.5% prefill, and small kernels are inflated by per-encoder overhead.
  Ablation mode is the cross-check. *Fundamental, workaround in place.*
- **Default Metal System Trace is too coarse.** Only per-encoder intervals,
  which for us is one ~19 ms bar per token. The shader timeline is off, and the
  only counter recorded is "RT Unit Active" (363k samples, useless). Patching
  the template plist (`shaderprofiler`, `counterprofile` keys) makes xctrace
  fail with *"Selected counter profile is not supported on target device"*,
  because counter profiles are device specific. Workaround: save a template
  once from the Instruments GUI. *Blocking Layer 2 until done.*
- **Serial vs concurrent dispatch.** llama.cpp lets kernels overlap; we
  don't. Split-encoder profiles also serialize everything. So per-kernel *sums*
  are comparable, but overlap and idle gaps can only be compared on xctrace
  timelines of normal, unsplit runs.
- **Runtime-compiled shaders, no offline Metal compiler.** May limit per-line
  shader profiling in Xcode. Fallback: the optional Metal Toolchain download.
  *Risk, not yet tested.*
- **Measurement noise**: other apps' GPU work (Chrome, WindowServer show up in
  every trace), battery power, cold page cache, clock ramp-up. Mitigated by
  the hygiene rules above.
- **Big exports**: about 113 MB of counter XML for a ~10 s recording, with
  interned values. Keep recordings short and parse in a streaming way.
- **`test-backend-ops` shape coverage**: the default perf cases may not
  include our shapes. Plan: `test-export-graph-ops` -> `--test-file`.
- **Incomplete web research**: a background docs search on Instruments/Xcode
  details (counter-set names, per-line profiling requirements) stopped before
  finishing. Everything marked *to verify* above still needs checking against
  the real tools.
