# profiling

Tools for measuring how fast Metal kernels run, ours and llama.cpp's, with reproducible numbers and the measurement noise recorded alongside them.

Run everything from `ios_kernels/`. Design, objectives, status and measured results are in
[PLAN.md](PLAN.md), and each subfolder's README says what it holds. Outputs go to
`$DATA_DIR/prof/` (`<repo>/.context/prof/` by default). Every tool writes the same `kprof/1`
JSON (see "Output format"), so any producer works with any consumer.

## Quickstart

```sh
uv run profiling/build.py                                            # Swift tools -> profiling/.build/
profiling/.build/kbench profiling/examples/roofline/bandwidth.json   # this machine's bandwidth roof
profiling/.build/kbench profiling/examples/roofline/compute.json     # ... and its FLOP/s roof
profiling/.build/kbench profiling/examples/matmul/matvec.json        # profile one kernel
profiling/.build/kbench profiling/examples/matmul/matvec.json --set M=8192 --set NR=2   # another shape / tunable
```

`kbench` prints one row per step:

| column | meaning |
|---|---|
| median / min / p90 us | GPU time per dispatch (timed encoder time divided by `repeat`) |
| spread% | (p90 − p10) / median. Above 10% the row is flagged as noisy; the median stays robust |
| GB/s, GFLOP/s | from the spec's `bytes` / `flops` per dispatch |
| %bw, %flop | against the measured roof in `$DATA_DIR/prof/peaks.json` |

The roof is the best the roofline specs have reached on this machine, not a theoretical
limit, so a well-tuned kernel can slightly exceed 100%. The table is followed by a noise line
(see "Noise") and the path of the `kprof` file.

## Single kernels: `kbench` specs

A spec describes one or more **steps** (kernel dispatches) over named **buffers**. Every size
can be an expression over `vars` and numeric `defines`:

```json
{
  "name": "matvec f16",
  "source": "matmul.metal",
  "defines": {"NR": 4, "NSG": 8},
  "vars": {"M": 4096, "K": 4096, "ROT": 4},
  "buffers": {
    "W": {"type": "half",  "count": "M * K", "init": "random", "rotate": "ROT"},
    "x": {"type": "float", "count": "K", "init": "random"},
    "y": {"type": "float", "count": "M"}
  },
  "steps": [
    {"name": "matvec", "kernel": "matvec_f16",
     "bind": [{"buf": "W"}, {"buf": "x"}, {"buf": "y"}, {"u32": "M"}, {"u32": "K"}],
     "threadgroups": ["M / (NR * NSG)"], "threads": ["32 * NSG"],
     "bytes": "M * K * 2 + K * 4 + M * 4", "flops": "2 * M * K",
     "repeat": 8}
  ],
  "warmup": 5, "reps": 30
}
```

| field | meaning |
|---|---|
| `source` | `.metal` file relative to the spec, compiled at runtime (no Metal toolchain needed). `fast_math` (default true) matches the kernels in `../kernels/` |
| `defines` | `#define NAME VALUE` lines prepended to the source. Override with `--set NAME=VALUE` |
| `vars` | named numbers or expressions (they may reference each other). Override with `--set` |
| `buffers.X.type` | `half float int uint short ushort char uchar` |
| `buffers.X.count` | element count |
| `buffers.X.init` | `zero` (default), `ones`, `random` (uniform [-1, 1)), `iota`, `{"random": [lo, hi]}`, `{"random_int": [lo, hi]}`, `{"const": v}`. Seeds are fixed, so runs are reproducible |
| `buffers.X.rotate` | N identical copies, cycled per dispatch, so consecutive dispatches don't hit in cache (real layers have distinct weights). `1` measures the cache-warm case |
| `steps[].bind` | the argument table in index order: `{"buf": name, "offset": bytes}`, a scalar `{"u32"/"i32"/"f32"/"f16"/"u16": expr}`, or `{"struct": [fields...]}` packed with MSL alignment |
| `steps[].threadgroups` + `threads` | `dispatchThreadgroups`. `grid` + `threads` uses `dispatchThreads` instead |
| `steps[].threadgroup_memory` | byte sizes, one per threadgroup-memory index |
| `steps[].bytes` / `flops` | work per dispatch, for GB/s and GFLOP/s. Count compulsory traffic: what has to move at least once |
| `steps[].repeat` | dispatches per timed encoder. Raise it for kernels under ~50 us, so encoder overhead is amortized |
| `warmup`, `reps` | untimed and timed repetitions (`--warmup`, `--reps` override them) |
| `roof` | `"bandwidth"` or `"flops"`: the spec measures the machine's roof (see "Roofline") |
| `loop` | `{"name": "layer", "count": expr}`: run the step list `count` times per rep. Use `rotate` on weight buffers so each iteration has its own copy |
| `steps[].group` | an extra nesting level for charts, e.g. `"attention"` / `"mlp"` |
| `whole` | `true`: also time each rep as **one** encoder, the way an engine encodes a token, and compare that with the sum of the split steps. The difference is what profiling costs |

Expressions support `+ - * / %`, parentheses, and `min max ceil floor round ceildiv pow log2`.
Integer contexts (counts, grids) fail loudly when a result isn't an integer.
`kbench SPEC --list` prints the resolved spec without running it.

## How timing works

- **One encoder per timed step.** Apple GPUs take timestamps only at encoder boundaries
  (`supportsCounterSampling(.atDispatchBoundary)` is false on M-series), so each step's
  dispatches go in their own encoder, with its start and end written to an
  `MTLCounterSampleBuffer`.
- **Timed encoders are chained with a fence.** Apple GPUs run *independent* encoders
  concurrently, even across command buffers. Metal only orders encoders that share
  resources. `RegionProfiler` makes every timed region wait for the previous one, so no
  region absorbs another's work.
- **Pipelined repetitions.** Each rep is its own command buffer, and they are queued back to
  back, so the GPU stays busy and clocked up.
- **Timestamps are ns** on the `CLOCK_UPTIME_RAW` clock (Apple silicon's GPU timestamps are
  already in that domain), so CPU and GPU events share one timeline.
- **Limits:** one counter sample buffer holds 2048 regions, and only about 32 can exist at
  once. `RegionProfiler.collect()` frees them.

## Roofline

`examples/roofline/` streams 512 MiB (read, copy, write) and runs `simdgroup_matrix` MMA and
FP32 FMA loops. Its specs set `roof`, so `kbench` records the best throughput in
`$DATA_DIR/prof/peaks.json`, and every later result reports a % of it. The file keeps the
best value ever measured, so a run slowed down by noise never lowers the roof.
`--reset-peaks` starts over, for example on a new machine.

## Noise

We don't require a quiet machine; every output records what was going on (`noise` in the
JSON, one summary line on stdout):

- **GPU idle baseline:** device-wide GPU busy % (IOAccelerator `PerformanceStatistics`, the
  value `ioreg -r -c IOAccelerator` shows, no sudo needed). It is sampled every 250 ms for
  `--idle-ms` (default 1000) *before* our run, which captures other processes' GPU load, and
  during the run, which includes our own work. The first read after a pause is stale, and
  polling faster than ~100 ms reads 100%, so the first sample is dropped.
- **Power source, Low Power Mode and thermal state** at start and end, the **load average**,
  and the **busiest CPU processes** at start.
- **`notes`:** plain-language warnings (on battery, thermal pressure, GPU busy before the run).

The idle baseline is measured before the run, so load that starts mid-run shows up only as
slow outliers. An Instruments (xctrace) recording can attribute GPU time to processes.

## Output format: `kprof/1`

JSON with snake_case keys, defined in `lib/KProf.swift`:

- `meta`: tool, name, created, device (name, model, gpu_cores, ...), os, git (`sha[-dirty]`),
  argv, spec, and params (the resolved defines and vars).
- `noise`: start/end snapshots, `gpu_util_idle` / `gpu_util_run`, and notes.
- `events`: one per timed region, with name, kernel, `stack` (the nesting path, outermost
  first, e.g. `["rep 3", "layer 7", "qkv"]`), track, start_ns, dur_ns, bytes, flops,
  variant and args. `dur_ns` covers all `repeat` dispatches of a region.
- `counters`: time series (e.g. GPU util %), one track each.
- `results`: aggregate rows, with n, median/mean/std/min/p10/p90 us per dispatch, gbps,
  gflops, pct_peak_bw, pct_peak_flops, and `method` (how the row was measured).
