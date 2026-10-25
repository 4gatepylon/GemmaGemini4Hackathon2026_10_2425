#!/usr/bin/env python3
"""Convert an Instruments trace (Metal System Trace) to kprof.

  xctrace_export.py TRACE [--process NAME] [--out JSON] [--counters] [--validate KPROF.json]

Works for any process (our engine, kbench, llama-bench). What it extracts:
  events   - the target process's GPU intervals, one per encoder, named by the
             encoder label (RegionProfiler labels encoders with the op name).
             An encoder preempted by other GPU work shows up as several
             intervals ("split"). Per-shader intervals too, if the template
             has the shader timeline on (see README, "Instruments template").
             Other processes' GPU work in the same window goes on "GPU other:"
             tracks, and GPU performance-state changes on their own track.
  results  - per encoder label: n, median/min/p90 us (split intervals summed).
  noise    - knoise's (TRACE.noise.json, written by xctrace_record.sh) plus
             `xctrace`: other processes' GPU busy time in our window, GPU
             performance states, thermal states, and device GPU busy %.
  counters - with --counters, GPU counter samples (large; needs a counter template).
--validate compares the per-label medians with a kbench/RegionProfiler kprof
file of the same run: an independent check of our timestamps.

Times are ns from the start of the recording (not the CLOCK_UPTIME_RAW base
kbench uses), so compare durations, not absolute times.
"""

import argparse
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import kprof


def export(trace, args_):
    r = subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), *args_], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"xctrace export failed: {r.stderr.strip()}")
    return r.stdout


def rows(path):
    """Stream rows of an exported table as {column: (raw, fmt)}. xctrace interns
    values: the first occurrence has id="N", later ones are <tag ref="N"/>."""
    cache, cols = {}, None

    def value(e):
        if "ref" in e.attrib:
            return cache[e.attrib["ref"]]
        v = (e.text, e.attrib.get("fmt"))
        for c in e:  # nested values (e.g. process -> pid) are cached too
            value(c)
        if "id" in e.attrib:
            cache[e.attrib["id"]] = v
        return v

    for _, e in ET.iterparse(path, events=("end",)):
        if e.tag == "schema":
            cols = [c.findtext("mnemonic") for c in e.findall("col")]
        elif e.tag == "row":
            yield dict(zip(cols, (value(c) for c in e), strict=False))  # trailing empty cells are omitted
            e.clear()


def num(v):
    try:
        return int(v[0])
    except (TypeError, ValueError):
        return None


LABEL = re.compile(r"^(?:Command Buffer \d+:)?(.*?)\s+\(\s*(.*?)\s*\((\d+)\)\s*\)\s*(0x[0-9a-f]+)?\s*$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--process", help="target process name (default: the launched process)")
    ap.add_argument("--out")
    ap.add_argument("--counters", action="store_true", help="also export GPU counter samples")
    ap.add_argument("--validate", help="kprof file from the same run (kbench) to compare medians against")
    args = ap.parse_args()
    trace = Path(args.trace)

    toc = ET.fromstring(export(trace, ["--toc"]))
    run = toc.find("run")
    launched = run.find(".//process[@type='launched']")
    target = args.process or (launched.get("name") if launched is not None else None)
    if not target:
        sys.exit("no launched process in the trace; pass --process NAME")
    schemas = {t.get("schema") for t in run.iter("table")}
    template = run.findtext(".//template-name")
    dev = run.find(".//device")

    wanted = [
        "metal-gpu-intervals",
        "metal-shader-profiler-intervals",
        "gpu-performance-state-intervals",
        "device-thermal-state-intervals",
        "metal-gpu-state-intervals",
    ]
    if args.counters:
        wanted += ["metal-gpu-counter-intervals"]
    tables = {}
    with tempfile.TemporaryDirectory() as tmp:
        for s in wanted:
            if s in schemas:
                p = Path(tmp) / f"{s}.xml"
                p.write_text(export(trace, ["--xpath", f'/trace-toc/run[@number="1"]/data/table[@schema="{s}"]']))
                tables[s] = list(rows(p))

    events, other = [], defaultdict(list)
    per_encoder = defaultdict(list)  # (label, encoder id) -> [(start, dur)]
    for r in tables.get("metal-gpu-intervals", []):
        proc = (r.get("process") or (None, ""))[1] or ""
        start, dur = num(r["start"]), num(r["duration"])
        if start is None or dur is None:
            continue
        m = LABEL.match(r["event-label"][1] or "")
        label = m.group(1).strip() if m else (r["event-label"][1] or "?")
        if proc.startswith(target + " ") or proc == target:
            enc = r["encoder-id"][1]
            per_encoder[(label, enc)].append((start, dur))
            events.append(
                {
                    "name": label,
                    "kernel": None,
                    "stack": [label],
                    "track": "GPU (xctrace)",
                    "start_ns": start,
                    "dur_ns": dur,
                    "args": {"channel": r["channel-name"][1], "encoder_id": enc, "cmdbuffer_id": r["cmdbuffer-id"][1]},
                }
            )
        else:
            other[proc or "?"].append((start, dur, r["channel-name"][1]))

    if not events:
        sys.exit(f"no GPU intervals for process {target!r} (processes with GPU work: {sorted(other)})")
    t0 = min(e["start_ns"] for e in events)
    t1 = max(e["start_ns"] + e["dur_ns"] for e in events)
    window = t1 - t0

    for r in tables.get("metal-shader-profiler-intervals", []):
        proc = (r.get("process") or (None, ""))[1] or ""
        if proc.startswith(target):
            events.append(
                {
                    "name": r["name"][1],
                    "kernel": r["name"][1],
                    "stack": [r["name"][1]],
                    "track": "GPU shaders (xctrace)",
                    "start_ns": num(r["start"]),
                    "dur_ns": num(r["duration"]),
                    "args": {"pipeline": (r.get("pso-label") or (None, ""))[1] or ""},
                }
            )

    # Other processes' GPU work overlapping our window: tracks + noise summary.
    other_summary = []
    for proc, iv in other.items():
        busy = 0
        for s, d, ch in iv:
            a, b = max(s, t0), min(s + d, t1)
            if b > a:
                busy += b - a
                events.append(
                    {
                        "name": proc,
                        "stack": [proc],
                        "track": f"GPU other: {proc}",
                        "start_ns": s,
                        "dur_ns": d,
                        "args": {"channel": ch},
                    }
                )
        if busy:
            other_summary.append({"process": proc, "busy_ms": busy / 1e6, "pct_of_window": 100 * busy / window})
    other_summary.sort(key=lambda x: -x["busy_ms"])

    perf_states = defaultdict(int)
    for r in tables.get("gpu-performance-state-intervals", []):
        s, d = num(r["start"]), num(r["duration"])
        st = r["gpu-performance-state"][1]
        a, b = max(s, t0), min(s + d, t1)
        if b > a:
            perf_states[st] += b - a
        events.append(
            {
                "name": st,
                "stack": [st],
                "track": "GPU performance state",
                "start_ns": s,
                "dur_ns": d,
                "args": {"narrative": (r.get("narrative") or (None, ""))[1] or ""},
            }
        )
    thermal = sorted({r["thermal-state"][1] for r in tables.get("device-thermal-state-intervals", [])})
    active = sum(
        min(num(r["start"]) + num(r["duration"]), t1) - max(num(r["start"]), t0)
        for r in tables.get("metal-gpu-state-intervals", [])
        if r["state"][1] == "Active" and num(r["start"]) < t1 and num(r["start"]) + num(r["duration"]) > t0
    )

    counters = []
    for r in tables.get("metal-gpu-counter-intervals", []):
        counters.append({"track": r["name"][1], "t_ns": num(r["start"]), "value": float(r["value"][0] or 0)})

    # Results per encoder label; an encoder split by preemption counts once (summed).
    by_label = defaultdict(list)
    splits = defaultdict(int)
    for (label, _enc), iv in per_encoder.items():
        by_label[label].append(sum(d for _, d in iv) / 1e3)
        splits[label] += len(iv) > 1
    results = []
    for label, us in sorted(by_label.items(), key=lambda kv: -sum(kv[1])):
        results.append(
            {
                "name": label,
                "variant": None,
                "n": len(us),
                "median_us": statistics.median(us),
                "mean_us": statistics.fmean(us),
                "std_us": statistics.stdev(us) if len(us) > 1 else 0.0,
                "min_us": min(us),
                "p10_us": sorted(us)[int(0.1 * (len(us) - 1))],
                "p90_us": sorted(us)[int(0.9 * (len(us) - 1))],
                "method": "xctrace Metal System Trace: GPU time per encoder (intervals split by preemption summed)",
                "extra": {"split_encoders": str(splits[label])},
            }
        )

    noise_path = Path(str(trace) + ".noise.json")
    base = kprof.load(noise_path) if noise_path.exists() else None
    noise = (base or {}).get("noise") or {}
    noise["xctrace"] = {
        "window_ms": window / 1e6,
        "device_gpu_busy_pct": 100 * active / window if window else None,
        "other_gpu_processes": other_summary,
        "gpu_performance_states_ms": {k: v / 1e6 for k, v in perf_states.items()},
        "thermal_states": thermal,
        "split_encoders": sum(splits.values()),
    }
    notes = noise.setdefault("notes", [])
    for o in other_summary[:3]:
        if o["pct_of_window"] >= 1:
            notes.append(f"{o['process']} used the GPU {o['pct_of_window']:.1f}% of our window")
    if sum(splits.values()):
        notes.append(f"{sum(splits.values())} of our encoders were split (preempted by other GPU work)")
    if any(t != "Nominal" for t in thermal):
        notes.append(f"thermal states during the trace: {thermal}")

    meta = (base or {}).get("meta") or {}
    meta.update(tool="xctrace_export", name=target, created=meta.get("created", ""), spec=str(trace.resolve()))
    meta.setdefault("device", {"name": dev.get("model") if dev is not None else "?"})
    meta["params"] = {
        **meta.get("params", {}),
        "template": template or "?",
        "process": target,
        "time_base": "ns since recording start",
    }
    out = {
        "schema": "kprof/1",
        "meta": meta,
        "noise": noise,
        "events": events,
        "counters": counters,
        "results": results,
    }
    path = kprof.save(out, args.out or Path(str(trace).removesuffix(".trace") + ".kprof.json"))

    print(f"{target}: {len(per_encoder)} encoders, {len(events)} events, template '{template}'")
    print(
        kprof.table(
            ["encoder label", "n", "median us", "min", "p90", "split"],
            [
                [
                    r["name"],
                    r["n"],
                    kprof.fmt(r["median_us"], "{:.2f}"),
                    kprof.fmt(r["min_us"], "{:.2f}"),
                    kprof.fmt(r["p90_us"], "{:.2f}"),
                    r["extra"]["split_encoders"],
                ]
                for r in results[:25]
            ],
        )
    )
    if len(results) > 25:
        print(f"... {len(results) - 25} more labels in the file")
    x = noise["xctrace"]
    print(
        f"window {x['window_ms']:.1f} ms, device GPU busy {kprof.fmt(x['device_gpu_busy_pct'])}%, "
        f"perf states {', '.join(f'{k} {v:.1f} ms' for k, v in x['gpu_performance_states_ms'].items()) or '-'}"
    )
    for o in other_summary[:5]:
        print(f"  other GPU: {o['process']} {o['busy_ms']:.2f} ms ({o['pct_of_window']:.1f}% of window)")
    for n in notes:
        print(f"  note: {n}")
    print(f"wrote {path}")

    if args.validate:
        k = kprof.load(args.validate)
        ours = defaultdict(list)
        for e in k["events"]:
            ours[e["name"]].append(e["dur_ns"] / 1e3)
        rows_ = []
        for r in results:
            if r["name"] in ours:
                a, am = statistics.median(ours[r["name"]]), statistics.fmean(ours[r["name"]])
                rows_.append(
                    [
                        r["name"],
                        f"{len(ours[r['name']])}/{r['n']}",
                        kprof.fmt(a, "{:.2f}"),
                        kprof.fmt(r["median_us"], "{:.2f}"),
                        kprof.fmt(100 * (r["median_us"] - a) / a, "{:+.2f}%"),
                        kprof.fmt(am, "{:.2f}"),
                        kprof.fmt(r["mean_us"], "{:.2f}"),
                        kprof.fmt(100 * (r["mean_us"] - am) / am, "{:+.2f}%"),
                    ]
                )
        print("\nvalidation: RegionProfiler timestamps vs xctrace GPU intervals (us per encoder)")
        print(
            kprof.table(["label", "n ours/xc", "median ours", "xctrace", "diff", "mean ours", "xctrace", "diff"], rows_)
        )
        print(
            "RegionProfiler spans encoder start..end, including time other processes preempted it;\n"
            "xctrace sums only the active pieces of a split encoder, so means can differ while medians agree."
        )


if __name__ == "__main__":
    main()
