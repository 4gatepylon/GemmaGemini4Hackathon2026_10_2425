"""Shared helpers for kprof/1 files (the common profiling format, see ../README.md).

Python tools here only post-process: anything that touches Metal is Swift
(../lib). Standard library only.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

PROFILING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROFILING_DIR.parents[1]
try:  # the shared layout (uv run from ios_kernels/); plain python3 falls back to the same default
    from surfaces.paths import DATA_DIR
except ImportError:
    DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / ".context"))
PROF_DIR = DATA_DIR / "prof"
BUILD_DIR = PROFILING_DIR / ".build"


def load(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != "kprof/1":
        sys.exit(f"{path}: not a kprof/1 file (schema={d.get('schema')!r})")
    return d


def save(d, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)
    return path


def stamp():
    return time.strftime("%Y%m%d-%H%M%S")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def default_out(tool, name):
    return PROF_DIR / tool / f"{slug(name)}-{stamp()}.json"


def peaks(path=None):
    p = Path(path) if path else PROF_DIR / "peaks.json"
    try:
        with open(p) as f:
            return json.load(f)
    except OSError:
        return {}


def add_roof_pct(result, pk):
    """Fill pct_peak_bw / pct_peak_flops from measured peaks (see examples/roofline)."""
    if result.get("gbps") is not None and pk.get("bandwidth_gbps"):
        result["pct_peak_bw"] = 100 * result["gbps"] / pk["bandwidth_gbps"]
    if result.get("gflops") is not None and pk.get("flops_gflops"):
        result["pct_peak_flops"] = 100 * result["gflops"] / pk["flops_gflops"]
    return result


def result_key(r):
    v = r.get("variant")
    return r["name"] if v in (None, "default") else f"{r['name']} [{v}]"


def noise_line(noise):
    if not noise:
        return "noise: not recorded"
    s = noise["start"]
    out = f"noise: {s['power']}, thermal {s['thermal']}->{noise['end']['thermal']}"
    i = noise.get("gpu_util_idle")
    if i:
        out += f", GPU idle-baseline {i['mean']:.0f}% (max {i['max']:.0f}%, n={i['n']})"
    out += f", load {s['loadavg'][0]:.1f}"
    if s.get("top_cpu"):
        out += ", top: " + ", ".join(s["top_cpu"][:3])
    return out


def fmt(v, f="{:.1f}"):
    return "-" if v is None else f.format(v)


def table(headers, rows, markdown=False):
    """Plain or markdown table; the first column is left-aligned, the rest right-aligned."""
    rows = [[str(c) for c in r] for r in rows]
    if markdown:
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(cells))

    return "\n".join([line(headers)] + [line(r) for r in rows])
