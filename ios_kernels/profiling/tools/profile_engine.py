#!/usr/bin/env python3
"""Profile the existing Metal engine's kernels next to llama.cpp, per decode token and per prompt chunk.

  uv run --extra profiling profiling/tools/profile_engine.py [--kernel NAME] [--gguf PATH] [--weights PATH]
      [--pos 64] [--t 512] [--skip-llama] [--measure] [--out-dir DIR]

1. Resolves the kernel folder through surfaces.paths (newest by default) and replays its own
   kernels.metal with kbench: ../engine/decode.json (one token at --pos) and ../engine/prefill.json
   (one --t-token chunk), with the engine's dispatch geometry and default tunables. Kernel folders
   are never modified: the source is passed with `kbench --source`.
2. With a GGUF (default: the model's baseline from surfaces.paths), records which ops llama.cpp dispatches
   for the same token / chunk (`llama_ops.py graph`), times each distinct op with test-backend-ops at
   the same shapes (`llama_ops.py export` + `bench`), and multiplies by how often it runs.
3. --measure also runs the real engine binary (needs --weights or $DATA_DIR/models/<kernel weights>)
   and llama-bench, for whole-token times to compare the sums against.
4. Prints and writes a per-category report (../engine/categories.json) and charts of our run to
   $DATA_DIR/prof/engine/<kernel>/.

Method differences are part of the report: our replay streams cold weights (rotated per layer),
while test-backend-ops reruns each op on the same buffers (cache-warm, unfused, no overlap between
ops), so llama.cpp's sum of op times is optimistic next to its measured whole token.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import kprof

TOOLS = Path(__file__).resolve().parent
ENGINE = kprof.PROFILING_DIR / "engine"


def run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def llama_ops(args, *rest):
    cmd = [sys.executable, TOOLS / "llama_ops.py"]
    if args.llama_build:
        cmd += ["--build", args.llama_build]
    return run([*cmd, *rest], stdout=subprocess.PIPE, text=True).stdout


# ---------------------------------------------------------------------------
# ours


def ours_per_rep(k):
    """{step: (us per rep, dispatches per rep)} from a kbench kprof file (GPU track, split steps)."""
    ev = [e for e in k["events"] if e["track"] == "GPU"]
    reps = {e["stack"][0] for e in ev}
    med = {r["name"]: r["median_us"] for r in k["results"]}
    count = defaultdict(int)
    for e in ev:
        count[e["name"]] += 1
    return {n: (med[n] * c / len(reps), c / len(reps)) for n, c in count.items()}


def whole_us(k):
    return next((r["median_us"] for r in k["results"] if r["name"] == "whole"), None)


# ---------------------------------------------------------------------------
# llama.cpp


def op_key(op, dst_ne, src0_ne):
    # The graph names ops by type (GLU), test-backend-ops by variant (SWIGLU, GEGLU, ...).
    op = "GLU" if op.endswith("GLU") else op
    return (op, tuple(dst_ne), tuple(src0_ne))


def llama_per_rep(graph, bench):
    """Sum test-backend-ops times over the ops llama.cpp actually dispatches.
    Returns ([(op, base, us)], unmatched op descriptions)."""
    sys.path.insert(0, str(TOOLS))
    from llama_ops import parse_params

    times = {}
    for r in bench["results"]:
        p = parse_params(r["extra"]["params"])
        if p["ne"] and p["sources"]:
            times.setdefault(op_key(r["extra"]["op"], p["ne"], p["sources"][0][1]), r["median_us"])
    rows, unmatched = [], []
    for o in graph["ops"]:
        key = op_key(o["op"], o.get("ne", []), o["srcs"][0]["ne"] if o["srcs"] else [])
        if key in times:
            rows.append((o["op"], o["base"], times[key]))
        else:
            unmatched.append(f"{o['op']} {o['name']} {o.get('ne')}")
    return rows, unmatched


def categorize_llama(op, base, rules):
    for r in rules:
        if re.fullmatch(r["op"], op) and re.search(r.get("name", ""), base):
            return r["category"]
    return "other"


# ---------------------------------------------------------------------------
# whole-token measurements


def measure_engine(kernel, weights, pos, t):
    manifest = json.loads((kernel / "kernel.json").read_text())
    binary = kernel / manifest["binary"]
    if not binary.exists():
        run([sys.executable, kernel / manifest["build"]])
    out = {}
    r = run(
        [binary, "--weights", weights, "--bench-tg", 2 * pos, "--bench-pp", t, "--reps", 3],
        capture_output=True,
        text=True,
    )
    for kind, n, txt in re.findall(r"^(pp|tg)(\d+): ([\d.]+)", r.stderr, re.M):
        out[kind] = 1e6 / float(txt) * (int(n) if kind == "pp" else 1)  # us per chunk (pp) / per token (tg)
    return out


def measure_llama(args, gguf, pos, t):
    from surfaces.paths import LLAMA_DIR

    dirs = [Path(args.llama_build)] if args.llama_build else [LLAMA_DIR / "build", LLAMA_DIR / "build-tests"]
    bench = next((d / "bin" / "llama-bench" for d in dirs if (d / "bin" / "llama-bench").exists()), None)
    if not bench:
        print("llama-bench not found: skipping its measurement", file=sys.stderr)
        return {}
    r = run(
        [bench, "-m", gguf, "-p", t, "-n", 2 * pos, "-r", 3, "-ngl", 99, "-o", "json"], capture_output=True, text=True
    )
    out = {}
    for row in json.loads(r.stdout):
        if row.get("n_prompt"):
            out["pp"] = 1e6 / row["avg_ts"] * row["n_prompt"]
        elif row.get("n_gen"):
            out["tg"] = 1e6 / row["avg_ts"]
    return out


# ---------------------------------------------------------------------------


def report(kind, ours, llama, cats, whole, measured, unmatched, ours_k, llama_meta):
    unit = "per token" if kind == "tg" else "per chunk"
    by_cat = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # ours us, ours dispatches, llama us, llama dispatches
    for step, (us, n) in ours.items():
        c = by_cat[cats["ours"].get(step, "other")]
        c[0] += us
        c[1] += n
    for op, base, us in llama or []:
        c = by_cat[categorize_llama(op, base, cats["llama"])]
        c[2] += us
        c[3] += 1
    order = [c for c in cats["order"] if c in by_cat] + [c for c in by_cat if c not in cats["order"]]
    rows = []
    for c in order:
        o_us, o_n, l_us, l_n = by_cat[c]
        rows.append(
            [
                c,
                kprof.fmt(o_us / 1e3, "{:.3f}"),
                f"{o_n:g}",
                kprof.fmt(l_us / 1e3 if llama else None, "{:.3f}"),
                f"{l_n}" if llama else "-",
                kprof.fmt(o_us / l_us if llama and l_us and o_us else None, "{:.2f}x"),
            ]
        )
    tot_o = sum(v[0] for v in by_cat.values())
    tot_l = sum(v[2] for v in by_cat.values())
    rows.append(
        [
            "**sum of ops**",
            kprof.fmt(tot_o / 1e3, "{:.3f}"),
            f"{sum(v[1] for v in by_cat.values()):g}",
            kprof.fmt(tot_l / 1e3 if llama else None, "{:.3f}"),
            str(sum(v[3] for v in by_cat.values())) if llama else "-",
            kprof.fmt(tot_o / tot_l if llama and tot_l else None, "{:.2f}x"),
        ]
    )
    lines = [
        f"### {'Decode' if kind == 'tg' else 'Prefill'} ({unit}, ms; ours / llama.cpp: below 1 means we are faster)",
        "",
    ]
    lines.append(
        kprof.table(
            ["category", "ours ms", "ours dispatches", "llama.cpp ms", "llama.cpp dispatches", "ours / llama"],
            rows,
            markdown=True,
        )
    )
    lines.append("")
    if whole:
        lines.append(f"- ours, whole {unit} as one encoder: {whole / 1e3:.3f} ms")
    if measured.get("ours") is not None:
        lines.append(f"- ours, real engine binary: {measured['ours'] / 1e3:.3f} ms {unit}")
    if measured.get("llama") is not None:
        lines.append(f"- llama.cpp, llama-bench: {measured['llama'] / 1e3:.3f} ms {unit}")
    if llama:
        lines.append(
            "- llama.cpp's column sums test-backend-ops times (host wall time, cache-warm, unfused, no overlap; "
            f"fastest of 2 runs) over the ops of its {llama_meta['graph']} graph "
            f"(attention over {llama_meta['n_kv']} KV slots)"
        )
    if unmatched:
        lines.append(
            f"- {len(unmatched)} llama.cpp ops had no matching timing: {', '.join(sorted(set(unmatched))[:6])}"
        )
    lines.append(f"- {kprof.noise_line(ours_k.get('noise'))}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", help="kernel folder name/substring/path (default: newest, via surfaces.paths)")
    ap.add_argument("--gguf", help="llama.cpp model (default: the kernel's model baseline, surfaces.paths.baseline)")
    ap.add_argument(
        "--weights", help="our engine's weights for --measure (default $DATA_DIR/models/<kernel.json weights>)"
    )
    ap.add_argument("--llama-build", help="llama.cpp build dir with the test tools (see llama_ops.py)")
    ap.add_argument("--pos", type=int, default=64, help="decode position (tg128 from an empty cache averages 64)")
    ap.add_argument("--t", type=int, default=512, help="prefill chunk length (512 = pp512)")
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--measure", action="store_true", help="also time the real engine binary and llama-bench")
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    from surfaces.paths import MODELS_DIR, baseline, resolve_kernel

    kernel = resolve_kernel(args.kernel)
    source = kernel / "kernels.metal"
    out_dir = Path(args.out_dir or kprof.PROF_DIR / "engine" / kernel.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    kbench = kprof.BUILD_DIR / "kbench"
    if not kbench.exists():
        sys.exit(f"{kbench} not found: run `uv run profiling/build.py`")
    cats = json.loads((ENGINE / "categories.json").read_text())

    ours = {}
    for kind, spec, var in (("tg", "decode.json", f"POS={args.pos}"), ("pp", "prefill.json", f"T={args.t}")):
        path = out_dir / f"{kind}.json"
        run([kbench, ENGINE / spec, "--source", source, "--set", var, "--out", path], stdout=subprocess.DEVNULL)
        ours[kind] = kprof.load(path)

    llama, graphs, unmatched = {}, {}, {}
    try:
        default_gguf = baseline("gguf", kernel)
    except KeyError:  # no llama.cpp baseline registered for this model
        default_gguf = Path("/nonexistent")
    gguf = Path(args.gguf) if args.gguf else default_gguf
    if not args.skip_llama and gguf.exists():
        for kind, flag in (("tg", ["--gen", "1"]), ("pp", ["--prompt", str(args.t)])):
            gpath = out_dir / f"llama-{kind}-graph.json"
            llama_ops(args, "graph", "--model", gguf, *flag, "--out", gpath)
            graphs[kind] = json.loads(gpath.read_text())
            ops_txt = out_dir / f"llama-{kind}-ops.txt"
            llama_ops(args, "export", "--model", gguf, "--ctx", graphs[kind]["n_kv"] or 4096, "--out", ops_txt)
            bpath = out_dir / f"llama-{kind}-ops.json"
            llama_ops(args, "bench", ops_txt, "--out", bpath, "--runs", "2")
            llama[kind], unmatched[kind] = llama_per_rep(graphs[kind], kprof.load(bpath))
    elif not args.skip_llama:
        print(f"no GGUF at {gguf}: skipping llama.cpp (pass --gguf)", file=sys.stderr)

    measured = {"tg": {}, "pp": {}}
    if args.measure:
        manifest = json.loads((kernel / "kernel.json").read_text())
        weights = Path(args.weights or MODELS_DIR / manifest["weights"])
        if weights.exists():
            for kind, us in measure_engine(kernel, weights, args.pos, args.t).items():
                measured[kind]["ours"] = us
        if gguf.exists():
            for kind, us in measure_llama(args, gguf, args.pos, args.t).items():
                measured[kind]["llama"] = us

    sections = [f"## Engine profile: {kernel.name}", ""]
    for kind in ("tg", "pp"):
        sections.append(
            report(
                kind,
                ours_per_rep(ours[kind]),
                llama.get(kind),
                cats,
                whole_us(ours[kind]),
                measured[kind],
                unmatched.get(kind),
                ours[kind],
                graphs.get(kind),
            )
        )
        sections.append("")
    text = "\n".join(sections)
    (out_dir / "report.md").write_text(text)
    print(text)

    viz = [sys.executable, TOOLS / "kviz.py"]
    try:
        import matplotlib  # noqa: F401

        run([*viz, "share", out_dir / "tg.json", "--out", out_dir / "tg.share.png"], stdout=subprocess.DEVNULL)
        run([*viz, "gantt", out_dir / "tg.json", "--out", out_dir / "tg.gantt.png"], stdout=subprocess.DEVNULL)
        run([*viz, "flame", out_dir / "pp.json", "--out", out_dir / "pp.flame.png"], stdout=subprocess.DEVNULL)
    except ImportError:
        print("(charts skipped: run with `uv run --extra profiling`)", file=sys.stderr)
    print(f"\nwrote {out_dir}/report.md and the kprof files and charts next to it")


if __name__ == "__main__":
    main()
