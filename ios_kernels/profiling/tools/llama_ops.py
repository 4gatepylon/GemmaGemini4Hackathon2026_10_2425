#!/usr/bin/env python3
"""llama.cpp per-op baseline, in kprof format (see ../README.md, "llama.cpp baseline").

  llama_ops.py build   [--src DIR]                      build test-backend-ops, test-export-graph-ops, llama-bench
  llama_ops.py export  --model GGUF [--ctx N] [--out OPS.txt]   the model's unique graph ops (pp and tg graphs)
  llama_ops.py mul-mat --shape M,N,K [...] [--type f16] [--out OPS.txt]   custom MUL_MAT cases
  llama_ops.py list    OPS.txt                          human-readable listing of a test file
  llama_ops.py bench   OPS.txt [--grep REGEX] [--out JSON] [--name NAME]
  llama_ops.py bench   --op MUL_MAT [--params REGEX]    llama.cpp's built-in perf cases instead

`bench` runs `test-backend-ops perf -b MTL0 ... --output sql` under knoise (so
the noise is recorded) and converts every row into a kprof result. Its timing
method differs from kbench's: test-backend-ops averages many back-to-back
runs of one op on the *same* buffers, so working sets that fit in the GPU's
caches are measured cache-warm (see the README).

Locations: --src/$LLAMA_SRC (default: the shared checkout, surfaces.paths.LLAMA_DIR), and
--build/$LLAMA_BUILD (default: <src>/build-tests, next to the benchmark build in <src>/build).
"""

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import kprof

TYPE_SIZE = {"f32": 4, "f16": 2, "bf16": 2}  # block size 1 types only (custom MUL_MAT lines)
try:
    from surfaces.paths import LLAMA_DIR
except ImportError:
    LLAMA_DIR = kprof.DATA_DIR / "deps" / "llama.cpp"


def build_dir(args):
    if args.build or os.environ.get("LLAMA_BUILD"):
        return Path(args.build or os.environ["LLAMA_BUILD"])
    return Path(args.src or os.environ.get("LLAMA_SRC") or LLAMA_DIR) / "build-tests"


def src_dir(args):
    if args.src or os.environ.get("LLAMA_SRC"):
        return Path(args.src or os.environ["LLAMA_SRC"])
    cache = build_dir(args) / "CMakeCache.txt"
    if cache.exists():  # a build dir knows its source
        m = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache.read_text(), re.M)
        if m:
            return Path(m.group(1))
    return Path(LLAMA_DIR)


def tool(args, name):
    p = build_dir(args) / "bin" / name
    if not p.exists():
        sys.exit(f"{p} not found: run `uv run profiling/tools/llama_ops.py build` first")
    return p


def ggml_enums(args):
    """(op name by id, type id by name) parsed from ggml.h, so test files match this llama.cpp build."""
    h = src_dir(args) / "ggml" / "include" / "ggml.h"
    if not h.exists():
        sys.exit(f"{h} not found (set --src or $LLAMA_SRC)")
    text = h.read_text()
    body = re.search(r"enum ggml_op \{(.*?)\};", text, re.S).group(1)
    ops = re.findall(r"^\s*GGML_OP_(\w+)\s*(?:=\s*0\s*)?,", body, re.M)  # GGML_OP_NONE = 0 first
    assert ops and ops[0] == "NONE", "unexpected enum ggml_op layout in ggml.h"
    tbody = re.search(r"enum ggml_type \{(.*?)\};", text, re.S).group(1)
    types = {n.lower(): int(v) for n, v in re.findall(r"GGML_TYPE_(\w+)\s*=\s*(\d+)", tbody)}
    return ops, types


# ---------------------------------------------------------------------------


def cmd_build(args):
    src, bld = src_dir(args), build_dir(args)
    if not src.exists():
        print(f"cloning llama.cpp into {src}")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(src)], check=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(src),
            "-B",
            str(bld),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_METAL=ON",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=ON",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(bld),
            "-j",
            "8",
            "--target",
            "test-backend-ops",
            "test-export-graph-ops",
            "llama-bench",
        ],
        check=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(src), "log", "-1", "--format=%h"], capture_output=True, text=True
    ).stdout.strip()
    print(f"built llama.cpp {sha} tools in {bld}/bin")


def cmd_export(args):
    out = Path(args.out or kprof.PROF_DIR / "llama_ops" / f"{Path(args.model).stem}-ctx{args.ctx}-ops.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    # n_ctx sets the KV length the attention ops are exported with (the model default can be 131072)
    subprocess.run(
        [str(tool(args, "test-export-graph-ops")), "-m", args.model, "-o", str(out), "-c", str(args.ctx)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"wrote {out} ({len(out.read_text().splitlines())} ops)")


def cmd_mul_mat(args):
    ops, types = ggml_enums(args)
    op = ops.index("MUL_MAT")
    ta, sa = types[args.type], TYPE_SIZE[args.type]
    lines = []
    for shape in args.shape:
        M, N, K = (int(x) for x in shape.split(","))
        name = f"mm_{args.type}_{M}x{N}x{K}"
        # dst f32 [M, N]; src0 = weights [K, M] (type_a); src1 = activations f32 [K, N]
        lines.append(
            f"{op} 0 {M} {N} 1 1 16 " + "0 " * 16 + f"2 {ta} {K} {M} 1 1 {sa} {sa * K} {sa * K * M} {sa * K * M} "
            f"0 {K} {N} 1 1 4 {4 * K} {4 * K * N} {4 * K * N} {name}"
        )
    out = Path(args.out or kprof.PROF_DIR / "llama_ops" / "mul_mat.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} MUL_MAT cases)")


def cmd_list(args):
    ops, types = ggml_enums(args)
    tname = {v: k for k, v in types.items()}
    for line in Path(args.file).read_text().splitlines():
        t = line.split()
        if not t:
            continue
        op, ty, ne = int(t[0]), int(t[1]), t[2:6]
        npar = int(t[6])
        i = 7 + npar
        nsrc = int(t[i])
        i += 1
        srcs = []
        for _ in range(nsrc):
            srcs.append(f"{tname.get(int(t[i]), t[i])}[{','.join(t[i + 1 : i + 5])}]")
            i += 9
        name = t[i] if i < len(t) else "-"
        print(f"{ops[op]:12s} {name:16s} -> {tname.get(ty, ty)}[{','.join(ne)}]  <- {', '.join(srcs)}")


# ---------------------------------------------------------------------------


def parse_params(p):
    """'name=Qcur-0,type=f32,ne=[2048,1,1,1],...,sources=f16[2048,2048,1,1],f32[2048,1,1,1]'"""
    out = {}
    m = re.search(r"(?:^|,)name=([^,]*)", p)
    out["name"] = m.group(1) if m else ""
    m = re.search(r"(?:^|,)ne=\[([\d,]+)\]", p)
    out["ne"] = [int(x) for x in m.group(1).split(",")] if m else []
    m = re.search(r"sources=(.*)$", p)
    out["sources"] = (
        [(t, [int(x) for x in d.split(",")]) for t, d in re.findall(r"(\w+)\[([\d,]+)\]", m.group(1))] if m else []
    )
    return out


def describe(op, p):
    """Short name and FLOPs (MUL_MAT only) for a test-backend-ops row."""
    pp = parse_params(p)
    tag = pp["name"] or ""
    if op == "MUL_MAT" and pp["ne"] and len(pp["sources"]) >= 2:
        M, N = pp["ne"][0], pp["ne"][1]
        ta, (K, *_) = pp["sources"][0][0], pp["sources"][0][1]
        return f"MUL_MAT {tag} {ta} M={M} N={N} K={K}".replace("  ", " "), 2.0 * M * N * K
    shape = "x".join(str(x) for x in pp["ne"] if x != 1) or "1"
    return f"{op} {tag} [{shape}]".replace("  ", " "), None


def cmd_bench(args):
    tbo = tool(args, "test-backend-ops")
    cmd = [str(tbo), "perf", "-b", "MTL0", "--output", "sql"]
    if args.file:
        cmd += ["--test-file", args.file]
    elif args.op:
        cmd += ["-o", args.op] + (["-p", args.params] if args.params else [])
    else:
        sys.exit("bench needs a test file (from export/mul-mat) or --op")
    knoise = kprof.BUILD_DIR / "knoise"
    if not knoise.exists():
        sys.exit(f"{knoise} not found: run `uv run profiling/build.py`")
    name = args.name or (Path(args.file).stem if args.file else args.op)
    with tempfile.TemporaryDirectory() as tmp:
        noise_path = Path(tmp) / "noise.json"
        print(f"running: {' '.join(cmd)}", file=sys.stderr)
        r = subprocess.run(
            [str(knoise), "--out", str(noise_path), "--name", "test-backend-ops", "--", *cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if not args.verbose else None,
            text=True,
        )
        if r.returncode != 0:
            sys.exit(f"test-backend-ops failed (exit {r.returncode})")
        k = kprof.load(noise_path)

    # stdout is SQL, but keep only the statements in case anything else gets printed there
    create = re.findall(r"(?s)CREATE TABLE.*?\);", r.stdout)[:1]
    inserts = re.findall(r"^INSERT INTO .*;$", r.stdout, re.M)
    if not create:
        sys.exit("test-backend-ops printed no SQL (try --verbose)")
    db = sqlite3.connect(":memory:")
    db.executescript("\n".join(create + inserts))
    rows = db.execute(
        "SELECT op_name, op_params, time_us, bandwidth_gb_s, memory_kb, n_runs, build_commit, supported "
        "FROM test_backend_ops WHERE test_mode = 'perf'"
    ).fetchall()
    pk = kprof.peaks(args.peaks)
    grep = re.compile(args.grep) if args.grep else None
    results, commit = [], ""
    for op, params, us, gbps, mem_kb, n_runs, row_commit, supported in rows:
        commit = row_commit
        if not supported or us is None or us <= 0:
            continue
        rname, flops = describe(op, params)
        if grep and not (grep.search(rname) or grep.search(params)):
            continue
        res = {
            "name": rname,
            "variant": None,
            "n": n_runs,
            "median_us": us,
            "mean_us": us,
            "std_us": 0.0,
            "min_us": us,
            "p10_us": us,
            "p90_us": us,
            "bytes": mem_kb * 1024,
            "flops": flops,
            "gbps": gbps,
            "gflops": flops / (us * 1e3) if flops else None,
            "method": f"test-backend-ops perf: mean over {n_runs} back-to-back runs of one op on the same buffers",
            "extra": {"op": op, "params": params},
        }
        results.append(kprof.add_roof_pct(res, pk))

    k["meta"].update(tool="llama_ops", name=name)
    k["meta"]["params"] = {
        **k["meta"].get("params", {}),
        "llama_cpp": commit or "?",
        "backend": "MTL0",
        "test_file": str(Path(args.file).resolve()) if args.file else "",
        "op": args.op or "",
    }
    k["results"] = results
    out = kprof.save(k, args.out or kprof.default_out("llama_ops", name))

    print(f"llama.cpp {commit} test-backend-ops perf ({len(results)} ops)")
    print(
        kprof.table(
            ["op", "us/run", "runs", "GB/s", "%bw", "GFLOP/s", "%flop"],
            [
                [
                    r["name"],
                    kprof.fmt(r["median_us"], "{:.2f}"),
                    r["n"],
                    kprof.fmt(r["gbps"]),
                    kprof.fmt(r.get("pct_peak_bw"), "{:.0f}"),
                    kprof.fmt(r["gflops"]),
                    kprof.fmt(r.get("pct_peak_flops"), "{:.0f}"),
                ]
                for r in results
            ],
        )
    )
    print(kprof.noise_line(k.get("noise")))
    for n in (k.get("noise") or {}).get("notes", []):
        print(f"  note: {n}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--build", help="llama.cpp build dir for the test tools (default $LLAMA_BUILD or <src>/build-tests)"
    )
    ap.add_argument("--src", help="llama.cpp source dir (default $LLAMA_SRC or the shared checkout)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("export")
    e.add_argument("--model", required=True)
    e.add_argument("--out")
    e.add_argument(
        "--ctx", type=int, default=4096, help="context length for the exported graph (KV length of attention ops)"
    )
    m = sub.add_parser("mul-mat")
    m.add_argument("--shape", action="append", required=True, help="M,N,K (repeatable)")
    m.add_argument("--type", default="f16", choices=sorted(TYPE_SIZE))
    m.add_argument("--out")
    li = sub.add_parser("list")
    li.add_argument("file")
    b = sub.add_parser("bench")
    b.add_argument("file", nargs="?")
    b.add_argument("--op")
    b.add_argument("--params")
    b.add_argument("--grep", help="keep only rows whose name/params match")
    b.add_argument("--out")
    b.add_argument("--name")
    b.add_argument("--peaks")
    b.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    {"build": cmd_build, "export": cmd_export, "mul-mat": cmd_mul_mat, "list": cmd_list, "bench": cmd_bench}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
