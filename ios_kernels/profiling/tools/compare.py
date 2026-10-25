#!/usr/bin/env python3
"""Side-by-side comparison of the results in two kprof files.

  compare.py A.json B.json [--pair 'A name=B name']... [--map MAP.json] [--labels ours,llama] [--md]

Rows are matched by name (results with a variant are keyed "name [variant]"),
or explicitly with --pair / --map (a JSON object {"A name": "B name"}).
"speedup" is B time / A time, so > 1 means A is faster. Both files' methods
and noise are printed, because a comparison is only as fair as its methods.

Typical uses: ours (kbench) vs llama.cpp (llama_ops.py bench); before vs after
an optimization; the same spec on a quiet vs a busy machine.
"""

import argparse
import json

import kprof


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pair", action="append", default=[], help="'A name=B name' (repeatable)")
    ap.add_argument("--map", help="JSON file {A name: B name}")
    ap.add_argument("--labels", default="A,B", help="column labels, e.g. ours,llama")
    ap.add_argument("--md", action="store_true", help="markdown table")
    args = ap.parse_args()

    A, B = kprof.load(args.a), kprof.load(args.b)
    la, lb = args.labels.split(",")
    ra = {kprof.result_key(r): r for r in A["results"]}
    rb = {kprof.result_key(r): r for r in B["results"]}
    pairs = []
    if args.map:
        with open(args.map) as f:
            pairs += list(json.load(f).items())
    for p in args.pair:
        x, _, y = p.partition("=")
        pairs.append((x.strip(), y.strip()))
    if not pairs:
        pairs = [(k, k) for k in ra if k in rb]
    if not pairs:
        raise SystemExit(
            f"no matching result names; use --pair 'A name=B name' or --map\n  {la}: {sorted(ra)}\n  {lb}: {sorted(rb)}"
        )

    rows = []
    for x, y in pairs:
        if x not in ra or y not in rb:
            missing = x if x not in ra else y
            raise SystemExit(f"no result named {missing!r} in {la if x not in ra else lb}")
        a, b = ra[x], rb[y]
        name = x if x == y else f"{x} vs {y}"
        rows.append(
            [
                name,
                kprof.fmt(a["median_us"], "{:.2f}"),
                kprof.fmt(b["median_us"], "{:.2f}"),
                kprof.fmt(b["median_us"] / a["median_us"], "{:.2f}x"),
                kprof.fmt(a.get("gbps")),
                kprof.fmt(b.get("gbps")),
                kprof.fmt(a.get("gflops")),
                kprof.fmt(b.get("gflops")),
            ]
        )
    print(
        kprof.table(
            [
                "result",
                f"{la} us",
                f"{lb} us",
                f"speedup ({la})",
                f"{la} GB/s",
                f"{lb} GB/s",
                f"{la} GFLOP/s",
                f"{lb} GFLOP/s",
            ],
            rows,
            markdown=args.md,
        )
    )
    print()
    for label, d, rs in ((la, A, ra), (lb, B, rb)):
        methods = sorted({rs[x if label == la else y]["method"] for x, y in pairs})
        m = d["meta"]
        print(f"{label}: {m['tool']} '{m['name']}' ({m['created']}, git {m.get('git')})")
        for meth in methods:
            print(f"  method: {meth}")
        print(f"  {kprof.noise_line(d.get('noise'))}")


if __name__ == "__main__":
    main()
