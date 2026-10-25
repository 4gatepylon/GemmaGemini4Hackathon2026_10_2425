#!/usr/bin/env python3
"""Charts from kprof files: interactive (Perfetto, speedscope) and static PNGs.

  kviz.py perfetto FILE... [--out TRACE.json] [--absolute] [--open]  Chrome Trace Event JSON -> https://ui.perfetto.dev
  kviz.py folded   FILE [--track T] [--keep-indices]        folded stacks -> https://www.speedscope.app, flamegraph.pl
  kviz.py gantt    FILE... [--frame 'rep 0' | --start-ms A --end-ms B] [--track T]... [--dark]   timeline PNG
                   (--start-ms/--end-ms count from the first event on the main GPU track)
  kviz.py flame    FILE [--track T] [--dark]                 aggregate icicle (flame graph) PNG
  kviz.py share    FILE [--track T] [--dark]                 where the time goes: one bar per kernel, PNG
  kviz.py table    FILE [--md]                               the results as a table (the charts' table view)

Events nest by their `stack` (e.g. rep 0 > layer 3 > attention > qkv). The
leaves (kernels) get categorical colors in a fixed order: first appearance in
time. Past 8 kernels the smallest fold into "Other". Parent frames are
neutral. Outputs default to the input path with a new suffix.

Requires matplotlib for the PNG commands (perfetto/folded/table are stdlib only).
"""

import argparse
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import kprof

# Validated categorical palette (8 slots, fixed order; dataviz reference palette,
# checked with validate_palette.js: CVD/normal-vision separation pass in both modes).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "parent": "#e9e8e2",
        "other": "#a9a79f",
        "foreign": "#52514e",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "parent": "#2c2c2a",
        "other": "#5f5e5a",
        "foreign": "#c3c2b7",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}


def out_path(args, inp, suffix):
    return Path(args.out) if args.out else Path(str(inp).removesuffix(".json") + suffix)


def collapse(frame, keep):
    """'layer 7' -> 'layer' so iterations aggregate (unless --keep-indices)."""
    return frame if keep else re.sub(r"\s+\d+$", "", frame)


def gpu_track(events, want=None):
    tracks = OrderedDict((e["track"], None) for e in events)
    if want:
        return [t for t in tracks if t in want]
    if "GPU" in tracks:
        return ["GPU"]
    gpu = [t for t in tracks if t.startswith("GPU") and "other" not in t and "state" not in t]
    return gpu[:1] or list(tracks)[:1]


# ---------------------------------------------------------------------------
# perfetto / folded / table (stdlib)


def cmd_perfetto(args):
    trace, pid = [], 0
    for path in args.files:
        d = kprof.load(path)
        pid += 1
        ev = [e for e in d["events"] if e.get("dur_ns") is not None]
        if not ev:
            continue
        base = 0 if args.absolute else min(e["start_ns"] for e in ev)

        def us(ns, base=base):
            return (ns - base) / 1e3

        label = f"{Path(path).stem}: {d['meta'].get('tool')} '{d['meta'].get('name')}'"
        trace.append({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": label}})
        # Flat tracks whose slices overlap (e.g. another process's vertex and fragment work)
        # are split into lanes: Perfetto drops partially overlapping slices on one thread.
        lane_of = {}
        for track in OrderedDict((e["track"], None) for e in ev):
            te = sorted((e for e in ev if e["track"] == track), key=lambda e: e["start_ns"])
            if all(len(e.get("stack") or [1]) == 1 for e in te):
                ends = []
                for e in te:
                    i = next((i for i, end in enumerate(ends) if end <= e["start_ns"]), len(ends))
                    if i == len(ends):
                        ends.append(e["start_ns"] + e["dur_ns"])
                    else:
                        ends[i] = e["start_ns"] + e["dur_ns"]
                    lane_of[id(e)] = f"{track} (lane {i + 1})" if i else track

        def trk(e, lane_of=lane_of):
            return lane_of.get(id(e), e["track"])

        tids = OrderedDict()
        for e in ev:
            tids.setdefault(trk(e), len(tids) + 1)
        for t, tid in tids.items():
            trace.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": t}})
            trace.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": tid}})
        # Parent slices: the span of every stack prefix, so Perfetto nests by containment.
        spans = {}
        for e in ev:
            st = e.get("stack") or [e["name"]]
            for k in range(1, len(st)):
                key = (e["track"], tuple(st[:k]))
                a, b = e["start_ns"], e["start_ns"] + e["dur_ns"]
                lo, hi = spans.get(key, (a, b))
                spans[key] = (min(lo, a), max(hi, b))
        for (t, st), (a, b) in spans.items():
            trace.append(
                {
                    "ph": "X",
                    "name": st[-1],
                    "cat": "frame",
                    "pid": pid,
                    "tid": tids[t],
                    "ts": us(a),
                    "dur": (b - a) / 1e3,
                }
            )
        for e in ev:
            a = {k: v for k, v in (e.get("args") or {}).items()}
            for k in ("kernel", "variant", "bytes", "flops"):
                if e.get(k) is not None:
                    a[k] = e[k]
            if e.get("bytes"):
                a["GB/s"] = round(e["bytes"] / e["dur_ns"], 2) if e["dur_ns"] else None
            if e.get("flops"):
                a["GFLOP/s"] = round(e["flops"] / e["dur_ns"], 2) if e["dur_ns"] else None
            trace.append(
                {
                    "ph": "X",
                    "name": e["name"],
                    "cat": e["track"],
                    "pid": pid,
                    "tid": tids[trk(e)],
                    "ts": us(e["start_ns"]),
                    "dur": e["dur_ns"] / 1e3,
                    "args": a,
                }
            )
        for c in d.get("counters", []):
            if c.get("t_ns", 0) >= base:
                trace.append(
                    {"ph": "C", "name": c["track"], "pid": pid, "ts": us(c["t_ns"]), "args": {"value": c["value"]}}
                )
    out = out_path(args, args.files[0], ".perfetto.json")
    with open(out, "w") as f:
        json.dump({"traceEvents": trace, "displayTimeUnit": "ns"}, f)
    print(
        f"wrote {out} ({len(trace)} trace events): open it at https://ui.perfetto.dev (Open trace file); "
        "it is processed locally in the browser"
    )
    if args.open:
        serve_once(out)


def serve_once(path, port=9001):
    """Open the trace in the Perfetto UI the way Perfetto's own open_trace_in_ui does:
    serve it once on 127.0.0.1:9001 with CORS and point ui.perfetto.dev at that URL.
    (Chrome may ask to allow access to the local network: allow it.)"""
    import http.server
    import subprocess
    import urllib.parse

    path = Path(path).resolve()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

        def do_POST(self):  # the UI probes :9001 for trace_processor's RPC; say no
            self.send_response(404)
            self.end_headers()

        def do_GET(self):
            if self.path.lstrip("/") != path.name:
                self.send_response(404)
                self.end_headers()
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.server.served = True

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    srv.served = False
    url = "https://ui.perfetto.dev/#!/?url=" + urllib.parse.quote(f"http://127.0.0.1:{port}/{path.name}", safe=":/")
    subprocess.run(["open", url])
    print(f"serving {path.name} on 127.0.0.1:{port} until the UI fetches it (Ctrl-C to stop)")
    while not srv.served:
        srv.handle_request()


def cmd_folded(args):
    d = kprof.load(args.file)
    tracks = gpu_track(d["events"], args.track)
    agg = defaultdict(int)
    for e in d["events"]:
        if e["track"] in tracks:
            st = [collapse(f, args.keep_indices) for f in (e.get("stack") or [e["name"]])]
            agg[";".join(s.replace(";", ",") for s in st)] += e["dur_ns"]
    out = out_path(args, args.file, ".folded")
    with open(out, "w") as f:
        for k, v in sorted(agg.items()):
            f.write(f"{k} {v}\n")
    print(f"wrote {out} ({len(agg)} stacks, values in ns): drop it on https://www.speedscope.app or run flamegraph.pl")


def cmd_table(args):
    d = kprof.load(args.file)
    rows = [
        [
            kprof.result_key(r),
            r["n"],
            kprof.fmt(r["median_us"], "{:.2f}"),
            kprof.fmt(r.get("p90_us"), "{:.2f}"),
            kprof.fmt(r.get("gbps")),
            kprof.fmt(r.get("pct_peak_bw"), "{:.0f}"),
            kprof.fmt(r.get("gflops")),
            kprof.fmt(r.get("pct_peak_flops"), "{:.0f}"),
        ]
        for r in d["results"]
    ]
    print(
        kprof.table(["result", "n", "median us", "p90 us", "GB/s", "%bw", "GFLOP/s", "%flop"], rows, markdown=args.md)
    )
    print(f"\n{kprof.noise_line(d.get('noise'))}")


# ---------------------------------------------------------------------------
# static PNGs (matplotlib)


def mpl(theme):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink2"],
            "xtick.color": t["muted"],
            "ytick.color": t["ink2"],
            "axes.edgecolor": t["axis"],
            "axes.linewidth": 0.8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )
    return plt, t


def assign_colors(names_in_order, totals, t):
    """Fixed-order categorical slots by first appearance; > 8 kernels fold the smallest into Other."""
    names = list(OrderedDict.fromkeys(names_in_order))
    if len(names) > 8:
        keep = set(sorted(names, key=lambda n: -totals.get(n, 0))[:7])
        names_kept = [n for n in names if n in keep]
        cmap = {n: t["series"][i] for i, n in enumerate(names_kept)}
        for n in names:
            cmap.setdefault(n, t["other"])
        return cmap, [*names_kept, "Other"]
    return {n: t["series"][i] for i, n in enumerate(names)}, names


def text_on(fill):
    r, g, b = (int(fill[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.55 else "#ffffff"


def legend(fig, cmap, names, t):
    """Figure legend along the bottom edge (identity is never color-only: slices also carry labels)."""
    from matplotlib.patches import Patch

    if not names:
        return
    handles = [
        Patch(facecolor=cmap.get(n, t["other"]) if n != "Other" else t["other"], edgecolor="none", label=n)
        for n in names
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        ncol=min(len(names), 8),
        frameon=False,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.4,
        labelcolor=t["ink2"],
        fontsize=8,
    )


def level_names(rows, spans):
    """Name each (track, depth) row: 'rep', 'layer', ... from its frames, 'kernel' for leaf rows."""
    frames = defaultdict(set)
    leafy = defaultdict(bool)
    for key, v in spans.items():
        track, st = key[0], key[1]
        frames[(track, len(st) - 1)].add(re.sub(r"\s+\d+$", "", st[-1]))
        leafy[(track, len(st) - 1)] |= v[2]
    out = []
    for track, dep in rows:
        f = frames[(track, dep)]
        lvl = "kernel" if leafy[(track, dep)] and len(f) > 2 else "/".join(sorted(f)) if len(f) <= 2 else "group"
        label = (track if lvl in track or track.startswith("GPU other") else f"{track} · {lvl}") if dep == 0 else lvl
        out.append(label if len(label) <= 34 else label[:33] + "…")
    return out


def title(fig, d, what, t):
    m = d["meta"]
    fig.text(0.01, 0.985, f"{what}: {m.get('name')}", ha="left", va="top", fontsize=11, color=t["ink"], weight="bold")
    dev = (m.get("device") or {}).get("name", "")
    sub = f"{m.get('tool')} on {dev}, {m.get('created', '')[:19]}  |  {kprof.noise_line(d.get('noise'))}"
    fig.text(
        0.01, 0.985 - 0.035 * (8 / fig.get_figheight()), sub[:220], ha="left", va="top", fontsize=7.5, color=t["ink2"]
    )


def cmd_gantt(args):
    plt, t = mpl(args.theme)
    files = [kprof.load(p) for p in args.files]
    panels = []
    for d in files:
        ev = [e for e in d["events"] if e.get("dur_ns")]
        if args.frame:
            sel = [e for e in ev if (e.get("stack") or [None])[0] == args.frame]
        elif args.start_ms is None and any(re.fullmatch(r"rep \d+", (e.get("stack") or [""])[0]) for e in ev):
            sel = [e for e in ev if (e.get("stack") or [None])[0] == "rep 0"]
        else:
            sel = ev
        if not sel:
            sys.exit(f"{d['meta']['name']}: no events for frame {args.frame!r}")
        main = gpu_track(ev)
        # --start-ms/--end-ms count from the first event on the main GPU track
        t0 = min(e["start_ns"] for e in ev if e["track"] in main)
        a = t0 + (args.start_ms or 0) * 1e6 if args.start_ms is not None else min(e["start_ns"] for e in sel)
        b = t0 + args.end_ms * 1e6 if args.end_ms is not None else max(e["start_ns"] + e["dur_ns"] for e in sel)
        tracks = args.track or list(OrderedDict((e["track"], None) for e in ev))
        win = [e for e in ev if e["track"] in tracks and e["start_ns"] < b and e["start_ns"] + e["dur_ns"] > a]
        if not win:
            sys.exit(
                f"{d['meta']['name']}: no events in the window ({(a - t0) / 1e6:.3f}..{(b - t0) / 1e6:.3f} ms "
                f"from the first '{main[0]}' event)"
            )
        panels.append((d, a, b, win))

    # Color follows the kernel: slots by first appearance over the whole file(s), so zooming
    # or windowing never repaints a kernel. The legend lists only kernels in view.
    leaves, totals, in_view = [], defaultdict(float), set()
    for d, _, _, win in panels:
        main = gpu_track(d["events"])
        for e in sorted((e for e in d["events"] if e["track"] in main), key=lambda e: e["start_ns"]):
            leaves.append(e["name"])
            totals[e["name"]] += e["dur_ns"] or 0
        in_view |= {e["name"] for e in win if e["track"] in main}
    cmap, legend_names = assign_colors(leaves, totals, t)
    legend_names = [n for n in legend_names if n in in_view or n == "Other"]

    # rows: one per (track, depth)
    layouts = []
    for d, a, b, win in panels:
        rows = OrderedDict()
        spans = {}  # parents: one span per (track, stack prefix); leaves: one per event
        for i, e in enumerate(win):
            st = e.get("stack") or [e["name"]]
            s, f = e["start_ns"], e["start_ns"] + e["dur_ns"]
            for k in range(1, len(st)):
                key = (e["track"], tuple(st[:k]))
                lo, hi = spans.get(key, (s, f))[:2]
                spans[key] = (min(lo, s), max(hi, f), False, st[k - 1])
                rows.setdefault((e["track"], k - 1), None)
            spans[(e["track"], tuple(st), i)] = (s, f, True, e["name"])
            rows.setdefault((e["track"], len(st) - 1), None)
        layouts.append((d, a, b, list(rows), spans))

    nrows = sum(len(r) for _, _, _, r, _ in layouts)
    fig_h = 1.5 + 0.3 * nrows + 0.55 * len(layouts) + 0.5
    fig, axes = plt.subplots(
        len(layouts),
        1,
        figsize=(14, fig_h),
        squeeze=False,
        gridspec_kw={"height_ratios": [max(len(r), 1) for _, _, _, r, _ in layouts]},
    )
    span_ms = max((b - a) / 1e6 for _, a, b, _, _ in layouts)
    fig.subplots_adjust(left=0.17, right=0.99, top=1 - 0.75 / fig_h, bottom=1.05 / fig_h, hspace=0.55)
    for ax, (d, a, b, rows, spans) in zip(axes[:, 0], layouts, strict=True):
        ax.set_xlim(0, span_ms)
        ax.set_ylim(len(rows), 0)
        px_per_ms = ax.get_window_extent().width / span_ms
        gap = 2 / px_per_ms
        for key, (s, f, leaf, name) in spans.items():
            track, st = key[0], key[1]
            y = rows.index((track, len(st) - 1))
            x0, w = (s - a) / 1e6, (f - s) / 1e6
            if x0 + w < 0 or x0 > span_ms:
                continue
            main = gpu_track(d["events"])
            fill = cmap.get(name, t["other"]) if leaf and track in main else t["parent"]
            if track.startswith("GPU other"):
                fill = t["foreign"]  # other processes: distinct from the legend's "Other" kernels
            ww = w - gap if w * px_per_ms > 4 else max(w, 0.6 / px_per_ms)
            ax.add_patch(plt.Rectangle((x0, y + 0.1), ww, 0.8, facecolor=fill, edgecolor="none"))
            label = st[-1]
            vis0, vis1 = max(x0, 0), min(x0 + ww, span_ms)  # label the visible part only
            if (vis1 - vis0) * px_per_ms > 7 * len(label) + 8:
                ax.text(
                    (vis0 + vis1) / 2,
                    y + 0.52,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_on(fill),
                    clip_on=True,
                )
        ax.set_yticks([i + 0.5 for i in range(len(rows))])
        ax.set_yticklabels(level_names(rows, spans), fontsize=7.5)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(
            f"{d['meta'].get('tool')} '{d['meta'].get('name')}'   window {(b - a) / 1e6:.2f} ms",
            loc="left",
            fontsize=8.5,
            color=t["ink2"],
            pad=4,
        )
    axes[-1, 0].set_xlabel("ms from window start")
    title(fig, files[0], "GPU timeline", t)
    legend(fig, cmap, legend_names, t)
    out = out_path(args, args.files[0], ".gantt.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


def build_tree(events, keep):
    root = {"name": "all", "value": 0, "children": OrderedDict()}
    for e in events:
        node = root
        root["value"] += e["dur_ns"]
        for f in [collapse(x, keep) for x in (e.get("stack") or [e["name"]])]:
            node = node["children"].setdefault(f, {"name": f, "value": 0, "children": OrderedDict()})
            node["value"] += e["dur_ns"]
    return root


def cmd_flame(args):
    plt, t = mpl(args.theme)
    d = kprof.load(args.file)
    tracks = gpu_track(d["events"], args.track)
    ev = [e for e in d["events"] if e["track"] in tracks and e.get("dur_ns")]
    root = build_tree(ev, args.keep_indices)
    leaves, totals = [], defaultdict(float)
    for e in sorted(ev, key=lambda e: e["start_ns"]):
        leaves.append(e["name"])
        totals[e["name"]] += e["dur_ns"]
    cmap, legend_names = assign_colors(leaves, totals, t)

    depth = 0

    def walk(n, dd):
        nonlocal depth
        depth = max(depth, dd)
        for c in n["children"].values():
            walk(c, dd + 1)

    walk(root, 0)
    fig_h = 1.7 + 0.34 * depth + 0.4
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.75 / fig_h, bottom=0.55 / fig_h)
    total = root["value"]
    ax.set_xlim(0, total)
    ax.set_ylim(depth, 0)
    px = ax.get_window_extent().width / total
    gap = 2 / px

    def draw(n, x, dd):
        kids = sorted(n["children"].values(), key=lambda c: -c["value"])  # left-heavy
        cx = x
        for c in kids:
            w = c["value"]
            leaf = not c["children"]
            fill = cmap.get(c["name"], t["other"]) if leaf else t["parent"]
            ax.add_patch(plt.Rectangle((cx, dd + 0.08), max(w - gap, 0.5 / px), 0.84, facecolor=fill, edgecolor="none"))
            label = f"{c['name']}  {100 * w / total:.1f}%"
            if w * px > 6.2 * len(label) + 8:
                ax.text(
                    cx + 4 / px,
                    dd + 0.52,
                    label,
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color=text_on(fill),
                    clip_on=True,
                )
            elif w * px > 6.2 * len(c["name"]) + 8:
                ax.text(
                    cx + 4 / px,
                    dd + 0.52,
                    c["name"],
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color=text_on(fill),
                    clip_on=True,
                )
            draw(c, cx, dd + 1)
            cx += w

    draw(root, 0, 0)
    ax.axis("off")
    title(fig, d, "Where GPU time goes (icicle; width = share of total)", t)
    reps = {(e.get("stack") or [""])[0] for e in ev}
    fig.text(
        0.99,
        0.02,
        f"total {total / 1e6:.2f} ms over {len(reps)} top-level frame(s), track '{', '.join(tracks)}'",
        ha="right",
        fontsize=7.5,
        color=t["muted"],
    )
    legend(fig, cmap, legend_names, t)
    out = out_path(args, args.file, ".flame.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


def cmd_share(args):
    plt, t = mpl(args.theme)
    d = kprof.load(args.file)
    tracks = gpu_track(d["events"], args.track)
    ev = [e for e in d["events"] if e["track"] in tracks and e.get("dur_ns")]
    tops = {(e.get("stack") or [""])[0] for e in ev}
    per = defaultdict(float)
    for e in ev:
        per[e["name"]] += e["dur_ns"]
    n_top = max(len(tops), 1) if all(re.fullmatch(r"rep \d+", x) for x in tops) else 1
    items = sorted(((k, v / n_top / 1e3) for k, v in per.items()), key=lambda kv: -kv[1])
    total = sum(v for _, v in items)
    res = {r["name"]: r for r in d.get("results", [])}
    fig_h = 1.3 + 0.3 * len(items)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.subplots_adjust(left=0.2, right=0.8, top=1 - 0.8 / fig_h, bottom=0.45 / fig_h)
    ys = list(range(len(items)))
    bar_h = min(0.6, 24 / (fig.dpi * fig_h * 0.8 / max(len(items), 1)))
    ax.barh(ys, [v for _, v in items], height=bar_h, color=t["series"][0], edgecolor="none")
    ax.set_yticks(ys)
    ax.set_yticklabels([k for k, _ in items])
    ax.set_ylim(len(items) - 0.5, -0.5)
    for y, (k, v) in zip(ys, items, strict=True):
        extra = ""
        r = res.get(k)
        if r and r.get("gbps"):
            extra = f"  |  {r['gbps']:.0f} GB/s" + (
                f" ({r['pct_peak_bw']:.0f}% of roof)" if r.get("pct_peak_bw") else ""
            )
        elif r and r.get("gflops"):
            extra = f"  |  {r['gflops']:.0f} GFLOP/s"
        ax.text(
            v + total * 0.01,
            y,
            f"{v / 1e3:.2f} ms  {100 * v / total:.1f}%{extra}",
            va="center",
            fontsize=7.5,
            color=t["ink2"],
        )
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    unit = "per rep" if n_top > 1 else "total"
    ax.set_xlabel(f"GPU time {unit} (us), track '{', '.join(tracks)}'")
    title(fig, d, f"GPU time by kernel ({total / 1e3:.2f} ms {unit})", t)
    out = out_path(args, args.file, ".share.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("perfetto")
    p.add_argument("files", nargs="+")
    p.add_argument("--out")
    p.add_argument("--absolute", action="store_true", help="keep raw timestamps (files on the same clock line up)")
    p.add_argument(
        "--open", action="store_true", help="open the result in the Perfetto UI (serves it once on 127.0.0.1:9001)"
    )
    f = sub.add_parser("folded")
    f.add_argument("file")
    f.add_argument("--out")
    f.add_argument("--track", action="append")
    f.add_argument("--keep-indices", action="store_true", help="keep 'layer 7' instead of aggregating to 'layer'")
    for name in ("gantt", "flame", "share"):
        g = sub.add_parser(name)
        g.add_argument("files" if name == "gantt" else "file", nargs="+" if name == "gantt" else None)
        g.add_argument("--out")
        g.add_argument("--track", action="append")
        g.add_argument("--dark", dest="theme", action="store_const", const="dark", default="light")
        g.add_argument("--dpi", type=int, default=150)
        if name == "gantt":
            g.add_argument("--frame", help="top-level frame to show, e.g. 'rep 3' (default: rep 0 if present)")
            g.add_argument("--start-ms", type=float)
            g.add_argument("--end-ms", type=float)
        if name == "flame":
            g.add_argument("--keep-indices", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("file")
    t.add_argument("--md", action="store_true")
    args = ap.parse_args()
    {
        "perfetto": cmd_perfetto,
        "folded": cmd_folded,
        "gantt": cmd_gantt,
        "flame": cmd_flame,
        "share": cmd_share,
        "table": cmd_table,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
