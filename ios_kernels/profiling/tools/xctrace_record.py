#!/usr/bin/env python3
"""Record any command with Instruments (headless), with the noise around it.

  uv run profiling/tools/xctrace_record.py [--template NAME|PATH] [--out TRACE] [--time-limit 60s] -- COMMAND [ARGS...]

The template defaults to profiling/templates/kprof.tracetemplate if it exists (saved once from
the Instruments GUI with the shader timeline and GPU counters), else the stock
"Metal System Trace". Writes TRACE and TRACE.noise.json (knoise), checks that the trace
is readable, and prints the export command (xctrace_export.py).
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import kprof


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit(__doc__)
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    custom = kprof.PROFILING_DIR / "templates" / "kprof.tracetemplate"
    ap.add_argument("--template", default=str(custom) if custom.exists() else "Metal System Trace")
    ap.add_argument("--out", help="trace path (default $DATA_DIR/prof/xctrace/<command>-<time>.trace)")
    ap.add_argument("--time-limit", default="60s")
    args = ap.parse_args(argv[:split])
    cmd = argv[split + 1 :]
    if not cmd:
        sys.exit("no command after --")

    exe = cmd[0] if "/" in cmd[0] else shutil.which(cmd[0])  # xctrace wants a path
    if not exe:
        sys.exit(f"command not found: {cmd[0]}")
    exe = str(Path(exe).resolve())
    out = Path(args.out or kprof.PROF_DIR / "xctrace" / f"{Path(exe).name}-{time.strftime('%Y%m%d-%H%M%S')}.trace")
    out = out.resolve()  # a relative --output has produced an unreadable trace
    out.parent.mkdir(parents=True, exist_ok=True)
    knoise = kprof.BUILD_DIR / "knoise"
    if not knoise.exists():
        sys.exit(f"{knoise} not found: run `uv run profiling/build.py`")

    print(f"recording '{args.template}' -> {out}", file=sys.stderr)
    r = subprocess.run(
        [
            str(knoise),
            "--out",
            f"{out}.noise.json",
            "--name",
            Path(exe).name,
            "--",
            "xcrun",
            "xctrace",
            "record",
            "--template",
            args.template,
            "--output",
            str(out),
            "--no-prompt",
            "--time-limit",
            args.time_limit,
            "--launch",
            "--",
            exe,
            *cmd[1:],
        ]
    )
    check = subprocess.run(["xcrun", "xctrace", "export", "--input", str(out), "--toc"], capture_output=True)
    if check.returncode != 0:
        sys.exit(f"error: {out} is not a readable trace (recording incomplete?); re-record")
    print(f"next: uv run profiling/tools/xctrace_export.py {out}", file=sys.stderr)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
