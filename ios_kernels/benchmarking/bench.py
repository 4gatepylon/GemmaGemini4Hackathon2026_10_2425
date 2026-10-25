"""Benchmark a kernel vs llama.cpp vs MLX on the same FP16 model (the kernel's).

    uv run --extra bench benchmarking/bench.py [--kernel NAME] [--pp 512] [--tg 128] [--reps 5]
        [--engine-args "--define MV_NR=2"]

Needs `benchmarking/setup.py` first.
"""

import argparse
import re
import subprocess
import sys

from surfaces.paths import LLAMA_DIR, MODELS_DIR, baseline, load_manifest, resolve_kernel


def show(cmd: list, pattern: str, merge_stderr: bool = False) -> None:
    """Run cmd and print only the output lines matching pattern."""
    out = subprocess.run(
        [str(c) for c in cmd],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
    ).stdout
    for line in out.splitlines():
        if re.search(pattern, line):
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", help="kernel folder name/substring/path (default: newest)")
    ap.add_argument("--pp", type=int, default=512, help="prompt tokens")
    ap.add_argument("--tg", type=int, default=128, help="generated tokens")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--engine-args", default="", help="extra args for our engine, e.g. '--define MV_NR=2'")
    args = ap.parse_args()
    kernel = resolve_kernel(args.kernel)
    manifest = load_manifest(kernel)
    pp, tg, reps = str(args.pp), str(args.tg), str(args.reps)

    subprocess.run([str(kernel / manifest["build"])], check=True, stdout=subprocess.DEVNULL)

    print(f"== ours: {kernel.name} ==", flush=True)
    show(
        [
            kernel / manifest["binary"],
            "--weights",
            MODELS_DIR / manifest["weights"],
            "--bench-pp",
            pp,
            "--bench-tg",
            tg,
            "--reps",
            reps,
            *args.engine_args.split(),
        ],
        r"^(pp|tg)",
        merge_stderr=True,
    )

    commit = subprocess.check_output(["git", "-C", str(LLAMA_DIR), "log", "-1", "--format=%h"], text=True).strip()
    print(f"== llama.cpp (Metal, {commit}) ==", flush=True)
    show(
        [
            LLAMA_DIR / "build/bin/llama-bench",
            "-m",
            baseline("gguf", kernel),
            "-p",
            pp,
            "-n",
            tg,
            "-r",
            reps,
            "-ngl",
            "99",
        ],
        r"pp|tg",
    )

    print("== MLX (mlx_lm; note its tg is measured after the pp prompt) ==", flush=True)
    show(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "benchmark",
            "--model",
            baseline("mlx", kernel),
            "-p",
            pp,
            "-g",
            tg,
            "-n",
            reps,
        ],
        r"Averages",
    )


if __name__ == "__main__":
    main()
