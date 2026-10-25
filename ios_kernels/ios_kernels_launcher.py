"""ios_kernels launcher: one entry point for setup, serving, chatting, testing and benchmarking.

    uv run ios_kernels_launcher.py <command> [flags]      (from ios_kernels/)
    make <command> [args="--flag value ..."]              (the Makefile is shorthand for this)
    uv run ios_kernels_launcher.py help <command>         (flags for one command)

It wraps the existing tools (benchmarking/*.sh, the openai-server / cli-chat entry points,
pytest, llama.cpp) so nobody has to remember paths or environment variables.
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent  # ios_kernels/


# --------------------------------------------------------------------------- helpers

def paths():
    """surfaces.paths, imported lazily so --data-dir (which sets $DATA_DIR) takes effect first."""
    from surfaces import paths as p

    return p


def run(cmd: list, env: dict | None = None, cwd: Path = HERE) -> int:
    """Run a command in the foreground. Ctrl-C stops the child (it gets SIGINT too) without a traceback."""
    print("+ " + " ".join(str(c) for c in cmd), file=sys.stderr, flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})})
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            continue  # the child got the same SIGINT; wait for it to shut down


def kernel_dir(name: str | None) -> Path:
    return paths().resolve_kernel(name)


def manifest(kdir: Path) -> dict:
    return json.loads((kdir / "kernel.json").read_text())


def base_url(args) -> str:
    return getattr(args, "base_url", None) or f"http://{args.host}:{args.port}/v1"


def wait_for_health(url: str, timeout: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise SystemExit(f"server not healthy after {timeout:.0f}s: {url}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- commands

def cmd_setup(args, extra) -> int:
    """One-time: build llama.cpp, convert weights (~7.5 GB into the data dir), build the kernel."""
    return run([HERE / "benchmarking/setup.sh", kernel_dir(args.kernel)])


def cmd_build(args, extra) -> int:
    """Build the kernel binary (Swift + runtime-compiled Metal)."""
    kdir = kernel_dir(args.kernel)
    return run([kdir / manifest(kdir)["build"]])


def server_cmd(args, extra, runner: str) -> list:
    cmd = ["uv", "run", "--project", HERE]
    if runner == "mlx":
        cmd += ["--extra", "mlx"]
    cmd += ["openai-server", "--runner", runner, "--host", args.host, "--port", args.port]
    if runner == "kernel" and args.kernel:
        cmd += ["--kernel", kernel_dir(args.kernel)]
    return cmd + extra


def cmd_serve(args, extra) -> int:
    """OpenAI-compatible server (default backend: newest kernel). Extra flags go to openai-server."""
    return run(server_cmd(args, extra, args.runner))


def cmd_serve_fake(args, extra) -> int:
    """`serve --runner fake`: canned echo replies, no GPU or weights needed."""
    return run(server_cmd(args, extra, "fake"))


def cmd_serve_mlx(args, extra) -> int:
    """`serve --runner mlx`: the mlx-lm backend (installs the mlx extra), to compare by hand."""
    return run(server_cmd(args, extra, "mlx"))


def cmd_serve_llamacpp(args, extra) -> int:
    """llama.cpp's own llama-server on the same FP16 model (default port 8080), for side-by-side chats."""
    p = paths()
    llama = p.DATA_DIR / "deps/llama.cpp"
    gguf = p.MODELS_DIR / "llama1b_f16.gguf"
    server = llama / "build/bin/llama-server"
    if not gguf.exists():
        raise SystemExit(f"{gguf} missing; run `make setup` first")
    if not server.exists():
        if code := run(["cmake", "--build", llama / "build", "-j", "10", "--target", "llama-server"]):
            return code
    print(f"chat with it: make chat args=\"--port {args.port}\"", file=sys.stderr)
    return run([server, "-m", gguf, "-ngl", "99", "--host", args.host, "--port", args.port, *extra])


def cmd_chat(args, extra) -> int:
    """Interactive terminal chat with the server at --base-url (or --host/--port). /reset, /exit."""
    from surfaces.cli_client import chat

    chat.main(["--base-url", base_url(args), *extra])
    return 0


def cmd_ask(args, extra) -> int:
    """One-shot: ask a question, print the reply and tok/s. Words are joined: `ask why is the sky blue`."""
    from surfaces.cli_client import chat

    chat.main(["--base-url", base_url(args), "--prompt", " ".join(args.prompt), *extra])
    return 0


def cmd_smoke(args, extra) -> int:
    """Start a server on a free port, ask one question, print the answer, shut down."""
    args.port, args.host = free_port(), "127.0.0.1"
    cmd = server_cmd(args, [], args.runner)
    print("+ " + " ".join(map(str, cmd)), file=sys.stderr, flush=True)
    server = subprocess.Popen([str(c) for c in cmd], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_health(f"http://127.0.0.1:{args.port}/health", timeout=120, proc=server)
        from surfaces.cli_client import chat

        chat.main(["--base-url", f"http://127.0.0.1:{args.port}/v1", "--prompt", args.prompt, "--max-tokens", "64"])
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


def cmd_test(args, extra) -> int:
    """Full pytest suite (API/CLI + end-to-end kernel/MLX tests that skip if weights are missing)."""
    return run([sys.executable, "-m", "pytest", "-q", *extra])


def cmd_test_fast(args, extra) -> int:
    """Only the fast tests: fake backend, launcher, Makefile (a few seconds, no GPU)."""
    return run([sys.executable, "-m", "pytest", "-q", "surfaces/tests/test_server.py", "tests/", *extra])


def cmd_bench(args, extra) -> int:
    """Ours vs llama.cpp vs MLX (benchmarking/bench.sh)."""
    env = {"PP": str(args.pp), "TG": str(args.tg), "REPS": str(args.reps)}
    return run([HERE / "benchmarking/bench.sh", kernel_dir(args.kernel), *extra], env=env)


def cmd_bench_ours(args, extra) -> int:
    """Our engine only: quick before/after numbers while iterating. --define sets kernel tunables."""
    kdir = kernel_dir(args.kernel)
    m = manifest(kdir)
    if code := run([kdir / m["build"]]):
        return code
    defines = [x for d in args.define for x in ("--define", d)]
    return run([kdir / m["binary"], "--weights", paths().MODELS_DIR / m["weights"],
                "--bench-pp", args.pp, "--bench-tg", args.tg, "--reps", args.reps, *defines, *extra])


def cmd_reference(args, extra) -> int:
    """Correctness: compare the kernel's logits and greedy tokens with HF transformers."""
    return run(["uv", "run", "--project", HERE, "--extra", "bench", "python",
                HERE / "benchmarking/reference.py", "--kernel", kernel_dir(args.kernel), *extra])


def cmd_kernels(args, extra) -> int:
    """List kernel folders (newest = default for every command)."""
    kernels = paths().list_kernels()
    for k in kernels:
        mark = "*" if k == kernels[-1] else " "
        print(f"{mark} {k.name:45s} {manifest(k).get('description', '')}")
    print("\n* = default (newest). Select another with --kernel <substring>.")
    return 0


def new_kernel(name: str, src: Path, kernels_dir: Path, now: datetime | None = None) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", name):
        raise SystemExit(f"bad name {name!r}: use lowercase letters, digits and underscores")
    dst = kernels_dir / f"kernel_{name}_{(now or datetime.now()).strftime('%Y%m%d-%H%M')}"
    if dst.exists():
        raise SystemExit(f"{dst.name} already exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".build", "__pycache__"))
    return dst


def cmd_new_kernel(args, extra) -> int:
    """Copy the newest kernel (or --from) to kernels/kernel_<name>_<YYYYMMDD-HHMM>/ to start a new version."""
    src = kernel_dir(args.from_kernel)
    dst = new_kernel(args.name, src, paths().KERNELS_DIR)
    print(f"created {dst.relative_to(HERE)} (copied from {src.name}); it is now the default kernel.")
    print("next: edit it, update its README/kernel.json, then `make test` and `make bench-ours`.")
    return 0


def cmd_clean(args, extra) -> int:
    """Remove build outputs and caches (kernels/*/.build, __pycache__, .pytest_cache). Never touches weights."""
    targets = [*HERE.glob("kernels/*/.build"), *HERE.glob(".pytest_cache")]
    targets += [p for p in HERE.rglob("__pycache__") if ".venv" not in p.parts]
    for t in targets:
        shutil.rmtree(t)
        print(f"removed {t.relative_to(HERE)}")
    return 0


# --------------------------------------------------------------------------- CLI

COMMANDS = {}  # name -> (function, configure-parser function); order = help order


def command(name, fn, *configure, passthrough=False):
    COMMANDS[name] = (fn, configure, passthrough)


def k(p):
    p.add_argument("--kernel", help="kernel folder: name substring or path (default: newest)")


def server_flags(p, port=8000):
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=port)


def client_flags(p):
    server_flags(p)
    p.add_argument("--base-url", help="overrides --host/--port, e.g. http://127.0.0.1:8080/v1")


def runner_flag(p):
    p.add_argument("--runner", default="kernel", choices=["kernel", "fake", "mlx"])


def bench_flags(p):
    p.add_argument("--pp", type=int, default=512, help="prompt tokens")
    p.add_argument("--tg", type=int, default=128, help="generated tokens")
    p.add_argument("--reps", type=int, default=5)


command("setup", cmd_setup, k)
command("build", cmd_build, k)
command("serve", cmd_serve, k, runner_flag, server_flags, passthrough=True)
command("serve-fake", cmd_serve_fake, server_flags, passthrough=True)
command("serve-mlx", cmd_serve_mlx, server_flags, passthrough=True)
command("serve-llamacpp", cmd_serve_llamacpp, lambda p: server_flags(p, port=8080), passthrough=True)
command("chat", cmd_chat, client_flags, passthrough=True)
command("ask", cmd_ask, client_flags, lambda p: p.add_argument("prompt", nargs="+"), passthrough=True)
command("smoke", cmd_smoke, k, runner_flag,
        lambda p: p.add_argument("--prompt", default="What is the capital of France? Answer in one sentence."))
command("test", cmd_test, passthrough=True)
command("test-fast", cmd_test_fast, passthrough=True)
command("bench", cmd_bench, k, bench_flags)
command("bench-ours", cmd_bench_ours, k, bench_flags,
        lambda p: p.add_argument("--define", action="append", default=[], metavar="NAME=VALUE",
                                 help="kernel tunable, repeatable (e.g. MV_NR=2)"))
command("reference", cmd_reference, k)
command("kernels", cmd_kernels)
command("new-kernel", cmd_new_kernel, lambda p: p.add_argument("name", help="e.g. metal_llama_q8"),
        lambda p: p.add_argument("--from", dest="from_kernel", help="kernel to copy (default: newest)"))
command("clean", cmd_clean)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help="weights / llama.cpp location (default: <repo>/.context)")

    parser = argparse.ArgumentParser(
        prog="ios_kernels_launcher.py", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  make setup                                  # one-time\n"
               "  make serve                                  # newest kernel on :8000\n"
               "  make chat                                   # in another terminal\n"
               "  make ask args=\"why is the sky blue\"\n"
               "  make serve args=\"--kernel 20261024 --port 9000\"\n"
               "  make bench-ours args=\"--define MV_NR=2 --reps 3\"\n"
               "  make help args=serve                        # flags for one command",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    sub.add_parser("help", help="show this help, or `help <command>` for one command's flags") \
        .add_argument("topic", nargs="?")
    for name, (fn, configure, passthrough) in COMMANDS.items():
        doc = fn.__doc__.strip()
        if passthrough:
            doc += " Unknown flags are passed through."
        p = sub.add_parser(name, parents=[common], help=doc.split(". ")[0].rstrip(".") + ".", description=doc)
        for c in configure:
            c(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command == "help":
        if args.topic:
            if args.topic not in COMMANDS:
                parser.error(f"unknown command {args.topic!r}")
            parser.parse_args([args.topic, "--help"])
        parser.print_help()
        return 0
    fn, _, passthrough = COMMANDS[args.command]
    if extra and not passthrough:
        parser.error(f"unrecognized arguments for {args.command}: {' '.join(extra)}")
    if args.data_dir:
        os.environ["DATA_DIR"] = str(Path(args.data_dir).resolve())  # inherited by every child process
    return fn(args, extra)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
