"""Runner for our hand-written kernels (any folder under ios_kernels/kernels/).

Each kernel folder is self-describing via kernel.json:
    {"hf_repo": ..., "weights": <file in DATA_DIR/models>, "build": "build.sh", "binary": ".build/llm"}
and its binary implements the `--serve` JSON-lines protocol (see surfaces/README.md).
This runner builds the binary if needed, keeps it running as a subprocess, and does
tokenization / chat templating / detokenization in Python.
"""

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ..paths import MODELS_DIR, resolve_kernel
from .base import Chunk, Message, Runner
from .hf_chat import ChatTokenizer


class KernelRunner(Runner):
    def __init__(self, kernel: str | None = None, weights: str | None = None, max_ctx: int = 4096):
        self.kernel_dir = resolve_kernel(kernel)
        manifest = json.loads((self.kernel_dir / "kernel.json").read_text())
        self.model_id = self.kernel_dir.name
        binary = self.kernel_dir / manifest["binary"]
        self._build_if_stale(binary, self.kernel_dir / manifest["build"])

        weights_path = Path(weights) if weights else MODELS_DIR / manifest["weights"]
        if not weights_path.exists():
            raise FileNotFoundError(f"{weights_path} missing; run ios_kernels/benchmarking/setup.sh")

        self.tok = ChatTokenizer(manifest["hf_repo"])
        # stderr is inherited so kernel logs show up in the server's terminal.
        self.proc = subprocess.Popen(
            [str(binary), "--weights", str(weights_path), "--serve", "--max-ctx", str(max_ctx)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"kernel did not start: {ready}")
        self.max_ctx = ready["max_ctx"]

    @staticmethod
    def _build_if_stale(binary: Path, build_script: Path) -> None:
        sources = list(build_script.parent.glob("*.swift"))
        if binary.exists() and all(s.stat().st_mtime <= binary.stat().st_mtime for s in sources):
            return
        subprocess.run([str(build_script)], check=True, stdout=subprocess.DEVNULL)

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"kernel process exited (code {self.proc.poll()})")
        return json.loads(line)

    def stream(self, messages: list[Message], max_tokens: int) -> Iterator[Chunk]:
        ids = self.tok.encode(messages)
        if len(ids) >= self.max_ctx:
            raise ValueError(f"prompt is {len(ids)} tokens; this kernel's context is {self.max_ctx}")
        req = {"ids": ids, "max_tokens": max_tokens, "stop_ids": self.tok.stop_ids}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

        dec = self.tok.decoder()
        done = False
        try:
            while True:
                msg = self._read()
                if "error" in msg:
                    done = True
                    raise ValueError(msg["error"])
                if "token" in msg:
                    if text := dec.add(msg["token"]):
                        yield Chunk(text)
                elif msg.get("done"):
                    done = True
                    yield Chunk(dec.flush(), finish_reason=msg["finish_reason"],
                                prompt_tokens=msg["prompt_tokens"], completion_tokens=msg["completion_tokens"])
                    return
        finally:
            # Consumer stopped early (e.g. client disconnected): drain this response so the
            # protocol stays in sync for the next request.
            while not done and self.proc.poll() is None:
                done = bool(self._read().get("done"))

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
