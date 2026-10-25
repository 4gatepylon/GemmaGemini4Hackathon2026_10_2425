"""mlx-lm backend (in-process), e.g. to compare answers against our kernels by hand.

Needs the `mlx` extra: `uv sync --extra mlx`. Greedy (mlx-lm's default sampler).

MLX GPU streams are per-thread, but the server calls `stream()` from arbitrary worker
threads. So all MLX work (loading and generation) runs on one dedicated thread owned by this
runner, and chunks are handed back through a queue.
"""

import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from ..paths import baseline
from .base import Chunk, Message, Runner

_END = object()


class MLXRunner(Runner):
    def __init__(self, model: str | None = None):
        model = model or str(baseline("mlx"))  # created by benchmarking/setup.py
        self.model_id = f"mlx:{Path(model).name}"
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, name="mlx", daemon=True).start()

        def load():
            from mlx_lm import load

            self.model, self.tok = load(model)

        self._call(load)

    def _worker(self) -> None:
        while (job := self._jobs.get()) is not None:
            job()

    def _call(self, fn: Callable[[], None]) -> None:
        """Run fn on the MLX thread and wait for it (re-raising its exception)."""
        done: queue.Queue = queue.Queue()
        self._jobs.put(lambda: done.put(_safe(fn)))
        if isinstance(err := done.get(), BaseException):
            raise err

    def stream(self, messages: list[Message], max_tokens: int) -> Iterator[Chunk]:
        out: queue.Queue = queue.Queue()
        cancelled = threading.Event()

        def generate():
            from mlx_lm import stream_generate

            # Thinking off, like the HF template our kernels use (mlx-lm turns it on where supported).
            prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
            last = None
            for r in stream_generate(self.model, self.tok, prompt, max_tokens=max_tokens):
                if cancelled.is_set():
                    return
                last = r
                if r.text:
                    out.put(Chunk(r.text))
            finish = (last.finish_reason if last else None) or "length"
            # mlx-lm counts the stop token; other runners don't, so drop it for consistent usage numbers.
            n = (last.generation_tokens if last else 0) - (finish == "stop")
            out.put(Chunk(finish_reason=finish, prompt_tokens=last.prompt_tokens if last else 0, completion_tokens=n))

        self._jobs.put(lambda: (out.put(_safe(generate)), out.put(_END)))
        try:
            while (item := out.get()) is not _END:
                if isinstance(item, BaseException):
                    raise item
                if item is not None:
                    yield item
        finally:
            cancelled.set()  # consumer stopped early: let the MLX thread stop at the next token

    def close(self) -> None:
        self._jobs.put(None)


def _safe(fn: Callable[[], None]) -> BaseException | None:
    try:
        fn()
    except BaseException as e:  # handed to the calling thread
        return e
    return None
