"""mlx-lm backend (in-process), e.g. to compare answers against our kernels by hand.

Needs the `mlx` extra: `uv sync --extra mlx`. Greedy (mlx-lm's default sampler).
"""

from collections.abc import Iterator
from pathlib import Path

from ..paths import MODELS_DIR
from .base import Chunk, Message, Runner


class MLXRunner(Runner):
    def __init__(self, model: str | None = None):
        from mlx_lm import load

        model = model or str(MODELS_DIR / "llama1b_mlx_f16")  # created by benchmarking/setup.sh
        self.model, self.tok = load(model)
        self.model_id = f"mlx:{Path(model).name}"

    def stream(self, messages: list[Message], max_tokens: int) -> Iterator[Chunk]:
        from mlx_lm import stream_generate

        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        last = None
        for r in stream_generate(self.model, self.tok, prompt, max_tokens=max_tokens):
            last = r
            if r.text:
                yield Chunk(r.text)
        finish = (last.finish_reason if last else None) or "length"
        # mlx-lm counts the stop token; other runners don't, so drop it for consistent usage numbers.
        n = (last.generation_tokens if last else 0) - (finish == "stop")
        yield Chunk(finish_reason=finish, prompt_tokens=last.prompt_tokens if last else 0, completion_tokens=n)
