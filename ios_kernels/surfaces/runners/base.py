"""The Runner interface: the one seam between surfaces (server, CLI) and inference backends.

A runner turns chat messages into a stream of text chunks. Everything backend-specific
(tokenization, chat template, process management, the kernels themselves) lives behind it,
so the OpenAI server never changes when a new kernel or runtime is added.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": str}


@dataclass
class Chunk:
    """A piece of generated text. The last chunk of a stream has `finish_reason` and usage set."""

    text: str = ""
    finish_reason: str | None = None  # "stop" (hit a stop token) or "length" (hit max_tokens)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Runner(ABC):
    model_id: str  # reported by /v1/models

    @abstractmethod
    def stream(self, messages: list[Message], max_tokens: int) -> Iterator[Chunk]:
        """Greedy-generate a reply. Must end with exactly one chunk whose finish_reason is set.
        Raise ValueError for bad requests (e.g. prompt longer than the context)."""

    def close(self) -> None:
        """Release resources (subprocesses, GPU memory)."""
