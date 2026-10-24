"""Deterministic stand-in backend for tests and UI work (no GPU, no model).

Algorithm: the reply is "Echo: " + the last user message, streamed one whitespace-separated
word per chunk ("token"). max_tokens truncates (finish_reason "length"); otherwise "stop".
prompt_tokens counts words across all messages.
"""

import re
import time
from collections.abc import Iterator

from .base import Chunk, Message, Runner


class FakeRunner(Runner):
    model_id = "fake-echo"

    def __init__(self, delay: float = 0.0):
        self.delay = delay  # seconds per token, to make streaming visible by eye

    @staticmethod
    def reply_for(messages: list[Message]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"Echo: {last_user}"

    def stream(self, messages: list[Message], max_tokens: int) -> Iterator[Chunk]:
        if not messages:
            raise ValueError("messages must not be empty")
        words = re.findall(r"\S+\s*", self.reply_for(messages))  # keep trailing whitespace
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        out = words[:max_tokens]
        for w in out:
            if self.delay:
                time.sleep(self.delay)
            yield Chunk(w)
        yield Chunk(finish_reason="length" if len(words) > max_tokens else "stop",
                    prompt_tokens=prompt_tokens, completion_tokens=len(out))
