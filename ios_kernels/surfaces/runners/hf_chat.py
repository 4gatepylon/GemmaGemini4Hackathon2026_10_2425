"""HF tokenizer helpers for runners whose backend only speaks token ids (like our kernels)."""

from transformers import AutoTokenizer, GenerationConfig

from .base import Message


class ChatTokenizer:
    def __init__(self, repo: str):
        self.tok = AutoTokenizer.from_pretrained(repo)
        eos = GenerationConfig.from_pretrained(repo).eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.stop_ids = sorted({*eos, self.tok.eos_token_id} - {None})

    def encode(self, messages: list[Message]) -> list[int]:
        # Render then encode: stable across transformers versions (the template adds BOS).
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tok.encode(text, add_special_tokens=False)

    def decoder(self) -> "IncrementalDecoder":
        return IncrementalDecoder(self.tok)


class IncrementalDecoder:
    """Turns a growing list of token ids into text deltas. Holds back output while the tail
    is an incomplete UTF-8 sequence (multi-token emoji etc.)."""

    def __init__(self, tok):
        self.tok = tok
        self.ids: list[int] = []
        self.sent = 0

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta

    def flush(self) -> str:
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        delta, self.sent = text[self.sent:], len(text)
        return delta
