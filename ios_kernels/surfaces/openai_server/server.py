"""Minimal OpenAI-compatible chat server over any Runner.

    uv run openai-server                      # newest kernel of the default model (MODEL in surfaces/paths.py)
    uv run openai-server --kernel <name>      # a specific kernel folder
    uv run openai-server --runner fake        # canned echo replies, no GPU
    uv run openai-server --runner mlx         # mlx-lm reference backend

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream and non-stream).
Decoding is always greedy: temperature/top_p/etc. are accepted and ignored.
"""

import argparse
import json
import threading
import time
import uuid
from collections.abc import Iterator

import anyio
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..runners import RUNNERS, Chunk, Runner, make_runner

SAMPLING_HEADER = {"x-sampling": "greedy; temperature/top_p ignored"}


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]  # plain text, or OpenAI content parts (text parts only)

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(p.get("text", "") for p in self.content if p.get("type") == "text")


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # temperature, top_p, ... accepted and ignored

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)


class ClosingStreamingResponse(StreamingResponse):
    """StreamingResponse that always close()s its sync iterator, including when the client
    disconnects mid-stream. Starlette alone leaves the generator for the GC, which would keep
    the runner lock held (and the backend mid-response) indefinitely."""

    def __init__(self, content: Iterator[str], **kwargs):
        self._content = content
        super().__init__(content, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self._content.close)


def create_app(runner: Runner, default_max_tokens: int = 512) -> FastAPI:
    app = FastAPI(title="ios_kernels OpenAI-compatible server")
    lock = threading.Lock()  # runners serve one request at a time

    def locked_stream(messages, max_tokens) -> Iterator[Chunk]:
        with lock:
            yield from runner.stream(messages, max_tokens)

    @app.exception_handler(RequestValidationError)
    def on_validation_error(_: Request, exc: RequestValidationError):
        return error(400, str(exc.errors()))

    @app.get("/health")
    def health():
        return {"status": "ok", "model": runner.model_id}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": runner.model_id, "object": "model", "owned_by": "ios_kernels"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        messages = [{"role": m.role, "content": m.text()} for m in req.messages]
        max_tokens = req.max_completion_tokens or req.max_tokens or default_max_tokens
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        base = {"id": cid, "created": created, "model": runner.model_id}

        chunks = locked_stream(messages, max_tokens)
        try:
            first = next(chunks)  # surface bad requests as HTTP 400 before streaming starts
        except ValueError as e:
            return error(400, str(e))

        def usage(c: Chunk) -> dict:
            return {
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "total_tokens": c.prompt_tokens + c.completion_tokens,
            }

        if not req.stream:
            parts = [first, *chunks]
            last = parts[-1]
            return JSONResponse(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": last.finish_reason,
                            "message": {"role": "assistant", "content": "".join(c.text for c in parts)},
                        }
                    ],
                    "usage": usage(last),
                },
                headers=SAMPLING_HEADER,
            )

        include_usage = bool(req.stream_options and req.stream_options.include_usage)

        def sse() -> Iterator[str]:
            def event(choices: list, **extra) -> str:
                chunk = {**base, "object": "chat.completion.chunk", "choices": choices, **extra}
                return f"data: {json.dumps(chunk)}\n\n"

            try:
                yield event([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
                for c in _chain(first, chunks):
                    if c.text:
                        yield event([{"index": 0, "delta": {"content": c.text}, "finish_reason": None}])
                    if c.finish_reason:
                        yield event([{"index": 0, "delta": {}, "finish_reason": c.finish_reason}])
                        if include_usage:
                            yield event([], usage=usage(c))
                yield "data: [DONE]\n\n"
            finally:
                chunks.close()  # releases the lock; runner drains its backend if we stopped early

        return ClosingStreamingResponse(sse(), media_type="text/event-stream", headers=SAMPLING_HEADER)

    return app


def _chain(first: Chunk, rest: Iterator[Chunk]) -> Iterator[Chunk]:
    yield first
    yield from rest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default="kernel", choices=sorted(RUNNERS))
    ap.add_argument("--kernel", help="[kernel] folder name/substring/path under kernels/<model>/ (default: newest)")
    ap.add_argument("--weights", help="[kernel] weights file (default: from the kernel's kernel.json)")
    ap.add_argument("--max-ctx", type=int, default=4096, help="[kernel] KV cache length")
    ap.add_argument(
        "--mlx-model", help="[mlx] model path or HF repo (default: the default model's MLX copy in .context/models/)"
    )
    ap.add_argument("--fake-delay", type=float, default=0.05, help="[fake] seconds per token")
    ap.add_argument("--default-max-tokens", type=int, default=512)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    kwargs = {
        "kernel": {"kernel": args.kernel, "weights": args.weights, "max_ctx": args.max_ctx},
        "fake": {"delay": args.fake_delay},
        "mlx": {"model": args.mlx_model},
    }[args.runner]
    runner = make_runner(args.runner, **kwargs)
    print(f"runner: {args.runner} ({runner.model_id}); serving http://{args.host}:{args.port}/v1", flush=True)

    import uvicorn

    try:
        uvicorn.run(create_app(runner, args.default_max_tokens), host=args.host, port=args.port)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
