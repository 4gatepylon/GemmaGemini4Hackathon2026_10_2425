"""Terminal chat client for any OpenAI-compatible server (ours, llama.cpp's llama-server, ...).

    uv run cli-chat                                   # interactive, talks to http://127.0.0.1:8000/v1
    uv run cli-chat --prompt "Why is the sky blue?"   # one-shot
    uv run cli-chat --base-url http://127.0.0.1:8080/v1   # e.g. llama-server, to compare by hand

In the REPL: /reset clears the conversation, /exit (or Ctrl-D) quits.
After each reply it prints time-to-first-token, tokens, and tokens/s.
"""

import argparse
import sys
import time

from openai import OpenAI


def reply(client: OpenAI, model: str, messages: list[dict], max_tokens: int, out=None) -> str:
    out = out or sys.stdout
    t0 = time.perf_counter()
    ttft, text, usage, finish = None, [], None, None
    stream = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        for choice in chunk.choices:
            if choice.delta.content:
                ttft = ttft or time.perf_counter() - t0
                text.append(choice.delta.content)
                out.write(choice.delta.content)
                out.flush()
            finish = choice.finish_reason or finish
    elapsed = time.perf_counter() - t0
    n = usage.completion_tokens if usage else len(text)
    rate = (n - 1) / (elapsed - ttft) if ttft and n > 1 and elapsed > ttft else 0.0
    out.write(f"\n[{n} tokens, ttft {1000 * (ttft or 0):.0f} ms, {rate:.1f} tok/s, finish={finish}]\n")
    return "".join(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", help="default: first model the server lists")
    ap.add_argument("--system", help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt", help="send one message, print the reply, exit")
    args = ap.parse_args(argv)

    client = OpenAI(base_url=args.base_url, api_key="unused")
    model = args.model or client.models.list().data[0].id
    system = [{"role": "system", "content": args.system}] if args.system else []
    history = list(system)

    if args.prompt is not None:
        reply(client, model, history + [{"role": "user", "content": args.prompt}], args.max_tokens)
        return

    print(f"model: {model} @ {args.base_url}   (/reset, /exit)")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/reset":
            history = list(system)
            print("(conversation cleared)")
            continue
        history.append({"role": "user", "content": line})
        print("bot> ", end="")
        try:
            history.append({"role": "assistant", "content": reply(client, model, history, args.max_tokens)})
        except KeyboardInterrupt:
            history.pop()
            print("\n(interrupted)")


if __name__ == "__main__":
    main()
