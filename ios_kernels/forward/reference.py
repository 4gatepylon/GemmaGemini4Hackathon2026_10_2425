"""Correctness check: compare the Metal engine against HF transformers (fp32, CPU).

    python reference.py --weights ../../.context/models/llama1b_f16.bin

Tokenizes a chat prompt, runs both engines, and reports last-position logit
error, top-5 agreement, and whether greedy generations match token for token.
"""

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def run_engine(weights: str, ids: list[int], gen: int, logits_path: str, extra: list[str]) -> dict:
    cmd = [os.path.join(HERE, ".build", "llm"), "--weights", weights,
           "--prompt-ids", ",".join(map(str, ids)), "--gen", str(gen),
           "--dump-logits", logits_path, *extra]
    return json.loads(subprocess.check_output(cmd).decode().strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--engine-args", default="", help="extra args for the Swift engine")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.repo)
    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True)
    print(f"prompt: {len(ids)} tokens")

    with tempfile.NamedTemporaryFile(suffix=".f32") as f:
        res = run_engine(args.weights, ids, args.gen, f.name, args.engine_args.split())
        ours = np.fromfile(f.name, dtype=np.float32)

    model = AutoModelForCausalLM.from_pretrained(args.repo, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0, -1].numpy()
        ref_gen = model.generate(torch.tensor([ids]), max_new_tokens=args.gen, do_sample=False,
                                 temperature=None, top_p=None)[0, len(ids):].tolist()

    diff = np.abs(ours - ref)
    top_ours, top_ref = np.argsort(-ours)[:5], np.argsort(-ref)[:5]
    n_match = next((i for i, (a, b) in enumerate(zip(res["tokens"], ref_gen)) if a != b),
                   min(len(res["tokens"]), len(ref_gen)))
    print(f"logits: max|diff| {diff.max():.4f}, mean|diff| {diff.mean():.5f}, ref range ±{np.abs(ref).max():.1f}")
    print(f"top5 ours {top_ours.tolist()} ref {top_ref.tolist()}")
    print(f"greedy: first {n_match}/{len(ref_gen)} tokens match")
    print("ours:", repr(tok.decode(res["tokens"])))
    print("ref: ", repr(tok.decode(ref_gen)))
    print(f"engine decode: {res['decode_tok_s']:.1f} tok/s")


if __name__ == "__main__":
    main()
