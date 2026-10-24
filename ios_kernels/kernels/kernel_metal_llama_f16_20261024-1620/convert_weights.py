"""Convert a HF Llama safetensors checkpoint into a flat, mmap-able fp16 file.

File layout:
    [0:8)     little-endian u64 = header length H
    [8:8+H)   UTF-8 JSON header: {"config": {...}, "tensors": {name: {"offset", "shape", "dtype"}}}
    ...       tensor data; every tensor starts on a 256-byte boundary, offsets are absolute
    file is padded to a 16 KiB multiple so Swift can wrap the whole mmap in one MTLBuffer.

Per layer we fuse Wq|Wk|Wv into one [q+2kv, D] matrix so decode needs one matvec launch.
Norm weights are stored as fp32 (they're tiny), everything else as fp16.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from safetensors import safe_open

ALIGN = 256
PAGE = 16384


def find_snapshot(repo: str) -> str:
    base = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots")
    for snap in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(snap, "model.safetensors")):
            return snap
    raise FileNotFoundError(f"no model.safetensors under {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap = find_snapshot(args.repo)
    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    st = safe_open(os.path.join(snap, "model.safetensors"), "pt")

    def get(name: str) -> torch.Tensor:
        return st.get_tensor(name).float()

    tensors: list[tuple[str, np.ndarray]] = []

    def add(name: str, t: torch.Tensor, dtype) -> None:
        if dtype == np.float16:
            m = t.abs().max().item()
            assert m < 65504, f"{name} overflows fp16 (max |w| = {m})"
        tensors.append((name, t.contiguous().numpy().astype(dtype)))

    add("embed", get("model.embed_tokens.weight"), np.float16)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        add(f"l{i}.attn_norm", get(p + "input_layernorm.weight"), np.float32)
        wqkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"], dim=0)
        add(f"l{i}.wqkv", wqkv, np.float16)
        add(f"l{i}.wo", get(p + "self_attn.o_proj.weight"), np.float16)
        add(f"l{i}.ffn_norm", get(p + "post_attention_layernorm.weight"), np.float32)
        add(f"l{i}.wgate", get(p + "mlp.gate_proj.weight"), np.float16)
        add(f"l{i}.wup", get(p + "mlp.up_proj.weight"), np.float16)
        add(f"l{i}.wdown", get(p + "mlp.down_proj.weight"), np.float16)
    add("final_norm", get("model.norm.weight"), np.float32)
    if not cfg.get("tie_word_embeddings", False):
        add("lm_head", get("lm_head.weight"), np.float16)

    # Lay out offsets; header size depends on offsets, so reserve a fixed prefix.
    header_reserve = 64 * 1024
    off = header_reserve
    meta = {}
    for name, arr in tensors:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        meta[name] = {"offset": off, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        off += arr.nbytes
    total = (off + PAGE - 1) // PAGE * PAGE
    header = json.dumps({"config": cfg, "tensors": meta}).encode()
    assert 8 + len(header) <= header_reserve

    with open(args.out, "wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
        for name, arr in tensors:
            f.seek(meta[name]["offset"])
            f.write(arr.tobytes())
        f.truncate(total)
    print(f"wrote {args.out}: {len(tensors)} tensors, {total / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
