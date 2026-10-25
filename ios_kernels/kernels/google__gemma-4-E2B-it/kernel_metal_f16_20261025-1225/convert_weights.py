"""Convert the text model of a HF Gemma 4 E2B/E4B checkpoint into a flat, mmap-able fp16 file.

File layout (same as the Llama kernel's):
    [0:8)     little-endian u64 = header length H
    [8:8+H)   UTF-8 JSON header: {"config": {...}, "engine": {...}, "tensors": {name: {"offset", "shape", "dtype"}}}
    ...       tensor data; every tensor starts on a 256-byte boundary, offsets are absolute
    file is padded to a 16 KiB multiple so Swift can wrap the whole mmap in one MTLBuffer.

"config" is the checkpoint's text_config. "engine" is what main.swift needs per layer, resolved
here so the Swift side has no model logic: head_dim, ffn_dim, sliding, kv_layer (the layer whose
KV cache this layer reads; itself unless KV-shared), RoPE theta and number of rotated pairs, and
layer_scalar.

Per layer, Wq|Wk|Wv are fused into one matrix (Wq only for KV-shared layers, whose K/V weights
are unused). Norm weights are fp32, everything else fp16. Vision and audio towers are skipped.
Tensors are converted one at a time (bf16 -> fp16), so peak memory stays small.
"""

import argparse
import glob
import json
import os

import torch
from safetensors import safe_open

ALIGN = 256
PAGE = 16384
PREFIX = "model.language_model."


def find_snapshot(repo: str) -> str:
    base = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots")
    for snap in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(snap, "model.safetensors")):
            return snap
    raise FileNotFoundError(f"no model.safetensors under {base}")


def engine_config(cfg: dict) -> dict:
    """Per-layer facts, mirroring transformers' modeling_gemma4.py."""
    n = cfg["num_hidden_layers"]
    types = cfg["layer_types"]
    first_shared = n - cfg.get("num_kv_shared_layers", 0)
    # A KV-shared layer reads the cache of the last non-shared layer of the same type.
    last_of_type = {t: i for i, t in enumerate(types[:first_shared])}
    layers = []
    for i, t in enumerate(types):
        sliding = t == "sliding_attention"
        hd = cfg["head_dim"] if sliding else cfg["global_head_dim"]
        rope = cfg["rope_parameters"][t]
        if rope["rope_type"] == "default":
            pairs = hd // 2
        else:
            assert rope["rope_type"] == "proportional", rope
            pairs = int(rope.get("partial_rotary_factor", 1.0) * hd // 2)
        shared = i >= first_shared
        double = cfg.get("use_double_wide_mlp", False) and shared
        layers.append(
            {
                "head_dim": hd,
                "ffn_dim": cfg["intermediate_size"] * (2 if double else 1),
                "sliding": sliding,
                "kv_layer": last_of_type[t] if shared else i,
                "rope_theta": rope["rope_theta"],
                "rope_pairs": pairs,
            }
        )
    return {"layers": layers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="google/gemma-4-E2B-it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap = find_snapshot(args.repo)
    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)["text_config"]
    assert not cfg.get("enable_moe_block") and not cfg.get("attention_k_eq_v"), "only the dense E2B/E4B layout"
    assert cfg["num_key_value_heads"] == 1 or cfg.get("num_global_key_value_heads") is None
    st = safe_open(os.path.join(snap, "model.safetensors"), "pt")
    engine = engine_config(cfg)

    # (name, shape, dtype, loader); loaders run one at a time while writing.
    plan: list[tuple[str, list[int], torch.dtype, object]] = []

    def shape(src: str) -> list[int]:
        return list(st.get_slice(PREFIX + src).get_shape())

    def add(name: str, src: str | list[str], dtype: torch.dtype = torch.float16) -> None:
        srcs = [src] if isinstance(src, str) else src
        shapes = [shape(s) for s in srcs]
        shp = [sum(s[0] for s in shapes), *shapes[0][1:]]
        plan.append((name, shp, dtype, lambda: torch.cat([st.get_tensor(PREFIX + s) for s in srcs], dim=0)))

    add("embed", "embed_tokens.weight")
    add("ple_embed", "embed_tokens_per_layer.weight")
    add("ple_proj", "per_layer_model_projection.weight")
    add("ple_norm", "per_layer_projection_norm.weight", torch.float32)
    for i, lay in enumerate(engine["layers"]):
        p = f"layers.{i}."
        lay["layer_scalar"] = st.get_tensor(PREFIX + p + "layer_scalar").float().item()
        own_kv = lay["kv_layer"] == i
        add(f"l{i}.attn_norm", p + "input_layernorm.weight", torch.float32)
        add(f"l{i}.wqkv", [p + f"self_attn.{n}_proj.weight" for n in ("qkv" if own_kv else "q")])
        add(f"l{i}.q_norm", p + "self_attn.q_norm.weight", torch.float32)
        if own_kv:
            add(f"l{i}.k_norm", p + "self_attn.k_norm.weight", torch.float32)
        add(f"l{i}.wo", p + "self_attn.o_proj.weight")
        add(f"l{i}.post_attn_norm", p + "post_attention_layernorm.weight", torch.float32)
        add(f"l{i}.ffn_norm", p + "pre_feedforward_layernorm.weight", torch.float32)
        add(f"l{i}.wgate", p + "mlp.gate_proj.weight")
        add(f"l{i}.wup", p + "mlp.up_proj.weight")
        add(f"l{i}.wdown", p + "mlp.down_proj.weight")
        add(f"l{i}.post_ffn_norm", p + "post_feedforward_layernorm.weight", torch.float32)
        add(f"l{i}.ple_gate", p + "per_layer_input_gate.weight")
        add(f"l{i}.ple_out", p + "per_layer_projection.weight")
        add(f"l{i}.post_ple_norm", p + "post_per_layer_input_norm.weight", torch.float32)
    add("final_norm", "norm.weight", torch.float32)
    assert cfg.get("tie_word_embeddings", True), "LM head is the tied embedding"

    # Lay out offsets; header size depends on offsets, so reserve a fixed prefix.
    header_reserve = 64 * 1024
    off = header_reserve
    meta = {}
    for name, shp, dtype, _ in plan:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        meta[name] = {"offset": off, "shape": shp, "dtype": str(dtype).removeprefix("torch.")}
        off += torch.Size(shp).numel() * dtype.itemsize
    total = (off + PAGE - 1) // PAGE * PAGE
    header = json.dumps({"config": cfg, "engine": engine, "tensors": meta}).encode()
    assert 8 + len(header) <= header_reserve

    with open(args.out, "wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
        for name, shp, dtype, load in plan:
            t = load().to(dtype)
            assert list(t.shape) == shp and torch.isfinite(t).all(), f"{name} overflows {dtype}"
            f.seek(meta[name]["offset"])
            f.write(t.contiguous().view(torch.uint8).numpy().tobytes())
        f.truncate(total)
    print(f"wrote {args.out}: {len(plan)} tensors, {total / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
