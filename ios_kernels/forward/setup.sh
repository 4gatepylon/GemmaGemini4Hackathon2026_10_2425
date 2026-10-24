#!/bin/sh
# One-time setup: build llama.cpp (Metal) and produce the three model files
# (our .bin, llama.cpp GGUF, MLX) from the same HF safetensors checkpoint.
# Large artifacts go to $DATA_DIR (default: <repo>/.context, gitignored).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/.context}"
HF_REPO="${HF_REPO:-unsloth/Llama-3.2-1B-Instruct}"
LLAMA_DIR="$DATA_DIR/deps/llama.cpp"
MODELS="$DATA_DIR/models"
mkdir -p "$DATA_DIR/deps" "$MODELS"

if [ ! -x "$LLAMA_DIR/build/bin/llama-bench" ]; then
    [ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_DIR/build" -j 10 --target llama-bench llama-cli
fi

# Make sure the full HF snapshot (weights + config + tokenizer) is in the cache.
python3 -c "from huggingface_hub import snapshot_download as s; print(s('$HF_REPO', allow_patterns=['*.json','*.safetensors']))" > "$DATA_DIR/snapshot_path"
SNAP="$(cat "$DATA_DIR/snapshot_path")"

[ -f "$MODELS/llama1b_f16.bin" ] || python3 "$HERE/convert_weights.py" --repo "$HF_REPO" --out "$MODELS/llama1b_f16.bin"
[ -f "$MODELS/llama1b_f16.gguf" ] || python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$SNAP" --outtype f16 --outfile "$MODELS/llama1b_f16.gguf"
[ -d "$MODELS/llama1b_mlx_f16" ] || python3 -m mlx_lm convert --hf-path "$HF_REPO" --mlx-path "$MODELS/llama1b_mlx_f16" --dtype float16

"$HERE/build.sh"
