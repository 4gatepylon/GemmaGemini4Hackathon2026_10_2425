#!/bin/sh
# One-time setup: build llama.cpp (Metal) and produce the three model files (our kernel's
# weights, llama.cpp GGUF, MLX) from the same HF safetensors checkpoint, then build the kernel.
# Usage: benchmarking/setup.sh [kernel folder]   (default: newest)
# Large artifacts go to $DATA_DIR (default: <repo>/.context, gitignored).
set -e
. "$(dirname "$0")/common.sh"
UV="uv run --project $IOS --extra bench"
mkdir -p "$DATA_DIR/deps" "$MODELS"

if [ ! -x "$LLAMA_DIR/build/bin/llama-bench" ]; then
    [ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_DIR/build" -j 10 --target llama-bench llama-cli llama-server
fi

# Full HF snapshot (weights + config + tokenizer) in the local cache.
SNAP="$($UV python -c "from huggingface_hub import snapshot_download as s; print(s('$HF_REPO', allow_patterns=['*.json','*.safetensors']))" | tail -1)"

[ -f "$WEIGHTS" ] || $UV python "$KERNEL/$(manifest convert)" --repo "$HF_REPO" --out "$WEIGHTS"
[ -f "$MODELS/llama1b_f16.gguf" ] || $UV python "$LLAMA_DIR/convert_hf_to_gguf.py" "$SNAP" --outtype f16 --outfile "$MODELS/llama1b_f16.gguf"
[ -d "$MODELS/llama1b_mlx_f16" ] || $UV python -m mlx_lm convert --hf-path "$HF_REPO" --mlx-path "$MODELS/llama1b_mlx_f16" --dtype float16

"$KERNEL/$(manifest build)"
