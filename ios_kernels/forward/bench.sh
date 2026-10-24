#!/bin/sh
# Benchmark our engine vs llama.cpp vs MLX on the same FP16 Llama-3.2-1B.
# Usage: bench.sh [extra engine args, e.g. --define MV_NR=2]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/.context}"
MODELS="$DATA_DIR/models"
PP="${PP:-512}"; TG="${TG:-128}"; REPS="${REPS:-5}"

"$HERE/build.sh" > /dev/null

echo "== ours (hand-written Metal) =="
"$HERE/.build/llm" --weights "$MODELS/llama1b_f16.bin" --bench-pp "$PP" --bench-tg "$TG" --reps "$REPS" "$@" 2>&1 | grep -E "^(pp|tg)"

echo "== llama.cpp (Metal, $(git -C "$DATA_DIR/deps/llama.cpp" log -1 --format=%h)) =="
"$DATA_DIR/deps/llama.cpp/build/bin/llama-bench" -m "$MODELS/llama1b_f16.gguf" -p "$PP" -n "$TG" -r "$REPS" -ngl 99 2>/dev/null | grep -E "pp|tg"

echo "== MLX (mlx_lm; note its tg is measured after the pp prompt) =="
python3 -m mlx_lm benchmark --model "$MODELS/llama1b_mlx_f16" -p "$PP" -g "$TG" -n "$REPS" 2>/dev/null | grep Averages
