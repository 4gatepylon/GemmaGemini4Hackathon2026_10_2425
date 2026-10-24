#!/bin/sh
# Benchmark a kernel vs llama.cpp vs MLX on the same FP16 Llama-3.2-1B.
# Usage: benchmarking/bench.sh [kernel folder] [extra engine args, e.g. --define MV_NR=2]
# Env: PP (default 512), TG (128), REPS (5).
set -e
. "$(dirname "$0")/common.sh"
[ $# -gt 0 ] && shift
PP="${PP:-512}"; TG="${TG:-128}"; REPS="${REPS:-5}"

"$KERNEL/$(manifest build)" > /dev/null

echo "== ours: $(basename "$KERNEL") =="
"$KERNEL/$(manifest binary)" --weights "$WEIGHTS" --bench-pp "$PP" --bench-tg "$TG" --reps "$REPS" "$@" 2>&1 | grep -E "^(pp|tg)"

echo "== llama.cpp (Metal, $(git -C "$LLAMA_DIR" log -1 --format=%h)) =="
"$LLAMA_DIR/build/bin/llama-bench" -m "$MODELS/llama1b_f16.gguf" -p "$PP" -n "$TG" -r "$REPS" -ngl 99 2>/dev/null | grep -E "pp|tg"

echo "== MLX (mlx_lm; note its tg is measured after the pp prompt) =="
uv run --project "$IOS" --extra bench python -m mlx_lm benchmark --model "$MODELS/llama1b_mlx_f16" -p "$PP" -g "$TG" -n "$REPS" 2>/dev/null | grep Averages
