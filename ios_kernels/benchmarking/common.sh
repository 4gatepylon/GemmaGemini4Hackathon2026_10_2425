# Shared by setup.sh / bench.sh. Sets IOS (ios_kernels/), KERNEL (kernel folder), DATA_DIR,
# MODELS, LLAMA_DIR, WEIGHTS (this kernel's weights file), HF_REPO.
# Kernel selection: first script argument, or $KERNEL, or the newest kernels/kernel_* folder
# (ordered by the trailing _YYYYMMDD-HHMM timestamp).
IOS="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$(dirname "$IOS")/.context}"
MODELS="$DATA_DIR/models"
LLAMA_DIR="$DATA_DIR/deps/llama.cpp"
KERNEL="${1:-${KERNEL:-$(for d in "$IOS"/kernels/kernel_*/; do echo "${d%/}"; done | awk -F_ '{print $NF" "$0}' | sort | tail -1 | cut -d' ' -f2-)}}"
case "$KERNEL" in /*) ;; *) [ -d "$IOS/kernels/$KERNEL" ] && KERNEL="$IOS/kernels/$KERNEL" ;; esac
[ -f "$KERNEL/kernel.json" ] || { echo "no kernel.json in '$KERNEL'" >&2; exit 1; }
manifest() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$KERNEL/kernel.json" "$1"; }
WEIGHTS="$MODELS/$(manifest weights)"
HF_REPO="$(manifest hf_repo)"
