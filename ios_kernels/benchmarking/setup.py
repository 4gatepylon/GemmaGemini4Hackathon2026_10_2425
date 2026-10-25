"""One-time setup: build llama.cpp (Metal) and produce the three model files (our kernel's
weights, llama.cpp GGUF, MLX) from the same HF safetensors checkpoint, then build the kernel.

    uv run --extra bench benchmarking/setup.py [--kernel NAME]   # default: newest kernel

Large artifacts go to $DATA_DIR (default: <repo>/.context, gitignored).
"""

import argparse
import subprocess
import sys

from huggingface_hub import snapshot_download

from surfaces.paths import LLAMA_DIR, MODELS_DIR, load_manifest, resolve_kernel


def run(*cmd) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", help="kernel folder name/substring/path (default: newest)")
    args = ap.parse_args()
    kernel = resolve_kernel(args.kernel)
    manifest = load_manifest(kernel)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not (LLAMA_DIR / "build/bin/llama-bench").exists():
        if not LLAMA_DIR.is_dir():
            run("git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_DIR)
        run(
            "cmake",
            "-S",
            LLAMA_DIR,
            "-B",
            LLAMA_DIR / "build",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_METAL=ON",
            "-DLLAMA_CURL=OFF",
        )
        run("cmake", "--build", LLAMA_DIR / "build", "-j", "10", "--target", "llama-bench", "llama-cli", "llama-server")

    # Full HF snapshot (weights + config + tokenizer) in the local cache.
    repo = manifest["hf_repo"]
    snap = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])

    weights = MODELS_DIR / manifest["weights"]
    gguf, mlx = MODELS_DIR / "llama1b_f16.gguf", MODELS_DIR / "llama1b_mlx_f16"
    if not weights.exists():
        run(sys.executable, kernel / manifest["convert"], "--repo", repo, "--out", weights)
    if not gguf.exists():
        run(sys.executable, LLAMA_DIR / "convert_hf_to_gguf.py", snap, "--outtype", "f16", "--outfile", gguf)
    if not mlx.is_dir():
        run(sys.executable, "-m", "mlx_lm", "convert", "--hf-path", repo, "--mlx-path", mlx, "--dtype", "float16")

    run(kernel / manifest["build"])


if __name__ == "__main__":
    main()
