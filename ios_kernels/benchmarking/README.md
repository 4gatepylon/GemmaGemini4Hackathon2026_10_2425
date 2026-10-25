# benchmarking

Scripts that set up the shared model files and measure our kernels' correctness and speed against HF transformers, llama.cpp, and MLX.

Run from `ios_kernels/` with `uv run --extra bench benchmarking/<script>.py`; `setup.py` runs
once first. Each script's docstring has its usage and flags. Large artifacts go in `$DATA_DIR`
(`<repo>/.context/` by default).
