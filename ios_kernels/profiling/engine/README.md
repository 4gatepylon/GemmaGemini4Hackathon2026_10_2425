# engine

`kbench` specs that replay the existing Metal engine's forward pass (one decode token, one prompt chunk) with the engine's own `kernels.metal`, plus the map that lines its kernels up with llama.cpp's ops; `../tools/profile_engine.py` runs them.

The specs mirror the engine's dispatch geometry and default tunables, with random weights rotated
per layer (cold, like the real model). They never modify a kernel folder: the kernel source is
passed with `kbench --source`. `categories.json` assigns our step names and llama.cpp's ops
(by op and name) to shared categories for the comparison.
