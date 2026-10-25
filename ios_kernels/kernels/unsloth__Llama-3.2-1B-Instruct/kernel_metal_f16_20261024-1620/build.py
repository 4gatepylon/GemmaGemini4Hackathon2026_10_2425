#!/usr/bin/env python3
"""Build the Swift engine. Kernels are compiled at runtime, so no Metal toolchain is needed.

    uv run kernels/<model>/<kernel>/build.py   (or run it directly)
"""

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
binary = HERE / ".build" / "llm"
binary.parent.mkdir(exist_ok=True)
# absolute path so #filePath finds kernels.metal
subprocess.run(["swiftc", "-O", "-o", str(binary), str(HERE / "main.swift")], check=True)
print(f"built {binary}")
