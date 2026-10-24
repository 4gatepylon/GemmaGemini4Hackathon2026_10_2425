#!/bin/sh
# Build the Swift engine. Kernels are compiled at runtime, so no Metal toolchain is needed.
set -e
cd "$(dirname "$0")"
mkdir -p .build
swiftc -O -o .build/llm "$(pwd)/main.swift"  # absolute path so #filePath finds kernels.metal
echo "built $(pwd)/.build/llm"
