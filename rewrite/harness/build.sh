#!/bin/bash
# Build run_kernel for gfx1151 (run on Ai-core, where hipcc lives).
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O2 -std=c++17 -o run_kernel run_kernel.cpp
echo "built $(pwd)/run_kernel"
