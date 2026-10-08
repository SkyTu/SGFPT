#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${CUDA_HOME:=/usr/local/cuda-11.8}"
: "${JOBS:=4}"
: "${CMAKE:=cmake}"
"$CMAKE" -S ext/sytorch -B ext/sytorch/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_PREFIX_PATH="$PWD/.deps/usr${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
"$CMAKE" --build ext/sytorch/build --target sytorch --parallel "$JOBS"
# CUTLASS headers are vendored; no separate CUTLASS build or system changes are needed.

