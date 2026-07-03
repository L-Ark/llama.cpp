#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/lfz/vendor/llama.cpp-deepseek-v4}
BUILD_DIR=${BUILD_DIR:-$ROOT/build-ds4-moe-stream}
OUT=${OUT:-/tmp/mxfp4_transient_repack_harness}
ITERS=${1:-200}

cd "$ROOT"

echo "[transient-harness] cpu flags"
grep -m1 '^flags' /proc/cpuinfo || true

echo "[transient-harness] compiler"
g++ --version | head -n 1

cmd=(
  g++ -O3 -march=native -std=c++17
  -Iggml/include -Iggml/src -Iggml/src/ggml-cpu
  .Agent/run-tools/mxfp4_transient_repack_harness.cpp
  -L"$BUILD_DIR/bin" -lggml-cpu -lggml-base
  -Wl,-rpath,"$BUILD_DIR/bin"
  -pthread -ldl -lm
  -o "$OUT"
)

echo "[transient-harness] compile command"
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "[transient-harness] run"
"$OUT" "$ITERS"
