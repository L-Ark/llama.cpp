#!/usr/bin/env bash
set -euo pipefail
unset GGML_CUDA_DISABLE_GRAPHS
exec /root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin/llama-cli "$@"
