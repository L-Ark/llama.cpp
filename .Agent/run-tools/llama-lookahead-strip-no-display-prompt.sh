#!/usr/bin/env bash
set -euo pipefail
REAL=/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin/llama-lookahead
args=()
for arg in "$@"; do
  case "$arg" in
    --no-display-prompt) ;;
    *) args+=("$arg") ;;
  esac
done
exec "$REAL" "${args[@]}"
