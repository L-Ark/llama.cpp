#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/taskset -c 0-19 /root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin/llama-cli "$@"
