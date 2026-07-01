#!/usr/bin/env bash
set -uo pipefail
cd /root/lfz/vendor/llama.cpp-deepseek-v4
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_NO_PINNED=1
export GGML_MOE_VRAM_CACHE_MIB=16384
export LLAMA_MMAP_LOW_RAM=1
stdbuf -o0 -e0 build-a101-nvcc/bin/llama-completion -m /root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf -p 'Question: Please introduce France in a short paragraph.

Answer:' -ngl 8 -c 512 -n 96 --ignore-eos --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-warmup --no-display-prompt -no-cnv -ub 1 -t 24 -tb 24 > /root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/runs/20260630-ds4-token-rate/baseline-n96/stdout.txt 2> /root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/runs/20260630-ds4-token-rate/baseline-n96/stderr.txt
rc=$?
printf '%s\n' "$rc" > /root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/runs/20260630-ds4-token-rate/baseline-n96/exit_code.txt
exit "$rc"
