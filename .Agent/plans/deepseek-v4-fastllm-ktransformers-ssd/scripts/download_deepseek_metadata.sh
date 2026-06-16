#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wici/lfz"
PLAN_DIR="$ROOT/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd"
MODEL_DIR="$PLAN_DIR/models/DeepSeek-V4-Flash-metadata"
HF_HOME_DIR="$PLAN_DIR/.hf"

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "$MODEL_DIR" "$HF_HOME_DIR"

PY="$ROOT/ktransformers/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

HF_CLI="$ROOT/ktransformers/.venv/bin/hf"
if [[ ! -x "$HF_CLI" ]]; then
  HF_CLI="hf"
fi

"$HF_CLI" download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "$MODEL_DIR" \
  --include "config.json" \
  --include "generation_config.json" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json" \
  --include "special_tokens_map.json" \
  --include "model.safetensors.index.json" \
  --include "*.model" \
  --include "*.tiktoken" \
  --include "README*" \
  --exclude "*.safetensors" \
  --exclude "*.bin" \
  --exclude "*.pt" \
  --exclude "*.pth"

echo "Downloaded metadata to $MODEL_DIR"
