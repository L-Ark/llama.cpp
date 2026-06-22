#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wici/lfz"
PLAN_DIR="$ROOT/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/DeepSeek-V4-Flash}"
PORT="${PORT:-30000}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
OUT_DIR="$PLAN_DIR/results"
mkdir -p "$OUT_DIR"

cd "$ROOT/ktransformers/third_party/sglang/python"

PY="$ROOT/ktransformers/.venv/bin/python"

"$PY" -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --random-range-ratio 0.0 \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate inf \
  --warmup-requests 1 \
  --output-details \
  --return-routed-experts \
  --output-file "$OUT_DIR/sglang-v4-flash-in${INPUT_LEN}-out${OUTPUT_LEN}-n${NUM_PROMPTS}.jsonl"
