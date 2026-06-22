#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wici/lfz"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/DeepSeek-V4-Flash}"
PORT="${PORT:-30000}"

export HF_HOME="${HF_HOME:-$ROOT/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd/.hf}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export SGLANG_DSV4_MODE="${SGLANG_DSV4_MODE:-2604}"
export SGLANG_DSV4_2604_SUBMODE="${SGLANG_DSV4_2604_SUBMODE:-2604B}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "MODEL_DIR does not exist: $MODEL_DIR" >&2
  exit 2
fi

cd "$ROOT/ktransformers"

numactl --interleave=all ./.venv/bin/python -m sglang.launch_server \
  --host 0.0.0.0 --port "$PORT" \
  --model "$MODEL_DIR" \
  --kt-weight-path "$MODEL_DIR" \
  --kt-method MXFP4 \
  --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS:-10}" \
  --kt-cpuinfer "${KT_CPUINFER:-60}" \
  --kt-threadpool-count "${KT_THREADPOOL_COUNT:-2}" \
  --kt-gpu-prefill-token-threshold "${KT_GPU_PREFILL_TOKEN_THRESHOLD:-4096}" \
  --kt-enable-dynamic-expert-update \
  --tensor-parallel-size 1 \
  --context-length "${CONTEXT_LENGTH:-16384}" \
  --attention-backend flashinfer \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS:-2048}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-2}" \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --skip-server-warmup
