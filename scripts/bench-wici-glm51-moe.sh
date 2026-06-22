#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build-cuda/bin/llama-cli}"
MODEL="${MODEL:-$ROOT/models/GLM-5.1-UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf}"
EXPERT_PACK="${EXPERT_PACK:-$ROOT/models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack}"
OUT_DIR="${OUT_DIR:-$ROOT/bench/wici-glm51-moe}"
TOKENS="${TOKENS:-16}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Write a concise explanation of why NVMe latency matters for MoE inference.}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)-n${TOKENS}}"
USE_DEFAULT_ARGS="${USE_DEFAULT_ARGS:-1}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-60}"
MLA="${MLA:-3}"
CPU_MOE="${CPU_MOE:-1}"
IGNORE_EOS="${IGNORE_EOS:-1}"

mkdir -p "$OUT_DIR"

STDOUT_LOG="$OUT_DIR/${RUN_NAME}.stdout.txt"
STDERR_LOG="$OUT_DIR/${RUN_NAME}.stderr.txt"
ROUTE_PROFILE="$OUT_DIR/${RUN_NAME}.route.csv"
ENV_LOG="$OUT_DIR/${RUN_NAME}.env.txt"

export GGML_MOE_BATCH_PROFILE="${GGML_MOE_BATCH_PROFILE:-1}"
export GGML_MOE_BATCH_PROFILE_OUT="${GGML_MOE_BATCH_PROFILE_OUT:-$ROUTE_PROFILE}"
export GGML_MOE_EXPERT_PACK="${GGML_MOE_EXPERT_PACK:-$EXPERT_PACK}"
export GGML_MOE_VRAM_PROFILE="${GGML_MOE_VRAM_PROFILE:-$ROOT/presets/moe/groundtruth/wici-glm51-interactive-n84.route.csv}"
export GGML_MOE_VRAM_PROFILE_PROTECT="${GGML_MOE_VRAM_PROFILE_PROTECT:-1}"
export GGML_MOE_VRAM_PROFILE_RESERVE_PCT="${GGML_MOE_VRAM_PROFILE_RESERVE_PCT:-10}"
export GGML_MOE_STREAM="${GGML_MOE_STREAM:-1}"
export GGML_MOE_STREAM_BATCH_ONLY="${GGML_MOE_STREAM_BATCH_ONLY:-1}"
export GGML_MOE_STREAM_DEFER="${GGML_MOE_STREAM_DEFER:-1}"
export GGML_MOE_STREAM_CPU_OPS="${GGML_MOE_STREAM_CPU_OPS:-1}"
export GGML_MOE_PARALLEL_EXPERTS="${GGML_MOE_PARALLEL_EXPERTS:-1}"
export GGML_MOE_STREAM_FUSED_UP_GATE="${GGML_MOE_STREAM_FUSED_UP_GATE:-1}"
export GGML_MOE_GPU_HANDOFF="${GGML_MOE_GPU_HANDOFF:-0}"
export GGML_MOE_VRAM_CACHE_SPLIT="${GGML_MOE_VRAM_CACHE_SPLIT:-1}"
export GGML_MOE_STREAM_ONE_CACHE_MIB="${GGML_MOE_STREAM_ONE_CACHE_MIB:-0}"
export GGML_MOE_VRAM_CACHE_MIB="${GGML_MOE_VRAM_CACHE_MIB:-3072}"
export GGML_MOE_VRAM_CACHE_UPGATE_PCT="${GGML_MOE_VRAM_CACHE_UPGATE_PCT:-63}"
export GGML_MOE_VRAM_CACHE_POLICY="${GGML_MOE_VRAM_CACHE_POLICY:-lfu_lru}"
export GGML_MOE_STAGE_PINNED="${GGML_MOE_STAGE_PINNED:-0}"
export GGML_MOE_STAGE_PINNED_SLOTS="${GGML_MOE_STAGE_PINNED_SLOTS:-16}"
export GGML_MOE_PREFETCH_DOWN="${GGML_MOE_PREFETCH_DOWN:-0}"
export GGML_MOE_PREFETCH_DOWN_DEPTH="${GGML_MOE_PREFETCH_DOWN_DEPTH:-8}"
export GGML_MOE_STREAM_UP_GATE_PARALLEL="${GGML_MOE_STREAM_UP_GATE_PARALLEL:-0}"
export GGML_MOE_IO_BACKEND="${GGML_MOE_IO_BACKEND:-mmap}"
export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-1}"

{
    printf 'BIN=%s\n' "$BIN"
    printf 'MODEL=%s\n' "$MODEL"
    printf 'TOKENS=%s\n' "$TOKENS"
    printf 'SEED=%s\n' "$SEED"
    printf 'PROMPT=%s\n' "$PROMPT"
    printf 'USE_DEFAULT_ARGS=%s\n' "$USE_DEFAULT_ARGS"
    printf 'CTX_SIZE=%s\n' "$CTX_SIZE"
    printf 'N_GPU_LAYERS=%s\n' "$N_GPU_LAYERS"
    printf 'MLA=%s\n' "$MLA"
    printf 'CPU_MOE=%s\n' "$CPU_MOE"
    printf 'IGNORE_EOS=%s\n' "$IGNORE_EOS"
    env | LC_ALL=C sort | grep -E '^(GGML_MOE_|GGML_CUDA_|CUDA_|LLAMA_)' || true
} > "$ENV_LOG"

args=()
if [[ "$USE_DEFAULT_ARGS" != "0" ]]; then
    args+=(-c "$CTX_SIZE" -ngl "$N_GPU_LAYERS" -fa on -mla "$MLA")
    if [[ "$CPU_MOE" != "0" ]]; then
        args+=(-cmoe)
    fi
    if [[ "$IGNORE_EOS" != "0" ]]; then
        args+=(--ignore-eos)
    fi
fi

"$BIN" \
    -m "$MODEL" \
    -n "$TOKENS" \
    --seed "$SEED" \
    -p "$PROMPT" \
    "${args[@]}" \
    "$@" \
    > "$STDOUT_LOG" \
    2> "$STDERR_LOG"

{
    echo "stdout=$STDOUT_LOG"
    echo "stderr=$STDERR_LOG"
    echo "route_profile=$ROUTE_PROFILE"
    echo "env=$ENV_LOG"
    grep -E 'llama_print_timings|tok/s|moe_stream|VRAM cache|profile:' "$STDERR_LOG" || true
}
