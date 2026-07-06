#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

MODEL=${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-4Expert-GGUF/ds4flash-4expert.gguf}
BINARY=${BINARY:-$REPO_ROOT/build-ds4-moe-stream-batch/bin/llama-cli}
OUT_ROOT=${OUT_ROOT:-/root/lfz/runs/vendor-ds4-16gb}
RUN_NAME=${RUN_NAME:-4expert-france-strict-smoke}
CPU_MOE=${CPU_MOE:-40}
VRAM_CACHE_GB=${VRAM_CACHE_GB:-0}
ONE_CACHE_MIB=${ONE_CACHE_MIB:-13568}
EXPECTED_SIZE=${EXPECTED_SIZE:-164465760544}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ART_DIR="$REPO_ROOT/.Agent/runs/20260705-vendor-ds4-coldstart"
READY_JSON="$ART_DIR/4expert-ready-validation-complete-${STAMP}.json"

mkdir -p "$ART_DIR"

if [ ! -x "$BINARY" ]; then
  echo "missing executable binary: $BINARY" >&2
  exit 2
fi

# This is the hard gate before any load/correctness/perf claim. It fails if the
# aria2 sidecar still exists, the size is wrong, or required GGUF metadata is absent.
.Agent/run-tools/validate_4expert_ready.py \
  --model "$MODEL" \
  --expected-size "$EXPECTED_SIZE" \
  --sha256 \
  --output "$READY_JSON"

python3 - <<PY
import json
from pathlib import Path
p = Path("$READY_JSON")
d = json.loads(p.read_text())
if not d.get("complete_and_ready_for_load_validation"):
    raise SystemExit(f"4Expert file is not ready: {d.get('failures')}")
print("ready_json", p)
print("sha256", d.get("sha256"))
PY

# First strict smoke is correctness-oriented. It uses the real complete file,
# cold drop_caches, MemoryMax=16000000000, MemorySwapMax=0, and records cgroup
# page cache in the same format as prior DS4 runs. It is not a generalized SOTA
# claim; held-out prompts remain locked until a candidate is frozen.
.Agent/run-tools/strict_ds4_runner.py \
  --binary "$BINARY" \
  --model "$MODEL" \
  --out-root "$OUT_ROOT" \
  --run-name "${STAMP}-${RUN_NAME}" \
  --prompt "Please introduce France in a short paragraph." \
  --case-name france-4expert \
  --cpu-moe "$CPU_MOE" \
  --vram-cache-gb "$VRAM_CACHE_GB" \
  --drop-caches-before-case \
  --env LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1 \
  --env GGML_MOE_STREAM_ONE_Q4K=1 \
  --env GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1 \
  --env GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps \
  --env GGML_MOE_STREAM_ONE_CACHE_MIB="$ONE_CACHE_MIB" \
  --env GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct
