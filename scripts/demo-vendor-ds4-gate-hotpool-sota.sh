#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

MODEL="${MODEL:-/home/wici/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
GATE_FULLPACK_PATH="${GATE_FULLPACK_PATH:-/home/wici/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.expert-pack}"
RUN_ROOT="${RUN_ROOT:-/home/wici/runs/vendor-ds4-16gb}"
MANIFEST="${GGML_MOE_STREAM_ONE_DIRECT_MANIFEST:-${RUN_ROOT}/ds4-gate-pack-first320.direct_manifest.csv}"
PROMPT="${PROMPT:-Explain quantum computing briefly.}"
MAX_TOKENS="${MAX_TOKENS:-96}"
RUN_LABEL="${RUN_LABEL:-gate-hotpool-first320-quantum-n96}"

[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }
[[ -f "$GATE_FULLPACK_PATH" ]] || { echo "missing expert pack: $GATE_FULLPACK_PATH" >&2; exit 2; }

mkdir -p "$(dirname "$MANIFEST")"
python3 scripts/ds4-expert-pack-direct-manifest.py \
  --pack "$GATE_FULLPACK_PATH" \
  --tensor-substr ffn_gate_exps \
  --limit 320 \
  --output "$MANIFEST"

bash -lc '
  systemctl stop display-manager || true
  pkill -f "[g]nome-shell" || true
  pkill -f "[X]org" || true
  pkill -f "/usr/lib/xorg/[X]org" || true
  pkill -f "[o]llama" || true
  pkill -f "[l]lama-cli" || true
  pkill -f "[m]ain" || true
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$f" 2>/dev/null || true
  done
  for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
    echo performance > "$f" 2>/dev/null || true
  done
'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true

MODEL="$MODEL" \
GATE_FULLPACK_PATH="$GATE_FULLPACK_PATH" \
RUN_LABEL="$RUN_LABEL" \
GGML_MOE_STREAM_ONE_DIRECT_MANIFEST="$MANIFEST" \
GGML_MOE_STREAM_ONE_DIRECT_MODEL="$GATE_FULLPACK_PATH" \
GGML_MOE_STREAM_ONE_DIRECT_IO=direct \
GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB=1400 \
GGML_MOE_STREAM_ONE_DIRECT_PREFILL_LIMIT=320 \
GGML_MOE_STREAM_ONE_DIRECT_PREFILL_ASYNC=1 \
GGML_MOE_ONE_PACK_READ_SUMMARY=1 \
./scripts/demo-vendor-ds4-general-sota.sh \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --gate-fullpack
