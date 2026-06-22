#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wici/lfz"
PLAN_DIR="$ROOT/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd"
LOG_DIR="$PLAN_DIR/results"
mkdir -p "$LOG_DIR"

TS="$(date -Iseconds)"
LOG="$LOG_DIR/resources-${TS//[:+]/_}.log"

{
  echo "# Resource Check"
  echo
  echo "- timestamp: $TS"
  echo "- host: $(hostname)"
  echo
  echo "## GPU"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
    echo
    echo "## GPU Processes"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "## Memory"
  free -h
  echo
  echo "## Disk"
  df -h "$ROOT"
} | tee "$LOG"

echo
echo "Wrote $LOG"
