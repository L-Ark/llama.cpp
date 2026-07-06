#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

MODEL=${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-4Expert-GGUF/ds4flash-4expert.gguf}
DOWNLOAD_SERVICE=${DOWNLOAD_SERVICE:-ds4-4expert-download.service}
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-28800}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG_DIR=${LOG_DIR:-/root/lfz/runs/vendor-ds4-16gb/${STAMP}-4expert-download-watch}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/watch.log"
READY_EXIT_FILE="$LOG_DIR/ready_exit_status.txt"
VALIDATION_EXIT_FILE="$LOG_DIR/validation_exit_status.txt"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "watch_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo=$REPO_ROOT"
echo "commit=$(git rev-parse HEAD)"
echo "model=$MODEL"
echo "download_service=$DOWNLOAD_SERVICE"
echo "log_dir=$LOG_DIR"

start_epoch=$(date +%s)
while true; do
  now_epoch=$(date +%s)
  elapsed=$((now_epoch - start_epoch))
  sidecar_present=0
  if [ -e "${MODEL}.aria2" ]; then
    sidecar_present=1
  fi
  service_state=$(systemctl is-active "$DOWNLOAD_SERVICE" 2>/dev/null || true)
  progress=$(tail -n 1 /root/lfz/models/DeepSeek-V4-Flash-4Expert-GGUF/aria2-4expert.log 2>/dev/null || true)
  stat_line=$(/usr/bin/stat --format='%n size=%s blocks=%b block_size=%B' "$MODEL" "${MODEL}.aria2" 2>/dev/null || true)
  echo "poll_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed=${elapsed}s service=${service_state:-unknown} sidecar=${sidecar_present} progress=${progress}"
  if [ -n "$stat_line" ]; then
    printf '%s\n' "$stat_line"
  fi

  if [ "$sidecar_present" -eq 0 ]; then
    echo "download_sidecar_absent=true"
    break
  fi
  if [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]; then
    echo "watch_timeout_after_seconds=$elapsed"
    exit 124
  fi
  if [ "$service_state" != "active" ] && [ "$sidecar_present" -eq 1 ]; then
    echo "download_service_not_active_but_sidecar_present=$service_state"
    systemctl --no-pager --full status "$DOWNLOAD_SERVICE" || true
    exit 125
  fi
  sleep "$SLEEP_SECONDS"
done

set +e
.Agent/run-tools/validate_4expert_ready.py \
  --model "$MODEL" \
  --sha256 \
  --output "$LOG_DIR/ready-validation.json"
ready_status=$?
set -e
echo "$ready_status" > "$READY_EXIT_FILE"
if [ "$ready_status" -ne 0 ]; then
  echo "ready_validation_failed=$ready_status"
  exit "$ready_status"
fi

set +e
RUN_NAME="4expert-france-strict-smoke-watch" \
  .Agent/run-tools/run_4expert_validation_after_download.sh
validation_status=$?
set -e
echo "$validation_status" > "$VALIDATION_EXIT_FILE"
echo "validation_exit_status=$validation_status"
echo "watch_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$validation_status"
