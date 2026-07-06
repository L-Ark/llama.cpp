#!/usr/bin/env bash
set -euo pipefail

# Default-safe helper for the AesSedai IQ2_XXS candidate.
#
# This script never deletes or downloads unless explicitly confirmed by env:
#   KIMI_IQ2_CLEANUP_OLD_PACKS=YES
#   KIMI_IQ2_DOWNLOAD=YES
#
# Typical flow after user approval:
#   KIMI_IQ2_CLEANUP_OLD_PACKS=YES .Agent/run-tools/kimi_iq2xxs_prepare_download.sh
#   KIMI_IQ2_DOWNLOAD=YES .Agent/run-tools/kimi_iq2xxs_prepare_download.sh

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MANIFEST="${MANIFEST:-$REPO_ROOT/.Agent/runs/20260707-gp18-iq2xxs-download-prep/download-manifest.json}"
TARGET_DIR="${TARGET_DIR:-/root/lfz/models/Kimi-K2.7-Code-GGUF-AesSedai-IQ2_XXS/IQ2_XXS}"
MIN_FREE_GIB="${MIN_FREE_GIB:-290}"
LOG_DIR="${LOG_DIR:-/root/lfz/runs/vendor-kimi-token-rate/iq2xxs-download-prep}"

OLD_PACKS=(
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/tmp-hot-upgate-pair-smoke.expert-pack
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-missing-down-overlay.expert-pack
)

PRESERVE=(
  /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack
  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack
)

bytes_free() {
  df -PB1 /root/lfz | awk 'NR == 2 { print $4 }'
}

gib() {
  awk -v b="$1" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }'
}

print_plan() {
  echo "repo_root=$REPO_ROOT"
  echo "manifest=$MANIFEST"
  echo "target_dir=$TARGET_DIR"
  echo "min_free_gib=$MIN_FREE_GIB"
  echo "free_gib=$(gib "$(bytes_free)")"
  echo
  echo "Preserve:"
  printf '  %s\n' "${PRESERVE[@]}"
  echo
  echo "Old pack cleanup candidates:"
  for path in "${OLD_PACKS[@]}"; do
    if [ -e "$path" ]; then
      du -h "$path"
    else
      echo "missing $path"
    fi
  done
}

cleanup_old_packs() {
  if [ "${KIMI_IQ2_CLEANUP_OLD_PACKS:-}" != "YES" ]; then
    echo "cleanup skipped: set KIMI_IQ2_CLEANUP_OLD_PACKS=YES to delete old packs"
    return
  fi
  mkdir -p "$LOG_DIR"
  {
    date -Is
    echo "Deleting old expert-pack candidates:"
    printf '%s\n' "${OLD_PACKS[@]}"
  } | tee "$LOG_DIR/cleanup.log"
  rm -f -- "${OLD_PACKS[@]}"
  df -h /root/lfz | tee -a "$LOG_DIR/cleanup.log"
}

check_space() {
  local free
  free="$(bytes_free)"
  local min_bytes
  min_bytes="$(awk -v g="$MIN_FREE_GIB" 'BEGIN { printf "%.0f", g * 1024 * 1024 * 1024 }')"
  if [ "$free" -lt "$min_bytes" ]; then
    echo "insufficient free space: free_gib=$(gib "$free"), required_gib=$MIN_FREE_GIB" >&2
    return 1
  fi
}

download_candidate() {
  if [ "${KIMI_IQ2_DOWNLOAD:-}" != "YES" ]; then
    echo "download skipped: set KIMI_IQ2_DOWNLOAD=YES to download candidate"
    return
  fi
  check_space
  mkdir -p "$TARGET_DIR" "$LOG_DIR"
  python3 - "$MANIFEST" "$TARGET_DIR" <<'PY'
import json
import pathlib
import subprocess
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
target = pathlib.Path(sys.argv[2])
for item in manifest["files"]:
    out = target / pathlib.Path(item["path"]).name
    cmd = [
        "curl", "-L", "--fail", "--retry", "10", "--continue-at", "-",
        "--output", str(out), item["url"],
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    actual = out.stat().st_size
    expected = int(item["bytes"])
    if actual != expected:
        raise SystemExit(f"size mismatch for {out}: got {actual}, expected {expected}")
PY
  find "$TARGET_DIR" -maxdepth 1 -type f -name '*.gguf' -printf '%s %p\n' | sort -n | tee "$LOG_DIR/downloaded-files.txt"
}

print_plan
cleanup_old_packs
download_candidate
