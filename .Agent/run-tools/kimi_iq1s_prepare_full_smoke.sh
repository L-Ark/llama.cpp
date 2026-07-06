#!/usr/bin/env bash
set -euo pipefail

# Default mode is a non-destructive dry run. Deleting old packs requires all of:
#   EXECUTE=1 DELETE_OLD_PACKS=1 CONFIRM_DELETE=DELETE_OLD_KIMI_NON_SOTA_PACKS

: "${REPO:=/root/lfz/tmp/vendor-kimi-speculative-gp33}"
: "${MODEL_DIR:=/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF}"
: "${MODEL_PATH:=$MODEL_DIR/Kimi-K2.7-Code.i1-IQ1_S.gguf}"
: "${RUN_ROOT:=/root/lfz/runs/vendor-kimi-token-rate}"
: "${RUN_TAG:=gp35-iq1s-france-n32}"
: "${MEMORY_MAX:=15900000000}"
: "${MEMORY_SWAP_MAX:=0}"
: "${N:=32}"
: "${THREADS:=32}"
: "${EXECUTE:=0}"
: "${DELETE_OLD_PACKS:=0}"
: "${CONFIRM_DELETE:=}"
: "${DOWNLOAD:=1}"
: "${RUN_SMOKE:=1}"
: "${VALIDATE_PARTS:=1}"
: "${MIN_FREE_AFTER_DOWNLOAD_GIB:=20}"

IQ1S_BYTES=204429739520
GIB=1073741824
MIN_FREE_AFTER_DOWNLOAD_BYTES=$((MIN_FREE_AFTER_DOWNLOAD_GIB * GIB))

PRESERVE_PATHS=(
  "/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S"
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack"
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack"
)

DELETE_CANDIDATES=(
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack"
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack"
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack"
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack"
)

PART_URLS=(
  "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5"
  "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part2of5"
  "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part3of5"
  "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part4of5"
  "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part5of5"
)

PART_BYTES=(
  41875931136
  41875931136
  41875931136
  41875931136
  36927147936
)

log() {
  printf '[kimi_iq1s_prepare] %s\n' "$*"
}

bytes_free_for_path() {
  local path="$1"
  local parent="$path"
  while [ ! -e "$parent" ]; do
    parent="$(dirname "$parent")"
  done
  df -B1 --output=avail "$parent" | tail -n 1 | tr -d ' '
}

path_size_bytes() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo 0
    return
  fi
  du -sb "$path" | awk '{print $1}'
}

print_inventory() {
  log "date=$(date -Is)"
  log "execute=$EXECUTE delete_old_packs=$DELETE_OLD_PACKS download=$DOWNLOAD run_smoke=$RUN_SMOKE validate_parts=$VALIDATE_PARTS"
  log "repo=$REPO"
  if [ -d "$REPO/.git" ]; then
    log "repo_branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    log "repo_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
  fi
  log "model_path=$MODEL_PATH"
  log "required_iq1s_bytes=$IQ1S_BYTES"
  log "free_before=$(bytes_free_for_path "$MODEL_DIR")"
  log "preserve paths:"
  for path in "${PRESERVE_PATHS[@]}"; do
    log "  preserve $(path_size_bytes "$path") $path"
  done
  log "delete candidates:"
  for path in "${DELETE_CANDIDATES[@]}"; do
    log "  candidate $(path_size_bytes "$path") $path"
  done
}

validate_part_metadata() {
  if [ "$VALIDATE_PARTS" != "1" ]; then
    log "part metadata validation skipped"
    return
  fi

  python3 - <<'PY'
import json
import sys
import urllib.request

repo = "mradermacher/Kimi-K2.7-Code-i1-GGUF"
expected = {
    "Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5": 41875931136,
    "Kimi-K2.7-Code.i1-IQ1_S.gguf.part2of5": 41875931136,
    "Kimi-K2.7-Code.i1-IQ1_S.gguf.part3of5": 41875931136,
    "Kimi-K2.7-Code.i1-IQ1_S.gguf.part4of5": 41875931136,
    "Kimi-K2.7-Code.i1-IQ1_S.gguf.part5of5": 36927147936,
}
url = f"https://huggingface.co/api/models/{repo}?blobs=true"
req = urllib.request.Request(url, headers={"User-Agent": "kimi-iq1s-prepare/1.0"})
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    print(f"[kimi_iq1s_prepare] ERROR part metadata validation failed: {exc}", file=sys.stderr)
    sys.exit(1)

sizes = {}
for item in data.get("siblings", []):
    name = item.get("rfilename")
    if not name:
        continue
    size = item.get("size")
    if size is None and isinstance(item.get("lfs"), dict):
        size = item["lfs"].get("size")
    if size is None and isinstance(item.get("blob"), dict):
        size = item["blob"].get("size")
    if size is not None:
        sizes[name] = int(size)

ok = True
total = 0
for name, want in expected.items():
    got = sizes.get(name)
    if got != want:
        print(f"[kimi_iq1s_prepare] ERROR part_size name={name} got={got} expected={want}", file=sys.stderr)
        ok = False
    else:
        print(f"[kimi_iq1s_prepare] part_size_ok name={name} bytes={got}")
        total += got

if total != sum(expected.values()):
    print(f"[kimi_iq1s_prepare] ERROR total_size got={total} expected={sum(expected.values())}", file=sys.stderr)
    ok = False
else:
    print(f"[kimi_iq1s_prepare] part_total_ok bytes={total}")

sys.exit(0 if ok else 1)
PY
}

validate_preserve_paths() {
  for path in "${PRESERVE_PATHS[@]}"; do
    if [ ! -e "$path" ]; then
      log "ERROR missing preserve path: $path"
      return 1
    fi
  done
}

maybe_delete_old_packs() {
  if [ "$DELETE_OLD_PACKS" != "1" ]; then
    log "delete step skipped"
    return
  fi
  if [ "$CONFIRM_DELETE" != "DELETE_OLD_KIMI_NON_SOTA_PACKS" ]; then
    log "ERROR deletion requested without exact CONFIRM_DELETE token"
    return 1
  fi
  validate_preserve_paths
  for path in "${DELETE_CANDIDATES[@]}"; do
    if [ ! -e "$path" ]; then
      log "delete candidate missing, skip: $path"
      continue
    fi
    log "delete candidate: $path"
    if [ "$EXECUTE" = "1" ]; then
      rm -f -- "$path"
    fi
  done
  log "free_after_delete=$(bytes_free_for_path "$MODEL_DIR")"
}

check_space() {
  local existing_size
  existing_size="$(path_size_bytes "$MODEL_PATH")"
  if [ "$existing_size" = "$IQ1S_BYTES" ]; then
    log "model already present with expected size"
    return
  fi

  local free_bytes required_bytes missing_bytes
  free_bytes="$(bytes_free_for_path "$MODEL_DIR")"
  required_bytes=$((IQ1S_BYTES + MIN_FREE_AFTER_DOWNLOAD_BYTES))
  if [ "$free_bytes" -lt "$required_bytes" ]; then
    missing_bytes=$((required_bytes - free_bytes))
    log "space_ready=0 free=$free_bytes required=$required_bytes missing=$missing_bytes"
    if [ "$EXECUTE" = "1" ]; then
      return 1
    fi
    return 0
  fi
  log "space_ready=1 free=$free_bytes required=$required_bytes"
}

download_model() {
  if [ "$DOWNLOAD" != "1" ]; then
    log "download step skipped"
    return
  fi

  if [ "$(path_size_bytes "$MODEL_PATH")" = "$IQ1S_BYTES" ]; then
    log "download skipped, model already complete"
    return
  fi

  if [ "$EXECUTE" != "1" ]; then
    log "dry-run download command:"
    log "  mkdir -p '$MODEL_DIR'"
    log "  stream ${#PART_URLS[@]} parts into '$MODEL_PATH'"
    return
  fi

  mkdir -p "$MODEL_DIR"
  local tmp_path="$MODEL_PATH.tmp"
  if [ -e "$tmp_path" ]; then
    log "ERROR temporary model already exists: $tmp_path"
    return 1
  fi
  if [ -e "$MODEL_PATH" ]; then
    log "ERROR model path already exists with unexpected size: $MODEL_PATH"
    return 1
  fi

  : > "$tmp_path"
  local i before after got expected
  for i in "${!PART_URLS[@]}"; do
    before="$(stat -c '%s' "$tmp_path")"
    log "download part $((i + 1)) expected=${PART_BYTES[$i]} url=${PART_URLS[$i]}"
    curl -L --fail --retry 5 --retry-delay 5 "${PART_URLS[$i]}" >> "$tmp_path"
    after="$(stat -c '%s' "$tmp_path")"
    got=$((after - before))
    expected="${PART_BYTES[$i]}"
    if [ "$got" -ne "$expected" ]; then
      log "ERROR part $((i + 1)) size mismatch got=$got expected=$expected"
      return 1
    fi
  done

  local final_size
  final_size="$(stat -c '%s' "$tmp_path")"
  if [ "$final_size" -ne "$IQ1S_BYTES" ]; then
    log "ERROR final size mismatch got=$final_size expected=$IQ1S_BYTES"
    return 1
  fi
  mv "$tmp_path" "$MODEL_PATH"
  sha256sum "$MODEL_PATH" > "$MODEL_PATH.sha256"
  log "download complete: $MODEL_PATH"
}

run_smoke() {
  if [ "$RUN_SMOKE" != "1" ]; then
    log "smoke step skipped"
    return
  fi
  local model_size
  model_size="$(path_size_bytes "$MODEL_PATH")"
  if [ "$model_size" != "$IQ1S_BYTES" ]; then
    log "smoke_ready=0 model_size=$model_size expected=$IQ1S_BYTES"
    if [ "$EXECUTE" = "1" ]; then
      return 1
    fi
    return 0
  fi

  local run_dir
  run_dir="$RUN_ROOT/$(date -u +%Y%m%d-%H%M%SZ)-$RUN_TAG"
  log "smoke_run=$run_dir"
  if [ "$EXECUTE" != "1" ]; then
    log "dry-run smoke command:"
    printf '%s\n' \
      "cd '$REPO'" \
      "systemd-run --wait --collect --same-dir -p MemoryMax=$MEMORY_MAX -p MemorySwapMax=$MEMORY_SWAP_MAX env REPO='$REPO' RUN='$run_dir' N='$N' THREADS='$THREADS' PROMPT_ID=dev_france_regression PROMPT_USER_TEXT='Please introduce France in a short paragraph.' QUALITY_KEYWORDS='france,europe|paris|eiffel|louvre|riviera|bordeaux' PROFILE=1 COPY_PROFILE=0 MODEL_PATH='$MODEL_PATH' .Agent/run-tools/kimi-general-prompt-repro.sh"
    return
  fi

  cd "$REPO"
  systemd-run --wait --collect --same-dir \
    -p "MemoryMax=$MEMORY_MAX" -p "MemorySwapMax=$MEMORY_SWAP_MAX" \
    env REPO="$REPO" RUN="$run_dir" N="$N" THREADS="$THREADS" \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france,europe|paris|eiffel|louvre|riviera|bordeaux" \
      PROFILE=1 COPY_PROFILE=0 MODEL_PATH="$MODEL_PATH" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
}

print_inventory
validate_part_metadata
maybe_delete_old_packs
check_space
download_model
run_smoke
