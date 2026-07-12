#!/usr/bin/env bash
set -euo pipefail

# Default mode is a non-destructive dry run. Deleting old packs requires all of:
#   EXECUTE=1 DELETE_OLD_PACKS=1 CONFIRM_DELETE=DELETE_OLD_KIMI_NON_SOTA_PACKS

: "${REPO:=/root/lfz/llama.cpp-vendor-kimi}"
: "${MODEL_DIR:=/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF}"
: "${MODEL_PATH:=$MODEL_DIR/Kimi-K2.7-Code.i1-IQ1_S.gguf}"
: "${RUN_ROOT:=/root/lfz/runs/vendor-kimi-token-rate}"
: "${RUN_TAG:=kimi-iq1s-dev-n32}"
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
: "${RESUME_DOWNLOAD:=1}"
: "${MIN_FREE_AFTER_DOWNLOAD_GIB:=50}"
: "${DELETE_MANIFEST_PATH:=}"
: "${CURL_BIN:=curl}"

: "${IQ1S_BYTES:=204430872480}"
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
  "/root/lfz/runs/vendor-kimi-token-rate/20260708-gp146-dev-grouped-overlay-shadow/gp146-france-routefirst-overlay.expert-pack"
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack"
  "/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack"
)

ACTIVE_REPRO_FILES=(
  ".Agent/run-tools/kimi-general-prompt-repro.sh"
  ".Agent/run-tools/kimi_phase0_io_trace_remote.sh"
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

if [ "${KIMI_IQ1S_SYNTHETIC_DOWNLOAD_TEST:-0}" = "1" ]; then
  IQ1S_BYTES="${KIMI_IQ1S_SYNTHETIC_IQ1S_BYTES:-16}"
  PART_URLS=("synthetic://iq1s-part1")
  PART_BYTES=("$IQ1S_BYTES")
  PRESERVE_PATHS=()
  DELETE_CANDIDATES=()
fi

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
  log "execute=$EXECUTE delete_old_packs=$DELETE_OLD_PACKS download=$DOWNLOAD run_smoke=$RUN_SMOKE validate_parts=$VALIDATE_PARTS resume_download=$RESUME_DOWNLOAD"
  log "repo=$REPO"
  if [ -d "$REPO/.git" ]; then
    log "repo_branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    log "repo_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
  fi
  log "model_path=$MODEL_PATH"
  log "required_iq1s_bytes=$IQ1S_BYTES"
  local free_before delete_total projected_free projected_leftover
  free_before="$(bytes_free_for_path "$MODEL_DIR")"
  delete_total=0
  log "free_before=$free_before"
  log "min_free_after_download_bytes=$MIN_FREE_AFTER_DOWNLOAD_BYTES"
  log "preserve paths:"
  for path in "${PRESERVE_PATHS[@]}"; do
    log "  preserve $(path_size_bytes "$path") $path"
  done
  log "delete candidates:"
  for path in "${DELETE_CANDIDATES[@]}"; do
    local candidate_size
    candidate_size="$(path_size_bytes "$path")"
    delete_total=$((delete_total + candidate_size))
    log "  candidate $candidate_size $path"
  done
  projected_free=$((free_before + delete_total))
  projected_leftover=$((projected_free - IQ1S_BYTES))
  log "delete_candidate_total=$delete_total"
  log "projected_free_after_candidate_delete=$projected_free"
  log "projected_leftover_after_iq1s_download=$projected_leftover"
  if [ "$projected_leftover" -ge "$MIN_FREE_AFTER_DOWNLOAD_BYTES" ]; then
    log "projected_space_ready=1"
  else
    log "projected_space_ready=0"
  fi
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
expected_total = 204430872480
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

if total != expected_total:
    print(f"[kimi_iq1s_prepare] ERROR total_size got={total} expected={expected_total}", file=sys.stderr)
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

validate_delete_candidates() {
  local candidate preserve refs file
  for candidate in "${DELETE_CANDIDATES[@]}"; do
    for preserve in "${PRESERVE_PATHS[@]}"; do
      if [ "$candidate" = "$preserve" ]; then
        log "ERROR delete candidate overlaps preserve path: $candidate"
        return 1
      fi
    done

    if [ -d "$REPO/.git" ]; then
      for file in "${ACTIVE_REPRO_FILES[@]}"; do
        if [ ! -f "$REPO/$file" ]; then
          continue
        fi
        refs="$(git -C "$REPO" grep -n -F -- "$candidate" -- "$file" 2>/dev/null || true)"
        if [ -n "$refs" ]; then
          log "ERROR delete candidate referenced by active repro file: $candidate"
          printf '%s\n' "$refs"
          return 1
        fi
      done
    fi
  done
  log "delete_candidate_safety_check=ok"
}

projected_space_check() {
  local free_before delete_total candidate_size projected_free projected_leftover
  free_before="$(bytes_free_for_path "$MODEL_DIR")"
  delete_total=0
  for candidate in "${DELETE_CANDIDATES[@]}"; do
    candidate_size="$(path_size_bytes "$candidate")"
    delete_total=$((delete_total + candidate_size))
  done
  projected_free=$((free_before + delete_total))
  projected_leftover=$((projected_free - IQ1S_BYTES))
  log "projected_space_check_free_before=$free_before"
  log "projected_space_check_delete_total=$delete_total"
  log "projected_space_check_free_after_delete=$projected_free"
  log "projected_space_check_leftover_after_iq1s_download=$projected_leftover"
  if [ "$projected_leftover" -lt "$MIN_FREE_AFTER_DOWNLOAD_BYTES" ]; then
    log "ERROR projected space check failed: leftover=$projected_leftover required=$MIN_FREE_AFTER_DOWNLOAD_BYTES"
    return 1
  fi
  log "projected_space_check=ok"
}

write_delete_manifest() {
  local manifest path size
  manifest="$DELETE_MANIFEST_PATH"
  if [ -z "$manifest" ]; then
    manifest="$MODEL_DIR/iq1s-delete-candidates-$(date -u +%Y%m%d-%H%M%SZ).tsv"
  fi
  mkdir -p "$(dirname "$manifest")"
  {
    printf 'kind\tbytes\tpath\n'
    for path in "${PRESERVE_PATHS[@]}"; do
      size="$(path_size_bytes "$path")"
      printf 'preserve\t%s\t%s\n' "$size" "$path"
    done
    for path in "${DELETE_CANDIDATES[@]}"; do
      size="$(path_size_bytes "$path")"
      printf 'delete_candidate\t%s\t%s\n' "$size" "$path"
    done
  } > "$manifest"
  log "delete_manifest=$manifest"
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
  validate_delete_candidates
  projected_space_check
  write_delete_manifest
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
    log "  resumable stream ${#PART_URLS[@]} parts into '$MODEL_PATH' via '$MODEL_PATH.tmp'"
    return
  fi

  mkdir -p "$MODEL_DIR"
  local tmp_path="$MODEL_PATH.tmp"
  if [ -e "$MODEL_PATH" ]; then
    log "ERROR model path already exists with unexpected size: $MODEL_PATH"
    return 1
  fi

  local tmp_size
  if [ -e "$tmp_path" ]; then
    tmp_size="$(stat -c '%s' "$tmp_path")"
    if [ "$tmp_size" -gt "$IQ1S_BYTES" ]; then
      log "ERROR temporary model is larger than expected: tmp_size=$tmp_size expected=$IQ1S_BYTES path=$tmp_path"
      return 1
    fi
    if [ "$RESUME_DOWNLOAD" != "1" ]; then
      log "ERROR temporary model exists but RESUME_DOWNLOAD is disabled: $tmp_path"
      return 1
    fi
    log "resume temporary model: path=$tmp_path bytes=$tmp_size"
  else
    : > "$tmp_path"
    tmp_size=0
  fi

  if [ "$tmp_size" -eq "$IQ1S_BYTES" ]; then
    log "temporary model already complete; finalizing"
    mv "$tmp_path" "$MODEL_PATH"
    sha256sum "$MODEL_PATH" > "$MODEL_PATH.sha256"
    return
  fi

  local i before after got expected part_offset completed_prefix
  completed_prefix="$tmp_size"
  for i in "${!PART_URLS[@]}"; do
    expected="${PART_BYTES[$i]}"
    if [ "$completed_prefix" -ge "$expected" ]; then
      log "skip completed part $((i + 1)) bytes=$expected"
      completed_prefix=$((completed_prefix - expected))
      continue
    fi
    part_offset="$completed_prefix"
    completed_prefix=0
    expected=$((expected - part_offset))
    before="$(stat -c '%s' "$tmp_path")"
    log "download part $((i + 1)) offset=$part_offset append_expected=$expected url=${PART_URLS[$i]}"
    local curl_rc
    set +e
    if [ "$part_offset" -gt 0 ]; then
      "$CURL_BIN" -L --fail --retry 5 --retry-delay 5 -r "${part_offset}-" "${PART_URLS[$i]}" >> "$tmp_path"
      curl_rc=$?
    else
      "$CURL_BIN" -L --fail --retry 5 --retry-delay 5 "${PART_URLS[$i]}" >> "$tmp_path"
      curl_rc=$?
    fi
    set -e
    if [ "$curl_rc" -ne 0 ]; then
      log "ERROR part $((i + 1)) curl failed rc=$curl_rc; truncating temp back to $before"
      truncate -s "$before" "$tmp_path"
      return 1
    fi
    after="$(stat -c '%s' "$tmp_path")"
    got=$((after - before))
    if [ "$got" -ne "$expected" ]; then
      log "ERROR part $((i + 1)) size mismatch got=$got expected=$expected"
      truncate -s "$before" "$tmp_path"
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
  local run_dir
  run_dir="$RUN_ROOT/$(date -u +%Y%m%d-%H%M%SZ)-$RUN_TAG"
  local model_size
  model_size="$(path_size_bytes "$MODEL_PATH")"
  if [ "$model_size" != "$IQ1S_BYTES" ]; then
    log "smoke_ready=0 model_size=$model_size expected=$IQ1S_BYTES"
    if [ "$EXECUTE" = "1" ]; then
      return 1
    fi
    log "dry-run smoke command after download:"
    printf '%s\n' \
      "cd '$REPO'" \
      "systemd-run --wait --collect --same-dir -p MemoryMax=$MEMORY_MAX -p MemorySwapMax=$MEMORY_SWAP_MAX env REPO='$REPO' RUN='$run_dir' N='$N' THREADS='$THREADS' PROMPT_ID=dev_france_regression PROMPT_USER_TEXT='Please introduce France in a short paragraph.' QUALITY_KEYWORDS='france,europe|paris|eiffel|louvre|riviera|bordeaux' PROFILE=1 COPY_PROFILE=0 MODEL_PATH='$MODEL_PATH' MOE_EXPERT_SOURCE=model .Agent/run-tools/kimi-general-prompt-repro.sh"
    return 0
  fi

  log "smoke_run=$run_dir"
  if [ "$EXECUTE" != "1" ]; then
    log "dry-run smoke command:"
    printf '%s\n' \
      "cd '$REPO'" \
      "systemd-run --wait --collect --same-dir -p MemoryMax=$MEMORY_MAX -p MemorySwapMax=$MEMORY_SWAP_MAX env REPO='$REPO' RUN='$run_dir' N='$N' THREADS='$THREADS' PROMPT_ID=dev_france_regression PROMPT_USER_TEXT='Please introduce France in a short paragraph.' QUALITY_KEYWORDS='france,europe|paris|eiffel|louvre|riviera|bordeaux' PROFILE=1 COPY_PROFILE=0 MODEL_PATH='$MODEL_PATH' MOE_EXPERT_SOURCE=model .Agent/run-tools/kimi-general-prompt-repro.sh"
    return
  fi

  cd "$REPO"
  systemd-run --wait --collect --same-dir \
    -p "MemoryMax=$MEMORY_MAX" -p "MemorySwapMax=$MEMORY_SWAP_MAX" \
    env REPO="$REPO" RUN="$run_dir" N="$N" THREADS="$THREADS" \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france,europe|paris|eiffel|louvre|riviera|bordeaux" \
      PROFILE=1 COPY_PROFILE=0 MODEL_PATH="$MODEL_PATH" MOE_EXPERT_SOURCE=model \
      .Agent/run-tools/kimi-general-prompt-repro.sh
}

print_inventory
validate_part_metadata
maybe_delete_old_packs
check_space
download_model
run_smoke
