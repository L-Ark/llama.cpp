#!/usr/bin/env bash
set -euo pipefail

# Demonstrate the current prompt-general vendor DeepSeek strict-16GB SOTA path.
#
# This script is intentionally prompt-general:
# - no prompt-specific expert pack
# - no prompt-derived cache admission profile
# - no France-only prefill/profile path
#
# Usage:
#   .Agent/examples/demo_current_sota.sh --prompt "AI infra is what?"
#   .Agent/examples/demo_current_sota.sh "AI infra is what?"
#   .Agent/examples/demo_current_sota.sh --stdin-prompt
#   .Agent/examples/demo_current_sota.sh --prompt-file prompt.txt
#   .Agent/examples/demo_current_sota.sh
#
# Useful options:
#   --n-predict 192      Decode length. Default matches the generalized baseline.
#   --warm               Do not drop page cache before this case.
#   --print-command      Print the strict runner command without executing it.
#
# Optional path overrides:
#   ROOT=/path/to/repo BINARY=/path/to/llama-cli MODEL=/path/to/model.gguf \
#     .Agent/examples/demo_current_sota.sh --prompt "..."

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
RUNNER="${RUNNER:-$ROOT/.Agent/run-tools/strict_ds4_runner.py}"
BINARY="${BINARY:-$ROOT/build-ds4-moe-stream/bin/llama-cli}"
MODEL="${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
OUT_ROOT="${OUT_ROOT:-/root/lfz/runs/vendor-ds4-16gb}"

PROMPT=""
PROMPT_FILE=""
STDIN_PROMPT=0
RUN_NAME="demo-generalized-sota"
CASE_NAME=""
N_PREDICT=192
COLD_START=1
PRINT_COMMAND=0

usage() {
  sed -n '1,29p' "$0"
}

die() {
  echo "error: $*" >&2
  exit 2
}

slugify() {
  python3 - "$1" <<'PY'
import hashlib
import re
import sys

text = sys.argv[1].strip().lower()
text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
if text:
    print(text[:48])
else:
    digest = hashlib.sha1(sys.argv[1].encode("utf-8")).hexdigest()[:10]
    print(f"prompt-{digest}")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      [[ $# -ge 2 ]] || die "missing value for --prompt"
      PROMPT="$2"
      shift 2
      ;;
    --prompt-file)
      [[ $# -ge 2 ]] || die "missing value for --prompt-file"
      PROMPT_FILE="$2"
      shift 2
      ;;
    --stdin-prompt)
      STDIN_PROMPT=1
      shift
      ;;
    --run-name)
      [[ $# -ge 2 ]] || die "missing value for --run-name"
      RUN_NAME="$2"
      shift 2
      ;;
    --case-name)
      [[ $# -ge 2 ]] || die "missing value for --case-name"
      CASE_NAME="$2"
      shift 2
      ;;
    --n-predict)
      [[ $# -ge 2 ]] || die "missing value for --n-predict"
      N_PREDICT="$2"
      shift 2
      ;;
    --warm|--no-drop-caches)
      COLD_START=0
      shift
      ;;
    --print-command)
      PRINT_COMMAND=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      die "unknown argument: $1"
      ;;
    *)
      [[ -z "$PROMPT" ]] || die "prompt already set; use one prompt argument or --prompt"
      PROMPT="$1"
      shift
      ;;
  esac
done

input_modes=0
[[ -n "$PROMPT" ]] && input_modes=$((input_modes + 1))
[[ -n "$PROMPT_FILE" ]] && input_modes=$((input_modes + 1))
[[ "$STDIN_PROMPT" -eq 1 ]] && input_modes=$((input_modes + 1))
[[ "$input_modes" -le 1 ]] || die "use only one of --prompt, --prompt-file, or --stdin-prompt"

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || die "prompt file not found: $PROMPT_FILE"
  PROMPT="$(<"$PROMPT_FILE")"
elif [[ "$STDIN_PROMPT" -eq 1 ]]; then
  PROMPT="$(cat)"
elif [[ -z "${PROMPT//[[:space:]]/}" ]]; then
  if [[ -t 0 ]]; then
    printf 'Prompt: ' >&2
    IFS= read -r PROMPT
  else
    die "missing prompt; pass --prompt TEXT, --prompt-file FILE, or --stdin-prompt"
  fi
fi

[[ -n "${PROMPT//[[:space:]]/}" ]] || die "prompt is empty"
[[ "$N_PREDICT" =~ ^[0-9]+$ ]] || die "--n-predict must be a positive integer"
[[ "$N_PREDICT" -ge 1 ]] || die "--n-predict must be a positive integer"

cd "$ROOT"

missing=0
for path in "$RUNNER" "$BINARY" "$MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required artifact: $path" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || exit 2

# Make the demo defensively prompt-general even when the caller's shell has
# leftovers from prompt-specific SOTA/reproduction experiments.
unset GGML_MOE_STREAM_ONE_EXPERT_PACK
unset GGML_MOE_STREAM_CACHE_ADMIT_PROFILE
unset GGML_MOE_STREAM_ONE_PREFILL_PROFILE
unset GGML_MOE_EXPERT_PACK
unset GGML_MOE_EXPERT_PACK_OVERLAY
unset GGML_MOE_EXPERT_GGUF_ALIAS_TSV
unset GGML_MOE_IO_ALIGNED_ALIAS_BATCH

if [[ -z "$CASE_NAME" ]]; then
  CASE_NAME="$(slugify "$PROMPT")"
fi

cmd=(
  python3 "$RUNNER"
  --binary "$BINARY"
  --model "$MODEL"
  --out-root "$OUT_ROOT"
  --run-name "$RUN_NAME"
  --prompt "$PROMPT"
  --case-name "$CASE_NAME"
  --cpu-moe 40
  --vram-cache-gb 0
  --memory-max-bytes 16000000000
  --ram-kill-threshold-bytes 16000000000
  --env CUDA_VISIBLE_DEVICES=0
  --env GGML_CUDA_DISABLE_GRAPHS=1
  --env GGML_MOE_STREAM=1
  --env GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1
  --env GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps
  --env GGML_MOE_STREAM_ONE_CACHE_MIB=13568
  --env GGML_MOE_STREAM_DONTNEED=1
  --env GGML_MOE_KEEP_TOPK_UPDOWN=4
  --env GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39
  --env GGML_MOE_KEEP_TOPK_LAYER_VALUE=3
  --extra-arg=-n
  --extra-arg="$N_PREDICT"
  --extra-arg=-c
  --extra-arg=256
  --extra-arg=-b
  --extra-arg=16
  --extra-arg=-ub
  --extra-arg=16
)

if [[ "$COLD_START" -eq 1 ]]; then
  cmd+=(--drop-caches-before-case)
fi

if [[ "$PRINT_COMMAND" -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

before=$(mktemp)
after=$(mktemp)
trap 'rm -f "$before" "$after"' EXIT
find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' 2>/dev/null | sort > "$before"

printf '[demo] current prompt-general vendor DeepSeek strict-16GB path\n'
printf '[demo] repo=%s\n' "$ROOT"
printf '[demo] commit=%s\n' "$(git rev-parse --short HEAD)"
printf '[demo] prompt=%s\n' "$PROMPT"
printf '[demo] n_predict=%s\n' "$N_PREDICT"
printf '[demo] target_context=random/generalized prompt, 16GB host RAM including page cache, 32GB RTX 5090\n'
printf '[demo] current_prompt_general_sota=min 1.8 tok/s, mean 2.18 tok/s, max 2.7 tok/s on calibration prompts\n'
printf '[demo] config=no prompt-specific pack/profile; gate one-stream cache only; cpu_moe=40; vram_cache=0\n'
if [[ "$COLD_START" -eq 1 ]]; then
  printf '[demo] memory_mode=cold strict cgroup, drop_caches before case, MemoryMax=16000000000, MemorySwapMax=0\n'
else
  printf '[demo] memory_mode=warm strict cgroup, MemoryMax=16000000000, MemorySwapMax=0\n'
fi

"${cmd[@]}"

find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' 2>/dev/null | sort > "$after"
run_dir=$(comm -13 "$before" "$after" | tail -n 1)
if [[ -z "$run_dir" ]]; then
  run_dir=$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-$RUN_NAME" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
case_dir="$run_dir/${CASE_NAME}-cpu40-vram0gb"
summary="$case_dir/summary.json"

if [[ ! -f "$summary" ]]; then
  echo "demo finished, but summary not found; run_dir=$run_dir case_dir=$case_dir" >&2
  exit 1
fi

python3 - "$summary" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
s = json.loads(summary.read_text())

keys = [
    "eval_tok_s",
    "prompt_tok_s",
    "ttft_estimate_ms",
    "elapsed_seconds",
    "memory_peak_bytes",
    "memory_file_bytes",
    "memory_max_events",
    "pgmajfault",
    "workingset_refault_file",
    "ram_ok",
    "ram_limit_killed",
    "oom_seen",
    "correctness_ok",
    "correctness_reason",
]

print("\n[demo] metrics")
for key in keys:
    print(f"{key}={s.get(key)}")

print("\n[demo] answer")
answer = (s.get("answer") or "").strip()
print(answer)

ram_ok = s.get("ram_ok") is True and not s.get("ram_limit_killed") and not s.get("oom_seen")
if not ram_ok:
    print("\n[demo] status=FAILED_RAM_GATE")
    raise SystemExit(1)

print("\n[demo] status=RUN_COMPLETED_STRICT_16GB")
print("[demo] note=For arbitrary prompts, review the printed answer for semantic correctness.")
if answer and answer[-1] not in ".!?。！？)]}\"'":
    print("[demo] answer_note=possibly_truncated; rerun with a larger --n-predict for longer answers.")
PY

printf '\n[demo] gate/cache counters\n'
grep -E 'VRAM cache: hits=|one expert pack: hits=|source stats:' "$case_dir/stderr.txt" || true

printf '\n[demo] exact command\n'
sed -n '1p' "$case_dir/exact_command.txt"

printf '\n[demo] run directory\n%s\n' "$case_dir"
