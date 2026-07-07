#!/usr/bin/env bash
set -euo pipefail

# Demo the current prompt-general vendor DeepSeek strict-16GB SOTA/reference path.
#
# This intentionally does NOT use France-specific expert packs, prompt-derived
# profiles, or cache-admission profiles. It is the no-prompt-specific path used
# for the generalized random-prompt target.
#
# Usage:
#   .Agent/examples/demo_current_sota.sh --prompt "AI infra is what?"
#   .Agent/examples/demo_current_sota.sh --stdin-prompt
#   printf 'Introduce Brazil briefly.\n' | .Agent/examples/demo_current_sota.sh --stdin-prompt
#   .Agent/examples/demo_current_sota.sh --prompt "Explain quantum computing briefly." --n-predict 96
#   .Agent/examples/demo_current_sota.sh --prompt "Today, what should I eat?" --print-command
#
# Optional path overrides:
#   ROOT=/path/to/checkout BINARY=/path/to/llama-cli MODEL=/path/to/model.gguf .Agent/examples/demo_current_sota.sh

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
RUNNER="${RUNNER:-$ROOT/.Agent/run-tools/strict_ds4_runner.py}"
BINARY="${BINARY:-$ROOT/build-ds4-moe-stream/bin/llama-cli}"
MODEL="${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
OUT_ROOT="${OUT_ROOT:-/root/lfz/runs/vendor-ds4-16gb}"

PROMPT=""
RUN_NAME="demo-general-sota"
CASE_NAME="custom"
N_PREDICT=128
PRINT_COMMAND=0
STDIN_PROMPT=0
COLD_START=1

usage() {
  sed -n '1,18p' "$0"
}

slugify() {
  python3 - "$1" <<'PY'
import re
import sys

text = sys.argv[1].strip().lower()
text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
print((text or "custom")[:40])
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:?missing value for --prompt}"
      shift 2
      ;;
    --stdin-prompt)
      STDIN_PROMPT=1
      shift
      ;;
    --run-name)
      RUN_NAME="${2:?missing value for --run-name}"
      shift 2
      ;;
    --case-name)
      CASE_NAME="${2:?missing value for --case-name}"
      shift 2
      ;;
    --n-predict)
      N_PREDICT="${2:?missing value for --n-predict}"
      shift 2
      ;;
    --warm)
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
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$STDIN_PROMPT" -eq 1 ]]; then
  PROMPT="$(cat)"
fi

if [[ -z "${PROMPT//[[:space:]]/}" ]]; then
  if [[ -t 0 ]]; then
    printf 'Prompt: ' >&2
    IFS= read -r PROMPT
  else
    echo "missing prompt; pass --prompt TEXT or --stdin-prompt" >&2
    exit 2
  fi
fi

if ! [[ "$N_PREDICT" =~ ^[0-9]+$ ]] || [[ "$N_PREDICT" -lt 1 ]]; then
  echo "--n-predict must be a positive integer" >&2
  exit 2
fi

cd "$ROOT"

missing=0
for path in "$RUNNER" "$BINARY" "$MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required artifact: $path" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 2
fi

if [[ "$CASE_NAME" == "custom" ]]; then
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

printf '[demo] prompt-general vendor DeepSeek SOTA/reference path\n'
printf '[demo] repo: %s\n' "$ROOT"
printf '[demo] commit: %s\n' "$(git rev-parse --short HEAD)"
printf '[demo] prompt: %s\n' "$PROMPT"
printf '[demo] n_predict: %s\n' "$N_PREDICT"
if [[ "$COLD_START" -eq 1 ]]; then
  printf '[demo] memory: strict cold run, drop_caches + 16GB cgroup including page cache\n'
else
  printf '[demo] memory: warm run, 16GB cgroup including page cache\n'
fi
printf '[demo] config: no prompt-specific expert pack/profile; gate one-stream cache only\n'

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

print("\n[demo] metrics")
for key in [
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
]:
    print(f"{key}={s.get(key)}")

print("\n[demo] answer")
print(s.get("answer", ""))
PY

printf '\n[demo] gate/cache counters\n'
grep -E 'VRAM cache: hits=|one expert pack: hits=|source stats:' "$case_dir/stderr.txt" || true

printf '\n[demo] exact command\n'
sed -n '1p' "$case_dir/exact_command.txt"

printf '\n[demo] run directory\n%s\n' "$case_dir"
