#!/usr/bin/env bash
set -euo pipefail

# Demo the current accepted vendor DeepSeek cold-start SOTA configuration.
# This runs the same strict 16GB cgroup setup used by the recorded SOTA audits:
#   - vendor framework
#   - cold drop_caches
#   - 16GB MemoryMax including page cache
#   - gate one-stream cache + one-prefill + O_DIRECT expert pack
#   - current top-k policy and small batch/context settings
#
# Usage:
#   .Agent/examples/demo_current_sota.sh
#   .Agent/examples/demo_current_sota.sh --print-command
#   .Agent/examples/demo_current_sota.sh --run-name my-demo

ROOT=/root/lfz/vendor/llama.cpp-deepseek-v4
RUNNER="$ROOT/.Agent/run-tools/strict_ds4_runner.py"
BINARY="$ROOT/build-ds4-moe-stream/bin/llama-cli"
OUT_ROOT=/root/lfz/runs/vendor-ds4-16gb
PROFILE="$ROOT/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv"
PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack
RUN_NAME="demo-current-sota"
PRINT_COMMAND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-command)
      PRINT_COMMAND=1
      shift
      ;;
    --run-name)
      RUN_NAME="${2:?missing value for --run-name}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,22p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

missing=0
for path in "$RUNNER" "$BINARY" "$PROFILE" "$PACK"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required artifact: $path" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 2
fi

cmd=(
  python3 "$RUNNER"
  --binary "$BINARY"
  --out-root "$OUT_ROOT"
  --run-name "$RUN_NAME"
  --cpu-moe 40
  --vram-cache-gb 0
  --memory-max-bytes 16000000000
  --ram-kill-threshold-bytes 16000000000
  --drop-caches-before-case
  --env CUDA_VISIBLE_DEVICES=0
  --env GGML_CUDA_DISABLE_GRAPHS=1
  --env GGML_MOE_STREAM=1
  --env GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1
  --env GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps
  --env GGML_MOE_STREAM_ONE_CACHE_MIB=13568
  --env GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000
  --env GGML_MOE_STREAM_ONE_PREFILL_PROFILE="$PROFILE"
  --env GGML_MOE_STREAM_DONTNEED=1
  --env GGML_MOE_KEEP_TOPK_UPDOWN=4
  --env GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39
  --env GGML_MOE_KEEP_TOPK_LAYER_VALUE=3
  --env GGML_MOE_STREAM_CACHE_ADMIT_PROFILE="$PROFILE"
  --env GGML_MOE_STREAM_ONE_EXPERT_PACK="$PACK"
  --env GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct
  --extra-arg=-c --extra-arg=256
  --extra-arg=-b --extra-arg=16
  --extra-arg=-ub --extra-arg=16
  --extra-arg=-t --extra-arg=20
  --extra-arg=-tb --extra-arg=20
)

if [[ "$PRINT_COMMAND" -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

before=$(mktemp)
after=$(mktemp)
trap 'rm -f "$before" "$after"' EXIT
find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort > "$before"

printf '[demo] running current vendor DeepSeek SOTA config\n'
printf '[demo] repo: %s\n' "$ROOT"
printf '[demo] this is a strict cold run: drop_caches + 16GB cgroup including page cache\n'
printf '[demo] expected strict-cold line: about 4.4 eval tok/s on this host\n'
"${cmd[@]}"

find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort > "$after"
run_dir=$(comm -13 "$before" "$after" | tail -n 1)
if [[ -z "$run_dir" ]]; then
  run_dir=$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-$RUN_NAME" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
case_dir="$run_dir/france-cpu40-vram0gb"
summary="$case_dir/summary.json"

if [[ ! -f "$summary" ]]; then
  echo "demo finished, but summary not found; run_dir=$run_dir" >&2
  exit 1
fi

python3 - "$summary" <<'PY'
import json
import sys
from pathlib import Path
summary = Path(sys.argv[1])
s = json.loads(summary.read_text())
print("\n[demo] summary")
for key in [
    "case_dir", "eval_tok_s", "prompt_tok_s", "ttft_estimate_ms",
    "elapsed_seconds", "memory_peak_bytes", "memory_file_bytes",
    "pgmajfault", "workingset_refault_file", "ram_ok",
    "ram_limit_killed", "correctness_ok", "correctness_reason",
]:
    print(f"{key}={s.get(key)}")
print("\n[demo] answer")
print(s.get("answer", ""))
PY

printf '\n[demo] gate/cache counters\n'
grep -E 'one expert pack: hits=|VRAM cache: hits=' "$case_dir/stderr.txt" || true

printf '\n[demo] exact command\n'
sed -n '1p' "$case_dir/exact_command.txt"

printf '\n[demo] run directory\n%s\n' "$case_dir"
