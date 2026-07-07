#!/usr/bin/env bash
set -euo pipefail

# Demo for the current prompt-general vendor DeepSeek strict-16GB baseline/SOTA.
#
# This is deliberately not the old France-specific SOTA path. It accepts any
# user prompt and clears prompt-derived packs/profiles before launching the
# strict runner. The run is cold by default: page cache is dropped inside the
# runner and the model process is placed in a 16GB cgroup including file cache.
#
# Usage:
#   .Agent/examples/demo_generalized_sota.sh --prompt "AI infra is what?"
#   .Agent/examples/demo_generalized_sota.sh "Today what should I eat?"
#   printf 'Introduce Japan briefly.\n' | .Agent/examples/demo_generalized_sota.sh --stdin-prompt
#   .Agent/examples/demo_generalized_sota.sh --prompt-file prompt.txt
#   .Agent/examples/demo_generalized_sota.sh
#
# Useful options:
#   --n-predict 192      Comparable decode length; default matches the baseline.
#   --fast-smoke         Use n_predict=32 only to prove the path runs.
#   --warm               Keep page cache; not a cold-start SOTA metric.
#   --print-command      Print the strict runner command and exit.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
RUNNER="${RUNNER:-$ROOT/.Agent/run-tools/strict_ds4_runner.py}"
BINARY="${BINARY:-$ROOT/build-ds4-moe-stream/bin/llama-cli}"
MODEL="${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
OUT_ROOT="${OUT_ROOT:-/root/lfz/runs/vendor-ds4-16gb}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/.Agent/runs/20260705-vendor-ds4-coldstart/general-prompt-baseline-no-prompt-specific-20260706.json}"

PROMPT=""
PROMPT_FILE=""
STDIN_PROMPT=0
RUN_NAME=""
CASE_NAME=""
N_PREDICT=192
COLD_START=1
PRINT_COMMAND=0
FAST_SMOKE=0

usage() {
  sed -n '1,28p' "$0"
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

raw = sys.argv[1]
text = raw.strip().lower()
text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
if text:
    print(text[:48])
else:
    print("prompt-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10])
PY
}

print_generalized_baseline() {
  python3 - "$BASELINE_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("[demo] generalized_baseline_artifact=missing")
    return_code = 0
    raise SystemExit(return_code)

data = json.loads(path.read_text())
agg = data.get("aggregate", {})
print(f"[demo] generalized_baseline_artifact={path}")
print("[demo] generalized_baseline_no_prompt_specific=true")
print(f"[demo] generalized_eval_tok_s_min={agg.get('min_eval_tok_s')}")
print(f"[demo] generalized_eval_tok_s_mean={agg.get('mean_eval_tok_s')}")
print(f"[demo] generalized_eval_tok_s_max={agg.get('max_eval_tok_s')}")
print(f"[demo] generalized_ram_ok_all={agg.get('all_ram_ok')}")
print(f"[demo] product_target_met_all_prompts_gt_5_tok_s={agg.get('target_met_all_prompts_gt_5_tok_s')}")
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
    --fast-smoke|--smoke)
      N_PREDICT=32
      FAST_SMOKE=1
      shift
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
      [[ -z "$PROMPT" ]] || die "prompt already set; pass only one prompt"
      PROMPT="$1"
      shift
      ;;
  esac
done

input_modes=0
[[ -n "$PROMPT" ]] && input_modes=$((input_modes + 1))
[[ -n "$PROMPT_FILE" ]] && input_modes=$((input_modes + 1))
[[ "$STDIN_PROMPT" -eq 1 ]] && input_modes=$((input_modes + 1))
[[ "$input_modes" -le 1 ]] || die "use only one of --prompt, --prompt-file, --stdin-prompt, or positional prompt"

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

for path in "$RUNNER" "$BINARY" "$MODEL"; do
  [[ -e "$path" ]] || die "missing required artifact: $path"
done

# Defensive cleanup: this demo must stay prompt-general even if the caller has
# leftovers from France/Kimi/GP experiments in the parent shell.
unset GGML_MOE_STREAM_ONE_EXPERT_PACK
unset GGML_MOE_STREAM_CACHE_ADMIT_PROFILE
unset GGML_MOE_STREAM_ONE_PREFILL_PROFILE
unset GGML_MOE_EXPERT_PACK
unset GGML_MOE_EXPERT_PACK_OVERLAY
unset GGML_MOE_EXPERT_GGUF_ALIAS_TSV
unset GGML_MOE_IO_ALIGNED_ALIAS_BATCH
unset GGML_MOE_STREAM_ONE_TRACE_IN
unset GGML_MOE_STREAM_ONE_TRACE_OUT

if [[ -z "$CASE_NAME" ]]; then
  CASE_NAME="$(slugify "$PROMPT")"
fi
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(date -u +%Y%m%dT%H%M%SZ)-demo-generalized-sota"
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

printf '[demo] prompt-general vendor DeepSeek strict-16GB demo\n'
printf '[demo] repo=%s\n' "$ROOT"
printf '[demo] branch=%s\n' "$(git rev-parse --abbrev-ref HEAD)"
printf '[demo] commit=%s\n' "$(git rev-parse --short HEAD)"
printf '[demo] prompt=%s\n' "$PROMPT"
printf '[demo] n_predict=%s\n' "$N_PREDICT"
printf '[demo] target=random prompt, 16GB host RAM including page cache, 32GB RTX 5090\n'
printf '[demo] config=vendor DeepSeek, no prompt-specific pack/profile, gate one-stream cache, cpu_moe=40, vram_cache=0\n'
print_generalized_baseline
if [[ "$FAST_SMOKE" -eq 1 ]]; then
  printf '[demo] metric_mode=fast_smoke; use --n-predict 192 for comparable baseline numbers\n'
else
  printf '[demo] metric_mode=comparable_default\n'
fi
if [[ "$COLD_START" -eq 1 ]]; then
  printf '[demo] memory_mode=cold, drop_caches before case, MemoryMax=16000000000, MemorySwapMax=0\n'
else
  printf '[demo] memory_mode=warm, MemoryMax=16000000000, MemorySwapMax=0; not a cold-start SOTA metric\n'
fi

"${cmd[@]}"

find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' 2>/dev/null | sort > "$after"
run_dir=$(comm -13 "$before" "$after" | tail -n 1)
if [[ -z "$run_dir" ]]; then
  run_dir=$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-$RUN_NAME" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
case_dir="$run_dir/${CASE_NAME}-cpu40-vram0gb"
summary="$case_dir/summary.json"

[[ -f "$summary" ]] || die "summary not found: $summary"

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
answer = (s.get("answer") or "").strip()
print(answer)

ram_ok = s.get("ram_ok") is True and not s.get("ram_limit_killed") and not s.get("oom_seen")
if not ram_ok:
    print("\n[demo] status=FAILED_STRICT_16GB_RAM_GATE")
    raise SystemExit(1)

print("\n[demo] status=RUN_COMPLETED_STRICT_16GB")
print("[demo] note=For arbitrary prompts, manually review the printed answer for semantic correctness.")
if answer and answer[-1] not in ".!?。！？)]}\"'":
    print("[demo] answer_note=possibly_truncated; increase --n-predict for a longer demo answer.")
PY

printf '\n[demo] cache/source counters\n'
grep -E 'VRAM cache: hits=|one expert pack: hits=|source stats:' "$case_dir/stderr.txt" || true

printf '\n[demo] exact command\n'
sed -n '1p' "$case_dir/exact_command.txt"

printf '\n[demo] run directory\n%s\n' "$case_dir"
