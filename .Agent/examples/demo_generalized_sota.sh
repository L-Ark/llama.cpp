#!/usr/bin/env bash
set -euo pipefail

# General-prompt demo for the current vendor DeepSeek SOTA baseline.
#
# This script is intentionally prompt-general:
#   - accepts any user prompt
#   - does not load prompt-specific expert packs, traces, or profiles
#   - runs under a strict 16 GB host RAM cgroup, including page cache
#   - disables swap through strict_ds4_runner.py/systemd-run
#   - uses cold start by default with drop_caches
#
# Examples:
#   .Agent/examples/demo_generalized_sota.sh --prompt "What does AI infrastructure do?"
#   .Agent/examples/demo_generalized_sota.sh "今天吃什么？"
#   printf 'Introduce Japan briefly.\n' | .Agent/examples/demo_generalized_sota.sh --stdin-prompt
#   .Agent/examples/demo_generalized_sota.sh --prompt-file prompt.txt --n-predict 192
#   .Agent/examples/demo_generalized_sota.sh --fast-smoke --prompt "Explain database indexes briefly."
#
# The default n_predict=192 is meant for comparable measurements. Use
# --fast-smoke only to check that the runnable path works.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
RUNNER="${RUNNER:-$ROOT/.Agent/run-tools/strict_ds4_runner.py}"
BINARY="${BINARY:-$ROOT/build-ds4-moe-stream/bin/llama-cli}"
MODEL="${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
OUT_ROOT="${OUT_ROOT:-/root/lfz/runs/vendor-ds4-16gb}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/.Agent/runs/20260705-vendor-ds4-coldstart/general-prompt-baseline-no-prompt-specific-20260706.json}"

PROMPT=""
PROMPT_FILE=""
STDIN_PROMPT=0
RUN_NAME="demo-generalized-sota"
CASE_NAME=""
N_PREDICT=192
CTX_SIZE=256
BATCH_SIZE=16
UBATCH_SIZE=16
COLD_START=1
FAST_SMOKE=0
PRINT_COMMAND=0
PRINT_JSON=0
MIN_DEMO_TOK_S="${MIN_DEMO_TOK_S:-0}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

usage() {
  sed -n '1,31p' "$0"
  cat <<'EOF'

Options:
  --prompt TEXT        Prompt text.
  --prompt-file FILE   Read prompt from FILE.
  --stdin-prompt       Read prompt from stdin.
  --n-predict N        Decode token budget, default 192.
  --fast-smoke         Use n_predict=32 for quick runnable validation.
  --warm               Do not drop page cache first; not a cold SOTA metric.
  --case-name NAME     Case directory prefix.
  --run-name NAME      Run directory suffix.
  --min-tok-s N        Fail the demo if eval_tok_s is below N.
  --print-command      Print strict runner command and exit.
  --json               Emit a final [demo-json] line.
EOF
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
    print(text[:56])
else:
    print("prompt-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10])
PY
}

json_quote() {
  python3 - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1], ensure_ascii=False))
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
    raise SystemExit(0)

data = json.loads(path.read_text(encoding="utf-8"))
agg = data.get("aggregate", {})
print(f"[demo] generalized_baseline_artifact={path}")
print(f"[demo] generalized_baseline_eval_tok_s_min={agg.get('min_eval_tok_s')}")
print(f"[demo] generalized_baseline_eval_tok_s_mean={agg.get('mean_eval_tok_s')}")
print(f"[demo] generalized_baseline_eval_tok_s_max={agg.get('max_eval_tok_s')}")
print(f"[demo] generalized_baseline_ram_ok_all={agg.get('all_ram_ok')}")
print(f"[demo] generalized_baseline_target_gt_5_all_prompts={agg.get('target_met_all_prompts_gt_5_tok_s')}")
PY
}

unset_prompt_specific_env() {
  for var in \
    GGML_MOE_STREAM_ONE_EXPERT_PACK \
    GGML_MOE_STREAM_CACHE_ADMIT_PROFILE \
    GGML_MOE_STREAM_ONE_PREFILL_PROFILE \
    GGML_MOE_STREAM_ONE_PREFILL_LIMIT \
    GGML_MOE_EXPERT_PACK \
    GGML_MOE_EXPERT_PACK_OVERLAY \
    GGML_MOE_EXPERT_GGUF_ALIAS_TSV \
    GGML_MOE_IO_ALIGNED_ALIAS_BATCH \
    GGML_MOE_STREAM_ONE_TRACE_IN \
    GGML_MOE_STREAM_ONE_TRACE_OUT \
    GGML_MOE_EXPERT_ADMISSION_PROFILE \
    GGML_MOE_PROMPT_PROFILE \
    DS4_NATIVE_RETAINED_DOWN_PROBE_OUT \
    DS4_FUSED_UP_GATE_REF \
    DS4_FUSED_UP_GATE_REF_DEBUG_EXPLICIT \
    GGML_MOE_UP_GATE_LIMIT_ALLOW_FUSED
  do
    unset "$var" || true
  done
}

positional=()
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
    --min-tok-s)
      [[ $# -ge 2 ]] || die "missing value for --min-tok-s"
      MIN_DEMO_TOK_S="$2"
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
    --json)
      PRINT_JSON=1
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
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ ${#positional[@]} -gt 0 ]]; then
  [[ -z "$PROMPT" ]] || die "prompt already set; use --prompt or positional text, not both"
  PROMPT="${positional[*]}"
fi

input_modes=0
[[ -n "$PROMPT" ]] && input_modes=$((input_modes + 1))
[[ -n "$PROMPT_FILE" ]] && input_modes=$((input_modes + 1))
[[ "$STDIN_PROMPT" -eq 1 ]] && input_modes=$((input_modes + 1))
[[ "$input_modes" -le 1 ]] || die "use only one prompt input mode"

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
    die "missing prompt; pass --prompt TEXT, --prompt-file FILE, --stdin-prompt, or positional text"
  fi
fi

[[ -n "${PROMPT//[[:space:]]/}" ]] || die "prompt is empty"
[[ "$N_PREDICT" =~ ^[0-9]+$ ]] || die "--n-predict must be a positive integer"
[[ "$N_PREDICT" -gt 0 ]] || die "--n-predict must be a positive integer"
[[ "$MIN_DEMO_TOK_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--min-tok-s must be numeric"

cd "$ROOT"
for path in "$RUNNER" "$BINARY" "$MODEL"; do
  [[ -e "$path" ]] || die "missing required artifact: $path"
done

unset_prompt_specific_env

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
  --extra-arg="$CTX_SIZE"
  --extra-arg=-b
  --extra-arg="$BATCH_SIZE"
  --extra-arg=-ub
  --extra-arg="$UBATCH_SIZE"
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

printf '[demo] task=prompt-general vendor DeepSeek SOTA demo\n'
printf '[demo] repo=%s\n' "$ROOT"
printf '[demo] branch=%s\n' "$(git rev-parse --abbrev-ref HEAD)"
printf '[demo] commit=%s\n' "$(git rev-parse --short HEAD)"
printf '[demo] prompt=%s\n' "$(json_quote "$PROMPT")"
printf '[demo] n_predict=%s\n' "$N_PREDICT"
printf '[demo] mode=%s\n' "$([[ "$FAST_SMOKE" -eq 1 ]] && printf fast_smoke || printf comparable_default)"
printf '[demo] cold_start=%s\n' "$([[ "$COLD_START" -eq 1 ]] && printf true || printf false)"
printf '[demo] memory_limit_bytes=16000000000\n'
printf '[demo] page_cache_counted_in_cgroup=true\n'
printf '[demo] swap=disabled\n'
printf '[demo] prompt_specific_optimization=false\n'
printf '[demo] config=vendor_ds4,cpu_moe=40,vram_cache=0,gate_cache_mib=13568,no_prompt_pack,no_profile,no_trace\n'
printf '[demo] product_target=random prompts stable >5 tok/s on 16GB host RAM + 32GB RTX 5090\n'
print_generalized_baseline
if [[ "$FAST_SMOKE" -eq 1 ]]; then
  printf '[demo] note=fast_smoke validates runnable path only; use default n_predict=192 for comparable token-rate numbers\n'
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

python3 - "$summary" "$PRINT_JSON" "$MIN_DEMO_TOK_S" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
print_json = sys.argv[2] == "1"
min_demo_tok_s = float(sys.argv[3])
s = json.loads(summary.read_text(encoding="utf-8"))

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

answer = (s.get("answer") or "").strip()
print("\n[demo] answer")
print(answer if answer else "<empty>")

ram_ok = s.get("ram_ok") is True and not s.get("ram_limit_killed") and not s.get("oom_seen")
has_answer = bool(answer)
eval_tok_s = s.get("eval_tok_s")
speed_ok = min_demo_tok_s <= 0 or (isinstance(eval_tok_s, (int, float)) and eval_tok_s >= min_demo_tok_s)
if not ram_ok:
    status = "FAILED_STRICT_16GB_RAM_GATE"
elif not has_answer:
    status = "FAILED_EMPTY_ANSWER"
elif not speed_ok:
    status = "FAILED_MIN_DEMO_TOK_S"
else:
    status = "RUN_COMPLETED_STRICT_16GB"

print(f"\n[demo] status={status}")
if s.get("correctness_ok") is not True:
    print("[demo] correctness_note=runner heuristic is conservative for arbitrary prompts; manually review the answer above")
if answer and answer[-1] not in ".!?。！？)]}\"'`":
    print("[demo] answer_note=possibly_truncated_by_n_predict; increase --n-predict for a longer answer")

payload = {
    "status": status,
    "summary": str(summary),
    "case_dir": s.get("case_dir"),
    "eval_tok_s": s.get("eval_tok_s"),
    "prompt_tok_s": s.get("prompt_tok_s"),
    "ttft_estimate_ms": s.get("ttft_estimate_ms"),
    "elapsed_seconds": s.get("elapsed_seconds"),
    "memory_peak_bytes": s.get("memory_peak_bytes"),
    "memory_file_bytes": s.get("memory_file_bytes"),
    "ram_ok": s.get("ram_ok"),
    "correctness_ok": s.get("correctness_ok"),
    "correctness_reason": s.get("correctness_reason"),
    "answer": answer,
}
if print_json:
    print("[demo-json] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))

if status != "RUN_COMPLETED_STRICT_16GB":
    raise SystemExit(1)
PY

printf '\n[demo] cache/source counters\n'
grep -E 'VRAM cache: hits=|one expert pack: hits=|source stats:' "$case_dir/stderr.txt" || true

printf '\n[demo] exact command\n'
sed -n '1p' "$case_dir/exact_command.txt"

printf '\n[demo] run directory\n%s\n' "$case_dir"
