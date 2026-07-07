#!/usr/bin/env bash
set -euo pipefail

# Prompt-general vendor DeepSeek SOTA demo under the project constraints.
#
# This demo intentionally does not use France-specific traces, prompt-specific
# expert packs, or prompt admission profiles. The user may pass any prompt; the
# run is cold by default and is executed by strict_ds4_runner.py inside a 16 GB
# cgroup that includes page cache.
#
# Examples:
#   .Agent/examples/demo_generalized_sota.sh --prompt "What does AI infrastructure do?"
#   .Agent/examples/demo_generalized_sota.sh "今天吃什么？"
#   printf 'Introduce Japan in one paragraph.\n' | .Agent/examples/demo_generalized_sota.sh --stdin-prompt
#   .Agent/examples/demo_generalized_sota.sh --prompt-file prompt.txt
#   .Agent/examples/demo_generalized_sota.sh
#
# Useful options:
#   --n-predict 192      Default comparable decode length.
#   --fast-smoke         Use n_predict=32 only to prove the path runs.
#   --warm               Keep page cache; not a cold-start SOTA metric.
#   --print-command      Print the strict runner command and exit.
#   --json               Print a final machine-readable JSON line.

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
PRINT_JSON=0
FAST_SMOKE=0

usage() {
  sed -n '1,31p' "$0"
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
print(text[:48] if text else "prompt-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10])
PY
}

print_baseline_banner() {
  python3 - "$BASELINE_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("[demo] generalized_baseline_artifact=missing")
    raise SystemExit(0)

data = json.loads(path.read_text())
agg = data.get("aggregate", {})
print(f"[demo] generalized_baseline_artifact={path}")
print("[demo] prompt_specific_optimization=disabled")
print(f"[demo] baseline_eval_tok_s_min={agg.get('min_eval_tok_s')}")
print(f"[demo] baseline_eval_tok_s_mean={agg.get('mean_eval_tok_s')}")
print(f"[demo] baseline_eval_tok_s_max={agg.get('max_eval_tok_s')}")
print(f"[demo] baseline_ram_ok_all={agg.get('all_ram_ok')}")
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
    die "missing prompt; pass --prompt TEXT, --prompt-file FILE, --stdin-prompt, or a positional prompt"
  fi
fi

[[ -n "${PROMPT//[[:space:]]/}" ]] || die "prompt is empty"
[[ "$N_PREDICT" =~ ^[0-9]+$ ]] || die "--n-predict must be a positive integer"
[[ "$N_PREDICT" -ge 1 ]] || die "--n-predict must be a positive integer"

cd "$ROOT"

for path in "$RUNNER" "$BINARY" "$MODEL"; do
  [[ -e "$path" ]] || die "missing required artifact: $path"
done

# Keep this demo prompt-general even if the caller's shell has leftovers from
# France/Kimi/GP experiments.
for var in \
  GGML_MOE_STREAM_ONE_EXPERT_PACK \
  GGML_MOE_STREAM_CACHE_ADMIT_PROFILE \
  GGML_MOE_STREAM_ONE_PREFILL_PROFILE \
  GGML_MOE_EXPERT_PACK \
  GGML_MOE_EXPERT_PACK_OVERLAY \
  GGML_MOE_EXPERT_GGUF_ALIAS_TSV \
  GGML_MOE_IO_ALIGNED_ALIAS_BATCH \
  GGML_MOE_STREAM_ONE_TRACE_IN \
  GGML_MOE_STREAM_ONE_TRACE_OUT \
  DS4_NATIVE_RETAINED_DOWN_PROBE_OUT \
  DS4_FUSED_UP_GATE_REF
do
  unset "$var" || true
done

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
printf '[demo] target=random prompt, stable >5 tok/s, 16GB host RAM including page cache, 32GB RTX 5090\n'
printf '[demo] current_status=generalized baseline/SOTA path, below product target; no prompt-specific optimization\n'
printf '[demo] config=vendor DeepSeek, cpu_moe=40, vram_cache=0, DS4 gate one-stream cache, cold strict cgroup\n'
print_baseline_banner
if [[ "$FAST_SMOKE" -eq 1 ]]; then
  printf '[demo] metric_mode=fast_smoke; use default --n-predict 192 for comparable numbers\n'
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

python3 - "$summary" "$PRINT_JSON" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
print_json = sys.argv[2] == "1"
s = json.loads(summary.read_text())

fields = [
    ("eval_tok_s", s.get("eval_tok_s")),
    ("prompt_tok_s", s.get("prompt_tok_s")),
    ("ttft_estimate_ms", s.get("ttft_estimate_ms")),
    ("elapsed_seconds", s.get("elapsed_seconds")),
    ("memory_peak_bytes", s.get("memory_peak_bytes")),
    ("memory_file_bytes", s.get("memory_file_bytes")),
    ("memory_max_events", s.get("memory_max_events")),
    ("pgmajfault", s.get("pgmajfault")),
    ("workingset_refault_file", s.get("workingset_refault_file")),
    ("ram_ok", s.get("ram_ok")),
    ("ram_limit_killed", s.get("ram_limit_killed")),
    ("oom_seen", s.get("oom_seen")),
    ("correctness_ok", s.get("correctness_ok")),
    ("correctness_reason", s.get("correctness_reason")),
]

print("\n[demo] metrics")
for key, value in fields:
    print(f"{key}={value}")

answer = (s.get("answer") or "").strip()
print("\n[demo] answer")
print(answer if answer else "<empty>")

ram_ok = s.get("ram_ok") is True and not s.get("ram_limit_killed") and not s.get("oom_seen")
has_answer = bool(answer)
if not ram_ok:
    status = "FAILED_STRICT_16GB_RAM_GATE"
elif not has_answer:
    status = "FAILED_EMPTY_ANSWER"
else:
    status = "RUN_COMPLETED_STRICT_16GB"

print(f"\n[demo] status={status}")
if s.get("correctness_ok") is not True:
    print("[demo] correctness_note=runner heuristic is conservative for arbitrary prompts; review the answer above.")
if answer and answer[-1] not in ".!?。！？)]}\"'":
    print("[demo] answer_note=possibly_truncated; increase --n-predict for a longer answer.")

if print_json:
    payload = {
        "status": status,
        "summary": str(summary),
        "eval_tok_s": s.get("eval_tok_s"),
        "prompt_tok_s": s.get("prompt_tok_s"),
        "ttft_estimate_ms": s.get("ttft_estimate_ms"),
        "memory_peak_bytes": s.get("memory_peak_bytes"),
        "memory_file_bytes": s.get("memory_file_bytes"),
        "ram_ok": s.get("ram_ok"),
        "correctness_ok": s.get("correctness_ok"),
        "correctness_reason": s.get("correctness_reason"),
        "answer": answer,
    }
    print("[demo-json] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))

if status != "RUN_COMPLETED_STRICT_16GB":
    raise SystemExit(1)
PY

printf '\n[demo] cache/source counters\n'
grep -E 'VRAM cache: hits=|one expert pack: hits=|source stats:' "$case_dir/stderr.txt" || true

printf '\n[demo] exact command\n'
sed -n '1p' "$case_dir/exact_command.txt"

printf '\n[demo] run directory\n%s\n' "$case_dir"
