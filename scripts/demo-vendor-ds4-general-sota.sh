#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/demo-vendor-ds4-general-sota.sh [--prompt TEXT | --prompt-file FILE | TEXT...] [options]

Options:
  --prompt TEXT          Prompt to run. Any user prompt is accepted.
  --prompt-file FILE     Read prompt from FILE. Use "-" for stdin.
  -n, --max-tokens N     Maximum generated tokens. Default: 128.
  --run-label LABEL      Label suffix for the run directory. Default: demo.
  --warm                 Skip drop_caches. Default is strict cold start.
  --print-command        Print the exact llama-cli command before execution.
  -h, --help             Show this help.

Purpose:
  Demo the current prompt-general vendor DeepSeek V4 path under strict
  16GB host RAM. This intentionally does not use the France-specialized
  pack/profile path. It rejects prompt-specific expert packs, route profiles,
  and alias TSVs so an operator can enter arbitrary prompts.

Artifacts:
  /root/lfz/runs/vendor-ds4-16gb/demo-general-sota/<timestamp>-<label>/
  contains prompt.txt, stdout.txt, stderr.txt, exact_command.txt,
  environment.txt, resource_samples.tsv, memory.* cgroup snapshots,
  config.json, and summary.json.
USAGE
}

fail() {
  echo "error: $*" >&2
  exit 2
}

json_quote() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read(), ensure_ascii=False))'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BINARY="${BINARY:-${REPO_DIR}/build-ds4-moe-stream/bin/llama-cli}"
MODEL="${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
RUN_ROOT="${RUN_ROOT:-/root/lfz/runs/vendor-ds4-16gb/demo-general-sota}"
BASELINE_ARTIFACT="${BASELINE_ARTIFACT:-${REPO_DIR}/.Agent/runs/20260705-vendor-ds4-coldstart/general-prompt-baseline-no-prompt-specific-20260706.json}"

MEMORY_MAX_BYTES=16000000000
MAX_TOKENS=128
RUN_LABEL="demo"
COLD=1
PRINT_COMMAND=0
PROMPT=""
PROMPT_FILE=""
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      [[ $# -ge 2 ]] || fail "missing value for --prompt"
      PROMPT="$2"
      shift 2
      ;;
    --prompt-file)
      [[ $# -ge 2 ]] || fail "missing value for --prompt-file"
      PROMPT_FILE="$2"
      shift 2
      ;;
    --max-tokens|-n)
      [[ $# -ge 2 ]] || fail "missing value for --max-tokens"
      MAX_TOKENS="$2"
      shift 2
      ;;
    --run-label)
      [[ $# -ge 2 ]] || fail "missing value for --run-label"
      RUN_LABEL="$2"
      shift 2
      ;;
    --warm)
      COLD=0
      shift
      ;;
    --print-command)
      PRINT_COMMAND=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      positional+=("$@")
      break
      ;;
    -*)
      fail "unknown option: $1"
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$PROMPT" && -n "$PROMPT_FILE" ]]; then
  fail "use only one of --prompt or --prompt-file"
fi
if [[ -n "$PROMPT" && "${#positional[@]}" -gt 0 ]]; then
  fail "use either --prompt or positional prompt text, not both"
fi
if [[ -n "$PROMPT_FILE" && "${#positional[@]}" -gt 0 ]]; then
  fail "use either --prompt-file or positional prompt text, not both"
fi
if [[ ! "$MAX_TOKENS" =~ ^[0-9]+$ ]] || [[ "$MAX_TOKENS" -le 0 ]]; then
  fail "--max-tokens must be a positive integer"
fi

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ "$PROMPT_FILE" == "-" ]]; then
    PROMPT="$(cat)"
  else
    [[ -f "$PROMPT_FILE" ]] || fail "prompt file not found: $PROMPT_FILE"
    PROMPT="$(cat "$PROMPT_FILE")"
  fi
elif [[ -n "$PROMPT" ]]; then
  :
elif [[ "${#positional[@]}" -gt 0 ]]; then
  PROMPT="${positional[*]}"
elif [[ -p /dev/stdin || ! -t 0 ]]; then
  PROMPT="$(cat)"
else
  echo "Enter prompt, then press Ctrl-D:" >&2
  PROMPT="$(cat)"
fi

PROMPT="$(printf '%s' "$PROMPT" | sed -e 's/[[:space:]]*$//')"
[[ -n "$PROMPT" ]] || fail "prompt is empty"
[[ -x "$BINARY" ]] || fail "missing executable: $BINARY"
[[ -f "$MODEL" ]] || fail "missing model: $MODEL"

prompt_specific_env=(
  GGML_MOE_STREAM_ONE_EXPERT_PACK
  GGML_MOE_EXPERT_PACK
  GGML_MOE_EXPERT_PACK_OVERLAY
  GGML_MOE_STREAM_CACHE_ADMIT_PROFILE
  GGML_MOE_STREAM_ONE_PREFILL_PROFILE
  GGML_MOE_EXPERT_GGUF_ALIAS_TSV
  GGML_MOE_IO_ALIGNED_ALIAS_BATCH
  GGML_MOE_STREAM_UP_DOWN_PROFILE
  GGML_MOE_STREAM_ONE_ROUTE_PROFILE
  GGML_DS4_GROUPED_RETAINED_ROUTE_PROFILE_OUT
  GGML_DS4_GROUPED_RETAINED_ROUTE_DETAIL_OUT
)

blocked=()
for key in "${prompt_specific_env[@]}"; do
  if [[ -n "${!key:-}" ]]; then
    blocked+=("$key=${!key}")
  fi
done
if [[ "${#blocked[@]}" -gt 0 ]]; then
  echo "Refusing prompt-specific environment variables:" >&2
  printf '  %s\n' "${blocked[@]}" >&2
  echo "Unset them before running the generalized demo." >&2
  exit 2
fi

safe_label="$(printf '%s' "$RUN_LABEL" | tr -cs 'A-Za-z0-9._-' '-' | sed -e 's/^-*//' -e 's/-*$//')"
[[ -n "$safe_label" ]] || safe_label="demo"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_ROOT}/${stamp}-${safe_label}"
mkdir -p "$RUN_DIR"
printf '%s\n' "$PROMPT" > "$RUN_DIR/prompt.txt"

cat > "$RUN_DIR/config.json" <<EOF_CFG
{
  "demo": "vendor-ds4-general-sota",
  "purpose": "Prompt-general DeepSeek V4 vendor demo for arbitrary user prompts under strict 16GB host RAM.",
  "current_generalized_status": "baseline_not_product_target",
  "known_no_prompt_specific_baseline": {
    "artifact": $(printf '%s' "$BASELINE_ARTIFACT" | json_quote),
    "min_eval_tok_s": 1.8,
    "mean_eval_tok_s": 2.18,
    "max_eval_tok_s": 2.7,
    "target_gt_5_tok_s_met": false
  },
  "not_france_specialized_path": true,
  "repo_dir": $(printf '%s' "$REPO_DIR" | json_quote),
  "source_head": "$(git -C "$REPO_DIR" rev-parse HEAD)",
  "source_branch": "$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)",
  "binary": $(printf '%s' "$BINARY" | json_quote),
  "model": $(printf '%s' "$MODEL" | json_quote),
  "max_tokens": ${MAX_TOKENS},
  "memory_max_bytes": ${MEMORY_MAX_BYTES},
  "memory_swap_max_bytes": 0,
  "cold_drop_caches": ${COLD},
  "prompt_general": true,
  "runtime_env_summary": {
    "GGML_MOE_STREAM": "1",
    "GGML_MOE_STREAM_DONTNEED": "1",
    "GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4": "1",
    "GGML_MOE_STREAM_ONE_NAME_FILTER": "ffn_gate_exps",
    "GGML_MOE_STREAM_ONE_CACHE_MIB": "13568",
    "GGML_MOE_VRAM_CACHE_GB": "0",
    "GGML_MOE_KEEP_TOPK_UPDOWN": "4",
    "GGML_MOE_KEEP_TOPK_LAYER_RANGE": "10-39",
    "GGML_MOE_KEEP_TOPK_LAYER_VALUE": "3"
  }
}
EOF_CFG

cat > "$RUN_DIR/runner.sh" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="__RUN_DIR__"
BINARY="__BINARY__"
MODEL="__MODEL__"
MAX_TOKENS="__MAX_TOKENS__"
MEMORY_MAX_BYTES="__MEMORY_MAX_BYTES__"
PRINT_COMMAND="__PRINT_COMMAND__"

cd "$RUN_DIR"
PROMPT="$(cat prompt.txt)"

for key in \
  GGML_MOE_STREAM_ONE_EXPERT_PACK \
  GGML_MOE_EXPERT_PACK \
  GGML_MOE_EXPERT_PACK_OVERLAY \
  GGML_MOE_STREAM_CACHE_ADMIT_PROFILE \
  GGML_MOE_STREAM_ONE_PREFILL_PROFILE \
  GGML_MOE_EXPERT_GGUF_ALIAS_TSV \
  GGML_MOE_IO_ALIGNED_ALIAS_BATCH \
  GGML_MOE_STREAM_UP_DOWN_PROFILE \
  GGML_MOE_STREAM_ONE_ROUTE_PROFILE \
  GGML_DS4_GROUPED_RETAINED_ROUTE_PROFILE_OUT \
  GGML_DS4_GROUPED_RETAINED_ROUTE_DETAIL_OUT; do
  unset "$key" || true
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39
export GGML_MOE_KEEP_TOPK_LAYER_VALUE=3
export GGML_MOE_KEEP_TOPK_UPDOWN=4
export GGML_MOE_STREAM=1
export GGML_MOE_STREAM_DONTNEED=1
export GGML_MOE_STREAM_ONE_CACHE_MIB=13568
export GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1
export GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps
export GGML_MOE_VRAM_CACHE_GB=0

cmd=(
  "$BINARY"
  -m "$MODEL"
  -p "$PROMPT"
  -n "$MAX_TOKENS"
  -c 256
  -b 16
  -ub 16
  -t 20
  -tb 20
  -ngl all
  --fit on
  -fa auto
  --temp 0
  --top-p 1
  --top-k 1
  --seed 1
  -st
  --simple-io
  --no-display-prompt
  --n-cpu-moe 40
  --defer-experts
)

printf '%q ' "${cmd[@]}" > exact_command.txt
printf '\n' >> exact_command.txt
env | sort > environment.txt
date -Is > start_time.txt
start_ns=$(date +%s%N)
printf '%s\n' "$start_ns" > start_ns.txt

if [[ "$PRINT_COMMAND" == "1" ]]; then
  echo "llama-cli command:"
  cat exact_command.txt
fi

cg_rel=$(awk -F: '$2 == "" { print $3 }' /proc/self/cgroup | tail -n 1)
cg_dir="/sys/fs/cgroup${cg_rel}"
printf '%s\n' "$cg_dir" > cgroup_path.txt

set +e
/usr/bin/time -v "${cmd[@]}" < /dev/null > stdout.txt 2> stderr.txt &
cmd_pid=$!

(
  printf 'time_epoch\tmemory_current\tmemory_peak\tgpu_mem_used_mib\tgpu_mem_free_mib\tgpu_util_pct\n'
  while kill -0 "$cmd_pid" 2>/dev/null; do
    now=$(date +%s)
    mem_cur=$(cat "$cg_dir/memory.current" 2>/dev/null || true)
    mem_peak=$(cat "$cg_dir/memory.peak" 2>/dev/null || true)
    gpu=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
    if [[ -n "$gpu" ]]; then
      printf '%s\t%s\t%s\t%s\n' "$now" "$mem_cur" "$mem_peak" "$(printf '%s' "$gpu" | tr ',' '\t')"
    else
      printf '%s\t%s\t%s\t\t\t\n' "$now" "$mem_cur" "$mem_peak"
    fi
    if [[ "$mem_cur" =~ ^[0-9]+$ ]] && [[ "$mem_cur" -gt "$MEMORY_MAX_BYTES" ]]; then
      printf 'memory.current=%s exceeded %s\n' "$mem_cur" "$MEMORY_MAX_BYTES" > ram_limit_exceeded.txt
      kill "$cmd_pid" 2>/dev/null || true
      exit 0
    fi
    stdout_bytes=$(stat -c%s stdout.txt 2>/dev/null || echo 0)
    if [[ "$stdout_bytes" =~ ^[0-9]+$ ]] && [[ "$stdout_bytes" -gt 16777216 ]]; then
      printf 'stdout_bytes=%s exceeded 16777216; killing probable runaway output\n' "$stdout_bytes" > stdout_limit_exceeded.txt
      kill "$cmd_pid" 2>/dev/null || true
      exit 0
    fi
    sleep 1
  done
) > resource_samples.tsv &
monitor_pid=$!

(
  while kill -0 "$cmd_pid" 2>/dev/null; do
    if [[ -s stdout.txt ]]; then
      first_ns=$(date +%s%N)
      python3 - "$start_ns" "$first_ns" > first_output_ms.txt <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000.0)
PY
      exit 0
    fi
    sleep 0.05
  done
) &
watcher_pid=$!

wait "$cmd_pid"
status=$?
set -e
kill "$watcher_pid" 2>/dev/null || true
wait "$watcher_pid" 2>/dev/null || true
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

date -Is > end_time.txt
printf '%s\n' "$status" > exit_status.txt
cp "$cg_dir/memory.current" memory.current 2>/dev/null || true
cp "$cg_dir/memory.peak" memory.peak 2>/dev/null || true
cp "$cg_dir/memory.events" memory.events 2>/dev/null || true
cp "$cg_dir/memory.stat" memory.stat 2>/dev/null || true
cat stdout.txt stderr.txt > combined.txt
exit "$status"
RUNNER

python3 - "$RUN_DIR/runner.sh" "$RUN_DIR" "$BINARY" "$MODEL" "$MAX_TOKENS" "$MEMORY_MAX_BYTES" "$PRINT_COMMAND" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
replacements = {
    "__RUN_DIR__": sys.argv[2],
    "__BINARY__": sys.argv[3],
    "__MODEL__": sys.argv[4],
    "__MAX_TOKENS__": sys.argv[5],
    "__MEMORY_MAX_BYTES__": sys.argv[6],
    "__PRINT_COMMAND__": sys.argv[7],
}
for key, value in replacements.items():
    text = text.replace(key, value)
path.write_text(text)
PY
chmod +x "$RUN_DIR/runner.sh"

if [[ "$COLD" -eq 1 ]]; then
  printf 'sync && echo 3 > /proc/sys/vm/drop_caches\n' > "$RUN_DIR/cold_start_procedure.txt"
  sync
  echo 3 > /proc/sys/vm/drop_caches
else
  printf 'warm run: drop_caches skipped by --warm\n' > "$RUN_DIR/cold_start_procedure.txt"
fi

unit="vendor-ds4-general-demo-${stamp}-${safe_label}"
unit="${unit:0:63}"
systemd_cmd=(
  systemd-run
  "--unit=${unit}.service"
  --collect
  --wait
  --property=MemoryMax="${MEMORY_MAX_BYTES}"
  --property=MemorySwapMax=0
  "$RUN_DIR/runner.sh"
)
printf '%q ' "${systemd_cmd[@]}" > "$RUN_DIR/systemd_command.txt"
printf '\n' >> "$RUN_DIR/systemd_command.txt"

cat <<EOF_RUN
=== Vendor DeepSeek V4 generalized SOTA demo ===
Run dir: $RUN_DIR
Prompt: $PROMPT
Mode: $([[ "$COLD" -eq 1 ]] && echo cold/drop_caches || echo warm/no-drop_caches)
Host RAM cgroup: MemoryMax=${MEMORY_MAX_BYTES}, MemorySwapMax=0
Current generalized baseline: 1.8-2.7 tok/s on dev prompt set; product target >5 tok/s is not yet met.
France-specialized 4.4 tok/s path is intentionally not used here.
EOF_RUN

set +e
"${systemd_cmd[@]}" > "$RUN_DIR/systemd-run.out" 2> "$RUN_DIR/systemd-run.err"
systemd_status=$?
set -e
printf '%s\n' "$systemd_status" > "$RUN_DIR/systemd_run_status.txt"
systemctl show "${unit}.service" > "$RUN_DIR/unit.properties" 2>/dev/null || true
journalctl -u "${unit}.service" --no-pager > "$RUN_DIR/journal.log" 2>/dev/null || true

python3 - "$RUN_DIR" "$MEMORY_MAX_BYTES" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
memory_max = int(sys.argv[2])

def read_text(name: str) -> str:
    p = run_dir / name
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def read_int(name: str):
    try:
        return int(read_text(name).strip())
    except Exception:
        return None

def parse_elapsed(stderr: str):
    m = re.search(r"Elapsed \(wall clock\) time[^\n]*\):\s*([0-9:]+(?:\.[0-9]+)?)", stderr)
    if not m:
        return None
    parts = m.group(1).split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return None

def parse_memory_stat(text: str):
    out = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) == 2:
            try:
                out[cols[0]] = int(cols[1])
            except ValueError:
                pass
    return out

def parse_memory_events(text: str):
    out = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) == 2:
            try:
                out[cols[0]] = int(cols[1])
            except ValueError:
                pass
    return out

stdout = read_text("stdout.txt").replace("\b", "").replace("\r", "\n")
stderr = read_text("stderr.txt")
prompt = read_text("prompt.txt").rstrip("\n")

answer = stdout
if prompt and prompt in answer:
    answer = answer.rsplit(prompt, 1)[-1]
answer = re.split(r"\n\[\s*Prompt:\s*[0-9.]+\s*t/s", answer, maxsplit=1)[0]
answer = re.sub(r"\[[^\n]*Prompt:\s*[0-9.]+\s*t/s[^\n]*\]", "", answer)
answer = re.sub(r"(?s)^.*?available commands:.*?\n\n", "", answer)
answer = re.sub(r"^[>\s|/\\\-\u2580-\u259f]+", "", answer).strip()

rate = re.search(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]", stdout)
prompt_tok_s = float(rate.group(1)) if rate else None
eval_tok_s = float(rate.group(2)) if rate else None

first_output_ms = None
try:
    first_output_ms = float(read_text("first_output_ms.txt").strip())
except Exception:
    pass

max_rss_kb = None
m = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", stderr)
if m:
    max_rss_kb = int(m.group(1))

mem_stat = parse_memory_stat(read_text("memory.stat"))
mem_events = parse_memory_events(read_text("memory.events"))
memory_peak = read_int("memory.peak")
answer_present = bool(re.search(r"[A-Za-z0-9]{2,}|[\u3400-\u9fff]", answer))

summary = {
    "run_dir": str(run_dir),
    "prompt": prompt,
    "answer": answer,
    "answer_present": answer_present,
    "prompt_general": True,
    "not_france_specialized": True,
    "eval_tok_s": eval_tok_s,
    "prompt_tok_s": prompt_tok_s,
    "first_output_ms": first_output_ms,
    "elapsed_seconds": parse_elapsed(stderr),
    "exit_status": read_int("exit_status.txt"),
    "systemd_status": read_int("systemd_run_status.txt"),
    "memory_max_bytes": memory_max,
    "memory_peak_bytes": memory_peak,
    "memory_current_bytes": read_int("memory.current"),
    "memory_file_bytes": mem_stat.get("file"),
    "memory_anon_bytes": mem_stat.get("anon"),
    "memory_events": mem_events,
    "memory_peak_ok": memory_peak is not None and memory_peak <= memory_max,
    "ram_ok": memory_peak is not None and memory_peak <= memory_max and mem_events.get("oom_kill", 0) == 0,
    "ram_limit_exceeded_file": (run_dir / "ram_limit_exceeded.txt").exists(),
    "stdout_limit_exceeded_file": (run_dir / "stdout_limit_exceeded.txt").exists(),
    "max_rss_kb": max_rss_kb,
    "known_general_baseline_tok_s": {"min": 1.8, "mean": 2.18, "max": 2.7},
    "target_gt_5_tok_s_met_by_this_run": eval_tok_s is not None and eval_tok_s > 5.0,
    "quality_note": "Demo only checks that output exists. Manual semantic review is required before accepting a benchmark/SOTA result.",
    "exact_command_file": str(run_dir / "exact_command.txt"),
    "environment_file": str(run_dir / "environment.txt"),
    "resource_samples_file": str(run_dir / "resource_samples.tsv"),
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print("\n=== Demo summary ===")
for key in [
    "eval_tok_s",
    "prompt_tok_s",
    "first_output_ms",
    "elapsed_seconds",
    "memory_peak_bytes",
    "memory_file_bytes",
    "ram_ok",
    "answer_present",
    "target_gt_5_tok_s_met_by_this_run",
]:
    print(f"{key}: {summary.get(key)}")
print(f"run_dir: {run_dir}")
print("\n=== Model answer ===")
print(answer)
PY

exit "$systemd_status"
