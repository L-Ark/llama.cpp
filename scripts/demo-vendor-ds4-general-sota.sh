#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/demo-vendor-ds4-general-sota.sh [--prompt TEXT | --prompt-file FILE] [options]

Options:
  --prompt TEXT                  Run one arbitrary user prompt.
  --prompt-file FILE             Read the prompt from a file. Use - for stdin.
  -n, --max-tokens N             Maximum generated tokens. Default: 192.
  --run-label NAME               Label added to the run directory.
  --warm                         Skip drop_caches. Default is cold-start.
  --allow-prompt-specific-env    Diagnostic only; do not use for SOTA demos.
  -h, --help                     Show this help.

This demos the current accepted prompt-general DeepSeek V4 vendor configuration
under the strict 16GB host RAM cgroup. It accepts arbitrary prompts and records
all files needed to reproduce the run.

Default behavior is intentionally prompt-general:
  - prompt-specific expert packs, aliases, and profiles are refused
  - MemoryMax=16000000000 and MemorySwapMax=0 via systemd-run
  - page cache is counted inside the cgroup
  - cold drop_caches before launch unless --warm is passed
  - vendor framework, cpu_moe=40, DS4 gate-only one-stream cache, vram_cache=0

Outputs:
  /root/lfz/runs/vendor-ds4-16gb/demo-general-sota/<timestamp>-<label>/
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BINARY="${BINARY:-${REPO_DIR}/build-ds4-moe-stream/bin/llama-cli}"
MODEL="${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
RUN_ROOT="${RUN_ROOT:-/root/lfz/runs/vendor-ds4-16gb/demo-general-sota}"

MAX_TOKENS=192
RUN_LABEL="interactive"
COLD=1
ALLOW_PROMPT_SPECIFIC_ENV=0
PROMPT=""
PROMPT_FILE=""

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
    --max-tokens|-n)
      [[ $# -ge 2 ]] || die "missing value for --max-tokens"
      MAX_TOKENS="$2"
      shift 2
      ;;
    --run-label)
      [[ $# -ge 2 ]] || die "missing value for --run-label"
      RUN_LABEL="$2"
      shift 2
      ;;
    --warm)
      COLD=0
      shift
      ;;
    --allow-prompt-specific-env)
      ALLOW_PROMPT_SPECIFIC_ENV=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -n "$PROMPT" && -n "$PROMPT_FILE" ]]; then
  die "use only one of --prompt or --prompt-file"
fi

if [[ ! "$MAX_TOKENS" =~ ^[0-9]+$ ]] || [[ "$MAX_TOKENS" -le 0 ]]; then
  die "--max-tokens must be a positive integer"
fi

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ "$PROMPT_FILE" == "-" ]]; then
    PROMPT="$(cat)"
  else
    [[ -f "$PROMPT_FILE" ]] || die "missing prompt file: $PROMPT_FILE"
    PROMPT="$(cat "$PROMPT_FILE")"
  fi
elif [[ -z "$PROMPT" ]]; then
  if [[ -t 0 ]]; then
    printf 'Enter prompt: ' >&2
    IFS= read -r PROMPT
  else
    PROMPT="$(cat)"
  fi
fi

PROMPT="$(printf '%s' "$PROMPT" | sed -e 's/[[:space:]]*$//')"
[[ -n "$PROMPT" ]] || die "prompt is empty"

[[ -x "$BINARY" ]] || { echo "missing executable: $BINARY" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 1; }

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

if [[ "$ALLOW_PROMPT_SPECIFIC_ENV" -eq 0 ]]; then
  found=()
  for key in "${prompt_specific_env[@]}"; do
    if [[ -n "${!key:-}" ]]; then
      found+=("$key=${!key}")
    fi
  done
  if [[ "${#found[@]}" -gt 0 ]]; then
    printf 'Refusing to run with prompt-specific or profiling env set:\n' >&2
    printf '  %s\n' "${found[@]}" >&2
    printf 'Unset these variables, or pass --allow-prompt-specific-env for diagnostics only.\n' >&2
    exit 2
  fi
fi

safe_label="$(printf '%s' "$RUN_LABEL" | tr -cs 'A-Za-z0-9._-' '-' | sed -e 's/^-*//' -e 's/-*$//')"
[[ -n "$safe_label" ]] || safe_label="interactive"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_ROOT}/${stamp}-${safe_label}"
mkdir -p "$RUN_DIR"

printf '%s\n' "$PROMPT" > "${RUN_DIR}/prompt.txt"
printf '/exit\n' > "${RUN_DIR}/stdin.txt"
cat > "${RUN_DIR}/config.json" <<EOF
{
  "demo": "vendor-ds4-general-sota",
  "description": "Current accepted prompt-general DeepSeek V4 vendor demo configuration. It deliberately excludes prompt-specific expert packs, aliases, and profiles.",
  "repo_dir": "${REPO_DIR}",
  "source_head": "$(git -C "$REPO_DIR" rev-parse HEAD)",
  "source_branch": "$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)",
  "binary": "${BINARY}",
  "model": "${MODEL}",
  "max_tokens": ${MAX_TOKENS},
  "memory_max_bytes": 16000000000,
  "memory_swap_max_bytes": 0,
  "cold_drop_caches": ${COLD},
  "prompt_general": true,
  "prompt_specific_env_allowed": ${ALLOW_PROMPT_SPECIFIC_ENV}
}
EOF

cat > "${RUN_DIR}/answer_started.py" <<'PY'
import re
import sys
from pathlib import Path

stdout = Path(sys.argv[1])
prompt = Path(sys.argv[2]).read_text(encoding="utf-8", errors="ignore").strip()
if not stdout.exists():
    raise SystemExit(1)

text = stdout.read_text(encoding="utf-8", errors="ignore")
text = text.replace("\b", "").replace("\r", "\n")
if prompt not in text:
    raise SystemExit(1)
text = text.rsplit(prompt, 1)[-1]
text = re.sub(r"\[[^\n]*Prompt:\s*[0-9.]+\s*t/s[^\n]*\]", "", text)
text = re.sub(r"^[>\s|/\\\-\u2580-\u259f]+", "", text).strip()

has_answer = re.search(r"[A-Za-z0-9]{2,}|[\u3400-\u9fff]{1,}", text) is not None
raise SystemExit(0 if has_answer else 1)
PY

cat > "${RUN_DIR}/summarize.py" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8", errors="ignore").rstrip("\n")

def read_text(name):
    path = run_dir / name
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

def read_int(name):
    text = read_text(name).strip()
    try:
        return int(text)
    except Exception:
        return None

stdout = read_text("stdout.txt").replace("\b", "").replace("\r", "\n")
stderr = read_text("stderr.txt")

answer = stdout.rsplit(prompt, 1)[-1] if prompt in stdout else stdout
answer = re.split(r"\n\[\s*Prompt:\s*[0-9.]+\s*t/s", answer, maxsplit=1)[0]
answer = re.sub(r"\[[^\n]*Prompt:\s*[0-9.]+\s*t/s[^\n]*\]", "", answer)
answer = re.sub(r"^[>\s|/\\\-\u2580-\u259f]+", "", answer).strip()

rate_re = re.search(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]", stdout)
prompt_tok_s = float(rate_re.group(1)) if rate_re else None
eval_tok_s = float(rate_re.group(2)) if rate_re else None

elapsed = None
elapsed_re = re.search(r"Elapsed \(wall clock\) time .*:\s*([0-9:]+(?:\.[0-9]+)?)", stderr)
if elapsed_re:
    parts = elapsed_re.group(1).split(":")
    try:
        if len(parts) == 3:
            elapsed = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            elapsed = int(parts[0]) * 60 + float(parts[1])
        else:
            elapsed = float(parts[0])
    except Exception:
        elapsed = None

max_rss_kb = None
rss_re = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", stderr)
if rss_re:
    max_rss_kb = int(rss_re.group(1))

memory_stat = {}
for line in read_text("memory.stat").splitlines():
    cols = line.split()
    if len(cols) == 2:
        try:
            memory_stat[cols[0]] = int(cols[1])
        except ValueError:
            pass

memory_events = {}
for line in read_text("memory.events").splitlines():
    cols = line.split()
    if len(cols) == 2:
        try:
            memory_events[cols[0]] = int(cols[1])
        except ValueError:
            pass

first_answer_ms = None
try:
    first_answer_ms = float(read_text("first_answer_ms.txt").strip())
except Exception:
    pass

answer_present = re.search(r"[A-Za-z0-9]{2,}|[\u3400-\u9fff]{1,}", answer) is not None
memory_peak = read_int("memory.peak")
summary = {
    "run_dir": str(run_dir),
    "prompt": prompt,
    "answer": answer,
    "answer_present": answer_present,
    "quality_note": "For arbitrary user prompts this script checks that text was produced; semantic correctness must be reviewed from the captured answer.",
    "eval_tok_s": eval_tok_s,
    "prompt_tok_s": prompt_tok_s,
    "ttft_estimate_ms": first_answer_ms,
    "elapsed_seconds": elapsed,
    "exit_status": read_int("exit_status.txt"),
    "systemd_status": read_int("systemd_run_status.txt"),
    "memory_peak_bytes": memory_peak,
    "memory_current_bytes": read_int("memory.current"),
    "memory_file_bytes": memory_stat.get("file"),
    "memory_anon_bytes": memory_stat.get("anon"),
    "memory_events": memory_events,
    "memory_peak_ok": memory_peak is not None and memory_peak <= 16000000000,
    "ram_ok": memory_peak is not None and memory_peak <= 16000000000 and memory_events.get("oom_kill", 0) == 0,
    "max_rss_kb": max_rss_kb,
    "cold_drop_caches": read_text("cold_start_procedure.txt").strip(),
    "exact_command": read_text("exact_command.txt").strip(),
    "environment_file": str(run_dir / "environment.txt"),
    "resource_samples": str(run_dir / "resource_samples.tsv"),
}

(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print("\n=== Demo summary ===")
for key in ("eval_tok_s", "prompt_tok_s", "ttft_estimate_ms", "elapsed_seconds", "memory_peak_bytes", "memory_file_bytes", "ram_ok", "answer_present"):
    print(f"{key}: {summary.get(key)}")
print(f"run_dir: {run_dir}")
print("\n=== Model answer ===")
print(answer)
PY

cat > "${RUN_DIR}/run_case.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="__RUN_DIR__"
BINARY="__BINARY__"
MODEL="__MODEL__"
MAX_TOKENS="__MAX_TOKENS__"
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
  --no-display-prompt
  --n-cpu-moe 40
  --defer-experts
)
printf '%q ' "${cmd[@]}" > exact_command.txt
printf '\n' >> exact_command.txt
env | sort > environment.txt
date -Is > start_time.txt
start_ns=$(date +%s%N)
echo "$start_ns" > start_ns.txt
CG_REL=$(awk -F: '$2 == "" { print $3 }' /proc/self/cgroup | tail -n 1)
CG_DIR="/sys/fs/cgroup${CG_REL}"
printf '%s\n' "$CG_DIR" > cgroup_path.txt
set +e
/usr/bin/time -v "${cmd[@]}" < stdin.txt > stdout.txt 2> stderr.txt &
CMD_PID=$!
(
  printf 'time_epoch\tmemory_current\tmemory_peak\tgpu_mem_used_mib\tgpu_mem_free_mib\tgpu_util_pct\n'
  while kill -0 "$CMD_PID" 2>/dev/null; do
    now=$(date +%s)
    mem_cur=$(cat "$CG_DIR/memory.current" 2>/dev/null || true)
    mem_peak=$(cat "$CG_DIR/memory.peak" 2>/dev/null || true)
    gpu=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
    if [[ -n "$gpu" ]]; then
      printf '%s\t%s\t%s\t%s\n' "$now" "$mem_cur" "$mem_peak" "$(printf '%s' "$gpu" | tr ',' '\t')"
    else
      printf '%s\t%s\t%s\t\t\t\n' "$now" "$mem_cur" "$mem_peak"
    fi
    if [[ -n "$mem_cur" ]] && [[ "$mem_cur" =~ ^[0-9]+$ ]] && [[ "$mem_cur" -gt 16000000000 ]]; then
      printf 'ram_limit_exceeded current=%s threshold=16000000000\n' "$mem_cur" > ram_limit_exceeded.txt
      kill "$CMD_PID" 2>/dev/null || true
      exit 0
    fi
    sleep 1
  done
) > resource_samples.tsv &
MONITOR_PID=$!
(
  while kill -0 "$CMD_PID" 2>/dev/null; do
    if python3 answer_started.py stdout.txt prompt.txt; then
      first_ns=$(date +%s%N)
      python3 - "$start_ns" "$first_ns" > first_answer_ms.txt <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1000000.0)
PY
      exit 0
    fi
    sleep 0.05
  done
) &
WATCHER_PID=$!
wait "$CMD_PID"
STATUS=$?
set -e
kill "$WATCHER_PID" 2>/dev/null || true
wait "$WATCHER_PID" 2>/dev/null || true
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
date -Is > end_time.txt
echo "$STATUS" > exit_status.txt
cp "$CG_DIR/memory.current" memory.current 2>/dev/null || true
cp "$CG_DIR/memory.peak" memory.peak 2>/dev/null || true
cp "$CG_DIR/memory.events" memory.events 2>/dev/null || true
cp "$CG_DIR/memory.stat" memory.stat 2>/dev/null || true
cat stdout.txt stderr.txt > combined.txt
exit "$STATUS"
SH

python3 - "$RUN_DIR/run_case.sh" "$RUN_DIR" "$BINARY" "$MODEL" "$MAX_TOKENS" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for key, value in {
    "__RUN_DIR__": sys.argv[2],
    "__BINARY__": sys.argv[3],
    "__MODEL__": sys.argv[4],
    "__MAX_TOKENS__": sys.argv[5],
}.items():
    text = text.replace(key, value)
path.write_text(text, encoding="utf-8")
PY
chmod +x "${RUN_DIR}/run_case.sh"

if [[ "$COLD" -eq 1 ]]; then
  echo "sync && echo 3 > /proc/sys/vm/drop_caches" > "${RUN_DIR}/cold_start_procedure.txt"
  sync
  echo 3 > /proc/sys/vm/drop_caches
else
  echo "warm run: drop_caches skipped by --warm" > "${RUN_DIR}/cold_start_procedure.txt"
fi

unit="vendor-ds4-demo-${stamp}-${safe_label}"
unit="${unit:0:63}"
systemd_cmd=(
  systemd-run
  "--unit=${unit}.service"
  --collect
  --wait
  --property=MemoryMax=16000000000
  --property=MemorySwapMax=0
  "${RUN_DIR}/run_case.sh"
)
printf '%q ' "${systemd_cmd[@]}" > "${RUN_DIR}/systemd_command.txt"
printf '\n' >> "${RUN_DIR}/systemd_command.txt"

set +e
"${systemd_cmd[@]}" > "${RUN_DIR}/systemd-run.out" 2> "${RUN_DIR}/systemd-run.err"
systemd_status=$?
set -e
echo "$systemd_status" > "${RUN_DIR}/systemd_run_status.txt"
systemctl show "${unit}.service" > "${RUN_DIR}/unit.properties" 2>/dev/null || true
journalctl -u "${unit}.service" --no-pager > "${RUN_DIR}/journal.log" 2>/dev/null || true

python3 "${RUN_DIR}/summarize.py" "$RUN_DIR"

exit "$systemd_status"
