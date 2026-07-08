#!/usr/bin/env bash
set -euo pipefail

# Demo the current prompt-general vendor DeepSeek V4 configuration.
# This script is intentionally not allowed to use France/prompt-specific packs,
# route profiles, admit profiles, or GGUF alias overlays. It is for arbitrary
# user prompts under the product constraint: 16GB host RAM including page cache.

usage() {
  cat <<'USAGE'
Usage:
  scripts/demo-vendor-ds4-general-sota.sh [--prompt TEXT | --prompt-file FILE | TEXT...] [options]

Prompt input:
  --prompt TEXT          Run exactly TEXT as the prompt.
  --prompt-file FILE     Read prompt from FILE. Use "-" for stdin.
  TEXT...                Positional prompt text.
  --multiline            Read a multi-line prompt from stdin until Ctrl-D.

Options:
  -n, --max-tokens N     Maximum generated tokens. Default: 96.
  --run-label LABEL      Label suffix for the artifact directory. Default: interactive.
  --warm                 Skip drop_caches. Default is cold start with drop_caches.
  --print-command        Print the exact llama-cli command used by the cgroup run.
  -h, --help             Show this help.

What this script demonstrates:
  - Current no-prompt-specific vendor DeepSeek V4 generalized path.
  - Strict host RAM cap: MemoryMax=16,000,000,000 bytes, MemorySwapMax=0.
  - Page cache is counted through cgroup memory.stat file bytes.
  - User may enter any prompt; this is not a France-specialized demo.

Current known generalized status:
  - Calibration/dev prompt range recorded in the repo: 2.2-3.2 tok/s, mean 2.78 tok/s.
  - Held-out v1 prompt range recorded in the repo: 2.4-3.0 tok/s, mean 2.76 tok/s.
  - Product target remains stable >5 tok/s for random prompts; not yet met.

Artifacts:
  /root/lfz/runs/vendor-ds4-16gb/demo-general-sota/<timestamp>-<label>/
USAGE
}

fail() {
  echo "error: $*" >&2
  exit 2
}

json_string() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read(), ensure_ascii=False))'
}

trim_trailing_space() {
  sed -e 's/[[:space:]]*$//'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${BINARY:-${REPO_DIR}/build-ds4-moe-stream/bin/llama-cli}"
MODEL="${MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
RUN_ROOT="${RUN_ROOT:-/root/lfz/runs/vendor-ds4-16gb/demo-general-sota}"
BASELINE_ARTIFACT="${BASELINE_ARTIFACT:-${REPO_DIR}/.Agent/runs/20260705-vendor-ds4-coldstart/general-prompt-baseline-no-prompt-specific-20260706.json}"
SOTA_ARTIFACT="${SOTA_ARTIFACT:-${REPO_DIR}/.Agent/runs/20260705-vendor-ds4-coldstart/updown-paired-read-generalized-sota-20260708.json}"
DS4_ALIAS_TSV="${DS4_ALIAS_TSV:-${REPO_DIR}/.Agent/profiles/vendor-ds4/ds4-native-full-gguf-alias-source-20260707.tsv}"

MEMORY_MAX_BYTES=16000000000
MAX_TOKENS=96
RUN_LABEL="interactive"
COLD=1
MULTILINE=0
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
    --multiline)
      MULTILINE=1
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
elif [[ "$MULTILINE" -eq 1 ]]; then
  echo "Enter any prompt, then press Ctrl-D:" >&2
  PROMPT="$(cat)"
elif [[ ! -t 0 ]]; then
  PROMPT="$(cat)"
else
  IFS= read -r -p "Prompt> " PROMPT || true
fi

PROMPT="$(printf '%s' "$PROMPT" | trim_trailing_space)"
[[ -n "$PROMPT" ]] || fail "prompt is empty"
[[ -x "$BINARY" ]] || fail "missing executable: $BINARY"
[[ -f "$MODEL" ]] || fail "missing model: $MODEL"
[[ -f "$DS4_ALIAS_TSV" ]] || fail "missing static DS4 alias TSV: $DS4_ALIAS_TSV"

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
  echo "Unset them before running this generalized demo." >&2
  exit 2
fi

safe_label="$(printf '%s' "$RUN_LABEL" | tr -cs 'A-Za-z0-9._-' '-' | sed -e 's/^-*//' -e 's/-*$//')"
[[ -n "$safe_label" ]] || safe_label="interactive"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_ROOT}/${stamp}-${safe_label}"
mkdir -p "$RUN_DIR"
printf '%s\n' "$PROMPT" > "$RUN_DIR/prompt.txt"

source_status="$(git -C "$REPO_DIR" status --short)"
printf '%s\n' "$source_status" > "$RUN_DIR/source_status.txt"
source_dirty=false
if [[ -n "$source_status" ]]; then
  source_dirty=true
fi

cat > "$RUN_DIR/config.json" <<EOF_CFG
{
  "demo": "vendor-ds4-generalized-sota-current",
  "purpose": "Run arbitrary prompts on the current no-prompt-specific vendor DeepSeek V4 path under strict 16GB host RAM.",
  "repo_dir": $(printf '%s' "$REPO_DIR" | json_string),
  "source_head": "$(git -C "$REPO_DIR" rev-parse HEAD)",
  "source_branch": "$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)",
  "source_dirty": ${source_dirty},
  "source_status_file": $(printf '%s' "$RUN_DIR/source_status.txt" | json_string),
  "binary": $(printf '%s' "$BINARY" | json_string),
  "model": $(printf '%s' "$MODEL" | json_string),
  "baseline_artifact": $(printf '%s' "$BASELINE_ARTIFACT" | json_string),
  "sota_artifact": $(printf '%s' "$SOTA_ARTIFACT" | json_string),
  "prompt_general": true,
  "prompt_specific_optimization": false,
  "france_specialized_path_used": false,
  "current_generalized_dev_tok_s": {"min": 2.2, "mean": 2.78, "max": 3.2},
  "current_held_out_v1_tok_s": {"min": 2.4, "mean": 2.76, "max": 3.0},
  "product_target_tok_s": 5.0,
  "product_target_currently_met": false,
  "max_tokens": ${MAX_TOKENS},
  "cold_drop_caches": ${COLD},
  "memory_max_bytes": ${MEMORY_MAX_BYTES},
  "memory_swap_max_bytes": 0,
  "runtime": {
    "n_cpu_moe": 40,
    "ngl": "all",
    "context": 256,
    "batch": 16,
    "ubatch": 16,
    "threads": 20,
    "GGML_CUDA_DISABLE_GRAPHS": "1",
    "GGML_MOE_STREAM": "1",
    "GGML_MOE_STREAM_DONTNEED": "1",
    "GGML_MOE_STAGE_PINNED_SLOTS": "8",
    "GGML_MOE_EXPERT_GGUF_ALIAS_TSV": $(printf '%s' "$DS4_ALIAS_TSV" | json_string),
    "GGML_MOE_IO_BACKEND": "iouring",
    "GGML_MOE_IO_ALIGNED_ALIAS_BATCH": "1",
    "GGML_MOE_IO_REFILL_BATCH": "4",
    "GGML_MOE_DOWN_PARALLEL_STAGE": "1",
    "GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4": "1",
    "GGML_MOE_STREAM_ONE_NAME_FILTER": "ffn_gate_exps",
    "GGML_MOE_STREAM_ONE_CACHE_MIB": "4096",
    "GGML_MOE_VRAM_CACHE_GB": "9",
    "GGML_MOE_STREAM_DOWN_BATCH": "1",
    "GGML_MOE_STREAM_DOWN_Q80_COMPAT_BATCH": "1",
    "GGML_MOE_STREAM_UP_Q80_COMPAT_BATCH": "1",
    "GGML_MOE_UPDOWN_PAIRED_READ": "1",
    "GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER": "1",
    "GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER_LANE8": "1",
    "GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER_LANE8_SHARED": "1",
    "GGML_MOE_KEEP_TOPK_UPDOWN": "4",
    "GGML_MOE_KEEP_TOPK_LAYER_RANGE": "10-39",
    "GGML_MOE_KEEP_TOPK_LAYER_VALUE": "3"
  }
}
EOF_CFG

cat > "$RUN_DIR/runner.sh" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=$(printf '%q' "$RUN_DIR")
BINARY=$(printf '%q' "$BINARY")
MODEL=$(printf '%q' "$MODEL")
MAX_TOKENS=$(printf '%q' "$MAX_TOKENS")
MEMORY_MAX_BYTES=$(printf '%q' "$MEMORY_MAX_BYTES")
PRINT_COMMAND=$(printf '%q' "$PRINT_COMMAND")
cd "\$RUN_DIR"
PROMPT="\$(cat prompt.txt)"

for key in ${prompt_specific_env[*]}; do
  unset "\$key" || true
done

export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_MOE_EXPERT_GGUF_ALIAS_TSV=$(printf '%q' "$DS4_ALIAS_TSV")
export GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1
export GGML_MOE_IO_REFILL_BATCH=4
export GGML_MOE_DOWN_PARALLEL_STAGE=1
export GGML_MOE_IO_BACKEND=iouring
export GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39
export GGML_MOE_KEEP_TOPK_LAYER_VALUE=3
export GGML_MOE_KEEP_TOPK_UPDOWN=4
export GGML_MOE_STREAM=1
export GGML_MOE_STAGE_PINNED_SLOTS=8
export GGML_MOE_STREAM_DONTNEED=1
export GGML_MOE_STREAM_ONE_CACHE_MIB=4096
export GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1
export GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps
export GGML_MOE_STREAM_DOWN_BATCH=1
export GGML_MOE_STREAM_DOWN_Q80_COMPAT_BATCH=1
export GGML_MOE_STREAM_UP_Q80_COMPAT_BATCH=1
export GGML_MOE_UPDOWN_PAIRED_READ=1
export GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER=1
export GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER_LANE8=1
export GGML_MOE_STREAM_DOWN_Q80_CPU_ORDER_LANE8_SHARED=1
export GGML_MOE_VRAM_CACHE_GB=9

cmd=(
  "\$BINARY"
  -m "\$MODEL"
  -p "\$PROMPT"
  -n "\$MAX_TOKENS"
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

printf '%q ' "\${cmd[@]}" > exact_command.txt
printf '\n' >> exact_command.txt
env | sort > environment.txt
date -Is > start_time.txt
start_ns="\$(date +%s%N)"
printf '%s\n' "\$start_ns" > start_ns.txt

if [[ "\$PRINT_COMMAND" == "1" ]]; then
  echo "llama-cli command:"
  cat exact_command.txt
fi

cg_rel="\$(awk -F: '\$2 == "" { print \$3 }' /proc/self/cgroup | tail -n 1)"
cg_dir="/sys/fs/cgroup\${cg_rel}"
printf '%s\n' "\$cg_dir" > cgroup_path.txt

set +e
/usr/bin/time -v "\${cmd[@]}" < /dev/null > stdout.txt 2> stderr.txt &
cmd_pid=\$!

(
  printf 'time_epoch\tmemory_current\tmemory_peak\tmemory_file\tgpu_mem_used_mib\tgpu_mem_free_mib\tgpu_util_pct\n'
  while kill -0 "\$cmd_pid" 2>/dev/null; do
    now="\$(date +%s)"
    mem_cur="\$(cat "\$cg_dir/memory.current" 2>/dev/null || true)"
    mem_peak="\$(cat "\$cg_dir/memory.peak" 2>/dev/null || true)"
    mem_file="\$(awk '\$1 == "file" {print \$2}' "\$cg_dir/memory.stat" 2>/dev/null || true)"
    gpu="\$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    if [[ -n "\$gpu" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "\$now" "\$mem_cur" "\$mem_peak" "\$mem_file" "\$(printf '%s' "\$gpu" | tr ',' '\t')"
    else
      printf '%s\t%s\t%s\t%s\t\t\t\n' "\$now" "\$mem_cur" "\$mem_peak" "\$mem_file"
    fi
    if [[ "\$mem_cur" =~ ^[0-9]+$ ]] && [[ "\$mem_cur" -gt "\$MEMORY_MAX_BYTES" ]]; then
      printf 'memory.current=%s exceeded %s\n' "\$mem_cur" "\$MEMORY_MAX_BYTES" > ram_limit_exceeded.txt
      kill "\$cmd_pid" 2>/dev/null || true
      exit 0
    fi
    sleep 1
  done
) > resource_samples.tsv &
monitor_pid=\$!

(
  while kill -0 "\$cmd_pid" 2>/dev/null; do
    if [[ -s stdout.txt ]]; then
      first_ns="\$(date +%s%N)"
      python3 - "\$start_ns" "\$first_ns" > first_output_ms.txt <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000.0)
PY
      exit 0
    fi
    sleep 0.05
  done
) &
watcher_pid=\$!

wait "\$cmd_pid"
status=\$?
set -e
kill "\$watcher_pid" 2>/dev/null || true
wait "\$watcher_pid" 2>/dev/null || true
kill "\$monitor_pid" 2>/dev/null || true
wait "\$monitor_pid" 2>/dev/null || true

date -Is > end_time.txt
printf '%s\n' "\$status" > exit_status.txt
cp "\$cg_dir/memory.current" memory.current 2>/dev/null || true
cp "\$cg_dir/memory.peak" memory.peak 2>/dev/null || true
cp "\$cg_dir/memory.events" memory.events 2>/dev/null || true
cp "\$cg_dir/memory.stat" memory.stat 2>/dev/null || true
cat stdout.txt stderr.txt > combined.txt
exit "\$status"
EOF_RUNNER
chmod +x "$RUN_DIR/runner.sh"

if [[ "$COLD" -eq 1 ]]; then
  printf 'sync && echo 3 > /proc/sys/vm/drop_caches\n' > "$RUN_DIR/cold_start_procedure.txt"
  sync
  echo 3 > /proc/sys/vm/drop_caches
else
  printf 'warm run: drop_caches skipped by --warm\n' > "$RUN_DIR/cold_start_procedure.txt"
fi

unit="vendor-ds4-demo-${stamp}-${safe_label}"
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

cat <<EOF_START
=== Vendor DeepSeek V4 generalized SOTA demo ===
Run dir: $RUN_DIR
Source: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)@$(git -C "$REPO_DIR" rev-parse --short HEAD) $([[ "$source_dirty" == true ]] && echo dirty || echo clean)
Mode: $([[ "$COLD" -eq 1 ]] && echo cold/drop_caches || echo warm/no-drop_caches)
Host RAM cgroup: MemoryMax=${MEMORY_MAX_BYTES}, MemorySwapMax=0
Known generalized dev range: 2.2-3.2 tok/s, mean 2.78 tok/s. Held-out v1: 2.4-3.0 tok/s, mean 2.76 tok/s.
Product target: stable >5 tok/s for random prompts. Current generalized path is not there yet.
Prompt-specific packs/profiles/aliases: disabled and refused.
Prompt:
$PROMPT
EOF_START

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

def read_text(name):
    p = run_dir / name
    return p.read_text(encoding='utf-8', errors='ignore') if p.exists() else ''

def read_int(name):
    try:
        return int(read_text(name).strip())
    except Exception:
        return None

def parse_elapsed(stderr):
    m = re.search(r'Elapsed \(wall clock\) time[^\n]*\):\s*([0-9:]+(?:\.[0-9]+)?)', stderr)
    if not m:
        return None
    parts = m.group(1).split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return None

def parse_kv_ints(text):
    out = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) == 2:
            try:
                out[cols[0]] = int(cols[1])
            except ValueError:
                pass
    return out

stdout = read_text('stdout.txt').replace('\b', '').replace('\r', '\n')
stderr = read_text('stderr.txt')
prompt = read_text('prompt.txt').rstrip('\n')
try:
    config = json.loads(read_text('config.json'))
except Exception:
    config = {}

answer = stdout
if prompt and prompt in answer:
    answer = answer.rsplit(prompt, 1)[-1]
answer = re.split(r'\n\[\s*Prompt:\s*[0-9.]+\s*t/s', answer, maxsplit=1)[0]
answer = re.sub(r'\[[^\n]*Prompt:\s*[0-9.]+\s*t/s[^\n]*\]', '', answer)
answer = re.sub(r'(?s)^.*?available commands:.*?\n\n', '', answer)
answer = re.sub(r'^[>\s|/\\\-\u2580-\u259f]+', '', answer).strip()

rate = re.search(r'\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]', stdout)
prompt_tok_s = float(rate.group(1)) if rate else None
eval_tok_s = float(rate.group(2)) if rate else None
try:
    first_output_ms = float(read_text('first_output_ms.txt').strip())
except Exception:
    first_output_ms = None

mem_stat = parse_kv_ints(read_text('memory.stat'))
mem_events = parse_kv_ints(read_text('memory.events'))
memory_peak = read_int('memory.peak')
answer_present = bool(re.search(r'[A-Za-z0-9]{2,}|[\u3400-\u9fff]', answer))
ram_ok = memory_peak is not None and memory_peak <= memory_max and mem_events.get('oom_kill', 0) == 0 and not (run_dir / 'ram_limit_exceeded.txt').exists()
exit_status = read_int('exit_status.txt')
systemd_status = read_int('systemd_run_status.txt')
run_ok = exit_status == 0 and systemd_status == 0 and ram_ok and answer_present

summary = {
    'run_dir': str(run_dir),
    'prompt': prompt,
    'answer': answer,
    'answer_present': answer_present,
    'run_ok': run_ok,
    'prompt_general': True,
    'prompt_specific_optimization': False,
    'france_specialized_path_used': False,
    'eval_tok_s': eval_tok_s,
    'prompt_tok_s': prompt_tok_s,
    'first_output_ms': first_output_ms,
    'elapsed_seconds': parse_elapsed(stderr),
    'exit_status': exit_status,
    'systemd_status': systemd_status,
    'memory_max_bytes': memory_max,
    'memory_peak_bytes': memory_peak,
    'memory_current_bytes': read_int('memory.current'),
    'memory_file_bytes': mem_stat.get('file'),
    'memory_anon_bytes': mem_stat.get('anon'),
    'memory_events': mem_events,
    'ram_ok': ram_ok,
    'known_generalized_dev_range_tok_s': {'min': 2.1, 'mean': 2.54, 'max': 3.0},
    'known_held_out_v1_range_tok_s': {'min': 2.3, 'mean': 2.56, 'max': 2.8},
    'product_target_gt_5_tok_s_met_by_this_run': eval_tok_s is not None and eval_tok_s > 5.0,
    'manual_quality_review_required': True,
    'exact_command_file': str(run_dir / 'exact_command.txt'),
    'environment_file': str(run_dir / 'environment.txt'),
    'resource_samples_file': str(run_dir / 'resource_samples.tsv'),
    'source_dirty': config.get('source_dirty'),
    'source_head': config.get('source_head'),
    'source_branch': config.get('source_branch'),
}
(run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
(run_dir / 'answer.txt').write_text(answer + ('\n' if answer else ''), encoding='utf-8')

print('\n=== Demo summary ===')
for key in ['run_ok', 'eval_tok_s', 'prompt_tok_s', 'first_output_ms', 'elapsed_seconds', 'memory_peak_bytes', 'memory_file_bytes', 'ram_ok', 'answer_present', 'product_target_gt_5_tok_s_met_by_this_run', 'source_dirty']:
    print(f'{key}: {summary.get(key)}')
print(f'run_dir: {run_dir}')
print('\n=== Model answer ===')
print(answer)
PY

exit "$systemd_status"
