#!/usr/bin/env bash
set -o pipefail

cd "${REPO:-/root/lfz/llama.cpp-vendor-kimi}" || exit 1

: "${RUN:?RUN required}"
: "${N:=96}"
: "${PINNED_SLOTS:=12}"
: "${VRAM_MIB:=15000}"
: "${THREADS:=32}"
: "${UPGATE_PCT:=62}"
: "${IQ2_UPGATE_PARALLEL:=1}"
: "${MOE_IO_DEPTH:=8}"
: "${MOE_IO_REFILL_BATCH:=4}"
: "${MOE_PREFETCH_DOWN_DEPTH:=2}"
: "${PROMPT_ID:=custom}"
: "${PROMPT_USER_TEXT:=Please introduce France in a short paragraph.}"
: "${QUALITY_KEYWORDS:=}"
: "${PROFILE:=0}"
: "${COPY_PROFILE:=0}"
: "${EXTRA_RUNTIME_ENV:=}"
: "${MODEL_PATH:=/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf}"
: "${EXPERT_GGUF_ALIAS_TSV:=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv}"
: "${MOE_IO_ALIGNED_ALIAS_BATCH:=1}"

PROMPT="<|im_user|>user<|im_middle|>${PROMPT_USER_TEXT}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"
mkdir -p "$RUN"

CG=/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
printf "%s\n" "$CG" > "$RUN/cgroup.txt"
cat "$CG/memory.max" > "$RUN/memory.max.txt" 2>/dev/null || true
cat "$CG/memory.swap.max" > "$RUN/memory.swap.max.txt" 2>/dev/null || true

REPO_DIR="${REPO:-/root/lfz/llama.cpp-vendor-kimi}"
{
  echo "repo=$REPO_DIR"
  echo "pwd=$PWD"
  echo "branch=$(env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" branch --show-current 2>&1)"
  echo "head=$(env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" rev-parse HEAD 2>&1)"
  echo "short=$(env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" rev-parse --short HEAD 2>&1)"
  echo "status_short:"
  env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" status --short 2>&1
  echo "diff_stat:"
  env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" diff --stat HEAD -- 2>&1
} > "$RUN/git.txt"
cp "$0" "$RUN/script.sh"

cat > "$RUN/env.txt" <<EOF
GGML_MOE_STREAM_SERIAL_STAGE_BATCH=1
GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack
GGML_MOE_EXPERT_PACK_OVERLAY=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack
GGML_MOE_IO_BACKEND=iouring
GGML_MOE_IO_BYTES=8388608
GGML_MOE_IO_DEPTH=$MOE_IO_DEPTH
GGML_MOE_IO_REFILL_BATCH=$MOE_IO_REFILL_BATCH
GGML_MOE_IO_SORT_OFFSET=1
GGML_MOE_IO_SQPOLL=1
GGML_MOE_MMAP_DONTNEED=1
GGML_MOE_PARALLEL_EXPERTS=1
GGML_MOE_PREFETCH_DOWN=1
GGML_MOE_PREFETCH_DOWN_DEPTH=$MOE_PREFETCH_DOWN_DEPTH
GGML_MOE_STAGE_PINNED=1
GGML_MOE_STAGE_PINNED_SLOTS=$PINNED_SLOTS
GGML_MOE_STREAM=1
GGML_MOE_STREAM_BATCH_ONLY=1
GGML_MOE_STREAM_DOWN_BATCH=1
GGML_MOE_STREAM_FUSED_UP_GATE=1
GGML_MOE_STREAM_FUSED_UP_GATE_MIXED_TYPES=1
GGML_MOE_MIXED_UP_GATE_PARALLEL_STAGE=1
GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
GGML_MOE_VRAM_CACHE_MIB=$VRAM_MIB
GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
GGML_MOE_VRAM_CACHE_SPLIT=1
GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6
GGML_MOE_VRAM_CACHE_UPGATE_PCT=$UPGATE_PCT
GGML_MOE_DOWN_PARALLEL_STAGE=1
GGML_MOE_DOWN_STAGE_SINGLE_RING=1
GGML_MOE_CURRENT_DOWN_OVERLAP=1
GGML_MOE_Q4_DOWN_BATCH=1
GGML_MOE_CPU_FALLBACK_PACK_MMAP=1
LLAMA_DROP_DENSE_MMAP_CACHE=1
LLAMA_DROP_EXPERT_MMAP_AFTER_PROMPT=1
LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1
GGML_MOE_EXPERT_GGUF_ALIAS_TSV=$EXPERT_GGUF_ALIAS_TSV
GGML_MOE_IO_ALIGNED_ALIAS_BATCH=$MOE_IO_ALIGNED_ALIAS_BATCH
EOF

if [ "$IQ2_UPGATE_PARALLEL" != "0" ]; then
  echo "GGML_MOE_STREAM_UP_GATE_PARALLEL=1" >> "$RUN/env.txt"
  echo "GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE=1" >> "$RUN/env.txt"
fi
if [ "$PROFILE" != "0" ]; then
  cat >> "$RUN/env.txt" <<EOF
GGML_KIMI_CPU_MOE_ELIGIBILITY_PROFILE=1
GGML_KIMI_CPU_MOE_NAME_PROFILE=1
GGML_KIMI_CPU_MOE_PROFILE=1
GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT=$RUN/fallback-profile.csv
GGML_MOE_DOWN_BATCH_PROFILE_OUT=$RUN/down-batch-profile.csv
GGML_MOE_UP_GATE_PROFILE_OUT=$RUN/up-gate-profile.csv
GGML_MOE_BATCH_PROFILE_OUT=$RUN/route-profile.csv
GGML_MOE_ROUTE_TRACE_OUT=$RUN/route-trace.csv
GGML_MOE_TTFT_TRACE_OUT=$RUN/ttft-trace.csv
GGML_MOE_BATCH_PROFILE=1
GGML_MOE_STREAM_DECLINE_DEBUG=1
GGML_MOE_TTFT_TRACE_MAX_EVENTS=120000
EOF
fi
if [ "$COPY_PROFILE" != "0" ]; then
  cat >> "$RUN/env.txt" <<EOF
GGML_MOE_COPY_PROFILE_OUT=$RUN/copy-profile.csv
GGML_MOE_COPY_PROFILE_H2D=1
EOF
fi
if [ -n "$EXTRA_RUNTIME_ENV" ]; then
  printf '%s\n' "$EXTRA_RUNTIME_ENV" >> "$RUN/env.txt"
fi

LLAMA_ARGS=(build-cuda-batch/bin/llama-completion --defer-experts --fit off -ngl 99 --special
  -m "$MODEL_PATH"
  -c 512 -n "$N" --temp 0 --top-p 1.0 --top-k 1 --seed 1
  --no-display-prompt -no-cnv -t "$THREADS" -tb "$THREADS" -p "$PROMPT")

{
  echo "PROMPT_ID=$PROMPT_ID"
  echo "PROMPT_USER_TEXT=$PROMPT_USER_TEXT"
  echo "QUALITY_KEYWORDS=$QUALITY_KEYWORDS"
  echo "RUN=$RUN"
  echo "N=$N"
  echo "PINNED_SLOTS=$PINNED_SLOTS"
  echo "VRAM_MIB=$VRAM_MIB"
  echo "THREADS=$THREADS"
  echo "UPGATE_PCT=$UPGATE_PCT"
  echo "IQ2_UPGATE_PARALLEL=$IQ2_UPGATE_PARALLEL"
  echo "MOE_IO_DEPTH=$MOE_IO_DEPTH"
  echo "MOE_IO_REFILL_BATCH=$MOE_IO_REFILL_BATCH"
  echo "MOE_PREFETCH_DOWN_DEPTH=$MOE_PREFETCH_DOWN_DEPTH"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "EXPERT_GGUF_ALIAS_TSV=$EXPERT_GGUF_ALIAS_TSV"
  echo "MOE_IO_ALIGNED_ALIAS_BATCH=$MOE_IO_ALIGNED_ALIAS_BATCH"
  echo "PROFILE=$PROFILE"
  echo "COPY_PROFILE=$COPY_PROFILE"
  printf '%q ' "${LLAMA_ARGS[@]}"
  echo
} > "$RUN/command.txt"

cat > "$RUN/README.md" <<EOF
# Kimi general prompt SOTA reproduction

Cold-start 16GB cgroup command shape:

systemd-run --wait --collect --same-dir \\
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \\
  env RUN=<new-run-dir> PROMPT_ID=<id> PROMPT_USER_TEXT='<prompt>' \\
      QUALITY_KEYWORDS='<keyword1|keyword2,...>' \\
      .Agent/run-tools/kimi-general-prompt-repro.sh
EOF

sync
echo 3 > /proc/sys/vm/drop_caches
date -Is > "$RUN/start.txt"
(
set -a
. "$RUN/env.txt"
set +a
/usr/bin/time -v "${LLAMA_ARGS[@]}"
) > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
rc=$?
date -Is > "$RUN/end.txt"
echo $rc > "$RUN/exit.txt"
cat "$CG/memory.current" > "$RUN/memory.current.final.txt" 2>/dev/null || true
cat "$CG/memory.peak" > "$RUN/memory.peak.txt" 2>/dev/null || true
cat "$CG/memory.events" > "$RUN/memory.events.txt" 2>/dev/null || true
cat "$CG/memory.stat" > "$RUN/memory.stat.final.txt" 2>/dev/null || true

python3 - "$RUN" "$PROMPT_ID" "$PROMPT_USER_TEXT" "$QUALITY_KEYWORDS" <<'PY'
import pathlib
import re
import sys

run = pathlib.Path(sys.argv[1])
prompt_id, prompt_text, keyword_spec = sys.argv[2], sys.argv[3], sys.argv[4]
stderr = (run / "stderr.txt").read_text(errors="replace") if (run / "stderr.txt").exists() else ""
stdout = (run / "stdout.txt").read_text(errors="replace").strip() if (run / "stdout.txt").exists() else ""

def clean_text(text):
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = text.replace("[end of text]", " ")
    return re.sub(r"\s+", " ", text).strip()

def quality(text, spec):
    clean = clean_text(text)
    lower = clean.lower()
    if len(clean) < 16:
        return False, "too_short"
    if re.search(r"\b(\w+)(?:\s+\1\b){4,}", lower):
        return False, "repeat_collapse"
    if clean.endswith(","):
        return False, "dangling_comma"
    if spec:
        groups = [g.strip() for g in spec.split(",") if g.strip()]
        for group in groups:
            choices = [c.strip().lower() for c in group.split("|") if c.strip()]
            if choices and not any(c in lower for c in choices):
                return False, f"missing_keyword:{group}"
    return True, "ok"

ok, reason = quality(stdout, keyword_spec)
metrics = [
    f"run={run}",
    f"prompt_id={prompt_id}",
    f"prompt={prompt_text}",
    f"exit={(run / 'exit.txt').read_text().strip() if (run / 'exit.txt').exists() else 'missing'}",
    f"quality={'pass' if ok else 'fail'}",
    f"quality_reason={reason}",
    f"output={clean_text(stdout)[:800]}",
]

mp = re.search(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", stderr)
me = re.search(r"eval time =\s*([0-9.]+) ms /\s*([0-9]+) runs\s*\([^\n]*?([0-9.]+) tokens per second\)", stderr)
if mp:
    metrics.append(f"ttft_ms={mp.group(1)}")
    metrics.append(f"prompt_tokens={mp.group(2)}")
if me:
    metrics.append(f"decode_ms={me.group(1)}")
    metrics.append(f"decode_runs={me.group(2)}")
    metrics.append(f"token_rate={me.group(3)}")

for fname, label in [
    ("memory.max.txt", "memory.max"),
    ("memory.swap.max.txt", "memory.swap.max"),
    ("memory.peak.txt", "memory.peak"),
    ("memory.current.final.txt", "memory.current.final"),
]:
    f = run / fname
    if f.exists():
        metrics.append(f"{label}={f.read_text().strip()}")

stat = {}
if (run / "memory.stat.final.txt").exists():
    for line in (run / "memory.stat.final.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            stat[parts[0]] = parts[1]
for key in ["anon", "file", "kernel", "inactive_file", "active_file", "pgmajfault", "pgfault", "workingset_refault_file"]:
    if key in stat:
        metrics.append(f"{key}={stat[key]}")

patterns = [
    (r"expert pack: hits=([^\n]+)", "expert_pack"),
    (r"expert pack iouring detail: ([^\n]+)", "expert_pack_iouring"),
    (r"pinned staging([^\n]+)", "pinned_staging"),
    (r"current down overlap: ([^\n]+)", "current_down_overlap"),
    (r"global expert sched shadow: ([^\n]+)", "global_expert_sched_shadow"),
    (r"VRAM cache down: ([^\n]+)", "vram_down"),
    (r"VRAM cache upgate: ([^\n]+)", "vram_upgate"),
]
for pattern, name in patterns:
    limit = 8 if name == "pinned_staging" else 4
    for i, match in enumerate(re.findall(pattern, stderr)[-limit:]):
        metrics.append(f"{name}_{i}={match}")

(run / "metrics.txt").write_text("\n".join(metrics) + "\n")
(run / "answer.txt").write_text(clean_text(stdout) + "\n")

def coerce(v):
    if v is None:
        return None
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v

json_metrics = {}
for item in metrics:
    if "=" not in item:
        continue
    key, value = item.split("=", 1)
    json_metrics[key] = coerce(value)

import json
(run / "metrics.json").write_text(json.dumps(json_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
PY

exit $rc
