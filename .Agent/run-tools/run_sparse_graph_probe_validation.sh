#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: run_sparse_graph_probe_validation.sh <defaultoff|probe> <case-dir>}"
case_dir="${2:?usage: run_sparse_graph_probe_validation.sh <defaultoff|probe> <case-dir>}"

repo=/root/lfz/vendor/llama.cpp-deepseek-v4
model=/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf
fixed_text=/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light/fixed-france-text.txt
baseline_top1=/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light/top1-baseline.json
profile_json="$repo/.Agent/profiles/vendor-ds4/current_sota_sparse_pair_top64_updown.profile.json"

mkdir -p "$case_dir"
cd "$repo"

export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39
export GGML_MOE_KEEP_TOPK_LAYER_VALUE=3
export GGML_MOE_KEEP_TOPK_UPDOWN=4
export GGML_MOE_STREAM=1
export GGML_MOE_STREAM_CACHE_ADMIT_PROFILE="$repo/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv"
export GGML_MOE_STREAM_DONTNEED=1
export GGML_MOE_STREAM_ONE_CACHE_MIB=13568
export GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1
export GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack
export GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct
export GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps
export GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000
export GGML_MOE_STREAM_ONE_PREFILL_PROFILE="$repo/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv"
export GGML_MOE_STREAM_ONE_TRACE_OUT="$case_dir/one-trace.csv"
export GGML_MOE_VRAM_CACHE_GB=0

if [[ "$mode" == "probe" ]]; then
    export DS4_SPARSE_PAIR_GRAPH_PROBE=1
    export DS4_SPARSE_PAIR_PROFILE_JSON="$profile_json"
    export DS4_SPARSE_PAIR_GRAPH_PROBE_OUT="$case_dir/sparse-graph-probe.csv"
elif [[ "$mode" != "defaultoff" ]]; then
    echo "unknown mode: $mode" >&2
    exit 2
fi

/usr/bin/time -v build-ds4-moe-stream/bin/llama-results \
  -m "$model" \
  -f "$fixed_text" \
  --sequential-logits -c 256 -b 16 -ub 16 -t 20 -tb 20 -ngl all --fit on -fa auto --n-cpu-moe 40 --defer-experts \
  --top1-report "$case_dir/top1.json" > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt"

cg=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
cgdir="/sys/fs/cgroup$cg"
cp "$cgdir/memory.peak" "$case_dir/memory.peak" || true
cp "$cgdir/memory.current" "$case_dir/memory.current" || true
cp "$cgdir/memory.events" "$case_dir/memory.events" || true
cp "$cgdir/memory.stat" "$case_dir/memory.stat" || true

python3 - "$case_dir" "$mode" "$baseline_top1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

case = Path(sys.argv[1])
mode = sys.argv[2]
baseline_path = Path(sys.argv[3])
base = json.loads(baseline_path.read_text(encoding="utf-8"))
cur = json.loads((case / "top1.json").read_text(encoding="utf-8"))
base_pos = base["positions"]
cur_pos = cur["positions"]
n = min(len(base_pos), len(cur_pos))
same = sum(1 for i in range(n) if base_pos[i]["top1"]["id"] == cur_pos[i]["top1"]["id"])
first = next((i for i in range(n) if base_pos[i]["top1"]["id"] != cur_pos[i]["top1"]["id"]), -1)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

events = (case / "memory.events").read_text(encoding="utf-8") if (case / "memory.events").exists() else ""
summary = {
    "mode": mode,
    "status": "pass" if same == len(base_pos) == len(cur_pos) and first == -1 and "oom 0" in events and "oom_kill 0" in events else "fail",
    "case_dir": str(case),
    "n_tokens": len(base_pos),
    "check_n_tokens": len(cur_pos),
    "same_top1": same,
    "first_mismatch_pos": first,
    "baseline_top1_report_sha256": sha256(baseline_path),
    "check_top1_report_sha256": sha256(case / "top1.json"),
    "memory_peak_bytes": int((case / "memory.peak").read_text().strip()) if (case / "memory.peak").exists() else None,
    "memory_current_bytes": int((case / "memory.current").read_text().strip()) if (case / "memory.current").exists() else None,
    "memory_events": events,
}
(case / "top1-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if summary["status"] == "pass" else 1)
PY

if [[ "$mode" == "probe" ]]; then
    "$repo/.Agent/run-tools/analyze_sparse_graph_probe.py" \
      --probe-csv "$case_dir/sparse-graph-probe.csv" \
      --profile-json "$profile_json" \
      --summary-out "$case_dir/sparse-graph-probe-summary.json"
fi
