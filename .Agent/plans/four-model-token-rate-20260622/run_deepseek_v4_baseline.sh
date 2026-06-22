#!/usr/bin/env bash
set -euo pipefail

IK="${IK:-/root/lfz/ik_llama}"
NATIVE_MODEL="${NATIVE_MODEL:-/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf}"
TARGET_DIR="${TARGET_DIR:-/root/lfz/models/DeepSeek-V4-Flash-GGUF}"
TARGET_MODEL="${TARGET_MODEL:-$TARGET_DIR/DeepSeek-V4-Flash-00001-of-00001.gguf}"
EXPECTED_BYTES="${EXPECTED_BYTES:-156148189760}"
RUN_DIR="${RUN_DIR:-/root/lfz/runs/ik_llama/deepseek-v4-$(date -u +%Y%m%d-%H%M%SZ)}"
N_CTX="${N_CTX:-512}"
N_PREDICT="${N_PREDICT:-256}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MEMORY_MAX="${MEMORY_MAX:-16G}"
DROP_DEFERRED_EXPERT_PAGES="${IK_LLAMA_DROP_DEFERRED_EXPERT_PAGES:-0}"
MOE_STREAM="${GGML_MOE_STREAM:-0}"
MOE_STREAM_DEFER="${GGML_MOE_STREAM_DEFER:-1}"
MOE_STREAM_FUSED_UP_GATE="${GGML_MOE_STREAM_FUSED_UP_GATE:-0}"
MOE_STREAM_BATCH_ONLY="${GGML_MOE_STREAM_BATCH_ONLY:-0}"
MOE_STREAM_CPU_OPS="${GGML_MOE_STREAM_CPU_OPS:-0}"
MOE_PREFETCH="${GGML_MOE_PREFETCH:-0}"
MOE_PREDICT="${GGML_MOE_PREDICT:-0}"
MOE_PARALLEL_EXPERTS="${GGML_MOE_PARALLEL_EXPERTS:-0}"
MOE_VRAM_CACHE_SPLIT="${GGML_MOE_VRAM_CACHE_SPLIT:-0}"
MOE_VRAM_CACHE_UPGATE_PCT="${GGML_MOE_VRAM_CACHE_UPGATE_PCT:-60}"
MOE_VRAM_PROFILE="${GGML_MOE_VRAM_PROFILE:-}"
MOE_VRAM_PROFILE_PROTECT="${GGML_MOE_VRAM_PROFILE_PROTECT:-0}"
MOE_VRAM_PROFILE_RESERVE_PCT="${GGML_MOE_VRAM_PROFILE_RESERVE_PCT:-10}"
MOE_ROUTE_TRACE_OUT="${GGML_MOE_ROUTE_TRACE_OUT:-}"
MOE_BATCH_PROFILE="${GGML_MOE_BATCH_PROFILE:-0}"
MOE_BATCH_PROFILE_OUT="${GGML_MOE_BATCH_PROFILE_OUT:-}"
MOE_EXPERT_PACK="${GGML_MOE_EXPERT_PACK:-}"
MOE_IO_BACKEND="${GGML_MOE_IO_BACKEND:-}"
MOE_IO_DEPTH="${GGML_MOE_IO_DEPTH:-16}"
MOE_IO_BYTES="${GGML_MOE_IO_BYTES:-2097152}"
MOE_STAGE_PINNED="${GGML_MOE_STAGE_PINNED:-0}"
MOE_STAGE_PINNED_SLOTS="${GGML_MOE_STAGE_PINNED_SLOTS:-16}"
MOE_GPU_HANDOFF="${GGML_MOE_GPU_HANDOFF:-0}"
MOE_STREAM_UP_GATE_PARALLEL="${GGML_MOE_STREAM_UP_GATE_PARALLEL:-0}"
MOE_STREAM_UP_GATE_PARALLEL_STAGE="${GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE:-0}"
MOE_STREAM_UP_GATE_STAGE_SPLIT="${GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT:-0}"
MOE_DOWN_PARALLEL_STAGE="${GGML_MOE_DOWN_PARALLEL_STAGE:-0}"
HOTEXP_CACHE_GB="${GGML_HOTEXP_CACHE_GB:-0}"
HOTEXP_DEBUG="${GGML_HOTEXP_DEBUG:-0}"
HOTEXP_PROFILE_OUT="${GGML_HOTEXP_PROFILE_OUT:-}"
HOTEXP_INSERT_ON_MISS="${GGML_HOTEXP_INSERT_ON_MISS:-1}"

mkdir -p "$RUN_DIR" "$TARGET_DIR"
date -u +%FT%TZ > "$RUN_DIR/start_utc.txt"

actual_bytes=0
if [[ -f "$NATIVE_MODEL" ]]; then
    actual_bytes="$(stat -c '%s' "$NATIVE_MODEL")"
fi

aria2_present=0
[[ -f "$NATIVE_MODEL.aria2" ]] && aria2_present=1

if [[ "$actual_bytes" != "$EXPECTED_BYTES" || "$aria2_present" != "0" ]]; then
    cat > "$RUN_DIR/not_ready.json" <<JSON
{
  "status": "not_ready",
  "native_model": "$NATIVE_MODEL",
  "actual_bytes": $actual_bytes,
  "expected_bytes": $EXPECTED_BYTES,
  "aria2_present": $aria2_present
}
JSON
    cat "$RUN_DIR/not_ready.json"
    exit 2
fi

ln -sfn "$NATIVE_MODEL" "$TARGET_MODEL"

MODEL_GLOB="$TARGET_DIR/DeepSeek-V4-Flash-00001-of-"*.gguf
matches=( $MODEL_GLOB )
if [[ "${#matches[@]}" -ne 1 ]]; then
    printf 'expected one model match, got %d\n' "${#matches[@]}" > "$RUN_DIR/error.txt"
    exit 3
fi
MODEL="${matches[0]}"

{
    echo "run_dir=$RUN_DIR"
    echo "git_sha=$(cd "$IK" && git rev-parse --short HEAD)"
    echo "branch=$(cd "$IK" && git branch --show-current)"
    echo "model=$MODEL"
    echo "native_model=$NATIVE_MODEL"
    echo "expected_bytes=$EXPECTED_BYTES"
    echo "n_ctx=$N_CTX"
    echo "n_predict=$N_PREDICT"
    echo "extra_args=$EXTRA_ARGS"
    echo "memory_max=$MEMORY_MAX"
    echo "drop_deferred_expert_pages=$DROP_DEFERRED_EXPERT_PAGES"
    echo "ggml_moe_stream=$MOE_STREAM"
    echo "ggml_moe_stream_defer=$MOE_STREAM_DEFER"
    echo "ggml_moe_stream_fused_up_gate=$MOE_STREAM_FUSED_UP_GATE"
    echo "ggml_moe_stream_batch_only=$MOE_STREAM_BATCH_ONLY"
    echo "ggml_moe_stream_cpu_ops=$MOE_STREAM_CPU_OPS"
    echo "ggml_moe_prefetch=$MOE_PREFETCH"
    echo "ggml_moe_predict=$MOE_PREDICT"
    echo "ggml_moe_parallel_experts=$MOE_PARALLEL_EXPERTS"
    echo "ggml_moe_vram_profile=$MOE_VRAM_PROFILE"
    echo "ggml_moe_route_trace_out=$MOE_ROUTE_TRACE_OUT"
    echo "ggml_moe_expert_pack=$MOE_EXPERT_PACK"
    echo "ggml_moe_io_backend=$MOE_IO_BACKEND"
    echo "ggml_hotexp_cache_gb=$HOTEXP_CACHE_GB"
    echo "ggml_hotexp_debug=$HOTEXP_DEBUG"
    echo "ggml_hotexp_profile_out=$HOTEXP_PROFILE_OUT"
    echo "cuda_visible_devices=0"
} > "$RUN_DIR/context.env"

nvidia-smi > "$RUN_DIR/nvidia-smi.before.txt" 2>&1 || true

cat > "$RUN_DIR/run_inside_cgroup.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
exec > "$RUN_DIR/llama.direct.log" 2>&1
cd "$IK"
extra_args=()
if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a extra_args <<< "$EXTRA_ARGS"
fi
exec /usr/bin/time -v env CUDA_VISIBLE_DEVICES=0 \\
    GGML_CUDA_NO_PINNED=1 \\
    GGML_MOE_RAM_TIER_MIB=0 \\
    GGML_MOE_RAM_TIER_SKIP=0 \\
    GGML_MOE_VRAM_CACHE_MIB=24576 \\
    GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1 \\
    GGML_MOE_VRAM_CACHE_SAFETY_MIB=512 \\
    GGML_MOE_VRAM_CACHE_POLICY=lfu_lru \\
    GGML_MOE_STREAM="$MOE_STREAM" \\
    GGML_MOE_STREAM_DEFER="$MOE_STREAM_DEFER" \\
    GGML_MOE_STREAM_FUSED_UP_GATE="$MOE_STREAM_FUSED_UP_GATE" \\
    GGML_MOE_STREAM_BATCH_ONLY="$MOE_STREAM_BATCH_ONLY" \\
    GGML_MOE_STREAM_CPU_OPS="$MOE_STREAM_CPU_OPS" \\
    GGML_MOE_PREFETCH="$MOE_PREFETCH" \\
    GGML_MOE_PREDICT="$MOE_PREDICT" \\
    GGML_MOE_PARALLEL_EXPERTS="$MOE_PARALLEL_EXPERTS" \\
    GGML_MOE_VRAM_CACHE_SPLIT="$MOE_VRAM_CACHE_SPLIT" \\
    GGML_MOE_VRAM_CACHE_UPGATE_PCT="$MOE_VRAM_CACHE_UPGATE_PCT" \\
    GGML_MOE_VRAM_PROFILE="$MOE_VRAM_PROFILE" \\
    GGML_MOE_VRAM_PROFILE_PROTECT="$MOE_VRAM_PROFILE_PROTECT" \\
    GGML_MOE_VRAM_PROFILE_RESERVE_PCT="$MOE_VRAM_PROFILE_RESERVE_PCT" \\
    GGML_MOE_ROUTE_TRACE_OUT="$MOE_ROUTE_TRACE_OUT" \\
    GGML_MOE_BATCH_PROFILE="$MOE_BATCH_PROFILE" \\
    GGML_MOE_BATCH_PROFILE_OUT="$MOE_BATCH_PROFILE_OUT" \\
    GGML_MOE_EXPERT_PACK="$MOE_EXPERT_PACK" \\
    GGML_MOE_IO_BACKEND="$MOE_IO_BACKEND" \\
    GGML_MOE_IO_DEPTH="$MOE_IO_DEPTH" \\
    GGML_MOE_IO_BYTES="$MOE_IO_BYTES" \\
    GGML_MOE_STAGE_PINNED="$MOE_STAGE_PINNED" \\
    GGML_MOE_STAGE_PINNED_SLOTS="$MOE_STAGE_PINNED_SLOTS" \\
    GGML_MOE_GPU_HANDOFF="$MOE_GPU_HANDOFF" \\
    GGML_MOE_STREAM_UP_GATE_PARALLEL="$MOE_STREAM_UP_GATE_PARALLEL" \\
    GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE="$MOE_STREAM_UP_GATE_PARALLEL_STAGE" \\
    GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT="$MOE_STREAM_UP_GATE_STAGE_SPLIT" \\
    GGML_MOE_DOWN_PARALLEL_STAGE="$MOE_DOWN_PARALLEL_STAGE" \\
    GGML_HOTEXP_CACHE_GB="$HOTEXP_CACHE_GB" \\
    GGML_HOTEXP_DEBUG="$HOTEXP_DEBUG" \\
    GGML_HOTEXP_PROFILE_OUT="$HOTEXP_PROFILE_OUT" \\
    GGML_HOTEXP_INSERT_ON_MISS="$HOTEXP_INSERT_ON_MISS" \\
    IK_LLAMA_DROP_DEFERRED_EXPERT_PAGES="$DROP_DEFERRED_EXPERT_PAGES" \\
    "$IK/build-cuda/bin/llama-cli" \\
      --defer-experts \\
      --fit \\
      -m "$MODEL" \\
      -ngl 999 \\
      -c "$N_CTX" \\
      -n "$N_PREDICT" \\
      --ignore-eos \\
      --temp 0 --top-p 1.0 --top-k 1 \\
      --seed 1 \\
      --no-display-prompt \\
      "\${extra_args[@]}" \\
      -p "The capital of France is"
SCRIPT
chmod +x "$RUN_DIR/run_inside_cgroup.sh"

systemd_args=(--wait --collect -p "WorkingDirectory=$IK")
if [[ "$MEMORY_MAX" != "0" && "$MEMORY_MAX" != "none" && "$MEMORY_MAX" != "NONE" ]]; then
    systemd_args+=(-p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0)
fi

set +e
/usr/bin/time -v systemd-run "${systemd_args[@]}" "$RUN_DIR/run_inside_cgroup.sh" \
  > "$RUN_DIR/systemd.log" 2> "$RUN_DIR/time.log"
rc=$?
set -e

unit="$(grep -hoE 'run-u[0-9]+\\.service' "$RUN_DIR/systemd.log" "$RUN_DIR/time.log" | tail -n 1 || true)"
if [[ -n "$unit" ]]; then
    journalctl -u "$unit" --no-pager > "$RUN_DIR/llama.log" 2>&1 || true
else
    : > "$RUN_DIR/llama.log"
fi
cat "$RUN_DIR/systemd.log" "$RUN_DIR/time.log" "$RUN_DIR/llama.log" "$RUN_DIR/llama.direct.log" > "$RUN_DIR/bench.log"

nvidia-smi > "$RUN_DIR/nvidia-smi.after.txt" 2>&1 || true
echo "$rc" > "$RUN_DIR/exit_code.txt"

python3 - "$RUN_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
text = (run / "bench.log").read_text(errors="replace")
summary = {
    "status": "ok" if (run / "exit_code.txt").read_text().strip() == "0" else "failed",
    "exit_code": int((run / "exit_code.txt").read_text().strip()),
    "log_path": str(run / "bench.log"),
}

patterns = {
    "eval": r"eval time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) runs.*?([0-9.]+) tokens per second",
    "prompt_eval": r"prompt eval time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) tokens.*?([0-9.]+) tokens per second",
    "total": r"total time\s*=\s*([0-9.]+) ms",
    "rss": r"Maximum resident set size \(kbytes\):\s*([0-9]+)",
}

m = re.search(patterns["eval"], text, re.S)
if m:
    summary["eval_ms"] = float(m.group(1))
    summary["gen_tokens"] = int(m.group(2))
    summary["eval_tok_s"] = float(m.group(3))

m = re.search(patterns["prompt_eval"], text, re.S)
if m:
    summary["prompt_eval_ms"] = float(m.group(1))
    summary["prompt_tokens"] = int(m.group(2))
    summary["prompt_eval_tok_s"] = float(m.group(3))

m = re.search(patterns["total"], text)
if m:
    summary["total_ms"] = float(m.group(1))

rss_values = [int(v) for v in re.findall(patterns["rss"], text)]
if rss_values:
    summary["host_rss_peak_mb"] = max(rss_values) / 1024.0

first = None
for line in text.splitlines():
    if line and not line.startswith(("llama_", "ggml_", "systemd-run", "llm_", "build:", "load_")):
        if "The capital of France is" not in line:
            first = line[:120]
            break
summary["first_visible_sample"] = first

(run / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

exit "$rc"
