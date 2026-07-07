#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-dev_france_regression}

ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260707-phase0-io-trace-n32
RUN=$ROOT/$MODE
REPO=/root/lfz/tmp/kimi-stage2m-align
MODEL=/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf

case "$MODE" in
    dev_france_regression)
        PROMPT_TEXT="Please introduce France in a short paragraph."
        QUALITY="france,europe|paris|eiffel|louvre|riviera|bordeaux"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

PROMPT="<|im_user|>user<|im_middle|>${PROMPT_TEXT}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"

mkdir -p "$RUN"
cd "$REPO"

cat > "$RUN/env.txt" <<EOF
GGML_MOE_STREAM_SERIAL_STAGE_BATCH=1
GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack
GGML_MOE_EXPERT_PACK_OVERLAY=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack
GGML_MOE_IO_BACKEND=iouring
GGML_MOE_IO_BYTES=8388608
GGML_MOE_IO_DEPTH=8
GGML_MOE_IO_REFILL_BATCH=4
GGML_MOE_IO_SORT_OFFSET=1
GGML_MOE_IO_SQPOLL=1
GGML_MOE_MMAP_DONTNEED=1
GGML_MOE_PARALLEL_EXPERTS=1
GGML_MOE_PREFETCH_DOWN=1
GGML_MOE_PREFETCH_DOWN_DEPTH=2
GGML_MOE_STAGE_PINNED=1
GGML_MOE_STAGE_PINNED_SLOTS=12
GGML_MOE_STREAM=1
GGML_MOE_STREAM_BATCH_ONLY=1
GGML_MOE_STREAM_DOWN_BATCH=1
GGML_MOE_STREAM_FUSED_UP_GATE=1
GGML_MOE_STREAM_FUSED_UP_GATE_MIXED_TYPES=1
GGML_MOE_MIXED_UP_GATE_PARALLEL_STAGE=1
GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
GGML_MOE_VRAM_CACHE_MIB=15000
GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
GGML_MOE_VRAM_CACHE_SPLIT=1
GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6
GGML_MOE_VRAM_CACHE_UPGATE_PCT=62
GGML_MOE_DOWN_PARALLEL_STAGE=1
GGML_MOE_DOWN_STAGE_SINGLE_RING=1
GGML_MOE_CURRENT_DOWN_OVERLAP=1
GGML_MOE_CPU_FALLBACK_PACK_MMAP=1
LLAMA_DROP_DENSE_MMAP_CACHE=1
LLAMA_DROP_EXPERT_MMAP_AFTER_PROMPT=1
LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1
GGML_MOE_STREAM_UP_GATE_PARALLEL=1
GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE=1
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
GGML_MOE_EXPERT_GGUF_ALIAS_TSV=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv
GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1
GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv
GGML_MOE_IO_READ_TRACE_OUT=$RUN/io-read-trace.csv
GGML_MOE_IO_LOCALITY_PROFILE_OUT=$RUN/io-locality-profile.csv
GGML_MOE_STAGE_GRANULARITY_PROFILE=1
EOF

set -a
source "$RUN/env.txt"
set +a

cat > "$RUN/command.txt" <<EOF
RUN=$RUN
N=32
build-cuda-batch/bin/llama-completion --defer-experts --fit off -ngl 99 --special -m $MODEL -c 512 -n 32 --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-display-prompt -no-cnv -t 32 -tb 32 -p '$PROMPT'
EOF

sync
echo 3 > /proc/sys/vm/drop_caches

set +e
systemd-run --scope -p MemoryMax=16000M -p MemorySwapMax=0 bash -lc \
    "cd '$REPO' && ./build-cuda-batch/bin/llama-completion --defer-experts --fit off -ngl 99 --special -m '$MODEL' -c 512 -n 32 --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-display-prompt -no-cnv -t 32 -tb 32 -p '$PROMPT' > '$RUN/answer.txt' 2> '$RUN/stderr.txt'"
rc=$?
set -e
echo "$rc" > "$RUN/exit.txt"
cat /proc/meminfo > "$RUN/meminfo.final.txt" || true

python3 - "$RUN" "$PROMPT_TEXT" "$QUALITY" <<'PY'
import pathlib
import re
import sys

run = pathlib.Path(sys.argv[1])
prompt = sys.argv[2]
quality = sys.argv[3]
stderr = (run / "stderr.txt").read_text(errors="ignore")
answer = (run / "answer.txt").read_text(errors="ignore")

out = [
    f"run={run}",
    f"prompt={prompt}",
    "exit=" + (run / "exit.txt").read_text().strip(),
    "quality_keywords=" + quality,
    "output=" + answer.replace("\n", " ")[:500],
]

for pattern, name in [
    (r"common_perf_print: prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", "prompt"),
    (r"common_perf_print:\s*eval time =\s*([0-9.]+) ms /\s*([0-9]+) runs\s*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\)", "decode"),
    (r"common_perf_print:\s*total time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", "total"),
]:
    m = re.search(pattern, stderr)
    if name == "prompt" and m:
        out.append(f"prompt_eval_ms={m.group(1)} prompt_tokens={m.group(2)}")
    elif name == "decode" and m:
        out.append(f"decode_ms={m.group(1)} decode_runs={m.group(2)} ms_per_token={m.group(3)} token_rate={m.group(4)}")
    elif name == "total" and m:
        out.append(f"total_ms={m.group(1)} total_tokens={m.group(2)}")

for pattern, name in [
    (r"\[moe_stream_batch\] expert pack: hits=.*", "expert_pack"),
    (r"\[moe_stream_batch\] expert pack iouring detail:.*", "expert_pack_iouring"),
    (r"\[moe_stream_batch\] pinned staging: copies=.*", "pinned_staging"),
    (r"\[moe_stream_batch\] pinned staging gate: copies=.*", "pinned_staging_gate"),
    (r"\[moe_stream_batch\] pinned staging granularity:.*", "stage_granularity"),
    (r"\[moe_stream_batch\] pinned staging gate granularity:.*", "stage_granularity_gate"),
    (r"\[moe_stream_batch\] current down overlap:.*", "current_down_overlap"),
]:
    matches = re.findall(pattern, stderr)
    if matches:
        out.append(name + "=" + matches[-1])

(run / "metrics.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
PY
