# MiniMax M3 ik_llama Deployment And Token Rate Plan

## Goal

Deploy MiniMax M3 with ik_llama, keep host RAM at or below 16 GB, fill RTX 5090 VRAM as much as the runtime can safely use, and optimize decode token rate with strict timing, metrics, commit, and push records.

This task must be completed on the provided benchmark server, not on the local laptop/workstation used to edit this plan. All benchmark commands, model paths, memory limits, GPU checks, commits, and pushes refer to the provided server environment.

The actual execution may happen on a new server. Do not assume the model files from any previous server are present; first verify the target server paths and download the required model shards again if missing. Model download time remains excluded from the integration timer.

When executing this model task on the server, create a dedicated branch from the latest `main` before any deployment work:

```bash
git switch main
git pull --ff-only origin main
git switch -c minimax-m3
git push -u origin minimax-m3
```

All MiniMax M3 integration commits, baseline records, token-rate improvements, and immediate improvement pushes must go to the `minimax-m3` branch.

Model source and target:

- Download name: `unsloth/MiniMax-M3-GGUF`
- Preferred GGUF: `UD-IQ3_XXS/MiniMax-M3-UD-IQ3_XXS-00001-of-00005.gguf`
- Target server path: `/root/lfz/models/MiniMax-M3-UD-IQ3_XXS/MiniMax-M3-UD-IQ3_XXS-00001-of-00005.gguf`
- If already present and runnable, skip download/conversion and immediately measure baseline.

## Timing And Records

Start timing after model files are present, at the first ik_llama integration action. Write `start_utc` before editing, configuring, or benchmarking. Model download and conversion time are not counted.

Every successful baseline or improved run must append a row with:

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Rules:

- First successful token rate is `baseline` and must be recorded even if slow.
- A token-rate improvement means `eval_tok_s > previous_best_eval_tok_s + 0.01`.
- After every improvement, update this plan, commit code/config/plan changes, then immediately `git push origin HEAD` before continuing.
- Raw logs stay under `/root/lfz/runs/ik_llama/minimax-m3-<utc>/` on the server; this plan records paths and parsed metrics.

## Baseline Command

Use a cgroup memory cap and no swap. Do not stop other users' GPU/RAM-heavy jobs; wait for them to finish naturally.

```bash
MODEL=/root/lfz/models/MiniMax-M3-UD-IQ3_XXS/MiniMax-M3-UD-IQ3_XXS-00001-of-00005.gguf
IK=/root/lfz/ik_llama
RUN_DIR=/root/lfz/runs/ik_llama/minimax-m3-$(date -u +%Y%m%d-%H%M%SZ)
mkdir -p "$RUN_DIR"
date -u +%FT%TZ > "$RUN_DIR/start_utc.txt"

/usr/bin/time -v systemd-run --pty --wait --collect \
  -p WorkingDirectory="$IK" \
  -p MemoryMax=16G \
  -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES=0 \
    GGML_MOE_RAM_TIER_MIB=0 \
    GGML_MOE_RAM_TIER_SKIP=0 \
    GGML_MOE_VRAM_CACHE_MIB=24576 \
    GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1 \
    GGML_MOE_VRAM_CACHE_SAFETY_MIB=512 \
    GGML_MOE_VRAM_CACHE_POLICY=lfu_lru \
    GGML_MOE_VRAM_CACHE_UPGATE_PCT=60 \
    "$IK/build-cuda/bin/llama-cli" \
      --defer-experts \
      -m "$MODEL" \
      -ngl 999 \
      -c 512 \
      -n 256 \
      --ignore-eos \
      --temp 0 --top-p 1.0 --top-k 1 \
      --seed 1 \
      --no-display-prompt \
      -p "The capital of France is" \
  > "$RUN_DIR/bench.log" 2>&1
```

## ik_llama Integration

- Follow ik_llama's native architecture path: arch registration, metadata mapping, tensor-name mapping, hparams, tensor shape checks, graph build, tokenizer, and chat template.
- Do not force MiniMax M3 through MiniMax M2 or fastllm shape assumptions. Validate expert count, top-k, gate/up/down tensor layout, attention dims, and norm placement from GGUF metadata.
- If a MatMul shape error appears, inspect tensor dimensions at the ik_llama tensor loader and graph node that created the failing matmul before changing math.

## Optimization Routes

Apply routes in this order and stop to record/push after each improvement:

- Baseline instrumentation: parse TTFT, first visible token time, prompt/eval tok/s, total ms, RSS peak, VRAM peak, cache hits, direct reads, and read failures from logs.
- VRAM fill scan: try `GGML_MOE_VRAM_CACHE_MIB` values `24576`, `26624`, `28672`, `30000`; if CUDA OOM occurs, reduce by `1024` MiB.
- Expert cache/profile route: reuse the GLM-style route profile and LFU/LRU VRAM cache pattern when MiniMax M3 tensors expose stable expert IDs.
- Startup preload route: preload early hot experts only if RSS remains under 16 GB and TTFT improves or does not regress materially.
- Direct read/io route: port only proven previous direct-read or io_uring ideas into ik_llama if MiniMax M3 is disk/pack limited; record read bandwidth and direct read counts.
- Decode prefetch route: prefetch next-token likely expert payloads without changing routing or top-k.

## Acceptance

- `llama-cli` builds.
- Benchmark exits with code `0`.
- Host RSS peak is `<= 16384 MB`.
- GPU VRAM is near full without CUDA OOM.
- `eval_tok_s`, `prompt_eval_tok_s`, `TTFT`, and `total_ms` are parsed.
- Baseline and every improvement are recorded and pushed.
