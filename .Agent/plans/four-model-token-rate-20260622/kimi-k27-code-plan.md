# Kimi K2.7 Code ik_llama Deployment And Token Rate Plan

## Goal

Deploy Kimi with ik_llama under a 16 GB host RAM cap, maximize RTX 5090 VRAM usage, and optimize decode token rate with full TTFT/cache/read telemetry.

This task must be completed on the provided benchmark server, not on the local laptop/workstation used to edit this plan. All benchmark commands, model paths, memory limits, GPU checks, commits, and pushes refer to the provided server environment.

The actual execution may happen on a new server. Do not assume the model files from any previous server are present; first verify the target server paths and download the required model shards again if missing. Model download time remains excluded from the integration timer.

When executing this model task on the server, create a dedicated branch from the latest `main` before any deployment work:

```bash
git switch main
git pull --ff-only origin main
git switch -c kimi-k2.7-code
git push -u origin kimi-k2.7-code
```

All Kimi K2.7 Code integration commits, baseline records, token-rate improvements, and immediate improvement pushes must go to the `kimi-k2.7-code` branch.

Model source and target:

- Default download name: `moonshotai/Kimi-K2.7-Code`
- Source files: `model-00001-of-000064.safetensors` through `model-00064-of-000064.safetensors`
- Target source path on server: `/root/lfz/models/Kimi-K2.7-Code/`
- ik_llama GGUF target on server: `/root/lfz/models/Kimi-K2.7-Code-GGUF/Kimi-K2.7-Code-00001-of-*.gguf`
- If the task owner later specifies Kimi K2.5, replace the model ID with `moonshotai/Kimi-K2.5` and keep the same workflow.

## Timing And Records

Start timing only after model files or converted GGUF files are present. Download and conversion time are excluded.

Every successful baseline or improved run must append a row with:

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Rules:

- First successful token rate is `baseline` and must be recorded.
- A token-rate improvement means `eval_tok_s > previous_best_eval_tok_s + 0.01`.
- After every improvement, update this plan, commit related changes, and immediately `git push origin HEAD`.
- Record output hash and smoke accuracy when a new best is claimed.

## Baseline Command

Use the converted GGUF path once available:

```bash
MODEL=/root/lfz/models/Kimi-K2.7-Code-GGUF/Kimi-K2.7-Code-00001-of-*.gguf
IK=/root/lfz/ik_llama
RUN_DIR=/root/lfz/runs/ik_llama/kimi-k27-code-$(date -u +%Y%m%d-%H%M%SZ)
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

- Start from ik_llama's existing Kimi tokenizer/chat-template support and MLA/MoE paths.
- Validate Kimi-specific metadata: tokenizer pre type, chat template, expert count, top-k, grouped routing fields, MLA dims, rope, and scale formats.
- Do not reuse DeepSeek or GLM routing constants unless Kimi metadata confirms the same values.
- For shape errors, trace tensor mapping and graph build dimensions before any arithmetic changes.

## Optimization Routes

- Baseline instrumentation: TTFT, first visible token, prompt/eval tok/s, total ms, RSS, VRAM, cache hit rates, direct reads, read bandwidth, and read failures.
- VRAM fill scan: tune `GGML_MOE_VRAM_CACHE_MIB` and dense layer placement; back off on CUDA OOM.
- Tokenizer/template cleanup: ensure no avoidable prompt overhead or wrong template path affects TTFT.
- Routing-profile VRAM cache: profile Kimi expert use and pin hot experts by benefit per byte.
- Expert pack/direct read: use contiguous expert pack and previous direct-read/io_uring ideas only when logs show read bandwidth or seeks dominate.
- Decode prefetch: overlap likely expert reads with compute without changing routing/top-k.
- Startup preload: preload hot early-layer experts only if it improves TTFT and keeps RSS under 16 GB.

## Acceptance

- `llama-cli` builds.
- Benchmark exits with code `0`.
- Host RSS peak is `<= 16384 MB`.
- No CUDA OOM, shape mismatch, tokenizer failure, or read failure.
- Baseline and every improvement include TTFT and throughput metrics.
- Every improvement has a pushed commit hash in this plan.
