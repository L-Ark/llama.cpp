# GLM 5.2 FP8 ik_llama Deployment And Token Rate Plan

## Goal

Deploy GLM 5.2 FP8 with ik_llama, keep host RAM at or below 16 GB, maximize RTX 5090 VRAM usage, and optimize token rate using the previous GLM/DeepSeek cache, profile, preload, and direct-read approaches where applicable.

This task must be completed on the provided benchmark server, not on the local laptop/workstation used to edit this plan. All benchmark commands, model paths, memory limits, GPU checks, commits, and pushes refer to the provided server environment.

The actual execution may happen on a new server. Do not assume the model files from any previous server are present; first verify the target server paths and download the required model shards again if missing. Model download time remains excluded from the integration timer.

When executing this model task on the server, create a dedicated branch from the latest `main` before any deployment work:

```bash
git switch main
git pull --ff-only origin main
git switch -c glm-5.2-fp8
git push -u origin glm-5.2-fp8
```

All GLM 5.2 FP8 integration commits, baseline records, token-rate improvements, and immediate improvement pushes must go to the `glm-5.2-fp8` branch.

All commits and pushes for this model task must use the GitHub account and commit identity:

```text
L-Ark <fliangae@connect.ust.hk>
```

Before the first task commit on the server, verify:

```bash
gh auth status
git config user.name "L-Ark"
git config user.email "fliangae@connect.ust.hk"
```

Optimization ideas may be chosen independently based on GLM 5.2 FP8's actual architecture and bottlenecks, and may reference previous ik_llama optimization experience such as GLM/DeepSeek expert packing, VRAM cache/profile placement, startup preload, direct-read/io_uring, prompt scheduling, and decode prefetch. Before starting any optimization attempt, write a concrete optimization plan in this file, including the hypothesis, code/config changes to try, benchmark command, success metric, rollback condition, and expected logs. During the attempt, record the full process in this same plan: commands, metrics, failures, reverted ideas, elapsed time, log paths, and whether the result was pushed. If any key metric regresses versus the previous recorded best, including eval tok/s, TTFT, first_visible_s, total_ms, RSS, VRAM stability, read_failures, or smoke accuracy, immediately roll back to the previous best commit/config and record the rollback reason and verification run in this plan before trying another route.

Model source and target:

- Download name: `zai-org/GLM-5.2-FP8`
- Source files: `model-00001-of-00141.safetensors` through `model-00141-of-00141.safetensors`
- Target source path on server: `/root/lfz/models/GLM-5.2-FP8/`
- ik_llama GGUF target on server: `/root/lfz/models/GLM-5.2-FP8-GGUF/GLM-5.2-FP8-00001-of-*.gguf`
- If an existing GLM 5.2 GGUF or deployment is found, first measure baseline before optimization.

## Timing And Records

Start timing only after model files or converted GGUF files are present. Download and conversion time are excluded.

Every successful baseline or improved run must append a row with:

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Rules:

- First successful token rate is `baseline` and must be recorded.
- A token-rate improvement means `eval_tok_s > previous_best_eval_tok_s + 0.01`.
- After every improvement, update this plan, commit related changes, and immediately `git push origin HEAD`.
- Best claims need at least one confirmation rerun; record p50 and worst when repeat runs are available.

## Baseline Command

Use the converted GGUF path once available:

```bash
MODEL=/root/lfz/models/GLM-5.2-FP8-GGUF/GLM-5.2-FP8-00001-of-*.gguf
IK=/root/lfz/ik_llama
RUN_DIR=/root/lfz/runs/ik_llama/glm-52-fp8-$(date -u +%Y%m%d-%H%M%SZ)
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

- Use existing GLM DSA/GLM MoE support as the nearest reference, but validate GLM 5.2 metadata directly.
- Check hparams and tensor shapes for MLA/DSA attention, routed experts, shared experts, sigmoid/router behavior, FP8 scales, MTP/NextN layers, and tokenizer/template fields.
- Reuse GLM 5.1 presets only as operational references; do not assume GLM 5.2 has identical dimensions or expert layout.
- For MatMul shape errors, log failing tensor names and `ne[]` dimensions before changing graph code.

## Optimization Routes

- Baseline instrumentation: TTFT, first visible token, prompt/eval tok/s, total ms, RSS, VRAM, cache hit rates, direct reads, read bandwidth, and read failures.
- GLM preset adaptation: start from previous GLM 5.1 low-host-RAM env patterns and update paths/metadata for GLM 5.2.
- VRAM fill scan: tune `GGML_MOE_VRAM_CACHE_MIB`, runtime safety margin, and dense layer placement; back off on CUDA OOM.
- Routing-profile VRAM cache: generate a GLM 5.2 route profile and pin hot experts by measured benefit per byte.
- Startup preload: preload first-prompt hot experts when TTFT improves and RSS remains under 16 GB.
- Dynamic prompt expert scheduling: reuse the previous GLM prompt MoE work-queue idea if prompt eval is the bottleneck.
- Expert pack/direct read/io_uring: use contiguous expert packs and direct-read/io_uring only when logs show read bandwidth or seeks dominate.
- Decode prefetch: overlap expert reads/H2D with compute without changing routing/top-k.

## Acceptance

- `llama-cli` builds.
- Benchmark exits with code `0`.
- Host RSS peak is `<= 16384 MB`.
- No CUDA OOM, shape mismatch, tokenizer failure, or read failure.
- Baseline and every improvement include TTFT and throughput metrics.
- Every improvement has a pushed commit hash in this plan.
