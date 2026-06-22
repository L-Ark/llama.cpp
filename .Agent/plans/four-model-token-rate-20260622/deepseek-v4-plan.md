# DeepSeek V4 Flash ik_llama Deployment And Token Rate Plan

## Goal

Deploy DeepSeek V4 Flash with ik_llama, keep host RAM at or below 16 GB, maximize RTX 5090 VRAM usage, and optimize decode token rate using the same record/push discipline as previous DeepSeek/GLM optimization tasks.

This task must be completed on the provided benchmark server, not on the local laptop/workstation used to edit this plan. All benchmark commands, model paths, memory limits, GPU checks, commits, and pushes refer to the provided server environment.

The actual execution may happen on a new server. Do not assume the model files from any previous server are present; first verify the target server paths and download the required model shards again if missing. Model download time remains excluded from the integration timer.

When executing this model task on the server, create a dedicated branch from the latest `main` before any deployment work:

```bash
git switch main
git pull --ff-only origin main
git switch -c deepseek-v4-flash
git push -u origin deepseek-v4-flash
```

All DeepSeek V4 Flash integration commits, baseline records, token-rate improvements, and immediate improvement pushes must go to the `deepseek-v4-flash` branch.

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

Optimization ideas may be chosen independently based on DeepSeek V4 Flash's actual architecture and bottlenecks. Before starting any optimization attempt, write a concrete optimization plan in this file, including the hypothesis, code/config changes to try, benchmark command, success metric, rollback condition, and expected logs. During the attempt, record the full process in this same plan: commands, metrics, failures, reverted ideas, elapsed time, log paths, and whether the result was pushed.

Model source and target:

- Download name: `deepseek-ai/DeepSeek-V4-Flash`
- Source files: `model-00001-of-00046.safetensors` through `model-00046-of-00046.safetensors`
- Target source path on server: `/root/lfz/models/DeepSeek-V4-Flash/`
- ik_llama GGUF target on server: `/root/lfz/models/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-00001-of-*.gguf`
- If an existing GGUF or prior deployment is found, first measure baseline before further work.

## Timing And Records

Start timing after model files or converted GGUF files are present. Download and conversion time are excluded; ik_llama architecture, tensor, graph, runtime, and benchmark work are included.

Every successful baseline or improved run must append a row with:

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Rules:

- First successful token rate is `baseline` and must be recorded.
- A token-rate improvement means `eval_tok_s > previous_best_eval_tok_s + 0.01`.
- After every improvement, update this plan, commit all effective tracked changes, and immediately `git push origin HEAD`.
- Record repeat metrics for any claimed best: p50, worst, and log paths.

## Baseline Command

Use the converted GGUF path once available:

```bash
MODEL=/root/lfz/models/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-00001-of-*.gguf
IK=/root/lfz/ik_llama
RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-$(date -u +%Y%m%d-%H%M%SZ)
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

- Use ik_llama's DeepSeek/MLA path as the starting point, not fastllm runtime assumptions.
- Verify DeepSeek V4 metadata against DeepSeek2/DSA support: attention head sizes, MLA latent dims, rope, routed expert count, shared experts, routing top-k, and FP8/MXFP scale formats.
- For MatMul shape errors, log the failing node, tensor names, and `ne[]` dimensions before changing tensor mapping or graph code.
- Do not reduce top-k or active expert count for reported token-rate improvements.

## Optimization Routes

Apply previous-task routes in this order:

- Baseline instrumentation: TTFT, first visible token, prompt/eval tok/s, total ms, RSS, VRAM, direct reads, read bandwidth, cache hit rates, read failures.
- VRAM fill scan: increase expert/KV/cache placement until near full VRAM; back off by `1024` MiB on CUDA OOM.
- Expert pack route: pack routed expert payloads into contiguous layout if the GGUF path is seek/read limited; record effective read GB/s.
- Direct read/io_uring route: port previous direct-read or io_uring worker ideas only if actual expert read bandwidth is limiting token rate.
- RAM cache route: allow a bounded CPU expert cache only if total RSS stays under 16 GB; record RAM hit rate and eviction.
- Routing-profile VRAM cache: select hot experts by measured routing profile benefit per byte.
- Decode prefetch pipeline: overlap route decision, SSD read, decompression/dequant, and H2D copy without changing model math.

## Acceptance

- `llama-cli` builds.
- Benchmark exits with code `0`.
- Host RSS peak is `<= 16384 MB`.
- No CUDA OOM, shape mismatch, or read failure.
- Baseline and every improvement include TTFT and throughput metrics.
- Every improvement has a pushed commit hash in this plan.
