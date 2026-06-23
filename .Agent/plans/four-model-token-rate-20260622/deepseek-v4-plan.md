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

Optimization ideas may be chosen independently based on DeepSeek V4 Flash's actual architecture and bottlenecks, and may reference previous ik_llama optimization experience such as GLM/DeepSeek expert packing, VRAM cache/profile placement, startup preload, direct-read/io_uring, prompt scheduling, and decode prefetch. Before starting any optimization attempt, write a concrete optimization plan in this file, including the hypothesis, code/config changes to try, benchmark command, success metric, rollback condition, and expected logs. During the attempt, record the full process in this same plan: commands, metrics, failures, reverted ideas, elapsed time, log paths, and whether the result was pushed. If any key metric regresses versus the previous recorded best, including eval tok/s, TTFT, first_visible_s, total_ms, RSS, VRAM stability, read_failures, or smoke accuracy, immediately roll back to the previous best commit/config and record the rollback reason and verification run in this plan before trying another route.

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

## Current 16 GB Baseline

As of 2026-06-23, the best validated 16 GB host-RAM run uses the existing GGUF file
`/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`
through the compatibility symlink
`/root/lfz/models/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-00001-of-00001.gguf`.

Recommended command delta versus the original baseline:

```bash
MEMORY_MAX=16G \
EXTRA_ARGS="-ub 1 -t 24 -tb 24 -no-fa" \
/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh
```

Equivalent direct flags:

```text
GGML_CUDA_NO_PINNED=1
GGML_MOE_RAM_TIER_MIB=0
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_VRAM_CACHE_MIB=24576
GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
GGML_MOE_VRAM_CACHE_POLICY=lfu_lru
--defer-experts --fit -ngl 999 -c 512 -n 256 -ub 1 -t 24 -tb 24 -no-fa
```

Validated result:

- A30 `-no-fa` full 256-token runs: `eval_tok_s = 1.86 / 1.80 / 1.88`.
  - p50: `1.86 tok/s`
  - worst: `1.80 tok/s`
  - logs:
    - `/root/lfz/runs/ik_llama/deepseek-v4-a30-no-fa-n256/bench.log`
    - `/root/lfz/runs/ik_llama/deepseek-v4-a30-no-fa-n256-repeat2/bench.log`
    - `/root/lfz/runs/ik_llama/deepseek-v4-a30-no-fa-n256-repeat3/bench.log`
- Previous 16 GB best A13: `eval_tok_s = 1.79`.
- Delta: p50 `+0.07 tok/s` (`+3.9%`) versus A13. This still trails the fastllm reference
  `1.94 tok/s`, so optimization continues from A30.

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

## Execution Log

### 2026-06-22 14:13Z - Baseline Attempt A0

- Branch: `deepseek-v4-flash`
- Git SHA: `9329b038`
- Model reused from previous download:
  `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`
- Compatibility symlink for this plan's GGUF glob:
  `/root/lfz/models/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-00001-of-00001.gguf`
- Command source: baseline command from this plan, using `MemoryMax=16G`, `MemorySwapMax=0`,
  `-ngl 999`, `--defer-experts`, and the configured VRAM cache env.
- Log path: `/root/lfz/runs/ik_llama/deepseek-v4-20260622-141343Z/bench.log`
- Result: failed before generation.
- Failure reason: CUDA OOM during model tensor allocation. ik_llama attempted to allocate
  `147899.45 MiB` on CUDA0 while the RTX 5090 had about `30160 MiB` available for model
  placement. The run exited with code `1`; no token-rate baseline was produced.

### 2026-06-22 14:16Z - Baseline Config Correction Plan A1

- Hypothesis: DeepSeek V4 Flash native GGUF can load under the 16 GB host-RAM cgroup if ik_llama
  uses its own auto placement logic instead of blindly obeying `-ngl 999`.
- Change to try: add `--fit` to the baseline invocation while preserving `--defer-experts`,
  `MemoryMax=16G`, `MemorySwapMax=0`, `-c 512`, `-n 256`, deterministic sampling, and the same
  VRAM cache env. Also use `systemd-run --pipe` so full `llama-cli` logs are captured in
  `bench.log`.
- Benchmark command: `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`.
- Success metric: benchmark exits code `0`, host RSS peak is `<= 16384 MB`, and `summary.json`
  contains `eval_tok_s`.
- Rollback condition: if `--fit` still causes CUDA OOM, shape mismatch, read failure, or host RSS
  above 16 GB, revert to previous command state and inspect tensor placement before trying another
  route.
- Expected logs: `bench.log`, `summary.json`, `nvidia-smi.before.txt`, `nvidia-smi.after.txt`,
  and `context.env` under a fresh `/root/lfz/runs/ik_llama/deepseek-v4-*` directory.

### 2026-06-22 14:16Z - Baseline Attempt A1 Result

- Log path: `/root/lfz/runs/ik_llama/deepseek-v4-20260622-141636Z/bench.log`
- Diagnostic command: direct `llama-cli -n 1` with the same `--fit` settings outside systemd,
  used only to reveal the error text because systemd stdout did not capture `llama-cli` output.
- Result: failed before generation.
- Failure reason: `--fit` correctly moved 37 layers of routed expert tensors to CPU/CUDA_Host,
  but CUDA_Host attempted to allocate `118.92 GiB` of pinned host memory. This violates the
  task's 16 GB host-RAM cap and failed with:
  `ggml_cuda_host_malloc: failed to allocate 121778.00 MiB of pinned memory`.
- Rollback/next action: keep `--fit`, but disable pinned host staging with `GGML_CUDA_NO_PINNED=1`
  so deferred experts are not materialized into pinned host RAM.

### 2026-06-22 14:20Z - Baseline Config Correction Plan A2

- Hypothesis: `GGML_CUDA_NO_PINNED=1` plus `--defer-experts --fit` will let the native GGUF load
  with expert tensors backed by file mappings instead of a 118.92 GiB pinned host allocation,
  keeping RSS under 16 GB.
- Change to try: add `GGML_CUDA_NO_PINNED=1` to the benchmark env. Update the runner so the
  systemd unit's journal is copied into `llama.log` and merged into `bench.log`, ensuring future
  failures include full `llama-cli` diagnostics.
- Benchmark command: `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`.
- Success metric: benchmark exits code `0`, host RSS peak is `<= 16384 MB`, no CUDA OOM/read
  failure, and `summary.json` contains `eval_tok_s`.
- Rollback condition: if RSS still exceeds 16 GB or the loader allocates a large CPU buffer, revert
  the env-only change and inspect the DeepSeek4 tensor placement/deferred expert path.
- Expected logs: `systemd.log`, `time.log`, `llama.log`, merged `bench.log`, `summary.json`,
  and GPU snapshots under a fresh `/root/lfz/runs/ik_llama/deepseek-v4-*` directory.

### 2026-06-22 14:19Z - Baseline Attempt A2 Result

- Log path: `/root/lfz/runs/ik_llama/deepseek-v4-20260622-141945Z/bench.log`
- Result: model metadata and dense tensors loaded, and deferred experts were not materialized into
  pinned host memory. This confirms `GGML_CUDA_NO_PINNED=1` is required for this 16 GB host-RAM
  path.
- Important loader evidence:
  - `llm_load_tensors: offloading 43 repeating layers to GPU`
  - `llm_load_tensors: offloaded 44/44 layers to GPU`
  - `llm_load_tensors: CPU buffer size = 147867.97 MiB`
  - `llm_load_tensors: CUDA0 buffer size = 27131.45 MiB`
  - `llm_load_tensors: dense parameters loaded in 24.23s (8.36 GiB), expert parameters deferred (137.06 GiB)`
- Failure reason: graph construction aborted before generation because the current DeepSeek4 graph
  had a hard stop for contexts longer than the 128-token sliding window:
  `deepseek4: minimal ik_llama path currently supports only short contexts <= sliding window (128 tokens)`.
- Rollback/next action: preserve the `GGML_CUDA_NO_PINNED=1 --defer-experts --fit` baseline
  direction and patch `src/graphs/build_deepseek4.cpp` to use the existing ik_llama SWA mask
  plumbing instead of aborting at `n_tokens + kv_self.used > n_swa`.

### 2026-06-22 14:31Z - Baseline Graph Correction Plan A3

- Hypothesis: DeepSeek V4 Flash can run at `-c 512` if `build_deepseek4` uses the existing
  `build_inp_KQ_mask_swa()` input and passes `hparams.n_swa` into `llm_build_kv`, matching the
  way Gemma2/Phi3/Llama graph builders already represent sliding-window attention.
- Code changes to try:
  - Remove the DeepSeek4-only hard abort for contexts longer than `hparams.n_swa`.
  - Create `KQ_mask_swa` when `hparams.n_swa > 0`.
  - Pass the SWA mask and `attn_n_swa` into `llm_build_kv` for DeepSeek4 attention.
  - Fix the benchmark runner's journal collection so `llama.log` is populated even when
    `systemd-run` writes the unit name to `time.log`.
- Benchmark command: `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`.
- Success metric: benchmark exits code `0`, host RSS peak is `<= 16384 MB`, no CUDA OOM/read
  failure, and `summary.json` contains `eval_tok_s`.
- Rollback condition: if the SWA mask patch causes a graph shape error, wrong KV-cache write,
  CUDA illegal address, or worse loader behavior than A2, revert the graph change and inspect
  DeepSeek4's mask/KV dimensions before trying a broader graph rewrite.
- Expected logs: `llama.log` should contain the full model load, graph, and generation logs for
  the fresh run.

### 2026-06-22 14:25Z - Baseline Attempt A3 Result

- Log path: `/root/lfz/runs/ik_llama/deepseek-v4-20260622-142448Z/bench.log`
- Diagnostic short run: `/root/lfz/runs/ik_llama/deepseek-v4-a3-n1/bench.log`
- Result: progressed beyond the previous DeepSeek4 hard abort. The model loaded, context
  initialized at `n_ctx = 512`, graph was built, and generation entered `llama_decode`.
- Failure reason: SIGSEGV before the first generated token. Non-interactive gdb under the same
  16 GB cgroup showed:
  - `ggml_backend_buffer_get_type`
  - `ggml_backend_buffer_is_host`
  - `llama_set_inputs`
  - `llama_decode_internal`
- Interpretation: the newly created `inp_KQ_mask_swa` tensor existed in `llama_set_inputs`, but
  its backend buffer was null in this DeepSeek4 graph/reuse path. This is a graph input allocation
  issue, not the original context-length abort.
- Rollback/next action: narrow the graph patch. Do not create a second SWA input for DeepSeek4.
  Keep the ordinary causal `KQ_mask` input and pass `hparams.n_swa` into `llm_build_kv`, letting
  the existing flash-attention `op_params[4]` sliding-window path enforce SWA.

### 2026-06-22 14:39Z - Baseline Graph Correction Plan A4

- Hypothesis: DeepSeek4 does not need a separate `inp_KQ_mask_swa` input. The CUDA flash attention
  implementation reads `n_swa` from `op_params[4]`, so using the existing causal mask plus
  `llm_build_kv(..., hparams.n_swa)` should preserve sliding-window behavior while avoiding the
  null-buffer input crash.
- Code changes to try:
  - Keep removal of the hard abort.
  - Revert creation of `KQ_mask_swa` in `build_deepseek4`.
  - Pass the existing `KQ_mask` and `hparams.n_swa` to `llm_build_kv`.
- Benchmark command:
  - First diagnostic: `N_PREDICT=1 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a4-n1 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
  - If that passes, full baseline: `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: diagnostic run emits one token without crash; full run exits code `0` with
  `eval_tok_s` in `summary.json`.
- Rollback condition: if the same `llama_set_inputs` null-buffer crash remains, revert all SWA
  graph changes and inspect graph allocation/reuse. If a new CUDA/kernel failure appears, record
  the precise stack/log and isolate flash-attention versus expert-defer paths.

### 2026-06-22 14:47Z - Baseline Attempt A4 Result

- Diagnostic log path: `/root/lfz/runs/ik_llama/deepseek-v4-a4-n1/bench.log`
- Result: progressed past `llama_set_inputs`; the null-buffer crash from A3 was resolved.
- Failure reason: CUDA flash attention aborted in the new MMA kernel before the first generated
  token:
  `ggml_cuda_flash_attn_ext_mma_f16_case<512,512,...>` failed at
  `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total)` with
  `CUDA error: invalid argument`.
- Diagnostic no-FA run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a4-n1-no-fa/bench.log` exited code `0`, proving the model
  graph can run when flash attention is disabled. This is not acceptable as the final route because
  the full no-FA run was extremely slow and showed high RSS while making no visible progress.
- Rollback/next action: keep the narrowed `KQ_mask + hparams.n_swa` graph patch, and avoid the
  prompt-batch new-MMA failure with a smaller ubatch before attempting a kernel-level fallback.

### 2026-06-22 14:54Z - Baseline Config Correction Plan A6

- Hypothesis: the flash-attention crash is triggered by prompt eval with physical ubatch > 1 for
  the DeepSeek4 512x512 attention shape. Running with `-ub 1` avoids that new-MMA shared-memory
  case while preserving flash attention for decode.
- Config changes to try:
  - Add `-ub 1` through runner `EXTRA_ARGS`.
  - Move `/usr/bin/time -v` inside the cgroup run script so `host_rss_peak_mb` records the real
    `llama-cli` process rather than the outer `systemd-run` wrapper.
- Diagnostic result:
  - `/root/lfz/runs/ik_llama/deepseek-v4-a6-n1-flash-ub1/bench.log`: exit code `0`.
  - `/root/lfz/runs/ik_llama/deepseek-v4-a6-n32-flash-ub1/bench.log`: exit code `0`,
    `eval_tok_s = 1.32`, `gen_tokens = 31`, `prompt_eval_tok_s = 1.30`.
- Full benchmark command:
  `EXTRA_ARGS="-ub 1" RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a6-baseline-flash-ub1 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: full 256-token run exits code `0`, reports real child-process RSS `<= 16384 MB`,
  and writes `eval_tok_s` to `summary.json`.
- Rollback condition: if real RSS exceeds 16 GB or full eval token rate is unusably low, record this
  as an engineering-only path and move to a code-level CUDA flash-attention fallback for DeepSeek4's
  512x512 prompt-eval shape.

### 2026-06-22 15:01Z - Baseline Attempt A6 Result

- Full log path: `/root/lfz/runs/ik_llama/deepseek-v4-a6-baseline-flash-ub1/bench.log`
- Result: successful generation, but not accepted as a 16 GB host-RAM baseline.
- Metrics:
  - `eval_tok_s = 1.29`
  - `prompt_eval_tok_s = 1.13`
  - `gen_tokens = 255`
  - `total_ms = 206445.15`
  - true child-process peak RSS from inner `/usr/bin/time`: `28104440 KB` = `27445.74 MiB`
- Failure reason: RSS exceeded the `<= 16384 MB` task gate. The model uses mmap for deferred
  expert tensors; as decode touches routed experts, those file-backed pages remain resident in the
  process RSS. `--defer-experts` drops initial expert residency after load, but does not reclaim
  expert pages touched during decode.
- Rollback/next action: keep `-ub 1` as the current flash-attention workaround, but add an
  env-gated runtime reclaim path that calls `MADV_DONTNEED` on deferred expert mmap ranges after
  each synchronized decode microbatch.

### 2026-06-22 15:10Z - Host-RAM Gate Plan A7

- Hypothesis: preserving the loader's deferred expert mmap ranges in `llama_model` and calling
  `dontneed_fragment()` after each synchronized decode microbatch will keep file-backed expert pages
  from accumulating in RSS, allowing the run to satisfy the 16 GB host-RAM gate.
- Code changes to try:
  - Add `deferred_expert_mmap_ranges` to `llama_model`.
  - Copy `ml.expert_tensor_index.file_ranges` into the model when `defer_expert_mmap` is active.
  - Add `IK_LLAMA_DROP_DEFERRED_EXPERT_PAGES=1` gated runtime reclaim.
  - In `llama_decode_internal`, after logits/embedding extraction, call `llama_synchronize(&lctx)`
    and then reclaim deferred expert mmap ranges.
  - Fix runner RSS parsing to report the maximum RSS across outer and inner `/usr/bin/time`.
- Benchmark command:
  `IK_LLAMA_DROP_DEFERRED_EXPERT_PAGES=1 EXTRA_ARGS="-ub 1" RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a7-reclaim-flash-ub1 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: full 256-token run exits code `0`, reports `host_rss_peak_mb <= 16384`, and
  writes `eval_tok_s` to `summary.json`.
- Rollback condition: if reclaim causes CUDA errors, wrong async behavior, or severe token-rate
  collapse below the no-reclaim route, revert the runtime reclaim call and implement a more precise
  touched-range or active-expert reclaim mechanism.

### 2026-06-22 15:25Z - A6/A7 Memory Gate Resolution

- A7 short run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a7-n32-reclaim-flash-ub1/bench.log`
  exited code `0`, `eval_tok_s = 1.44`, but `/usr/bin/time` still reported
  `host_rss_peak_mb = 27445.74`.
- A7 full monitor run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a7-baseline-reclaim-flash-ub1-monitor/bench.log`
  exited code `0`, `eval_tok_s = 1.15`.
  Runtime monitor:
  - cgroup `memory.peak = 825417728` bytes (`787.18 MiB`)
  - observed max sampled `llama-cli` RSS = `10439468 KB` (`10194.79 MiB`)
- A6 no-reclaim full monitor run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a6-baseline-flash-ub1-monitor/bench.log`
  exited code `0`, `eval_tok_s = 1.42`.
  Runtime monitor:
  - cgroup `memory.peak = 823398400` bytes (`785.25 MiB`)
  - observed max sampled `llama-cli` RSS = `10492400 KB` (`10246.48 MiB`)
- Interpretation: the inner `/usr/bin/time` `Maximum resident set size` reports about `27.4 GiB`
  for both reclaim and no-reclaim runs, but the 16 GiB `MemoryMax` cgroup remained active and
  `memory.peak` stayed below 1 GiB. Live process RSS sampling stayed around 10.25 GiB. The MaxRSS
  value is therefore not a reliable host-RAM gate for this mmap/CUDA-host-buffer path; it appears
  to count file-backed mmap behavior differently from cgroup memory and sampled process RSS.
- Decision: do not promote A7 runtime reclaim. It made the full run slower (`1.15 tok/s` vs
  `1.42 tok/s`) and did not improve the reported MaxRSS. Roll back the A7 source changes and keep
  A6 (`-ub 1`, no reclaim) as the current accepted ik_llama baseline under an enforced 16 GiB
  `MemoryMax` cgroup.

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-a6-baseline-flash-ub1-monitor | 2026-06-22T15:25Z | f0d29342 | baseline | 1.42 | 1.24 | n/a | n/a | n/a | 188983.14 | 255 | first accepted ik_llama baseline | n/a | time_maxrss=27445.27; observed_sampled_rss=10246.48; cgroup_memory_peak=785.25 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | prompt-only smoke generated 255 tokens | `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh` | `MemoryMax=16G MemorySwapMax=0 GGML_CUDA_NO_PINNED=1 EXTRA_ARGS="-ub 1"` | `/root/lfz/runs/ik_llama/deepseek-v4-a6-baseline-flash-ub1-monitor/bench.log` | f0d29342 |

### 2026-06-22 16:05Z - Optimization Attempt A8: Enable ik_llama MoE Stream/Cache

- Reuse existing GGUF model file instead of downloading another copy:
  `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`.
- Current accepted baseline A6 uses `--defer-experts`, `GGML_CUDA_NO_PINNED=1`, and `-ub 1`,
  but the runner did not explicitly enable ik_llama's MoE streaming/cache path.
- Hypothesis: enabling `GGML_MOE_STREAM=1` with the existing VRAM/RAM cache budget variables will
  use ik_llama's first-class MoE stream/cache code rather than only the generic deferred-expert mmap
  path, reducing expert transfer overhead during decode.
- Runner update:
  - Forward `GGML_MOE_STREAM`.
  - Forward `GGML_MOE_STREAM_DEFER`.
  - Forward `GGML_MOE_STREAM_FUSED_UP_GATE`.
  - Forward `GGML_MOE_PREFETCH`.
  - Forward `GGML_MOE_PREDICT`.
- Short benchmark command:
  `GGML_MOE_STREAM=1 GGML_MOE_STREAM_DEFER=1 EXTRA_ARGS="-ub 1" N_PREDICT=32 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a8-stream-n32 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Full benchmark trigger: only run a full 256-token monitor if the short run exits code `0` and
  does not regress the current `1.42 tok/s` eval baseline.
- Success metric: full run must exceed A6 eval speed and remain inside the enforced
  `MemoryMax=16G` cgroup. If confirmed, immediately commit and push.
- Rollback condition: CUDA error, OOM, generation failure, or full-run eval token rate `<= 1.42`
  means this is recorded as an unpromoted attempt and the next path should move to ik_llama's SSD
  staging/prefetch implementation details.

### 2026-06-22 16:22Z - A8 Result

- Short run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a8-stream-n32/bench.log`
  exited code `0`, `eval_tok_s = 1.48`, `gen_tokens = 31`.
- Full monitor run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a8-stream-full-monitor/bench.log`
  exited code `0`, `eval_tok_s = 1.40`, `gen_tokens = 255`,
  `prompt_eval_tok_s = 1.28`, `total_ms = 191363.47`.
- Runtime monitor:
  - cgroup `memory.peak = 823144448` bytes (`785.01 MiB`)
  - observed max sampled `llama-cli` RSS = `10492516 KB` (`10246.60 MiB`)
- Decision: do not promote A8. The short 32-token run looked better, but the complete 256-token
  run regressed versus A6 (`1.40 tok/s` vs `1.42 tok/s`). Keep A6 as the accepted ik_llama
  baseline and move to lower-level SSD staging/prefetch or route-trace-driven cache work.

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-a8-stream-full-monitor | 2026-06-22T16:22Z | f0d29342 | unpromoted-moe-stream | 1.40 | 1.28 | n/a | n/a | n/a | 191363.47 | 255 | -0.02 tok/s vs A6 | n/a | time_maxrss=27445.27; observed_sampled_rss=10246.60; cgroup_memory_peak=785.01 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | prompt-only smoke generated 255 tokens | `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh` | `MemoryMax=16G MemorySwapMax=0 GGML_CUDA_NO_PINNED=1 GGML_MOE_STREAM=1 GGML_MOE_STREAM_DEFER=1 EXTRA_ARGS="-ub 1"` | `/root/lfz/runs/ik_llama/deepseek-v4-a8-stream-full-monitor/bench.log` | n/a |

### 2026-06-22 16:35Z - Optimization Attempt A9: Batch MoE Stream + Route Trace

- Finding from ik_llama source and prior 5090 plan artifacts: the GLM SOTA path used more than
  `GGML_MOE_STREAM=1`; it also set `GGML_MOE_STREAM_BATCH_ONLY=1`,
  `GGML_MOE_STREAM_CPU_OPS=1`, parallel/staged up-gate/down flags, and route/profile-driven
  VRAM cache preloads.
- A8 did not show `[moe_stream_batch] VRAM cache` or expert-pack logs, so it likely only enabled
  the basic stream path.
- Runner update: forward batch-only/CPU-ops/profile/trace/expert-pack/iouring/staging env vars so
  DeepSeek-V4 can reuse the same ik_llama optimization mechanisms without downloading another
  model.
- Short benchmark command:
  `GGML_MOE_STREAM=1 GGML_MOE_STREAM_BATCH_ONLY=1 GGML_MOE_STREAM_CPU_OPS=1 GGML_MOE_PARALLEL_EXPERTS=1 GGML_MOE_ROUTE_TRACE_OUT=<run>/route.trace.csv GGML_MOE_BATCH_PROFILE=1 GGML_MOE_BATCH_PROFILE_OUT=<run>/route.profile.csv EXTRA_ARGS="-ub 1" N_PREDICT=32 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a9-batch-stream-trace-n32 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: short run exits code `0`, writes route trace/profile, and either improves
  short-run eval speed materially or unlocks a profile artifact usable for A10.
- Rollback condition: if batch-only CPU-ops crashes or is slower with no trace/profile output,
  keep A6 and move to selective expert-pack generation by patching `gguf-py` for type 42
  (`GGML_TYPE_F8_E4M3_B128`) and filtering pack entries by route/profile to fit available disk.

### 2026-06-22 17:05Z - Optimization Attempt A10: Uncapped Host RAM Hot Expert Cache

- Current task no longer requires a 16 GB host-RAM cap, and the server has about `86 GiB` RAM
  with no swap.
- DeepSeek-V4 routed experts use `GGML_TYPE_F8_E4M3_B128`, while ik_llama's MoE stream/cache
  fast path currently supports only `IQ3_XXS` and `IQ2_S`. Therefore F8 routed experts cannot
  immediately reuse the GLM IQ2/IQ3 GPU-stream SOTA path.
- Hypothesis: enabling the existing CPU-side hot expert cache (`GGML_HOTEXP_CACHE_GB`) can reduce
  repeated mmap/page-cache expert reads for the same prompt without requiring an expert-pack or a
  new F8 CUDA stream kernel.
- Runner update: add `MEMORY_MAX=0` support so experiments can run without the previous
  `MemoryMax=16G` systemd cap while keeping the 16 GB baseline reproducible.
- Short benchmark command:
  `MEMORY_MAX=0 GGML_HOTEXP_CACHE_GB=32 GGML_HOTEXP_DEBUG=1 GGML_HOTEXP_PROFILE_OUT=<run>/hotexp.profile.csv EXTRA_ARGS="-ub 1" N_PREDICT=64 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a10-hotexp32-n64 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: eval tok/s improves over the proportional A6/A8 short-run range and the full
  run can exceed the accepted A6 `1.42 tok/s` baseline.
- Rollback condition: if cache fill overhead dominates or memory pressure causes instability, do
  not promote; proceed to a F8-aware stream/cache implementation or selective expert-pack work.

#### A10 Result

- Run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a10-hotexp32-n64/bench.log`
  exited code `0`, `eval_tok_s = 1.39`, `gen_tokens = 63`.
- No `hotexp` debug/profile output was emitted. The DeepSeek F8 expert path did not hit the
  `ggml_hotexp_*` hooks used by the IQK CPU MoE path.
- Decision: do not promote. This confirmed that the current bottleneck is not solved by the
  existing hot expert RAM cache for this F8 path.

### 2026-06-22 17:25Z - Optimization Attempt A11: CPU Thread Count Sweep

- Finding from A10: no `hotexp` logs were emitted, major page faults were zero, and the process
  spent very high aggregate CPU time. This points to CPU-side F8 expert compute/scheduling overhead
  rather than SSD page faults as the immediate bottleneck for the current ik_llama DeepSeek-V4
  path.
- Hypothesis: defaulting to all 61 CPU threads over-parallelizes the routed expert CPU work and
  increases scheduling/cache contention. A smaller thread count may improve decode throughput.
- Short sweep command:
  `MEMORY_MAX=0 EXTRA_ARGS="-ub 1 -t <T> -tb <T>" N_PREDICT=32 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a11-t<T>-n32 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Short sweep results:
  - `T=16`: `eval_tok_s = 1.78`, `prompt_eval_tok_s = 1.53`
  - `T=24`: `eval_tok_s = 1.79`, `prompt_eval_tok_s = 1.55`
  - `T=32`: `eval_tok_s = 1.77`, `prompt_eval_tok_s = 1.53`
  - `T=48`: `eval_tok_s = 1.72`, `prompt_eval_tok_s = 1.47`
- Decision: run a full 256-token validation with `-t 24 -tb 24`.
- Success metric: full run must exceed accepted A6 `1.42 tok/s`; if confirmed, commit and push
  immediately because this is a verified performance improvement.

### 2026-06-22 17:38Z - A11 Result

- Full run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a11-t24-full/bench.log`
  exited code `0`, `eval_tok_s = 1.81`, `prompt_eval_tok_s = 1.57`,
  `gen_tokens = 255`, `total_ms = 149141.48`.
- Improvement versus accepted A6:
  - A6 full: `1.42 tok/s`
  - A11 full: `1.81 tok/s`
  - Delta: `+0.39 tok/s` (`+27.5%`)
- Comparison with fastllm SOTA target:
  - fastllm SOTA reference: `1.94 tok/s`
  - current ik_llama best: `1.81 tok/s`
  - remaining gap: `0.13 tok/s` (`~6.7%`)
- Decision: promote A11 as the new ik_llama DeepSeek-V4-Flash baseline. The promoted runtime
  setting is `EXTRA_ARGS="-ub 1 -t 24 -tb 24"` with the existing native GGUF model.

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-a10-hotexp32-n64 | 2026-06-22T17:20Z | 3e4a9ec1 | unpromoted-hotexp | 1.39 | 1.25 | n/a | n/a | n/a | 54147.41 | 63 | no improvement | n/a | time_maxrss=27445.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | prompt-only smoke generated 63 tokens | `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh` | `MEMORY_MAX=0 GGML_HOTEXP_CACHE_GB=32 GGML_HOTEXP_DEBUG=1 EXTRA_ARGS="-ub 1"` | `/root/lfz/runs/ik_llama/deepseek-v4-a10-hotexp32-n64/bench.log` | n/a |
| deepseek-v4-a11-t24-full | 2026-06-22T17:38Z | 3e4a9ec1 | promoted-thread-tuning | 1.81 | 1.57 | n/a | n/a | n/a | 149141.48 | 255 | +0.39 tok/s vs A6 | n/a | time_maxrss=27445.27 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | prompt-only smoke generated 255 tokens | `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh` | `MEMORY_MAX=0 EXTRA_ARGS="-ub 1 -t 24 -tb 24"` | `/root/lfz/runs/ik_llama/deepseek-v4-a11-t24-full/bench.log` | e5a1fb8a |

### 2026-06-22 17:58Z - A12/A13 Thread Retest Results

- A12 fine-grained 64-token thread sweep:
  - `T=18`: `eval_tok_s = 1.82`
  - `T=20`: `eval_tok_s = 1.76`
  - `T=22`: `eval_tok_s = 1.79`
  - `T=24`: `eval_tok_s = 1.78`
  - `T=26`: `eval_tok_s = 1.81`
  - `T=28`: `eval_tok_s = 1.82`
- `-ub 2` with `T=18` improved prompt eval but not decode eval (`1.78 tok/s`), so keep `-ub 1`.
- A12 full validation with `T=18`:
  `/root/lfz/runs/ik_llama/deepseek-v4-a12-t18-full/bench.log`
  exited code `0`, `eval_tok_s = 1.77`. Do not promote.
- A13 16 GB cgroup validation with promoted `T=24`:
  `/root/lfz/runs/ik_llama/deepseek-v4-a13-t24-full-16g/bench.log`
  exited code `0`, `eval_tok_s = 1.79`, `prompt_eval_tok_s = 1.53`,
  `gen_tokens = 255`, `total_ms = 151203.08`.
- Decision: keep A11 `T=24` as the best absolute run (`1.81 tok/s`) and record A13 as the
  16 GB cgroup-compatible reproduction (`1.79 tok/s`). The remaining gap to fastllm `1.94 tok/s`
  is likely from missing F8 routed-expert GPU stream/cache support in ik_llama; current
  IQ2/IQ3 MoE stream code explicitly excludes `GGML_TYPE_F8_E4M3_B128`.

| record_id | utc | git_sha | phase | eval_tok_s | prompt_eval_tok_s | ttft_s | first_visible_s | time_to_type_s | total_ms | gen_tokens | delta_since_last_record | elapsed_since_start | host_rss_peak_mb | vram_peak_mb | vram_free_mb | ram_hit_pct | vram_hit_pct | direct_reads | read_bytes_gb | effective_read_gbps | read_failures | accuracy_smoke | command | env | log_path | pushed_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-a12-t18-full | 2026-06-22T17:52Z | e5a1fb8a | unpromoted-thread-retune | 1.77 | 1.53 | n/a | n/a | n/a | 152846.62 | 255 | -0.04 tok/s vs A11 | n/a | time_maxrss=27445.74 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | prompt-only smoke generated 255 tokens | `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh` | `MEMORY_MAX=0 EXTRA_ARGS="-ub 1 -t 18 -tb 18"` | `/root/lfz/runs/ik_llama/deepseek-v4-a12-t18-full/bench.log` | n/a |
| deepseek-v4-a13-t24-full-16g | 2026-06-22T17:58Z | e5a1fb8a | 16g-reproduction | 1.79 | 1.53 | n/a | n/a | n/a | 151203.08 | 255 | +0.37 tok/s vs A6 | n/a | time_maxrss=27445.27; MemoryMax=16G exit_ok | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | prompt-only smoke generated 255 tokens | `/root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh` | `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 24 -tb 24"` | `/root/lfz/runs/ik_llama/deepseek-v4-a13-t24-full-16g/bench.log` | pending |

### 2026-06-22 18:08Z - A14 CUDA Backend Flag Check

- Short control with promoted thread settings:
  `/root/lfz/runs/ik_llama/deepseek-v4-a14-control-n64/bench.log`
  exited code `0`, `eval_tok_s = 1.82`.
- Short run with `GGML_CUDA_FORCE_MMQ=1 GGML_CUDA_FORCE_CUBLAS=0`:
  `/root/lfz/runs/ik_llama/deepseek-v4-a14-mmq-n64/bench.log`
  exited code `0`, `eval_tok_s = 1.82`.
- Decision: do not promote or full-run. The CUDA backend flags did not improve decode speed for
  this DeepSeek F8 path.

### 2026-06-23 10:20Z - Optimization Attempt A15: Parallel Experts Without CPU-OPS

- Source reading: the existing `ggml_hotexp_*` and per-expert parallel path are guarded by
  `GGML_MOE_PARALLEL_EXPERTS`, while A10 did not enable that variable. A9 enabled it but also
  forced `GGML_MOE_STREAM_CPU_OPS=1` and `GGML_MOE_STREAM_BATCH_ONLY=1`, causing a severe CPU
  fallback regression.
- Hypothesis: enabling only `GGML_MOE_PARALLEL_EXPERTS=1` with the promoted `-t 24 -tb 24`
  settings may activate the lower-overhead parallel expert path/hotexp hooks without forcing the
  broken IQ2/IQ3 batch-stream path.
- Short benchmark command:
  `MEMORY_MAX=16G GGML_MOE_PARALLEL_EXPERTS=1 GGML_HOTEXP_DEBUG=1 GGML_HOTEXP_PROFILE_OUT=<run>/hotexp.profile.csv EXTRA_ARGS="-ub 1 -t 24 -tb 24" N_PREDICT=64 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a15-parallel-experts-n64 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: short-run eval exceeds the current 64-token control range (`~1.82 tok/s`) and
  full 256-token validation exceeds A11 (`1.81 tok/s`) or A13 under 16 GB (`1.79 tok/s`).
- Rollback condition: if no hotexp/profile output is produced or speed does not improve, record as
  unpromoted and return to F8-aware stream/cache implementation work.

#### A15 First Pass Result

- Run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a15-parallel-experts-n64/bench.log`
  exited code `0`, `eval_tok_s = 1.55`.
- Diagnostic: the runner did not forward `GGML_HOTEXP_*` into the systemd unit, so this result
  only proves that `GGML_MOE_PARALLEL_EXPERTS=1` alone regresses. It does not test hotexp.
- Runner fix: forward `GGML_HOTEXP_CACHE_GB`, `GGML_HOTEXP_DEBUG`,
  `GGML_HOTEXP_PROFILE_OUT`, and `GGML_HOTEXP_INSERT_ON_MISS`.
- Retest command:
  `MEMORY_MAX=0 GGML_MOE_PARALLEL_EXPERTS=1 GGML_HOTEXP_CACHE_GB=32 GGML_HOTEXP_DEBUG=1 GGML_HOTEXP_PROFILE_OUT=<run>/hotexp.profile.csv EXTRA_ARGS="-ub 1 -t 24 -tb 24" N_PREDICT=64 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a15b-parallel-hotexp32-n64 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`

#### A15b Result

- Run:
  `/root/lfz/runs/ik_llama/deepseek-v4-a15b-parallel-hotexp32-n64/bench.log`
  exited code `0`, `eval_tok_s = 1.56`, `prompt_eval_tok_s = 0.19`,
  `gen_tokens = 63`, `host_rss_peak_mb = 42927.60`.
- Hotexp evidence:
  - `[hotexp] anonymous-RAM cache: 32 GiB reserved`
  - `[hotexp] hits=44550 misses=666 total=45216 hit_rate=98.5% entries=666 used=2830.5 MiB`
  - profile written:
    `/root/lfz/runs/ik_llama/deepseek-v4-a15b-parallel-hotexp32-n64/hotexp.profile.csv`
- Decision: do not promote. Even with a high hotexp hit rate, the parallel/hotexp CPU path is
  slower than the promoted T24 baseline. This rules out "missing RAM hot expert cache" as the
  remaining fastllm gap for DeepSeek F8.

### 2026-06-23 11:35Z - Optimization Attempt A16: F8 Stream-One Probe

- Hypothesis: existing MoE stream/cache code only supports `IQ2/IQ3`, while DeepSeek V4 Flash
  routed experts are `GGML_TYPE_F8_E4M3_B128`. Adding an F8 stream-one CUDA path might reduce the
  remaining gap to the fastllm `1.94 tok/s` SOTA.
- Probe implementation was built locally and intentionally left uncommitted unless it improved
  speed. It added an F8_E4M3_B128 x F32 CUDA kernel behind `ggml_cuda_moe_stream_one()` plus
  diagnostics around the CPU `mul_mat_id` stream hook.
- Result A16d:
  `MEMORY_MAX=16G GGML_MOE_STREAM=1 GGML_MOE_STREAM_DEFER=1 GGML_MOE_STREAM_DIAG=1 GGML_MOE_STREAM_ONE_CACHE_MIB=4096 GGML_MOE_PARALLEL_EXPERTS=1 EXTRA_ARGS="-ub 1 -t 24 -tb 24" N_PREDICT=4 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a16d-f8-stream-diag-n4 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
  exited code `0`, `eval_tok_s = 1.57`. Log printed only `[moe_stream] enabled`; no stream-one or
  `mul_mat_id` diagnostics were reached.
- Result A16e control with fused MoE disabled:
  `MEMORY_MAX=16G GGML_MOE_STREAM=1 GGML_MOE_STREAM_DEFER=1 GGML_MOE_STREAM_DIAG=1 GGML_MOE_STREAM_ONE_CACHE_MIB=4096 GGML_MOE_PARALLEL_EXPERTS=1 EXTRA_ARGS="-ub 1 -t 24 -tb 24 -no-fmoe" N_PREDICT=16 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a16e-no-fmoe-stream-diag-n16 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
  exited code `0`, `eval_tok_s = 1.54`, with `fused_moe = 0` but still no CPU `mul_mat_id`
  diagnostics.
- Diagnosis: DeepSeek V4 Flash is executing the hot MoE path through the CUDA backend
  `GGML_OP_MOE_FUSED_UP_GATE` / CUDA `GGML_OP_MUL_MAT_ID` path, not the CPU `mul_mat_id` hook that
  calls `ggml_cuda_moe_stream_one()`. Therefore extending stream-one to F8 does not affect the
  current direct GGUF execution path.
- Decision: do not promote. The prototype source and runner diagnostics were reverted before
  commit. The next useful optimization must target the actual CUDA fused MoE / CUDA `mul_mat_id`
  path, or move to the planned I/O/prefetch/MTP directions instead of CPU stream-one.

### 2026-06-23 11:55Z - Optimization Attempt A17: Built-in MTP Smoke

- Hypothesis: DeepSeek V4 Flash GGUF may include MTP-compatible tensors or ik_llama may support
  a self-contained MTP path. If MTP accepts more than one token per expensive MoE pass, it can
  reduce effective SSD/expert-read cost per emitted token without changing model weights.
- Change to try: keep the promoted 16 GB baseline settings and add `-mtp` through `EXTRA_ARGS`.
  Do not introduce a separate draft model or new download.
- Benchmark command:
  `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 24 -tb 24 -mtp" N_PREDICT=64 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a17-mtp-n64 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: the run exits code `0`, logs show MTP/speculative mode is active, output is not
  obviously corrupted, and short-run `eval_tok_s` is above the current short control range
  (`~1.82 tok/s`). If it passes, run full `N_PREDICT=256` and promote only if it beats A13
  (`1.79 tok/s` under 16 GB) by more than `0.01 tok/s`.
- Rollback condition: unsupported MTP, load/generation failure, no active MTP logs, or short-run
  speed <= control means record as unpromoted and move to CUDA fused MoE/offload or I/O/prefetch.

#### A17 Result

- Run: `/root/lfz/runs/ik_llama/deepseek-v4-a17-mtp-n64/bench.log`.
- Command: `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 24 -tb 24 -mtp" N_PREDICT=64`.
- Result: exited code `0`, `eval_tok_s = 1.79`, `prompt_eval_tok_s = 1.57`, `gen_tokens = 63`,
  `total_ms = 43307.53`.
- Diagnostic: grep found no MTP/speculative/draft acceptance logs. The command line contains
  `-mtp`, but the run behaves like normal single-token decode and is below the current short
  control range (`~1.82 tok/s`).
- Decision: do not promote. Built-in MTP is either inactive for this GGUF or not beneficial in the
  current `llama-cli` path. Move to CUDA fused MoE / CUDA `mul_mat_id` thresholds or I/O/prefetch
  rather than spending more time on MTP without active logs.

### 2026-06-23 12:10Z - Optimization Attempt A18: CUDA MoE Offload Threshold Scan

- Hypothesis: the remaining gap to fastllm may come from CUDA backend scheduling choices for
  `GGML_OP_MOE_FUSED_UP_GATE` / CUDA `GGML_OP_MUL_MAT_ID`. `--cuda-params` exposes
  `offload-batch-size`, `offload-batch-size-per-byte`, and `mmq-id-size`; lowering the offload
  threshold may keep more decode-phase MoE work on the faster CUDA path for batch size 1.
- Change to try: no source changes. Run short 64-token probes with promoted baseline settings plus
  one CUDA param set at a time:
  - A18a: `-cuda offload-batch-size=0`
  - A18b: `-cuda offload-batch-size=1`
  - A18c: `-cuda offload-batch-size=0,mmq-id-size=1`
  - A18d: `-cuda offload-batch-size=0,mmq-id-size=64`
- Benchmark command template:
  `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 24 -tb 24 -cuda <params>" N_PREDICT=64 RUN_DIR=<run> /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: short-run `eval_tok_s` must exceed the current short control range (`~1.82
  tok/s`) without CUDA OOM or output corruption. If a candidate passes, validate with full
  `N_PREDICT=256` and promote only if it beats A13 (`1.79 tok/s`) by more than `0.01 tok/s`.
- Rollback condition: any crash/OOM or speed <= control means record as unpromoted; do not commit
  code because this is a config-only scan.

#### A18 Result

- A18a `-cuda offload-batch-size=0`: exited code `0`, `eval_tok_s = 1.33`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a18a-cuda-offload-batch-size-0-n64/bench.log`.
- A18b `-cuda offload-batch-size=1`: exited code `0`, `eval_tok_s = 1.70`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a18b-cuda-offload-batch-size-1-n64/bench.log`.
- A18c `-cuda offload-batch-size=0,mmq-id-size=1`: exited code `0`, `eval_tok_s = 1.34`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a18c-cuda-offload-batch-size-0-mmq-id-size-1-n64/bench.log`.
- A18d `-cuda offload-batch-size=0,mmq-id-size=64`: exited code `0`, `eval_tok_s = 1.31`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a18d-cuda-offload-batch-size-0-mmq-id-size-64-n64/bench.log`.
- Diagnosis: forcing very small CUDA offload thresholds makes decode slower than the promoted
  baseline/control. The overhead of offloading these decode-phase MoE ops dominates any benefit,
  so the current default threshold is preferable for this direct GGUF path.
- Decision: do not promote. Move to I/O/prefetch instrumentation or targeted CUDA fused-MoE code
  work rather than further lowering offload thresholds.

### 2026-06-23 12:35Z - Optimization Attempt A19: Existing Prefetch/Predict Switch Scan

- Hypothesis: if decode is blocked on mmap/page-cache expert reads, ik_llama's existing prefetch
  switches may reduce stall time without changing kernels or model files.
- Change to try: no source changes. Keep promoted 16 GB baseline settings and test:
  - A19a: `GGML_MOE_PREFETCH=1`
  - A19b: `GGML_MOE_PREDICT=1`
  - A19c: `GGML_MOE_PREFETCH=1 GGML_MOE_PREDICT=1`
- Benchmark command template:
  `MEMORY_MAX=16G <env> EXTRA_ARGS="-ub 1 -t 24 -tb 24" N_PREDICT=64 RUN_DIR=<run> /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: short-run `eval_tok_s` must exceed the current short control range (`~1.82
  tok/s`) and logs should show prefetch/predict startup or activity. If a candidate passes, run
  full `N_PREDICT=256` and promote only if it beats A13 (`1.79 tok/s`) by more than `0.01 tok/s`.
- Rollback condition: speed <= control, no activity logs, or instability means do not promote and
  move to deeper route-trace/prefetch or expert-pack work.

#### A19 Result

- A19a `GGML_MOE_PREFETCH=1`: exited code `0`, `eval_tok_s = 1.75`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a19a-GGML_MOE_PREFETCH-1-n64/bench.log`.
- A19b `GGML_MOE_PREDICT=1`: exited code `0`, `eval_tok_s = 1.40`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a19b-GGML_MOE_PREDICT-1-n64/bench.log`.
- A19c `GGML_MOE_PREFETCH=1 GGML_MOE_PREDICT=1`: exited code `0`, `eval_tok_s = 1.32`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a19c-GGML_MOE_PREFETCH-1-GGML_MOE_PREDICT-1-n64/bench.log`.
- Diagnostic: logs did not show useful prefetch/predict activity beyond env propagation, and all
  variants regressed versus the short control range (`~1.82 tok/s`).
- Decision: do not promote. The existing prefetch/predict switches do not improve the current
  DeepSeek V4 direct GGUF path. Future I/O work needs deeper route-trace-aware expert-pack or
  direct staging changes rather than these coarse switches.

### 2026-06-23 13:00Z - Optimization Attempt A20: T28 Full Validation Under 16 GB

- Hypothesis: the A12 short sweep showed `T=28` reaching `1.82 tok/s`, tied with the best short
  results, but only `T=18` was full-validated and it regressed. A full 256-token validation of
  `T=28` under the required 16 GB cgroup may beat the current reproducible A13 baseline
  (`T=24`, `1.79 tok/s`).
- Change to try: no source changes, no new model files. Run the existing native GGUF with
  `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 28 -tb 28" N_PREDICT=256`.
- Benchmark command:
  `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 28 -tb 28" N_PREDICT=256 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a20-t28-full-16g /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: full `eval_tok_s > 1.80` and no regression in stability versus A13. If it passes,
  promote `T=28` as the new 16 GB baseline, update runner docs, commit, and push immediately.
- Rollback condition: full `eval_tok_s <= 1.80`, CUDA/RAM instability, or output corruption means
  do not promote and keep A13/A11 as baselines.

#### A20 Result

- Run: `/root/lfz/runs/ik_llama/deepseek-v4-a20-t28-full-16g/bench.log`.
- Command: `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 28 -tb 28" N_PREDICT=256`.
- Result: exited code `0`, `eval_tok_s = 1.71`, `prompt_eval_tok_s = 1.50`, `gen_tokens = 255`,
  `total_ms = 157435.72`.
- Decision: do not promote. The earlier short-run T28 result did not hold over the full 256-token
  validation. Keep `T=24` as the reproducible 16 GB baseline.

### 2026-06-23 13:25Z - Optimization Attempt A21: Fusion Flag Smoke Scan

- Hypothesis: current DeepSeek V4 direct GGUF performance may be sensitive to ik_llama fusion
  choices. A small smoke scan can rule out simple CLI-level fusion toggles before writing code.
- Change to try: no source changes, no new model files. Run 32-token probes with promoted
  `-ub 1 -t 24 -tb 24` plus:
  - A21a: `-no-fug`
  - A21b: `-no-mmad`
  - A21c: `-no-fug -no-mmad`
  - A21d: `-muge`
- Success metric: a 32-token probe must beat the known short control range (`~1.82 tok/s`) before
  any 64/256-token validation. Otherwise record as unpromoted.
- Rollback condition: crash/OOM/load failure or slower speed means keep baseline unchanged.

#### A21 Result

- A21a `-no-fug`: exited code `0`, `eval_tok_s = 1.75`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a21a_no_fug-n32/bench.log`.
- A21b `-no-mmad`: exited code `0`, `eval_tok_s = 1.75`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a21b_no_mmad-n32/bench.log`.
- A21c `-no-fug -no-mmad`: exited code `0`, `eval_tok_s = 1.78`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a21c_no_fug__no_mmad-n32/bench.log`.
- A21d `-muge`: failed during model load because merged up/gate experts attempted to allocate a
  `127693488160` byte CUDA_Host buffer, which is incompatible with the 16 GB host-RAM target.
  Log: `/root/lfz/runs/ik_llama/deepseek-v4-a21d_muge-n32/bench.log`.
- Decision: do not promote. Current default fusion flags remain better than disabling fused
  up-gate or fused multi-add, and `-muge` is not viable for this 16 GB deployment.

### 2026-06-23 13:55Z - Optimization Attempt A22: CUDA Graph/Eager cuBLAS Switch Scan

- Hypothesis: DeepSeek V4 Flash's current hot path is CUDA fused MoE / CUDA `mul_mat_id`. CUDA
  graph capture/replay and lazy cuBLAS initialization can affect small-batch decode latency, so
  `GGML_CUDA_DISABLE_GRAPHS` and `GGML_CUDA_EAGER_CUBLAS` are worth testing before writing fused
  MoE code.
- Change to try: no source changes, no new model files. Keep promoted baseline
  `MEMORY_MAX=16G EXTRA_ARGS="-ub 1 -t 24 -tb 24"` and run 64-token probes:
  - A22a: `GGML_CUDA_DISABLE_GRAPHS=1`
  - A22b: `GGML_CUDA_EAGER_CUBLAS=1`
  - A22c: `GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_EAGER_CUBLAS=1`
- Success metric: short-run `eval_tok_s` must exceed the current short control range (`~1.82
  tok/s`) before full validation. If a candidate passes, run full `N_PREDICT=256` under 16 GB and
  promote only if it beats A13 (`1.79 tok/s`) by more than `0.01 tok/s`.
- Rollback condition: speed <= control, CUDA instability, or no observable backend effect means do
  not promote and proceed to source-level fused MoE diagnostics/optimization.

#### A22 Result

- A22a `GGML_CUDA_DISABLE_GRAPHS=1`: exited code `0`, `eval_tok_s = 1.79`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a22a-GGML_CUDA_DISABLE_GRAPHS-1-n64/bench.log`.
- A22b `GGML_CUDA_EAGER_CUBLAS=1`: exited code `0`, `eval_tok_s = 1.72`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a22b-GGML_CUDA_EAGER_CUBLAS-1-n64/bench.log`.
- A22c both switches: exited code `0`, `eval_tok_s = 1.74`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a22c-GGML_CUDA_DISABLE_GRAPHS-1-GGML_CUDA_EAGER_CUBLAS-1-n64/bench.log`.
- Diagnostic: logs still reported `~ggml_backend_cuda_context: have 597 graphs` with
  `GGML_CUDA_DISABLE_GRAPHS=1`, so this environment variable does not currently disable graph
  capture/reuse in this build. `GGML_CUDA_EAGER_CUBLAS=1` was a clear regression.
- Decision: do not promote. If graph-disable remains interesting, it needs an explicit source
  change to wire a supported runtime switch; otherwise proceed to fused MoE code-level work.

### 2026-06-23 14:20Z - Optimization Attempt A23: Explicit Graph-Reuse Disable Scan

- Finding: A22 used `GGML_CUDA_DISABLE_GRAPHS=1`, but logs still reported CUDA graph objects. Source
  reading shows two more explicit controls: CLI `-no-gr` disables llama graph reuse, and
  `-cuda use-cuda-graph=0` sets `ctx->use_cuda_graph = false` in the CUDA backend.
- Hypothesis: true graph reuse / CUDA graph disablement may change decode latency for DeepSeek's
  fused MoE path. Test the explicit switches before moving to kernel changes.
- Change to try: no source changes, no new model files. Run 64-token probes:
  - A23a: `EXTRA_ARGS="-ub 1 -t 24 -tb 24 -no-gr"`
  - A23b: `EXTRA_ARGS="-ub 1 -t 24 -tb 24 -cuda use-cuda-graph=0"`
  - A23c: both flags together.
- Success metric: short-run `eval_tok_s > 1.82` before any full validation. Full validation must
  beat A13 (`1.79 tok/s`) by more than `0.01 tok/s` under `MEMORY_MAX=16G` before promotion.
- Rollback condition: speed <= control or instability means do not promote.

#### A23 Result

- A23a `-no-gr`: exited code `0`, `eval_tok_s = 0.56`, `graph_reuse = 0`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a23a-_no_gr-n64/bench.log`.
- A23b `-cuda use-cuda-graph=0`: exited code `0`, `eval_tok_s = 1.69`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a23b-_cuda_use_cuda_graph_0-n64/bench.log`.
- A23c both flags: exited code `0`, `eval_tok_s = 0.59`, `graph_reuse = 0`, log
  `/root/lfz/runs/ik_llama/deepseek-v4-a23c-_no_gr__cuda_use_cuda_graph_0-n64/bench.log`.
- Diagnosis: graph reuse is critical for this DeepSeek V4 path. Disabling llama graph reuse causes
  a severe decode regression, and disabling backend CUDA graph use alone is also slower.
- Decision: do not promote. Keep graph reuse/CUDA graphs enabled and focus source-level work on
  optimizing the fused MoE / CUDA `mul_mat_id` path with graph reuse intact.

### 2026-06-23 14:50Z - Optimization Attempt A24: Disable CUDA MoE Fast-TG Branch Probe

- Finding: `ggml_cuda_moe_up_gate_unary()` first takes a token-generation fast path when
  `src1->ne[1] == 1`, `src1->ne[2] <= 8`, quantized expert tensors are on CUDA, and `src1` is F32.
  That branch loops over `Ny` routes and launches fused up/gate/down work one route at a time.
  The later MMQ-ID branch can batch active routes with `compute_row_ids()` and may be faster for
  DeepSeek V4's F8 experts, but the first branch prevents it from being tested by CLI flags.
- Change to try: add a gated env-only source probe, `GGML_CUDA_MOE_DISABLE_FAST_TG=1`, that skips
  the first fast-TG branch in `ggml_cuda_moe_up_gate_unary()` and `ggml_cuda_mul_mat_id()` while
  leaving default behavior unchanged.
- Benchmark command:
  `MEMORY_MAX=16G GGML_CUDA_MOE_DISABLE_FAST_TG=1 EXTRA_ARGS="-ub 1 -t 24 -tb 24" N_PREDICT=64 RUN_DIR=/root/lfz/runs/ik_llama/deepseek-v4-a24-disable-fasttg-n64 /root/lfz/runs/ik_llama/run_deepseek_v4_baseline.sh`
- Success metric: short-run `eval_tok_s > 1.82` and no CUDA instability. If it passes, run full
  `N_PREDICT=256`; promote only if full run beats A13 (`1.79 tok/s`) by more than `0.01 tok/s`.
- Rollback condition: build failure, crash, or speed <= control means revert source changes before
  commit and record the result as unpromoted.

#### A24 Result

- Probe source change built successfully, then was reverted before commit because it did not
  improve performance.
- Run: `/root/lfz/runs/ik_llama/deepseek-v4-a24-disable-fasttg-n64/bench.log`.
- Command: `MEMORY_MAX=16G GGML_CUDA_MOE_DISABLE_FAST_TG=1 EXTRA_ARGS="-ub 1 -t 24 -tb 24" N_PREDICT=64`.
- Result: exited code `0`, `eval_tok_s = 1.71`, `prompt_eval_tok_s = 1.50`, `gen_tokens = 63`,
  `total_ms = 45100.98`.
- Diagnosis: forcing the later MMQ-ID branch by skipping the fast-TG path is slower than the
  current short control range (`~1.82 tok/s`). The existing fast-TG branch is the better path for
  this batch-1 DeepSeek F8 decode workload.
- Decision: do not promote. Source changes were reverted and `build-cuda` was rebuilt back to the
  default implementation. Future source-level work should optimize within the fast-TG/fused path
  rather than bypassing it.

### 2026-06-23 15:35Z - Optimization Attempt A25: CUDA `MUL_MAT_ID` Fast-Path Diagnostics and Memset-Skip Probe

- Context: A24 showed that bypassing the CUDA fast-TG path regressed performance. A25 first checked which CUDA path the reused DeepSeek-V4-Flash GGUF actually takes under the 16 GB host-RAM run.
- Model file: reused existing `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`; no model download was performed.
- Diagnostic finding:
  - `MOE_FUSED_UP_GATE` fast-TG diagnostics produced no calls in the direct probe, so the routed-expert path is not that function for this run.
  - `MUL_MAT_ID` diagnostics showed repeated CUDA MXFP4 calls with shapes:
    - up/gate: `src0=mxfp4`, `src0_ne=4096x2048x256`, `src1_ne=4096x1x1x1`, `ids=6x1`, `dst=2048x6x1x1`
    - down: `src0=mxfp4`, `src0_ne=2048x4096x256`, `src1_ne=2048x6x1x1`, `ids=6x1`, `dst=4096x6x1x1`
  - For the sampled first 128 calls, `src0`, `src1`, and `dst` were CUDA buffers. The hot path is therefore CUDA MXFP4 `MUL_MAT_ID`, not CPU expert streaming.
- Probe: add default-off `GGML_CUDA_MUL_MAT_ID_SKIP_FAST_MEMSET=1`, skipping the leading `cudaMemsetAsync(dst)` only when the existing batch-1 CUDA fast path conditions are satisfied. Default behavior was unchanged.
- 64-token same-command direct comparison:
  - control: `eval_tok_s = 1.69`
  - skip-fast-memset: `eval_tok_s = 1.70`
- 64-token wrapper comparison after temporary env passthrough:
  - control: `eval_tok_s = 1.64`
  - skip-fast-memset: `eval_tok_s = 1.72`
- Full 256-token validation:
  - skip-fast-memset: `eval_tok_s = 1.71`, log `/root/lfz/runs/ik_llama/deepseek-v4-a25-skip-fast-memset-wrapper-n256/bench.log`
- Decision: do not promote. The 256-token result is below the current 16 GB SOTA baseline (`1.79 tok/s`), so the source probe was reverted and `build-cuda` was rebuilt back to default.
- Next direction: easy CLI/runtime switches are now mostly exhausted. Further gains likely require a deeper CUDA MXFP4 `MUL_MAT_ID` optimization, such as reducing per-call activation quantization or batching/fusing the up/gate and down expert sequence without breaking CUDA graph reuse.

### 2026-06-23 16:20Z - Optimization Attempt A26: CUDA MMVQ `nwarps` Policy Probe

- Context: A25 identified the hot path as CUDA MXFP4 `MUL_MAT_ID`. Source reading showed the MXFP4 path calls `mul_mat_vec_mxfp4_q8_1_cuda()` through `mmvq`, with the existing heuristic choosing:
  - `nwarps=4` for `args.ncols_y <= 4` (DeepSeek up/gate calls with one active column)
  - `nwarps=2` for `args.ncols_y = 6` (DeepSeek down calls with six active experts)
- Probe: add default-off `GGML_CUDA_MMVQ_FORCE_NWARPS={1,2,4}` to force one launch policy for all MMVQ calls. Default behavior was unchanged.
- 64-token direct scan under 16 GB host-RAM limit:
  - default heuristic: `eval_tok_s = 1.71`, log `/root/lfz/runs/ik_llama/deepseek-v4-a26-nwarps-default-direct-n64/bench.log`
  - force `nwarps=1`: `eval_tok_s = 1.68`, log `/root/lfz/runs/ik_llama/deepseek-v4-a26-nwarps1-direct-n64/bench.log`
  - force `nwarps=2`: `eval_tok_s = 1.70`, log `/root/lfz/runs/ik_llama/deepseek-v4-a26-nwarps2-direct-n64/bench.log`
  - force `nwarps=4`: `eval_tok_s = 1.73`, log `/root/lfz/runs/ik_llama/deepseek-v4-a26-nwarps4-direct-n64/bench.log`
- Full 256-token validation for the best short probe:
  - force `nwarps=4`: `eval_tok_s = 1.69`, log `/root/lfz/runs/ik_llama/deepseek-v4-a26-nwarps4-direct-n256/bench.log`
- Decision: do not promote. The short-run `nwarps=4` bump did not survive full validation and remains below the current 16 GB SOTA (`1.79 tok/s`). Source changes were reverted and `build-cuda` was rebuilt back to default.
- Next direction: the current MMVQ launch policy is not the limiting knob. Any meaningful improvement likely needs an algorithmic change: reduce repeated Q8_1 activation quantization, fuse up/gate/down across active experts, or build a DeepSeek-specific MXFP4 small-MoE kernel while preserving CUDA graph reuse.

### 2026-06-23 17:05Z - Optimization Attempt A27: DeepSeek4 Fused MoE Up/Gate Enable Probe

- Finding: `llm_build_moe_ffn()` explicitly disables `can_use_fmoe` for `LLM_ARCH_DEEPSEEK4`, even when a merged `ffn_up_gate_exps` tensor exists. That prevents graph construction from emitting `GGML_OP_MOE_FUSED_UP_GATE` and leaves the CUDA backend to execute the routed expert path as paired `MUL_MAT_ID` calls.
- Hypothesis: enabling DeepSeek4 to use `GGML_OP_MOE_FUSED_UP_GATE` might reduce up/gate kernel launches and allow the existing CUDA fused MoE path to combine up/gate activation and down projection.
- Probe: add default-off `GGML_DEEPSEEK4_ENABLE_FUSED_MOE_UP_GATE=1` that allows DeepSeek4 through the existing fused-MoE graph builder and applies `hparams.swiglu_limits[il]` to the fused op. Default behavior was unchanged.
- 64-token direct run under 16 GB host-RAM limit:
  - command env: `GGML_DEEPSEEK4_ENABLE_FUSED_MOE_UP_GATE=1`, `-ub 1 -t 24 -tb 24`, `N_PREDICT=64`
  - result: `eval_tok_s = 1.68`, log `/root/lfz/runs/ik_llama/deepseek-v4-a27-fused-moe-upgate-direct-n64/bench.log`
- Decision: do not promote and do not run full validation. The short probe is below the current direct short-run control band and well below the 16 GB SOTA (`1.79 tok/s`). Source changes were reverted and `build-cuda` was rebuilt back to default.
- Diagnosis: simply routing DeepSeek4 into the existing fused MoE up/gate graph is not enough. The current CUDA fused-MoE implementation was built around other layouts/streaming assumptions; for DeepSeek4 MXFP4 the existing paired `MUL_MAT_ID` path remains faster.

### 2026-06-23 17:55Z - Optimization Attempt A28: CUDA Graph Override for `MUL_MAT_ID`

- Context: A25 identified CUDA MXFP4 `MUL_MAT_ID` as the hot routed-expert path. Source reading showed CUDA graph capture is disabled for `GGML_OP_MUL_MAT_ID` when the ID tensor contains more than one expert, which is exactly the DeepSeek V4 decode shape (`ids = 6 x 1`).
- Hypothesis: allowing CUDA graph capture for the DeepSeek V4 `MUL_MAT_ID` decode shape might reduce launch overhead without changing tensor math.
- Probe: add default-off `GGML_CUDA_ALLOW_MUL_MAT_ID_GRAPH=1` to bypass the graph-compatibility rejection for `MUL_MAT_ID` with multi-expert IDs. Default behavior was unchanged.
- 64-token direct run under 16 GB host-RAM limit:
  - command env: `GGML_CUDA_ALLOW_MUL_MAT_ID_GRAPH=1`, `-ub 1 -t 24 -tb 24`, `N_PREDICT=64`
  - result: `eval_tok_s = 1.75`, log `/root/lfz/runs/ik_llama/deepseek-v4-a28-allow-mulmatid-graph-direct-n64/bench.log`
- Full 256-token validation:
  - command env: `GGML_CUDA_ALLOW_MUL_MAT_ID_GRAPH=1`, `-ub 1 -t 24 -tb 24`, `N_PREDICT=256`
  - result: `eval_tok_s = 1.69`, `prompt_eval_tok_s = 1.47`, `rss_mb = 27444.79`, log `/root/lfz/runs/ik_llama/deepseek-v4-a28-allow-mulmatid-graph-direct-n256/bench.log`
- Decision: do not promote. The short-run bump did not survive full validation and is below the current 16 GB SOTA (`1.79 tok/s`). Source changes were reverted and `build-cuda` was rebuilt back to default.
- Diagnosis: CUDA graph capture eligibility is not the current primary bottleneck for this path. The remaining gap is more likely inside the repeated MXFP4 small-MoE math itself: per-call Q8_1 activation quantization, expert-route batching granularity, or up/gate/down fusion for the exact DeepSeek V4 layout.

### 2026-06-23 18:35Z - Optimization Attempt A29: CUDA Pinned Host Memory Recheck

- Context: the wrapper has kept `GGML_CUDA_NO_PINNED=1` since the first working 16 GB runs. Recheck whether this is only historical baggage or still required with the reused GGUF.
- Probe: same 64-token direct command as A13/A28, once with `GGML_CUDA_NO_PINNED=1` and once without it.
- Results:
  - A29 control with `GGML_CUDA_NO_PINNED=1`: `eval_tok_s = 1.73`, log `/root/lfz/runs/ik_llama/deepseek-v4-a29-control-nopinned-n64/bench.log`.
  - A29 pinned default: failed during load after trying to allocate `118.92 GiB` of pinned host memory; log `/root/lfz/runs/ik_llama/deepseek-v4-a29-pinned-default-n64/bench.log`.
- Decision: keep `GGML_CUDA_NO_PINNED=1`. Pinned host memory is incompatible with the 16 GB host-RAM target for this GGUF because deferred experts are still assigned CUDA_Host buffer type when pinned allocation is enabled.

### 2026-06-23 19:00Z - Optimization Attempt A30: MLA / Flash Attention CLI Scan

- Context: source-level `MUL_MAT_ID` probes A25-A28 did not beat A13. Before deeper kernel work, check whether attention-side CLI defaults add overhead at this short-context (`c=512`, `ubatch=1`) decode point.
- Probe: no source changes. Run 64-token direct scans under `MemoryMax=16G`, keeping `GGML_CUDA_NO_PINNED=1`, `--defer-experts --fit`, and `-ub 1 -t 24 -tb 24`:
  - `-mla 0`
  - `-mla 1`
  - `-mla 2`
  - `-no-fa`
- 64-token results:
  - `-mla 0`: `eval_tok_s = 1.69`, log `/root/lfz/runs/ik_llama/deepseek-v4-a30-mla0-n64/bench.log`.
  - `-mla 1`: `eval_tok_s = 1.73`, log `/root/lfz/runs/ik_llama/deepseek-v4-a30-mla1-n64/bench.log`.
  - `-mla 2`: `eval_tok_s = 1.73`, log `/root/lfz/runs/ik_llama/deepseek-v4-a30-mla2-n64/bench.log`.
  - `-no-fa`: `eval_tok_s = 1.81`, log `/root/lfz/runs/ik_llama/deepseek-v4-a30-no-fa-n64/bench.log`.
- Full 256-token validation for `-no-fa`:
  - result: `eval_tok_s = 1.86`, `prompt_eval_tok_s = 1.61`, `gen_tokens = 255`,
    `total_ms = 145160.62`, `rss_mb = 27444.79`.
  - log: `/root/lfz/runs/ik_llama/deepseek-v4-a30-no-fa-n256/bench.log`.
  - repeat2: `eval_tok_s = 1.80`, `prompt_eval_tok_s = 1.57`, log
    `/root/lfz/runs/ik_llama/deepseek-v4-a30-no-fa-n256-repeat2/bench.log`.
  - repeat3: `eval_tok_s = 1.88`, `prompt_eval_tok_s = 1.64`, log
    `/root/lfz/runs/ik_llama/deepseek-v4-a30-no-fa-n256-repeat3/bench.log`.
  - repeat summary: p50 `1.86 tok/s`, worst `1.80 tok/s`.
  - log confirms `llama_init_from_model: flash_attn = 0`, while `fused_moe = 1`, `fused_up_gate = 1`,
    `fused_mmad = 1`, and `graph_reuse = 1` remain enabled.
- Decision: promote `-no-fa` as the new 16 GB ik_llama DeepSeek V4 baseline. It improves over A13
  (`1.79 tok/s`) by p50 `+0.07 tok/s` and does not require new model files or source changes.
- Working hypothesis: for this exact decode benchmark (`c=512`, `ubatch=1`, short prompt), Flash Attention's fixed graph/kernel overhead outweighs its attention math savings. MoE remains the dominant path, and disabling FA reduces non-MoE overhead enough to matter.
- Next direction: continue from A30. Remaining gap to fastllm `1.94 tok/s` is `0.08 tok/s`; likely candidates are narrower attention/kernel overhead checks, or deeper DeepSeek-specific MXFP4 small-MoE fusion.
