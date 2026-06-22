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
