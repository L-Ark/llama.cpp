# Vendor DeepSeek V4 16GB Token Rate Optimization Plan

Date: 2026-07-01
Target repo: `/root/lfz/vendor/llama.cpp-deepseek-v4`
Target branch: `feat/ds4-moe-stream-on-vendor`
Target model: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`

## Goal

Optimize DeepSeek V4 token rate in the vendor llama.cpp framework. ik_llama may be used only as a reference implementation or source of ideas; accepted results must run from the vendor repo and vendor binary.

## Current Priority: Cold Start

From this point forward, optimization must focus on **cold-start behavior** under the 16 GB host RAM gate. Single-prompt warm/steady-state results may be recorded as diagnostics, but they are not the primary SOTA target unless they are explicitly labeled as warm-only.

Definitions:

- **Cold start** means the run begins without relying on previously warmed OS page cache for model/expert pages. The default validation method is `sync; echo 3 > /proc/sys/vm/drop_caches` before the experiment, followed by the strict 16 GB cgroup run.
- **Steady state** means model/expert pages, VRAM cache, CUDA context, or storage cache may already be warm from a previous run. Steady-state results must be labeled separately and cannot replace cold-start SOTA.
- **Constrained warmup** is allowed only if the warmup itself runs inside the same 16 GB cgroup workflow and is recorded as part of the benchmark. It must not depend on page cache created outside the 16 GB constraint.

Latest evidence:

- Accepted single-prompt warm result: `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=4`, `eval_tok_s=8.2`, `memory_peak_bytes=15793479680`, `memory_max_events=0`.
- Cold `vram5` rerun after `drop_caches` regressed to `eval_tok_s=1.2`, `ttft_estimate_ms=43188.133285`, and `memory.events max=83388`.
- Prompt-set same-process test after one France warmup failed to generalize: generation rates were `1.6`, `1.8`, `1.3`, `1.0`, `1.2`, `0.9` tok/s, with `memory.events max=422741`, `pgmajfault=4774376`, and `workingset_refault_file=147025029`.
- Path tracing shows the vendor DS4 model currently falls back to CPU expert compute for MXFP4/F8 experts. The CUDA `moe_stream_one` path is present but only correct for `IQ3_XXS`; simply allowing MXFP4/F8 entered the path but produced garbled output and was rejected.

Therefore the current bottleneck is not raw decode compute. It is cold-start and prompt-dependent expert page churn under a 16 GB cgroup. The next accepted improvements must reduce page faults, cgroup reclaim pressure, and TTFT for cold runs while preserving correctness.

## Hard Gates

1. Host RAM must stay below 16 GB, including page cache and all non-process memory. Every accepted run must execute inside a cgroup with `MemoryMax=16000000000` and `MemorySwapMax=0`, and must archive cgroup `memory.current`, `memory.peak`, `memory.events`, and `memory.stat`.
2. VRAM should be used as fully as practical without OOM or correctness loss. Each experiment must record GPU memory and utilization samples.
3. Correctness is mandatory. For the prompt `Please introduce France in a short paragraph.`, the answer must be semantically correct, coherent, and free of obvious degeneration. Large token-rate gains require manual answer review in addition to script checks.
4. TTFT must not increase by more than 20% versus the accepted baseline. The runner records a consistent TTFT estimate from load time plus prompt eval time; any candidate near the limit needs a live first-token probe before acceptance.
5. Cold-start claims must be backed by a cold-state procedure, normally `sync; echo 3 > /proc/sys/vm/drop_caches`, or a documented equivalent. Warm-cache results must be committed only with an explicit `warm_only`, `rejected`, or `needs_cold_validation` classification.

## Baseline Reset

Previous high vendor token rate was observed outside the 16 GB host RAM gate, so it is not a valid baseline. The first implementation step is to rerun baseline in the vendor repo under the strict 16 GB cgroup.

Initial baseline matrix:

| Case | Purpose |
| --- | --- |
| `cpu_moe=40` | Reproduce previous clean 16 GB reference, expected around 1.1 tok/s. |
| `cpu_moe=45` | Check whether a small CPU expert increase is still within the RAM gate. |
| `cpu_moe=50` | Compare against previous clean 16 GB run. |
| `cpu_moe=60` | Confirm whether previous slightly-over-limit result can be made valid with current page dropping. |

Baseline command family:

```bash
GGML_MOE_STREAM=1 \
GGML_MOE_VRAM_CACHE_GB=2 \
GGML_CUDA_DISABLE_GRAPHS=1 \
GGML_MOE_STREAM_DONTNEED=1 \
llama-cli \
  -m "$MODEL" \
  -n 192 -c 512 -b 64 -ub 64 \
  -t 20 -tb 20 \
  -ngl all --fit on -fa auto \
  --temp 0 --top-p 1 --top-k 1 --seed 1 \
  --no-display-prompt \
  --n-cpu-moe "$CPU_MOE" \
  --defer-experts
```

## Optimization Loop

The work alternates between design and execution.

### Design Step

Before each optimization, identify the current bottleneck with an experiment that breaks down per-token time:

- total generation time per token
- prompt eval and TTFT estimate
- GPU memory use and GPU utilization
- cgroup memory peak and page-cache pressure
- major page faults, `workingset_refault_file`, `inactive_file`/`active_file`, and `memory.events max`
- expert streaming / CPU expert behavior from logs where available

Rank candidates by expected token-rate gain and implementation risk. Prioritize the largest compressible time first, especially:

- streamed expert transfer and residency behavior
- VRAM cache sizing and eviction
- page cache pressure under the 16 GB cgroup
- cold-start page-fault count and reclaim behavior
- prompt-dependent expert page churn
- CPU/GPU expert split only if it lowers cold page churn without breaking VRAM fit
- up/down expert compute speed after I/O and page-cache pressure are no longer dominant
- batch and microbatch settings only after the memory bottleneck is understood

### Execution Step

For each method:

1. State why it can improve token rate.
2. Compute a theoretical upper bound from hard limits such as transfer bandwidth, expert size, number of active experts, or measured kernel time.
3. Run the strict gated experiment.
4. Compare measured result with the bound.
5. If the result misses expectation, debug the gap from first principles instead of guessing. Examples: verify whether two operations are actually parallel, whether page cache eviction is working, whether host memory is throttling, whether expert transfer overlaps compute, and whether kernels are using the intended path.
6. Record all metrics and the exact answer.
7. If the candidate passes RAM, VRAM, correctness, TTFT, and cold-start gates and improves token rate, commit and push immediately.
8. If token rate improves but TTFT or cold-start gates fail, commit the experiment only as `rejected` or `needs_ttft_recovery` when it teaches a useful next step.
9. If it regresses token rate, correctness, or memory gates without useful diagnostic value, revert the candidate change and keep only the experiment record.

## Next Cold-Start Optimization Candidates

These supersede the earlier warm single-prompt sweep order.

1. **Cold-start profile with page-fault accounting**
   - Hypothesis: cold runs are dominated by major faults and cgroup reclaim, not GPU compute.
   - Bound: maximum possible speedup is the portion of wall time attributable to major page faults, file reads, and reclaim stalls.
   - Run `drop_caches` cold baselines for current accepted config (`cpu_moe=40`, `vram_cache=4`) and a small neighbor set. Record `pgmajfault`, `workingset_refault_file`, `memory.events max`, file input bytes, VRAM samples, and answer correctness.

2. **Constrained multi-prompt warmup design**
   - Hypothesis: France-only warmup does not cover enough expert pages. A short, diverse warmup set may populate a higher-value subset of expert pages while staying inside 16 GB.
   - Bound: useful gain is capped by the reduction in subsequent major faults and `workingset_refault_file`.
   - Warmup must run inside the same 16 GB cgroup and be recorded as part of the run. It is accepted only if the formal post-warmup prompt set improves and the warmup cost is documented.

3. **Selective expert residency / hot expert pinning**
   - Hypothesis: the 16 GB cgroup cannot keep all expert pages hot, so we should reserve RAM/VRAM for high-frequency experts and avoid low-value page cache churn.
   - Bound: estimate from active expert count, bytes per expert, and measured miss/refault rate.
   - Use traces to identify repeated expert IDs across the prompt set, then test pinning/caching only those experts or changing eviction policy.

4. **Correct DS4 MXFP4/F8 CUDA expert stream**
   - Hypothesis: the CPU fallback dominates cold expert time because DS4 MXFP4/F8 experts do not use the single-expert CUDA stream path. A correct stream implementation can shift expert matvec compute to GPU and make VRAM residency useful for DS4.
   - Bound: each observed expert tensor is about `4.25 MiB` (`src0_bytes=4456448`). The cold lower bound is unique expert bytes divided by storage/page-fault plus H2D bandwidth; the warm lower bound is the measured CUDA kernel plus D2H/scatter time when cached in VRAM.
   - Required first step: compare the existing `moe_stream_one` call layout against the working CUDA `mmvq` implementation for `GGML_TYPE_MXFP4` and `GGML_TYPE_F8_E4M3_B128`. Do not accept a type-gate-only change; that already failed correctness with degenerate output.
   - Updated evidence: isolating only `ffn_gate_exps.weight` into the experimental MXFP4 stream still failed correctness, even after passing the true `nb01` stride into the CUDA MMVQ helper. Replacing the non-ids single-column MMVQ call with a compact-rows MoE ids-kernel helper also failed with the same degeneration. The next CUDA step must be a small numeric CPU-vs-GPU harness for one captured expert/input, not another full-model type-gate trial.
   - Numeric compare evidence: a default-off CPU-vs-GPU compare hook for the experimental gate stream shows that some `GGML_TYPE_MXFP4` experts match CPU to about `1e-6`, while other experts in the same tensors have large mismatches (`max_abs` up to `10.390495`) or GPU zeros. Therefore the bug is expert/cache/layout specific rather than a uniform output scatter failure. Follow-up isolation with `GGML_MOE_STREAM_DONTNEED=0`, with `GGML_MOE_VRAM_CACHE_GB=0`, and with `GGML_MOE_STREAM_ONE_DS4_NONIDS=1` produced the same first-eight compare rows, so the mismatch is not caused by source page discard, VRAM cache copy/slot behavior, or the dedicated MoE ids kernel. A block-level trace using the exact CPU fallback `wdata` q8_0 row reproduced CPU totals for all compared rows, while GPU still returned zeros or wrong-sign values; post-src1 re-quantization also matched `wdata`. Testing CUDA `VDR_MXFP4_Q8_1_MMVQ=2` to match SYCL did not fix correctness or the core compare rows, so the next step must be a CUDA-side single-row debug kernel that writes per-block partial sums for one failing row and one matching row.
   - Acceptance requires France output to remain semantic/coherent and TTFT to stay within the configured gate, or be committed only as rejected/needs_ttft_recovery.

5. **Streaming overlap and cold prefetch**
   - Hypothesis: cold expert page faults are serialized with decode. Predictive prefetch or async expert reads can hide part of the transfer/reclaim time behind GPU compute.
   - Bound: maximum gain is the measured transfer/fault time that overlaps with compute.
   - Updated evidence: broad CPU `MADV_WILLNEED` over active experts reduced major faults in one cold vram2 run but lowered generation rate from the cold baseline range to `1.0 tok/s`. Do not continue broad prefetch as-is; any future prefetch must be narrower or asynchronous enough to avoid reclaim contention.
   - Validate with timeline/resource evidence before accepting; do not infer overlap from tok/s alone.

6. **VRAM cache and CPU-MoE tuning under cold validation**
   - Hypothesis: `vram_cache=4` is best for warm single prompt, but cold prompt sets may need a different balance.
   - Bound: constrained by RTX 5090 VRAM fit and 16 GB cgroup page-cache pressure.
   - Continue sweeps only after each candidate is cold-validated. Warm-only improvements such as `vram5` must be labeled `needs_cold_validation` or `rejected`.

## Required Record For Every Run

Each run directory must contain:

- exact command and environment
- git commit and branch
- prompt
- full stdout and stderr
- parsed prompt token rate, generation token rate, timings, TTFT estimate, wall time
- cgroup memory current/peak/events/stat
- GPU memory/utilization samples
- correctness check result and full answer text
- pass/fail against every hard gate
- timestamped notes in `.Agent/progress/20260701-vendor-ds4-16gb-token-rate-progress.md`
