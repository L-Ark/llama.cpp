# Vendor DeepSeek V4 16GB Token Rate Optimization Plan

Date: 2026-07-01
Target repo: `/root/lfz/vendor/llama.cpp-deepseek-v4`
Target branch: `feat/ds4-moe-stream-on-vendor`
Target model: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`

## Goal

Optimize DeepSeek V4 token rate in the vendor llama.cpp framework. ik_llama may be used only as a reference implementation or source of ideas; accepted results must run from the vendor repo and vendor binary.

## Hard Gates

1. Host RAM must stay below 16 GB, including page cache and all non-process memory. Every accepted run must execute inside a cgroup with `MemoryMax=16000000000` and `MemorySwapMax=0`, and must archive cgroup `memory.current`, `memory.peak`, `memory.events`, and `memory.stat`.
2. VRAM should be used as fully as practical without OOM or correctness loss. Each experiment must record GPU memory and utilization samples.
3. Correctness is mandatory. For the prompt `Please introduce France in a short paragraph.`, the answer must be semantically correct, coherent, and free of obvious degeneration. Large token-rate gains require manual answer review in addition to script checks.
4. TTFT must not increase by more than 20% versus the accepted baseline. The runner records a consistent TTFT estimate from load time plus prompt eval time; any candidate near the limit needs a live first-token probe before acceptance.

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
- expert streaming / CPU expert behavior from logs where available

Rank candidates by expected token-rate gain and implementation risk. Prioritize the largest compressible time first, especially:

- up/down expert compute speed
- streamed expert transfer and residency behavior
- CPU/GPU expert split
- VRAM cache sizing and eviction
- page cache pressure under the 16 GB cgroup
- batch and microbatch settings only after the memory bottleneck is understood

### Execution Step

For each method:

1. State why it can improve token rate.
2. Compute a theoretical upper bound from hard limits such as transfer bandwidth, expert size, number of active experts, or measured kernel time.
3. Run the strict gated experiment.
4. Compare measured result with the bound.
5. If the result misses expectation, debug the gap from first principles instead of guessing. Examples: verify whether two operations are actually parallel, whether page cache eviction is working, whether host memory is throttling, whether expert transfer overlaps compute, and whether kernels are using the intended path.
6. Record all metrics and the exact answer.
7. If the candidate passes RAM, VRAM, correctness, and TTFT gates and improves token rate, commit and push immediately.
8. If it regresses token rate, correctness, TTFT, or memory gates, revert the candidate change and keep only the experiment record.

## First Optimization Candidates

These are ordered after the baseline reset unless the baseline profile points elsewhere.

1. **VRAM cache sweep**
   - Hypothesis: the clean 16 GB baseline uses too little VRAM cache and pays repeated streamed expert load cost.
   - Bound: best possible gain is limited by the fraction of generation time spent waiting on expert load/transfer.
   - Sweep `GGML_MOE_VRAM_CACHE_GB` values that fit RTX 5090 VRAM, starting from 2 GB and increasing conservatively.

2. **CPU-MoE split sweep**
   - Hypothesis: current split is not optimal under strict RAM. More resident CPU experts may reduce streaming cost but can exceed cgroup/page-cache limits.
   - Bound: improvement is limited by the per-token time currently spent on streamed experts that become resident.
   - Sweep near the best valid baseline value first.

3. **Up/down compute path profiling**
   - Hypothesis: expert up/down compute is a significant part of decode time and may have a vendor-kernel or scheduling gap compared with ik_llama.
   - Bound: use measured up/down kernel time per token as the maximum removable time.
   - Compare vendor kernels with corresponding ik_llama reference behavior only to identify vendor changes to port.

4. **Streaming overlap and prefetch**
   - Hypothesis: token generation may serialize expert load with compute.
   - Bound: maximum gain is the portion of transfer time that can be hidden behind compute.
   - Validate with traces before and after; a reported speedup is not accepted unless the timeline shows actual overlap.

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

