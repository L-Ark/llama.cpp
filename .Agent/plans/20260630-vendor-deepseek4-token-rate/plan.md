# Vendor DeepSeek V4 Token-Rate Optimization Plan

Date: 2026-06-30

## Repository And Push Target

This plan is for the vendor repository only, not `ik_llama_migrated`.

- Server: `ssh -p 57990 root@111.237.107.89`
- Working directory: `/root/lfz/vendor/llama.cpp-deepseek-v4`
- Branch: `wip/deepseek-v4-support`
- Push remote: `lark`
- Push URL: `git@github.com:L-Ark/llama.cpp.git`
- Push command for accepted commits: `git push lark wip/deepseek-v4-support`

The `origin` remote currently points at `https://github.com/nisparks/llama.cpp` and is not the target for this optimization work unless explicitly changed later.

## Current Task

Continue optimizing DeepSeek V4 token rate in the vendor `llama.cpp` repository. The objective is to improve decode token rate while preserving correctness and the strict memory budget.

This replaces the earlier mistaken task framing that targeted `ik_llama_migrated`. `ik_llama_migrated` remains useful only as a reference for experiments and wrapper ideas; implementation, measurement, commits, and pushes for this task happen in the vendor repository above.

## Hard Constraints

1. Host RAM must stay strictly below 16 GB, including page cache and memory outside the model process.
2. Use `MemoryMax=14G` and `MemorySwapMax=0` for strict acceptance runs unless a tighter validated setup is introduced.
3. VRAM should be used as fully as possible without violating correctness or TTFT constraints.
4. Every accepted step must preserve semantic correctness.
5. For the prompt `Please introduce France in a short paragraph.`, the answer must be coherent, semantically correct, and free of malformed tail output.
6. `n64` may be used for fast speed screening and profiling only.
7. Final acceptance for an optimization requires at least `n96`; the `n96` tail must not contain garbled bytes, random symbols, broken multilingual fragments, repeated punctuation artifacts, or incoherent suffix text.
8. TTFT must not increase by more than 20% versus the strict vendor baseline.
9. If speed improves but correctness fails, RAM exceeds the limit, or TTFT exceeds the limit, the change is rejected or reverted.

## Existing Baseline Evidence

Vendor DeepSeek V4 has already run successfully on this server. Existing reference runs include:

- `n96` vendor memory-fit baseline, coherent France paragraph, decode around `0.57 tok/s`, prompt eval around `0.50 tok/s`, cgroup memory peak around 15 GiB, VRAM peak around 25.9 GiB.
- Later simple vendor `n96` run under `MemoryMax=14G`, coherent output, decode around `0.59 tok/s`, prompt eval around `0.57 tok/s`, runtime around 7m32s.

Because the strict requirement now includes total host RAM below 16 GB including page cache, the next accepted baseline must be rerun with live memory accounting and preserved logs. Older runs are references, not sufficient final acceptance evidence.

## Required Metrics For Every Experiment

Record all of the following in the run directory before judging the result:

- Git commit SHA and dirty status.
- Full command line and environment variables.
- Prompt, `n_predict`, context size, GPU layer settings, cache settings, mmap/mlock settings, batch settings, and any DeepSeek-specific flags.
- Exact output text.
- Correctness judgment: pass/fail, with a short reason.
- TTFT.
- Prompt eval token rate.
- Decode eval token rate.
- Total wall time.
- Per-token or per-stage timing when profiling is enabled.
- `memory.current`, `memory.peak`, and `memory.stat` fields for anon, file, kernel, slab, pagetables, and related counters.
- GPU memory usage and utilization sampled during the run.
- CPU utilization and I/O indicators when relevant.

## Workflow

The work must alternate between design and execution.

### Step 1: Strict Vendor Baseline Design

Before changing code, define the exact baseline command and monitoring wrapper. The baseline must:

- Run from `/root/lfz/vendor/llama.cpp-deepseek-v4`.
- Use the current branch `wip/deepseek-v4-support`.
- Use `MemoryMax=14G` and `MemorySwapMax=0`.
- Capture cgroup memory and GPU metrics continuously, not only after process exit.
- Run the France prompt at `n96` for acceptance.
- Optionally run `n64` first only to verify the command path and shorten iteration time.

### Step 2: Strict Vendor Baseline Execution

Run the baseline and write a summary JSON or markdown record under a dated run directory. This establishes the TTFT, token rate, correctness, RAM peak, and VRAM peak used for all later comparisons.

If the baseline violates the strict host RAM requirement, do not optimize speed yet. First adjust model loading, mmap, cache, offload, or page-cache behavior until the baseline is valid.

### Step 3: Bottleneck Profiling Design

Profile one valid baseline run to split per-token time into meaningful components. At minimum, identify time spent in:

- Up/down projection compute.
- Expert routing and expert matmuls.
- Attention and KV cache access.
- CPU-GPU transfer or host paging behavior.
- Scheduler/graph overhead.
- Sampling and tokenization overhead.

Rank bottlenecks by expected token-rate gain. Prioritize large, measurable bottlenecks before small cleanups.

### Step 4: Bottleneck Profiling Execution

Implement or enable profiling with minimal code changes. Validate that profiling overhead does not distort the baseline too much. If overhead is significant, compare profiler-on and profiler-off runs and use profiler output only for relative distribution.

### Step 5: Optimization Design

For each candidate optimization, write down before implementation:

- Which measured bottleneck it targets.
- Why it should improve token rate.
- A theoretical upper bound based on hard metrics such as memory bandwidth, transfer size, expert size, arithmetic throughput, or graph launch count.
- Expected impact on TTFT, RAM, VRAM, and correctness.
- The exact acceptance test to run after implementation.

Reference optimization areas include up/down compute speed, expert compute placement, KV/cache behavior, graph scheduling overhead, transfer reduction, and page-cache control.

### Step 6: Optimization Execution

Implement one optimization at a time. Run `n64` for quick screening when appropriate, then run `n96` for acceptance. If the result differs from the theoretical expectation, analyze the gap before moving on.

Accepted changes must satisfy all hard constraints and improve token rate. Immediately commit and push accepted commits:
