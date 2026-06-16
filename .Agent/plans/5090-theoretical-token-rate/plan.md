# 5090 Theoretical Token Rate

## Goal

Improve GLM-5.1 MoE inference on the RTX 5090 by reusing as many RTX 5060 Ti
optimizations as possible and by making better use of the 5090's 32 GB VRAM,
especially by placing more useful experts in VRAM. The target is to move toward
the GPU-scaled theoretical token rate, not just a small local microbenchmark
win.

## Hard Constraints

- Work only under `/home/wici/lfz`.
- Keep all task plans, scripts, logs, profiles, and summaries in this task
  directory.
- Do not delete or modify files outside `/home/wici/lfz`.
- Default 5090 line remains strict host RAM:
  `MemoryMax=2G`, `MemorySwapMax=0`, `GGML_MOE_RAM_TIER_MIB=0`.
- Any non-strict experiment, such as re-enabling RAM tier, must be clearly
  labeled as a diagnostic and must not be reported as strict-2GB performance.
- Do not claim the goal is achieved until a current 5090 run demonstrates the
  target token rate and the exact workload/config is recorded.
- Another parallel task may use the RTX 5090 or host RAM. Before running
  experiments, check GPU/RAM/process occupancy. If other work is active, do not
  stop it; wait for it to finish naturally before starting this task's run.

## Baseline Evidence

RTX 5060 Ti accepted/past configs:

- `wici-glm51-0p97`: accepted 5060 Ti groundtruth, `0.97 tok/s`,
  `GGML_MOE_RAM_TIER_MIB=4096`, `GGML_MOE_VRAM_CACHE_MIB=2048`.
- `wici-glm51-interactive-n84`: 5060 Ti interactive n84 preset,
  `GGML_MOE_RAM_TIER_MIB=3072`, `GGML_MOE_VRAM_CACHE_MIB=1536`,
  `GGML_MOE_PROMPT_DYNAMIC_EXPERTS=1`,
  `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=3`.
- Historical n84 5060 Ti-style reproduction targets recorded in
  `.Agent/plans/5090-fast-ssd-first-three-repro/plan.md`:
  - pre-opt baseline: `1.02 eval tok/s`, `RAM tier=0`, `VRAM hit=6.0%`;
  - optimized candidate1: `1.25 eval tok/s`, `RAM tier=0`,
    `VRAM hit=29.1%`;
  - optimized candidate2: `1.01 eval tok/s`, `RAM tier=0`,
    `VRAM hit=31.8%`.

Current 5090 evidence:

- Strict n84 I/O sweep direct/iouring is around `1.05-1.06 eval tok/s`, with
  low `VRAM hit=7.1%`.
- Dynamic-k smoke with staging+iouring reaches `1.107 eval tok/s`, but only
  `13.5%` VRAM hit and no meaningful wall-time improvement on the short smoke.
- Current 5090 does not appear GPU-bound; expert placement and storage/staging
  dominate.

Theoretical target:

- 5090 vs 5060 Ti theoretical GPU scale is roughly `4x-4.4x` depending on
  bandwidth/AI throughput assumptions.
- From 5060 Ti `0.97 tok/s`, the GPU-bound target would be about
  `3.9-4.3 tok/s`.
- From 5060 Ti `1.25 tok/s`, the upper target would be about `5.0-5.5 tok/s`.
- These targets are not expected unless the workload becomes mostly GPU-bound.

## 5060 Ti Optimization Checklist

Compatible with strict 2GB unless proven otherwise:

- `GGML_MOE_STREAM=1`
- `GGML_MOE_STREAM_BATCH_ONLY=1`
- `GGML_MOE_STREAM_DEFER=1`
- `GGML_MOE_STREAM_CPU_OPS=1`
- `GGML_MOE_PARALLEL_EXPERTS=1`
- `GGML_MOE_STREAM_FUSED_UP_GATE=1`
- `GGML_MOE_GPU_HANDOFF=1`
- `GGML_MOE_VRAM_CACHE_SPLIT=1`
- `GGML_MOE_STREAM_ONE_CACHE_MIB=0`
- `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`
- `GGML_MOE_STAGE_PINNED_SLOTS=16`
- `GGML_MOE_STREAM_UP_GATE_PARALLEL=1`
- `GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE=1`
- `GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT=1`
- `GGML_MOE_DOWN_PARALLEL_STAGE=1`
- `GGML_MOE_PREFETCH_DOWN=0`
- `GGML_MOE_PREFETCH_DOWN_DEPTH=8`
- route/profile preload hooks, if supported by the command path.

Potentially incompatible with strict 2GB:

- `GGML_MOE_RAM_TIER_MIB=3072/4096`;
- `GGML_MOE_STAGE_PINNED=1`, depending on host pinned allocation behavior;
- chat-specific startup preload if using raw `llama-cli` rather than chat
  wrapper.

## Experiment Plan

### Phase 0: Reproducible Runner

Create a task-local runner that can:

- run the fixed n84 long prompt and the smoke4 dataset;
- enforce strict 2GB by default;
- toggle each 5060 optimization explicitly;
- sweep VRAM cache size and profile choice;
- optionally run non-strict diagnostics;
- parse timings, VRAM hit, RAM hit, direct reads, io_uring reads, read
  failures, and output paths.

### Phase 1: Config-Only Strict 2GB Matrix

Run the strict line first, no core code changes:

1. current best strict staging+iouring baseline;
2. add `GGML_MOE_GPU_HANDOFF=1`;
3. add `GGML_MOE_PROMPT_DYNAMIC_EXPERTS=1`;
4. try profile/preload envs supported by raw prompt:
   `LLAMA_PROMPT_STARTUP_PROFILE_PRELOAD=1`,
   `LLAMA_PROMPT_TTFT_MARKERS=1`;
5. sweep `GGML_MOE_VRAM_CACHE_MIB`:
   `12288`, `16384`, `20480`, `24576`, and highest stable value with reserve;
6. compare route profile vs top8 runtime profile:
   - `wici-glm51-interactive-n84.route.csv`;
   - `wici-glm51-interactive-n84-expert-top8.runtime.csv`.

Acceptance for a candidate:

- no OOM;
- read failures `0`;
- `VRAM hit` improves materially;
- eval token rate improves on the same workload.

### Phase 2: Same-Prompt Expert Residency Upper Bound

Use route traces from the exact n84 prompt to build an oracle/same-prompt
profile that places all or most used experts into VRAM, then run the same
prompt with large 5090 VRAM cache.

Purpose:

- estimate the real upper bound if expert placement is solved;
- determine whether `4 tok/s` is possible with this runtime once SSD misses are
  mostly removed.

This is explicitly an upper-bound diagnostic, not a general benchmark.

### Phase 3: Generalizable Profile Improvement

If same-prompt residency is much faster:

- build a larger top-K or frequency-weighted profile from multiple prompts;
- test on held-out smoke/daily prompts;
- improve cache loader/profile interpretation if current 5090 is failing to
  reproduce historical `29-32%` VRAM hit.

### Phase 4: Code Changes Only After Config Evidence

Possible code directions:

- load more profile rows into 5090 VRAM using a real byte budget rather than a
  row-count or conservative cap;
- fix any mismatch between profile tensor names and runtime lookup;
- add 5090-aware cache admission/protection for top experts;
- reduce runtime SSD read path only after hit rate is improved.

### Phase 5: Trace-Guided Future Prefetch Diagnostic

Implement a diagnostic-only prefetch path that can consume an exact route trace
from the same prompt and request upcoming expert rows before they are needed.

Purpose:

- estimate whether better cross-layer/token prediction can still improve the
  strict 2 GB 5090 line beyond the oracle static profile result;
- separate "cache contents are wrong" from "miss path cannot overlap enough";
- avoid reporting this as a general benchmark, because the trace is generated
  from the same prompt.

Acceptance:

- strict `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- no read failures and no CUDA OOM;
- exact trace/profile/log paths recorded;
- compare against `replay-oracle-n84-up65-safety256` (`1.88 eval tok/s`).

## Progress Log

- 2026-06-12: Created task. Initial evidence shows current 5090 is not
  GPU-bound and does not fully reuse all 5060 Ti optimizations.
- 2026-06-12: Added task-local runner
  `.Agent/plans/5090-theoretical-token-rate/run_5090_matrix.py`.
  It enforces strict 2 GB by default and can toggle 5060-compatible options,
  profile choice, VRAM cache size, io_uring, and diagnostics.
- 2026-06-12: Ran strict 2 GB 5060-compatible top8 baseline with
  `GGML_MOE_GPU_HANDOFF=1`, `GGML_MOE_PROMPT_DYNAMIC_EXPERTS=1`,
  `LLAMA_PROMPT_STARTUP_PROFILE_PRELOAD=1`,
  `LLAMA_PROMPT_TTFT_MARKERS=1`, `GGML_CUDA_EAGER_CUBLAS=1`,
  `GGML_MOE_VRAM_CACHE_MIB=12288`, `RAM tier=0`, `backend=iouring`.
  - `strict-compat-5060-n4`: `eval tok/s=1.13`, `prompt tok/s=1.69`,
    `VRAM hit=14.8%`, `direct_reads=1800`, `iouring_reads=4620`,
    `read_failures=0`.
  - `strict-compat-5060-n84`: `eval tok/s=1.07`, `prompt tok/s=1.68`,
    `total_ms=98795.03`, `VRAM hit=7.1%`, `direct_reads=1800`,
    `iouring_reads=138750`, `read_failures=0`.
  - Interpretation: reusing these 5060-compatible flags does not make the long
    n84 run GPU-bound. The run is still dominated by runtime expert reads and
    very low VRAM hit.
- 2026-06-12: Ran strict 2 GB, top8 profile, `GGML_MOE_VRAM_CACHE_MIB=20480`
  with the same 5060-compatible flags.
  - `strict-vram20480-top8-n84`: `eval tok/s=1.05`,
    `prompt tok/s=1.80`, `total_ms=99765.84`, `VRAM hit=70.6%`,
    `direct_reads=1200`, `iouring_reads=29245`, `read_failures=0`.
  - This proves the 5090 can hold many more experts and dramatically reduce
    runtime SSD reads under the strict 2 GB host-RAM line.
  - However, token rate did not improve. The next bottleneck is no longer only
    raw SSD read count; profile precision, cache lookup/admission overhead,
    GPU/CPU MoE compute scheduling, or H2D/cache interaction must be examined.
- 2026-06-12: Ran strict 2 GB, route profile,
  `GGML_MOE_VRAM_CACHE_MIB=20480`.
  - `strict-vram20480-route-n84`: `eval tok/s=0.82`,
    `prompt tok/s=1.64`, `total_ms=123270.67`, `VRAM hit=9.1%`,
    `direct_reads=2876`, `iouring_reads=90502`, `read_failures=0`.
  - The 5060 Ti route profile is worse than the top8 runtime profile for this
    strict raw n84 run. Keep top8 runtime for the current path.
- 2026-06-12: Ran strict 2 GB, top8 profile,
  `GGML_MOE_VRAM_CACHE_MIB=24576`.
  - `strict-vram24576-top8-n84`: `eval tok/s=0.68`,
    `prompt tok/s=1.91`, `total_ms=118956.89`, `VRAM hit=41.2%`,
    `direct_reads=600`, `iouring_reads=23635`, `read_failures=0`.
  - This was worse than 20 GB and generated only 67 eval tokens before stop,
    so 24 GB is not currently a stable win.
- 2026-06-12: Important bottleneck evidence from stderr:
  - `strict-vram20480-top8-n84` logs
    `batched CUDA MoE down path declined; falling back to CPU path`.
  - `strict-vram24576-top8-n84` logs
    `batched CUDA MoE up/gate path declined; falling back to CPU path`.
  - Therefore the 5090 is still not fully used even when VRAM hit is high.
    The next implementation step is to add/enable decline-reason diagnostics
    for the down path and determine why CUDA MoE kernels are not taking over.
- 2026-06-12: Added down-path decline diagnostics and confirmed the 20 GB
  top8 run declines because `batch_cache_get()` returns null for
  `ffn_down_exps.weight`.
  - Root cause from logs: after model load only about `13754 MiB` is reported
    free. The requested `20480 MiB` cache is split into `12.0 GiB` up/gate
    first, then down cache `cudaMalloc 8.0 GiB FAILED`.
  - A short `auto-clamp` diagnostic with `safety=512 MiB` succeeded and created
    both pools: up/gate `7.8 GiB`, down `5.2 GiB`. It avoided down decline but
    only reached `20.0%` VRAM hit on the short n8 smoke.
  - Added cache allocation retry so a failed second pool can shrink instead of
    disabling that cache class. A no-clamp 20 GB n8 run then allocated up/gate
    `12.0 GiB` and down `1.4 GiB`, proving the down path can get a cache, but
    the run later OOMed at CUDA graph launch. This means any 5090 cache policy
    must leave explicit graph/compute headroom, not just allocate until
    `cudaMalloc` succeeds.
- 2026-06-12: Stable strict 2 GB split/clamp run with top8 profile:
  - `clamp20480-up75-safety512-n84`: `1.13 eval tok/s`,
    `16.1%` VRAM hit, `125394` io_uring reads.
  - Compared with the unclamped high-upgate run, this avoids down fallback but
    the long n84 route distribution is still poorly covered by the old profile.
- 2026-06-12: Captured same-prompt oracle route profile:
  `.Agent/plans/5090-theoretical-token-rate/oracle-n84.route.csv`.
  - It contains `19167` expert rows, about `76.6 GiB` total, so full residency
    is impossible under the current `~13 GiB` post-load MoE cache budget.
  - Replaying the same prompt with this oracle profile and strict 2 GB:
    `replay-oracle-n84-up75-safety512` reached `1.83 eval tok/s`,
    `57.5%` VRAM hit, and reduced io_uring reads from `125394` to `63552`.
  - This proves profile/expert selection is a major bottleneck; current 5090
    compute is not the limiting factor yet.
- 2026-06-12: SER diagnostics with oracle profile:
  - `ser=1,1.05` / `ser=1,1.10`: `6.26-6.27 eval tok/s`, but output is visibly
    corrupted/degenerate. This is only a speed upper bound, not acceptable model
    quality.
  - `ser=4,1.05`: `2.24 eval tok/s`, still visibly degraded.
  - `ser=6,1.05`: `1.69 eval tok/s`, slower than oracle baseline and degraded.
  - Naive SER is not a valid completion path without an accuracy/quality gate.
- 2026-06-12: Best strict non-SER result so far is the same-prompt oracle
  replay with cache auto-clamp and `safety=256 MiB`:
  `replay-oracle-n84-up65-safety256`, `1.88 eval tok/s`, `58.1%` VRAM hit,
  `62582` io_uring reads. Reducing headroom to `128 MiB` OOMs around CUDA
  graph launch, so the current stable post-load cache budget is about
  `13.5 GiB` with at least `256 MiB` safety.
- 2026-06-12: Ring depth/slot sweep and existing down-prefetch did not improve
  the oracle replay. `depth=32, slots=32` remained around `1.84 eval tok/s`,
  and `GGML_MOE_PREFETCH_DOWN=1` remained around `1.88 eval tok/s`. This
  indicates the miss path is dependency-limited and/or insufficiently
  predictive, not simply under-provisioned on io_uring queue depth.
- 2026-06-12: Next implementation step is a diagnostic trace-guided future
  expert prefetch path using `oracle-n84.trace.csv`. This is not a valid
  general benchmark, but it can show whether correct early prefetch can close
  more of the gap toward the 5090 theoretical target.
- 2026-06-12: Implemented an env-gated trace-guided future prefetch diagnostic
  (`GGML_MOE_TRACE_PREFETCH`, `GGML_MOE_TRACE_PREFETCH_WINDOW`,
  `GGML_MOE_TRACE_PREFETCH_MAX_LOADS`) and added runner flags for it. Build
  passes.
  - First naive main-stream version was safe but much too aggressive:
    `trace-prefetch-n8-smoke`, `0.12 eval tok/s`, `94918` direct reads.
  - Cursor-limited main-stream version still regressed:
    `trace-prefetch-cursor-n8-smoke`, `0.83 eval tok/s`,
    `45.6%` VRAM hit.
  - Added pending-slot CUDA events so prefetch-stream copies are safe: a cache
    hit on a pending slot waits for the slot's ready event before use.
  - Async trace prefetch with `window=24,max_loads=4` still regressed:
    `trace-prefetch-async-w24-l4-n8`, `1.10 eval tok/s`,
    `38.8%` VRAM hit, `2258` async waits.
  - Pack-backed async trace prefetch removed the registered-tensor limitation
    (`missing_tensor=0`) but still regressed:
    `trace-prefetch-pack-w24-l4-n8`, `1.11 eval tok/s`,
    `35.6%` VRAM hit.
  - No-trace n8 control under the same oracle/cache settings:
    `oracle-no-trace-n8-control`, `1.45 eval tok/s`, `41.5%` VRAM hit.
  - Interpretation: per-row future prefetch is the wrong primitive here. It
    adds extra direct reads and cache churn. The next implementation must batch
    trace-prefetch loads through the existing io_uring multi-job path, or avoid
    prefetching rows that would displace profile-pinned/high-frequency rows.
- 2026-06-12: Reworked trace-prefetch to use pending-slot CUDA events and then
  batch same-size future rows through `expert_pack_iouring_copy_jobs()`.
  - `trace-prefetch-batch-w24-l4-n8`: direct reads fixed (`2977`, same as
    control), but iouring reads increased to `9910` and eval fell to
    `1.14 tok/s`.
  - `trace-prefetch-batch-noevict-w24-l4-n8`: no cache eviction during
    prefetch, `1.39 tok/s`, `38.9%` VRAM hit, `8043` iouring reads.
  - `trace-prefetch-batch-noevict-w12-l1-n8`: same `1.39 tok/s`, `39.0%` VRAM
    hit, `8026` iouring reads.
  - Conclusion: even safe, batched, no-evict trace prefetch does not beat the
    no-trace oracle control (`1.45 tok/s` n8) because it adds read work without
    improving hit rate. Do not pursue trace-prefetch further until there is a
    stronger predictor/admission policy. Shift next work to cache admission and
    profile residency.
- 2026-06-12: Ran profile reserve/admission sweep.
  - n8 controls:
    - `oracle-no-trace-n8-control`: `1.45 eval tok/s`, `41.5%` VRAM hit,
      `7382` io_uring reads.
    - `oracle-n8-reserve0`: `0.48 eval tok/s`, `100.0%` VRAM hit, but only
      `727` cache lookups and no io_uring misses. This is a misleading high
      hit-rate case; profile pins fill the cache and the runtime takes a slow
      path.
    - `oracle-n8-reserve20`: `1.43 eval tok/s`, `39.4%` VRAM hit,
      `7652` io_uring reads.
    - `oracle-n8-reserve40`: `1.52 eval tok/s`, `44.1%` VRAM hit,
      `7061` io_uring reads.
  - Long n84 validation for reserve 40:
    - `oracle-n84-reserve40`: `1.66 eval tok/s`, `49.8%` VRAM hit,
      `74937` io_uring reads.
    - This is worse than the current best strict non-SER run:
      `replay-oracle-n84-up65-safety256`, `1.88 eval tok/s`, `58.1%` VRAM hit,
      `62582` io_uring reads.
  - `GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1` n8 smoke:
    `oracle-n8-reserve10-evict`, `1.49 eval tok/s`, same hit/miss counts as
    the no-trace n8 control, so it is not a strong candidate for long n84.
- 2026-06-12: Current best strict 2 GB, RAM tier 0, non-SER result remains:
  `replay-oracle-n84-up65-safety256`, `1.88 eval tok/s`, `58.1%` VRAM hit,
  `62582` io_uring reads, `read_failures=0`. This is an improvement over
  top8/clamped strict long n84 (`1.13 eval tok/s`) but still below the
  theoretical 5090 target (`~3.9-5.5 tok/s` depending on 5060 Ti baseline).

## Current Bottleneck

The 5090 line is still not compute-bound. The best non-SER same-prompt oracle
profile cuts runtime SSD reads roughly in half relative to the old profile, but
still performs more than `62k` expert-pack io_uring reads over n84. Simple
future prefetch, even with exact future trace and safe async slot events,
regresses because it adds read work and cache churn. Reserve sweeps show that
freeing more runtime cache can help short n8 but hurts long n84 by reducing
useful profile residency.

The next useful implementation direction is not more blind prefetch. It should
be profile/admission quality:

- use multiple representative route traces to build a held-out-valid profile;
- make admission frequency-aware for runtime misses, so one-off future rows do
  not evict profile/high-hit rows;
- simulate cache policies from `oracle-n84.trace.csv` before implementing more
  CUDA/I/O changes;
- only revisit prefetch when the predictor/admission policy can prove it
  reduces misses without increasing total reads.

## Phase 6 Results

- 2026-06-12: Added
  `.Agent/plans/5090-theoretical-token-rate/simulate_cache_policy.py` for
  offline cache policy simulation from `oracle-n84.trace.csv` and
  `oracle-n84.route.csv`.
  - Baseline LFU/LRU simulation with `upgate=2282`, `down=1028`,
    `reserve=10%`: `62550` misses, `58.139%` hit rate. This closely matches
    the best real run (`62582` io_uring reads, `58.1%` hit rate).
  - `profile_lfu_lru` simulation predicted `57368` misses, `61.607%` hit
    rate, about `5.2k` fewer reads.
  - Full sweep saved to
    `.Agent/plans/5090-theoretical-token-rate/cache-policy-sweep.json`.
- 2026-06-12: Implemented env-gated runtime cache policy
  `GGML_MOE_VRAM_CACHE_POLICY=profile_lfu_lru` and added runner flag
  `--cache-policy`.
  - First implementation only assigned profile counts to preloaded slots, so
    n84 was effectively unchanged: `profile-lfu-lru-n84`,
    `1.89 eval tok/s`, `62582` io_uring reads, `58.1%` VRAM hit.
  - Fixed runtime miss insertion to look up profile counts by hashed cache key.
    This made n8 worse: `profile-lfu-lru-counts-n8`,
    `1.37 eval tok/s`, `8371` io_uring reads, `33.7%` VRAM hit, compared with
    no-trace/control `1.45 eval tok/s`, `7382` reads, `41.5%` hit.
  - Interpretation: the offline simulator matched the long-run aggregate, but
    `profile_lfu_lru` over-protects globally frequent rows and hurts early
    token locality. Do not promote this policy as a win. The simulator needs a
    short-prefix/locality objective, not only full-trace miss minimization.
- 2026-06-12: Extended simulator with prefix metrics and a
  `hybrid_profile_lfu_lru` policy. The baseline simulation reproduces the n8
  prefix exactly: `7382` prefix misses, matching the no-trace n8 control.
  - Single-point simulation for `hybrid_profile_lfu_lru --profile-after 12624`
    predicted no prefix regression and fewer full-trace misses:
    `57808` misses vs baseline `62550`.
  - Runtime n8 for `hybrid_profile_lfu_lru --cache-profile-after 12624`
    preserved the baseline prefix behavior: `7382` reads, `41.5%` VRAM hit,
    `1.52 eval tok/s`.
  - Runtime n84 for the same policy regressed badly:
    `hybrid-profile-after12624-n84`, `1.50 eval tok/s`, `39.2%` VRAM hit,
    `90869` io_uring reads.
  - Diagnostics show profile counts are present but not useful enough for
    victim selection in runtime: `profile_count_lookups=90869`, `hits=57485`,
    `inserted_avg=6.25`, `victim_avg=5.01`. Count-based victim choice churns
    local rows and increases misses.
  - Full prefix-aware sweep saved to
    `.Agent/plans/5090-theoretical-token-rate/cache-policy-prefix-sweep.json`,
    but simulator predictions remain too optimistic for profile-count victim
    policies. Keep `profile_lfu_lru` / `hybrid_profile_lfu_lru` env-gated only;
    default remains `lfu_lru`.

## Updated Direction

Profile-count victim selection is not the next path to the theoretical 5090
rate. The remaining gap is dominated by miss-path cost and batching, not simple
cache replacement. Evidence:

- best strict non-SER still performs `62582` io_uring reads over n84;
- trace-prefetch and profile-count policies both increase reads or miss wait;
- io_uring inflight remains low (`~2.3-2.8`, max `8`) because runtime batches
  are small/dependency-limited.

Next implementation should target miss-path batching/coalescing:

- reduce per-miss CPU/cache lookup overhead;
- group same-layer or same-size miss loads into larger io_uring submissions;
- avoid extra reads and avoid evicting current local working sets;
- use real n8/n84 read counts as the acceptance gate, not only simulator miss
  counts.

## Phase 7: Miss-Path Batching

Profile-count cache replacement and trace-prefetch both failed to improve the
strict long n84 line. The next target is the existing miss path itself:

- inspect up/gate parallel staging and down parallel staging;
- look for places where misses are split into smaller io_uring batches than the
  dependency graph requires;
- avoid any optimization that adds reads or changes expert selection;
- accept only if strict n8 does not regress and strict n84 reduces total time or
  io_uring wait at similar read counts.

### Phase 7 Progress

- 2026-06-12 22:49 CST: Inspected the up/gate and down staging paths. Found a
  likely configuration mismatch: the task runner sets
  `GGML_MOE_STREAM_UP_GATE_SPLIT_STAGE=1`, while the CUDA path checks
  `GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT`. This means the intended 5060 Ti
  up/gate split-stage optimization may not have been enabled in the current
  5090 runs. Next step is a minimal alias fix, then strict n8/n84 validation
  against the current best oracle baseline.
- 2026-06-12 22:50 CST: Added a compatibility alias so both
  `GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT` and
  `GGML_MOE_STREAM_UP_GATE_SPLIT_STAGE` enable up/gate split-stage. Validation
  shows this is not a win on the current 5090/oracle workload:
  - `split-stage-alias-n8`: `1.48 eval tok/s`, `41.5%` VRAM hit, `7382`
    io_uring reads, `read_failures=0`. It did not regress the n8 smoke and
    confirmed `up_aux`/`gate_aux` staging is active.
  - `split-stage-alias-n84`: `1.83 eval tok/s`, `58.1%` VRAM hit, `62582`
    io_uring reads, `read_failures=0`.
  - Compared with current best `replay-oracle-n84-up65-safety256`
    (`1.88 eval tok/s`, `62582` reads), split-stage keeps the same miss count
    but slows down. The likely cause is smaller io_uring batches:
    `inflight_avg=1.67`, `inflight_max=4`, versus the previous best around
    `inflight_avg=2.38`, `inflight_max=8`.
  - Decision: keep the code alias for diagnostics, but change the task runner
    so up/gate split-stage is opt-in via `--up-gate-stage-split`, not part of
    the default 5090 baseline.
- 2026-06-12 22:56 CST: Verified the default runner after making split-stage
  opt-in:
  - `default-no-split-n8-verify`: `1.48 eval tok/s`, `41.5%` VRAM hit, `7382`
    io_uring reads, `read_failures=0`.
  - Only `main` and `gate` staging rings are active. `inflight_avg=2.70`,
    `inflight_max=8`, and batch histogram includes `586` batches in the `5-8`
    bucket. This confirms the default path has better batching than the
    split-stage diagnostic.
- 2026-06-12 22:56 CST: Implemented an io_uring CQE drain attempt: after one
  blocking `io_uring_wait_cqe`, non-blockingly drain already-completed CQEs
  before blocking again. This does not change read count or cache behavior.
  - `cqe-drain-n8`: `1.50 eval tok/s`, `41.5%` VRAM hit, `7382` io_uring
    reads, `read_failures=0`.
  - `wait_calls` barely changed (`7382` to `7370`), so CQEs rarely arrive
    clustered enough for this to matter. Do not spend a long n84 run on this
    alone.
- 2026-06-12 22:58 CST: Added an opt-in io_uring submission-order diagnostic:
  `GGML_MOE_IO_SORT_OFFSET=1` / runner flag `--iouring-sort-offset`. It sorts
  jobs within each same-size io_uring batch by expert-pack file offset before
  submission, without changing expert selection, read count, cache slots, or
  the strict 2 GB/RAM-tier-0 line.
  - `offset-sort-n8`: `1.59 eval tok/s`, `41.5%` VRAM hit, `7382` io_uring
    reads, `read_failures=0`.
  - `offset-sort-n84`: `1.92 eval tok/s`, `58.1%` VRAM hit, `62582`
    io_uring reads, `read_failures=0`, `total_ms=64918.76`.
  - This improves the current best strict non-SER result from
    `replay-oracle-n84-up65-safety256` (`1.88 eval tok/s`, same `62582`
    reads) to `1.92 eval tok/s`, with no increase in reads. It does not close
    the theoretical 5090 gap, but it is the first Phase 7 change that improves
    the long strict n84 run.
  - Keep it opt-in for now because only the oracle n84 workload has been
    validated. Recommended current command adds `--iouring-sort-offset` to the
    previous best config.
- 2026-06-12 23:00 CST: Tested SQPOLL as another io_uring submission-overhead
  diagnostic, combined with offset sorting:
  - `offset-sort-sqpoll-n8`: `1.53 eval tok/s`, `41.5%` VRAM hit, `7382`
    io_uring reads, `read_failures=0`.
  - `iouring_submit_us` dropped sharply (`563252` to `13129` on n8), but
    `iouring_wait_us` increased and token rate regressed versus
    `offset-sort-n8` (`1.59`). Do not run long n84 or adopt SQPOLL for this
    workload.

## Current Best Strict 2GB Result

As of 2026-06-12 23:00 CST, the best strict, RAM-tier-disabled, non-SER
oracle n84 result in this task is:

- label: `offset-sort-n84`
- command/env/log directory:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase7-miss-batching/offset-sort-n84/`
- strict host RAM: `MemoryMax=2G`, `MemorySwapMax=0`
- RAM tier: `GGML_MOE_RAM_TIER_MIB=0`
- profile: `.Agent/plans/5090-theoretical-token-rate/oracle-n84.route.csv`
- VRAM cache request: `20480 MiB`, auto-clamped to `13498 MiB`,
  up/gate split `65%`, safety `256 MiB`
- optimization delta: `GGML_MOE_IO_SORT_OFFSET=1`
- `eval tok/s=1.92`, `prompt_eval tok/s=1.60`, `total_ms=64918.76`
- `VRAM hit=58.1%`, `RAM hit=0.0%` / disabled
- `direct_reads=2978`, `io_uring_reads=62582`, `read_failures=0`
- io_uring: `batches=22560`, `submit_calls=22560`, `wait_calls=62518`,
  `inflight_avg=2.38`, `inflight_max=8`

Remaining bottleneck:

- read count is unchanged at `62582`, so offset sorting only reduces cost per
  miss; it does not solve expert residency or prediction;
- the workload is still far below the GPU-scaled 5090 target because it is
  still I/O/miss-path dominated;
- further gains likely require reducing miss count, increasing batch size
  across dependency boundaries, or introducing a quality-safe approximation
  rather than more syscall-level tuning.

## Phase 8: Cache Split / More Useful Experts

Offset sorting improved cost per miss, but the run still performs `62582`
runtime expert reads. The next attempt returns to the original goal of using
the 5090's VRAM more effectively:

- keep strict `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep the same oracle profile/trace diagnostic for now, because it gives a
  reproducible upper-bound workload;
- use offline simulation to sweep the post-load cache split between up/gate
  rows and down rows under the observed clamped cache budget (`13498 MiB`);
- only run GPU validation for candidates that predict fewer misses than the
  current `62582` read line and do not degrade the n8 prefix too much;
- keep `--iouring-sort-offset` enabled for validation because it is currently
  the best miss-path implementation.

Acceptance gate:

- strict n8 must stay at least near the current `offset-sort-n8` line
  (`1.59 eval tok/s`, `7382` reads) or explain why the long-run tradeoff is
  worth testing;
- strict n84 must beat `offset-sort-n84` (`1.92 eval tok/s`, `62582` reads)
  or reduce reads materially enough to justify follow-up.

### Phase 8 Progress

- 2026-06-12 23:05 CST: Added
  `.Agent/plans/5090-theoretical-token-rate/sweep_cache_split.py` and swept
  the observed clamped cache budget (`13498 MiB`) across up/gate split
  percentages.
  - Coarse sweep saved to `cache-split-sweep-13498.json`.
  - Fine sweep saved to `cache-split-sweep-13498-fine.json`.
  - Best simulated split was `70%` up/gate: `62421` misses versus the current
    `65%` line at `62550` simulated misses. This is only about `129` fewer
    misses and not a material reduction.
- 2026-06-12 23:05 CST: Validated the best simulated split on strict n8 with
  offset sorting:
  - `up70-offset-sort-n8`: `1.52 eval tok/s`, `41.6%` VRAM hit, `7375`
    io_uring reads, `read_failures=0`.
  - It reduced n8 reads by `7` but regressed speed versus `offset-sort-n8`
    (`1.59 eval tok/s`). Because the long-run simulated gain is tiny, do not
    spend a long n84 run on this split. Keep the current `65%` split as the
    best validated setting.
- 2026-06-12 23:06 CST: Next hypothesis is larger effective cache budget. The
  logs show auto-clamp is already using almost all free post-load VRAM:
  `free=13754 MiB`, `actual=13498 MiB` with `safety=256 MiB`. Earlier
  `safety=128 MiB` OOMed around CUDA graph launch. Test whether disabling graph
  reuse (`--no-graph-reuse`) lets us safely reduce safety and place more expert
  rows in VRAM.
- 2026-06-12 23:07 CST: `--no-graph-reuse` did not make `safety=128 MiB`
  viable:
  - `no-gr-safety128-offset-sort-n8`: failed before producing timings.
  - stderr shows `actual=13626 MiB`, then CUDA OOM at
    `evaluate_and_capture_cuda_graph` / `cudaGraphLaunch`.
  - Do not use this path. Next test is reducing over-provisioned context/batch
    memory (`-c 2048 -b 2048`) for this short n84 workload, to see whether the
    freed VRAM can hold more expert rows without CUDA graph OOM.
- 2026-06-12 23:08 CST: Reducing batch alone did not increase expert cache
  budget:
  - `batch512-offset-sort-n8`: `1.50 eval tok/s`, `41.5%` VRAM hit, `7382`
    io_uring reads, `read_failures=0`.
  - VRAM cache budget remained `actual=13498 MiB`, same as the best run.
  - Next test is reducing context (`-c`) because KV/cache allocation is a more
    plausible source of reclaimable VRAM for this short n84 prompt.
- 2026-06-12 23:11 CST: Reducing context to `512` did increase effective
  expert cache budget and slightly improved decode:
  - `ctx512-offset-sort-n8`: `1.54 eval tok/s`, `41.8%` VRAM hit, `7343`
    io_uring reads, `read_failures=0`, cache budget `actual=13630 MiB`.
  - `ctx512-offset-sort-n84`: `1.93 eval tok/s`, `58.4%` VRAM hit, `62129`
    io_uring reads, `read_failures=0`, cache budget `actual=13630 MiB`.
  - Compared with `offset-sort-n84`, this reduces runtime reads by `453` and
    improves decode from `1.92` to `1.93 tok/s`, but prompt/total time is worse
    in this run (`total_ms=66336.61` vs `64918.76`). Treat it as a decode-path
    diagnostic win, not a full wall-time win.
- 2026-06-12 23:14 CST: Reducing context further to `256` placed more experts
  in VRAM but reduced token rate:
  - `ctx256-offset-sort-n8`: `1.51 eval tok/s`, `42.1%` VRAM hit, `7304`
    io_uring reads, `read_failures=0`, cache budget `actual=13810 MiB`.
  - `ctx256-offset-sort-n84`: `1.78 eval tok/s`, `58.8%` VRAM hit, `61508`
    io_uring reads, `read_failures=0`, cache budget `actual=13810 MiB`.
  - This proves more expert residency alone is not sufficient. It reduces reads
    by `1074` versus `offset-sort-n84`, but wait time rises and decode speed
    regresses. Do not use `ctx=256` as the current best.

Updated Phase 8 interpretation:

- The 5090 can hold more experts by reducing context, and this does lower miss
  count, but the current miss path still waits too much for the remaining reads.
- Current best decode-only strict line is `ctx512-offset-sort-n84`
  (`1.93 eval tok/s`, `62129` reads).
- Current best overall total-time strict line remains `offset-sort-n84`
  (`1.92 eval tok/s`, `total_ms=64918.76`, `62582` reads).
- The theoretical target is still not achieved. The next useful direction is
  reducing wait cost for the remaining reads or changing the computation
  workload safely; simply adding a few hundred MiB of expert cache is too weak.

## Phase 9: Remaining Wait-Cost Diagnostics

The first Phase 9 hypothesis was that `GGML_MOE_IO_BYTES=2MiB` might be
inflating O_DIRECT reads. Code inspection disproves that for the current
batched io_uring path:

- `expert_pack_iouring_copy_jobs()` uses
  `read_sz = align_up(expert_bytes, expert_pack_direct_alignment())`;
- `GGML_MOE_IO_BYTES` is printed and used by older direct paths, but it does
  not determine current batched io_uring read size;
- therefore changing `--io-bytes` is not expected to reduce current wait time.

The next diagnostic is to make summaries preserve per-ring io_uring details
from stderr. Current summary only reports aggregate expert-pack counters, while
the logs also contain separate `main` and `gate` staging ring lines. Comparing
`offset-sort-n84`, `ctx512-offset-sort-n84`, and `ctx256-offset-sort-n84` should
show whether wait cost regresses primarily in up/gate, down, or both.

### Phase 9 Progress

- 2026-06-12 23:18 CST: Extended
  `.Agent/plans/5090-theoretical-token-rate/run_5090_matrix.py` to parse
  per-ring pinned-staging io_uring details, and added
  `.Agent/plans/5090-theoretical-token-rate/compare_runs.py`.
  Comparison saved to `wait-cost-compare.json`.
  - `offset-sort-n84`: `1.92 eval tok/s`, `62582` reads,
    `wait_us=50943841`, `inflight_avg=2.38`.
  - `ctx512-offset-sort-n84`: `1.93 eval tok/s`, `62129` reads,
    `wait_us=47897342`, `inflight_avg=2.37`.
  - `ctx256-offset-sort-n84`: `1.78 eval tok/s`, `61508` reads,
    `wait_us=53633437`, `inflight_avg=2.35`.
  - Per-ring batch shape is broadly similar, so the `ctx256` regression is not
    explained by obvious batch fragmentation. It may be read-position/cache
    content effects or run-to-run variation.
- 2026-06-12 23:19 CST: Before treating `ctx512` as a real decode improvement,
  repeat the current best `offset-sort-n84` once under the same strict config to
  estimate run-to-run noise. Do not claim a new best unless the repeat confirms
  the margin.
- 2026-06-12 23:20 CST: Repeated the current strict baseline:
  - `offset-sort-n84-repeat1`: `1.91 eval tok/s`, `58.1%` VRAM hit, `62582`
    io_uring reads, `read_failures=0`, `total_ms=63851.16`.
  - This shows the `1.92` vs `1.93` decode difference is within run-to-run
    noise. Keep reporting the robust current line as about `1.9 tok/s`, not as
    a meaningful `1.93` breakthrough.

## Phase 10: Conservative Expert Reduction

I/O/cache tuning has not moved the 5090 close to the theoretical `~4-5 tok/s`
target. The only prior runs that approached that range used SER, but the
outputs were visibly degraded. Current SER semantics:

- `--smart-expert-reduction MIN,THRESH` uses `ggml_top_k_thresh`;
- it always keeps at least `MIN` experts;
- additional experts are kept only if their routing score is at least
  `THRESH * max_score`.

Previous tested SER settings were too aggressive (`min=1`, `4`, `6`) and were
not run on the latest offset-sort baseline. Phase 10 tests conservative SER as
a quality-gated diagnostic:

- start with `min=7, thresh=1.05` on n8 only;
- inspect output and counters before any long run;
- only run n84 if n8 output remains plausible and speed improves materially.

### Phase 10 Progress

- 2026-06-12 23:30 CST: Ran `ser7-105-offset-sort-n8` under the current strict
  offset-sort config (`MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`,
  oracle profile, clamped VRAM cache, `GGML_MOE_IO_SORT_OFFSET=1`).
  - Result: `1.54 eval tok/s`, `34.8%` VRAM hit, `7205` io_uring reads,
    `read_failures=0`, `total_ms=24453.97`.
  - This reduced reads versus `offset-sort-n8` (`7382`) but was slower than
    the non-SER offset-sort n8 line (`1.59 eval tok/s`).
  - The generated text was only a truncated prompt continuation, so free-form
    output is not a usable quality signal for this short diagnostic.
- Next step: use the existing 4-item CEval/CMMLU smoke set as a small quality
  gate with the current optimized config. The smoke set is not a broad
  benchmark; it only checks whether an expert-reduction candidate is obviously
  unsafe before any long n84 run. Run non-SER and `SER=7,1.05` with identical
  current offset-sort settings, then compare accuracy, reads, VRAM hit,
  eval/prompt tok/s, and total time.
- 2026-06-12 23:39 CST: Added
  `.Agent/plans/5090-theoretical-token-rate/run_smoke_current.py` so the smoke
  gate uses the same current optimized runtime settings as the n84 runner
  (`io_uring`, offset-sort, profile preload, cache auto-clamp, strict 2 GB,
  `RAM tier=0`). This avoids comparing the current work against the older
  direct-I/O 12 GB cache smoke runner.
- 2026-06-12 23:45 CST: Ran current-config 4-item smoke baseline and
  `SER=7,1.05`:

  | config | accuracy | io_uring reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s | read_failures |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | current non-SER | 3/4 | 15529 | 28.4% | 3.806 | 1.130 | 153317.44 | 159.10 | 0 |
  | current `SER=7,1.05` | 3/4 | 13168 | 30.7% | 3.883 | 1.273 | 152028.56 | 157.45 | 0 |

  Both configs answered the same three questions correctly and missed
  `cmmlu_chinese_civil_service_exam_0150` (`expected=C`, output `D`).
  `SER=7,1.05` cut io_uring reads by `15.2%` and improved aggregate eval
  token rate by `12.6%`, but wall time improved only `1.0%` because this smoke
  is dominated by repeated process/model startup and prompt work.
- Interpretation: `SER=7,1.05` is not obviously quality-unsafe on this tiny
  smoke gate, but its n8 free-form diagnostic was slower and the smoke wall
  gain is small. Run one same-prompt n84 diagnostic only to measure long-decode
  effect; do not treat it as a general benchmark.
- 2026-06-12 23:50 CST: Ran `ser7-105-offset-sort-n84` as that single
  same-prompt long-decode diagnostic.
  - Result: `1.30 eval tok/s`, `30.3%` VRAM hit, `91123` io_uring reads,
    `read_failures=0`, `total_ms=82223.19`, `wall_s=83.83`.
  - This is much worse than the non-SER `offset-sort-n84` line
    (`~1.9 eval tok/s`, `58.1%` VRAM hit, `62582` io_uring reads).
  - Interpretation: `SER=7,1.05` changes the long generated path enough that
    the same-prompt oracle profile no longer matches the access stream. It
    should not be pursued as the current theoretical-token-rate path.
- Next conservative action: retest the earlier quality-preserving
  `SER=1,0.96` candidate under the current optimized io_uring/offset-sort
  runtime. It previously preserved `3/4` smoke accuracy under the old direct
  config, but must be remeasured under the current code and cache behavior.
- 2026-06-12 23:57 CST: Ran current-config 4-item smoke for `SER=1,0.96`.
  - Result: `2/4` accuracy, `10259` io_uring reads, `30.6%` VRAM hit,
    `3.505` prompt tok/s, `1.703` eval tok/s, `162623.75 total_ms`,
    `168.17 wall_s`, `read_failures=0`.
  - It changed `ceval_computer_network_0009` from the correct `D` to `C`.
  - Interpretation: this setting is faster in decode and reduces reads by
    `33.9%` versus current non-SER smoke, but it fails the small quality gate
    under the current optimized runtime. Do not run a long n84 diagnostic for
    `SER=1,0.96`.

Phase 10 conclusion:

- Global SER is not the missing lever for the current 5090 target.
- `SER=7,1.05` preserves the tiny smoke score but regresses long n84 because
  it changes generated-route behavior and destroys the oracle-profile cache
  match.
- `SER=1,0.96` gives stronger read reduction but fails the smoke quality gate
  under the current config.
- The next credible path is not a global expert-count threshold. It should be
  route/profile/cache-aware, for example reducing only experts that are both
  low-score and cold/miss-heavy, or only applying approximation where the
  selected expert set remains profile-compatible.

## Phase 11: Profile/Cache-Aware Miss Reduction

Global SER failed for two different reasons: conservative `SER=7,1.05`
preserved the tiny smoke result but changed the long generated route stream
enough to lose oracle-profile locality, while aggressive `SER=1,0.96` reduced
reads but failed the smoke quality gate. Phase 11 therefore changes the
optimization rule:

- do not reduce experts globally by count or score threshold alone;
- first use the existing route/profile/cache traces to find whether misses are
  concentrated in low-profile or low-reuse rows;
- prefer strategies that preserve profile-hot experts and avoid changing the
  long generated route distribution;
- only implement runtime changes after an offline simulator predicts a
  meaningful miss reduction over the current strict non-SER line
  (`~62582` io_uring reads) without relying on extra host RAM;
- keep the validation line strict: `MemoryMax=2G`, `MemorySwapMax=0`,
  `RAM tier=0`, current io_uring offset-sort, same model/pack/profile.

Candidate mechanisms to evaluate offline before CUDA/runtime edits:

- cache-aware admission: bypass one-off or low-profile rows so they do not
  evict profile-hot rows;
- profile-compatible selective reduction: only drop candidates that are both
  outside the startup profile hot set and have low router score margin;
- layer-specific policy: if some layers dominate misses with little reuse,
  apply more aggressive admission/reduction there only;
- longer horizon prefetch/order grouping: reduce wait cost without changing
  selected experts, but only if trace order shows read batches can be enlarged
  without violating dependencies.

Acceptance gate:

- offline simulation must predict a material miss reduction, not just a few
  hundred reads;
- smoke accuracy must stay at the current non-SER `3/4` line before any long
  same-prompt n84 claim;
- n84 must beat the robust current non-SER line (`~1.9 eval tok/s`, `58.1%`
  VRAM hit, `~62582` io_uring reads) or explain clearly why the measured
  tradeoff is useful.

### Phase 11 Progress

- 2026-06-13 00:05 CST: Extended
  `.Agent/plans/5090-theoretical-token-rate/simulate_cache_policy.py` with a
  disabled-by-default `--ghost-admit-after` offline admission model. This
  tests whether bypassing first-touch rows can reduce cache pollution without
  changing selected experts or output.
  - `ghost_admit_after=2`: `62265` simulated misses, only `285` fewer than
    baseline `62550`.
  - `ghost_admit_after=3`: `61565` simulated misses, `985` fewer.
  - `ghost_admit_after=4`: `61430` simulated misses, `1120` fewer, but prefix
    misses regress versus `ghost=3`.
  - Pure admission is therefore too weak to explain the gap to the
    `~4-5 tok/s` target.
- 2026-06-13 00:08 CST: Added
  `.Agent/plans/5090-theoretical-token-rate/analyze_trace_reuse.py` and saved
  baseline reuse analysis to `trace-reuse-baseline.json`.
  - Replayed misses match the current simulator baseline: `62550`.
  - Miss reuse distance buckets: `1025-4096=8384`, `gt4096=37979`,
    `never=16187`; there are no short-distance miss reuses in the current
    bucketization.
  - Prefix miss reuse buckets: `1025-4096=2449`, `gt4096=3079`,
    `never=1854`.
  - Interpretation: most misses are not caused by short-term cache pollution.
    Cache admission/replacement tweaks can help at the margin but cannot move
    the run from `~1.9 tok/s` to the theoretical target.

Updated direction from Phase 11:

- Keep `lfu_lru` as default. Do not implement runtime ghost admission yet; the
  predicted gain is too small.
- Shift the next implementation check from "fewer misses through replacement"
  to "cheaper miss path": inspect whether expert-pack offsets within a batch
  are adjacent enough to coalesce multiple O_DIRECT/io_uring reads into fewer
  larger sequential reads, then scatter H2D into the existing cache slots.
- 2026-06-13 00:16 CST: Added
  `.Agent/plans/5090-theoretical-token-rate/analyze_pack_coalesce.py` for
  read-only expert-pack index parsing and offset coalescing analysis. Initial
  version incorrectly assumed a 96-byte tensor field and tried to read the
  entire pack; fixed it to stream only the pack index and use the actual
  `expert_pack_entry` layout (`tensor[128]`, `int32`, padding, `offset`,
  `nbytes`).
  - Saved result: `pack-coalesce-g8.json`.
  - On the full same-prompt trace, grouping by 8 same-size route events gives
    `149424` jobs and `143481` contiguous offset runs.
  - Ideal read reduction is only `5943` / `3.98%` over all trace jobs, and this
    is an upper bound before cache hits are removed. Actual runtime miss reads
    would benefit less.
  - Interpretation: implementing large-read coalescing plus scatter-H2D would
    add complexity for a small upper-bound gain. Keep offset sorting as the
    practical low-risk win; do not implement coalesced scatter reads yet.

Phase 11 conclusion:

- Cache replacement/admission and offset coalescing both look marginal under
  current evidence.
- The current bottleneck is not an obvious missed local reuse or adjacent-read
  packing problem. Remaining progress toward the theoretical 5090 target
  likely requires changing dependency timing: load future experts earlier with
  a predictor that does not add many wasted reads, or restructure execution so
  more miss reads are in flight across layer/token boundaries.

## Phase 12: Dependency-Timed Prefetch

Phase 12 tests whether the miss path can be started earlier without changing
the selected experts. Previous trace-prefetch attempts were too blunt and
increased reads or wait cost. The narrower hypothesis now is:

- keep selected experts unchanged;
- keep cache policy and VRAM residency unchanged;
- only prefetch experts that the runtime will very likely request soon;
- start the read earlier than the blocking runtime load, but avoid filling the
  cache with far-future rows that evict useful rows.

Implementation principles:

- use env-gated code only;
- first inspect the existing runtime job formation and trace-prefetch hooks;
- prefer a small short-window prefetch that uses the same expert-pack/io_uring
  path and current cache insertion, so it can be disabled cleanly;
- validate with strict n8 before any n84 run.

Acceptance gate:

- strict n8 must not regress versus `offset-sort-n8` (`1.59 eval tok/s`,
  `7382` reads, `41.5%` VRAM hit) unless diagnostics clearly show why a long
  n84 run is still useful;
- wasted reads must be visible in counters/logs;
- no `read_failures`;
- keep `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`, io_uring offset-sort,
  same profile/cache settings.

### Phase 12 Progress

- 2026-06-13 00:24 CST: Reviewed the existing trace-prefetch path and previous
  Phase 3 logs. The best safe no-evict variants still regressed on n8:
  `trace-prefetch-batch-noevict-w12-l1-n8` and `w24-l4` were both about
  `1.39 eval tok/s` versus the later offset-sort n8 line at `1.59 tok/s`.
  The logs show `async prefetch waits` equal to the number of loaded prefetch
  rows, which means rows were often demanded before the prefetch stream had
  completed. The prior implementation was too near-term.
- 2026-06-13 00:27 CST: Added an env-gated lead offset to the existing
  trace-prefetch diagnostic:
  - env: `GGML_MOE_TRACE_PREFETCH_LEAD_EVENTS`;
  - runner flag: `--trace-prefetch-lead-events`;
  - default: `0`, preserving old behavior;
  - behavior: prefetch starts at `cursor + lead_events` and scans
    `lead_events + window`, so diagnostic reads can be launched further ahead
    of demand.
- Build/validation:
  - `python3 -m py_compile` passed for the task scripts;
  - `git diff --check` passed;
  - `cmake --build build-cuda --target llama-cli -j 8` passed.
- 2026-06-13 00:34 CST: Ran two strict n8 diagnostics with the new lead
  parameter, both with current io_uring offset-sort, strict 2 GB host RAM,
  `RAM tier=0`, oracle profile, and `GGML_MOE_TRACE_PREFETCH_WINDOW=24`,
  `MAX_LOADS=4`.

  | config | eval tok/s | VRAM hit | io_uring reads | iouring wait us | async waits | total_ms | read_failures |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `lead256-w24-l4-n8` | 1.26 | 43.9% | 7257 | 6865727 | 174 | 30073.96 | 0 |
  | `lead1024-w24-l4-n8` | 1.47 | 41.5% | 7534 | 5329791 | 152 | 27726.57 | 0 |

  Interpretation:
  - The lead parameter works mechanically: `async prefetch waits` dropped from
    the old no-evict w24 line's `330` to `174` / `152`.
  - It still does not beat `offset-sort-n8` (`1.59 eval tok/s`, `7382` reads,
    `41.5%` VRAM hit). `lead256` improves hit rate but fragments/worsens the
    wait path; `lead1024` returns to the same hit rate and remains slower.
  - Do not run n84 for this fixed-lead trace-prefetch path. The issue is no
    longer merely that prefetch starts too late; fixed-distance future rows do
    not align with useful cache residency and runtime batch shape well enough.

Phase 12 conclusion:

- Fixed lead-events trace prefetch is a useful diagnostic but not a solution.
- The next implementation direction must either predict exact near-future
  misses with cache residency awareness, or restructure the runtime loader to
  overlap the already-known current-layer loads with independent compute. More
  trace-prefetch parameter sweeps are unlikely to close the gap.

## Phase 13: Current-Miss / Compute Overlap

The remaining strict 2 GB path is still dominated by blocking runtime loads.
Phase 13 focuses on overlap inside the current layer call rather than
future-trace prediction:

- inspect the existing `UP/GATE_PARALLEL_STAGE` and `DOWN_PARALLEL_STAGE`
  paths to see whether they actually overlap I/O with independent GPU compute
  or only split staging work;
- add diagnostics before changing behavior if current timing is opaque;
- prefer a small env-gated reordering that starts known current miss reads as
  early as possible and then performs independent quantize/compute while those
  reads are in flight;
- keep selected experts, cache policy, and profile residency unchanged.

Acceptance gate:

- strict n8 must beat or at least match `offset-sort-n8` (`1.59 eval tok/s`,
  `7382` reads, `41.5%` VRAM hit) before any n84 run;
- no extra read failures and no OOM;
- if a diagnostic shows there is no independent compute window to overlap,
  record that explicitly and do not add complex code.

### Phase 13 Progress

- 2026-06-13 00:43 CST: Inspected the current up/gate and down staging code.
  - Up/gate `parallel_stage` already overlaps up and gate staging via separate
    CPU threads/streams. Gate staging can run while up compute starts.
  - Down `DOWN_PARALLEL_STAGE` splits down staging across two CPU threads and
    streams, but the down compute begins only after the down staging jobs have
    completed.
  - The likely question is not whether the mechanism exists, but whether there
    is enough independent GPU compute to hide the expert-pack read latency.
- 2026-06-13 00:49 CST: Ran a profile-only strict n8 diagnostic:
  `offset-sort-profile-n8` in
  `.Agent/plans/5090-theoretical-token-rate/runs/phase13-overlap-profile/`.
  This enables `GGML_MOE_BATCH_PROFILE=1` and keeps the same offset-sort,
  strict 2 GB, `RAM tier=0`, oracle-profile config.
  - It preserves the same read/hit shape as `offset-sort-n8`: `7382`
    io_uring reads, `41.5%` VRAM hit, `read_failures=0`.
  - Profiling overhead slows eval to `1.27 tok/s`, so use the profile only for
    phase proportions, not token-rate comparison.
  - Up/gate profile:
    `calls=526`, `avg_active=8.00`, `stage=0.044 ms`, `quant=0.001 ms`,
    `up_wait=5.061 ms`, `gate_wait=5.320 ms`, `up_compute=0.147 ms`,
    `gate_compute=0.121 ms`, `kernel=5.470 ms`, `total=5.515 ms/call`.
  - Down profile:
    `calls=526`, `avg_active=8.00`, `stage=2.607 ms`, `quant=0.008 ms`,
    `kernel=0.330 ms`, `total=2.973 ms/call`.

Phase 13 conclusion:

- Current-stage reordering has very limited headroom. Up/gate already overlaps
  the two staging paths, but useful compute is only about `0.1-0.15 ms` while
  each call waits about `5 ms` for staging. Down compute is about `0.33 ms`
  against `2.6 ms` staging.
- Do not add a complex same-layer scheduling rewrite yet; the profile says
  there is not enough independent GPU work to hide the current miss latency.
- The remaining high-leverage options are outside simple reordering:
  materially fewer runtime reads, materially cheaper reads, or more expert
  residency. Under strict `RAM tier=0`, the earlier evidence says read count
  and residency are already near this design's limits.

## Phase 6: Offline Cache Policy Simulation

Before changing runtime cache admission again, simulate policies against
`oracle-n84.trace.csv` and the exact oracle profile. The goal is to find a
policy that reduces long n84 misses without adding prefetch reads or degrading
profile residency.

Candidate policies:

- reproduce current profile-pinned LFU/LRU behavior approximately;
- no-admit or bypass for rows with low remaining frequency;
- frequency-threshold admission based on same-prompt profile counts;
- protect top profile rows from runtime eviction while allowing lower profile
  rows to age out;
- split-specific thresholds for up/gate and down caches.

Only implement a runtime policy if the simulator predicts fewer misses than the
current best strict non-SER line (`62582` io_uring reads / runtime misses) under
the same approximate slot budgets.

## Phase 14: 5060 Ti RAM-Tier Diagnostic

The accepted 5060 Ti presets were not strict `RAM tier=0` runs:

- `wici-glm51-interactive-n84`: `GGML_MOE_RAM_TIER_MIB=3072`,
  `GGML_MOE_RAM_TIER_SKIP=447`, `N_GPU_LAYERS=60`, direct I/O,
  `GGML_MOE_STAGE_PINNED=1`, route profile, `VRAM cache=1536 MiB`.
- `wici-glm51-0p97`: `GGML_MOE_RAM_TIER_MIB=4096`,
  `GGML_MOE_RAM_TIER_SKIP=500`, `N_GPU_LAYERS=60`, direct I/O,
  route profile, `VRAM cache=2048 MiB`.

Therefore "reuse 5060 Ti optimizations" has two different meanings:

- strict 5090 comparison: keep `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- non-strict diagnostic: re-enable the 5060 Ti RAM tier and pinned staging to
  measure how much of the 5060 result came from host-RAM residency rather than
  raw GPU speed.

Phase 14 will run only when GPU and host RAM are idle. It must not stop any
other process. The first run is a short `n8` diagnostic with:

- `MemoryMax=8G`, `MemorySwapMax=0`;
- `GGML_MOE_RAM_TIER_MIB=3072`, `GGML_MOE_RAM_TIER_SKIP=447`;
- route profile first, because that matches the 5060 Ti accepted n84 preset;
- current 5090 runtime flags retained: io_uring offset-sort, GPU handoff,
  prompt dynamic experts, auto-clamped VRAM cache, split up/gate/down staging.

Acceptance gate before n84:

- no OOM and `read_failures=0`;
- RAM tier must initialize and report non-zero RAM hit;
- short n8 eval rate or total runtime must materially improve over the strict
  comparable line. If not, do not spend a long n84 run on this path.

### Phase 14 Progress

- 2026-06-13 01:24 CST: Ran the first non-strict 5060 Ti RAM-tier diagnostic
  with `MemoryMax=8G`, `RAM tier=3072 MiB`, `RAM_TIER_SKIP=447`, route profile,
  and `STAGE_PINNED=1`.
  - `ramtier3072-route-n8`: `1.21 eval tok/s`, `total_ms=29255.13`,
    `VRAM hit=21.9%`, `io_uring_reads=9859`, `read_failures=0`,
    `RAM tier hit=25.2%`.
  - This is slower than the strict oracle/offset-sort n8 line and uses a worse
    route-profile residency shape.
- 2026-06-13 01:28 CST: Re-ran with the same oracle profile as the strict n8
  baseline to isolate RAM tier.
  - `ramtier3072-oracle-n8`: `1.49 eval tok/s`, `total_ms=27006.41`,
    `VRAM hit=41.5%`, `io_uring_reads=7382`, `read_failures=0`.
  - Disabling explicit `STAGE_PINNED=1` in
    `ramtier3072-oracle-n8-nopinned` reached only `1.51 eval tok/s` and still
    had `io_uring_reads=7382`.
- 2026-06-13 01:37 CST: Added a runtime fix so the batch io_uring path first
  tries RAM-tier H2D for each job and only submits the remaining jobs to SSD.
  This preserves strict `RAM tier=0` behavior.
  - Build passed: `cmake --build build-cuda --target llama-cli -j 8`.
  - `ramtier3072-oracle-n8-nopinned-ramfilter` still had
    `io_uring_reads=7382` and `1.52 eval tok/s`.
  - The new counter shape showed the real issue: RAM tier was tried for all
    `10360` pack hits, but `skip=447` only hit `750` entries that overlap the
    profile preload/direct-read phase, not the runtime io_uring misses.
- 2026-06-13 01:43 CST: Tested alternate RAM tier placement.
  - `RAM_TIER_SKIP=0`: `1.45 eval tok/s`, `total_ms=29840.02`,
    `io_uring_reads=7382`, `RAM tier hits=750`; worse because it duplicates
    the VRAM-preloaded hottest rows.
  - `RAM_TIER_SKIP=2978`: `1.58 eval tok/s`, `total_ms=25242.76`,
    `io_uring_reads=6845`, `iouring_wait_us=7153048`, `RAM tier hits=606`.
    This is the first configuration that actually reduces runtime SSD reads,
    but token rate only ties the strict n8 baseline (`1.59 eval tok/s`) and
    does not clear the n84 acceptance gate.

Phase 14 conclusion:

- The accepted 5060 Ti RAM tier is not directly reusable on 5090 with a large
  20 GB VRAM cache. The old `skip=447`/`skip=0` placements mostly duplicate
  rows that the 5090 already preloads into VRAM.
- Skipping past the 5090 VRAM-preloaded rows (`skip≈2978`) makes RAM tier hit
  actual runtime misses and reduces n8 io_uring reads by `537` (`7382 -> 6845`),
  but it does not materially improve eval token rate. Do not run the long n84
  diagnostic until RAM-tier placement predicts a larger read reduction.
- Next useful work is an offline planner that chooses RAM-tier rows from the
  residual miss set after VRAM profile preload, instead of a fixed skip count.
  That planner should estimate read reduction before any new long GPU run.

## Phase 15: Residual-Miss RAM Tier Planner

Phase 15 makes RAM tier useful for the 5090 shape instead of copying the old
5060 Ti fixed-skip heuristic.

Hypothesis:

- The 5090 VRAM cache already preloads the hottest profile rows.
- RAM tier should therefore be filled from rows that remain frequent in the
  route trace after simulated VRAM preload/cache behavior, not from a fixed
  rank offset.
- A separate RAM-tier profile is needed because the best VRAM ordering and best
  RAM ordering are different once the 5090 cache is large.

Implementation plan:

1. Add an offline planner that replays `oracle-n84.trace.csv` with the same
   approximate split cache budgets used by the strict oracle run.
2. Count residual misses by `(tensor, expert, bytes)` and write a RAM-tier CSV
   ordered by predicted miss reduction per byte.
3. Add an env-gated runtime input `GGML_MOE_RAM_TIER_PROFILE`; when set, RAM
   tier loads from that profile instead of `GGML_MOE_VRAM_PROFILE`.
4. Keep `GGML_MOE_RAM_TIER_SKIP` working for compatibility, but use `skip=0`
   for planner-generated residual profiles.
5. Validate on short non-strict n8 only. Do not run n84 unless n8 shows a
   material read reduction and token-rate or total-runtime gain over
   `ramtier3072-skip2978-oracle-n8-ramfilter`.

Acceptance gate:

- build passes;
- strict `RAM tier=0` behavior is unchanged by default;
- non-strict `n8` with residual RAM profile has `read_failures=0`;
- `io_uring_reads` drops materially below `6845`, and either `eval tok/s`
  improves beyond the strict `1.59` line or `total_ms` improves enough to make
  an n84 diagnostic worthwhile.

### Phase 15 Progress

- 2026-06-13 02:05 CST: Added
  `.Agent/plans/5090-theoretical-token-rate/plan_ram_tier_profile.py`.
  It replays `oracle-n84.trace.csv` with the same approximate VRAM cache slots
  as the current oracle run (`upgate=2282`, `down=1028`, `reserve=10`) and
  ranks residual misses by miss reduction per byte.
  - Output profile:
    `.Agent/plans/5090-theoretical-token-rate/ramtier-residual-3072.route.csv`.
  - Summary:
    `.Agent/plans/5090-theoretical-token-rate/ramtier-residual-3072.summary.json`.
  - Offline prediction: `3072 MiB` RAM tier can hold `767` entries and cover
    `10626 / 62550` residual misses (`16.988%`) on the full n84 trace.
- 2026-06-13 02:10 CST: Added runtime support for
  `GGML_MOE_RAM_TIER_PROFILE`. When set, RAM tier loads from this independent
  profile instead of `GGML_MOE_VRAM_PROFILE`. This keeps VRAM and RAM tier
  ordering separate and preserves old behavior when the new env is unset.
- 2026-06-13 02:14 CST: Added `--ram-tier-profile` to
  `run_5090_matrix.py` and added RAM-tier metric parsing
  (`ram_tier_hits`, `ram_tier_total`, `ram_tier_hit_rate_pct`,
  `ram_tier_resident_mib`) to summaries.
- 2026-06-13 02:20 CST: First residual-profile n8 attempt showed
  `RAM tier profile preload: loaded 767 entries` but `RAM tier: loaded 0
  entries`. Root cause: Python `csv.writer` emitted CRLF and the C `sscanf`
  `%[^\n]` captured trailing `\r` in tensor names, so expert-pack lookup did
  not match.
  - Fixed planner output to use LF line endings.
  - Hardened C profile parsing by trimming trailing `\r` / `\n`.
  - Expanded profile/trace tensor buffers from `96` to `128` to match
    `expert_pack_entry::tensor`.
  - Build passed after the fix:
    `cmake --build build-cuda --target llama-cli -j 8`.
- 2026-06-13 02:33 CST: The 5090 was occupied by another
  `sglang.launch_server` task using about `22024 MiB` VRAM. Per `Agent.md`, I
  did not stop it and waited for it to finish naturally before running the GPU
  diagnostic.
- 2026-06-13 02:42 CST: Ran fixed non-strict n8 residual RAM-tier diagnostic:
  `residual-ramtier3072-oracle-n8-fixed`.
  - Config: `MemoryMax=8G`, `MemorySwapMax=0`, `RAM tier=3072 MiB`,
    `GGML_MOE_RAM_TIER_PROFILE=ramtier-residual-3072.route.csv`,
    `RAM_TIER_SKIP=0`, VRAM profile still `oracle-n84.route.csv`.
  - Result: `1.64 eval tok/s`, `VRAM hit=41.5%`,
    `io_uring_reads=6789`, `read_failures=0`, `RAM tier hits=593`,
    `RAM tier hit=5.7%`, `total_ms=28997.27`.
  - This beats the strict n8 decode line (`1.59 eval tok/s`) and reduces
    runtime io_uring reads below `6845`, so it clears the gate for one n84
    diagnostic.
- 2026-06-13 02:45 CST: Ran fixed non-strict n84 residual RAM-tier diagnostic:
  `residual-ramtier3072-oracle-n84`.
  - Result: `2.22 eval tok/s`, `total_ms=60635.02`,
    `VRAM hit=58.1%`, `io_uring_reads=51956`, `read_failures=0`,
    `RAM tier hits=10626`, `RAM tier hit=16.2%`, `resident=3072 MiB`.
  - Comparison to strict best/repeat:
    - `offset-sort-n84`: `1.92 eval tok/s`, `62582` reads,
      `total_ms=64918.76`;
    - `offset-sort-n84-repeat1`: `1.91 eval tok/s`, `62582` reads,
      `total_ms=63851.16`;
    - residual RAM tier: `2.22 eval tok/s`, `51956` reads,
      `total_ms=60635.02`.
  - Decode rate improved about `15-16%` and runtime reads dropped by exactly
    the offline-predicted `10626` misses. This is the strongest current 5090
    result, but it is explicitly non-strict because it uses a 3 GiB host RAM
    tier.

Phase 15 conclusion:

- Residual RAM-tier planning works and closes part of the gap:
  `~1.9 tok/s -> 2.22 tok/s`.
- The result is still well below the earlier GPU-scaled theoretical target
  (`~3.9-5.5 tok/s`). The remaining bottleneck is the large residual SSD miss
  set (`51956` reads) plus small batch/io dependency shape.
- Next promising directions:
  - test a larger non-strict RAM tier budget if host RAM is available and no
    other task is active, using the same residual planner (`4096`, `6144`,
    possibly `8192 MiB`);
  - split residual RAM tier by upgate/down or prefix-vs-decode utility to
    improve early token speed;
  - consider an analogous residual planner for additional VRAM if model/graph
    headroom can be recovered.

## Phase 16: Larger Residual RAM Tier Budgets

Phase 16 extends the residual planner to larger non-strict host-RAM budgets.
This does not change the strict 2 GB line; it is a diagnostic for how much of
the gap to the theoretical 5090 target can be closed by eliminating more SSD
runtime misses.

Constraints:

- Do not run GPU experiments while another task owns the 5090 or host RAM.
- Keep all generated profiles and summaries in this task directory.
- Treat all RAM-tier budgets above `0` as non-strict.

Plan:

1. Generate residual RAM-tier profiles for `4096`, `6144`, and `8192 MiB` using
   the same replay settings as Phase 15.
2. Compare predicted covered misses and coverage per GiB.
3. Only run GPU diagnostics for budgets that predict a meaningful read
   reduction beyond the proven `3072 MiB` line (`10626` covered misses).
4. Start with n8 if resources are tight; run n84 only after the short line
   confirms the RAM-tier profile loads and reduces reads.

### Phase 16 Progress

- 2026-06-13 03:00 CST: Current machine resources were occupied again by an
  external `sglang.launch_server` task using about `20 GiB` VRAM and high host
  RAM/swap. Per `/home/wici/lfz/Agent.md`, I did not stop it and limited work
  to offline planning.
- Generated larger residual RAM-tier profiles:
  - `.Agent/plans/5090-theoretical-token-rate/ramtier-residual-4096.route.csv`
  - `.Agent/plans/5090-theoretical-token-rate/ramtier-residual-6144.route.csv`
  - `.Agent/plans/5090-theoretical-token-rate/ramtier-residual-8192.route.csv`

Offline prediction table:

| RAM tier MiB | entries | selected MiB | covered misses | covered miss % | covered prefix misses | predicted io_uring reads |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3072 | 767 | 3069.656 | 10626 | 16.988% | 593 | 51924 |
| 4096 | 1018 | 4095.938 | 13664 | 21.845% | 803 | 48886 |
| 6144 | 1512 | 6141.750 | 19147 | 30.611% | 1104 | 43403 |
| 8192 | 2006 | 8188.312 | 23857 | 38.141% | 1558 | 38693 |
| 10240 | 2513 | 10237.594 | 28003 | 44.769% | 1974 | 34547 |
| 12288 | 3011 | 12286.031 | 31659 | 50.614% | 2254 | 30891 |

Interpretation:

- The 8 GiB residual tier has the strongest predicted read reduction:
  `62550 -> ~38693` residual reads, about `38%` coverage.
- Marginal predicted read reductions remain worthwhile but diminish:
  - `0 -> 3 GiB`: `10626` fewer reads, about `3542/GiB`;
  - `3 -> 4 GiB`: `3038` fewer reads, about `3038/GiB`;
  - `4 -> 6 GiB`: `5483` fewer reads, about `2742/GiB`;
  - `6 -> 8 GiB`: `4710` fewer reads, about `2355/GiB`.
  - `8 -> 10 GiB`: `4146` fewer reads, about `2073/GiB`;
  - `10 -> 12 GiB`: `3656` fewer reads, about `1828/GiB`.
- Because the machine has only about `15 GiB` host RAM and other tasks may run
  concurrently, the next GPU experiment must be chosen based on resource state:
  run the 8 GiB n8 diagnostic only if host RAM is clearly available and no
  external task is active; otherwise use 6 GiB as the safer next diagnostic.
- Do not run n84 for 6/8 GiB until n8 confirms the profile actually loads and
  read reductions track the offline prediction.
- As of the latest check, an external `sglang.launch_server` task was still
  active and using the 5090/host RAM, so no new GPU run was started.
- 10/12 GiB profiles are useful as upper-bound planning artifacts, but 12 GiB
  is too risky to run on this 15 GiB host while any other task is active. It
  would leave little room for model/runtime allocations and page cache.
- 2026-06-13 03:18 CST: The external task ended and resources briefly became
  available, so I ran `residual-ramtier8192-oracle-n8`.
  - Config: `MemoryMax=12G`, `RAM tier=8192 MiB`,
    `GGML_MOE_RAM_TIER_PROFILE=ramtier-residual-8192.route.csv`,
    `VRAM cache=20480 MiB`, `safety=256`.
  - Result: `1.82 eval tok/s`, `io_uring_reads=5824`,
    `RAM tier hits=1558`, `RAM tier hit=15.0%`, `read_failures=0`.
  - This confirms the 8 GiB residual profile loads and follows the offline
    prefix prediction, but startup/prompt total time is slower because loading
    and registering 8 GiB of host RAM is expensive.
- 2026-06-13 03:22 CST: Tried `residual-ramtier8192-oracle-n84` with the same
  8 GiB RAM tier. It failed before timings with CUDA graph launch OOM:
  `cudaGraphLaunch(...): out of memory`.
  - Auto-clamp saw only `13028 MiB` free after model load and allocated about
    `8.1 GiB` upgate plus `4.4 GiB` down cache, leaving insufficient graph
    headroom.
- 2026-06-13 03:25 CST: Tried a more conservative n8 headroom variant:
  `residual-ramtier8192-vram18432-safety768-n8`
  (`VRAM cache=18432`, `safety=768`). It still failed with the same CUDA graph
  OOM after allocating about `8.2 GiB` upgate plus `4.4 GiB` down cache.
- After that, a new external `sglang.launch_server` task started and occupied
  the 5090/host RAM again, so no further GPU experiments were started.

Updated Phase 16 direction:

- The 8 GiB residual profile is useful for decode when it runs, but it needs
  stricter VRAM headroom control before n84 can be measured.
- Next GPU attempt, when resources are free, should either:
  - use the newly exposed runner flag `--cache-graph-reserve-mib` to set
    `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB` and reserve graph headroom during
    auto-clamp;
  - use `--no-graph-reuse` as a diagnostic to bypass graph capture/launch OOM;
  - or reduce the effective VRAM cache much more aggressively, likely
    `VRAM cache <= 12 GiB` or a lower upgate/down split, then re-run n8 before
    n84.
- 2026-06-13 03:31 CST: Exposed
  `--cache-graph-reserve-mib` in
  `.Agent/plans/5090-theoretical-token-rate/run_5090_matrix.py`; static checks
  passed (`python3 -m py_compile`, `git diff --check`). No GPU run was started
  because an external `sglang.launch_server` task was active.
- 2026-06-13 03:39 CST: Resources briefly became available and I ran
  `residual-ramtier8192-graphreserve2048-n8`.
  - Config: `RAM tier=8192 MiB`, residual RAM profile, `MemoryMax=12G`,
    `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=2048`, `safety=256`.
  - Result: `1.84 eval tok/s`, `io_uring_reads=6215`,
    `RAM tier hits=1558`, `RAM tier hit=15.1%`, `read_failures=0`.
  - Graph reserve fixed the CUDA graph OOM. Effective VRAM cache was clamped
    to `11450 MiB` (`7.3 GiB` upgate, `3.9 GiB` down), so VRAM hit dropped to
    `38.4%`; even so, decode speed stayed slightly above the no-reserve 8 GiB
    n8 line (`1.82 tok/s`).
- 2026-06-13 03:43 CST: A new external `sglang.launch_server` task started
  before the n84 follow-up. Per `Agent.md`, I did not stop it and did not start
  the n84 run. The next pending run is the same graph-reserve config at n84.
- 2026-06-13 00:54 CST: Resumed after resource recheck. GPU usage was only
  `41 MiB`, no matching `sglang.launch_server` / `llama-cli` / matrix-runner
  process was active, and host available RAM was about `12 GiB`. Starting the
  pending non-strict 8 GiB residual RAM-tier n84 follow-up:
  `residual-ramtier8192-graphreserve2048-n84`.
  - This is not a strict 2 GB result: it uses `MemoryMax=12G` and
    `GGML_MOE_RAM_TIER_MIB=8192`.
  - Config keeps the residual RAM-tier profile, oracle VRAM profile,
    io_uring offset sort, cache auto-clamp, `safety=256`, and
    `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=2048`.
- 2026-06-13 00:55 CST: `residual-ramtier8192-graphreserve2048-n84` completed
  successfully.
  - Result: `2.44 eval tok/s`, `1.05 prompt tok/s`, `total_ms=59345.24`,
    `wall_s=61.24`.
  - Reads/cache: `io_uring_reads=46272`, `direct_reads=2526`,
    `read_failures=0`, `VRAM hit=53.1%`.
  - RAM tier: `23857` hits / `72655` total (`32.8%`), resident
    `8192 MiB`.
  - Effective graph-reserved VRAM cache was smaller than the strict best:
    upgate `7.3 GiB` plus down `3.9 GiB` (`2808` total slots), versus the
    strict best's larger `~13.5 GiB` cache. This explains why VRAM hit falls
    below the strict oracle line (`58.1%`) even though SSD reads still drop.
  - Comparison:
    - strict best/repeat: `1.91-1.92 tok/s`, `62582` io_uring reads,
      `RAM tier=0`;
    - 3 GiB residual RAM tier: `2.22 tok/s`, `51956` io_uring reads,
      `10626` RAM-tier hits;
    - 8 GiB residual RAM tier with graph reserve: `2.44 tok/s`, `46272`
      io_uring reads, `23857` RAM-tier hits.
  - Interpretation: larger residual RAM tier continues to help decode, but
    the gain is sublinear because reserving graph headroom reduces VRAM
    residency and because remaining reads still have low batching
    (`inflight_avg=1.54`, `inflight_max=4`). This is still far below the
    theoretical `~3.9-5.5 tok/s` target and remains a non-strict result.
- 2026-06-13 00:56 CST: Ran an offline residual RAM-tier replan using the
  actual graph-reserved cache slot counts from the successful n84 run
  (`upgate=1936`, `down=872`, reserve `10%`).
  - New profiles:
    - `ramtier-residual-graphreserve2048-8192.route.csv`
    - `ramtier-residual-graphreserve2048-10240.route.csv`
  - Prediction with graph-reserved VRAM slots:
    - 8 GiB: `70093` residual misses, `27185` covered misses,
      `1514` covered prefix misses;
    - 10 GiB: `70093` residual misses, `31832` covered misses,
      `2074` covered prefix misses.
  - The successful 8 GiB run used a RAM-tier profile planned for the larger
    strict-cache slot shape and measured `23857` RAM-tier hits. The
    graph-reserve-aware 8 GiB profile predicts about `3328` additional covered
    misses at the same host RAM budget, so it is a lower-risk next GPU
    diagnostic than jumping directly to a 10 GiB tier.
- 2026-06-13 00:57 CST: Before launching the graph-reserve-aware 8 GiB
  validation, resources were checked again. A new external
  `sglang.launch_server` task was active under `/home/wici/lfz/ktransformers`;
  GPU memory was still low at that instant, but the process may allocate more
  GPU or host RAM. Per `/home/wici/lfz/Agent.md`, no new GPU/RAM experiment was
  started and the external process was not stopped.
- 2026-06-13 00:58 CST: Offline comparison found that the successful
  `residual-ramtier8192-graphreserve2048-n84` run still used
  `--up-gate-stage-split`. Earlier strict Phase 7 evidence showed split-stage
  reduces io_uring batching on this workload. The new run also shows that
  shape: `inflight_avg=1.54`, `inflight_max=4`, no `5-8` batches, versus the
  strict no-split offset-sort line's `inflight_avg=2.38`, `inflight_max=8`,
  and `3475` batches in the `5-8` bucket.
  - Next validation should not add more RAM first. It should run the
    graph-reserve-aware 8 GiB residual profile without up/gate split-stage, so
    it combines the better RAM-tier placement with the better batching shape.
- 2026-06-13 01:00 CST: External `sglang.launch_server` ended naturally.
  GPU usage was back to `41 MiB`, no matching workload process was active, and
  host available RAM was about `12 GiB`. Starting short non-strict validation:
  `residual-ramtier8192-graphprofile-nosplit-n8`.
  - Changes versus the previous successful graph-reserve n8/n84:
    use `ramtier-residual-graphreserve2048-8192.route.csv` and do not enable
    up/gate split-stage.
- 2026-06-13 01:01 CST: `residual-ramtier8192-graphprofile-nosplit-n8`
  completed.
  - Result: `1.68 eval tok/s`, `1.07 prompt tok/s`, `total_ms=32365.92`,
    `wall_s=34.35`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=6259`, `VRAM hit=38.4%`,
    `RAM tier hits=1514` (`14.7%`), resident `8192 MiB`.
  - Batching improved versus the previous split-stage graph-reserve n8:
    `inflight_avg=2.50`, `inflight_max=8`, `396` batches in `5-8`, and
    `iouring_wait_us=4481868`, versus split-stage's `inflight_avg=1.75`,
    `inflight_max=4`, no `5-8` batches, and `wait_us=6432350`.
  - However, the graph-reserve-aware profile covers fewer prefix misses than
    the old profile (`1514` vs `1558`), and the n8 decode rate is lower
    (`1.68` vs `1.84`) despite better total time. Do not treat this profile as
    a proven win yet.
  - Better isolation for the next long run: keep the old validated 8 GiB
    residual profile and disable up/gate split-stage. That tests whether the
    `2.44 tok/s` n84 result was held back by split-stage batching without
    changing RAM-tier contents.
- 2026-06-13 01:02 CST: Resources remained idle (`41 MiB` GPU used, about
  `12 GiB` host RAM available, no matching external workload process).
  Starting `residual-ramtier8192-oldprofile-nosplit-n84`, a non-strict
  isolation run that keeps the already-validated
  `ramtier-residual-8192.route.csv` profile and disables up/gate split-stage.
- 2026-06-13 01:04 CST:
  `residual-ramtier8192-oldprofile-nosplit-n84` completed.
  - Result: `2.45 eval tok/s`, `0.81 prompt tok/s`, `total_ms=65762.20`,
    `wall_s=67.45`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=46272`, `VRAM hit=53.1%`,
    `RAM tier hits=23857` (`32.8%`), resident `8192 MiB`.
  - Compared with split-stage `residual-ramtier8192-graphreserve2048-n84`:
    read counts and RAM hits are identical, batching improves
    (`inflight_avg=2.06`, `inflight_max=8`, `1690` `5-8` batches), and
    `iouring_wait_us` drops from `46761771` to `32318432`.
  - Decode rate improves only from `2.44` to `2.45 tok/s`, and total/wall time
    are worse because load and prompt timings were slower in this run. Treat
    no-split as a useful batching cleanup, not a major performance lever by
    itself.
- 2026-06-13 01:05 CST: Resources were still idle. Starting
  `residual-ramtier8192-graphprofile-nosplit-n84`, keeping the same non-strict
  8 GiB RAM-tier budget and no-split batching, but switching RAM tier to the
  graph-reserve-aware residual profile. Offline prediction says this profile
  should cover `27185` misses under the actual `upgate=1936` / `down=872`
  cache shape, versus `23857` hits measured with the old profile.
- 2026-06-13 01:06 CST:
  `residual-ramtier8192-graphprofile-nosplit-n84` completed and is the current
  best measured non-strict result.
  - Result: `2.46 eval tok/s`, `1.18 prompt tok/s`, `total_ms=56792.21`,
    `wall_s=58.57`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=42944`, `direct_reads=2526`,
    `VRAM hit=53.1%`.
  - RAM tier: `27185` hits / `72655` total (`37.4%`), resident `8192 MiB`.
  - This exactly matches the graph-reserve-aware offline prediction for
    covered misses (`27185`), validating the replanner under the actual
    graph-reserved slot shape.
  - Comparison to current key lines:
    - strict best/repeat: `1.91-1.92 tok/s`, `62582` io_uring reads;
    - 3 GiB residual RAM tier: `2.22 tok/s`, `51956` reads;
    - 8 GiB old profile + graph reserve: `2.44-2.45 tok/s`, `46272` reads;
    - 8 GiB graph-reserve-aware profile + no-split: `2.46 tok/s`,
      `42944` reads.
  - Remaining gap: still far below the `~3.9-5.5 tok/s` theoretical target.
    The remaining `42944` SSD reads and low effective in-flight depth
    (`avg=2.07`) remain the bottleneck.
- 2026-06-13 01:07 CST: Considered a 10 GiB graph-reserve-aware RAM-tier n8
  diagnostic. The offline prediction for the same graph-reserved slot shape is
  `31832` covered misses, only `4647` more than the validated 8 GiB profile.
  Before launching, a new external `sglang.launch_server` task appeared under
  `/home/wici/lfz/ktransformers` and host available RAM dropped to about
  `9.3 GiB`. Per `Agent.md`, I did not start the 10 GiB run. It remains a
  pending non-strict diagnostic for a fully idle window.
- 2026-06-13 01:09 CST: Resumed and rechecked resources. GPU usage was
  `41 MiB`, no matching external workload process was active, and host
  available RAM was about `12 GiB`. Starting the pending short non-strict
  10 GiB graph-reserve-aware RAM-tier diagnostic:
  `residual-ramtier10240-graphprofile-nosplit-n8`.
  - Config: `MemoryMax=14G`, `MemorySwapMax=0`,
    `GGML_MOE_RAM_TIER_MIB=10240`,
    `GGML_MOE_RAM_TIER_PROFILE=ramtier-residual-graphreserve2048-10240.route.csv`,
    graph reserve `2048 MiB`, no up/gate split-stage.
  - Gate: only run n84 if n8 loads successfully, has `read_failures=0`, and
    RAM-tier prefix hits track the offline prediction (`2074` covered prefix
    misses).
- 2026-06-13 01:10 CST:
  `residual-ramtier10240-graphprofile-nosplit-n8` completed successfully.
  - Result: `1.99 eval tok/s`, `1.01 prompt tok/s`, `total_ms=31118.15`,
    `wall_s=33.95`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=5699`, `VRAM hit=38.4%`,
    `RAM tier hits=2074` (`20.1%`), resident `10240 MiB`.
  - Batching remained healthy: `inflight_avg=2.34`, `inflight_max=8`.
  - This exactly matches the offline prefix prediction (`2074`) and improves
    the previous 8 GiB graph-reserve-aware n8 (`1.68 tok/s`, `6259` reads,
    `1514` RAM hits). Gate passed for one n84 diagnostic if resources remain
    idle.
- 2026-06-13 01:11 CST: Rechecked resources before launching the corresponding
  10 GiB n84 run. A new external `sglang.launch_server` task was active under
  `/home/wici/lfz/ktransformers`, using about `13.9 GiB` VRAM; host available
  RAM was about `8.4 GiB` with swap already in use. Per `Agent.md`, I did not
  start the 10 GiB n84 run and did not stop the external process. The 10 GiB
  n84 diagnostic remains pending for the next fully idle window.
- 2026-06-13 01:13 CST: Waited through several resource checks. The external
  `sglang.launch_server` work continued to start new attempts, with GPU/host
  RAM usage changing over time. I continued to avoid stopping or competing with
  it.
  - Offline expectation for the pending 10 GiB n84 run under the graph-reserved
    slot shape: `70093` residual misses, `31832` RAM-tier-covered misses,
    predicted `38261` io_uring reads.
  - Compared with the current best 8 GiB graph-reserve-aware n84
    (`42944` reads, `2.46 tok/s`), 10 GiB is expected to remove only about
    `4683` more reads. This may improve decode, but it is unlikely by itself
    to close the gap to the `~3.9-5.5 tok/s` theoretical target.
  - Current pending command shape for the next idle window:
    `residual-ramtier10240-graphprofile-nosplit-n84`, same as the successful
    10 GiB n8 but with `--predict-tokens 84`.
- 2026-06-13 01:15 CST: Resources became idle again: GPU usage `41 MiB`,
  no matching workload process, host available RAM about `12 GiB`. Starting
  the pending non-strict 10 GiB n84 diagnostic:
  `residual-ramtier10240-graphprofile-nosplit-n84`.
- 2026-06-13 01:16 CST:
  `residual-ramtier10240-graphprofile-nosplit-n84` failed before timings.
  - systemd journal shows `run-u161.service: A process of this unit has been
    killed by the OOM killer`, exit `status=9/KILL`, result `oom-kill`.
  - cgroup peak was reported as `12.0G` despite the requested
    `MemoryMax=14G`, and stderr showed `RAM tier: cudaHostRegister succeeded
    (10239.00 MiB pinned)` followed by startup/decode activation but no llama
    timings or expert-pack final counters.
  - This is a host-memory/cgroup OOM, not a CUDA graph OOM. The 10 GiB n8
    remains valid, but 10 GiB n84 is not currently safe on this 15 GiB host
    with `MemorySwapMax=0`.
  - Next safer step: generate a `9216 MiB` graph-reserve-aware residual profile
    and validate it. It should sit between the stable 8 GiB n84 and the OOMing
    10 GiB n84.
- 2026-06-13 01:17 CST: Generated
  `ramtier-residual-graphreserve2048-9216.route.csv` using the actual
  graph-reserved slot shape (`upgate=1936`, `down=872`).
  - Prediction: `9213.844 MiB`, `2265` entries, `29591` covered misses
    (`42.217%`) out of `70093` residual misses.
  - Predicted n84 io_uring reads: `40502`.
  - Prefix prediction: `1774` covered prefix misses.
  - This is expected to save about `2442` reads versus the current best 8 GiB
    n84 (`42944` reads) while avoiding the 10 GiB n84 host-memory OOM.
- 2026-06-13 01:18 CST: Resources remained idle (`41 MiB` GPU used, no
  matching workload process, about `12 GiB` host available RAM). Starting
  short validation `residual-ramtier9216-graphprofile-nosplit-n8` with
  `MemoryMax=13G`, `MemorySwapMax=0`, `RAM tier=9216 MiB`.
- 2026-06-13 01:19 CST:
  `residual-ramtier9216-graphprofile-nosplit-n8` completed successfully.
  - Result: `1.84 eval tok/s`, `0.67 prompt tok/s`, `total_ms=39965.95`,
    `wall_s=41.86`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=5999`, `VRAM hit=38.4%`,
    `RAM tier hits=1774` (`17.2%`), resident `9216 MiB`.
  - This exactly matches the offline prefix prediction (`1774`) and falls
    between the 8 GiB and 10 GiB n8 read counts (`6259`, `5699`).
  - Gate passed for one 9 GiB n84 diagnostic if resources remain idle.
- 2026-06-13 01:20 CST: Resources remained idle. Starting
  `residual-ramtier9216-graphprofile-nosplit-n84` with `MemoryMax=13G`,
  `MemorySwapMax=0`, `RAM tier=9216 MiB`, graph reserve `2048 MiB`, and no
  up/gate split-stage.
- 2026-06-13 01:21 CST:
  `residual-ramtier9216-graphprofile-nosplit-n84` failed before producing any
  output tokens or MoE cache/RAM-tier counters.
  - stderr starts with `CUDA0: using device CUDA0 - 21726 MiB free`, much lower
    than an idle 5090 (`~31500 MiB free`), and then fails during dense model
    load with `cudaMalloc failed: out of memory` for a small `14.88 MiB`
    allocation.
  - systemd reports `status=11/SEGV`, not a clean llama timing failure.
  - A new external `sglang.launch_server` process was active immediately after
    the run. Treat this as an invalid resource-contention attempt, not a
    conclusive 9 GiB RAM-tier result. Per `Agent.md`, do not keep retrying
    while external work is active.
- 2026-06-13 01:22 CST: Static checks still pass
  (`python3 -m py_compile`, `git diff --check`). External `sglang` work became
  active again and occupied about `20.9 GiB` VRAM, so no further GPU run was
  started.
- 2026-06-13 01:23 CST: Offline upper-bound check for graph-reserve-aware
  larger RAM tiers:
  - 8 GiB validated n84: `42944` actual io_uring reads, `2.46 tok/s`.
  - 9 GiB predicted n84 reads: `40502` (`2442` fewer than 8 GiB actual).
  - 10 GiB predicted n84 reads: `38261` (`4683` fewer than 8 GiB actual), but
    the real n84 attempt was host OOM.
  - 12 GiB predicted n84 reads: `34176`, but this is not practical on the
    current 15 GiB host because 10 GiB already OOMs with `MemorySwapMax=0`.
  - Conclusion: simply increasing RAM tier is no longer a credible path to the
    theoretical `~3.9-5.5 tok/s` target on this host. The next useful work is
    to reduce memory pressure/loading footprint or find a way to cut/overlap
    the remaining reads without requiring another multi-GiB RAM tier.
- 2026-06-13 01:24 CST: A final resource recheck showed the machine idle
  again (`41 MiB` GPU used, about `12 GiB` host available RAM, no matching
  workload process). Since the previous 9 GiB n84 attempt started with only
  `21726 MiB` GPU free and is therefore invalid, retrying once with a distinct
  label: `residual-ramtier9216-graphprofile-nosplit-n84-repeat1`.
- 2026-06-13 01:25 CST:
  `residual-ramtier9216-graphprofile-nosplit-n84-repeat1` completed
  successfully and is the new best non-strict decode result.
  - Result: `2.65 eval tok/s`, `0.84 prompt tok/s`, `total_ms=62934.80`,
    `wall_s=64.78`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=40538`, `direct_reads=2526`,
    `VRAM hit=53.1%`.
  - RAM tier: `29591` hits / `72655` total (`40.7%`), resident `9216 MiB`.
  - The RAM-tier hits exactly match the offline prediction and reduce SSD
    reads from the 8 GiB best's `42944` to `40538`.
  - Decode rate improves from the 8 GiB best's `2.46 tok/s` to `2.65 tok/s`,
    but total time is worse than the 8 GiB run (`62934.80 ms` vs
    `56792.21 ms`) because load/prompt time is higher and the machine is under
    substantial host-memory/swap pressure.
  - Interpretation: 9 GiB RAM tier confirms that removing more residual reads
    improves decode, but host-memory cost dominates overall latency. Further
    RAM-tier growth is not the main path to the theoretical target on this
    15 GiB host; the next path should reduce runtime/load memory pressure or
    reduce/overlap the remaining `~40k` SSD reads without requiring a larger
    resident host tier.

## Phase 17: Reduce Runtime Memory Pressure

Phase 16 showed that larger RAM tiers improve decode but are not enough to
reach the theoretical 5090 target, and they worsen host-memory pressure. Phase
17 tests whether reducing runtime memory pressure can preserve the 9 GiB
decode improvement while improving load/prompt/total time.

First candidate:

- keep the validated `9216 MiB` graph-reserve-aware RAM-tier profile;
- keep no up/gate split-stage, io_uring offset sort, graph reserve `2048 MiB`;
- reduce context from `2048` to `512`, because the n84 prompt/generation is
  short and earlier strict diagnostics showed lower `ctx` reduces model/KV
  pressure;
- validate with n8 first before n84.

Acceptance gate:

- n8 must complete with `read_failures=0`;
- RAM-tier hits should still match the prefix prediction (`1774`) or explain
  any difference;
- if n8 is not worse on decode and improves load/prompt/total enough, run one
  n84 validation.

- 2026-06-13 01:27 CST: Runner already supports `--ctx` and `--batch`; no code
  change needed for the first validation.
- 2026-06-13 01:28 CST: Resources idle (`41 MiB` GPU used, no matching
  workload process, about `12 GiB` host available RAM). Starting
  `residual-ramtier9216-graphprofile-nosplit-ctx512-n8`.
- 2026-06-13 01:29 CST:
  `residual-ramtier9216-graphprofile-nosplit-ctx512-n8` completed.
  - Result: `1.85 eval tok/s`, `0.80 prompt tok/s`, `total_ms=35259.17`,
    `wall_s=37.12`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=5999`, `VRAM hit=38.6%`,
    `RAM tier hits=1776` (`17.2%`), resident `9216 MiB`.
  - Compared with the ctx2048 9 GiB n8 (`1.84 tok/s`, `total_ms=39965.95`),
    decode is unchanged but total time improves by about `4.7s`. Gate passed
    for one ctx512 n84 validation if resources remain idle.
- 2026-06-13 01:30 CST: Resources still idle. Starting
  `residual-ramtier9216-graphprofile-nosplit-ctx512-n84`.
- 2026-06-13 01:31 CST:
  `residual-ramtier9216-graphprofile-nosplit-ctx512-n84` failed before
  timings.
  - It started with enough GPU memory (`CUDA0 ... 30838 MiB free`) and loaded
    the 9 GiB RAM tier, so this is not the earlier resource-contention failure.
  - stderr reports CUDA graph launch OOM:
    `evaluate_and_capture_cuda_graph ... cudaGraphLaunch ... out of memory`.
  - systemd also reports OOM kill at `13.0G` memory peak.
  - Reducing `ctx` from `2048` to `512` improves n8 total time but does not
    make the long 9 GiB n84 stable. Next diagnostic is larger graph reserve
    (`3072 MiB`) to leave more CUDA graph/compute headroom, starting with n8.
- 2026-06-13 01:32 CST: Tried
  `residual-ramtier9216-graphprofile-nosplit-gr3072-n8` as a safer graph
  headroom diagnostic. It failed before timings.
  - systemd reports OOM kill at `13.0G` memory peak.
  - stderr shows the larger graph reserve clamped effective VRAM cache to
    `9360 MiB`; the down cache first failed to allocate `3.2 GiB` and retried
    to only `0.5 GiB`, then CUDA graph launch still OOMed.
  - This is worse than the validated 9 GiB graph-reserve-2048 path. Increasing
    graph reserve is not a viable fix because it both worsens VRAM residency
    and still hits host/cgroup pressure.
  - Updated direction: investigate/reduce host memory peak and pinned
    allocation pressure for n84, especially why 9 GiB n8 fits while 9 GiB n84
    reaches the `13G` cgroup peak. Do not keep sweeping graph reserve.
- 2026-06-13 01:34 CST: Added an env-gated RAM-tier pinning control:
  `GGML_MOE_RAM_TIER_PIN=0`. Default behavior remains unchanged
  (`cudaHostRegister` still attempted when the env is unset). The task runner
  gained `--ram-tier-no-pin` to make this reproducible.
  - Build passed: `cmake --build build-cuda --target llama-cli -j 8`.
  - Static script check passed: `python3 -m py_compile`.
  - Purpose: test whether avoiding a multi-GiB pinned RAM-tier registration
    reduces cgroup/pinned-memory pressure enough to make larger RAM-tier n84
    stable, while measuring any H2D performance penalty.
- 2026-06-13 01:35 CST: Before running the no-pin validation, a new external
  `sglang.launch_server` task was active under `/home/wici/lfz/ktransformers`;
  GPU usage was about `6.7 GiB` and host available RAM about `8.8 GiB`.
  Per `Agent.md`, no GPU/RAM experiment was started and the external process
  was not stopped. Pending next run when idle:
  `residual-ramtier9216-graphprofile-nosplit-nopin-n8`, followed by n84 only
  if n8 shows the unpinned RAM-tier path works with acceptable speed and
  reduced memory pressure.
- 2026-06-13 01:37 CST: External task ended naturally. GPU usage returned to
  `41 MiB`, no matching workload process was active, and host available RAM
  was about `12 GiB`. Starting
  `residual-ramtier9216-graphprofile-nosplit-nopin-n8`.
- 2026-06-13 01:38 CST:
  `residual-ramtier9216-graphprofile-nosplit-nopin-n8` completed successfully.
  - Result: `1.61 eval tok/s`, `0.87 prompt tok/s`, `total_ms=32290.11`,
    `wall_s=33.93`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=5999`, `VRAM hit=38.4%`,
    `RAM tier hits=1774` (`17.2%`), resident `9216 MiB`.
  - stderr confirms the new path:
    `RAM tier: cudaHostRegister disabled, using unpinned path`.
  - Compared with pinned 9 GiB n8, decode is slower (`1.61` vs `1.84`), but
    load/total improve (`load_ms=27941` vs `36160`, `total_ms=32290` vs
    `39966`). This clears the gate for one n84 validation because the goal is
    to reduce host/pinned pressure.
- 2026-06-13 01:39 CST: Before launching no-pin n84, a new external
  `sglang.launch_server` task started under `/home/wici/lfz/ktransformers`.
  Per `Agent.md`, I did not start the n84 run. Pending next run when the
  machine is fully idle:
  `residual-ramtier9216-graphprofile-nosplit-nopin-n84`.
- 2026-06-13 01:41 CST: Waited one more resource-check interval. The same
  external `sglang.launch_server` was still active, GPU memory was about
  `21.0 GiB` used, and host available RAM had dropped to about `2.7 GiB`.
  No additional GPU/RAM experiment was started.
- 2026-06-13 02:05 CST: Resources are idle again (`41 MiB` GPU used, about
  `12 GiB` host RAM available, no matching GLM/sglang workload process).
  Starting the gated long validation:
  `residual-ramtier9216-graphprofile-nosplit-nopin-n84`.
  - Config: 9 GiB residual RAM-tier profile, `GGML_MOE_RAM_TIER_PIN=0`,
    graph reserve `2048 MiB`, no up/gate split-stage, io_uring offset sort,
    `MemoryMax=13G`, `MemorySwapMax=0`, `n_predict=84`.
  - Purpose: determine whether avoiding multi-GiB `cudaHostRegister` pressure
    can keep the 9 GiB RAM-tier n84 run stable and improve total latency,
    while measuring the decode penalty from unpinned H2D copies.
- 2026-06-13 02:07 CST:
  `residual-ramtier9216-graphprofile-nosplit-nopin-n84` completed.
  - Result: `1.98 eval tok/s`, `1.02 prompt tok/s`, `total_ms=67598.72`,
    `load_ms=25592.57`, `wall_s=69.27`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=40538`, `direct_reads=2526`,
    `VRAM hit=53.1%`, `RAM tier hits=29591/72655` (`40.7%`), resident
    `9216 MiB`.
  - stderr confirms the intended path:
    `RAM tier: cudaHostRegister disabled, using unpinned path (9213.84 MiB)`.
  - Compared with pinned 9 GiB n84 repeat:
    `2.65 eval tok/s`, `0.84 prompt tok/s`, `total_ms=62934.80`,
    `load_ms=31592.17`, same `40538` SSD reads and same RAM-tier hit count.
  - Interpretation: no-pin reduces load time and io wait
    (`29.5s -> 24.3s`) but loses too much decode throughput due to pageable
    RAM-tier H2D copies. Pure no-pin is not the path to the target. The next
    direction is either partial pinning of the RAM-tier hot subset, so H2D
    remains fast without pinning the full 9 GiB, or improving VRAM residency
    so fewer residual rows need RAM/SSD at all.

## Phase 18: Partial RAM-Tier Pinning

Phase 17 proved both sides of the tradeoff:

- full 9 GiB pinning gives the best decode rate (`2.65 tok/s`) but high
  load/host pressure;
- zero pinning lowers load pressure but decode falls to `1.98 tok/s`.

The next experiment should keep the 9 GiB RAM-tier contents unchanged but pin
only a prefix of the residual RAM-tier profile. The residual profile is already
ordered by predicted usefulness, so prefix pinning should accelerate the most
frequent RAM-tier hits while avoiding a full 9 GiB `cudaHostRegister`.

Implementation plan:

- add env `GGML_MOE_RAM_TIER_PIN_MIB=<MiB>` with default preserving existing
  behavior;
- if `GGML_MOE_RAM_TIER_PIN=0`, pin nothing;
- if `GGML_MOE_RAM_TIER_PIN_MIB` is set, `cudaHostRegister` only the first
  contiguous prefix of RAM-tier entries whose resident bytes fit the budget;
- report pinned MiB and entry count in stderr;
- add runner flag `--ram-tier-pin-mib`;
- validate with n8 at `2048 MiB` and `4096 MiB` pin budgets, then one n84
  validation if n8 decode/load tradeoff is promising.

- 2026-06-13 02:14 CST: Implemented partial RAM-tier pinning.
  - CUDA env: `GGML_MOE_RAM_TIER_PIN_MIB=<MiB>` pins only a contiguous RAM-tier
    prefix whose complete entries fit the budget. Default remains full pinning
    when `GGML_MOE_RAM_TIER_PIN_MIB` is unset and `GGML_MOE_RAM_TIER_PIN` is
    not `0`.
  - `GGML_MOE_RAM_TIER_PIN=0` still disables pinning entirely.
  - Runner flag: `--ram-tier-pin-mib`.
  - Verification passed:
    `cmake --build build-cuda --target llama-cli -j 8` and
    `python3 -m py_compile`.
- 2026-06-13 02:15 CST: Before launching the first partial-pin validation,
  another `sglang.launch_server` task was active under
  `/home/wici/lfz/ktransformers`, using about `20.8 GiB` VRAM; host available
  RAM was about `5.7 GiB`. Per `/home/wici/lfz/Agent.md`, no experiment was
  started and the external process was not stopped. Pending when idle:
  `residual-ramtier9216-graphprofile-pin2048-n8`.
- 2026-06-13 02:16 CST: The external task ended naturally. Resources returned
  to idle (`41 MiB` GPU used, about `12 GiB` host RAM available). Starting
  `residual-ramtier9216-graphprofile-pin2048-n8`.
- 2026-06-13 02:17 CST:
  `residual-ramtier9216-graphprofile-pin2048-n8` completed.
  - Result: `1.65 eval tok/s`, `1.11 prompt tok/s`, `total_ms=28432.55`,
    `load_ms=24186.49`, `wall_s=30.13`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=5999`, `VRAM hit=38.4%`,
    `RAM tier hits=1774/10299` (`17.2%`).
  - stderr: pinned prefix was `2045.62 MiB`, `516` entries, with
    `9213.84 MiB` resident.
  - Interpretation: 2 GiB prefix pin keeps the load/total benefit of no-pin
    and slightly improves decode (`1.65` vs `1.61`), but is still far below
    full 9 GiB pin n8 decode (`1.84`). Running 4 GiB prefix next to check
    whether decode recovers smoothly with more pinned hot rows.
- 2026-06-13 02:18 CST: Resources remained idle. Starting
  `residual-ramtier9216-graphprofile-pin4096-n8`.
- 2026-06-13 02:19 CST:
  `residual-ramtier9216-graphprofile-pin4096-n8` completed.
  - Result: `1.74 eval tok/s`, `1.08 prompt tok/s`, `total_ms=28541.96`,
    `load_ms=24518.08`, `wall_s=30.25`, `read_failures=0`.
  - Reads/cache: same as the 2 GiB prefix n8 (`5999` io_uring reads,
    `1774/10299` RAM-tier hits, `38.4%` VRAM hit).
  - stderr: pinned prefix was `4095.56 MiB`, `1014` entries, with
    `9213.84 MiB` resident.
  - Interpretation: decode improves smoothly as more RAM-tier hot rows are
    pinned (`1.61 no-pin -> 1.65 pin2G -> 1.74 pin4G -> 1.84 full-pin`),
    while load/total remain near the low-pressure no-pin path. A 6 GiB prefix
    is the next useful n8 point before choosing an n84 validation.
- 2026-06-13 02:20 CST: A new external `sglang.launch_server` task started
  under `/home/wici/lfz/ktransformers`. Per `/home/wici/lfz/Agent.md`, no
  additional GPU/RAM experiment was started and the external process was not
  stopped. Pending when idle:
  `residual-ramtier9216-graphprofile-pin6144-n8`.
- 2026-06-13 02:24 CST: The same external `sglang.launch_server` task was
  still active, using about `20.8 GiB` VRAM. Host memory was under pressure
  (`~2.2 GiB` available, swap mostly used). No additional experiment was
  started.
- 2026-06-13 02:29 CST: The previous external task ended naturally, but a new
  `sglang.launch_server` attempt started immediately under
  `/home/wici/lfz/ktransformers` (`attempt34_cpu8_tp2`). GPU usage had not yet
  ramped at the instant of the check, so no experiment was started to avoid
  racing the new workload. Pending remains:
  `residual-ramtier9216-graphprofile-pin6144-n8`.
- 2026-06-13 02:33 CST: The external workload continued as
  `attempt35_cpu8`, using about `20.8 GiB` VRAM, with host RAM under heavy
  pressure and swap nearly full. No experiment was started.
- 2026-06-13 02:37 CST: External workload ended naturally. Resources returned
  to idle (`41 MiB` GPU used, about `12 GiB` host RAM available). Starting
  `residual-ramtier9216-graphprofile-pin6144-n8`.
- 2026-06-13 02:38 CST:
  `residual-ramtier9216-graphprofile-pin6144-n8` completed.
  - Result: `1.70 eval tok/s`, `1.03 prompt tok/s`, `total_ms=30459.39`,
    `load_ms=26335.87`, `wall_s=32.34`, `read_failures=0`.
  - stderr: pinned prefix was `6142.78 MiB`, `1523` entries, with
    `9213.84 MiB` resident.
  - Interpretation: 6 GiB prefix is worse than 4 GiB prefix on both decode
    and total in the n8 smoke. The best short-run partial-pin point is now
    4 GiB (`1.74 eval tok/s`, `total_ms=28541.96`), so run one n84 validation
    with `--ram-tier-pin-mib 4096`.
- 2026-06-13 02:39 CST: Resources remained idle. Starting
  `residual-ramtier9216-graphprofile-pin4096-n84`.
- 2026-06-13 02:40 CST:
  `residual-ramtier9216-graphprofile-pin4096-n84` completed but is a negative
  result.
  - Result: `1.72 eval tok/s`, `1.13 prompt tok/s`, `total_ms=71350.92`,
    `load_ms=23024.92`, `wall_s=73.06`, `read_failures=0`.
  - stderr: pinned prefix was `4095.56 MiB`, `1014` entries, with
    `9213.84 MiB` resident.
  - The output drifted from the original oracle-n84 path. As a consequence,
    profile coverage collapsed: `io_uring_reads=66760` instead of the expected
    `40538`, `VRAM hit=40.2%` instead of `53.1%`, and RAM-tier hits were
    `22588/91874` instead of `29591/72655`.
  - Interpretation: partial pinning is not currently a reliable n84 win. It
    can lower load pressure, but if generation drifts, the same oracle/profile
    no longer covers the long run and the extra SSD reads dominate. Before
    using this path further, re-run the default full-pin path with the new code
    to verify default behavior is unchanged.
- 2026-06-13 02:42 CST: Resources remained idle. Starting default full-pin
  verification `residual-ramtier9216-graphprofile-fullpin-n84-verify` with no
  `GGML_MOE_RAM_TIER_PIN_MIB` set.
- 2026-06-13 02:43 CST:
  `residual-ramtier9216-graphprofile-fullpin-n84-verify` completed.
  - Result: `2.56 eval tok/s`, `0.88 prompt tok/s`, `total_ms=60276.70`,
    `load_ms=27797.23`, `wall_s=62.14`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=40538`, `direct_reads=2526`,
    `VRAM hit=53.1%`, `RAM tier hits=29591/72655` (`40.7%`).
  - This matches the original full-pin route/profile coverage:
    same `40538` SSD reads and same RAM-tier hits as the previous 9 GiB
    full-pin repeat (`2.65 tok/s`, `total_ms=62934.80`).
  - Conclusion: the new `GGML_MOE_RAM_TIER_PIN_MIB` implementation preserves
    default full-pin behavior when the env is unset. Partial pinning remains a
    useful diagnostic but is not yet a win for the long n84 benchmark because
    it can change the generated route and lose oracle profile coverage.

## Phase 19: Next Throughput Lever

Current best non-strict line remains the 9 GiB RAM-tier full-pin family:

- best decode observed: `2.65 eval tok/s` (`total_ms=62934.80`);
- best full-pin verify total: `2.56 eval tok/s` (`total_ms=60276.70`);
- residual SSD reads still high: `40538` io_uring reads for n84;
- strict 2 GiB line remains much lower (`~1.9 tok/s`) because RAM tier is
  disabled.

Partial pinning reduced load time but cannot solve the main bottleneck unless
route/profile coverage remains stable. The next practical lever is to reduce
the remaining `~40k` SSD reads by increasing VRAM-resident expert rows, but
the current `20480 MiB` requested cache is auto-clamped to about `11450 MiB`
after dense tensors, KV, compute buffer, graph reserve, and safety. Next
checks:

- inspect actual clamped VRAM budget and split in the latest full-pin stderr;
- test whether disabling graph reuse or lowering graph reserve can safely
  increase effective VRAM cache without CUDA OOM;
- only run n8 first, and only validate n84 if `read_failures=0` and actual
  VRAM hit improves.

- 2026-06-13 02:45 CST: Latest full-pin verify shows the VRAM cache request
  was clamped from `20480 MiB` to `11450 MiB` because free VRAM after dense
  tensors was `13754 MiB` and auto-clamp reserved `2048 MiB` for CUDA graph
  plus `256 MiB` safety. The next n8 diagnostic disables graph reuse and leaves
  graph reserve unset/zero, to see whether the extra VRAM cache headroom
  improves hit rate enough to offset any compute slowdown:
  `residual-ramtier9216-fullpin-nograph-n8`.
- 2026-06-13 02:46 CST:
  `residual-ramtier9216-fullpin-nograph-n8` completed.
  - Result: `1.52 eval tok/s`, `1.02 prompt tok/s`, `total_ms=30295.75`,
    `load_ms=25693.78`, `wall_s=32.21`, `read_failures=0`.
  - Actual VRAM cache increased from `11450 MiB` to `13498 MiB`
    (`graph_reserve=0`), with preloads increasing from `2526` to `2978`.
  - VRAM hit improved only modestly on n8 (`38.4% -> 41.5%`), but decode was
    worse than graph-reuse full-pin n8 (`1.84 tok/s`) and worse than partial
    pin 4 GiB n8 (`1.74 tok/s`). `io_uring_wait_us` also increased.
  - Conclusion: freeing graph reserve to expand cache is not worth it in this
    configuration. Do not run the n84 version unless the graph/cache behavior
    changes.

## Current Status

Implemented:

- `GGML_MOE_RAM_TIER_PIN=0` to disable RAM-tier `cudaHostRegister`;
- `GGML_MOE_RAM_TIER_PIN_MIB=<MiB>` to pin only a contiguous RAM-tier prefix;
- runner flags `--ram-tier-no-pin` and `--ram-tier-pin-mib`.

Validation:

- build passed: `cmake --build build-cuda --target llama-cli -j 8`;
- script check passed: `python3 -m py_compile`;
- `git diff --check` passed;
- default full-pin path was revalidated and remains compatible with the
  previous best route/profile coverage.

Best current non-strict result remains full 9 GiB RAM-tier pinning:

- latest full-pin verify: `2.56 eval tok/s`, `0.88 prompt tok/s`,
  `total_ms=60276.70`, `io_uring_reads=40538`, `VRAM hit=53.1%`,
  `RAM tier hits=29591/72655`;
- previous best decode repeat: `2.65 eval tok/s`, `total_ms=62934.80`;
- strict 2 GiB/RAM=0 line remains around `1.9 tok/s`.

Open issue:

- The model is still far below the rough 5090 theoretical target because the
  long n84 path still needs about `40538` SSD reads even with 9 GiB RAM tier.
  Attempts to trade memory pressure for lower load time either lose H2D speed
  (no-pin/partial-pin) or lose graph performance (no-graph). The remaining
  plausible path is better route-stable VRAM/RAM profile generation or a real
  overlap/prefetch mechanism that removes the `~28s` io_uring wait without
  changing generation.

## Phase 20: Route-Stable Profile Regeneration

Partial pinning failed on n84 largely because the long generation drifted away
from the original `oracle-n84` route, so the profile stopped covering the
actual accesses. Phase 20 tests whether a profile generated from the current
configuration's own route can recover coverage and throughput.

Plan:

- capture route profile and route trace for the 4 GiB partial-pin n84
  configuration, writing them under
  `.Agent/plans/5090-theoretical-token-rate/phase20-route-stable/`;
- generate a 9 GiB residual RAM-tier profile from that trace using the actual
  graph-reserve cache shape: `upgate_slots=1936`, `down_slots=872`,
  `reserve_pct=10`;
- replay the 4 GiB partial-pin n84 using the newly captured route profile and
  newly generated residual RAM-tier profile;
- compare against the failed partial-pin n84 and the full-pin verify.

Acceptance gate:

- `read_failures=0`;
- `io_uring_reads` must fall back near the covered full-pin level (`~40538`)
  rather than the drifted partial-pin level (`66760`);
- VRAM hit should recover toward `53.1%`;
- throughput must beat the current full-pin total or decode line before this
  path is considered useful.

- 2026-06-13 02:51 CST: Resources idle (`41 MiB` GPU used, about `12 GiB`
  host RAM available). Starting Phase 20 capture with 4 GiB partial pin.
- 2026-06-13 02:56 CST:
  `capture-pin4096-n84-route` completed and wrote:
  - `phase20-route-stable/pin4096-n84.route.csv`;
  - `phase20-route-stable/pin4096-n84.trace.csv`.
  - Capture result stayed on the drifted path: `1.63 eval tok/s`,
    `total_ms=76972.98`, `io_uring_reads=66760`, `VRAM hit=40.2%`,
    `RAM tier hits=22588/91874`.
  - Next: generate a residual RAM-tier profile from this captured trace with
    `upgate_slots=1936`, `down_slots=872`, `ram_tier_mib=9216`.
- 2026-06-13 02:57 CST: Generated
  `phase20-route-stable/pin4096-n84-ramtier9216.route.csv`.
  - Offline replay with captured route/profile and the actual graph-reserve
    cache shape predicted `77149` residual misses and only `25877` misses
    covered by the 9 GiB RAM tier (`33.542%`).
  - This is better than the drifted run's observed `22588` RAM-tier hits, but
    still worse than the original full-pin oracle path (`29591` RAM-tier hits
    and `40538` SSD reads). Expected SSD reads remain too high for the
    theoretical target.
  - Run one replay anyway to measure real impact, but this path is unlikely to
    beat the full-pin baseline.
- 2026-06-13 02:59 CST:
  `replay-pin4096-n84-route-stable` completed.
  - Result: `1.79 eval tok/s`, `0.93 prompt tok/s`, `total_ms=74017.62`,
    `load_ms=27565.26`, `wall_s=76.01`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=59998`, `direct_reads=2526`,
    `VRAM hit=43.6%`, `RAM tier hits=24340/86864` (`28.0%`).
  - This is better than the drifted partial-pin capture (`66760` reads), but
    still much worse than full-pin verify (`40538` reads, `53.1%` VRAM hit,
    `2.56 eval tok/s`).
  - Conclusion: route-stable regeneration helps only modestly because the
    drifted path has a much larger residual working set. Stop this path for
    now. The next bottleneck is the actual I/O wait: even the best full-pin
    path spends about `28.5s` in io_uring wait with average inflight only
    about `2.0` despite `depth=16`.

## Phase 21: I/O Concurrency Diagnosis

Hypothesis:

- The current io_uring path uses depth/slots=16, but batches are limited by
  the number of missing active experts available inside one MoE call.
- Reported batch histograms are dominated by `1` and `2-4`, with no `9-16`
  batches, so increasing `depth` alone cannot create enough queue depth.
- To get closer to 5090 theoretical token rate, the runtime needs route-aware
  lookahead/prefetch across upcoming MoE calls or token steps, not just faster
  per-call reads.

Immediate inspection:

- locate the io_uring batch submission loops;
- identify whether trace-prefetch currently only preloads into VRAM cache and
  whether it can be repurposed for pack-to-staging prefetch;
- decide on the smallest implementation that overlaps future SSD reads while
  preserving deterministic route/profile coverage.

- 2026-06-13 03:04 CST: Inspection findings:
  - `expert_pack_iouring_copy_jobs()` batches only the current MoE call's
    missing jobs. That explains why `depth=16` still reports
    `inflight_avg≈2` and no `9-16` batches: most calls do not expose enough
    same-call misses.
  - Existing `trace_prefetch_on_hit()` already uses future route trace entries
    to preload into VRAM cache on `g_batch.prefetch_stream`, but previous n8
    tests were mostly no-graph/cache-headroom variants and were slower.
  - Because `trace_prefetch` inserts with `allow_evict=false`, it should not
    evict existing profile-pinned cache entries. The next low-risk diagnostic
    is to test trace-prefetch on the current best graph-reserve full-pin
    configuration, first with n8 only.
- 2026-06-13 03:05 CST: Starting
  `fullpin-traceprefetch-lead1024-w24-l4-n8` with the current full-pin
  graph-reserve configuration.
- 2026-06-13 03:06 CST:
  `fullpin-traceprefetch-lead1024-w24-l4-n8` completed.
  - Result: `1.45 eval tok/s`, `1.01 prompt tok/s`, `total_ms=30783.34`,
    `load_ms=25949.12`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=6091`, `VRAM hit=38.4%`,
    `RAM tier hits=1812/10429`.
  - Trace prefetch stats: `loads=130`, `cached=5093`, `async prefetch
    waits=130`.
  - Compared with the current full-pin n8 baseline (`1.84 eval tok/s`,
    `5999` reads), this is worse. Existing trace-prefetch performs too little
    useful prefetch and adds wait/IO interference. Do not run the n84 version.
- 2026-06-13 03:07 CST: Run one direct concurrency diagnostic with
  `depth=32`, `slots=32` on the current full-pin n8 baseline. This should
  confirm whether queue depth is limited by the number of per-call misses.
- 2026-06-13 03:08 CST:
  `fullpin-depth32-slots32-n8` completed.
  - Result: `1.73 eval tok/s`, `0.80 prompt tok/s`, `total_ms=34955.10`,
    `load_ms=30912.96`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=5999`, `VRAM hit=38.4%`,
    `RAM tier hits=1774/10299`.
  - Concurrency stats stayed unchanged from baseline:
    `iouring_inflight_avg=2.42`, `iouring_inflight_max=8`, same batch
    histogram shape, despite `depth=32` and `slots=32`.
  - Conclusion: single-call batch size is the practical concurrency limiter.
    Raising depth/slots alone does not expose more I/O parallelism and should
    not be used for the long n84 line.

## Phase 21 Conclusion

The current implementation cannot reach the 5090 theoretical target just by
tuning `depth`, `slots`, trace-prefetch parameters, or RAM-tier pinning:

- per-call io_uring batches expose only about two concurrent reads on average;
- existing trace-prefetch preloads too few useful rows and adds async waits;
- partial/no pin reduce load pressure but lose H2D speed or route coverage;
- no-graph increases VRAM cache but loses graph execution speed.

Next implementation direction:

- build a real cross-call route prefetch queue from the route trace that reads
  future pack rows into dedicated pinned host buffers before the MoE call needs
  them;
- runtime loads should first consume completed prefetched host buffers, then
  fall back to RAM tier or io_uring;
- the queue must be bounded by host memory and should not insert into VRAM
  cache early, to avoid cache pollution and async wait on cache hits;
- validate with n8 first using the full-pin oracle path, then n84 only if
  runtime `io_uring_wait_us` falls without reducing VRAM hit or changing the
  output route.

Checks after this phase:

- `python3 -m py_compile` passed for task scripts;
- `git diff --check` passed.

## Phase 22: Cross-Call Host Prefetch Queue

Phase 21 showed that per-call io_uring batches are too small to exploit the
SSD or 5090. The next implementation is a trace-driven host prefetch queue:

- read future route-trace entries ahead of the current cursor;
- load future expert-pack rows into a bounded pool of pinned host buffers;
- do not insert these rows into VRAM cache early, avoiding cache pollution and
  async waits on cache hits;
- runtime miss handling first checks whether the row is already prefetched in
  host memory, copies it H2D, and releases the buffer;
- fallback remains unchanged: RAM tier, then current io_uring batch, then
  existing staged read path.

Initial env knobs:

- `GGML_MOE_HOST_PREFETCH=<trace.csv>` enables the queue;
- `GGML_MOE_HOST_PREFETCH_LEAD_EVENTS` controls how far ahead to keep the
  producer cursor;
- `GGML_MOE_HOST_PREFETCH_SLOTS` controls pinned host buffers;
- `GGML_MOE_HOST_PREFETCH_MAX_MIB` bounds total pinned MiB.

Acceptance gate for n8:

- no correctness failures or read failures;
- host-prefetch stats must show nonzero hits;
- `io_uring_wait_us` or eval time must improve over current full-pin n8
  (`1.84 eval tok/s`, `5999` reads, `4.45s` io wait) before running n84.

- 2026-06-13 03:12 CST: Resources idle. Starting implementation.
- 2026-06-13 03:17 CST: Implemented initial env-gated host-prefetch queue in
  `moe_stream_batch.cu` and added runner flags:
  `--host-prefetch`, `--host-prefetch-lead-events`,
  `--host-prefetch-slots`, `--host-prefetch-max-mib`.
  - Default path is unchanged unless `GGML_MOE_HOST_PREFETCH` is set.
  - Build passed: `cmake --build build-cuda --target llama-cli -j 8`.
- 2026-06-13 03:20 CST:
  `fullpin-hostprefetch-lead2048-s64-m512-n8` completed.
  - Result: `1.60 eval tok/s`, `1.01 prompt tok/s`, `total_ms=30717.17`,
    `read_failures=0`.
  - Host-prefetch stats: `submitted=3893`, `hits=56`, `misses=5943`,
    `evicted=3773`, `duplicate_skips=362`.
  - Reads/cache: `io_uring_reads=5943` vs baseline `5999`, but
    `io_uring_wait_us=4938170` worsened vs baseline `4454103`.
  - Interpretation: the producer is filling slots with early/prompt rows that
    are mostly not consumed. Added `GGML_MOE_HOST_PREFETCH_SKIP_EVENTS` /
    runner `--host-prefetch-skip-events`, and changed the consumer to
    synchronize the H2D before releasing a prefetched slot to the producer.
  - Build and py_compile passed after the fix. Next smoke skips the first
    `12624` trace events to target decode.
- 2026-06-13 03:24 CST:
  `fullpin-hostprefetch-skip12624-lead2048-s64-m512-n8` completed.
  - Result: `1.72 eval tok/s`, `1.01 prompt tok/s`, `total_ms=30464.57`,
    `read_failures=0`.
  - Host-prefetch stats: `submitted=849`, `hits=0`, `misses=5999`,
    `evicted=785`, `cursor=12624`, `produce_cursor=13473`.
  - Reads/cache: same `5999` io_uring reads as baseline.
  - Interpretation: n8 only reaches the first `12624` route calls, so skipping
    the first `12624` trace events intentionally targets decode events that
    n8 does not consume. This validates stability but cannot prove usefulness.
    Run one n84 validation with the same skip configuration.
- 2026-06-13 03:27 CST:
  `fullpin-hostprefetch-skip12624-lead2048-s64-m512-n84` completed.
  - Result: `2.22 eval tok/s`, `0.81 prompt tok/s`, `total_ms=67866.60`,
    `read_failures=0`.
  - Host-prefetch stats: `submitted=35985`, `hits=10`, `misses=40528`,
    `evicted=35911`.
  - Reads/cache: `io_uring_reads=40528` vs full-pin baseline `40538`, but
    `io_uring_wait_us=34488533` worsened vs `28466292`; VRAM/RAM coverage was
    unchanged (`VRAM hit=53.1%`, `RAM hits=29591/72655`).
  - Interpretation: the implementation is stable, but `64` slots cannot hold
    enough of the lookahead window, so prefetched rows are overwritten before
    use. Fixed host-prefetch memory accounting and will run one larger-slot
    diagnostic (`512` slots, `2048 MiB` max) to test whether slot pressure is
    the limiting factor. This is still a non-strict host-RAM experiment.
  - Build, py_compile, and `git diff --check` passed after the accounting fix.
- 2026-06-13 03:31 CST:
  `fullpin-hostprefetch-skip12624-lead2048-s512-m2048-n84` completed.
  - Result: `2.03 eval tok/s`, `0.99 prompt tok/s`, `total_ms=67673.73`,
    `read_failures=0`.
  - Host-prefetch stats: `submitted=719`, `hits=227`, `misses=40311`,
    `no_slot=119201`, `duplicate_skips=15735`, `used=2047.88 MiB`.
  - Reads/cache: `io_uring_reads=40311`, only `227` fewer than the full-pin
    baseline, with unchanged VRAM/RAM coverage.
  - Interpretation: larger slots confirm slot pressure, but blind route-trace
    host prefetch is still not a viable accelerator. It wastes buffers on rows
    that would be VRAM/RAM hits or duplicate future route events, and the
    synchronized H2D consume path hurts decode. Stop blind host-prefetch
    sweeping for now.

## Phase 22 Conclusion

The first host-prefetch implementation is stable and env-gated, but it is not
yet a performance win:

- 64 slots: only `10` n84 hits;
- 512 slots / 2 GiB: `227` n84 hits, still far too low;
- decode falls from the full-pin baseline (`2.56-2.65 tok/s`) to `2.03-2.22`
  because producer I/O and synchronous H2D consume overhead outweigh the small
  read reduction.

Next implementation direction:

- generate an offline residual-miss trace for the exact cache + RAM-tier
  replay, not the raw route trace;
- host-prefetch should consume only rows predicted to miss both VRAM cache and
  RAM tier, avoiding duplicate/covered route events;
- make H2D consume asynchronous with a slot lifetime event instead of
  `cudaStreamSynchronize`;
- only then rerun n8/n84 gates.

## Phase 23: SSD-Miss Trace Filtering

Phase 22 showed raw route traces are too noisy for host prefetch. Phase 23
builds a filtered trace that contains only events predicted to miss both VRAM
cache and RAM tier under the current full-pin oracle configuration.

Plan:

- reuse the cache replay model from `simulate_cache_policy.py`;
- preload VRAM cache from `oracle-n84.route.csv` with actual graph-reserve
  slots: `upgate=1936`, `down=872`, `reserve_pct=10`;
- load the RAM-tier selected keys from
  `ramtier-residual-graphreserve2048-9216.route.csv`;
- replay `oracle-n84.trace.csv`;
- output only residual misses not present in RAM tier to
  `phase23-ssd-miss-trace/oracle-n84-ssd-misses.trace.csv`;
- use that filtered trace as `GGML_MOE_HOST_PREFETCH`, with no skip needed,
  and validate whether host-prefetch hits rise enough to lower
  `io_uring_reads` / `io_uring_wait_us`.

Acceptance gate:

- offline trace length should be close to the observed full-pin SSD reads
  (`40538`);
- n84 host-prefetch with filtered trace must reduce runtime `io_uring_reads`
  materially without lowering decode speed.

- 2026-06-13 03:36 CST: Resources idle. Starting offline trace filtering.
- 2026-06-13 03:38 CST: Added
  `make_ssd_miss_trace.py` and generated
  `phase23-ssd-miss-trace/oracle-n84-ssd-misses.trace.csv`.
  - Offline result: `40502` predicted SSD events, close to observed full-pin
    `40538` io_uring reads.
  - Replay details: `vram_hits=79331`, `vram_misses=70093`,
    `ram_hits=29591`, `prefix_ssd_events=5999`.
  - This confirms the filtered trace is aligned with the current full-pin
    oracle path and is a much better host-prefetch input than raw route trace.
- 2026-06-13 03:42 CST:
  `fullpin-hostprefetch-ssdmiss-s512-m2048-n84` produced an invalid negative
  result: output/route drifted, `VRAM hit` collapsed to `27.7%`, and
  `io_uring_reads` jumped to `90628`.
  - Root cause identified in the host-prefetch consumer: on a hit it released
    the slot back to the producer before the H2D copy had completed, allowing
    the producer to overwrite the pinned host buffer during the copy.
  - Fixed the slot lifetime by marking the slot `in_use` during the H2D copy
    and only releasing it after `cudaStreamSynchronize`.
  - Build, py_compile, and `git diff --check` passed after the fix. Run a
    filtered-trace n8 correctness smoke before retrying n84.
- 2026-06-13 03:45 CST:
  `fullpin-hostprefetch-ssdmiss-s512-m2048-n8-verify` completed correctly.
  - Result: `1.61 eval tok/s`, `1.02 prompt tok/s`, `total_ms=30597.07`,
    `read_failures=0`.
  - Host-prefetch stats: `submitted=3156`, `hits=2691`, `misses=3308`,
    `no_slot=1832`.
  - Reads dropped from `5999` to `3308`, proving the SSD-miss filtered trace
    is effective. Decode still fell vs baseline (`1.84 tok/s`) because the
    consumer was still synchronizing after every H2D.
  - Changed host-prefetch consume to asynchronous H2D with a per-slot CUDA
    event; producer waits on that event before reusing the slot. Build,
    py_compile, and `git diff --check` passed. Next run repeats the n8
    filtered-trace smoke.
- 2026-06-13 03:48 CST:
  `fullpin-hostprefetch-ssdmiss-async-s512-m2048-n8` completed.
  - Result: `1.70 eval tok/s`, `0.81 prompt tok/s`, `total_ms=34614.09`,
    `read_failures=0`.
  - Host-prefetch stats: `submitted=2250`, `hits=1775`, `misses=4224`,
    `no_slot=3077`.
  - Reads/cache: `io_uring_reads=4224`, `io_uring_wait_us=3253245`,
    unchanged `VRAM hit=38.4%`, `RAM hits=1774/10299`.
  - Compared with the previous synchronized version, async H2D improves decode
    (`1.61 -> 1.70`) but lowers prefetch coverage (`3308 -> 4224` reads).
    Compared with the full-pin n8 baseline (`1.84 tok/s`, `5999` reads),
    it still fails the gate because decode remains slower.
  - Do not run n84 yet. The next fix should improve producer targeting and
    slot pressure: filtered trace is correct, but the producer still advances
    too far and fills the 2 GiB buffer before the consumer reaches those rows.

## Phase 23 Current Status

Achieved:

- built an offline SSD-miss trace generator;
- produced a trace whose length (`40502`) matches the observed full-pin SSD
  read count (`40538`);
- proved filtered host-prefetch can cut n8 SSD reads materially
  (`5999 -> 3308/4224`) while preserving output correctness.

Not achieved:

- decode speed is still below baseline due to host-prefetch overhead and slot
  pressure;
- n84 filtered host-prefetch is not yet valid to rerun until n8 beats or at
  least matches baseline decode.

Next implementation:

- make producer deduplicate and prioritize the nearest future SSD-miss keys
  instead of linearly filling from `produce_cursor`;
- add counters for producer wait / slot reuse waits;
- test smaller lead windows (`256`, `512`) on n8 after producer targeting,
  because current `2048` lead overfills 2 GiB before consumption catches up.

## Phase 24: Nearest-Future Host Prefetch Targeting

Phase 23 proved the SSD-miss trace is the right input, but the current producer
still linearly fills the queue until it hits the memory cap. With a 2 GiB cap,
this leaves many near-future rows uncovered and produces `no_slot` pressure.

Plan:

- change the producer to scan a window starting at the current matched cursor;
- submit only the nearest future keys not already ready/in-use;
- keep `produce_cursor` as a progress hint, but do not let it permanently run
  far ahead of the consumer;
- add counters for skipped in-use/ready rows and scan passes;
- validate first with filtered-trace n8 at smaller lead windows (`512`, then
  `256` if needed);
- run n84 only if n8 decode is at least near baseline while preserving the
  reduced SSD reads.

- 2026-06-13 03:53 CST: Resources idle. Starting producer targeting change.
- 2026-06-13 04:05 CST: Resuming Phase 24 after implementing nearest-future
  host-prefetch targeting. The code is still env-gated and diagnostic-only.
  Validation order:
  - rebuild/check current tree;
  - run a non-strict full RAM-tier n8 gate using the filtered SSD-miss trace,
    `lead_events=512`, `slots=512`, `max=2048 MiB`;
  - compare against the full-pin n8 baseline (`1.84 eval tok/s`,
    `5999` SSD reads) and the previous async filtered n8 run
    (`1.70 eval tok/s`, `4224` SSD reads).
- 2026-06-13 04:08 CST:
  `fullpin-hostprefetch-ssdmiss-targeted-lead512-s512-m2048-n8` completed
  correctly.
  - Result: `1.74 eval tok/s`, `0.82 prompt tok/s`, `total_ms=34478.41`,
    `read_failures=0`.
  - Reads improved versus the previous async filtered n8 run:
    `io_uring_reads=3838` vs `4224`, and `io_uring_wait_us=2865588` vs
    `3253245`.
  - Host-prefetch stats: `submitted=2635`, `hits=2161`, `misses=3838`,
    `duplicate_skips=3023458`, `reserved_skips=707`, `no_slot=1651129`,
    `scan_passes=1653764`, `used=2047.69 MiB`.
  - Interpretation: nearest-future targeting is directionally better, but the
    producer busy-scans when the 2 GiB host-prefetch buffer is full. This likely
    explains why decode is still below the full-pin n8 baseline
    (`1.84 tok/s`) despite materially fewer SSD reads. Next fix: make the
    producer sleep on slot pressure and wake it only when a slot is released or
    the cursor advances.
- 2026-06-13 04:12 CST: Code inspection found an additional host-prefetch slot
  selection bug. When the prefetch buffer is near `max_bytes`, the allocator can
  pick the first ready slot even if that slot is too small for the selected
  expert row, then fail without scanning later slots that already have enough
  capacity. Fixing this should reduce artificial `no_slot` pressure without
  changing expert selection or the filtered trace input.
- 2026-06-13 04:17 CST:
  `fullpin-hostprefetch-ssdmiss-slotfix-lead512-s512-m2048-n8` was a failed
  attempt.
  - Result: `1.57 eval tok/s`, `5997` io_uring reads, only `2`
    host-prefetch hits.
  - Host-prefetch stats: `submitted=4159`, `hits=2`, `misses=5997`,
    `evicted=4156`, `no_slot=0`, `scan_passes=4159`, `used=9.19 MiB`.
  - Root cause: the revised slot picker preferred a ready slot with enough
    capacity before considering unused slots, so it repeatedly evicted the same
    few ready rows instead of filling the 512-slot queue. Next correction:
    prefer unused/non-ready slots first, then ready slots; within each class,
    prefer already-large-enough capacity before expandable capacity.
- 2026-06-13 04:22 CST:
  `fullpin-hostprefetch-ssdmiss-slotpick2-lead512-s512-m2048-n8` was also a
  failed attempt.
  - Result: `1.44 eval tok/s`, `3906` io_uring reads,
    `iouring_wait_us=4355199`.
  - Host-prefetch stats: `submitted=3397`, `hits=2093`, `misses=3906`,
    `evicted=824`, `no_slot=0`, `scan_passes=3397`, `used=2047.50 MiB`.
  - It preserved most of the read reduction but made the remaining wait path
    worse than the first targeted run (`3838` reads, `2865588` wait us,
    `1.74 eval tok/s`). Revert these slot/wait adjustments and keep the better
    producer-targeting baseline for the next parameter check. Next run:
    lead window `256`, same filtered SSD-miss trace, same non-strict full-pin
    diagnostic line.
- 2026-06-13 04:27 CST:
  Reverted the failed slot/wait adjustments and ran
  `fullpin-hostprefetch-ssdmiss-targeted-lead256-s512-m2048-n8`.
  - Result: `1.44 eval tok/s`, `1.00 prompt tok/s`, `total_ms=31363.32`,
    `read_failures=0`.
  - Reads were the lowest so far: `io_uring_reads=3791`, but wait worsened:
    `iouring_wait_us=4335720`.
  - Host-prefetch stats: `submitted=2682`, `hits=2208`, `misses=3791`,
    `duplicate_skips=5456817`, `reserved_skips=707`, `no_slot=1431799`,
    `scan_passes=1434481`, `used=2047.69 MiB`.
  - Interpretation: smaller lead improves hit count/read count, but it still
    causes heavy producer scan/slot pressure and worsens the remaining miss
    wait path. Since n8 is below both the full-pin baseline (`1.84 tok/s`) and
    the best targeted attempt (`1.74 tok/s`), do not run n84 for this path.

## Phase 24 Current Conclusion

The SSD-miss filtered host-prefetch path is mechanically correct and can reduce
fallback SSD reads on n8 (`5999 -> 3791-3838`), but it has not improved token
rate. Best Phase 24 n8 remains:

- `fullpin-hostprefetch-ssdmiss-targeted-lead512-s512-m2048-n8`:
  `1.74 eval tok/s`, `3838` io_uring reads, output prefix correct.

This is still below the full-pin no-host-prefetch n8 baseline
(`1.84 eval tok/s`, `5999` reads). The bottleneck is no longer just read count:
the extra producer thread, pinned host buffer pressure, and changed remaining
io_uring wait shape outweigh the saved reads. Keep host-prefetch env-gated as a
diagnostic, but do not promote it as a 5090 speedup and do not spend an n84 run
until an n8 configuration matches or beats the baseline.

Recommended next direction:

- replace the single producer with a bounded batch/queue design that sleeps
  without busy-scanning and preserves read order/batch shape;
- or step back from host-prefetch and focus on reducing H2D/CPU overhead in the
  existing blocking miss path, because current host-prefetch reduces reads but
  increases wait/coordination cost.

## Phase 25: Larger RAM-Tier Upper Bound

Phase 24 showed that reducing SSD reads through host-prefetch is not enough if
the implementation adds producer/coordination overhead. The cleaner upper-bound
diagnostic is to use more host RAM as a read-through tier and avoid the
remaining SSD miss path without a producer thread.

This is explicitly non-strict:

- not `MemoryMax=2G`;
- not `GGML_MOE_RAM_TIER_MIB=0`;
- results must not be reported as strict 2 GB performance.

Purpose:

- determine whether the current 5090 runtime can approach the theoretical
  token rate if residual SSD reads are mostly eliminated;
- separate "storage miss path is the remaining bottleneck" from "GPU/runtime
  compute path is still too slow even when reads are removed";
- evaluate whether more expert residency in RAM is a viable direction on
  machines that can spare host memory.

Plan:

- use offline cache replay to generate RAM-tier profiles larger than the
  current `9216 MiB` full-pin line;
- start with `11264 MiB`, because the machine currently has about `12 GiB`
  available and we must not interfere with other tasks;
- compare predicted residual coverage against the current `9216 MiB` profile;
- if prediction is meaningful and resources stay idle, run n8 first with
  `MemoryMax=15G`, `MemorySwapMax=0`, `GGML_MOE_RAM_TIER_MIB=11264`;
- only run n84 if n8 improves over the current full-pin n8 baseline
  (`1.84 eval tok/s`, `5999` SSD reads) or clearly shows lower read/wait cost.

Acceptance:

- output prefix correct;
- `read_failures=0`;
- report `RAM tier` size/hit rate separately;
- do not mix these numbers into strict 2GB/RAM=0 results.

- 2026-06-13 04:34 CST: Resources idle
  (`GPU 41 MiB used, 0% util`; host available about `12 GiB`). Starting
  offline planning for an `11264 MiB` RAM-tier diagnostic.
- 2026-06-13 04:36 CST: Generated larger RAM-tier residual profiles under
  `phase25-ram-tier-upper/`.
  - `10240 MiB`: `31832/70093` residual misses covered (`45.414%`),
    prefix coverage `2074/7773` (`26.682%`).
  - `11264 MiB`: `33960/70093` residual misses covered (`48.450%`),
    prefix coverage `2095/7773` (`26.952%`).
  - `12288 MiB`: `35917/70093` residual misses covered (`51.242%`),
    prefix coverage `2459/7773` (`31.635%`).
  - Interpretation: even 12 GiB RAM tier does not get close to eliminating
    residual SSD misses under the current VRAM cache split/profile; however,
    11 GiB is safe enough to run as a non-strict n8 diagnostic while the
    machine is idle. Do not run 12 GiB yet because host available memory is
    only about 12 GiB and another task may start.
- 2026-06-13 04:39 CST:
  `fullpin-ram11264-offset-sort-n8` completed but was not a win.
  - Result: `1.70 eval tok/s`, `0.85 prompt tok/s`, `total_ms=33381.76`,
    `read_failures=0`.
  - Reads/cache: `io_uring_reads=5678`, `VRAM hit=38.4%`,
    `RAM tier hits=2095/10299` (`20.3%`), resident `11264 MiB`.
  - This is slower than the older 10 GiB n8 line
    `residual-ramtier10240-graphprofile-nosplit-n8` (`1.99 eval tok/s`,
    `5699` reads, `2074` RAM hits), despite almost identical read/hit shape.
    Treat it as run-to-run/config noise plus higher memory pressure, not as a
    reason to run 11 GiB n84.
- 2026-06-13 04:43 CST: Re-read Phase 16 history. The most promising
  non-strict line remains:
  `residual-ramtier10240-graphprofile-nosplit-n8`, `1.99 eval tok/s`, but its
  n84 counterpart was killed by host cgroup OOM with full `10240 MiB` pinned
  RAM tier. Next attempt is to lower context memory (`--ctx 512`) while keeping
  the 10 GiB RAM tier/profile, then run n84 only if the short gate succeeds and
  resources remain idle.
- 2026-06-13 04:47 CST:
  `fullpin-ram10240-ctx512-offset-sort-n8` completed but failed the gate.
  - Result: `1.57 eval tok/s`, `0.98 prompt tok/s`, `total_ms=30908.24`,
    `read_failures=0`.
  - Reads/cache: `io_uring_reads=5699`, `VRAM hit=38.6%`,
    `RAM tier hits=2076/10303` (`20.1%`), resident `10240 MiB`.
  - This has the same read/RAM-hit shape as the successful 10 GiB n8, but
    decode is much worse (`1.57` vs `1.99`). Do not run the ctx512 n84. Next
    step is code inspection for RAM-tier initialization memory pressure, aiming
    to make the 10 GiB full-pin n84 fit without changing the fast runtime path.
- 2026-06-13 04:51 CST: Inspected RAM-tier initialization. The code does not
  retain an obvious multi-GiB temporary structure after loading; the previous
  10 GiB n84 failure is most likely the full pinned RAM-tier plus runtime
  memory peak exceeding this 15 GiB host's cgroup/physical limit. Since the old
  failed run used `MemoryMax=14G`, do one final controlled retry with the same
  fast full-pin runtime path and `MemoryMax=15G`, only while the machine is
  idle. If this still OOMs, stop the 10 GiB full-pin n84 line on this host.
- 2026-06-13 04:55 CST:
  `fullpin-ram10240-mem15g-offset-sort-n84` completed successfully.
  - Result: `2.50 eval tok/s`, `0.98 prompt tok/s`, `total_ms=59720.99`,
    `load_ms=26429.07`, `read_failures=0`.
  - Reads/cache: `io_uring_reads=38297`, `direct_reads=2526`,
    `VRAM hit=53.1%`, `RAM tier hits=31832/72655` (`43.8%`), resident
    `10240 MiB`.
  - This exactly matches the offline 10 GiB residual prediction (`31832`
    RAM-tier-covered misses) and reduces SSD reads versus the 9 GiB full-pin
    line (`40538 -> 38297`).
  - It does not improve decode over the best 9 GiB full-pin result
    (`2.65 eval tok/s`). The remaining io_uring batch shape got worse:
    `inflight_avg=1.97` versus about `2.3-2.4` on the 9 GiB best. Total time is
    better than the older 9 GiB repeat, but this still does not approach the
    theoretical `~3.9-5.5 tok/s` target.
  - Next gate: test whether higher io_uring depth/slot count restores batching
    on the 10 GiB profile. Run n8 with `depth=32`, `slots=32`; only run n84 if
    n8 improves over the 10 GiB depth16 n8 line.
- 2026-06-13 04:58 CST:
  `fullpin-ram10240-depth32-slots32-offset-sort-n8` completed.
  - Result: `1.86 eval tok/s`, `0.97 prompt tok/s`, `total_ms=30770.66`,
    `read_failures=0`.
  - Reads/cache: `io_uring_reads=5699`, `VRAM hit=38.4%`,
    `RAM tier hits=2074/10299` (`20.1%`), resident `10240 MiB`.
  - io_uring shape did not improve: `inflight_avg=2.34`, same as the depth16
    10 GiB n8 line. It also did not beat the historical depth16 10 GiB n8
    (`1.99 eval tok/s`).
  - Do not run depth32 n84. Larger queue/slot counts are not the missing
    lever; runtime batches remain dependency-shaped and small.

## Phase 25 Current Conclusion

10 GiB full-pinned RAM tier can now run n84 if `MemoryMax=15G`, and it reduces
SSD reads as predicted:

- `fullpin-ram10240-mem15g-offset-sort-n84`: `2.50 eval tok/s`,
  `38297` io_uring reads, `31832` RAM-tier hits, `53.1%` VRAM hit.

However, this does not beat the best 9 GiB full-pin decode result
(`2.65 eval tok/s`, `40538` reads). The extra RAM tier reduces reads but also
changes the remaining read/batch shape enough that decode does not improve.
The current non-strict upper-bound line is therefore still about `2.5-2.65
tok/s`, below the theoretical `~3.9-5.5 tok/s`.

Implication:

- simply adding more host RAM tier is not sufficient on this runtime;
- io_uring depth/slot increases are not sufficient because batches stay small;
- the next credible path must reduce per-layer dependency stalls or reduce the
  number of expert rows used/computed in a quality-safe, route-compatible way,
  rather than just moving more residual rows from SSD to RAM.

## Phase 26: Cache-Aware Selective Expert Reduction

Global SER was already rejected: `SER=7,1.05` preserved the tiny smoke score but
changed the long generated route enough to destroy oracle-profile locality, and
`SER=1,0.96` failed the smoke quality gate. Phase 25 also showed that moving
more residual rows from SSD to RAM is not enough. The next direction is a more
targeted reduction:

- do not reduce experts solely by score threshold;
- only consider dropping lower-rank selected experts that are predicted to miss
  both VRAM and RAM tiers;
- preserve profile-hot / resident experts to keep the generated route as close
  as possible to the oracle path;
- prove offline that this would remove a material number of SSD reads before
  adding another runtime path.

Plan:

- treat the oracle route trace's per-layer expert order as a top-k rank proxy;
- replay the current cache + RAM-tier profiles to label events as VRAM hit,
  RAM hit, or SSD miss;
- estimate saved rows if dropping only rank-tail SSD misses, for example
  rank >= 7 or rank >= 6 within each 8-expert group;
- only implement an env-gated runtime selective reduction if the offline
  reduction is large enough to plausibly close part of the gap and does not
  imply dropping too many profile/resident experts.

- 2026-06-13 05:03 CST: Starting offline rank-tail SSD-miss reduction analysis
  before touching runtime expert selection.
- 2026-06-13 05:06 CST: Added
  `analyze_tail_drop.py` and ran offline rank-tail analysis for the 9 GiB and
  10 GiB non-strict upper-bound lines.
  - 9 GiB RAM tier (`40502` predicted SSD misses):
    - keeping first 7 of 8 experts would save only `5081` SSD misses
      (`12.545%` of current SSD misses);
    - but it would also drop `13597` resident events already served by
      VRAM/RAM (`9392` VRAM hits + `4205` RAM hits).
  - 10 GiB RAM tier (`38261` predicted SSD misses):
    - keeping first 7 of 8 experts would save only `4786` SSD misses
      (`12.509%`);
    - but it would also drop `13892` resident events
      (`9392` VRAM hits + `4500` RAM hits).
  - More aggressive keep-first-N settings save more SSD reads but drop about
    `72-74%` resident events among removed events. The tail rank is not a good
    proxy for "cold SSD miss".
  - Interpretation: do not implement a pure tail/rank selective reduction.
    It would likely repeat global SER's failure mode by changing useful
    resident expert computation far more than it reduces SSD stalls. A runtime
    selective reduction would need actual score margins and cache residency
    together, not rank alone.

## Phase 27: Router Score/Margin Trace for Residency-Aware Reduction

Phase 26 showed that rank alone is a poor proxy for SSD misses: dropping the
last selected expert would save only about 12.5% of SSD reads while also
dropping many VRAM/RAM-resident events. The next diagnostic is to capture the
router scores/margins attached to the selected experts, then join those scores
with the existing cache replay labels.

Plan:

- first locate where selected MoE expert IDs are materialized during CUDA
  graph execution (`top_k` / `argsort` / `argsort_thresh`);
- add only env-gated diagnostics, with default runtime behavior unchanged;
- begin with a low-risk debug print of relevant argsort tensor names and
  shapes, capped to avoid noisy logs;
- if the source score tensor and selected ID buffer are available at that
  point, add a trace writer keyed by layer/row/rank/expert/score;
- run only a short n8 diagnostic first, then join the score trace with the
  existing route/cache replay to estimate whether low-score + SSD-miss experts
  can be skipped without discarding many resident/high-score experts;
- implement no runtime selective reduction unless the offline evidence is
  materially better than the Phase 26 rank-tail result.

Acceptance:

- default builds/runs produce no extra output unless trace env vars are set;
- diagnostic run has `read_failures=0` and a correct output prefix;
- trace records enough rows to align with route events;
- plan records exact env vars, log paths, and the offline selectivity result.

- 2026-06-13 03:23 CST: Starting Phase 27. The immediate goal is a diagnostic
  score/margin trace, not another SER runtime path. This avoids changing model
  behavior until we know whether router score and cache residency actually
  isolate a useful subset of SSD-miss expert loads.
- 2026-06-13 03:23 CST: Located the GLM MoE selection path:
  `llama-build-context.cpp` builds `ffn_moe_probs` / optional
  `ffn_moe_probs_biased`, then `ggml_top_k` or `ggml_top_k_thresh` produces
  `ffn_moe_topk`. CUDA executes the underlying `GGML_OP_ARGSORT` or
  `GGML_OP_ARGSORT_THRESH` in `ggml/src/ggml-cuda/argsort.cu`. Existing route
  traces are later recorded in `moe_stream_batch.cu` after only the selected
  expert ids are visible, so score capture needs to happen at argsort/top-k
  time or via a new propagated side channel.
- 2026-06-13 03:23 CST: Added a default-off diagnostic print controlled by
  `GGML_MOE_SCORE_TRACE_DEBUG=1`. It logs only likely MoE argsort node names
  and shapes, capped at 96 rows, and should not affect default runtime
  behavior.
- 2026-06-13 03:23 CST: Added runner plumbing for reproducibility:
  `--score-trace-debug` sets `GGML_MOE_SCORE_TRACE_DEBUG=1`, and
  `--score-trace-out` sets `GGML_MOE_SCORE_TRACE_OUT=<path>` for the next CSV
  trace implementation.
- 2026-06-13 03:23 CST: Ran `argsort-debug-n8` under strict 2GB/RAM=0. It
  completed with `read_failures=0`, `1.47 eval tok/s`, `7501` io_uring reads,
  `40.6%` VRAM hit, and route trace `12624` events. No argsort debug lines
  appeared, confirming this GLM path is not using the plain CUDA
  `ggml_cuda_op_argsort` entry.
- 2026-06-13 03:23 CST: Found the actual optimized path:
  `cuda_glm45moe_experts()` in `argsort.cu`. The graph fusion sees the
  top-k view, the underlying full top-k id buffer, the router probability
  tensor, the expert bias tensor, and the output weights. Added default-off
  `GGML_MOE_SCORE_TRACE_OUT=<csv>` support there, plus fused-path debug prints.
  This diagnostic synchronizes/copies device data only when the trace output
  env var is set.
- 2026-06-13 03:29 CST: First `score-trace-n8` failed before generation.
  Fused-path debug confirmed the needed tensors are visible:
  `topk_view=ffn_moe_topk-3`, `topk=ffn_moe_probs_biased-3 (sort)`,
  `probs=ffn_moe_logits-3`, `bias=blk.3.exp_probs_b.bias`, `n_experts=256`,
  `topk=8`. Failure reason was `operation not permitted when stream is
  capturing`: graph reuse captures the CUDA stream, so the diagnostic cannot
  synchronize/copy from inside capture. Added a guard to skip score trace rows
  during capture and will rerun the diagnostic with `--no-graph-reuse`.
- 2026-06-13 03:31 CST: `score-trace-n8-nograph` completed but wrote only the
  CSV header. Even with `--no-graph-reuse`, CUDA backend graph capture still
  occurs, and the capture guard skips all rows. Added runner flag
  `--cuda-disable-graphs` to set `GGML_CUDA_DISABLE_GRAPHS=1` for this
  diagnostic only.
- 2026-06-13 03:32 CST: `score-trace-n8-disable-cudagraphs` succeeded and
  wrote `14272` score rows, with `read_failures=0`, `7496` io_uring reads, and
  `40.6%` VRAM hit. However, the run used `--profile oracle-n84`, which
  resolves to `/home/wici/lfz/ik_llama/oracle-n84`; preload failed with
  `profile preload: open failed`. This run validates the score trace
  mechanism but must not be used for cache/residency conclusions. Rerun with
  the correct profile path
  `.Agent/plans/5090-theoretical-token-rate/oracle-n84.route.csv`.
- 2026-06-13 03:34 CST: Reran with the correct oracle profile:
  `score-trace-oracle-n8-disable-cudagraphs`.
  - Config: strict `MemoryMax=2G`, `MemorySwapMax=0`, RAM tier `0`,
    io_uring offset-sort, oracle route profile, `GGML_CUDA_DISABLE_GRAPHS=1`
    and `--no-graph-reuse` for diagnostic score copying.
  - Result: `1.41 eval tok/s`, `1.73 prompt tok/s`, `total_ms=24866.74`,
    `read_failures=0`, `7811` io_uring reads, `2513` direct preload reads,
    `38.1%` VRAM hit.
  - Trace files:
    `runs/phase27-score-trace/score-trace-oracle-n8-disable-cudagraphs/score-trace.csv`
    (`14272` score rows) and `route-trace.csv` (`12624` route events).
  - Note: diagnostic CUDA graph disabling and score copies make this a trace
    capture, not a performance result.
- 2026-06-13 03:35 CST: Added `analyze_score_trace.py` and matched score
  groups to route groups. With `--match-window 4096`, all `526/526` route
  groups matched. Summary:
  `runs/phase27-score-trace/score-trace-oracle-n8-disable-cudagraphs/score-analysis-window4096.json`.
  - Total replay: `12624` events, `4813` VRAM hits, `7811` SSD misses, RAM `0`.
  - Rank is correlated with SSD miss: rank0 `38.3%` SSD, rank7 `77.6%` SSD.
    Keeping first 7/8 would save `1225` SSD misses (`15.7%` of current SSD)
    while dropping `353` resident VRAM events (`22.4%` of dropped events).
  - Weight ratio is also correlated but not clean. Dropping
    `weight/max_weight < 0.20` would save `1541` SSD misses (`19.7%`) but drop
    `640` resident events (`29.3%` of dropped events). At `<0.30`, it saves
    `3016` SSD misses (`38.6%`) but drops `1259` resident events.
  - An oracle rule "drop only SSD-miss rows with low weight" would save the
    same SSD reads with zero resident drops, but that requires runtime cache
    residency awareness before committing to the expert set. Pure score or rank
    thresholding is still too blunt.

## Phase 27 Current Conclusion

The new score trace answers the main question from Phase 26:

- router rank and normalized weight are meaningfully correlated with SSD
  misses;
- the correlation is not clean enough for a pure score/rank SER threshold;
- a useful runtime path would need to check cache/RAM residency before pruning,
  e.g. "only skip a selected expert if it is not in VRAM, not in RAM tier, and
  its normalized weight is below a threshold";
- for the strict RAM=0 line this becomes "skip low-weight selected experts
  only when they would trigger an SSD expert-pack read";
- the likely implementation point is not `argsort.cu`; it is the MoE streaming
  path where `active_experts[j]`, tensor name, VRAM cache lookup, RAM tier, and
  pack lookup are all visible.

Do not run a long n84 with pure score/rank reduction. The next implementation
candidate should be env-gated cache-aware pruning in `moe_stream_batch.cu`,
with an n8 gate first:

- keep all resident experts;
- keep all high-weight experts;
- optionally skip only low-weight experts that would miss VRAM/RAM and read
  from SSD;
- report skipped count, skipped SSD reads, and any change in output prefix.

- 2026-06-13 03:38 CST: Build and Python checks pass after score trace and
  analysis tooling:
  `cmake --build build-cuda --target llama-cli -j 8` and
  `python3 -m py_compile run_5090_matrix.py analyze_score_trace.py`.
- 2026-06-13 03:38 CST: Inspected runtime implementation point. The CUDA
  streaming entrypoints in `moe_stream_batch.cu` currently receive active
  expert ids via `matrix_row_counts` / `matrix_rows`, but not router weights.
  The selected weights are available earlier in the graph (`ffn_moe_weights`)
  and in the fused score diagnostic, but they are not propagated into the MoE
  streaming calls. Therefore a real cache-aware low-weight skip needs an
  additional data path from `ggml.c` matrix-row construction into
  `ggml_cuda_moe_stream_up_gate_batch` / `ggml_cuda_moe_stream_batch`.
  Implementing that directly is possible but higher risk than the diagnostic
  work because it changes cross-backend op plumbing.
- 2026-06-13 03:38 CST: Safer next step identified: add an env-gated online
  estimator in `ggml.c` row grouping before changing execution. It can use
  selected expert rank and `ggml_cuda_moe_stream_cache_contains()` to estimate
  how many low-rank expert rows would be skipped only when they are currently
  non-resident. This does not capture true router weight, but it can validate
  on n84 whether the n8 score/rank signal persists under the exact runtime
  cache state. Avoid changing `ggml_moe_row_mapping` ABI until that estimator
  justifies the risk.

## Phase 28: Online Cache-Aware Skip Estimator

Before changing execution or extending `ggml_moe_row_mapping`, add a
default-off estimator that only counts hypothetical skips:

- enable with `GGML_MOE_SKIP_ESTIMATE=1`;
- choose rank cutoff with `GGML_MOE_SKIP_ESTIMATE_KEEP=<N>`, default `7`,
  meaning ranks `>= N` are candidates;
- run after profile preload and before CUDA streaming so `cache_contains()`
  observes the runtime VRAM cache state;
- count up, gate, and down tensor accesses separately;
- report candidate events, resident-preserved events, and nonresident
  low-rank events that would likely become skipped SSD reads;
- do not change `matrix_row_counts`, expert ids, output rows, cache admission,
  or model behavior.

Acceptance:

- default runs have no new output;
- enabled n8 run completes with `read_failures=0` and same output prefix;
- estimator counters are printed at exit and recorded here;
- only if n8/n84 estimates are materially better than Phase 27 offline pure
  rank/score results should we consider adding real weight/rank metadata to
  the streaming ABI.

- 2026-06-13 03:39 CST: Implemented the default-off estimator in `ggml.c`.
  It adds `GGML_MOE_SKIP_ESTIMATE=1` and `GGML_MOE_SKIP_ESTIMATE_KEEP=<N>`.
  The estimator runs only in the CUDA streaming path after preload and before
  the streaming call, uses `ggml_cuda_moe_stream_cache_contains()` to classify
  low-rank candidates as resident/nonresident, and prints a single
  `[moe_skip_est]` summary at exit. It does not mutate `matrix_row_counts`,
  selected expert ids, cache state, or output tensors.
- 2026-06-13 03:40 CST: Ran strict n8 estimator
  `skip-est-keep7-oracle-n8`.
  - Config: strict `MemoryMax=2G`, `MemorySwapMax=0`, RAM tier `0`,
    oracle route profile, io_uring offset-sort, normal CUDA graphs enabled,
    `GGML_MOE_SKIP_ESTIMATE=1`, `GGML_MOE_SKIP_ESTIMATE_KEEP=7`.
  - Result: output prefix correct, `read_failures=0`, `1.42 eval tok/s`,
    `1.66 prompt tok/s`, `total_ms=25261.97`, `7812` io_uring reads,
    `38.1%` VRAM hit.
  - Estimator:
    `groups=8416 events=12624 keep=7 candidates=1578 nonresident=1225
    resident=353 nonresident_pct_events=9.7 resident_pct_candidates=22.4
    upgate_nonresident=822 down_nonresident=403`.
  - Interpretation: online estimator matches the offline score/rank replay.
    A real "skip only low-rank nonresident" path could avoid about `1225/7812`
    (`15.7%`) runtime reads for this n8 case while preserving resident
    candidates. This is helpful but not enough by itself to close the 5090
    theoretical gap; n84 estimator is needed before changing execution.
- 2026-06-13 03:42 CST: Ran strict n84 estimator
  `skip-est-keep7-oracle-n84`.
  - Config: same strict 2GB/RAM=0 oracle-profile line, normal CUDA graphs,
    `GGML_MOE_SKIP_ESTIMATE=1`, `GGML_MOE_SKIP_ESTIMATE_KEEP=7`.
  - Result: output prefix correct, `read_failures=0`, `1.74 eval tok/s`,
    `1.64 prompt tok/s`, `total_ms=68012.98`, `70525` io_uring reads,
    `52.8%` VRAM hit.
  - Estimator:
    `groups=99616 events=149424 keep=7 candidates=18678 nonresident=13066
    resident=5612 nonresident_pct_events=8.7 resident_pct_candidates=30.0
    upgate_nonresident=8808 down_nonresident=4258`.
  - Interpretation: a real low-rank/nonresident skip would avoid about
    `13066/70525` (`18.5%`) runtime reads on this n84 route if implemented
    perfectly. That is nontrivial, but still not enough by itself to explain
    the full 5090 theoretical gap. Also, actual skipping cannot be done by
    independently pruning only one streaming call: up/gate and down rows must
    remain consistent, and skipped intermediate rows must be explicitly zeroed
    or represented as skipped across both ops.

## Phase 29: Wider Rank-Cutoff Estimator Sweep

Before implementing real skipping, quantify whether more aggressive rank
cutoffs are worth the risk. Keep this diagnostic-only:

- run strict 5090 line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- reuse the oracle n84 route profile and io_uring offset-sort configuration;
- run `GGML_MOE_SKIP_ESTIMATE_KEEP=6` and `KEEP=5`;
- first verify n8 output prefix and read failures, then run n84 if the machine
  is idle;
- report eval tok/s, prompt tok/s, total_ms, direct/io_uring reads, VRAM hit,
  read_failures, and estimator counters;
- do not stop or disturb any other process if the GPU or host RAM is occupied.

Acceptance:

- all runs remain behavior-preserving (`read_failures=0`, expected output
  prefix);
- estimator output is recorded here with exact log paths;
- decide whether rank-only nonresident skipping has enough read-saving upside
  to justify implementation, or whether the next improvement should target the
  I/O pipeline / cache residency instead.

- 2026-06-13 03:49 CST: Started an initial n8 `KEEP=6` run but marked it
  invalid for comparison because the command accidentally changed the cache
  setup: `vram_cache_mib=13498`, `profile_reserve_pct=0`,
  `cache_profile_after=0`, `hybrid_profile_lfu_lru`, no GPU handoff, and no
  prompt dynamic experts. It produced `100.0%` VRAM hit and `0` io_uring
  runtime reads, so it is not comparable with the Phase 28 strict baseline.
  Log path kept only for audit:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase29-rank-cutoff-est/skip-est-keep6-oracle-n8/`.
  Re-running with the exact Phase 28 baseline shape next.
- 2026-06-13 03:50 CST: Ran strict n8 `KEEP=6` with the Phase 28 baseline
  config:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase29-rank-cutoff-est/skip-est-keep6-oracle-n8-baselinecfg/`.
  Result: `1.44 eval tok/s`, `1.66 prompt tok/s`, `total_ms=25510.09`,
  `read_failures=0`, `7812` io_uring reads, `38.1%` VRAM hit.
  Estimator:
  `groups=8416 events=12624 keep=6 candidates=3156 nonresident=2405
  resident=751 nonresident_pct_events=19.1 resident_pct_candidates=23.8
  upgate_nonresident=1616 down_nonresident=789`.
  Interpretation: perfect low-rank/nonresident skipping would avoid
  `2405/7812` (`30.8%`) runtime reads on n8, but rank-only candidates include
  `751` resident events.
- 2026-06-13 03:51 CST: Ran strict n8 `KEEP=5` with the same baseline config:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase29-rank-cutoff-est/skip-est-keep5-oracle-n8-baselinecfg/`.
  Result: `1.43 eval tok/s`, `1.66 prompt tok/s`, `total_ms=25933.12`,
  `read_failures=0`, `7812` io_uring reads, `38.1%` VRAM hit.
  Estimator:
  `groups=8416 events=12624 keep=5 candidates=4734 nonresident=3518
  resident=1216 nonresident_pct_events=27.9 resident_pct_candidates=25.7
  upgate_nonresident=2364 down_nonresident=1154`.
  Interpretation: theoretical skipped SSD reads rise to `45.0%`, but the
  candidate set is broad enough that a plain rank cutoff would likely harm
  quality unless it is strictly residency-aware.
- 2026-06-13 03:53 CST: Ran strict n84 `KEEP=6` with the Phase 28 baseline
  config:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase29-rank-cutoff-est/skip-est-keep6-oracle-n84-baselinecfg/`.
  Result: `1.77 eval tok/s`, `1.71 prompt tok/s`, `total_ms=67199.15`,
  `read_failures=0`, `70525` io_uring reads, `52.8%` VRAM hit.
  Estimator:
  `groups=99616 events=149424 keep=6 candidates=37356 nonresident=25019
  resident=12337 nonresident_pct_events=16.7 resident_pct_candidates=33.0
  upgate_nonresident=16882 down_nonresident=8137`.
  Interpretation: perfect low-rank/nonresident skipping would avoid
  `25019/70525` (`35.5%`) runtime reads on n84.
- 2026-06-13 03:54 CST: Ran strict n84 `KEEP=5` with the same baseline config:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase29-rank-cutoff-est/skip-est-keep5-oracle-n84-baselinecfg/`.
  Result: `1.76 eval tok/s`, `1.65 prompt tok/s`, `total_ms=67859.81`,
  `read_failures=0`, `70525` io_uring reads, `52.8%` VRAM hit.
  Estimator:
  `groups=99616 events=149424 keep=5 candidates=56034 nonresident=36116
  resident=19918 nonresident_pct_events=24.2 resident_pct_candidates=35.5
  upgate_nonresident=24420 down_nonresident=11696`.
  Interpretation: the optimistic read-saving ceiling is `36116/70525`
  (`51.2%`), but it would touch more than one third of all candidate events
  that are already resident. The next implementation must therefore be
  residency-aware and keep up/gate/down consistency; a rank-only execution
  cutoff is not acceptable.

## Phase 29 Current Decision

Rank is useful enough to continue, but only as a candidate selector. The next
candidate implementation should be default-off and constrained:

- skip only `rank >= KEEP`;
- skip only if the expert row is not already resident in VRAM cache;
- preserve resident low-rank rows;
- apply one consistent decision across up, gate, and down for the same selected
  token/layer/rank/expert, rather than pruning each tensor independently;
- first test on n8 and compare output prefix / read failures before any n84
  run.

If the existing row plumbing cannot carry a consistent skip decision without
large ABI churn, stop and switch to I/O pipeline/cache improvements instead.

## Phase 30: Default-Off Consistent Low-Rank Nonresident Skip Prototype

Prototype an execution-changing path only after the Phase 29 estimator sweep.

Design constraints:

- env gate: `GGML_MOE_SKIP_NONRESIDENT=1`;
- cutoff: `GGML_MOE_SKIP_NONRESIDENT_KEEP=<N>`, same meaning as estimator;
- default behavior must remain unchanged;
- only consider decode rows where each selected rank maps one token row;
- skip candidate only when `rank >= KEEP`;
- require all related tensors for the same token/layer/rank/expert to be
  nonresident before skipping:
  - up tensor not in VRAM;
  - gate tensor not in VRAM;
  - matching down tensor not in VRAM;
- zero the skipped up/gate output row so downstream math sees a zero
  contribution;
- rebuild down `matrix_rows` with the same decision rule so down does not read
  an expert for a row that was zeroed upstream;
- counters must report total events, skipped routes, resident-preserved routes,
  and missing-down-reg cases;
- first test only n8 strict 2GB/RAM=0 and compare output prefix/read failures.

If this requires changing shared `ggml_moe_row_mapping` ABI or CUDA function
signatures, stop and do not implement in this phase.

- 2026-06-13 04:10 CST: Implemented a default-off prototype without changing
  CUDA function signatures or `ggml_moe_row_mapping` ABI.
  - New envs: `GGML_MOE_SKIP_NONRESIDENT=1`,
    `GGML_MOE_SKIP_NONRESIDENT_KEEP=<N>`.
  - Up/gate row construction skips only rank candidates whose up, gate, and
    matching down tensors are all nonresident in VRAM cache; skipped up/gate
    output rows are zeroed.
  - Down row construction skips only rows that were recorded as skipped by the
    matching up/gate decision, preserving cross-op consistency.
  - CUDA up/gate handoff path now zeros the fused device buffer when this env
    is enabled so filtered rows cannot leave stale device data for down.
  - Build passed:
    `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`.
  - Next: strict n8 smoke with `KEEP=7`; only proceed to longer runs if output
    prefix and `read_failures=0` look acceptable.
- 2026-06-13 04:12 CST: First strict n8 execution smoke
  `skip-exec-keep7-oracle-n8-baselinecfg` completed but was marked
  non-comparable. It ran, `read_failures=0`, and reduced io_uring reads only
  from `7812` to `7710`, but it also applied skip during prompt grouping and
  changed the prompt/cache path:
  - `1.39 eval tok/s`, `1.73 prompt tok/s`, `total_ms=25422.27`;
  - `7710` io_uring reads, `32.2%` VRAM hit;
  - `[moe_skip_exec] routes=28544 keep=7 candidates=3568 skipped=838
    preserved_resident=106 preserved_missing_meta=1259 upgate_skipped=419
    down_skipped=419`.
  The low VRAM hit and output change made it unsuitable as a baseline
  comparison. Fixed the prototype to apply only when `ids->ne[1] == 1`
  (decode rows), rebuilt successfully, and will rerun the n8 smoke.
- 2026-06-13 04:14 CST: Decode-only rerun
  `skip-exec-keep7-oracle-n8-decodeonly` still showed the same weak read
  reduction (`7710` io_uring reads) and low `32.2%` VRAM hit. Counter:
  `[moe_skip_exec] routes=8416 keep=7 candidates=1052 skipped=838
  preserved_resident=106 preserved_missing_meta=1 upgate_skipped=419
  down_skipped=419`.
  Diagnosis: skip filtering was still performed before profile preload, so it
  could classify rows as nonresident before profile preload filled the VRAM
  cache. Moved filtering to after the profile-preload barrier and rebuilt
  successfully. Next rerun should compare against Phase 28/29 baseline shape.
- 2026-06-13 04:16 CST: Post-preload-filter rerun
  `skip-exec-keep7-oracle-n8-afterpreload` produced the same shape:
  `1.43 eval tok/s`, `1.75 prompt tok/s`, `total_ms=24765.68`,
  `read_failures=0`, `7710` io_uring reads, `32.2%` VRAM hit.
  Counter remained:
  `[moe_skip_exec] routes=8416 keep=7 candidates=1052 skipped=838
  preserved_resident=106 preserved_missing_meta=1 upgate_skipped=419
  down_skipped=419`.
  Deeper diagnosis: the authoritative profile preload happens inside the CUDA
  streaming functions (`preload_profile_for_tensor()` in
  `moe_stream_batch.cu`), after the CPU-side row filter runs. Therefore a
  CPU-side filter cannot see the final cache state. This explains why the
  execution skip path still skips rows that would otherwise become profile
  resident, lowers reported VRAM hit, and fails to deliver the estimator's
  expected read savings.

## Phase 30 Current Decision

Do not run n84 with the current execution skip prototype. It is default-off and
builds, but its decision point is too early relative to CUDA internal profile
preload. A correct implementation would need to move the skip decision inside
`ggml_cuda_moe_stream_up_gate_batch()` / `ggml_cuda_moe_stream_batch()` after
their profile preload, while preserving a shared up/gate/down decision. That is
a larger CUDA-side change and should not be mixed with the I/O work.

For the current 5090 token-rate goal, the better next step is to focus on the
observed I/O bottleneck:

- n84 strict baseline still has `70525` io_uring reads and inflight average
  only `~1.75` despite configured depth 16;
- Phase 29 suggests read elimination has a ceiling, but current CPU-side skip
  cannot realize it correctly;
- improving batching/coalescing/prefetch inside the CUDA streaming path is more
  likely to help without changing model quality.
- 2026-06-13 04:18 CST: Verified default-off safety after the skip code:
  `default-off-baseline-n8-after-skip-code`.
  Result returned to the Phase 28/29 baseline shape: `1.40 eval tok/s`,
  `1.77 prompt tok/s`, `total_ms=24552.66`, `read_failures=0`, `7812`
  io_uring reads, `38.1%` VRAM hit, output prefix back to
  `(Hint: answer D; see section ...)`. This confirms the current prototype does
  not affect normal runs unless `GGML_MOE_SKIP_NONRESIDENT=1` is set.

## Phase 31: CUDA Streaming I/O Pipeline Utilization

The CPU-side skip prototype is not a good path for n84 because its decision
point is earlier than CUDA internal profile preload. Shift the next attempt to
I/O utilization inside the CUDA streaming path, preserving model behavior.

Problem evidence:

- strict n84 baseline still performs `70525` io_uring reads;
- configured `GGML_MOE_IO_DEPTH=16`, but observed inflight average is only
  about `1.75`, max `4`;
- most batches fall into size `1` or `2-4`; no `5-8`/higher batches;
- io wait time dominates enough that reducing per-read latency or increasing
  read concurrency is more likely to help than CPU-side row filtering.

Plan:

- inspect `expert_pack_iouring_copy_jobs()` and its callers for why jobs are
  submitted in very small batches;
- prefer an env-gated or conservative change that does not alter selected
  experts or model outputs;
- try to group more pending stage jobs per tensor/call before waiting;
- keep strict line unchanged: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- first verify with n8, then run n84 only if n8 improves or preserves output
  with no read failures.

Acceptance:

- default/off behavior remains reproducible;
- new run has `read_failures=0` and stable output prefix;
- report eval tok/s, prompt tok/s, total_ms, io_uring reads, submit/wait time,
  inflight average/max, batch histogram, VRAM hit, and log paths;
- if the code path cannot raise batch size/inflight without intrusive
  scheduler changes, record that and move to cache/profile tuning instead.

- 2026-06-13 04:20 CST: Inspected `expert_pack_iouring_copy_jobs()` and its
  callers. The io_uring loop can submit up to `min(IO_DEPTH, ring.slots)` jobs
  in one batch, but the callers often pass very small job vectors:
  - up/gate parallel stage first splits into up and gate jobs;
  - with `GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT=1`, each of those is split again
    over an auxiliary copy stream, yielding four small vectors;
  - down parallel staging splits down jobs across two copy streams;
  - this matches observed histograms: mostly batch size `1` and `2-4`, never
    `5-8` or higher.

First low-risk test: do not change code; run the strict n8 baseline without
`--up-gate-stage-split` so each up/gate tensor keeps a larger job vector. If
batch size/inflight improves and eval speed does not regress, test n84. If it
regresses, keep the split path and consider a CUDA-side aggregate submit
mechanism instead.
- 2026-06-13 04:21 CST: Ran strict n8 no-code config
  `no-upgate-stage-split-n8`:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase31-io-pipeline/no-upgate-stage-split-n8/`.
  Result: `1.38 eval tok/s`, `1.75 prompt tok/s`, `total_ms=25124.98`,
  `read_failures=0`, `7812` io_uring reads, `38.1%` VRAM hit.
  I/O changed as intended:
  - batches `3046 -> 2038`;
  - inflight avg `~1.92 -> 2.84`;
  - inflight max `4 -> 8`;
  - batch histogram gained `5-8:686`;
  - io wait fell from about `8.08s` to `5.53s`.
  But eval speed regressed versus default-off baseline (`1.40-1.44`), likely
  because removing the up/gate split sacrifices parallel CPU staging / CUDA
  stream overlap. Do not adopt this as the default. It does prove that the
  small batches are a caller-side splitting artifact.
- 2026-06-13 04:22 CST: Added runner flag `--no-down-parallel-stage` and ran
  strict n8 `no-down-parallel-stage-n8`:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase31-io-pipeline/no-down-parallel-stage-n8/`.
  Result: `1.08 eval tok/s`, `1.77 prompt tok/s`, `total_ms=26037.01`,
  `read_failures=0`, `5282` io_uring reads, `5035` direct reads, `38.1%`
  VRAM hit. Disabling down parallel staging badly hurts decode despite fewer
  io_uring reads, because direct/staging work shifts back into the synchronous
  path. Do not adopt.

Conclusion from no-code tests:

- up/gate stage split and down parallel stage are important for overlap;
- simply increasing per-ring batch size by disabling split regresses speed;
- a useful change would need to preserve parallel H2D/compute overlap while
  aggregating or pre-submitting the underlying pack reads.

Next low-cost diagnostic: test `GGML_MOE_IO_SQPOLL=1` on the strict baseline.
This keeps the split/overlap structure unchanged and may reduce io_uring
submission/wakeup overhead on the fast SSD. Run n8 first; only run n84 if n8
improves or at least does not regress.
- 2026-06-13 04:24 CST: Ran strict n8 `sqpoll-n8`:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase31-io-pipeline/sqpoll-n8/`.
  Result: `1.55 eval tok/s`, `1.76 prompt tok/s`, `total_ms=24202.20`,
  `read_failures=0`, `7812` io_uring reads, `38.1%` VRAM hit.
  I/O stats:
  - same batch shape/inflight (`inflight_avg=1.91`, `max=4`);
  - submit time dropped sharply (`~0.98s -> 0.013s`);
  - wait time rose (`~8.08s -> 9.18s`);
  - net decode improved over default-off n8 (`1.40-1.44 -> 1.55 tok/s`).
  This is worth an n84 strict validation.
- 2026-06-13 04:26 CST: Ran strict n84 `sqpoll-n84`:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase31-io-pipeline/sqpoll-n84/`.
  Result: `1.92 eval tok/s`, `1.74 prompt tok/s`, `total_ms=63290.47`,
  `read_failures=0`, `70525` io_uring reads, `52.8%` VRAM hit.
  I/O stats:
  - same reads and batch shape as baseline (`inflight_avg=1.75`, `max=4`);
  - submit time dropped sharply (`~7.26s -> 0.097s`);
  - wait time rose (`~77.5s -> 85.3s`);
  - decode improved versus recent strict baseline (`1.74-1.77 -> 1.92 tok/s`).
  Conclusion: `GGML_MOE_IO_SQPOLL=1` is a real strict-line win on this fast
  SSD despite unchanged batch size. It should be included in the current best
  strict 5090 configuration unless future runs show instability.

Follow-up diagnostic: combine SQPOLL with no up/gate stage split on n8 only.
Without SQPOLL, removing split raised inflight but regressed eval. With SQPOLL,
submission overhead is much lower, so the tradeoff might shift. Do not run n84
unless n8 beats `sqpoll-n8`.
- 2026-06-13 04:27 CST: Ran strict n8
  `sqpoll-no-upgate-stage-split-n8`.
  Result: `1.52 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24498.60`,
  `read_failures=0`, `7812` io_uring reads, `38.1%` VRAM hit.
  It improved over no-SQPOLL/no-split (`1.38`) but did not beat SQPOLL with
  the existing split (`1.55`). Do not run n84 for this combination.

## Phase 31 Current Best Strict Config

For strict 5090 (`MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`) the best
configuration found in this phase is the Phase 28 baseline plus
`GGML_MOE_IO_SQPOLL=1`:

- keep `GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT=1` /
  `GGML_MOE_STREAM_UP_GATE_SPLIT_STAGE=1`;
- keep `GGML_MOE_DOWN_PARALLEL_STAGE=1`;
- keep io_uring offset sort;
- add `GGML_MOE_IO_SQPOLL=1`.

Best strict n84 so far in this phase:

- `sqpoll-n84`: `1.92 eval tok/s`, `1.74 prompt tok/s`,
  `total_ms=63290.47`, `read_failures=0`, `70525` io_uring reads,
  `52.8%` VRAM hit.

This recovers the earlier strict best range but is still far below the
theoretical 5090 target. Remaining gap is not from submit overhead alone; the
batch-size/inflight ceiling and total SSD read count still dominate.

- 2026-06-13 04:28 CST: Verification after Phase 31:
  - `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
    passed.
  - `python3 -m py_compile run_5090_matrix.py analyze_score_trace.py` passed.
  - `git diff --check` passed.

Next implementation choices:

- Make SQPOLL part of the recommended strict 5090 config in future runners and
  comparisons.
- For further speedups, avoid disabling split/parallel staging; instead design
  a cross-ring aggregate pack-read scheduler or a cache-profile change that
  reduces total SSD reads.
- The current CPU-side skip prototype remains default-off and should not be
  used for performance claims until the decision is moved inside CUDA after
  profile preload.

## Phase 32: Strict VRAM Cache Headroom Sweep With SQPOLL

Phase 31 recovered strict n84 to `1.92 tok/s` via SQPOLL, but read count remains
`70525`. The next direct route toward the goal is to reduce SSD reads by using
more of the 5090's 32GB VRAM for expert cache, while preserving the strict host
RAM line.

Hypothesis:

- Previous strict baselines used `GGML_MOE_VRAM_CACHE_MIB=12288` plus graph
  reserve/clamp, yielding actual cache about `11.45 GiB`;
- the 5090 still reports substantial free VRAM at startup, but CUDA graphs and
  dense model memory need reserve;
- a moderate cache increase may preload more oracle-profile experts, raise
  VRAM hit, and reduce io_uring reads without touching host RAM;
- SQPOLL should stay enabled because it is a strict-line win.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep Phase 31 best flags: SQPOLL, offset sort, up/gate stage split, down
  parallel stage, GPU handoff, prompt dynamic experts;
- sweep `GGML_MOE_VRAM_CACHE_MIB` upward with auto clamp and graph reserve:
  start n8 at `14336`, then `16384`, then stop if OOM/clamp instability or
  speed regression is clear;
- keep `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=10` and `cache_profile_after=12624`
  unless evidence suggests profile reserve is the limiter;
- run n84 only for the best n8 candidate.

Acceptance:

- `read_failures=0`;
- output prefix remains stable;
- report requested vs actual cache, VRAM hit, io_uring reads, eval tok/s,
  prompt tok/s, total_ms, and logs;
- do not use host RAM tier or any non-strict memory setting.

- 2026-06-13 04:31 CST: Ran strict n8 `sqpoll-vram14336-n8`.
  Result: `1.54 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24358.35`,
  `read_failures=0`, `7812` io_uring reads, `38.1%` VRAM hit.
  The requested cache increase had no effect because auto-clamp reported
  `requested=14336 MiB actual=11450 MiB`, identical to the `12288` request.
  Actual split remained `upgate=1787 slots`, `down=997 slots`. Do not continue
  increasing requested cache without changing clamp/reserve; it will not add
  experts.

Next diagnostic: keep strict host RAM, SQPOLL, and auto-clamp, but reduce
`GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB` from `2048` to `1024` for n8. If actual
cache increases and the run remains stable, test n84; otherwise stop this
cache-headroom path.
- 2026-06-13 04:32 CST: Ran strict n8 `sqpoll-graphreserve1024-n8`.
  Result: `1.58 eval tok/s`, `1.73 prompt tok/s`, `total_ms=24189.24`,
  `read_failures=0`, `7578` io_uring reads, `40.0%` VRAM hit.
  Cache actually increased:
  - requested `14336 MiB`, actual `12474 MiB`;
  - upgate cache `1787 -> 1947` slots;
  - down cache `997 -> 1086` slots.
  Compared with `sqpoll-n8`, this reduced runtime io_uring reads by `234`
  (`3.0%`) and improved decode `1.55 -> 1.58 tok/s`. Run n84 strict validation.
- 2026-06-13 04:34 CST: Ran strict n84 `sqpoll-graphreserve1024-n84`.
  Result: `2.00 eval tok/s`, `1.71 prompt tok/s`, `total_ms=61698.27`,
  `read_failures=0`, `66627` io_uring reads, `55.4%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `12474 MiB`;
  - upgate cache `1947` slots, hit `54.3%`;
  - down cache `1086` slots, hit `57.7%`.
  Compared with `sqpoll-n84`, this reduces io_uring reads by `3898`
  (`5.5%`), raises VRAM hit `52.8% -> 55.4%`, and improves decode
  `1.92 -> 2.00 tok/s`. This is the new strict current best.

Next diagnostic: n8 only with `graph_reserve=512` to see whether even more
VRAM cache can be used safely. Do not run n84 unless n8 improves and there are
no graph/OOM symptoms.
- 2026-06-13 04:37 CST: Ran strict n8 `sqpoll-graphreserve512-n8`.
  Result: `1.60 eval tok/s`, `1.69 prompt tok/s`, `total_ms=24478.28`,
  `read_failures=0`, `7472` io_uring reads, `40.8%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `12986 MiB`;
  - upgate cache `2026` slots;
  - down cache `1130` slots.
  Compared with graphreserve1024 n8, this reduced another `106` io_uring reads
  and improved decode `1.58 -> 1.60 tok/s`, with no OOM/graph error. Run n84
  validation; watch prompt speed and graph stability.
- 2026-06-13 04:39 CST: Ran strict n84 `sqpoll-graphreserve512-n84`.
  Result: `2.03 eval tok/s`, `1.70 prompt tok/s`, `total_ms=60859.87`,
  `read_failures=0`, `64798` io_uring reads, `56.6%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `12986 MiB`;
  - upgate cache `2026` slots, hit `55.5%`;
  - down cache `1130` slots, hit `58.9%`.
  Compared with graphreserve1024 n84, this reduces io_uring reads by `1829`
  and improves decode `2.00 -> 2.03 tok/s`. New strict current best.

Next diagnostic: n8 only with `graph_reserve=256`. This is closer to the CUDA
graph safety margin, so stop immediately if any OOM/graph/cache instability is
observed.
- 2026-06-13 04:41 CST: Ran strict n8 `sqpoll-graphreserve256-n8`.
  Result: `1.60 eval tok/s`, `1.69 prompt tok/s`, `total_ms=24387.58`,
  `read_failures=0`, `7424` io_uring reads, `41.2%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13242 MiB`;
  - upgate cache `2066` slots;
  - down cache `1153` slots.
  This is stable and reduces another `48` reads vs graphreserve512 n8, but the
  decode gain is within noise. Run n84 once to verify whether the larger
  working set helps long decode.
- 2026-06-13 04:51 CST: Ran strict n84 `sqpoll-graphreserve256-n84`.
  Result: `2.06 eval tok/s`, `1.71 prompt tok/s`, `total_ms=60288.95`,
  `read_failures=0`, `63886` io_uring reads, `2896` direct reads,
  `57.2%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13242 MiB`;
  - graph reserve `256 MiB`, safety `256 MiB`, auto clamp enabled;
  - upgate cache `2066` slots, hit `56.1%`;
  - down cache `1153` slots, hit `59.5%`.
  I/O details:
  - `iouring_inflight_avg=1.68`, `iouring_inflight_max=4`;
  - `iouring_submit_us=81863`, `iouring_wait_us=78985873`;
  - batch histogram remained small (`1:11955`, `2-4:20708`,
    no `5-8` or larger batches).
  Compared with graphreserve512 n84, this reduces runtime io_uring reads by
  another `912`, raises VRAM hit `56.6% -> 57.2%`, and improves decode
  `2.03 -> 2.06 tok/s`. This is the new strict current best.

Next diagnostic: n8 only with `graph_reserve=0`, keeping `safety=256` and
auto-clamp. This may free a little more 5090 VRAM for expert cache, but is
closer to the CUDA graph/runtime headroom limit. Run only after confirming no
other task is occupying the 5090 or host RAM, and stop if there is any
OOM/graph/cache instability.
- 2026-06-13 14:29 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 14:30 CST: Ran strict n8 `sqpoll-graphreserve0-n8`.
  Result: `1.62 eval tok/s`, `1.71 prompt tok/s`, `total_ms=24111.12`,
  `read_failures=0`, `7387` io_uring reads, `2952` direct reads,
  `41.5%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - graph reserve `0 MiB`, safety `256 MiB`, auto clamp enabled;
  - upgate cache `2106` slots;
  - down cache `1175` slots.
  I/O details:
  - `iouring_inflight_avg=1.86`, `iouring_inflight_max=4`;
  - `iouring_submit_us=11373`, `iouring_wait_us=8789900`.
  This is stable on the short run and improves over graphreserve256 n8
  (`1.60 -> 1.62 tok/s`, `7424 -> 7387` io_uring reads). Run one strict n84
  validation to check whether the smaller graph headroom remains stable and
  beneficial for the longer decode.
- 2026-06-13 14:32 CST: Ran strict n84 `sqpoll-graphreserve0-n84`.
  Result: `2.08 eval tok/s`, `1.70 prompt tok/s`, `total_ms=60207.22`,
  `read_failures=0`, `62977` io_uring reads, `2952` direct reads,
  `57.9%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - graph reserve `0 MiB`, safety `256 MiB`, auto clamp enabled;
  - upgate cache `2106` slots, hit `56.7%`;
  - down cache `1175` slots, hit `60.1%`.
  I/O details:
  - `iouring_inflight_avg=1.67`, `iouring_inflight_max=4`;
  - `iouring_submit_us=78807`, `iouring_wait_us=77443593`;
  - batch histogram still has no batches larger than 4 jobs.
  Compared with graphreserve256 n84, this reduces io_uring reads by `909`,
  raises VRAM hit `57.2% -> 57.9%`, and improves decode
  `2.06 -> 2.08 tok/s`. No OOM, CUDA graph, or expert-pack read failure was
  observed. This is the new strict current best for this task.

## Phase 32 Current Best Strict Config

For strict 5090 (`MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`), the best
configuration found so far is:

- `GGML_MOE_IO_SQPOLL=1`;
- `GGML_MOE_VRAM_CACHE_MIB=14336`;
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
- `GGML_MOE_VRAM_CACHE_SAFETY_MIB=256`;
- `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
- keep profile preload from `oracle-n84.route.csv`;
- keep up/gate stage split, down parallel stage, offset sort, GPU handoff, and
  prompt dynamic experts.

Best strict result:

- `sqpoll-graphreserve0-n84`: `2.08 eval tok/s`, `1.70 prompt tok/s`,
  `total_ms=60207.22`, `read_failures=0`, `62977` io_uring reads,
  `57.9%` VRAM hit, actual VRAM cache `13498 MiB`.

This is a measurable improvement over the Phase 31 strict best
(`1.92 tok/s`, `70525` reads, `52.8%` VRAM hit), but it remains far below the
theoretical 5090 target. The remaining bottleneck is still runtime expert
streaming from SSD: even the best strict run performs about `63k` runtime
io_uring reads, and the caller-side split/parallel overlap structure caps
per-ring batch depth at 4 jobs in this path.

## Phase 33: Profile Reserve Sweep With Max Strict VRAM Cache

The best strict Phase 32 config uses actual VRAM cache `13498 MiB` but still
sets `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=10`. The implementation computes
profile preload budget as `n_slots - reserve`, where reserve is this percentage
of each split cache. That means some 5090 cache slots are intentionally left
available for runtime replacement instead of being filled by the oracle
profile.

Hypothesis:

- On this same-prompt/oracle-profile diagnostic, reducing profile reserve can
  place more known-hot experts into VRAM at startup;
- this should reduce runtime io_uring reads and may improve eval token rate;
- reserve too low may hurt if runtime replacement needs space, so validate
  with n8 before n84.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep current best Phase 32 config:
  SQPOLL, `vram_cache_mib=14336`, auto clamp, safety `256`,
  graph reserve `0`, offset sort, up/gate stage split, down parallel stage,
  GPU handoff, prompt dynamic experts;
- run n8 with `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=0`;
- if stable and read count / speed improve, run n84;
- optionally test `profile_reserve_pct=5` if reserve 0 regresses or looks
  unstable.

Acceptance:

- report actual cache, preloads/pinned, VRAM hit, io_uring reads, eval tok/s,
  prompt tok/s, total_ms, and read_failures;
- keep host RAM strict and RAM tier disabled;
- do not claim this as a general benchmark beyond the same-prompt/oracle
  profile diagnostic.
- 2026-06-13 14:40 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 14:42 CST: Ran strict n8 `reserve0-n8`
  (`GGML_MOE_VRAM_PROFILE_RESERVE_PCT=0`) with the Phase 32 best cache
  settings.
  Result: `0.49 eval tok/s`, `1.68 prompt tok/s`, `total_ms=34241.49`,
  `read_failures=0`, `0` io_uring reads, `3281` direct reads,
  `100.0%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2106` slots, all `2106` preloaded/pinned;
  - down cache `1175` slots, all `1175` preloaded/pinned.
  Interpretation:
  - runtime SSD reads disappear, but this is not a win;
  - stderr shows both `batched CUDA MoE up/gate path declined` and
    `batched CUDA MoE down path declined`, falling back to CPU path;
  - output prefix changed from the previous deterministic prefix;
  - overfilling/pinning all cache slots is therefore a bad configuration.

Next diagnostic: test `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=5` on n8. This should
preload more than the current best reserve `10`, but still leave some unpinned
runtime replacement space for CUDA batched MoE to avoid the all-pinned decline.
- 2026-06-13 14:44 CST: Ran strict n8 `reserve5-n8`
  (`GGML_MOE_VRAM_PROFILE_RESERVE_PCT=5`) with the Phase 32 best cache
  settings.
  Result: `1.63 eval tok/s`, `1.68 prompt tok/s`, `total_ms=24384.81`,
  `read_failures=0`, `7256` io_uring reads, `3116` direct reads,
  `42.5%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2106` slots, `2000` preloaded/pinned, hit `41.8%`;
  - down cache `1175` slots, `1116` preloaded/pinned, hit `43.9%`.
  Compared with reserve `10` n8, this preloads `164` more expert rows and
  reduces runtime io_uring reads by `131` (`7387 -> 7256`). Decode is slightly
  higher (`1.62 -> 1.63 tok/s`), but total time is slightly worse
  (`24111.12 -> 24384.81 ms`) and prompt speed is lower. No batched MoE decline
  was observed. Run n84 once because long decode may benefit more from the
  lower read count.
- 2026-06-13 14:46 CST: Ran strict n84 `reserve5-n84`.
  Result: `2.14 eval tok/s`, `1.66 prompt tok/s`, `total_ms=59235.53`,
  `read_failures=0`, `60448` io_uring reads, `3116` direct reads,
  `59.5%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2106` slots, `2000` preloaded/pinned, hit `58.4%`;
  - down cache `1175` slots, `1116` preloaded/pinned, hit `61.8%`.
  I/O details:
  - `iouring_inflight_avg=1.65`, `iouring_inflight_max=4`;
  - `iouring_submit_us=76530`, `iouring_wait_us=74811819`.
  Compared with the Phase 32 best reserve `10` n84, this reduces runtime
  io_uring reads by `2529` (`62977 -> 60448`), raises VRAM hit
  `57.9% -> 59.5%`, and improves decode `2.08 -> 2.14 tok/s`. No batched MoE
  decline was observed. This is the new strict current best.

Next diagnostic: run n8 with `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=3`. This is
more aggressive than reserve `5` but should still leave non-pinned slots
(`~64` upgate and `~36` down) unlike reserve `0`.
- 2026-06-13 14:48 CST: Ran strict n8 `reserve3-n8`.
  Result: `1.63 eval tok/s`, `1.65 prompt tok/s`, `total_ms=24511.08`,
  `read_failures=0`, `7230` io_uring reads, `3181` direct reads,
  `42.7%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2106` slots, `2042` preloaded/pinned, hit `42.0%`;
  - down cache `1175` slots, `1139` preloaded/pinned, hit `44.2%`.
  This reduces only `26` reads versus reserve `5` n8, while prompt speed and
  total time both regress. No batched MoE decline was observed, but the n8
  signal is too weak to justify n84.

## Phase 33 Current Best Strict Config

For strict 5090 (`MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`) the best
configuration after the profile reserve sweep is:

- `GGML_MOE_IO_SQPOLL=1`;
- `GGML_MOE_VRAM_CACHE_MIB=14336`;
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
- `GGML_MOE_VRAM_CACHE_SAFETY_MIB=256`;
- `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=5`;
- keep profile preload from `oracle-n84.route.csv`;
- keep up/gate stage split, down parallel stage, offset sort, GPU handoff, and
  prompt dynamic experts.

Best strict result:

- `reserve5-n84`: `2.14 eval tok/s`, `1.66 prompt tok/s`,
  `total_ms=59235.53`, `read_failures=0`, `60448` io_uring reads,
  `59.5%` VRAM hit, actual VRAM cache `13498 MiB`.

Key lesson: adding more profile-pinned expert rows helps only while enough
unpinned runtime cache space remains. Reserve `0` made runtime misses vanish
but filled both split caches entirely with pinned rows, which caused CUDA MoE
batched paths to decline and regress to `0.49 tok/s` on n8. Reserve `5` is the
best tested balance so far.

## Phase 34: Split Cache Ratio Sweep

The current strict best uses a split VRAM cache with
`GGML_MOE_VRAM_CACHE_UPGATE_PCT=60`. In `reserve5-n84`, upgate still has
`41440` misses while down has `19008` misses. Upgate rows are smaller
(`3.84 MiB`) but appear more frequently; down rows are larger (`4.59 MiB`) and
have a higher hit rate. The fixed 60/40 split may not be optimal for the 5090
strict cache budget.

Hypothesis:

- shifting more cache budget to upgate may reduce more total runtime reads;
- shifting too much away from down may increase larger down reads and hurt
  speed;
- an n8 sweep can identify promising split ratios before spending n84 runs.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep Phase 33 best config:
  SQPOLL, `vram_cache_mib=14336`, auto clamp, safety `256`,
  graph reserve `0`, profile reserve `5`, offset sort, up/gate stage split,
  down parallel stage, GPU handoff, prompt dynamic experts;
- run n8 with `vram_upgate_pct=65`;
- if promising, run n84;
- if not, test `vram_upgate_pct=55` to check whether down capacity is more
  valuable.

Acceptance:

- report actual split slots, preloads/pinned, per-cache hit/miss, total VRAM
  hit, io_uring reads, eval tok/s, prompt tok/s, total_ms, and read_failures;
- keep strict host RAM and RAM tier disabled;
- stop any direction that causes batched MoE decline, output instability, or
  read failures.
- 2026-06-13 15:01 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 15:03 CST: Ran strict n8 `upgate65-n8`
  (`GGML_MOE_VRAM_CACHE_UPGATE_PCT=65`) with Phase 33 best settings.
  Result: `1.64 eval tok/s`, `1.67 prompt tok/s`, `total_ms=24478.06`,
  `read_failures=0`, `7227` io_uring reads, `3143` direct reads,
  `42.8%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2282` slots, `2167` preloaded/pinned, hit `43.5%`;
  - down cache `1028` slots, `976` preloaded/pinned, hit `41.3%`.
  Compared with the Phase 33 best n8 split (`60/40`), this reduces runtime
  io_uring reads by `29` (`7256 -> 7227`) and raises decode
  `1.63 -> 1.64 tok/s`, but total time remains slightly worse than the best
  short run. No batched MoE decline was observed. Run n84 to check whether the
  larger upgate cache helps over long decode.
- 2026-06-13 15:05 CST: Ran strict n84 `upgate65-n84`.
  Result: `2.14 eval tok/s`, `1.67 prompt tok/s`, `total_ms=58854.13`,
  `read_failures=0`, `60038` io_uring reads, `3143` direct reads,
  `59.8%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2282` slots, `2167` preloaded/pinned, hit `60.9%`;
  - down cache `1028` slots, `976` preloaded/pinned, hit `57.6%`.
  I/O details:
  - `iouring_inflight_avg=1.65`, `iouring_inflight_max=4`;
  - `iouring_submit_us=73011`, `iouring_wait_us=70960454`.
  Compared with the Phase 33 best n84 split (`60/40`), this lowers total
  io_uring reads by `410` (`60448 -> 60038`) and total time by `381 ms`, while
  decode remains rounded to `2.14 tok/s`. Per-cache misses shift as expected:
  upgate misses drop (`41440 -> 38935`) but down misses rise
  (`19008 -> 21103`). This is a small but real strict-line improvement and is
  the current best tested split.

Next diagnostic: run n8 with `GGML_MOE_VRAM_CACHE_UPGATE_PCT=55` to confirm
whether giving more slots to down is worse or whether the larger row size makes
down capacity more valuable than the miss count suggests.
- 2026-06-13 15:07 CST: Ran strict n8 `upgate55-n8`.
  Result: `1.64 eval tok/s`, `1.67 prompt tok/s`, `total_ms=24473.46`,
  `read_failures=0`, `7271` io_uring reads, `3089` direct reads,
  `42.4%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `1931` slots, `1834` preloaded/pinned, hit `40.4%`;
  - down cache `1322` slots, `1255` preloaded/pinned, hit `46.5%`.
  This improves down hits but increases total runtime reads versus both
  `upgate65-n8` and the Phase 33 `60/40` split. Do not run n84 for this split.

## Phase 34 Current Best Strict Config

For strict 5090 (`MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`) the best
configuration after the split-cache sweep is:

- `GGML_MOE_IO_SQPOLL=1`;
- `GGML_MOE_VRAM_CACHE_MIB=14336`;
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
- `GGML_MOE_VRAM_CACHE_SAFETY_MIB=256`;
- `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=5`;
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=65`;
- keep profile preload from `oracle-n84.route.csv`;
- keep up/gate stage split, down parallel stage, offset sort, GPU handoff, and
  prompt dynamic experts.

Best strict result:

- `upgate65-n84`: `2.14 eval tok/s`, `1.67 prompt tok/s`,
  `total_ms=58854.13`, `read_failures=0`, `60038` io_uring reads,
  `59.8%` VRAM hit, actual VRAM cache `13498 MiB`.

The improvement over Phase 33 is small but aligned with the goal:
`60448 -> 60038` runtime io_uring reads and `59235.53 -> 58854.13 ms` total
time. The split sweep suggests this workload benefits from slightly more
upgate cache, but not enough to resolve the remaining gap to the theoretical
5090 target. Runtime SSD reads remain about `60k`.

## Phase 35: Upgate Split Refinement

Phase 34 showed that shifting the split cache from `60/40` to `65/35` slightly
reduced total runtime SSD reads. The per-cache miss shift suggests upgate still
benefits from more cache, but down misses rise as down cache shrinks. Test one
more point before changing implementation direction.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep Phase 34 best config except set `GGML_MOE_VRAM_CACHE_UPGATE_PCT=70`;
- run n8 first;
- run n84 only if n8 reduces reads or improves speed without batched MoE
  decline.

Acceptance:

- report split slots, preloads/pinned, per-cache misses, total reads, VRAM hit,
  eval tok/s, prompt tok/s, total_ms, and read_failures;
- keep RAM tier disabled and strict host RAM;
- stop if output prefix changes, read failures occur, or batched CUDA MoE
  declines.
- 2026-06-13 15:13 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 15:14 CST: Ran strict n8 `upgate70-n8`.
  Result: `1.63 eval tok/s`, `1.68 prompt tok/s`, `total_ms=24585.62`,
  `read_failures=0`, `7214` io_uring reads, `3171` direct reads,
  `42.9%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2458` slots, `2335` preloaded/pinned, hit `45.1%`;
  - down cache `881` slots, `836` preloaded/pinned, hit `38.4%`.
  This reduces only `13` reads versus `upgate65-n8`, but decode and total time
  regress. The down cache is now too small (`2593` down misses on n8). No
  batched MoE decline was observed, but the n8 signal is too weak and too
  noisy to justify n84. Keep `upgate65` as the best tested split.

## Phase 35 Current Best Strict Config

No change from Phase 34. The best strict result remains:

- `upgate65-n84`: `2.14 eval tok/s`, `1.67 prompt tok/s`,
  `total_ms=58854.13`, `read_failures=0`, `60038` io_uring reads,
  `59.8%` VRAM hit, actual VRAM cache `13498 MiB`.

Further split tuning is unlikely to close the gap: `55`, `60`, `65`, and `70`
show that total reads are fairly flat near the optimum, while shrinking down
too far increases larger down misses. The next useful work should target the
remaining ~`60k` runtime SSD reads through better scheduling/prefetching or a
more accurate preload profile, not just more split-ratio tuning.

## Phase 36: Strict Cache Safety Margin Sweep

The current best strict configuration still leaves
`GGML_MOE_VRAM_CACHE_SAFETY_MIB=256` with `graph_reserve=0`. Earlier graph
reserve sweeps showed that reducing reserve from `2048` to `0` steadily
increased actual expert cache and reduced reads without OOM. The remaining
safety margin may be conservative on this 5090 for the fixed same-prompt
diagnostic.

Hypothesis:

- reducing safety from `256 MiB` to `128 MiB` may add a small number of expert
  cache slots and reduce runtime SSD reads;
- lowering safety too far can risk CUDA allocation/graph instability, so use
  n8 first and stop on any error.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep Phase 34 best config:
  SQPOLL, `vram_cache_mib=14336`, auto clamp, graph reserve `0`, profile
  reserve `5`, upgate pct `65`, offset sort, up/gate stage split, down
  parallel stage, GPU handoff, prompt dynamic experts;
- run n8 with `GGML_MOE_VRAM_CACHE_SAFETY_MIB=128`;
- if stable and better, run n84;
- do not try `safety=0` until `128` is proven stable and useful.

Acceptance:

- report actual cache, split slots, preloads/pinned, per-cache hit/miss,
  io_uring reads, eval tok/s, prompt tok/s, total_ms, and read_failures;
- stop immediately on OOM, allocation errors, graph errors, output prefix
  instability, or batched CUDA MoE decline.
- 2026-06-13 15:22 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 15:23 CST: Ran strict n8 `safety128-n8`.
  Result: failed with return code `1` before timings were emitted.
  Cache details before failure:
  - requested `14336 MiB`, actual `13626 MiB`;
  - upgate cache `2304` slots;
  - down cache `1038` slots.
  Failure details:
  - stderr reports `CUDA error: out of memory`;
  - failure occurs at `evaluate_and_capture_cuda_graph` during
    `cudaGraphLaunch(graph->instance, cuda_ctx->stream())`;
  - stdout contains the debugger/ptrace attach failure text after the CUDA
    crash, not a model output.
  Conclusion: `GGML_MOE_VRAM_CACHE_SAFETY_MIB=128` is too aggressive on this
  strict 5090 setup with graph reuse enabled. Do not test `safety=0`, and keep
  `safety=256` as the minimum stable margin found so far.

## Phase 36 Current Decision

No change from Phase 34/35. The best strict result remains:

- `upgate65-n84`: `2.14 eval tok/s`, `1.67 prompt tok/s`,
  `total_ms=58854.13`, `read_failures=0`, `60038` io_uring reads,
  `59.8%` VRAM hit, actual VRAM cache `13498 MiB`.

The CUDA OOM at `safety=128` shows that the previous `safety=256` plus
`graph_reserve=0` setting is already near the practical VRAM limit for this
graph-reuse run. Further gains should not come from reducing safety margin.

## Phase 37: Combined Split And Profile Reserve Check

The best split is `upgate_pct=65`, while the profile reserve sweep that found
`reserve=5` was done before changing the split from `60/40`. Since the split
changes how many slots each cache has, `reserve=3` may be viable with
`upgate65` even though it was not worth n84 at `60/40`.

Hypothesis:

- `upgate65 + profile_reserve=3` may preload a few more hot upgate rows and
  lower reads without filling the down cache too aggressively;
- if pinned rows get too close to total slots, CUDA MoE can still decline or
  replacement space can become too small.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep Phase 34 best config except set
  `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=3`;
- run n8 first;
- run n84 only if n8 improves reads and does not regress decode/total time or
  trigger batched MoE decline.

Acceptance:

- report split slots, preloads/pinned, per-cache hit/miss, total reads, VRAM
  hit, eval tok/s, prompt tok/s, total_ms, and read_failures;
- keep safety at `256` and graph reserve at `0`;
- stop if output prefix changes, read failures occur, CUDA errors occur, or
  batched CUDA MoE declines.
- 2026-06-13 15:28 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 15:29 CST: Ran strict n8 `upgate65-reserve3-n8`.
  Result: `1.64 eval tok/s`, `1.67 prompt tok/s`, `total_ms=24651.29`,
  `read_failures=0`, `7171` io_uring reads, `3210` direct reads,
  `43.2%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2282` slots, `2213` preloaded/pinned, hit `43.9%`;
  - down cache `1028` slots, `997` preloaded/pinned, hit `41.8%`.
  Compared with `upgate65-reserve5-n8`, this reduces reads
  (`7227 -> 7171`) and raises VRAM hit (`42.8% -> 43.2%`), but total time
  regresses (`24478.06 -> 24651.29 ms`). No batched MoE decline was observed.
  Run n84 once, because previous reserve tuning showed the longer decode can
  benefit from lower read count even when n8 total time is noisy.
- 2026-06-13 15:31 CST: Ran strict n84 `upgate65-reserve3-n84`.
  Result: `2.17 eval tok/s`, `1.67 prompt tok/s`, `total_ms=58432.63`,
  `read_failures=0`, `59049` io_uring reads, `3210` direct reads,
  `60.5%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2282` slots, `2213` preloaded/pinned, hit `61.6%`;
  - down cache `1028` slots, `997` preloaded/pinned, hit `58.3%`.
  I/O details:
  - `iouring_inflight_avg=1.64`, `iouring_inflight_max=4`;
  - `iouring_submit_us=74681`, `iouring_wait_us=70161916`.
  Compared with the previous strict best `upgate65-reserve5-n84`, this reduces
  runtime io_uring reads by `989` (`60038 -> 59049`), raises VRAM hit
  `59.8% -> 60.5%`, lowers total time `58854.13 -> 58432.63 ms`, and improves
  decode `2.14 -> 2.17 tok/s`. No batched MoE decline or read failure was
  observed. This is the new strict current best.

## Phase 37 Current Best Strict Config

For strict 5090 (`MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`) the best
configuration after the split/reserve combination sweep is:

- `GGML_MOE_IO_SQPOLL=1`;
- `GGML_MOE_VRAM_CACHE_MIB=14336`;
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
- `GGML_MOE_VRAM_CACHE_SAFETY_MIB=256`;
- `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=3`;
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=65`;
- keep profile preload from `oracle-n84.route.csv`;
- keep up/gate stage split, down parallel stage, offset sort, GPU handoff, and
  prompt dynamic experts.

Best strict result:

- `upgate65-reserve3-n84`: `2.17 eval tok/s`, `1.67 prompt tok/s`,
  `total_ms=58432.63`, `read_failures=0`, `59049` io_uring reads,
  `60.5%` VRAM hit, actual VRAM cache `13498 MiB`.

This is another incremental gain, but the remaining gap is still dominated by
runtime SSD reads and small io_uring batches: the current best still performs
about `59k` runtime expert-pack reads, with `iouring_inflight_max=4`.

## Phase 38: Profile Reserve 2 Probe

Phase 37 showed `upgate65 + profile_reserve=3` is a valid improvement over
reserve `5`. There are still some unpinned slots left:

- upgate: `2282 - 2213 = 69` slots;
- down: `1028 - 997 = 31` slots.

Trying reserve `2` should pin only a small number of additional profile rows,
while still leaving some replacement space. Reserve `0` is known bad because it
fills all cache slots and causes CUDA MoE batched paths to decline, so do not
approach it without an n8 guard.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep Phase 37 best config except set
  `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=2`;
- run n8 first;
- run n84 only if n8 reduces reads without output instability, CUDA errors, or
  batched CUDA MoE decline.

Acceptance:

- report split slots, preloads/pinned, per-cache hit/miss, total reads, VRAM
  hit, eval tok/s, prompt tok/s, total_ms, and read_failures;
- keep safety at `256` and graph reserve at `0`;
- stop if any batched CUDA MoE decline appears, because reserve `0` already
  proved that over-pinning can force CPU fallback.
- 2026-06-13 15:38 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 15:39 CST: Ran strict n8 `upgate65-reserve2-n8`.
  Result: `1.64 eval tok/s`, `1.65 prompt tok/s`, `total_ms=24750.68`,
  `read_failures=0`, `7169` io_uring reads, `3243` direct reads,
  `43.2%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13498 MiB`;
  - upgate cache `2282` slots, `2236` preloaded/pinned, hit `43.9%`;
  - down cache `1028` slots, `1007` preloaded/pinned, hit `41.9%`.
  Compared with `upgate65-reserve3-n8`, this only reduces `2` io_uring reads
  (`7171 -> 7169`) and does not improve VRAM hit meaningfully, while prompt
  speed and total time regress. No batched MoE decline was observed, but the n8
  signal is too weak to justify n84. Keep reserve `3`.

## Phase 38 Current Decision

No change from Phase 37. The best strict result remains:

- `upgate65-reserve3-n84`: `2.17 eval tok/s`, `1.67 prompt tok/s`,
  `total_ms=58432.63`, `read_failures=0`, `59049` io_uring reads,
  `60.5%` VRAM hit, actual VRAM cache `13498 MiB`.

Reserve `2` is likely too close to the over-pinned regime: it pins more rows
but only removes noise-level runtime reads on n8 and slows prompt/total time.
Reserve `3` remains the best tested profile reserve for `upgate_pct=65`.

## Phase 39: No-Graphs Cache Headroom Probe

Phase 36 showed that `cache_safety_mib=128` OOMs during CUDA graph launch, even
though it would add cache slots. This suggests CUDA graph memory may be the
limiting headroom. It is worth testing whether disabling CUDA graphs lets the
5090 use the larger expert cache and whether the reduced SSD reads compensate
for losing graph reuse.

Hypothesis:

- `GGML_CUDA_DISABLE_GRAPHS=1` may avoid the graph-launch OOM at
  `cache_safety_mib=128`;
- if the larger cache reduces enough runtime SSD reads, no-graphs could
  improve decode despite higher per-token launch overhead;
- if n8 is slower or still OOMs, keep graph reuse enabled.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`,
  `GGML_MOE_RAM_TIER_MIB=0`;
- keep Phase 37 best config, but set `--cuda-disable-graphs` and
  `--cache-safety-mib 128`;
- run n8 first;
- run n84 only if n8 is stable and shows a credible speed/read improvement.

Acceptance:

- report actual cache, split slots, preloads/pinned, per-cache hit/miss,
  io_uring reads, eval tok/s, prompt tok/s, total_ms, and read_failures;
- stop on CUDA errors, output prefix instability, or batched CUDA MoE decline;
- do not adopt no-graphs unless speed improves, not merely because reads fall.
- 2026-06-13 16:58 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 17:02 CST: Ran strict n8 `nograph-safety128-n8`.
  Result: `1.64 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24052.45`,
  `read_failures=0`, `7169` io_uring reads, `3249` direct reads,
  `43.2%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13664 MiB`;
  - upgate cache `2310` slots, `2240` preloaded/pinned, hit `43.9%`;
  - down cache `1041` slots, `1009` preloaded/pinned, hit `41.9%`.
  This run was stable, showed no batched CUDA MoE decline, and avoided the
  Phase 36 graph-launch OOM. Compared with the current best n8 guard
  `upgate65-reserve3-n8`, it reduces two reads (`7171 -> 7169`) and improves
  prompt/total time (`24651.29 -> 24052.45 ms`). Run n84 once to see whether
  the extra cache headroom offsets disabling graphs on the longer decode.
- 2026-06-13 17:05 CST: Ran strict n84 `nograph-safety128-n84`.
  Result: `2.18 eval tok/s`, `1.74 prompt tok/s`, `total_ms=57795.16`,
  `read_failures=0`, `58489` io_uring reads, `3249` direct reads,
  `60.9%` VRAM hit.
  Cache details:
  - requested `14336 MiB`, actual `13664 MiB`;
  - upgate cache `2310` slots, `2240` preloaded/pinned, hit `61.9%`;
  - down cache `1041` slots, `1009` preloaded/pinned, hit `58.7%`.
  I/O details:
  - `iouring_inflight_avg=1.64`, `iouring_inflight_max=4`;
  - `iouring_submit_us=71261`, `iouring_wait_us=69854828`;
  - pinned staging fallbacks and expert-pack io_uring fallbacks were all `0`.
  `GGML_CUDA_DISABLE_GRAPHS=1` was present in the recorded environment and
  stderr ended with `have 0 graphs`, so this is a true no-CUDA-graphs run. The
  runner's `summary.json` field `no_graph_reuse=false` is the separate
  `--no-graph-reuse` argument and should not be used to infer CUDA graph state.
  Compared with the previous strict best `upgate65-reserve3-n84`, this reduces
  runtime io_uring reads by `560` (`59049 -> 58489`), raises VRAM hit
  `60.5% -> 60.9%`, lowers total time `58432.63 -> 57795.16 ms`, raises prompt
  `1.67 -> 1.74 tok/s`, and slightly raises decode `2.17 -> 2.18 tok/s`. This
  is the new strict current best, but the gain is small; the remaining bottleneck
  is still about `58.5k` runtime expert-pack reads with small io_uring batches.

## Phase 39 Current Best Strict Config

For strict 5090 (`MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`) the best
configuration after the no-graphs cache headroom probe is:

- `GGML_CUDA_DISABLE_GRAPHS=1`;
- `GGML_MOE_IO_SQPOLL=1`;
- `GGML_MOE_VRAM_CACHE_MIB=14336`;
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
- `GGML_MOE_VRAM_CACHE_SAFETY_MIB=128`;
- `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=3`;
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=65`;
- keep profile preload from `oracle-n84.route.csv`;
- keep up/gate stage split, down parallel stage, offset sort, GPU handoff, and
  prompt dynamic experts.

Best strict result:

- `nograph-safety128-n84`: `2.18 eval tok/s`, `1.74 prompt tok/s`,
  `total_ms=57795.16`, `read_failures=0`, `58489` io_uring reads,
  `60.9%` VRAM hit, actual VRAM cache `13664 MiB`.

This supersedes Phase 37 by a narrow margin. The core conclusion has not
changed: larger VRAM cache helps, but even with CUDA graphs disabled and
`cache_safety_mib=128`, the strict 2 GB RAM line still leaves too many cold
expert rows to stream synchronously from SSD.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the Phase 39 runner/helper scripts;
- `git diff --check` passed.

Next direction:

- further gains likely require reducing the number of runtime expert-pack reads
  more substantially, or making the remaining reads overlap better than the
  current `iouring_inflight_avg=1.64`, rather than continuing to tune small cache
  reserve margins.

## Phase 40: Stage Profiling Before Cross-Split I/O Aggregation

Phase 31 already proved that disabling up/gate stage split raises io_uring
batch size/inflight but regresses speed because it loses CPU staging / CUDA
stream overlap. Therefore, do not pursue a no-split configuration. A useful
implementation would need to preserve the current split-stage overlap while
reducing per-vector I/O waits or aggregating reads underneath it.

Before changing the scheduler, measure where the current Phase 39 best spends
time with the existing built-in batch profiling:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 39 best config:
  `GGML_CUDA_DISABLE_GRAPHS=1`, `cache_safety_mib=128`,
  `upgate_pct=65`, `profile_reserve_pct=3`, SQPOLL, offset sort,
  split up/gate staging, down parallel staging, GPU handoff, prompt dynamic
  experts;
- enable `GGML_MOE_BATCH_PROFILE=1` through `--route-profile-out` to also turn
  on `g_bprof/g_uprof`;
- run n8 only first, because profiling changes CUDA event creation and is a
  diagnostic, not a comparable performance line;
- use the result to decide whether the next implementation should target
  cross-split io_uring aggregation, host-read-ahead, or cache/profile changes.

Acceptance:

- report `batch upgate profile`, `batch down profile`, pinned-stage timing
  (`slot_wait`, `host_stage`, `enqueue`, `h2d`) if present, and normal I/O
  metrics;
- output must remain stable, `read_failures=0`, and no CUDA MoE decline/fallback
  should appear;
- do not adopt a profiling result as a new best, even if token rate changes.
- 2026-06-13 17:28 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 17:29 CST: Ran strict profiling n8
  `nograph-safety128-profile-n8`.
  Result: `1.62 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24091.23`,
  `read_failures=0`, `7169` io_uring reads, `3249` direct reads,
  `43.2%` VRAM hit. This is diagnostic only and is not a new best.
  Cache details:
  - requested `14336 MiB`, actual `13664 MiB`;
  - upgate cache `2310` slots, `2240` preloaded/pinned, hit `43.9%`;
  - down cache `1041` slots, `1009` preloaded/pinned, hit `41.9%`.
  I/O details:
  - total `iouring_wait_us=8414445`, `iouring_submit_us=17274`;
  - `iouring_inflight_avg=1.83`, `iouring_inflight_max=4`;
  - batch histogram remains limited to `1` and `2-4`.
  Pinned-stage timing:
  - main/down ring: `copies=5917`, `host_stage=2012.497 ms`,
    `h2d=805.242 ms`, `enqueue=63.932 ms`, `slot_wait=7.418 ms`;
  - gate ring: `copies=2407`, `h2d=383.207 ms`;
  - up_aux: `copies=1047`, `h2d=149.465 ms`;
  - gate_aux: `copies=1047`, `h2d=152.450 ms`;
  - all staging fallbacks were `0`.
  Batch profiles:
  - up/gate: `calls=526`, `avg_active=8`, `up_wait=3.857 ms/call`,
    `gate_wait=4.118 ms/call`, `up_compute=0.128 ms/call`,
    `gate_compute=0.136 ms/call`, `kernel=4.275 ms/call`;
  - down: `calls=526`, `avg_active=8`, `stage=2.552 ms/call`,
    `kernel=0.329 ms/call`, `total=2.914 ms/call`.
  Interpretation: compute kernels are not the near-term limiter. The useful
  target is the split-stage I/O wait: preserve current split-stage CUDA overlap,
  but make the underlying pack reads less fragmented or more overlapped. This
  supports a default-off cross-split read aggregation prototype before further
  cache micro-tuning.

## Phase 41: Recheck No-Split Tradeoff Under Phase 39 Best

Before implementing a cross-split aggregation prototype, recheck the old
Phase 31 no-split diagnostic under the current Phase 39 best:

- Phase 31 used the older cache shape and graph setting;
- the current best disables CUDA graphs and uses a slightly larger cache
  (`cache_safety_mib=128`);
- if no up/gate stage split is still slower, the implementation must preserve
  split-stage overlap and only aggregate/advance the underlying reads;
- if no-split improves under Phase 39, it is a cheaper configuration win and
  may also simplify the next implementation.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 39 best, but omit `--up-gate-stage-split`;
- run n8 only;
- do not run n84 unless n8 clearly beats `nograph-safety128-n8` on total time
  and eval speed while keeping stable output and `read_failures=0`.

Acceptance:

- compare token rate, total time, io_uring read count, wait time, inflight
  average/max, batch histogram, and VRAM hit;
- if no-split loses, proceed to a default-off implementation that keeps the
  split-stage streams.
- 2026-06-13 17:35 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 17:36 CST: Ran strict n8
  `nograph-safety128-no-upgate-split-n8`.
  Result: `1.63 eval tok/s`, `1.77 prompt tok/s`, `total_ms=23882.66`,
  `read_failures=0`, `7169` io_uring reads, `3249` direct reads,
  `43.2%` VRAM hit.
  I/O details:
  - `iouring_wait_us=5574581`, down from `8386616` in the Phase 39 split n8;
  - `iouring_inflight_avg=2.66`, `iouring_inflight_max=8`;
  - batch histogram now includes `5-8:538` batches;
  - batches fell from `3034` to `2038`.
  This confirms the no-split path still produces much healthier io_uring
  batches under the Phase 39 no-graphs cache shape. The performance signal is
  mixed: eval is slightly below the split n8 (`1.64 -> 1.63`), but prompt and
  total time improve (`24052.45 -> 23882.66 ms`). Because n84 is dominated more
  by repeated runtime I/O and the I/O wait reduction is large, run one n84
  validation before deciding whether no-split is now preferable.
- 2026-06-13 17:39 CST: Ran strict n84
  `nograph-safety128-no-upgate-split-n84`.
  Result: `1.92 eval tok/s`, `1.73 prompt tok/s`, `total_ms=63094.02`,
  `read_failures=0`, `58489` io_uring reads, `3249` direct reads,
  `60.9%` VRAM hit.
  I/O details:
  - `iouring_wait_us=54299949`, down from `69854828` in the Phase 39 split n84;
  - `iouring_inflight_avg=2.31`, `iouring_inflight_max=8`;
  - batch histogram includes `5-8:3074` batches.
  Despite much healthier I/O batching and lower measured io_uring wait, decode
  regresses badly against Phase 39 split n84 (`2.18 -> 1.92 tok/s`) and total
  time worsens (`57795.16 -> 63094.02 ms`). No read failure or staging fallback
  occurred, so this is a real overlap tradeoff rather than an error.

Phase 41 decision:

- do not adopt no up/gate stage split;
- keep the Phase 39 split-stage config as the current strict best;
- the next implementation must preserve split-stage CUDA overlap and target
  earlier pack reads or cross-ring prefetch, not simply merge up/gate staging
  into larger synchronous io_uring batches.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the Phase 40/41 runner/helper scripts;
- `git diff --check` passed.

Next implementation direction:

- build a default-off read-ahead mechanism that keeps the current split-stage
  streams but issues pack reads earlier from route-trace knowledge or from the
  already planned stage jobs;
- avoid replacing split staging with larger synchronous batches, because Phase
  41 showed better io_uring metrics can still lose substantial decode speed
  when overlap is reduced.

## Phase 42: Default-Off Planned Host Read-Ahead Prototype

Phase 41 rejected simply disabling up/gate split. The next prototype should
preserve the existing split-stage stream overlap but move some pack reads
earlier. The least invasive implementation is to reuse the existing
`host_prefetch` buffer/cache machinery, but feed it from the already planned
current-call stage jobs instead of from a long blind route trace:

- after `plan_tensor()` has produced `up_jobs` and `gate_jobs`, the runtime
  knows the exact miss set for this call;
- a default-off env gate can submit those jobs to the host-prefetch worker
  before splitting them across the four copy threads;
- the later split copy threads can hit `host_prefetch_copy_h2d()` and only do
  H2D from pinned host memory;
- if a prefetch is not ready, the current code falls back to normal io_uring
  path, so correctness should be preserved.

Plan:

- add env gate `GGML_MOE_PLANNED_HOST_PREFETCH=1`;
- reuse `GGML_MOE_HOST_PREFETCH_SLOTS` and
  `GGML_MOE_HOST_PREFETCH_MAX_MIB` for buffer sizing, but do not require a
  trace file;
- add counters/reporting for planned submissions, hits, misses/reserved, and
  failures;
- integrate only in up/gate parallel-stage path first, because Phase 40 showed
  up/gate waits dominate;
- keep default/off behavior unchanged;
- run strict n8 guard under Phase 39 best config with
  `--up-gate-stage-split` preserved;
- run n84 only if n8 is stable and improves either total time or eval speed
  without read failures, CUDA errors, or output drift.

Acceptance:

- default build succeeds and default/off behavior still builds;
- planned-prefetch run reports planned counters;
- output prefix remains stable and `read_failures=0`;
- report token rate, total time, host prefetch hit/miss/reserved counts,
  io_uring reads/wait/inflight, VRAM hit, and fallback counts.
- 2026-06-13 18:04 CST: Implemented the default-off prototype:
  - `GGML_MOE_PLANNED_HOST_PREFETCH=1` enables planned host read-ahead;
  - it reuses the existing host-prefetch worker, pinned host buffers, and
    `host_prefetch_copy_h2d()` path;
  - it can run without a route trace file;
  - after up/gate `plan_tensor()` produces current-call miss jobs, those
    pack-backed jobs are queued to the host-prefetch worker before the existing
    split-stage copy threads run;
  - runner flag `--planned-host-prefetch` sets the env and records parsed
    host-prefetch counters.
  The implementation is intentionally default-off. It does not change selected
  experts or cache slot allocation. If a planned host read is not ready, the
  existing split copy path falls back to normal io_uring.
- 2026-06-13 18:06 CST: Build/format checks after implementation:
  - `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
    passed;
  - `python3 -m py_compile run_5090_matrix.py` passed;
  - `git diff --check` passed.
- 2026-06-13 18:08 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 18:09 CST: Ran strict n8 `planned-host-prefetch-n8`.
  Result: `1.51 eval tok/s`, `1.75 prompt tok/s`, `total_ms=24359.58`,
  `read_failures=0`, `7123` io_uring reads, `3249` direct reads,
  `43.2%` VRAM hit.
  Host-prefetch counters:
  - `planned_enqueued=4129`, `planned_dequeued=3771`, `submitted=3771`;
  - only `hits=46`, with `misses=7123` and `reserved_skips=3`;
  - `evicted=3661`, `read_failures=0`, `alloc_failures=0`, `no_slot=0`;
  - `used=246 MiB`, `slots=64`.
  I/O details:
  - `iouring_reads` fell by only `46` versus Phase 39 n8 (`7169 -> 7123`);
  - `iouring_wait_us` worsened (`8386616 -> 9055744`);
  - `iouring_inflight_avg=1.82`, `iouring_inflight_max=4`.
  Decision: do not run n84 and do not adopt this prototype. It is correct and
  stable, but current-call planned prefetch is too late: most split copy
  threads reach `host_prefetch_copy_h2d()` before the worker has finished the
  corresponding host read. The worker also churns slots heavily (`3661`
  evictions) for only `46` hits, so the thread/buffer overhead dominates.

Phase 42 decision:

- keep the code default-off for now as an instrumentation/prototype path;
- keep the Phase 39 split-stage config as current strict best;
- the next useful implementation must prefetch earlier than the current call's
  split-copy point, for example by queueing next-call misses from route history
  or using a more accurate trace-driven producer with stronger filtering.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed after the n8 guard;
- `python3 -m py_compile` passed for the Phase 42 runner/helper scripts;
- `git diff --check` passed;
- no llama/matrix experiment process remained after the run.

## Phase 43: Strict SSD-Miss Trace Host-Prefetch Recheck

Phase 42 showed current-call planned host-prefetch is too late. Historical
Phases 23/24 showed SSD-miss filtered trace prefetch is mechanically effective
but was tested mostly on non-strict full RAM-tier lines and with large 2 GiB
host-prefetch buffers that created producer/slot pressure.

For the current strict Phase 39 shape, generate the SSD-miss trace using the
same cache slots and profile reserve:

- upgate slots: `2310`;
- down slots: `1041`;
- reserve: `3%`;
- RAM tier disabled;
- `profile_after=12624`.

Offline analysis:

- generated
  `phase43-read-ahead-analysis/phase39-predicted-ssd.trace.csv`;
- predicted SSD events: `58460`, close to observed Phase 39 n84
  `58489` io_uring reads;
- prefix SSD events: `7169`, matching n8 exactly;
- window simulation over this filtered SSD-miss sequence shows that in theory
  a near-future miss window is compact:
  - window `16`: `73.59 MiB`, `99.97%` hit upper bound;
  - window `64`: `273.09 MiB`, `99.89%` hit upper bound;
  - window `128`: `539.34 MiB`, `99.78%` hit upper bound.

Plan:

- no new code first;
- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 39 best config with split-stage overlap;
- use `GGML_MOE_HOST_PREFETCH=<phase39-predicted-ssd.trace.csv>` rather than
  raw route trace or current-call planned queue;
- do not enable `GGML_MOE_PLANNED_HOST_PREFETCH`;
- start with a small strict buffer: `slots=64`, `max=512 MiB`,
  `lead_events=64`;
- run n8 guard only;
- run n84 only if n8 lowers reads materially without decode/total-time
  regression.

Acceptance:

- report host-prefetch hits/misses/evictions/no_slot, io_uring reads/wait,
  token rates, total time, VRAM hit, and read failures;
- output prefix must remain stable;
- if n8 is slower, do not run n84 and keep Phase 39 as best.
- 2026-06-13 18:20 CST: Generated strict Phase 39 SSD-miss trace:
  `phase43-read-ahead-analysis/phase39-predicted-ssd.trace.csv`.
  Prediction:
  - `58460` SSD events, close to Phase 39 n84's observed `58489` io_uring
    reads;
  - prefix SSD events `7169`, matching Phase 39 n8;
  - simulated read-ahead windows show that if the consumer could prefetch the
    true future SSD-miss sequence, window `16-64` would fit in roughly
    `74-273 MiB` and cover almost all misses.
- 2026-06-13 18:23 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 18:24 CST: Ran strict n8
  `ssdmiss-lead64-s64-m512-n8`.
  Result: `1.61 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24184.23`,
  `read_failures=0`, `6376` io_uring reads, `3249` direct reads,
  `43.2%` VRAM hit.
  Host-prefetch stats:
  - `matched=7169`, `submitted=3696`, `hits=793`, `misses=6376`;
  - `evicted=2839`, `reserved_skips=959`, `no_slot=0`;
  - `used=284.25 MiB`, `slots=64`;
  - `read_failures=0`, `alloc_failures=0`.
  I/O details: `iouring_wait_us=8109632`,
  `iouring_inflight_avg=1.72`, `iouring_inflight_max=4`.
  Compared with Phase 39 n8, reads fall materially (`7169 -> 6376`) and wait
  is slightly lower (`8386616 -> 8109632`), but eval/total regress
  (`1.64 -> 1.61`, `24052.45 -> 24184.23`). Do not run n84 yet; try a smaller
  lead window to reduce producer/eviction overhead.
- 2026-06-13 18:27 CST: Ran strict n8
  `ssdmiss-lead16-s64-m512-n8`.
  Result: `1.60 eval tok/s`, `1.75 prompt tok/s`, `total_ms=24079.95`,
  `read_failures=0`, `6484` io_uring reads, `3249` direct reads,
  `43.2%` VRAM hit.
  Host-prefetch stats:
  - `matched=7169`, `submitted=3761`, `hits=685`, `misses=6484`;
  - `evicted=3012`, `reserved_skips=938`, `no_slot=0`;
  - `used=284.25 MiB`, `slots=64`;
  - `read_failures=0`, `alloc_failures=0`.
  I/O details: `iouring_wait_us=8257539`,
  `iouring_inflight_avg=1.73`, `iouring_inflight_max=4`.
  This nearly matches total time but still loses eval and saves fewer reads than
  lead `64`. Do not run n84.

Phase 43 decision:

- filtered SSD-miss trace is much better than raw trace/current-call queue and
  does reduce strict n8 SSD reads by `685-793`;
- however, the host-prefetch consume path still does not improve n8 token rate;
- keep Phase 39 as current strict best;
- the next credible path should avoid the extra host-prefetch producer/consume
  overhead, for example by integrating future SSD-miss lookahead into the
  existing split-stage io_uring rings rather than copying through a separate
  host-prefetch queue.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the runner and Phase 43 analysis helpers;
- `git diff --check` passed;
- no llama/matrix experiment process remained after the runs.

## Phase 44: Filtered SSD-Miss Trace Prefetch Recheck

Phase 43 showed that filtered SSD-miss trace predicts real SSD reads well, but
the separate host-prefetch queue adds producer/consume overhead. A lower-risk
next check is to reuse the existing `GGML_MOE_TRACE_PREFETCH` path, which
prefetches directly into the VRAM cache via the existing io_uring multi-job
copy path. Earlier trace-prefetch attempts used raw route traces and failed
because they added extra read work/cache churn. The filtered SSD-miss trace
should avoid most of that noise.

Plan:

- no code changes first;
- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 39 best config with split-stage overlap;
- use
  `GGML_MOE_TRACE_PREFETCH=phase43-read-ahead-analysis/phase39-predicted-ssd.trace.csv`;
- start conservatively: `window=16`, `max_loads=4`, `lead_events=0`;
- run n8 guard only;
- run n84 only if n8 improves total time or eval speed without increasing
  reads, read failures, CUDA errors, or output drift.

Acceptance:

- report trace-prefetch stats, io_uring reads/wait, VRAM hit, token rates,
  total time, and read failures;
- output prefix must remain stable;
- if n8 is slower, stop this path and keep Phase 39 as best.
- 2026-06-13 18:36 CST: Checked resources before running: 5090 showed only
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 18:37 CST: Ran strict n8
  `ssdmiss-traceprefetch-w16-l4-n8`.
  Result: `1.43 eval tok/s`, `1.76 prompt tok/s`, `total_ms=24723.28`,
  `read_failures=0`, `8438` io_uring reads, `3249` direct reads,
  `34.0%` VRAM hit.
  Output drifted: prefix changed to
  `(1 0 0 3 ...)` instead of the expected
  `(Hint: answer D; see section ...)`.
  Trace-prefetch stats:
  - `calls=12624`, `matched=353`, `resync=116`;
  - `loads=101`, `cached=0`, `missing_tensor=0`;
  - `cursor=601`, `prefetch_cursor=617/58460`.
  Interpretation: even with filtered SSD-miss trace, the existing
  trace-prefetch path is not aligned with runtime consumption. It advances a
  separate cursor slowly, inserts only `101` rows, pollutes/replaces useful
  cache rows, drops VRAM hit `43.2% -> 34.0%`, increases reads
  `7169 -> 8438`, and changes the generated route/output. Do not run n84 and
  do not pursue this trace-prefetch path further.

Phase 44 decision:

- filtered trace-prefetch is rejected;
- keep Phase 39 as current strict best;
- the next viable implementation needs ring-local future-miss staging that does
  not insert speculative rows into VRAM cache and does not use the separate
  host-prefetch producer/consume path.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the runner and analysis helpers;
- `git diff --check` passed;
- no llama/matrix experiment process remained after the run.

## Phase 45: No-Graph VRAM Cache Headroom Probe

Phase 39 established that disabling CUDA graphs allows a larger strict VRAM
cache than the graph-enabled path, and `cache_safety_mib=128` is the current
strict best. Before implementing a more invasive ring-local future-read cache,
try the simpler 5090-specific route: spend a little more of the 32 GB VRAM on
resident experts by reducing the no-graph cache safety margin.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep the Phase 39 best config:
  - CUDA graphs disabled;
  - `GGML_MOE_IO_SQPOLL=1`;
  - `GGML_MOE_VRAM_CACHE_MIB=14336`;
  - `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
  - `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
  - `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=3`;
  - `GGML_MOE_VRAM_CACHE_UPGATE_PCT=65`;
  - route profile `oracle-n84.route.csv`;
  - up/gate split-stage overlap, down parallel stage, offset sort, GPU handoff,
    and prompt dynamic experts enabled;
- change only `cache_safety_mib` from `128` to `64` first;
- run n8 guard before n84;
- if n8 OOMs, falls back, changes output, or regresses clearly, stop and keep
  Phase 39 as best;
- if n8 is stable and reduces reads or improves runtime, run n84.

Acceptance:

- report actual cache size/slots, VRAM hit, io_uring reads/wait, token rates,
  total time, read failures, and any CUDA/OOM/fallback messages;
- compare to Phase 39 n8 (`1.64 eval tok/s`, `24052.45 ms`, `7169` reads,
  `43.2%` VRAM hit) and Phase 39 n84 (`2.18 eval tok/s`, `57795.16 ms`,
  `58489` reads, `60.9%` VRAM hit);
- keep this as a headroom probe unless n84 beats the Phase 39 strict best.

- 2026-06-13 05:32 CST: Checked resources before Phase 45. 5090 showed
  `41 MiB` used and `0%` GPU utilization; host RAM had about `11 GiB`
  available; no other llama/matrix experiment process was present.
- 2026-06-13 05:33 CST: Ran strict n8
  `nograph-safety64-n8`.
  Result: `1.66 eval tok/s`, `1.74 prompt tok/s`, `total_ms=23940.45`,
  `read_failures=0`, `7154` io_uring reads, `3264` direct reads,
  `43.3%` VRAM hit.
  Compared with Phase 39 n8:
  - reads improved slightly (`7169 -> 7154`);
  - total time improved slightly (`24052.45 -> 23940.45 ms`);
  - eval improved slightly (`1.64 -> 1.66 tok/s`);
  - output prefix stayed stable and no CUDA/OOM/fallback message was observed
    in the runner result.
  Decision: run strict n84 because the n8 guard is stable and slightly better.
- 2026-06-13 05:35 CST: Ran strict n84
  `nograph-safety64-n84`.
  Result: `2.18 eval tok/s`, `1.76 prompt tok/s`, `total_ms=57763.92`,
  `read_failures=0`, `58275` io_uring reads, `3264` direct reads,
  `61.0%` VRAM hit.
  Stderr confirms:
  - actual VRAM cache `13728 MiB` with `safety=64 MiB`;
  - upgate cache `2321` slots, down cache `1045` slots;
  - `~ggml_backend_cuda_context: have 0 graphs`;
  - no direct/io_uring fallbacks or read failures.
  Compared with Phase 39 n84:
  - reads improved slightly (`58489 -> 58275`);
  - total time improved slightly (`57795.16 -> 57763.92 ms`);
  - VRAM hit improved slightly (`60.9% -> 61.0%`);
  - eval remains rounded to `2.18 tok/s`.
  Decision: this is the new strict best by total time and reads, but the gain
  is too small to explain the 5090 gap. Continue the same headroom probe with
  an n8-only `cache_safety_mib=0` guard. If it OOMs or regresses, keep
  `safety=64` as the best headroom setting.
- 2026-06-13 05:39 CST: Ran strict n8
  `nograph-safety0-n8`.
  Result: failed with return code `1`; not a valid performance result.
  Stderr shows the real failure:
  - down batched decode path declined and fell back;
  - `ggml_cuda_compute_forward: MUL_MULTI_ADD failed`;
  - `CUDA error: out of memory`;
  - the later `ptrace`/attach text was only crash-handler output.
  Decision: `cache_safety_mib=0` is too aggressive for this no-graph strict
  config. Try one intermediate n8 guard with `cache_safety_mib=32`; if it is
  unstable or only neutral, keep `safety=64` as the practical strict best.
- 2026-06-13 05:42 CST: Ran strict n8
  `nograph-safety32-n8`.
  Result: failed with return code `1`; not a valid performance result.
  Stderr confirms:
  - actual VRAM cache `13760 MiB` with `safety=32 MiB`;
  - down batched CUDA MoE path declined;
  - `ggml_cuda_compute_forward: MUL_MULTI_ADD failed`;
  - `CUDA error: out of memory`.
  Decision: the no-graph strict stable cache boundary on this idle 5090 is
  between `13728 MiB` and `13760 MiB`; `cache_safety_mib=64` is the practical
  stable headroom setting for this config. Stop this direction here because it
  only gives a small read/runtime improvement and cannot close the theoretical
  5090 token-rate gap by itself.

Phase 45 decision:

- New strict best by total/runtime/read count is
  `nograph-safety64-n84`:
  - `2.18 eval tok/s`;
  - `1.76 prompt tok/s`;
  - `total_ms=57763.92`;
  - `read_failures=0`;
  - `58275` io_uring reads and `3264` direct reads;
  - `61.0%` VRAM hit;
  - actual cache `13728 MiB`, upgate `2321` slots, down `1045` slots.
- It is only marginally better than Phase 39
  (`57795.16 -> 57763.92 ms`, `58489 -> 58275` reads), so the main remaining
  bottleneck is still SSD/I/O staging wait, not lack of a few extra VRAM cache
  slots.
- `cache_safety_mib=32` and `0` both OOM under the same strict config, so do
  not use them for final strict results.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the Phase 45 runner/helper scripts;
- `git diff --check` passed;
- after the runs, 5090 returned to `41 MiB` used and no llama/matrix experiment
  process remained.

## Phase 46: No-Graph Reserve 2 Recheck At Safety 64

Phase 38 rejected `profile_reserve_pct=2` in the graph-reuse/safety256 cache
shape because it removed only two n8 reads and slowed the run. Phase 45 changed
the stable cache boundary: no CUDA graphs with `cache_safety_mib=64` adds more
VRAM cache slots (`upgate=2321`, `down=1045`). Recheck whether reserve `2`
becomes useful in this larger stable no-graph cache shape.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 45 best config:
  - CUDA graphs disabled;
  - `GGML_MOE_IO_SQPOLL=1`;
  - `GGML_MOE_VRAM_CACHE_MIB=14336`;
  - `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
  - `GGML_MOE_VRAM_CACHE_SAFETY_MIB=64`;
  - `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
  - `GGML_MOE_VRAM_CACHE_UPGATE_PCT=65`;
  - up/gate split-stage, down parallel stage, offset sort, GPU handoff, and
    prompt dynamic experts enabled;
- change only `GGML_MOE_VRAM_PROFILE_RESERVE_PCT` from `3` to `2`;
- run n8 first;
- run n84 only if n8 reduces reads materially without output drift, CUDA OOM,
  batched MoE decline, or total-time regression.

Acceptance:

- compare against Phase 45 safety64 n8 (`7154` reads, `23940.45 ms`,
  `1.66 eval tok/s`) and n84 (`58275` reads, `57763.92 ms`,
  `2.18 eval tok/s`);
- report actual cache, preloaded/pinned slots, VRAM hit, io_uring reads/wait,
  token rates, total time, and read failures.

- 2026-06-13 05:39 CST: Checked resources before Phase 46. 5090 showed
  `41 MiB` used and `0%` GPU utilization; host RAM had about `11 GiB`
  available; no other llama/matrix experiment process was present.
- 2026-06-13 05:40 CST: Ran strict n8
  `nograph-safety64-reserve2-n8`.
  Result: `1.65 eval tok/s`, `1.76 prompt tok/s`, `total_ms=23797.36`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads,
  `43.5%` VRAM hit.
  Compared with Phase 45 n8:
  - reads improved (`7154 -> 7137`);
  - total time improved (`23940.45 -> 23797.36 ms`);
  - VRAM hit improved (`43.3% -> 43.5%`);
  - output prefix stayed stable and no fallback/read failure was reported.
  Decision: run strict n84 because the n8 guard is stable and improves the
  main read/total metrics.
- 2026-06-13 05:41 CST: Ran strict n84
  `nograph-safety64-reserve2-n84`.
  Result: `2.24 eval tok/s`, `1.75 prompt tok/s`, `total_ms=56916.70`,
  `read_failures=0`, `57788` io_uring reads, `3298` direct reads,
  `61.3%` VRAM hit.
  Stderr confirms:
  - actual VRAM cache `13728 MiB` with `safety=64 MiB`;
  - upgate cache `2321` slots, `2274` preloaded/pinned, hit `62.4%`;
  - down cache `1045` slots, `1024` preloaded/pinned, hit `59.1%`;
  - `~ggml_backend_cuda_context: have 0 graphs`;
  - no direct/io_uring fallbacks or read failures.
  Compared with Phase 45 safety64 reserve3 n84:
  - reads improved (`58275 -> 57788`);
  - VRAM hit improved (`61.0% -> 61.3%`);
  - total time improved (`57763.92 -> 56916.70 ms`);
  - eval improved (`2.18 -> 2.24 tok/s`).

Phase 46 decision:

- New strict best is `nograph-safety64-reserve2-n84`:
  - `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
  - `GGML_CUDA_DISABLE_GRAPHS=1`;
  - `GGML_MOE_IO_SQPOLL=1`;
  - `GGML_MOE_VRAM_CACHE_MIB=14336`;
  - `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`;
  - `GGML_MOE_VRAM_CACHE_SAFETY_MIB=64`;
  - `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=0`;
  - `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=2`;
  - `GGML_MOE_VRAM_CACHE_UPGATE_PCT=65`;
  - up/gate split-stage, down parallel stage, offset sort, GPU handoff, and
    prompt dynamic experts enabled.
- This is the first material strict n84 improvement after the no-graphs cache
  headroom change: `2.24 tok/s` and `56916.70 ms`.
- It still does not reach the theoretical 5090 target. Remaining evidence:
  `57788` runtime io_uring reads, `iouring_inflight_avg=1.63`,
  `iouring_inflight_max=4`, and most batches are still size `1-4`.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the Phase 46 runner/helper scripts;
- `git diff --check` passed;
- after the runs, 5090 returned to `41 MiB` used and no llama/matrix experiment
  process remained.

## Phase 47: No-Graph Reserve 1 Guard

Phase 46 showed that reserve `2` becomes useful once no-graphs/safety64 adds
enough cache slots. It pins `3298` profile rows into `3366` total slots
(`2321 + 1045`), leaving `68` non-pinned slots. Reserve `1` should pin closer
to the profile upper bound while still leaving roughly half that replacement
space. This is close enough to the over-pinned regime that it needs an n8 guard.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep the Phase 46 best config exactly, except set
  `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=1`;
- run n8 first;
- run n84 only if n8 is stable and improves reads/total time without output
  drift, CUDA OOM, batched MoE decline, or fallback.

Acceptance:

- compare against Phase 46 reserve2 n8 (`7137` reads, `23797.36 ms`,
  `43.5%` VRAM hit) and n84 (`57788` reads, `56916.70 ms`,
  `2.24 eval tok/s`);
- report actual cache, preloaded/pinned rows, remaining replacement slots,
  per-cache hit/miss, token rates, total time, and read failures.

- 2026-06-13 05:43 CST: Checked resources before Phase 47. 5090 showed
  `41 MiB` used and `0%` GPU utilization; host RAM had about `11 GiB`
  available; no other llama/matrix experiment process was present.
- 2026-06-13 05:44 CST: Ran strict n8
  `nograph-safety64-reserve1-n8`.
  Result: `1.66 eval tok/s`, `1.74 prompt tok/s`, `total_ms=23833.84`,
  `read_failures=0`, `7098` io_uring reads, `3331` direct reads,
  `43.8%` VRAM hit.
  Compared with Phase 46 reserve2 n8:
  - reads improved (`7137 -> 7098`);
  - VRAM hit improved (`43.5% -> 43.8%`);
  - total time regressed slightly (`23797.36 -> 23833.84 ms`);
  - output prefix stayed stable and no CUDA/OOM/fallback was reported.
  Decision: run n84 despite the small n8 total-time regression, because the
  read reduction is clear and n8 total timing has been noisy relative to long
  n84 decode behavior. Stop reserve lowering if n84 regresses or shows any
  batched MoE decline.
- 2026-06-13 05:46 CST: Ran strict n84
  `nograph-safety64-reserve1-n84`.
  Result: `2.22 eval tok/s`, `1.75 prompt tok/s`, `total_ms=57111.63`,
  `read_failures=0`, `57316` io_uring reads, `3331` direct reads,
  `61.6%` VRAM hit.
  Stderr confirms:
  - actual VRAM cache `13728 MiB`;
  - upgate cache `2321` slots, `2297` preloaded/pinned, hit `62.8%`;
  - down cache `1045` slots, `1034` preloaded/pinned, hit `59.4%`;
  - only `35` replacement slots remained across both caches;
  - `~ggml_backend_cuda_context: have 0 graphs`;
  - no direct/io_uring fallbacks or read failures.
  Compared with Phase 46 reserve2 n84:
  - reads improved (`57788 -> 57316`);
  - VRAM hit improved (`61.3% -> 61.6%`);
  - total time regressed (`56916.70 -> 57111.63 ms`);
  - eval regressed (`2.24 -> 2.22 tok/s`).

Phase 47 decision:

- Do not adopt reserve `1`; keep Phase 46 reserve `2` as the strict best.
- The result is useful diagnostically: pinning more rows can lower SSD read
  count, but once only about `35` replacement slots remain, long-run token rate
  regresses despite fewer reads. This points to replacement flexibility and
  staging overlap, not just raw hit count.
- Do not test reserve `0` under this strict no-graph/safety64 config unless a
  new cache policy is implemented, because it would almost certainly over-pin
  and risk the known batched MoE decline/OOM regime.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the Phase 47 runner/helper scripts;
- `git diff --check` passed;
- after the runs, 5090 returned to `41 MiB` used and no llama/matrix experiment
  process remained.

## Phase 48: Hybrid Profile LFU/LRU Recheck On Current Strict Best

Earlier cache policy tests rejected profile-count replacement under older cache
shapes. Phase 46/47 changed the useful region: reserve `2` is best, while
reserve `1` shows that fewer reads can still be slower when replacement space
gets too tight. Recheck the existing env-gated
`GGML_MOE_VRAM_CACHE_POLICY=hybrid_profile_lfu_lru` on the current strict best
shape, where the policy only affects runtime evictions after
`GGML_MOE_VRAM_CACHE_PROFILE_AFTER`.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best config exactly:
  - no CUDA graphs;
  - `cache_safety_mib=64`;
  - `profile_reserve_pct=2`;
  - `upgate_pct=65`;
  - SQPOLL, offset sort, up/gate split-stage, down parallel stage, GPU handoff,
    and prompt dynamic experts enabled;
- add only:
  - `GGML_MOE_VRAM_CACHE_POLICY=hybrid_profile_lfu_lru`;
  - keep default `GGML_MOE_VRAM_CACHE_PROFILE_AFTER=12624`;
- run n8 first;
- run n84 only if n8 preserves output and improves either total time or a
  meaningful combination of reads/wait without fallback/OOM.

Acceptance:

- compare against Phase 46 reserve2 n8 and n84;
- report cache policy diag, VRAM hit, reads, wait time, token rates, total time,
  read failures, and any fallback/decline messages;
- do not adopt if policy reduces reads but regresses total time like reserve1.

- 2026-06-13 05:52 CST: Checked resources before Phase 48. 5090 showed
  `41 MiB` used and `0%` GPU utilization; host RAM had about `11 GiB`
  available; no other llama/matrix experiment process was present.
- 2026-06-13 05:53 CST: Ran strict n8
  `nograph-safety64-reserve2-hybrid-n8`.
  Result: `1.66 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24277.05`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads,
  `43.5%` VRAM hit.
  Stderr confirms:
  - actual VRAM cache `13728 MiB`;
  - `~ggml_backend_cuda_context: have 0 graphs`;
  - no direct/io_uring fallbacks or read failures;
  - cache policy diag:
    `profile_count_lookups=7137`, `hits=7137`,
    `inserted_nonzero=10435`, `inserted_avg=12.11`,
    `evictions=7069`, `victim_nonzero=7069`, `victim_avg=4.87`.
  Compared with Phase 46 reserve2 n8:
  - reads unchanged (`7137`);
  - VRAM hit unchanged (`43.5%`);
  - total time regressed (`23797.36 -> 24277.05 ms`);
  - eval is unchanged after rounding.

Phase 48 decision:

- Do not run n84 and do not adopt `hybrid_profile_lfu_lru` for the current
  strict best. The policy actively changes eviction choice, but it does not
  change the n8 miss sequence and only adds overhead.
- The next meaningful work should target read-ahead/aggregation of the remaining
  small io_uring batches, not another profile-score replacement policy.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the Phase 48 runner/helper scripts;
- `git diff --check` passed;
- after the run, 5090 returned to `41 MiB` used and no llama/matrix experiment
  process remained.

## Phase 49: Profile Current Strict Best Before Read-Ahead Work

Phase 40 profiling was done on the older Phase 39 best. The current strict best
is Phase 46 (`no-graphs`, `safety=64`, `reserve=2`) and has different cache
residency. Before implementing another read-ahead/aggregation path, profile the
current best to confirm whether the bottleneck is still split-stage I/O wait.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best config exactly:
  - `GGML_CUDA_DISABLE_GRAPHS=1`;
  - `GGML_MOE_IO_SQPOLL=1`;
  - `cache_safety_mib=64`;
  - `profile_reserve_pct=2`;
  - `upgate_pct=65`;
  - up/gate split-stage, down parallel stage, offset sort, GPU handoff, and
    prompt dynamic experts enabled;
- enable profiling with `GGML_MOE_BATCH_PROFILE=1`;
- run n8 only;
- do not treat profiling token rate as a new best because profiling adds CUDA
  events/timing overhead.

Acceptance:

- report up/gate and down profile lines, pinned-stage timing, iouring
  wait/submit, batch histogram, token rates, total time, VRAM hit, and
  read failures;
- use the result to decide whether the next implementation should target
  cross-ring read-ahead, cache admission, or another runtime path.

- 2026-06-13 05:53 CST: Checked resources before Phase 49. 5090 showed
  `41 MiB` used and `0%` GPU utilization; host RAM had about `11 GiB`
  available; no other llama/matrix experiment process was present.
- 2026-06-13 05:54 CST: Ran strict profiling n8
  `nograph-safety64-reserve2-profile-n8`.
  Result: `1.62 eval tok/s`, `1.73 prompt tok/s`, `total_ms=24242.77`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads,
  `43.5%` VRAM hit. This is diagnostic only and is not a new best.
  I/O details:
  - `iouring_wait_us=8415752`, `iouring_submit_us=18020`;
  - `iouring_inflight_avg=1.82`, `iouring_inflight_max=4`;
  - batch histogram remains only `1` and `2-4`;
  - `3034` total io_uring batches for `7137` reads.
  Pinned-stage timing:
  - main/down ring: `copies=5958`, `host_stage=2051.735 ms`,
    `h2d=805.654 ms`, `enqueue=63.228 ms`, `slot_wait=7.585 ms`;
  - gate ring: `copies=2398`, `h2d=383.833 ms`;
  - up_aux: `copies=1040`, `h2d=146.810 ms`;
  - gate_aux: `copies=1039`, `h2d=154.554 ms`;
  - all staging fallbacks were `0`.
  Batch profiles:
  - up/gate: `calls=526`, `avg_active=8`, `up_wait=3.872 ms/call`,
    `gate_wait=4.085 ms/call`, `up_compute=0.147 ms/call`,
    `gate_compute=0.140 ms/call`, `kernel=4.247 ms/call`;
  - down: `calls=526`, `avg_active=8`, `stage=2.556 ms/call`,
    `kernel=0.329 ms/call`, `total=2.919 ms/call`.
  The run also wrote
  `runs/phase49-current-best-profile/profile-n8.route.csv` with `6444`
  profile entries.

Phase 49 decision:

- The current Phase 46 best is still dominated by I/O/staging wait:
  up/gate and down compute kernels are small relative to staging waits.
- Cache policy and reserve tuning are at the point of diminishing returns; the
  next implementation should target earlier or less fragmented pack reads while
  preserving split-stage overlap.
- Do not treat this profiling result as a new performance line.

## Phase 50: Current-Prompt Profile Preload Probe

Phase 49 wrote a current-prompt n8 route profile with `6444` entries. This is a
same-prompt diagnostic, not a general benchmark, but it can answer whether the
remaining gap is mostly profile mismatch. If a short current-prompt profile
materially reduces n84 reads under the current strict best config, then a
better online predictor/profile builder may be more valuable than low-level
io_uring changes. If it does not help, continue toward read-ahead/aggregation.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best config exactly:
  - no CUDA graphs;
  - `cache_safety_mib=64`;
  - `profile_reserve_pct=2`;
  - `upgate_pct=65`;
  - SQPOLL, offset sort, up/gate split-stage, down parallel stage, GPU handoff,
    and prompt dynamic experts enabled;
- change only the profile file from `oracle-n84.route.csv` to
  `runs/phase49-current-best-profile/profile-n8.route.csv`;
- run n8 guard first;
- run n84 only if n8 is stable and reduces reads or total time.

Acceptance:

- clearly label this as a same-prompt/current-prompt profile diagnostic;
- compare against Phase 46 best with `oracle-n84.route.csv`;
- report profile file, preloads/pinned rows, VRAM hit, reads, token rates,
  total time, output prefix stability, and read failures.

- 2026-06-13 05:56 CST: Checked resources before Phase 50. 5090 showed
  `41 MiB` used and `0%` GPU utilization; host RAM had about `11 GiB`
  available; no other llama/matrix experiment process was present.
- 2026-06-13 05:57 CST: Ran strict same-prompt diagnostic n8
  `nograph-safety64-reserve2-currentprofile-n8`.
  Profile file:
  `runs/phase49-current-best-profile/profile-n8.route.csv`.
  Result: `3.02 eval tok/s`, `1.73 prompt tok/s`, `total_ms=22122.95`,
  `read_failures=0`, `3146` io_uring reads, `3298` direct reads,
  `75.1%` VRAM hit.
  Compared with Phase 46 oracle-profile n8:
  - reads improved sharply (`7137 -> 3146`);
  - VRAM hit improved (`43.5% -> 75.1%`);
  - total time improved (`23797.36 -> 22122.95 ms`);
  - eval improved (`1.65 -> 3.02 tok/s`);
  - output prefix stayed stable.
  Decision: run n84 to measure the same-prompt upper bound. Do not report this
  as a general benchmark because the profile was generated from the same prompt.
- 2026-06-13 05:59 CST: Ran strict same-prompt diagnostic n84
  `nograph-safety64-reserve2-currentprofile-n84`.
  Profile file:
  `runs/phase49-current-best-profile/profile-n8.route.csv`.
  Result: `1.50 eval tok/s`, `1.74 prompt tok/s`, `total_ms=75267.70`,
  `read_failures=0`, `96931` io_uring reads, `3298` direct reads,
  `35.1%` VRAM hit.
  Stderr confirms:
  - actual VRAM cache `13728 MiB`;
  - upgate cache `2321` slots, `2274` preloaded/pinned, hit `35.7%`;
  - down cache `1045` slots, `1024` preloaded/pinned, hit `34.0%`;
  - no direct/io_uring fallbacks or read failures.
  Compared with Phase 46 oracle-profile n84:
  - reads regressed badly (`57788 -> 96931`);
  - VRAM hit regressed (`61.3% -> 35.1%`);
  - total time regressed (`56916.70 -> 75267.70 ms`);
  - eval regressed (`2.24 -> 1.50 tok/s`).

Phase 50 decision:

- The short current-prompt profile is excellent for the first 8 decode tokens
  but harmful for n84. It overfits the prefix and evicts too many rows needed
  later in the route stream.
- This is not a viable general optimization and not a new best.
- The useful lesson is that better online prediction could give large gains if
  it remains accurate beyond the prefix; a naive short same-prompt profile is
  worse than the existing oracle-n84 profile for long decode.
- Continue toward earlier/online read-ahead or route prediction rather than
  replacing the startup profile with a short current-prompt profile.

Verification:

- `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed;
- `python3 -m py_compile` passed for the Phase 50 runner/helper scripts;
- `git diff --check` passed;
- after the runs, 5090 returned to `41 MiB` used and no llama/matrix experiment
  process remained.

## Phase 51: Prompt-Profile Overlay Instead Of Profile Replacement

Phase 50 proved that replacing `oracle-n84.route.csv` with a short current
prompt profile overfits the first 8 tokens and destroys n84. The runtime already
has a separate `GGML_MOE_VRAM_PROFILE_PROMPT` path that preloads prompt-profile
rows without pinning and without eviction. Test this safer overlay mode:

- keep the main `oracle-n84.route.csv` pinned profile as Phase 46 best;
- add the Phase 49 `profile-n8.route.csv` only as `GGML_MOE_VRAM_PROFILE_PROMPT`;
- because prompt-profile insertions use `allow_evict=false` and `pin=false`,
  they can only fill empty/unpinned space and should not evict the oracle-pinned
  rows that keep n84 stable.

Plan:

- add a runner argument for `GGML_MOE_VRAM_PROFILE_PROMPT`;
- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best config exactly;
- add prompt-profile overlay:
  `runs/phase49-current-best-profile/profile-n8.route.csv`;
- run n8 guard first;
- run n84 only if n8 preserves output and does not increase reads/total time.

Acceptance:

- compare against Phase 46 best and Phase 50 replacement results;
- report whether prompt-profile rows were loaded, VRAM hit, reads, token rates,
  total time, read failures, and output prefix stability;
- adopt only if n84 does not regress while n8 improves or stays neutral.

- 2026-06-13 06:05 CST: Added runner argument `--prompt-profile`, mapping to
  `GGML_MOE_VRAM_PROFILE_PROMPT`.
- 2026-06-13 06:06 CST: Checked resources before Phase 51. 5090 showed
  `41 MiB` used and `0%` GPU utilization; no other llama/matrix experiment
  process was present.
- 2026-06-13 06:07 CST: Ran strict n8
  `nograph-safety64-reserve2-promptoverlay-n8`.
  Result: `2.18 eval tok/s`, `1.72 prompt tok/s`, `total_ms=23118.68`,
  `read_failures=0`, `4926` io_uring reads, `3347` direct reads,
  `61.0%` VRAM hit.
  Compared with Phase 46 n8:
  - reads improved (`7137 -> 4926`);
  - total time improved (`23797.36 -> 23118.68 ms`);
  - eval improved (`1.65 -> 2.18 tok/s`);
  - output prefix stayed stable.
  However stderr exposed a correctness problem in the overlay mechanics:
  - prompt-profile rows loaded first and occupied cache slots;
  - main oracle profile then skipped rows that were already present instead of
    upgrading them to protected/pinned rows;
  - final pinned count was only `85` (`26` down, `59` upgate), not the expected
    Phase 46 `3298` pinned rows.
  Decision: do not run n84 for this unsafe overlay shape. Fix the profile
  preload logic so a later protected profile preload can upgrade an existing
  unpinned prompt-profile slot to pinned/profile-count, then retest the overlay.
- 2026-06-13 06:25 CST: After the first overlay fix, ran strict n8
  `nograph-safety64-reserve2-promptoverlay-fixed-n8` under the Phase 46 best
  config plus prompt-profile overlay. Result: `2.19 eval tok/s`,
  `1.74 prompt tok/s`, `total_ms=22876.11`, `read_failures=0`, `4840`
  io_uring reads, `3347` direct reads, `61.7%` VRAM hit. This is better than
  the first unsafe overlay n8 but still not valid for n84 adoption: final pinned
  rows remained far below the Phase 46 oracle baseline (`478` total pinned vs
  expected `3298`). Decision: do not run n84 yet. Continue debugging why the
  prompt-profile overlay prevents the main oracle profile from restoring its
  protected/pinned preload set.
- 2026-06-13 06:06 CST: Root-caused the unsafe prompt overlay. In prompt mode,
  `ggml.c` called the prompt-profile preload entry instead of the regular
  oracle profile preload entry. Because prompt preload uses `pin=false` and
  `allow_evict=false`, it filled most cache slots first; the later oracle
  profile could only promote overlapping rows and could not rebuild the Phase 46
  protected set. Patched `ggml_cuda_moe_stream_preload_tensor_prompt()` to run
  the regular protected oracle profile preload before loading the evictable
  prompt overlay. This keeps the main `oracle-n84.route.csv` pinning as the
  primary cache shape and lets prompt profile fill only remaining opportunity.
- 2026-06-13 06:08 CST: The first reordered overlay guard still regressed
  (`1.40 eval tok/s`, `8964` io_uring reads, `29.0%` VRAM hit, only `1950`
  pinned rows). Stderr showed prompt-profile rows were inserted while the main
  oracle profile was still building its protected budget, so unpinned prompt
  rows consumed slots before oracle rows could be pinned. Added a second guard:
  prompt-profile overlay now returns until the corresponding cache reaches its
  protected oracle preload budget. This should make prompt overlay strictly a
  reserve-space filler rather than a competitor with the main profile.
- 2026-06-13 06:11 CST: Rebuilt and ran strict guarded n8
  `nograph-safety64-reserve2-promptoverlay-guarded-n8`. Result:
  `1.65 eval tok/s`, `1.77 prompt tok/s`, `total_ms=23860.03`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit. Cache shape matched the Phase 46 n8 baseline exactly:
  `3298` preloads/pinned total, down `1024` pinned, upgate `2274` pinned.
  This confirms the guard made overlay safe, but also that the prompt overlay no
  longer adds useful rows under the current `profile_reserve_pct=2` cache shape.
  Decision: do not run n84 for the guarded overlay because n8 is identical to
  baseline and cannot improve the current Phase 46 best. Do not adopt prompt
  overlay as a performance improvement in this form.

Phase 51 decision:

- Unsafe overlay can reduce n8 reads but breaks oracle pinning and is not valid
  for long decode.
- Safe overlay preserves oracle pinning but gives no measured benefit.
- Keep the implementation guard only as a correctness fix for future explicit
  `GGML_MOE_VRAM_PROFILE_PROMPT` experiments; do not claim it improves 5090
  throughput.
- Next work should target real read latency/parallelism reduction: either better
  online route prediction/read-ahead or deeper aggregation of pack reads, because
  Phase 49 shows wait time is still dominated by small staged io_uring reads.

## Phase 52: Route-Trace Read-Ahead Guards

Phase 51 showed that profile overlay is not the right lever. Phase 49 showed the
current best is dominated by staged I/O waits, with small io_uring batches and
low effective inflight depth. The next guarded probe is to reuse the already
implemented route-trace read-ahead paths under the strict Phase 46 config:

- `GGML_MOE_TRACE_PREFETCH=<oracle-n84.trace.csv>`: direct prefetch from the
  route trace into the VRAM cache using the prefetch stream;
- if that does not help, `GGML_MOE_HOST_PREFETCH=<oracle-n84.trace.csv>` or
  `GGML_MOE_PLANNED_HOST_PREFETCH=1`: move pack reads earlier into host pinned
  buffers so runtime H2D can avoid synchronous SSD reads.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best config unchanged (`cache=14336`, safety `64`, reserve `2`,
  upgate `65`, no CUDA graphs, SQPOLL, offset sort, split-stage, GPU handoff,
  prompt dynamic experts);
- run n8 guard with `trace_prefetch` first;
- run n84 only if n8 reduces reads or total time without read failures;
- otherwise test host-prefetch with small memory (`<=512 MiB`) and n8 only.
- 2026-06-13 06:16 CST: Ran strict n8
  `nograph-safety64-reserve2-traceprefetch-n8` with
  `GGML_MOE_TRACE_PREFETCH=oracle-n84.trace.csv`, window `96`, max loads `8`.
  Result: `1.61 eval tok/s`, `1.75 prompt tok/s`, `total_ms=24153.25`,
  `read_failures=0`, `7143` io_uring reads, `3297` direct reads, `44.0%`
  VRAM hit. Trace-prefetch diagnostics: `calls=12624`, `matched=12624`,
  `resync=0`, but only `loads=68` and `cached=5585`. Interpretation: the oracle
  trace matches perfectly, but with the current protected profile occupying the
  cache, trace prefetch can only use tiny reserve space and does not reduce the
  runtime miss stream. Decision: do not run n84 for this setting; test host
  prefetch next.
- 2026-06-13 06:19 CST: Ran strict n8
  `nograph-safety64-reserve2-hostprefetch-n8` with
  `GGML_MOE_HOST_PREFETCH=oracle-n84.trace.csv`, lead `2048`, slots `64`, max
  `512 MiB`. Result: `1.50 eval tok/s`, `1.72 prompt tok/s`,
  `total_ms=24380.35`, `read_failures=0`, `7082` io_uring reads, `3298` direct
  reads, `43.5%` VRAM hit. Host-prefetch diagnostics: `submitted=3789`, but only
  `hits=55`, `misses=7082`, `evicted=3670`, `reserved_skips=573`, `used=270 MiB`.
  Interpretation: offline trace host-prefetch matches the route, but the worker
  produces too much low-use data and too few timely hits for the runtime miss
  stream. Decision: do not run n84; test planned host-prefetch next because it
  uses active stage-copy jobs directly instead of scanning the oracle trace.
- 2026-06-13 06:22 CST: Ran strict n8
  `nograph-safety64-reserve2-plannedhostprefetch-n8` with
  `GGML_MOE_PLANNED_HOST_PREFETCH=1`, slots `64`, max `512 MiB`. Result:
  `1.50 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24391.36`,
  `read_failures=0`, `7093` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit. Planned host-prefetch diagnostics: `planned_enqueued=4166`,
  `planned_dequeued=3897`, `submitted=3897`, but only `hits=44` and
  `misses=7093`. Interpretation: planned jobs are submitted, but the read
  completion is too late to be used by the same stage-copy wave. Decision: do
  not run n84 for the current planned host-prefetch implementation; inspect and
  move the submit point earlier if continuing this direction.

Phase 52 next implementation attempt:

- Problem: both offline and planned host-prefetch read thousands of entries but
  hit only tens of runtime loads. The offline worker currently scans route trace
  entries without checking whether a candidate is already resident in the VRAM
  expert cache. Since the Phase 46 profile pins `3298` hot rows, many early
  trace events are already cache hits and should not consume host-prefetch slots.
- Change: while selecting offline host-prefetch candidates, skip rows whose
  `(tensor, expert_idx, expert_bytes)` key is already present in the appropriate
  VRAM cache. This should focus host-prefetch bandwidth and slots on predicted
  future misses.
- Guard: rerun only strict n8 first with the same host-prefetch config. Run n84
  only if n8 host-prefetch hits rise materially and total time does not regress.
- 2026-06-13 06:29 CST: Implemented a focused offline host-prefetch filter:
  protected profile preload now records the actual keys successfully pinned in
  VRAM cache, and the offline host-prefetch worker skips those profile-pinned
  keys while scanning the route trace. This keeps host-prefetch bandwidth and
  slots for predicted future misses instead of rereading rows already resident
  from the Phase 46 oracle profile. Added `profile_pinned_skips` to the host
  prefetch diagnostics and updated the runner parser. Build and `py_compile`
  passed.
- 2026-06-13 06:32 CST: Ran strict n8
  `nograph-safety64-reserve2-hostprefetch-skip-pinned-n8` after filtering
  offline host-prefetch away from profile-pinned VRAM rows. Result:
  `1.64 eval tok/s`, `1.74 prompt tok/s`, `total_ms=23876.15`,
  `read_failures=0`, `6189` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit. Host-prefetch improved materially versus the previous offline
  version: `hits=948` instead of `55`; `misses=6189` instead of `7082`;
  `profile_pinned_skips=38196`; `submitted=3645`; `used=287.25 MiB`. This cuts
  runtime io_uring reads by about `13.3%` versus the Phase 46 n8 baseline
  (`7137 -> 6189`) but total time is only neutral (`23797.36 ms` baseline vs
  `23876.15 ms`). Decision: run n84 once, because a longer decode may benefit
  more from the reduced runtime read count.
- 2026-06-13 06:36 CST: Ran strict n84
  `nograph-safety64-reserve2-hostprefetch-skip-pinned-n84` with the filtered
  offline host-prefetch. Result: `1.95 eval tok/s`, `1.75 prompt tok/s`,
  `total_ms=62223.35`, `read_failures=0`, `54746` io_uring reads, `3298`
  direct reads, `61.3%` VRAM hit. Host-prefetch diagnostics:
  `submitted=32779`, `hits=3042`, `misses=54746`,
  `profile_pinned_skips=272290`, `evicted=29674`, `used=287.25 MiB`. Compared
  with Phase 46 best n84, runtime io_uring reads improved (`57788 -> 54746`,
  about `5.3%` fewer), but total time regressed badly (`56916.70 -> 62223.35`
  ms) and eval regressed (`2.24 -> 1.95 tok/s`). Interpretation: the filtered
  host-prefetch now predicts useful misses, but its background direct reads
  compete with the main io_uring stream and add enough CPU/SSD pressure to lose
  wall-clock time. Decision: do not adopt host-prefetch in its current form.

Phase 52 decision:

- `trace_prefetch` matches the oracle route but cannot load enough useful rows
  because the protected profile already fills the cache.
- unfiltered host-prefetch has too few timely hits;
- filtered host-prefetch raises hit count and reduces runtime io_uring reads,
  but slows n84 due to competing background reads and extra synchronization.
- The next viable direction is not more independent host direct-read prefetch;
  it should either merge prefetch jobs into the existing io_uring scheduler so
  total inflight depth and ordering are controlled in one queue, or create a
  token/layer lookahead that submits future jobs early enough without duplicating
  pack traffic.

## Phase 53: io_uring Refill Batching Prototype

Phase 52 ruled out independent host direct-read prefetch: it can reduce runtime
io_uring read count, but the extra background reads compete with the main read
stream and slow n84. The next improvement should stay inside the main io_uring
path so read ordering, inflight depth, and SSD pressure are controlled by one
queue.

Observation from Phase 46/49/52:

- per-ring batches are still mostly `1` and `2-4` jobs;
- effective inflight is low (`~1.6-1.8`) even with configured depth `16`;
- `depth=32` and more staging slots did not help, because the number of misses
  available per call is small;
- the current copy loop refills and submits immediately after each CQE, which
  preserves low latency but creates many tiny queue refills.

Prototype:

- add default-off `GGML_MOE_IO_REFILL_BATCH=<N>` to the existing
  `expert_pack_iouring_copy_jobs()` path;
- after consuming one or more CQEs, refill up to `N` new SQEs before one
  `io_uring_submit()` instead of submitting after every CQE;
- keep default behavior unchanged (`N=1`);
- first test `N=4` on strict n8 under the Phase 46 best config;
- run n84 only if n8 improves or at least preserves total time while reducing
  submit/wait overhead.

Acceptance:

- no output divergence/read failures;
- report eval tok/s, total_ms, reads, wait_us, submit_calls, batch hist, and
  per-ring iouring stats;
- do not adopt if n84 regresses.
- 2026-06-13 06:46 CST: Implemented default-off io_uring refill batching via
  `GGML_MOE_IO_REFILL_BATCH`. Default `1` preserves the original immediate
  refill path. Values greater than `1` drain currently available CQEs first and
  then refill up to `N` SQEs before a single submit. Updated the Phase runner
  with `--io-refill-batch`.
- 2026-06-13 06:49 CST: Ran strict n8 default guard
  `nograph-safety64-reserve2-refill1-n8`. Result: `1.65 eval tok/s`,
  `1.72 prompt tok/s`, `total_ms=24261.80`, `read_failures=0`, `7137`
  io_uring reads, `3298` direct reads, `43.5%` VRAM hit. Batch/read shape
  matches the Phase 46 n8 baseline (`7137` reads, `3034` batches,
  `inflight_avg=1.82`), confirming `GGML_MOE_IO_REFILL_BATCH=1` preserves the
  original path. Continue with refill batch `4` guard.
- 2026-06-13 06:52 CST: Ran strict n8
  `nograph-safety64-reserve2-refill4-n8` with `GGML_MOE_IO_REFILL_BATCH=4`.
  Result: `1.65 eval tok/s`, `1.77 prompt tok/s`, `total_ms=23709.54`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit. io_uring shape was effectively unchanged versus refill1/Phase 46:
  `3034` batches, `3034` submit calls, `inflight_avg=1.82`, max `4`. This means
  the loop rarely has multiple CQEs available to drain before refill, so batching
  refills does not materially change the runtime. Decision: do not run n84 for
  refill batch `4`; keep it as a diagnostic/default-off knob only.
- 2026-06-13 06:55 CST: Added a read-only pack/trace adjacency analyzer and
  checked `oracle-n84.trace.csv` against `glm51-iq3xxs.expert-pack`. Results:
  `149424/149424` trace events resolved in the pack; strict-adjacent grouped
  runs were `143481`, so ideal same-window read coalescing would reduce request
  count by only about `4.0%`. `window=8/16/32` and `max_gap=4096` did not change
  the run count. Decision: do not implement complex adjacent pack-read
  aggregation now; the available locality is too weak to close the remaining
  5090 gap.
- 2026-06-13 06:58 CST: Rechecked the low-rank/nonresident skip path as the
  next possible way to reduce required SSD reads. Existing Phase 28/29 estimator
  says a correct CUDA-side skip could theoretically avoid `18.5%` reads at
  keep7 or `35.5%` at keep6 on the old graph-enabled n84 baseline. Phase 30's
  CPU-side execution prototype failed because it made the residency decision
  before CUDA-internal profile preload, so it skipped rows that would later
  become resident. A correct implementation must make the skip decision inside
  or after CUDA profile preload. However, the CUDA streaming entry points only
  receive active expert ids, not rank/order metadata; the rank cutoff decision
  currently exists in CPU row construction. Therefore a correct implementation
  needs either:
  - an ABI extension carrying per-active rank/order into
    `ggml_cuda_moe_stream_up_gate_batch()` and the matching down call; or
  - a CPU-side explicit profile preload step before skip filtering so CPU
    residency queries see the final profile-pinned cache.
  This is larger than a safe end-of-turn patch. Do not revive the earlier
  CPU-side skip prototype as-is; it is known to make decisions too early.

## Phase 54: Recheck Residency-Aware Skip On Current Strict Best

Earlier skip work was done on an older graph-enabled/cache-smaller baseline. The
current strict best is Phase 46: no CUDA graphs, `cache_safety_mib=64`,
`profile_reserve_pct=2`, `upgate_pct=65`, SQPOLL, offset sort, split-stage,
GPU handoff, and prompt dynamic experts. It preloads/pins `3298` rows and has a
higher VRAM hit rate than the Phase 28/30 baseline.

Before changing ABI, rerun the existing default-off skip paths under the current
Phase 46 best config:

- estimator n8 with `GGML_MOE_SKIP_ESTIMATE=1`, keep `7`, to quantify the new
  true candidate/nonresident ceiling after current profile preload/cache shape;
- execution n8 with `GGML_MOE_SKIP_NONRESIDENT=1`, keep `7`, to see whether the
  existing post-preload CPU path still changes output/cache shape or whether the
  current explicit preload path now makes it viable;
- run n84 only if n8 output prefix is stable and total/read count improves.

Acceptance:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 config exactly otherwise;
- report read failures, prefix stability, eval/prompt rates, total_ms, VRAM hit,
  direct/io_uring reads, and skip counters;
- do not adopt if output prefix changes or n8 does not improve.
- Phase 54 resume: Ran strict n8 estimator
  `nograph-safety64-reserve2-skipest-keep7-n8` with the current Phase 46 best
  config plus `GGML_MOE_SKIP_ESTIMATE=1`, keep `7`. Result:
  `1.65 eval tok/s`, `1.75 prompt tok/s`, `total_ms=24027.02`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit, `3298` preloaded/pinned rows. Output prefix stayed at the smoke
  prompt prefix. Skip estimator counter:
  `groups=8416`, `events=12624`, `candidates=1578`, `nonresident=1171`,
  `resident=407`, `nonresident_pct_events=9.3`, `resident_pct_candidates=25.8`,
  `upgate_nonresident=776`, `down_nonresident=395`. Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase54-skip-current-best/nograph-safety64-reserve2-skipest-keep7-n8/`.
  Interpretation: under the current strict best cache/profile shape, keep7 can
  at most remove about `9.3%` of routed events before accounting for correctness
  and compute-side effects, lower than the older graph-enabled estimator.
  Continue with n8 execution guard only.
- Phase 54 resume: Ran strict n8 execution guard
  `nograph-safety64-reserve2-skipexec-keep7-n8` with
  `GGML_MOE_SKIP_NONRESIDENT=1`, keep `7`. Result: `1.66 eval tok/s`,
  `1.75 prompt tok/s`, `total_ms=24102.48`, `read_failures=0`, `7193`
  io_uring reads, `3298` direct reads, `36.8%` VRAM hit. Output prefix changed
  from the baseline/estimator hint text (`answer D; see section...`) to
  `consider the bandwidth/through...`, so this is not a same-output
  optimization. Skip execution counter: `routes=8416`, `candidates=1052`,
  `skipped=824`, `preserved_resident=113`, `upgate_skipped=412`,
  `down_skipped=412`. Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase54-skip-current-best/nograph-safety64-reserve2-skipexec-keep7-n8/`.

Phase 54 decision:

- Do not adopt the current CPU-side skip execution path.
- It changes output, lowers VRAM hit (`43.5% -> 36.8%` on n8), and does not
  reduce runtime reads (`7137 -> 7193`).
- A correctness-preserving skip would need a model-quality/accuracy acceptance
  target and likely CUDA-side rank/residency handling; it is not a valid route
  for reproducing the current smoke with same behavior.

## Phase 55: More Replacement Slots For Trace Prefetch

Phase 52 trace-prefetch matched the oracle route exactly but loaded only `68`
rows under Phase 46 reserve `2`, because nearly all cache slots were pinned by
the startup profile. Phase 54 skip execution did not produce a valid speedup.
The next conservative probe is to keep the strict 2 GB line and current no-graph
headroom, but leave more replacement slots so trace-prefetch can actually place
future rows into the VRAM cache.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best otherwise: cache `14336`, safety `64`, graph reserve `0`,
  no CUDA graphs, SQPOLL, offset sort, split-stage, GPU handoff, prompt dynamic
  experts;
- change only `profile_reserve_pct` plus trace-prefetch:
  - first guard: reserve `5`, trace window `96`, max loads `8`, n8;
  - if reads fall enough and total time does not regress badly, run n84;
  - optionally test reserve `8` only if reserve `5` shows trace-prefetch loads
    rise but cache hit/time remain marginal.

Acceptance:

- no read failures;
- output prefix must stay comparable to the baseline smoke;
- report trace-prefetch diagnostics, pinned rows/slots, VRAM hit, read count,
  eval/prompt rates, and total_ms;
- do not adopt if the lost profile pins cost more than trace-prefetch gains.
- Phase 55 resume: Ran strict n8
  `nograph-safety64-reserve5-traceprefetch-n8`. Result:
  `1.46 eval tok/s`, `1.75 prompt tok/s`, `total_ms=24619.12`,
  `read_failures=0`, `8113` io_uring reads, `3195` direct reads, `37.1%`
  VRAM hit. Actual cache stayed `13728 MiB` (`requested=14336`, free `13792`,
  safety `64`). Trace-prefetch loads rose from the Phase 52 reserve2 value
  (`68`) to `168`, but at the cost of fewer protected profile-pinned rows:
  `pinned=3196` instead of `3298`. Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase55-traceprefetch-reserve/nograph-safety64-reserve5-traceprefetch-n8/`.

Phase 55 decision:

- Do not run n84 and do not adopt reserve5 trace-prefetch.
- Leaving more replacement slots makes trace-prefetch slightly more active, but
  losing oracle profile pins hurts much more: reads regress (`7137 -> 8113` on
  n8), VRAM hit regresses (`43.5% -> 37.1%`), and total time regresses.
- The trace-prefetch bottleneck is not solved by taking slots away from the
  protected profile.

## Phase 56: Free KV/Context VRAM For More Expert Cache

Phase 46 already requests more VRAM cache than the runtime can allocate:
`requested=14336 MiB`, `actual=13728 MiB`, because free VRAM after model/context
allocation is only `13792 MiB` and `cache_safety_mib=64`. Therefore simply
raising `GGML_MOE_VRAM_CACHE_MIB` to `16384` will not add expert slots.

The current smoke prompt is tiny (`17` prompt eval tokens), so `ctx=2048` may be
over-reserving KV/context memory for this benchmark. Earlier pre-no-graph tests
with smaller context were mixed, but they did not use the current Phase 46
no-graph/safety64/reserve2 strict best. Recheck the current best with smaller
context and a high cache request so any freed VRAM is converted into expert
cache.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best flags: no CUDA graphs, safety `64`, graph reserve `0`,
  reserve `2`, SQPOLL, offset sort, split-stage, GPU handoff, prompt dynamic
  experts;
- set `ctx=512` and request `vram_cache_mib=20480` with auto-clamp;
- run n8 guard first;
- run n84 only if actual cache slots increase and n8 output/read/time do not
  regress materially;
- if ctx512 is stable but neutral, do not test smaller ctx unless there is a
  clear free-VRAM/cache-slot gain.

Acceptance:

- no read failures, no CUDA OOM, no batched MoE decline;
- output prefix comparable to Phase 46;
- report actual cache budget, upgate/down slots and pins, VRAM hit, io_uring
  reads, eval/prompt rates, and total_ms.
- Phase 56 resume: Ran strict n8
  `nograph-safety64-reserve2-ctx512-cache20480-n8`. Result:
  `1.66 eval tok/s`, `1.77 prompt tok/s`, `total_ms=23779.12`,
  `read_failures=0`, `7099` io_uring reads, `3330` direct reads, `43.8%`
  VRAM hit. Output prefix matched the baseline smoke prefix. Actual VRAM cache
  increased from Phase 46's `13728 MiB` to `13860 MiB`
  (`requested=20480`, free `13924`, safety `64`), with upgate/down slots
  `2343/1056` instead of `2321/1045`. Profile preloads/pins increased
  `3298 -> 3330`. Compared with Phase 46 n8, reads improved (`7137 -> 7099`)
  and total time improved very slightly (`23797.36 -> 23779.12 ms`). No CUDA
  OOM, batched MoE decline, direct fallback, or read failure was observed. Log
  path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase56-ctx-cache-headroom/nograph-safety64-reserve2-ctx512-cache20480-n8/`.
  Decision: run n84 validation because the n8 guard is stable and actual cache
  slots increased.
- Phase 56 resume: Ran strict n84
  `nograph-safety64-reserve2-ctx512-cache20480-n84`. Result:
  `2.22 eval tok/s`, `1.75 prompt tok/s`, `total_ms=57364.60`,
  `read_failures=0`, `57330` io_uring reads, `3330` direct reads, `61.6%`
  VRAM hit. Actual VRAM cache stayed `13860 MiB`, with upgate/down slots
  `2343/1056` and `3330` protected profile rows pinned. Compared with Phase 46
  n84, this confirms the intended cache effect: reads fall (`57788 -> 57330`),
  VRAM hit rises (`61.3% -> 61.6%`), and io_uring wait falls
  (`69499383 us -> 68157256 us`). However, total time regresses
  (`56916.70 -> 57364.60 ms`) and eval drops (`2.24 -> 2.22 tok/s`), so
  `ctx512` is not a new strict best. Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase56-ctx-cache-headroom/nograph-safety64-reserve2-ctx512-cache20480-n84/`.

Phase 56 next guard:

- Test one middle point, `ctx=1024`, with the same high cache request
  (`20480 MiB`) and Phase 46 flags.
- Rationale: `ctx512` proves that freeing context memory can add expert slots,
  but may alter runtime shape enough to lose wall-clock time. `ctx1024` may
  retain more of the Phase 46 execution shape while still adding some expert
  cache.
- Run n8 only first. Run n84 only if n8 improves reads and total time versus
  Phase 46 n8 without output drift or fallback.
- Phase 56 resume: Ran strict n8
  `nograph-safety64-reserve2-ctx1024-cache20480-n8`. Result:
  `1.66 eval tok/s`, `1.76 prompt tok/s`, `total_ms=24216.91`,
  `read_failures=0`, `7112` io_uring reads, `3319` direct reads, `43.7%`
  VRAM hit. Actual cache was `13816 MiB`, upgate/down slots `2336/1052`,
  preloads/pinned `3319`. Output prefix matched the baseline smoke prefix, and
  there were no read failures/fallbacks. Compared with Phase 46 n8, reads fall
  only slightly (`7137 -> 7112`) but total time regresses badly
  (`23797.36 -> 24216.91 ms`). Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase56-ctx-cache-headroom/nograph-safety64-reserve2-ctx1024-cache20480-n8/`.

Phase 56 decision:

- Reducing context can free a small amount of VRAM for more expert slots, but
  the available gain is tiny (`+22/+11` slots for ctx512 versus Phase 46) and
  does not become a wall-clock win on n84.
- Keep Phase 46 as strict best. Do not adopt ctx512 or ctx1024.

## Phase 57: Free Batch Workspace VRAM For More Expert Cache

Phase 56 showed that freeing context/KV VRAM adds expert cache slots but changes
runtime shape enough to lose wall-clock time. Another way to free VRAM is to
reduce `-b` while keeping `ctx=2048`. The smoke prompt is only `17` prompt eval
tokens, so `batch=2048` may reserve more workspace than needed for this
benchmark. Earlier pre-Phase-46 batch tests were not under the current
no-graph/safety64/reserve2 best, so recheck only a guarded n8.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 flags and `ctx=2048`;
- set `batch=512`, `vram_cache_mib=20480`, auto-clamp, safety `64`;
- run n8 guard first;
- run n84 only if actual cache increases and n8 improves total time or at
  least reduces reads without a total-time regression.

Acceptance:

- no read failures, CUDA OOM, or batched MoE decline;
- output prefix comparable to Phase 46;
- report actual cache budget, slots/pins, VRAM hit, reads, eval/prompt rates,
  and total_ms.
- Phase 57 resume: Ran strict n8
  `nograph-safety64-reserve2-batch512-cache20480-n8`. Result:
  `1.37 eval tok/s`, `1.60 prompt tok/s`, `total_ms=25999.58`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit. Actual cache did not increase despite the higher request:
  `requested=20480 MiB`, `actual=13728 MiB`, upgate/down slots `2321/1045`,
  preloads/pinned `3298`. Output prefix matched the baseline smoke prefix and
  there were no read failures, but the reduced batch size made both prompt and
  decode much slower while preserving the same miss count. Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase57-batch-cache-headroom/nograph-safety64-reserve2-batch512-cache20480-n8/`.

Phase 57 decision:

- Do not run n84 and do not adopt `batch=512`.
- Reducing batch does not free usable VRAM for the expert cache in this current
  strict shape and directly hurts compute/runtime speed.

## Current Status After Phase 57

- Current strict best remains Phase 46:
  `nograph-safety64-reserve2-n84`, `2.24 eval tok/s`, `1.75 prompt tok/s`,
  `total_ms=56916.70`, `read_failures=0`, `57788` io_uring reads,
  `3298` direct reads, `61.3%` VRAM hit, actual cache `13728 MiB`.
- Newly tested after the resume:
  - Phase 54 skip execution: invalid because output changed and reads/hit rate
    regressed;
  - Phase 55 reserve5 trace-prefetch: more trace-prefetch loads, but fewer
    protected profile pins and worse reads/time;
  - Phase 56 ctx512/ctx1024: can add a small number of expert slots and reduce
    reads slightly, but not wall-clock time;
  - Phase 57 batch512: no extra expert cache and much slower runtime.
- The remaining gap is still the same structural bottleneck: many runtime
  expert-pack reads (`~57.8k` on n84) with low effective io_uring inflight
  (`~1.6`) and batches mostly size `1-4`, plus limited opportunity to add more
  expert cache without destabilizing or slowing the execution shape.

Verification after resume:

- `git diff --check` passed;
- `python3 -m py_compile` passed for the Phase runner and analysis helpers.

## Phase 58: Eager cuBLAS Recheck On Current Strict Best

The 5060 Ti-compatible startup matrix used `GGML_CUDA_EAGER_CUBLAS=1`, but that
was before the current Phase 46 strict best (`no graphs`, safety `64`, reserve
`2`, larger protected oracle profile, SQPOLL, split-stage, GPU handoff, prompt
dynamic experts). Recheck this single 5060-compatible knob under the current
best before moving to more invasive code changes.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best exactly:
  `cache=14336`, safety `64`, graph reserve `0`, profile reserve `2`,
  upgate pct `65`, no CUDA graphs, SQPOLL, offset sort, up/gate split-stage,
  down parallel stage, GPU handoff, prompt dynamic experts;
- add only `GGML_CUDA_EAGER_CUBLAS=1`;
- run n8 guard first;
- run n84 only if n8 preserves output/read shape and improves total time or
  eval speed without CUDA errors.

Acceptance:

- no read failures, CUDA OOM, or batched MoE decline;
- output prefix comparable to Phase 46;
- report eval/prompt rates, total_ms, VRAM hit, direct/io_uring reads, and
  cache shape;
- do not adopt if the difference is only noise or total time regresses.
- Phase 58 resume: Ran strict n8
  `nograph-safety64-reserve2-eagercublas-n8`. Result:
  `1.64 eval tok/s`, `1.76 prompt tok/s`, `total_ms=24435.85`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit. Output prefix matched the baseline smoke prefix, and cache/read
  shape matched Phase 46 exactly. Compared with Phase 46 n8, this is a clear
  total-time regression (`23797.36 -> 24435.85 ms`) with no reduction in reads.
  Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase58-eager-cublas/nograph-safety64-reserve2-eagercublas-n8/`.

Phase 58 decision:

- Do not run n84 and do not adopt `GGML_CUDA_EAGER_CUBLAS=1` for the current
  strict best.
- The 5060-compatible eager cuBLAS knob does not improve the current
  no-graph/safety64/reserve2 runtime shape.

## Phase 59: Recheck Up/Gate No-Split Under Current Strict Best

vramctl's directly transferable io_uring evidence is mixed for this VM:
SQPOLL is already enabled in Phase 46, while fixed buffers and fixed files were
documented as slower in vramctl Wave 2. The largest remaining runtime issue in
Phase 46 is still the tiny per-call batch shape (`inflight_avg~1.6`, max `4`,
mostly batch size `1-4`). Earlier no-split tests increased batch size but lost
overlap and regressed under an older cache shape. Recheck the same scheduling
tradeoff under the current Phase 46 strict best.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best config exactly except omit `GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT`;
- keep down parallel stage enabled;
- run n8 guard first;
- run n84 only if n8 improves total time or materially improves I/O wait/batch
  shape without a total-time regression.

Acceptance:

- no read failures, CUDA OOM, or batched MoE decline;
- output prefix comparable to Phase 46;
- report eval/prompt rates, total_ms, VRAM hit, direct/io_uring reads, wait_us,
  batch histogram/inflight, and cache shape;
- do not adopt if larger batches come at the cost of wall-clock slowdown.
- Phase 59 resume: Ran strict n8
  `nograph-safety64-reserve2-upgate-nosplit-n8`. Result:
  `1.62 eval tok/s`, `1.73 prompt tok/s`, `total_ms=24635.18`,
  `read_failures=0`, `7137` io_uring reads, `3298` direct reads, `43.5%`
  VRAM hit. Output prefix matched Phase 46 and cache/read count were unchanged.
  The scheduling effect was visible: batches improved `3034 -> 2038`,
  `inflight_avg` improved `1.82 -> 2.65`, max inflight improved `4 -> 8`, and
  io_uring wait fell `8402086 us -> 5527675 us`. However total time regressed
  (`23797.36 -> 24635.18 ms`) and eval regressed (`1.65 -> 1.62 tok/s`).
  Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase59-upgate-nosplit-current-best/nograph-safety64-reserve2-upgate-nosplit-n8/`.

Phase 59 decision:

- Do not run n84 and do not adopt up/gate no-split.
- This confirms the earlier pattern under the current strict best: larger
  synchronous I/O batches are not sufficient if they lose the split-stage
  overlap that hides some staging work.

## Phase 60: Recheck Down Parallel Stage Under Current Strict Best

Phase 59 showed that disabling up/gate split-stage improves I/O batch shape but
loses enough overlap to slow wall-clock time. The current Phase 46 best also
enables down parallel staging (`GGML_MOE_DOWN_PARALLEL_STAGE=1`). Recheck that
knob independently under the current strict best before making more invasive
scheduler changes.

Plan:

- strict line only: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`;
- keep Phase 46 best config exactly except disable down parallel stage via the
  runner's `--no-down-parallel-stage`;
- keep up/gate split-stage enabled;
- run n8 guard first;
- run n84 only if n8 improves wall-clock time or gives a clearly better
  compute/I/O balance without output drift.

Acceptance:

- no read failures, CUDA OOM, or batched MoE decline;
- output prefix comparable to Phase 46;
- report eval/prompt rates, total_ms, VRAM hit, direct/io_uring reads, wait_us,
  batch histogram/inflight, and cache shape;
- do not adopt if read shape is unchanged and total time regresses.
- Phase 60 resume: Ran strict n8
  `nograph-safety64-reserve2-no-down-parallel-n8`. Result:
  `1.54 eval tok/s`, `1.74 prompt tok/s`, `total_ms=24600.17`,
  `read_failures=0`, `4695` io_uring reads, `5740` direct reads, `43.5%`
  VRAM hit. Output prefix matched Phase 46 and total cache misses were
  unchanged, but disabling down parallel stage moved down-path misses out of
  the batched io_uring path and into synchronous direct reads. Compared with
  Phase 46 n8, eval regressed (`1.65 -> 1.54`) and total regressed
  (`23797.36 -> 24600.17 ms`). Log path:
  `.Agent/plans/5090-theoretical-token-rate/runs/phase60-down-stage-current-best/nograph-safety64-reserve2-no-down-parallel-n8/`.

Phase 60 decision:

- Do not run n84 and do not disable down parallel stage.
- Down parallel staging is required for the current strict best because it keeps
  down misses in the staged io_uring path rather than the slower synchronous
  direct-read path.

## Current Status After Phase 60

- Current strict best remains Phase 46:
  `nograph-safety64-reserve2-n84`, `2.24 eval tok/s`, `1.75 prompt tok/s`,
  `total_ms=56916.70`, `read_failures=0`, `57788` io_uring reads,
  `3298` direct reads, `61.3%` VRAM hit, actual cache `13728 MiB`.
- Additional resumed checks:
  - Phase 58 eager cuBLAS: no read/cache change and slower n8;
  - Phase 59 up/gate no-split: larger I/O batches but slower wall-clock;
  - Phase 60 no down parallel stage: more synchronous direct reads and slower
    wall-clock.
- The strict-line bottleneck remains structural: the fastest current shape is
  split-stage overlap plus many small residual io_uring batches. Config-only
  toggles that enlarge batches or alter staging have not closed the
  `~3.9-5.5 tok/s` theoretical gap.
