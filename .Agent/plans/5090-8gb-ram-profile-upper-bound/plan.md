# 5090 8GB RAM Profile Upper Bound

## Goal

Evaluate whether relaxing host RAM from the strict 2GB line to an 8GB limit can
raise GLM-5.1 MoE inference on the RTX 5090 toward `5 eval tok/s` for broad
prompts, and whether improved profiles can make that speed generalize.

This task is explicitly not the strict 2GB result line. Every result here must
be labeled as `8GB RAM` or `non-strict`.

## Hard Constraints

- Work only under `/home/wici/lfz`.
- Do not delete or modify files outside `/home/wici/lfz`.
- Store all plans, scripts, logs, profiles, and summaries for this task under:
  `.Agent/plans/5090-8gb-ram-profile-upper-bound/`.
- Before any run, check whether the RTX 5090 or host RAM is occupied by another
  task. If occupied, do not stop other processes; wait for them to finish.
- Use `MemoryMax=8G`, `MemorySwapMax=0` for the main 8GB line.
- Keep the old strict 2GB Phase 46 result only as a comparison baseline, not as
  part of this task's acceptance.
- Do not claim broad-prompt success from same-prompt or oracle-profile runs.

## Current Reference Points

Strict 2GB current best from the previous task:

- `nograph-safety64-reserve2-n84`
- `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`
- `2.24 eval tok/s`, `1.75 prompt tok/s`, `total_ms=56916.70`
- `read_failures=0`
- `57788` io_uring reads, `3298` direct reads
- `61.3%` VRAM hit

Best non-strict historical RAM-tier evidence:

- 9GiB RAM tier full-pin family reached about `2.65 eval tok/s`.
- 10GiB RAM tier reduced reads further but did not improve decode beyond the
  9GiB result.
- This suggests host RAM helps but does not automatically make the runtime
  GPU-bound.

Target for this task:

- Primary target: determine whether `5 eval tok/s` is reachable under an 8GB
  host-RAM cap.
- Practical milestone: exceed the previous non-strict `~2.65 eval tok/s` line
  while preserving broad-prompt behavior.
- Negative result is acceptable if supported by clear upper-bound evidence.

## Metrics To Record

For every meaningful run:

- `eval tok/s`
- `prompt_eval tok/s`
- `total_ms`, `load_ms`, `eval_ms`
- `direct_reads`
- `io_uring_reads`
- `io_uring_wait_us`
- io_uring batch histogram and inflight average/max
- `VRAM hit`
- `RAM hit`, RAM tier resident MiB, and whether RAM tier is pinned
- `read_failures`
- output prefix stability
- profile path and profile construction method
- prompt set / dataset split
- exact command, env, stdout/stderr/summary paths

## Phase 0: Runner And Baseline Reuse

Use the existing runner first:

`../5090-theoretical-token-rate/run_5090_matrix.py`

If it is sufficient, do not fork it. If this task needs broad prompt loops or
profile/train-test automation, add task-local wrapper scripts under this
directory.

Baseline checks:

1. Re-run one short strict Phase 46 n8 baseline only if needed to confirm the
   machine is stable.
2. Run a matching 8GB n8 baseline with `RAM tier=0`:
   - `MemoryMax=8G`, `MemorySwapMax=0`;
   - same Phase 46 config otherwise;
   - purpose: measure whether the looser cgroup alone changes performance.

Gate:

- If `RAM tier=0` with `MemoryMax=8G` is the same as strict 2GB, proceed to RAM
  tier/profile tests.
- If it differs materially, record it as machine/cgroup sensitivity before
  interpreting later results.

## Phase 1: 8GB RAM Tier Upper Bound On The Fixed n84 Prompt

Purpose: learn the best possible single-prompt speed under `MemoryMax=8G` before
spending time on broad profiles.

Candidate configurations:

1. `RAM tier=3072 MiB`
   - 5060 Ti-compatible budget.
   - Use residual RAM-tier profile rather than old fixed skip if available.
2. `RAM tier=4096 MiB`
   - 5060 Ti accepted budget.
   - Residual profile after current VRAM pins.
3. `RAM tier=6144 MiB`
   - Main 8GB upper-bound candidate.
   - Leave room for process overhead under `MemoryMax=8G`.
4. Optional `RAM tier=7168 MiB`
   - Only if host memory pressure and cgroup overhead are safe in n8.
   - Do not run n84 unless n8 is stable.

For each RAM tier size:

- generate or reuse a residual profile that covers rows not already pinned in
  VRAM under Phase 46 cache shape;
- run n8 first;
- run n84 only if n8 reduces runtime reads and does not regress total time;
- keep `RAM tier pinning` explicit:
  - full pin if it fits;
  - otherwise test `--ram-tier-no-pin` or a small pin prefix only as separate
    labeled variants.

Acceptance:

- `read_failures=0`;
- RAM tier reports non-zero useful hits;
- no cgroup OOM;
- n84 improves over strict Phase 46 and ideally over historical `2.65 tok/s`.

Decision point:

- If fixed-prompt 8GB cannot exceed `~2.65 tok/s`, reaching `5 tok/s` for broad
  prompts is very unlikely without larger architectural changes.
- If fixed-prompt 8GB approaches or exceeds `4 tok/s`, proceed aggressively to
  broad-profile generalization.

## Phase 2: Same-Prompt / Oracle Upper Bound

Purpose: estimate the absolute speed ceiling when profile placement is ideal
for the fixed n84 prompt.

Runs:

1. Capture route/trace for the fixed n84 prompt under the best 8GB RAM-tier
   candidate.
2. Build an oracle profile that assigns:
   - most useful rows to VRAM profile first;
   - residual frequent rows to RAM tier;
   - leaves enough replacement space to avoid batched MoE decline.
3. Replay the same prompt.

Interpretation:

- This is an upper-bound diagnostic only.
- It cannot prove broad-prompt performance.
- If this oracle line still cannot approach `5 tok/s`, then broad-prompt `5
  tok/s` is not credible on the current runtime.

Acceptance:

- Same output prefix behavior as the baseline prompt.
- No read failures.
- Record exact profile generation inputs.
- Report VRAM/RAM hit split and remaining SSD reads.

## Phase 3: Broad-Prompt Profile Construction

Purpose: test whether a profile can generalize across prompts rather than
overfit the fixed prompt.

Prompt set:

- Use existing smoke/evaluation prompts already present in the task history if
  available.
- If no suitable broad set exists locally, create a small task-local prompt
  suite with at least:
  - short factual prompt;
  - reasoning prompt;
  - code/math prompt;
  - instruction-following prompt;
  - longer context prompt.

Profile split:

- training prompts: used to build VRAM/RAM profiles;
- held-out prompts: used only for evaluation.

Profile variants:

1. Frequency profile:
   - count expert rows across training prompts.
2. Layer-balanced profile:
   - avoid overfilling a few layers while leaving others cold.
3. Upgate/down split-aware profile:
   - tune VRAM split and RAM residual profile together.
4. Optional prompt-class profile:
   - only if prompt classes show distinct expert distributions.

Evaluation:

- n8 guard on all held-out prompts first.
- n84 on the best 1-2 candidates.
- Report mean, median, min, and worst-case token rate.
- Report read counts and hit rates per prompt, not only aggregate averages.

Broad-prompt acceptance:

- Must improve held-out prompts, not only training prompts.
- Worst-case should not collapse below strict Phase 46 by more than noise.
- To claim target success, held-out n84 mean should be near `5 eval tok/s`, and
  worst-case should remain clearly above previous non-strict `~2.65 tok/s`.

## Phase 4: Feasibility Decision

After Phase 1-3, classify the result:

- `Target likely`: broad held-out prompts reach or approach `5 tok/s`, with low
  remaining SSD reads and stable output.
- `Profile-limited`: oracle/same-prompt is fast, but broad held-out prompts do
  not generalize. Next work should be profile generation, prompt clustering, or
  dynamic routing prediction.
- `Runtime-limited`: even oracle/profile-heavy fixed-prompt runs stay well
  below `5 tok/s`. Next work must change runtime scheduling, expert reduction,
  pack layout, or model execution strategy.
- `Memory-limited`: 8GB RAM tier cannot hold enough residual rows; reaching the
  target would require more RAM, more VRAM, or fewer active experts.

## Initial Hypothesis

Based on prior evidence, `5 eval tok/s` under `MemoryMax=8G` is unlikely on
broad prompts with the current runtime. The expected useful outcome is a tighter
upper-bound measurement:

- whether 8GB RAM can beat the previous strict `2.24 tok/s`;
- whether it can beat the historical non-strict `~2.65 tok/s`;
- whether profile quality or runtime scheduling is the dominant remaining
  limiter.

## Next Action

Before running experiments:

1. Check GPU/RAM/process occupancy.
2. Confirm existing residual profile scripts/outputs and runner options.
3. Run the Phase 0 8GB `RAM tier=0` baseline.
4. Generate or select the first `4096 MiB` and `6144 MiB` residual RAM-tier
   profiles.

## Progress Log

- 2026-06-15: Started implementation. Confirmed the existing task runner
  `.Agent/plans/5090-theoretical-token-rate/run_5090_matrix.py` supports the
  required knobs: `--memory-max`, `--memory-swap-max`, `--ram-tier-mib`,
  `--ram-tier-profile`, `--ram-tier-no-pin`, `--ram-tier-pin-mib`, profile
  paths, cache sizing, SQPOLL, split-stage, and n8/n84 token counts.
- 2026-06-15: Confirmed existing residual RAM-tier profiles are available from
  the previous task, including:
  - `ramtier-residual-3072.route.csv`;
  - `ramtier-residual-4096.route.csv`;
  - `ramtier-residual-6144.route.csv`;
  - `ramtier-residual-8192.route.csv`;
  - graph-reserve-aware 8192/9216/10240 profiles.
- 2026-06-15: Phase 0 first run will be `MemoryMax=8G`,
  `MemorySwapMax=0`, `RAM tier=0`, keeping the Phase 46 strict-best runtime
  config otherwise. This isolates cgroup/RAM-limit effects before enabling RAM
  tier.
- 2026-06-15: Phase 0 8GB cgroup baseline completed:
  `mem8g-ram0-nograph-safety64-reserve2-n8`, `MemoryMax=8G`,
  `MemorySwapMax=0`, `RAM tier=0`, profile
  `.Agent/plans/5090-theoretical-token-rate/oracle-n84.route.csv`.
  Result: `1.68 eval tok/s`, `1.68 prompt_eval tok/s`, `total_ms=23876.75`,
  `load_ms=19700.78`, `eval_ms=4170.53`, `direct_reads=3298`,
  `io_uring_reads=7137`, `io_uring_wait_us=8317662`,
  `io_uring_inflight_avg=1.82`, `io_uring_inflight_max=4`,
  `VRAM hit=43.5%`, `read_failures=0`. Logs:
  `.Agent/plans/5090-8gb-ram-profile-upper-bound/runs/phase0-8gb-baseline/mem8g-ram0-nograph-safety64-reserve2-n8/`.
  Interpretation: relaxing the cgroup to 8GB without RAM tier does not
  materially change the n8 baseline versus the strict 2GB line (`1.65 eval
  tok/s`, `total_ms=23797.36`), so later gains should be attributed to RAM
  tier/profile behavior rather than cgroup looseness alone.
- 2026-06-15: Phase 1 `RAM tier=4096 MiB` n8 guard completed:
  `mem8g-ram4096-residual-n8`, residual profile
  `.Agent/plans/5090-theoretical-token-rate/ramtier-residual-4096.route.csv`.
  Result: `1.81 eval tok/s`, `1.39 prompt_eval tok/s`, `total_ms=24678.94`,
  `load_ms=20803.16`, `eval_ms=3870.28`, `direct_reads=2980`,
  `io_uring_reads=6579`, `io_uring_wait_us=7645455`, `RAM hit=8.4%`,
  `RAM resident=4096.0 MiB`, `VRAM hit=43.5%`, `read_failures=0`. Logs:
  `.Agent/plans/5090-8gb-ram-profile-upper-bound/runs/phase1-ramtier/mem8g-ram4096-residual-n8/`.
  Interpretation: decode improved slightly versus Phase 0, and SSD reads fell
  from `7137` to `6579`, but total time worsened due to higher load and slower
  prompt evaluation. This is not strong enough to promote directly to n84.
- 2026-06-15: Phase 1 `RAM tier=6144 MiB` n8 guard completed:
  `mem8g-ram6144-residual-n8`, residual profile
  `.Agent/plans/5090-theoretical-token-rate/ramtier-residual-6144.route.csv`.
  Result: `1.87 eval tok/s`, `1.21 prompt_eval tok/s`, `total_ms=26339.51`,
  `load_ms=22600.82`, `eval_ms=3733.38`, `direct_reads=2980`,
  `io_uring_reads=6278`, `io_uring_wait_us=7216661`, `RAM hit=11.3%`,
  `RAM resident=6144.0 MiB`, `VRAM hit=43.5%`, `read_failures=0`. Logs:
  `.Agent/plans/5090-8gb-ram-profile-upper-bound/runs/phase1-ramtier/mem8g-ram6144-residual-n8/`.
  Interpretation: the larger RAM tier further reduces SSD reads and improves
  short decode speed, but worsens load and prompt-eval enough that total n8
  time is worse than RAM tier off. Because the target is sustained eval token
  rate, run one n84 despite the n8 total-time gate and label it as an
  exploratory upper-bound check.
- 2026-06-15: Phase 1 exploratory `RAM tier=6144 MiB` n84 completed:
  `mem8g-ram6144-residual-n84`. Result: `2.75 eval tok/s`, `1.19
  prompt_eval tok/s`, `total_ms=52922.96`, `load_ms=22736.77`,
  `eval_ms=30145.18`, `direct_reads=2980`, `io_uring_reads=43403`,
  `io_uring_wait_us=47831876`, `io_uring_inflight_avg=1.53`,
  `io_uring_inflight_max=4`, `RAM hit=24.1%`, `RAM resident=6144.0 MiB`,
  `VRAM hit=61.3%`, `read_failures=0`. Logs:
  `.Agent/plans/5090-8gb-ram-profile-upper-bound/runs/phase1-ramtier/mem8g-ram6144-residual-n84/`.
  Interpretation: 6GB RAM tier beats the old strict Phase 46 n84 line (`2.24
  eval tok/s`) and slightly exceeds the previous non-strict `~2.65 eval tok/s`
  reference. It is still far from `5 eval tok/s`: residual SSD reads remain
  high (`43403` io_uring reads) and accumulated io_uring wait is `47.8s`.
- 2026-06-15: Optional `RAM tier=7168 MiB` n8 guard attempted:
  `mem8g-ram7168-residual8192prefix-n8`, using the 8192 MiB residual profile
  as a prefix. It returned `1` before timings. Stdout only reached the prompt
  prefix, and stderr ended around first decode after warnings:
  `madvise(..., MADV_DONTNEED) failed: Cannot allocate memory`. Logs:
  `.Agent/plans/5090-8gb-ram-profile-upper-bound/runs/phase1-ramtier/mem8g-ram7168-residual8192prefix-n8/`.
  Interpretation: under `MemoryMax=8G`, a fully resident 7 GiB RAM tier is too
  close to the cgroup limit for this runtime shape. The practical pinned RAM
  tier ceiling is therefore near 6 GiB.
- 2026-06-15: Re-read RAM tier implementation in
  `ggml/src/ggml-cuda/moe_stream_batch.cu`. `ram_tier_init()` allocates an
  anonymous mmap of `GGML_MOE_RAM_TIER_MIB`, reads selected expert-pack entries
  into it, and then optionally calls `cudaHostRegister` on the loaded prefix.
  `--ram-tier-no-pin` only disables `cudaHostRegister`; it does not avoid the
  resident host-memory budget. Therefore no-pin is not a credible way to fit a
  7 GiB tier inside an 8 GiB cgroup, though it remains useful for separating
  pinning overhead from residency overhead at larger memory limits.

## Interim Result And Analysis

Best result so far under the 8GB RAM line:

- `mem8g-ram6144-residual-n84`
- `MemoryMax=8G`, `MemorySwapMax=0`, `RAM tier=6144 MiB`
- `2.75 eval tok/s`, `1.19 prompt_eval tok/s`, `total_ms=52922.96`
- `VRAM hit=61.3%`, `RAM hit=24.1%`
- `direct_reads=2980`, `io_uring_reads=43403`, `io_uring_wait_us=47831876`
- `read_failures=0`

Comparison:

- 8GB cgroup with `RAM tier=0` is effectively unchanged from strict 2GB:
  `1.68 eval tok/s` on n8.
- 4GB RAM tier improves n8 decode to `1.81 eval tok/s` but worsens total time.
- 6GB RAM tier improves n8 decode to `1.87 eval tok/s`, and n84 to `2.75 eval
  tok/s`.
- 7GB resident RAM tier does not complete under `MemoryMax=8G`.

Feasibility reading:

- The 6GB tier reduced n84 SSD reads from the strict Phase 46 reference
  `57788` to `43403`, about a 25% read-count reduction, and improved decode
  from `2.24` to `2.75 tok/s`, about a 23% speedup.
- Reaching `5 tok/s` for 83 eval tokens would require reducing eval time from
  `30145 ms` to about `16600 ms`, a further ~45% reduction from the current
  best 8GB result.
- The remaining `43403` SSD reads are still too many. Since 7GB resident tier
  does not fit and 6GB same-prompt residual placement still only reaches
  `24.1%` RAM hit, profile-only gains under an 8GB cap do not currently look
  capable of closing the gap.
- This result is already same-prompt/profile-favorable. Broad prompts should
  be expected to do worse unless the profile generator becomes prompt-adaptive
  or the runtime changes how many residual expert rows need SSD service.

Current classification:

- `Runtime-limited` plus `Memory-limited`.
- RAM tier/profile work helps and now slightly beats the previous non-strict
  `~2.65 tok/s` line, but the fixed-prompt upper-bound result is not near the
  `4-5 tok/s` region. It is not yet worth claiming or prioritizing broad-prompt
  `5 tok/s` profile construction without another runtime mechanism that cuts
  residual SSD reads much more aggressively.

Recommended next experiments if continuing:

- Build a task-local SSD-miss trace for the current 6GB cache shape before any
  host-prefetch retest; the older Phase 23 trace targets a different
  graph-reserve/9GB shape.
- Use n8 gate only for host-prefetch: accept it only if it preserves output and
  beats `1.87 eval tok/s` while lowering reads.
- If host-prefetch still loses decode speed, the next meaningful work is not
  profile tuning but runtime scheduling: larger effective batch windows,
  fewer synchronized copy waits, better near-future miss targeting, or reducing
  expert materialization per token.
