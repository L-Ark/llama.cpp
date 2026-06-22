# Expert Predictor Prefetch Feasibility Plan

## Goal

Quickly verify whether future MoE expert usage is predictable enough to improve
GLM eval token rate without changing model outputs.

The core idea is to predict near-future expert accesses from recent routing
history, then use that prediction to improve VRAM cache retention and prefetch.
This plan is deliberately staged so we can reject the idea early if there is no
offline signal.

## Constraints

- All work must stay under `/home/wici/lfz`.
- Do not modify or delete files outside the working directory.
- Do not stop other users' GPU or RAM-consuming processes. If resources are
  occupied, wait for them to finish naturally before running experiments.
- Do not use same-prompt oracle profiles as the main result.
- Do not change GLM output semantics. Predictor output may only affect cache,
  preload, or prefetch scheduling.
- Keep train/tuning prompts separate from test prompts.
- Preserve the strict RAM setting when comparing with the current non-oracle
  baseline unless explicitly running a separate ablation.

## Current Baselines

Use these as reference points for this task:

- Current non-oracle interactive result on this machine: about `1.06 eval tok/s`
  with TTFT about `10.6s`.
- Current same-prompt oracle/profile result: `2.24 eval tok/s`, used only as an
  upper-bound reference for routing/cache preparedness, not as a valid
  generalized result.

## Hypothesis

Future expert usage has enough short-range structure that a lightweight
predictor can improve cache hit rate and reduce SSD reads on the eval critical
path.

If true, the best early signal should appear in offline route-trace replay:
simple predictors should improve simulated VRAM hit rate and reduce simulated
misses on held-out prompts. If simple predictors show no signal, a neural
predictor is unlikely to be worth implementing immediately.

## Directory Layout

Use this task directory:

```text
/home/wici/lfz/ik_llama/.Agent/plans/expert-predictor-prefetch-feasibility/
  plan.md
  traces/
    train/
    test/
  replay/
  results/
  notes/
```

## Phase 1: Collect Non-Oracle Route Traces

Collect route traces from non-oracle prompts.

Prompt split:

- `train_prompts`: 20-50 short prompts for building predictors.
- `test_prompts`: 10-20 held-out prompts for final feasibility checks.

Record at least:

- prompt id
- token position
- layer id
- selected expert ids
- router score or top-k score if available
- expert row or offset if available
- VRAM hit or miss if available
- SSD read event if available
- token latency or layer latency if available

Required output:

- `traces/train/*.route.csv`
- `traces/test/*.route.csv`
- a short note describing the exact command/config used to generate traces

Success criteria:

- Complete traces can be collected without oracle profiles.
- The trace format is sufficient to reconstruct the expert access sequence.

## Phase 2: Offline Predictor Baselines

Start with simple non-neural predictors.

Predictors:

1. `last-token-repeat`
   - For each layer, predict that the next token will reuse the previous token's
     experts from the same layer.

2. `recent-hot`
   - For each layer, keep expert counts from the last `W` tokens.
   - Test `W = 4, 8, 16`.
   - Predict top `M` experts.

3. `layer-transition`
   - Use early-layer expert ids from the current token to predict later-layer
     expert ids for the same token.
   - This is useful only if the prediction arrives early enough to overlap I/O.

4. `global-hot-per-layer`
   - For each layer, use the most frequent experts from train prompts.
   - This is the simplest static non-oracle baseline.

5. `oracle-upper-bound`
   - Use true future accesses from the held-out trace.
   - This is only a ceiling for comparison and must not be reported as a
     generalized result.

Evaluate lookahead windows:

- `lookahead = 1 token`
- `lookahead = 2 tokens`
- `lookahead = 4 tokens`

Evaluate prefetch widths:

- `M = 2, 4, 8, 16`

Metrics:

- `top_m_recall`: fraction of true future experts covered by predictions
- `precision`: fraction of predicted experts actually used
- `wasted_prefetch`: predicted experts not used in the lookahead window
- `miss_reduction_sim`: simulated reduction in SSD/VRAM misses
- `prediction_rows`: predicted expert rows, if row-level data is available

Decision gate:

- If held-out `top_m_recall` is below `20%-30%` for practical `M`, stop before
  online prefetch work.
- If simple predictors reach about `40%+` recall with manageable waste, continue
  to cache replay.

## Phase 3: Cache Replay Simulator

Replay held-out route traces and compare cache policies.

Policies:

- current baseline policy
- `global-hot-per-layer`
- `last-token-repeat`
- `recent-hot`
- `layer-transition`
- `oracle-upper-bound`

Budgets:

- current effective VRAM cache budget
- `8 GiB`
- `12 GiB`
- `14 GiB`
- `20 GiB` as an exploratory non-strict ablation only

Prefetch caps:

- per-token cap: `M = 4, 8, 16`
- inflight cap: `16, 32`

Replay metrics:

- baseline simulated VRAM hit
- predicted simulated VRAM hit
- oracle simulated VRAM hit
- simulated SSD miss count
- extra reads caused by wrong predictions
- wasted cache evictions
- estimated impact on eval token rate

Decision gate:

- Continue only if held-out prompts show at least one of:
  - `+15%-20% absolute` simulated VRAM hit improvement
  - `25%+` simulated miss reduction
- Stop or redesign if predicted prefetch evicts useful hot experts too often.

## Phase 4: Online Shadow Mode

Integrate the best lightweight predictor in shadow mode only.

Shadow mode must:

- compute predictions during real inference
- not issue prefetch reads
- not change cache state
- log whether predictions would have covered future misses
- log when the prediction became available

Timing metrics:

- prediction compute cost
- lead time before the true expert miss
- estimated read/copy overlap window
- whether the lead time is enough for SSD read plus H2D copy

Decision gate:

- Prediction overhead should be below `1%-3%` of token latency.
- Useful predictions must arrive early enough to overlap I/O.
- If predictions arrive too late, revise the feature source before implementing
  real prefetch.

## Phase 5: Limited Online Prefetch

Only after offline replay and shadow mode pass, enable conservative prefetch.

Implementation constraints:

- Use the main io_uring scheduler where possible.
- Do not add a competing direct-read background path unless an experiment
  explicitly measures the contention.
- Assign prefetch lower priority than demand reads.
- Stop prefetch automatically when the demand read queue is backlogged.
- Do not evict pinned or recently proven hot experts for low-confidence
  prefetches.
- Keep a strict per-token and global inflight prefetch budget.

Initial experiment matrix:

```text
lookahead = 1
recent_window = 8
prefetch_top_m = 4, 8
max_prefetch_inflight = 16, 32
```

Compare:

- baseline non-oracle
- shadow predictor only
- online prefetch top-4
- online prefetch top-8
- same-prompt oracle/profile only as an upper-bound reference

Runtime metrics:

- eval tok/s
- prompt tok/s
- TTFT
- total_ms
- VRAM hit rate
- io_uring reads
- direct reads
- read failures
- read queue wait
- output equivalence versus baseline

Success criteria:

- Output remains unchanged.
- TTFT does not regress materially.
- eval tok/s improves by at least `10%-15%` over the non-oracle baseline.
- A result around `1.2-1.4 eval tok/s` would validate the direction.
- A result around `1.6+ eval tok/s` would justify building a neural predictor.

## Phase 6: Tiny Neural Predictor

Do not start here. Only do this if phases 2-5 show clear signal.

Minimal model:

- input: recent `N` tokens' per-layer expert ids, layer id, and position bucket
- optional input: router scores or compact hidden-state features if cheap
- output: multi-label future expert top-M prediction
- architecture: embedding plus small MLP first; avoid a Transformer initially
- loss: multi-label classification over experts per layer

Deployment requirement:

- Predictor compute must be much cheaper than the SSD miss latency it avoids.
- Predictor must be usable in the critical path without reducing eval tok/s.

## Fastest Feasibility Path

The shortest useful path is:

1. Collect train/test non-oracle route traces.
2. Run offline simple predictors on held-out traces.
3. Run cache replay to estimate VRAM hit and miss reduction.
4. Only if replay is positive, add online shadow mode.
5. Only if shadow timing is positive, add conservative online prefetch.

This sequence should quickly answer whether expert prediction is worth pursuing
without committing to a complex neural predictor too early.

## Progress Log

- 2026-06-15: Started implementing the plan. Current task scope is the fastest
  feasibility path through Phase 1-3: collect strict non-oracle train/test route
  traces, run offline simple predictors on held-out prompts, and run a cache
  replay estimate. Online shadow mode and online prefetch will only be started
  if the offline decision gates pass.
- 2026-06-15: Added task-local prompt suite and scripts:
  - `prompts.json`;
  - `collect_traces.py`;
  - `offline_predictors.py`.
  These scripts stay under the task directory and do not change runtime code.
- 2026-06-15: Collected `20` train and `10` held-out test traces using strict
  non-oracle config: `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`, top8
  profile, `VRAM cache=12288 MiB`, `backend=iouring`, `predict_tokens=8`.
  All 30 traces completed with `read_failures=0`; every trace has `12624` row
  events (`7` eval-token blocks of `1800` row events). Aggregate collection
  metrics: average `eval tok/s=1.0623`, average `VRAM hit=9.18%`,
  total `io_uring_reads=343983`.
- 2026-06-15: Ran offline predictor metrics and cache replay. Full results are
  in `results/offline_predictors.json`; condensed results are in
  `results/summary.md`.
  - Best practical held-out recall at lookahead 1:
    `last-token-repeat, M=8`: `46.81%` recall and precision; `recent-hot-w8,
    M=8`: `39.03%` recall and precision.
  - Wider recent-hot `M=16` reaches `51-52%` recall but with low precision
    (`~30%`) and high waste.
  - Exact-signature `layer-transition` produced no hits on this small train
    set.
  - Current-budget (`12288 MiB`) best non-oracle replay is only `8.35%` miss
    reduction, far below the `25%+` gate, and requires `42385` extra prefetch
    reads.
  - Best non-oracle replay across tested budgets is `14336 MiB`
    `last-token-repeat, M=4`: `+6.69 pp` hit rate and `8.67%` miss reduction,
    still below the `+15%-20 pp` / `25%+` gate.
- 2026-06-15: Decision: stop before Phase 4 online shadow mode and Phase 5
  online prefetch. The offline signal is not strong enough; online prefetch
  would likely add I/O and eviction pressure without improving eval token rate.
