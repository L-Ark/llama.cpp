# Kimi Scheduler-Only Closure

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `6d9b4fd11`

## Purpose

Decide whether the next implementation cycle should still be a scheduler-only
same-layer up/gate/down read A/B, after the small RAM-tier admission was
rejected.

This report does not run a new benchmark and does not claim SOTA. It consolidates
the current evidence to avoid repeating already rejected scheduler-only paths.

## Evidence

### Exact Queue Continuity Bound

Artifact:

- `.Agent/runs/20260712-exact-queue-continuity-bound/report.md`

Relevant results:

- prompt set: `dev_france_regression`, `dev_intelligence_general`;
- queue-empty ratio during decode-like waits: `0.000` for both prompts;
- low-inflight ratio: `0.045` and `0.068` for decode-like waits;
- `next_job_done` ratio: `1.000`;
- inflight avg/max: about `4.5 / 8`;
- current expert movement: `6.393-6.689 GiB/token`;
- exact-byte floor+transfer ceiling: about `1.50 tok/s`;
- up/gate fusion optimistic bound: `1.66-1.73 tok/s`;
- required byte ratio for `2 tok/s`: about `0.715x-0.748x`;
- required byte ratio for `5 tok/s`: about `0.249x-0.260x`.

Interpretation:

- The exposed wait is mostly waiting for work that has already been submitted,
  not waiting because the scheduler has no jobs.
- Exact-byte scheduling cannot reach the `>2 tok/s` target without reducing
  moved expert bytes.
- Up/gate fusion may reduce job count, but it does not reduce semantic bytes
  enough to be the next primary implementation.

### Standalone Gate/Up/Down Cosubmit

Artifact:

- `.Agent/plans/kimi-cpu-defer-gpu-extension-cache-plan.md`, Phase 4A.

Relevant results:

- env: `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`;
- candidate quality passed but endpoint regressed:
  - control mean `1.85 tok/s`;
  - candidate mean `1.81 tok/s`;
  - TTFT median `8537.31 ms -> 8878.44 ms`;
- no actual atexit cosubmit counter appeared;
- the hook lives on the standalone gate path, while current Kimi uses the fused
  up/gate batch path.

Decision:

- keep `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` default-off;
- do not retry it for the current Kimi SOTA path.

### Fused-Path Cosubmit Shadow

Artifact:

- `.Agent/plans/kimi-cpu-defer-gpu-extension-cache-plan.md`, Phase 4B.

Relevant results:

- env:
  `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1`;
- rows: `5583`;
- active expert rows: `44664`;
- up cache hit rate: `44.08%`;
- gate cache hit rate: `44.17%`;
- down cache hit rate: `32.10%`;
- down-overlap plannable rows: `25094 / 44664 = 56.18%`;
- pack hits were present for measured up/gate/down rows.

Interpretation:

- There are same-layer miss opportunities, but many down rows are already
  plannable by current-down overlap.
- A real fused cosubmit path would need to avoid duplicating current-down
  overlap and still cannot fix the byte/token ceiling by itself.

### Early Current-Down Overlap

Artifact:

- `.Agent/plans/kimi-cpu-defer-gpu-extension-cache-plan.md`, Phase 4C.

Relevant results:

- env: `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1`;
- two N32 dev prompts failed with CUDA OOM;
- the surviving prompt slowed down to `0.91 tok/s`;
- up/gate VRAM slots collapsed because early overlap changed memory pressure.

Decision:

- reject `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1`;
- do not run N96;
- do not enable it in reproduction scripts.

### Small Static RAM-Tier Admission

Artifact:

- `.Agent/runs/20260712-current-goal-small-ram-admission/report.md`

Relevant results:

- `512MiB` candidate:
  - route coverage `2.18%`;
  - copy io/wall coverage `0.88%`;
  - optimistic endpoint upper bound `1.604 tok/s`;
- `1024MiB` candidate:
  - route coverage `3.57%`;
  - copy io/wall coverage `1.77%`;
  - optimistic endpoint upper bound `1.680 tok/s`.

Interpretation:

- Adding small scattered RAM residency also does not expose enough endpoint
  benefit; it is not a substitute for byte reduction or a coherent layout.

## Decision

Close scheduler-only same-layer up/gate/down submission as the next primary
implementation path.

Do not run a new runtime A/B for:

- `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`;
- `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1`;
- a naive fused same-layer up/gate/down co-submit that does not reduce bytes;
- scattered static RAM tier used only to improve nominal hit rate.

Scheduler work is still allowed later as a secondary multiplier after a
byte-reducing or layout-changing path exists. At that point, up/gate fusion or
same-layer grouping may help convert reduced bytes into better endpoint latency.

## Next Implementation Direction

The next implementation cycle should prioritize one of:

1. lower-byte/v2 expert representation with output-error and quality gates;
2. pack/layout work that reduces exposed wait by making active same-tensor or
   same-layer expert reads contiguous enough to increase useful batch size;
3. RAM/VRAM cache only if it replaces measured low-value decode file cache with
   batchable layer/role slabs and has a stronger exposed-wait bound than the
   rejected scattered tiers.

The target remains unchanged:

- cold-start generalized prompts;
- host RAM `<16 GB`;
- TTFT `<= +20%`;
- France quality pass;
- all SOTA claims must be reproducible from a pushed commit with exact commands,
  prompt split, metrics, output text, RAM/VRAM data, and rollback point.
