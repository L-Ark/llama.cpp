# Exact Queue Continuity Bound

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Input root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717`

This is a dev-only offline/profiling bound. It does not change runtime behavior
and does not claim SOTA.

## Goal

After rejecting broad CPU fallback work and static pack relayout, test whether
exact non-speculative scheduling can be the next implementation path:

- no future expert prediction;
- no wrong-path overfetch;
- only better queue continuity/co-submit for already known current-layer work;
- include the specific up/gate fusion idea because up/gate expert IDs are paired
  for the same active experts.

## Inputs

- prompts: `dev_france_regression`, `dev_intelligence_general`;
- profile files: `copy-profile.csv`, `io-batch-profile.csv`,
  `io-wait-trace.csv`, `io-read-trace.csv`, `ttft-trace.csv`;
- both prompts were dev prompts, not held-out/test prompts;
- both prompts passed the lightweight quality gate in the source run.

## Queue Evidence

France:

- token rate: `1.42 tok/s`;
- decode: `21839.52 ms / 31 runs`;
- iouring: `38796 reads`, `222635671552 bytes`, `16755483 us wait`;
- iouring batch hist: `1:182`, `2-4:2579`, `5-8:2818`, `gt32:177`;
- inflight avg/max: `4.52 / 8`;
- full IO wait trace:
  - wait ms `16762.233`;
  - queue empty ratio `0.000`;
  - low inflight ratio `0.037`;
- decode-like wait trace (`read_jobs <= 8`):
  - wait ms `11882.550`;
  - queue empty ratio `0.000`;
  - low inflight ratio `0.045`;
  - next-job-done ratio `1.000`.

Intelligence:

- token rate: `1.50 tok/s`;
- decode: `20687.58 ms / 31 runs`;
- iouring: `37043 reads`, `212793262080 bytes`, `15102679 us wait`;
- iouring batch hist: `1:205`, `2-4:2478`, `5-8:2872`, `gt32:177`;
- inflight avg/max: `4.47 / 8`;
- full IO wait trace:
  - wait ms `15108.997`;
  - queue empty ratio `0.000`;
  - low inflight ratio `0.056`;
- decode-like wait trace (`read_jobs <= 8`):
  - wait ms `11123.981`;
  - queue empty ratio `0.000`;
  - low inflight ratio `0.068`;
  - next-job-done ratio `1.000`.

Interpretation:

- The queue is not frequently empty in these traces.
- The runtime already reaches `inflight_max=8`, with average inflight around
  `4.5`.
- Most decode-like wait rows are waiting for completion of already-submitted
  work rather than waiting because the scheduler had no jobs.
- Therefore a no-prediction scheduler cannot create enough new independent work
  to approach `2 tok/s`, unless it also reduces moved bytes or changes the
  storage representation.

## Up/Gate Fusion Bound

France:

- current up+gate staged bytes: `75.34 GiB`;
- current up+gate staged jobs: `15864`;
- fused jobs: `7934`;
- job reduction: `7930` (`50.0%`);
- rows with both up and gate staged: `1859/1859` (`100.0%`);
- current exposed up+gate bandwidth: `7.24 GiB/s`;
- ideal up+gate time at `10.4 GiB/s`: `7244.12 ms`;
- ideal saved decode time: `3158.96 ms`;
- ideal token rate if all saved: `1.66 tok/s`.

Intelligence:

- current up+gate staged bytes: `76.59 GiB`;
- current up+gate staged jobs: `16076`;
- fused jobs: `8040`;
- job reduction: `8036` (`50.0%`);
- rows with both up and gate staged: `1847/1847` (`100.0%`);
- current exposed up+gate bandwidth: `7.57 GiB/s`;
- ideal up+gate time at `10.4 GiB/s`: `7364.70 ms`;
- ideal saved decode time: `2746.61 ms`;
- ideal token rate if all saved: `1.73 tok/s`.

Interpretation:

- Up/gate fusion can cut job count by half, but it cannot cut semantic bytes.
- Even the optimistic upper bound remains below the `>2 tok/s` short-term
  target.
- It may still be useful as a secondary cleanup after byte reduction, but it is
  not enough as the next primary implementation.

## Exact-Byte Ceiling

Assumptions:

- movement bandwidth ceiling: `10.40 GiB/s`;
- all-hit MoE floor: `40.1 ms/token`;
- no byte reduction.

Target `2 tok/s`:

- measured mean token rate: `1.460 tok/s`;
- transfer-only mean ceiling: `1.591 tok/s`;
- floor+transfer mean ceiling: `1.495 tok/s`;
- best prompt floor+transfer ceiling: `1.527 tok/s`;
- required mean byte ratio to reach `2 tok/s`: `0.732x`.

Target `5 tok/s`:

- floor+transfer mean ceiling: `1.495 tok/s`;
- best prompt floor+transfer ceiling: `1.527 tok/s`;
- required mean byte ratio to reach `5 tok/s`: `0.254x`.

Per-prompt movement:

| prompt | moved GiB/token | floor+transfer ceiling | byte ratio for 2 tok/s | byte ratio for 5 tok/s |
|---|---:|---:|---:|---:|
| `dev_france_regression` | `6.689` | `1.464 tok/s` | `0.715x` | `0.249x` |
| `dev_intelligence_general` | `6.393` | `1.527 tok/s` | `0.748x` | `0.260x` |

## Decision

Reject exact-byte queue continuity as the next primary runtime A/B.

Reasons:

- The queue is not empty often enough for a scheduler-only fix to expose a
  large gain.
- Up/gate fusion reduces jobs but not bytes; the optimistic bound is only
  `1.66-1.73 tok/s`.
- Exact-byte scheduling at `10.4 GiB/s` still tops out around `1.50 tok/s`
  with the current `6.4-6.7 GiB/token` expert movement.
- Reaching `2 tok/s` already requires roughly `25-30%` byte reduction, even
  before targeting `5 tok/s`.
- Reaching `5 tok/s` requires roughly `74-75%` byte reduction or an equivalent
  reduction in exact expert movement/compute.

## Next Direction

The next implementation cycle should not be scheduler-only. It should choose a
candidate that reduces bytes/token while preserving quality:

1. lower-byte expert representation with current-pack output-error gates;
2. RAM/VRAM tiering only if it replaces low-value file cache and reduces endpoint
   decode time, not only hit rate;
3. after a byte-reducing path exists, revisit up/gate fusion or queue continuity
   as a secondary multiplier.

CPU/defer GPU-extension acceptance and `0` fallback rows remain mandatory
regression gates for every future run.
