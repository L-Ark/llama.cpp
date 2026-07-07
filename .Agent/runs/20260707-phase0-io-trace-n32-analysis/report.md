# GP66 Phase 0 IO Trace Profile

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`
Commit tested: `5c29b1a82`
Plan: `.Agent/plans/kimi-next-expert-transfer-optimization-plan.md`

## Goal

Run the Phase 0 trace requested by the next optimization plan:

- n32 cold start;
- 16 GB host RAM cgroup;
- correct GP4 alias/env;
- output correctness check on `Please introduce France in a short paragraph.`;
- decompose expert-pack IO batch size, wait behavior, locality, and staging granularity.

This is a profiling run, not a SOTA run. Extra CSV tracing adds overhead.

## Reproduction

Remote repo:

```text
/root/lfz/tmp/kimi-stage2m-align
```

Remote run root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260707-phase0-io-trace-n32/dev_france_regression
```

Command wrapper:

```bash
.Agent/run-tools/kimi_phase0_io_trace_remote.sh dev_france_regression
```

Additional trace env over the normal GP4/SOTA env:

```text
GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv
GGML_MOE_IO_READ_TRACE_OUT=$RUN/io-read-trace.csv
GGML_MOE_IO_LOCALITY_PROFILE_OUT=$RUN/io-locality-profile.csv
GGML_MOE_STAGE_GRANULARITY_PROFILE=1
```

Analysis command:

```bash
python3 .Agent/run-tools/kimi_phase0_analyze_io_trace.py \
  /root/lfz/runs/vendor-kimi-token-rate/20260707-phase0-io-trace-n32/dev_france_regression
```

## Result

Quality passed. Output begins:

```text
France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous
```

Timing with trace overhead:

```text
prompt_eval_ms=87319.21 prompt_tokens=17
decode_ms=18641.99 decode_runs=31
ms_per_token=601.35
token_rate=1.66
total_ms=105982.99 total_tokens=48
```

Trace overhead makes this slower than the normal n32 baseline, so this number must not be used as SOTA.

## IO Batch Summary

Files:

```text
io-batch-profile.csv    5361 rows
io-wait-trace.csv      18532 rows
io-read-trace.csv      23502 rows
io-locality-profile.csv 5361 rows
```

Key batch stats:

```text
batch.read_jobs: avg=4.384 p50=4 p90=7 p99=8 max=8
batch.initial_submit_jobs: avg=4.384 p50=4 p90=7 p99=8 max=8
batch.submit_calls: avg=1.000 max=1
batch.wait_calls: avg=3.457 p50=3 p90=5 p99=7 max=8
batch.inflight_avg: avg=2.876 p50=3 p90=4 p99=5.2 max=8
batch.wait_ms: sum=12043.682 avg=2.247 p50=2.151 p90=3.064 p99=4.487 max=12.360
batch.wall_ms: sum=14331.415 avg=2.673 p50=2.410 p90=3.404 p99=5.009 max=150.657
```

Read job histogram:

```text
read_jobs_hist=1:176,2:586,3:957,4:1231,5:1069,6:673,7:377,8:292
```

Interpretation:

- The io_uring depth is 8, but the runtime usually has fewer than 8 real read jobs available.
- `initial_submit_jobs == read_jobs` and `submit_calls == 1`, so most batches have no refill opportunity.
- Increasing `GGML_MOE_IO_DEPTH` alone cannot help much because most batches do not have enough independent reads to fill depth 8.

## Wait Breakdown

Wait by in-flight depth:

```text
wait_inflight_before_hist=1:3736,2:3957,3:3901,4:3137,5:1955,6:1047,7:507,8:292
wait_inflight_before_ms=1:1105.933,2:1809.727,3:2369.772,4:2472.421,5:1873.443,6:1160.528,7:664.559,8:587.299
```

Wait position:

```text
wait.first_cqe: sum=7817.067 avg=1.458 p50=1.404 p90=1.875 p99=2.501 max=9.173
wait.later_cqe: sum=2703.849 avg=0.328 p50=0.262 p90=0.687 p99=1.145 max=2.479
wait.tail_cqe:  sum=1522.767 avg=0.310 p50=0.250 p90=0.660 p99=1.119 max=3.140
```

Interpretation:

- Most exposed IO wait is waiting for the first completion of each small batch.
- Later completions are much cheaper.
- This supports a cross-op scheduling/prefetch design: if the next useful batch can already be in flight, first-CQE wait is the time most likely to shrink.

## Locality

Locality stats:

```text
locality.read_bytes: avg=24.46 MB p50=23.51 MB p90=37.85 MB p99=54.59 MB max=62.39 MB
locality.span_bytes: avg=586.99 MB p50=597.18 MB p90=921.63 MB p99=1302.40 MB max=2314.98 MB
locality.gap_bytes: avg=562.53 MB p50=573.67 MB p90=889.41 MB p99=1263.40 MB max=2296.05 MB
locality.adjacent_pairs: avg=0.098 p50=0 p90=0 p99=1 max=2
```

Interpretation:

- The selected experts within a tensor are usually far apart in the pack.
- Sorting by offset is enabled, but it cannot fix sparse physical layout.
- Co-occurrence-aware pack layout can plausibly improve locality, but by itself it will not increase the number of known future jobs.

## By Tensor Kind

```text
kind.down: batches=1649 read_jobs=7194 wait_ms=3601.815 wall_ms=4391.124
kind.gate: batches=1856 read_jobs=8154 wait_ms=4270.306 wall_ms=5039.276
kind.up:   batches=1856 read_jobs=8153 wait_ms=4171.562 wall_ms=4901.014
```

up/gate still account for the largest exposed demand wait. down overlap helps but does not eliminate down movement.

## Decision

The next implementation target should not be larger `GGML_MOE_IO_DEPTH` or 2 MiB staging alignment.

The profile points to two higher-priority directions:

1. **Global expert transfer scheduler**
   - Goal: reduce repeated first-CQE waits by keeping useful future work in flight across op/layer boundaries.
   - Required: demand priority, prefetch priority, deduplication, stale cancellation, and demand stealing of in-flight prefetch.
   - Must start default-off and report useful hits, wasted bytes, duplicate suppression, and demand wait reduction.

2. **Co-occurrence-aware pack layout**
   - Goal: reduce huge physical gaps among selected experts.
   - Must be generated from dev routes only and validated on held-out routes.
   - Likely needs read coalescing or at least locality replay to prove it can affect real wall time.

Immediate next step:

- Implement only the scheduler observability/control plane first:
  - task key;
  - queued/in-flight/ready state;
  - demand dedup/steal path;
  - counters;
  - no speculative reads initially.

This keeps behavior close to current SOTA while creating the mechanism needed for later safe prefetch.
