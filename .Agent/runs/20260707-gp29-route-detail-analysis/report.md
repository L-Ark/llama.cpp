# GP29 route-detail same-layer stability analysis

Timestamp: `2026-07-07T06:45:00+0800`

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Commit under test: `b95ef6df95b3fa317de7993c45656c9e6e7d5db2`

Status: dev-only feasibility result; not SOTA.

## Purpose

GP27 rejected coarse recent-window route prediction because coverage was too low
and false-prefetch bytes were too high. GP28 added exact, env-gated
`GGML_MOE_ROUTE_DETAIL_OUT` instrumentation with token/layer/kind fields. GP29
uses that trace to test a tighter predictor:

- for each decode `(kind, layer, tensor)` group, predict the next call's expert
  set from the previous `1`, `2`, or `4` calls in the same group;
- measure byte coverage, false-prefetch byte ratio, net byte multiplier,
  Jaccard overlap, and exact-repeat rate;
- implement no runtime prefetch unless the predictor passes the gate.

The analysis uses a dev prompt only. Held-out test prompts were not used.

## Reproduction

Remote worktree:

```text
/root/lfz/tmp/gp29-route-detail-run
```

Run directory:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260707-gp29-route-detail-dev-france-n32-batch
```

Important build note:

- A first diagnostic run was invalid because the CMake build did not enable
  `GGML_CUDA_MOE_STREAM_BATCH`; it fell back to the stub path and produced no
  route-detail trace.
- The valid run was rebuilt with `-DGGML_CUDA_MOE_STREAM_BATCH=ON`.

Valid command environment is preserved in:

- `command.txt`
- `env.txt`
- `git.txt`

Analyzer command:

```bash
python3 .Agent/run-tools/kimi_route_detail_stability.py \
  --detail-csv .Agent/runs/20260707-gp29-route-detail-analysis/route-detail.csv \
  --out-json .Agent/runs/20260707-gp29-route-detail-analysis/route-stability.json \
  --out-csv .Agent/runs/20260707-gp29-route-detail-analysis/route-stability-by-layer.csv
```

## Runtime run metrics

Prompt:

```text
Please introduce France in a short paragraph.
```

Key metrics:

- `N=32`
- quality: `pass`
- output: `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `ttft_ms=73885.36`
- `decode_ms=23975.93`
- `decode_runs=31`
- `token_rate=1.29`
- `memory.max=15899996160`
- `memory.swap.max=0`
- `memory.peak=15899996160`
- `memory.current.final=15071965184`
- route detail rows: `42928`
- expert pack: `iouring_bytes=126391910400`
- expert pack wait: `iouring_wait_us=20452709`
- iouring inflight: average `3.33`, max `8`
- iouring batch histogram: `1:176`, `2-4:2679`, `5-8:2323`,
  `9-16:0`, `17-32:0`, `gt32:0`
- VRAM cache total hit rate: `53.9%`
- down hit rate: `73.4%`
- up/gate hit rate: `45.2%`

This run is instrumentation-heavy and n32-only, so it is diagnostic. It does
not replace the current n96 SOTA.

## Predictor gate

Runtime predictor implementation requires at least one setting with:

- byte coverage >= `0.60`
- false-prefetch ratio <= `0.25`
- net byte multiplier <= `1.25`

## Results

No setting passed.

Aggregate results:

| setting | byte coverage | false-prefetch ratio | net byte multiplier | avg Jaccard | transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth1_all | 0.356 | 0.644 | 1.644 | 0.237 | 5193 |
| depth1_down | 0.359 | 0.641 | 1.641 | 0.241 | 1591 |
| depth1_gate | 0.357 | 0.643 | 1.643 | 0.235 | 1801 |
| depth1_up | 0.352 | 0.648 | 1.648 | 0.235 | 1801 |
| depth2_all | 0.469 | 1.158 | 2.158 | 0.232 | 5193 |
| depth2_down | 0.470 | 1.154 | 2.154 | 0.235 | 1591 |
| depth2_gate | 0.470 | 1.157 | 2.157 | 0.231 | 1801 |
| depth2_up | 0.466 | 1.165 | 2.165 | 0.231 | 1801 |
| depth4_all | 0.548 | 2.003 | 3.003 | 0.195 | 5193 |
| depth4_down | 0.550 | 1.998 | 2.998 | 0.196 | 1591 |
| depth4_gate | 0.549 | 2.000 | 3.000 | 0.194 | 1801 |
| depth4_up | 0.546 | 2.014 | 3.014 | 0.194 | 1801 |

Best by byte coverage:

- `depth4_down`
- byte coverage: `0.5495`
- false-prefetch ratio: `1.9977`
- net byte multiplier: `2.9977`

## Decision

Reject same-layer previous-call route prediction as a runtime prefetch path.

Reason:

- depth 1 has low byte coverage and still adds about `64%` false bytes;
- depth 4 approaches only `55%` byte coverage while tripling total bytes;
- exact-repeat rate is `0.0`, so active expert sets are not repeating exactly
  at the granularity needed for cheap prefetch;
- implementing this predictor would likely increase IO and TTFT risk without
  reliably reducing critical-path wait.

Next optimization should not be another naive recency predictor. The useful
paths are byte reduction, a real compatible draft/predictor model, or a much
tighter route classifier with bounded candidate sets and a measured acceptance
gate before runtime changes.
