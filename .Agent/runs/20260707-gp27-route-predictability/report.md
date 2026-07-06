# GP27 route/expert demand predictability report

Timestamp: `2026-07-07T05:55:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Plan commit: `5fc132888`

## Scope

This is a dev-only feasibility analysis. It does not use held-out test prompts
and does not claim a new SOTA.

The goal is to test whether recent expert movement events can predict upcoming
expert movement events well enough to justify a runtime prefetcher that raises
IO queue depth.

## Command

```bash
python3 .Agent/run-tools/kimi_route_window_predictability.py \
  --traces-glob '.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct/*/route-trace.csv' \
  --out-json .Agent/runs/20260707-gp27-route-predictability/window-predictability.json \
  --out-csv .Agent/runs/20260707-gp27-route-predictability/window-predictability.csv
```

## Inputs

- Dev route traces only:
  `.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct/*/route-trace.csv`.
- No held-out test traces.

## Caveat

The available `route-trace.csv` files contain ordered expert movement events,
not explicit token ids or a clean token/layer routing matrix. This analysis is
therefore a fixed-window demand-predictability estimate, not exact token-level
router accuracy.

## Gate

A runtime prototype requires at least one setting with:

- byte coverage >= `0.60`;
- false-prefetch byte ratio <= `0.25`;
- net byte multiplier <= `1.25`;
- minimum prompt byte coverage >= `0.35`.

## Results

No setting passed the gate.

Best byte coverage:

- window size: `2048` events;
- history depth: `4` windows;
- byte coverage: `0.531`;
- minimum prompt byte coverage: `0.461`;
- false-prefetch byte ratio: `2.004`;
- net byte multiplier: `3.004`.

Other high-coverage settings:

```text
w=1536 d=4 cov=0.507 false=2.165 net=3.165 min=0.432
w=1024 d=4 cov=0.466 false=2.502 net=3.502 min=0.382
w=2048 d=2 cov=0.447 false=1.132 net=2.132 min=0.371
w=1536 d=2 cov=0.424 false=1.208 net=2.208 min=0.331
w=1024 d=2 cov=0.363 false=1.418 net=2.418 min=0.253
w=2048 d=1 cov=0.344 false=0.618 net=1.618 min=0.250
```

## Decision

- Reject simple recent-window expert prefetch as the next primary runtime path.
- It does not cover enough upcoming bytes, and the settings with useful
  coverage add `1.1x-2.5x` extra false-prefetch bytes.
- Do not implement this prefetcher in the Kimi runtime.

## Next implication

Useful predictor work needs stronger information than recent movement windows:

- explicit token/layer route instrumentation;
- per-layer router-output stability measurement;
- a learned or rule-based route predictor with a small candidate set;
- or a different optimization path that reduces expert bytes instead of trying
  to prefetch many uncertain experts.
