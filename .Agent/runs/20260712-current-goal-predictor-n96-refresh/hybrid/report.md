# Kimi Hybrid Future Predictor Admission

- input_glob: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-slab-iotrace-n96-074132/dev_*/route-trace.csv`
- prompts: `2`
- horizons: `1,2,3`
- budgets: `8,16,32`
- recent window: `4`
- wait reference: `504.616 ms/token`

This is a dev-only offline screen. It does not use held-out/test prompts and makes no SOTA claim.

## Parse Summary

| prompt | rows | decode tokens | prompt-like | incomplete |
|---|---:|---:|---:|---:|
| `dev_france_regression` | 146496 | 85 | 1 | 0 |
| `dev_intelligence_general` | 155232 | 95 | 1 | 0 |

## Admission Gate

- Required: `>=65%` all-role byte recall, `<=1.35x` predicted/actual bytes, and `>=40%` full-step coverage.
- Passing rows: `0`.

## Top All-Role Predictors

| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | useful GiB/token | waste GiB/token | linear wait cover ms/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_recent | 3 | 32 | 0.6051 | 0.1513 | 3.48% | 4.00x | 84.029 | 12.684 | 71.345 | 305.3 |
| hybrid_recent | 2 | 32 | 0.6023 | 0.1506 | 3.46% | 4.00x | 85.547 | 12.851 | 72.696 | 303.9 |
| hybrid_recent | 1 | 32 | 0.6017 | 0.1504 | 3.31% | 4.00x | 87.066 | 13.067 | 73.999 | 303.6 |
| hybrid_balanced | 3 | 32 | 0.5999 | 0.1500 | 3.36% | 4.00x | 84.029 | 12.574 | 71.455 | 302.7 |
| hybrid_prev | 3 | 32 | 0.5992 | 0.1498 | 3.35% | 4.00x | 84.029 | 12.559 | 71.470 | 302.4 |
| hybrid_balanced | 2 | 32 | 0.5976 | 0.1494 | 3.26% | 4.00x | 85.547 | 12.749 | 72.798 | 301.5 |
| hybrid_balanced | 1 | 32 | 0.5971 | 0.1493 | 3.10% | 4.00x | 87.066 | 12.967 | 74.098 | 301.3 |
| hybrid_prev | 2 | 32 | 0.5968 | 0.1492 | 3.23% | 4.00x | 85.547 | 12.733 | 72.814 | 301.2 |
| hybrid_prev | 1 | 32 | 0.5964 | 0.1491 | 3.07% | 4.00x | 87.066 | 12.952 | 74.114 | 300.9 |
| hybrid_cooc | 3 | 32 | 0.5729 | 0.1432 | 2.17% | 4.00x | 84.029 | 12.005 | 72.024 | 289.1 |
| hybrid_cooc | 2 | 32 | 0.5707 | 0.1427 | 2.26% | 4.00x | 85.547 | 12.174 | 73.373 | 288.0 |
| hybrid_cooc | 1 | 32 | 0.5698 | 0.1424 | 2.26% | 4.00x | 87.066 | 12.373 | 74.693 | 287.5 |
| recent_lfu | 3 | 32 | 0.5313 | 0.2034 | 1.60% | 2.62x | 54.961 | 11.134 | 43.827 | 268.1 |
| recent_lfu | 2 | 32 | 0.5291 | 0.2020 | 1.59% | 2.62x | 56.113 | 11.287 | 44.827 | 267.0 |
| recent_lfu | 1 | 32 | 0.5279 | 0.2011 | 1.58% | 2.63x | 57.224 | 11.461 | 45.764 | 266.4 |
| hybrid_recent | 3 | 16 | 0.4898 | 0.2449 | 0.58% | 2.00x | 42.015 | 10.260 | 31.754 | 247.2 |
| hybrid_recent | 2 | 16 | 0.4881 | 0.2440 | 0.65% | 2.00x | 42.774 | 10.405 | 32.368 | 246.3 |
| hybrid_recent | 1 | 16 | 0.4867 | 0.2434 | 0.60% | 2.00x | 43.533 | 10.561 | 32.971 | 245.6 |
| hybrid_balanced | 3 | 16 | 0.4856 | 0.2428 | 0.49% | 2.00x | 42.015 | 10.171 | 31.843 | 245.0 |
| hybrid_prev | 3 | 16 | 0.4848 | 0.2424 | 0.48% | 2.00x | 42.015 | 10.155 | 31.860 | 244.6 |

## Top Up/Gate Bundle Predictors

| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | waste GiB/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_recent | 3 | 32 | 0.6051 | 0.1513 | 3.48% | 4.00x | 50.191 | 42.621 |
| hybrid_recent | 2 | 32 | 0.6023 | 0.1506 | 3.46% | 4.00x | 51.164 | 43.485 |
| hybrid_recent | 1 | 32 | 0.6017 | 0.1504 | 3.31% | 4.00x | 52.136 | 44.320 |
| hybrid_balanced | 3 | 32 | 0.5999 | 0.1500 | 3.36% | 4.00x | 50.191 | 42.688 |
| hybrid_prev | 3 | 32 | 0.5992 | 0.1498 | 3.35% | 4.00x | 50.191 | 42.696 |
| hybrid_balanced | 2 | 32 | 0.5976 | 0.1494 | 3.26% | 4.00x | 51.164 | 43.546 |
| hybrid_balanced | 1 | 32 | 0.5971 | 0.1493 | 3.10% | 4.00x | 52.136 | 44.381 |
| hybrid_prev | 2 | 32 | 0.5968 | 0.1492 | 3.23% | 4.00x | 51.164 | 43.556 |
| hybrid_prev | 1 | 32 | 0.5964 | 0.1491 | 3.07% | 4.00x | 52.136 | 44.390 |
| hybrid_cooc | 3 | 32 | 0.5729 | 0.1432 | 2.17% | 4.00x | 50.191 | 43.027 |

Decision: no hybrid route-history predictor passes the offline admission gate.

Interpretation:

- Same-prompt recency/history improves the best all-role recall over plain cooc, but not enough to make speculative future-layer prefetch demand-safe.
- A runtime prefetch prototype should not be implemented from route traces alone.
- The next predictor attempt needs a stronger signal such as router logits, hidden-state features, or a draft/router model; otherwise the optimization should return to lower-byte expert representation or storage-format changes.

## Artifacts

- `parse_summary.csv`
- `fold_metrics.csv`
- `aggregate_metrics.csv`

