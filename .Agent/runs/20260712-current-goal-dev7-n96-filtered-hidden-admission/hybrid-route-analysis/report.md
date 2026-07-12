# Kimi Hybrid Future Predictor Admission

- input_glob: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-n96-filtered-hidden-corpus/dev_*/route-trace.csv`
- prompts: `7`
- horizons: `1,2,3`
- budgets: `8,16,32`
- recent window: `4`
- wait reference: `371.100 ms/token`

This is a dev-only offline screen. It does not use held-out/test prompts and makes no SOTA claim.

## Parse Summary

| prompt | rows | decode tokens | prompt-like | incomplete |
|---|---:|---:|---:|---:|
| `dev_france_regression` | 146496 | 85 | 1 | 0 |
| `dev_japan_factual` | 136416 | 78 | 1 | 0 |
| `dev_linear_equation` | 82968 | 34 | 1 | 0 |
| `dev_mixed_summary` | 104568 | 49 | 1 | 0 |
| `dev_photosynthesis_factual` | 155208 | 94 | 1 | 0 |
| `dev_python_reverse` | 162312 | 95 | 1 | 0 |
| `dev_zh_france` | 91824 | 49 | 1 | 0 |

## Admission Gate

- Required: `>=65%` all-role byte recall, `<=1.35x` predicted/actual bytes, and `>=40%` full-step coverage.
- Passing rows: `0`.

## Top All-Role Predictors

| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | useful GiB/token | waste GiB/token | linear wait cover ms/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_recent | 3 | 32 | 0.5959 | 0.1490 | 3.57% | 4.00x | 64.556 | 9.600 | 54.956 | 221.1 |
| hybrid_balanced | 3 | 32 | 0.5942 | 0.1486 | 3.60% | 4.00x | 64.556 | 9.574 | 54.982 | 220.5 |
| hybrid_recent | 2 | 32 | 0.5939 | 0.1485 | 3.53% | 4.00x | 65.722 | 9.738 | 55.984 | 220.4 |
| hybrid_recent | 1 | 32 | 0.5937 | 0.1484 | 3.54% | 4.00x | 66.888 | 9.911 | 56.977 | 220.3 |
| hybrid_prev | 3 | 32 | 0.5929 | 0.1482 | 3.51% | 4.00x | 64.556 | 9.553 | 55.003 | 220.0 |
| hybrid_balanced | 1 | 32 | 0.5923 | 0.1481 | 3.58% | 4.00x | 66.888 | 9.887 | 57.001 | 219.8 |
| hybrid_balanced | 2 | 32 | 0.5920 | 0.1480 | 3.62% | 4.00x | 65.722 | 9.709 | 56.013 | 219.7 |
| hybrid_prev | 2 | 32 | 0.5910 | 0.1478 | 3.54% | 4.00x | 65.722 | 9.693 | 56.029 | 219.3 |
| hybrid_prev | 1 | 32 | 0.5909 | 0.1477 | 3.49% | 4.00x | 66.888 | 9.864 | 57.024 | 219.3 |
| hybrid_cooc | 3 | 32 | 0.5875 | 0.1469 | 3.97% | 4.00x | 64.556 | 9.467 | 55.088 | 218.0 |
| hybrid_cooc | 2 | 32 | 0.5853 | 0.1463 | 3.88% | 4.00x | 65.722 | 9.601 | 56.121 | 217.2 |
| hybrid_cooc | 1 | 32 | 0.5852 | 0.1463 | 4.02% | 4.00x | 66.888 | 9.771 | 57.117 | 217.2 |
| recent_lfu | 3 | 32 | 0.4993 | 0.1879 | 1.04% | 2.66x | 42.937 | 8.041 | 34.896 | 185.3 |
| recent_lfu | 2 | 32 | 0.4972 | 0.1867 | 1.03% | 2.67x | 43.810 | 8.152 | 35.658 | 184.5 |
| recent_lfu | 1 | 32 | 0.4970 | 0.1865 | 1.02% | 2.67x | 44.610 | 8.292 | 36.318 | 184.4 |
| hybrid_balanced | 3 | 16 | 0.4832 | 0.2416 | 0.67% | 2.00x | 32.278 | 7.782 | 24.496 | 179.3 |
| hybrid_prev | 3 | 16 | 0.4824 | 0.2412 | 0.65% | 2.00x | 32.278 | 7.769 | 24.509 | 179.0 |
| hybrid_balanced | 2 | 16 | 0.4818 | 0.2409 | 0.59% | 2.00x | 32.861 | 7.898 | 24.963 | 178.8 |
| hybrid_balanced | 1 | 16 | 0.4813 | 0.2407 | 0.57% | 2.00x | 33.444 | 8.032 | 25.413 | 178.6 |
| hybrid_prev | 2 | 16 | 0.4810 | 0.2405 | 0.58% | 2.00x | 32.861 | 7.885 | 24.976 | 178.5 |

## Top Up/Gate Bundle Predictors

| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | waste GiB/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_recent | 3 | 32 | 0.5959 | 0.1490 | 3.57% | 4.00x | 38.560 | 32.832 |
| hybrid_balanced | 3 | 32 | 0.5942 | 0.1486 | 3.60% | 4.00x | 38.560 | 32.847 |
| hybrid_recent | 2 | 32 | 0.5939 | 0.1485 | 3.53% | 4.00x | 39.307 | 33.489 |
| hybrid_recent | 1 | 32 | 0.5937 | 0.1484 | 3.54% | 4.00x | 40.054 | 34.127 |
| hybrid_prev | 3 | 32 | 0.5929 | 0.1482 | 3.51% | 4.00x | 38.560 | 32.860 |
| hybrid_balanced | 1 | 32 | 0.5923 | 0.1481 | 3.58% | 4.00x | 40.054 | 34.141 |
| hybrid_balanced | 2 | 32 | 0.5920 | 0.1480 | 3.62% | 4.00x | 39.307 | 33.507 |
| hybrid_prev | 2 | 32 | 0.5910 | 0.1478 | 3.54% | 4.00x | 39.307 | 33.516 |
| hybrid_prev | 1 | 32 | 0.5909 | 0.1477 | 3.49% | 4.00x | 40.054 | 34.155 |
| hybrid_cooc | 3 | 32 | 0.5875 | 0.1469 | 3.97% | 4.00x | 38.560 | 32.912 |

Decision: no hybrid route-history predictor passes the offline admission gate.

Interpretation:

- Same-prompt recency/history improves the best all-role recall over plain cooc, but not enough to make speculative future-layer prefetch demand-safe.
- A runtime prefetch prototype should not be implemented from route traces alone.
- The next predictor attempt needs a stronger signal such as router logits, hidden-state features, or a draft/router model; otherwise the optimization should return to lower-byte expert representation or storage-format changes.

## Artifacts

- `parse_summary.csv`
- `fold_metrics.csv`
- `aggregate_metrics.csv`

