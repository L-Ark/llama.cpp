# Kimi Hybrid Future Predictor Admission

- input_glob: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-route-n32-001009/dev_*/route-trace.csv`
- prompts: `7`
- horizons: `1,2,3`
- budgets: `8,16,32`
- recent window: `4`
- wait reference: `371.100 ms/token`

This is a dev-only offline screen. It does not use held-out/test prompts and makes no SOTA claim.

## Parse Summary

| prompt | rows | decode tokens | prompt-like | incomplete |
|---|---:|---:|---:|---:|
| `dev_france_regression` | 68736 | 31 | 1 | 0 |
| `dev_japan_factual` | 68736 | 31 | 1 | 0 |
| `dev_linear_equation` | 78648 | 31 | 1 | 0 |
| `dev_mixed_summary` | 78648 | 31 | 1 | 0 |
| `dev_photosynthesis_factual` | 64488 | 31 | 1 | 0 |
| `dev_python_reverse` | 70152 | 31 | 1 | 0 |
| `dev_zh_france` | 65904 | 31 | 1 | 0 |

## Admission Gate

- Required: `>=65%` all-role byte recall, `<=1.35x` predicted/actual bytes, and `>=40%` full-step coverage.
- Passing rows: `0`.

## Top All-Role Predictors

| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | useful GiB/token | waste GiB/token | linear wait cover ms/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_recent | 3 | 32 | 0.5883 | 0.1471 | 3.09% | 4.00x | 28.943 | 4.251 | 24.693 | 218.3 |
| hybrid_recent | 1 | 32 | 0.5880 | 0.1470 | 3.05% | 4.00x | 29.989 | 4.403 | 25.586 | 218.2 |
| hybrid_recent | 2 | 32 | 0.5874 | 0.1468 | 3.10% | 4.00x | 29.466 | 4.321 | 25.145 | 218.0 |
| hybrid_balanced | 1 | 32 | 0.5822 | 0.1456 | 3.09% | 4.00x | 29.989 | 4.360 | 25.629 | 216.1 |
| hybrid_balanced | 3 | 32 | 0.5815 | 0.1454 | 3.02% | 4.00x | 28.943 | 4.202 | 24.741 | 215.8 |
| hybrid_balanced | 2 | 32 | 0.5815 | 0.1454 | 3.11% | 4.00x | 29.466 | 4.278 | 25.189 | 215.8 |
| hybrid_prev | 1 | 32 | 0.5811 | 0.1453 | 2.99% | 4.00x | 29.989 | 4.351 | 25.638 | 215.6 |
| hybrid_prev | 3 | 32 | 0.5807 | 0.1452 | 2.93% | 4.00x | 28.943 | 4.197 | 24.747 | 215.5 |
| hybrid_prev | 2 | 32 | 0.5801 | 0.1450 | 3.07% | 4.00x | 29.466 | 4.267 | 25.199 | 215.3 |
| hybrid_cooc | 2 | 32 | 0.5666 | 0.1416 | 2.97% | 4.00x | 29.466 | 4.168 | 25.298 | 210.3 |
| hybrid_cooc | 3 | 32 | 0.5665 | 0.1416 | 2.89% | 4.00x | 28.943 | 4.093 | 24.850 | 210.2 |
| hybrid_cooc | 1 | 32 | 0.5663 | 0.1416 | 3.13% | 4.00x | 29.989 | 4.240 | 25.749 | 210.1 |
| recent_lfu | 3 | 32 | 0.4893 | 0.1912 | 1.11% | 2.56x | 18.540 | 3.535 | 15.004 | 181.6 |
| recent_lfu | 1 | 32 | 0.4878 | 0.1902 | 1.09% | 2.57x | 19.246 | 3.651 | 15.594 | 181.0 |
| recent_lfu | 2 | 32 | 0.4878 | 0.1902 | 1.10% | 2.57x | 18.907 | 3.587 | 15.319 | 181.0 |
| hybrid_balanced | 3 | 16 | 0.4776 | 0.2388 | 0.65% | 2.00x | 14.472 | 3.450 | 11.022 | 177.2 |
| hybrid_recent | 3 | 16 | 0.4771 | 0.2385 | 0.68% | 2.00x | 14.472 | 3.446 | 11.025 | 177.0 |
| hybrid_balanced | 1 | 16 | 0.4768 | 0.2384 | 0.68% | 2.00x | 14.995 | 3.569 | 11.426 | 177.0 |
| hybrid_prev | 3 | 16 | 0.4766 | 0.2383 | 0.64% | 2.00x | 14.472 | 3.443 | 11.029 | 176.9 |
| hybrid_balanced | 2 | 16 | 0.4765 | 0.2382 | 0.63% | 2.00x | 14.733 | 3.503 | 11.230 | 176.8 |

## Top Up/Gate Bundle Predictors

| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | waste GiB/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_recent | 3 | 32 | 0.5883 | 0.1471 | 3.09% | 4.00x | 17.288 | 14.753 |
| hybrid_recent | 1 | 32 | 0.5880 | 0.1470 | 3.05% | 4.00x | 17.958 | 15.326 |
| hybrid_recent | 2 | 32 | 0.5874 | 0.1468 | 3.10% | 4.00x | 17.623 | 15.043 |
| hybrid_balanced | 1 | 32 | 0.5822 | 0.1456 | 3.09% | 4.00x | 17.958 | 15.352 |
| hybrid_balanced | 3 | 32 | 0.5815 | 0.1454 | 3.02% | 4.00x | 17.288 | 14.782 |
| hybrid_balanced | 2 | 32 | 0.5815 | 0.1454 | 3.11% | 4.00x | 17.623 | 15.069 |
| hybrid_prev | 1 | 32 | 0.5811 | 0.1453 | 2.99% | 4.00x | 17.958 | 15.357 |
| hybrid_prev | 3 | 32 | 0.5807 | 0.1452 | 2.93% | 4.00x | 17.288 | 14.785 |
| hybrid_prev | 2 | 32 | 0.5801 | 0.1450 | 3.07% | 4.00x | 17.623 | 15.075 |
| hybrid_cooc | 2 | 32 | 0.5666 | 0.1416 | 2.97% | 4.00x | 17.623 | 15.135 |

Decision: no hybrid route-history predictor passes the offline admission gate.

Interpretation:

- Same-prompt recency/history improves the best all-role recall over plain cooc, but not enough to make speculative future-layer prefetch demand-safe.
- A runtime prefetch prototype should not be implemented from route traces alone.
- The next predictor attempt needs a stronger signal such as router logits, hidden-state features, or a draft/router model; otherwise the optimization should return to lower-byte expert representation or storage-format changes.

## Artifacts

- `parse_summary.csv`
- `fold_metrics.csv`
- `aggregate_metrics.csv`

## Reproduce

```bash
cd /root/lfz/llama.cpp-vendor-kimi
OUT=.Agent/runs/20260712-hybrid-future-predictor-admission
rm -rf "$OUT"
timeout 600 python3 .Agent/run-tools/kimi_future_hybrid_predictor_admission.py \
  --input-glob "/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-route-n32-001009/dev_*/route-trace.csv" \
  --out "$OUT" \
  --layers 60 \
  --max-decode-experts 8 \
  --horizons 1 2 3 \
  --budgets 8 16 32 \
  --bundles all,upgate,down \
  --recent-window 4 \
  --io-wait-ms-per-token 371.1
```
