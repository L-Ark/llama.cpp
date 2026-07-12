# Kimi Exact Layout And Future-Prefetch Bound

This is a dev-only offline report. It does not rewrite packs, implement
prefetch, inspect held-out prompts, or claim SOTA.

## Inputs

- Branch: `vendor/kimi-deepseek-41d205-additive`
- Previous commit: `9d6d7fb76`
- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717`
- Prompts:
  - `dev_france_regression`
  - `dev_intelligence_general`
- Baseline decode: `39422.98 ms / 62 = 1.573 tok/s`

## Pack Layout

Same-prompt static greedy-pair looked high:

| prompt | saved ms upper | bounded tok/s |
|---|---:|---:|
| `dev_france_regression` | `5175.30` | `1.86` |
| `dev_intelligence_general` | `5319.35` | `2.02` |

Leave-one-prompt-out bound rejected it:

| layout | saved ms/token | bounded tok/s |
|---|---:|---:|
| `greedy_pair` | `19.175` | `1.622` |
| `first_use` | `15.448` | `1.612` |
| `frequency` | `9.281` | `1.596` |
| `expert_id` | `4.521` | `1.584` |

Decision: do not implement static prompt-general pack relayout from these
traces. The same-prompt bound is mostly prompt-order overfitting.

## Future Prefetch

The optimistic future-window bound is large:

| prompt | wait ms/token | window2 saved | window4 saved |
|---|---:|---:|---:|
| `dev_france_regression` | `383.308` | `165.778` | `260.042` |
| `dev_intelligence_general` | `358.838` | `159.097` | `248.467` |

Route-history predictor admission rejected the actionable version:

- passing rows: `0`
- best all-role predictor: `hybrid_recent`, horizon `3`, budget `32`
- recall: `0.6217`
- precision: `0.1554`
- full-step coverage: `3.54%`
- predicted/actual bytes: `4.00x`
- waste: `24.453 GiB/token`

Decision: do not implement route-history-only future prefetch. A future
prefetch attempt needs a stronger shadow signal such as router logits,
hidden-state features, or a dedicated draft/router model.

## Reproduce

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
OUT=$ROOT/analysis/exact-layout-prefetch-bound

.Agent/run-tools/kimi_pack_layout_loo_bound.py \
  --input-root "$ROOT" \
  --out-json "$OUT/pack-layout-loo.json" \
  --out-md "$OUT/pack-layout-loo.md" \
  --roles up,gate,down \
  --max-jobs 8 \
  --max-gap-mib 1.0 \
  --baseline-decode-ms 39422.98

python3 .Agent/run-tools/kimi_future_hybrid_predictor_admission.py \
  --input-glob "$ROOT/*/route-trace.csv" \
  --out "$OUT/future-hybrid-predictor-admission" \
  --layers 60 \
  --max-decode-experts 16 \
  --horizons 1 2 3 \
  --budgets 4 8 12 16 32 \
  --bundles all,upgate,down \
  --recent-window 4 \
  --io-wait-ms-per-token 371.073
```
