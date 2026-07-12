# Kimi Prefix-Neighbor Predictor Admission

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit before this report: `fcefe5eb5a81b40de6504e4b3385efcd81cf684d`

## Goal

Test whether route-ID prefix similarity is strong enough to turn the
future-layer prefetch oracle into a runtime candidate. This is stricter than
simple LFU/history because it uses the current token's already-computed layer
expert sets as a nearest-neighbor query.

This report uses dev traces only and does not inspect held-out/test prompts.

## Input

Dev7 route root:

`/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-route-n32-001009`

Prompts:

- `dev_france_regression`
- `dev_japan_factual`
- `dev_photosynthesis_factual`
- `dev_linear_equation`
- `dev_python_reverse`
- `dev_zh_france`
- `dev_mixed_summary`

All seven runs were cold-start N32, under `MemoryMax=15900000000` and
`MemorySwapMax=0`, with quality pass.

## Tool

`.Agent/run-tools/kimi_prefix_neighbor_predictor_admission.py`

Method:

- leave-one-dev-prompt-out;
- build a prefix signature from active expert sets in already-computed layers;
- find nearest prefixes in the other dev prompts using weighted Jaccard;
- vote over future-layer target experts;
- evaluate byte recall, predicted/actual bytes and full-step coverage.

Admission gate:

- all-role byte recall `>=65%`;
- predicted/actual bytes `<=1.35x`;
- full-step coverage `>=40%`.

## Runs

Smoke:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-route-n32-001009
OUT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-prefix-neighbor-smoke-030927
python3 .Agent/run-tools/kimi_prefix_neighbor_predictor_admission.py \
  --runs-root "$ROOT" \
  --out "$OUT/report.md" \
  --horizons 1 \
  --widths 1,2 \
  --budgets 8,16,32 \
  --neighbors 3 \
  --min-similarities 0.0
```

Focused max-recall:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-route-n32-001009
OUT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-prefix-neighbor-focused-031038
python3 .Agent/run-tools/kimi_prefix_neighbor_predictor_admission.py \
  --runs-root "$ROOT" \
  --out "$OUT/report.md" \
  --horizons 1,2,3 \
  --widths 1,2,3 \
  --budgets 32 \
  --neighbors 9 \
  --min-similarities 0.0
```

## Result

Smoke best under byte-budget gate:

| horizon | width | budget | neighbors | byte recall | full steps | pred/actual |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `2` | `8` | `3` | `22.95%` | `0.09%` | `1.00x` |

Focused best all-role recall:

| horizon | width | budget | neighbors | byte recall | full steps | pred/actual |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `3` | `32` | `9` | `40.99%` | `2.38%` | `4.00x` |
| `2` | `3` | `32` | `9` | `40.87%` | `2.38%` | `4.00x` |
| `3` | `3` | `32` | `9` | `40.74%` | `2.55%` | `4.00x` |

Passing all-role rows: `0`.

## Decision

Reject route-ID prefix-neighbor future prefetch as a standalone runtime
candidate. It improves over the smallest smoke rows, but remains far from the
admission gate and requires `4.00x` predicted bytes for only about `41%` byte
recall.

Future prefetch still needs a stronger signal, such as router logits, hidden
states, or a draft/router model. Otherwise the next major path should return to
lower-byte expert representation or a storage format that reduces bytes moved.
