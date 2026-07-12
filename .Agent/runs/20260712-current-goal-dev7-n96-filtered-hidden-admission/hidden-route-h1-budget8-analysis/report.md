# Kimi hidden-vector future expert admission

This is a dev-only offline screen. It does not change runtime behavior or claim SOTA.

## Corpus

- root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-n96-filtered-hidden-corpus`
- prompts: `7`
- feature dim: `128`
- horizons: `1`
- budgets: `8`

| prompt | route tokens | activation tokens | paired tokens | activation rows | f32 MiB | samples |
|---|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | `85` | `85` | `85` | `5101` | `139.5` | `5100` |
| `dev_japan_factual` | `78` | `78` | `78` | `4681` | `128.0` | `4680` |
| `dev_linear_equation` | `34` | `34` | `34` | `2041` | `55.8` | `2040` |
| `dev_mixed_summary` | `49` | `49` | `49` | `2941` | `80.4` | `2940` |
| `dev_photosynthesis_factual` | `94` | `94` | `94` | `5641` | `154.2` | `5640` |
| `dev_python_reverse` | `95` | `95` | `95` | `5701` | `155.9` | `5700` |
| `dev_zh_france` | `49` | `49` | `49` | `2941` | `80.4` | `2940` |

## Admission Gate

- recall >= `0.65`
- predicted/actual bytes <= `1.35`
- full-step coverage >= `0.4`
- passing rows: `0`

## Top Rows

| H | policy | budget | recall | precision | pred/actual | full steps | samples |
|---:|---|---:|---:|---:|---:|---:|---:|
| `1` | `hidden_knn5` | `8` | `0.236955` | `0.236955` | `1.000000` | `0.000981` | `28556` |
| `1` | `hidden_knn3` | `8` | `0.234097` | `0.234097` | `1.000000` | `0.002451` | `28556` |
| `1` | `hidden_knn1` | `8` | `0.226393` | `0.226393` | `1.000000` | `0.002907` | `28556` |
| `1` | `static_target_layer` | `8` | `0.125801` | `0.125801` | `1.000000` | `0.000000` | `28556` |
| `1` | `source_route_plus_static` | `8` | `0.020153` | `0.020153` | `1.000000` | `0.000000` | `28556` |

## Decision

Reject hidden-vector future prefetch as the next runtime path: best row H=`1` `hidden_knn5` budget `8` reaches recall `0.236955` and full-step `0.000981` with pred/actual `1.000000`.

This means filtered hidden vectors improve neither complete-batch coverage nor byte efficiency enough to justify runtime prefetch reads.

## Reproduce

```bash
.Agent/run-tools/kimi_hidden_route_future_admission.py --root /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-n96-filtered-hidden-corpus --out-dir .Agent/runs/20260712-current-goal-dev7-n96-filtered-hidden-admission/hidden-route-h1-budget8-analysis --layers 60 --horizons 1 --budgets 8 --neighbors 1,3,5 --feature-dim 128
```
