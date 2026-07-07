# Kimi layer full-hit cache simulation

This is a dev-only offline simulation. It does not change runtime behavior or claim SOTA.

- route root: `.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct`
- prompts: `7`
- io depth: `8`
- upgate budget GiB: `9.10`
- down budget GiB: `5.60`

## Aggregate

| strategy | mean wave saved | worst wave saved | mean full-hit calls | worst full-hit calls | mean bytes saved | up/gate full-hit | down full-hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| `front_layers_0_2` | `2.54%` | `0.97%` | `2.54%` | `0.97%` | `3.29%` | `2.44%` | `2.77%` |
| `front_layers_0_4` | `1.78%` | `0.49%` | `1.78%` | `0.49%` | `5.92%` | `1.72%` | `1.92%` |
| `front_layers_0_1` | `1.39%` | `0.66%` | `1.39%` | `0.66%` | `1.57%` | `1.33%` | `1.51%` |
| `front_layers_0_8` | `0.71%` | `0.10%` | `0.71%` | `0.10%` | `7.55%` | `0.45%` | `1.28%` |
| `front_layers_0_12` | `0.37%` | `0.03%` | `0.37%` | `0.03%` | `8.85%` | `0.26%` | `0.63%` |
| `front_layers_0_16` | `0.27%` | `0.02%` | `0.27%` | `0.02%` | `10.37%` | `0.16%` | `0.53%` |
| `fullhit_greedy` | `0.19%` | `0.00%` | `0.19%` | `0.00%` | `15.13%` | `0.21%` | `0.14%` |
| `global_hotset` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `19.53%` | `0.00%` | `0.01%` |

## Decision

`front_layers_0_2` beats global_hotset on dev leave-one-out IO waves. Generate a static profile or env-gated admission policy before held-out testing.

## Reproduce

```bash
.Agent/run-tools/kimi_layer_fullhit_cache_sim.py --route-root .Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct --out-dir .Agent/runs/20260708-gp79-layer-fullhit-cache-sim --io-depth 8 --upgate-budget-gib 9.1 --down-budget-gib 5.6 --front-layers 1,2,4,8,12,16
```
