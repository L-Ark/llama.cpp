# Kimi quant-scaled VRAM cache bound

This is a dev-only offline route-trace simulation. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-08T03:20:54+0800`
- runs_root: `.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct`
- prompts: `7`

## Mean Bound

| ratio | global LFU hit | global LFU miss GiB/tok | prompt LFU hit | prompt LFU miss GiB/tok | LRU hit | LRU miss GiB/tok | farthest-next hit | farthest-next miss GiB/tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 28.9% | 5.123 | 48.7% | 3.702 | 41.7% | 4.211 | n/a | n/a |
| 0.827 | 32.2% | 4.041 | 53.1% | 2.805 | 45.5% | 3.258 | n/a | n/a |
| 0.696 | 35.4% | 3.241 | 57.1% | 2.158 | 48.8% | 2.573 | n/a | n/a |
| 0.563 | 39.6% | 2.453 | 62.4% | 1.533 | 52.5% | 1.932 | n/a | n/a |
| 0.505 | 41.9% | 2.118 | 65.1% | 1.276 | 54.5% | 1.661 | n/a | n/a |
| 0.400 | 47.1% | 1.528 | 71.1% | 0.840 | 58.6% | 1.197 | n/a | n/a |
| 0.300 | 54.2% | 0.993 | 78.7% | 0.465 | 63.3% | 0.794 | n/a | n/a |

## Interpretation

- `global LFU` is prompt-general over the dev set and is the only policy in this table that resembles a deployable static hotset.
- `prompt LFU` and `farthest-next` are oracles. They are upper bounds, not acceptable SOTA policies.
- Miss GiB/token includes both up/gate and down groups after applying the quant byte ratio.
- A `5 tok/s` non-overlap transfer budget at `10.4 GiB/s` and `40 ms` compute floor is about `1.664 GiB/token`.
- A perfect-overlap transfer budget is about `2.08 GiB/token`.

## Reproduce

```bash
.Agent/run-tools/kimi_quant_scaled_cache_bound.py --runs-root .Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct --ratios 1.0,0.827,0.696,0.563,0.505,0.4,0.3 --out-json .Agent/runs/20260708-gp93-quant-scaled-cache-bound/report.json --out-md .Agent/runs/20260708-gp93-quant-scaled-cache-bound/report.md
```
