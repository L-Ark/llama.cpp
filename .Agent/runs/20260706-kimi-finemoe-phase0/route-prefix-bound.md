# Kimi FineMoE route-prefix bound

- generated_at: `2026-07-06T15:23:39+0800`
- trace: `.Agent/runs/20260706-kimi-finemoe-phase0/route_trace.csv`
- segments_used: `63`
- max_prefetch_distance: `8`

| distance | prefix predictor recall | prefix useful/fp | global recall | global useful/fp | windows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4330 | 0.764 | 0.3471 | 0.532 | 3658 |
| 2 | 0.4324 | 0.762 | 0.3486 | 0.536 | 3596 |
| 3 | 0.4320 | 0.760 | 0.3497 | 0.538 | 3534 |
| 4 | 0.4312 | 0.758 | 0.3507 | 0.541 | 3472 |
| 5 | 0.4307 | 0.756 | 0.3518 | 0.543 | 3410 |
| 6 | 0.4298 | 0.754 | 0.3524 | 0.545 | 3348 |
| 7 | 0.4291 | 0.752 | 0.3533 | 0.547 | 3286 |
| 8 | 0.4285 | 0.750 | 0.3541 | 0.549 | 3224 |

## Decision

Phase 0 does not justify runtime FineMoE prefetch/protection yet. The prefix predictor does not recover enough future expert bytes with a strong useful-to-false-positive ratio on this accepted trace.

## Reproduce

```bash
.Agent/run-tools/kimi_finemoe_route_prefix_bound.py --trace .Agent/runs/20260706-kimi-finemoe-phase0/route_trace.csv --out-json .Agent/runs/20260706-kimi-finemoe-phase0/route-prefix-bound.json --out-md .Agent/runs/20260706-kimi-finemoe-phase0/route-prefix-bound.md --max-distance 8
```
