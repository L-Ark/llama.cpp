# Kimi FineMoE route-prefix bound

- generated_at: `2026-07-12T04:45:37+0000`
- trace: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_france_regression/route-trace.csv`
- segments_used: `32`
- max_prefetch_distance: `8`

| distance | prefix predictor recall | prefix useful/fp | global recall | global useful/fp | windows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4230 | 0.501 | 0.3308 | 0.502 | 1829 |
| 2 | 0.4224 | 0.498 | 0.3322 | 0.505 | 1798 |
| 3 | 0.4218 | 0.496 | 0.3330 | 0.507 | 1767 |
| 4 | 0.4207 | 0.494 | 0.3342 | 0.510 | 1736 |
| 5 | 0.4198 | 0.492 | 0.3354 | 0.513 | 1705 |
| 6 | 0.4185 | 0.490 | 0.3362 | 0.514 | 1674 |
| 7 | 0.4172 | 0.488 | 0.3371 | 0.516 | 1643 |
| 8 | 0.4160 | 0.486 | 0.3377 | 0.518 | 1612 |

## Decision

Phase 0 does not justify runtime FineMoE prefetch/protection yet. The prefix predictor does not recover enough future expert bytes with a strong useful-to-false-positive ratio on this accepted trace.

## Reproduce

```bash
.Agent/run-tools/kimi_finemoe_route_prefix_bound.py --trace /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_france_regression/route-trace.csv --out-json .Agent/runs/20260712-finemoe-prefix-expert-id-admission/dev_france_regression.json --out-md .Agent/runs/20260712-finemoe-prefix-expert-id-admission/dev_france_regression.md --max-distance 8 --min-segment-layers 50
```
