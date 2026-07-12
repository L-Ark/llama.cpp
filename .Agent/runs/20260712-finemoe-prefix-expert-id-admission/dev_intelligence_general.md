# Kimi FineMoE route-prefix bound

- generated_at: `2026-07-12T04:46:33+0000`
- trace: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_intelligence_general/route-trace.csv`
- segments_used: `32`
- max_prefetch_distance: `8`

| distance | prefix predictor recall | prefix useful/fp | global recall | global useful/fp | windows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4311 | 0.549 | 0.3019 | 0.439 | 1829 |
| 2 | 0.4311 | 0.547 | 0.3040 | 0.443 | 1798 |
| 3 | 0.4322 | 0.549 | 0.3064 | 0.448 | 1767 |
| 4 | 0.4330 | 0.550 | 0.3086 | 0.453 | 1736 |
| 5 | 0.4335 | 0.551 | 0.3104 | 0.457 | 1705 |
| 6 | 0.4340 | 0.552 | 0.3121 | 0.461 | 1674 |
| 7 | 0.4340 | 0.552 | 0.3133 | 0.463 | 1643 |
| 8 | 0.4339 | 0.552 | 0.3141 | 0.465 | 1612 |

## Decision

Phase 0 does not justify runtime FineMoE prefetch/protection yet. The prefix predictor does not recover enough future expert bytes with a strong useful-to-false-positive ratio on this accepted trace.

## Reproduce

```bash
.Agent/run-tools/kimi_finemoe_route_prefix_bound.py --trace /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_intelligence_general/route-trace.csv --out-json .Agent/runs/20260712-finemoe-prefix-expert-id-admission/dev_intelligence_general.json --out-md .Agent/runs/20260712-finemoe-prefix-expert-id-admission/dev_intelligence_general.md --max-distance 8 --min-segment-layers 50
```
