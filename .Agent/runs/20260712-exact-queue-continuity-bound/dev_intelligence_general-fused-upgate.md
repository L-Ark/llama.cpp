# Kimi fused up/gate IO bound

This is a dev-only offline bound. It does not use held-out test prompts and does not change runtime code.

- run_dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_intelligence_general`
- prompt: `dev_intelligence_general`
- quality: `pass`
- current token rate: `1.50 tok/s`
- decode: `20687.58 ms / 31 tokens`
- peak bandwidth assumption: `10.40 GiB/s`

## Bound

| metric | value |
|---|---:|
| current up+gate staged bytes | 76.59 GiB |
| current up+gate staged jobs | 16076 |
| fused up+gate jobs | 8040 |
| job reduction | 8036 (50.0%) |
| rows with both up and gate staged | 1847/1847 (100.0%) |
| current exposed up+gate bandwidth | 7.57 GiB/s |
| current exposed up+gate time | 10111.31 ms |
| ideal up+gate time at peak | 7364.70 ms |
| ideal saved decode time at peak | 2746.61 ms |
| ideal token rate if all saved | 1.73 tok/s |

## Interpretation

- Fusing up+gate can reduce job count but cannot reduce moved bytes.
- The `ideal token rate` row is an optimistic upper bound: it assumes all exposed up+gate time above pure IO peak disappears from decode.
- If this upper bound remains far below 5 tok/s, fused up/gate pack is not enough as a primary path.
