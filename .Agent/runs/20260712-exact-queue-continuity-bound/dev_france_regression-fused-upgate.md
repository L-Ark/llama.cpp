# Kimi fused up/gate IO bound

This is a dev-only offline bound. It does not use held-out test prompts and does not change runtime code.

- run_dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_france_regression`
- prompt: `dev_france_regression`
- quality: `pass`
- current token rate: `1.42 tok/s`
- decode: `21839.52 ms / 31 tokens`
- peak bandwidth assumption: `10.40 GiB/s`

## Bound

| metric | value |
|---|---:|
| current up+gate staged bytes | 75.34 GiB |
| current up+gate staged jobs | 15864 |
| fused up+gate jobs | 7934 |
| job reduction | 7930 (50.0%) |
| rows with both up and gate staged | 1859/1859 (100.0%) |
| current exposed up+gate bandwidth | 7.24 GiB/s |
| current exposed up+gate time | 10403.08 ms |
| ideal up+gate time at peak | 7244.12 ms |
| ideal saved decode time at peak | 3158.96 ms |
| ideal token rate if all saved | 1.66 tok/s |

## Interpretation

- Fusing up+gate can reduce job count but cannot reduce moved bytes.
- The `ideal token rate` row is an optimistic upper bound: it assumes all exposed up+gate time above pure IO peak disappears from decode.
- If this upper bound remains far below 5 tok/s, fused up/gate pack is not enough as a primary path.
