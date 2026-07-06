# Kimi fused up/gate IO bound

This is a dev-only offline bound. It does not use held-out test prompts and does not change runtime code.

- run_dir: `/Users/spark/llama.cpp-vendor-kimi/.Agent/runs/20260707-gp12-fused-upgate-bound/dev_france_regression`
- prompt: `dev_france_regression`
- quality: `pass`
- current token rate: `1.28 tok/s`
- decode: `24175.55 ms / 31 tokens`
- peak bandwidth assumption: `10.40 GiB/s`

## Bound

| metric | value |
|---|---:|
| current up+gate staged bytes | 38.08 GiB |
| current up+gate staged jobs | 8109 |
| fused up+gate jobs | 4055 |
| job reduction | 4054 (50.0%) |
| rows with both up and gate staged | 867/867 (100.0%) |
| current exposed up+gate bandwidth | 6.27 GiB/s |
| current exposed up+gate time | 6069.18 ms |
| ideal up+gate time at peak | 3661.39 ms |
| ideal saved decode time at peak | 2407.79 ms |
| ideal token rate if all saved | 1.42 tok/s |

## Interpretation

- Fusing up+gate can reduce job count but cannot reduce moved bytes.
- The `ideal token rate` row is an optimistic upper bound: it assumes all exposed up+gate time above pure IO peak disappears from decode.
- If this upper bound remains far below 5 tok/s, fused up/gate pack is not enough as a primary path.
