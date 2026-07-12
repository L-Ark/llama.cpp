# Kimi exact-byte scheduler ceiling

This is an offline ceiling analysis. It does not change runtime behavior or claim SOTA.

- profile root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717`
- movement bandwidth ceiling: `10.40 GiB/s`
- all-hit MoE floor: `40.1 ms/token`

## Summary

- prompts: `2`
- measured mean token rate: `1.460 tok/s`
- transfer-only mean ceiling: `1.591 tok/s`
- floor+transfer mean ceiling: `1.495 tok/s`
- best prompt floor+transfer ceiling: `1.527 tok/s`
- worst prompt floor+transfer ceiling: `1.464 tok/s`
- mean byte ratio needed for 5.0 tok/s after floor: `0.254x`

## Per Prompt

| prompt | measured tok/s | moved GiB/token | transfer-only tok/s | floor+transfer tok/s | required ratio for 5.0 tok/s | call wall ms/token | runtime-load copy ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | `1.420` | `6.689` | `1.555` | `1.464` | `0.249` | `1019.4` | `3853.4` |
| `dev_intelligence_general` | `1.500` | `6.393` | `1.627` | `1.527` | `0.260` | `948.3` | `3549.1` |

## Decision

Exact-byte scheduling without byte reduction is not a primary 5.0 tok/s path: even at 10.4 GiB/s and a 40.1 ms/token all-hit floor, the best held-out prompt ceiling is 1.53 tok/s and the mean ceiling is 1.50 tok/s. Continue with structural byte reduction.

## Reproduce

```bash
.Agent/run-tools/kimi_exact_byte_scheduler_ceiling.py --profile-root /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717 --out-json .Agent/runs/20260712-exact-queue-continuity-bound/exact-byte-ceiling-5tps.json --out-md .Agent/runs/20260712-exact-queue-continuity-bound/exact-byte-ceiling-5tps.md --bandwidth-gib-s 10.4 --all-hit-floor-ms-per-token 40.1 --target-tok-s 5
```
