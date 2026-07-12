# Kimi Phase 5C RAM slab screen

This is a dev-only offline screen. It does not use held-out prompts and does not claim a SOTA result.

## Inputs

- prompts: `2`
- decode-like rows: `150482`
- decode-like batches: `32264`
- decode runs: `180`
- total batch wait: `0.000 ms`
- total batch wait per decode token: `0.000 ms/token`
- max jobs: `8`
- simulated VRAM hotset entries: `2548`

## Top Candidates

| kind | name | prompts | resident MiB | read GiB | wait ms/token | wait/GiB | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `layer_role` | `blk.60.up` | 2 | 1393.86 | 4.722 | 0.000 | 0.000 | 179 | 179 | 0 | 64.32 |
| `layer_upgate` | `blk.60.upgate` | 2 | 2787.73 | 9.449 | 0.000 | 0.000 | 358 | 358 | 0 | 128.64 |
| `layer_all` | `blk.60.all` | 2 | 4352.21 | 15.530 | 0.000 | 0.000 | 540 | 540 | 0 | 158.72 |
| `layer_role` | `blk.60.gate` | 2 | 1393.86 | 4.727 | 0.000 | 0.000 | 179 | 179 | 0 | 64.32 |
| `layer_role` | `blk.60.down` | 2 | 1564.49 | 6.081 | 0.000 | 0.000 | 182 | 182 | 0 | 30.08 |
| `layer_role` | `blk.1.up` | 2 | 1116.98 | 4.385 | 0.000 | 0.000 | 180 | 180 | 0 | 49.33 |
| `layer_upgate` | `blk.1.upgate` | 2 | 2233.96 | 8.769 | 0.000 | 0.000 | 360 | 360 | 0 | 98.66 |
| `layer_all` | `blk.1.all` | 2 | 3731.85 | 15.719 | 0.000 | 0.000 | 540 | 540 | 0 | 146.79 |
| `layer_role` | `blk.1.gate` | 2 | 1116.98 | 4.385 | 0.000 | 0.000 | 180 | 180 | 0 | 49.33 |
| `layer_role` | `blk.1.down` | 2 | 1497.89 | 6.950 | 0.000 | 0.000 | 180 | 180 | 0 | 48.12 |
| `layer_role` | `blk.2.up` | 2 | 1270.56 | 4.758 | 0.000 | 0.000 | 180 | 180 | 0 | 128.64 |
| `layer_upgate` | `blk.2.upgate` | 2 | 2541.12 | 9.517 | 0.000 | 0.000 | 360 | 360 | 0 | 257.29 |
| `layer_all` | `blk.2.all` | 2 | 3966.83 | 16.167 | 0.000 | 0.000 | 540 | 540 | 0 | 293.38 |
| `layer_role` | `blk.2.gate` | 2 | 1270.56 | 4.758 | 0.000 | 0.000 | 180 | 180 | 0 | 128.64 |
| `layer_role` | `blk.2.down` | 2 | 1425.70 | 6.650 | 0.000 | 0.000 | 180 | 180 | 0 | 36.09 |
| `layer_role` | `blk.3.up` | 2 | 1441.93 | 5.119 | 0.000 | 0.000 | 180 | 180 | 0 | 85.77 |
| `layer_upgate` | `blk.3.upgate` | 2 | 2883.85 | 10.238 | 0.000 | 0.000 | 360 | 360 | 0 | 171.53 |
| `layer_all` | `blk.3.all` | 2 | 4502.31 | 16.960 | 0.000 | 0.000 | 540 | 540 | 0 | 201.62 |
| `layer_role` | `blk.3.gate` | 2 | 1441.93 | 5.119 | 0.000 | 0.000 | 180 | 180 | 0 | 85.77 |
| `layer_role` | `blk.3.down` | 2 | 1618.46 | 6.721 | 0.000 | 0.000 | 180 | 180 | 0 | 30.09 |
| `layer_role` | `blk.4.up` | 2 | 1393.74 | 5.350 | 0.000 | 0.000 | 180 | 180 | 0 | 91.11 |
| `layer_upgate` | `blk.4.upgate` | 2 | 2787.48 | 10.694 | 0.000 | 0.000 | 360 | 360 | 0 | 182.23 |
| `layer_all` | `blk.4.all` | 2 | 4721.54 | 19.345 | 0.000 | 0.000 | 540 | 540 | 0 | 234.29 |
| `layer_role` | `blk.4.gate` | 2 | 1393.74 | 5.344 | 0.000 | 0.000 | 180 | 180 | 0 | 91.11 |
| `layer_role` | `blk.4.down` | 2 | 1934.05 | 8.651 | 0.000 | 0.000 | 180 | 180 | 0 | 52.07 |
| `layer_role` | `blk.5.up` | 2 | 1361.55 | 5.475 | 0.000 | 0.000 | 180 | 180 | 0 | 91.12 |
| `layer_upgate` | `blk.5.upgate` | 2 | 2723.09 | 10.950 | 0.000 | 0.000 | 360 | 360 | 0 | 182.24 |
| `layer_all` | `blk.5.all` | 2 | 4251.33 | 17.889 | 0.000 | 0.000 | 540 | 540 | 0 | 224.36 |
| `layer_role` | `blk.5.gate` | 2 | 1361.55 | 5.475 | 0.000 | 0.000 | 180 | 180 | 0 | 91.12 |
| `layer_role` | `blk.5.down` | 2 | 1528.23 | 6.939 | 0.000 | 0.000 | 180 | 180 | 0 | 42.11 |
| `layer_role` | `blk.6.up` | 2 | 1211.51 | 4.743 | 0.000 | 0.000 | 180 | 180 | 0 | 107.20 |
| `layer_upgate` | `blk.6.upgate` | 2 | 2423.02 | 9.485 | 0.000 | 0.000 | 360 | 360 | 0 | 214.40 |
| `layer_all` | `blk.6.all` | 2 | 4203.07 | 17.768 | 0.000 | 0.000 | 540 | 540 | 0 | 308.91 |
| `layer_role` | `blk.6.gate` | 2 | 1211.51 | 4.743 | 0.000 | 0.000 | 180 | 180 | 0 | 107.20 |
| `layer_role` | `blk.6.down` | 2 | 1780.04 | 8.283 | 0.000 | 0.000 | 180 | 180 | 0 | 94.51 |
| `layer_role` | `blk.7.up` | 2 | 1243.66 | 4.512 | 0.000 | 0.000 | 179 | 179 | 0 | 117.92 |
| `layer_upgate` | `blk.7.upgate` | 2 | 2284.31 | 8.288 | 0.000 | 0.000 | 358 | 358 | 0 | 194.16 |
| `layer_all` | `blk.7.all` | 2 | 4111.59 | 16.486 | 0.000 | 0.000 | 538 | 538 | 0 | 280.79 |
| `layer_role` | `blk.7.gate` | 2 | 1040.66 | 3.775 | 0.000 | 0.000 | 179 | 179 | 0 | 76.24 |
| `layer_role` | `blk.7.down` | 2 | 1827.28 | 8.199 | 0.000 | 0.000 | 180 | 180 | 0 | 86.63 |

## Layer/Role Slab Upper Bound

These rows are more relevant for a pageable RAM slab than the tiny contiguous windows because they turn whole layer/role demand batches into RAM hits.

| kind | name | prompts | resident MiB | wait ms/token | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_upgate` | `blk.60.upgate` | 2 | 2787.73 | 0.000 | 358 | 358 | 0 | 128.64 |
| `layer_upgate` | `blk.1.upgate` | 2 | 2233.96 | 0.000 | 360 | 360 | 0 | 98.66 |
| `layer_upgate` | `blk.2.upgate` | 2 | 2541.12 | 0.000 | 360 | 360 | 0 | 257.29 |
| `layer_upgate` | `blk.3.upgate` | 2 | 2883.85 | 0.000 | 360 | 360 | 0 | 171.53 |
| `layer_upgate` | `blk.4.upgate` | 2 | 2787.48 | 0.000 | 360 | 360 | 0 | 182.23 |
| `layer_upgate` | `blk.5.upgate` | 2 | 2723.09 | 0.000 | 360 | 360 | 0 | 182.24 |
| `layer_upgate` | `blk.6.upgate` | 2 | 2423.02 | 0.000 | 360 | 360 | 0 | 214.40 |
| `layer_upgate` | `blk.7.upgate` | 2 | 2284.31 | 0.000 | 358 | 358 | 0 | 194.16 |
| `layer_role` | `blk.60.up` | 2 | 1393.86 | 0.000 | 179 | 179 | 0 | 64.32 |
| `layer_role` | `blk.60.gate` | 2 | 1393.86 | 0.000 | 179 | 179 | 0 | 64.32 |
| `layer_role` | `blk.60.down` | 2 | 1564.49 | 0.000 | 182 | 182 | 0 | 30.08 |
| `layer_role` | `blk.1.up` | 2 | 1116.98 | 0.000 | 180 | 180 | 0 | 49.33 |
| `layer_role` | `blk.1.gate` | 2 | 1116.98 | 0.000 | 180 | 180 | 0 | 49.33 |
| `layer_role` | `blk.1.down` | 2 | 1497.89 | 0.000 | 180 | 180 | 0 | 48.12 |
| `layer_role` | `blk.2.up` | 2 | 1270.56 | 0.000 | 180 | 180 | 0 | 128.64 |
| `layer_role` | `blk.2.gate` | 2 | 1270.56 | 0.000 | 180 | 180 | 0 | 128.64 |
| `layer_all` | `blk.60.all` | 2 | 4352.21 | 0.000 | 540 | 540 | 0 | 158.72 |
| `layer_all` | `blk.1.all` | 2 | 3731.85 | 0.000 | 540 | 540 | 0 | 146.79 |
| `layer_all` | `blk.2.all` | 2 | 3966.83 | 0.000 | 540 | 540 | 0 | 293.38 |
| `layer_all` | `blk.3.all` | 2 | 4502.31 | 0.000 | 540 | 540 | 0 | 201.62 |
| `layer_all` | `blk.4.all` | 2 | 4721.54 | 0.000 | 540 | 540 | 0 | 234.29 |
| `layer_all` | `blk.5.all` | 2 | 4251.33 | 0.000 | 540 | 540 | 0 | 224.36 |
| `layer_all` | `blk.6.all` | 2 | 4203.07 | 0.000 | 540 | 540 | 0 | 308.91 |
| `layer_all` | `blk.7.all` | 2 | 4111.59 | 0.000 | 538 | 538 | 0 | 280.79 |

### Greedy `layer_upgate` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 2048 | 1 | 1803.3 | 0.00 |
| 4096 | 1 | 2787.7 | 0.00 |
| 8192 | 3 | 7562.8 | 0.00 |
| 10240 | 4 | 9985.8 | 0.00 |

### Greedy `layer_all` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 4096 | 1 | 3731.8 | 0.00 |
| 8192 | 2 | 8084.1 | 0.00 |
| 10240 | 2 | 8084.1 | 0.00 |


## Decision Notes

- Prefer candidates with high `wait/GiB`, broad dev prompt coverage, low VRAM overlap, and many dominant/ram-only batches.
- Treat high `mixed_risk_batches` as a fragmentation warning: those candidates can reduce SSD bytes while making remaining SSD batches smaller and less efficient.
- If the greedy budget bound is only a few tens of ms/token, RAM slabs alone cannot bridge the current gap to `2 tok/s`; they should be treated as a small candidate or rejected in favor of lower-byte expert representation.
- Runtime A/B should start with pageable RAM and must reject candidates that raise TTFT, refaults, direct reclaim, or aggregate demand wait.
