# Kimi Phase 5C RAM slab screen

This is a dev-only offline screen. It does not use held-out prompts and does not claim a SOTA result.

## Inputs

- prompts: `2`
- decode-like rows: `150482`
- decode-like batches: `32264`
- decode runs: `180`
- total batch wait: `92473.847 ms`
- total batch wait per decode token: `513.744 ms/token`
- max jobs: `8`
- simulated VRAM hotset entries: `2548`

## Top Candidates

| kind | name | prompts | resident MiB | read GiB | wait ms/token | wait/GiB | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contig_window` | `src0.gate.win512MiB.rank1` | 2 | 4.48 | 0.101 | 0.076 | 3135.202 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank1` | 2 | 4.48 | 0.101 | 0.076 | 3135.202 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank1` | 2 | 4.48 | 0.101 | 0.075 | 3094.950 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank1` | 2 | 4.48 | 0.101 | 0.075 | 3094.950 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.gate.win512MiB.rank2` | 2 | 4.48 | 0.070 | 0.053 | 2184.035 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank2` | 2 | 4.48 | 0.070 | 0.053 | 2184.035 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank2` | 2 | 4.48 | 0.070 | 0.052 | 2153.586 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank2` | 2 | 4.48 | 0.070 | 0.052 | 2153.586 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank3` | 2 | 8.97 | 0.114 | 0.100 | 2047.278 | 21 | 0 | 21 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank3` | 2 | 8.97 | 0.114 | 0.100 | 2047.278 | 21 | 0 | 21 | 4.48 |
| `contig_window` | `src0.down.win512MiB.rank1` | 2 | 6.02 | 0.147 | 0.065 | 2005.368 | 24 | 0 | 24 | 0.00 |
| `contig_window` | `src0.down.win1024MiB.rank1` | 2 | 6.02 | 0.147 | 0.065 | 2005.368 | 24 | 0 | 24 | 0.00 |
| `contig_window` | `src0.down.win512MiB.rank2` | 2 | 7.88 | 0.177 | 0.076 | 1789.190 | 22 | 0 | 22 | 7.88 |
| `contig_window` | `src0.down.win1024MiB.rank2` | 2 | 7.88 | 0.177 | 0.076 | 1789.190 | 22 | 0 | 22 | 7.88 |
| `contig_window` | `src0.gate.win512MiB.rank3` | 2 | 10.72 | 0.136 | 0.103 | 1772.311 | 21 | 0 | 21 | 10.72 |
| `contig_window` | `src0.gate.win1024MiB.rank3` | 2 | 10.72 | 0.136 | 0.103 | 1772.311 | 21 | 0 | 21 | 10.72 |
| `contig_window` | `src0.up.win512MiB.rank4` | 2 | 8.97 | 0.105 | 0.082 | 1675.974 | 22 | 0 | 22 | 8.97 |
| `contig_window` | `src0.up.win1024MiB.rank4` | 2 | 8.97 | 0.105 | 0.082 | 1675.974 | 22 | 0 | 22 | 8.97 |
| `contig_window` | `src0.up.win512MiB.rank5` | 1 | 4.48 | 0.048 | 0.040 | 1657.253 | 10 | 0 | 10 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank5` | 1 | 4.48 | 0.048 | 0.040 | 1657.253 | 10 | 0 | 10 | 4.48 |
| `contig_window` | `src0.down.win512MiB.rank3` | 2 | 12.03 | 0.211 | 0.105 | 1610.120 | 29 | 0 | 29 | 6.02 |
| `contig_window` | `src0.down.win1024MiB.rank3` | 2 | 12.03 | 0.211 | 0.105 | 1610.120 | 29 | 0 | 29 | 6.02 |
| `contig_window` | `src0.gate.win512MiB.rank4` | 2 | 10.72 | 0.126 | 0.085 | 1469.477 | 22 | 0 | 22 | 10.72 |
| `contig_window` | `src0.gate.win1024MiB.rank4` | 2 | 10.72 | 0.126 | 0.085 | 1469.477 | 22 | 0 | 22 | 10.72 |
| `contig_window` | `src0.gate.win512MiB.rank5` | 1 | 5.36 | 0.058 | 0.041 | 1427.152 | 10 | 0 | 10 | 5.36 |
| `contig_window` | `src0.gate.win1024MiB.rank5` | 1 | 5.36 | 0.058 | 0.041 | 1427.152 | 10 | 0 | 10 | 5.36 |
| `contig_window` | `src0.up.win512MiB.rank7` | 2 | 13.45 | 0.136 | 0.104 | 1420.312 | 28 | 0 | 28 | 8.97 |
| `contig_window` | `src0.up.win1024MiB.rank7` | 2 | 13.45 | 0.136 | 0.104 | 1420.312 | 28 | 0 | 28 | 8.97 |
| `contig_window` | `src0.gate.win512MiB.rank6` | 2 | 8.97 | 0.092 | 0.069 | 1416.855 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank6` | 2 | 8.97 | 0.092 | 0.069 | 1416.855 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank8` | 1 | 4.48 | 0.044 | 0.034 | 1383.263 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank8` | 1 | 4.48 | 0.044 | 0.034 | 1383.263 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank6` | 2 | 8.97 | 0.092 | 0.067 | 1374.810 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank6` | 2 | 8.97 | 0.092 | 0.067 | 1374.810 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.down.win512MiB.rank4` | 2 | 12.03 | 0.200 | 0.089 | 1371.052 | 32 | 0 | 32 | 0.00 |
| `contig_window` | `src0.down.win1024MiB.rank4` | 2 | 12.03 | 0.200 | 0.089 | 1371.052 | 32 | 0 | 32 | 0.00 |
| `contig_window` | `src0.gate.win512MiB.rank9` | 2 | 13.45 | 0.123 | 0.097 | 1329.302 | 24 | 0 | 24 | 0.00 |
| `contig_window` | `src0.gate.win1024MiB.rank9` | 2 | 13.45 | 0.123 | 0.097 | 1329.302 | 24 | 0 | 24 | 0.00 |
| `contig_window` | `src0.up.win512MiB.rank9` | 2 | 13.45 | 0.123 | 0.094 | 1293.421 | 24 | 0 | 24 | 0.00 |
| `contig_window` | `src0.up.win1024MiB.rank9` | 2 | 13.45 | 0.123 | 0.094 | 1293.421 | 24 | 0 | 24 | 0.00 |

## Layer/Role Slab Upper Bound

These rows are more relevant for a pageable RAM slab than the tiny contiguous windows because they turn whole layer/role demand batches into RAM hits.

| kind | name | prompts | resident MiB | wait ms/token | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_upgate` | `blk.1.upgate` | 2 | 2233.96 | 8.350 | 360 | 360 | 0 | 98.66 |
| `layer_upgate` | `blk.9.upgate` | 2 | 2569.97 | 6.826 | 360 | 360 | 0 | 185.20 |
| `layer_upgate` | `blk.7.upgate` | 2 | 2284.31 | 6.736 | 358 | 358 | 0 | 194.16 |
| `layer_upgate` | `blk.28.upgate` | 2 | 2195.63 | 6.693 | 358 | 358 | 0 | 175.34 |
| `layer_upgate` | `blk.29.upgate` | 2 | 2185.88 | 6.525 | 360 | 360 | 0 | 183.45 |
| `layer_upgate` | `blk.8.upgate` | 2 | 2264.60 | 6.463 | 360 | 360 | 0 | 153.90 |
| `layer_upgate` | `blk.30.upgate` | 2 | 2107.11 | 6.454 | 360 | 360 | 0 | 188.79 |
| `layer_upgate` | `blk.32.upgate` | 2 | 2097.26 | 6.334 | 360 | 360 | 0 | 185.19 |
| `layer_role` | `blk.1.gate` | 2 | 1116.98 | 4.257 | 180 | 180 | 0 | 49.33 |
| `layer_role` | `blk.1.up` | 2 | 1116.98 | 4.092 | 180 | 180 | 0 | 49.33 |
| `layer_role` | `blk.4.down` | 2 | 1934.05 | 3.784 | 180 | 180 | 0 | 52.07 |
| `layer_role` | `blk.6.down` | 2 | 1780.04 | 3.778 | 180 | 180 | 0 | 94.51 |
| `layer_role` | `blk.7.down` | 2 | 1827.28 | 3.764 | 180 | 180 | 0 | 86.63 |
| `layer_role` | `blk.10.down` | 2 | 1913.96 | 3.637 | 180 | 180 | 0 | 86.63 |
| `layer_role` | `blk.9.down` | 2 | 2055.75 | 3.618 | 180 | 180 | 0 | 78.75 |
| `layer_role` | `blk.8.down` | 2 | 1811.52 | 3.514 | 180 | 180 | 0 | 86.62 |
| `layer_all` | `blk.1.all` | 2 | 3731.85 | 11.502 | 540 | 540 | 0 | 146.79 |
| `layer_all` | `blk.7.all` | 2 | 4111.59 | 10.500 | 538 | 538 | 0 | 280.79 |
| `layer_all` | `blk.9.all` | 2 | 4625.72 | 10.444 | 540 | 540 | 0 | 263.95 |
| `layer_all` | `blk.8.all` | 2 | 4076.12 | 9.977 | 540 | 540 | 0 | 240.52 |
| `layer_all` | `blk.10.all` | 2 | 4094.03 | 9.851 | 540 | 540 | 0 | 221.18 |
| `layer_all` | `blk.28.all` | 2 | 3537.36 | 9.540 | 538 | 538 | 0 | 211.43 |
| `layer_all` | `blk.25.all` | 2 | 3478.91 | 9.517 | 540 | 540 | 0 | 182.01 |
| `layer_all` | `blk.29.all` | 2 | 3521.64 | 9.348 | 540 | 540 | 0 | 243.62 |

### Greedy `layer_upgate` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 2048 | 1 | 1794.2 | 5.91 |
| 4096 | 2 | 4028.2 | 14.26 |
| 8192 | 4 | 7338.6 | 25.00 |
| 10240 | 5 | 9240.5 | 31.14 |

### Greedy `layer_all` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 4096 | 1 | 3731.8 | 11.50 |
| 8192 | 2 | 6729.4 | 20.13 |
| 10240 | 3 | 9732.5 | 28.58 |


## Decision Notes

- Prefer candidates with high `wait/GiB`, broad dev prompt coverage, low VRAM overlap, and many dominant/ram-only batches.
- Treat high `mixed_risk_batches` as a fragmentation warning: those candidates can reduce SSD bytes while making remaining SSD batches smaller and less efficient.
- If the greedy budget bound is only a few tens of ms/token, RAM slabs alone cannot bridge the current gap to `2 tok/s`; they should be treated as a small candidate or rejected in favor of lower-byte expert representation.
- Runtime A/B should start with pageable RAM and must reject candidates that raise TTFT, refaults, direct reclaim, or aggregate demand wait.
