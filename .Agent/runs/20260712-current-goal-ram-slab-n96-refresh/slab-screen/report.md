# Kimi Phase 5C RAM slab screen

This is a dev-only offline screen. It does not use held-out prompts and does not claim a SOTA result.

## Inputs

- prompts: `2`
- decode-like rows: `150482`
- decode-like batches: `32264`
- decode runs: `180`
- total batch wait: `90830.872 ms`
- total batch wait per decode token: `504.616 ms/token`
- max jobs: `8`
- simulated VRAM hotset entries: `2458`

## Top Candidates

| kind | name | prompts | resident MiB | read GiB | wait ms/token | wait/GiB | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contig_window` | `src0.up.win512MiB.rank1` | 2 | 4.48 | 0.101 | 0.072 | 2957.720 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank1` | 2 | 4.48 | 0.101 | 0.072 | 2957.720 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.gate.win512MiB.rank1` | 2 | 4.48 | 0.101 | 0.071 | 2928.885 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank1` | 2 | 4.48 | 0.101 | 0.071 | 2928.885 | 22 | 0 | 22 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank2` | 2 | 4.48 | 0.070 | 0.057 | 2336.866 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank2` | 2 | 4.48 | 0.070 | 0.057 | 2336.866 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.gate.win512MiB.rank2` | 2 | 4.48 | 0.070 | 0.056 | 2288.632 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank2` | 2 | 4.48 | 0.070 | 0.056 | 2288.632 | 15 | 1 | 14 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank3` | 2 | 8.97 | 0.114 | 0.098 | 2011.759 | 21 | 0 | 21 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank3` | 2 | 8.97 | 0.114 | 0.098 | 2011.759 | 21 | 0 | 21 | 4.48 |
| `contig_window` | `src0.down.win512MiB.rank1` | 2 | 6.02 | 0.147 | 0.061 | 1865.869 | 24 | 0 | 24 | 6.02 |
| `contig_window` | `src0.down.win1024MiB.rank1` | 2 | 6.02 | 0.147 | 0.061 | 1865.869 | 24 | 0 | 24 | 6.02 |
| `contig_window` | `src0.down.win512MiB.rank2` | 2 | 7.88 | 0.177 | 0.076 | 1780.201 | 22 | 0 | 22 | 7.88 |
| `contig_window` | `src0.down.win1024MiB.rank2` | 2 | 7.88 | 0.177 | 0.076 | 1780.201 | 22 | 0 | 22 | 7.88 |
| `contig_window` | `src0.up.win512MiB.rank4` | 2 | 8.97 | 0.105 | 0.086 | 1762.610 | 22 | 0 | 22 | 8.97 |
| `contig_window` | `src0.up.win1024MiB.rank4` | 2 | 8.97 | 0.105 | 0.086 | 1762.610 | 22 | 0 | 22 | 8.97 |
| `contig_window` | `src0.gate.win512MiB.rank3` | 2 | 10.72 | 0.136 | 0.102 | 1751.713 | 21 | 0 | 21 | 5.36 |
| `contig_window` | `src0.gate.win1024MiB.rank3` | 2 | 10.72 | 0.136 | 0.102 | 1751.713 | 21 | 0 | 21 | 5.36 |
| `contig_window` | `src0.down.win512MiB.rank3` | 2 | 12.03 | 0.211 | 0.103 | 1584.204 | 29 | 0 | 29 | 6.02 |
| `contig_window` | `src0.down.win1024MiB.rank3` | 2 | 12.03 | 0.211 | 0.103 | 1584.204 | 29 | 0 | 29 | 6.02 |
| `contig_window` | `src0.gate.win512MiB.rank4` | 2 | 10.72 | 0.126 | 0.092 | 1581.962 | 22 | 0 | 22 | 10.72 |
| `contig_window` | `src0.gate.win1024MiB.rank4` | 2 | 10.72 | 0.126 | 0.092 | 1581.962 | 22 | 0 | 22 | 10.72 |
| `contig_window` | `src0.up.win512MiB.rank5` | 1 | 4.48 | 0.048 | 0.037 | 1525.749 | 10 | 0 | 10 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank5` | 1 | 4.48 | 0.048 | 0.037 | 1525.749 | 10 | 0 | 10 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank7` | 2 | 13.45 | 0.136 | 0.110 | 1504.812 | 28 | 0 | 28 | 8.97 |
| `contig_window` | `src0.up.win1024MiB.rank7` | 2 | 13.45 | 0.136 | 0.110 | 1504.812 | 28 | 0 | 28 | 8.97 |
| `contig_window` | `src0.up.win512MiB.rank6` | 2 | 8.97 | 0.092 | 0.073 | 1496.840 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank6` | 2 | 8.97 | 0.092 | 0.073 | 1496.840 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.gate.win512MiB.rank6` | 2 | 8.97 | 0.092 | 0.071 | 1466.598 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank6` | 2 | 8.97 | 0.092 | 0.071 | 1466.598 | 18 | 1 | 17 | 4.48 |
| `contig_window` | `src0.gate.win512MiB.rank7` | 2 | 16.08 | 0.162 | 0.117 | 1336.246 | 28 | 0 | 28 | 10.72 |
| `contig_window` | `src0.gate.win1024MiB.rank7` | 2 | 16.08 | 0.162 | 0.117 | 1336.246 | 28 | 0 | 28 | 10.72 |
| `contig_window` | `src0.down.win512MiB.rank4` | 2 | 12.03 | 0.200 | 0.087 | 1329.904 | 32 | 0 | 32 | 6.02 |
| `contig_window` | `src0.down.win1024MiB.rank4` | 2 | 12.03 | 0.200 | 0.087 | 1329.904 | 32 | 0 | 32 | 6.02 |
| `contig_window` | `src0.gate.win512MiB.rank5` | 1 | 5.36 | 0.058 | 0.038 | 1309.438 | 10 | 0 | 10 | 5.36 |
| `contig_window` | `src0.gate.win1024MiB.rank5` | 1 | 5.36 | 0.058 | 0.038 | 1309.438 | 10 | 0 | 10 | 5.36 |
| `contig_window` | `src0.up.win512MiB.rank8` | 1 | 4.48 | 0.044 | 0.031 | 1268.391 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank8` | 1 | 4.48 | 0.044 | 0.031 | 1268.391 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.up.win2048MiB.rank7` | 2 | 17.94 | 0.136 | 0.120 | 1229.350 | 25 | 0 | 25 | 0.00 |
| `contig_window` | `src0.gate.win512MiB.rank8` | 1 | 5.36 | 0.052 | 0.035 | 1211.352 | 9 | 0 | 9 | 5.36 |

## Layer/Role Slab Upper Bound

These rows are more relevant for a pageable RAM slab than the tiny contiguous windows because they turn whole layer/role demand batches into RAM hits.

| kind | name | prompts | resident MiB | wait ms/token | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_upgate` | `blk.1.upgate` | 2 | 2233.96 | 8.451 | 360 | 360 | 0 | 80.72 |
| `layer_upgate` | `blk.9.upgate` | 2 | 2569.97 | 6.867 | 360 | 360 | 0 | 139.59 |
| `layer_upgate` | `blk.7.upgate` | 2 | 2284.31 | 6.690 | 358 | 358 | 0 | 169.11 |
| `layer_upgate` | `blk.28.upgate` | 2 | 2195.63 | 6.633 | 358 | 358 | 0 | 160.13 |
| `layer_upgate` | `blk.29.upgate` | 2 | 2185.88 | 6.562 | 360 | 360 | 0 | 163.77 |
| `layer_upgate` | `blk.8.upgate` | 2 | 2264.60 | 6.413 | 360 | 360 | 0 | 144.93 |
| `layer_upgate` | `blk.30.upgate` | 2 | 2107.11 | 6.355 | 360 | 360 | 0 | 172.70 |
| `layer_upgate` | `blk.48.upgate` | 2 | 2442.00 | 6.196 | 360 | 360 | 0 | 134.23 |
| `layer_role` | `blk.1.gate` | 2 | 1116.98 | 4.254 | 180 | 180 | 0 | 40.36 |
| `layer_role` | `blk.1.up` | 2 | 1116.98 | 4.197 | 180 | 180 | 0 | 40.36 |
| `layer_role` | `blk.4.down` | 2 | 1934.05 | 3.749 | 180 | 180 | 0 | 89.25 |
| `layer_role` | `blk.6.down` | 2 | 1780.04 | 3.715 | 180 | 180 | 0 | 110.26 |
| `layer_role` | `blk.7.down` | 2 | 1827.28 | 3.690 | 180 | 180 | 0 | 133.88 |
| `layer_role` | `blk.9.down` | 2 | 2055.75 | 3.584 | 180 | 180 | 0 | 118.14 |
| `layer_role` | `blk.10.down` | 2 | 1913.96 | 3.578 | 180 | 180 | 0 | 118.13 |
| `layer_role` | `blk.9.up` | 2 | 1399.17 | 3.524 | 180 | 180 | 0 | 85.76 |
| `layer_all` | `blk.1.all` | 2 | 3731.85 | 11.530 | 540 | 540 | 0 | 128.84 |
| `layer_all` | `blk.9.all` | 2 | 4625.72 | 10.452 | 540 | 540 | 0 | 257.72 |
| `layer_all` | `blk.7.all` | 2 | 4111.59 | 10.380 | 538 | 538 | 0 | 302.99 |
| `layer_all` | `blk.8.all` | 2 | 4076.12 | 9.889 | 540 | 540 | 0 | 270.93 |
| `layer_all` | `blk.10.all` | 2 | 4094.03 | 9.766 | 540 | 540 | 0 | 243.71 |
| `layer_all` | `blk.28.all` | 2 | 3537.36 | 9.382 | 538 | 538 | 0 | 226.31 |
| `layer_all` | `blk.29.all` | 2 | 3521.64 | 9.321 | 540 | 540 | 0 | 235.96 |
| `layer_all` | `blk.25.all` | 2 | 3478.91 | 9.230 | 540 | 540 | 0 | 196.89 |

### Greedy `layer_upgate` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 2048 | 1 | 1794.2 | 5.91 |
| 4096 | 2 | 4028.2 | 14.36 |
| 8192 | 4 | 7338.6 | 25.00 |
| 10240 | 5 | 9177.6 | 30.86 |

### Greedy `layer_all` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 4096 | 1 | 3731.8 | 11.53 |
| 8192 | 2 | 6729.4 | 20.08 |
| 10240 | 3 | 9732.5 | 28.37 |


## Decision Notes

- Prefer candidates with high `wait/GiB`, broad dev prompt coverage, low VRAM overlap, and many dominant/ram-only batches.
- Treat high `mixed_risk_batches` as a fragmentation warning: those candidates can reduce SSD bytes while making remaining SSD batches smaller and less efficient.
- If the greedy budget bound is only a few tens of ms/token, RAM slabs alone cannot bridge the current gap to `2 tok/s`; they should be treated as a small candidate or rejected in favor of lower-byte expert representation.
- Runtime A/B should start with pageable RAM and must reject candidates that raise TTFT, refaults, direct reclaim, or aggregate demand wait.
