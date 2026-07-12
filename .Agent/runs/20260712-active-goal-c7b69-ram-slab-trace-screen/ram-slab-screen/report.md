# Kimi Phase 5C RAM slab screen

This is a dev-only offline screen. It does not use held-out prompts and does not claim a SOTA result.

## Inputs

- prompts: `2`
- decode-like rows: `51596`
- decode-like batches: `11134`
- decode runs: `62`
- total batch wait: `30673.200 ms`
- total batch wait per decode token: `494.729 ms/token`
- max jobs: `8`
- simulated VRAM hotset entries: `2458`

## Top Candidates

| kind | name | prompts | resident MiB | read GiB | wait ms/token | wait/GiB | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contig_window` | `src0.gate.win512MiB.rank1` | 2 | 4.48 | 0.044 | 0.085 | 1203.635 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank1` | 2 | 4.48 | 0.044 | 0.085 | 1203.635 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank1` | 2 | 4.48 | 0.044 | 0.085 | 1198.349 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank1` | 2 | 4.48 | 0.044 | 0.085 | 1198.349 | 9 | 0 | 9 | 4.48 |
| `contig_window` | `src0.gate.win512MiB.rank2` | 2 | 4.48 | 0.031 | 0.067 | 952.645 | 6 | 0 | 6 | 0.00 |
| `contig_window` | `src0.gate.win1024MiB.rank2` | 2 | 4.48 | 0.031 | 0.067 | 952.645 | 6 | 0 | 6 | 0.00 |
| `contig_window` | `src0.down.win512MiB.rank1` | 2 | 7.44 | 0.087 | 0.109 | 933.880 | 11 | 0 | 11 | 7.44 |
| `contig_window` | `src0.down.win1024MiB.rank1` | 2 | 7.44 | 0.087 | 0.109 | 933.880 | 11 | 0 | 11 | 7.44 |
| `contig_window` | `src0.up.win512MiB.rank2` | 2 | 4.48 | 0.031 | 0.065 | 913.069 | 6 | 0 | 6 | 0.00 |
| `contig_window` | `src0.up.win1024MiB.rank2` | 2 | 4.48 | 0.031 | 0.065 | 913.069 | 6 | 0 | 6 | 0.00 |
| `contig_window` | `src0.gate.win512MiB.rank3` | 2 | 4.48 | 0.031 | 0.060 | 849.293 | 6 | 0 | 6 | 4.48 |
| `contig_window` | `src0.gate.win1024MiB.rank3` | 2 | 4.48 | 0.031 | 0.060 | 849.293 | 6 | 0 | 6 | 4.48 |
| `contig_window` | `src0.up.win512MiB.rank3` | 2 | 4.48 | 0.031 | 0.059 | 837.406 | 6 | 0 | 6 | 4.48 |
| `contig_window` | `src0.up.win1024MiB.rank3` | 2 | 4.48 | 0.031 | 0.059 | 837.406 | 6 | 0 | 6 | 4.48 |
| `contig_window` | `src0.up.win2048MiB.rank1` | 2 | 4.48 | 0.031 | 0.059 | 837.406 | 6 | 0 | 6 | 4.48 |
| `contig_window` | `src0.gate.win512MiB.rank4` | 2 | 8.97 | 0.057 | 0.117 | 829.298 | 10 | 0 | 10 | 0.00 |
| `contig_window` | `src0.gate.win1024MiB.rank4` | 2 | 8.97 | 0.057 | 0.117 | 829.298 | 10 | 0 | 10 | 0.00 |
| `contig_window` | `src0.up.win512MiB.rank4` | 2 | 8.97 | 0.057 | 0.116 | 819.450 | 10 | 0 | 10 | 0.00 |
| `contig_window` | `src0.up.win1024MiB.rank4` | 2 | 8.97 | 0.057 | 0.116 | 819.450 | 10 | 0 | 10 | 0.00 |
| `contig_window` | `src0.down.win512MiB.rank2` | 2 | 6.02 | 0.065 | 0.076 | 803.984 | 10 | 0 | 10 | 6.02 |
| `contig_window` | `src0.down.win1024MiB.rank2` | 2 | 6.02 | 0.065 | 0.076 | 803.984 | 10 | 0 | 10 | 6.02 |
| `contig_window` | `src0.gate.win512MiB.rank5` | 1 | 4.48 | 0.026 | 0.054 | 766.169 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.gate.win1024MiB.rank5` | 1 | 4.48 | 0.026 | 0.054 | 766.169 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.up.win512MiB.rank5` | 1 | 4.48 | 0.026 | 0.053 | 743.669 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.up.win1024MiB.rank5` | 1 | 4.48 | 0.026 | 0.053 | 743.669 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.gate.win512MiB.rank6` | 2 | 4.48 | 0.026 | 0.051 | 719.622 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.gate.win1024MiB.rank6` | 2 | 4.48 | 0.026 | 0.051 | 719.622 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.up.win512MiB.rank6` | 2 | 4.48 | 0.026 | 0.050 | 710.580 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.up.win1024MiB.rank6` | 2 | 4.48 | 0.026 | 0.050 | 710.580 | 5 | 0 | 5 | 0.00 |
| `contig_window` | `src0.up.win512MiB.rank15` | 1 | 4.48 | 0.022 | 0.049 | 699.526 | 4 | 0 | 4 | 0.00 |
| `contig_window` | `src0.up.win1024MiB.rank15` | 1 | 4.48 | 0.022 | 0.049 | 699.526 | 4 | 0 | 4 | 0.00 |
| `contig_window` | `src0.up.win512MiB.rank7` | 2 | 8.97 | 0.048 | 0.097 | 688.053 | 7 | 0 | 7 | 8.97 |
| `contig_window` | `src0.up.win1024MiB.rank7` | 2 | 8.97 | 0.048 | 0.097 | 688.053 | 7 | 0 | 7 | 8.97 |
| `contig_window` | `src0.up.win512MiB.rank14` | 1 | 4.48 | 0.022 | 0.048 | 685.031 | 4 | 0 | 4 | 0.00 |
| `contig_window` | `src0.up.win1024MiB.rank14` | 1 | 4.48 | 0.022 | 0.048 | 685.031 | 4 | 0 | 4 | 0.00 |
| `contig_window` | `src0.down.win512MiB.rank4` | 2 | 12.03 | 0.100 | 0.129 | 681.862 | 12 | 0 | 12 | 12.03 |
| `contig_window` | `src0.down.win1024MiB.rank4` | 2 | 12.03 | 0.100 | 0.129 | 681.862 | 12 | 0 | 12 | 12.03 |
| `contig_window` | `src0.gate.win512MiB.rank9` | 2 | 13.45 | 0.070 | 0.143 | 672.686 | 12 | 0 | 12 | 0.00 |
| `contig_window` | `src0.gate.win1024MiB.rank8` | 2 | 13.45 | 0.070 | 0.143 | 672.686 | 12 | 0 | 12 | 0.00 |
| `contig_window` | `src0.down.win512MiB.rank3` | 2 | 7.88 | 0.069 | 0.082 | 660.954 | 8 | 0 | 8 | 0.00 |

## Layer/Role Slab Upper Bound

These rows are more relevant for a pageable RAM slab than the tiny contiguous windows because they turn whole layer/role demand batches into RAM hits.

| kind | name | prompts | resident MiB | wait ms/token | hit batches | ram-only | mixed risk | VRAM overlap MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_upgate` | `blk.9.upgate` | 2 | 1407.88 | 7.174 | 124 | 124 | 0 | 138.69 |
| `layer_upgate` | `blk.1.upgate` | 2 | 1426.35 | 7.042 | 124 | 124 | 0 | 89.69 |
| `layer_upgate` | `blk.29.upgate` | 2 | 1467.01 | 6.438 | 124 | 124 | 0 | 146.69 |
| `layer_upgate` | `blk.48.upgate` | 2 | 1496.62 | 6.392 | 124 | 124 | 0 | 120.75 |
| `layer_upgate` | `blk.28.upgate` | 2 | 1417.67 | 6.249 | 124 | 124 | 0 | 126.11 |
| `layer_upgate` | `blk.7.upgate` | 2 | 1230.66 | 6.158 | 122 | 122 | 0 | 163.75 |
| `layer_upgate` | `blk.32.upgate` | 2 | 1398.07 | 6.110 | 124 | 124 | 0 | 134.20 |
| `layer_upgate` | `blk.33.upgate` | 2 | 1348.86 | 6.093 | 124 | 124 | 0 | 153.03 |
| `layer_role` | `blk.9.down` | 2 | 1126.24 | 3.721 | 62 | 62 | 0 | 118.12 |
| `layer_role` | `blk.10.down` | 2 | 1181.39 | 3.685 | 62 | 62 | 0 | 133.88 |
| `layer_role` | `blk.7.down` | 2 | 984.47 | 3.678 | 62 | 62 | 0 | 141.76 |
| `layer_role` | `blk.1.gate` | 2 | 713.18 | 3.632 | 62 | 62 | 0 | 44.84 |
| `layer_role` | `blk.6.down` | 2 | 1031.71 | 3.626 | 62 | 62 | 0 | 149.63 |
| `layer_role` | `blk.9.up` | 2 | 766.50 | 3.602 | 62 | 62 | 0 | 80.39 |
| `layer_role` | `blk.4.down` | 2 | 1130.60 | 3.580 | 62 | 62 | 0 | 89.25 |
| `layer_role` | `blk.9.gate` | 2 | 641.38 | 3.571 | 62 | 62 | 0 | 58.30 |
| `layer_all` | `blk.9.all` | 2 | 2534.12 | 10.895 | 186 | 186 | 0 | 256.81 |
| `layer_all` | `blk.1.all` | 2 | 2382.84 | 10.043 | 186 | 186 | 0 | 149.84 |
| `layer_all` | `blk.7.all` | 2 | 2215.12 | 9.836 | 184 | 184 | 0 | 305.51 |
| `layer_all` | `blk.10.all` | 2 | 2526.98 | 9.700 | 186 | 186 | 0 | 277.40 |
| `layer_all` | `blk.29.all` | 2 | 2363.48 | 9.319 | 186 | 186 | 0 | 200.83 |
| `layer_all` | `blk.26.all` | 2 | 2477.84 | 9.301 | 186 | 186 | 0 | 162.53 |
| `layer_all` | `blk.8.all` | 2 | 2445.48 | 9.231 | 186 | 186 | 0 | 256.82 |
| `layer_all` | `blk.23.all` | 2 | 2001.82 | 9.131 | 186 | 186 | 0 | 211.76 |

### Greedy `layer_upgate` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 2048 | 2 | 1812.1 | 10.16 |
| 4096 | 4 | 3767.6 | 20.59 |
| 8192 | 8 | 8001.5 | 42.73 |
| 10240 | 9 | 9409.4 | 49.90 |

### Greedy `layer_all` Budget Bound

| budget MiB | selected | resident MiB | wait ms/token |
|---:|---:|---:|---:|
| 4096 | 2 | 3218.9 | 16.00 |
| 8192 | 4 | 6613.9 | 31.94 |
| 10240 | 5 | 8468.0 | 40.51 |


## Decision Notes

- Prefer candidates with high `wait/GiB`, broad dev prompt coverage, low VRAM overlap, and many dominant/ram-only batches.
- Treat high `mixed_risk_batches` as a fragmentation warning: those candidates can reduce SSD bytes while making remaining SSD batches smaller and less efficient.
- If the greedy budget bound is only a few tens of ms/token, RAM slabs alone cannot bridge the current gap to `2 tok/s`; they should be treated as a small candidate or rejected in favor of lower-byte expert representation.
- Runtime A/B should start with pageable RAM and must reject candidates that raise TTFT, refaults, direct reclaim, or aggregate demand wait.
