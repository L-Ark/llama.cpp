# Kimi copy-profile RAM tier oracle

This is a dev-only offline upper bound. It does not change runtime behavior or claim SOTA.

- root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-dev-n32-035310`
- decode: `40585.85 ms / 62`
- baseline tok/s: `1.528`
- target tok/s: `2.000`
- needed saved: `9585.85 ms` total, `154.610 ms/token`
- candidate keys after min-prompt filter: `13473`
- candidate traffic: `405.525 GiB`
- candidate io_wait: `253126.55 ms`

## Budget Bound

| budget MiB | resident GiB | entries | traffic GiB | opt saved ms/token | opt tok/s | conservative saved ms/token | conservative tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.996 | 193 | 10.892 | 153.018 | 1.994 | 145.698 | 1.965 |
| 2048 | 1.997 | 381 | 20.321 | 262.855 | 2.553 | 249.198 | 2.467 |
| 4096 | 3.996 | 764 | 36.376 | 439.381 | 4.646 | 414.934 | 4.172 |
| 6144 | 5.997 | 1146 | 50.299 | 585.334 | 14.435 | 551.531 | 9.701 |
| 8192 | 7.998 | 1523 | 62.437 | 712.007 | 62000.000 | 670.047 | 62000.000 |
| 10240 | 9.996 | 1913 | 73.231 | 825.744 | 62000.000 | 776.530 | 62000.000 |

## Decision

- `candidate_for_io_trace_or_runtime_ab`: conservative bound clears target; require IO batch fragmentation analysis next.

Caveats:

- Row-level optimistic saving assumes each selected RAM entry removes its full profiled `io_wait_ms`.
- Conservative saving subtracts RAM->VRAM H2D at the configured bandwidth, because CPU/RAM residency does not remove H2D.
- This oracle does not model mixed SSD/RAM batch fragmentation; if a budget looks promising, the next step is an `io-read-trace.csv` complete-batch oracle before runtime A/B.

## Best Budget Top Entries

Best optimistic budget: `8192 MiB`

| rank | tensor | expert | prompts | calls | traffic GiB | io wait ms | score ms/MiB |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | `blk.1.ffn_up_exps.weight` | 178 | dev_france_regression;dev_intelligence_general | 4 | 0.018 | 124.79 | 27.828 |
| 2 | `blk.1.ffn_up_exps.weight` | 265 | dev_france_regression;dev_intelligence_general | 3 | 0.013 | 102.03 | 22.751 |
| 3 | `blk.1.ffn_up_exps.weight` | 257 | dev_france_regression;dev_intelligence_general | 3 | 0.013 | 90.36 | 20.149 |
| 4 | `blk.1.ffn_up_exps.weight` | 121 | dev_france_regression;dev_intelligence_general | 6 | 0.026 | 83.02 | 18.514 |
| 5 | `blk.1.ffn_up_exps.weight` | 34 | dev_france_regression;dev_intelligence_general | 4 | 0.018 | 82.43 | 18.381 |
| 6 | `blk.1.ffn_up_exps.weight` | 134 | dev_france_regression;dev_intelligence_general | 6 | 0.026 | 69.46 | 15.489 |
| 7 | `blk.1.ffn_up_exps.weight` | 163 | dev_france_regression;dev_intelligence_general | 2 | 0.009 | 66.97 | 14.934 |
| 8 | `blk.1.ffn_up_exps.weight` | 116 | dev_france_regression;dev_intelligence_general | 9 | 0.039 | 60.49 | 13.489 |
| 9 | `blk.1.ffn_up_exps.weight` | 261 | dev_france_regression;dev_intelligence_general | 4 | 0.018 | 60.29 | 13.445 |
| 10 | `blk.48.ffn_gate_exps.weight` | 363 | dev_france_regression;dev_intelligence_general | 6 | 0.031 | 67.57 | 12.608 |
| 11 | `blk.3.ffn_up_exps.weight` | 17 | dev_france_regression;dev_intelligence_general | 5 | 0.026 | 66.28 | 12.367 |
| 12 | `blk.7.ffn_down_exps.weight` | 288 | dev_france_regression;dev_intelligence_general | 17 | 0.131 | 93.64 | 11.891 |
| 13 | `blk.29.ffn_down_exps.weight` | 363 | dev_france_regression;dev_intelligence_general | 19 | 0.112 | 71.28 | 11.850 |
| 14 | `blk.7.ffn_down_exps.weight` | 235 | dev_france_regression;dev_intelligence_general | 20 | 0.154 | 92.91 | 11.797 |
| 15 | `blk.29.ffn_up_exps.weight` | 363 | dev_france_regression;dev_intelligence_general | 13 | 0.057 | 52.63 | 11.735 |
| 16 | `blk.32.ffn_down_exps.weight` | 353 | dev_france_regression;dev_intelligence_general | 19 | 0.112 | 70.39 | 11.702 |
| 17 | `blk.1.ffn_down_exps.weight` | 116 | dev_france_regression;dev_intelligence_general | 22 | 0.129 | 69.07 | 11.481 |
| 18 | `blk.6.ffn_down_exps.weight` | 356 | dev_france_regression;dev_intelligence_general | 17 | 0.131 | 90.32 | 11.469 |
| 19 | `blk.1.ffn_gate_exps.weight` | 23 | dev_france_regression;dev_intelligence_general | 9 | 0.039 | 50.79 | 11.326 |
| 20 | `blk.6.ffn_down_exps.weight` | 84 | dev_france_regression;dev_intelligence_general | 21 | 0.161 | 88.96 | 11.296 |
| 21 | `blk.60.ffn_up_exps.weight` | 298 | dev_france_regression;dev_intelligence_general | 2 | 0.010 | 60.30 | 11.251 |
| 22 | `blk.29.ffn_gate_exps.weight` | 363 | dev_france_regression;dev_intelligence_general | 13 | 0.068 | 60.06 | 11.206 |
| 23 | `blk.24.ffn_gate_exps.weight` | 344 | dev_france_regression;dev_intelligence_general | 11 | 0.048 | 50.22 | 11.198 |
| 24 | `blk.8.ffn_down_exps.weight` | 356 | dev_france_regression;dev_intelligence_general | 12 | 0.092 | 85.29 | 10.830 |
| 25 | `blk.17.ffn_gate_exps.weight` | 317 | dev_france_regression;dev_intelligence_general | 11 | 0.048 | 48.14 | 10.734 |
| 26 | `blk.9.ffn_gate_exps.weight` | 256 | dev_france_regression;dev_intelligence_general | 11 | 0.048 | 47.95 | 10.692 |
| 27 | `blk.13.ffn_down_exps.weight` | 270 | dev_france_regression;dev_intelligence_general | 20 | 0.117 | 64.14 | 10.662 |
| 28 | `blk.25.ffn_gate_exps.weight` | 340 | dev_france_regression;dev_intelligence_general | 12 | 0.053 | 47.76 | 10.651 |
| 29 | `blk.25.ffn_up_exps.weight` | 340 | dev_france_regression;dev_intelligence_general | 12 | 0.053 | 47.54 | 10.602 |
| 30 | `blk.24.ffn_up_exps.weight` | 344 | dev_france_regression;dev_intelligence_general | 11 | 0.048 | 47.45 | 10.582 |