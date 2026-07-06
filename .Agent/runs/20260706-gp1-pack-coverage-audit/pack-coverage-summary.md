# Kimi expert-pack coverage audit

This report compares route-trace events against expert-pack index keys.
It does not run inference and does not inspect held-out test prompts.

## Pack Sources

| path | entries | new unique keys | duplicates | new unique payload GiB |
|---|---:|---:|---:|---:|
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | 30831 | 30831 | 0 | 163.10 |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | 768 | 768 | 0 | 4.51 |

## Prompt Coverage

| prompt | events | event hit | GiB | byte hit | miss GiB |
|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | 106592 | 99.4% | 553.28 | 99.4% | 3.20 |
| `dev_japan_factual` | 117664 | 82.3% | 610.75 | 82.4% | 107.51 |
| `dev_linear_equation` | 47080 | 54.1% | 244.38 | 54.1% | 112.22 |
| `dev_mixed_summary` | 74760 | 58.3% | 388.05 | 58.5% | 161.17 |
| `dev_photosynthesis_factual` | 130120 | 58.9% | 675.40 | 59.0% | 276.74 |
| `dev_python_reverse` | 131504 | 53.9% | 682.58 | 54.0% | 313.99 |
| `dev_zh_france` | 67840 | 79.5% | 352.13 | 79.4% | 72.41 |

## Aggregate

- events: `675560`
- event hit rate: `70.1%`
- routed bytes: `3506.57 GiB`
- byte hit rate: `70.1%`
- miss bytes: `1047.24 GiB`

## By Role

| role | events | event hit | GiB | byte hit | miss GiB |
|---|---:|---:|---:|---:|---:|
| down | 206968 | 70.7% | 1280.91 | 70.7% | 374.92 |
| gate | 234296 | 69.8% | 1156.20 | 69.7% | 350.30 |
| up | 234296 | 69.8% | 1069.46 | 69.9% | 322.02 |

## Top Missing Layer/Role Buckets

| layer | role | miss events | miss GiB | hit events | byte hit |
|---:|---|---:|---:|---:|---:|
| 57 | down | 1294 | 9.40 | 2610 | 66.9% |
| 19 | down | 1266 | 9.20 | 2638 | 67.6% |
| 58 | down | 1266 | 9.20 | 2638 | 67.6% |
| 16 | down | 1241 | 9.01 | 2663 | 68.2% |
| 21 | down | 1221 | 8.87 | 2683 | 68.7% |
| 43 | down | 1501 | 8.82 | 2403 | 61.6% |
| 45 | down | 1496 | 8.79 | 2408 | 61.7% |
| 40 | down | 1456 | 8.55 | 2448 | 62.7% |
| 46 | down | 1436 | 8.44 | 2468 | 63.2% |
| 39 | down | 1405 | 8.25 | 2499 | 64.0% |
| 52 | down | 1354 | 7.95 | 2550 | 65.3% |
| 44 | down | 1351 | 7.94 | 2553 | 65.4% |
| 43 | gate | 1501 | 7.86 | 2403 | 61.6% |
| 37 | down | 1334 | 7.84 | 2570 | 65.8% |
| 45 | gate | 1496 | 7.83 | 2408 | 61.7% |
| 55 | down | 1329 | 7.81 | 2575 | 66.0% |
| 60 | down | 1321 | 7.76 | 2639 | 66.6% |
| 13 | down | 1319 | 7.75 | 2585 | 66.2% |
| 26 | down | 1061 | 7.71 | 2843 | 72.8% |
| 35 | down | 1302 | 7.65 | 2602 | 66.6% |
| 40 | gate | 1456 | 7.62 | 2448 | 62.7% |
| 23 | down | 1043 | 7.58 | 2861 | 73.3% |
| 46 | gate | 1436 | 7.52 | 2468 | 63.2% |
| 48 | down | 1275 | 7.49 | 2629 | 67.3% |
| 34 | down | 1264 | 7.43 | 2640 | 67.6% |
| 39 | gate | 1405 | 7.35 | 2499 | 64.0% |
| 24 | down | 1011 | 7.34 | 2893 | 74.1% |
| 41 | down | 1244 | 7.31 | 2660 | 68.1% |
| 25 | down | 1006 | 7.31 | 2898 | 74.2% |
| 54 | down | 1239 | 7.28 | 2665 | 68.3% |
| 17 | down | 1234 | 7.25 | 2670 | 68.4% |
| 29 | down | 1230 | 7.23 | 2674 | 68.5% |
| 30 | down | 1230 | 7.23 | 2674 | 68.5% |
| 51 | down | 1225 | 7.20 | 2679 | 68.6% |
| 56 | down | 1221 | 7.17 | 2683 | 68.7% |
| 47 | down | 1217 | 7.15 | 2687 | 68.8% |
| 52 | gate | 1354 | 7.09 | 2550 | 65.3% |
| 44 | gate | 1351 | 7.07 | 2553 | 65.4% |
| 20 | down | 969 | 7.04 | 2935 | 75.2% |
| 49 | down | 1197 | 7.03 | 2707 | 69.3% |

## Top Missing Keys

| count | GiB | tensor | expert | bytes |
|---:|---:|---|---:|---:|
| 142 | 0.62 | `blk.45.ffn_up_exps.weight` | 90 | 4702208 |
| 142 | 0.74 | `blk.45.ffn_gate_exps.weight` | 90 | 5619712 |
| 142 | 0.83 | `blk.45.ffn_down_exps.weight` | 90 | 6307840 |
| 115 | 0.50 | `blk.55.ffn_up_exps.weight` | 310 | 4702208 |
| 115 | 0.60 | `blk.55.ffn_gate_exps.weight` | 310 | 5619712 |
| 115 | 0.68 | `blk.55.ffn_down_exps.weight` | 310 | 6307840 |
| 111 | 0.49 | `blk.35.ffn_up_exps.weight` | 167 | 4702208 |
| 111 | 0.58 | `blk.35.ffn_gate_exps.weight` | 167 | 5619712 |
| 111 | 0.65 | `blk.35.ffn_down_exps.weight` | 167 | 6307840 |
| 106 | 0.46 | `blk.36.ffn_up_exps.weight` | 19 | 4702208 |
| 106 | 0.55 | `blk.36.ffn_gate_exps.weight` | 19 | 5619712 |
| 106 | 0.62 | `blk.36.ffn_down_exps.weight` | 19 | 6307840 |
| 103 | 0.45 | `blk.39.ffn_up_exps.weight` | 260 | 4702208 |
| 103 | 0.54 | `blk.39.ffn_gate_exps.weight` | 260 | 5619712 |
| 103 | 0.61 | `blk.39.ffn_down_exps.weight` | 260 | 6307840 |
| 103 | 0.45 | `blk.43.ffn_up_exps.weight` | 346 | 4702208 |
| 103 | 0.54 | `blk.43.ffn_gate_exps.weight` | 346 | 5619712 |
| 103 | 0.61 | `blk.43.ffn_down_exps.weight` | 346 | 6307840 |
| 100 | 0.52 | `blk.5.ffn_up_exps.weight` | 187 | 5619712 |
| 100 | 0.52 | `blk.5.ffn_gate_exps.weight` | 187 | 5619712 |
| 100 | 0.59 | `blk.5.ffn_down_exps.weight` | 187 | 6307840 |
| 97 | 0.42 | `blk.38.ffn_up_exps.weight` | 361 | 4702208 |
| 97 | 0.51 | `blk.38.ffn_gate_exps.weight` | 361 | 5619712 |
| 97 | 0.57 | `blk.38.ffn_down_exps.weight` | 361 | 6307840 |
| 97 | 0.42 | `blk.43.ffn_up_exps.weight` | 157 | 4702208 |
| 97 | 0.51 | `blk.43.ffn_gate_exps.weight` | 157 | 5619712 |
| 97 | 0.57 | `blk.43.ffn_down_exps.weight` | 157 | 6307840 |
| 95 | 0.42 | `blk.51.ffn_up_exps.weight` | 7 | 4702208 |
| 95 | 0.50 | `blk.51.ffn_gate_exps.weight` | 7 | 5619712 |
| 95 | 0.56 | `blk.51.ffn_down_exps.weight` | 7 | 6307840 |
| 94 | 0.41 | `blk.37.ffn_up_exps.weight` | 248 | 4702208 |
| 94 | 0.49 | `blk.37.ffn_gate_exps.weight` | 248 | 5619712 |
| 94 | 0.55 | `blk.37.ffn_down_exps.weight` | 248 | 6307840 |
| 93 | 0.49 | `blk.3.ffn_up_exps.weight` | 259 | 5619712 |
| 93 | 0.49 | `blk.3.ffn_gate_exps.weight` | 259 | 5619712 |
| 93 | 0.55 | `blk.3.ffn_down_exps.weight` | 259 | 6307840 |
| 92 | 0.40 | `blk.40.ffn_up_exps.weight` | 382 | 4702208 |
| 92 | 0.48 | `blk.40.ffn_gate_exps.weight` | 382 | 5619712 |
| 92 | 0.54 | `blk.40.ffn_down_exps.weight` | 382 | 6307840 |
| 91 | 0.48 | `blk.56.ffn_up_exps.weight` | 124 | 5619712 |

## Interpretation

- `pack_hit` coverage is independent from VRAM-cache coverage. This audit
  only measures whether a routed expert can use the optimized expert-pack
  source path after it misses VRAM.
- High miss bytes here mean the runtime must materialize selected experts
  from GGUF-backed tensor storage for those route events.
- A model-wide same-quant pack would remove these misses but currently needs
  separate disk feasibility. A GGUF-offset alias pack would target the same
  miss bytes without duplicating the payload.
