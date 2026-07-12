# Kimi Budgeted Hot Expert Profile

This is a dev-only offline candidate profile. It does not use held-out
prompts and does not claim a SOTA result.

## Inputs

- route profiles: `2`
- selected layer/role buckets: `24`
- budgets MiB: `{"down": 128.0, "upgate": 384.0}`

## Selected By Role

| role | entries | MiB | route count | budget MiB |
|---|---:|---:|---:|---:|
| down | 16 | 122.94 | 1355 | 128.0 |
| upgate | 79 | 383.14 | 5213 | 384.0 |

## Top Selected Buckets

| layer | role | entries | MiB | route count | source ms/token |
|---:|---|---:|---:|---:|---:|
| 32 | upgate | 10 | 48.34 | 592 | 9.324 |
| 34 | upgate | 9 | 43.86 | 582 | 9.010 |
| 14 | upgate | 8 | 39.38 | 644 | 10.021 |
| 29 | upgate | 8 | 38.50 | 452 | 9.567 |
| 31 | upgate | 7 | 34.02 | 530 | 9.292 |
| 33 | upgate | 7 | 34.02 | 481 | 9.123 |
| 28 | upgate | 7 | 34.02 | 392 | 9.633 |
| 30 | upgate | 7 | 33.14 | 430 | 9.449 |
| 54 | upgate | 6 | 29.53 | 382 | 9.464 |
| 10 | down | 3 | 23.62 | 245 | 5.619 |
| 48 | upgate | 4 | 19.69 | 372 | 9.063 |
| 53 | upgate | 4 | 19.69 | 262 | 8.982 |
| 8 | down | 2 | 15.75 | 216 | 5.425 |
| 15 | down | 2 | 15.75 | 154 | 5.162 |
| 25 | down | 2 | 14.88 | 180 | 5.221 |
| 23 | down | 2 | 14.88 | 149 | 5.118 |
| 52 | upgate | 2 | 8.97 | 94 | 9.155 |
| 6 | down | 1 | 7.88 | 79 | 5.808 |
| 7 | down | 1 | 7.88 | 71 | 5.691 |
| 26 | down | 1 | 7.44 | 90 | 5.354 |
| 4 | down | 1 | 7.44 | 80 | 5.884 |
| 24 | down | 1 | 7.44 | 91 | 5.122 |

## Top Entries

| rank | role | layer | tensor | expert | count | MiB | score |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | upgate | 14 | `blk.14.ffn_up_exps.weight` | 110 | 111 | 4.48 | 248.046 |
| 2 | upgate | 14 | `blk.14.ffn_up_exps.weight` | 272 | 94 | 4.48 | 210.057 |
| 3 | upgate | 48 | `blk.48.ffn_up_exps.weight` | 5 | 103 | 4.48 | 208.165 |
| 4 | upgate | 14 | `blk.14.ffn_gate_exps.weight` | 110 | 111 | 5.36 | 207.549 |
| 5 | upgate | 31 | `blk.31.ffn_up_exps.weight` | 330 | 97 | 4.48 | 200.992 |
| 6 | upgate | 14 | `blk.14.ffn_gate_exps.weight` | 272 | 94 | 5.36 | 175.762 |
| 7 | upgate | 48 | `blk.48.ffn_gate_exps.weight` | 5 | 103 | 5.36 | 174.179 |
| 8 | upgate | 33 | `blk.33.ffn_up_exps.weight` | 25 | 85 | 4.48 | 172.924 |
| 9 | upgate | 30 | `blk.30.ffn_up_exps.weight` | 246 | 82 | 4.48 | 172.782 |
| 10 | upgate | 31 | `blk.31.ffn_gate_exps.weight` | 330 | 97 | 5.36 | 168.177 |
| 11 | upgate | 48 | `blk.48.ffn_up_exps.weight` | 340 | 83 | 4.48 | 167.744 |
| 12 | upgate | 54 | `blk.54.ffn_up_exps.weight` | 33 | 73 | 4.48 | 154.062 |
| 13 | upgate | 31 | `blk.31.ffn_up_exps.weight` | 257 | 73 | 4.48 | 151.262 |
| 14 | upgate | 34 | `blk.34.ffn_up_exps.weight` | 175 | 75 | 4.48 | 150.690 |
| 15 | upgate | 34 | `blk.34.ffn_up_exps.weight` | 246 | 75 | 4.48 | 150.690 |
| 16 | upgate | 32 | `blk.32.ffn_up_exps.weight` | 37 | 72 | 4.48 | 149.704 |
| 17 | upgate | 31 | `blk.31.ffn_up_exps.weight` | 371 | 71 | 4.48 | 147.118 |
| 18 | upgate | 33 | `blk.33.ffn_gate_exps.weight` | 25 | 85 | 5.36 | 144.691 |
| 19 | upgate | 30 | `blk.30.ffn_gate_exps.weight` | 246 | 82 | 5.36 | 144.572 |
| 20 | upgate | 33 | `blk.33.ffn_up_exps.weight` | 263 | 70 | 4.48 | 142.408 |
| 21 | upgate | 48 | `blk.48.ffn_gate_exps.weight` | 340 | 83 | 5.36 | 140.358 |
| 22 | upgate | 28 | `blk.28.ffn_up_exps.weight` | 213 | 64 | 4.48 | 137.480 |
| 23 | upgate | 29 | `blk.29.ffn_up_exps.weight` | 305 | 64 | 4.48 | 136.538 |
| 24 | upgate | 53 | `blk.53.ffn_up_exps.weight` | 115 | 68 | 4.48 | 136.201 |
| 25 | upgate | 32 | `blk.32.ffn_up_exps.weight` | 354 | 65 | 4.48 | 135.149 |
| 26 | upgate | 14 | `blk.14.ffn_up_exps.weight` | 286 | 60 | 4.48 | 134.079 |
| 27 | upgate | 54 | `blk.54.ffn_gate_exps.weight` | 33 | 73 | 5.36 | 128.909 |
| 28 | upgate | 54 | `blk.54.ffn_up_exps.weight` | 20 | 61 | 4.48 | 128.737 |
| 29 | upgate | 14 | `blk.14.ffn_up_exps.weight` | 132 | 57 | 4.48 | 127.375 |
| 30 | upgate | 31 | `blk.31.ffn_gate_exps.weight` | 257 | 73 | 5.36 | 126.566 |
| 31 | upgate | 30 | `blk.30.ffn_up_exps.weight` | 204 | 60 | 4.48 | 126.426 |
| 32 | upgate | 53 | `blk.53.ffn_up_exps.weight` | 132 | 63 | 4.48 | 126.186 |
| 33 | upgate | 34 | `blk.34.ffn_gate_exps.weight` | 175 | 75 | 5.36 | 126.087 |
| 34 | upgate | 34 | `blk.34.ffn_gate_exps.weight` | 246 | 75 | 5.36 | 126.087 |
| 35 | upgate | 29 | `blk.29.ffn_up_exps.weight` | 130 | 59 | 4.48 | 125.871 |
| 36 | upgate | 32 | `blk.32.ffn_gate_exps.weight` | 37 | 72 | 5.36 | 125.262 |
| 37 | upgate | 31 | `blk.31.ffn_gate_exps.weight` | 371 | 71 | 5.36 | 123.099 |
| 38 | upgate | 28 | `blk.28.ffn_up_exps.weight` | 124 | 57 | 4.48 | 122.443 |
| 39 | upgate | 33 | `blk.33.ffn_up_exps.weight` | 360 | 60 | 4.48 | 122.064 |
| 40 | upgate | 29 | `blk.29.ffn_up_exps.weight` | 363 | 57 | 4.48 | 121.604 |
| 41 | upgate | 34 | `blk.34.ffn_up_exps.weight` | 289 | 60 | 4.48 | 120.552 |
| 42 | upgate | 54 | `blk.54.ffn_up_exps.weight` | 245 | 57 | 4.48 | 120.295 |
| 43 | upgate | 33 | `blk.33.ffn_gate_exps.weight` | 263 | 70 | 5.36 | 119.158 |
| 44 | upgate | 32 | `blk.32.ffn_up_exps.weight` | 92 | 57 | 4.48 | 118.516 |
| 45 | upgate | 28 | `blk.28.ffn_gate_exps.weight` | 213 | 64 | 5.36 | 115.034 |
| 46 | upgate | 29 | `blk.29.ffn_gate_exps.weight` | 305 | 64 | 5.36 | 114.246 |
| 47 | upgate | 53 | `blk.53.ffn_gate_exps.weight` | 115 | 68 | 5.36 | 113.964 |
| 48 | upgate | 32 | `blk.32.ffn_gate_exps.weight` | 354 | 65 | 5.36 | 113.084 |
| 49 | upgate | 34 | `blk.34.ffn_up_exps.weight` | 67 | 56 | 4.48 | 112.515 |
| 50 | upgate | 14 | `blk.14.ffn_gate_exps.weight` | 286 | 60 | 5.36 | 112.188 |
| 51 | upgate | 28 | `blk.28.ffn_up_exps.weight` | 138 | 52 | 4.48 | 111.703 |
| 52 | upgate | 32 | `blk.32.ffn_up_exps.weight` | 353 | 53 | 4.48 | 110.199 |
| 53 | upgate | 54 | `blk.54.ffn_gate_exps.weight` | 20 | 61 | 5.36 | 107.719 |
| 54 | upgate | 30 | `blk.30.ffn_up_exps.weight` | 185 | 51 | 4.48 | 107.462 |
| 55 | upgate | 14 | `blk.14.ffn_gate_exps.weight` | 132 | 57 | 5.36 | 106.579 |
| 56 | upgate | 30 | `blk.30.ffn_gate_exps.weight` | 204 | 60 | 5.36 | 105.785 |
| 57 | upgate | 53 | `blk.53.ffn_gate_exps.weight` | 132 | 63 | 5.36 | 105.584 |
| 58 | upgate | 30 | `blk.30.ffn_up_exps.weight` | 363 | 50 | 4.48 | 105.355 |
| 59 | upgate | 29 | `blk.29.ffn_gate_exps.weight` | 130 | 59 | 5.36 | 105.321 |
| 60 | upgate | 29 | `blk.29.ffn_up_exps.weight` | 217 | 49 | 4.48 | 104.537 |
| 61 | upgate | 32 | `blk.32.ffn_up_exps.weight` | 270 | 50 | 4.48 | 103.961 |
| 62 | upgate | 33 | `blk.33.ffn_up_exps.weight` | 241 | 51 | 4.48 | 103.754 |
| 63 | upgate | 28 | `blk.28.ffn_gate_exps.weight` | 124 | 57 | 5.36 | 102.452 |
| 64 | upgate | 33 | `blk.33.ffn_gate_exps.weight` | 360 | 60 | 5.36 | 102.135 |
| 65 | upgate | 29 | `blk.29.ffn_gate_exps.weight` | 363 | 57 | 5.36 | 101.750 |
| 66 | upgate | 34 | `blk.34.ffn_gate_exps.weight` | 289 | 60 | 5.36 | 100.870 |
| 67 | upgate | 54 | `blk.54.ffn_gate_exps.weight` | 245 | 57 | 5.36 | 100.655 |
| 68 | upgate | 34 | `blk.34.ffn_up_exps.weight` | 123 | 50 | 4.48 | 100.460 |
| 69 | upgate | 52 | `blk.52.ffn_up_exps.weight` | 284 | 49 | 4.48 | 100.035 |
| 70 | upgate | 32 | `blk.32.ffn_up_exps.weight` | 246 | 48 | 4.48 | 99.803 |
| 71 | upgate | 31 | `blk.31.ffn_up_exps.weight` | 42 | 48 | 4.48 | 99.460 |
| 72 | upgate | 32 | `blk.32.ffn_gate_exps.weight` | 92 | 57 | 5.36 | 99.166 |
| 73 | upgate | 28 | `blk.28.ffn_up_exps.weight` | 200 | 46 | 4.48 | 98.814 |
| 74 | upgate | 30 | `blk.30.ffn_up_exps.weight` | 55 | 45 | 4.48 | 94.819 |
| 75 | upgate | 34 | `blk.34.ffn_gate_exps.weight` | 67 | 56 | 5.36 | 94.145 |
| 76 | upgate | 28 | `blk.28.ffn_gate_exps.weight` | 138 | 52 | 5.36 | 93.465 |
| 77 | upgate | 32 | `blk.32.ffn_gate_exps.weight` | 353 | 53 | 5.36 | 92.207 |
| 78 | upgate | 52 | `blk.52.ffn_up_exps.weight` | 33 | 45 | 4.48 | 91.869 |
| 79 | upgate | 29 | `blk.29.ffn_up_exps.weight` | 92 | 43 | 4.48 | 91.737 |
| 80 | down | 8 | `blk.8.ffn_down_exps.weight` | 356 | 112 | 7.88 | 77.156 |

Interpretation:

- This profile is intentionally smaller than whole-layer admission.
- A runtime A/B must verify whether these pinned/preloaded entries reduce
  decode wall without increasing TTFT, RAM pressure, or residual time.
