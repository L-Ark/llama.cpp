# Kimi Low-Byte Expert Bound

## Inputs

- traces: `2`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_france_regression/io-read-trace.csv`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_intelligence_general/io-read-trace.csv`
- batches: `11134`
- decode runs: `62`
- baseline decode: `42527.100 ms`
- baseline token rate: `1.458 tok/s`
- total profiled IO wait: `23006.531 ms`
- total profiled payload: `277.515 GiB`

## Role Payload

| role | payload GiB | max linear wait ms | max linear wait ms/token | eliminate-role bound tok/s |
|---|---:|---:|---:|---:|
| down | 125.563 | 7026.910 | 113.337 | 1.746 |
| gate | 78.777 | 8398.003 | 135.452 | 1.817 |
| up | 73.175 | 7581.618 | 122.284 | 1.774 |

## Byte Reduction Scenarios

| scenario | byte reduction | saved ms/token | bounded tok/s |
|---|---:|---:|---:|
| gate | 40% | 54.181 | 1.583 |
| gate | 42% | 56.890 | 1.590 |
| gate | 45% | 60.953 | 1.600 |
| gate | 48% | 65.017 | 1.611 |
| gate | 50% | 67.726 | 1.618 |
| up | 40% | 48.914 | 1.570 |
| up | 42% | 51.359 | 1.576 |
| up | 45% | 55.028 | 1.585 |
| up | 48% | 58.696 | 1.594 |
| up | 50% | 61.142 | 1.601 |
| down | 40% | 45.335 | 1.561 |
| down | 42% | 47.602 | 1.567 |
| down | 45% | 51.002 | 1.575 |
| down | 48% | 54.402 | 1.583 |
| down | 50% | 56.669 | 1.589 |
| up+gate | 40% | 103.094 | 1.716 |
| up+gate | 42% | 108.249 | 1.731 |
| up+gate | 45% | 115.981 | 1.755 |
| up+gate | 48% | 123.713 | 1.779 |
| up+gate | 50% | 128.868 | 1.795 |
| all | 40% | 148.429 | 1.860 |
| all | 42% | 155.851 | 1.887 |
| all | 45% | 166.983 | 1.927 |
| all | 48% | 178.115 | 1.969 |
| all | 50% | 185.537 | 1.998 |

## Target Feasibility

| target tok/s | required saving ms/token | all-role required reduction | possible by IO-byte reduction only |
|---:|---:|---:|---|
| 2.0 | 185.921 | 50.1% | True |
| 5.0 | 485.921 | 131.0% | False |

## Top Layer/Role Linear Contributions

| rank | layer_role | payload GiB | max linear wait ms/token | eliminate bound tok/s |
|---:|---|---:|---:|---:|
| 1 | `blk.1.gate` | 1.402 | 3.195 | 1.465 |
| 2 | `blk.16.gate` | 1.143 | 2.866 | 1.464 |
| 3 | `blk.1.up` | 1.402 | 2.818 | 1.464 |
| 4 | `blk.29.gate` | 1.602 | 2.752 | 1.464 |
| 5 | `blk.39.gate` | 1.262 | 2.702 | 1.464 |
| 6 | `blk.32.gate` | 1.507 | 2.699 | 1.464 |
| 7 | `blk.28.gate` | 1.555 | 2.682 | 1.464 |
| 8 | `blk.9.up` | 1.555 | 2.655 | 1.464 |
| 9 | `blk.48.gate` | 1.466 | 2.605 | 1.463 |
| 10 | `blk.33.gate` | 1.513 | 2.553 | 1.463 |
| 11 | `blk.30.gate` | 1.487 | 2.551 | 1.463 |
| 12 | `blk.9.gate` | 1.305 | 2.516 | 1.463 |
| 13 | `blk.34.gate` | 1.471 | 2.510 | 1.463 |
| 14 | `blk.51.gate` | 1.408 | 2.474 | 1.463 |
| 15 | `blk.31.gate` | 1.361 | 2.472 | 1.463 |
| 16 | `blk.52.gate` | 1.377 | 2.461 | 1.463 |
| 17 | `blk.7.down` | 2.853 | 2.453 | 1.463 |
| 18 | `blk.7.up` | 1.387 | 2.437 | 1.463 |
| 19 | `blk.36.gate` | 1.403 | 2.435 | 1.463 |
| 20 | `blk.2.gate` | 1.523 | 2.428 | 1.463 |
| 21 | `blk.53.gate` | 1.298 | 2.417 | 1.463 |
| 22 | `blk.14.up` | 1.038 | 2.409 | 1.463 |
| 23 | `blk.43.gate` | 1.303 | 2.396 | 1.463 |
| 24 | `blk.38.gate` | 1.293 | 2.395 | 1.463 |
| 25 | `blk.7.gate` | 1.161 | 2.394 | 1.463 |
| 26 | `blk.35.gate` | 1.340 | 2.391 | 1.463 |
| 27 | `blk.6.down` | 2.861 | 2.384 | 1.463 |
| 28 | `blk.14.gate` | 1.241 | 2.382 | 1.463 |
| 29 | `blk.12.gate` | 1.288 | 2.378 | 1.463 |
| 30 | `blk.8.up` | 1.387 | 2.375 | 1.463 |
| 31 | `blk.54.gate` | 1.225 | 2.355 | 1.463 |
| 32 | `blk.10.gate` | 1.345 | 2.348 | 1.463 |
| 33 | `blk.44.gate` | 1.324 | 2.341 | 1.463 |
| 34 | `blk.26.gate` | 1.323 | 2.338 | 1.463 |
| 35 | `blk.10.down` | 2.792 | 2.323 | 1.463 |
| 36 | `blk.41.gate` | 1.241 | 2.322 | 1.463 |
| 37 | `blk.49.gate` | 1.251 | 2.313 | 1.463 |
| 38 | `blk.8.gate` | 1.161 | 2.310 | 1.463 |
| 39 | `blk.40.gate` | 1.277 | 2.308 | 1.463 |
| 40 | `blk.9.down` | 2.730 | 2.305 | 1.463 |

## Interpretation

- All-role 50% byte reduction is bounded at 1.998 tok/s; 75% reduction is bounded at 2.453 tok/s.
- Eliminating all profiled IO wait entirely is bounded at 3.176 tok/s, so reaching 5 tok/s cannot be achieved by IO-byte reduction alone on this baseline.
- This is a linear exposed-wait bound, not a quality result. Any lower-byte runtime change still must pass cold-start generalized-prompt quality, TTFT and 16GB host-RAM gates before it can be SOTA.
