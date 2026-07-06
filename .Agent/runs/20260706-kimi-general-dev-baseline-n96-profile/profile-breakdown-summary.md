# Kimi general dev profile breakdown

This report uses dev baseline profile artifacts only. It does not use held-out test prompts.

## Prompt Summary

| prompt | tok/s | decode ms | down wall ms | upgate wall ms | fallback ms | top decline reason |
|---|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | 1.26 | 61015 | 11238 | 16799 | 78497 | multirow_not_supported:52 |
| `dev_japan_factual` | 0.42 | 202575 | 36077 | 54372 | 84353 | multirow_not_supported:52 |
| `dev_linear_equation` | 0.16 | 218566 | 41978 | 57519 | 88862 | multirow_not_supported:52 |
| `dev_mixed_summary` | 0.20 | 266891 | 52047 | 72471 | 99895 | multirow_not_supported:52 |
| `dev_photosynthesis_factual` | 0.24 | 393774 | 74034 | 102503 | 76611 | multirow_not_supported:52 |
| `dev_python_reverse` | 0.17 | 566553 | 109370 | 145502 | 96938 | multirow_not_supported:52 |
| `dev_zh_france` | 0.38 | 128862 | 24344 | 36166 | 73254 | multirow_not_supported:52 |

## Down Batch By Type

| src0_type | role | calls | wall ms | stage ms | kernel ms | staged jobs | hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | down | 5856 | 194381 | 193221 | 554 | 29035 | 38.0% |
| 11 | down | 20015 | 154705 | 150476 | 2690 | 30220 | 81.1% |

## Up/Gate Batch By Type Pair

| type pair | calls | wall ms | stage ms | kernel ms | up ms | gate ms | up wait ms | gate wait ms | hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| up18_gate18 | 4887 | 265666 | 285 | 264981 | 138429 | 126475 | 0 | 0 | 34.2% |
| up22_gate22 | 8784 | 219665 | 440 | 218734 | 206250 | 12280 | 204775 | 217496 | 39.0% |

## Fallback By Phase / Type / Role

| phase | src0_type | role | fallback ms | calls | GGUF calls | pack mmap calls | bytes GiB |
|---|---:|---|---:|---:|---:|---:|---:|
| prompt | 11 | down | 155266 | 22570 | 22570 | 0 | 132.59 |
| prompt | 22 | up | 109051 | 25834 | 25834 | 0 | 113.13 |
| prompt | 18 | gate | 101867 | 21828 | 21828 | 0 | 114.24 |
| decode | 2 | down | 78911 | 27328 | 7774 | 19554 | 210.16 |
| prompt | 23 | down | 55655 | 6433 | 6433 | 0 | 46.72 |
| prompt | 22 | gate | 43320 | 10508 | 10508 | 0 | 46.02 |
| prompt | 18 | up | 27605 | 6572 | 6572 | 0 | 34.40 |
| prompt | 2 | down | 26735 | 3403 | 3403 | 0 | 26.17 |

## Top CPU/MOE Name Profile Rows

| prompt | layer | role | name | total ms/call | decode ms/call | decode fallback ms/call | src0_type | unsupported |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `dev_python_reverse` | 57 | down | `blk.57.ffn_down_exps.weight` | 80.555 | 72.577 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 58 | down | `blk.58.ffn_down_exps.weight` | 79.640 | 72.226 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 60 | down | `blk.60.ffn_down_exps.weight` | 62.866 | 62.866 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 4 | down | `blk.4.ffn_down_exps.weight` | 64.392 | 59.606 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 56 | down | `blk.56.ffn_down_exps.weight` | 63.624 | 56.695 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 21 | down | `blk.21.ffn_down_exps.weight` | 61.773 | 54.796 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 23 | down | `blk.23.ffn_down_exps.weight` | 61.800 | 54.629 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 59 | down | `blk.59.ffn_down_exps.weight` | 59.544 | 53.923 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 26 | down | `blk.26.ffn_down_exps.weight` | 60.765 | 53.307 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 24 | down | `blk.24.ffn_down_exps.weight` | 61.806 | 51.601 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 25 | down | `blk.25.ffn_down_exps.weight` | 60.138 | 51.564 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 58 | down | `blk.58.ffn_down_exps.weight` | 55.506 | 51.171 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 20 | down | `blk.20.ffn_down_exps.weight` | 56.741 | 50.043 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 16 | down | `blk.16.ffn_down_exps.weight` | 56.140 | 49.475 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 19 | down | `blk.19.ffn_down_exps.weight` | 55.060 | 49.441 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 17 | down | `blk.17.ffn_down_exps.weight` | 52.356 | 46.986 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 27 | down | `blk.27.ffn_down_exps.weight` | 51.713 | 46.349 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 22 | down | `blk.22.ffn_down_exps.weight` | 51.285 | 45.836 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 45.397 | 45.397 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 49.692 | 44.195 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 49.393 | 43.914 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 13 | down | `blk.13.ffn_down_exps.weight` | 47.118 | 41.296 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 5 | down | `blk.5.ffn_down_exps.weight` | 43.670 | 39.439 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 3 | down | `blk.3.ffn_down_exps.weight` | 43.771 | 39.408 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 59 | down | `blk.59.ffn_down_exps.weight` | 44.689 | 39.774 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 56 | down | `blk.56.ffn_down_exps.weight` | 42.459 | 38.564 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 16 | down | `blk.16.ffn_down_exps.weight` | 42.391 | 37.591 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 12 | down | `blk.12.ffn_down_exps.weight` | 41.396 | 36.811 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 21 | down | `blk.21.ffn_down_exps.weight` | 41.180 | 36.574 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 23 | down | `blk.23.ffn_down_exps.weight` | 41.893 | 35.860 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 18 | down | `blk.18.ffn_down_exps.weight` | 41.826 | 35.016 | 34.979 | 2 | 96 |
| `dev_python_reverse` | 6 | down | `blk.6.ffn_down_exps.weight` | 39.000 | 34.646 | 34.548 | 2 | 96 |
| `dev_photosynthesis_factual` | 22 | down | `blk.22.ffn_down_exps.weight` | 39.370 | 34.594 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 25 | down | `blk.25.ffn_down_exps.weight` | 40.388 | 34.288 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 15 | down | `blk.15.ffn_down_exps.weight` | 39.239 | 33.381 | 33.337 | 2 | 96 |
| `dev_photosynthesis_factual` | 26 | down | `blk.26.ffn_down_exps.weight` | 39.690 | 33.307 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 11 | down | `blk.11.ffn_down_exps.weight` | 35.466 | 32.509 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 20 | down | `blk.20.ffn_down_exps.weight` | 35.672 | 32.308 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 35.950 | 31.923 | 31.887 | 2 | 95 |
| `dev_photosynthesis_factual` | 5 | down | `blk.5.ffn_down_exps.weight` | 36.256 | 31.890 | 0.003 | 11 | 0 |
| `dev_mixed_summary` | 21 | down | `blk.21.ffn_down_exps.weight` | 73.834 | 55.005 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 8 | down | `blk.8.ffn_down_exps.weight` | 36.379 | 30.589 | 30.557 | 2 | 96 |
| `dev_photosynthesis_factual` | 27 | down | `blk.27.ffn_down_exps.weight` | 35.090 | 30.371 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 57 | down | `blk.57.ffn_down_exps.weight` | 64.876 | 52.366 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 3 | down | `blk.3.ffn_down_exps.weight` | 32.978 | 29.792 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 58 | down | `blk.58.ffn_down_exps.weight` | 64.218 | 50.759 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 23 | down | `blk.23.ffn_down_exps.weight` | 61.965 | 50.531 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 24 | down | `blk.24.ffn_down_exps.weight` | 34.218 | 28.911 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 58 | down | `blk.58.ffn_down_exps.weight` | 99.586 | 79.688 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 19 | down | `blk.19.ffn_down_exps.weight` | 33.585 | 28.745 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 20 | down | `blk.20.ffn_down_exps.weight` | 60.808 | 49.584 | 0.003 | 23 | 0 |
| `dev_mixed_summary` | 60 | down | `blk.60.ffn_down_exps.weight` | 48.666 | 48.666 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 12 | down | `blk.12.ffn_down_exps.weight` | 31.703 | 28.370 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 16 | down | `blk.16.ffn_down_exps.weight` | 60.348 | 49.354 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 59 | down | `blk.59.ffn_down_exps.weight` | 58.352 | 47.859 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 4 | down | `blk.4.ffn_down_exps.weight` | 57.941 | 47.396 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 18 | down | `blk.18.ffn_down_exps.weight` | 33.152 | 26.741 | 26.713 | 2 | 95 |
| `dev_python_reverse` | 7 | down | `blk.7.ffn_down_exps.weight` | 31.078 | 26.257 | 26.220 | 2 | 96 |
| `dev_mixed_summary` | 56 | down | `blk.56.ffn_down_exps.weight` | 57.532 | 46.109 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 22 | down | `blk.22.ffn_down_exps.weight` | 58.029 | 45.694 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 11 | down | `blk.11.ffn_down_exps.weight` | 28.192 | 26.038 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 13 | down | `blk.13.ffn_down_exps.weight` | 29.187 | 25.556 | 0.002 | 11 | 0 |
| `dev_japan_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 34.787 | 27.951 | 27.918 | 2 | 86 |
| `dev_linear_equation` | 60 | down | `blk.60.ffn_down_exps.weight` | 67.734 | 67.734 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 15 | down | `blk.15.ffn_down_exps.weight` | 29.451 | 25.040 | 25.008 | 2 | 95 |
| `dev_photosynthesis_factual` | 17 | down | `blk.17.ffn_down_exps.weight` | 28.528 | 24.998 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 9 | down | `blk.9.ffn_down_exps.weight` | 29.663 | 24.617 | 24.587 | 2 | 96 |
| `dev_mixed_summary` | 26 | down | `blk.26.ffn_down_exps.weight` | 57.459 | 42.157 | 0.002 | 23 | 0 |
| `dev_zh_france` | 57 | down | `blk.57.ffn_down_exps.weight` | 56.834 | 46.294 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 25 | down | `blk.25.ffn_down_exps.weight` | 84.832 | 66.195 | 0.004 | 23 | 0 |
| `dev_japan_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 25.943 | 25.943 | 0.003 | 11 | 0 |
| `dev_linear_equation` | 21 | down | `blk.21.ffn_down_exps.weight` | 95.552 | 65.525 | 0.003 | 23 | 0 |
| `dev_japan_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 33.181 | 26.116 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 19 | down | `blk.19.ffn_down_exps.weight` | 49.814 | 40.654 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 10 | down | `blk.10.ffn_down_exps.weight` | 28.066 | 23.016 | 22.985 | 2 | 96 |
| `dev_japan_factual` | 58 | down | `blk.58.ffn_down_exps.weight` | 33.310 | 25.698 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 10 | down | `blk.10.ffn_down_exps.weight` | 26.965 | 23.062 | 23.035 | 2 | 95 |
| `dev_japan_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 33.277 | 25.374 | 0.003 | 23 | 0 |
| `dev_linear_equation` | 57 | down | `blk.57.ffn_down_exps.weight` | 89.396 | 63.168 | 0.003 | 23 | 0 |
| `dev_linear_equation` | 4 | down | `blk.4.ffn_down_exps.weight` | 80.579 | 62.899 | 0.003 | 23 | 0 |

## Down Batch Decline Reasons

| reason | count |
|---|---:|
| multirow_not_supported | 364 |

Interpretation:

- The slow general prompts are not explained by iouring wait alone.
- Down batch profile and name profile should be used to identify whether
  the dominant cost is eligible GPU down batch wall time, unsupported CPU
  fallback types, or prompt/prefill fallback materialization.
- Any source optimization must first name the rows it is expected to reduce
  and compute the best-case bound from this report.
