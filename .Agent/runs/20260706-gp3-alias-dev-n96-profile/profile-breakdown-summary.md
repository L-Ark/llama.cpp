# Kimi general dev profile breakdown

This report uses dev baseline profile artifacts only. It does not use held-out test prompts.

## Prompt Summary

| prompt | tok/s | decode ms | down wall ms | upgate wall ms | fallback ms | top decline reason |
|---|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | 1.35 | 57125 | 10555 | 15859 | 74299 | multirow_not_supported:52 |
| `dev_japan_factual` | 1.02 | 83541 | 15894 | 23804 | 77395 | multirow_not_supported:52 |
| `dev_linear_equation` | 0.60 | 56532 | 10604 | 15114 | 94028 | multirow_not_supported:52 |
| `dev_mixed_summary` | 0.71 | 75722 | 16046 | 20743 | 100909 | multirow_not_supported:52 |
| `dev_photosynthesis_factual` | 0.78 | 120051 | 23373 | 33803 | 61174 | multirow_not_supported:52 |
| `dev_python_reverse` | 0.71 | 134295 | 25775 | 37285 | 82628 | multirow_not_supported:52 |
| `dev_zh_france` | 0.84 | 58258 | 12253 | 16412 | 71189 | multirow_not_supported:52 |

## Down Batch By Type

| src0_type | role | calls | wall ms | stage ms | kernel ms | staged jobs | hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | down | 5856 | 59213 | 58192 | 514 | 29035 | 38.0% |
| 11 | down | 20015 | 55285 | 51514 | 2552 | 30220 | 81.1% |

## Up/Gate Batch By Type Pair

| type pair | calls | wall ms | stage ms | kernel ms | up ms | gate ms | up wait ms | gate wait ms | hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| up18_gate18 | 4887 | 82902 | 204 | 82378 | 41794 | 40543 | 0 | 0 | 34.2% |
| up22_gate22 | 8784 | 80118 | 340 | 79351 | 76629 | 2651 | 75167 | 78217 | 39.0% |

## Fallback By Phase / Type / Role

| phase | src0_type | role | fallback ms | calls | GGUF calls | pack mmap calls | bytes GiB |
|---|---:|---|---:|---:|---:|---:|---:|
| prompt | 11 | down | 158550 | 22570 | 22570 | 0 | 132.59 |
| prompt | 22 | up | 111384 | 25834 | 25834 | 0 | 113.13 |
| prompt | 18 | gate | 104121 | 21828 | 21828 | 0 | 114.24 |
| prompt | 23 | down | 54998 | 6433 | 6433 | 0 | 46.72 |
| prompt | 22 | gate | 44763 | 10411 | 10411 | 0 | 45.59 |
| decode | 2 | down | 30687 | 27328 | 0 | 27328 | 210.16 |
| prompt | 18 | up | 29589 | 6572 | 6572 | 0 | 34.40 |
| prompt | 2 | down | 27530 | 3403 | 3403 | 0 | 26.17 |

## Top CPU/MOE Name Profile Rows

| prompt | layer | role | name | total ms/call | decode ms/call | decode fallback ms/call | src0_type | unsupported |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `dev_python_reverse` | 16 | down | `blk.16.ffn_down_exps.weight` | 22.698 | 15.602 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 57 | down | `blk.57.ffn_down_exps.weight` | 23.048 | 15.417 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 16 | down | `blk.16.ffn_down_exps.weight` | 18.840 | 15.417 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 4 | down | `blk.4.ffn_down_exps.weight` | 18.826 | 13.869 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 60 | down | `blk.60.ffn_down_exps.weight` | 13.656 | 13.656 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 58 | down | `blk.58.ffn_down_exps.weight` | 20.561 | 13.346 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 18.327 | 13.124 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 58 | down | `blk.58.ffn_down_exps.weight` | 16.787 | 12.673 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 12.217 | 12.217 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 27 | down | `blk.27.ffn_down_exps.weight` | 17.395 | 11.963 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 12 | down | `blk.12.ffn_down_exps.weight` | 16.077 | 12.088 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 16.703 | 11.936 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 56 | down | `blk.56.ffn_down_exps.weight` | 17.730 | 11.612 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 12 | down | `blk.12.ffn_down_exps.weight` | 17.399 | 11.529 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 24 | down | `blk.24.ffn_down_exps.weight` | 19.102 | 11.471 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 25 | down | `blk.25.ffn_down_exps.weight` | 19.330 | 11.242 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 59 | down | `blk.59.ffn_down_exps.weight` | 17.158 | 11.035 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 27 | down | `blk.27.ffn_down_exps.weight` | 15.513 | 10.955 | 0.003 | 11 | 0 |
| `dev_zh_france` | 57 | down | `blk.57.ffn_down_exps.weight` | 33.365 | 20.989 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 20 | down | `blk.20.ffn_down_exps.weight` | 16.180 | 10.794 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 23 | down | `blk.23.ffn_down_exps.weight` | 17.483 | 10.751 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 26 | down | `blk.26.ffn_down_exps.weight` | 18.750 | 10.654 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 59 | down | `blk.59.ffn_down_exps.weight` | 15.384 | 10.643 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 20 | down | `blk.20.ffn_down_exps.weight` | 15.633 | 10.484 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 19 | down | `blk.19.ffn_down_exps.weight` | 15.519 | 10.355 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 21 | down | `blk.21.ffn_down_exps.weight` | 16.176 | 10.335 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 4 | down | `blk.4.ffn_down_exps.weight` | 31.734 | 18.018 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 21 | down | `blk.21.ffn_down_exps.weight` | 16.456 | 10.286 | 0.003 | 23 | 0 |
| `dev_mixed_summary` | 16 | down | `blk.16.ffn_down_exps.weight` | 33.175 | 17.870 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 18.946 | 11.064 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 26 | down | `blk.26.ffn_down_exps.weight` | 14.621 | 9.948 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 17 | down | `blk.17.ffn_down_exps.weight` | 15.856 | 9.832 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 3 | down | `blk.3.ffn_down_exps.weight` | 14.477 | 9.788 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 8 | down | `blk.8.ffn_down_exps.weight` | 15.740 | 9.756 | 9.731 | 2 | 96 |
| `dev_photosynthesis_factual` | 56 | down | `blk.56.ffn_down_exps.weight` | 14.225 | 9.834 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 5 | down | `blk.5.ffn_down_exps.weight` | 14.151 | 9.679 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 25 | down | `blk.25.ffn_down_exps.weight` | 15.618 | 9.704 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 19 | down | `blk.19.ffn_down_exps.weight` | 14.280 | 9.588 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 22 | down | `blk.22.ffn_down_exps.weight` | 16.089 | 9.430 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 13 | down | `blk.13.ffn_down_exps.weight` | 16.268 | 9.357 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 23 | down | `blk.23.ffn_down_exps.weight` | 14.965 | 9.291 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 5 | down | `blk.5.ffn_down_exps.weight` | 13.576 | 9.211 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 7 | down | `blk.7.ffn_down_exps.weight` | 13.279 | 9.041 | 9.011 | 2 | 95 |
| `dev_photosynthesis_factual` | 11 | down | `blk.11.ffn_down_exps.weight` | 12.028 | 9.041 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 22 | down | `blk.22.ffn_down_exps.weight` | 13.217 | 8.969 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 9 | down | `blk.9.ffn_down_exps.weight` | 13.550 | 8.867 | 8.843 | 2 | 96 |
| `dev_python_reverse` | 7 | down | `blk.7.ffn_down_exps.weight` | 13.287 | 8.687 | 8.659 | 2 | 96 |
| `dev_japan_factual` | 10 | down | `blk.10.ffn_down_exps.weight` | 15.541 | 9.576 | 9.551 | 2 | 86 |
| `dev_mixed_summary` | 20 | down | `blk.20.ffn_down_exps.weight` | 31.542 | 15.069 | 0.002 | 23 | 0 |
| `dev_zh_france` | 60 | down | `blk.60.ffn_down_exps.weight` | 16.271 | 16.271 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 15 | down | `blk.15.ffn_down_exps.weight` | 14.838 | 8.534 | 8.510 | 2 | 96 |
| `dev_photosynthesis_factual` | 3 | down | `blk.3.ffn_down_exps.weight` | 11.296 | 8.582 | 0.003 | 11 | 0 |
| `dev_japan_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 9.322 | 9.322 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 24 | down | `blk.24.ffn_down_exps.weight` | 12.881 | 8.436 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 18 | down | `blk.18.ffn_down_exps.weight` | 15.536 | 8.265 | 8.239 | 2 | 96 |
| `dev_zh_france` | 58 | down | `blk.58.ffn_down_exps.weight` | 26.837 | 15.936 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 8 | down | `blk.8.ffn_down_exps.weight` | 12.576 | 8.306 | 8.278 | 2 | 95 |
| `dev_photosynthesis_factual` | 10 | down | `blk.10.ffn_down_exps.weight` | 12.437 | 8.284 | 8.254 | 2 | 95 |
| `dev_japan_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 15.457 | 9.105 | 9.079 | 2 | 86 |
| `dev_photosynthesis_factual` | 15 | down | `blk.15.ffn_down_exps.weight` | 11.849 | 8.140 | 8.114 | 2 | 95 |
| `dev_python_reverse` | 11 | down | `blk.11.ffn_down_exps.weight` | 11.348 | 8.017 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 58 | down | `blk.58.ffn_down_exps.weight` | 25.508 | 14.055 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 57 | down | `blk.57.ffn_down_exps.weight` | 26.918 | 13.979 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 17 | down | `blk.17.ffn_down_exps.weight` | 11.106 | 8.010 | 0.002 | 11 | 0 |
| `dev_zh_france` | 4 | down | `blk.4.ffn_down_exps.weight` | 28.390 | 15.253 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 23 | down | `blk.23.ffn_down_exps.weight` | 28.064 | 13.757 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 21 | down | `blk.21.ffn_down_exps.weight` | 29.436 | 13.752 | 0.002 | 23 | 0 |
| `dev_france_regression` | 8 | down | `blk.8.ffn_down_exps.weight` | 14.694 | 9.589 | 9.558 | 2 | 78 |
| `dev_japan_factual` | 12 | down | `blk.12.ffn_down_exps.weight` | 14.488 | 8.660 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 22 | down | `blk.22.ffn_down_exps.weight` | 26.839 | 13.459 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 16 | down | `blk.16.ffn_down_exps.weight` | 15.052 | 8.503 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 16.202 | 8.474 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 9 | down | `blk.9.ffn_down_exps.weight` | 31.758 | 13.290 | 13.266 | 2 | 55 |
| `dev_python_reverse` | 6 | down | `blk.6.ffn_down_exps.weight` | 11.581 | 7.554 | 7.526 | 2 | 96 |
| `dev_photosynthesis_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 12.035 | 7.609 | 7.581 | 2 | 95 |
| `dev_japan_factual` | 3 | down | `blk.3.ffn_down_exps.weight` | 14.346 | 8.356 | 0.002 | 11 | 0 |
| `dev_japan_factual` | 7 | down | `blk.7.ffn_down_exps.weight` | 14.201 | 8.307 | 8.277 | 2 | 86 |
| `dev_mixed_summary` | 19 | down | `blk.19.ffn_down_exps.weight` | 26.478 | 13.044 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 13 | down | `blk.13.ffn_down_exps.weight` | 11.237 | 7.432 | 0.002 | 11 | 0 |
| `dev_japan_factual` | 21 | down | `blk.21.ffn_down_exps.weight` | 16.299 | 8.214 | 0.002 | 23 | 0 |

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
