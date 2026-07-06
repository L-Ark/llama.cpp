# Kimi general dev profile breakdown

This report uses dev baseline profile artifacts only. It does not use held-out test prompts.

## Prompt Summary

| prompt | tok/s | decode ms | down wall ms | upgate wall ms | fallback ms | top decline reason |
|---|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | 1.37 | 56157 | 10339 | 15378 | 75416 | multirow_not_supported:52 |
| `dev_japan_factual` | 1.01 | 84104 | 15818 | 24066 | 79357 | multirow_not_supported:52 |
| `dev_linear_equation` | 0.47 | 71705 | 16057 | 17310 | 96400 | multirow_not_supported:52 |
| `dev_mixed_summary` | 0.62 | 86560 | 17584 | 23380 | 103918 | multirow_not_supported:52 |
| `dev_photosynthesis_factual` | 0.76 | 123763 | 24377 | 33734 | 66202 | multirow_not_supported:52 |
| `dev_python_reverse` | 0.67 | 142312 | 27714 | 38851 | 78759 | multirow_not_supported:52 |
| `dev_zh_france` | 0.93 | 52427 | 10307 | 15073 | 68775 | multirow_not_supported:52 |

## Down Batch By Type

| src0_type | role | calls | wall ms | stage ms | kernel ms | staged jobs | hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | down | 5856 | 64126 | 63098 | 516 | 29035 | 38.0% |
| 11 | down | 20015 | 58070 | 54294 | 2558 | 30220 | 81.1% |

## Up/Gate Batch By Type Pair

| type pair | calls | wall ms | stage ms | kernel ms | up ms | gate ms | up wait ms | gate wait ms | hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| up22_gate22 | 8784 | 84038 | 349 | 83258 | 80134 | 3017 | 78683 | 82111 | 39.0% |
| up18_gate18 | 4887 | 83754 | 207 | 83211 | 42070 | 41098 | 0 | 0 | 34.2% |

## Fallback By Phase / Type / Role

| phase | src0_type | role | fallback ms | calls | GGUF calls | pack mmap calls | bytes GiB |
|---|---:|---|---:|---:|---:|---:|---:|
| prompt | 11 | down | 161684 | 22570 | 22570 | 0 | 132.59 |
| prompt | 22 | up | 113286 | 25834 | 25834 | 0 | 113.13 |
| prompt | 18 | gate | 104721 | 21828 | 21828 | 0 | 114.24 |
| prompt | 23 | down | 56219 | 6433 | 6433 | 0 | 46.72 |
| prompt | 22 | gate | 45720 | 10578 | 10578 | 0 | 46.32 |
| decode | 2 | down | 31017 | 27328 | 0 | 27328 | 210.16 |
| prompt | 2 | down | 28238 | 3403 | 3403 | 0 | 26.17 |
| prompt | 18 | up | 27943 | 6572 | 6572 | 0 | 34.40 |

## Top CPU/MOE Name Profile Rows

| prompt | layer | role | name | total ms/call | decode ms/call | decode fallback ms/call | src0_type | unsupported |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `dev_python_reverse` | 57 | down | `blk.57.ffn_down_exps.weight` | 24.603 | 18.335 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 16 | down | `blk.16.ffn_down_exps.weight` | 23.734 | 17.485 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 16 | down | `blk.16.ffn_down_exps.weight` | 21.906 | 16.771 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 27 | down | `blk.27.ffn_down_exps.weight` | 20.173 | 14.967 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 60 | down | `blk.60.ffn_down_exps.weight` | 14.352 | 14.352 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 4 | down | `blk.4.ffn_down_exps.weight` | 21.574 | 14.188 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 58 | down | `blk.58.ffn_down_exps.weight` | 20.677 | 14.153 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 17.753 | 13.907 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 12 | down | `blk.12.ffn_down_exps.weight` | 17.051 | 13.421 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 12 | down | `blk.12.ffn_down_exps.weight` | 18.246 | 13.266 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 18.587 | 13.245 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 56 | down | `blk.56.ffn_down_exps.weight` | 18.494 | 12.760 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 12.574 | 12.574 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 58 | down | `blk.58.ffn_down_exps.weight` | 17.531 | 12.545 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 16 | down | `blk.16.ffn_down_exps.weight` | 36.984 | 21.183 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 24 | down | `blk.24.ffn_down_exps.weight` | 20.489 | 12.018 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 27 | down | `blk.27.ffn_down_exps.weight` | 16.826 | 12.002 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 25 | down | `blk.25.ffn_down_exps.weight` | 18.882 | 11.834 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 59 | down | `blk.59.ffn_down_exps.weight` | 17.586 | 11.714 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 20 | down | `blk.20.ffn_down_exps.weight` | 17.139 | 11.430 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 19 | down | `blk.19.ffn_down_exps.weight` | 17.853 | 11.272 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 23 | down | `blk.23.ffn_down_exps.weight` | 17.185 | 11.160 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 26 | down | `blk.26.ffn_down_exps.weight` | 19.198 | 11.153 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 16 | down | `blk.16.ffn_down_exps.weight` | 54.005 | 30.901 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 20 | down | `blk.20.ffn_down_exps.weight` | 16.922 | 11.154 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 21 | down | `blk.21.ffn_down_exps.weight` | 16.634 | 10.923 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 21 | down | `blk.21.ffn_down_exps.weight` | 15.546 | 10.694 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 59 | down | `blk.59.ffn_down_exps.weight` | 15.377 | 10.511 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 4 | down | `blk.4.ffn_down_exps.weight` | 28.716 | 18.212 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 17 | down | `blk.17.ffn_down_exps.weight` | 14.478 | 10.289 | 0.002 | 11 | 0 |
| `dev_linear_equation` | 58 | down | `blk.58.ffn_down_exps.weight` | 52.818 | 28.666 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 26 | down | `blk.26.ffn_down_exps.weight` | 16.588 | 10.205 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 8 | down | `blk.8.ffn_down_exps.weight` | 15.253 | 10.097 | 10.067 | 2 | 96 |
| `dev_photosynthesis_factual` | 25 | down | `blk.25.ffn_down_exps.weight` | 16.901 | 10.113 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 5 | down | `blk.5.ffn_down_exps.weight` | 14.384 | 9.989 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 22 | down | `blk.22.ffn_down_exps.weight` | 16.219 | 9.974 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 18.764 | 11.101 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 19 | down | `blk.19.ffn_down_exps.weight` | 14.514 | 10.001 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 3 | down | `blk.3.ffn_down_exps.weight` | 16.195 | 9.855 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 23 | down | `blk.23.ffn_down_exps.weight` | 15.511 | 9.873 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 25 | down | `blk.25.ffn_down_exps.weight` | 49.346 | 27.036 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 56 | down | `blk.56.ffn_down_exps.weight` | 14.326 | 9.775 | 0.002 | 11 | 0 |
| `dev_linear_equation` | 57 | down | `blk.57.ffn_down_exps.weight` | 54.279 | 26.945 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 13 | down | `blk.13.ffn_down_exps.weight` | 14.981 | 9.459 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 11 | down | `blk.11.ffn_down_exps.weight` | 12.899 | 9.536 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 60 | down | `blk.60.ffn_down_exps.weight` | 16.221 | 16.221 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 5 | down | `blk.5.ffn_down_exps.weight` | 12.878 | 9.413 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 22 | down | `blk.22.ffn_down_exps.weight` | 15.033 | 9.406 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 9 | down | `blk.9.ffn_down_exps.weight` | 35.924 | 16.272 | 16.233 | 2 | 55 |
| `dev_mixed_summary` | 57 | down | `blk.57.ffn_down_exps.weight` | 31.575 | 16.249 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 3 | down | `blk.3.ffn_down_exps.weight` | 12.672 | 9.196 | 0.002 | 11 | 0 |
| `dev_mixed_summary` | 21 | down | `blk.21.ffn_down_exps.weight` | 31.192 | 15.875 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 21 | down | `blk.21.ffn_down_exps.weight` | 52.016 | 25.065 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 60 | down | `blk.60.ffn_down_exps.weight` | 24.347 | 24.347 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 24 | down | `blk.24.ffn_down_exps.weight` | 15.440 | 9.021 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 4 | down | `blk.4.ffn_down_exps.weight` | 46.621 | 24.842 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 7 | down | `blk.7.ffn_down_exps.weight` | 15.067 | 8.811 | 8.783 | 2 | 96 |
| `dev_python_reverse` | 18 | down | `blk.18.ffn_down_exps.weight` | 15.776 | 8.775 | 8.748 | 2 | 96 |
| `dev_python_reverse` | 11 | down | `blk.11.ffn_down_exps.weight` | 12.908 | 8.684 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 10 | down | `blk.10.ffn_down_exps.weight` | 11.849 | 8.595 | 8.569 | 2 | 95 |
| `dev_mixed_summary` | 22 | down | `blk.22.ffn_down_exps.weight` | 28.933 | 14.847 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 6 | down | `blk.6.ffn_down_exps.weight` | 12.874 | 8.399 | 8.369 | 2 | 96 |
| `dev_mixed_summary` | 7 | down | `blk.7.ffn_down_exps.weight` | 28.324 | 14.761 | 14.727 | 2 | 55 |
| `dev_mixed_summary` | 58 | down | `blk.58.ffn_down_exps.weight` | 29.405 | 14.747 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 9.248 | 9.248 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 17 | down | `blk.17.ffn_down_exps.weight` | 13.441 | 8.428 | 0.002 | 11 | 0 |
| `dev_japan_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 16.614 | 9.249 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 6 | down | `blk.6.ffn_down_exps.weight` | 25.473 | 14.445 | 14.407 | 2 | 55 |
| `dev_mixed_summary` | 20 | down | `blk.20.ffn_down_exps.weight` | 33.431 | 14.349 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 16.412 | 9.063 | 9.036 | 2 | 86 |
| `dev_mixed_summary` | 12 | down | `blk.12.ffn_down_exps.weight` | 23.775 | 14.255 | 0.002 | 11 | 0 |
| `dev_linear_equation` | 26 | down | `blk.26.ffn_down_exps.weight` | 47.215 | 22.540 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 12.444 | 8.121 | 8.093 | 2 | 95 |
| `dev_mixed_summary` | 10 | down | `blk.10.ffn_down_exps.weight` | 28.010 | 13.985 | 13.953 | 2 | 55 |
| `dev_mixed_summary` | 23 | down | `blk.23.ffn_down_exps.weight` | 25.481 | 13.985 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 9 | down | `blk.9.ffn_down_exps.weight` | 13.029 | 7.895 | 7.868 | 2 | 96 |
| `dev_python_reverse` | 10 | down | `blk.10.ffn_down_exps.weight` | 13.032 | 7.884 | 7.857 | 2 | 96 |
| `dev_linear_equation` | 27 | down | `blk.27.ffn_down_exps.weight` | 40.500 | 22.012 | 0.002 | 11 | 0 |
| `dev_japan_factual` | 12 | down | `blk.12.ffn_down_exps.weight` | 14.466 | 8.674 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 13 | down | `blk.13.ffn_down_exps.weight` | 11.392 | 7.842 | 0.002 | 11 | 0 |

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
