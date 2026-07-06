# Kimi general dev profile breakdown

This report uses dev baseline profile artifacts only. It does not use held-out test prompts.

## Prompt Summary

| prompt | tok/s | decode ms | down wall ms | upgate wall ms | fallback ms | top decline reason |
|---|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | 1.37 | 56051 | 10355 | 15522 | 76084 | multirow_not_supported:52 |
| `dev_japan_factual` | 1.00 | 84600 | 15847 | 24534 | 76137 | multirow_not_supported:52 |
| `dev_linear_equation` | 0.52 | 65968 | 14682 | 16176 | 93489 | multirow_not_supported:52 |
| `dev_mixed_summary` | 0.74 | 73139 | 14107 | 20732 | 95633 | multirow_not_supported:52 |
| `dev_photosynthesis_factual` | 0.80 | 118019 | 22757 | 33760 | 66942 | multirow_not_supported:52 |
| `dev_python_reverse` | 0.70 | 135110 | 25929 | 37961 | 81719 | multirow_not_supported:52 |
| `dev_zh_france` | 0.84 | 58142 | 12799 | 15956 | 66568 | multirow_not_supported:52 |

## Down Batch By Type

| src0_type | role | calls | wall ms | stage ms | kernel ms | staged jobs | hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | down | 5856 | 60287 | 59270 | 515 | 29035 | 38.0% |
| 11 | down | 20015 | 56187 | 52440 | 2551 | 30220 | 81.1% |

## Up/Gate Batch By Type Pair

| type pair | calls | wall ms | stage ms | kernel ms | up ms | gate ms | up wait ms | gate wait ms | hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| up18_gate18 | 4887 | 83534 | 201 | 82980 | 42130 | 40807 | 0 | 0 | 34.2% |
| up22_gate22 | 8784 | 81107 | 339 | 80350 | 77625 | 2642 | 76140 | 79214 | 39.0% |

## Fallback By Phase / Type / Role

| phase | src0_type | role | fallback ms | calls | GGUF calls | pack mmap calls | bytes GiB |
|---|---:|---|---:|---:|---:|---:|---:|
| prompt | 11 | down | 157699 | 22570 | 22570 | 0 | 132.59 |
| prompt | 22 | up | 110078 | 25834 | 25834 | 0 | 113.13 |
| prompt | 18 | gate | 101718 | 21828 | 21828 | 0 | 114.24 |
| prompt | 23 | down | 54765 | 6433 | 6433 | 0 | 46.72 |
| prompt | 22 | gate | 44638 | 10578 | 10578 | 0 | 46.32 |
| decode | 2 | down | 29898 | 27328 | 0 | 27328 | 210.16 |
| prompt | 2 | down | 29613 | 3403 | 3403 | 0 | 26.17 |
| prompt | 18 | up | 28164 | 6572 | 6572 | 0 | 34.40 |

## Top CPU/MOE Name Profile Rows

| prompt | layer | role | name | total ms/call | decode ms/call | decode fallback ms/call | src0_type | unsupported |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `dev_python_reverse` | 57 | down | `blk.57.ffn_down_exps.weight` | 23.501 | 15.882 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 16 | down | `blk.16.ffn_down_exps.weight` | 20.094 | 15.083 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 16 | down | `blk.16.ffn_down_exps.weight` | 21.051 | 14.843 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 60 | down | `blk.60.ffn_down_exps.weight` | 14.314 | 14.314 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 4 | down | `blk.4.ffn_down_exps.weight` | 20.350 | 13.641 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 58 | down | `blk.58.ffn_down_exps.weight` | 20.028 | 13.361 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 19.101 | 13.226 | 0.004 | 23 | 0 |
| `dev_python_reverse` | 56 | down | `blk.56.ffn_down_exps.weight` | 18.189 | 11.950 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 27 | down | `blk.27.ffn_down_exps.weight` | 17.997 | 11.897 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 12 | down | `blk.12.ffn_down_exps.weight` | 15.482 | 11.892 | 0.003 | 11 | 0 |
| `dev_zh_france` | 57 | down | `blk.57.ffn_down_exps.weight` | 36.448 | 22.806 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 17.127 | 11.737 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 12 | down | `blk.12.ffn_down_exps.weight` | 24.057 | 11.566 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 11.557 | 11.557 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 58 | down | `blk.58.ffn_down_exps.weight` | 16.344 | 11.641 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 24 | down | `blk.24.ffn_down_exps.weight` | 18.439 | 11.432 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 59 | down | `blk.59.ffn_down_exps.weight` | 16.309 | 11.185 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 25 | down | `blk.25.ffn_down_exps.weight` | 19.582 | 11.121 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 20 | down | `blk.20.ffn_down_exps.weight` | 17.054 | 10.786 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 23 | down | `blk.23.ffn_down_exps.weight` | 16.875 | 10.680 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 27 | down | `blk.27.ffn_down_exps.weight` | 16.169 | 10.646 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 21 | down | `blk.21.ffn_down_exps.weight` | 17.320 | 10.503 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 26 | down | `blk.26.ffn_down_exps.weight` | 17.445 | 10.387 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 19 | down | `blk.19.ffn_down_exps.weight` | 17.391 | 10.345 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 20 | down | `blk.20.ffn_down_exps.weight` | 15.242 | 10.266 | 0.003 | 23 | 0 |
| `dev_zh_france` | 60 | down | `blk.60.ffn_down_exps.weight` | 19.059 | 19.059 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 17 | down | `blk.17.ffn_down_exps.weight` | 15.652 | 9.938 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 21 | down | `blk.21.ffn_down_exps.weight` | 16.076 | 9.930 | 0.003 | 23 | 0 |
| `dev_photosynthesis_factual` | 59 | down | `blk.59.ffn_down_exps.weight` | 14.849 | 9.860 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 26 | down | `blk.26.ffn_down_exps.weight` | 15.755 | 9.788 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 5 | down | `blk.5.ffn_down_exps.weight` | 14.738 | 9.669 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 25 | down | `blk.25.ffn_down_exps.weight` | 16.234 | 9.769 | 0.003 | 23 | 0 |
| `dev_python_reverse` | 3 | down | `blk.3.ffn_down_exps.weight` | 15.083 | 9.583 | 0.002 | 11 | 0 |
| `dev_zh_france` | 58 | down | `blk.58.ffn_down_exps.weight` | 30.632 | 18.487 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 4 | down | `blk.4.ffn_down_exps.weight` | 18.851 | 10.646 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 22 | down | `blk.22.ffn_down_exps.weight` | 15.734 | 9.451 | 0.001 | 23 | 0 |
| `dev_photosynthesis_factual` | 19 | down | `blk.19.ffn_down_exps.weight` | 13.047 | 9.404 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 16 | down | `blk.16.ffn_down_exps.weight` | 43.801 | 25.831 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 56 | down | `blk.56.ffn_down_exps.weight` | 13.877 | 9.270 | 0.003 | 11 | 0 |
| `dev_python_reverse` | 8 | down | `blk.8.ffn_down_exps.weight` | 14.273 | 9.153 | 9.126 | 2 | 96 |
| `dev_photosynthesis_factual` | 23 | down | `blk.23.ffn_down_exps.weight` | 15.848 | 9.167 | 0.003 | 23 | 0 |
| `dev_japan_factual` | 8 | down | `blk.8.ffn_down_exps.weight` | 16.093 | 10.133 | 10.105 | 2 | 86 |
| `dev_python_reverse` | 13 | down | `blk.13.ffn_down_exps.weight` | 15.793 | 9.051 | 0.001 | 11 | 0 |
| `dev_photosynthesis_factual` | 10 | down | `blk.10.ffn_down_exps.weight` | 13.688 | 9.070 | 9.042 | 2 | 95 |
| `dev_photosynthesis_factual` | 11 | down | `blk.11.ffn_down_exps.weight` | 12.182 | 9.008 | 0.002 | 11 | 0 |
| `dev_japan_factual` | 60 | down | `blk.60.ffn_down_exps.weight` | 9.791 | 9.791 | 0.002 | 11 | 0 |
| `dev_photosynthesis_factual` | 22 | down | `blk.22.ffn_down_exps.weight` | 13.642 | 8.951 | 0.003 | 23 | 0 |
| `dev_linear_equation` | 58 | down | `blk.58.ffn_down_exps.weight` | 40.641 | 24.691 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 4 | down | `blk.4.ffn_down_exps.weight` | 29.806 | 15.410 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 13.597 | 8.832 | 8.792 | 2 | 95 |
| `dev_mixed_summary` | 16 | down | `blk.16.ffn_down_exps.weight` | 28.583 | 15.073 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 9 | down | `blk.9.ffn_down_exps.weight` | 13.723 | 8.527 | 8.502 | 2 | 96 |
| `dev_photosynthesis_factual` | 5 | down | `blk.5.ffn_down_exps.weight` | 13.874 | 8.597 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 3 | down | `blk.3.ffn_down_exps.weight` | 12.189 | 8.539 | 0.003 | 11 | 0 |
| `dev_photosynthesis_factual` | 24 | down | `blk.24.ffn_down_exps.weight` | 15.122 | 8.516 | 0.003 | 23 | 0 |
| `dev_linear_equation` | 4 | down | `blk.4.ffn_down_exps.weight` | 45.959 | 23.524 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 57 | down | `blk.57.ffn_down_exps.weight` | 40.780 | 23.478 | 0.002 | 23 | 0 |
| `dev_linear_equation` | 60 | down | `blk.60.ffn_down_exps.weight` | 22.681 | 22.681 | 0.002 | 11 | 0 |
| `dev_japan_factual` | 6 | down | `blk.6.ffn_down_exps.weight` | 15.763 | 9.221 | 9.190 | 2 | 86 |
| `dev_photosynthesis_factual` | 15 | down | `blk.15.ffn_down_exps.weight` | 15.280 | 8.335 | 8.306 | 2 | 95 |
| `dev_python_reverse` | 11 | down | `blk.11.ffn_down_exps.weight` | 12.478 | 8.174 | 0.001 | 11 | 0 |
| `dev_japan_factual` | 10 | down | `blk.10.ffn_down_exps.weight` | 14.340 | 9.099 | 9.071 | 2 | 86 |
| `dev_python_reverse` | 6 | down | `blk.6.ffn_down_exps.weight` | 13.285 | 8.141 | 8.113 | 2 | 96 |
| `dev_photosynthesis_factual` | 9 | down | `blk.9.ffn_down_exps.weight` | 13.802 | 8.221 | 8.189 | 2 | 95 |
| `dev_photosynthesis_factual` | 17 | down | `blk.17.ffn_down_exps.weight` | 11.625 | 8.075 | 0.002 | 11 | 0 |
| `dev_zh_france` | 4 | down | `blk.4.ffn_down_exps.weight` | 28.677 | 15.409 | 0.002 | 23 | 0 |
| `dev_python_reverse` | 18 | down | `blk.18.ffn_down_exps.weight` | 14.879 | 7.936 | 7.866 | 2 | 96 |
| `dev_japan_factual` | 9 | down | `blk.9.ffn_down_exps.weight` | 15.640 | 8.799 | 8.770 | 2 | 86 |
| `dev_photosynthesis_factual` | 8 | down | `blk.8.ffn_down_exps.weight` | 12.749 | 7.946 | 7.917 | 2 | 95 |
| `dev_python_reverse` | 7 | down | `blk.7.ffn_down_exps.weight` | 13.995 | 7.693 | 7.665 | 2 | 96 |
| `dev_japan_factual` | 57 | down | `blk.57.ffn_down_exps.weight` | 16.670 | 8.574 | 0.002 | 23 | 0 |
| `dev_mixed_summary` | 6 | down | `blk.6.ffn_down_exps.weight` | 25.609 | 13.494 | 13.466 | 2 | 55 |
| `dev_linear_equation` | 21 | down | `blk.21.ffn_down_exps.weight` | 45.009 | 21.201 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 12 | down | `blk.12.ffn_down_exps.weight` | 15.026 | 8.470 | 0.002 | 11 | 0 |
| `dev_python_reverse` | 15 | down | `blk.15.ffn_down_exps.weight` | 14.059 | 7.556 | 7.529 | 2 | 96 |
| `dev_linear_equation` | 26 | down | `blk.26.ffn_down_exps.weight` | 43.753 | 21.096 | 0.002 | 23 | 0 |
| `dev_photosynthesis_factual` | 7 | down | `blk.7.ffn_down_exps.weight` | 12.464 | 7.539 | 7.446 | 2 | 95 |
| `dev_mixed_summary` | 9 | down | `blk.9.ffn_down_exps.weight` | 32.762 | 13.120 | 13.093 | 2 | 55 |
| `dev_linear_equation` | 25 | down | `blk.25.ffn_down_exps.weight` | 43.948 | 20.743 | 0.002 | 23 | 0 |
| `dev_japan_factual` | 3 | down | `blk.3.ffn_down_exps.weight` | 15.041 | 8.296 | 0.002 | 11 | 0 |

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
