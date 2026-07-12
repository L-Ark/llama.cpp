# Kimi Queue/Scheduler Evidence

Input: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_intelligence_general`

## Run Metrics

- quality: `pass`
- token_rate: `1.5`
- TTFT ms: `10052.75`
- decode: `20687.58 ms / 31 runs`
- expert_pack: `37043 misses=0 read_failures=0 direct_reads=0 direct_fallbacks=0 iouring_reads=37043 iouring_bytes=212793262080 iouring_fallbacks=0 iouring_submit_us=123080 iouring_wait_us=15102679 iouring_h2d_enqueues=37043 entries=69120`
- expert_pack_iouring: `batches=5732 submit_calls=8466 wait_calls=12850 cqes=37043 inflight_avg=4.47 inflight_max=8 batch_hist=1:205,2-4:2478,5-8:2872,9-16:0,17-32:0,gt32:177`

## Copy By Role

| role | rows | GiB | io wait ms | io wait/row | H2D ms | H2D/row | slot wait ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down` | 13587 | 86.525 | 46934.629 | 3.454 | 4830.403 | 0.356 | 0.000 | 46934.629 |
| `gate` | 11730 | 58.023 | 39841.241 | 3.397 | 3335.196 | 0.284 | 0.000 | 39841.241 |
| `up` | 11726 | 53.631 | 36700.815 | 3.130 | 3130.030 | 0.267 | 0.000 | 36700.815 |

## IO Batch By Op

| op | rows | read jobs | avg read jobs | avg initial submit | wait calls | wait ms | wait/read job | inflight avg | inflight max | slot wait ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4833 | 32529 | 6.731 | 4.733 | 11588 | 13578.843 | 0.417 | 4.088 | 8 | 54.278 | 27763.014 |
| `current_down_overlap` | 899 | 4514 | 5.021 | 5.021 | 1262 | 1530.154 | 0.339 | 4.698 | 8 | 9.512 | 3243.500 |

## Decode-Like IO Batch By Op (`read_jobs <= 8`)

| op | rows | read jobs | avg read jobs | avg initial submit | wait calls | wait ms | wait/read job | inflight avg | inflight max | slot wait ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4656 | 21459 | 4.609 | 4.609 | 8552 | 9593.827 | 0.447 | 4.029 | 8 | 41.303 | 19269.299 |
| `current_down_overlap` | 899 | 4514 | 5.021 | 5.021 | 1262 | 1530.154 | 0.339 | 4.698 | 8 | 9.512 | 3243.500 |

## IO Wait Trace

- wait rows: `12850`
- wait ms: `15108.997`
- wait p50/p95/p99 ms: `1.276` / `2.413` / `3.083`
- low inflight ratio: `0.056`
- queue empty ratio: `0.000`
- zero drain ratio: `0.000`
- next-job-done ratio: `0.787`

## Decode-Like IO Wait Trace (`read_jobs <= 8`)

- wait rows: `9814`
- wait ms: `11123.981`
- wait p50/p95/p99 ms: `1.265` / `2.377` / `2.912`
- low inflight ratio: `0.068`
- queue empty ratio: `0.000`
- zero drain ratio: `0.000`
- next-job-done ratio: `1.000`

## H2D Coalesce By Op

| op | rows | read jobs | copy count | coalesced ratio | bytes GiB | saved copies |
|---|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4833 | 32529 | 32529 | 1.000 | 171.661 | 0 |
| `current_down_overlap` | 899 | 4514 | 4514 | 1.000 | 26.518 | 0 |

## IO Locality By Op

| op | rows | read jobs | read GiB | span/read | gap/read | source switches/row | adjacent pairs/read job | same tensor ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4833 | 32529 | 171.695 | 23.869 | 22.869 | 2.059 | 0.084 | 1.000 |
| `current_down_overlap` | 899 | 4514 | 26.524 | 32.966 | 31.966 | 1.722 | 0.013 | 1.000 |

## Decode-Like IO Locality By Op (`read_jobs <= 8`)

| op | rows | read jobs | read GiB | span/read | gap/read | source switches/row | adjacent pairs/read job | same tensor ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4656 | 21459 | 113.232 | 32.182 | 31.182 | 1.415 | 0.014 | 1.000 |
| `current_down_overlap` | 899 | 4514 | 26.524 | 32.966 | 31.966 | 1.722 | 0.013 | 1.000 |

## Top Locality Gaps

| op | first tensor | read jobs | sources | switches | read MiB | span MiB | gap/read | adjacent pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | `blk.46.ffn_gate_exps.weight` | 2 | 1 | 0 | 10.727 | 1988.332 | 184.365 | 0 |
| `runtime_load` | `blk.46.ffn_up_exps.weight` | 2 | 1 | 0 | 8.977 | 1663.707 | 184.339 | 0 |
| `runtime_load` | `blk.48.ffn_gate_exps.weight` | 2 | 1 | 0 | 10.727 | 1650.691 | 152.888 | 0 |
| `runtime_load` | `blk.48.ffn_up_exps.weight` | 2 | 1 | 0 | 8.977 | 1381.191 | 152.866 | 0 |
| `runtime_load` | `blk.15.ffn_up_exps.weight` | 2 | 1 | 0 | 8.977 | 1206.301 | 133.383 | 0 |
| `runtime_load` | `blk.15.ffn_gate_exps.weight` | 2 | 1 | 0 | 8.977 | 1206.301 | 133.383 | 0 |
| `runtime_load` | `blk.51.ffn_gate_exps.weight` | 4 | 2 | 2 | 21.445 | 2701.129 | 124.954 | 0 |
| `runtime_load` | `blk.51.ffn_up_exps.weight` | 4 | 2 | 2 | 17.945 | 2260.129 | 124.945 | 0 |
| `current_down_overlap` | `blk.46.ffn_down_exps.weight` | 3 | 1 | 0 | 18.059 | 2231.801 | 122.587 | 0 |
| `runtime_load` | `blk.29.ffn_gate_exps.weight` | 3 | 2 | 2 | 16.086 | 1934.738 | 119.275 | 0 |
| `runtime_load` | `blk.29.ffn_up_exps.weight` | 3 | 2 | 2 | 13.461 | 1618.863 | 119.264 | 0 |
| `current_down_overlap` | `blk.41.ffn_down_exps.weight` | 3 | 2 | 2 | 18.055 | 2087.426 | 114.617 | 0 |
| `runtime_load` | `blk.21.ffn_down_exps.weight` | 4 | 2 | 3 | 29.758 | 3421.254 | 113.970 | 0 |
| `runtime_load` | `blk.38.ffn_gate_exps.weight` | 3 | 2 | 2 | 16.086 | 1838.270 | 113.278 | 0 |
| `runtime_load` | `blk.38.ffn_up_exps.weight` | 3 | 2 | 2 | 13.461 | 1538.145 | 113.267 | 0 |
| `runtime_load` | `blk.37.ffn_gate_exps.weight` | 2 | 1 | 0 | 10.727 | 1205.863 | 111.418 | 0 |

## Up/Gate By Type

| type | calls | wall ms | wall/call | up miss/call | gate miss/call | up wait/call | gate wait/call | up compute/call | gate compute/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `up=22/gate=18/parallel=1` | 899 | 8259.500 | 9.187 | 4.192 | 4.194 | 4.746 | 5.186 | 0.153 | 0.097 |
| `up=22/gate=22/parallel=1` | 558 | 2967.232 | 5.318 | 4.165 | 4.167 | 4.894 | 5.088 | 0.161 | 0.119 |
| `up=18/gate=18/parallel=0` | 311 | 2385.914 | 7.672 | 4.923 | 4.929 | 0.000 | 0.000 | 0.000 | 0.000 |
| `up=18/gate=22/parallel=1` | 93 | 527.056 | 5.667 | 4.430 | 4.430 | 5.346 | 5.453 | 0.116 | 0.104 |

## Top Copy Tensors By IO Wait

| tensor | role | rows | GiB | io wait ms | io wait/row | H2D ms |
|---|---|---:|---:|---:|---:|---:|
| `blk.4.ffn_down_exps.weight` | `down` | 264 | 1.917 | 1212.779 | 4.594 | 112.381 |
| `blk.1.ffn_up_exps.weight` | `up` | 208 | 0.911 | 1182.848 | 5.687 | 60.592 |
| `blk.6.ffn_down_exps.weight` | `down` | 245 | 1.884 | 1169.771 | 4.775 | 109.801 |
| `blk.3.ffn_down_exps.weight` | `down` | 285 | 1.674 | 1118.625 | 3.925 | 98.007 |
| `blk.7.ffn_down_exps.weight` | `down` | 252 | 1.938 | 1062.233 | 4.215 | 111.218 |
| `blk.9.ffn_down_exps.weight` | `down` | 243 | 1.869 | 1061.803 | 4.370 | 108.399 |
| `blk.26.ffn_down_exps.weight` | `down` | 265 | 1.925 | 1050.034 | 3.962 | 108.432 |
| `blk.20.ffn_down_exps.weight` | `down` | 245 | 1.779 | 978.592 | 3.994 | 100.641 |
| `blk.1.ffn_gate_exps.weight` | `gate` | 208 | 0.911 | 969.600 | 4.662 | 60.871 |
| `blk.1.ffn_down_exps.weight` | `down` | 241 | 1.416 | 902.455 | 3.745 | 79.871 |
| `blk.10.ffn_down_exps.weight` | `down` | 225 | 1.730 | 889.945 | 3.955 | 91.176 |
| `blk.5.ffn_down_exps.weight` | `down` | 258 | 1.516 | 885.864 | 3.434 | 89.755 |
| `blk.15.ffn_down_exps.weight` | `down` | 216 | 1.661 | 882.479 | 4.086 | 94.787 |
| `blk.9.ffn_up_exps.weight` | `up` | 217 | 1.136 | 872.985 | 4.023 | 65.004 |
| `blk.25.ffn_down_exps.weight` | `down` | 217 | 1.576 | 855.049 | 3.940 | 89.384 |
| `blk.22.ffn_down_exps.weight` | `down` | 217 | 1.576 | 854.160 | 3.936 | 89.298 |
| `blk.58.ffn_down_exps.weight` | `down` | 231 | 1.678 | 854.133 | 3.698 | 92.876 |
| `blk.2.ffn_down_exps.weight` | `down` | 248 | 1.457 | 852.047 | 3.436 | 84.806 |
| `blk.18.ffn_down_exps.weight` | `down` | 212 | 1.630 | 838.595 | 3.956 | 89.273 |
| `blk.32.ffn_gate_exps.weight` | `gate` | 213 | 1.115 | 831.748 | 3.905 | 66.225 |

## Interpretation Rules

- Low inflight or queue-empty waits point toward queue continuity/co-submit.
- High inflight with long waits points toward storage latency/throughput or refill policy.
- Large H2D time with poor coalescing points toward staging layout.
- High locality gap/read with same-tensor batches points toward expert pack layout or route-order locality, not larger cross-role batches.
- Large upgate wait without corresponding IO wait points toward CUDA stream sync or cache slot readiness.

This report is profiling evidence only and makes no SOTA claim.
