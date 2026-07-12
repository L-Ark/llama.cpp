# Kimi Queue/Scheduler Evidence

Input: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_france_regression`

## Run Metrics

- quality: `pass`
- token_rate: `1.42`
- TTFT ms: `11949.34`
- decode: `21839.52 ms / 31 runs`
- expert_pack: `38796 misses=0 read_failures=0 direct_reads=0 direct_fallbacks=0 iouring_reads=38796 iouring_bytes=222635671552 iouring_fallbacks=0 iouring_submit_us=120398 iouring_wait_us=16755483 iouring_h2d_enqueues=38796 entries=69120`
- expert_pack_iouring: `batches=5756 submit_calls=9046 wait_calls=13722 cqes=38796 inflight_avg=4.52 inflight_max=8 batch_hist=1:182,2-4:2579,5-8:2818,9-16:0,17-32:0,gt32:177`

## Copy By Role

| role | rows | GiB | io wait ms | io wait/row | H2D ms | H2D/row | slot wait ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down` | 14150 | 90.245 | 50551.993 | 3.573 | 4978.752 | 0.352 | 0.000 | 50551.993 |
| `gate` | 12324 | 60.790 | 42976.471 | 3.487 | 3422.712 | 0.278 | 0.000 | 42976.471 |
| `up` | 12322 | 56.311 | 38661.432 | 3.138 | 3205.031 | 0.260 | 0.000 | 38661.432 |

## IO Batch By Op

| op | rows | read jobs | avg read jobs | avg initial submit | wait calls | wait ms | wait/read job | inflight avg | inflight max | slot wait ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4857 | 34445 | 7.092 | 4.671 | 12344 | 15205.671 | 0.441 | 4.032 | 8 | 56.923 | 29767.590 |
| `current_down_overlap` | 899 | 4351 | 4.840 | 4.840 | 1378 | 1556.562 | 0.358 | 4.450 | 8 | 13.134 | 3207.118 |

## Decode-Like IO Batch By Op (`read_jobs <= 8`)

| op | rows | read jobs | avg read jobs | avg initial submit | wait calls | wait ms | wait/read job | inflight avg | inflight max | slot wait ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4680 | 21272 | 4.545 | 4.545 | 8785 | 10325.987 | 0.485 | 3.969 | 8 | 38.455 | 19637.901 |
| `current_down_overlap` | 899 | 4351 | 4.840 | 4.840 | 1378 | 1556.562 | 0.358 | 4.450 | 8 | 13.134 | 3207.118 |

## IO Wait Trace

- wait rows: `13722`
- wait ms: `16762.233`
- wait p50/p95/p99 ms: `1.272` / `2.564` / `3.635`
- low inflight ratio: `0.037`
- queue empty ratio: `0.000`
- zero drain ratio: `0.000`
- next-job-done ratio: `0.760`

## Decode-Like IO Wait Trace (`read_jobs <= 8`)

- wait rows: `10163`
- wait ms: `11882.550`
- wait p50/p95/p99 ms: `1.263` / `2.509` / `3.317`
- low inflight ratio: `0.045`
- queue empty ratio: `0.000`
- zero drain ratio: `0.000`
- next-job-done ratio: `1.000`

## H2D Coalesce By Op

| op | rows | read jobs | copy count | coalesced ratio | bytes GiB | saved copies |
|---|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4857 | 34445 | 34445 | 1.000 | 181.785 | 0 |
| `current_down_overlap` | 899 | 4351 | 4351 | 1.000 | 25.561 | 0 |

## IO Locality By Op

| op | rows | read jobs | read GiB | span/read | gap/read | source switches/row | adjacent pairs/read job | same tensor ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4857 | 34445 | 181.787 | 15.976 | 14.976 | 0.221 | 0.163 | 1.000 |
| `current_down_overlap` | 899 | 4351 | 25.561 | 22.631 | 21.631 | 0.058 | 0.025 | 1.000 |

## Decode-Like IO Locality By Op (`read_jobs <= 8`)

| op | rows | read jobs | read GiB | span/read | gap/read | source switches/row | adjacent pairs/read job | same tensor ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 4680 | 21272 | 112.198 | 23.500 | 22.500 | 0.051 | 0.023 | 1.000 |
| `current_down_overlap` | 899 | 4351 | 25.561 | 22.631 | 21.631 | 0.058 | 0.025 | 1.000 |

## Top Locality Gaps

| op | first tensor | read jobs | sources | switches | read MiB | span MiB | gap/read | adjacent pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | `blk.2.ffn_down_exps.weight` | 3 | 1 | 0 | 18.047 | 1648.281 | 90.333 | 0 |
| `runtime_load` | `blk.2.ffn_down_exps.weight` | 4 | 1 | 0 | 24.062 | 2141.562 | 88.000 | 0 |
| `runtime_load` | `blk.4.ffn_up_exps.weight` | 2 | 1 | 0 | 10.719 | 932.531 | 86.000 | 0 |
| `runtime_load` | `blk.4.ffn_gate_exps.weight` | 2 | 1 | 0 | 10.719 | 932.531 | 86.000 | 0 |
| `runtime_load` | `blk.57.ffn_up_exps.weight` | 2 | 1 | 0 | 10.719 | 927.172 | 85.500 | 0 |
| `runtime_load` | `blk.57.ffn_gate_exps.weight` | 2 | 1 | 0 | 10.719 | 927.172 | 85.500 | 0 |
| `runtime_load` | `blk.57.ffn_down_exps.weight` | 2 | 1 | 0 | 14.875 | 1286.688 | 85.500 | 0 |
| `runtime_load` | `blk.2.ffn_down_exps.weight` | 3 | 1 | 0 | 18.047 | 1503.906 | 82.333 | 0 |
| `runtime_load` | `blk.59.ffn_up_exps.weight` | 2 | 1 | 0 | 10.719 | 857.500 | 79.000 | 0 |
| `runtime_load` | `blk.59.ffn_gate_exps.weight` | 2 | 1 | 0 | 10.719 | 857.500 | 79.000 | 0 |
| `runtime_load` | `blk.38.ffn_up_exps.weight` | 2 | 1 | 0 | 8.969 | 659.203 | 72.500 | 0 |
| `runtime_load` | `blk.38.ffn_gate_exps.weight` | 2 | 1 | 0 | 10.719 | 787.828 | 72.500 | 0 |
| `current_down_overlap` | `blk.38.ffn_down_exps.weight` | 2 | 1 | 0 | 12.031 | 884.297 | 72.500 | 0 |
| `runtime_load` | `blk.22.ffn_up_exps.weight` | 2 | 1 | 0 | 8.969 | 654.719 | 72.000 | 0 |
| `runtime_load` | `blk.22.ffn_gate_exps.weight` | 2 | 1 | 0 | 8.969 | 654.719 | 72.000 | 0 |
| `current_down_overlap` | `blk.55.ffn_down_exps.weight` | 2 | 1 | 0 | 12.031 | 866.250 | 71.000 | 0 |

## Up/Gate By Type

| type | calls | wall ms | wall/call | up miss/call | gate miss/call | up wait/call | gate wait/call | up compute/call | gate compute/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `up=22/gate=18/parallel=1` | 899 | 8410.555 | 9.355 | 3.981 | 3.981 | 4.717 | 5.341 | 0.176 | 0.151 |
| `up=22/gate=22/parallel=1` | 558 | 3180.855 | 5.700 | 4.484 | 4.484 | 5.058 | 5.467 | 0.212 | 0.119 |
| `up=18/gate=18/parallel=0` | 311 | 2383.677 | 7.665 | 4.614 | 4.617 | 0.000 | 0.000 | 0.000 | 0.000 |
| `up=18/gate=22/parallel=1` | 93 | 553.037 | 5.947 | 4.462 | 4.473 | 5.448 | 5.650 | 0.161 | 0.123 |

## Top Copy Tensors By IO Wait

| tensor | role | rows | GiB | io wait ms | io wait/row | H2D ms |
|---|---|---:|---:|---:|---:|---:|
| `blk.20.ffn_down_exps.weight` | `down` | 240 | 1.743 | 1333.267 | 5.555 | 116.043 |
| `blk.4.ffn_down_exps.weight` | `down` | 280 | 2.034 | 1257.249 | 4.490 | 117.295 |
| `blk.1.ffn_up_exps.weight` | `up` | 238 | 1.042 | 1240.790 | 5.213 | 65.834 |
| `blk.6.ffn_down_exps.weight` | `down` | 250 | 1.923 | 1206.142 | 4.825 | 111.648 |
| `blk.1.ffn_gate_exps.weight` | `gate` | 238 | 1.042 | 1149.140 | 4.828 | 65.420 |
| `blk.7.ffn_down_exps.weight` | `down` | 257 | 1.976 | 1145.919 | 4.459 | 111.874 |
| `blk.1.ffn_down_exps.weight` | `down` | 283 | 1.663 | 1102.031 | 3.894 | 95.574 |
| `blk.10.ffn_down_exps.weight` | `down` | 247 | 1.900 | 1097.158 | 4.442 | 108.919 |
| `blk.24.ffn_down_exps.weight` | `down` | 259 | 1.881 | 1057.556 | 4.083 | 103.188 |
| `blk.17.ffn_down_exps.weight` | `down` | 232 | 1.363 | 1014.171 | 4.371 | 76.173 |
| `blk.29.ffn_gate_exps.weight` | `gate` | 236 | 1.235 | 1000.663 | 4.240 | 70.209 |
| `blk.9.ffn_down_exps.weight` | `down` | 247 | 1.900 | 998.306 | 4.042 | 98.788 |
| `blk.23.ffn_down_exps.weight` | `down` | 254 | 1.845 | 996.113 | 3.922 | 101.085 |
| `blk.16.ffn_down_exps.weight` | `down` | 247 | 1.794 | 981.991 | 3.976 | 96.585 |
| `blk.26.ffn_down_exps.weight` | `down` | 247 | 1.794 | 976.517 | 3.954 | 98.831 |
| `blk.18.ffn_down_exps.weight` | `down` | 235 | 1.807 | 966.004 | 4.111 | 98.952 |
| `blk.5.ffn_down_exps.weight` | `down` | 288 | 1.692 | 964.755 | 3.350 | 89.093 |
| `blk.28.ffn_gate_exps.weight` | `gate` | 240 | 1.256 | 964.421 | 4.018 | 75.297 |
| `blk.21.ffn_down_exps.weight` | `down` | 241 | 1.750 | 961.007 | 3.988 | 98.124 |
| `blk.29.ffn_down_exps.weight` | `down` | 274 | 1.610 | 959.189 | 3.501 | 88.425 |

## Interpretation Rules

- Low inflight or queue-empty waits point toward queue continuity/co-submit.
- High inflight with long waits points toward storage latency/throughput or refill policy.
- Large H2D time with poor coalescing points toward staging layout.
- High locality gap/read with same-tensor batches points toward expert pack layout or route-order locality, not larger cross-role batches.
- Large upgate wait without corresponding IO wait points toward CUDA stream sync or cache slot readiness.

This report is profiling evidence only and makes no SOTA claim.
