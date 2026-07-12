# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-n96`
- prompt id: `dev_france_regression`
- token rate: `1.54 tok/s`
- TTFT: `11828.71 ms`
- decode: `55246.55 ms / 85`
- memory peak: `12812201984`

## High-Level Split

- total profiled copy wall: `276366 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `276366 ms` over `452.62 GiB`
- iouring wall: `276366 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 26908 | 132.52 | 0 | 90819 | 7417 | 411 | 0 | 90819 |
| `runtime_load` | up | 1 | 1 | 26905 | 123.20 | 0 | 82956 | 6825 | 500 | 0 | 82956 |
| `runtime_load` | down | 1 | 1 | 19053 | 127.85 | 0 | 68659 | 6794 | 250 | 0 | 68659 |
| `current_down_overlap` | down | 1 | 1 | 11754 | 69.05 | 0 | 33932 | 3707 | 155 | 0 | 33932 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 2800 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 2534 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 2413 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 2390 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 2379 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 2370 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 2256 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 2220 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 2218 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 2181 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 2119 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 2110 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 2097 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 2066 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 2060 | `runtime_load` | `blk.29.ffn_gate_exps.weight` | 1 | 1 |
| 2036 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 1966 | `runtime_load` | `blk.21.ffn_down_exps.weight` | 1 | 1 |
| 1944 | `runtime_load` | `blk.16.ffn_down_exps.weight` | 1 | 1 |
| 1930 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 1912 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 1909 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 1893 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 1865 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 1 | 1 |
| 1864 | `runtime_load` | `blk.22.ffn_down_exps.weight` | 1 | 1 |
| 1859 | `runtime_load` | `blk.15.ffn_down_exps.weight` | 1 | 1 |
| 1853 | `runtime_load` | `blk.30.ffn_gate_exps.weight` | 1 | 1 |
| 1841 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 1 |
| 1839 | `runtime_load` | `blk.7.ffn_up_exps.weight` | 1 | 1 |
| 1818 | `runtime_load` | `blk.9.ffn_up_exps.weight` | 1 | 1 |
| 1815 | `runtime_load` | `blk.33.ffn_gate_exps.weight` | 1 | 1 |

## Interpretation

- This run has no `pack_hit=0` copy rows. The profiled movement is already
  served through the expert-pack/io_uring path.
- The remaining cost is therefore not GGUF fallback or missing expert-pack
  coverage. It is VRAM-cache miss handling: expert-pack io_uring read,
  pinned staging, H2D enqueue/copy, and synchronization around demand reads.
- The next candidate should target batch depth, queue continuity, cache
  admission, RAM/VRAM tiering, or a lower-byte expert representation.
  Rebuilding another prompt-specific pack is not justified by this profile.
- Token rate from this diagnostic run should not be used as SOTA because
  `COPY_PROFILE_H2D=1` adds synchronization for measurement.
