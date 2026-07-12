# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-head-cpudefer-n96-050308/dev_france_regression`
- prompt id: `dev_france_regression`
- token rate: `1.52 tok/s`
- TTFT: `11867.62 ms`
- decode: `55887.45 ms / 85`
- memory peak: `12809277440`

## High-Level Split

- total profiled copy wall: `274544 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `274544 ms` over `452.62 GiB`
- iouring wall: `274544 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 26908 | 132.52 | 0 | 90972 | 7459 | 406 | 0 | 90972 |
| `runtime_load` | up | 1 | 1 | 26905 | 123.20 | 0 | 81705 | 6927 | 477 | 0 | 81705 |
| `runtime_load` | down | 1 | 1 | 19053 | 127.85 | 0 | 68326 | 6883 | 249 | 0 | 68326 |
| `current_down_overlap` | down | 1 | 1 | 11754 | 69.05 | 0 | 33542 | 3699 | 163 | 0 | 33542 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 2672 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 2479 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 2436 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 2407 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 2372 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 2330 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 2255 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 2240 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 2181 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 2160 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 2158 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 2120 | `runtime_load` | `blk.29.ffn_gate_exps.weight` | 1 | 1 |
| 2090 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 2078 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 2070 | `runtime_load` | `blk.31.ffn_gate_exps.weight` | 1 | 1 |
| 2045 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 2003 | `runtime_load` | `blk.7.ffn_up_exps.weight` | 1 | 1 |
| 1977 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 1963 | `runtime_load` | `blk.15.ffn_down_exps.weight` | 1 | 1 |
| 1962 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 1962 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 1956 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 1938 | `runtime_load` | `blk.16.ffn_down_exps.weight` | 1 | 1 |
| 1927 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 1918 | `runtime_load` | `blk.21.ffn_down_exps.weight` | 1 | 1 |
| 1907 | `runtime_load` | `blk.22.ffn_down_exps.weight` | 1 | 1 |
| 1872 | `runtime_load` | `blk.30.ffn_gate_exps.weight` | 1 | 1 |
| 1835 | `runtime_load` | `blk.33.ffn_gate_exps.weight` | 1 | 1 |
| 1823 | `runtime_load` | `blk.9.ffn_up_exps.weight` | 1 | 1 |
| 1811 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 1 |

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
