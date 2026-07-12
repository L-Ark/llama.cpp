# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855/dev_france_regression`
- prompt id: `dev_france_regression`
- token rate: `1.56 tok/s`
- TTFT: `12251.96 ms`
- decode: `54611.50 ms / 85`
- memory peak: `12808163328`

## High-Level Split

- total profiled copy wall: `271907 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `271907 ms` over `452.62 GiB`
- iouring wall: `271907 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 26908 | 132.52 | 0 | 89888 | 7406 | 404 | 0 | 89888 |
| `runtime_load` | up | 1 | 1 | 26905 | 123.20 | 0 | 80232 | 6880 | 481 | 0 | 80232 |
| `runtime_load` | down | 1 | 1 | 19053 | 127.85 | 0 | 68341 | 6824 | 242 | 0 | 68341 |
| `current_down_overlap` | down | 1 | 1 | 11754 | 69.05 | 0 | 33445 | 3620 | 147 | 0 | 33445 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 2768 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 2561 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 2447 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 2434 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 2398 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 2386 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 2259 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 2222 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 2193 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 2185 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 2152 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 2116 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 2077 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 2037 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 2009 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 2000 | `runtime_load` | `blk.29.ffn_gate_exps.weight` | 1 | 1 |
| 1960 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 1953 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 1935 | `runtime_load` | `blk.21.ffn_down_exps.weight` | 1 | 1 |
| 1930 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 1899 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 1895 | `runtime_load` | `blk.16.ffn_down_exps.weight` | 1 | 1 |
| 1892 | `runtime_load` | `blk.15.ffn_down_exps.weight` | 1 | 1 |
| 1891 | `runtime_load` | `blk.22.ffn_down_exps.weight` | 1 | 1 |
| 1822 | `runtime_load` | `blk.33.ffn_gate_exps.weight` | 1 | 1 |
| 1807 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 1 |
| 1804 | `runtime_load` | `blk.7.ffn_up_exps.weight` | 1 | 1 |
| 1799 | `runtime_load` | `blk.10.ffn_gate_exps.weight` | 1 | 1 |
| 1796 | `runtime_load` | `blk.2.ffn_down_exps.weight` | 1 | 1 |
| 1786 | `runtime_load` | `blk.9.ffn_up_exps.weight` | 1 | 1 |

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
