# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-baseline-cpudefer-n96-082136/dev_france_regression`
- prompt id: `dev_france_regression`
- token rate: `1.53 tok/s`
- TTFT: `12004.97 ms`
- decode: `55478.89 ms / 85`
- memory peak: `12812529664`

## High-Level Split

- total profiled copy wall: `278508 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `278508 ms` over `452.62 GiB`
- iouring wall: `278508 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 26908 | 132.52 | 0 | 91295 | 7586 | 412 | 0 | 91295 |
| `runtime_load` | up | 1 | 1 | 26905 | 123.20 | 0 | 83822 | 7045 | 490 | 0 | 83822 |
| `runtime_load` | down | 1 | 1 | 19053 | 127.85 | 0 | 69655 | 6981 | 251 | 0 | 69655 |
| `current_down_overlap` | down | 1 | 1 | 11754 | 69.05 | 0 | 33737 | 3734 | 149 | 0 | 33737 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 2661 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 2564 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 2558 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 2475 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 2447 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 2433 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 2336 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 2315 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 2209 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 2179 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 2164 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 2147 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 2095 | `runtime_load` | `blk.29.ffn_gate_exps.weight` | 1 | 1 |
| 2087 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 2064 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 2048 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 2026 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 1980 | `runtime_load` | `blk.16.ffn_down_exps.weight` | 1 | 1 |
| 1971 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 1969 | `runtime_load` | `blk.22.ffn_down_exps.weight` | 1 | 1 |
| 1941 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 1929 | `runtime_load` | `blk.21.ffn_down_exps.weight` | 1 | 1 |
| 1928 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 1897 | `runtime_load` | `blk.10.ffn_gate_exps.weight` | 1 | 1 |
| 1886 | `runtime_load` | `blk.7.ffn_up_exps.weight` | 1 | 1 |
| 1884 | `runtime_load` | `blk.9.ffn_up_exps.weight` | 1 | 1 |
| 1859 | `runtime_load` | `blk.30.ffn_gate_exps.weight` | 1 | 1 |
| 1859 | `runtime_load` | `blk.10.ffn_up_exps.weight` | 1 | 1 |
| 1850 | `runtime_load` | `blk.2.ffn_down_exps.weight` | 1 | 1 |
| 1847 | `runtime_load` | `blk.33.ffn_gate_exps.weight` | 1 | 1 |

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
