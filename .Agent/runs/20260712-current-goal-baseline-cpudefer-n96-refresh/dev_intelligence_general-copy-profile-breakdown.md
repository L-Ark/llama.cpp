# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-baseline-cpudefer-n96-082136/dev_intelligence_general`
- prompt id: `dev_intelligence_general`
- token rate: `1.52 tok/s`
- TTFT: `9863.44 ms`
- decode: `62548.34 ms / 95`
- memory peak: `12639948800`

## High-Level Split

- total profiled copy wall: `296108 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `296108 ms` over `482.88 GiB`
- iouring wall: `296108 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 28350 | 140.03 | 0 | 97467 | 8057 | 446 | 0 | 97467 |
| `runtime_load` | up | 1 | 1 | 28354 | 130.03 | 0 | 86743 | 7452 | 508 | 0 | 86743 |
| `runtime_load` | down | 1 | 1 | 19867 | 133.32 | 0 | 71081 | 7273 | 255 | 0 | 71081 |
| `current_down_overlap` | down | 1 | 1 | 13534 | 79.51 | 0 | 40816 | 4279 | 176 | 0 | 40816 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 2946 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 2787 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 2770 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 2632 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 2547 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 2387 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 2380 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 2360 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 2335 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 2325 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 2275 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 1 | 1 |
| 2266 | `runtime_load` | `blk.2.ffn_down_exps.weight` | 1 | 1 |
| 2235 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 2188 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 1 |
| 2121 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 2107 | `runtime_load` | `blk.54.ffn_gate_exps.weight` | 1 | 1 |
| 2104 | `runtime_load` | `blk.59.ffn_down_exps.weight` | 1 | 1 |
| 2093 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 2043 | `runtime_load` | `blk.53.ffn_gate_exps.weight` | 1 | 1 |
| 1997 | `runtime_load` | `blk.9.ffn_up_exps.weight` | 1 | 1 |
| 1984 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 1975 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 1975 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 1969 | `runtime_load` | `blk.15.ffn_down_exps.weight` | 1 | 1 |
| 1952 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 1947 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 1939 | `runtime_load` | `blk.52.ffn_gate_exps.weight` | 1 | 1 |
| 1897 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 1896 | `runtime_load` | `blk.30.ffn_gate_exps.weight` | 1 | 1 |
| 1856 | `runtime_load` | `blk.48.ffn_gate_exps.weight` | 1 | 1 |

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
