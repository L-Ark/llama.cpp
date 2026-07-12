# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-dev-n32-035310/dev_france_regression`
- prompt id: `dev_france_regression`
- token rate: `1.51 tok/s`
- TTFT: `13538.01 ms`
- decode: `20513.44 ms / 31`
- memory peak: `12773163008`

## High-Level Split

- total profiled copy wall: `131336 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `131336 ms` over `207.35 GiB`
- iouring wall: `131336 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 12324 | 60.79 | 0 | 42296 | 3405 | 232 | 0 | 42296 |
| `runtime_load` | up | 1 | 1 | 12322 | 56.31 | 0 | 38907 | 3233 | 390 | 0 | 38907 |
| `runtime_load` | down | 1 | 1 | 9799 | 64.68 | 0 | 37778 | 3555 | 190 | 0 | 37778 |
| `current_down_overlap` | down | 1 | 1 | 4351 | 25.56 | 0 | 12355 | 1362 | 57 | 0 | 12355 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 1425 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 1285 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 1190 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 1150 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 1058 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 1053 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 1047 | `runtime_load` | `blk.48.ffn_gate_exps.weight` | 1 | 1 |
| 1046 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 1040 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 1039 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 1 |
| 1036 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 1 | 1 |
| 1022 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 993 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 991 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 990 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 980 | `runtime_load` | `blk.16.ffn_down_exps.weight` | 1 | 1 |
| 962 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 959 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 954 | `runtime_load` | `blk.29.ffn_gate_exps.weight` | 1 | 1 |
| 954 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 952 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 946 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 941 | `runtime_load` | `blk.2.ffn_down_exps.weight` | 1 | 1 |
| 912 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 912 | `runtime_load` | `blk.59.ffn_down_exps.weight` | 1 | 1 |
| 908 | `runtime_load` | `blk.21.ffn_down_exps.weight` | 1 | 1 |
| 899 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 881 | `runtime_load` | `blk.32.ffn_gate_exps.weight` | 1 | 1 |
| 869 | `runtime_load` | `blk.3.ffn_up_exps.weight` | 1 | 1 |
| 863 | `runtime_load` | `blk.4.ffn_gate_exps.weight` | 1 | 1 |

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
