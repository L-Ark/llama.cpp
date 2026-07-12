# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855/dev_intelligence_general`
- prompt id: `dev_intelligence_general`
- token rate: `1.58 tok/s`
- TTFT: `9435.44 ms`
- decode: `60059.09 ms / 95`
- memory peak: `12638240768`

## High-Level Split

- total profiled copy wall: `286726 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `286726 ms` over `482.88 GiB`
- iouring wall: `286726 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 28350 | 140.03 | 0 | 93905 | 7886 | 433 | 0 | 93905 |
| `runtime_load` | up | 1 | 1 | 28354 | 130.03 | 0 | 84075 | 7459 | 494 | 0 | 84075 |
| `runtime_load` | down | 1 | 1 | 19867 | 133.32 | 0 | 69072 | 7155 | 253 | 0 | 69072 |
| `current_down_overlap` | down | 1 | 1 | 13534 | 79.51 | 0 | 39675 | 4293 | 173 | 0 | 39675 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 2839 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 2696 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 2649 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 2601 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 2523 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 2310 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 2306 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 2306 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 2237 | `runtime_load` | `blk.2.ffn_down_exps.weight` | 1 | 1 |
| 2221 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 2198 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 2182 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 2166 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 1 | 1 |
| 2052 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 2048 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 1 |
| 2041 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 2027 | `runtime_load` | `blk.59.ffn_down_exps.weight` | 1 | 1 |
| 1999 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 1988 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 1984 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 1976 | `runtime_load` | `blk.15.ffn_down_exps.weight` | 1 | 1 |
| 1976 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 1966 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 1860 | `runtime_load` | `blk.9.ffn_up_exps.weight` | 1 | 1 |
| 1858 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 1844 | `runtime_load` | `blk.30.ffn_gate_exps.weight` | 1 | 1 |
| 1826 | `runtime_load` | `blk.7.ffn_up_exps.weight` | 1 | 1 |
| 1814 | `runtime_load` | `blk.12.ffn_down_exps.weight` | 1 | 1 |
| 1806 | `runtime_load` | `blk.26.ffn_gate_exps.weight` | 1 | 1 |
| 1802 | `runtime_load` | `blk.32.ffn_gate_exps.weight` | 1 | 1 |

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
