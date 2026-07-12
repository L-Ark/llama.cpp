# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-dev-n32-035310/dev_intelligence_general`
- prompt id: `dev_intelligence_general`
- token rate: `1.54 tok/s`
- TTFT: `10050.75 ms`
- decode: `20072.41 ms / 31`
- memory peak: `12602880000`

## High-Level Split

- total profiled copy wall: `121791 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `121791 ms` over `198.18 GiB`
- iouring wall: `121791 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 11730 | 58.02 | 0 | 39404 | 3325 | 181 | 0 | 39404 |
| `runtime_load` | up | 1 | 1 | 11726 | 53.63 | 0 | 35526 | 3102 | 271 | 0 | 35526 |
| `runtime_load` | down | 1 | 1 | 9073 | 60.01 | 0 | 33365 | 3305 | 121 | 0 | 33365 |
| `current_down_overlap` | down | 1 | 1 | 4514 | 26.52 | 0 | 13496 | 1435 | 58 | 0 | 13496 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 1214 | `runtime_load` | `blk.6.ffn_down_exps.weight` | 1 | 1 |
| 1135 | `runtime_load` | `blk.3.ffn_down_exps.weight` | 1 | 1 |
| 1114 | `runtime_load` | `blk.4.ffn_down_exps.weight` | 1 | 1 |
| 1110 | `runtime_load` | `blk.7.ffn_down_exps.weight` | 1 | 1 |
| 1072 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 1045 | `runtime_load` | `blk.26.ffn_down_exps.weight` | 1 | 1 |
| 1040 | `runtime_load` | `blk.9.ffn_down_exps.weight` | 1 | 1 |
| 998 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 965 | `runtime_load` | `blk.20.ffn_down_exps.weight` | 1 | 1 |
| 905 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 1 |
| 895 | `runtime_load` | `blk.10.ffn_down_exps.weight` | 1 | 1 |
| 891 | `runtime_load` | `blk.23.ffn_down_exps.weight` | 1 | 1 |
| 889 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 1 | 1 |
| 876 | `runtime_load` | `blk.5.ffn_down_exps.weight` | 1 | 1 |
| 861 | `runtime_load` | `blk.15.ffn_down_exps.weight` | 1 | 1 |
| 843 | `runtime_load` | `blk.24.ffn_down_exps.weight` | 1 | 1 |
| 838 | `runtime_load` | `blk.2.ffn_down_exps.weight` | 1 | 1 |
| 830 | `runtime_load` | `blk.34.ffn_gate_exps.weight` | 1 | 1 |
| 829 | `runtime_load` | `blk.19.ffn_down_exps.weight` | 1 | 1 |
| 827 | `runtime_load` | `blk.18.ffn_down_exps.weight` | 1 | 1 |
| 822 | `runtime_load` | `blk.43.ffn_gate_exps.weight` | 1 | 1 |
| 820 | `runtime_load` | `blk.9.ffn_up_exps.weight` | 1 | 1 |
| 814 | `runtime_load` | `blk.25.ffn_down_exps.weight` | 1 | 1 |
| 811 | `runtime_load` | `blk.8.ffn_down_exps.weight` | 1 | 1 |
| 811 | `runtime_load` | `blk.59.ffn_down_exps.weight` | 1 | 1 |
| 793 | `runtime_load` | `blk.32.ffn_gate_exps.weight` | 1 | 1 |
| 793 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 792 | `runtime_load` | `blk.40.ffn_gate_exps.weight` | 1 | 1 |
| 791 | `runtime_load` | `blk.30.ffn_gate_exps.weight` | 1 | 1 |
| 789 | `runtime_load` | `blk.48.ffn_gate_exps.weight` | 1 | 1 |

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
