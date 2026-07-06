# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260706-133600Z-gp2-alias-python-reverse-n32`
- prompt id: `dev_python_reverse`
- token rate: `0.53 tok/s`
- TTFT: `90800.03 ms`
- decode: `58299.67 ms / 31`
- memory peak: `15899996160`

## High-Level Split

- total profiled copy wall: `74820 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `74820 ms` over `155.00 GiB`
- iouring wall: `10094 ms`
- non-iouring wall: `64726 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 0 | 9612 | 47.64 | 20031 | 0 | 2029 | 232 | 26 | 22384 |
| `runtime_load` | up | 1 | 0 | 9612 | 43.81 | 19430 | 0 | 1849 | 285 | 24 | 21653 |
| `current_down_overlap` | down | 1 | 0 | 4922 | 28.91 | 10638 | 0 | 1175 | 119 | 17 | 11933 |
| `runtime_load` | down | 1 | 0 | 3425 | 22.78 | 7770 | 0 | 915 | 72 | 9 | 8756 |
| `runtime_load` | gate | 1 | 1 | 673 | 3.25 | 0 | 3243 | 134 | 11 | 0 | 3243 |
| `runtime_load` | down | 1 | 1 | 723 | 4.48 | 0 | 3200 | 179 | 11 | 0 | 3200 |
| `runtime_load` | up | 1 | 1 | 673 | 3.14 | 0 | 3078 | 128 | 11 | 0 | 3078 |
| `current_down_overlap` | down | 1 | 1 | 169 | 0.99 | 0 | 573 | 40 | 3 | 0 | 573 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 895 | `runtime_load` | `blk.1.ffn_down_exps.weight` | 1 | 1 |
| 731 | `runtime_load` | `blk.2.ffn_down_exps.weight` | 1 | 1 |
| 647 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 1 | 0 |
| 634 | `runtime_load` | `blk.16.ffn_down_exps.weight` | 1 | 0 |
| 553 | `current_down_overlap` | `blk.42.ffn_down_exps.weight` | 1 | 0 |
| 549 | `runtime_load` | `blk.42.ffn_gate_exps.weight` | 1 | 0 |
| 544 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 1 | 0 |
| 532 | `current_down_overlap` | `blk.51.ffn_down_exps.weight` | 1 | 0 |
| 532 | `runtime_load` | `blk.51.ffn_gate_exps.weight` | 1 | 0 |
| 507 | `runtime_load` | `blk.57.ffn_gate_exps.weight` | 1 | 0 |
| 506 | `runtime_load` | `blk.60.ffn_down_exps.weight` | 1 | 0 |
| 494 | `runtime_load` | `blk.51.ffn_up_exps.weight` | 1 | 0 |
| 491 | `runtime_load` | `blk.60.ffn_up_exps.weight` | 1 | 0 |
| 486 | `runtime_load` | `blk.57.ffn_up_exps.weight` | 1 | 0 |
| 470 | `runtime_load` | `blk.60.ffn_gate_exps.weight` | 1 | 0 |
| 465 | `current_down_overlap` | `blk.34.ffn_down_exps.weight` | 1 | 0 |
| 462 | `runtime_load` | `blk.16.ffn_up_exps.weight` | 1 | 0 |
| 461 | `runtime_load` | `blk.55.ffn_gate_exps.weight` | 1 | 0 |
| 461 | `current_down_overlap` | `blk.36.ffn_down_exps.weight` | 1 | 0 |
| 457 | `current_down_overlap` | `blk.43.ffn_down_exps.weight` | 1 | 0 |
| 456 | `runtime_load` | `blk.42.ffn_up_exps.weight` | 1 | 0 |
| 455 | `current_down_overlap` | `blk.44.ffn_down_exps.weight` | 1 | 0 |
| 454 | `runtime_load` | `blk.46.ffn_gate_exps.weight` | 1 | 0 |
| 452 | `runtime_load` | `blk.16.ffn_gate_exps.weight` | 1 | 0 |
| 452 | `runtime_load` | `blk.43.ffn_gate_exps.weight` | 1 | 0 |
| 449 | `runtime_load` | `blk.36.ffn_gate_exps.weight` | 1 | 0 |
| 446 | `runtime_load` | `blk.45.ffn_gate_exps.weight` | 1 | 0 |
| 441 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 0 |
| 440 | `runtime_load` | `blk.27.ffn_down_exps.weight` | 1 | 0 |
| 439 | `runtime_load` | `blk.35.ffn_gate_exps.weight` | 1 | 0 |

## Interpretation

- The largest measured copy cost comes from `pack_hit=0` paths. These are
  tensors absent from the current expert pack and therefore materialized
  through the slower GGUF-backed path rather than the optimized pack path.
- This is different from a VRAM hotset miss. A tensor can miss VRAM cache
  but still be served efficiently if it exists in the expert pack. The
  current general-prompt profile shows many misses are also expert-pack
  misses, which makes fixed VRAM hotset tuning insufficient.
- A prompt-specific/dev-union pack would be useful only as a diagnostic
  bound. An accepted candidate must use prompt-independent coverage, such
  as a model-wide expert pack or a GGUF-offset direct-read pack alias.
- The next implementation should first remove or reduce the pack-miss path
  for general prompts, then rerun the strict n96 dev baseline before any
  held-out test evaluation.
