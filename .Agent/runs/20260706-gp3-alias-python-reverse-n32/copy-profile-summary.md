# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `/root/lfz/runs/vendor-kimi-token-rate/20260706-145000Z-gp3-alias-python-reverse-n32`
- prompt id: `dev_python_reverse`
- token rate: `1.08 tok/s`
- TTFT: `93022.15 ms`
- decode: `28647.98 ms / 31`
- memory peak: `15899996160`

## High-Level Split

- total profiled copy wall: `135905 ms`
- expert-pack miss wall: `0 ms` over `0.00 GiB`
- expert-pack hit wall: `135905 ms` over `155.00 GiB`
- iouring wall: `135905 ms`
- non-iouring wall: `0 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 1 | 1 | 10285 | 50.89 | 0 | 51838 | 2091 | 168 | 0 | 51838 |
| `runtime_load` | up | 1 | 1 | 10285 | 46.95 | 0 | 46801 | 1938 | 234 | 0 | 46801 |
| `current_down_overlap` | down | 1 | 1 | 5091 | 29.91 | 0 | 19565 | 1197 | 73 | 0 | 19565 |
| `runtime_load` | down | 1 | 1 | 4148 | 27.26 | 0 | 17701 | 1084 | 58 | 0 | 17701 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 1307 | `runtime_load` | `blk.30.ffn_gate_exps.weight` | 1 | 1 |
| 1180 | `runtime_load` | `blk.1.ffn_up_exps.weight` | 1 | 1 |
| 1152 | `runtime_load` | `blk.51.ffn_gate_exps.weight` | 1 | 1 |
| 1100 | `runtime_load` | `blk.45.ffn_gate_exps.weight` | 1 | 1 |
| 1100 | `runtime_load` | `blk.34.ffn_gate_exps.weight` | 1 | 1 |
| 1091 | `runtime_load` | `blk.30.ffn_up_exps.weight` | 1 | 1 |
| 1077 | `runtime_load` | `blk.55.ffn_gate_exps.weight` | 1 | 1 |
| 1053 | `runtime_load` | `blk.28.ffn_gate_exps.weight` | 1 | 1 |
| 1042 | `runtime_load` | `blk.40.ffn_gate_exps.weight` | 1 | 1 |
| 1041 | `runtime_load` | `blk.43.ffn_gate_exps.weight` | 1 | 1 |
| 1036 | `runtime_load` | `blk.1.ffn_gate_exps.weight` | 1 | 1 |
| 1032 | `runtime_load` | `blk.35.ffn_gate_exps.weight` | 1 | 1 |
| 1023 | `runtime_load` | `blk.53.ffn_gate_exps.weight` | 1 | 1 |
| 1020 | `runtime_load` | `blk.8.ffn_up_exps.weight` | 1 | 1 |
| 1016 | `runtime_load` | `blk.29.ffn_gate_exps.weight` | 1 | 1 |
| 1001 | `runtime_load` | `blk.42.ffn_gate_exps.weight` | 1 | 1 |
| 990 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 1 | 1 |
| 979 | `runtime_load` | `blk.46.ffn_gate_exps.weight` | 1 | 1 |
| 978 | `runtime_load` | `blk.36.ffn_gate_exps.weight` | 1 | 1 |
| 974 | `runtime_load` | `blk.60.ffn_down_exps.weight` | 1 | 1 |
| 971 | `runtime_load` | `blk.51.ffn_up_exps.weight` | 1 | 1 |
| 971 | `runtime_load` | `blk.44.ffn_gate_exps.weight` | 1 | 1 |
| 969 | `runtime_load` | `blk.32.ffn_gate_exps.weight` | 1 | 1 |
| 968 | `runtime_load` | `blk.48.ffn_gate_exps.weight` | 1 | 1 |
| 963 | `runtime_load` | `blk.50.ffn_gate_exps.weight` | 1 | 1 |
| 962 | `runtime_load` | `blk.25.ffn_gate_exps.weight` | 1 | 1 |
| 955 | `runtime_load` | `blk.52.ffn_gate_exps.weight` | 1 | 1 |
| 953 | `runtime_load` | `blk.54.ffn_gate_exps.weight` | 1 | 1 |
| 946 | `runtime_load` | `blk.17.ffn_gate_exps.weight` | 1 | 1 |
| 943 | `runtime_load` | `blk.8.ffn_gate_exps.weight` | 1 | 1 |

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
