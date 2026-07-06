# Kimi copy-profile breakdown

This diagnostic report uses one dev prompt run with copy profiling enabled.
It is not a held-out test result and is not an accepted SOTA metric.

Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which
synchronizes H2D copies for measurement. Token rate from this run is not
directly comparable to normal SOTA runs, but the split between expert-pack
hits and misses identifies where host-side materialization time is spent.

## Run

- run dir: `.Agent/runs/20260706-kimi-copy-profile-python-reverse-n32`
- prompt id: `dev_python_reverse`
- token rate: `0.16 tok/s`
- TTFT: `77103.44 ms`
- decode: `198848.52 ms / 31`
- memory peak: `15899996160`

## High-Level Split

- total profiled copy wall: `256862 ms`
- expert-pack miss wall: `207424 ms` over `64.85 GiB`
- expert-pack hit wall: `49438 ms` over `87.64 GiB`
- iouring wall: `11811 ms`
- non-iouring wall: `245051 ms`

## By Copy Path

| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 0 | 0 | 4354 | 21.63 | 69941 | 0 | 925 | 171 | 18 | 71033 |
| `runtime_load` | up | 0 | 0 | 4354 | 19.83 | 67191 | 0 | 853 | 174 | 18 | 68220 |
| `current_down_overlap` | down | 0 | 0 | 2282 | 13.41 | 39220 | 0 | 565 | 93 | 11 | 39880 |
| `runtime_load` | down | 0 | 0 | 1503 | 9.98 | 27821 | 0 | 415 | 58 | 6 | 28291 |
| `runtime_load` | gate | 1 | 0 | 5109 | 25.32 | 11612 | 0 | 1091 | 141 | 20 | 12895 |
| `runtime_load` | up | 1 | 0 | 5109 | 23.36 | 11252 | 0 | 1008 | 165 | 19 | 12478 |
| `current_down_overlap` | down | 1 | 0 | 2542 | 14.93 | 6042 | 0 | 613 | 71 | 12 | 6732 |
| `runtime_load` | down | 1 | 0 | 1886 | 12.50 | 4961 | 0 | 509 | 49 | 8 | 5522 |
| `runtime_load` | gate | 1 | 1 | 650 | 3.14 | 0 | 3853 | 130 | 14 | 0 | 3853 |
| `runtime_load` | up | 1 | 1 | 650 | 3.00 | 0 | 3718 | 124 | 27 | 0 | 3718 |
| `runtime_load` | down | 1 | 1 | 681 | 4.23 | 0 | 3470 | 171 | 14 | 0 | 3470 |
| `current_down_overlap` | down | 1 | 1 | 197 | 1.16 | 0 | 769 | 47 | 4 | 0 | 769 |

## Top Pack-Miss Layer/Role Buckets

| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `runtime_load` | 58 | down | 0 | 93 | 0.68 | 1956 | 28 | 1988 |
| `runtime_load` | 43 | gate | 0 | 104 | 0.54 | 1918 | 23 | 1945 |
| `runtime_load` | 46 | gate | 0 | 106 | 0.55 | 1911 | 24 | 1939 |
| `runtime_load` | 57 | down | 0 | 84 | 0.61 | 1872 | 25 | 1900 |
| `runtime_load` | 52 | gate | 0 | 97 | 0.51 | 1863 | 22 | 1888 |
| `current_down_overlap` | 43 | down | 0 | 104 | 0.61 | 1767 | 26 | 1797 |
| `current_down_overlap` | 46 | down | 0 | 106 | 0.62 | 1753 | 26 | 1783 |
| `runtime_load` | 46 | up | 0 | 106 | 0.46 | 1705 | 20 | 1729 |
| `runtime_load` | 43 | up | 0 | 104 | 0.46 | 1693 | 20 | 1717 |
| `current_down_overlap` | 52 | down | 0 | 97 | 0.57 | 1662 | 24 | 1690 |
| `runtime_load` | 40 | gate | 0 | 84 | 0.44 | 1648 | 19 | 1671 |
| `runtime_load` | 60 | down | 0 | 89 | 0.52 | 1638 | 22 | 1664 |
| `runtime_load` | 52 | up | 0 | 97 | 0.42 | 1641 | 18 | 1663 |
| `runtime_load` | 42 | gate | 0 | 85 | 0.44 | 1631 | 19 | 1654 |
| `runtime_load` | 38 | gate | 0 | 78 | 0.41 | 1631 | 17 | 1651 |
| `runtime_load` | 32 | gate | 0 | 74 | 0.39 | 1629 | 17 | 1649 |
| `current_down_overlap` | 44 | down | 0 | 76 | 0.45 | 1617 | 19 | 1639 |
| `runtime_load` | 56 | down | 0 | 81 | 0.48 | 1603 | 20 | 1627 |
| `runtime_load` | 55 | gate | 0 | 84 | 0.44 | 1589 | 19 | 1611 |
| `current_down_overlap` | 42 | down | 0 | 85 | 0.50 | 1562 | 21 | 1586 |
| `runtime_load` | 49 | gate | 0 | 86 | 0.45 | 1559 | 19 | 1582 |
| `current_down_overlap` | 40 | down | 0 | 84 | 0.49 | 1510 | 21 | 1535 |
| `runtime_load` | 44 | gate | 0 | 76 | 0.40 | 1498 | 17 | 1518 |
| `runtime_load` | 49 | up | 0 | 86 | 0.38 | 1493 | 17 | 1513 |
| `runtime_load` | 59 | down | 0 | 84 | 0.49 | 1484 | 21 | 1508 |
| `current_down_overlap` | 55 | down | 0 | 84 | 0.49 | 1478 | 21 | 1502 |
| `runtime_load` | 40 | up | 0 | 84 | 0.37 | 1480 | 16 | 1500 |
| `runtime_load` | 29 | gate | 0 | 83 | 0.43 | 1478 | 18 | 1499 |
| `runtime_load` | 53 | gate | 0 | 77 | 0.40 | 1477 | 17 | 1498 |
| `runtime_load` | 2 | gate | 0 | 67 | 0.35 | 1476 | 14 | 1493 |

## Top Tensor Copy Wall

| wall ms | op | tensor | pack hit | iouring |
|---:|---|---|---:|---:|
| 1988 | `runtime_load` | `blk.58.ffn_down_exps.weight` | 0 | 0 |
| 1945 | `runtime_load` | `blk.43.ffn_gate_exps.weight` | 0 | 0 |
| 1939 | `runtime_load` | `blk.46.ffn_gate_exps.weight` | 0 | 0 |
| 1900 | `runtime_load` | `blk.57.ffn_down_exps.weight` | 0 | 0 |
| 1888 | `runtime_load` | `blk.52.ffn_gate_exps.weight` | 0 | 0 |
| 1797 | `current_down_overlap` | `blk.43.ffn_down_exps.weight` | 0 | 0 |
| 1783 | `current_down_overlap` | `blk.46.ffn_down_exps.weight` | 0 | 0 |
| 1729 | `runtime_load` | `blk.46.ffn_up_exps.weight` | 0 | 0 |
| 1717 | `runtime_load` | `blk.43.ffn_up_exps.weight` | 0 | 0 |
| 1690 | `current_down_overlap` | `blk.52.ffn_down_exps.weight` | 0 | 0 |
| 1671 | `runtime_load` | `blk.40.ffn_gate_exps.weight` | 0 | 0 |
| 1664 | `runtime_load` | `blk.60.ffn_down_exps.weight` | 0 | 0 |
| 1663 | `runtime_load` | `blk.52.ffn_up_exps.weight` | 0 | 0 |
| 1654 | `runtime_load` | `blk.42.ffn_gate_exps.weight` | 0 | 0 |
| 1651 | `runtime_load` | `blk.38.ffn_gate_exps.weight` | 0 | 0 |
| 1649 | `runtime_load` | `blk.32.ffn_gate_exps.weight` | 0 | 0 |
| 1639 | `current_down_overlap` | `blk.44.ffn_down_exps.weight` | 0 | 0 |
| 1627 | `runtime_load` | `blk.56.ffn_down_exps.weight` | 0 | 0 |
| 1611 | `runtime_load` | `blk.55.ffn_gate_exps.weight` | 0 | 0 |
| 1586 | `current_down_overlap` | `blk.42.ffn_down_exps.weight` | 0 | 0 |
| 1582 | `runtime_load` | `blk.49.ffn_gate_exps.weight` | 0 | 0 |
| 1535 | `current_down_overlap` | `blk.40.ffn_down_exps.weight` | 0 | 0 |
| 1518 | `runtime_load` | `blk.44.ffn_gate_exps.weight` | 0 | 0 |
| 1513 | `runtime_load` | `blk.49.ffn_up_exps.weight` | 0 | 0 |
| 1508 | `runtime_load` | `blk.59.ffn_down_exps.weight` | 0 | 0 |
| 1502 | `current_down_overlap` | `blk.55.ffn_down_exps.weight` | 0 | 0 |
| 1500 | `runtime_load` | `blk.40.ffn_up_exps.weight` | 0 | 0 |
| 1499 | `runtime_load` | `blk.29.ffn_gate_exps.weight` | 0 | 0 |
| 1498 | `runtime_load` | `blk.53.ffn_gate_exps.weight` | 0 | 0 |
| 1493 | `runtime_load` | `blk.2.ffn_gate_exps.weight` | 0 | 0 |

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
