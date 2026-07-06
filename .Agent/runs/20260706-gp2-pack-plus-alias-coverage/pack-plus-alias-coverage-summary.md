# Kimi expert-pack coverage audit

This report compares route-trace events against expert-pack index keys.
It does not run inference and does not inspect held-out test prompts.

## Pack Sources

| path | entries | new unique keys | duplicates | new unique payload GiB |
|---|---:|---:|---:|---:|
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | 30831 | 30831 | 0 | 163.10 |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | 768 | 768 | 0 | 4.51 |
| `/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv` | 69120 | 37521 | 31599 | 197.88 |

## Prompt Coverage

| prompt | events | event hit | GiB | byte hit | miss GiB |
|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | 106592 | 100.0% | 553.28 | 100.0% | 0.00 |
| `dev_japan_factual` | 117664 | 100.0% | 610.75 | 100.0% | 0.00 |
| `dev_linear_equation` | 47080 | 100.0% | 244.38 | 100.0% | 0.00 |
| `dev_mixed_summary` | 74760 | 100.0% | 388.05 | 100.0% | 0.00 |
| `dev_photosynthesis_factual` | 130120 | 100.0% | 675.40 | 100.0% | 0.00 |
| `dev_python_reverse` | 131504 | 100.0% | 682.58 | 100.0% | 0.00 |
| `dev_zh_france` | 67840 | 100.0% | 352.13 | 100.0% | 0.00 |

## Aggregate

- events: `675560`
- event hit rate: `100.0%`
- routed bytes: `3506.57 GiB`
- byte hit rate: `100.0%`
- miss bytes: `0.00 GiB`

## By Role

| role | events | event hit | GiB | byte hit | miss GiB |
|---|---:|---:|---:|---:|---:|
| down | 206968 | 100.0% | 1280.91 | 100.0% | 0.00 |
| gate | 234296 | 100.0% | 1156.20 | 100.0% | 0.00 |
| up | 234296 | 100.0% | 1069.46 | 100.0% | 0.00 |

## Top Missing Layer/Role Buckets

| layer | role | miss events | miss GiB | hit events | byte hit |
|---:|---|---:|---:|---:|---:|
| 60 | up | 0 | 0.00 | 3960 | 100.0% |
| 60 | gate | 0 | 0.00 | 3960 | 100.0% |
| 60 | down | 0 | 0.00 | 3960 | 100.0% |
| 1 | up | 0 | 0.00 | 3904 | 100.0% |
| 1 | gate | 0 | 0.00 | 3904 | 100.0% |
| 1 | down | 0 | 0.00 | 3904 | 100.0% |
| 2 | up | 0 | 0.00 | 3904 | 100.0% |
| 2 | gate | 0 | 0.00 | 3904 | 100.0% |
| 2 | down | 0 | 0.00 | 3904 | 100.0% |
| 3 | up | 0 | 0.00 | 3904 | 100.0% |
| 3 | gate | 0 | 0.00 | 3904 | 100.0% |
| 3 | down | 0 | 0.00 | 3904 | 100.0% |
| 4 | up | 0 | 0.00 | 3904 | 100.0% |
| 4 | gate | 0 | 0.00 | 3904 | 100.0% |
| 4 | down | 0 | 0.00 | 3904 | 100.0% |
| 5 | up | 0 | 0.00 | 3904 | 100.0% |
| 5 | gate | 0 | 0.00 | 3904 | 100.0% |
| 5 | down | 0 | 0.00 | 3904 | 100.0% |
| 6 | up | 0 | 0.00 | 3904 | 100.0% |
| 6 | gate | 0 | 0.00 | 3904 | 100.0% |
| 7 | up | 0 | 0.00 | 3904 | 100.0% |
| 7 | gate | 0 | 0.00 | 3904 | 100.0% |
| 8 | up | 0 | 0.00 | 3904 | 100.0% |
| 8 | gate | 0 | 0.00 | 3904 | 100.0% |
| 9 | up | 0 | 0.00 | 3904 | 100.0% |
| 9 | gate | 0 | 0.00 | 3904 | 100.0% |
| 10 | up | 0 | 0.00 | 3904 | 100.0% |
| 10 | gate | 0 | 0.00 | 3904 | 100.0% |
| 11 | up | 0 | 0.00 | 3904 | 100.0% |
| 11 | gate | 0 | 0.00 | 3904 | 100.0% |
| 11 | down | 0 | 0.00 | 3904 | 100.0% |
| 12 | up | 0 | 0.00 | 3904 | 100.0% |
| 12 | gate | 0 | 0.00 | 3904 | 100.0% |
| 12 | down | 0 | 0.00 | 3904 | 100.0% |
| 13 | up | 0 | 0.00 | 3904 | 100.0% |
| 13 | gate | 0 | 0.00 | 3904 | 100.0% |
| 13 | down | 0 | 0.00 | 3904 | 100.0% |
| 14 | up | 0 | 0.00 | 3904 | 100.0% |
| 14 | gate | 0 | 0.00 | 3904 | 100.0% |
| 14 | down | 0 | 0.00 | 3904 | 100.0% |

## Top Missing Keys

| count | GiB | tensor | expert | bytes |
|---:|---:|---|---:|---:|

## Interpretation

- `pack_hit` coverage is independent from VRAM-cache coverage. This audit
  only measures whether a routed expert can use the optimized expert-pack
  source path after it misses VRAM.
- High miss bytes here mean the runtime must materialize selected experts
  from GGUF-backed tensor storage for those route events.
- A model-wide same-quant pack would remove these misses but currently needs
  separate disk feasibility. A GGUF-offset alias pack would target the same
  miss bytes without duplicating the payload.
