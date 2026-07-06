# Kimi route-detail cache-wave bound

This is a dev-only offline diagnostic. It is not SOTA and does not use held-out test prompts.

## Inputs

- run_dir: `.Agent/runs/20260707-gp30-route-detail-dev-n32`
- prompts: `7`
- io_depth: `8`
- current upgate slots: `1735`
- current down slots: `766`

## Runtime Metrics

| prompt | quality | tok/s | TTFT ms | decode ms | memory peak | route rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | pass | 1.33 | 66329.46 | 23354.81 | 15899996160 | 42928 |
| `dev_japan_factual` | pass | 0.43 | 73191.37 | 72255.59 | 15899996160 | 42928 |
| `dev_linear_equation` | fail | 0.15 | 83494.76 | 205431.05 | 15899996160 | 42928 |
| `dev_mixed_summary` | pass | 0.20 | 101048.20 | 152855.86 | 15899996160 | 42928 |
| `dev_photosynthesis_factual` | pass | 0.30 | 59685.97 | 103730.58 | 15899996160 | 42928 |
| `dev_python_reverse` | pass | 0.16 | 74750.65 | 190100.26 | 15899996160 | 42928 |
| `dev_zh_france` | pass | 0.35 | 66773.41 | 87496.45 | 15899996160 | 42928 |

## Strategy Summary

| pool | strategy | slots | waves | full-hit rate | missed GiB | aggregate wave reduction vs LFU | worst-prompt wave reduction | worst miss-byte change | pass gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| upgate | global_lfu | 1735 | 26052 | 0.0001 | 697.90 | 0.0000 | 0.0000 | 0.0000 | no |
| upgate | front_layers_1_2 | 1735 | 25186 | 0.0333 | 797.25 | 0.0332 | 0.0331 | 0.1945 | no |
| upgate | front_layers_1_4 | 1735 | 24769 | 0.0493 | 922.95 | 0.0492 | 0.0438 | 0.5968 | no |
| upgate | front_layers_1_8 | 1735 | 25424 | 0.0242 | 881.76 | 0.0241 | 0.0134 | 0.5096 | no |
| upgate | front_layers_1_12 | 1735 | 25661 | 0.0151 | 855.99 | 0.0150 | 0.0040 | 0.4492 | no |
| upgate | front_layers_1_16 | 1735 | 25815 | 0.0092 | 832.85 | 0.0091 | 0.0024 | 0.3935 | no |
| upgate | setcover_fullhit | 1735 | 24751 | 0.0500 | 935.06 | 0.0499 | 0.0494 | 0.6194 | no |
| upgate | split_62_lfu | 1734 | 26052 | 0.0001 | 698.00 | 0.0000 | 0.0000 | 0.0007 | no |
| upgate | split_68_lfu | 1902 | 26050 | 0.0002 | 682.17 | 0.0001 | 0.0000 | -0.0149 | no |
| upgate | split_72_lfu | 2014 | 26049 | 0.0002 | 672.07 | 0.0001 | 0.0000 | -0.0277 | no |
| upgate | split_76_lfu | 2126 | 26047 | 0.0003 | 662.23 | 0.0002 | 0.0000 | -0.0383 | no |
| upgate | split_80_lfu | 2238 | 26042 | 0.0005 | 652.86 | 0.0004 | 0.0000 | -0.0467 | no |
| down | global_lfu | 766 | 11507 | 0.0001 | 401.52 | 0.0000 | 0.0000 | 0.0000 | no |
| down | front_layers_1_2 | 766 | 11074 | 0.0377 | 470.36 | 0.0376 | 0.0371 | 0.2289 | no |
| down | front_layers_1_4 | 766 | 10993 | 0.0448 | 529.41 | 0.0447 | 0.0341 | 0.5924 | no |
| down | front_layers_1_8 | 766 | 11109 | 0.0347 | 523.05 | 0.0346 | 0.0225 | 0.5705 | no |
| down | front_layers_1_12 | 766 | 11245 | 0.0229 | 512.27 | 0.0228 | 0.0158 | 0.5299 | no |
| down | front_layers_1_16 | 766 | 11318 | 0.0165 | 496.12 | 0.0164 | 0.0073 | 0.4683 | no |
| down | setcover_fullhit | 766 | 10887 | 0.0540 | 532.48 | 0.0539 | 0.0487 | 0.6036 | no |
| down | split_62_lfu | 766 | 11507 | 0.0001 | 401.52 | 0.0000 | 0.0000 | 0.0000 | no |
| down | split_68_lfu | 645 | 11507 | 0.0001 | 417.62 | 0.0000 | 0.0000 | 0.0548 | no |
| down | split_72_lfu | 564 | 11508 | 0.0000 | 429.48 | -0.0001 | -0.0006 | 0.0878 | no |
| down | split_76_lfu | 483 | 11508 | 0.0000 | 442.09 | -0.0001 | -0.0006 | 0.1254 | no |
| down | split_80_lfu | 403 | 11508 | 0.0000 | 455.70 | -0.0001 | -0.0006 | 0.1705 | no |

## Decision

No offline strategy passes the GP30 gate. Reject front-layer/full-hit cache reallocation as the next primary runtime path under the current entry budget.

## Reproduce

```bash
.Agent/run-tools/kimi_route_detail_cache_wave_bound.py --run-dir .Agent/runs/20260707-gp30-route-detail-dev-n32 --out-json .Agent/runs/20260707-gp30-route-detail-dev-n32/cache-wave-bound.json --out-md .Agent/runs/20260707-gp30-route-detail-dev-n32/report.md
```
