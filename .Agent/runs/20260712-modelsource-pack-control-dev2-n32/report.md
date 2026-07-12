# Kimi Fresh Pack-Source Control

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Commit: `6721481ba93f2abf8ae50ce6616cd7f62417e49f`

## Goal

Verify that the new `MOE_EXPERT_SOURCE=pack|model` source gate did not change
the default SOTA `pack` path, then test whether simply increasing `io_uring`
depth/refill improves endpoint decode.

This report makes no SOTA claim.

## Fresh Control

External profile root:

`/root/lfz/runs/vendor-kimi-token-rate/20260712-modelsource-pack-control-dev2-n32-025300`

Runs:

- France:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-modelsource-pack-control-france-n32-024906`
- Intelligence:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-modelsource-pack-control-intelligence-n32-025149`

Config:

- cold start per prompt;
- `MemoryMax=15900000000`, `MemorySwapMax=0`;
- `N=32`, `VRAM_MIB=15000`, `UPGATE_PCT=72`;
- `MOE_EXPERT_SOURCE=pack`;
- `PROFILE=1`, `COPY_PROFILE=1`;
- default `MOE_IO_DEPTH=8`, `MOE_IO_REFILL_BATCH=4`.

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak |
|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.58` | `11336.22` | `19558.88 / 31` | `12768227328` |
| `dev_intelligence_general` | pass | `1.51` | `9832.15` | `20464.24 / 31` | `12602351616` |

Aggregate:

- weighted token rate: `1.549 tok/s`;
- aggregate `io_uring` throughput: `10.132 GiB/s`;
- direct reads: `0`;
- fallback rows: `0`;
- copy-profile `pack_hit=0` rows: `0`;
- up/gate wall: `440.722 ms/token`;
- down wall: `169.687 ms/token`.

## IO Depth A/B

Base:

`/root/lfz/runs/vendor-kimi-token-rate/20260712-modelsource-pack-control-france-n32-024906`

Candidate:

`/root/lfz/runs/vendor-kimi-token-rate/20260712-modelsource-pack-depth16-france-n32-025416`

Only changed env:

- `MOE_IO_DEPTH=16`
- `MOE_IO_REFILL_BATCH=8`

| config | quality | tok/s | TTFT ms | decode ms | iouring wait s | inflight avg/max |
|---|---:|---:|---:|---:|---:|---:|
| `depth8/refill4` | pass | `1.58` | `11336.22` | `19558.88` | `15.998` | `4.59/8` |
| `depth16/refill8` | pass | `1.50` | `10849.12` | `20718.04` | `15.424` | `5.05/12` |

Decision: reject `depth16/refill8`. It reduces raw `iouring_wait` by only
`0.574s`, but endpoint decode regresses by `1159.16 ms`.

## Next Decision

The next useful candidate should reduce bytes or improve future knowledge/cache
admission. Pure IO-depth tuning is not sufficient because the default path is
already near the measured IO ceiling and has full expert-pack coverage.
