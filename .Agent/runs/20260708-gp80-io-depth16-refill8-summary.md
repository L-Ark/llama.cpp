# Kimi GP80 IO depth16/refill8 smoke

This is a dev-only runtime parameter smoke. It does not claim SOTA.

## Goal

Test whether increasing iouring depth/refill can reduce exposed wait without source changes.

## A/B

| run | IO depth | refill | quality | token rate | TTFT ms | decode ms / runs | RAM peak | iouring wait us | inflight avg/max |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| GP79 baseline | `8` | `4` | pass | `1.72` | `77778.06` | `18051.43 / 31` | `15899996160` | `11929606` | `3.12 / 8` |
| GP80 candidate | `16` | `8` | pass | `1.69` | `83751.31` | `18315.23 / 31` | `15899996160` | `12066182` | `3.10 / 8` |

VRAM hit rates and moved bytes were unchanged:

- baseline upgate/down hit: `45.2% / 73.4%`
- candidate upgate/down hit: `45.2% / 73.4%`
- both runs: `iouring_bytes=129268056064`

## Decision

Reject. Increasing `MOE_IO_DEPTH` and `MOE_IO_REFILL_BATCH` does not expose deeper queues in this runtime path:

- token rate regressed from `1.72` to `1.69 tok/s`;
- TTFT increased by `7.7%`, still within the 20% limit but worse;
- iouring wait increased rather than decreased;
- `inflight_max` stayed at `8`, and batch hist stayed identical.

This supports the model that current decode is gated by per-layer route availability and active expert count, not by the configured io_uring queue depth.

## Reproduce

```bash
cd /root/lfz/tmp/kimi-stage2m-align
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp80-io-depth16-refill8-n32 \
REPO=/root/lfz/tmp/kimi-stage2m-align N=32 MIN_PROFILE=1 MEMORY_MAX=15900000000 MEMORY_SWAP_MAX=0 \
PINNED_SLOTS=12 VRAM_MIB=15000 THREADS=32 UPGATE_PCT=62 MOE_IO_DEPTH=16 MOE_IO_REFILL_BATCH=8 \
scripts/kimi-phase7og-priority-repro.sh
```
