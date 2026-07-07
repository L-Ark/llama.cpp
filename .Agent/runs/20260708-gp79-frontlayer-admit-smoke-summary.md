# Kimi GP79 front-layer admission smoke

This is a dev-only runtime smoke. It does not claim SOTA.

## Inputs

- simulator report: `.Agent/runs/20260708-gp79-layer-fullhit-cache-sim/summary.md`
- admission profile: `.Agent/runs/20260708-gp79-layer-admit-profile/front-layers-0-2.tsv`
- profile entries: `1974`
- remote profile: `/root/lfz/runs/vendor-kimi-token-rate/gp79-front-layers-0-2-admit.tsv`
- prompt: `Please introduce France in a short paragraph.`
- output length: `N=32`
- cold start: yes
- host RAM cgroup: `MemoryMax=15900000000`, `MemorySwapMax=0`

## Runtime A/B

| run | quality | token rate | TTFT ms | decode ms / runs | RAM peak | upgate hit | down hit | iouring wait us |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | pass | `1.72` | `77778.06` | `18051.43 / 31` | `15899996160` | `45.2%` | `73.4%` | `11929606` |
| front-layer admit | pass | `1.69` | `82742.77` | `18354.19 / 31` | `15899996160` | `45.2%` | `73.4%` | `12001517` |

Both runs produced the same semantically valid France prefix:

```text
France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous
```

## Decision

Reject the existing `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE` hook for this Kimi split-cache path:

- the candidate is slower (`1.69` vs `1.72 tok/s`);
- TTFT increases by about `6.4%`;
- VRAM hit counters are identical, so the admission profile did not affect the active Kimi split upgate/down cache path;
- no held-out validation should be run for this candidate.

The next runtime attempt needs a Kimi-specific split-cache admission hook or a different optimization path. The current accepted SOTA remains GP4.

## Reproduce

```bash
cd /root/lfz/tmp/kimi-stage2m-align

RUN=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp79-frontlayer-admit-baseline-n32 \
REPO=/root/lfz/tmp/kimi-stage2m-align N=32 MIN_PROFILE=1 MEMORY_MAX=15900000000 MEMORY_SWAP_MAX=0 \
PINNED_SLOTS=12 VRAM_MIB=15000 THREADS=32 UPGATE_PCT=62 \
scripts/kimi-phase7og-priority-repro.sh

RUN=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp79-frontlayer-admit-profile-n32 \
REPO=/root/lfz/tmp/kimi-stage2m-align N=32 MIN_PROFILE=1 MEMORY_MAX=15900000000 MEMORY_SWAP_MAX=0 \
PINNED_SLOTS=12 VRAM_MIB=15000 THREADS=32 UPGATE_PCT=62 \
EXTRA_RUNTIME_ENV=$'GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/gp79-front-layers-0-2-admit.tsv' \
scripts/kimi-phase7og-priority-repro.sh
```
