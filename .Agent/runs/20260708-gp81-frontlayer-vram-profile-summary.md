# Kimi GP81 front-layer split-cache profile preload smoke

This is a dev-only runtime parameter smoke. It does not claim SOTA.

## Goal

Use the real Kimi split-cache profile path (`GGML_MOE_VRAM_PROFILE`) for the GP79 front-layer candidate. GP79 used `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE`, which did not affect this runtime path.

## Profile

- local profile: `.Agent/runs/20260708-gp81-layer-profile-preload/front-layers-1-2-profile.csv`
- remote profile: `/root/lfz/runs/vendor-kimi-token-rate/gp81-front-layers-1-2-profile.csv`
- entries: `1974`
- layers: `1..2`
- source: dev route traces only
- held-out prompts used: `0`

## A/B

| run | quality | token rate | TTFT ms | decode ms / runs | RAM peak | upgate hit | down hit | preloads | iouring wait us |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GP79 baseline | pass | `1.72` | `77778.06` | `18051.43 / 31` | `15899996160` | `45.2%` | `73.4%` | `3673` | `11929606` |
| GP81 profile | pass | `1.20` | `84064.88` | `25905.23 / 31` | `15899996160` | `45.4%` | `73.4%` | `4989` | `12060373` |

The model answer remained semantically valid:

```text
France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous
```

## Evidence That The Hook Worked

`stderr.txt` contains:

```text
[moe_stream_batch] profile preload: loaded 1974 entries from /root/lfz/runs/vendor-kimi-token-rate/gp81-front-layers-1-2-profile.csv
[moe_stream_batch] profile preload: blk.1.ffn_up_exps.weight loaded=329
[moe_stream_batch] profile preload: blk.1.ffn_gate_exps.weight loaded=329
[moe_stream_batch] profile preload: blk.1.ffn_down_exps.weight loaded=329
[moe_stream_batch] profile preload: blk.2.ffn_up_exps.weight loaded=329
[moe_stream_batch] profile preload: blk.2.ffn_gate_exps.weight loaded=329
[moe_stream_batch] profile preload: blk.2.ffn_down_exps.weight loaded=329
```

## Decision

Reject. The real split-cache profile path changes behavior, but the gain is too small and the overhead is large:

- upgate hit rate improves only `+0.2pp`;
- down hit rate is unchanged;
- decode time regresses by `+43.5%`;
- token rate drops from `1.72` to `1.20 tok/s`;
- extra preloads/direct reads do not reduce exposed iouring wait.

Do not continue front-layer full-hit preload as a token-rate path. Future cache work needs either a low-overhead admission/protection hook that does not preload thousands of entries, or a materially stronger offline wave-saving candidate.

## Reproduce

```bash
cd /root/lfz/tmp/kimi-stage2m-align
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp81-frontlayer-vram-profile-n32 \
REPO=/root/lfz/tmp/kimi-stage2m-align N=32 MIN_PROFILE=1 MEMORY_MAX=15900000000 MEMORY_SWAP_MAX=0 \
PINNED_SLOTS=12 VRAM_MIB=15000 THREADS=32 UPGATE_PCT=62 \
EXTRA_RUNTIME_ENV=$'GGML_MOE_VRAM_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/gp81-front-layers-1-2-profile.csv\nGGML_MOE_VRAM_PROFILE_PROTECT=0' \
scripts/kimi-phase7og-priority-repro.sh
```
