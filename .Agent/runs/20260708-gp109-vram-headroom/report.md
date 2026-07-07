# GP109 VRAM headroom cache expansion probe

Date: 2026-07-08

This is a dev-only n32 probe. It does not claim SOTA and does not change the
default runtime configuration yet.

## Purpose

Check whether the current accepted GP4-style Kimi SOTA leaves enough RTX 5090
VRAM headroom to slightly increase `GGML_MOE_VRAM_CACHE_MIB` without violating
the strict 16 GB host RAM limit, TTFT limit, or output quality gate.

## Setup

- Remote repo: `/root/lfz/tmp/kimi-stage2m-align`
- Branch: `vendor/kimi-speculative-general-token-rate-16gb`
- Prompt: `Please introduce France in a short paragraph.`
- `N=32`
- Cold start: yes, runner executes `sync` and `drop_caches`.
- Cgroup:
  - `MemoryMax=15900000000`
  - `MemorySwapMax=0`
- Quality keywords:
  - `France|French`
  - `Europe|European`
  - `Paris`
- Compared configs:
  - baseline: `VRAM_MIB=15000`
  - candidate: `VRAM_MIB=15350`
- Additional telemetry:
  - `nvidia-smi` sampled every 0.5 s into each run directory.

## Results

| config | quality | tok/s | decode ms/runs | TTFT ms | RAM peak | max VRAM used MiB | min VRAM free MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `VRAM_MIB=15000` | pass | `1.79` | `17325.68 / 31` | `81917.01` | `15899996160` | `31293` | `817` |
| `VRAM_MIB=15350` | pass | `1.83` | `16937.10 / 31` | `84380.62` | `15899996160` | `31641` | `469` |

Cache allocation changed as expected:

| config | actual cache MiB | upgate slots | down slots |
|---|---:|---:|---:|
| `15000` | `15000` | `1735` | `766` |
| `15350` | `15350` | `1775` | `784` |

Hit-rate movement was small:

| config | upgate hit | down hit | iouring bytes |
|---|---:|---:|---:|
| `15000` | `45.2%` | `73.4%` | `131119579136` |
| `15350` | `45.3%` | `73.6%` | `130782969856` |

The candidate saved about `388.58 ms` over 31 decode steps, about
`12.54 ms/token`, and raised TTFT by about `3.0%`. This is within the TTFT gate
but may be noise at n32 scale.

## Decision

`VRAM_MIB=15350` is a valid candidate for further testing, but not accepted
yet.

Acceptance gate before changing defaults:

1. Run at least three dev n32 prompts with paired or comparable cold-start
   evidence.
2. If the candidate is neutral or better and quality passes, run held-out n96.
3. Accept only if held-out mean token rate improves, all held-out qualities
   pass, TTFT increase remains under 20%, host RAM peak stays below 16 GB, and
   `nvidia-smi` shows no OOM/near-zero free instability.

## Reproduce

```bash
ssh -p 51056 root@92.180.27.82 'cd /root/lfz/tmp/kimi-stage2m-align && for cfg in 15000 15350; do
  RUN=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp109-vram-headroom/france-n32-vram${cfg}
  mkdir -p "$RUN"
  systemd-run --wait --collect --same-dir \
    -p MemoryMax=15900000000 \
    -p MemorySwapMax=0 \
    -p RuntimeMaxSec=900 \
    bash -lc '"'"'RUN="$0"; VRAM_MIB="$1"; mkdir -p "$RUN";
      (while true; do
        date -Is | tr -d "\n" >> "$RUN/nvidia-smi.csv";
        printf "," >> "$RUN/nvidia-smi.csv";
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits >> "$RUN/nvidia-smi.csv" 2>/dev/null || true;
        sleep 0.5;
      done) & mon=$!;
      env REPO=/root/lfz/tmp/kimi-stage2m-align RUN="$RUN" N=32 VRAM_MIB="$VRAM_MIB" \
        PROMPT_ID=gp109_france_vram_${VRAM_MIB} \
        PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
        QUALITY_KEYWORDS="France|French,Europe|European,Paris" \
        PROFILE=0 \
        .Agent/run-tools/kimi-general-prompt-repro.sh;
      rc=$?;
      kill "$mon" 2>/dev/null || true;
      wait "$mon" 2>/dev/null || true;
      exit "$rc"'"'"' "$RUN" "$cfg"
done'
```

