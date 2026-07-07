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

## Multi-Prompt N32 Follow-Up

After the France-only candidate looked mildly positive, two more prompts were
run with the same paired cold-start setup:

- `Briefly explain why the Moon has phases.`
- `What is AI infrastructure? Answer in one short paragraph.`

| prompt | config | quality | tok/s | decode ms/runs | TTFT ms | RAM peak | max VRAM used MiB | min VRAM free MiB | upgate hit | down hit |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| France | `15000` | pass | `1.79` | `17325.68 / 31` | `81917.01` | `15899996160` | `31293` | `817` | `45.2%` | `73.4%` |
| France | `15350` | pass | `1.83` | `16937.10 / 31` | `84380.62` | `15899996160` | `31641` | `469` | `45.3%` | `73.6%` |
| Moon phases | `15000` | pass | `1.77` | `17501.27 / 31` | `82570.64` | `15899996160` | `31293` | `817` | `40.5%` | `71.1%` |
| Moon phases | `15350` | pass | `1.74` | `17779.16 / 31` | `94543.96` | `15899996160` | `31641` | `469` | `41.4%` | `71.1%` |
| AI infra | `15000` | auto fail | `1.77` | `17507.24 / 31` | `98662.00` | `15899996160` | `31293` | `817` | `44.0%` | `72.3%` |
| AI infra | `15350` | auto fail | `1.79` | `17288.09 / 31` | `88187.85` | `15899996160` | `31641` | `469` | `45.1%` | `73.1%` |

Notes:

- The AI infra answers are semantically reasonable, but the automated keyword
  gate was too strict for this prompt, so these runs are not usable for
  acceptance.
- Mean n32 token rate across the three paired prompts:
  - `15000`: about `1.777 tok/s`;
  - `15350`: about `1.787 tok/s`.
- The mean lift is only about `0.6%`, with one clear regression:
  - France: `+2.2%`;
  - Moon phases: `-1.7%`;
  - AI infra: `+1.1%`, but auto quality failed.
- `15350` leaves only about `469 MiB` minimum sampled free VRAM on all runs.

## Decision

Reject `VRAM_MIB=15350` as a default/SOTA change.

Reason:

- The multi-prompt n32 result is not consistently positive.
- The average gain is too small to justify a held-out n96 sweep.
- The candidate leaves little VRAM margin.
- No default env or runtime code should be changed.

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

Additional paired prompts used the same command shape with:

```text
PROMPT_ID=gp109_moon_vram_<cfg>
PROMPT_USER_TEXT="Briefly explain why the Moon has phases."
QUALITY_KEYWORDS="Moon|moon,Earth|earth,Sun|sun"

PROMPT_ID=gp109_aiinfra_vram_<cfg>
PROMPT_USER_TEXT="What is AI infrastructure? Answer in one short paragraph."
QUALITY_KEYWORDS="AI|artificial,intfrastructure|infrastructure,computing|systems"
```
