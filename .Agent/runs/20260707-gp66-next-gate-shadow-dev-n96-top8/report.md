# GP66 next-gate shadow dev N96 TopK=8

Status: rejected / stopped early.

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`
Commit on remote run repo: `300516836`

## Purpose

Confirm whether the GP64/GP65 next-gate shadow signal remains usable at `N=96`
before planning any real bounded prefetch path.

This was a dev-only shadow run. It did not issue extra prefetch reads and did
not change accepted SOTA runtime behavior.

## Runtime

- Remote root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp66-next-gate-shadow-dev-n96-top8`
- Prompt file:
  `.Agent/evals/kimi-general-dev-prompts.jsonl`
- `N=96`
- `GGML_MOE_NEXT_GATE_SHADOW_TOPK=8`
- `GGML_MOE_NEXT_GATE_SHADOW_OUT=$RUN/next-gate-shadow.csv`
- Cold-start per prompt through `systemd-run`
- Cgroup:
  - `MemoryMax=15900000000`
  - `MemorySwapMax=0`

## Result

The run was stopped early because the second prompt exceeded the TTFT gate by a
large margin.

Completed prompt:

| prompt | quality | tok/s | TTFT ms | decode ms | decode runs | RAM peak |
|---|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.90` | `88618.6` | `40632.75` | `77` | `15899996160` |

Stopped / incomplete prompts:

| prompt | observed state | decision |
|---|---|---|
| `dev_japan_factual` | no metrics; stdout only 5 bytes; killed after systemd runtime `7min 36.857s` | TTFT/early-latency failure |
| `dev_photosynthesis_factual` | no metrics; transiently started during cleanup; killed | ignored |
| `dev_linear_equation` | no metrics; transiently started during cleanup; killed | ignored |

## Decision

Do not implement real TopK=8 next-gate prefetch from this result.

GP65 showed useful route recall, but GP66 shows the N96 shadow path is not yet
stable enough under the strict TTFT rule. The next predictor/prefetch step must
first reduce the shadow/predictor overhead or prove the long TTFT was unrelated
with a paired baseline. Until then, this path stays diagnostic-only.

## Reproduce

Use the committed GP4 general-prompt runner and add:

```bash
EXTRA_RUNTIME_ENV='GGML_MOE_NEXT_GATE_SHADOW_OUT=$RUN/next-gate-shadow.csv
GGML_MOE_NEXT_GATE_SHADOW_TOPK=8'
N=96
```

Run each dev prompt through:

```bash
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO=/root/lfz/tmp/kimi-stage2m-align \
      RUN=<prompt-run-dir> \
      N=96 \
      PROMPT_ID=<dev-id> \
      PROMPT_USER_TEXT='<dev-prompt>' \
      QUALITY_KEYWORDS='<keywords>' \
      EXTRA_RUNTIME_ENV="$EXTRA_RUNTIME_ENV" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```
