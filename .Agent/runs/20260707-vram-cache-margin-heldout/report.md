# Kimi VRAM cache margin held-out check

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`
Commit: `9a3ef27e9`

## Purpose

Test whether increasing `GGML_MOE_VRAM_CACHE_MIB` from `15000` to `15500`
improves prompt-general held-out token rate while preserving all constraints.

This is a config-only experiment. No runtime code changed.

## Runs

Dev n32 A/B:

- run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-vram15500-n32-ab`
- baseline: `VRAM_MIB=15000`
- candidate: `VRAM_MIB=15500`

Held-out n96 candidate:

```bash
cd /root/lfz/tmp/kimi-stage2m-align
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/tmp/kimi-stage2m-align \
  --prompt-file .Agent/evals/kimi-general-test-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260707-vram15500-heldout-n96-candidate \
  --mode test \
  --n 96 \
  --profile \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 900 \
  --vram-mib 15500
```

Paired held-out n96 baseline:

```bash
cd /root/lfz/tmp/kimi-stage2m-align
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/tmp/kimi-stage2m-align \
  --prompt-file .Agent/evals/kimi-general-test-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260707-vram15000-paired-heldout-n96-baseline \
  --mode test \
  --n 96 \
  --profile \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 900 \
  --vram-mib 15000
```

## Dev N32 Result

| mode | quality | tok/s | decode ms/runs | TTFT ms | RAM peak | direct reads | iouring bytes | upgate hit | down hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `15000` | pass | 1.78 | 17437.84 / 31 | 93581.89 | 15899996160 | 0 | 131119579136 | 45.2% | 73.4% |
| `15500` | pass | 1.82 | 17053.06 / 31 | 93185.55 | 15899996160 | 0 | 130773565440 | 45.3% | 73.6% |

Dev result was positive but small, so n96 held-out validation was required.

## Held-Out N96 Paired Result

| prompt | quality | historical GP4 tok/s | paired 15000 tok/s | candidate 15500 tok/s | candidate vs paired | candidate TTFT vs paired | candidate TTFT vs historical | RAM peak | direct reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | pass | 1.44 | 1.78 | 1.81 | +1.69% | +6.80% | +32.60% | 15899996160 | 0 |
| `test_coding_01` | pass | 1.14 | 1.50 | 1.49 | -0.67% | -8.63% | +29.60% | 15899996160 | 0 |
| `test_english_factual_01` | pass | 1.48 | 1.81 | 1.82 | +0.55% | -2.09% | +52.93% | 15899996160 | 0 |
| `test_english_factual_02` | pass | 1.49 | 1.75 | 1.80 | +2.86% | +1.20% | +38.74% | 15899996160 | 0 |
| `test_mixed_instruction_01` | pass | 1.33 | 1.70 | 1.71 | +0.59% | -1.75% | +43.84% | 15899996160 | 0 |
| `test_reasoning_math_01` | pass | 1.32 | 1.66 | 1.62 | -2.41% | +6.47% | +3.12% | 15899996160 | 0 |

Aggregate:

- historical GP4 mean: `1.3667 tok/s`
- paired `15000` mean: `1.7000 tok/s`
- candidate `15500` mean: `1.7083 tok/s`
- candidate vs paired mean: `+0.49%`
- candidate vs paired min token rate: `1.49` vs `1.50`

## Decision

Reject `VRAM_MIB=15500` as a new default:

- quality, RAM, and direct-read gates passed;
- paired TTFT gate passed;
- but token-rate gain versus paired `15000` was only `+0.49%`;
- `test_coding_01` and `test_reasoning_math_01` regressed versus paired
  `15000`;
- TTFT versus the historical GP4 accepted baseline exceeded +20% for most
  prompts, so it cannot be accepted against the historical SOTA gate either.

Important follow-up:

- The paired `VRAM_MIB=15000` run itself reproduced `1.70 tok/s` mean, much
  higher than the historical GP4 `1.37 tok/s`.
- This is not a `15500` improvement. Before claiming a new SOTA, re-run or
  audit the current `15000` baseline for reproducibility and identify whether
  the difference is code, binary, environment, or measurement variance.

## Historical vs Paired Baseline Audit

The paired `15000` run used the same moved bytes and the same hit rates as the
historical GP4 run, but the iouring wait time was much lower:

| prompt | historical tok/s | paired tok/s | historical iouring wait ms | paired iouring wait ms | historical GiB | paired GiB | up hit | down hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | 1.44 | 1.78 | 66449.2 | 38436.9 | 378.0 | 378.0 | 44.4% | 73.5% |
| `test_coding_01` | 1.14 | 1.50 | 86362.9 | 46755.5 | 505.7 | 505.7 | 26.3% | 66.7% |
| `test_english_factual_01` | 1.48 | 1.81 | 63217.2 | 37368.3 | 372.8 | 372.8 | 45.3% | 73.4% |
| `test_english_factual_02` | 1.49 | 1.75 | 62733.6 | 39021.3 | 394.6 | 394.6 | 42.0% | 71.1% |
| `test_mixed_instruction_01` | 1.33 | 1.70 | 71271.6 | 39975.7 | 405.1 | 405.1 | 40.7% | 71.6% |
| `test_reasoning_math_01` | 1.32 | 1.66 | 42333.6 | 24164.3 | 255.1 | 255.1 | 35.8% | 69.1% |

Interpretation:

- The faster paired baseline is not caused by more VRAM hits or fewer bytes.
- It is an IO wait/runtime-state difference, likely storage-device or system
  state after repeated reads.
- Under the strict cold-start requirement, this cannot be treated as a new
  accepted SOTA without a stronger cold-start reproduction protocol.
- Keep the historical GP4 SOTA as the accepted baseline for now.
