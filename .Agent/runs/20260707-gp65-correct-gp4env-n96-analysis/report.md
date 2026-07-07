# GP65 correct GP4 env N96 shadow confirmation

## Run roots

- Baseline: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n96-baseline`
- TopK=8 shadow: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n96-shadow-top8`
- Analysis: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n96-analysis`
- Branch/head: `vendor/kimi-speculative-general-token-rate-16gb` / `74bcf7eacc6fc9bde889d58763dc86dedd89c6f0`
- Prompt file SHA256: `ef204b612ff4e3089dcdcffc98cfabfa3c8c2b1fd9e21d9047abd156493425ea`
- Baseline anomaly note: `dev_photosynthesis_factual` first baseline attempt hung and was preserved under `dev_photosynthesis_factual.hung-*`; the clean rerun at the normal prompt path succeeded and is used for comparison.

## Aggregate

- Prompts with shadow CSV: `7/7`
- Dev quality pass baseline/shadow: `True`
- Expert/byte recall: `77.25%`
- Expert precision: `77.25%`
- False/actual bytes: `0.2275`
- Max record overhead: `2718 us`
- Mean token-rate ratio shadow/baseline: `0.993`
- Max TTFT ratio shadow/baseline: `1.085`
- Max decode ratio shadow/baseline: `1.027`
- Max memory peak: `15899996160`
- Direct reads all zero: `True`
- GP65 shadow gate passed: `True`

## Per prompt

| prompt | quality | tok/s base -> shadow | TTFT ratio | decode ratio | direct_reads | rows |
|---|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass/pass | 1.82 -> 1.78 | 1.002 | 1.021 | 0 | 9164 |
| `dev_japan_factual` | pass/pass | 1.79 -> 1.80 | 0.992 | 0.992 | 0 | 10116 |
| `dev_linear_equation` | pass/pass | 1.44 -> 1.44 | 0.956 | 1.003 | 0 | 4047 |
| `dev_mixed_summary` | pass/pass | 1.63 -> 1.59 | 1.085 | 1.027 | 0 | 6427 |
| `dev_photosynthesis_factual` | pass/pass | 1.75 -> 1.75 | 0.946 | 1.000 | 0 | 11187 |
| `dev_python_reverse` | pass/pass | 1.62 -> 1.62 | 1.074 | 1.000 | 0 | 11306 |
| `dev_zh_france` | pass/pass | 1.77 -> 1.76 | 1.007 | 1.004 | 0 | 5832 |

## Decision

- N96 shadow gate passes for predictor signal, direct-read integrity, quality, memory, and TTFT overhead.
- This still does not prove token-rate improvement because shadow mode does not prefetch. It proves that a bounded runtime prefetch experiment is justified.
- Next implementation should be default-off `CAP_PER_LAYER=1`, using only expert pack + io_uring, with hard backpressure and extra-byte caps.
