# GP65 correct GP4 env N32 shadow confirmation

## Run roots

- Baseline: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n32-baseline`
- TopK=8 shadow: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n32-shadow-top8`
- Analysis: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n32-analysis`
- Branch/head: `vendor/kimi-speculative-general-token-rate-16gb` / `74bcf7eacc6fc9bde889d58763dc86dedd89c6f0`
- Prompt file SHA256: `ef204b612ff4e3089dcdcffc98cfabfa3c8c2b1fd9e21d9047abd156493425ea`

## Aggregate

- Prompts with shadow CSV: `7/7`
- Expert/byte recall: `77.35%`
- Expert precision: `77.35%`
- False/actual bytes: `0.2265`
- Max record overhead: `17089 us`
- Mean token-rate ratio shadow/baseline: `0.995`
- Max TTFT ratio shadow/baseline: `1.062`
- Max decode ratio shadow/baseline: `1.038`
- Max memory peak: `15899996160`
- Direct reads all zero: `True`

## Per prompt

| prompt | quality | tok/s base -> shadow | TTFT ratio | decode ratio | direct_reads | rows |
|---|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass ok | 1.69 -> 1.71 | 0.979 | 0.990 | 0 | 3690 |
| `dev_japan_factual` | pass ok | 1.65 -> 1.67 | 0.968 | 0.990 | 0 | 3690 |
| `dev_linear_equation` | fail missing_keyword:7|seven | 1.42 -> 1.42 | 0.997 | 0.994 | 0 | 3690 |
| `dev_mixed_summary` | pass ok | 1.54 -> 1.52 | 1.062 | 1.013 | 0 | 3690 |
| `dev_photosynthesis_factual` | pass ok | 1.62 -> 1.56 | 1.016 | 1.038 | 0 | 3690 |
| `dev_python_reverse` | pass ok | 1.55 -> 1.54 | 1.032 | 1.006 | 0 | 3690 |
| `dev_zh_france` | pass ok | 1.74 -> 1.74 | 1.000 | 1.000 | 0 | 3690 |

## Decision

- N32 shadow gate passes for predictor signal, direct-read integrity, memory, and TTFT overhead.
- N32 is not a final quality gate because `dev_linear_equation` is truncated before keyword `7`; run N96 confirmation before promoting runtime prefetch.
