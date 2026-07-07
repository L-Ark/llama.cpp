# Kimi prediction-recall decode ceiling

This is an offline bound. It does not change runtime behavior or claim SOTA.

- profile root: `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- policy source: `.Agent/runs/20260707-gp71-next-gate-prefetch-policy-bound/report.json`
- target token rate: `5.00 tok/s`
- prompts: `6`
- current aggregate token rate: `1.356 tok/s`
- exposed up/gate wall fraction: `0.249`
- required up/gate hide recall for target: `2.922`

## Aggregate Policies

| policy | recall | saved s | token rate | speedup |
|---|---:|---:|---:|---:|
| `cap1_all` | `0.1243` | `12.09` | `1.399` | `1.032x` |
| `cap2_all` | `0.2457` | `23.92` | `1.444` | `1.065x` |
| `cap4_all` | `0.4714` | `45.88` | `1.536` | `1.133x` |
| `cap8_all` | `0.7725` | `75.19` | `1.679` | `1.239x` |
| `perfect_upgate_hide` | `1.0000` | `97.33` | `1.806` | `1.332x` |

## Per Prompt

| prompt | current tok/s | exposed upgate % decode | req recall for 5 | perfect-hide tok/s | cap8 tok/s |
|---|---:|---:|---:|---:|---:|
| `test_chinese_01` | `1.439` | `0.260` | `2.744` | `1.944` | `1.800` |
| `test_coding_01` | `1.135` | `0.238` | `3.243` | `1.490` | `1.391` |
| `test_english_factual_01` | `1.477` | `0.247` | `2.851` | `1.962` | `1.825` |
| `test_english_factual_02` | `1.492` | `0.246` | `2.847` | `1.980` | `1.843` |
| `test_mixed_instruction_01` | `1.331` | `0.258` | `2.842` | `1.794` | `1.662` |
| `test_reasoning_math_01` | `1.317` | `0.249` | `2.962` | `1.753` | `1.630` |

## Decision

Even perfect up/gate hide cannot reach the target token rate. Prediction/prefetch is secondary and must be combined with byte reduction or a representation change.

## Reproduce

```bash
.Agent/run-tools/kimi_prediction_recall_decode_ceiling.py --profile-root .Agent/runs/20260707-gp4-postcommit-test-n96-profile --policy-json .Agent/runs/20260707-gp71-next-gate-prefetch-policy-bound/report.json --out-json .Agent/runs/20260708-gp83-prediction-recall-decode-ceiling/report.json --out-md .Agent/runs/20260708-gp83-prediction-recall-decode-ceiling/report.md --target-tps 5.0
```
