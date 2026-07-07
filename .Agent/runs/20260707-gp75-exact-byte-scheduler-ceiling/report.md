# Kimi exact-byte scheduler ceiling

This is an offline ceiling analysis. It does not change runtime behavior or claim SOTA.

- profile root: `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- movement bandwidth ceiling: `10.40 GiB/s`
- all-hit MoE floor: `40.1 ms/token`

## Summary

- prompts: `6`
- measured mean token rate: `1.367 tok/s`
- transfer-only mean ceiling: `2.396 tok/s`
- floor+transfer mean ceiling: `2.184 tok/s`
- best prompt floor+transfer ceiling: `2.395 tok/s`
- worst prompt floor+transfer ceiling: `1.812 tok/s`
- mean byte ratio needed for 5 tok/s after floor: `0.383x`

## Per Prompt

| prompt | measured tok/s | moved GiB/token | transfer-only tok/s | floor+transfer tok/s | required ratio for 5 tok/s | call wall ms/token | runtime-load copy ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | `1.440` | `3.979` | `2.614` | `2.366` | `0.418` | `471.5` | `1939.2` |
| `test_coding_01` | `1.140` | `5.324` | `1.954` | `1.812` | `0.312` | `566.3` | `2741.8` |
| `test_english_factual_01` | `1.480` | `3.925` | `2.650` | `2.395` | `0.424` | `466.4` | `1936.9` |
| `test_english_factual_02` | `1.490` | `4.197` | `2.478` | `2.254` | `0.396` | `463.5` | `1919.6` |
| `test_mixed_instruction_01` | `1.330` | `4.265` | `2.439` | `2.221` | `0.390` | `508.6` | `2144.6` |
| `test_reasoning_math_01` | `1.320` | `4.639` | `2.242` | `2.057` | `0.359` | `657.1` | `2963.8` |

## Decision

Exact-byte scheduling without byte reduction is not a primary 5 tok/s path: even at 10.4 GiB/s and a 40.1 ms/token all-hit floor, the best held-out prompt ceiling is 2.40 tok/s and the mean ceiling is 2.18 tok/s. Continue with structural byte reduction.

## Reproduce

```bash
.Agent/run-tools/kimi_exact_byte_scheduler_ceiling.py --profile-root .Agent/runs/20260707-gp4-postcommit-test-n96-profile --out-json .Agent/runs/20260707-gp75-exact-byte-scheduler-ceiling/report.json --out-md .Agent/runs/20260707-gp75-exact-byte-scheduler-ceiling/report.md
```
