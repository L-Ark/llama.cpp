# Kimi cold-start SOTA audit

- generated_at: `2026-07-07T13:37:21+0000`
- baseline: `historical-gp4-accepted` -> `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- candidate: `paired-vram15000-rerun` -> `/root/lfz/runs/vendor-kimi-token-rate/20260707-vram15000-paired-heldout-n96-baseline`
- prompts compared: `6`

## Summary

- baseline mean/min tok/s: `1.3667` / `1.1400`
- candidate mean/min tok/s: `1.7000` / `1.5000`
- mean token-rate delta: `24.39%`
- max TTFT delta: `56.18%`
- common quality/RAM/direct-read gate: `True`
- TTFT gate: `False`
- token-rate gate: `True`
- wait-only suspicion: `True`
- acceptance safe: `False`

Suspicious prompts:

- `test_chinese_01`
- `test_coding_01`
- `test_english_factual_01`
- `test_english_factual_02`
- `test_mixed_instruction_01`
- `test_reasoning_math_01`

## Prompt Comparison

| prompt | quality | base tok/s | cand tok/s | tok/s delta | TTFT delta | bytes delta | wait delta | up hit delta | down hit delta | same bytes/hits | direct reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | pass | 1.440 | 1.780 | 23.61% | 24.16% | 0.00% | -42.16% | 0.00 pp | 0.00 pp | True | 0 |
| `test_coding_01` | pass | 1.140 | 1.500 | 31.58% | 41.85% | 0.00% | -45.86% | 0.00 pp | 0.00 pp | True | 0 |
| `test_english_factual_01` | pass | 1.480 | 1.810 | 22.30% | 56.18% | 0.00% | -40.89% | 0.00 pp | 0.00 pp | True | 0 |
| `test_english_factual_02` | pass | 1.490 | 1.750 | 17.45% | 37.09% | 0.00% | -37.80% | 0.00 pp | 0.00 pp | True | 0 |
| `test_mixed_instruction_01` | pass | 1.330 | 1.700 | 27.82% | 46.41% | 0.00% | -43.91% | 0.00 pp | 0.00 pp | True | 0 |
| `test_reasoning_math_01` | pass | 1.320 | 1.660 | 25.76% | -3.15% | 0.00% | -42.92% | 0.00 pp | 0.00 pp | True | 0 |
## Interpretation Rule

- A candidate that improves token rate only because `iouring_wait` drops while moved bytes and hit rates stay unchanged is not accepted as a runtime/model optimization.
- Such a result must be reproduced under a stronger cold-device protocol before it can replace the accepted SOTA.
- This audit does not replace semantic quality review; it only checks metric consistency and cold-start risk.

## Reproduce

```bash
kimi_cold_start_sota_audit.py --baseline-root .Agent/runs/20260707-gp4-postcommit-test-n96-profile --baseline-label historical-gp4-accepted --candidate-root /root/lfz/runs/vendor-kimi-token-rate/20260707-vram15000-paired-heldout-n96-baseline --candidate-label paired-vram15000-rerun --out-json .Agent/runs/20260707-cold-start-sota-audit/paired-vs-historical.json --out-md .Agent/runs/20260707-cold-start-sota-audit/paired-vs-historical.md
```
