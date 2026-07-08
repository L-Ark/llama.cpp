# Kimi cold-start SOTA audit

- generated_at: `2026-07-08T08:12:59+0800`
- baseline: `gp112_q4_off_paired` -> `.Agent/runs/20260708-gp112-q4-ttft-source/heldout-q4-off-root`
- candidate: `gp112_q4_on_paired` -> `.Agent/runs/20260708-gp112-q4-ttft-source/heldout-q4-on-root`
- prompts compared: `6`

## Summary

- baseline mean/min tok/s: `1.7133` / `1.5200`
- candidate mean/min tok/s: `1.8200` / `1.6000`
- mean token-rate delta: `6.23%`
- max TTFT delta: `13.05%`
- common quality/RAM/direct-read gate: `True`
- TTFT gate: `True`
- token-rate gate: `True`
- wait-only suspicion: `False`
- acceptance safe: `True`

## Prompt Comparison

| prompt | quality | base tok/s | cand tok/s | tok/s delta | TTFT delta | bytes delta | wait delta | up hit delta | down hit delta | same bytes/hits | direct reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | pass | 1.810 | 1.880 | 3.87% | 0.77% | 11.77% | 7.58% | 0.00 pp | -7.70 pp | False | 0 |
| `test_coding_01` | pass | 1.520 | 1.600 | 5.26% | -0.30% | 7.60% | 6.50% | 0.10 pp | -6.10 pp | False | 0 |
| `test_english_factual_01` | pass | 1.840 | 1.960 | 6.52% | 13.05% | 10.50% | 6.49% | 0.90 pp | -7.30 pp | False | 0 |
| `test_english_factual_02` | pass | 1.790 | 1.850 | 3.35% | 5.74% | -16.27% | -17.29% | -0.40 pp | -6.70 pp | False | 0 |
| `test_mixed_instruction_01` | pass | 1.720 | 1.860 | 8.14% | -7.35% | 8.10% | 5.61% | 1.40 pp | -6.50 pp | False | 0 |
| `test_reasoning_math_01` | pass | 1.600 | 1.770 | 10.62% | -10.62% | 21.85% | 17.11% | -0.10 pp | -6.60 pp | False | 0 |
## Interpretation Rule

- A candidate that improves token rate only because `iouring_wait` drops while moved bytes and hit rates stay unchanged is not accepted as a runtime/model optimization.
- Such a result must be reproduced under a stronger cold-device protocol before it can replace the accepted SOTA.
- This audit does not replace semantic quality review; it only checks metric consistency and cold-start risk.

## Reproduce

```bash
kimi_cold_start_sota_audit.py --baseline-root .Agent/runs/20260708-gp112-q4-ttft-source/heldout-q4-off-root --candidate-root .Agent/runs/20260708-gp112-q4-ttft-source/heldout-q4-on-root --baseline-label gp112_q4_off_paired --candidate-label gp112_q4_on_paired --out-json .Agent/runs/20260708-gp112-q4-ttft-source/audit/report.json --out-md .Agent/runs/20260708-gp112-q4-ttft-source/audit/report.md
```
