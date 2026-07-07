# Kimi cold-start SOTA audit

- generated_at: `2026-07-07T13:37:04+0000`
- baseline: `paired-vram15000` -> `/root/lfz/runs/vendor-kimi-token-rate/20260707-vram15000-paired-heldout-n96-baseline`
- candidate: `vram15500-candidate` -> `/root/lfz/runs/vendor-kimi-token-rate/20260707-vram15500-heldout-n96-candidate`
- prompts compared: `6`

## Summary

- baseline mean/min tok/s: `1.7000` / `1.5000`
- candidate mean/min tok/s: `1.7083` / `1.4900`
- mean token-rate delta: `0.49%`
- max TTFT delta: `6.80%`
- common quality/RAM/direct-read gate: `True`
- TTFT gate: `True`
- token-rate gate: `False`
- wait-only suspicion: `False`
- acceptance safe: `False`

## Prompt Comparison

| prompt | quality | base tok/s | cand tok/s | tok/s delta | TTFT delta | bytes delta | wait delta | up hit delta | down hit delta | same bytes/hits | direct reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | pass | 1.780 | 1.810 | 1.69% | 6.80% | -0.24% | -1.21% | 0.20 pp | 0.00 pp | False | 0 |
| `test_coding_01` | pass | 1.500 | 1.490 | -0.67% | -8.63% | -0.97% | 1.38% | 0.80 pp | 0.30 pp | False | 0 |
| `test_english_factual_01` | pass | 1.810 | 1.820 | 0.55% | -2.09% | -0.26% | 1.62% | 0.10 pp | 0.10 pp | True | 0 |
| `test_english_factual_02` | pass | 1.750 | 1.800 | 2.86% | 1.20% | -1.47% | -1.53% | 0.80 pp | 0.50 pp | False | 0 |
| `test_mixed_instruction_01` | pass | 1.700 | 1.710 | 0.59% | -1.75% | -0.20% | 0.56% | 0.10 pp | 0.10 pp | True | 0 |
| `test_reasoning_math_01` | pass | 1.660 | 1.620 | -2.41% | 6.47% | -0.71% | -0.41% | 0.40 pp | 0.20 pp | False | 0 |

## Reference Comparison

- reference: `historical-gp4-accepted` -> `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- candidate mean/min tok/s vs reference: `1.7083` / `1.4900`
- mean token-rate delta vs reference: `25.00%`
- max TTFT delta vs reference: `52.93%`
- wait-only suspicion vs reference: `True`
- acceptance safe vs reference: `False`

## Interpretation Rule

- A candidate that improves token rate only because `iouring_wait` drops while moved bytes and hit rates stay unchanged is not accepted as a runtime/model optimization.
- Such a result must be reproduced under a stronger cold-device protocol before it can replace the accepted SOTA.
- This audit does not replace semantic quality review; it only checks metric consistency and cold-start risk.

## Reproduce

```bash
kimi_cold_start_sota_audit.py --baseline-root /root/lfz/runs/vendor-kimi-token-rate/20260707-vram15000-paired-heldout-n96-baseline --baseline-label paired-vram15000 --candidate-root /root/lfz/runs/vendor-kimi-token-rate/20260707-vram15500-heldout-n96-candidate --candidate-label vram15500-candidate --reference-root .Agent/runs/20260707-gp4-postcommit-test-n96-profile --reference-label historical-gp4-accepted --out-json .Agent/runs/20260707-cold-start-sota-audit/report.json --out-md .Agent/runs/20260707-cold-start-sota-audit/report.md
```
