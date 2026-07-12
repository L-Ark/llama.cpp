# Kimi lower-byte candidate gate

This report is generated from committed planning artifacts. It does not change runtime behavior and does not use held-out prompts for candidate selection.

- target total moved-byte ratio: `0.30x-0.40x`

## Candidate Results

| candidate | evidence | decision |
|---|---|---|
| naive blockwise 1-bit re-encode | ratio `0.347x-0.488x`, rel L2 `1.777-2.275` | reject |
| naive blockwise 2-bit re-encode | ratio `0.673x-0.878x`, rel L2 `0.687-0.791` | reject |
| D2MoE clustered base, top32 sample | 16-cluster rank128 error down `0.5002`, gate `0.3064`, base `448 MiB` per tensor | reject as primary |
| selected IQ1_S hotsets | all-dev candidate hybrid ratio `0.5205x`; small-budget worst ratio `0.6503x` | reject as direct 5 tok/s path |
| selected v2 hotsets | best transfer-only row needs packed `0.276x`, hybrid `0.276x`, ideal `5.018 tok/s` | reject as runtime path without more margin |
| down activation block skipping | threshold `0.1`: skip `0.198`, mean rel error `0.210`; threshold `0.2`: skip `0.345`, mean rel error `0.368` | reject as primary |

## Decision

Do not implement runtime support for the rejected candidates as the next primary optimization.

The next viable experiment must measure activation-output error for compressed up/gate/down compute on real hidden states and must reduce up/gate and down movement together.

## Evidence Inputs

- quant_report: `.Agent/runs/20260707-gp11-quant-reencode-bound/report.md`
- d2moe_down_report: `.Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-down-top32.md`
- d2moe_gate_report: `.Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-gate-top32.md`
- shadow_error_csv: `.Agent/runs/20260707-gp62-shadow-error-multiprompt-n32/summary.csv`
- iq1_report: `.Agent/runs/20260707-gp34-iq1s-budgeted-hotset-bound/report.md`
- v2_hotset_report: `.Agent/runs/20260707-gp46-v2-hotset-sweep/report.md`

## Reproduce

```bash
.Agent/run-tools/kimi_lower_byte_candidate_gate.py --out-json .Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/lower-byte-candidate-gate.json --out-md .Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/lower-byte-candidate-gate.md
```
