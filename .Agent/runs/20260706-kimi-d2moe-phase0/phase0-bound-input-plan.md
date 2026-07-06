# Kimi D2MoE Phase 0 Bound Input Plan

- repo_head: `1072f177d5d5cbba6e329779385d78e638ec3eaf`
- inventory: `.Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv`
- selected_layers: `1, 13, 15, 26, 27, 29, 31, 56, 57, 60`
- selected_tensors: `30`
- bf16_base_resident_gib: `0.820`

## Decision

This artifact does not claim D2MoE is viable yet. It selects the first
Kimi tensors and experts for a real residual-SVD pass and estimates the
payload ratios for candidate delta ranks.

## Top Profile Layers

| layer | count | fallback_ms | weighted_gib |
| --- | ---: | ---: | ---: |
| 13 | 1152 | 1688.616 | 5.619 |
| 27 | 1152 | 1576.036 | 5.619 |
| 26 | 1152 | 1558.253 | 6.152 |
| 15 | 1152 | 1473.718 | 6.316 |
| 57 | 1152 | 1450.086 | 6.809 |
| 56 | 1152 | 1449.268 | 6.275 |
| 52 | 1152 | 1420.204 | 5.947 |
| 58 | 1152 | 1419.533 | 6.809 |
| 9 | 1152 | 1417.257 | 6.645 |
| 44 | 1152 | 1412.545 | 5.947 |
| 28 | 1152 | 1375.971 | 5.947 |
| 50 | 1152 | 1366.074 | 5.947 |
| 14 | 1152 | 1361.697 | 5.947 |
| 35 | 1152 | 1359.274 | 5.947 |
| 47 | 1152 | 1341.416 | 5.947 |
| 49 | 1152 | 1312.090 | 5.947 |
| 24 | 1152 | 1304.227 | 6.152 |
| 10 | 1152 | 1301.835 | 6.316 |
| 37 | 1152 | 1284.405 | 5.947 |
| 8 | 1152 | 1278.271 | 6.645 |

## Selected Tensor Rank Estimates

| tensor | type | expert MiB | base MiB | rank64 ratio | rank128 ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `blk.1.ffn_down_exps.weight` | Q3_K | 6.02 | 28.00 | 0.187 | 0.374 |
| `blk.1.ffn_gate_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.1.ffn_up_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.13.ffn_down_exps.weight` | Q3_K | 6.02 | 28.00 | 0.187 | 0.374 |
| `blk.13.ffn_gate_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.13.ffn_up_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.15.ffn_down_exps.weight` | Q4_0 | 7.88 | 28.00 | 0.143 | 0.286 |
| `blk.15.ffn_gate_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.15.ffn_up_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.26.ffn_down_exps.weight` | IQ4_XS | 7.44 | 28.00 | 0.151 | 0.303 |
| `blk.26.ffn_gate_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.26.ffn_up_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.27.ffn_down_exps.weight` | Q3_K | 6.02 | 28.00 | 0.187 | 0.374 |
| `blk.27.ffn_gate_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.27.ffn_up_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.29.ffn_down_exps.weight` | Q3_K | 6.02 | 28.00 | 0.187 | 0.374 |
| `blk.29.ffn_gate_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |
| `blk.29.ffn_up_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.31.ffn_down_exps.weight` | Q3_K | 6.02 | 28.00 | 0.187 | 0.374 |
| `blk.31.ffn_gate_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |
| `blk.31.ffn_up_exps.weight` | IQ2_S | 4.48 | 28.00 | 0.251 | 0.502 |
| `blk.56.ffn_down_exps.weight` | Q3_K | 6.02 | 28.00 | 0.187 | 0.374 |
| `blk.56.ffn_gate_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |
| `blk.56.ffn_up_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |
| `blk.57.ffn_down_exps.weight` | IQ4_XS | 7.44 | 28.00 | 0.151 | 0.303 |
| `blk.57.ffn_gate_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |
| `blk.57.ffn_up_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |
| `blk.60.ffn_down_exps.weight` | Q3_K | 6.02 | 28.00 | 0.187 | 0.374 |
| `blk.60.ffn_gate_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |
| `blk.60.ffn_up_exps.weight` | IQ3_XXS | 5.36 | 28.00 | 0.210 | 0.420 |

## Next Step

Implement the real residual reader/dequantizer for the selected tensor set,
then compute weighted base tensors and residual SVD curves. Keep this path
offline and default-off until a real Kimi n96 cold-start run improves over
the same-host baseline under the existing quality and memory gates.
