# Kimi D2MoE dequant sample

- generated_at: `2026-07-06T06:33:59+0000`
- libggml_base: `build-cuda-batch/bin/libggml-base.so`
- samples: `5`

| tensor | expert | type | finite | abs_mean | abs_max | dequant_s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| blk.1.ffn_down_exps.weight | 116 | Q3_K | 1.000000 | 0.0150864 | 0.45459 | 0.019714 |
| blk.1.ffn_gate_exps.weight | 116 | IQ2_S | 1.000000 | 0.0154649 | 0.361034 | 0.058429 |
| blk.15.ffn_down_exps.weight | 336 | Q4_0 | 1.000000 | 0.0135468 | 0.156265 | 0.021698 |
| blk.26.ffn_down_exps.weight | 293 | IQ4_XS | 1.000000 | 0.0138902 | 0.10416 | 0.017912 |
| blk.29.ffn_gate_exps.weight | 227 | IQ3_XXS | 1.000000 | 0.0144246 | 0.106312 | 0.056349 |

## Reproduce

```bash
.Agent/run-tools/kimi_d2moe_dequant_sample.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260706-kimi-d2moe-phase0/dequant-sample-summary.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/dequant-sample-summary.md --max-per-type 1
```

This only verifies byte addressing and ggml dequantization. It is not a runtime token-rate run and it does not make a quality claim.
