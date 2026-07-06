# Kimi D2MoE residual rank sample

- generated_at: `2026-07-06T06:43:04+0000`
- tensor: `blk.1.ffn_down_exps.weight`
- experts: `4`
- libggml_base: `build-cuda-batch/bin/libggml-base.so`

| expert | weight | dequant_s | r16 | r32 | r64 | r128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 116 | 0.4912 | 0.152096 | 0.9923 | 0.9850 | 0.9712 | 0.9464 |
| 23 | 0.3158 | 0.127901 | 0.9924 | 0.9851 | 0.9714 | 0.9467 |
| 160 | 0.1053 | 0.130416 | 0.9922 | 0.9850 | 0.9712 | 0.9465 |
| 137 | 0.0877 | 0.155777 | 0.9925 | 0.9853 | 0.9716 | 0.9470 |

## Byte Estimate

| rank | bf16_delta_bytes | delta_vs_full_quant |
| ---: | ---: | ---: |
| 16 | 294912 | 0.0468 |
| 32 | 589824 | 0.0935 |
| 64 | 1179648 | 0.1870 |
| 128 | 2359296 | 0.3740 |

## Reproduce

```bash
.Agent/run-tools/kimi_d2moe_residual_rank.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-sample.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-sample.md --preferred-kind down --max-experts 4 --ranks 16,32,64,128 --oversample 8 --niter 1 --torch-threads 8
```

This is an offline residual-compressibility bound. It is not a runtime token-rate or quality claim.
