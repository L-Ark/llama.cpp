# Kimi D2MoE residual rank sample

- generated_at: `2026-07-06T06:48:01+0000`
- tensor: `blk.1.ffn_up_exps.weight`
- experts: `4`
- libggml_base: `build-cuda-batch/bin/libggml-base.so`

| expert | weight | dequant_s | r16 | r32 | r64 | r128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 116 | 0.4912 | 0.213087 | 0.9922 | 0.9847 | 0.9706 | 0.9455 |
| 23 | 0.3158 | 0.210101 | 0.9922 | 0.9848 | 0.9708 | 0.9457 |
| 160 | 0.1053 | 0.204900 | 0.9920 | 0.9845 | 0.9705 | 0.9454 |
| 137 | 0.0877 | 0.202867 | 0.9923 | 0.9850 | 0.9712 | 0.9462 |

## Byte Estimate

| rank | bf16_delta_bytes | delta_vs_full_quant |
| ---: | ---: | ---: |
| 16 | 294912 | 0.0627 |
| 32 | 589824 | 0.1254 |
| 64 | 1179648 | 0.2509 |
| 128 | 2359296 | 0.5017 |

## Reproduce

```bash
.Agent/run-tools/kimi_d2moe_residual_rank.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-blk1-up.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-blk1-up.md --tensor blk.1.ffn_up_exps.weight --max-experts 4 --ranks 16,32,64,128 --oversample 8 --niter 1 --torch-threads 8
```

This is an offline residual-compressibility bound. It is not a runtime token-rate or quality claim.
