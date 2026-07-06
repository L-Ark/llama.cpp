# Kimi D2MoE residual rank sample

- generated_at: `2026-07-06T06:48:04+0000`
- tensor: `blk.56.ffn_down_exps.weight`
- experts: `4`
- libggml_base: `build-cuda-batch/bin/libggml-base.so`

| expert | weight | dequant_s | r16 | r32 | r64 | r128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 0.1200 | 0.149117 | 0.9925 | 0.9853 | 0.9716 | 0.9470 |
| 202 | 0.6800 | 0.154935 | 0.9924 | 0.9852 | 0.9716 | 0.9469 |
| 337 | 0.1000 | 0.153399 | 0.9925 | 0.9852 | 0.9715 | 0.9468 |
| 7 | 0.1000 | 0.150433 | 0.9925 | 0.9853 | 0.9717 | 0.9471 |

## Byte Estimate

| rank | bf16_delta_bytes | delta_vs_full_quant |
| ---: | ---: | ---: |
| 16 | 294912 | 0.0468 |
| 32 | 589824 | 0.0935 |
| 64 | 1179648 | 0.1870 |
| 128 | 2359296 | 0.3740 |

## Reproduce

```bash
.Agent/run-tools/kimi_d2moe_residual_rank.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-blk56-down.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-blk56-down.md --tensor blk.56.ffn_down_exps.weight --max-experts 4 --ranks 16,32,64,128 --oversample 8 --niter 1 --torch-threads 8
```

This is an offline residual-compressibility bound. It is not a runtime token-rate or quality claim.
