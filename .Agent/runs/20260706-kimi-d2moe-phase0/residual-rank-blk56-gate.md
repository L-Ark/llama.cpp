# Kimi D2MoE residual rank sample

- generated_at: `2026-07-06T06:48:07+0000`
- tensor: `blk.56.ffn_gate_exps.weight`
- experts: `4`
- libggml_base: `build-cuda-batch/bin/libggml-base.so`

| expert | weight | dequant_s | r16 | r32 | r64 | r128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 0.1200 | 0.192644 | 0.9905 | 0.9830 | 0.9691 | 0.9442 |
| 202 | 0.6800 | 0.216828 | 0.9912 | 0.9840 | 0.9702 | 0.9455 |
| 337 | 0.1000 | 0.213721 | 0.9913 | 0.9841 | 0.9704 | 0.9458 |
| 7 | 0.1000 | 0.219270 | 0.9910 | 0.9838 | 0.9701 | 0.9455 |

## Byte Estimate

| rank | bf16_delta_bytes | delta_vs_full_quant |
| ---: | ---: | ---: |
| 16 | 294912 | 0.0525 |
| 32 | 589824 | 0.1050 |
| 64 | 1179648 | 0.2099 |
| 128 | 2359296 | 0.4198 |

## Reproduce

```bash
.Agent/run-tools/kimi_d2moe_residual_rank.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-blk56-gate.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-blk56-gate.md --tensor blk.56.ffn_gate_exps.weight --max-experts 4 --ranks 16,32,64,128 --oversample 8 --niter 1 --torch-threads 8
```

This is an offline residual-compressibility bound. It is not a runtime token-rate or quality claim.
