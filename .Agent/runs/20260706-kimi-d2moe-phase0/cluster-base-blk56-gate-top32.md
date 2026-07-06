# Kimi D2MoE clustered-base bound

- generated_at: `2026-07-06T07:56:57+0000`
- tensor: `blk.56.ffn_gate_exps.weight`
- experts: `32`
- type: `IQ3_XXS`
- shape: `(7168, 2048, 384)`

| clusters | raw residual/weight | rank64 error/weight | rank128 error/weight | rank128 residual ratio | base bf16 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0001 | 0.9264 | 0.9026 | 0.9444 | 28.00 |
| 4 | 0.9726 | 0.8038 | 0.7832 | 0.9449 | 112.00 |
| 8 | 0.9017 | 0.6188 | 0.6029 | 0.9449 | 224.00 |
| 16 | 0.6650 | 0.3145 | 0.3064 | 0.9448 | 448.00 |

## Reproduce

```bash
.Agent/run-tools/kimi_d2moe_cluster_base_bound.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --route-profile /root/lfz/runs/ik_llama/kimi-iq3s-16gb-accepted-default-fulltrace-n64-001-20260627-011039Z/route_profile.csv --libggml-base build-cuda-batch/bin/libggml-base.so --tensor blk.56.ffn_gate_exps.weight --out-json .Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-gate-top32.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-gate-top32.md --max-experts 32 --clusters 1,4,8,16 --ranks 64,128 --max-svd-experts 8 --torch-threads 8 --sketch-dim 8192 --kmeans-iters 20
```

This is an offline bound only. It is not a runtime token-rate or quality result.
