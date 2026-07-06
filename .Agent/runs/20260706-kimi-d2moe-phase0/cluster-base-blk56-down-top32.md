# Kimi D2MoE clustered-base bound

- generated_at: `2026-07-06T07:53:33+0000`
- tensor: `blk.56.ffn_down_exps.weight`
- experts: `32`
- type: `Q3_K`
- shape: `(2048, 7168, 384)`

| clusters | raw residual/weight | rank64 error/weight | rank128 error/weight | rank128 residual ratio | base bf16 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9998 | 0.9277 | 0.9040 | 0.9464 | 28.00 |
| 4 | 0.9756 | 0.8109 | 0.7902 | 0.9466 | 112.00 |
| 8 | 0.9105 | 0.5873 | 0.5723 | 0.9467 | 224.00 |
| 16 | 0.6407 | 0.5133 | 0.5002 | 0.9467 | 448.00 |

## Reproduce

```bash
.Agent/run-tools/kimi_d2moe_cluster_base_bound.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --route-profile /root/lfz/runs/ik_llama/kimi-iq3s-16gb-accepted-default-fulltrace-n64-001-20260627-011039Z/route_profile.csv --libggml-base build-cuda-batch/bin/libggml-base.so --tensor blk.56.ffn_down_exps.weight --out-json .Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-down-top32.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-down-top32.md --max-experts 32 --clusters 1,4,8,16 --ranks 64,128 --max-svd-experts 8 --torch-threads 8 --sketch-dim 8192 --kmeans-iters 20
```

This is an offline bound only. It is not a runtime token-rate or quality result.
