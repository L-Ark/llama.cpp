# Kimi activation-weighted cluster-base residual oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T19:55:33+0000`
- prompt roots: `3`
- max records per prompt: `64`
- tensors evaluated: `1`
- bits: `[1, 2]`
- block: `256`

## Summary

| tensor | clusters | candidate | moved ratio | base BF16 MiB | row mean rel L2 | group mean rel L2 | group max rel L2 |
|---|---:|---|---:|---:|---:|---:|---:|
| `blk.60.ffn_down_exps.weight` | `1` | `base_only` | `0.0000` | `28.00` | `0.946337` | `0.937965` | `0.944217` |
| `blk.60.ffn_down_exps.weight` | `1` | `residual_1bit` | `0.3091` | `28.00` | `2.150013` | `2.130529` | `2.149083` |
| `blk.60.ffn_down_exps.weight` | `1` | `residual_2bit` | `0.6000` | `28.00` | `0.768445` | `0.760023` | `0.766711` |
| `blk.60.ffn_down_exps.weight` | `4` | `base_only` | `0.0000` | `112.00` | `0.751338` | `0.735754` | `0.765620` |
| `blk.60.ffn_down_exps.weight` | `4` | `residual_1bit` | `0.3091` | `112.00` | `1.707213` | `1.659475` | `1.721163` |
| `blk.60.ffn_down_exps.weight` | `4` | `residual_2bit` | `0.6000` | `112.00` | `0.604427` | `0.591922` | `0.609699` |
| `blk.60.ffn_down_exps.weight` | `8` | `base_only` | `0.0000` | `224.00` | `0.460191` | `0.508142` | `0.544927` |
| `blk.60.ffn_down_exps.weight` | `8` | `residual_1bit` | `0.3091` | `224.00` | `1.035591` | `1.142634` | `1.227380` |
| `blk.60.ffn_down_exps.weight` | `8` | `residual_2bit` | `0.6000` | `224.00` | `0.369914` | `0.405364` | `0.438060` |

## Decision

Reject as primary: best target-ratio candidate blk.60.ffn_down_exps.weight clusters=8 residual_1bit has group mean rel L2 1.142634, above the 0.10 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_cluster_base_activation_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp99-cluster-base-activation-smoke/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp99-cluster-base-activation-smoke/report.md --max-records-per-prompt 64 --max-tensors 1 --clusters 1,4,8 --bits 1,2 --block 256 --sketch-dim 4096 --torch-threads 8
```

Executed on the remote host from `/root/lfz/tmp/kimi-stage2m-align` under
`systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0`.
