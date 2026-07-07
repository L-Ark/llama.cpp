# Kimi activation-weighted cluster-base residual oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T19:56:45+0000`
- prompt roots: `3`
- max records per prompt: `192`
- tensors evaluated: `2`
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
| `blk.12.ffn_down_exps.weight` | `1` | `base_only` | `0.0000` | `28.00` | `0.964768` | `0.965383` | `0.969936` |
| `blk.12.ffn_down_exps.weight` | `1` | `residual_1bit` | `0.3091` | `28.00` | `2.154796` | `2.169308` | `2.178746` |
| `blk.12.ffn_down_exps.weight` | `1` | `residual_2bit` | `0.6000` | `28.00` | `0.777755` | `0.783945` | `0.785082` |
| `blk.12.ffn_down_exps.weight` | `4` | `base_only` | `0.0000` | `112.00` | `0.872376` | `0.874265` | `0.905801` |
| `blk.12.ffn_down_exps.weight` | `4` | `residual_1bit` | `0.3091` | `112.00` | `1.968934` | `1.989792` | `2.048534` |
| `blk.12.ffn_down_exps.weight` | `4` | `residual_2bit` | `0.6000` | `112.00` | `0.701966` | `0.706330` | `0.731654` |
| `blk.12.ffn_down_exps.weight` | `8` | `base_only` | `0.0000` | `224.00` | `0.646318` | `0.718243` | `0.836564` |
| `blk.12.ffn_down_exps.weight` | `8` | `residual_1bit` | `0.3091` | `224.00` | `1.450863` | `1.606279` | `1.900464` |
| `blk.12.ffn_down_exps.weight` | `8` | `residual_2bit` | `0.6000` | `224.00` | `0.520164` | `0.572879` | `0.668787` |

## Decision

Reject as primary: best target-ratio candidate blk.60.ffn_down_exps.weight clusters=8 residual_1bit has group mean rel L2 1.142634, above the 0.10 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_cluster_base_activation_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp99-cluster-base-activation-expanded/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp99-cluster-base-activation-expanded/report.md --max-records-per-prompt 192 --max-tensors 2 --clusters 1,4,8 --bits 1,2 --block 256 --sketch-dim 4096 --torch-threads 8
```

Executed on the remote host from `/root/lfz/tmp/kimi-stage2m-align` under:

```bash
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  python3 .Agent/run-tools/kimi_cluster_base_activation_oracle.py \
    --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse \
    --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual \
    --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary \
    --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
    --libggml-base build-cuda-batch/bin/libggml-base.so \
    --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp99-cluster-base-activation-expanded/report.json \
    --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp99-cluster-base-activation-expanded/report.md \
    --max-records-per-prompt 192 --max-tensors 2 \
    --clusters 1,4,8 --bits 1,2 --block 256 \
    --sketch-dim 4096 --torch-threads 8
```
