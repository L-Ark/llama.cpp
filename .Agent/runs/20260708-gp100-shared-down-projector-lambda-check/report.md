# Kimi shared down-projector oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T20:07:51+0000`
- prompts: `3`
- groups: `96`
- layers evaluated: `32`
- max down records per prompt: `256`

## Summary

| mode | lambda | rows | mean rel L2 | max rel L2 | BF16 MiB/layer | BF16 GiB/60 layers |
|---|---:|---:|---:|---:|---:|---:|
| `sum_h_abs_sq` | `1` | `96` | `2.717026` | `34.433532` | `84.00` | `4.92` |
| `sum_h_abs_sq` | `10` | `96` | `1.480783` | `8.715341` | `84.00` | `4.92` |
| `sum_h_abs_sq` | `100` | `96` | `1.260756` | `1.774349` | `84.00` | `4.92` |

## Worst Layer Rows

| layer | mode | lambda | rows | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|---:|
| `blk.51` | `sum_h_abs_sq` | `1` | `3` | `13.728706` | `34.433532` |
| `blk.37` | `sum_h_abs_sq` | `1` | `3` | `8.925278` | `21.128317` |
| `blk.52` | `sum_h_abs_sq` | `1` | `3` | `5.015026` | `7.478861` |
| `blk.51` | `sum_h_abs_sq` | `10` | `3` | `3.820551` | `8.715341` |
| `blk.34` | `sum_h_abs_sq` | `1` | `3` | `3.697706` | `7.470466` |
| `blk.12` | `sum_h_abs_sq` | `1` | `3` | `3.666510` | `8.085127` |
| `blk.23` | `sum_h_abs_sq` | `1` | `3` | `3.118368` | `5.737922` |
| `blk.35` | `sum_h_abs_sq` | `1` | `3` | `3.103685` | `6.622766` |
| `blk.48` | `sum_h_abs_sq` | `1` | `3` | `3.002271` | `4.329905` |
| `blk.37` | `sum_h_abs_sq` | `10` | `3` | `2.695591` | `5.335369` |
| `blk.27` | `sum_h_abs_sq` | `1` | `3` | `2.607411` | `3.897687` |
| `blk.45` | `sum_h_abs_sq` | `1` | `3` | `2.533693` | `4.155237` |
| `blk.55` | `sum_h_abs_sq` | `1` | `3` | `2.353618` | `2.882385` |
| `blk.60` | `sum_h_abs_sq` | `1` | `3` | `2.236163` | `3.199937` |
| `blk.59` | `sum_h_abs_sq` | `1` | `3` | `2.018265` | `3.307923` |
| `blk.21` | `sum_h_abs_sq` | `1` | `3` | `1.909948` | `3.134343` |
| `blk.16` | `sum_h_abs_sq` | `1` | `3` | `1.892288` | `2.907500` |
| `blk.28` | `sum_h_abs_sq` | `1` | `3` | `1.858356` | `2.490289` |
| `blk.41` | `sum_h_abs_sq` | `1` | `3` | `1.830657` | `2.419915` |
| `blk.20` | `sum_h_abs_sq` | `1` | `3` | `1.808180` | `2.576608` |
| `blk.52` | `sum_h_abs_sq` | `10` | `3` | `1.777007` | `2.269350` |
| `blk.30` | `sum_h_abs_sq` | `1` | `3` | `1.750466` | `2.600138` |
| `blk.58` | `sum_h_abs_sq` | `1` | `3` | `1.731800` | `2.312113` |
| `blk.24` | `sum_h_abs_sq` | `1` | `3` | `1.718743` | `2.195225` |
| `blk.31` | `sum_h_abs_sq` | `1` | `3` | `1.714526` | `2.398051` |
| `blk.38` | `sum_h_abs_sq` | `1` | `3` | `1.671127` | `1.935911` |
| `blk.34` | `sum_h_abs_sq` | `10` | `3` | `1.662263` | `2.238090` |
| `blk.49` | `sum_h_abs_sq` | `1` | `3` | `1.606641` | `2.018289` |
| `blk.12` | `sum_h_abs_sq` | `10` | `3` | `1.599650` | `2.254539` |
| `blk.14` | `sum_h_abs_sq` | `1` | `3` | `1.583983` | `1.962560` |
| `blk.35` | `sum_h_abs_sq` | `10` | `3` | `1.580651` | `2.391579` |
| `blk.5` | `sum_h_abs_sq` | `1` | `3` | `1.538321` | `1.852116` |
| `blk.11` | `sum_h_abs_sq` | `1` | `3` | `1.532175` | `2.111643` |
| `blk.56` | `sum_h_abs_sq` | `1` | `3` | `1.526398` | `1.827410` |
| `blk.55` | `sum_h_abs_sq` | `10` | `3` | `1.485304` | `1.592518` |
| `blk.45` | `sum_h_abs_sq` | `10` | `3` | `1.477680` | `1.929669` |
| `blk.23` | `sum_h_abs_sq` | `10` | `3` | `1.457715` | `1.806683` |
| `blk.48` | `sum_h_abs_sq` | `10` | `3` | `1.435049` | `1.703464` |
| `blk.2` | `sum_h_abs_sq` | `1` | `3` | `1.423056` | `1.696107` |
| `blk.5` | `sum_h_abs_sq` | `10` | `3` | `1.395501` | `1.786718` |

## Decision

Reject as primary: best shared down projector sum_h_abs_sq lambda=100 has mean rel L2 1.260756, above the 0.10 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_shared_down_projector_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp100-shared-down-projector-lambda-check/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp100-shared-down-projector-lambda-check/report.md --max-down-records-per-prompt 256 --modes sum_h_abs_sq --lambdas 1.0,10.0,100.0 --torch-threads 8
```

Executed on the remote host from `/root/lfz/tmp/kimi-stage2m-align` under
`systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0`.
