# Kimi shared down-projector oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T20:24:57+0000`
- prompts: `3`
- groups: `96`
- layers evaluated: `32`
- max down records per prompt: `256`

## Summary

| mode | lambda | rows | mean rel L2 | max rel L2 | BF16 MiB/layer | BF16 GiB/60 layers |
|---|---:|---:|---:|---:|---:|---:|
| `route_hash4_sum_h` | `10` | `96` | `1.415892` | `4.710977` | `140.00` | `8.20` |
| `route_hash4_sum_h` | `100` | `96` | `1.257192` | `1.769298` | `140.00` | `8.20` |
| `route_hash4_sum_h` | `1000` | `96` | `1.253637` | `1.772551` | `140.00` | `8.20` |
| `route_scalar_sum_h` | `10` | `96` | `1.448896` | `5.512427` | `84.00` | `4.92` |
| `route_scalar_sum_h` | `100` | `96` | `1.255206` | `1.765178` | `84.00` | `4.92` |
| `route_scalar_sum_h` | `1000` | `96` | `1.253345` | `1.772039` | `84.00` | `4.92` |

## Worst Layer Rows

| layer | mode | lambda | rows | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|---:|
| `blk.37` | `route_scalar_sum_h` | `10` | `3` | `2.661882` | `5.512427` |
| `blk.49` | `route_hash4_sum_h` | `10` | `3` | `2.443668` | `4.710977` |
| `blk.24` | `route_scalar_sum_h` | `10` | `3` | `2.202180` | `3.939357` |
| `blk.48` | `route_hash4_sum_h` | `10` | `3` | `2.168869` | `3.852451` |
| `blk.37` | `route_hash4_sum_h` | `10` | `3` | `1.959776` | `3.268942` |
| `blk.51` | `route_scalar_sum_h` | `10` | `3` | `1.853923` | `3.048927` |
| `blk.42` | `route_hash4_sum_h` | `10` | `3` | `1.772413` | `2.683740` |
| `blk.52` | `route_scalar_sum_h` | `10` | `3` | `1.769090` | `2.637064` |
| `blk.41` | `route_scalar_sum_h` | `10` | `3` | `1.719659` | `2.279617` |
| `blk.44` | `route_scalar_sum_h` | `10` | `3` | `1.631217` | `2.431661` |
| `blk.55` | `route_scalar_sum_h` | `10` | `3` | `1.563209` | `2.252904` |
| `blk.21` | `route_scalar_sum_h` | `10` | `3` | `1.551515` | `2.424205` |
| `blk.35` | `route_scalar_sum_h` | `10` | `3` | `1.550478` | `2.337810` |
| `blk.58` | `route_scalar_sum_h` | `10` | `3` | `1.529881` | `1.925565` |
| `blk.23` | `route_hash4_sum_h` | `10` | `3` | `1.528765` | `2.188485` |
| `blk.59` | `route_hash4_sum_h` | `10` | `3` | `1.512106` | `1.607641` |
| `blk.31` | `route_hash4_sum_h` | `10` | `3` | `1.508908` | `2.038656` |
| `blk.24` | `route_hash4_sum_h` | `10` | `3` | `1.488505` | `1.759654` |
| `blk.48` | `route_scalar_sum_h` | `10` | `3` | `1.481833` | `1.755755` |
| `blk.14` | `route_scalar_sum_h` | `10` | `3` | `1.478680` | `2.040909` |
| `blk.12` | `route_scalar_sum_h` | `10` | `3` | `1.447490` | `1.847145` |
| `blk.24` | `route_scalar_sum_h` | `100` | `3` | `1.434793` | `1.686397` |
| `blk.41` | `route_hash4_sum_h` | `10` | `3` | `1.413235` | `1.586830` |
| `blk.44` | `route_hash4_sum_h` | `10` | `3` | `1.395328` | `1.717090` |
| `blk.35` | `route_hash4_sum_h` | `10` | `3` | `1.389236` | `1.514292` |
| `blk.24` | `route_scalar_sum_h` | `1000` | `3` | `1.375864` | `1.696628` |
| `blk.24` | `route_hash4_sum_h` | `100` | `3` | `1.373875` | `1.704557` |
| `blk.24` | `route_hash4_sum_h` | `1000` | `3` | `1.370535` | `1.698490` |
| `blk.5` | `route_hash4_sum_h` | `1000` | `3` | `1.362246` | `1.772551` |
| `blk.5` | `route_scalar_sum_h` | `1000` | `3` | `1.361727` | `1.772039` |
| `blk.5` | `route_hash4_sum_h` | `100` | `3` | `1.361003` | `1.769298` |
| `blk.3` | `route_hash4_sum_h` | `1000` | `3` | `1.359394` | `1.604701` |
| `blk.38` | `route_scalar_sum_h` | `10` | `3` | `1.359360` | `1.634831` |
| `blk.3` | `route_scalar_sum_h` | `1000` | `3` | `1.359096` | `1.604213` |
| `blk.52` | `route_hash4_sum_h` | `10` | `3` | `1.358376` | `1.536495` |
| `blk.3` | `route_hash4_sum_h` | `100` | `3` | `1.357141` | `1.608016` |
| `blk.5` | `route_scalar_sum_h` | `100` | `3` | `1.356275` | `1.765178` |
| `blk.5` | `route_hash4_sum_h` | `10` | `3` | `1.355774` | `1.756796` |
| `blk.3` | `route_scalar_sum_h` | `100` | `3` | `1.354110` | `1.603210` |
| `blk.3` | `route_hash4_sum_h` | `10` | `3` | `1.353529` | `1.636821` |

## Decision

Reject as primary: best shared down projector route_scalar_sum_h lambda=1000 has mean rel L2 1.253345, above the 0.10 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_shared_down_projector_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp102-route-conditioned-down-surrogate/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp102-route-conditioned-down-surrogate/report.md --max-down-records-per-prompt 256 --modes route_scalar_sum_h,route_hash4_sum_h --lambdas 10.0,100.0,1000.0 --torch-threads 8
```

Executed on the remote host from `/root/lfz/tmp/kimi-stage2m-align` under
`systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0`.
