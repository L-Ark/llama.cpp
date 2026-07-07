# Kimi shared down-projector oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T20:05:24+0000`
- prompts: `3`
- groups: `96`
- layers evaluated: `32`
- max down records per prompt: `256`

## Summary

| mode | lambda | rows | mean rel L2 | max rel L2 | BF16 MiB/layer | BF16 GiB/60 layers |
|---|---:|---:|---:|---:|---:|---:|
| `sum_h` | `0.001` | `96` | `3.891767` | `92.493989` | `28.00` | `1.64` |
| `sum_h` | `0.01` | `96` | `3.877807` | `92.079398` | `28.00` | `1.64` |
| `sum_h` | `0.1` | `96` | `3.745093` | `88.128874` | `28.00` | `1.64` |
| `sum_h` | `1` | `96` | `2.874318` | `61.661736` | `28.00` | `1.64` |
| `sum_h_abs` | `0.001` | `96` | `4.291087` | `80.018142` | `56.00` | `3.28` |
| `sum_h_abs` | `0.01` | `96` | `4.275506` | `79.690436` | `56.00` | `3.28` |
| `sum_h_abs` | `0.1` | `96` | `4.124345` | `76.280672` | `56.00` | `3.28` |
| `sum_h_abs` | `1` | `96` | `3.129784` | `53.417229` | `56.00` | `3.28` |
| `sum_h_abs_sq` | `0.001` | `96` | `3.655398` | `51.568411` | `84.00` | `4.92` |
| `sum_h_abs_sq` | `0.01` | `96` | `3.642652` | `51.351340` | `84.00` | `4.92` |
| `sum_h_abs_sq` | `0.1` | `96` | `3.520159` | `49.155112` | `84.00` | `4.92` |
| `sum_h_abs_sq` | `1` | `96` | `2.717026` | `34.433532` | `84.00` | `4.92` |

## Worst Layer Rows

| layer | mode | lambda | rows | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|---:|
| `blk.37` | `sum_h` | `0.001` | `3` | `33.619005` | `92.493989` |
| `blk.37` | `sum_h` | `0.01` | `3` | `33.469379` | `92.079398` |
| `blk.37` | `sum_h` | `0.1` | `3` | `32.043884` | `88.128874` |
| `blk.51` | `sum_h_abs` | `0.001` | `3` | `30.020791` | `80.018142` |
| `blk.51` | `sum_h_abs` | `0.01` | `3` | `29.896668` | `79.690436` |
| `blk.51` | `sum_h_abs` | `0.1` | `3` | `28.618505` | `76.280672` |
| `blk.37` | `sum_h` | `1` | `3` | `22.509372` | `61.661736` |
| `blk.51` | `sum_h_abs_sq` | `0.001` | `3` | `20.537772` | `51.568411` |
| `blk.51` | `sum_h_abs_sq` | `0.01` | `3` | `20.450397` | `51.351340` |
| `blk.51` | `sum_h_abs` | `1` | `3` | `20.067936` | `53.417229` |
| `blk.51` | `sum_h_abs_sq` | `0.1` | `3` | `19.575478` | `49.155112` |
| `blk.37` | `sum_h_abs` | `0.001` | `3` | `17.155288` | `46.043397` |
| `blk.37` | `sum_h_abs` | `0.01` | `3` | `17.079869` | `45.836867` |
| `blk.37` | `sum_h_abs` | `0.1` | `3` | `16.361546` | `43.868940` |
| `blk.51` | `sum_h_abs_sq` | `1` | `3` | `13.728706` | `34.433532` |
| `blk.37` | `sum_h_abs_sq` | `0.001` | `3` | `13.198610` | `31.703731` |
| `blk.37` | `sum_h_abs_sq` | `0.01` | `3` | `13.140931` | `31.561474` |
| `blk.37` | `sum_h_abs_sq` | `0.1` | `3` | `12.591581` | `30.206037` |
| `blk.37` | `sum_h_abs` | `1` | `3` | `11.569231` | `30.686392` |
| `blk.12` | `sum_h` | `0.001` | `3` | `11.433545` | `30.742535` |
| `blk.12` | `sum_h` | `0.01` | `3` | `11.385375` | `30.604898` |
| `blk.12` | `sum_h` | `0.1` | `3` | `10.926572` | `29.293459` |
| `blk.37` | `sum_h_abs_sq` | `1` | `3` | `8.925278` | `21.128317` |
| `blk.12` | `sum_h` | `1` | `3` | `7.863171` | `20.509628` |
| `blk.34` | `sum_h_abs` | `0.001` | `3` | `7.611884` | `16.198194` |
| `blk.34` | `sum_h_abs` | `0.01` | `3` | `7.580143` | `16.126548` |
| `blk.52` | `sum_h_abs_sq` | `0.001` | `3` | `7.349910` | `11.093366` |
| `blk.52` | `sum_h_abs_sq` | `0.01` | `3` | `7.318277` | `11.044613` |
| `blk.34` | `sum_h_abs` | `0.1` | `3` | `7.277969` | `15.443915` |
| `blk.52` | `sum_h_abs_sq` | `0.1` | `3` | `7.017093` | `10.580208` |
| `blk.12` | `sum_h_abs` | `0.001` | `3` | `6.653467` | `16.169501` |
| `blk.12` | `sum_h_abs` | `0.01` | `3` | `6.626267` | `16.097275` |
| `blk.12` | `sum_h_abs` | `0.1` | `3` | `6.367409` | `15.409106` |
| `blk.34` | `sum_h_abs` | `1` | `3` | `5.271473` | `10.875904` |
| `blk.34` | `sum_h_abs_sq` | `0.001` | `3` | `5.163380` | `11.070568` |
| `blk.34` | `sum_h_abs_sq` | `0.01` | `3` | `5.143410` | `11.022038` |
| `blk.12` | `sum_h_abs_sq` | `0.001` | `3` | `5.120747` | `12.080499` |
| `blk.12` | `sum_h_abs_sq` | `0.01` | `3` | `5.101005` | `12.026660` |
| `blk.52` | `sum_h_abs_sq` | `1` | `3` | `5.015026` | `7.478861` |
| `blk.34` | `sum_h_abs_sq` | `0.1` | `3` | `4.953417` | `10.559741` |

## Decision

Reject as primary: best shared down projector sum_h_abs_sq lambda=1 has mean rel L2 2.717026, above the 0.10 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_shared_down_projector_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp100-shared-down-projector-smoke/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp100-shared-down-projector-smoke/report.md --max-down-records-per-prompt 256 --modes sum_h,sum_h_abs,sum_h_abs_sq --lambdas 0.001,0.01,0.1,1.0 --torch-threads 8
```

Executed on the remote host from `/root/lfz/tmp/kimi-stage2m-align` under
`systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0`.
