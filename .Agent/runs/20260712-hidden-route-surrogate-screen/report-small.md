# Kimi input-route MoE-output surrogate oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-12T04:36:50+0000`
- prompts: `3`
- groups: `75`
- layers evaluated: `25`
- max activation records per prompt: `720`
- complete-group only: `True`

## Summary

| method | rows | mean rel L2 | max rel L2 |
|---|---:|---:|---:|
| `krr:input_route_gated:lambda=10` | `75` | `1.064523` | `1.492350` |
| `krr:input_route_stats:lambda=10` | `75` | `1.129687` | `1.903881` |
| `krr:input_stats:lambda=10` | `75` | `1.130047` | `1.906169` |
| `krr:input:lambda=10` | `75` | `1.130147` | `1.906549` |
| `nn_raw:input_route_gated` | `75` | `1.159443` | `1.648673` |
| `nn_raw:input` | `75` | `1.190653` | `1.648673` |
| `nn_raw:input_route_stats` | `75` | `1.190653` | `1.648673` |
| `nn_raw:input_stats` | `75` | `1.190653` | `1.648673` |
| `krr:input_route_gated:lambda=1` | `75` | `1.221120` | `2.898693` |
| `nn_scaled:input_route_gated` | `75` | `1.542301` | `4.461015` |
| `krr:input_route_stats:lambda=1` | `75` | `1.705877` | `7.301841` |
| `krr:input_stats:lambda=1` | `75` | `1.707780` | `7.323120` |
| `krr:input:lambda=1` | `75` | `1.708253` | `7.327095` |
| `nn_scaled:input_route_stats` | `75` | `2.234803` | `5.496183` |
| `nn_scaled:input_stats` | `75` | `2.235346` | `5.496183` |
| `nn_scaled:input` | `75` | `2.235376` | `5.496183` |

## Prototype Resident Estimate

| feature mode | feature dim | BF16 bytes/prototype with output | MiB/layer @ K=64 | GiB/60 layers @ K=64 |
|---|---:|---:|---:|---:|
| `input` | `2048` | `8192` | `0.500` | `0.029` |
| `input_stats` | `2052` | `8200` | `0.500` | `0.029` |
| `input_route_stats` | `2073` | `8242` | `0.503` | `0.029` |
| `input_route_gated` | `12288` | `28672` | `1.750` | `0.103` |

## Worst Layer Rows

| layer | method | rows | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|
| `blk.29` | `krr:input:lambda=1` | `3` | `3.880853` | `7.327095` |
| `blk.29` | `krr:input_stats:lambda=1` | `3` | `3.878449` | `7.323120` |
| `blk.29` | `krr:input_route_stats:lambda=1` | `3` | `3.868735` | `7.301841` |
| `blk.29` | `nn_scaled:input` | `3` | `3.787203` | `5.173819` |
| `blk.29` | `nn_scaled:input_stats` | `3` | `3.784052` | `5.173819` |
| `blk.29` | `nn_scaled:input_route_stats` | `3` | `3.781503` | `5.173819` |
| `blk.20` | `nn_scaled:input` | `3` | `3.197821` | `4.771026` |
| `blk.20` | `nn_scaled:input_stats` | `3` | `3.196408` | `4.768009` |
| `blk.20` | `nn_scaled:input_route_stats` | `3` | `3.186841` | `4.752153` |
| `blk.12` | `nn_scaled:input` | `3` | `3.040888` | `5.496183` |
| `blk.12` | `nn_scaled:input_stats` | `3` | `3.040670` | `5.496183` |
| `blk.12` | `nn_scaled:input_route_stats` | `3` | `3.039179` | `5.496183` |
| `blk.30` | `nn_scaled:input` | `3` | `3.014295` | `3.841330` |
| `blk.30` | `nn_scaled:input_stats` | `3` | `3.013927` | `3.841330` |
| `blk.30` | `nn_scaled:input_route_stats` | `3` | `3.012408` | `3.841330` |
| `blk.4` | `nn_scaled:input_stats` | `3` | `2.841172` | `4.054580` |
| `blk.4` | `nn_scaled:input_route_stats` | `3` | `2.840157` | `4.054580` |
| `blk.4` | `nn_scaled:input` | `3` | `2.830510` | `4.054580` |
| `blk.31` | `nn_scaled:input_route_gated` | `3` | `2.778043` | `4.461015` |
| `blk.4` | `krr:input_stats:lambda=1` | `3` | `2.616866` | `3.415746` |
| `blk.4` | `krr:input:lambda=1` | `3` | `2.615832` | `3.417599` |
| `blk.4` | `krr:input_route_stats:lambda=1` | `3` | `2.612015` | `3.409623` |
| `blk.28` | `nn_scaled:input` | `3` | `2.551456` | `5.483152` |
| `blk.28` | `nn_scaled:input_route_stats` | `3` | `2.550242` | `5.483152` |
| `blk.28` | `nn_scaled:input_stats` | `3` | `2.550190` | `5.483152` |
| `blk.19` | `nn_scaled:input_route_stats` | `3` | `2.521827` | `3.744818` |
| `blk.19` | `nn_scaled:input` | `3` | `2.512992` | `3.744818` |
| `blk.19` | `nn_scaled:input_stats` | `3` | `2.512187` | `3.744818` |
| `blk.21` | `nn_scaled:input` | `3` | `2.509357` | `3.260228` |
| `blk.21` | `nn_scaled:input_route_stats` | `3` | `2.506193` | `3.247553` |
| `blk.21` | `nn_scaled:input_stats` | `3` | `2.505624` | `3.250626` |
| `blk.31` | `krr:input:lambda=1` | `3` | `2.480096` | `5.340666` |
| `blk.31` | `krr:input_stats:lambda=1` | `3` | `2.479336` | `5.338674` |
| `blk.31` | `krr:input_route_stats:lambda=1` | `3` | `2.475426` | `5.327352` |
| `blk.12` | `krr:input:lambda=1` | `3` | `2.440423` | `4.881862` |
| `blk.12` | `krr:input_stats:lambda=1` | `3` | `2.439450` | `4.879154` |
| `blk.4` | `nn_scaled:input_route_gated` | `3` | `2.438641` | `3.330886` |
| `blk.12` | `krr:input_route_stats:lambda=1` | `3` | `2.434484` | `4.865479` |
| `blk.5` | `nn_scaled:input_route_stats` | `3` | `2.401674` | `3.549739` |
| `blk.5` | `nn_scaled:input` | `3` | `2.401364` | `3.549739` |
| `blk.5` | `nn_scaled:input_stats` | `3` | `2.400322` | `3.549739` |
| `blk.60` | `nn_scaled:input` | `3` | `2.393241` | `3.008044` |
| `blk.60` | `nn_scaled:input_stats` | `3` | `2.391064` | `3.005383` |
| `blk.60` | `nn_scaled:input_route_stats` | `3` | `2.389874` | `3.006957` |
| `blk.14` | `nn_scaled:input_route_stats` | `3` | `2.300829` | `3.421887` |
| `blk.28` | `krr:input_stats:lambda=1` | `3` | `2.295498` | `4.807624` |
| `blk.28` | `krr:input:lambda=1` | `3` | `2.295347` | `4.806864` |
| `blk.28` | `krr:input_route_stats:lambda=1` | `3` | `2.291384` | `4.795088` |
| `blk.31` | `nn_scaled:input` | `3` | `2.288681` | `4.461015` |
| `blk.31` | `nn_scaled:input_stats` | `3` | `2.288532` | `4.461015` |

## Decision

Reject as primary: best input-route surrogate krr:input_route_gated:lambda=10 has mean rel L2 1.064523, above the 0.1 gate. This closes the small prototype/kernel full-MoE-output surrogate family for now.

## Reproduce

```bash
.Agent/run-tools/kimi_input_route_moe_surrogate_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_france_regression --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_photosynthesis_factual --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-hidden-route-surrogate-screen/report-small.json --out-md .Agent/runs/20260712-hidden-route-surrogate-screen/report-small.md --max-records-per-prompt 720 --modes input,input_stats,input_route_stats,input_route_gated --lambdas 1.0,10.0 --complete-group-only --target-rel-l2 0.10 --torch-threads 8
```
