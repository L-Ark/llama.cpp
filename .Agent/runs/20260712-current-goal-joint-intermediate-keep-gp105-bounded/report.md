# Kimi joint intermediate-dimension keep oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-12T09:22:09+0000`
- prompts: `3`
- keep fractions: `[0.45, 0.48, 0.5, 0.52, 0.55, 0.58, 0.6]`

## Aggregate

| keep | byte ratio | row mean rel L2 | row max rel L2 | group mean rel L2 | group max rel L2 |
|---:|---:|---:|---:|---:|---:|
| `0.45` | `0.4500` | `0.139522` | `0.231181` | `0.132250` | `0.164276` |
| `0.48` | `0.4800` | `0.123822` | `0.207401` | `0.117306` | `0.146492` |
| `0.5` | `0.5000` | `0.113979` | `0.192250` | `0.107984` | `0.135587` |
| `0.52` | `0.5200` | `0.104647` | `0.177216` | `0.099207` | `0.125209` |
| `0.55` | `0.5500` | `0.091733` | `0.157773` | `0.086941` | `0.110633` |
| `0.58` | `0.5800` | `0.079656` | `0.137870` | `0.075411` | `0.096500` |
| `0.6` | `0.6000` | `0.072216` | `0.126524` | `0.068359` | `0.087230` |

## Decision

Reject joint intermediate-dimension partial reads as the next primary runtime path: at keep <= 0.5, grouped down-output mean rel L2 is 0.107984, above the 0.1 gate.

## Prompt Summaries

### dev_france_regression

- records: `384`
- groups: `48`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.45` | `0.140084` | `0.132344` |
| `0.48` | `0.124346` | `0.117445` |
| `0.5` | `0.114435` | `0.108105` |
| `0.52` | `0.105104` | `0.099393` |
| `0.55` | `0.092124` | `0.087107` |
| `0.58` | `0.080024` | `0.075481` |
| `0.6` | `0.072552` | `0.068422` |

### dev_japan_factual

- records: `384`
- groups: `48`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.45` | `0.138015` | `0.130233` |
| `0.48` | `0.122481` | `0.115371` |
| `0.5` | `0.112718` | `0.106211` |
| `0.52` | `0.103469` | `0.097564` |
| `0.55` | `0.090669` | `0.085434` |
| `0.58` | `0.078733` | `0.074155` |
| `0.6` | `0.071368` | `0.067238` |

### dev_photosynthesis_factual

- records: `384`
- groups: `48`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.45` | `0.140465` | `0.134173` |
| `0.48` | `0.124638` | `0.119101` |
| `0.5` | `0.114784` | `0.109637` |
| `0.52` | `0.105369` | `0.100664` |
| `0.55` | `0.092405` | `0.088282` |
| `0.58` | `0.080211` | `0.076597` |
| `0.6` | `0.072727` | `0.069416` |

## Reproduce

```bash
.Agent/run-tools/kimi_joint_intermediate_keep_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_france_regression --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_photosynthesis_factual --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-current-goal-joint-intermediate-keep-gp105-bounded/report.json --out-md .Agent/runs/20260712-current-goal-joint-intermediate-keep-gp105-bounded/report.md --keep-fracs 0.45,0.48,0.50,0.52,0.55,0.58,0.60 --max-records-per-prompt 384 --target-keep-frac 0.5 --target-group-rel-l2 0.10 --torch-threads 8
```
