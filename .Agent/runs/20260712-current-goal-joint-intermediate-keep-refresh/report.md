# Kimi joint intermediate-dimension keep oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-12T07:35:26+0000`
- prompts: `3`
- keep fractions: `[0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6]`

## Aggregate

| keep | byte ratio | row mean rel L2 | row max rel L2 | group mean rel L2 | group max rel L2 |
|---:|---:|---:|---:|---:|---:|
| `0.4` | `0.4000` | `0.169726` | `0.287683` | `0.159495` | `0.221885` |
| `0.45` | `0.4500` | `0.140134` | `0.240430` | `0.131601` | `0.186090` |
| `0.48` | `0.4800` | `0.124388` | `0.215395` | `0.116727` | `0.166445` |
| `0.5` | `0.5000` | `0.114506` | `0.200821` | `0.107407` | `0.152523` |
| `0.52` | `0.5200` | `0.105154` | `0.186497` | `0.098653` | `0.140934` |
| `0.55` | `0.5500` | `0.092173` | `0.168037` | `0.086354` | `0.126843` |
| `0.6` | `0.6000` | `0.072670` | `0.137818` | `0.068085` | `0.102728` |

## Decision

Reject joint intermediate-dimension partial reads as the next primary runtime path: at keep <= 0.5, grouped down-output mean rel L2 is 0.107407, above the 0.1 gate.

## Prompt Summaries

### dev_japan_factual

- records: `192`
- groups: `24`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.4` | `0.164731` | `0.153551` |
| `0.45` | `0.135888` | `0.126659` |
| `0.48` | `0.120563` | `0.112286` |
| `0.5` | `0.110918` | `0.103247` |
| `0.52` | `0.101791` | `0.094751` |
| `0.55` | `0.089189` | `0.082955` |
| `0.6` | `0.070267` | `0.065378` |

### dev_mixed_summary

- records: `192`
- groups: `24`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.4` | `0.171549` | `0.161984` |
| `0.45` | `0.141781` | `0.133896` |
| `0.48` | `0.125909` | `0.118902` |
| `0.5` | `0.115994` | `0.109440` |
| `0.52` | `0.106630` | `0.100583` |
| `0.55` | `0.093567` | `0.088074` |
| `0.6` | `0.073860` | `0.069498` |

### dev_python_reverse

- records: `192`
- groups: `24`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.4` | `0.172898` | `0.162951` |
| `0.45` | `0.142731` | `0.134250` |
| `0.48` | `0.126691` | `0.118993` |
| `0.5` | `0.116605` | `0.109533` |
| `0.52` | `0.107041` | `0.100624` |
| `0.55` | `0.093763` | `0.088035` |
| `0.6` | `0.073883` | `0.069379` |

## Reproduce

```bash
.Agent/run-tools/kimi_joint_intermediate_keep_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-current-goal-joint-intermediate-keep-refresh/report.json --out-md .Agent/runs/20260712-current-goal-joint-intermediate-keep-refresh/report.md --keep-fracs 0.4,0.45,0.48,0.5,0.52,0.55,0.6 --max-records-per-prompt 192 --target-keep-frac 0.5 --target-group-rel-l2 0.10 --torch-threads 8
```
