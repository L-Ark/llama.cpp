# Kimi scalar-corrected intermediate keep oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T19:37:16+0000`
- prompts: `3`
- keep fractions: `[0.35, 0.4, 0.45, 0.5]`

## Aggregate Group Error

| keep | raw mean rel L2 | row-scalar-sum mean rel L2 | group-scalar mean rel L2 | group-scalar max rel L2 | mean group alpha |
|---:|---:|---:|---:|---:|---:|
| `0.35` | `0.188441` | `0.188425` | `0.188425` | `0.234301` | `1.000491` |
| `0.4` | `0.157275` | `0.157262` | `0.157266` | `0.198101` | `1.000280` |
| `0.45` | `0.129764` | `0.129756` | `0.129755` | `0.165924` | `1.000023` |
| `0.5` | `0.105683` | `0.105676` | `0.105678` | `0.138143` | `0.999897` |

## Aggregate Row Error

| keep | raw mean rel L2 | row-scalar mean rel L2 | row-scalar max rel L2 | mean row alpha |
|---:|---:|---:|---:|---:|
| `0.35` | `0.201473` | `0.201460` | `0.335930` | `0.999767` |
| `0.4` | `0.168237` | `0.168226` | `0.287631` | `0.999778` |
| `0.45` | `0.138887` | `0.138877` | `0.240428` | `0.999935` |
| `0.5` | `0.113524` | `0.113517` | `0.200817` | `0.999921` |

## Decision

Reject scalar correction as a rescue for the 0.4x intermediate top-k path: even the optimal group scalar has mean rel L2 0.157266, above the 0.1 gate.

## Prompt Summaries

### dev_japan_factual

- records: `64`
- groups: `8`

| keep | raw group rel L2 | group-scalar rel L2 | mean alpha |
|---:|---:|---:|---:|
| `0.35` | `0.178138` | `0.178131` | `0.999931` |
| `0.4` | `0.147883` | `0.147880` | `0.999853` |
| `0.45` | `0.121810` | `0.121805` | `0.999539` |
| `0.5` | `0.099021` | `0.099018` | `0.999726` |

### dev_mixed_summary

- records: `64`
- groups: `8`

| keep | raw group rel L2 | group-scalar rel L2 | mean alpha |
|---:|---:|---:|---:|
| `0.35` | `0.189049` | `0.189021` | `1.002257` |
| `0.4` | `0.157720` | `0.157699` | `1.001382` |
| `0.45` | `0.129774` | `0.129758` | `1.001116` |
| `0.5` | `0.105551` | `0.105545` | `1.000626` |

### dev_python_reverse

- records: `64`
- groups: `8`

| keep | raw group rel L2 | group-scalar rel L2 | mean alpha |
|---:|---:|---:|---:|
| `0.35` | `0.198136` | `0.198124` | `0.999284` |
| `0.4` | `0.166223` | `0.166218` | `0.999604` |
| `0.45` | `0.137707` | `0.137702` | `0.999415` |
| `0.5` | `0.112477` | `0.112472` | `0.999337` |

## Reproduce

```bash
.Agent/run-tools/kimi_joint_intermediate_scalar_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp96-intermediate-scalar-oracle-smoke64/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp96-intermediate-scalar-oracle-smoke64/report.md --keep-fracs 0.35,0.4,0.45,0.5 --max-records-per-prompt 64 --torch-threads 8
```
