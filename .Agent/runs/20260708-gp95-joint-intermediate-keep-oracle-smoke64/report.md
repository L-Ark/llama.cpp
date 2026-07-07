# Kimi joint intermediate-dimension keep oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T19:32:34+0000`
- prompts: `3`
- keep fractions: `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]`

## Aggregate

| keep | byte ratio | row mean rel L2 | row max rel L2 | group mean rel L2 | group max rel L2 |
|---:|---:|---:|---:|---:|---:|
| `0.05` | `0.0500` | `0.607228` | `0.812293` | `0.585457` | `0.651932` |
| `0.1` | `0.1000` | `0.487549` | `0.702728` | `0.466323` | `0.532228` |
| `0.2` | `0.2000` | `0.338423` | `0.526710` | `0.319685` | `0.379036` |
| `0.3` | `0.3000` | `0.240277` | `0.392086` | `0.225629` | `0.275613` |
| `0.4` | `0.4000` | `0.168237` | `0.287683` | `0.157275` | `0.198107` |
| `0.5` | `0.5000` | `0.113524` | `0.200821` | `0.105683` | `0.138145` |

## Decision

Reject joint intermediate-dimension partial reads as the next primary runtime path: at keep <= 0.4, grouped down-output mean rel L2 is 0.157275, above the 0.1 gate.

## Prompt Summaries

### dev_japan_factual

- records: `64`
- groups: `8`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.05` | `0.591218` | `0.571221` |
| `0.1` | `0.470430` | `0.449375` |
| `0.2` | `0.323554` | `0.304031` |
| `0.3` | `0.228790` | `0.213636` |
| `0.4` | `0.159363` | `0.147883` |
| `0.5` | `0.107198` | `0.099021` |

### dev_mixed_summary

- records: `64`
- groups: `8`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.05` | `0.608064` | `0.588558` |
| `0.1` | `0.489435` | `0.470958` |
| `0.2` | `0.340345` | `0.321884` |
| `0.3` | `0.241663` | `0.226580` |
| `0.4` | `0.169128` | `0.157720` |
| `0.5` | `0.114115` | `0.105551` |

### dev_python_reverse

- records: `64`
- groups: `8`

| keep | row mean rel L2 | group mean rel L2 |
|---:|---:|---:|
| `0.05` | `0.622403` | `0.596593` |
| `0.1` | `0.502782` | `0.478636` |
| `0.2` | `0.351369` | `0.333142` |
| `0.3` | `0.250378` | `0.236672` |
| `0.4` | `0.176221` | `0.166223` |
| `0.5` | `0.119260` | `0.112477` |

## Reproduce

```bash
.Agent/run-tools/kimi_joint_intermediate_keep_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp95-joint-intermediate-keep-oracle-smoke64/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp95-joint-intermediate-keep-oracle-smoke64/report.md --keep-fracs 0.05,0.1,0.2,0.3,0.4,0.5 --max-records-per-prompt 64 --torch-threads 8
```
