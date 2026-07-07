# Kimi layer MoE output subspace oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T19:42:33+0000`
- prompts: `3`
- groups: `24`
- ranks: `[0, 1, 2, 4]`

## Leave-One-Prompt-Out Summary

| rank | rows | mean rel L2 | max rel L2 |
|---:|---:|---:|---:|
| `0` | `24` | `1.212535` | `1.711725` |
| `1` | `24` | `1.194453` | `1.710250` |
| `2` | `24` | `1.194453` | `1.710250` |
| `4` | `24` | `1.194453` | `1.710250` |

## Worst Layer Rows

| layer | rank | rows | mean rel L2 | max rel L2 |
|---|---:|---:|---:|---:|
| `blk.56` | `0` | `3` | `1.312402` | `1.711725` |
| `blk.35` | `0` | `3` | `1.296180` | `1.493571` |
| `blk.49` | `0` | `3` | `1.262699` | `1.387242` |
| `blk.42` | `0` | `3` | `1.253511` | `1.402318` |
| `blk.28` | `0` | `3` | `1.244880` | `1.455049` |
| `blk.21` | `0` | `3` | `1.198206` | `1.359525` |
| `blk.12` | `0` | `3` | `1.192494` | `1.300121` |
| `blk.60` | `0` | `3` | `0.939910` | `1.072112` |
| `blk.56` | `1` | `3` | `1.293650` | `1.710250` |
| `blk.35` | `1` | `3` | `1.248659` | `1.441919` |
| `blk.42` | `1` | `3` | `1.238382` | `1.392813` |
| `blk.28` | `1` | `3` | `1.236412` | `1.453501` |
| `blk.49` | `1` | `3` | `1.233149` | `1.351921` |
| `blk.21` | `1` | `3` | `1.194629` | `1.354727` |
| `blk.12` | `1` | `3` | `1.179497` | `1.276713` |
| `blk.60` | `1` | `3` | `0.931249` | `1.058098` |
| `blk.56` | `2` | `3` | `1.293650` | `1.710250` |
| `blk.35` | `2` | `3` | `1.248659` | `1.441919` |
| `blk.42` | `2` | `3` | `1.238382` | `1.392813` |
| `blk.28` | `2` | `3` | `1.236412` | `1.453501` |
| `blk.49` | `2` | `3` | `1.233149` | `1.351921` |
| `blk.21` | `2` | `3` | `1.194629` | `1.354727` |
| `blk.12` | `2` | `3` | `1.179497` | `1.276713` |
| `blk.60` | `2` | `3` | `0.931249` | `1.058098` |
| `blk.56` | `4` | `3` | `1.293650` | `1.710250` |
| `blk.35` | `4` | `3` | `1.248659` | `1.441919` |
| `blk.42` | `4` | `3` | `1.238382` | `1.392813` |
| `blk.28` | `4` | `3` | `1.236412` | `1.453501` |
| `blk.49` | `4` | `3` | `1.233149` | `1.351921` |
| `blk.21` | `4` | `3` | `1.194629` | `1.354727` |

## Decision

Reject tiny layer-output subspace as the next primary path: rank 4 leave-one-prompt-out mean rel L2 is 1.194453, above the 0.1 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_layer_output_subspace_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp97-layer-output-subspace-oracle-smoke64/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp97-layer-output-subspace-oracle-smoke64/report.md --ranks 0,1,2,4 --max-down-records-per-prompt 64 --torch-threads 8
```
