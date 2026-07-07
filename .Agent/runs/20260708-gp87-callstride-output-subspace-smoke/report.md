# Kimi output-subspace oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T18:06:09+0000`
- prompts: `1`
- total activation records: `512`
- ranks: `[1, 2, 3, 4]`

## Summary

| role | rank | rank ratio | groups | mean sum rel L2 | max sum rel L2 | mean row Fro rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| `down` | `1` | `0.1250` | `19` | `0.855157` | `0.938100` | `0.869534` |
| `down` | `2` | `0.2500` | `19` | `0.745677` | `0.931850` | `0.758020` |
| `down` | `3` | `0.3750` | `19` | `0.659468` | `0.812333` | `0.653076` |
| `down` | `4` | `0.5000` | `19` | `0.546865` | `0.708816` | `0.545378` |
| `fused_up_gate` | `1` | `0.1304` | `23` | `0.892103` | `0.998401` | `0.919437` |
| `fused_up_gate` | `2` | `0.2609` | `23` | `0.819362` | `0.977782` | `0.837577` |
| `fused_up_gate` | `3` | `0.3913` | `23` | `0.768264` | `0.969907` | `0.750320` |
| `fused_up_gate` | `4` | `0.5217` | `23` | `0.648302` | `0.904567` | `0.645836` |

## Decision

Rank <=3 dynamic output-subspace reconstruction does not pass the dev gate for both roles. Do not implement output-subspace runtime kernels as the next primary path.

## Reproduce

```bash
.Agent/run-tools/kimi_output_subspace_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp87-callstride-activation-smoke/dev_france_regression --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp87-callstride-output-subspace-smoke/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp87-callstride-output-subspace-smoke/report.md --ranks 1,2,3,4 --max-records-per-prompt 512 --torch-threads 8
```
