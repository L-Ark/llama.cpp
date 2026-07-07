# Kimi output-subspace oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T17:58:17+0000`
- prompts: `3`
- total activation records: `1536`
- ranks: `[1, 2, 3, 4]`

## Summary

| role | rank | rank ratio | groups | mean sum rel L2 | max sum rel L2 | mean row Fro rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| `down` | `1` | `0.9296` | `405` | `0.079175` | `0.804627` | `0.083017` |
| `down` | `2` | `1.0000` | `405` | `0.000000` | `0.000000` | `0.000000` |
| `down` | `3` | `1.0000` | `405` | `0.000000` | `0.000000` | `0.000000` |
| `down` | `4` | `1.0000` | `405` | `0.000000` | `0.000000` | `0.000000` |

## Decision

Rank <=3 dynamic output-subspace reconstruction does not pass the dev gate for both roles. Do not implement output-subspace runtime kernels as the next primary path.

## Reproduce

```bash
.Agent/run-tools/kimi_output_subspace_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp85-strided-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp85-strided-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp85-strided-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp86-output-subspace-gp85-corpus/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp86-output-subspace-gp85-corpus/report.md --ranks 1,2,3,4 --max-records-per-prompt 512 --torch-threads 8
```
