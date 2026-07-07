# Kimi output-subspace oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T17:29:10+0000`
- prompts: `3`
- total activation records: `216`
- ranks: `[1, 2, 3, 4]`

## Summary

| role | rank | rank ratio | groups | mean sum rel L2 | max sum rel L2 | mean row Fro rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| `down` | `1` | `0.1250` | `9` | `0.825773` | `0.897026` | `0.847456` |
| `down` | `2` | `0.2500` | `9` | `0.727538` | `0.879749` | `0.738998` |
| `down` | `3` | `0.3750` | `9` | `0.614090` | `0.823425` | `0.623778` |
| `down` | `4` | `0.5000` | `9` | `0.490611` | `0.628282` | `0.509238` |
| `fused_up_gate` | `1` | `0.1250` | `9` | `0.916734` | `0.999286` | `0.920679` |
| `fused_up_gate` | `2` | `0.2500` | `9` | `0.846610` | `0.966139` | `0.838878` |
| `fused_up_gate` | `3` | `0.3750` | `9` | `0.728007` | `0.873224` | `0.754610` |
| `fused_up_gate` | `4` | `0.5000` | `9` | `0.670923` | `0.840100` | `0.663399` |

## Decision

Rank <=3 dynamic output-subspace reconstruction does not pass the dev gate for both roles. Do not implement output-subspace runtime kernels as the next primary path.

## Reproduce

```bash
.Agent/run-tools/kimi_output_subspace_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp84-output-subspace-oracle/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp84-output-subspace-oracle/report.md --ranks 1,2,3,4 --max-records-per-prompt 72 --torch-threads 8
```
