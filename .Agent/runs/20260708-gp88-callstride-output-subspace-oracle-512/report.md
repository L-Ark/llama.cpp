# Kimi output-subspace oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T18:33:01+0000`
- prompts: `3`
- total activation records: `1536`
- ranks: `[1, 2, 3, 4]`

## Summary

| role | rank | rank ratio | groups | mean sum rel L2 | max sum rel L2 | mean row Fro rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| `down` | `1` | `0.1250` | `57` | `0.860694` | `0.996028` | `0.868387` |
| `down` | `2` | `0.2500` | `57` | `0.755510` | `0.913147` | `0.758699` |
| `down` | `3` | `0.3750` | `57` | `0.649566` | `0.872197` | `0.652107` |
| `down` | `4` | `0.5000` | `57` | `0.555836` | `0.786431` | `0.551791` |
| `fused_up_gate` | `1` | `0.1304` | `69` | `0.925092` | `0.999989` | `0.919424` |
| `fused_up_gate` | `2` | `0.2609` | `69` | `0.815937` | `0.990440` | `0.837297` |
| `fused_up_gate` | `3` | `0.3913` | `69` | `0.732757` | `0.964901` | `0.749547` |
| `fused_up_gate` | `4` | `0.5217` | `69` | `0.636106` | `0.919917` | `0.644597` |

## Decision

Rank <=3 dynamic output-subspace reconstruction does not pass the dev gate for both roles. Do not implement output-subspace runtime kernels as the next primary path.

## Reproduce

```bash
.Agent/run-tools/kimi_output_subspace_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-output-subspace-oracle-512/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-output-subspace-oracle-512/report.md --ranks 1,2,3,4 --max-records-per-prompt 512 --torch-threads 8
```
