# Kimi partial-exact contribution oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T17:17:39+0000`
- prompts: `3`
- total activation records: `216`
- max rel L2 gate: `0.1`

## Role Summary

| role | groups | keep | mean rel L2 | p90 rel L2 | max rel L2 | mean exact byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| `down` | `9` | `1` | `0.847724` | `0.896996` | `0.914022` | `0.1250` |
| `down` | `9` | `2` | `0.739492` | `0.818302` | `0.827969` | `0.2500` |
| `down` | `9` | `4` | `0.509682` | `0.638770` | `0.642436` | `0.5000` |
| `down` | `9` | `6` | `0.320483` | `0.429386` | `0.432093` | `0.7500` |
| `down` | `9` | `8` | `0.000000` | `0.000000` | `0.000000` | `1.0000` |
| `fused_up_gate` | `9` | `1` | `0.921727` | `0.925826` | `0.926794` | `0.1250` |
| `fused_up_gate` | `9` | `2` | `0.841827` | `0.848558` | `0.848792` | `0.2500` |
| `fused_up_gate` | `9` | `4` | `0.667950` | `0.676934` | `0.681643` | `0.5000` |
| `fused_up_gate` | `9` | `6` | `0.458450` | `0.470295` | `0.470987` | `0.7500` |
| `fused_up_gate` | `9` | `8` | `0.000000` | `0.000000` | `0.000000` | `1.0000` |

## Exact Experts Needed For Error Gate

| role | mean count | p90 count | max count | mean exact byte ratio | p90 byte ratio | max byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| `down` | `8.00` | `8.00` | `8.00` | `1.0000` | `1.0000` | `1.0000` |
| `fused_up_gate` | `8.00` | `8.00` | `8.00` | `1.0000` | `1.0000` | `1.0000` |

## Energy Targets

| role | energy | mean count | mean exact byte ratio | p90 exact byte ratio |
|---|---:|---:|---:|---:|
| `down` | `0.9` | `6.22` | `0.7778` | `0.8750` |
| `down` | `0.95` | `7.22` | `0.9028` | `1.0000` |
| `down` | `0.99` | `8.00` | `1.0000` | `1.0000` |
| `fused_up_gate` | `0.9` | `7.78` | `0.9722` | `1.0000` |
| `fused_up_gate` | `0.95` | `8.00` | `1.0000` | `1.0000` |
| `fused_up_gate` | `0.99` | `8.00` | `1.0000` | `1.0000` |

## Decision

The oracle requires too many exact active expert contributions to meet the rel L2 gate inside the 0.30x-0.40x byte budget. Do not implement partial-exact topK retention as a primary runtime path.

## Reproduce

```bash
.Agent/run-tools/kimi_partial_exact_contribution_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp82-partial-exact-contribution-oracle/summary.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp82-partial-exact-contribution-oracle/summary.md --max-records-per-prompt 72 --torch-threads 8
```
