# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-07T16:37:41+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.f32`
- records_loaded: `72`
- unique tensor experts: `72`

## Gate Result

- target byte ratio: `<= 0.40x`
- max mean rel L2: `<= 0.10`
- passing matvec candidates: `0`
- passing fused candidates: `0`
- advance blockwise low-bit path: `False`

## Matvec Output Error

| candidate | rows | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `down:aw_codebook:bits1:block256` | `24` | `0.3273` | `0.571013` | `0.578527` | `0.0489232` | `1.22657` |
| `gate:aw_codebook:bits1:block256` | `24` | `0.3912` | `0.592142` | `0.613728` | `0.102035` | `1.31754` |
| `up:aw_codebook:bits1:block256` | `24` | `0.3912` | `0.595450` | `0.617253` | `0.104344` | `1.34325` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_codebook:bits1:block256` | `24` | `0.3912` | `0.773949` | `0.814748` | `0.0274149` | `1.6834` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp78a-aw-codebook-block256-smoke/dev_japan_factual/screen.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp78a-aw-codebook-block256-smoke/dev_japan_factual/report.md --bits 1 --blocks 256 --scale-modes aw_codebook --max-records 72 --torch-threads 8
```
