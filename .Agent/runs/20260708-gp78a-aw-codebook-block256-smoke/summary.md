# Kimi GP78a aw_codebook block256 smoke

This is a dev-only offline representation screen. It does not change runtime behavior or claim SOTA.

- remote root: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp78a-aw-codebook-block256-smoke`
- prompt evaluated: `dev_japan_factual`
- candidate: `aw_codebook:bits1:block256`
- records loaded: `72`
- unique tensor experts: `72`
- held-out prompts used: `0`

## Result

| candidate | byte ratio | mean rel L2 | max rel L2 | decision |
|---|---:|---:|---:|---|
| `down:aw_codebook:bits1:block256` | `0.3273` | `0.571013` | `0.578527` | reject |
| `gate:aw_codebook:bits1:block256` | `0.3912` | `0.592142` | `0.613728` | reject |
| `up:aw_codebook:bits1:block256` | `0.3912` | `0.595450` | `0.617253` | reject |
| `fused_up_gate:aw_codebook:bits1:block256` | `0.3912` | `0.773949` | `0.814748` | reject |

## Decision

Stop the `aw_codebook:bits1:block256` path early. It reaches the byte budget but makes fused up/gate error worse than GP77's best fused up/gate candidate (`0.773949` vs `0.598174` mean rel L2), so running the remaining dev prompts would not change the implementation decision.

The bottleneck is representation quality, not scheduling or byte ratio. Do not write runtime kernels for this codebook family unless a new variant shows an order-of-magnitude lower fused up/gate activation-output error.

## Reproduce

```bash
cd /root/lfz/tmp/kimi-stage2m-align
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp78a-aw-codebook-block256-smoke
rm -rf "$ROOT"
mkdir -p "$ROOT/dev_japan_factual"
python3 .Agent/run-tools/kimi_activation_output_compression_screen.py \
  --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.csv \
  --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.f32 \
  --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
  --libggml-base build-cuda-batch/bin/libggml-base.so \
  --out-json "$ROOT/dev_japan_factual/screen.json" \
  --out-md "$ROOT/dev_japan_factual/report.md" \
  --bits 1 \
  --blocks 256 \
  --scale-modes aw_codebook \
  --max-records 72 \
  --torch-threads 8
```
