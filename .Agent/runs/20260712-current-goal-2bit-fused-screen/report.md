# Kimi 2-bit Fused Up/Gate Screen

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `01d321dcb`

## Purpose

The previous non-destructive low-byte closure showed that the existing 1-bit
blockwise/activation-aware family fails mainly because fused up/gate output
error is too high. This screen checks whether moving fused up/gate to 2-bit
blockwise compression gives a meaningful quality/byte tradeoff before any
runtime work.

This is dev-only and non-destructive:

- no runtime path changed;
- no model or pack was downloaded;
- no file was deleted;
- no SOTA claim is made.

## Inputs

Activation corpus:

- `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual`
- `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary`
- `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse`

Screen config:

- bits: `2`;
- blocks: `64,128,256`;
- scale modes: `maxabs,aw_mse`;
- max records per prompt: `48`;
- torch threads: `8`;
- inventory:
  `.Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv`;
- library:
  `build-cuda-batch/bin/libggml-base.so`.

## Artifacts

- `.Agent/runs/20260712-current-goal-2bit-fused-screen/dev_japan_factual-bits2.json`
- `.Agent/runs/20260712-current-goal-2bit-fused-screen/dev_japan_factual-bits2.md`
- `.Agent/runs/20260712-current-goal-2bit-fused-screen/dev_mixed_summary-bits2.json`
- `.Agent/runs/20260712-current-goal-2bit-fused-screen/dev_mixed_summary-bits2.md`
- `.Agent/runs/20260712-current-goal-2bit-fused-screen/dev_python_reverse-bits2.json`
- `.Agent/runs/20260712-current-goal-2bit-fused-screen/dev_python_reverse-bits2.md`
- `.Agent/runs/20260712-current-goal-2bit-fused-screen/summary.json`
- `.Agent/runs/20260712-current-goal-2bit-fused-screen/summary.md`

## Result

Aggregated fused up/gate result:

| candidate | prompts | pairs | byte ratio | mean rel-L2 | max rel-L2 |
|---|---:|---:|---:|---:|---:|
| `aw_mse:bits2:block64` | `3` | `48` | `0.7673x` | `0.575718` | `0.640693` |
| `aw_mse:bits2:block128` | `3` | `48` | `0.7247x` | `0.581918` | `0.640225` |
| `aw_mse:bits2:block256` | `3` | `48` | `0.7034x` | `0.583060` | `0.645486` |
| `maxabs:bits2:block64` | `3` | `48` | `0.7673x` | `1.032088` | `1.125518` |
| `maxabs:bits2:block128` | `3` | `48` | `0.7247x` | `1.084651` | `1.187158` |
| `maxabs:bits2:block256` | `3` | `48` | `0.7034x` | `1.112494` | `1.266916` |

Comparison against the previous 1-bit screen:

- best 1-bit fused up/gate candidate:
  - `aw_mse_keep_input0p1:bits1:block256`;
  - byte ratio `0.4326x`;
  - mean rel-L2 `0.598174`;
- best 2-bit fused up/gate candidate:
  - `aw_mse:bits2:block64`;
  - byte ratio `0.7673x`;
  - mean rel-L2 `0.575718`.

2-bit reduces fused up/gate mean rel-L2 by only about `0.0225` absolute, while
raising moved bytes by about `77%` relative to the best 1-bit candidate. It also
misses the short-term `0.50x` movement target by a wide margin.

## Decision

Reject 2-bit blockwise fused up/gate compression as the next runtime path.

Reasons:

- byte ratio is `0.70x-0.77x`, above the required all-role `~0.50x` target for
  the `2 tok/s` milestone;
- fused up/gate mean rel-L2 remains around `0.58`, far above the `<=0.10`
  output-error gate;
- 2-bit gives only a small error improvement over the already rejected 1-bit
  keep-input candidate;
- `maxabs` 2-bit is worse than AW-MSE and should not be pursued.

This closes another simple extension of the blockwise low-bit family. Future
non-destructive work must use a qualitatively different representation, for
example an activation-output trained representation, a structured factorization
with much lower fused up/gate error, or the explicitly approved complete-model
`i1-IQ1_S` smoke.

## Reproduce

```bash
cd /root/lfz/llama.cpp-vendor-kimi
OUT=.Agent/runs/20260712-current-goal-2bit-fused-screen
mkdir -p "$OUT"

for P in dev_japan_factual dev_mixed_summary dev_python_reverse; do
  ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/$P
  python3 .Agent/run-tools/kimi_activation_output_compression_screen.py \
    --activation-csv "$ROOT/act/activations.csv" \
    --activation-bin "$ROOT/act/activations.f32" \
    --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
    --libggml-base build-cuda-batch/bin/libggml-base.so \
    --out-json "$OUT/$P-bits2.json" \
    --out-md "$OUT/$P-bits2.md" \
    --bits 2 \
    --blocks 64,128,256 \
    --scale-modes maxabs,aw_mse \
    --max-records 48 \
    --torch-threads 8
done

python3 /tmp/summarize_2bit_fused.py "$OUT"
```
