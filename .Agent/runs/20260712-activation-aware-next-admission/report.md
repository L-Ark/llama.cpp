# Kimi activation-aware next admission

generated_at: `2026-07-12T05:51:14+0000`

dev-only non-destructive smoke; no runtime change; no SOTA claim

screen config: `bits=1,2 blocks=256 scale=aw_mse keep_input=0.05,0.10 max_records=48`

## Gate

- primary compressed-runtime target: `<=0.40x` moved bytes and `mean rel-L2 <=0.10`
- short-term 2 tok/s warning target: `<=0.50x`; still requires `mean rel-L2 <=0.10` before runtime work
- prompts: `dev_japan_factual`, `dev_mixed_summary`, `dev_python_reverse`

## Summary

| role | byte target | prompts | mean best rel-L2 | max best rel-L2 | mean ratio | pass all |
|---|---:|---:|---:|---:|---:|---|
| `down` | `0.40x` | `3` | `0.226993` | `0.233521` | `0.3782` | `False` |
| `down` | `0.50x` | `3` | `0.226993` | `0.233521` | `0.3782` | `False` |
| `gate` | `0.40x` | `3` | `0.591026` | `0.592525` | `0.3808` | `False` |
| `gate` | `0.50x` | `3` | `0.442266` | `0.448092` | `0.4427` | `False` |
| `up` | `0.40x` | `3` | `0.511000` | `0.512534` | `0.3796` | `False` |
| `up` | `0.50x` | `3` | `0.450705` | `0.452721` | `0.4122` | `False` |
| `fused_up_gate` | `0.40x` | `3` | `0.673783` | `0.679132` | `0.3942` | `False` |
| `fused_up_gate` | `0.50x` | `3` | `0.602443` | `0.607491` | `0.4261` | `False` |

## Per-Prompt Best Candidates

| prompt | role | target | candidate | ratio | mean rel-L2 | max rel-L2 |
|---|---|---:|---|---:|---:|---:|
| `dev_japan_factual` | `down` | `0.40x` | `down:aw_mse_keep_input0p1:bits1:block256` | `0.3782` | `0.225126` | `0.273773` |
| `dev_japan_factual` | `down` | `0.50x` | `down:aw_mse_keep_input0p1:bits1:block256` | `0.3782` | `0.225126` | `0.273773` |
| `dev_japan_factual` | `gate` | `0.40x` | `gate:aw_mse:bits1:block256` | `0.3808` | `0.591011` | `0.615154` |
| `dev_japan_factual` | `gate` | `0.50x` | `gate:aw_mse_keep_input0p1:bits1:block256` | `0.4427` | `0.440230` | `0.456074` |
| `dev_japan_factual` | `up` | `0.40x` | `up:aw_mse_keep_input0p05:bits1:block256` | `0.3796` | `0.508876` | `0.523152` |
| `dev_japan_factual` | `up` | `0.50x` | `up:aw_mse_keep_input0p1:bits1:block256` | `0.4122` | `0.449129` | `0.464097` |
| `dev_japan_factual` | `fused_up_gate` | `0.40x` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3942` | `0.679132` | `0.705427` |
| `dev_japan_factual` | `fused_up_gate` | `0.50x` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.4261` | `0.607491` | `0.627326` |
| `dev_mixed_summary` | `down` | `0.40x` | `down:aw_mse_keep_input0p1:bits1:block256` | `0.3782` | `0.233521` | `0.284719` |
| `dev_mixed_summary` | `down` | `0.50x` | `down:aw_mse_keep_input0p1:bits1:block256` | `0.3782` | `0.233521` | `0.284719` |
| `dev_mixed_summary` | `gate` | `0.40x` | `gate:aw_mse:bits1:block256` | `0.3808` | `0.589540` | `0.611814` |
| `dev_mixed_summary` | `gate` | `0.50x` | `gate:aw_mse_keep_input0p1:bits1:block256` | `0.4427` | `0.438476` | `0.461745` |
| `dev_mixed_summary` | `up` | `0.40x` | `up:aw_mse_keep_input0p05:bits1:block256` | `0.3796` | `0.512534` | `0.544691` |
| `dev_mixed_summary` | `up` | `0.50x` | `up:aw_mse_keep_input0p1:bits1:block256` | `0.4122` | `0.450265` | `0.473392` |
| `dev_mixed_summary` | `fused_up_gate` | `0.40x` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3942` | `0.671834` | `0.733526` |
| `dev_mixed_summary` | `fused_up_gate` | `0.50x` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.4261` | `0.597495` | `0.659500` |
| `dev_python_reverse` | `down` | `0.40x` | `down:aw_mse_keep_input0p1:bits1:block256` | `0.3782` | `0.222332` | `0.268733` |
| `dev_python_reverse` | `down` | `0.50x` | `down:aw_mse_keep_input0p1:bits1:block256` | `0.3782` | `0.222332` | `0.268733` |
| `dev_python_reverse` | `gate` | `0.40x` | `gate:aw_mse:bits1:block256` | `0.3808` | `0.592525` | `0.621596` |
| `dev_python_reverse` | `gate` | `0.50x` | `gate:aw_mse_keep_input0p1:bits1:block256` | `0.4427` | `0.448092` | `0.473336` |
| `dev_python_reverse` | `up` | `0.40x` | `up:aw_mse_keep_input0p05:bits1:block256` | `0.3796` | `0.511590` | `0.528367` |
| `dev_python_reverse` | `up` | `0.50x` | `up:aw_mse_keep_input0p1:bits1:block256` | `0.4122` | `0.452721` | `0.470593` |
| `dev_python_reverse` | `fused_up_gate` | `0.40x` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3942` | `0.670384` | `0.715023` |
| `dev_python_reverse` | `fused_up_gate` | `0.50x` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.4261` | `0.602344` | `0.630390` |

## Decision

reject activation-aware AW-MSE/input-correction as next primary runtime path: no role passes rel-L2<=0.10 at <=0.40x or <=0.50x across dev prompts; fused up/gate remains far above the quality gate.

This reinforces earlier rejections of exact input-channel keep, partial exact contribution, output subspace, joint intermediate keep/scalar, and cluster-base residual candidates. The next runtime implementation should not be an activation-aware AW-MSE/input-correction compressed expert path.

## Reproduce

```bash
cd /root/lfz/llama.cpp-vendor-kimi
OUT=.Agent/runs/20260712-activation-aware-next-admission
for P in dev_japan_factual dev_mixed_summary dev_python_reverse; do
  ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/$P
  timeout 600 python3 .Agent/run-tools/kimi_activation_output_compression_screen.py \
    --activation-csv "$ROOT/act/activations.csv" \
    --activation-bin "$ROOT/act/activations.f32" \
    --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
    --libggml-base build-cuda-batch/bin/libggml-base.so \
    --out-json "$OUT/$P-aw-mse-smoke48.json" \
    --out-md "$OUT/$P-aw-mse-smoke48.md" \
    --bits 1,2 --blocks 256 --scale-modes aw_mse \
    --keep-input-fracs 0.05,0.10 --max-records 48 --torch-threads 8
done
python3 .Agent/run-tools/kimi_activation_admission_summary.py
```
