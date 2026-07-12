# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-12T07:21:22+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.f32`
- records_loaded: `48`
- unique tensor experts: `48`

## Gate Result

- target byte ratio: `<= 0.40x`
- max mean rel L2: `<= 0.10`
- passing matvec candidates: `0`
- passing fused candidates: `0`
- advance blockwise low-bit path: `False`

## Matvec Output Error

| candidate | rows | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `down:aw_mse:bits2:block128` | `16` | `0.6182` | `0.447470` | `0.452350` | `0.0498402` | `0.69385` |
| `down:aw_mse:bits2:block256` | `16` | `0.6000` | `0.448876` | `0.457284` | `0.0499992` | `0.809667` |
| `down:aw_mse:bits2:block64` | `16` | `0.6545` | `0.445271` | `0.452899` | `0.049805` | `0.722307` |
| `down:maxabs:bits2:block128` | `16` | `0.6182` | `0.776215` | `0.791151` | `0.0869792` | `1.3868` |
| `down:maxabs:bits2:block256` | `16` | `0.6000` | `0.814648` | `0.837003` | `0.091122` | `1.28491` |
| `down:maxabs:bits2:block64` | `16` | `0.6545` | `0.730950` | `0.743861` | `0.081543` | `1.46815` |
| `gate:aw_mse:bits2:block128` | `16` | `0.7616` | `0.426120` | `0.442957` | `0.109571` | `1.10447` |
| `gate:aw_mse:bits2:block256` | `16` | `0.7392` | `0.429481` | `0.450211` | `0.110539` | `1.06024` |
| `gate:aw_mse:bits2:block64` | `16` | `0.8064` | `0.423395` | `0.448249` | `0.109358` | `0.980083` |
| `gate:maxabs:bits2:block128` | `16` | `0.7616` | `0.746837` | `0.774188` | `0.191535` | `1.66573` |
| `gate:maxabs:bits2:block256` | `16` | `0.7392` | `0.791347` | `0.822082` | `0.202784` | `1.81797` |
| `gate:maxabs:bits2:block64` | `16` | `0.8064` | `0.694038` | `0.721814` | `0.1776` | `1.48749` |
| `up:aw_mse:bits2:block128` | `16` | `0.6939` | `0.424763` | `0.444394` | `0.109001` | `0.92513` |
| `up:aw_mse:bits2:block256` | `16` | `0.6735` | `0.427107` | `0.441856` | `0.109573` | `1.03646` |
| `up:aw_mse:bits2:block64` | `16` | `0.7347` | `0.425776` | `0.435957` | `0.109656` | `0.97663` |
| `up:maxabs:bits2:block128` | `16` | `0.6939` | `0.754309` | `0.785298` | `0.194664` | `1.65892` |
| `up:maxabs:bits2:block256` | `16` | `0.6735` | `0.797009` | `0.836368` | `0.205499` | `1.94986` |
| `up:maxabs:bits2:block64` | `16` | `0.7347` | `0.704399` | `0.744941` | `0.181241` | `1.57798` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits2:block128` | `16` | `0.7247` | `0.579812` | `0.614970` | `0.0307859` | `1.33804` |
| `fused_up_gate:aw_mse:bits2:block256` | `16` | `0.7034` | `0.581987` | `0.633574` | `0.0311798` | `1.19156` |
| `fused_up_gate:aw_mse:bits2:block64` | `16` | `0.7673` | `0.575935` | `0.633592` | `0.0307703` | `1.45033` |
| `fused_up_gate:maxabs:bits2:block128` | `16` | `0.7247` | `1.077378` | `1.184663` | `0.0572167` | `2.92986` |
| `fused_up_gate:maxabs:bits2:block256` | `16` | `0.7034` | `1.092515` | `1.184150` | `0.0588649` | `1.73491` |
| `fused_up_gate:maxabs:bits2:block64` | `16` | `0.7673` | `1.028322` | `1.125518` | `0.0539042` | `1.84515` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-current-goal-2bit-fused-screen/dev_python_reverse-bits2.json --out-md .Agent/runs/20260712-current-goal-2bit-fused-screen/dev_python_reverse-bits2.md --bits 2 --blocks 64,128,256 --scale-modes maxabs,aw_mse --max-records 48 --torch-threads 8
```
