# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-12T05:47:06+0000`
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
| `down:aw_mse:bits1:block256` | `16` | `0.3091` | `0.501182` | `0.513722` | `0.0551799` | `1.08231` |
| `down:aw_mse:bits2:block256` | `16` | `0.6000` | `0.448876` | `0.457284` | `0.0499992` | `0.809667` |
| `down:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3436` | `0.283288` | `0.337431` | `0.0291997` | `0.443399` |
| `down:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.3782` | `0.222332` | `0.268733` | `0.0223671` | `0.304719` |
| `gate:aw_mse:bits1:block256` | `16` | `0.3808` | `0.592525` | `0.621596` | `0.153227` | `1.40263` |
| `gate:aw_mse:bits2:block256` | `16` | `0.7392` | `0.429481` | `0.450211` | `0.110539` | `1.06024` |
| `gate:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.4117` | `0.504553` | `0.538500` | `0.130636` | `1.0555` |
| `gate:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4427` | `0.448092` | `0.473336` | `0.115517` | `1.05938` |
| `up:aw_mse:bits1:block256` | `16` | `0.3469` | `0.602855` | `0.620655` | `0.156678` | `1.37018` |
| `up:aw_mse:bits2:block256` | `16` | `0.6735` | `0.427107` | `0.441856` | `0.109573` | `1.03646` |
| `up:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3796` | `0.511590` | `0.528367` | `0.132421` | `1.22429` |
| `up:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4122` | `0.452721` | `0.470593` | `0.116991` | `0.981719` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits1:block256` | `16` | `0.3624` | `0.769125` | `0.816864` | `0.0410947` | `1.59656` |
| `fused_up_gate:aw_mse:bits2:block256` | `16` | `0.7034` | `0.581987` | `0.633574` | `0.0311798` | `1.19156` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3942` | `0.670384` | `0.715023` | `0.0361313` | `1.43262` |
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4261` | `0.602344` | `0.630390` | `0.0323944` | `1.39068` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-activation-aware-next-admission/dev_python_reverse-aw-mse-smoke48.json --out-md .Agent/runs/20260712-activation-aware-next-admission/dev_python_reverse-aw-mse-smoke48.md --bits 1,2 --blocks 256 --scale-modes aw_mse --keep-input-fracs 0.05,0.10 --max-records 48 --torch-threads 8
```
