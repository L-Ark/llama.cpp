# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-07T16:24:25+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_python_reverse/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_python_reverse/act/activations.f32`
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
| `down:aw_mse:bits1:block256` | `24` | `0.3091` | `0.498145` | `0.503601` | `0.035483` | `1.08231` |
| `down:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3229` | `0.352920` | `0.407149` | `0.0231171` | `0.630824` |
| `down:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.3436` | `0.291744` | `0.347193` | `0.0186722` | `0.443399` |
| `down:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.3782` | `0.228214` | `0.284013` | `0.0142758` | `0.304719` |
| `gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.595993` | `0.621596` | `0.102251` | `1.40263` |
| `gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.550171` | `0.578022` | `0.0944799` | `1.38077` |
| `gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.505936` | `0.538500` | `0.0871152` | `1.0555` |
| `gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.444554` | `0.473336` | `0.076744` | `1.05938` |
| `up:aw_mse:bits1:block256` | `24` | `0.3695` | `0.596324` | `0.622302` | `0.104238` | `1.37018` |
| `up:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.550590` | `0.571047` | `0.0961671` | `1.25415` |
| `up:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.503721` | `0.528367` | `0.0880298` | `1.22429` |
| `up:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.444083` | `0.470593` | `0.077702` | `0.981719` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.767664` | `0.816864` | `0.0269629` | `1.59656` |
| `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.719618` | `0.758893` | `0.0252902` | `1.58383` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.669909` | `0.715023` | `0.0237124` | `1.43262` |
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.596316` | `0.630390` | `0.021249` | `1.39068` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_python_reverse/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_python_reverse/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen/dev_python_reverse/screen.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen/dev_python_reverse/report.md --bits 1 --blocks 256 --scale-modes aw_mse --keep-input-fracs 0.02,0.05,0.10 --max-records 72 --torch-threads 8
```
