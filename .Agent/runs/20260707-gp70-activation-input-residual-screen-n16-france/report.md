# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-07T15:06:49+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp70-activation-input-residual-screen-n16-france/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp70-activation-input-residual-screen-n16-france/act/activations.f32`
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
| `down:aw_mse:bits1:block256` | `24` | `0.3091` | `0.498586` | `0.506834` | `0.0427402` | `0.995317` |
| `down:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3229` | `0.358662` | `0.417127` | `0.0269555` | `0.643738` |
| `down:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.3436` | `0.297049` | `0.351045` | `0.0218028` | `0.523757` |
| `down:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.3782` | `0.234399` | `0.281071` | `0.0169393` | `0.396246` |
| `gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.593778` | `0.623770` | `0.102318` | `1.21561` |
| `gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.545335` | `0.580931` | `0.0938765` | `1.20614` |
| `gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.500029` | `0.528231` | `0.0859158` | `1.1678` |
| `gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.437931` | `0.462781` | `0.0749603` | `0.960123` |
| `up:aw_mse:bits1:block256` | `24` | `0.3695` | `0.595762` | `0.623160` | `0.10276` | `1.35713` |
| `up:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.546429` | `0.579281` | `0.0944626` | `1.45386` |
| `up:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.499323` | `0.534195` | `0.086321` | `1.20614` |
| `up:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.439421` | `0.468540` | `0.0757254` | `1.0418` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.769351` | `0.804690` | `0.0270338` | `1.57457` |
| `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.719166` | `0.763279` | `0.0253734` | `1.32189` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.668969` | `0.707214` | `0.0237682` | `1.09025` |
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.594271` | `0.632797` | `0.0211193` | `0.930015` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp70-activation-input-residual-screen-n16-france/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp70-activation-input-residual-screen-n16-france/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260707-gp70-activation-input-residual-screen-n16-france/screen.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260707-gp70-activation-input-residual-screen-n16-france/report.md --bits 1 --blocks 256 --scale-modes aw_mse --keep-input-fracs 0.02,0.05,0.10 --max-records 72 --torch-threads 8
```
