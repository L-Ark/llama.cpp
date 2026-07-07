# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-07T16:23:38+0000`
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
| `down:aw_mse:bits1:block256` | `24` | `0.3091` | `0.498341` | `0.505538` | `0.0423581` | `1.10824` |
| `down:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3229` | `0.358507` | `0.415322` | `0.0265273` | `0.726583` |
| `down:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.3436` | `0.298594` | `0.353800` | `0.021698` | `0.555651` |
| `down:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.3782` | `0.236081` | `0.288870` | `0.0169733` | `0.435595` |
| `gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.595660` | `0.615154` | `0.102535` | `1.32003` |
| `gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.545449` | `0.568452` | `0.0945591` | `1.26953` |
| `gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.500187` | `0.519848` | `0.0867376` | `1.21602` |
| `gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.437701` | `0.459150` | `0.0760596` | `0.981788` |
| `up:aw_mse:bits1:block256` | `24` | `0.3695` | `0.598225` | `0.617493` | `0.104715` | `1.33947` |
| `up:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.548852` | `0.571385` | `0.0968993` | `1.1822` |
| `up:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.501102` | `0.521471` | `0.0885961` | `1.18436` |
| `up:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.439723` | `0.461914` | `0.0778434` | `1.18131` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.776314` | `0.813760` | `0.0275098` | `1.70894` |
| `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.724481` | `0.767928` | `0.0259963` | `1.74457` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.673145` | `0.706927` | `0.0241872` | `1.53387` |
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.598353` | `0.629257` | `0.0216409` | `1.3019` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_japan_factual/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen/dev_japan_factual/screen.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen/dev_japan_factual/report.md --bits 1 --blocks 256 --scale-modes aw_mse --keep-input-fracs 0.02,0.05,0.10 --max-records 72 --torch-threads 8
```
