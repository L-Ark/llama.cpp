# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-12T05:45:02+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.f32`
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
| `down:aw_mse:bits1:block256` | `16` | `0.3091` | `0.498162` | `0.502448` | `0.0647476` | `1.10824` |
| `down:aw_mse:bits2:block256` | `16` | `0.6000` | `0.446924` | `0.453731` | `0.0587103` | `0.858395` |
| `down:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3436` | `0.282413` | `0.342376` | `0.033266` | `0.555651` |
| `down:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.3782` | `0.225126` | `0.273773` | `0.0260333` | `0.435595` |
| `gate:aw_mse:bits1:block256` | `16` | `0.3808` | `0.591011` | `0.615154` | `0.152443` | `1.32003` |
| `gate:aw_mse:bits2:block256` | `16` | `0.7392` | `0.427596` | `0.440562` | `0.109864` | `0.915012` |
| `gate:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.4117` | `0.499047` | `0.519848` | `0.129085` | `1.21602` |
| `gate:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4427` | `0.440230` | `0.456074` | `0.113453` | `0.981788` |
| `up:aw_mse:bits1:block256` | `16` | `0.3469` | `0.603281` | `0.622859` | `0.156301` | `1.33947` |
| `up:aw_mse:bits2:block256` | `16` | `0.6735` | `0.427127` | `0.443954` | `0.110292` | `0.97912` |
| `up:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3796` | `0.508876` | `0.523152` | `0.132411` | `1.18436` |
| `up:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4122` | `0.449129` | `0.464097` | `0.116441` | `1.18131` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits1:block256` | `16` | `0.3624` | `0.781699` | `0.813760` | `0.0417491` | `1.70894` |
| `fused_up_gate:aw_mse:bits2:block256` | `16` | `0.7034` | `0.585079` | `0.620837` | `0.0317021` | `1.01361` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3942` | `0.679132` | `0.705427` | `0.0367067` | `1.53387` |
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4261` | `0.607491` | `0.627326` | `0.0328526` | `1.3019` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-activation-aware-next-admission/dev_japan_factual-aw-mse-smoke48.json --out-md .Agent/runs/20260712-activation-aware-next-admission/dev_japan_factual-aw-mse-smoke48.md --bits 1,2 --blocks 256 --scale-modes aw_mse --keep-input-fracs 0.05,0.10 --max-records 48 --torch-threads 8
```
