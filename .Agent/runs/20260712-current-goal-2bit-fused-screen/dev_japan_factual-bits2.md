# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-12T07:16:38+0000`
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
| `down:aw_mse:bits2:block128` | `16` | `0.6182` | `0.445050` | `0.450738` | `0.0586852` | `0.862768` |
| `down:aw_mse:bits2:block256` | `16` | `0.6000` | `0.446924` | `0.453731` | `0.0587103` | `0.858395` |
| `down:aw_mse:bits2:block64` | `16` | `0.6545` | `0.444655` | `0.450826` | `0.058635` | `0.914162` |
| `down:maxabs:bits2:block128` | `16` | `0.6182` | `0.774511` | `0.787276` | `0.102098` | `1.44709` |
| `down:maxabs:bits2:block256` | `16` | `0.6000` | `0.812411` | `0.831655` | `0.107333` | `1.4752` |
| `down:maxabs:bits2:block64` | `16` | `0.6545` | `0.727618` | `0.740241` | `0.0961561` | `1.37283` |
| `gate:aw_mse:bits2:block128` | `16` | `0.7616` | `0.426780` | `0.447656` | `0.109922` | `1.02704` |
| `gate:aw_mse:bits2:block256` | `16` | `0.7392` | `0.427596` | `0.440562` | `0.109864` | `0.915012` |
| `gate:aw_mse:bits2:block64` | `16` | `0.8064` | `0.422513` | `0.446222` | `0.109232` | `1.15326` |
| `gate:maxabs:bits2:block128` | `16` | `0.7616` | `0.750907` | `0.800556` | `0.192906` | `1.67762` |
| `gate:maxabs:bits2:block256` | `16` | `0.7392` | `0.791681` | `0.831778` | `0.203467` | `1.85665` |
| `gate:maxabs:bits2:block64` | `16` | `0.8064` | `0.698331` | `0.736789` | `0.179308` | `1.49797` |
| `up:aw_mse:bits2:block128` | `16` | `0.6939` | `0.426365` | `0.441581` | `0.109562` | `0.951958` |
| `up:aw_mse:bits2:block256` | `16` | `0.6735` | `0.427127` | `0.443954` | `0.110292` | `0.97912` |
| `up:aw_mse:bits2:block64` | `16` | `0.7347` | `0.427094` | `0.441868` | `0.110095` | `1.04308` |
| `up:maxabs:bits2:block128` | `16` | `0.6939` | `0.745716` | `0.763032` | `0.193764` | `1.52418` |
| `up:maxabs:bits2:block256` | `16` | `0.6735` | `0.789929` | `0.809486` | `0.204326` | `1.81172` |
| `up:maxabs:bits2:block64` | `16` | `0.7347` | `0.694486` | `0.722838` | `0.180098` | `1.64817` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits2:block128` | `16` | `0.7247` | `0.584991` | `0.621643` | `0.0316969` | `1.02146` |
| `fused_up_gate:aw_mse:bits2:block256` | `16` | `0.7034` | `0.585079` | `0.620837` | `0.0317021` | `1.01361` |
| `fused_up_gate:aw_mse:bits2:block64` | `16` | `0.7673` | `0.575900` | `0.604534` | `0.0312797` | `0.844047` |
| `fused_up_gate:maxabs:bits2:block128` | `16` | `0.7247` | `1.098310` | `1.187158` | `0.0583159` | `1.72508` |
| `fused_up_gate:maxabs:bits2:block256` | `16` | `0.7034` | `1.126448` | `1.213644` | `0.0605145` | `1.92952` |
| `fused_up_gate:maxabs:bits2:block64` | `16` | `0.7673` | `1.038414` | `1.125029` | `0.0551223` | `1.58016` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-current-goal-2bit-fused-screen/dev_japan_factual-bits2.json --out-md .Agent/runs/20260712-current-goal-2bit-fused-screen/dev_japan_factual-bits2.md --bits 2 --blocks 64,128,256 --scale-modes maxabs,aw_mse --max-records 48 --torch-threads 8
```
