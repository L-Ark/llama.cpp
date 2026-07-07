# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-07T16:25:14+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_mixed_summary/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_mixed_summary/act/activations.f32`
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
| `down:aw_mse:bits1:block256` | `24` | `0.3091` | `0.501344` | `0.506396` | `0.0359407` | `0.807677` |
| `down:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3229` | `0.362857` | `0.417975` | `0.0233839` | `0.597524` |
| `down:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.3436` | `0.301351` | `0.353182` | `0.0193336` | `0.474492` |
| `down:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.3782` | `0.240162` | `0.289019` | `0.0152286` | `0.348499` |
| `gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.594334` | `0.618666` | `0.0994651` | `1.27994` |
| `gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.545788` | `0.569266` | `0.0917273` | `1.29159` |
| `gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.498865` | `0.529448` | `0.0842152` | `1.01411` |
| `gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.441676` | `0.465267` | `0.0737992` | `0.913203` |
| `up:aw_mse:bits1:block256` | `24` | `0.3695` | `0.594070` | `0.633636` | `0.100845` | `1.32464` |
| `up:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.546703` | `0.591509` | `0.0931706` | `1.1369` |
| `up:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.501646` | `0.544691` | `0.0854484` | `1.05396` |
| `up:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.443303` | `0.473392` | `0.0747841` | `1.02959` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits1:block256` | `24` | `0.3695` | `0.769910` | `0.826939` | `0.0243015` | `1.52114` |
| `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `24` | `0.3821` | `0.721379` | `0.793158` | `0.0230107` | `1.41899` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `24` | `0.4010` | `0.667810` | `0.733526` | `0.0214463` | `1.15945` |
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `24` | `0.4326` | `0.599852` | `0.659500` | `0.0191795` | `1.11494` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_mixed_summary/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/dev_mixed_summary/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen/dev_mixed_summary/screen.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen/dev_mixed_summary/report.md --bits 1 --blocks 256 --scale-modes aw_mse --keep-input-fracs 0.02,0.05,0.10 --max-records 72 --torch-threads 8
```
