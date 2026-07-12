# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-12T05:46:03+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.f32`
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
| `down:aw_mse:bits1:block256` | `16` | `0.3091` | `0.499619` | `0.505923` | `0.0552173` | `0.807677` |
| `down:aw_mse:bits2:block256` | `16` | `0.6000` | `0.449277` | `0.459352` | `0.049538` | `0.774187` |
| `down:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3436` | `0.291814` | `0.350008` | `0.0298271` | `0.474492` |
| `down:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.3782` | `0.233521` | `0.284719` | `0.0235069` | `0.348499` |
| `gate:aw_mse:bits1:block256` | `16` | `0.3808` | `0.589540` | `0.611814` | `0.146114` | `1.27994` |
| `gate:aw_mse:bits2:block256` | `16` | `0.7392` | `0.426635` | `0.444956` | `0.104777` | `1.17233` |
| `gate:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.4117` | `0.498975` | `0.529448` | `0.123885` | `1.01411` |
| `gate:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4427` | `0.438476` | `0.461745` | `0.108441` | `0.913203` |
| `up:aw_mse:bits1:block256` | `16` | `0.3469` | `0.604825` | `0.633636` | `0.148725` | `1.32464` |
| `up:aw_mse:bits2:block256` | `16` | `0.6735` | `0.431109` | `0.451011` | `0.105965` | `0.899362` |
| `up:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3796` | `0.512534` | `0.544691` | `0.126093` | `1.05396` |
| `up:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4122` | `0.450265` | `0.473392` | `0.110291` | `1.02959` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits1:block256` | `16` | `0.3624` | `0.770394` | `0.826939` | `0.0368268` | `1.52114` |
| `fused_up_gate:aw_mse:bits2:block256` | `16` | `0.7034` | `0.582113` | `0.645486` | `0.0283121` | `1.68192` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `16` | `0.3942` | `0.671834` | `0.733526` | `0.0325003` | `1.15945` |
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `16` | `0.4261` | `0.597495` | `0.659500` | `0.0290562` | `1.11494` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-activation-aware-next-admission/dev_mixed_summary-aw-mse-smoke48.json --out-md .Agent/runs/20260712-activation-aware-next-admission/dev_mixed_summary-aw-mse-smoke48.md --bits 1,2 --blocks 256 --scale-modes aw_mse --keep-input-fracs 0.05,0.10 --max-records 48 --torch-threads 8
```
