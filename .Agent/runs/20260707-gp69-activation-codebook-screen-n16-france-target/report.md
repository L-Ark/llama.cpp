# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-07T15:01:10+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp69-activation-codebook-screen-n16-france-target/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp69-activation-codebook-screen-n16-france-target/act/activations.f32`
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
| `down:aw_codebook:bits1:block128` | `24` | `0.3636` | `0.568261` | `0.579050` | `0.0490165` | `1.06301` |
| `down:aw_codebook:bits1:block256` | `24` | `0.3273` | `0.571083` | `0.582493` | `0.0492555` | `1.06557` |
| `gate:aw_codebook:bits1:block128` | `24` | `0.4347` | `0.588626` | `0.615065` | `0.101209` | `1.18512` |
| `gate:aw_codebook:bits1:block256` | `24` | `0.3912` | `0.591504` | `0.620760` | `0.101659` | `1.25257` |
| `up:aw_codebook:bits1:block128` | `24` | `0.4347` | `0.590740` | `0.619448` | `0.101822` | `1.23309` |
| `up:aw_codebook:bits1:block256` | `24` | `0.3912` | `0.593947` | `0.620593` | `0.102377` | `1.34151` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_codebook:bits1:block128` | `24` | `0.4347` | `0.763645` | `0.796962` | `0.0267488` | `1.64147` |
| `fused_up_gate:aw_codebook:bits1:block256` | `24` | `0.3912` | `0.767573` | `0.799110` | `0.0268667` | `1.5257` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp69-activation-codebook-screen-n16-france-target/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp69-activation-codebook-screen-n16-france-target/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260707-gp69-activation-codebook-screen-n16-france-target/screen.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260707-gp69-activation-codebook-screen-n16-france-target/report.md --bits 1 --blocks 128,256 --scale-modes aw_codebook --max-records 72 --torch-threads 8
```
