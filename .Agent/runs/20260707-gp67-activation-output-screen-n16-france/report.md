# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-07T14:32:19+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp67-activation-output-screen-n16-france/act/activations.csv`
- activation_bin: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp67-activation-output-screen-n16-france/act/activations.f32`
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
| `down:bits1:block128` | `24` | `0.3273` | `1.747889` | `1.820015` | `0.154098` | `3.22217` |
| `down:bits1:block256` | `24` | `0.3091` | `1.932912` | `2.023435` | `0.171129` | `3.59344` |
| `down:bits1:block64` | `24` | `0.3636` | `1.551744` | `1.598351` | `0.135838` | `2.93139` |
| `down:bits2:block128` | `24` | `0.6182` | `0.773293` | `0.791669` | `0.0675224` | `1.56371` |
| `down:bits2:block256` | `24` | `0.6000` | `0.809577` | `0.833019` | `0.070927` | `1.54843` |
| `down:bits2:block64` | `24` | `0.6545` | `0.728578` | `0.746158` | `0.0635269` | `1.38078` |
| `gate:bits1:block128` | `24` | `0.3912` | `2.055933` | `2.154627` | `0.350011` | `4.17247` |
| `gate:bits1:block256` | `24` | `0.3695` | `2.263926` | `2.376220` | `0.384487` | `4.71089` |
| `gate:bits1:block64` | `24` | `0.4347` | `1.841133` | `1.939932` | `0.313492` | `4.04098` |
| `gate:bits2:block128` | `24` | `0.7390` | `0.744470` | `0.779613` | `0.126599` | `1.72708` |
| `gate:bits2:block256` | `24` | `0.7173` | `0.785383` | `0.810206` | `0.133475` | `1.85017` |
| `gate:bits2:block64` | `24` | `0.7825` | `0.696058` | `0.728174` | `0.118921` | `1.56997` |
| `up:bits1:block128` | `24` | `0.3912` | `2.070950` | `2.131014` | `0.355489` | `4.61357` |
| `up:bits1:block256` | `24` | `0.3695` | `2.278700` | `2.348801` | `0.39023` | `5.13238` |
| `up:bits1:block64` | `24` | `0.4347` | `1.854208` | `1.915393` | `0.319342` | `4.23826` |
| `up:bits2:block128` | `24` | `0.7390` | `0.750930` | `0.782478` | `0.128186` | `1.58619` |
| `up:bits2:block256` | `24` | `0.7173` | `0.795203` | `0.827649` | `0.135411` | `1.91843` |
| `up:bits2:block64` | `24` | `0.7825` | `0.698469` | `0.734119` | `0.119163` | `1.52259` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:bits1:block128` | `24` | `0.3912` | `7.630776` | `10.149129` | `0.242463` | `17.8918` |
| `fused_up_gate:bits1:block256` | `24` | `0.3695` | `8.975913` | `11.880800` | `0.283112` | `19.4178` |
| `fused_up_gate:bits1:block64` | `24` | `0.4347` | `6.311197` | `8.284465` | `0.202033` | `14.5024` |
| `fused_up_gate:bits2:block128` | `24` | `0.7390` | `1.083192` | `1.217540` | `0.0381595` | `2.51024` |
| `fused_up_gate:bits2:block256` | `24` | `0.7173` | `1.106137` | `1.184785` | `0.0390676` | `2.06225` |
| `fused_up_gate:bits2:block64` | `24` | `0.7825` | `1.045023` | `1.184614` | `0.0361841` | `1.92842` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp67-activation-output-screen-n16-france/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp67-activation-output-screen-n16-france/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260707-gp67-activation-output-screen-n16-france/screen.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260707-gp67-activation-output-screen-n16-france/report.md --bits 1,2 --blocks 64,128,256 --max-records 72 --torch-threads 8
```
