# Kimi activation-output compression screen

This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.

- generated_at: `2026-07-12T07:18:57+0000`
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
| `down:aw_mse:bits2:block128` | `16` | `0.6182` | `0.447064` | `0.455114` | `0.0494061` | `0.827363` |
| `down:aw_mse:bits2:block256` | `16` | `0.6000` | `0.449277` | `0.459352` | `0.049538` | `0.774187` |
| `down:aw_mse:bits2:block64` | `16` | `0.6545` | `0.446223` | `0.454726` | `0.0492467` | `0.7319` |
| `down:maxabs:bits2:block128` | `16` | `0.6182` | `0.775653` | `0.788197` | `0.086816` | `1.26089` |
| `down:maxabs:bits2:block256` | `16` | `0.6000` | `0.814252` | `0.825387` | `0.0909895` | `1.25116` |
| `down:maxabs:bits2:block64` | `16` | `0.6545` | `0.729465` | `0.738946` | `0.0817675` | `1.12434` |
| `gate:aw_mse:bits2:block128` | `16` | `0.7616` | `0.424700` | `0.444969` | `0.104706` | `1.17122` |
| `gate:aw_mse:bits2:block256` | `16` | `0.7392` | `0.426635` | `0.444956` | `0.104777` | `1.17233` |
| `gate:aw_mse:bits2:block64` | `16` | `0.8064` | `0.421994` | `0.439465` | `0.104053` | `0.883779` |
| `gate:maxabs:bits2:block128` | `16` | `0.7616` | `0.748206` | `0.773478` | `0.18408` | `1.52631` |
| `gate:maxabs:bits2:block256` | `16` | `0.7392` | `0.788559` | `0.816048` | `0.19393` | `1.88691` |
| `gate:maxabs:bits2:block64` | `16` | `0.8064` | `0.695835` | `0.720756` | `0.170525` | `1.47157` |
| `up:aw_mse:bits2:block128` | `16` | `0.6939` | `0.429443` | `0.446405` | `0.105489` | `0.858701` |
| `up:aw_mse:bits2:block256` | `16` | `0.6735` | `0.431109` | `0.451011` | `0.105965` | `0.899362` |
| `up:aw_mse:bits2:block64` | `16` | `0.7347` | `0.428700` | `0.444868` | `0.105444` | `0.871613` |
| `up:maxabs:bits2:block128` | `16` | `0.6939` | `0.748689` | `0.788375` | `0.185838` | `1.53404` |
| `up:maxabs:bits2:block256` | `16` | `0.6735` | `0.792525` | `0.820917` | `0.195952` | `1.59758` |
| `up:maxabs:bits2:block64` | `16` | `0.7347` | `0.700496` | `0.735576` | `0.173035` | `1.65061` |

## Fused Up/Gate Output Error

| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits2:block128` | `16` | `0.7247` | `0.580951` | `0.640225` | `0.0282398` | `1.29919` |
| `fused_up_gate:aw_mse:bits2:block256` | `16` | `0.7034` | `0.582113` | `0.645486` | `0.0283121` | `1.68192` |
| `fused_up_gate:aw_mse:bits2:block64` | `16` | `0.7673` | `0.575319` | `0.640693` | `0.0280742` | `1.01539` |
| `fused_up_gate:maxabs:bits2:block128` | `16` | `0.7247` | `1.078265` | `1.168349` | `0.0527979` | `1.36476` |
| `fused_up_gate:maxabs:bits2:block256` | `16` | `0.7034` | `1.118518` | `1.266916` | `0.0546001` | `1.90138` |
| `fused_up_gate:maxabs:bits2:block64` | `16` | `0.7673` | `1.029527` | `1.107082` | `0.0497221` | `1.3134` |

## Decision

- This is a screen only; it does not produce a runtime pack or SOTA claim.
- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.
- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_output_compression_screen.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260712-current-goal-2bit-fused-screen/dev_mixed_summary-bits2.json --out-md .Agent/runs/20260712-current-goal-2bit-fused-screen/dev_mixed_summary-bits2.md --bits 2 --blocks 64,128,256 --scale-modes maxabs,aw_mse --max-records 48 --torch-threads 8
```
