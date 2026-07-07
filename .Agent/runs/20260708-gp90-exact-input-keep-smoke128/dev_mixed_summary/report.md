# Kimi exact input-channel keep oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T18:43:40+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.csv`
- records_loaded: `128`
- unique tensor experts: `128`
- keep fractions: `[0.01, 0.02, 0.05, 0.1, 0.2, 0.4]`

## Matvec Output Error

| role | keep | rows | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|---:|
| `down` | `0.01` | `32` | `0.0100` | `0.777905` | `0.886031` | `0.0564957` | `1.21281` |
| `down` | `0.02` | `32` | `0.0200` | `0.712026` | `0.821825` | `0.0514281` | `1.06998` |
| `down` | `0.05` | `32` | `0.0500` | `0.596969` | `0.706566` | `0.0427602` | `0.834707` |
| `down` | `0.1` | `32` | `0.1000` | `0.479176` | `0.580120` | `0.0339196` | `0.736171` |
| `down` | `0.2` | `32` | `0.2000` | `0.330728` | `0.409489` | `0.0228929` | `0.458546` |
| `down` | `0.4` | `32` | `0.4000` | `0.162025` | `0.228983` | `0.0109709` | `0.189955` |
| `gate` | `0.01` | `48` | `0.0100` | `0.957000` | `0.967894` | `0.222381` | `2.38602` |
| `gate` | `0.02` | `48` | `0.0200` | `0.923691` | `0.941964` | `0.214628` | `2.19125` |
| `gate` | `0.05` | `48` | `0.0500` | `0.845077` | `0.879234` | `0.196434` | `1.98607` |
| `gate` | `0.1` | `48` | `0.1000` | `0.743575` | `0.781557` | `0.172673` | `1.61346` |
| `gate` | `0.2` | `48` | `0.2000` | `0.587342` | `0.622091` | `0.13632` | `1.18266` |
| `gate` | `0.4` | `48` | `0.4000` | `0.358405` | `0.378865` | `0.0831837` | `0.826545` |
| `up` | `0.01` | `48` | `0.0100` | `0.956207` | `0.970100` | `0.223018` | `2.0714` |
| `up` | `0.02` | `48` | `0.0200` | `0.924319` | `0.943895` | `0.215371` | `2.01286` |
| `up` | `0.05` | `48` | `0.0500` | `0.846010` | `0.870237` | `0.197386` | `2.00265` |
| `up` | `0.1` | `48` | `0.1000` | `0.744320` | `0.772408` | `0.173863` | `1.75315` |
| `up` | `0.2` | `48` | `0.2000` | `0.589302` | `0.627098` | `0.137473` | `1.3298` |
| `up` | `0.4` | `48` | `0.4000` | `0.354371` | `0.370712` | `0.0825602` | `0.792402` |

## Fused Up/Gate Output Error

| keep | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---:|---:|---:|---:|---:|---:|---:|
| `0.01` | `48` | `0.0100` | `0.996376` | `1.000251` | `0.0339863` | `1.95539` |
| `0.02` | `48` | `0.0200` | `0.989607` | `0.999187` | `0.0340719` | `1.9558` |
| `0.05` | `48` | `0.0500` | `0.958808` | `0.980956` | `0.0336646` | `1.83903` |
| `0.1` | `48` | `0.1000` | `0.894497` | `0.930792` | `0.0321479` | `1.71157` |
| `0.2` | `48` | `0.2000` | `0.757861` | `0.802694` | `0.0278459` | `1.19946` |
| `0.4` | `48` | `0.4000` | `0.489464` | `0.544297` | `0.0182539` | `0.882902` |

## Decision

No exact input-channel keep candidate passes the offline error gate within the target byte ratio. Do not implement activation-guided partial exact reads as the next primary runtime path.

## Reproduce

```bash
.Agent/run-tools/kimi_exact_input_keep_oracle.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp90-exact-input-keep-smoke128/dev_mixed_summary/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp90-exact-input-keep-smoke128/dev_mixed_summary/report.md --keep-fracs 0.01,0.02,0.05,0.1,0.2,0.4 --max-records 128 --torch-threads 8
```
