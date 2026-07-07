# Kimi exact input-channel keep oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T18:43:06+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.csv`
- records_loaded: `128`
- unique tensor experts: `128`
- keep fractions: `[0.01, 0.02, 0.05, 0.1, 0.2, 0.4]`

## Matvec Output Error

| role | keep | rows | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|---:|
| `down` | `0.01` | `32` | `0.0100` | `0.779546` | `0.879173` | `0.0568799` | `1.25779` |
| `down` | `0.02` | `32` | `0.0200` | `0.708828` | `0.812853` | `0.0512744` | `0.987598` |
| `down` | `0.05` | `32` | `0.0500` | `0.588836` | `0.703514` | `0.0417796` | `0.823789` |
| `down` | `0.1` | `32` | `0.1000` | `0.468760` | `0.591127` | `0.0326644` | `0.554848` |
| `down` | `0.2` | `32` | `0.2000` | `0.321097` | `0.425596` | `0.0218482` | `0.432281` |
| `down` | `0.4` | `32` | `0.4000` | `0.157352` | `0.219291` | `0.0106482` | `0.17958` |
| `gate` | `0.01` | `48` | `0.0100` | `0.959090` | `0.970110` | `0.224594` | `1.96044` |
| `gate` | `0.02` | `48` | `0.0200` | `0.927635` | `0.944197` | `0.21732` | `1.98356` |
| `gate` | `0.05` | `48` | `0.0500` | `0.852238` | `0.869593` | `0.1996` | `1.77866` |
| `gate` | `0.1` | `48` | `0.1000` | `0.753539` | `0.776187` | `0.17641` | `1.66093` |
| `gate` | `0.2` | `48` | `0.2000` | `0.592539` | `0.611808` | `0.139054` | `1.35194` |
| `gate` | `0.4` | `48` | `0.4000` | `0.359070` | `0.375330` | `0.084289` | `0.802232` |
| `up` | `0.01` | `48` | `0.0100` | `0.956682` | `0.966804` | `0.226252` | `2.26178` |
| `up` | `0.02` | `48` | `0.0200` | `0.927018` | `0.939871` | `0.219218` | `2.35517` |
| `up` | `0.05` | `48` | `0.0500` | `0.851324` | `0.872054` | `0.201112` | `1.82964` |
| `up` | `0.1` | `48` | `0.1000` | `0.751348` | `0.774010` | `0.177532` | `1.59907` |
| `up` | `0.2` | `48` | `0.2000` | `0.591844` | `0.621306` | `0.139542` | `1.36713` |
| `up` | `0.4` | `48` | `0.4000` | `0.359309` | `0.374064` | `0.0849837` | `0.815464` |

## Fused Up/Gate Output Error

| keep | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---:|---:|---:|---:|---:|---:|---:|
| `0.01` | `48` | `0.0100` | `0.996933` | `1.002272` | `0.0347938` | `2.42595` |
| `0.02` | `48` | `0.0200` | `0.990271` | `0.997293` | `0.0348219` | `2.44067` |
| `0.05` | `48` | `0.0500` | `0.961803` | `0.979450` | `0.0343896` | `2.3023` |
| `0.1` | `48` | `0.1000` | `0.902445` | `0.927868` | `0.0329245` | `2.30113` |
| `0.2` | `48` | `0.2000` | `0.762676` | `0.809772` | `0.0284493` | `2.00522` |
| `0.4` | `48` | `0.4000` | `0.492035` | `0.545970` | `0.0187343` | `0.970985` |

## Decision

No exact input-channel keep candidate passes the offline error gate within the target byte ratio. Do not implement activation-guided partial exact reads as the next primary runtime path.

## Reproduce

```bash
.Agent/run-tools/kimi_exact_input_keep_oracle.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp90-exact-input-keep-smoke128/dev_python_reverse/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp90-exact-input-keep-smoke128/dev_python_reverse/report.md --keep-fracs 0.01,0.02,0.05,0.1,0.2,0.4 --max-records 128 --torch-threads 8
```
