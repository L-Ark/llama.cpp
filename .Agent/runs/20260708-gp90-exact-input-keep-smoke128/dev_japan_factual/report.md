# Kimi exact input-channel keep oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T18:42:31+0000`
- activation_csv: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.csv`
- records_loaded: `128`
- unique tensor experts: `128`
- keep fractions: `[0.01, 0.02, 0.05, 0.1, 0.2, 0.4]`

## Matvec Output Error

| role | keep | rows | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---|---:|---:|---:|---:|---:|---:|---:|
| `down` | `0.01` | `32` | `0.0100` | `0.783893` | `0.923715` | `0.0584673` | `1.44615` |
| `down` | `0.02` | `32` | `0.0200` | `0.716694` | `0.877477` | `0.0525212` | `1.33376` |
| `down` | `0.05` | `32` | `0.0500` | `0.599976` | `0.762073` | `0.0430556` | `1.03562` |
| `down` | `0.1` | `32` | `0.1000` | `0.479482` | `0.640905` | `0.0339601` | `0.782062` |
| `down` | `0.2` | `32` | `0.2000` | `0.328933` | `0.448805` | `0.0226147` | `0.460099` |
| `down` | `0.4` | `32` | `0.4000` | `0.160916` | `0.227084` | `0.0108879` | `0.228901` |
| `gate` | `0.01` | `48` | `0.0100` | `0.954885` | `0.967031` | `0.205175` | `2.05301` |
| `gate` | `0.02` | `48` | `0.0200` | `0.923815` | `0.940033` | `0.198362` | `1.90947` |
| `gate` | `0.05` | `48` | `0.0500` | `0.843426` | `0.862085` | `0.18136` | `1.72074` |
| `gate` | `0.1` | `48` | `0.1000` | `0.742997` | `0.768417` | `0.15976` | `1.7078` |
| `gate` | `0.2` | `48` | `0.2000` | `0.585032` | `0.615381` | `0.125446` | `1.26094` |
| `gate` | `0.4` | `48` | `0.4000` | `0.354779` | `0.368605` | `0.0760994` | `0.803035` |
| `up` | `0.01` | `48` | `0.0100` | `0.953743` | `0.966665` | `0.206816` | `2.01705` |
| `up` | `0.02` | `48` | `0.0200` | `0.922740` | `0.934526` | `0.199934` | `1.90735` |
| `up` | `0.05` | `48` | `0.0500` | `0.843988` | `0.864989` | `0.182725` | `1.88007` |
| `up` | `0.1` | `48` | `0.1000` | `0.742629` | `0.759442` | `0.16085` | `1.75282` |
| `up` | `0.2` | `48` | `0.2000` | `0.585242` | `0.610031` | `0.126448` | `1.38936` |
| `up` | `0.4` | `48` | `0.4000` | `0.354999` | `0.375803` | `0.0767182` | `0.776666` |

## Fused Up/Gate Output Error

| keep | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |
|---:|---:|---:|---:|---:|---:|---:|
| `0.01` | `48` | `0.0100` | `0.996184` | `1.000512` | `0.030315` | `1.78027` |
| `0.02` | `48` | `0.0200` | `0.988786` | `0.997049` | `0.0303624` | `1.73928` |
| `0.05` | `48` | `0.0500` | `0.957113` | `0.974527` | `0.029982` | `1.63948` |
| `0.1` | `48` | `0.1000` | `0.893635` | `0.927991` | `0.028585` | `1.58946` |
| `0.2` | `48` | `0.2000` | `0.755722` | `0.805050` | `0.0247114` | `1.40608` |
| `0.4` | `48` | `0.4000` | `0.489026` | `0.514360` | `0.0161966` | `0.837685` |

## Decision

No exact input-channel keep candidate passes the offline error gate within the target byte ratio. Do not implement activation-guided partial exact reads as the next primary runtime path.

## Reproduce

```bash
.Agent/run-tools/kimi_exact_input_keep_oracle.py --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.csv --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual/act/activations.f32 --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp90-exact-input-keep-smoke128/dev_japan_factual/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp90-exact-input-keep-smoke128/dev_japan_factual/report.md --keep-fracs 0.01,0.02,0.05,0.1,0.2,0.4 --max-records 128 --torch-threads 8
```
