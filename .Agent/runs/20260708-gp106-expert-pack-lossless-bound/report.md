# Kimi expert-pack lossless compression bound

This is an offline non-destructive bound. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T21:34:12+0000`
- sampled entries: `80`
- sampled raw bytes: `451870720`
- target ratio: `0.4`

## Pack Sources

| pack | entries | payload GiB | sampled entries |
|---|---:|---:|---:|
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | `30831` | `163.101` | `72` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | `768` | `4.512` | `8` |

## Raw Payload Compression

| codec | entries | raw MiB | compressed MiB | aggregate ratio | mean ratio | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lzma6` | `80` | `430.94` | `428.58` | `0.9945` | `0.9961` | `0.9198` | `1.0001` |
| `zlib6` | `80` | `430.94` | `427.98` | `0.9931` | `0.9945` | `0.9206` | `1.0003` |
| `zlib9` | `80` | `430.94` | `427.98` | `0.9931` | `0.9945` | `0.9206` | `1.0003` |

### By Role

| role | codec | entries | aggregate ratio | mean ratio |
|---|---|---:|---:|---:|
| `down` | `lzma6` | `32` | `0.9880` | `0.9900` |
| `down` | `zlib6` | `32` | `0.9884` | `0.9904` |
| `down` | `zlib9` | `32` | `0.9884` | `0.9904` |
| `gate` | `lzma6` | `24` | `1.0001` | `1.0001` |
| `gate` | `zlib6` | `24` | `0.9969` | `0.9969` |
| `gate` | `zlib9` | `24` | `0.9969` | `0.9969` |
| `up` | `lzma6` | `24` | `1.0001` | `1.0001` |
| `up` | `zlib6` | `24` | `0.9975` | `0.9976` |
| `up` | `zlib9` | `24` | `0.9975` | `0.9976` |

## Base-XOR Residual Compression

| codec | entries | raw MiB | compressed MiB | aggregate ratio | mean ratio | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lzma6` | `80` | `430.94` | `323.24` | `0.7501` | `0.7501` | `0.0002` | `1.0001` |
| `zlib6` | `80` | `430.94` | `323.39` | `0.7504` | `0.7504` | `0.0010` | `1.0003` |
| `zlib9` | `80` | `430.94` | `323.39` | `0.7504` | `0.7504` | `0.0010` | `1.0003` |

### By Role

| role | codec | entries | aggregate ratio | mean ratio |
|---|---|---:|---:|---:|
| `down` | `lzma6` | `32` | `0.7501` | `0.7501` |
| `down` | `zlib6` | `32` | `0.7504` | `0.7504` |
| `down` | `zlib9` | `32` | `0.7504` | `0.7504` |
| `gate` | `lzma6` | `24` | `0.7501` | `0.7501` |
| `gate` | `zlib6` | `24` | `0.7505` | `0.7505` |
| `gate` | `zlib9` | `24` | `0.7505` | `0.7505` |
| `up` | `lzma6` | `24` | `0.7501` | `0.7501` |
| `up` | `zlib6` | `24` | `0.7505` | `0.7505` |
| `up` | `zlib9` | `24` | `0.7505` | `0.7505` |

## Base-XOR Residual Compression, Excluding Base-Self Rows

| codec | entries | raw MiB | compressed MiB | aggregate ratio | mean ratio | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lzma6` | `60` | `323.20` | `323.22` | `1.0001` | `1.0001` | `1.0001` | `1.0001` |
| `zlib6` | `60` | `323.20` | `323.28` | `1.0003` | `1.0003` | `0.9995` | `1.0003` |
| `zlib9` | `60` | `323.20` | `323.28` | `1.0003` | `1.0003` | `0.9995` | `1.0003` |

### By Role

| role | codec | entries | aggregate ratio | mean ratio |
|---|---|---:|---:|---:|
| `down` | `lzma6` | `24` | `1.0001` | `1.0001` |
| `down` | `zlib6` | `24` | `1.0002` | `1.0002` |
| `down` | `zlib9` | `24` | `1.0002` | `1.0002` |
| `gate` | `lzma6` | `18` | `1.0001` | `1.0001` |
| `gate` | `zlib6` | `18` | `1.0003` | `1.0003` |
| `gate` | `zlib9` | `18` | `1.0003` | `1.0003` |
| `up` | `lzma6` | `18` | `1.0001` | `1.0001` |
| `up` | `zlib6` | `18` | `1.0003` | `1.0003` |
| `up` | `zlib9` | `18` | `1.0003` | `1.0003` |

## Best Rows

| mode | codec | aggregate ratio | decision |
|---|---|---:|---|
| `raw_summary` | `zlib6` | `0.9931` | `fail` |
| `xor_summary` | `lzma6` | `0.7501` | `fail` |
| `xor_nonbase_summary` | `lzma6` | `1.0001` | `fail` |

## Decision

Reject as primary: neither raw lossless compression nor exact base-XOR residual compression approaches the 0.40 target ratio. Generic exact expert-byte compression cannot close the Kimi 5 tok/s byte gap.

## Reproduce

```bash
.Agent/run-tools/kimi_expert_pack_lossless_bound.py --pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack --pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp106-expert-pack-lossless-bound/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp106-expert-pack-lossless-bound/report.md --per-role 24 --per-tensor 4 --seed 1 --codecs zlib6,zlib9,lzma6 --target-ratio 0.40
```
