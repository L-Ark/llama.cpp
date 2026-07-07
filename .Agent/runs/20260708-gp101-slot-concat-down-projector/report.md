# Kimi shared down-projector oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T20:12:56+0000`
- prompts: `3`
- groups: `96`
- layers evaluated: `32`
- max down records per prompt: `256`

## Summary

| mode | lambda | rows | mean rel L2 | max rel L2 | BF16 MiB/layer | BF16 GiB/60 layers |
|---|---:|---:|---:|---:|---:|---:|
| `slot_concat_h` | `1` | `96` | `1.887801` | `8.662453` | `224.00` | `13.12` |
| `slot_concat_h` | `10` | `96` | `1.314453` | `2.442733` | `224.00` | `13.12` |
| `slot_concat_h` | `100` | `96` | `1.252990` | `1.778483` | `224.00` | `13.12` |

## Worst Layer Rows

| layer | mode | lambda | rows | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|---:|
| `blk.52` | `slot_concat_h` | `1` | `3` | `4.504939` | `8.662453` |
| `blk.55` | `slot_concat_h` | `1` | `3` | `3.867292` | `8.150792` |
| `blk.35` | `slot_concat_h` | `1` | `3` | `2.760233` | `5.273747` |
| `blk.38` | `slot_concat_h` | `1` | `3` | `2.274098` | `3.118184` |
| `blk.48` | `slot_concat_h` | `1` | `3` | `2.244226` | `3.296380` |
| `blk.59` | `slot_concat_h` | `1` | `3` | `2.208012` | `3.674954` |
| `blk.60` | `slot_concat_h` | `1` | `3` | `2.193784` | `4.637424` |
| `blk.28` | `slot_concat_h` | `1` | `3` | `2.130744` | `2.854859` |
| `blk.49` | `slot_concat_h` | `1` | `3` | `2.101486` | `3.642164` |
| `blk.42` | `slot_concat_h` | `1` | `3` | `2.075508` | `2.782992` |
| `blk.41` | `slot_concat_h` | `1` | `3` | `2.054149` | `2.869256` |
| `blk.56` | `slot_concat_h` | `1` | `3` | `1.988509` | `2.865535` |
| `blk.23` | `slot_concat_h` | `1` | `3` | `1.819206` | `2.080922` |
| `blk.20` | `slot_concat_h` | `1` | `3` | `1.807289` | `2.463355` |
| `blk.30` | `slot_concat_h` | `1` | `3` | `1.776256` | `2.499513` |
| `blk.31` | `slot_concat_h` | `1` | `3` | `1.762718` | `2.256167` |
| `blk.51` | `slot_concat_h` | `1` | `3` | `1.760742` | `2.588003` |
| `blk.44` | `slot_concat_h` | `1` | `3` | `1.705167` | `2.559763` |
| `blk.52` | `slot_concat_h` | `10` | `3` | `1.690452` | `2.442733` |
| `blk.24` | `slot_concat_h` | `1` | `3` | `1.642797` | `2.290845` |
| `blk.5` | `slot_concat_h` | `1` | `3` | `1.575066` | `2.333500` |
| `blk.55` | `slot_concat_h` | `10` | `3` | `1.542969` | `2.010849` |
| `blk.35` | `slot_concat_h` | `10` | `3` | `1.502081` | `2.124470` |
| `blk.34` | `slot_concat_h` | `1` | `3` | `1.467537` | `1.540955` |
| `blk.27` | `slot_concat_h` | `1` | `3` | `1.430480` | `1.490571` |
| `blk.58` | `slot_concat_h` | `1` | `3` | `1.429852` | `1.651632` |
| `blk.59` | `slot_concat_h` | `10` | `3` | `1.420098` | `1.518674` |
| `blk.56` | `slot_concat_h` | `10` | `3` | `1.411319` | `1.737059` |
| `blk.21` | `slot_concat_h` | `1` | `3` | `1.397089` | `1.583402` |
| `blk.5` | `slot_concat_h` | `10` | `3` | `1.387597` | `1.844800` |
| `blk.24` | `slot_concat_h` | `10` | `3` | `1.385203` | `1.811530` |
| `blk.48` | `slot_concat_h` | `10` | `3` | `1.381128` | `1.516703` |
| `blk.24` | `slot_concat_h` | `100` | `3` | `1.369983` | `1.709637` |
| `blk.5` | `slot_concat_h` | `100` | `3` | `1.364212` | `1.778483` |
| `blk.3` | `slot_concat_h` | `100` | `3` | `1.358068` | `1.602630` |
| `blk.41` | `slot_concat_h` | `10` | `3` | `1.350053` | `1.590671` |
| `blk.3` | `slot_concat_h` | `10` | `3` | `1.346525` | `1.590152` |
| `blk.16` | `slot_concat_h` | `1` | `3` | `1.345807` | `1.587623` |
| `blk.11` | `slot_concat_h` | `1` | `3` | `1.342165` | `1.517668` |
| `blk.28` | `slot_concat_h` | `10` | `3` | `1.340508` | `1.514274` |

## Decision

Reject as primary: best shared down projector slot_concat_h lambda=100 has mean rel L2 1.252990, above the 0.10 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_shared_down_projector_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp101-slot-concat-down-projector/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp101-slot-concat-down-projector/report.md --max-down-records-per-prompt 256 --modes slot_concat_h --lambdas 1.0,10.0,100.0 --torch-threads 8
```

Executed on the remote host from `/root/lfz/tmp/kimi-stage2m-align` under
`systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0`.
