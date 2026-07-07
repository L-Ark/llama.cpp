# Kimi shared down-projector oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T20:27:44+0000`
- prompts: `3`
- groups: `96`
- layers evaluated: `32`
- max down records per prompt: `256`

## Summary

| mode | lambda | rows | mean rel L2 | max rel L2 | BF16 MiB/layer | BF16 GiB/60 layers |
|---|---:|---:|---:|---:|---:|---:|
| `slot_expert_scalar_h` | `10` | `96` | `1.276001` | `1.915882` | `448.00` | `26.25` |
| `slot_expert_scalar_h` | `100` | `96` | `1.252953` | `1.774497` | `448.00` | `26.25` |
| `slot_expert_scalar_h` | `1000` | `96` | `1.253520` | `1.773099` | `448.00` | `26.25` |

## Worst Layer Rows

| layer | mode | lambda | rows | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|---:|
| `blk.24` | `slot_expert_scalar_h` | `10` | `3` | `1.443512` | `1.915882` |
| `blk.56` | `slot_expert_scalar_h` | `10` | `3` | `1.385290` | `1.741974` |
| `blk.24` | `slot_expert_scalar_h` | `100` | `3` | `1.377272` | `1.718914` |
| `blk.52` | `slot_expert_scalar_h` | `10` | `3` | `1.371843` | `1.635358` |
| `blk.24` | `slot_expert_scalar_h` | `1000` | `3` | `1.371013` | `1.699897` |
| `blk.3` | `slot_expert_scalar_h` | `10` | `3` | `1.366658` | `1.620373` |
| `blk.5` | `slot_expert_scalar_h` | `1000` | `3` | `1.362362` | `1.773099` |
| `blk.5` | `slot_expert_scalar_h` | `100` | `3` | `1.362088` | `1.774497` |
| `blk.5` | `slot_expert_scalar_h` | `10` | `3` | `1.361360` | `1.788374` |
| `blk.3` | `slot_expert_scalar_h` | `100` | `3` | `1.360482` | `1.606180` |
| `blk.3` | `slot_expert_scalar_h` | `1000` | `3` | `1.359756` | `1.604515` |
| `blk.35` | `slot_expert_scalar_h` | `10` | `3` | `1.355832` | `1.502615` |
| `blk.41` | `slot_expert_scalar_h` | `10` | `3` | `1.335800` | `1.556115` |
| `blk.55` | `slot_expert_scalar_h` | `1000` | `3` | `1.330849` | `1.531336` |
| `blk.16` | `slot_expert_scalar_h` | `10` | `3` | `1.327106` | `1.583176` |
| `blk.42` | `slot_expert_scalar_h` | `10` | `3` | `1.320449` | `1.554724` |
| `blk.56` | `slot_expert_scalar_h` | `100` | `3` | `1.317470` | `1.713267` |
| `blk.55` | `slot_expert_scalar_h` | `100` | `3` | `1.315973` | `1.525889` |
| `blk.56` | `slot_expert_scalar_h` | `1000` | `3` | `1.312871` | `1.711858` |
| `blk.16` | `slot_expert_scalar_h` | `1000` | `3` | `1.311193` | `1.615485` |
| `blk.16` | `slot_expert_scalar_h` | `100` | `3` | `1.306790` | `1.611022` |
| `blk.48` | `slot_expert_scalar_h` | `10` | `3` | `1.306530` | `1.529032` |
| `blk.45` | `slot_expert_scalar_h` | `10` | `3` | `1.300694` | `1.496101` |
| `blk.20` | `slot_expert_scalar_h` | `10` | `3` | `1.300404` | `1.454254` |
| `blk.35` | `slot_expert_scalar_h` | `100` | `3` | `1.300281` | `1.494599` |
| `blk.41` | `slot_expert_scalar_h` | `100` | `3` | `1.298460` | `1.544149` |
| `blk.41` | `slot_expert_scalar_h` | `1000` | `3` | `1.298308` | `1.551470` |
| `blk.59` | `slot_expert_scalar_h` | `1000` | `3` | `1.297300` | `1.549150` |
| `blk.35` | `slot_expert_scalar_h` | `1000` | `3` | `1.296559` | `1.493675` |
| `blk.45` | `slot_expert_scalar_h` | `100` | `3` | `1.296399` | `1.471173` |
| `blk.45` | `slot_expert_scalar_h` | `1000` | `3` | `1.295970` | `1.468382` |
| `blk.59` | `slot_expert_scalar_h` | `100` | `3` | `1.290121` | `1.542596` |
| `blk.55` | `slot_expert_scalar_h` | `10` | `3` | `1.286733` | `1.485246` |
| `blk.59` | `slot_expert_scalar_h` | `10` | `3` | `1.285559` | `1.497040` |
| `blk.38` | `slot_expert_scalar_h` | `10` | `3` | `1.278418` | `1.294403` |
| `blk.48` | `slot_expert_scalar_h` | `100` | `3` | `1.278123` | `1.514528` |
| `blk.48` | `slot_expert_scalar_h` | `1000` | `3` | `1.276571` | `1.514662` |
| `blk.34` | `slot_expert_scalar_h` | `10` | `3` | `1.276257` | `1.432833` |
| `blk.58` | `slot_expert_scalar_h` | `1000` | `3` | `1.273191` | `1.456992` |
| `blk.58` | `slot_expert_scalar_h` | `100` | `3` | `1.265305` | `1.454276` |

## Decision

Reject as primary: best shared down projector slot_expert_scalar_h lambda=100 has mean rel L2 1.252953, above the 0.10 gate.

## Reproduce

```bash
.Agent/run-tools/kimi_shared_down_projector_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_python_reverse --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/dev_mixed_summary --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp102-slot-expert-scalar-down-surrogate/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp102-slot-expert-scalar-down-surrogate/report.md --max-down-records-per-prompt 256 --modes slot_expert_scalar_h --lambdas 10.0,100.0,1000.0 --torch-threads 8
```

Executed on the remote host from `/root/lfz/tmp/kimi-stage2m-align` under
`systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0`.
