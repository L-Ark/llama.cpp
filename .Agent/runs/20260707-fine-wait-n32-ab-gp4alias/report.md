# Kimi Down Prefetch Fine-Wait Probe

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`
Remote repo: `/root/lfz/tmp/kimi-stage2m-align`

## Objective

Test whether skipping the down-batch entry `cudaStreamSynchronize(prefetch_stream)` can reduce exposed latency when `GGML_MOE_CURRENT_DOWN_OVERLAP=1` is already responsible for current-token down expert overlap.

The probe was default-off:

```bash
GGML_MOE_DOWN_PREFETCH_FINE_WAIT=1
```

## Correct n32 A/B

Run root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260707-fine-wait-n32-ab-gp4alias
```

Prompt:

```text
Please introduce France in a short paragraph.
```

Both runs used the GP4 alias env:

```bash
GGML_MOE_EXPERT_GGUF_ALIAS_TSV=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv
GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1
```

| config | quality | tok/s | decode ms/runs | TTFT ms | RAM peak bytes | direct reads | missing pack |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | pass | 1.74 | 17836.48/31 | 78236.87 | 15899996160 | 0 | 0 |
| fine-wait | pass | 1.74 | 17809.93/31 | 85794.85 | 15899996160 | 0 | 0 |

Key counters were otherwise identical:

- `entries=69120`
- `iouring_reads=23501`
- `iouring_bytes=131119579136`
- down hit rate: `73.4%`
- upgate hit rate: `45.2%`

## Decision

Rejected as a performance optimization.

The decode improvement was only `26.55 ms` over 31 decode runs, while TTFT increased by about `9.7%`. The runtime code was reverted to the prior SOTA behavior.

## Invalid Run To Ignore

Run root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260707-fine-wait-heldout-n96-test
```

This held-out n96 sweep omitted the GP4 alias env, so it did not reproduce the accepted SOTA path. It produced `direct_reads`/large `missing_pack` and very low token rates. It is not comparable to `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`.

The reproduction runner was updated after this to include the GP4 alias env by default.
