# Kimi Global Expert Scheduler Shadow Probe

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`
Remote repo: `/root/lfz/tmp/kimi-stage2m-align`

## Objective

Implement Phase 2A.0 from `.Agent/plans/kimi-next-expert-transfer-optimization-plan.md`: a default-off shadow task table for expert-pack io_uring reads.

The probe is enabled with:

```bash
GGML_MOE_GLOBAL_EXPERT_SCHED_SHADOW=1
```

It does not change routing, read submission, wait order, H2D copies, cache policy, or output. It records active physical read keys:

```text
(source_idx, read_offset, read_size)
```

## Run

Run root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260707-global-sched-shadow-n32-ab
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

## Result

| config | quality | tok/s | decode ms/runs | TTFT ms | RAM peak bytes | direct reads | missing pack |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | pass | 1.68 | 18474.01/31 | 87776.30 | 15899996160 | 0 | 0 |
| shadow | pass | 1.71 | 18114.25/31 | 87553.93 | 15899996160 | 0 | 0 |

The small timing difference is noise; the probe is diagnostic, not a SOTA claim.

Shadow counters:

```text
batches=5361
tasks=23501
demand=19828
prefetch=3673
active_duplicates=0
demand_hit_prefetch=0
prefetch_hit_demand=0
demand_hit_demand=0
prefetch_hit_prefetch=0
max_active=16
max_batch=8
batch_hist=1:176,2-4:2774,5-8:2411,9-16:0,17-32:0,gt32:0
```

## Interpretation

No active duplicate physical read task appeared in this n32 France trace. A dedup/steal-only global scheduler is therefore unlikely to improve token rate by itself. The current runtime can expose up to 16 active read tasks across concurrent rings/threads, but they are distinct keys.

The next useful scheduler work must create useful future tasks or reduce bytes:

- prediction or known-future enqueue;
- co-occurrence-aware pack layout;
- smaller expert representation;
- or another mechanism that makes more work available before demand stalls.

## Decision

Keep the shadow probe default-off and committed as observability. Do not promote any token-rate SOTA from this run.
