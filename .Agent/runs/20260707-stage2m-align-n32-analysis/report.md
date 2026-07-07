# GP66 2 MiB Pinned Staging Alignment Probe

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`
Plan: `.Agent/plans/kimi-next-expert-transfer-optimization-plan.md`

## Goal

Test the Phase 1 idea from the external repo survey: 2 MiB-align expert-pack pinned staging buffers to see whether larger host buffer alignment improves real Kimi decode throughput.

This was intentionally run as a small n32 cold-start A/B before any n96 or held-out expansion.

## Implementation Tested

Temporary code was applied only in a clean remote clone:

```text
/root/lfz/tmp/kimi-stage2m-align
```

The implementation added a default-off env:

```text
GGML_MOE_STAGE_2M_ALIGN=1
```

When enabled, each pinned staging slot allocated `slot_size + 2 MiB - 1`, then used a 2 MiB-aligned pointer inside that pinned allocation for io_uring reads and H2D copies. Default behavior was later corrected to allocate exactly the original slot size when the env was not enabled.

The test build succeeded:

```text
cmake --build build-cuda-batch -j$(nproc) --target llama-completion
```

Warnings were pre-existing unused/missing-declaration warnings in CUDA MoE files.

## Reproduction

Common run constraints:

- cold start: `sync; echo 3 > /proc/sys/vm/drop_caches`
- cgroup: `systemd-run --scope -p MemoryMax=16000M -p MemorySwapMax=0`
- prompt: `Please introduce France in a short paragraph.`
- n_predict: `32`
- model: `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf`

Critical env:

```text
GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack
GGML_MOE_EXPERT_PACK_OVERLAY=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack
GGML_MOE_EXPERT_GGUF_ALIAS_TSV=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv
GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1
GGML_MOE_IO_BACKEND=iouring
GGML_MOE_IO_BYTES=8388608
GGML_MOE_IO_DEPTH=8
GGML_MOE_IO_REFILL_BATCH=4
GGML_MOE_IO_SORT_OFFSET=1
GGML_MOE_IO_SQPOLL=1
GGML_MOE_STAGE_PINNED=1
GGML_MOE_STAGE_PINNED_SLOTS=12
GGML_MOE_VRAM_CACHE_MIB=15000
GGML_MOE_VRAM_CACHE_UPGATE_PCT=62
GGML_MOE_CURRENT_DOWN_OVERLAP=1
```

Full remote run roots:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260707-stage2m-align-n32-default/dev_france_regression
/root/lfz/runs/vendor-kimi-token-rate/20260707-stage2m-align-n32-align2m/dev_france_regression
```

## Results

| Mode | Prompt eval | Decode | Token rate | Total | Quality |
| --- | ---: | ---: | ---: | ---: | --- |
| default staging | 87135.53 ms | 18032.14 ms | 1.72 tok/s | 105190.00 ms | pass |
| 2 MiB aligned staging | 87774.79 ms | 18086.40 ms | 1.71 tok/s | 105881.67 ms | pass |

Both outputs started with:

```text
France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous
```

## Transfer Counters

Default:

```text
expert pack: iouring_reads=23501 iouring_bytes=131119579136 iouring_wait_us=11963044 iouring_h2d_enqueues=23501
expert pack iouring: batches=5361 submit_calls=5361 wait_calls=18566 cqes=23501 inflight_avg=3.13 inflight_max=8
pinned staging: copies=16849 waits=16801 slots=12 slot=7.44 MiB slot_wait=39.089 ms enqueue=358.658 ms h2d=5268.353 ms
pinned staging gate: copies=6652 waits=6616 slots=12 slot=5.36 MiB slot_wait=13.285 ms enqueue=107.127 ms h2d=2283.592 ms
```

2 MiB aligned:

```text
expert pack: iouring_reads=23501 iouring_bytes=131119579136 iouring_wait_us=11950761 iouring_h2d_enqueues=23501
expert pack iouring: batches=5361 submit_calls=5361 wait_calls=18653 cqes=23501 inflight_avg=3.13 inflight_max=8
pinned staging: copies=16849 waits=16801 slots=12 slot=7.44 MiB align=2.00 MiB slot_wait=33.439 ms enqueue=381.957 ms h2d=5286.851 ms
pinned staging gate: copies=6652 waits=6616 slots=12 slot=5.36 MiB align=2.00 MiB slot_wait=13.406 ms enqueue=103.483 ms h2d=2274.537 ms
```

## Analysis

The alignment change did not reduce real decode time:

- decode regressed by `54.26 ms` over n32;
- token rate changed from `1.72` to `1.71 tok/s`;
- prompt eval also regressed by `639.26 ms`;
- iouring wait was essentially unchanged: `11963.044 ms` vs `11950.761 ms`;
- H2D was not improved: main staging `5268.353 -> 5286.851 ms`, gate staging `2283.592 -> 2274.537 ms`.

This indicates the current runtime bottleneck is not host pointer alignment at the staging slot level. The earlier diagnosis still holds: real decode is limited by small op-local IO batches and insufficient cross-layer queue depth, not by O_DIRECT buffer alignment.

## Decision

Rejected.

The code change was not committed and was reverted locally. No SOTA claim is made.

Next step remains Phase 0/Phase 2 from the plan:

1. add gate/topK-to-read-submit observability;
2. then prototype a default-off global expert transfer scheduler with demand/prefetch deduplication and demand stealing.
