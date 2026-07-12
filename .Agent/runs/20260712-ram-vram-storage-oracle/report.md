# Kimi RAM/VRAM Storage Rebalance Oracle

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Head while running analysis: `01375939f`

## Goal

Evaluate whether replacing low-value Linux file/page cache with explicit RAM
expert residency can reduce Kimi critical-path io_uring wait enough to justify
a runtime A/B under the existing constraints:

- generalized dev prompts only;
- cold start;
- host RAM under 16GB including page cache/RSS/pinned/RAM tier;
- no prompt-specific SOTA claim;
- quality must remain correct before any SOTA acceptance.

## Inputs

Score/copy-profile dev root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-dev-n32-035310
```

IO read-trace dev root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-ram-vram-io-trace-n32-040559
```

Prompts:

- `dev_france_regression`: `Please introduce France in a short paragraph.`
- `dev_intelligence_general`: `What is intelligence?`

## Baseline Observations

The current path is already expert-pack/io_uring based:

- France copy profile: `207.35 GiB` profiled copy traffic, all `pack_hit=1`,
  all `iouring=1`, no pack miss.
- Intelligence copy profile: `198.18 GiB` profiled copy traffic, all
  `pack_hit=1`, all `iouring=1`, no pack miss.
- Therefore RAM tier is not fixing GGUF fallback. It can only replace some
  SSD/io_uring demand reads with RAM->VRAM copies.

Wait-weighted layer/role screen:

- best single layer/role: `blk.14 upgate`, `11.832 ms/token`, observed
  footprint `1575 MiB`;
- best down layer/role: about `5.98 ms/token`;
- a single layer or role slab is far below the saving needed for `2 tok/s`.

## Row-Level RAM Tier Bound

Artifact:

```text
.Agent/runs/20260712-ram-vram-storage-oracle/copy-profile-ram-tier-oracle/oracle.md
```

This bound ranks individual expert entries by profiled io_wait per resident
byte. It is intentionally optimistic because it assumes each selected expert
row independently removes its full io_wait.

Result:

| budget | optimistic saved | conservative saved | conservative tok/s |
|---:|---:|---:|---:|
| 1GB | `153.018 ms/token` | `145.698 ms/token` | `1.965` |
| 2GB | `262.855 ms/token` | `249.198 ms/token` | `2.467` |
| 4GB | `439.381 ms/token` | `414.934 ms/token` | `4.172` |

Interpretation:

- Row-level selection says RAM could matter if each chosen expert miss were an
  independent stall.
- That assumption is too strong for the real runtime because demand reads are
  batched; if one expert in a batch remains on SSD, the batch still has SSD
  wait and RAM hits may fragment the effective IO queue.

## Complete-Batch RAM Cover Oracle

Artifact:

```text
.Agent/runs/20260712-ram-vram-storage-oracle/batch-cover-ram-oracle/oracle.md
```

This oracle uses `io-read-trace.csv` and only counts saved wait when a complete
IO batch is covered by RAM-resident experts. It is still dev-overfit and
therefore an upper bound.

Input summary:

- traces: `2`;
- prompts: `2`;
- batches: `11134`;
- unique expert keys: `24053`;
- decode runs: `62`;
- total traced wait: `501.193 ms/token`.

Greedy complete-batch cover result:

| budget | resident GiB | covered batches | saved ms/token | bounded tok/s |
|---:|---:|---:|---:|---:|
| 1GB | `0.997` | `146` | `5.839` | `1.691` |
| 2GB | `1.999` | `291` | `10.771` | `1.705` |
| 4GB | `3.998` | `539` | `21.573` | `1.737` |
| 6GB | `5.998` | `817` | `32.511` | `1.771` |
| 8GB | `7.999` | `1052` | `42.761` | `1.803` |
| 10GB | `9.999` | `1286` | `53.441` | `1.839` |

The IO-trace baseline is `37031.89 ms / 62`, or about `1.674 tok/s`. Reaching
`2 tok/s` would require about `97.3 ms/token` saving in this traced run. Even a
dev-overfit 10GB complete-batch RAM resident set saves only `53.441 ms/token`.

## Decision

Do not build a runtime A/B for static RAM hot expert tier or static whole-layer
RAM slabs from the current traces.

Reasons:

- single layer/role slabs are too small in critical-path impact;
- row-level RAM tier looks promising only under an unrealistic independence
  assumption;
- complete-batch coverage, which models mixed SSD/RAM batch fragmentation more
  honestly, does not reach the `2 tok/s` gate even with 10GB resident RAM and
  dev-trace overfitting;
- this would also risk TTFT increase from cold preload and host RAM pressure.

## Next Direction

RAM is still useful only if paired with a mechanism that improves batch
dominance:

- a stronger draft-router/hidden-state predictor that preloads complete future
  batches;
- a storage format that lowers bytes for all active experts in a batch;
- a lower-byte expert representation that reduces SSD/H2D traffic without
  relying on Linux page cache;
- or a runtime scheduler that can keep SSD queues saturated despite RAM hits,
  which must be proven with IO trace before A/B.

Given the current evidence, the next higher-value path is lower-byte expert
representation or a stronger predictor, not static RAM/VRAM storage rebalance.
