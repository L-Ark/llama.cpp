# ik_llama / GLM5.1 SSD MoE Scheduler Plan

## 1. Goal

Use the existing `ik_llama` GLM5.1 SSD MoE implementation as the first practical landing point for the scheduler idea.

The objective is to improve decode latency and expert-cache hit rate for GLM5.1 without changing model semantics or answer quality.

Primary target:

- GLM5.1 UD-IQ3_XXS in `ik_llama`

Existing assets to reuse:

- `.expert-pack` cold expert storage
- direct SSD expert reads
- VRAM expert cache
- RAM tier controls
- route traces
- cache simulation scripts
- low-host-RAM profile work

Do not use this plan as the first implementation path for DeepSeek V4-Flash.

## 2. Constraints

- CPU RAM tier budget: 2GB.
- VRAM: use as much as practical; do not apply a 2GB VRAM cap.
- SSD: full cold expert tier.
- Quality: no expert reduction, no top-k reduction, no SER-like lossy behavior.
- No prompt-specific oracle routing.
- Static profile seeding is allowed only as a baseline or bootstrap; the target is runtime scheduling.

## 3. Existing ik_llama Fit

`ik_llama` is a good fit for this part because it already has the main components needed by the plan:

- `glm51-iq3xxs.expert-pack`
- expert-pack indexing
- SSD direct read path
- VRAM cache hit/miss accounting
- RAM tier configuration
- route/cache simulation scripts
- low-host-RAM operating profile

The work should therefore be an incremental scheduler upgrade, not a new runtime.

## 4. Baseline

Start from the accepted GLM5.1 low-host-RAM profile and current interactive profile.

Track at least:

- TTFT
- decode tokens/sec
- p50/p95 token latency
- direct SSD reads
- SSD read bytes
- VRAM cache hit rate
- RAM tier hit rate
- read failures
- output hash or canary verdict
- peak RSS
- peak VRAM

The key baseline comparison is:

- existing static/profile-seeded VRAM cache
- no RAM tier
- 2GB RAM tier
- dynamic scheduler with 2GB RAM tier

## 5. Scheduler Design

The scheduler manages three tiers:

- VRAM: hot experts used for fast compute
- CPU RAM: 2GB staging and small hot/candidate tier
- SSD: full cold expert store

Every expert key should include:

- tensor name
- expert id
- layer id
- tensor kind: up, gate, upgate, or down
- byte size
- pack offset

Every runtime access should update:

- last access time
- access count
- recent-window count
- miss penalty estimate
- SSD read time
- RAM hit/miss
- VRAM hit/miss

## 6. Replacement Policy

Promote an expert when its expected future miss cost is higher than the cost of the VRAM slot it consumes.

Score should include:

- recent activation frequency
- layer-local reuse
- cross-layer route correlation if measured useful
- SSD miss latency
- expert byte size
- current VRAM pressure
- RAM tier availability
- whether the expert was needed during decode rather than only prefill

Evict experts with:

- low recent use
- low predicted reuse
- large VRAM footprint relative to benefit
- no protection from static bootstrap profile

Keep the first implementation simple:

- LFU/LRU hybrid
- profile-seeded protected slots
- runtime unprotected slots
- separate budgets for up/gate/upgate and down if current code already uses that split

## 7. RAM Tier

The 2GB CPU RAM tier should be treated as a staging and candidate tier, not as a full expert store.

Use it for:

- pinned staging buffers
- recently missed experts likely to recur
- experts selected for near-future prefetch
- optional decoded/repacked payloads if that avoids repeated conversion

Do not let the RAM tier hide bad scheduling decisions. All RAM hits, misses, evictions, and bytes must be logged.

## 8. SSD Prefetch

Do not add speculative prefetch before route traces show it can help.

First implementation:

- synchronous correctness path
- async read path behind a flag
- exact expert prefetch from already-known route events where possible
- bounded queue depth
- cancellation or drop behavior when predictions become stale

Only enable speculative prefetch when:

- route predictor byte recall is high enough
- SSD load can be overlapped with GPU compute
- RAM tier has staging room
- prefetch does not increase p95 latency

## 9. KV Tradeoff

For GLM5.1, do not start with KV offload/recompute.

First prove:

- expert miss cost is still dominant
- VRAM expert cache size materially changes decode latency
- 2GB RAM tier improves or fails to improve the miss path

Only revisit KV offload/recompute if long context or multi-request scenarios create real competition between KV cache and expert cache.

## 10. Implementation Steps

1. Add a unified runtime expert-access telemetry row.
2. Add a scheduler simulation mode that consumes real route traces and outputs predicted hit/miss behavior.
3. Implement LFU/LRU dynamic replacement in the existing VRAM cache.
4. Add a hard 2GB RAM tier budget and per-entry accounting.
5. Add RAM tier hit/miss logging.
6. Add synchronous SSD-to-RAM-to-VRAM promotion path for correctness.
7. Add async SSD prefetch behind a flag.
8. Add score-based promotion and eviction.
9. Compare against current static/profile-seeded cache.
10. Keep all changes gated by environment variables or preset fields until validated.

## 11. Experiments

Run:

- current static profile baseline
- current low-host-RAM baseline
- dynamic VRAM cache without RAM tier
- dynamic VRAM cache with 2GB RAM tier
- async SSD prefetch off
- async SSD prefetch on
- static profile seed off
- static profile seed on

Use:

- deterministic canary
- representative interactive prompt
- at least 3 repeated runs per config
- long enough decode to expose steady-state behavior

## 12. Success Criteria

This plan succeeds if:

- output quality/canary remains unchanged
- read failures remain zero
- p95 token latency improves against the current static cache baseline
- VRAM hit rate improves without increasing TTFT unacceptably
- the 2GB RAM tier benefit is quantified
- the scheduler can explain its own promotion and eviction choices from logs

## 13. Stop Conditions

Stop or de-scope if:

- dynamic replacement causes correctness failures
- async prefetch increases p95 latency
- 2GB RAM tier shows no measurable benefit after telemetry confirms it is being used correctly
- route predictability is too low for speculative prefetch

If stopped:

- keep static/profile-seeded VRAM cache as the production path
- preserve telemetry and simulator improvements for future work

## 14. 2026-06-15 Strict 2GB Scheduler Validation

### Objective

Validate the narrower scheduler goal:

- strict host RAM: `MemoryMax=2G`, `MemorySwapMax=0`;
- RAM tier disabled: `GGML_MOE_RAM_TIER_MIB=0`;
- no SER / no lossy expert reduction;
- verify whether scheduler/read-order changes can reduce tail stalls and
  produce a real `5-10%` eval token-rate gain.

### Baseline and Candidate

Use the current best strict non-SER line from the 5090 task as the comparison
base:

- profile: `.Agent/plans/5090-theoretical-token-rate/oracle-n84.route.csv`;
- VRAM cache request: `20480 MiB`;
- cache auto-clamp: on;
- cache safety: `256 MiB`;
- up/gate split: `65%`;
- backend: `io_uring`;
- profile reserve: `10%`;
- cache policy: `lfu_lru`;
- context: `2048`;
- prompt: fixed n84 prompt from `run_5090_matrix.py`;
- seed: `42`.

Compare:

1. `baseline`: same config without `GGML_MOE_IO_SORT_OFFSET`.
2. `scheduler-offset-sort`: same config with `GGML_MOE_IO_SORT_OFFSET=1`.

The current implementation treats offset sorting as a conservative scheduler
primitive: it does not change selected experts, cache admission, profile rows,
or read count; it only orders same-size io_uring jobs by expert-pack offset
inside the existing batch.

### Metrics

Report per run:

- `eval_tokens_per_s`;
- `prompt_eval_tokens_per_s`;
- `total_ms`;
- `expert_pack_iouring_reads`;
- `expert_pack_read_failures`;
- `vram_cache_hit_rate_pct`;
- `ram_tier_hit_rate_pct` or disabled;
- aggregate `iouring_wait_us`, `iouring_wait_calls`, `inflight_avg`,
  `inflight_max`;
- per-ring pinned-staging io_uring details when parsed.

Tail-stall proxy for this validation:

- primary: `iouring_wait_us` and `iouring_wait_us / eval token`;
- secondary: `total_ms`, `eval tok/s`, and io_uring batch histogram.

### Execution Plan

1. Build-check `llama-cli`.
2. Run n8 smoke gate once for baseline and candidate to catch crashes/OOM.
3. If both pass with `read_failures=0`, run n84 for baseline and candidate.
4. Repeat n84 once more only if the first delta is near the expected noise
   range.
5. Summarize whether the candidate reaches a stable `5-10%` real eval
   token-rate improvement and whether tail-stall proxy metrics improve.

### Progress

- 2026-06-15: Created this validation section after re-reading repository and
  scheduler plan constraints. Initial resource check: GPU memory `41 MiB` used,
  `32071 MiB` free, GPU util `0%`; no active `llama-cli` / `moe-run.py`
  experiment process found. Proceeding with build check and n8 gate.
- 2026-06-15: Build check passed:
  `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`.
- 2026-06-15: n8 gate passed for both strict configs:

  | label | offset sort | eval tok/s | total ms | io_uring reads | io_uring wait us | VRAM hit | read failures | log |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | `baseline-nosort-n8` | no | `1.41` | `26098.89` | `7382` | `5202000` | `41.5%` | `0` | `runs/strict2gb-validation/baseline-nosort-n8/` |
  | `offset-sort-n8` | yes | `1.42` | `26148.60` | `7382` | `5451877` | `41.5%` | `0` | `runs/strict2gb-validation/offset-sort-n8/` |

  Interpretation: the short n8 gate confirms correctness and no read-count
  regression, but it does not show a meaningful tail-stall improvement. Proceed
  to n84 long decode.
- 2026-06-15: First n84 A/B completed:

  | label | offset sort | eval tok/s | total ms | io_uring reads | io_uring wait us | wait calls | inflight avg/max | VRAM hit | read failures | log |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | `baseline-nosort-n84` | no | `1.73` | `68574.03` | `62582` | `51364174` | `62514` | `2.38/8` | `58.1%` | `0` | `runs/strict2gb-validation/baseline-nosort-n84/` |
  | `offset-sort-n84` | yes | `1.74` | `68249.45` | `62582` | `51730280` | `62498` | `2.38/8` | `58.1%` | `0` | `runs/strict2gb-validation/offset-sort-n84/` |

  Interpretation: first long run shows only `+0.6%` eval-token-rate change and
  no read-count change; aggregate wait time is slightly worse. This does not
  satisfy the `5-10%` target. Repeat once to check noise/stability.
- 2026-06-15: n84 repeat completed:

  | label | offset sort | eval tok/s | total ms | io_uring reads | io_uring wait us | wait calls | inflight avg/max | VRAM hit | read failures | log |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | `baseline-nosort-n84-repeat1` | no | `1.70` | `69543.50` | `62582` | `53840354` | `62482` | `2.38/8` | `58.1%` | `0` | `runs/strict2gb-validation/baseline-nosort-n84-repeat1/` |
  | `offset-sort-n84-repeat1` | yes | `1.73` | `70732.79` | `62582` | `52366939` | `62490` | `2.38/8` | `58.1%` | `0` | `runs/strict2gb-validation/offset-sort-n84-repeat1/` |

### 2026-06-15 Result

Across the two n84 pairs:

| group | runs | avg eval tok/s | avg total ms | avg io_uring reads | avg io_uring wait us | VRAM hit | read failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline, no offset sort | `2` | `1.715` | `69058.77` | `62582` | `52602264` | `58.1%` | `0` |
| scheduler offset sort | `2` | `1.735` | `69491.12` | `62582` | `52048609.5` | `58.1%` | `0` |

Measured deltas:

- eval token rate: `+1.17%`;
- aggregate io_uring wait time: `-1.05%`;
- total runtime: `+0.63%` worse;
- read count: unchanged;
- VRAM hit: unchanged;
- io_uring batch shape: unchanged (`inflight_avg=2.38`, `inflight_max=8`);
- correctness/safety: all runs returned `0`, `read_failures=0`.

Conclusion:

- The conservative scheduler/read-order primitive is safe under strict 2GB RAM,
  but it does **not** achieve the validation goal.
- It does not stably reduce tail stalls in a meaningful way: the only positive
  tail proxy is about `1%` lower aggregate wait time, while total runtime does
  not improve.
- It does not reach the requested `5-10%` real eval token-rate improvement.
- The limiting factor remains the unchanged `62582` runtime expert reads and
  dependency-limited small io_uring batches, not only within-batch read order.

Recommended next step:

- Do not promote offset sorting as the scheduler solution by itself.
- Keep it as a low-risk optional I/O ordering knob.
- Further scheduler work should target reducing miss count or increasing
  cross-dependency batching without extra reads; replacement/order-only tweaks
  at the current batch shape are unlikely to meet the `5-10%` target.
