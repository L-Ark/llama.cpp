# GP65 Planned Host-Prefetch N32 Analysis

Timestamp: `2026-07-07T18:32:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Purpose:

- Test whether the existing `GGML_MOE_PLANNED_HOST_PREFETCH=1` path can improve
  the current GP4-env prompt-general SOTA before implementing the deeper GP65
  next-layer prefetch path.
- This is a dev-only check. No held-out prompts were used.

Run roots:

- Baseline:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n32-baseline`
- Planned host-prefetch:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-planned-host-prefetch-n32`

Required GP4 env:

```text
GGML_MOE_EXPERT_GGUF_ALIAS_TSV=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv
GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1
```

Planned host-prefetch env:

```text
GGML_MOE_PLANNED_HOST_PREFETCH=1
GGML_MOE_HOST_PREFETCH_SLOTS=32
GGML_MOE_HOST_PREFETCH_MAX_MIB=512
```

Result summary:

| prompt | baseline tok/s | prefetch tok/s | baseline decode ms | prefetch decode ms | quality |
| --- | ---: | ---: | ---: | ---: | --- |
| `dev_france_regression` | 1.69 | 1.65 | 18326.06 | 18753.90 | pass |
| `dev_japan_factual` | 1.65 | 1.62 | 18776.77 | 19193.82 | pass |

Both runs stayed within the 16GB cgroup and kept `direct_reads=0`, but planned
host-prefetch regressed decode time on both prompts.

Key counters:

| prompt | submitted | useful hits | misses | evicted | read failures | host-prefetch bytes | host-prefetch wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | 9172 | 32 | 23469 | 9108 | 4099 | 40.516 GiB | 11694.968 ms |
| `dev_japan_factual` | 9194 | 45 | 23715 | 9117 | 4174 | 40.607 GiB | 13776.394 ms |

Demand-path expert-pack reads did not fall materially:

| prompt | baseline iouring bytes | prefetch iouring bytes | baseline wait | prefetch wait |
| --- | ---: | ---: | ---: | ---: |
| `dev_france_regression` | 131.120 GB | 130.951 GB | 12237.546 ms | 12632.243 ms |
| `dev_japan_factual` | 132.609 GB | 132.373 GB | 12278.905 ms | 12957.835 ms |

Profile deltas:

| prompt | profile | baseline wall | prefetch wall | delta |
| --- | --- | ---: | ---: | ---: |
| `dev_france_regression` | up/gate | 4838.308 ms | 5144.769 ms | +306.461 ms |
| `dev_france_regression` | down | 3534.729 ms | 3647.490 ms | +112.761 ms |
| `dev_japan_factual` | up/gate | 5099.989 ms | 5019.401 ms | -80.588 ms |
| `dev_japan_factual` | down | 3718.545 ms | 3753.681 ms | +35.136 ms |

Root cause:

- The existing planned host-prefetch producer is too late and too weakly tied to
  actual future demand. It enqueues planned jobs close to demand copy creation,
  so most prefetched entries are either not ready when needed or are evicted
  before a matching demand miss consumes them.
- Useful hits are effectively zero relative to submitted work:
  `32/9172` and `45/9194`.
- The prefetch worker reads about `40.5 GiB` per prompt into host-prefetch slots
  without reducing demand-path iouring bytes or H2D enqueues. This extra work
  competes with current-layer expert-pack reads and H2D/copy scheduling.
- The slower decode is therefore not caused by output quality, TTFT, missing
  GP4 alias env, or increased GGUF direct fallback. It is caused by wasted
  planned host-prefetch IO plus cache/slot churn and demand-path contention.

Decision:

- Reject `GGML_MOE_PLANNED_HOST_PREFETCH=1` as a SOTA candidate.
- Do not continue this N32 sweep or run held-out tests with this env.
- A valid GP65 runtime prefetch still needs a predictor that becomes available
  early enough in the CUDA/MoE execution path, plus hard backpressure and
  useful-hit accounting. Generic planned host-prefetch is not sufficient.
