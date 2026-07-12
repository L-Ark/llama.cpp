# Kimi Small Static RAM-Tier Admission

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `77343bd99`

## Purpose

Close the next RAM-tier gate before running more runtime A/B tests. The test
asks whether a smaller `512 MiB` or `1024 MiB` prompt-agnostic static RAM tier
has enough coverage to justify cold-start runtime validation under the strict
16 GB host-RAM constraint.

This is an offline admission screen only. It does not use held-out prompts and
does not claim a SOTA result.

## Inputs

- profile root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-head-cpudefer-n96-050308`
- dev profiles:
  - `dev_france_regression`
  - `dev_intelligence_general`
- decode runs: `180`
- weighted decode baseline from screen: `651.099 ms/token`, `1.536 tok/s`

## Generated Artifacts

- `.Agent/runs/20260712-current-goal-small-ram-admission/wait-weighted-screen.csv`
- `.Agent/runs/20260712-current-goal-small-ram-admission/wait-weighted-screen.md`
- `.Agent/runs/20260712-current-goal-small-ram-admission/small-ram-512-profile.csv`
- `.Agent/runs/20260712-current-goal-small-ram-admission/small-ram-512-profile.md`
- `.Agent/runs/20260712-current-goal-small-ram-admission/small-ram-1024-profile.csv`
- `.Agent/runs/20260712-current-goal-small-ram-admission/small-ram-1024-profile.md`

## Candidate Summary

| candidate | entries | resident MiB | layer/role buckets | route coverage | copy rows | copy bytes | copy io/wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `512MiB` | `95` | `506.08` | `24` | `2.18%` | `0.81%` | `0.86%` | `0.88%` |
| `1024MiB` | `191` | `1018.28` | `24` | `3.57%` | `1.65%` | `1.73%` | `1.77%` |

Per-prompt copy/wall coverage:

| candidate | prompt | copy io/wall coverage |
|---|---|---:|
| `512MiB` | `dev_france_regression` | `0.93%` |
| `512MiB` | `dev_intelligence_general` | `0.83%` |
| `1024MiB` | `dev_france_regression` | `1.85%` |
| `1024MiB` | `dev_intelligence_general` | `1.69%` |

Optimistic endpoint upper bound:

| candidate | selected copy wall | selected wall/token | optimistic ms/token | optimistic tok/s |
|---|---:|---:|---:|---:|
| `512MiB` | `4966.414 ms` | `27.591 ms/token` | `623.508` | `1.604` |
| `1024MiB` | `10042.437 ms` | `55.791 ms/token` | `595.308` | `1.680` |

The optimistic bound assumes every selected copy-profile wall sample is fully
exposed on the decode critical path and can be removed for free. This is more
favorable than runtime reality because it ignores:

- cold-start preload and TTFT cost;
- extra RAM residency under the 16 GB cap;
- RAM-tier H2D and staging cost;
- possible fragmentation of the remaining SSD/io_uring batches;
- loss of overlap or cache pressure.

## Decision

Reject `512MiB` and `1024MiB` scattered static RAM-tier runtime A/B as the next
optimization step.

Reason:

- route coverage is only `2.18%` and `3.57%`;
- actual copy-profile io/wall coverage is only `0.88%` and `1.77%`;
- even the favorable endpoint upper bound reaches only about `1.60-1.68 tok/s`,
  below the `>2 tok/s` milestone;
- previous larger RAM-tier and whole-layer experiments already showed that
  reducing bytes or increasing hit rate does not reliably reduce generalized
  endpoint time when the remaining SSD reads still keep the queue fragmented.

## Next Action

Do not run another scattered static RAM-tier runtime A/B from this admission.
Move to one of:

1. scheduler-only same-layer up/gate/down read submission that does not add
   host RAM pressure;
2. lower-byte/v2 expert representation after quality-safe admission;
3. explicit RAM/VRAM cache only if it replaces measured low-value file cache
   with batchable layer/role slabs and has a stronger exposed-wait bound.

## Reproduce

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-head-cpudefer-n96-050308
OUT=.Agent/runs/20260712-current-goal-small-ram-admission
mkdir -p "$OUT"

python3 .Agent/run-tools/kimi_wait_weighted_admission_screen.py \
  --profile-root "$ROOT" \
  --out-csv "$OUT/wait-weighted-screen.csv" \
  --out-md "$OUT/wait-weighted-screen.md" \
  --top 80

python3 .Agent/run-tools/kimi_make_budgeted_hot_expert_profile.py \
  --route-root "$ROOT" \
  --screen-csv "$OUT/wait-weighted-screen.csv" \
  --out-profile "$OUT/small-ram-512-profile.csv" \
  --out-report "$OUT/small-ram-512-profile.md" \
  --upgate-budget-mib 384 \
  --down-budget-mib 128 \
  --min-prompts 2 \
  --max-layers-per-role 12

python3 .Agent/run-tools/kimi_make_budgeted_hot_expert_profile.py \
  --route-root "$ROOT" \
  --screen-csv "$OUT/wait-weighted-screen.csv" \
  --out-profile "$OUT/small-ram-1024-profile.csv" \
  --out-report "$OUT/small-ram-1024-profile.md" \
  --upgate-budget-mib 768 \
  --down-budget-mib 256 \
  --min-prompts 2 \
  --max-layers-per-role 12
```
