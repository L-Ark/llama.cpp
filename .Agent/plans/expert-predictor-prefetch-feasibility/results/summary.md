# Expert Predictor Prefetch Feasibility Results

## Trace Collection

- Collected `20` train route traces and `10` held-out test route traces.
- Config: strict `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`, top8
  non-oracle profile, `VRAM cache=12288 MiB`, `upgate_pct=60`,
  `profile_reserve_pct=10`, `backend=iouring`, `predict_tokens=8`.
- Every trace has `12624` row events, reconstructed as `7` eval-token blocks
  of `1800` row events.
- Aggregate run metrics over 30 traces:
  - average `eval tok/s = 1.0623`;
  - average `VRAM hit = 9.18%`;
  - total `io_uring_reads = 343983`;
  - `read_failures = 0`.

Primary artifacts:

- Prompt suite: `prompts.json`
- Trace manifest: `notes/trace_collection_manifest.json`
- Train traces: `traces/train/*.route.csv`
- Test traces: `traces/test/*.route.csv`
- Full offline result: `results/offline_predictors.json`

## Predictor Recall

Held-out, lookahead `1` token:

| predictor | M | recall | precision | wasted predictions |
| --- | ---: | ---: | ---: | ---: |
| recent-hot-w4 | 16 | 52.18% | 30.68% | 42446 |
| recent-hot-w8 | 16 | 51.39% | 29.94% | 43280 |
| last-token-repeat | 8 | 46.81% | 46.81% | 19148 |
| recent-hot-w4 | 8 | 39.73% | 39.73% | 21698 |
| recent-hot-w8 | 8 | 39.03% | 39.03% | 21951 |
| recent-hot-w4 | 4 | 24.93% | 49.86% | 9025 |
| last-token-repeat | 4 | 23.53% | 47.07% | 9528 |

Interpretation:

- There is real short-range structure.
- Practical `M=4` is only about `23-25%` recall.
- `M=8` reaches about `39-47%` recall, but predicts many rows.
- `M=16` reaches `51-52%` for recent-hot but with high waste and low precision.
- The implemented exact-signature layer-transition baseline produced no hits;
  the signature is too sparse for this small train set.

## Cache Replay

Baseline simulated hit rates on held-out traces:

| budget | baseline hit | baseline misses |
| ---: | ---: | ---: |
| 8192 MiB | 8.99% | 114674 |
| 12288 MiB | 9.04% | 114606 |
| 14336 MiB | 22.89% | 97157 |
| 20480 MiB | 53.61% | 58457 |

Best non-oracle replay rows:

| budget | predictor | M | hit delta | miss reduction | extra prefetch reads |
| ---: | --- | ---: | ---: | ---: | ---: |
| 14336 MiB | last-token-repeat | 4 | +6.69 pp | 8.67% | 2603 |
| 12288 MiB | global-hot-per-layer | 4 | +7.60 pp | 8.35% | 42385 |
| 12288 MiB | recent-hot-w8 | 4 | +7.48 pp | 8.23% | 29131 |
| 12288 MiB | last-token-repeat | 4 | +7.28 pp | 8.01% | 25779 |
| 20480 MiB | global-hot-per-layer | 8 | +1.92 pp | 4.13% | 3198 |

Oracle upper-bound replay remains much higher, for example:

- `20480 MiB`, oracle `M=8`, lookahead `1`: `77.74%` miss reduction.
- `14336 MiB`, oracle `M=4`, lookahead `1`: `35.90%` miss reduction.

This confirms the replay machinery can show large gains when the future is
known, but the simple non-oracle predictors do not capture enough of it.

## Decision

Stop before online shadow mode and online prefetch.

Reason:

- Phase 2 recall has some signal, but practical widths do not give enough
  coverage with low waste.
- Phase 3 cache replay does not meet either gate:
  - no non-oracle predictor gives `+15%-20%` absolute hit-rate improvement;
  - no non-oracle predictor gives `25%+` simulated miss reduction.
- Several configurations require tens of thousands of extra prefetch reads for
  only about `8%` miss reduction, which is unlikely to improve real eval
  tok/s on the current I/O-bound runtime.

Recommended next step is not online prefetch yet. A better predictor feature
source is needed first, such as router-score/top-k traces, hidden-state-derived
features, or a prompt-adaptive profile trained on more diverse traces.
