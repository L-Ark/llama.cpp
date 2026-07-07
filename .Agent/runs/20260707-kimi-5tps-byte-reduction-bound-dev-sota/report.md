# Kimi byte-reduction target bound

Evidence scope: `current-sota-dev-profile-only`.

This is a bound report only. It must not be used to select prompt-specific experts or tune held-out prompts.

- target token rate: `5.00 tok/s`
- optimistic sustained movement bandwidth: `10.40 GiB/s`

## Prompt Bound

| prompt | tok/s | decode tok | decode s | moved GiB | moved GiB/token | active GiB/token | eff GiB/s | transfer-only ratio @ peak | floor ms/token | IO budget GiB/token @ peak | ratio @ peak after floor | reduction @ peak after floor | ratio @ current eff after floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | 1.40 | 77 | 54.87 | 308.38 | 4.00 | 8.10 | 5.62 | 0.52x | 0.0 | 2.08 | 0.52x | 48.1% | 0.28x |
| `dev_japan_factual` | 1.43 | 85 | 59.60 | 336.58 | 3.96 | 8.10 | 5.65 | 0.53x | 0.0 | 2.08 | 0.53x | 47.5% | 0.29x |
| `dev_linear_equation` | 1.10 | 34 | 30.96 | 181.08 | 5.33 | 8.10 | 5.85 | 0.39x | 0.0 | 2.08 | 0.39x | 60.9% | 0.22x |
| `dev_mixed_summary` | 1.34 | 54 | 40.41 | 246.62 | 4.57 | 8.10 | 6.10 | 0.46x | 0.0 | 2.08 | 0.46x | 54.5% | 0.27x |
| `dev_photosynthesis_factual` | 1.35 | 94 | 69.56 | 400.17 | 4.26 | 8.10 | 5.75 | 0.49x | 0.0 | 2.08 | 0.49x | 51.1% | 0.27x |
| `dev_python_reverse` | 1.29 | 95 | 73.82 | 440.16 | 4.63 | 8.10 | 5.96 | 0.45x | 0.0 | 2.08 | 0.45x | 55.1% | 0.26x |
| `dev_zh_france` | 1.41 | 49 | 34.70 | 185.72 | 3.79 | 8.10 | 5.35 | 0.55x | 0.0 | 2.08 | 0.55x | 45.1% | 0.28x |

## Aggregate Bound

- median moved bytes: `4.26 GiB/token`; mean `4.36 GiB/token`.
- median active expert footprint before cache: `8.10 GiB/token`; mean `8.10 GiB/token`.
- median all-hit MoE floor estimate: `0.0 ms/token`; mean `0.0 ms/token`.
- median required byte ratio at `10.40 GiB/s` after floor: `0.49x`; worst `0.39x`.

## Miss Bytes By Role

| role | miss GiB | misses | share | estimated rows |
|---|---:|---:|---:|---:|
| gate | 691.99 | 140135 | 40.8% | 420/420 |
| up | 640.08 | 140135 | 37.7% | 420/420 |
| down | 365.59 | 59111 | 21.5% | 371/371 |

## Top Layer/Role Miss Bytes

| layer | role | miss GiB | misses | share | estimated rows |
|---:|---|---:|---:|---:|---:|
| 60 | up | 12.41 | 2370 | 0.7% | 7/7 |
| 60 | gate | 12.41 | 2370 | 0.7% | 7/7 |
| 2 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 2 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 3 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 3 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 4 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 4 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 5 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 5 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 6 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 6 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 7 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 8 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 9 | up | 12.23 | 2335 | 0.7% | 7/7 |
| 14 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 28 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 29 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 30 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 31 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 32 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 33 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 34 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 35 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 36 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 37 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 38 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 39 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 40 | gate | 12.23 | 2335 | 0.7% | 7/7 |
| 41 | gate | 12.23 | 2335 | 0.7% | 7/7 |

## Interpretation

- The peak-bandwidth ratio is an optimistic lower bound: it assumes expert movement is the only remaining bottleneck
  and that runtime can sustain the pure IO bench bandwidth during decode.
- The after-floor ratio is stricter: it reserves time for an optimistic all-hit MoE wall estimate before assigning
  the remaining token budget to expert movement.
- If the required byte ratio is far below 1.0, queue tuning and prediction cannot be enough; expert bytes must shrink.
- When down/upgate miss profiles are absent, role/layer miss bytes are estimated from route trace bytes
  scaled by the prompt-level VRAM cache hit rates in metrics.json.
- A viable representation change must preserve quality first, then reduce moved expert bytes by the required ratio on dev before held-out testing.
