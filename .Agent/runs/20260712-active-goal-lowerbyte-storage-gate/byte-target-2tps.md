# Kimi byte-reduction target bound

Evidence scope: `current-copyio-dev-generalized`.

This is a bound report only. It must not be used to select prompt-specific experts or tune held-out prompts.

- target token rate: `2.00 tok/s`
- optimistic sustained movement bandwidth: `10.30 GiB/s`

## Prompt Bound

| prompt | tok/s | decode tok | decode s | moved GiB | moved GiB/token | active GiB/token | eff GiB/s | transfer-only ratio @ peak | floor ms/token | IO budget GiB/token @ peak | ratio @ peak after floor | reduction @ peak after floor | ratio @ current eff after floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | 1.42 | 31 | 21.84 | 207.35 | 6.69 | 13.21 | 9.49 | 0.77x | 233.4 | 2.75 | 0.41x | 58.9% | 0.38x |
| `dev_intelligence_general` | 1.50 | 31 | 20.69 | 198.18 | 6.39 | 12.15 | 9.58 | 0.81x | 134.6 | 3.76 | 0.59x | 41.1% | 0.55x |

## Aggregate Bound

- median moved bytes: `6.54 GiB/token`; mean `6.54 GiB/token`.
- median active expert footprint before cache: `12.68 GiB/token`; mean `12.68 GiB/token`.
- median all-hit MoE floor estimate: `184.0 ms/token`; mean `184.0 ms/token`.
- median required byte ratio at `10.30 GiB/s` after floor: `0.50x`; worst `0.41x`.

## Miss Bytes By Role

| role | miss GiB | misses | share | estimated rows |
|---|---:|---:|---:|---:|
| down | 124.69 | 18872 | 35.3% | 0/2042 |
| gate | 118.81 | 24054 | 33.6% | 0/3824 |
| up | 109.94 | 24048 | 31.1% | 0/3824 |

## Top Layer/Role Miss Bytes

| layer | role | miss GiB | misses | share | estimated rows |
|---:|---|---:|---:|---:|---:|
| 4 | down | 3.95 | 544 | 1.1% | 0/64 |
| 7 | down | 3.91 | 509 | 1.1% | 0/64 |
| 6 | down | 3.81 | 495 | 1.1% | 0/64 |
| 9 | down | 3.77 | 490 | 1.1% | 0/64 |
| 26 | down | 3.72 | 512 | 1.1% | 0/64 |
| 10 | down | 3.63 | 472 | 1.0% | 0/64 |
| 20 | down | 3.52 | 485 | 1.0% | 0/64 |
| 23 | down | 3.52 | 484 | 1.0% | 0/64 |
| 24 | down | 3.46 | 476 | 1.0% | 0/64 |
| 18 | down | 3.44 | 447 | 1.0% | 0/64 |
| 58 | down | 3.40 | 468 | 1.0% | 0/64 |
| 8 | down | 3.38 | 439 | 1.0% | 0/64 |
| 15 | down | 3.32 | 432 | 0.9% | 0/64 |
| 25 | down | 3.32 | 457 | 0.9% | 0/64 |
| 16 | down | 3.28 | 451 | 0.9% | 0/64 |
| 19 | down | 3.28 | 451 | 0.9% | 0/64 |
| 57 | down | 3.28 | 451 | 0.9% | 0/64 |
| 3 | down | 3.27 | 557 | 0.9% | 0/64 |
| 21 | down | 3.22 | 444 | 0.9% | 0/64 |
| 5 | down | 3.21 | 546 | 0.9% | 0/64 |
| 22 | down | 3.19 | 439 | 0.9% | 0/64 |
| 1 | down | 3.08 | 524 | 0.9% | 0/64 |
| 27 | down | 2.93 | 499 | 0.8% | 0/64 |
| 2 | down | 2.90 | 494 | 0.8% | 0/64 |
| 59 | down | 2.84 | 483 | 0.8% | 0/64 |
| 56 | down | 2.83 | 482 | 0.8% | 0/64 |
| 12 | down | 2.78 | 473 | 0.8% | 0/64 |
| 13 | down | 2.53 | 431 | 0.7% | 0/64 |
| 3 | gate | 2.52 | 482 | 0.7% | 0/64 |
| 3 | up | 2.52 | 482 | 0.7% | 0/64 |

## Interpretation

- The peak-bandwidth ratio is an optimistic lower bound: it assumes expert movement is the only remaining bottleneck
  and that runtime can sustain the pure IO bench bandwidth during decode.
- The after-floor ratio is stricter: it reserves time for an optimistic all-hit MoE wall estimate before assigning
  the remaining token budget to expert movement.
- If the required byte ratio is far below 1.0, queue tuning and prediction cannot be enough; expert bytes must shrink.
- When down/upgate miss profiles are absent, role/layer miss bytes are estimated from route trace bytes
  scaled by the prompt-level VRAM cache hit rates in metrics.json.
- A viable representation change must preserve quality first, then reduce moved expert bytes by the required ratio on dev before held-out testing.
