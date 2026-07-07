# Kimi byte-reduction target bound

Evidence scope: `accepted-heldout-sota-profile`.

This is a bound report only. It must not be used to select prompt-specific experts or tune held-out prompts.

- target token rate: `5.00 tok/s`
- optimistic sustained movement bandwidth: `10.40 GiB/s`

## Prompt Bound

| prompt | tok/s | decode tok | decode s | moved GiB | moved GiB/token | active GiB/token | eff GiB/s | transfer-only ratio @ peak | floor ms/token | IO budget GiB/token @ peak | ratio @ peak after floor | reduction @ peak after floor | ratio @ current eff after floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | 1.44 | 95 | 66.01 | 377.96 | 3.98 | 8.10 | 5.73 | 0.52x | 35.9 | 1.71 | 0.43x | 57.1% | 0.24x |
| `test_coding_01` | 1.14 | 95 | 83.68 | 505.74 | 5.32 | 8.10 | 6.04 | 0.39x | 46.0 | 1.60 | 0.30x | 69.9% | 0.17x |
| `test_english_factual_01` | 1.48 | 95 | 64.33 | 372.85 | 3.92 | 8.10 | 5.80 | 0.53x | 41.5 | 1.65 | 0.42x | 58.0% | 0.23x |
| `test_english_factual_02` | 1.49 | 94 | 63.01 | 394.56 | 4.20 | 8.10 | 6.26 | 0.50x | 40.7 | 1.66 | 0.39x | 60.5% | 0.24x |
| `test_mixed_instruction_01` | 1.33 | 95 | 71.38 | 405.15 | 4.26 | 8.10 | 5.68 | 0.49x | 40.3 | 1.66 | 0.39x | 61.1% | 0.21x |
| `test_reasoning_math_01` | 1.32 | 55 | 41.77 | 255.12 | 4.64 | 8.10 | 6.11 | 0.45x | 36.1 | 1.71 | 0.37x | 63.2% | 0.22x |

## Aggregate Bound

- median moved bytes: `4.23 GiB/token`; mean `4.39 GiB/token`.
- median active expert footprint before cache: `8.10 GiB/token`; mean `8.10 GiB/token`.
- median all-hit MoE floor estimate: `40.5 ms/token`; mean `40.1 ms/token`.
- median required byte ratio at `10.40 GiB/s` after floor: `0.39x`; worst `0.30x`.

## Miss Bytes By Role

| role | miss GiB | misses | share | estimated rows |
|---|---:|---:|---:|---:|
| down | 426.10 | 65094 | 37.6% | 0/12754 |
| up | 352.90 | 74909 | 31.2% | 0/14786 |
| gate | 352.89 | 74907 | 31.2% | 0/14786 |

## Top Layer/Role Miss Bytes

| layer | role | miss GiB | misses | share | estimated rows |
|---:|---|---:|---:|---:|---:|
| 4 | down | 22.47 | 3094 | 2.0% | 0/529 |
| 57 | down | 20.37 | 2805 | 1.8% | 0/529 |
| 58 | down | 20.35 | 2802 | 1.8% | 0/525 |
| 25 | down | 19.57 | 2695 | 1.7% | 0/529 |
| 5 | down | 19.28 | 3282 | 1.7% | 0/529 |
| 19 | down | 18.73 | 2579 | 1.7% | 0/527 |
| 22 | down | 18.48 | 2545 | 1.6% | 0/528 |
| 24 | down | 18.46 | 2541 | 1.6% | 0/527 |
| 20 | down | 18.35 | 2526 | 1.6% | 0/528 |
| 3 | down | 18.32 | 3118 | 1.6% | 0/529 |
| 16 | down | 18.22 | 2508 | 1.6% | 0/527 |
| 23 | down | 18.19 | 2504 | 1.6% | 0/525 |
| 60 | down | 18.18 | 3095 | 1.6% | 0/535 |
| 21 | down | 17.95 | 2472 | 1.6% | 0/529 |
| 1 | down | 17.88 | 3044 | 1.6% | 0/528 |
| 26 | down | 17.44 | 2401 | 1.5% | 0/528 |
| 5 | gate | 17.16 | 3278 | 1.5% | 0/529 |
| 5 | up | 17.15 | 3277 | 1.5% | 0/529 |
| 59 | down | 16.98 | 2890 | 1.5% | 0/527 |
| 3 | up | 16.22 | 3100 | 1.4% | 0/529 |
| 3 | gate | 16.22 | 3100 | 1.4% | 0/529 |
| 4 | up | 16.19 | 3093 | 1.4% | 0/529 |
| 4 | gate | 16.19 | 3093 | 1.4% | 0/529 |
| 60 | up | 16.15 | 3086 | 1.4% | 0/535 |
| 60 | gate | 16.15 | 3086 | 1.4% | 0/535 |
| 2 | down | 15.98 | 2721 | 1.4% | 0/528 |
| 12 | down | 15.69 | 2670 | 1.4% | 0/529 |
| 11 | down | 15.44 | 2629 | 1.4% | 0/527 |
| 59 | up | 15.13 | 2891 | 1.3% | 0/527 |
| 59 | gate | 15.13 | 2890 | 1.3% | 0/527 |

## Interpretation

- The peak-bandwidth ratio is an optimistic lower bound: it assumes expert movement is the only remaining bottleneck
  and that runtime can sustain the pure IO bench bandwidth during decode.
- The after-floor ratio is stricter: it reserves time for an optimistic all-hit MoE wall estimate before assigning
  the remaining token budget to expert movement.
- If the required byte ratio is far below 1.0, queue tuning and prediction cannot be enough; expert bytes must shrink.
- When down/upgate miss profiles are absent, role/layer miss bytes are estimated from route trace bytes
  scaled by the prompt-level VRAM cache hit rates in metrics.json.
- A viable representation change must preserve quality first, then reduce moved expert bytes by the required ratio on dev before held-out testing.
