# Kimi byte-reduction target bound

Evidence scope: `current HEAD 564b887 n96 dev_france/intelligence plus France endpoint`.

This is a bound report only. It must not be used to select prompt-specific experts or tune held-out prompts.

- target token rate: `2.00 tok/s`
- optimistic sustained movement bandwidth: `10.40 GiB/s`

## Prompt Bound

| prompt | tok/s | decode tok | decode s | moved GiB | moved GiB/token | active GiB/token | eff GiB/s | transfer-only ratio @ peak | floor ms/token | IO budget GiB/token @ peak | ratio @ peak after floor | reduction @ peak after floor | ratio @ current eff after floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | 1.56 | 85 | 54.61 | 452.62 | 5.32 | 10.36 | 8.29 | 0.98x | 198.0 | 3.14 | 0.59x | 41.0% | 0.47x |
| `dev_france_regression_iobatch` | 1.72 | 85 | 49.51 | 452.62 | 5.32 | 10.36 | 9.14 | 0.98x | 163.1 | 3.50 | 0.66x | 34.2% | 0.58x |
| `dev_intelligence_general` | 1.58 | 95 | 60.06 | 482.88 | 5.08 | 9.84 | 8.04 | 1.00x | 163.0 | 3.51 | 0.69x | 31.0% | 0.53x |

## Aggregate Bound

- median moved bytes: `5.32 GiB/token`; mean `5.24 GiB/token`.
- median active expert footprint before cache: `10.36 GiB/token`; mean `10.18 GiB/token`.
- median all-hit MoE floor estimate: `163.1 ms/token`; mean `174.7 ms/token`.
- median required byte ratio at `10.40 GiB/s` after floor: `0.66x`; worst `0.59x`.

## Miss Bytes By Role

| role | miss GiB | misses | share | estimated rows |
|---|---:|---:|---:|---:|
| gate | 405.06 | 82166 | 34.6% | 0/15999 |
| down | 389.01 | 57973 | 33.2% | 0/8392 |
| up | 376.44 | 82164 | 32.2% | 0/16000 |

## Top Layer/Role Miss Bytes

| layer | role | miss GiB | misses | share | estimated rows |
|---:|---|---:|---:|---:|---:|
| 4 | down | 14.30 | 1969 | 1.2% | 0/268 |
| 7 | down | 13.76 | 1789 | 1.2% | 0/268 |
| 6 | down | 13.63 | 1772 | 1.2% | 0/268 |
| 10 | down | 13.37 | 1738 | 1.1% | 0/268 |
| 9 | down | 13.23 | 1720 | 1.1% | 0/268 |
| 8 | down | 12.50 | 1625 | 1.1% | 0/268 |
| 24 | down | 12.22 | 1682 | 1.0% | 0/268 |
| 25 | down | 12.22 | 1682 | 1.0% | 0/268 |
| 26 | down | 12.09 | 1664 | 1.0% | 0/268 |
| 23 | down | 12.04 | 1658 | 1.0% | 0/268 |
| 20 | down | 11.79 | 1623 | 1.0% | 0/268 |
| 58 | down | 11.64 | 1602 | 1.0% | 0/268 |
| 18 | down | 11.60 | 1509 | 1.0% | 0/268 |
| 19 | down | 11.56 | 1592 | 1.0% | 0/268 |
| 15 | down | 11.47 | 1492 | 1.0% | 0/268 |
| 5 | down | 11.44 | 1947 | 1.0% | 0/268 |
| 21 | down | 11.40 | 1570 | 1.0% | 0/268 |
| 1 | down | 11.40 | 1941 | 1.0% | 0/268 |
| 57 | down | 11.29 | 1555 | 1.0% | 0/268 |
| 3 | down | 11.21 | 1908 | 1.0% | 0/268 |
| 16 | down | 11.16 | 1537 | 1.0% | 0/268 |
| 22 | down | 11.05 | 1522 | 0.9% | 0/268 |
| 2 | down | 10.61 | 1806 | 0.9% | 0/268 |
| 12 | down | 9.92 | 1688 | 0.8% | 0/268 |
| 27 | down | 9.74 | 1658 | 0.8% | 0/268 |
| 59 | down | 9.62 | 1637 | 0.8% | 0/268 |
| 5 | gate | 9.18 | 1754 | 0.8% | 0/268 |
| 5 | up | 9.18 | 1754 | 0.8% | 0/268 |
| 4 | up | 9.05 | 1729 | 0.8% | 0/268 |
| 4 | gate | 9.04 | 1727 | 0.8% | 0/268 |

## Interpretation

- The peak-bandwidth ratio is an optimistic lower bound: it assumes expert movement is the only remaining bottleneck
  and that runtime can sustain the pure IO bench bandwidth during decode.
- The after-floor ratio is stricter: it reserves time for an optimistic all-hit MoE wall estimate before assigning
  the remaining token budget to expert movement.
- If the required byte ratio is far below 1.0, queue tuning and prediction cannot be enough; expert bytes must shrink.
- When down/upgate miss profiles are absent, role/layer miss bytes are estimated from route trace bytes
  scaled by the prompt-level VRAM cache hit rates in metrics.json.
- A viable representation change must preserve quality first, then reduce moved expert bytes by the required ratio on dev before held-out testing.
