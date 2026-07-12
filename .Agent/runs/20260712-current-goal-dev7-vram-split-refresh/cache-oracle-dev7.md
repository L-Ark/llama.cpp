# Kimi dev route-cache oracle bound

This is an offline bound from dev route traces. It does not use held-out test prompts.

| prompt | group | slots | events | unique | current hit | global LFU hit | prompt LFU hit | Belady hit | current miss GiB | global miss GiB | prompt miss GiB | Belady miss GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | upgate | 2015 | 97664 | 21562 | 44.9% | 36.7% | 46.4% | 63.4% | 255.52 | 294.42 | 249.20 | 170.29 |
| `dev_france_regression` | down | 533 | 48832 | 10781 | 61.2% | 27.0% | 34.4% | 54.5% | 120.61 | 227.00 | 204.05 | 141.68 |
| `dev_japan_factual` | upgate | 2015 | 90944 | 21274 | 46.3% | 35.3% | 46.3% | 64.4% | 231.89 | 279.79 | 232.35 | 154.02 |
| `dev_japan_factual` | down | 533 | 45472 | 10637 | 62.1% | 24.9% | 34.2% | 55.4% | 109.70 | 217.83 | 190.93 | 129.32 |
| `dev_linear_equation` | upgate | 2015 | 55312 | 18546 | 39.4% | 19.4% | 42.6% | 57.1% | 159.09 | 212.17 | 151.22 | 112.79 |
| `dev_linear_equation` | down | 533 | 27656 | 9273 | 56.2% | 11.4% | 30.7% | 49.9% | 77.13 | 155.36 | 121.40 | 87.93 |
| `dev_mixed_summary` | upgate | 2015 | 69712 | 20614 | 42.5% | 21.8% | 41.1% | 58.4% | 190.28 | 258.98 | 195.54 | 138.12 |
| `dev_mixed_summary` | down | 533 | 34856 | 10307 | 58.7% | 13.3% | 29.7% | 51.3% | 91.65 | 192.29 | 155.94 | 107.89 |
| `dev_photosynthesis_factual` | upgate | 2015 | 103472 | 22894 | 44.2% | 29.8% | 45.3% | 62.4% | 274.17 | 344.96 | 269.01 | 185.04 |
| `dev_photosynthesis_factual` | down | 533 | 51736 | 11447 | 61.9% | 20.3% | 34.3% | 53.7% | 125.47 | 262.53 | 216.41 | 152.33 |
| `dev_python_reverse` | upgate | 2015 | 108208 | 26704 | 38.4% | 17.9% | 35.6% | 57.6% | 316.51 | 422.51 | 332.17 | 218.74 |
| `dev_python_reverse` | down | 533 | 54104 | 13352 | 61.1% | 10.5% | 25.5% | 49.2% | 133.97 | 307.20 | 255.30 | 174.11 |
| `dev_zh_france` | upgate | 2015 | 61216 | 17156 | 48.5% | 34.0% | 49.0% | 64.0% | 149.68 | 192.41 | 148.39 | 104.74 |
| `dev_zh_france` | down | 533 | 30608 | 8578 | 61.0% | 23.9% | 36.5% | 56.2% | 75.99 | 148.39 | 124.30 | 85.62 |

Interpretation:

- `global LFU` is the best fixed dev-wide frequency hotset with the current slot counts.
- `prompt LFU` is an oracle upper bound for a prompt-adaptive static hotset.
- `Belady` is an offline replacement upper bound for the same trace and capacity.
- If prompt LFU/Belady materially beat global LFU on slow prompts, fixed
  prompt-agnostic cache is insufficient and runtime-adaptive or byte-reduction
  work should be prioritized.
