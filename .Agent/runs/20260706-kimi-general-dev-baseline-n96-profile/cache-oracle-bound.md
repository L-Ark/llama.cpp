# Kimi dev route-cache oracle bound

This is an offline bound from dev route traces. It does not use held-out test prompts.

| prompt | group | slots | events | unique | current hit | global LFU hit | prompt LFU hit | Belady hit | current miss GiB | global miss GiB | prompt miss GiB | Belady miss GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | upgate | 1735 | 73936 | 16270 | 44.1% | 38.3% | 48.8% | 63.1% | 196.30 | 217.21 | 179.95 | 129.97 |
| `dev_france_regression` | down | 766 | 32656 | 7086 | 73.0% | 38.8% | 49.3% | 63.6% | 54.57 | 123.62 | 102.51 | 73.68 |
| `dev_japan_factual` | upgate | 1735 | 81616 | 17294 | 44.7% | 37.1% | 48.6% | 64.0% | 214.37 | 244.54 | 199.62 | 140.01 |
| `dev_japan_factual` | down | 766 | 36048 | 7514 | 73.3% | 37.6% | 49.1% | 64.5% | 59.57 | 139.23 | 113.72 | 79.29 |
| `dev_linear_equation` | upgate | 1735 | 32656 | 12574 | 26.0% | 12.9% | 44.5% | 52.3% | 114.78 | 135.23 | 86.26 | 74.06 |
| `dev_linear_equation` | down | 766 | 14424 | 5566 | 65.8% | 12.3% | 44.2% | 52.2% | 30.53 | 78.11 | 49.71 | 42.68 |
| `dev_mixed_summary` | upgate | 1735 | 51856 | 15262 | 36.3% | 19.5% | 45.1% | 56.8% | 156.89 | 198.43 | 135.56 | 106.71 |
| `dev_mixed_summary` | down | 766 | 22904 | 6681 | 69.5% | 19.5% | 45.7% | 57.4% | 43.23 | 114.07 | 76.86 | 60.38 |
| `dev_photosynthesis_factual` | upgate | 1735 | 90256 | 19686 | 40.9% | 29.5% | 46.1% | 61.3% | 253.35 | 302.36 | 231.58 | 166.39 |
| `dev_photosynthesis_factual` | down | 766 | 39864 | 8703 | 71.9% | 29.4% | 46.1% | 61.3% | 69.33 | 174.18 | 133.00 | 95.49 |
| `dev_python_reverse` | upgate | 1735 | 91216 | 23932 | 35.8% | 14.9% | 34.3% | 55.7% | 278.14 | 368.93 | 285.62 | 192.70 |
| `dev_python_reverse` | down | 766 | 40288 | 10711 | 69.9% | 14.0% | 33.3% | 55.0% | 75.05 | 214.06 | 166.20 | 112.10 |
| `dev_zh_france` | upgate | 1735 | 47056 | 12188 | 47.1% | 35.7% | 54.1% | 64.5% | 118.23 | 144.41 | 102.54 | 79.32 |
| `dev_zh_france` | down | 766 | 20784 | 5297 | 74.1% | 36.3% | 54.9% | 65.1% | 33.32 | 81.98 | 57.98 | 44.94 |

Interpretation:

- `global LFU` is the best fixed dev-wide frequency hotset with the current slot counts.
- `prompt LFU` is an oracle upper bound for a prompt-adaptive static hotset.
- `Belady` is an offline replacement upper bound for the same trace and capacity.
- If prompt LFU/Belady materially beat global LFU on slow prompts, fixed
  prompt-agnostic cache is insufficient and runtime-adaptive or byte-reduction
  work should be prioritized.
