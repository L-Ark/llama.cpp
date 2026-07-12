# Kimi CPU/defer GPU-extension audit

Input root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-head-cpudefer-n96-050308`
Branch at source run: `vendor/kimi-deepseek-41d205-additive`
Commit at source run: `97d1c177f`
Runs: `2`

## Verdict

- Decision: `fallback_hook_not_next`.
- True CPU fallback rows: `0`; fallback ms: `0.000`.
- Runs with CPU/defer extension miss: `0`.
- Weighted decode: `651.099 ms/token`, `1.536 tok/s`.
- Up/gate GPU-extension wall: `441.220 ms/token`.
- Up/gate exposed wait proxy: `252.286 ms/token`.
- Decode down GPU-extension wall: `171.086 ms/token`.

Interpretation:

- The DeepSeek CPU/defer GPU-extension pattern is already active for Kimi in this profiled SOTA path.
- `up_gate` and `down` CPU/defer ops are accepted by the CUDA batch extension; the fallback CSV is empty.
- The next implementation should not be another broad CPU fallback rewrite.
- The next bottleneck to attack is exposed up/gate staging and io_uring demand-read starvation, followed by down staging.

## Per Prompt

| Prompt | quality | tok/s | decode ms/token | TTFT ms | RAM peak GiB | fallback rows | extension miss | up/gate ms/token | up/gate wait ms/token | down ms/token | down stage ms/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | pass | 1.520 | 657.499 | 11867.620 | 11.930 | 0 | 0 | 442.823 | 256.395 | 171.889 | 160.302 |
| `dev_intelligence_general` | pass | 1.550 | 645.373 | 9917.750 | 11.769 | 0 | 0 | 439.785 | 248.609 | 170.367 | 159.169 |

## CPU/defer Op Acceptance

| Prompt | op | calls | batch accept | batch decline | accept rate | decode calls | prompt calls | decode cuda ms/call | decode fallback ms/call | prompt cuda ms/call | prompt fallback ms/call |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | `down` | 5278 | 5278 | 0 | 1.000 | 5101 | 177 | 2.912 | 0.001 | 63.172 | 0.002 |
| `dev_france_regression` | `up_gate` | 5101 | 5101 | 0 | 1.000 | 5101 | 0 | 7.437 | 0.001 | 0.000 | 0.000 |
| `dev_intelligence_general` | `down` | 5878 | 5878 | 0 | 1.000 | 5701 | 177 | 2.886 | 0.002 | 54.215 | 0.002 |
| `dev_intelligence_general` | `up_gate` | 5701 | 5701 | 0 | 1.000 | 5701 | 0 | 7.387 | 0.001 | 0.000 | 0.000 |

## Top 20 Up/Gate Layers By Wall

| layer | calls | wall ms | wait ms | stage ms | kernel ms | up miss/call | gate miss/call | avg stage jobs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 180 | 1803.794 | 1110.749 | 7.541 | 1147.185 | 4.094 | 4.094 | 8.189 |
| 28 | 180 | 1733.943 | 986.548 | 7.653 | 1023.530 | 4.835 | 4.835 | 9.670 |
| 29 | 180 | 1722.035 | 972.582 | 7.618 | 1009.392 | 4.733 | 4.733 | 9.466 |
| 54 | 180 | 1703.601 | 1014.783 | 7.592 | 1051.117 | 4.194 | 4.189 | 8.382 |
| 30 | 180 | 1700.841 | 972.765 | 7.795 | 1009.422 | 4.618 | 4.618 | 9.236 |
| 32 | 180 | 1678.388 | 948.553 | 7.514 | 984.349 | 4.475 | 4.475 | 8.949 |
| 31 | 180 | 1672.628 | 977.411 | 7.562 | 1014.285 | 4.412 | 4.412 | 8.824 |
| 52 | 180 | 1647.860 | 930.592 | 7.802 | 966.855 | 4.346 | 4.346 | 8.692 |
| 33 | 180 | 1642.137 | 941.909 | 7.735 | 978.386 | 4.444 | 4.444 | 8.889 |
| 48 | 180 | 1631.381 | 934.604 | 7.559 | 970.458 | 4.310 | 4.310 | 8.620 |
| 34 | 180 | 1621.754 | 917.049 | 7.436 | 953.243 | 4.205 | 4.205 | 8.409 |
| 53 | 180 | 1616.755 | 922.071 | 7.531 | 959.385 | 4.314 | 4.314 | 8.628 |
| 36 | 180 | 1613.952 | 925.952 | 7.501 | 961.907 | 4.328 | 4.322 | 8.650 |
| 55 | 180 | 1588.447 | 892.288 | 7.472 | 929.332 | 4.172 | 4.172 | 8.343 |
| 51 | 180 | 1585.188 | 898.088 | 8.055 | 935.302 | 4.177 | 4.183 | 8.360 |
| 43 | 180 | 1576.633 | 897.089 | 7.634 | 933.909 | 4.030 | 4.030 | 8.061 |
| 44 | 180 | 1556.668 | 882.046 | 7.707 | 917.840 | 3.976 | 3.976 | 7.952 |
| 50 | 180 | 1553.582 | 882.265 | 7.632 | 917.831 | 3.979 | 3.979 | 7.959 |
| 38 | 180 | 1533.992 | 868.781 | 7.335 | 904.136 | 3.954 | 3.954 | 7.909 |
| 49 | 180 | 1524.450 | 869.494 | 7.426 | 905.300 | 3.964 | 3.964 | 7.928 |

## Top 20 Decode Down Layers By Wall

| layer | calls | wall ms | stage ms | kernel ms | miss/call | staged jobs/call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 180 | 1059.131 | 1030.449 | 17.366 | 6.613 | 6.613 |
| 6 | 180 | 1045.415 | 1016.685 | 17.503 | 5.981 | 5.981 |
| 9 | 180 | 1038.909 | 1010.079 | 17.664 | 5.688 | 5.688 |
| 7 | 180 | 1024.368 | 995.674 | 17.520 | 5.928 | 5.928 |
| 10 | 180 | 1011.425 | 982.811 | 17.513 | 5.826 | 5.826 |
| 8 | 180 | 976.519 | 947.639 | 17.821 | 5.422 | 5.422 |
| 26 | 180 | 963.798 | 935.085 | 17.455 | 5.485 | 5.485 |
| 25 | 180 | 939.712 | 910.722 | 17.669 | 5.547 | 5.547 |
| 15 | 180 | 929.227 | 900.241 | 17.733 | 4.949 | 4.949 |
| 24 | 180 | 922.039 | 893.270 | 17.581 | 5.475 | 5.475 |
| 23 | 180 | 921.317 | 892.422 | 17.556 | 5.415 | 5.415 |
| 58 | 180 | 912.721 | 884.277 | 17.258 | 5.272 | 5.272 |
| 20 | 180 | 909.179 | 880.523 | 17.640 | 5.268 | 5.268 |
| 18 | 180 | 909.034 | 879.976 | 17.942 | 4.977 | 4.977 |
| 19 | 180 | 908.308 | 879.318 | 17.721 | 5.297 | 5.297 |
| 1 | 180 | 904.935 | 869.856 | 23.492 | 6.574 | 6.574 |
| 57 | 180 | 902.310 | 873.436 | 17.510 | 5.096 | 5.096 |
| 16 | 180 | 883.392 | 853.895 | 18.016 | 5.012 | 5.012 |
| 5 | 180 | 879.530 | 844.975 | 23.139 | 6.561 | 6.561 |
| 3 | 180 | 876.475 | 840.382 | 23.061 | 6.354 | 6.354 |

## Evidence Scope

- This audit uses existing `metrics.json`, `stderr.txt`, `fallback-profile.csv`, `up-gate-profile.csv`, and `down-batch-profile.csv` artifacts.
- It proves CPU fallback and CUDA batch-extension acceptance for the profiled runs.
- Runs with `copy-profile.csv`: `2/2`.
- For runs collected with `COPY_PROFILE=1`, use `kimi_copy_profile_breakdown.py` to split expert-pack/io_uring wait from H2D enqueue/copy.
- A later source-change A/B must still run fresh cold-start profiles with copy/io traces before claiming SOTA.

## Next Action

1. Skip broad Kimi fallback hook work unless a fresh fallback-reason profile contradicts this audit.
2. Profile with `COPY_PROFILE=1` and IO batch traces on France plus a held-out prompt.
3. Design the next default-off A/B around reducing up/gate demand-read wait and preserving down overlap.
4. Reject any cache/RAM policy that improves hit rate but does not reduce endpoint decode time under the 16GB RAM gate.
