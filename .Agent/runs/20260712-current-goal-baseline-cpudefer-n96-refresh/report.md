# Kimi CPU/defer GPU-extension audit

Input root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-baseline-cpudefer-n96-082136`
Branch at source run: `vendor/kimi-deepseek-41d205-additive`
Commit at source run: `2d0487c94`
Runs: `2`

## Verdict

- Decision: `fallback_hook_not_next`.
- True CPU fallback rows: `0`; fallback ms: `0.000`.
- Runs with CPU/defer extension miss: `0`.
- Weighted decode: `655.707 ms/token`, `1.525 tok/s`.
- Up/gate GPU-extension wall: `448.261 ms/token`.
- Up/gate exposed wait proxy: `254.810 ms/token`.
- Decode down GPU-extension wall: `171.924 ms/token`.

Interpretation:

- The DeepSeek CPU/defer GPU-extension pattern is already active for Kimi in this profiled SOTA path.
- `up_gate` and `down` CPU/defer ops are accepted by the CUDA batch extension; the fallback CSV is empty.
- The next implementation should not be another broad CPU fallback rewrite.
- The next bottleneck to attack is exposed up/gate staging and io_uring demand-read starvation, followed by down staging.

## Per Prompt

| Prompt | quality | tok/s | decode ms/token | TTFT ms | RAM peak GiB | fallback rows | extension miss | up/gate ms/token | up/gate wait ms/token | down ms/token | down stage ms/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | pass | 1.530 | 652.693 | 12004.970 | 11.933 | 0 | 0 | 445.737 | 257.324 | 172.304 | 161.357 |
| `dev_intelligence_general` | pass | 1.520 | 658.404 | 9863.440 | 11.772 | 0 | 0 | 450.519 | 252.561 | 171.583 | 160.551 |

## CPU/defer Op Acceptance

| Prompt | op | calls | batch accept | batch decline | accept rate | decode calls | prompt calls | decode cuda ms/call | decode fallback ms/call | prompt cuda ms/call | prompt fallback ms/call |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | `down` | 5278 | 5278 | 0 | 1.000 | 5101 | 177 | 2.918 | 0.001 | 64.519 | 0.002 |
| `dev_france_regression` | `up_gate` | 5101 | 5101 | 0 | 1.000 | 5101 | 0 | 7.484 | 0.001 | 0.000 | 0.000 |
| `dev_intelligence_general` | `down` | 5878 | 5878 | 0 | 1.000 | 5701 | 177 | 2.908 | 0.001 | 54.061 | 0.002 |
| `dev_intelligence_general` | `up_gate` | 5701 | 5701 | 0 | 1.000 | 5701 | 0 | 7.567 | 0.001 | 0.000 | 0.000 |

## Top 20 Up/Gate Layers By Wall

| layer | calls | wall ms | wait ms | stage ms | kernel ms | up miss/call | gate miss/call | avg stage jobs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54 | 180 | 1796.309 | 1066.828 | 7.914 | 1102.776 | 4.194 | 4.189 | 8.382 |
| 14 | 180 | 1787.115 | 1086.125 | 7.536 | 1122.792 | 4.094 | 4.094 | 8.189 |
| 28 | 180 | 1742.840 | 989.586 | 7.817 | 1027.646 | 4.835 | 4.835 | 9.670 |
| 29 | 180 | 1736.321 | 980.560 | 7.897 | 1017.742 | 4.733 | 4.733 | 9.466 |
| 53 | 180 | 1713.809 | 987.500 | 8.068 | 1025.032 | 4.314 | 4.314 | 8.628 |
| 52 | 180 | 1702.640 | 972.491 | 7.926 | 1008.499 | 4.346 | 4.346 | 8.692 |
| 30 | 180 | 1698.782 | 969.699 | 7.760 | 1005.680 | 4.618 | 4.618 | 9.236 |
| 48 | 180 | 1691.286 | 965.299 | 8.193 | 1001.592 | 4.310 | 4.310 | 8.620 |
| 32 | 180 | 1665.089 | 942.228 | 7.716 | 978.774 | 4.475 | 4.475 | 8.949 |
| 55 | 180 | 1658.863 | 952.211 | 10.494 | 989.784 | 4.172 | 4.172 | 8.343 |
| 51 | 180 | 1651.611 | 937.878 | 7.813 | 973.288 | 4.177 | 4.183 | 8.360 |
| 33 | 180 | 1649.562 | 948.765 | 7.797 | 985.096 | 4.444 | 4.444 | 8.889 |
| 31 | 180 | 1632.016 | 931.364 | 7.978 | 967.670 | 4.412 | 4.412 | 8.824 |
| 50 | 180 | 1628.117 | 926.343 | 7.924 | 961.973 | 3.979 | 3.979 | 7.959 |
| 34 | 180 | 1621.490 | 916.671 | 7.788 | 953.076 | 4.205 | 4.205 | 8.409 |
| 43 | 180 | 1620.860 | 922.761 | 7.867 | 958.898 | 4.030 | 4.030 | 8.061 |
| 49 | 180 | 1608.206 | 923.644 | 7.977 | 959.052 | 3.964 | 3.964 | 7.928 |
| 36 | 180 | 1606.168 | 915.162 | 7.619 | 951.149 | 4.328 | 4.322 | 8.650 |
| 44 | 180 | 1578.633 | 898.859 | 7.840 | 934.349 | 3.976 | 3.976 | 7.952 |
| 38 | 180 | 1554.642 | 873.715 | 7.647 | 909.532 | 3.954 | 3.954 | 7.909 |

## Top 20 Decode Down Layers By Wall

| layer | calls | wall ms | stage ms | kernel ms | miss/call | staged jobs/call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 180 | 1079.671 | 1051.683 | 17.173 | 6.613 | 6.613 |
| 6 | 180 | 1053.661 | 1025.010 | 17.685 | 5.981 | 5.981 |
| 7 | 180 | 1040.326 | 1011.622 | 17.662 | 5.928 | 5.928 |
| 10 | 180 | 1021.811 | 993.070 | 17.585 | 5.826 | 5.826 |
| 9 | 180 | 1015.287 | 986.339 | 17.814 | 5.688 | 5.688 |
| 8 | 180 | 974.872 | 946.259 | 17.816 | 5.422 | 5.422 |
| 25 | 180 | 938.008 | 909.620 | 17.684 | 5.547 | 5.547 |
| 58 | 180 | 933.334 | 905.442 | 17.099 | 5.272 | 5.272 |
| 24 | 180 | 932.927 | 904.313 | 17.857 | 5.475 | 5.475 |
| 26 | 180 | 932.819 | 904.166 | 17.630 | 5.485 | 5.485 |
| 57 | 180 | 929.448 | 900.418 | 17.368 | 5.096 | 5.096 |
| 15 | 180 | 921.247 | 892.448 | 17.782 | 4.949 | 4.949 |
| 1 | 180 | 919.649 | 885.075 | 23.492 | 6.574 | 6.574 |
| 23 | 180 | 917.882 | 889.456 | 17.782 | 5.415 | 5.415 |
| 20 | 180 | 907.954 | 879.606 | 17.593 | 5.268 | 5.268 |
| 3 | 180 | 906.423 | 872.213 | 23.087 | 6.354 | 6.354 |
| 5 | 180 | 905.777 | 871.167 | 23.153 | 6.561 | 6.561 |
| 18 | 180 | 901.076 | 872.680 | 17.560 | 4.977 | 4.977 |
| 19 | 180 | 899.993 | 871.691 | 17.561 | 5.297 | 5.297 |
| 2 | 180 | 887.323 | 853.192 | 23.022 | 6.273 | 6.273 |

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
