# Kimi CPU/defer GPU-extension audit

Input root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855`
Branch at source run: `vendor/kimi-deepseek-41d205-additive`
Commit at source run: `564b88773`
Runs: `3`

## Verdict

- Decision: `fallback_hook_not_next`.
- True CPU fallback rows: `0`; fallback ms: `0.000`.
- Runs with CPU/defer extension miss: `0`.
- Weighted decode: `619.557 ms/token`, `1.614 tok/s`.
- Up/gate GPU-extension wall: `416.688 ms/token`.
- Up/gate exposed wait proxy: `246.390 ms/token`.
- Decode down GPU-extension wall: `164.982 ms/token`.

Interpretation:

- The DeepSeek CPU/defer GPU-extension pattern is already active for Kimi in this profiled SOTA path.
- `up_gate` and `down` CPU/defer ops are accepted by the CUDA batch extension; the fallback CSV is empty.
- The next implementation should not be another broad CPU fallback rewrite.
- The next bottleneck to attack is exposed up/gate staging and io_uring demand-read starvation, followed by down staging.

## Per Prompt

| Prompt | quality | tok/s | decode ms/token | TTFT ms | RAM peak GiB | fallback rows | extension miss | up/gate ms/token | up/gate wait ms/token | down ms/token | down stage ms/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | pass | 1.560 | 642.488 | 12251.960 | 11.929 | 0 | 0 | 437.948 | 253.242 | 169.885 | 159.181 |
| `dev_france_regression_iobatch` | pass | 1.720 | 582.496 | 9746.780 | 11.923 | 0 | 0 | 379.089 | 243.411 | 158.363 | 148.174 |
| `dev_intelligence_general` | pass | 1.580 | 632.201 | 9435.440 | 11.770 | 0 | 0 | 431.308 | 242.925 | 166.516 | 155.752 |

## CPU/defer Op Acceptance

| Prompt | op | calls | batch accept | batch decline | accept rate | decode calls | prompt calls | decode cuda ms/call | decode fallback ms/call | prompt cuda ms/call | prompt fallback ms/call |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | `down` | 5278 | 5278 | 0 | 1.000 | 5101 | 177 | 2.878 | 0.001 | 65.385 | 0.002 |
| `dev_france_regression` | `up_gate` | 5101 | 5101 | 0 | 1.000 | 5101 | 0 | 7.355 | 0.001 | 0.000 | 0.000 |
| `dev_france_regression_iobatch` | `down` | 5278 | 5278 | 0 | 1.000 | 5101 | 177 | 2.875 | 0.001 | 51.454 | 0.003 |
| `dev_france_regression_iobatch` | `up_gate` | 5101 | 5101 | 0 | 1.000 | 5101 | 0 | 6.372 | 0.001 | 0.000 | 0.000 |
| `dev_intelligence_general` | `down` | 5878 | 5878 | 0 | 1.000 | 5701 | 177 | 2.820 | 0.002 | 51.599 | 0.002 |
| `dev_intelligence_general` | `up_gate` | 5701 | 5701 | 0 | 1.000 | 5701 | 0 | 7.243 | 0.001 | 0.000 | 0.000 |

## Top 24 Up/Gate Layers By Wall

| layer | calls | wall ms | wait ms | stage ms | kernel ms | up miss/call | gate miss/call | avg stage jobs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 265 | 2417.370 | 1531.553 | 11.106 | 1574.920 | 4.094 | 4.094 | 8.189 |
| 29 | 265 | 2386.450 | 1418.192 | 11.632 | 1461.802 | 4.896 | 4.896 | 9.793 |
| 28 | 265 | 2367.113 | 1433.013 | 11.132 | 1476.693 | 4.878 | 4.878 | 9.756 |
| 54 | 265 | 2347.110 | 1492.539 | 11.403 | 1538.799 | 4.090 | 4.086 | 8.176 |
| 30 | 265 | 2305.324 | 1387.932 | 11.199 | 1431.750 | 4.659 | 4.659 | 9.318 |
| 33 | 265 | 2274.659 | 1380.018 | 11.728 | 1423.185 | 4.543 | 4.543 | 9.086 |
| 31 | 265 | 2267.875 | 1378.964 | 11.969 | 1422.364 | 4.482 | 4.487 | 8.969 |
| 32 | 265 | 2257.772 | 1345.205 | 11.242 | 1387.999 | 4.489 | 4.489 | 8.978 |
| 48 | 265 | 2251.466 | 1370.526 | 11.768 | 1413.699 | 4.336 | 4.336 | 8.672 |
| 52 | 265 | 2226.637 | 1327.941 | 11.318 | 1371.257 | 4.356 | 4.356 | 8.712 |
| 51 | 265 | 2213.934 | 1331.674 | 11.245 | 1377.597 | 4.275 | 4.279 | 8.554 |
| 53 | 265 | 2207.497 | 1339.601 | 11.325 | 1385.681 | 4.300 | 4.300 | 8.599 |
| 36 | 265 | 2196.061 | 1331.244 | 11.224 | 1374.412 | 4.328 | 4.321 | 8.649 |
| 34 | 265 | 2182.763 | 1306.748 | 11.410 | 1350.098 | 4.133 | 4.133 | 8.265 |
| 55 | 265 | 2182.692 | 1313.239 | 11.296 | 1357.980 | 4.267 | 4.267 | 8.535 |
| 50 | 265 | 2114.603 | 1268.239 | 11.277 | 1310.666 | 3.955 | 3.955 | 7.910 |
| 43 | 265 | 2107.570 | 1258.416 | 11.304 | 1301.693 | 3.946 | 3.946 | 7.891 |
| 38 | 265 | 2091.410 | 1251.151 | 11.338 | 1293.761 | 3.915 | 3.915 | 7.830 |
| 49 | 265 | 2088.480 | 1265.089 | 11.341 | 1307.481 | 3.913 | 3.913 | 7.826 |
| 44 | 265 | 2081.556 | 1244.118 | 11.261 | 1286.918 | 3.866 | 3.866 | 7.733 |
| 5 | 265 | 2074.681 | 0.000 | 12.198 | 2049.577 | 5.837 | 5.837 | 11.675 |
| 4 | 265 | 2046.188 | 0.000 | 12.381 | 2020.896 | 5.688 | 5.680 | 11.368 |
| 39 | 265 | 2042.841 | 1229.300 | 11.215 | 1271.959 | 3.689 | 3.689 | 7.379 |
| 45 | 265 | 2041.962 | 1230.231 | 11.210 | 1273.241 | 3.772 | 3.772 | 7.545 |

## Top 24 Decode Down Layers By Wall

| layer | calls | wall ms | stage ms | kernel ms | miss/call | staged jobs/call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 265 | 1517.104 | 1477.902 | 24.183 | 6.589 | 6.589 |
| 6 | 265 | 1495.320 | 1455.807 | 24.268 | 5.968 | 5.968 |
| 10 | 265 | 1479.099 | 1439.311 | 24.923 | 5.947 | 5.947 |
| 7 | 265 | 1476.494 | 1435.672 | 25.227 | 5.960 | 5.960 |
| 9 | 265 | 1439.190 | 1399.111 | 24.921 | 5.713 | 5.713 |
| 8 | 265 | 1390.872 | 1350.881 | 24.966 | 5.489 | 5.489 |
| 25 | 265 | 1344.492 | 1304.431 | 24.838 | 5.627 | 5.627 |
| 24 | 265 | 1337.743 | 1297.185 | 25.168 | 5.591 | 5.591 |
| 23 | 265 | 1320.829 | 1280.741 | 24.871 | 5.508 | 5.508 |
| 20 | 265 | 1316.988 | 1276.710 | 24.948 | 5.379 | 5.379 |
| 18 | 265 | 1314.515 | 1274.470 | 24.873 | 5.040 | 5.040 |
| 15 | 265 | 1312.056 | 1271.518 | 25.018 | 4.974 | 4.974 |
| 26 | 265 | 1301.792 | 1261.695 | 25.010 | 5.398 | 5.398 |
| 19 | 265 | 1288.171 | 1248.133 | 24.851 | 5.312 | 5.312 |
| 58 | 265 | 1286.469 | 1246.954 | 24.289 | 5.174 | 5.174 |
| 57 | 265 | 1285.703 | 1246.224 | 24.212 | 5.036 | 5.036 |
| 1 | 265 | 1277.874 | 1228.672 | 33.594 | 6.587 | 6.587 |
| 22 | 265 | 1258.816 | 1218.304 | 25.052 | 5.093 | 5.093 |
| 16 | 265 | 1256.153 | 1215.991 | 25.069 | 5.087 | 5.087 |
| 3 | 265 | 1253.623 | 1205.342 | 32.942 | 6.346 | 6.346 |
| 5 | 265 | 1250.948 | 1203.097 | 32.812 | 6.562 | 6.562 |
| 21 | 265 | 1249.144 | 1208.916 | 25.000 | 5.150 | 5.150 |
| 2 | 265 | 1228.611 | 1180.545 | 32.834 | 6.178 | 6.178 |
| 60 | 268 | 1182.479 | 1133.851 | 33.167 | 5.631 | 5.631 |

## Evidence Scope

- This audit uses existing `metrics.json`, `stderr.txt`, `fallback-profile.csv`, `up-gate-profile.csv`, and `down-batch-profile.csv` artifacts.
- It proves CPU fallback and CUDA batch-extension acceptance for the profiled runs.
- Runs with `copy-profile.csv`: `2/3`.
- For runs collected with `COPY_PROFILE=1`, use `kimi_copy_profile_breakdown.py` to split expert-pack/io_uring wait from H2D enqueue/copy.
- A later source-change A/B must still run fresh cold-start profiles with copy/io traces before claiming SOTA.

## Next Action

1. Skip broad Kimi fallback hook work unless a fresh fallback-reason profile contradicts this audit.
2. Profile with `COPY_PROFILE=1` and IO batch traces on France plus a held-out prompt.
3. Design the next default-off A/B around reducing up/gate demand-read wait and preserving down overlap.
4. Reject any cache/RAM policy that improves hit rate but does not reduce endpoint decode time under the 16GB RAM gate.
