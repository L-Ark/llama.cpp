# Kimi Wait-Weighted Admission Screen

- profile root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-baseline-cpudefer-n96-082136`
- runs: `2`
- decode runs: `180`
- decode ms/token: `655.707`
- decode down filter: `n_active <= 8`

This is an offline screen only. It does not select held-out prompts and
does not claim a SOTA result.

| layer | role | prompts | ms/token | stage ms/token | hit rate | misses | unique experts | observed MiB | ideal tok/s if removed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 54 | upgate | 2 | 9.979 | 0.044 | 47.40% | 1515 | 556 | 2736.56 | 1.549 |
| 14 | upgate | 2 | 9.928 | 0.042 | 48.82% | 1474 | 468 | 2303.44 | 1.549 |
| 28 | upgate | 2 | 9.682 | 0.043 | 39.65% | 1738 | 486 | 2392.03 | 1.548 |
| 29 | upgate | 2 | 9.646 | 0.044 | 41.18% | 1694 | 494 | 2431.41 | 1.548 |
| 53 | upgate | 2 | 9.521 | 0.045 | 46.04% | 1554 | 546 | 2687.34 | 1.548 |
| 52 | upgate | 2 | 9.459 | 0.044 | 45.69% | 1564 | 536 | 2638.12 | 1.547 |
| 30 | upgate | 2 | 9.438 | 0.043 | 42.36% | 1660 | 482 | 2372.34 | 1.547 |
| 48 | upgate | 2 | 9.396 | 0.046 | 46.18% | 1550 | 528 | 2598.75 | 1.547 |
| 32 | upgate | 2 | 9.250 | 0.043 | 44.10% | 1610 | 484 | 2382.19 | 1.547 |
| 55 | upgate | 2 | 9.216 | 0.058 | 48.06% | 1496 | 538 | 2647.97 | 1.547 |
| 51 | upgate | 2 | 9.176 | 0.043 | 47.95% | 1499 | 506 | 2490.47 | 1.547 |
| 33 | upgate | 2 | 9.164 | 0.043 | 44.65% | 1594 | 484 | 2382.19 | 1.547 |
| 31 | upgate | 2 | 9.067 | 0.044 | 45.00% | 1584 | 472 | 2323.12 | 1.546 |
| 50 | upgate | 2 | 9.045 | 0.044 | 50.21% | 1434 | 510 | 2510.16 | 1.546 |
| 34 | upgate | 2 | 9.008 | 0.043 | 47.29% | 1518 | 466 | 2293.59 | 1.546 |
| 43 | upgate | 2 | 9.005 | 0.044 | 49.44% | 1456 | 506 | 2490.47 | 1.546 |
| 49 | upgate | 2 | 8.934 | 0.044 | 50.35% | 1430 | 534 | 2628.28 | 1.546 |
| 36 | upgate | 2 | 8.923 | 0.042 | 45.94% | 1557 | 484 | 2382.19 | 1.546 |
| 44 | upgate | 2 | 8.770 | 0.044 | 50.07% | 1438 | 514 | 2529.84 | 1.546 |
| 38 | upgate | 2 | 8.637 | 0.042 | 50.49% | 1426 | 452 | 2224.69 | 1.545 |
| 45 | upgate | 2 | 8.569 | 0.043 | 52.36% | 1372 | 512 | 2520.00 | 1.545 |
| 47 | upgate | 2 | 8.533 | 0.046 | 52.92% | 1356 | 496 | 2441.25 | 1.545 |
| 39 | upgate | 2 | 8.526 | 0.043 | 53.40% | 1342 | 470 | 2313.28 | 1.545 |
| 40 | upgate | 2 | 8.499 | 0.044 | 54.24% | 1318 | 472 | 2323.12 | 1.545 |
| 46 | upgate | 2 | 8.474 | 0.043 | 53.47% | 1340 | 486 | 2392.03 | 1.545 |
| 41 | upgate | 2 | 8.328 | 0.043 | 54.44% | 1312 | 486 | 2392.03 | 1.545 |
| 42 | upgate | 2 | 8.169 | 0.044 | 56.88% | 1242 | 466 | 2293.59 | 1.544 |
| 4 | upgate | 2 | 8.163 | 0.047 | 29.06% | 2043 | 554 | 2969.09 | 1.544 |
| 5 | upgate | 2 | 8.145 | 0.048 | 27.36% | 2092 | 528 | 2829.75 | 1.544 |
| 35 | upgate | 2 | 8.088 | 0.043 | 53.51% | 1339 | 468 | 2303.44 | 1.544 |

Interpretation:

- Prefer candidates that are high ms/token, appear in all dev prompts,
  have substantial misses, and have an observed footprint small enough to
  fit in VRAM or RAM without displacing higher-value entries.
- This screen ranks layer/role buckets, not individual experts. A runtime
  candidate still needs a profile-generation step and paired cold-start
  A/B validation.
