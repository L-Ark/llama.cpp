# Kimi CPU/defer GPU-extension audit

Input root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-audit-input`
Branch at source run: `vendor/kimi-deepseek-41d205-additive`
Commit at source run: `a4076b7a4`
Runs: `1`

## Verdict

- Decision: `fallback_hook_not_next`.
- True CPU fallback rows: `0`; fallback ms: `0.000`.
- Runs with CPU/defer extension miss: `0`.
- Weighted decode: `649.959 ms/token`, `1.539 tok/s`.
- Up/gate GPU-extension wall: `443.978 ms/token`.
- Up/gate exposed wait proxy: `255.369 ms/token`.
- Decode down GPU-extension wall: `170.954 ms/token`.

Interpretation:

- The DeepSeek CPU/defer GPU-extension pattern is already active for Kimi in this profiled SOTA path.
- `up_gate` and `down` CPU/defer ops are accepted by the CUDA batch extension; the fallback CSV is empty.
- The next implementation should not be another broad CPU fallback rewrite.
- The next bottleneck to attack is exposed up/gate staging and io_uring demand-read starvation, followed by down staging.

## Per Prompt

| Prompt | quality | tok/s | decode ms/token | TTFT ms | RAM peak GiB | fallback rows | extension miss | up/gate ms/token | up/gate wait ms/token | down ms/token | down stage ms/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | pass | 1.540 | 649.959 | 11828.710 | 11.932 | 0 | 0 | 443.978 | 255.369 | 170.954 | 159.927 |

## CPU/defer Op Acceptance

| Prompt | op | calls | batch accept | batch decline | accept rate | decode calls | prompt calls | decode cuda ms/call | decode fallback ms/call | prompt cuda ms/call | prompt fallback ms/call |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | `down` | 5278 | 5278 | 0 | 1.000 | 5101 | 177 | 2.896 | 0.001 | 63.259 | 0.002 |
| `dev_france_regression` | `up_gate` | 5101 | 5101 | 0 | 1.000 | 5101 | 0 | 7.457 | 0.001 | 0.000 | 0.000 |

## Top 20 Up/Gate Layers By Wall

| layer | calls | wall ms | wait ms | stage ms | kernel ms | up miss/call | gate miss/call | avg stage jobs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 29 | 85 | 854.252 | 484.685 | 3.232 | 501.791 | 5.224 | 5.224 | 10.447 |
| 14 | 85 | 834.170 | 513.356 | 3.171 | 530.141 | 4.094 | 4.094 | 8.188 |
| 33 | 85 | 827.266 | 483.011 | 3.326 | 499.992 | 4.741 | 4.741 | 9.482 |
| 51 | 85 | 826.527 | 471.318 | 3.327 | 488.449 | 4.471 | 4.471 | 8.941 |
| 28 | 85 | 825.530 | 470.502 | 3.188 | 487.991 | 4.965 | 4.965 | 9.929 |
| 54 | 85 | 816.138 | 503.089 | 3.333 | 519.689 | 3.882 | 3.882 | 7.765 |
| 30 | 85 | 804.599 | 458.359 | 3.393 | 475.632 | 4.741 | 4.741 | 9.482 |
| 55 | 85 | 798.142 | 459.137 | 4.365 | 476.615 | 4.459 | 4.459 | 8.918 |
| 32 | 85 | 794.507 | 445.210 | 3.334 | 461.999 | 4.518 | 4.518 | 9.035 |
| 31 | 85 | 791.405 | 451.778 | 3.351 | 468.791 | 4.624 | 4.635 | 9.259 |
| 48 | 85 | 790.838 | 452.477 | 3.472 | 469.301 | 4.388 | 4.388 | 8.776 |
| 52 | 85 | 790.777 | 457.835 | 3.578 | 474.730 | 4.376 | 4.376 | 8.753 |
| 36 | 85 | 773.870 | 444.217 | 3.262 | 461.159 | 4.329 | 4.318 | 8.647 |
| 53 | 85 | 757.435 | 439.638 | 3.410 | 456.744 | 4.271 | 4.271 | 8.541 |
| 43 | 85 | 738.380 | 421.785 | 3.285 | 438.556 | 3.776 | 3.776 | 7.553 |
| 34 | 85 | 731.764 | 416.115 | 3.284 | 432.866 | 3.988 | 3.988 | 7.976 |
| 50 | 85 | 729.069 | 410.739 | 3.470 | 427.315 | 3.906 | 3.906 | 7.812 |
| 47 | 85 | 718.276 | 416.293 | 3.339 | 432.693 | 3.788 | 3.788 | 7.576 |
| 41 | 85 | 715.435 | 405.230 | 3.300 | 421.710 | 3.741 | 3.741 | 7.482 |
| 38 | 85 | 714.238 | 406.247 | 3.324 | 422.651 | 3.835 | 3.835 | 7.671 |

## Top 20 Decode Down Layers By Wall

| layer | calls | wall ms | stage ms | kernel ms | miss/call | staged jobs/call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 85 | 505.330 | 491.969 | 8.141 | 6.541 | 6.541 |
| 10 | 85 | 492.885 | 479.449 | 8.302 | 6.188 | 6.188 |
| 7 | 85 | 484.203 | 470.627 | 8.487 | 6.024 | 6.024 |
| 6 | 85 | 483.376 | 470.237 | 8.215 | 5.941 | 5.941 |
| 9 | 85 | 477.747 | 464.282 | 8.320 | 5.765 | 5.765 |
| 18 | 85 | 472.949 | 458.553 | 8.333 | 5.165 | 5.165 |
| 24 | 85 | 469.402 | 456.033 | 8.322 | 5.824 | 5.824 |
| 8 | 85 | 463.049 | 449.650 | 8.337 | 5.624 | 5.624 |
| 25 | 85 | 450.740 | 437.427 | 8.273 | 5.788 | 5.788 |
| 23 | 85 | 445.498 | 432.000 | 8.460 | 5.694 | 5.694 |
| 20 | 85 | 440.284 | 426.408 | 8.406 | 5.600 | 5.600 |
| 19 | 85 | 431.097 | 417.648 | 8.328 | 5.341 | 5.341 |
| 16 | 85 | 428.650 | 415.274 | 8.351 | 5.235 | 5.235 |
| 15 | 85 | 427.616 | 414.157 | 8.392 | 5.024 | 5.024 |
| 1 | 85 | 426.810 | 410.674 | 11.067 | 6.612 | 6.612 |
| 22 | 85 | 424.551 | 411.172 | 8.319 | 5.282 | 5.282 |
| 21 | 85 | 421.864 | 408.478 | 8.262 | 5.282 | 5.282 |
| 58 | 85 | 420.853 | 407.881 | 8.043 | 4.976 | 4.976 |
| 57 | 85 | 418.317 | 404.980 | 8.192 | 4.918 | 4.918 |
| 26 | 85 | 417.113 | 403.498 | 8.243 | 5.224 | 5.224 |

## Evidence Scope

- This audit uses existing `metrics.json`, `stderr.txt`, `fallback-profile.csv`, `up-gate-profile.csv`, and `down-batch-profile.csv` artifacts.
- It proves CPU fallback and CUDA batch-extension acceptance for the profiled runs.
- Runs with `copy-profile.csv`: `1/1`.
- For runs collected with `COPY_PROFILE=1`, use `kimi_copy_profile_breakdown.py` to split expert-pack/io_uring wait from H2D enqueue/copy.
- A later source-change A/B must still run fresh cold-start profiles with copy/io traces before claiming SOTA.

## Next Action

1. Skip broad Kimi fallback hook work unless a fresh fallback-reason profile contradicts this audit.
2. Profile with `COPY_PROFILE=1` and IO batch traces on France plus a held-out prompt.
3. Design the next default-off A/B around reducing up/gate demand-read wait and preserving down overlap.
4. Reject any cache/RAM policy that improves hit rate but does not reduce endpoint decode time under the 16GB RAM gate.
