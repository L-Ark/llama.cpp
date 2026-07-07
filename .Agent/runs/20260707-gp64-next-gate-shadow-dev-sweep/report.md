# GP64 next-gate shadow dev sweep

Status: dev-only shadow profiling. No runtime prefetch was implemented.

Branch:

- `vendor/kimi-speculative-general-token-rate-16gb`
- pushed remote: `wici/vendor/kimi-speculative-general-token-rate-16gb`
- code head for remote runs: `042b67c11`

Remote validation repo:

- `/root/lfz/llama.cpp-vendor-kimi-gp64-shadow`

Prompt discipline:

- Used only dev prompts:
  `.Agent/evals/kimi-general-dev-prompts.jsonl`
- Did not use held-out test prompts.

Runtime shape:

- Cold start per prompt via `systemd-run`.
- Host RAM cgroup:
  `MemoryMax=15900000000`, `MemorySwapMax=0`.
- `N=16`.
- Existing general-prompt SOTA reproduction script:
  `.Agent/run-tools/kimi-general-prompt-repro.sh`.
- Shadow env:
  `GGML_MOE_NEXT_GATE_SHADOW_OUT=$RUN/next-gate-shadow.csv`
  plus `GGML_MOE_NEXT_GATE_SHADOW_TOPK=<4|8>`.

Runs:

- no-shadow baseline:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp64-next-gate-shadow-dev-n16-baseline`
- TopK=4 shadow:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp64-next-gate-shadow-dev-n16-top4`
- TopK=8 shadow:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp64-next-gate-shadow-dev-n16-top8`
- overhead comparison:
  `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp64-next-gate-shadow-dev-n16-overhead`

Aggregate shadow results:

| config | prompts | expert recall | precision | byte recall | false/actual bytes | max record overhead | avg shadow tok/s | max memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TopK=4 | 7/7 | 47.26% | 94.53% | 47.26% | 0.0274 | 2359 us | 0.547 | 15899996160 |
| TopK=8 | 7/7 | 77.79% | 77.79% | 77.79% | 0.2221 | 2540 us | 0.541 | 15899996160 |

Overhead versus no-shadow baseline:

| config | avg token-rate ratio | avg TTFT ratio | max TTFT ratio | avg decode ratio | max decode ratio |
|---|---:|---:|---:|---:|---:|
| TopK=4 | 1.017 | 0.987 | 1.267 | 0.990 | 1.096 |
| TopK=8 | 1.003 | 1.120 | 1.245 | 1.006 | 1.069 |

Quality notes:

- `dev_france_regression` passed in baseline, TopK=4, and TopK=8.
- `dev_linear_equation` and `dev_mixed_summary` failed keyword checks for all
  N=16 variants because outputs were truncated before the required keyword
  appeared.
- This means the N=16 shadow sweep is useful for routing/predictor metrics, but
  it is not sufficient as a final quality gate.

Interpretation:

- TopK=4 is very precise and has low false bytes, but recall is too low to
  justify next-step prefetch decisions by itself.
- TopK=8 has useful recall and stays below the initial false/actual byte target
  of `<= 0.25` on the dev sweep.
- TTFT comparison is noisy because the shadow predictor only runs during
  decode, yet the cold-start TTFT ratios still show per-prompt excursions above
  `1.20`. This fails the strict acceptance gate for moving directly to runtime
  bounded prefetch.
- Therefore GP64 shadow has demonstrated signal, but has **not** passed all
  gates needed to implement bounded prefetch.

Next required step:

- Run a longer dev quality-confirmation shadow sweep, preferably `N=32` or
  `N=96`, for the selected predictor setting before any held-out test run.
- If quality is clean and TTFT remains within the strict gate, then freeze the
  shadow configuration and run held-out test once.
- Only after that should bounded runtime prefetch be planned.
