# Kimi multi-size mixed up/gate scheduler bound

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Head before report: `9f8abca0c`

This report decides whether the next runtime implementation should be a
multi-size mixed up/gate IO scheduler. It uses existing cold-start traces only
and does not claim SOTA.

## Inputs

N96 France current baseline:

- run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-n96-iobatch`
- prompt: `Please introduce France in a short paragraph.`
- quality: pass
- decode: `50692.07 ms / 85`, `1.68 tok/s`
- TTFT: `8498.12 ms`
- RAM peak: `12802908160 bytes`
- CPU fallback: `0`

N32 same-type combined A/B:

- run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-mixed-combined-stage-n32-ab`
- baseline: `baseline`
- candidate: `sametype-combined`
- report:
  `.Agent/runs/20260712-current-goal-mixed-combined-stage-n32-ab/report.md`

## Required Savings

Current N96 France endpoint:

- decode per generated token: `50692.07 / 85 = 596.38 ms/token`;
- token rate: `1.68 tok/s`.

To reach `2 tok/s` on this prompt:

- target decode: `85 * 500 = 42500 ms`;
- required endpoint saving: `8192.07 ms`;
- required saving per generated token: about `96.38 ms/token`.

This is only the near milestone. The product target `5 tok/s` would need a
much larger change in moved bytes or computation form.

## Up/Gate Type Contribution

From the N96 `up-gate-profile.csv`:

| type pair | calls | wall ms | wall/call | up wait ms | gate wait ms | up jobs/call | gate jobs/call |
|---|---:|---:|---:|---:|---:|---:|---:|
| `22/18` | `2465` | `17624.2` | `7.150` | `10869.1` | `11823.0` | `4.08` | `4.08` |
| `22/22` | `1530` | `7968.1` | `5.208` | `7069.2` | `7629.8` | `4.61` | `4.61` |
| `18/18` | `851` | `6062.4` | `7.124` | `0.0` | `0.0` | `4.87` | `4.87` |
| `18/22` | `255` | `1459.9` | `5.725` | `1345.2` | `1351.5` | `4.89` | `4.90` |

Interpretation:

- the dominant exposed up/gate wall is mixed `22/18`, not same-type `22/22`;
- mixed `18/22` is smaller but still nonzero;
- same-size combined staging cannot cover `22/18` or `18/22`;
- `18/18` has no measured wait because that type pair follows a different
  path where the profile attributes wait differently.

## Observed Same-Type Combined Result

The existing same-type path was tested with:

- `GGML_MOE_UP_GATE_COMBINED_STAGE=1`.

It activated and moved IO metrics in the expected direction:

| run | endpoint tok/s | decode ms / runs | total iouring wait | decode-only IO wait | avg read jobs/batch | inflight |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `1.70` | `18206.89 / 31` | `18034153 us` | `15622.196 ms` | `4.593` | `3.479` |
| same-type combined | `1.70` | `18241.14 / 31` | `16124716 us` | `13160.784 ms` | `4.719` | `3.544` |

The IO wait reduction was real:

- total `io_uring_wait_us` improved by about `1.91 s`;
- decode-only IO wait improved by about `2.46 s`.

But endpoint decode did not improve:

- baseline decode: `18206.89 ms`;
- candidate decode: `18241.14 ms`;
- endpoint delta: `-34.25 ms`, slightly worse.

This means IO-wait counter reductions cannot be treated as endpoint savings.
The likely losses are extra main-stage-ring pressure and reduced up/gate
copy-compute overlap.

## Multi-Size Mixed Scheduler Bound

A useful Kimi mixed scheduler would need to handle different expert sizes in
the same IO planning phase:

- `22/18`;
- `18/22`.

A naive implementation that simply places mixed-size jobs into one ring and
waits for all copies before launching both up and gate is unlikely to help.
The same-type A/B already shows that losing overlap can erase IO wait gains.

For a mixed scheduler to be worth implementation, it must satisfy all of:

- submit mixed-size `up+gate` reads together so the SSD queue sees a larger
  batch;
- copy each completed read to the correct role stream or otherwise make each
  role ready independently;
- allow up compute to begin when up bytes are ready, without waiting for all
  gate reads;
- allow gate compute to begin when gate bytes are ready, without waiting for
  all up reads;
- preserve the existing current-down overlap behavior;
- keep endpoint quality/RAM/TTFT/fallback gates.

The N96 type contribution gives a loose upper bound:

- mixed `22/18 + 18/22` wall is about `19084 ms`;
- reaching `2 tok/s` from this prompt needs about `8192 ms` endpoint saving;
- therefore a mixed scheduler would need to convert roughly `43%` of the
  entire mixed up/gate wall into endpoint saving.

That is not plausible for a submit/wait scheduling-only change after the
same-type A/B showed zero endpoint improvement despite a large IO wait
reduction. This does not mean multi-size scheduling is impossible, but it is
not the best next implementation without a stronger design.

## Decision

Do not implement the full multi-size mixed up/gate scheduler as the next step.

Reasons:

- same-type combined already tested the core scheduling hypothesis and did not
  improve endpoint token rate;
- Kimi's dominant up/gate wait is mixed-size, so a correct implementation is
  materially more complex than the same-type path;
- the required endpoint saving for `2 tok/s` is about `8.2 s` on this N96
  prompt, and current evidence does not support that much endpoint gain from
  scheduler-only changes;
- changing this path risks breaking the carefully working CPU/defer GPU
  extension where CPU fallback is already `0`.

## Next Priority

Move the next implementation cycle away from exact-byte scheduler-only work.

Priority order:

1. Lower-byte expert movement with hard quality gates.
   The current exact-byte path moves too much data. Any candidate must reduce
   moved bytes enough to affect endpoint time, not just IO wait counters.

2. Stronger future-expert admission.
   Prior route-history and hidden-KNN admissions failed. Only revisit if using
   a materially stronger signal such as router logits/top-k score history or a
   draft/router model.

3. RAM/VRAM storage redesign only if it reduces exposed wait under a
   prompt-general policy.
   Do not use broad hotsets or global LFU/LRU; those were already rejected.

## Reproduce Analysis

Type contribution command:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
python3 - <<'PY'
import csv, pathlib, collections
paths={
  "n96": pathlib.Path("/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-n96-iobatch/up-gate-profile.csv"),
  "n32_base": pathlib.Path("/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-mixed-combined-stage-n32-ab/baseline/up-gate-profile.csv"),
  "n32_same": pathlib.Path("/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-mixed-combined-stage-n32-ab/sametype-combined/up-gate-profile.csv"),
}
fields=["wall_ms","up_wait_ms","gate_wait_ms","up_stage_jobs","gate_stage_jobs"]
for name,p in paths.items():
    rows=list(csv.DictReader(p.open()))
    by=collections.defaultdict(lambda:collections.defaultdict(float))
    for r in rows:
        k=(r["mode"],r["up_type"],r["gate_type"])
        by[k]["calls"]+=1
        for f in fields:
            by[k][f]+=float(r[f] or 0)
    print("==", name)
    for k,d in sorted(by.items(), key=lambda kv:-kv[1]["wall_ms"]):
        c=d["calls"]
        print(k, int(c), d["wall_ms"], d["up_wait_ms"], d["gate_wait_ms"],
              d["up_stage_jobs"]/c, d["gate_stage_jobs"]/c)
PY
```
