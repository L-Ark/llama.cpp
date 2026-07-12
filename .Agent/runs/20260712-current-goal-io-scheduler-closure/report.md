# Kimi IO Scheduler-Only Closure

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `39194d59b`

Status: completed and rejected as the next primary runtime direction.

This is an evidence consolidation from current N96 traces plus prior A/B
results. It does not change runtime behavior and does not claim SOTA.

## Purpose

The current goal requires reducing exposed expert movement and io_uring wait.
After CPU fallback, RAM second-tier, and known lower-byte candidates were
rejected, the remaining low-risk question was whether pure io_uring scheduler
knobs such as larger `MOE_IO_DEPTH` or `MOE_IO_REFILL_BATCH` still have enough
headroom.

## Current Trace Input

Trace root:

`/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-admission-ioread-n96-083914`

Prompts:

- `dev_france_regression`
- `dev_intelligence_general`

Config:

- `N=96`;
- `PROFILE=1`;
- `COPY_PROFILE=0`;
- `MOE_IO_DEPTH=8`;
- `MOE_IO_REFILL_BATCH=4`;
- `GGML_MOE_IO_BATCH_PROFILE_OUT=<run>/io-batch-profile.csv`;
- cold start under `MemoryMax=15900000000`, `MemorySwapMax=0`.

Quality and endpoint metrics:

| prompt | quality | tok/s | TTFT ms | RAM peak GiB |
|---|---|---:|---:|---:|
| `dev_france_regression` | pass | `1.72` | `8487.43` | `11.93` |
| `dev_intelligence_general` | pass | `1.69` | `7034.72` | `11.77` |

Weighted decode is about `587.45 ms/token`, or `1.70 tok/s`. Reaching `2 tok/s`
requires roughly `87.45 ms/token` saved before any additional overhead.

## Current Batch Depth Evidence

Aggregate across both N96 prompts:

- total IO batches: `32618`;
- total read jobs: `174725`;
- avg read jobs/batch: `5.357`;
- weighted inflight avg: `3.990`;
- total profiled IO wait: `96770.782 ms`;
- batches with `read_jobs <= 8`: `32264` (`98.915%`);
- batches with `read_jobs > 8`: `354` (`1.085%`);
- wait in `read_jobs <= 8` batches: `92473.847 ms` (`95.560%`);
- wait in `read_jobs > 8` batches: `4296.934 ms` (`4.440%`).

By op/role:

| op/role | batches | avg read jobs | inflight avg | batches >8 | wait >8 frac |
|---|---:|---:|---:|---:|---:|
| `runtime_load:gate` | `10853` | `5.091` | `4.017` | `118` | `4.263%` |
| `runtime_load:up` | `10854` | `5.091` | `3.839` | `118` | `3.910%` |
| `runtime_load:down` | `5698` | `6.830` | `4.515` | `118` | `8.679%` |
| `current_down_overlap:down` | `5213` | `4.851` | `3.456` | `0` | `0.000%` |

Interpretation:

- Increasing configured depth above `8` can only directly affect the `354`
  batches with more than `8` read jobs.
- Even an impossible upper bound that eliminates all wait in those `>8` batches
  saves only `4296.934 ms / 180 tokens = 23.872 ms/token`.
- That upper bound would move the current trace from about `587.45 ms/token` to
  about `563.58 ms/token`, or roughly `1.77 tok/s`, still below `2 tok/s`.
- Real depth/refill changes cannot eliminate all `>8` wait and may add overhead
  or alter scheduling, so the practical bound is smaller.

## Prior A/B Evidence

Existing committed runtime A/B results already tested the obvious knob:
`MOE_IO_DEPTH=16`, `MOE_IO_REFILL_BATCH=8`.

GP80 N32 smoke:

- artifact: `.Agent/runs/20260708-gp80-io-depth16-refill8-summary.md`;
- baseline `depth8/refill4`: `1.72 tok/s`;
- candidate `depth16/refill8`: `1.69 tok/s`;
- TTFT increased from `77778.06 ms` to `83751.31 ms`;
- iouring wait increased from `11929606 us` to `12066182 us`;
- inflight max stayed `8`.

Fresh pack-source control N32:

- artifact: `.Agent/runs/20260712-modelsource-pack-control-dev2-n32/report.md`;
- baseline `depth8/refill4`: `1.58 tok/s`, `19558.88 ms` decode;
- candidate `depth16/refill8`: `1.50 tok/s`, `20718.04 ms` decode;
- raw iouring wait decreased by only `0.574 s`, but endpoint decode regressed
  by `1159.16 ms`.

## Decision

Reject pure IO-depth/refill tuning as the next primary Kimi optimization.

Reasons:

- Current N96 traces expose too few batches with more than `8` read jobs.
- The theoretical upper bound from deeper queues is below the gap to `2 tok/s`.
- Prior runtime A/B confirms that `depth16/refill8` does not improve endpoint
  token rate and can regress decode/TTFT.
- The root issue is not the configured queue depth; it is demand order and
  expert byte volume. The runtime usually knows only the current layer's routed
  experts and cannot keep enough independent future work in flight.

Next action:

- Do not continue scheduler-only depth/refill tuning.
- Keep `MOE_IO_DEPTH=8`, `MOE_IO_REFILL_BATCH=4`, `VRAM_MIB=15000`, and
  `UPGATE_PCT=72` as the reproduction defaults.
- Further progress requires either:
  - explicit approval for the complete-model `i1-IQ1_S` smoke; or
  - a genuinely new lower-byte expert representation with prompt-level
    output-error evidence; or
  - a new algorithm that gives future expert knowledge early enough to create
    large independent batches without breaking quality.

## Reproduce The Depth Ceiling Calculation

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-admission-ioread-n96-083914

python3 - <<'PY'
import csv, pathlib, collections

root = pathlib.Path("/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-admission-ioread-n96-083914")
total = collections.defaultdict(float)

for run in sorted(root.iterdir()):
    path = run / "io-batch-profile.csv"
    if not path.exists():
        continue
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            read_jobs = int(float(row.get("read_jobs") or 0))
            wait = float(row.get("wait_ms") or 0)
            total["batches"] += 1
            total["read_jobs"] += read_jobs
            total["wait_ms"] += wait
            if read_jobs <= 8:
                total["batches_le8"] += 1
                total["wait_le8"] += wait
            else:
                total["batches_gt8"] += 1
                total["wait_gt8"] += wait

print(dict(total))
print("avg_read_jobs", total["read_jobs"] / total["batches"])
print("batch_gt8_rate", total["batches_gt8"] / total["batches"])
print("wait_gt8_frac", total["wait_gt8"] / total["wait_ms"])
print("gt8_upper_bound_ms_per_token", total["wait_gt8"] / 180)
PY
```
