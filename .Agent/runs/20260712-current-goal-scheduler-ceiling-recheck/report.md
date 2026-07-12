# Kimi Current Scheduler Ceiling Recheck

This is a dev-only COPY/IO diagnostic analysis. It does not modify runtime
behavior and does not claim SOTA.

## Input

- Branch: `vendor/kimi-deepseek-41d205-additive`
- Previous commit: `12e3b8942`
- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717`
- Prompts:
  - `dev_france_regression`
  - `dev_intelligence_general`

## IO Queue Summary

- weighted diagnostic token rate: `1.458 tok/s`
- aggregate `io_uring` throughput: `9.536 GiB/s`
- pure IO peak reference: `10.300 GiB/s`
- peak utilization: `0.926`
- direct read ratio: `0.000`
- weighted inflight avg: `4.496`
- `io_uring` wait/decode fraction: `0.749`

## Exact-Byte Scheduler Ceiling

| prompt | moved GiB/token | transfer-only tok/s | floor+transfer tok/s |
|---|---:|---:|---:|
| `dev_france_regression` | `6.689` | `1.540` | `1.450` |
| `dev_intelligence_general` | `6.393` | `1.611` | `1.513` |

Summary:

- transfer-only mean ceiling: `1.576 tok/s`
- floor+transfer mean ceiling: `1.482 tok/s`
- best floor+transfer ceiling: `1.513 tok/s`
- byte ratio needed for `5 tok/s`: about `0.252x`

## Decision

Do not implement pure exact-byte scheduler or gate/up/down co-submit as the
next runtime A/B. At current moved bytes, the NVMe-to-VRAM path is already near
the measured pure-IO reference, and perfect scheduling without byte reduction
does not reach `2 tok/s`.

Future scheduler/co-submit work should be secondary, after byte reduction or a
stronger predictor/shadow signal.

## Reproduce

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
OUT=$ROOT/analysis/scheduler-current

python3 .Agent/run-tools/kimi_io_queue_metrics_summary.py \
  --metrics "$ROOT/dev_france_regression/metrics.json" \
  --metrics "$ROOT/dev_intelligence_general/metrics.json" \
  --out-json "$OUT/io-queue-summary.json" \
  --out-csv "$OUT/io-queue-summary.csv" \
  --out-md "$OUT/io-queue-summary.md" \
  --peak-gib-s 10.3

python3 .Agent/run-tools/kimi_exact_byte_scheduler_ceiling.py \
  --profile-root "$ROOT" \
  --out-json "$OUT/exact-byte-scheduler-ceiling.json" \
  --out-md "$OUT/exact-byte-scheduler-ceiling.md" \
  --bandwidth-gib-s 10.3 \
  --all-hit-floor-ms-per-token 40.1 \
  --target-tok-s 5.0
```
