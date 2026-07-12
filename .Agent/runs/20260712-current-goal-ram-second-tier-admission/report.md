# Kimi RAM Second-Tier Admission

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `48629e900`

Status: completed and rejected as the next runtime A/B.

This is a dev-only offline admission screen. It does not use held-out prompts
and does not claim a SOTA result.

## Purpose

The current N96 baseline shows no broad CPU fallback and no expert-pack miss.
The exposed cost is demand-loaded expert payload, especially up/gate. This
screen checks whether replacing low-value file cache with a RAM-resident expert
second tier is likely to improve endpoint token rate before implementing a
runtime path.

The admission gate is not "higher RAM hit rate". A useful RAM tier must:

- keep host RAM below `16 GB`;
- reduce exposed wait enough to approach `>2 tok/s`;
- preserve large batchable transfers instead of fragmenting SSD batches;
- avoid a TTFT preload cost that exceeds the `+20%` gate;
- stay prompt-general, using only dev prompts for selection.

## Inputs

Trace root:

`/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-admission-ioread-n96-083914`

Prompts:

- `dev_france_regression`: `Please introduce France in a short paragraph.`
- `dev_intelligence_general`: `What is intelligence?`

Run config:

- `N=96`;
- `PROFILE=1`;
- `COPY_PROFILE=0`;
- `GGML_MOE_IO_READ_TRACE_OUT=<run>/io-read-trace.csv`;
- `GGML_MOE_IO_BATCH_PROFILE_OUT=<run>/io-batch-profile.csv`;
- cold start under `MemoryMax=15900000000`, `MemorySwapMax=0`.

Quality and resource results:

| prompt | quality | tok/s | decode ms | runs | TTFT ms | RAM peak GiB |
|---|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.72` | `49468.56` | `85` | `8487.43` | `11.93` |
| `dev_intelligence_general` | pass | `1.69` | `56272.65` | `95` | `7034.72` | `11.77` |

Weighted decode is about `587.45 ms/token`, or `1.70 tok/s`. Reaching `2 tok/s`
requires roughly `87.45 ms/token` saved before TTFT and RAM overheads.

## Artifacts

- `upgate-budget-<mib>m/report.json`
- `allroles-budget-<mib>m/report.json`
- `phase5c-slab-screen-with-wait/report.md`
- generated wait traces in the external trace root:
  `<prompt>/io-wait-trace.csv`

The `io-wait-trace.csv` files were generated from `io-batch-profile.csv` using
`batch_seq=seq` and `wait_ms=wait_ms`. The runtime uses the same batch sequence
source for these profiles in this configuration.

## Dynamic Expert RAM Tier Screen

Screen config:

- budgets: `2048,4096,6144,8192,10240 MiB`;
- `min_prompts=2`;
- `min_count=2`;
- `max_jobs=8`;
- simulated current VRAM hotset excluded with `upgate_slots=2015`,
  `down_slots=533`;
- families:
  - `upgate`: roles `up,gate`;
  - `allroles`: roles `up,gate,down`.

| family | budget MiB | selected MiB | entries | weighted GiB | hit batches | RAM-dominant batches | RAM-only batches |
|---|---:|---:|---:|---:|---:|---:|---:|
| `upgate` | `2048` | `2046.1` | `419` | `23.708` | `4294` | `2` | `213` |
| `upgate` | `4096` | `4095.5` | `844` | `42.003` | `7038` | `14` | `429` |
| `upgate` | `6144` | `6141.9` | `1265` | `57.703` | `9082` | `41` | `567` |
| `upgate` | `8192` | `8189.5` | `1682` | `71.700` | `10650` | `85` | `809` |
| `upgate` | `10240` | `10236.9` | `2080` | `84.126` | `11846` | `138` | `1016` |
| `allroles` | `2048` | `2047.1` | `325` | `32.242` | `4243` | `1` | `73` |
| `allroles` | `4096` | `4095.2` | `683` | `56.658` | `7474` | `12` | `200` |
| `allroles` | `6144` | `6143.8` | `1032` | `77.763` | `9740` | `46` | `313` |
| `allroles` | `8192` | `8191.5` | `1393` | `96.808` | `11805` | `79` | `471` |
| `allroles` | `10240` | `10239.6` | `1763` | `114.231` | `13745` | `116` | `641` |

Interpretation:

- The selected experts cover meaningful weighted traffic, but hits are widely
  scattered across batches.
- Even at `10 GiB`, RAM-dominant batches are tiny relative to the total:
  - upgate: `138 / 21145`;
  - allroles: `116 / 31903`.
- This is exactly the fragmentation risk discussed earlier: mixed RAM+SSD
  batches can reduce SSD bytes while shrinking remaining SSD batches and hurting
  effective IO throughput.

## Layer/Role Slab Wait Bound

`phase5c-slab-screen-with-wait` uses the same traces and a wait trace derived
from the io-batch profile.

Total measured batch wait:

- `92473.847 ms`;
- `513.744 ms/token`.

Best layer-slab upper bounds:

| candidate | resident MiB | wait ms/token |
|---|---:|---:|
| `blk.1.upgate` | `2233.96` | `8.350` |
| `blk.9.upgate` | `2569.97` | `6.826` |
| `blk.7.upgate` | `2284.31` | `6.736` |
| `blk.28.upgate` | `2195.63` | `6.693` |
| `blk.1.all` | `3731.85` | `11.502` |
| `blk.7.all` | `4111.59` | `10.500` |

Greedy upper bound:

| family | budget MiB | resident MiB | saved wait ms/token |
|---|---:|---:|---:|
| `layer_upgate` | `2048` | `1803.3` | `5.91` |
| `layer_upgate` | `4096` | `4028.2` | `14.26` |
| `layer_upgate` | `8192` | `7338.6` | `25.00` |
| `layer_upgate` | `10240` | `9240.5` | `31.14` |
| `layer_all` | `4096` | `3731.8` | `11.50` |
| `layer_all` | `8192` | `6729.4` | `20.13` |
| `layer_all` | `10240` | `9732.5` | `28.58` |

The most favorable `10 GiB` slab bound saves only about `31 ms/token`, while
the current trace needs about `87 ms/token` to reach `2 tok/s`. This is an
upper bound before TTFT preload, cgroup pressure, page-cache displacement,
runtime RAM lookup overhead, and mixed-source scheduling overhead.

## Decision

Reject RAM second-tier as the next primary runtime A/B.

Reasons:

- Dynamic top-expert RAM tier has poor batch dominance even at `10 GiB`.
- Whole layer/role slabs are batch-cleaner, but the saved wait bound is too
  small relative to the `>2 tok/s` gap.
- A runtime implementation would consume most remaining host RAM and add TTFT
  or asynchronous preload complexity without a strong enough endpoint bound.
- This does not mean RAM is useless; it means RAM second-tier alone is not the
  next high-value path under the current representation and cache split.

Next action:

- Do not implement RAM second-tier runtime now.
- Keep `VRAM_MIB=15000`, `UPGATE_PCT=72` as the reproduction default.
- The remaining credible path is either:
  - explicit approval for the complete-model `i1-IQ1_S` smoke; or
  - a genuinely new lower-byte expert representation with prompt-level
    output-error evidence, not a retune of the rejected families.

## Reproduce

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-admission-ioread-n96-083914
OUT=.Agent/runs/20260712-current-goal-ram-second-tier-admission

# Generate io-wait-trace.csv from io-batch-profile.csv.
python3 - <<'PY'
import csv, pathlib
root = pathlib.Path("/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-admission-ioread-n96-083914")
for run in sorted(root.iterdir()):
    src = run / "io-batch-profile.csv"
    dst = run / "io-wait-trace.csv"
    if not src.exists():
        continue
    with src.open(newline="") as f, dst.open("w", newline="") as g:
        reader = csv.DictReader(f)
        fields = ["seq", "batch_seq", "op", "first_tensor", "last_tensor",
                  "read_jobs", "completed_before", "inflight_before",
                  "next_job", "depth", "refill_batch", "wait_ms",
                  "drained_cqes", "enqueue_ms", "completed_after",
                  "inflight_after"]
        writer = csv.DictWriter(g, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            writer.writerow({
                "seq": row.get("seq", ""),
                "batch_seq": row.get("seq", ""),
                "op": row.get("op", ""),
                "first_tensor": row.get("first_tensor", ""),
                "last_tensor": row.get("last_tensor", ""),
                "read_jobs": row.get("read_jobs", ""),
                "depth": row.get("depth", ""),
                "refill_batch": row.get("refill_batch", ""),
                "wait_ms": row.get("wait_ms", "0"),
                "drained_cqes": row.get("cqes", ""),
                "enqueue_ms": row.get("enqueue_ms", ""),
            })
PY

for B in 2048 4096 6144 8192 10240; do
  python3 .Agent/run-tools/kimi_ram_candidate_multidev_screen.py \
    --io-read-trace "$ROOT/dev_france_regression/io-read-trace.csv" \
    --io-read-trace "$ROOT/dev_intelligence_general/io-read-trace.csv" \
    --route-profile "$ROOT/dev_france_regression/route-profile.csv" \
    --route-profile "$ROOT/dev_intelligence_general/route-profile.csv" \
    --out-profile "$OUT/upgate-budget-${B}m/ram-profile.csv" \
    --out-report "$OUT/upgate-budget-${B}m/report.json" \
    --out-csv "$OUT/upgate-budget-${B}m/candidates.csv" \
    --budget-mib "$B" --roles up,gate --max-jobs 8 \
    --min-count 2 --min-prompts 2 \
    --upgate-slots 2015 --down-slots 533 --exclude-vram

  python3 .Agent/run-tools/kimi_ram_candidate_multidev_screen.py \
    --io-read-trace "$ROOT/dev_france_regression/io-read-trace.csv" \
    --io-read-trace "$ROOT/dev_intelligence_general/io-read-trace.csv" \
    --route-profile "$ROOT/dev_france_regression/route-profile.csv" \
    --route-profile "$ROOT/dev_intelligence_general/route-profile.csv" \
    --out-profile "$OUT/allroles-budget-${B}m/ram-profile.csv" \
    --out-report "$OUT/allroles-budget-${B}m/report.json" \
    --out-csv "$OUT/allroles-budget-${B}m/candidates.csv" \
    --budget-mib "$B" --roles up,gate,down --max-jobs 8 \
    --min-count 2 --min-prompts 2 \
    --upgate-slots 2015 --down-slots 533 --exclude-vram
done

python3 .Agent/run-tools/kimi_phase5c_ram_slab_screen.py \
  --input-root "$ROOT" --route-profile-root "$ROOT" \
  --out-dir "$OUT/phase5c-slab-screen-with-wait" \
  --roles up,gate,down --max-jobs 8 \
  --down-slots 533 --upgate-slots 2015 \
  --contig-window-mib 512 1024 2048 \
  --contig-top-per-source 12
```
