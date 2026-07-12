# Kimi RAM slab screen from per-read IO traces

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Commit: `c7b69a97e`

This is a dev-only offline screen. It does not change runtime behavior and
does not claim SOTA.

## Purpose

The previous next-candidate screen showed that coarse layer/role RAM residency
looked too weak to reach `2 tok/s`, but it lacked `io-read-trace.csv`. This
run collects per-read traces on generalized dev prompts and reruns the RAM
screen with actual read locality and complete-batch coverage.

The question is whether replacing low-yield page cache with explicit RAM
expert data can be a primary path to `2 tok/s`.

## Trace Collection

Source root:

`/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-c7b69-io-read-trace-n32-114538`

Prompts:

- `dev_france_regression`: `Please introduce France in a short paragraph.`
- `dev_intelligence_general`: `What is intelligence?`

Both runs used:

- `N=32`;
- `MemoryMax=15900000000`;
- `MemorySwapMax=0`;
- cold start via the existing repro script's `sync` and `drop_caches`;
- `PROFILE=1`;
- `COPY_PROFILE=0`;
- `GGML_MOE_IO_READ_TRACE_OUT`;
- `GGML_MOE_IO_WAIT_TRACE_OUT`;
- `GGML_MOE_IO_BATCH_PROFILE_OUT`;
- `GGML_MOE_IO_LOCALITY_PROFILE_OUT`.

Quality and resource gates:

| prompt | quality | tok/s | decode | TTFT ms | RAM peak | fallback | direct reads |
|---|---|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.68` | `18399.58 ms / 31` | `8790.36` | `12773388288` | `0` | `0` |
| `dev_intelligence_general` | pass | `1.67` | `18592.35 ms / 31` | `6965.02` | `12609302528` | `0` | `0` |

Trace sizes:

- France `io-read-trace.csv`: `38798` lines;
- Intelligence `io-read-trace.csv`: `37045` lines.

## Complete-Batch RAM Cover Oracle

Artifact:

- `.Agent/runs/20260712-active-goal-c7b69-ram-slab-trace-screen/batch-cover-oracle.md`

This is a dev-overfit upper bound: it greedily selects arbitrary complete IO
batches from the profiled prompts. A real prompt-general RAM policy should do
worse unless it can predict the same future batches.

Inputs:

- traces: `2`;
- prompts: `2`;
- decode runs: `62`;
- baseline decode: `36991.93 ms`;
- baseline rate: `1.676 tok/s`;
- profiled IO wait: `30673.200 ms`, `494.729 ms/token`;
- max jobs: `8`.

Greedy complete-batch cover bound:

| RAM budget | resident GiB | covered batches | saved ms/token | bounded tok/s |
|---:|---:|---:|---:|---:|
| `2048 MiB` | `2.000` | `267` | `11.694` | `1.710` |
| `4096 MiB` | `3.999` | `525` | `22.300` | `1.741` |
| `6144 MiB` | `5.998` | `782` | `32.549` | `1.773` |
| `8192 MiB` | `8.000` | `1033` | `42.962` | `1.806` |
| `10240 MiB` | `9.996` | `1267` | `53.571` | `1.841` |
| `12288 MiB` | `11.998` | `1493` | `63.659` | `1.876` |

Decision:

- Even with unrealistic dev-batch overfitting and `12 GiB` RAM, the bound is
  only `1.876 tok/s`, below the `2 tok/s` milestone.
- Real RAM residency still pays RAM-to-VRAM H2D and can fragment remaining SSD
  batches, so measured runtime would likely be lower than the bound.

## Layer and Slab Screen

Artifact:

- `.Agent/runs/20260712-active-goal-c7b69-ram-slab-trace-screen/ram-slab-screen/report.md`

Top reusable whole-layer candidates:

| candidate | resident MiB | wait ms/token | hit batches | mixed risk |
|---|---:|---:|---:|---:|
| `blk.9.upgate` | `1407.88` | `7.174` | `124` | `0` |
| `blk.1.upgate` | `1426.35` | `7.042` | `124` | `0` |
| `blk.29.upgate` | `1467.01` | `6.438` | `124` | `0` |
| `blk.48.upgate` | `1496.62` | `6.392` | `124` | `0` |
| `blk.28.upgate` | `1417.67` | `6.249` | `124` | `0` |
| `blk.9.all` | `2534.12` | `10.895` | `186` | `0` |
| `blk.1.all` | `2382.84` | `10.043` | `186` | `0` |
| `blk.7.all` | `2215.12` | `9.836` | `184` | `0` |

Greedy layer slab bounds from the screen:

- `layer_upgate`, `10240 MiB`: `49.90 ms/token`;
- `layer_all`, `10240 MiB`: `40.51 ms/token`.

Decision:

- Whole layer/role slabs are cleaner than scattered RAM entries because they
  have `mixed risk = 0` in the screen.
- Their wait savings are still too small to bridge the current `2 tok/s` gap.
- A small pageable RAM slab A/B may be useful later as an auxiliary
  optimization, but it should not precede lower-byte movement.

## Final Decision

RAM/VRAM storage redesign remains useful for cleanup and incremental gains, but
this per-read trace screen rejects RAM slabs as the next primary route to
`2 tok/s`.

Next priority stays:

1. Lower-byte movement with target moved-byte ratio `<=0.6x` and strict
   output-quality gates.
2. If an approved complete-model lower-byte smoke is allowed, test
   `i1-IQ1_S` dev-only under the same RAM/TTFT/fallback gates.
3. If approval is not available, design a new non-destructive lower-byte
   representation; do not implement the already rejected blockwise or
   structured candidates.
4. Treat RAM slabs as auxiliary only after lower-byte or prediction reduces
   the remaining movement enough that `40-60 ms/token` matters.

## Reproduce

Trace collection:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-c7b69-io-read-trace-n32-114538

run_one() {
  id="$1"
  prompt="$2"
  keywords="$3"
  run="$ROOT/$id"
  trace_env="GGML_MOE_IO_BATCH_PROFILE_OUT=$run/io-batch-profile.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$run/io-wait-trace.csv
GGML_MOE_IO_READ_TRACE_OUT=$run/io-read-trace.csv
GGML_MOE_IO_LOCALITY_PROFILE_OUT=$run/io-locality-profile.csv
GGML_MOE_STAGE_GRANULARITY_PROFILE=1"
  systemd-run --wait --collect --same-dir \
    -p MemoryMax=15900000000 -p MemorySwapMax=0 \
    env RUN="$run" N=32 PROMPT_ID="$id" \
        PROMPT_USER_TEXT="$prompt" \
        QUALITY_KEYWORDS="$keywords" \
        PROFILE=1 COPY_PROFILE=0 VRAM_MIB=15000 PINNED_SLOTS=12 \
        UPGATE_PCT=72 IQ2_UPGATE_PARALLEL=1 \
        MOE_IO_DEPTH=8 MOE_IO_REFILL_BATCH=4 MOE_PREFETCH_DOWN_DEPTH=2 \
        EXTRA_RUNTIME_ENV="$trace_env" \
        .Agent/run-tools/kimi-general-prompt-repro.sh
}

run_one dev_france_regression \
  "Please introduce France in a short paragraph." \
  "france|french,europe|paris|country"

run_one dev_intelligence_general \
  "What is intelligence?" \
  "intelligence,learn|reason|adapt|solve"
```

Offline reports:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-c7b69-io-read-trace-n32-114538
OUT=.Agent/runs/20260712-active-goal-c7b69-ram-slab-trace-screen
BASE=$(python3 - <<'PY' "$ROOT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(sum(float(json.load(open(p)).get("decode_ms", 0)) for p in root.glob("*/metrics.json")))
PY
)

python3 .Agent/run-tools/kimi_batch_cover_ram_oracle.py \
  --input-root "$ROOT" \
  --out-json "$OUT/batch-cover-oracle.json" \
  --out-md "$OUT/batch-cover-oracle.md" \
  --roles up,gate,down \
  --max-jobs 8 \
  --budgets-mib 2048,4096,6144,8192,10240,12288 \
  --baseline-decode-ms "$BASE"

python3 .Agent/run-tools/kimi_phase5c_ram_slab_screen.py \
  --input-root "$ROOT" \
  --route-profile-root "$ROOT" \
  --out-dir "$OUT/ram-slab-screen" \
  --roles up,gate,down \
  --max-jobs 8 \
  --contig-window-mib 512 1024 2048 4096 \
  --contig-top-per-source 20
```
