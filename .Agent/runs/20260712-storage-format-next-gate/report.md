# Kimi storage-format next gate

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`

This is a dev-only offline decision report. It does not rewrite packs, delete
files, download models, change runtime behavior, inspect held-out/test prompts,
or claim SOTA.

## Goal

After rejecting broad CPU fallback work, static RAM hot tiers, activation-aware
local compression, and route-history prefetch, decide whether storage-format
work that keeps the current IQ3_S payload can still justify a runtime A/B.

The key distinction is:

- useful storage format: reduces moved bytes or exposed wait on generalized
  prompts under cold start;
- weak storage format: only reorders the same bytes or improves a prompt-specific
  trace while failing leave-one-prompt-out generalization.

## Inputs

Primary profile root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
```

Prompts:

- `dev_france_regression`
- `dev_intelligence_general`

Supporting artifacts:

- `.Agent/runs/20260712-exact-layout-storage-bound/dev_france_regression-layout.md`
- `.Agent/runs/20260712-exact-layout-storage-bound/dev_intelligence_general-layout.md`
- `.Agent/runs/20260712-gate-updown-layout-loo-bound/report.md`
- `.Agent/runs/20260708-gp106-expert-pack-lossless-bound/report.md`
- `.Agent/runs/20260707-gp23-lowbyte-subpack-feasibility/report.md`

## Exact Layout Bound

Per-prompt IO-trace layout screens show that a perfect per-batch layout could
reduce read extents enough to exceed `2 tok/s` on the two dev prompts:

| prompt | current read GiB | unique exact batches | repeated exact batches | ideal bounded tok/s | static greedy-pair tok/s |
|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | `137.759` | `5575` | `8` | `2.29` | `1.86` |
| `dev_intelligence_general` | `139.756` | `5549` | `11` | `2.19` | `2.02` |

Interpretation:

- The ideal layout is an upper bound, not a static pack design.
- In the tool, ideal rows with the same source/tensor/layer grouping become
  adjacent with zero gap for that batch. That is equivalent to knowing the
  active experts for each batch and arranging them together.
- Batch signatures are almost all unique, so a finite static pack cannot realize
  the ideal layout without duplicating massive payloads.
- The ideal layout does not reduce moved bytes: read traffic remains about
  `138-140 GiB` per N32 prompt. It only reduces read extent count.

## Static Layout Generalization

Leave-one-prompt-out static layout already rejects prompt-general pack relayout:

| layout | saved ms/token | bounded tok/s |
|---|---:|---:|
| `greedy_pair` | `19.175` | `1.500` |
| `first_use` | `15.448` | `1.491` |
| `frequency` | `9.281` | `1.478` |
| `expert_id` | `4.521` | `1.468` |

Role split does not rescue it:

- `up,gate` saves at most `11.371 ms/token`;
- `down` saves at most `7.804 ms/token`;
- both are far below what is needed for stable `>2 tok/s`.

Decision: do not implement static expert-order pack relayout as the next
runtime A/B. Same-prompt gains are prompt-order overfit.

## Lossless Payload Compression

Generic lossless compression on sampled expert-pack payloads does not reduce
bytes enough:

| mode | best codec | aggregate ratio | decision |
|---|---|---:|---|
| raw payload | `zlib6` | `0.9931` | fail |
| base-XOR residual, including base-self rows | `lzma6` | `0.7501` | fail |
| base-XOR residual, excluding base-self rows | `lzma6` | `1.0001` | fail |

Interpretation:

- Current quantized expert payloads are already near entropy-dense.
- Base-XOR only looks better when counting base-self rows that collapse to near
  zero; real non-base residual rows do not compress.
- Even `0.75x` is weaker than the roughly `0.50x` all-role movement needed for
  the short-term `2 tok/s` target and would add decompression overhead.

Decision: do not implement generic compressed-pack runtime.

## Split Lower-Byte Subpacks

Prior dry-run shows selected lower-byte subpacks can fit individually:

| subpack | payload GiB | fits current filesystem in dry-run |
|---|---:|---|
| `up+gate` | `61.797` | yes |
| `down` | `53.418` | yes |

But this is not directly runnable with the current IQ3_S GGUF:

- entries have runtime byte/type mismatches;
- current runtime lookup expects current GGUF expert byte size/type;
- a lower-byte override path would need new kernels/type handling and a quality
  gate, or the real lower-byte GGUF must be loaded.

Decision: selected lower-byte subpack is not a safe next runtime A/B unless it
is paired with an explicit lower-byte expert override design and quality gate.

## Decision

Storage-format work that keeps the current IQ3_S payload is rejected as the next
primary path.

Reasons:

- exact per-batch layout can cross `2 tok/s` only as an unrealizable upper bound;
- static prompt-general layout saves only about `19 ms/token`;
- lossless compression does not materially reduce expert bytes;
- split packs alone do not reduce bytes and lower-byte split packs need new type
  override support before they can run.

The next accepted storage/payload direction must satisfy one of:

1. load and smoke a complete lower-bit model, such as `i1-IQ1_S`, after explicit
   cleanup/download approval;
2. implement a real lower-byte expert override path, default-off, with direct
   kernels and current-pack output quality gates before runtime SOTA testing;
3. introduce a stronger predictor that proves complete-batch coverage before
   issuing runtime reads.

## Reproduce

Per-prompt exact layout screens:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
OUT=.Agent/runs/20260712-exact-layout-storage-bound
mkdir -p "$OUT"
for P in dev_france_regression dev_intelligence_general; do
  python3 .Agent/run-tools/kimi_io_trace_pack_layout_screen.py \
    --io-read-trace "$ROOT/$P/io-read-trace.csv" \
    --baseline-metrics "$ROOT/$P/metrics.txt" \
    --out-json "$OUT/$P-layout.json" \
    --out-md "$OUT/$P-layout.md" \
    --roles up,gate,down \
    --max-jobs 8 \
    --max-gap-mib 1.0
done
```

Static leave-one-prompt-out layout:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
OUT=.Agent/runs/20260712-gate-updown-layout-loo-bound
BASE=$(python3 - <<'PY' "$ROOT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(sum(float(json.load(open(p)).get("decode_ms", 0)) for p in root.glob("*/metrics.json")))
PY
)
python3 .Agent/run-tools/kimi_pack_layout_loo_bound.py \
  --input-root "$ROOT" \
  --out-json "$OUT/loo-bound.json" \
  --out-md "$OUT/loo-bound.md" \
  --roles up,gate,down \
  --max-jobs 8 \
  --max-gap-mib 1.0 \
  --baseline-decode-ms "$BASE"
```
