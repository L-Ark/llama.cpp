# Lower-Byte Next Admission

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Input root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717`

This is a dev-only offline admission report. It does not change runtime behavior
and does not claim SOTA.

## Goal

After rejecting broad fallback work, static pack relayout, and scheduler-only
queue continuity, quantify the lower-byte path needed for the next Kimi token
rate milestone.

The practical question is whether any current lower-byte family can plausibly
reach the short-term `>2 tok/s` target while preserving quality.

## Exposed-Wait Byte Bound

Command:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
OUT=.Agent/runs/20260712-lowerbyte-next-admission
BASE=$(python3 - <<'PY' "$ROOT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(sum(float(json.load(open(p)).get("decode_ms", 0)) for p in root.glob("*/metrics.json")))
PY
)

python3 .Agent/run-tools/kimi_low_byte_expert_bound.py \
  --input-root "$ROOT" \
  --out-json "$OUT/low-byte-bound.json" \
  --out-md "$OUT/low-byte-bound.md" \
  --roles up,gate,down \
  --max-jobs 8 \
  --keep-ratios 0.75,0.732,0.50,0.254,0.25,0.0 \
  --targets 2.0,5.0 \
  --baseline-decode-ms "$BASE"
```

Inputs:

- traces: `2`;
- batches: `11134`;
- decode runs: `62`;
- baseline decode: `42527.100 ms`;
- baseline token rate: `1.458 tok/s`;
- profiled IO wait: `23006.531 ms`;
- profiled payload: `277.515 GiB`.

Role contribution:

| role | payload GiB | max linear wait ms/token | eliminate-role bound |
|---|---:|---:|---:|
| `gate` | `78.777` | `135.452` | `1.817 tok/s` |
| `up` | `73.175` | `122.284` | `1.774 tok/s` |
| `down` | `125.563` | `113.337` | `1.746 tok/s` |

Scenarios:

| scenario | byte reduction | bounded tok/s |
|---|---:|---:|
| `all` | `50%` | `1.998` |
| `all` | `75%` | `2.444-2.453` |
| `all` | `100%` | `3.176` |
| `up+gate` | `50%` | `1.795` |
| `up+gate` | `75%` | `2.026-2.030` |
| `down` | `100%` | `1.746` |

Target feasibility:

| target | required saving | required all-role reduction | possible by IO-byte reduction only |
|---|---:|---:|---|
| `2 tok/s` | `185.921 ms/token` | `50.1%` | yes, barely |
| `5 tok/s` | `485.921 ms/token` | `131.0%` | no |

Interpretation:

- `2 tok/s` requires roughly all-role `0.50x` effective movement, not just a
  small hotset/cache cleanup.
- A role-only reduction is not enough; even eliminating all down movement is
  bounded below `1.75 tok/s`.
- `5 tok/s` is not reachable from IO byte reduction alone on this baseline.

## Split-Pool Bound Under Lower-Byte Ratios

Command:

```bash
python3 .Agent/run-tools/kimi_quant_split_sweep_bound.py \
  --runs-root "$ROOT" \
  --ratios 1.0,0.732,0.50,0.254 \
  --pcts 40,45,50,55,60,62,65,70,75,80 \
  --current-pct 62 \
  --out-json "$OUT/quant-split-sweep.json" \
  --out-md "$OUT/quant-split-sweep.md"
```

Result:

| byte ratio | current split miss GiB/tok | best split | best miss GiB/tok | relative gain |
|---:|---:|---:|---:|---:|
| `1.000` | `7.004` | `62` | `7.004` | `0.00%` |
| `0.732` | `4.558` | `62` | `4.558` | `0.00%` |
| `0.500` | `2.584` | `62` | `2.584` | `0.00%` |
| `0.254` | `0.786` | `60` | `0.785` | `0.25%` |

Interpretation:

- The current `62%` upgate split is already near-optimal for this dev root.
- Lower-byte entries improve hit rates because more experts fit in VRAM, but
  changing the split itself is not the main source of gain.
- For a future `0.50x` runtime smoke, keep the split near `62%` initially.

## Error Gate For Existing Blockwise Lower-Byte Family

Command:

```bash
python3 .Agent/run-tools/kimi_mixed_role_byte_error_budget.py \
  --screen-json .Agent/runs/20260712-current-goal-lowbyte-freeze-screen/gp77-target050-summary.json \
  --out-json "$OUT/mixed-role-target050.json" \
  --out-md "$OUT/mixed-role-target050.md" \
  --target-global-ratio 0.50 \
  --target-mean-rel-l2 0.10
```

Result:

- combinations under `0.50x`: `16`;
- passing combinations: `0`;
- best under-budget combination:
  - global ratio `0.3861`;
  - down mean rel-L2 `0.499277`;
  - fused up/gate mean rel-L2 `0.598174`;
  - worst mean rel-L2 `0.598174`;
  - decision `reject`.

Interpretation:

- The existing GP68-GP77 blockwise 1-bit residual family is rejected even when
  the byte target is relaxed to the `0.50x` needed for the `2 tok/s` milestone.
- The failure is driven by up/gate/fused-upgate output error, not storage size.

## Complete-Model Candidate State

Existing committed storage gate:

- `.Agent/runs/20260712-active-goal-lowerbyte-storage-gate/report.md`

Relevant conclusion:

- `i1-IQ1_S` complete model size: `204430872480 bytes`;
- ratio versus current IQ3_S model directory: `0.504x`;
- it is the only current complete-model candidate worth a `2 tok/s` smoke;
- it is not a `5 tok/s` solution;
- current free space is insufficient unless non-SOTA cleanup candidates are
  deleted;
- the guarded script defaults to dry-run and requires explicit confirmation:
  `EXECUTE=1 DELETE_OLD_PACKS=1 CONFIRM_DELETE=DELETE_OLD_KIMI_NON_SOTA_PACKS`.

Current check in this admission:

- `/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF` is not present;
- `/root/lfz` free space is about `106 GiB`;
- no deletion or download was performed.

## Decision

Lower-byte remains the next primary direction, but only with strict admission:

1. Do not implement the existing blockwise residual family as runtime code.
   It fails the `0.50x` byte/error gate.
2. A practical `2 tok/s` runtime smoke requires roughly all-role `0.50x`
   effective movement and must include up/gate, not only down.
3. The only concrete complete-model smoke candidate currently identified is
   `i1-IQ1_S`; running it requires explicit storage cleanup/download action.
4. For any future byte-reduced runtime path:
   - keep CPU/defer GPU-extension `batch_accept == calls`;
   - keep true fallback rows at `0`;
   - cold start under `16 GB` host RAM;
   - France answer must remain semantic and coherent;
   - TTFT ratio must stay `<=1.20`;
   - results must be committed and pushed with exact reproduction commands.

## Next Action

If storage cleanup/download is explicitly approved, run the guarded
`i1-IQ1_S` complete-model smoke first as dev-only:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
EXECUTE=1 \
DELETE_OLD_PACKS=1 \
CONFIRM_DELETE=DELETE_OLD_KIMI_NON_SOTA_PACKS \
DOWNLOAD=1 \
RUN_SMOKE=1 \
VALIDATE_PARTS=1 \
N=32 \
.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh
```

If that is not approved, the next non-destructive work should design a new
activation-aware lower-byte representation that targets `<=0.50x` effective
movement with mean output rel-L2 much closer to `<=0.10` on multi-prompt dev
activations.
