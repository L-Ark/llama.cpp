# Gate/Up/Down Static Layout LOO Bound

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Input root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717`

This is a dev-only offline bound. It does not rewrite any expert pack and does
not claim SOTA.

## Goal

After the current-HEAD CPU/defer audit showed `0` true fallback rows and full
GPU-extension acceptance, test whether a static gate/up/down pack layout can
reduce io_uring read fragmentation enough to justify a runtime A/B.

The important constraint is prompt generalization: layout order is trained on
one dev prompt and evaluated on the other prompt. Held-out/test prompts are not
used.

## Reproduce

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

python3 .Agent/run-tools/kimi_pack_layout_loo_bound.py \
  --input-root "$ROOT" \
  --out-json "$OUT/loo-bound-up_gate.json" \
  --out-md "$OUT/loo-bound-up_gate.md" \
  --roles up,gate \
  --max-jobs 8 \
  --max-gap-mib 1.0 \
  --baseline-decode-ms "$BASE"

python3 .Agent/run-tools/kimi_pack_layout_loo_bound.py \
  --input-root "$ROOT" \
  --out-json "$OUT/loo-bound-down.json" \
  --out-md "$OUT/loo-bound-down.md" \
  --roles down \
  --max-jobs 8 \
  --max-gap-mib 1.0 \
  --baseline-decode-ms "$BASE"
```

## Results

Baseline decode input:

- prompts: `dev_france_regression`, `dev_intelligence_general`
- combined decode ms: `42527.100`
- combined decode runs: `62`

All roles together:

| layout | saved reads | saved ms upper | saved ms/token | bounded tok/s |
|---|---:|---:|---:|---:|
| `greedy_pair` | `2845` | `1188.869` | `19.175` | `1.500` |
| `first_use` | `2298` | `957.777` | `15.448` | `1.491` |
| `frequency` | `1376` | `575.416` | `9.281` | `1.478` |
| `expert_id` | `677` | `280.297` | `4.521` | `1.468` |

Up/gate only:

| layout | saved reads | saved ms upper | saved ms/token | bounded tok/s |
|---|---:|---:|---:|---:|
| `greedy_pair` | `1687` | `705.017` | `11.371` | `1.482` |
| `first_use` | `1388` | `578.567` | `9.332` | `1.478` |
| `frequency` | `808` | `337.987` | `5.451` | `1.470` |
| `expert_id` | `372` | `153.940` | `2.483` | `1.463` |

Down only:

| layout | saved reads | saved ms upper | saved ms/token | bounded tok/s |
|---|---:|---:|---:|---:|
| `greedy_pair` | `1158` | `483.852` | `7.804` | `1.475` |
| `first_use` | `910` | `379.210` | `6.116` | `1.471` |
| `frequency` | `568` | `237.429` | `3.830` | `1.466` |
| `expert_id` | `305` | `126.357` | `2.038` | `1.462` |

## Decision

Reject static prompt-general pack relayout as the next runtime A/B.

Reasons:

- Best optimistic leave-one-prompt-out bound is only `19.175 ms/token` saved.
- Bounded token rate reaches only `1.500 tok/s`, far below the current short-term
  target of `>2 tok/s`.
- Role split confirms no single static layout target is enough:
  `up,gate` saves at most `11.371 ms/token`, and `down` saves at most
  `7.804 ms/token`.
- This bound is optimistic because it assumes fewer read extents linearly reduce
  exposed io_uring wait and ignores pack rebuild/runtime overhead.
- Same-prompt layout gains are therefore prompt-specific overfit unless a future
  cross-prompt/held-out result contradicts this LOO screen.

## Next Direction

Do not spend the next implementation cycle on static expert order relayout.
Focus on changes that can remove much larger cost:

1. reduce expert bytes/token with a quality-gated lower-byte representation;
2. improve runtime queue continuity using exact known current-layer work without
   adding overfetch;
3. test RAM/VRAM tiering only when it replaces low-value page cache and reduces
   endpoint decode time, not only hit rate;
4. keep CPU/defer GPU-extension acceptance and `0` fallback as regression gates.
