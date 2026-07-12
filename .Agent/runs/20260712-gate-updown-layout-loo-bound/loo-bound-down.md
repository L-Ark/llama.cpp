# Kimi Pack Layout Leave-One-Prompt-Out Bound

This is a dev-only offline screen. It does not rewrite packs or claim SOTA.

- traces: `2`
- roles: `down`
- max jobs: `8`
- max gap: `1.00 MiB`

## LOO Summary

| layout | saved reads | saved ms upper | saved ms/token | bounded tok/s | missing |
|---|---:|---:|---:|---:|---:|
| `greedy_pair` | `1158` | `483.852` | `7.804` | `1.475` | `0` |
| `first_use` | `910` | `379.210` | `6.116` | `1.471` | `0` |
| `frequency` | `568` | `237.429` | `3.830` | `1.466` | `0` |
| `expert_id` | `305` | `126.357` | `2.038` | `1.462` | `0` |

## Per Prompt

| train prompts | test prompt | layout | current extents | layout extents | saved reads | saved ms upper | bounded tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| `dev_intelligence_general` | `dev_france_regression` | `expert_id` | `9500` | `9417` | `83` | `35.847` | `1.422` |
| `dev_intelligence_general` | `dev_france_regression` | `frequency` | `9500` | `9258` | `242` | `104.517` | `1.426` |
| `dev_intelligence_general` | `dev_france_regression` | `first_use` | `9500` | `9161` | `339` | `146.410` | `1.429` |
| `dev_intelligence_general` | `dev_france_regression` | `greedy_pair` | `9500` | `9015` | `485` | `209.465` | `1.433` |
| `dev_france_regression` | `dev_intelligence_general` | `expert_id` | `9727` | `9505` | `222` | `90.511` | `1.505` |
| `dev_france_regression` | `dev_intelligence_general` | `frequency` | `9727` | `9401` | `326` | `132.912` | `1.508` |
| `dev_france_regression` | `dev_intelligence_general` | `first_use` | `9727` | `9156` | `571` | `232.801` | `1.516` |
| `dev_france_regression` | `dev_intelligence_general` | `greedy_pair` | `9727` | `9054` | `673` | `274.387` | `1.519` |

## Interpretation

- Best leave-one-prompt-out static layout is greedy_pair with 7.804 ms/token saved and bounded 1.475 tok/s.
- This is still an optimistic per-read-wait upper bound: it assumes fewer extents translate linearly to less exposed iouring wait and ignores pack rebuild overhead.
- A runtime A/B is justified only if the LOO bound is comfortably above the target and a default-off pack layout can be built without prompt-specific held-out tuning.

## Reproduce

```bash
.Agent/run-tools/kimi_pack_layout_loo_bound.py --input-root /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717 --out-json .Agent/runs/20260712-gate-updown-layout-loo-bound/loo-bound-down.json --out-md .Agent/runs/20260712-gate-updown-layout-loo-bound/loo-bound-down.md --roles down --max-jobs 8 --max-gap-mib 1.0 --baseline-decode-ms 42527.100000000006
```
