# Kimi Pack Layout Leave-One-Prompt-Out Bound

This is a dev-only offline screen. It does not rewrite packs or claim SOTA.

- traces: `2`
- roles: `gate,up`
- max jobs: `8`
- max gap: `1.00 MiB`

## LOO Summary

| layout | saved reads | saved ms upper | saved ms/token | bounded tok/s | missing |
|---|---:|---:|---:|---:|---:|
| `greedy_pair` | `1687` | `705.017` | `11.371` | `1.482` | `0` |
| `first_use` | `1388` | `578.567` | `9.332` | `1.478` | `0` |
| `frequency` | `808` | `337.987` | `5.451` | `1.470` | `0` |
| `expert_id` | `372` | `153.940` | `2.483` | `1.463` | `0` |

## Per Prompt

| train prompts | test prompt | layout | current extents | layout extents | saved reads | saved ms upper | bounded tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| `dev_intelligence_general` | `dev_france_regression` | `expert_id` | `15517` | `15423` | `94` | `40.597` | `1.422` |
| `dev_intelligence_general` | `dev_france_regression` | `frequency` | `15517` | `15163` | `354` | `152.888` | `1.429` |
| `dev_intelligence_general` | `dev_france_regression` | `first_use` | `15517` | `14993` | `524` | `226.309` | `1.434` |
| `dev_intelligence_general` | `dev_france_regression` | `greedy_pair` | `15517` | `14805` | `712` | `307.503` | `1.440` |
| `dev_france_regression` | `dev_intelligence_general` | `expert_id` | `15850` | `15572` | `278` | `113.342` | `1.507` |
| `dev_france_regression` | `dev_intelligence_general` | `frequency` | `15850` | `15396` | `454` | `185.099` | `1.512` |
| `dev_france_regression` | `dev_intelligence_general` | `first_use` | `15850` | `14986` | `864` | `352.259` | `1.524` |
| `dev_france_regression` | `dev_intelligence_general` | `greedy_pair` | `15850` | `14875` | `975` | `397.514` | `1.528` |

## Interpretation

- Best leave-one-prompt-out static layout is greedy_pair with 11.371 ms/token saved and bounded 1.482 tok/s.
- This is still an optimistic per-read-wait upper bound: it assumes fewer extents translate linearly to less exposed iouring wait and ignores pack rebuild overhead.
- A runtime A/B is justified only if the LOO bound is comfortably above the target and a default-off pack layout can be built without prompt-specific held-out tuning.

## Reproduce

```bash
.Agent/run-tools/kimi_pack_layout_loo_bound.py --input-root /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717 --out-json .Agent/runs/20260712-gate-updown-layout-loo-bound/loo-bound-up_gate.json --out-md .Agent/runs/20260712-gate-updown-layout-loo-bound/loo-bound-up_gate.md --roles up,gate --max-jobs 8 --max-gap-mib 1.0 --baseline-decode-ms 42527.100000000006
```
