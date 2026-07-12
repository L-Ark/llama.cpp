# Kimi Pack Layout Leave-One-Prompt-Out Bound

This is a dev-only offline screen. It does not rewrite packs or claim SOTA.

- traces: `2`
- roles: `down,gate,up`
- max jobs: `8`
- max gap: `1.00 MiB`

## LOO Summary

| layout | saved reads | saved ms upper | saved ms/token | bounded tok/s | missing |
|---|---:|---:|---:|---:|---:|
| `greedy_pair` | `2845` | `1188.869` | `19.175` | `1.500` | `0` |
| `first_use` | `2298` | `957.777` | `15.448` | `1.491` | `0` |
| `frequency` | `1376` | `575.416` | `9.281` | `1.478` | `0` |
| `expert_id` | `677` | `280.297` | `4.521` | `1.468` | `0` |

## Per Prompt

| train prompts | test prompt | layout | current extents | layout extents | saved reads | saved ms upper | bounded tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| `dev_intelligence_general` | `dev_france_regression` | `expert_id` | `25017` | `24840` | `177` | `76.444` | `1.424` |
| `dev_intelligence_general` | `dev_france_regression` | `frequency` | `25017` | `24421` | `596` | `257.405` | `1.436` |
| `dev_intelligence_general` | `dev_france_regression` | `first_use` | `25017` | `24154` | `863` | `372.718` | `1.444` |
| `dev_intelligence_general` | `dev_france_regression` | `greedy_pair` | `25017` | `23820` | `1197` | `516.969` | `1.454` |
| `dev_france_regression` | `dev_intelligence_general` | `expert_id` | `25577` | `25077` | `500` | `203.853` | `1.513` |
| `dev_france_regression` | `dev_intelligence_general` | `frequency` | `25577` | `24797` | `780` | `318.011` | `1.522` |
| `dev_france_regression` | `dev_intelligence_general` | `first_use` | `25577` | `24142` | `1435` | `585.059` | `1.542` |
| `dev_france_regression` | `dev_intelligence_general` | `greedy_pair` | `25577` | `23929` | `1648` | `671.901` | `1.549` |

## Interpretation

- Best leave-one-prompt-out static layout is greedy_pair with 19.175 ms/token saved and bounded 1.500 tok/s.
- This is still an optimistic per-read-wait upper bound: it assumes fewer extents translate linearly to less exposed iouring wait and ignores pack rebuild overhead.
- A runtime A/B is justified only if the LOO bound is comfortably above the target and a default-off pack layout can be built without prompt-specific held-out tuning.

## Reproduce

```bash
.Agent/run-tools/kimi_pack_layout_loo_bound.py --input-root /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717 --out-json .Agent/runs/20260712-gate-updown-layout-loo-bound/loo-bound.json --out-md .Agent/runs/20260712-gate-updown-layout-loo-bound/loo-bound.md --roles up,gate,down --max-jobs 8 --max-gap-mib 1.0 --baseline-decode-ms 42527.100000000006
```
