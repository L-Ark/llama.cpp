# Kimi IO-trace pack layout screen

This is a dev-only offline screen. It does not rewrite packs or claim SOTA.

- traces: `1`
- rows: `25973`
- batches: `5555`
- max gap: `1.00 MiB`

## Batch Signature Reuse

- unique exact signatures: `5549`
- repeated exact batches: `11`
- unique layer signatures: `5549`
- repeated layer batches: `11`

## Current Offset Coalescing

- read jobs: `25973`
- coalesced extents with current offsets: `25577`
- read bytes: `139.756 GiB`
- span bytes with max-gap coalescing: `139.756 GiB`
- gap bytes: `0.000 GiB`

## Ideal Layouts

| layout | extents | read reduction | read GiB | span GiB | gap GiB |
|---|---:|---:|---:|---:|---:|
| `tensor` | `9564` | `16013` | `139.756` | `139.756` | `0.000` |
| `layer_role` | `9564` | `16013` | `139.756` | `139.756` | `0.000` |
| `layer` | `9564` | `16013` | `139.756` | `139.756` | `0.000` |
| `source` | `9564` | `16013` | `139.756` | `139.756` | `0.000` |

## Static Per-Tensor Clustered Layouts

| layout | extents | read reduction | read GiB | span GiB | gap GiB | missing |
|---|---:|---:|---:|---:|---:|---:|
| `expert_id` | `24673` | `904` | `139.756` | `139.756` | `0.000` | `0` |
| `frequency` | `23746` | `1831` | `139.756` | `139.756` | `0.000` | `0` |
| `first_use` | `14431` | `11146` | `139.756` | `139.756` | `0.000` | `0` |
| `greedy_pair` | `12530` | `13047` | `139.756` | `139.756` | `0.000` | `0` |

## Decode Bound

| layout | saved reads | saved ms upper | bounded decode ms | bounded tok/s |
|---|---:|---:|---:|---:|
| `ideal:tensor` | `16013.0` | `6528.61` | `14158.97` | `2.19` |
| `ideal:layer_role` | `16013.0` | `6528.61` | `14158.97` | `2.19` |
| `ideal:layer` | `16013.0` | `6528.61` | `14158.97` | `2.19` |
| `ideal:source` | `16013.0` | `6528.61` | `14158.97` | `2.19` |
| `static:expert_id` | `904.0` | `368.57` | `20319.01` | `1.53` |
| `static:frequency` | `1831.0` | `746.51` | `19941.07` | `1.55` |
| `static:first_use` | `11146.0` | `4544.30` | `16143.28` | `1.92` |
| `static:greedy_pair` | `13047.0` | `5319.35` | `15368.23` | `2.02` |

## Decision Rule

- If ideal same-layer/source bounds are small, pack layout cannot be the next runtime path.
- If the bound is large, build a default-off pack-layout A/B and validate N32 before N96.

