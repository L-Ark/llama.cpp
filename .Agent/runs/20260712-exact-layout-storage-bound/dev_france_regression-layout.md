# Kimi IO-trace pack layout screen

This is a dev-only offline screen. It does not rewrite packs or claim SOTA.

- traces: `1`
- rows: `25623`
- batches: `5579`
- max gap: `1.00 MiB`

## Batch Signature Reuse

- unique exact signatures: `5575`
- repeated exact batches: `8`
- unique layer signatures: `5575`
- repeated layer batches: `8`

## Current Offset Coalescing

- read jobs: `25623`
- coalesced extents with current offsets: `25017`
- read bytes: `137.759 GiB`
- span bytes with max-gap coalescing: `137.759 GiB`
- gap bytes: `0.000 GiB`

## Ideal Layouts

| layout | extents | read reduction | read GiB | span GiB | gap GiB |
|---|---:|---:|---:|---:|---:|
| `tensor` | `5771` | `19246` | `137.759` | `137.759` | `0.000` |
| `layer_role` | `5771` | `19246` | `137.759` | `137.759` | `0.000` |
| `layer` | `5771` | `19246` | `137.759` | `137.759` | `0.000` |
| `source` | `5771` | `19246` | `137.759` | `137.759` | `0.000` |

## Static Per-Tensor Clustered Layouts

| layout | extents | read reduction | read GiB | span GiB | gap GiB | missing |
|---|---:|---:|---:|---:|---:|---:|
| `expert_id` | `24342` | `675` | `137.759` | `137.759` | `0.000` | `0` |
| `frequency` | `23575` | `1442` | `137.759` | `137.759` | `0.000` | `0` |
| `first_use` | `15111` | `9906` | `137.759` | `137.759` | `0.000` | `0` |
| `greedy_pair` | `13034` | `11983` | `137.759` | `137.759` | `0.000` | `0` |

## Decode Bound

| layout | saved reads | saved ms upper | bounded decode ms | bounded tok/s |
|---|---:|---:|---:|---:|
| `ideal:tensor` | `19246.0` | `8312.09` | `13527.43` | `2.29` |
| `ideal:layer_role` | `19246.0` | `8312.09` | `13527.43` | `2.29` |
| `ideal:layer` | `19246.0` | `8312.09` | `13527.43` | `2.29` |
| `ideal:source` | `19246.0` | `8312.09` | `13527.43` | `2.29` |
| `static:expert_id` | `675.0` | `291.52` | `21548.00` | `1.44` |
| `static:frequency` | `1442.0` | `622.78` | `21216.74` | `1.46` |
| `static:first_use` | `9906.0` | `4278.27` | `17561.25` | `1.77` |
| `static:greedy_pair` | `11983.0` | `5175.30` | `16664.22` | `1.86` |

## Decision Rule

- If ideal same-layer/source bounds are small, pack layout cannot be the next runtime path.
- If the bound is large, build a default-off pack-layout A/B and validate N32 before N96.

