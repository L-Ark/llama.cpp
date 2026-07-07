# Kimi next-gate bounded prefetch policy bound

This is an offline dev-only screen. It does not implement runtime prefetch and does not claim SOTA.

- source root: `.Agent/runs/20260707-gp65-correct-gp4env-n96-shadow-top8`
- prompts: `7`
- caps: `[1, 2, 4, 8]`
- min precisions: `[0.7, 0.75, 0.8, 0.85, 0.9]`
- max false/actual gate for layer selection: `0.15`

## Policy Summary

| policy | pairs | selected layers | pred/actual | recall | precision | false/actual |
|---|---:|---:|---:|---:|---:|---:|
| `cap1_all` | `28792` | `all` | `0.1250` | `0.1243` | `0.9942` | `0.0007` |
| `cap1_loo_layer_prec0.70` | `28792` | `59.0` | `0.1250` | `0.1243` | `0.9942` | `0.0007` |
| `cap1_loo_layer_prec0.75` | `28792` | `59.0` | `0.1250` | `0.1243` | `0.9942` | `0.0007` |
| `cap1_loo_layer_prec0.80` | `28792` | `59.0` | `0.1250` | `0.1243` | `0.9942` | `0.0007` |
| `cap1_loo_layer_prec0.85` | `28792` | `59.0` | `0.1250` | `0.1243` | `0.9942` | `0.0007` |
| `cap1_loo_layer_prec0.90` | `28304` | `58.0` | `0.1250` | `0.1245` | `0.9960` | `0.0005` |
| `cap2_all` | `28792` | `all` | `0.2500` | `0.2457` | `0.9829` | `0.0043` |
| `cap2_loo_layer_prec0.70` | `28792` | `59.0` | `0.2500` | `0.2457` | `0.9829` | `0.0043` |
| `cap2_loo_layer_prec0.75` | `28581` | `58.6` | `0.2500` | `0.2460` | `0.9842` | `0.0040` |
| `cap2_loo_layer_prec0.80` | `28304` | `58.0` | `0.2500` | `0.2467` | `0.9869` | `0.0033` |
| `cap2_loo_layer_prec0.85` | `28304` | `58.0` | `0.2500` | `0.2467` | `0.9869` | `0.0033` |
| `cap2_loo_layer_prec0.90` | `28304` | `58.0` | `0.2500` | `0.2467` | `0.9869` | `0.0033` |
| `cap4_all` | `28792` | `all` | `0.5000` | `0.4714` | `0.9429` | `0.0286` |
| `cap4_loo_layer_prec0.70` | `28304` | `58.0` | `0.5000` | `0.4745` | `0.9491` | `0.0255` |
| `cap4_loo_layer_prec0.75` | `28304` | `58.0` | `0.5000` | `0.4745` | `0.9491` | `0.0255` |
| `cap4_loo_layer_prec0.80` | `28030` | `57.6` | `0.5000` | `0.4751` | `0.9501` | `0.0249` |
| `cap4_loo_layer_prec0.85` | `25959` | `53.1` | `0.5000` | `0.4803` | `0.9606` | `0.0197` |
| `cap4_loo_layer_prec0.90` | `25376` | `52.0` | `0.5000` | `0.4815` | `0.9630` | `0.0185` |
| `cap8_all` | `28792` | `all` | `1.0000` | `0.7725` | `0.7725` | `0.2275` |
| `cap8_loo_layer_prec0.70` | `95` | `0.1` | `1.0000` | `0.8092` | `0.8092` | `0.1908` |
| `cap8_loo_layer_prec0.75` | `95` | `0.1` | `1.0000` | `0.8092` | `0.8092` | `0.1908` |
| `cap8_loo_layer_prec0.80` | `95` | `0.1` | `1.0000` | `0.8092` | `0.8092` | `0.1908` |
| `cap8_loo_layer_prec0.85` | `95` | `0.1` | `1.0000` | `0.8092` | `0.8092` | `0.1908` |
| `cap8_loo_layer_prec0.90` | `0` | `0.0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |

## Decision Notes

- `cap1_all` is the strict low-extra-byte bound; it cannot exceed about 12.5% recall because each actual set has 8 experts.
- A runtime candidate needs low false bytes and enough recall to hide exposed IO. Low false bytes alone is not enough.
- These metrics are optimistic because they do not subtract experts already resident in VRAM.

## Reproduce

```bash
python3 .Agent/run-tools/kimi_next_gate_prefetch_policy_bound.py --root .Agent/runs/20260707-gp65-correct-gp4env-n96-shadow-top8 --caps 1,2,4,8 --min-precisions 0.70,0.75,0.80,0.85,0.90 --max-false-over-actual 0.15 --min-layer-pairs 64 --out-json .Agent/runs/20260707-gp71-next-gate-prefetch-policy-bound/report.json --out-md .Agent/runs/20260707-gp71-next-gate-prefetch-policy-bound/report.md
```
