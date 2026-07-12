# Kimi Route Score Predictor Offline Analysis

Trace root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-dev-n32-035310`

Prompts: `dev_france_regression, dev_intelligence_general`

Decode route-score rows: `3720`

## Policy Results

| policy | rows | eligible | recall | precision | pred/actual | full-step |
|---|---:|---:|---:|---:|---:|---:|
| prev_token_top2 | 3720 | 3600 | 0.1254 | 0.5185 | 0.242 | 0.0000 |
| prev_token_top4 | 3720 | 3600 | 0.2203 | 0.4552 | 0.484 | 0.0000 |
| prev_token_top8 | 3720 | 3600 | 0.3398 | 0.3511 | 0.968 | 0.0000 |
| prev_layer_top2 | 3720 | 3658 | 0.0042 | 0.0169 | 0.246 | 0.0000 |
| prev_layer_top4 | 3720 | 3658 | 0.0093 | 0.0189 | 0.492 | 0.0000 |
| prev_layer_top8 | 3720 | 3658 | 0.0192 | 0.0195 | 0.983 | 0.0000 |
| hybrid_layer4_token4 | 3720 | 3718 | 0.2280 | 0.2348 | 0.971 | 0.0000 |
| hybrid_layer8_token8 | 3720 | 3718 | 0.3528 | 0.1826 | 1.932 | 0.0000 |

## Decision

Best full-step policy is `hybrid_layer8_token8` with recall `0.3528`, pred/actual `1.932`, and full-step cover `0.0000`.

This does not pass the runtime prefetch gate. The next runtime A/B should not be built from these simple score/history policies unless a stronger hidden-state or draft-router signal is added.

## Margin Bins

| bin | rows | margin min | margin max | avg margin | avg entropy | next-token same-layer recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 900 | 0.000113 | 0.057104 | 0.026198 | 2.012032 | 0.3381 |
| 2 | 900 | 0.057155 | 0.134092 | 0.093400 | 1.990254 | 0.3463 |
| 3 | 900 | 0.134137 | 0.259763 | 0.190492 | 1.972149 | 0.3442 |
| 4 | 900 | 0.260098 | 0.836320 | 0.398208 | 1.907510 | 0.3758 |
