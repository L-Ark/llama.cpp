# Kimi Route Score Predictor Offline Analysis

Trace root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-route-score-n96-080423`

Prompts: `dev_france_regression, dev_intelligence_general`

Decode route-score rows: `10800`

## Policy Results

| policy | rows | eligible | recall | precision | pred/actual | full-step |
|---|---:|---:|---:|---:|---:|---:|
| prev_token_top2 | 10800 | 10680 | 0.1316 | 0.5323 | 0.247 | 0.0000 |
| prev_token_top4 | 10800 | 10680 | 0.2291 | 0.4633 | 0.494 | 0.0000 |
| prev_token_top8 | 10800 | 10680 | 0.3505 | 0.3544 | 0.989 | 0.0008 |
| prev_layer_top2 | 10800 | 10620 | 0.0047 | 0.0193 | 0.246 | 0.0000 |
| prev_layer_top4 | 10800 | 10620 | 0.0097 | 0.0198 | 0.492 | 0.0000 |
| prev_layer_top8 | 10800 | 10620 | 0.0198 | 0.0202 | 0.983 | 0.0000 |
| hybrid_layer4_token4 | 10800 | 10798 | 0.2369 | 0.2414 | 0.981 | 0.0000 |
| hybrid_layer8_token8 | 10800 | 10798 | 0.3637 | 0.1862 | 1.953 | 0.0011 |

## Decision

Best full-step policy is `hybrid_layer8_token8` with recall `0.3637`, pred/actual `1.953`, and full-step cover `0.0011`.

This does not pass the runtime prefetch gate. The next runtime A/B should not be built from these simple score/history policies unless a stronger hidden-state or draft-router signal is added.

## Margin Bins

| bin | rows | margin min | margin max | avg margin | avg entropy | next-token same-layer recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2670 | 0.000006 | 0.056292 | 0.026511 | 2.015458 | 0.3375 |
| 2 | 2670 | 0.056306 | 0.136351 | 0.093941 | 1.996387 | 0.3453 |
| 3 | 2670 | 0.136421 | 0.265625 | 0.194943 | 1.972152 | 0.3586 |
| 4 | 2670 | 0.265663 | 0.836320 | 0.404537 | 1.912267 | 0.3763 |
