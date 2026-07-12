# Kimi Route Score Predictor Offline Analysis

Trace root: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-n96-filtered-hidden-corpus`

Prompts: `dev_france_regression, dev_japan_factual, dev_linear_equation, dev_mixed_summary, dev_photosynthesis_factual, dev_python_reverse, dev_zh_france`

Decode route-score rows: `29040`

## Policy Results

| policy | rows | eligible | recall | precision | pred/actual | full-step |
|---|---:|---:|---:|---:|---:|---:|
| prev_token_top2 | 29040 | 28620 | 0.1300 | 0.5274 | 0.246 | 0.0000 |
| prev_token_top4 | 29040 | 28620 | 0.2241 | 0.4548 | 0.493 | 0.0000 |
| prev_token_top8 | 29040 | 28620 | 0.3366 | 0.3416 | 0.986 | 0.0006 |
| prev_layer_top2 | 29040 | 28556 | 0.0050 | 0.0203 | 0.246 | 0.0000 |
| prev_layer_top4 | 29040 | 28556 | 0.0098 | 0.0200 | 0.492 | 0.0000 |
| prev_layer_top8 | 29040 | 28556 | 0.0198 | 0.0202 | 0.983 | 0.0000 |
| hybrid_layer4_token4 | 29040 | 29033 | 0.2318 | 0.2366 | 0.980 | 0.0000 |
| hybrid_layer8_token8 | 29040 | 29033 | 0.3496 | 0.1794 | 1.949 | 0.0008 |

## Decision

Best full-step policy is `hybrid_layer8_token8` with recall `0.3496`, pred/actual `1.949`, and full-step cover `0.0008`.

This does not pass the runtime prefetch gate. The next runtime A/B should not be built from these simple score/history policies unless a stronger hidden-state or draft-router signal is added.

## Margin Bins

| bin | rows | margin min | margin max | avg margin | avg entropy | next-token same-layer recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7155 | 0.000006 | 0.054173 | 0.025110 | 2.017897 | 0.3258 |
| 2 | 7155 | 0.054186 | 0.134279 | 0.091814 | 1.999672 | 0.3415 |
| 3 | 7155 | 0.134285 | 0.263791 | 0.192497 | 1.978913 | 0.3474 |
| 4 | 7155 | 0.263837 | 0.852541 | 0.398090 | 1.926878 | 0.3516 |
