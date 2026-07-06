# Kimi expert route predictability

- Input traces: `7`
- Aggregate calls: `84445`

Predictor: previous call for the same tensor predicts the current active expert set.

## Aggregate

| kind | calls | predictable calls | recall | precision | byte recall | predicted/actual bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 84445 | 0.9857 | 0.3378 | 0.3427 | 0.3373 | 0.9857 |
| down | 25871 | 0.9857 | 0.3404 | 0.3454 | 0.3388 | 0.9857 |
| gate | 29287 | 0.9857 | 0.3367 | 0.3416 | 0.3380 | 0.9857 |
| up | 29287 | 0.9857 | 0.3367 | 0.3416 | 0.3345 | 0.9857 |

## Per Prompt

| prompt | calls | recall | precision | byte recall |
| --- | ---: | ---: | ---: | ---: |
| dev_france_regression | 13324 | 0.3632 | 0.3680 | 0.3629 |
| dev_japan_factual | 14708 | 0.3808 | 0.3854 | 0.3803 |
| dev_linear_equation | 5885 | 0.2229 | 0.2296 | 0.2225 |
| dev_mixed_summary | 9345 | 0.3104 | 0.3162 | 0.3097 |
| dev_photosynthesis_factual | 16265 | 0.3362 | 0.3398 | 0.3357 |
| dev_python_reverse | 16438 | 0.3065 | 0.3097 | 0.3056 |
| dev_zh_france | 8480 | 0.3972 | 0.4055 | 0.3967 |
