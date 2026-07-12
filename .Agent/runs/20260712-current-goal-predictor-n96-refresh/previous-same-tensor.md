# Kimi expert route predictability

- Input traces: `2`
- Aggregate calls: `32760`

Predictor: previous call for the same tensor predicts the current active expert set.

## Aggregate

| kind | calls | predictable calls | recall | precision | byte recall | predicted/actual bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 32760 | 0.9890 | 0.3272 | 0.3306 | 0.3260 | 0.9898 |
| down | 10920 | 0.9890 | 0.3272 | 0.3306 | 0.3252 | 0.9898 |
| gate | 10920 | 0.9890 | 0.3272 | 0.3306 | 0.3283 | 0.9898 |
| up | 10920 | 0.9890 | 0.3272 | 0.3306 | 0.3245 | 0.9898 |

## Per Prompt

| prompt | calls | recall | precision | byte recall |
| --- | ---: | ---: | ---: | ---: |
| dev_france_regression | 15480 | 0.3248 | 0.3283 | 0.3236 |
| dev_intelligence_general | 17280 | 0.3295 | 0.3327 | 0.3282 |
