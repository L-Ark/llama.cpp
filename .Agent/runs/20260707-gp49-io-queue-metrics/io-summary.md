# Kimi IO queue metrics summary

- Prompts: `7`
- Weighted token rate: `0.265 tok/s`
- Aggregate iouring throughput: `0.343 GiB/s`
- Pure IO peak reference: `10.300 GiB/s`
- Peak utilization: `0.033`
- Direct read ratio: `0.572`
- Weighted iouring inflight avg: `3.335`
- Iouring wait/decode fraction: `0.066`

| prompt | tok/s | iouring GiB/s | iouring read ratio | direct reads | inflight avg | upgate hit | down hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev_france_regression | 1.260 | 4.814 | 0.962 | 2229 | 3.390 | 0.441 | 0.730 |
| dev_japan_factual | 0.420 | 0.675 | 0.502 | 26007 | 3.290 | 0.447 | 0.733 |
| dev_linear_equation | 0.160 | 0.059 | 0.126 | 16872 | 3.800 | 0.260 | 0.658 |
| dev_mixed_summary | 0.200 | 0.120 | 0.203 | 23826 | 3.500 | 0.363 | 0.695 |
| dev_photosynthesis_factual | 0.240 | 0.159 | 0.246 | 36641 | 3.290 | 0.409 | 0.719 |
| dev_python_reverse | 0.170 | 0.059 | 0.133 | 40761 | 3.210 | 0.358 | 0.699 |
| dev_zh_france | 0.380 | 0.452 | 0.425 | 15188 | 3.090 | 0.471 | 0.741 |
