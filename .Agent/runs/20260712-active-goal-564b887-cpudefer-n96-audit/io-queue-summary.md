# Kimi IO queue metrics summary

- Prompts: `3`
- Weighted token rate: `1.614 tok/s`
- Aggregate iouring throughput: `8.455 GiB/s`
- Pure IO peak reference: `10.300 GiB/s`
- Peak utilization: `0.821`
- Direct read ratio: `0.000`
- Weighted iouring inflight avg: `4.053`
- Iouring wait/decode fraction: `0.716`

| prompt | tok/s | iouring GiB/s | iouring read ratio | direct reads | inflight avg | upgate hit | down hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev_france_regression | 1.560 | 8.288 | 1.000 | 0 | 4.260 | 0.449 | 0.612 |
| dev_france_regression_iobatch | 1.720 | 9.142 | 1.000 | 0 | 3.720 | 0.449 | 0.612 |
| dev_intelligence_general | 1.580 | 8.040 | 1.000 | 0 | 4.170 | 0.452 | 0.618 |
