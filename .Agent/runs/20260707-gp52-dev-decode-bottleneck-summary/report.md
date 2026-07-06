# GP52 Dev Decode Bottleneck Summary

Input: `.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile`

Runs: `7`
Total decode runs: `488`
Weighted decode ms/token: `3766.876`
Weighted token rate: `0.265 tok/s`
Direct read ratio: `0.572`
Iouring read ratio: `0.428`
Iouring throughput over decode wall: `0.343 GiB/s`

## Per Prompt

| Prompt | Quality | tok/s | decode ms/token | TTFT ms | up/gate ms/token | down ms/token | fallback ms/token | residual ms/token | direct read ratio | iouring GiB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | pass | 1.260 | 792.397 | 78633.750 | 218.163 | 145.949 | 59.651 | 368.634 | 0.038 | 4.814 |
| `dev_japan_factual` | pass | 0.420 | 2383.232 | 74493.280 | 639.671 | 424.433 | 150.238 | 1168.890 | 0.498 | 0.675 |
| `dev_linear_equation` | pass | 0.160 | 6428.414 | 85340.380 | 1691.729 | 1234.646 | 225.125 | 3276.914 | 0.874 | 0.059 |
| `dev_mixed_summary` | pass | 0.200 | 4942.434 | 91143.370 | 1342.055 | 963.825 | 215.759 | 2420.794 | 0.797 | 0.120 |
| `dev_photosynthesis_factual` | pass | 0.240 | 4189.084 | 66719.530 | 1090.455 | 787.591 | 168.161 | 2142.878 | 0.754 | 0.159 |
| `dev_python_reverse` | pass | 0.170 | 5963.713 | 84157.320 | 1531.595 | 1151.263 | 207.188 | 3073.666 | 0.867 | 0.059 |
| `dev_zh_france` | pass | 0.380 | 2629.834 | 72540.090 | 738.082 | 496.810 | 137.805 | 1257.137 | 0.575 | 0.452 |

## Components

| Component | total ms | ms/token | decode share | ideal tok/s if removed |
| --- | ---: | ---: | ---: | ---: |
| `residual_unattributed` | 924907.019 | 1895.301 | 0.503 | 0.534 |
| `upgate_wall` | 485330.658 | 994.530 | 0.264 | 0.361 |
| `down_wall` | 349086.609 | 715.341 | 0.190 | 0.328 |
| `down_stage` | 343697.931 | 704.299 | 0.187 | 0.327 |
| `iouring_wait_reported` | 120463.828 | 246.852 | 0.066 | 0.284 |
| `decode_fallback` | 78911.024 | 161.703 | 0.043 | 0.277 |

## Top Detail Rows

| Group | Item | total ms | ms/token |
| --- | --- | ---: | ---: |
| `upgate_type` | `up=18/gate=18` | 265666.008 | 544.398 |
| `upgate_type` | `up=22/gate=22` | 219664.650 | 450.132 |
| `down_type` | `type=23` | 194381.155 | 398.322 |
| `down_type` | `type=11` | 154705.454 | 317.019 |
| `fallback_type` | `type=2` | 78911.024 | 161.703 |
| `fallback_role` | `down` | 78911.024 | 161.703 |
| `upgate_layer` | `blk.60` | 33028.009 | 67.680 |
| `upgate_layer` | `blk.58` | 30465.863 | 62.430 |
| `upgate_layer` | `blk.57` | 29429.032 | 60.305 |
| `upgate_layer` | `blk.56` | 28449.493 | 58.298 |
| `upgate_layer` | `blk.59` | 27379.248 | 56.105 |
| `upgate_layer` | `blk.2` | 26149.055 | 53.584 |
| `upgate_layer` | `blk.6` | 25024.177 | 51.279 |
| `upgate_layer` | `blk.4` | 24223.669 | 49.639 |
| `upgate_layer` | `blk.5` | 22007.280 | 45.097 |
| `down_layer` | `blk.58` | 21424.476 | 43.903 |
| `down_layer` | `blk.57` | 20792.171 | 42.607 |
| `down_layer` | `blk.60` | 19827.648 | 40.630 |
| `upgate_layer` | `blk.3` | 19510.181 | 39.980 |
| `down_layer` | `blk.4` | 18642.304 | 38.201 |

## Decision

The largest prompt-agnostic component is `residual_unattributed`, not the seven unsupported Q4_0 down tensors.
The slowest dev prompts also have the highest direct-read ratios and the lowest iouring throughput, while the France prompt is fast and mostly iouring-backed.
This means the next implementation target should first explain and reduce broad direct-read/residual time on non-France dev prompts.
A standalone Q4_0 down fallback implementation is lower priority because `decode_fallback` is only a small fraction of the weighted decode wall.

Recommended next step: run current GP50 code on slow dev prompts with `COPY_PROFILE=1`, rank direct-read copy paths by tensor/type, then implement the highest-byte path that can move from direct reads to batched iouring without changing output.
This report does not change runtime behavior and makes no SOTA claim.
