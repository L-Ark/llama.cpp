# Kimi IO batch-profile breakdown

This report summarizes `GGML_MOE_IO_BATCH_PROFILE_OUT` rows. It measures
how much independent read work the runtime actually exposes to io_uring.

## Aggregate

- Batches: `32618`
- Read jobs: `174725`
- Avg read jobs/batch: `5.357`
- Weighted inflight avg: `4.864`
- Max inflight: `8`
- Wait: `74271.192 ms`
- Wall: `143747.549 ms`
- Wait/read job: `0.425 ms`
- Read batches <=4 jobs: `0.472`
- Read batches <=8 jobs: `0.989`

## By Prompt

| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | 15449 | 84620 | 5.477 | 4.888 | 35598.774 | 69331.213 | 0.467 |
| `dev_intelligence_general` | 17169 | 90105 | 5.248 | 4.841 | 38672.418 | 74416.336 | 0.476 |

## By Op/Role

| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 10853 | 55258 | 5.091 | 4.626 | 26527.255 | 48849.178 | 0.532 |
| `runtime_load` | `up` | 10854 | 55259 | 5.091 | 4.410 | 24065.299 | 45317.504 | 0.532 |
| `runtime_load` | `down` | 5698 | 38920 | 6.830 | 5.653 | 14654.988 | 31441.748 | 0.271 |
| `current_down_overlap` | `down` | 5213 | 25288 | 4.851 | 5.161 | 9023.650 | 18139.118 | 0.442 |

## Top 20 Layer/Role Buckets By Wait

| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 1 | 182 | 1127 | 6.192 | 5.106 | 649.313 | 1484.336 |
| `runtime_load` | `up` | 1 | 182 | 1127 | 6.192 | 4.808 | 557.440 | 1176.822 |
| `runtime_load` | `gate` | 53 | 181 | 930 | 5.138 | 4.441 | 534.824 | 927.990 |
| `runtime_load` | `gate` | 54 | 182 | 905 | 4.973 | 4.460 | 523.609 | 989.882 |
| `runtime_load` | `gate` | 52 | 180 | 937 | 5.206 | 4.658 | 510.226 | 907.953 |
| `runtime_load` | `gate` | 30 | 182 | 978 | 5.374 | 4.782 | 508.018 | 915.473 |
| `runtime_load` | `gate` | 51 | 178 | 902 | 5.067 | 4.492 | 507.407 | 883.395 |
| `runtime_load` | `gate` | 28 | 181 | 1016 | 5.613 | 4.904 | 502.780 | 931.083 |
| `runtime_load` | `down` | 6 | 182 | 1200 | 6.593 | 5.979 | 502.153 | 1329.767 |
| `runtime_load` | `gate` | 48 | 182 | 916 | 5.033 | 4.589 | 500.914 | 890.014 |
| `runtime_load` | `gate` | 50 | 181 | 867 | 4.790 | 4.339 | 499.503 | 869.222 |
| `runtime_load` | `gate` | 29 | 182 | 979 | 5.379 | 4.831 | 499.212 | 912.479 |
| `runtime_load` | `gate` | 55 | 182 | 901 | 4.951 | 4.447 | 498.182 | 870.309 |
| `runtime_load` | `down` | 4 | 182 | 1334 | 7.330 | 6.357 | 494.267 | 1334.376 |
| `runtime_load` | `down` | 7 | 182 | 1204 | 6.615 | 5.942 | 493.323 | 1048.782 |
| `runtime_load` | `gate` | 49 | 178 | 870 | 4.888 | 4.464 | 492.094 | 860.185 |
| `runtime_load` | `up` | 9 | 182 | 1022 | 5.615 | 4.801 | 489.688 | 947.065 |
| `runtime_load` | `gate` | 43 | 181 | 867 | 4.790 | 4.430 | 489.662 | 859.472 |
| `runtime_load` | `down` | 9 | 182 | 1158 | 6.363 | 5.676 | 485.265 | 1022.877 |
| `runtime_load` | `gate` | 32 | 182 | 962 | 5.286 | 4.730 | 485.040 | 875.496 |

## Interpretation Rules

- If avg read jobs/batch and weighted inflight are far below the pure IO
  bench depth, the runtime is not exposing enough continuous independent
  read work.
- If wait is concentrated in `runtime_load gate/up`, earlier admission or
  lower-byte up/gate representation is more promising than another down
  overlap change.
- If wait is concentrated in `current_down_overlap`, down prefetch depth or
  down placement should be investigated, but only if endpoint TTFT/RAM
  gates remain valid.
