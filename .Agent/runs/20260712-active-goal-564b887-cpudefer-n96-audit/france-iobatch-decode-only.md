# Kimi IO batch-profile breakdown

This report summarizes `GGML_MOE_IO_BATCH_PROFILE_OUT` rows. It measures
how much independent read work the runtime actually exposes to io_uring.

## Filters

- min jobs: `None`
- max jobs: `8`

## Aggregate

- Batches: `15272`
- Read jobs: `71447`
- Avg read jobs/batch: `4.678`
- Weighted inflight avg: `3.406`
- Max inflight: `8`
- Wait: `43111.496 ms`
- Wall: `45064.212 ms`
- Wait/read job: `0.603 ms`
- Read batches <=4 jobs: `0.473`
- Read batches <=8 jobs: `1.000`

## By Prompt

| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression_iobatch` | 15272 | 71447 | 4.678 | 3.406 | 43111.496 | 45064.212 | 0.473 |

## By Op/Role

| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 5087 | 22517 | 4.426 | 3.493 | 14687.123 | 15481.526 | 0.530 |
| `runtime_load` | `up` | 5087 | 22514 | 4.426 | 3.206 | 14323.879 | 14900.892 | 0.530 |
| `runtime_load` | `down` | 2635 | 14662 | 5.564 | 3.654 | 8088.851 | 8400.667 | 0.263 |
| `current_down_overlap` | `down` | 2463 | 11754 | 4.772 | 3.313 | 6011.644 | 6281.126 | 0.460 |

## Top 24 Layer/Role Buckets By Wait

| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 1 | 85 | 477 | 5.612 | 4.125 | 344.913 | 435.663 |
| `runtime_load` | `up` | 1 | 85 | 477 | 5.612 | 3.637 | 333.858 | 402.481 |
| `runtime_load` | `down` | 7 | 85 | 512 | 6.024 | 3.901 | 319.740 | 330.185 |
| `runtime_load` | `down` | 10 | 85 | 526 | 6.188 | 3.958 | 316.088 | 327.630 |
| `runtime_load` | `down` | 4 | 85 | 556 | 6.541 | 4.006 | 313.633 | 324.740 |
| `runtime_load` | `down` | 6 | 85 | 505 | 5.941 | 3.961 | 312.010 | 322.808 |
| `runtime_load` | `down` | 9 | 85 | 490 | 5.765 | 3.771 | 303.877 | 313.885 |
| `runtime_load` | `gate` | 29 | 85 | 444 | 5.224 | 4.063 | 303.594 | 313.267 |
| `runtime_load` | `down` | 8 | 85 | 478 | 5.624 | 3.684 | 301.263 | 311.292 |
| `runtime_load` | `up` | 7 | 85 | 423 | 4.976 | 3.535 | 298.076 | 308.082 |
| `runtime_load` | `gate` | 33 | 85 | 403 | 4.741 | 3.846 | 297.549 | 306.781 |
| `runtime_load` | `gate` | 28 | 85 | 422 | 4.965 | 3.812 | 295.035 | 304.625 |
| `runtime_load` | `gate` | 7 | 85 | 423 | 4.976 | 3.545 | 294.731 | 305.581 |
| `runtime_load` | `up` | 9 | 85 | 419 | 4.929 | 3.553 | 289.750 | 299.138 |
| `runtime_load` | `up` | 29 | 85 | 444 | 5.224 | 3.436 | 289.579 | 299.095 |
| `runtime_load` | `gate` | 10 | 85 | 465 | 5.471 | 3.837 | 286.084 | 296.318 |
| `runtime_load` | `gate` | 31 | 84 | 394 | 4.690 | 3.797 | 285.835 | 294.887 |
| `runtime_load` | `up` | 33 | 85 | 403 | 4.741 | 3.342 | 285.670 | 294.554 |
| `runtime_load` | `up` | 10 | 85 | 465 | 5.471 | 3.577 | 285.562 | 295.547 |
| `runtime_load` | `up` | 8 | 85 | 406 | 4.776 | 3.419 | 285.498 | 294.902 |
| `runtime_load` | `gate` | 9 | 85 | 420 | 4.941 | 3.438 | 284.519 | 294.509 |
| `runtime_load` | `down` | 25 | 85 | 492 | 5.788 | 3.618 | 284.321 | 293.856 |
| `runtime_load` | `down` | 23 | 85 | 484 | 5.694 | 3.593 | 284.052 | 293.822 |
| `runtime_load` | `down` | 24 | 85 | 495 | 5.824 | 3.630 | 283.882 | 293.569 |

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
