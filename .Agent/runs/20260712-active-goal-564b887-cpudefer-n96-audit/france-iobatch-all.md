# Kimi IO batch-profile breakdown

This report summarizes `GGML_MOE_IO_BATCH_PROFILE_OUT` rows. It measures
how much independent read work the runtime actually exposes to io_uring.

## Filters

- min jobs: `None`
- max jobs: `None`

## Aggregate

- Batches: `15449`
- Read jobs: `84620`
- Avg read jobs/batch: `5.477`
- Weighted inflight avg: `4.032`
- Max inflight: `8`
- Wait: `45427.522 ms`
- Wall: `52108.482 ms`
- Wait/read job: `0.537 ms`
- Read batches <=4 jobs: `0.467`
- Read batches <=8 jobs: `0.989`

## By Prompt

| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression_iobatch` | 15449 | 84620 | 5.477 | 4.032 | 45427.522 | 52108.482 | 0.467 |

## By Op/Role

| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 5146 | 26908 | 5.229 | 4.135 | 15462.054 | 17592.558 | 0.524 |
| `runtime_load` | `up` | 5146 | 26905 | 5.228 | 3.890 | 15051.684 | 16917.653 | 0.524 |
| `runtime_load` | `down` | 2694 | 19053 | 7.072 | 4.528 | 8902.140 | 11317.144 | 0.257 |
| `current_down_overlap` | `down` | 2463 | 11754 | 4.772 | 3.313 | 6011.644 | 6281.126 | 0.460 |

## Top 24 Layer/Role Buckets By Wait

| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 1 | 86 | 547 | 6.360 | 4.542 | 361.564 | 544.979 |
| `runtime_load` | `down` | 6 | 86 | 572 | 6.651 | 4.378 | 351.865 | 515.651 |
| `runtime_load` | `down` | 4 | 86 | 635 | 7.384 | 4.428 | 347.866 | 511.684 |
| `runtime_load` | `up` | 1 | 86 | 547 | 6.360 | 4.138 | 343.679 | 431.507 |
| `runtime_load` | `down` | 7 | 86 | 585 | 6.802 | 4.346 | 336.616 | 379.949 |
| `runtime_load` | `down` | 10 | 86 | 584 | 6.791 | 4.272 | 330.026 | 357.658 |
| `runtime_load` | `gate` | 55 | 86 | 470 | 5.465 | 4.410 | 320.809 | 359.250 |
| `runtime_load` | `down` | 9 | 86 | 562 | 6.535 | 4.277 | 318.220 | 361.242 |
| `runtime_load` | `gate` | 29 | 86 | 515 | 5.988 | 4.511 | 315.285 | 346.046 |
| `runtime_load` | `down` | 8 | 86 | 540 | 6.279 | 4.124 | 312.992 | 350.748 |
| `runtime_load` | `up` | 7 | 86 | 496 | 5.767 | 4.132 | 310.084 | 340.772 |
| `runtime_load` | `gate` | 33 | 86 | 476 | 5.535 | 4.407 | 307.303 | 341.389 |
| `runtime_load` | `gate` | 28 | 86 | 502 | 5.837 | 4.418 | 307.032 | 341.437 |
| `runtime_load` | `up` | 25 | 86 | 497 | 5.779 | 4.061 | 305.665 | 332.926 |
| `runtime_load` | `gate` | 7 | 86 | 496 | 5.767 | 4.116 | 305.321 | 328.406 |
| `runtime_load` | `up` | 8 | 86 | 468 | 5.442 | 3.871 | 300.957 | 329.205 |
| `runtime_load` | `up` | 29 | 86 | 515 | 5.988 | 3.999 | 299.676 | 329.197 |
| `runtime_load` | `down` | 25 | 86 | 560 | 6.512 | 4.075 | 299.497 | 330.911 |
| `runtime_load` | `up` | 9 | 86 | 491 | 5.709 | 4.170 | 298.456 | 331.289 |
| `runtime_load` | `gate` | 30 | 86 | 484 | 5.628 | 4.365 | 297.863 | 349.904 |
| `runtime_load` | `down` | 23 | 86 | 554 | 6.442 | 4.083 | 297.231 | 336.259 |
| `runtime_load` | `down` | 18 | 86 | 501 | 5.826 | 3.888 | 297.056 | 332.831 |
| `runtime_load` | `gate` | 31 | 85 | 471 | 5.541 | 4.427 | 295.988 | 329.877 |
| `runtime_load` | `gate` | 9 | 86 | 492 | 5.721 | 4.035 | 293.921 | 321.979 |

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
