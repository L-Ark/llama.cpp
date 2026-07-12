# Kimi IO batch-profile breakdown

This report summarizes `GGML_MOE_IO_BATCH_PROFILE_OUT` rows. It measures
how much independent read work the runtime actually exposes to io_uring.

## Aggregate

- Batches: `15449`
- Read jobs: `84620`
- Avg read jobs/batch: `5.477`
- Weighted inflight avg: `4.087`
- Max inflight: `8`
- Wait: `46154.564 ms`
- Wall: `52282.207 ms`
- Wait/read job: `0.545 ms`
- Read batches <=4 jobs: `0.467`
- Read batches <=8 jobs: `0.989`

## By Prompt

| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260712-current-goal-a4076-baseline-france-n96-iobatch` | 15449 | 84620 | 5.477 | 4.087 | 46154.564 | 52282.207 | 0.467 |

## By Op/Role

| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 5146 | 26908 | 5.229 | 4.118 | 15495.967 | 17501.357 | 0.524 |
| `runtime_load` | `up` | 5146 | 26905 | 5.228 | 3.937 | 15124.512 | 16812.847 | 0.524 |
| `runtime_load` | `down` | 2694 | 19053 | 7.072 | 4.640 | 9344.942 | 11482.182 | 0.257 |
| `current_down_overlap` | `down` | 2463 | 11754 | 4.772 | 3.459 | 6189.143 | 6485.821 | 0.460 |

## Top 30 Layer/Role Buckets By Wait

| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 1 | 86 | 547 | 6.360 | 4.362 | 367.863 | 537.167 |
| `runtime_load` | `down` | 6 | 86 | 572 | 6.651 | 4.436 | 357.418 | 520.930 |
| `runtime_load` | `up` | 1 | 86 | 547 | 6.360 | 4.215 | 345.886 | 422.863 |
| `runtime_load` | `down` | 4 | 86 | 635 | 7.384 | 4.557 | 344.211 | 499.183 |
| `runtime_load` | `down` | 10 | 86 | 584 | 6.791 | 4.335 | 342.216 | 378.938 |
| `runtime_load` | `down` | 7 | 86 | 585 | 6.802 | 4.482 | 338.615 | 378.032 |
| `runtime_load` | `down` | 9 | 86 | 562 | 6.535 | 4.388 | 331.809 | 363.130 |
| `runtime_load` | `gate` | 29 | 86 | 515 | 5.988 | 4.425 | 327.572 | 359.260 |
| `runtime_load` | `down` | 8 | 86 | 540 | 6.279 | 4.196 | 326.426 | 361.074 |
| `runtime_load` | `down` | 25 | 86 | 560 | 6.512 | 4.244 | 319.742 | 348.993 |
| `runtime_load` | `up` | 7 | 86 | 496 | 5.767 | 4.218 | 314.037 | 348.619 |
| `runtime_load` | `up` | 9 | 86 | 491 | 5.709 | 4.031 | 313.326 | 338.584 |
| `runtime_load` | `up` | 29 | 86 | 515 | 5.988 | 3.969 | 311.841 | 339.843 |
| `runtime_load` | `down` | 24 | 86 | 566 | 6.581 | 4.290 | 308.097 | 348.681 |
| `runtime_load` | `down` | 23 | 86 | 554 | 6.442 | 4.205 | 307.411 | 345.734 |
| `runtime_load` | `gate` | 28 | 86 | 502 | 5.837 | 4.418 | 306.206 | 342.369 |
| `runtime_load` | `gate` | 7 | 86 | 496 | 5.767 | 4.148 | 306.013 | 337.108 |
| `runtime_load` | `down` | 20 | 86 | 544 | 6.326 | 4.117 | 304.457 | 343.293 |
| `runtime_load` | `gate` | 9 | 86 | 492 | 5.721 | 3.987 | 304.131 | 329.754 |
| `runtime_load` | `down` | 1 | 86 | 632 | 7.349 | 4.560 | 302.029 | 436.832 |
| `runtime_load` | `gate` | 33 | 86 | 476 | 5.535 | 4.363 | 301.233 | 333.410 |
| `runtime_load` | `gate` | 30 | 86 | 484 | 5.628 | 4.375 | 300.996 | 327.091 |
| `runtime_load` | `down` | 18 | 86 | 501 | 5.826 | 3.978 | 300.285 | 327.567 |
| `runtime_load` | `up` | 8 | 86 | 468 | 5.442 | 4.009 | 299.654 | 321.742 |
| `runtime_load` | `gate` | 10 | 86 | 523 | 6.081 | 4.183 | 295.893 | 323.766 |
| `runtime_load` | `down` | 19 | 86 | 520 | 6.047 | 4.183 | 294.699 | 332.179 |
| `runtime_load` | `gate` | 31 | 85 | 471 | 5.541 | 4.307 | 294.687 | 320.054 |
| `runtime_load` | `down` | 26 | 86 | 524 | 6.093 | 4.178 | 294.198 | 326.312 |
| `runtime_load` | `up` | 10 | 86 | 523 | 6.081 | 4.061 | 293.465 | 318.687 |
| `runtime_load` | `down` | 22 | 86 | 510 | 5.930 | 4.156 | 293.426 | 321.677 |

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
