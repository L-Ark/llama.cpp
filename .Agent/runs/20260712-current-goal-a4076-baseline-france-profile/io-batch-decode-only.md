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
- Weighted inflight avg: `3.470`
- Max inflight: `8`
- Wait: `43730.248 ms`
- Wall: `45773.007 ms`
- Wait/read job: `0.612 ms`
- Read batches <=4 jobs: `0.473`
- Read batches <=8 jobs: `1.000`

## By Prompt

| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260712-current-goal-a4076-baseline-france-n96-iobatch` | 15272 | 71447 | 4.678 | 3.470 | 43730.248 | 45773.007 | 0.473 |

## By Op/Role

| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 5087 | 22517 | 4.426 | 3.477 | 14706.035 | 15503.109 | 0.530 |
| `runtime_load` | `up` | 5087 | 22514 | 4.426 | 3.252 | 14442.326 | 15051.388 | 0.530 |
| `runtime_load` | `down` | 2635 | 14662 | 5.564 | 3.805 | 8392.745 | 8732.688 | 0.263 |
| `current_down_overlap` | `down` | 2463 | 11754 | 4.772 | 3.459 | 6189.143 | 6485.821 | 0.460 |

## Top 30 Layer/Role Buckets By Wait

| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 1 | 85 | 477 | 5.612 | 3.981 | 341.774 | 418.751 |
| `runtime_load` | `up` | 1 | 85 | 477 | 5.612 | 3.709 | 338.677 | 397.244 |
| `runtime_load` | `down` | 4 | 85 | 556 | 6.541 | 4.163 | 324.641 | 337.284 |
| `runtime_load` | `down` | 7 | 85 | 512 | 6.024 | 4.022 | 324.270 | 335.955 |
| `runtime_load` | `down` | 10 | 85 | 526 | 6.188 | 4.046 | 324.067 | 334.838 |
| `runtime_load` | `down` | 6 | 85 | 505 | 5.941 | 4.078 | 319.646 | 330.925 |
| `runtime_load` | `gate` | 29 | 85 | 444 | 5.224 | 3.967 | 314.490 | 325.188 |
| `runtime_load` | `down` | 9 | 85 | 490 | 5.765 | 3.948 | 312.795 | 323.253 |
| `runtime_load` | `down` | 8 | 85 | 478 | 5.624 | 3.808 | 309.131 | 319.956 |
| `runtime_load` | `up` | 7 | 85 | 423 | 4.976 | 3.614 | 302.401 | 314.207 |
| `runtime_load` | `up` | 9 | 85 | 419 | 4.929 | 3.475 | 301.005 | 311.586 |
| `runtime_load` | `up` | 29 | 85 | 444 | 5.224 | 3.413 | 300.638 | 311.223 |
| `runtime_load` | `down` | 25 | 85 | 492 | 5.788 | 3.757 | 299.847 | 310.204 |
| `runtime_load` | `gate` | 9 | 85 | 420 | 4.941 | 3.440 | 293.170 | 304.516 |
| `runtime_load` | `gate` | 7 | 85 | 423 | 4.976 | 3.557 | 293.141 | 305.366 |
| `runtime_load` | `gate` | 28 | 85 | 422 | 4.965 | 3.850 | 292.087 | 302.450 |
| `runtime_load` | `down` | 24 | 85 | 495 | 5.824 | 3.816 | 290.988 | 301.641 |
| `runtime_load` | `down` | 23 | 85 | 484 | 5.694 | 3.716 | 290.829 | 301.071 |
| `runtime_load` | `gate` | 33 | 85 | 403 | 4.741 | 3.799 | 288.522 | 299.068 |
| `runtime_load` | `up` | 8 | 85 | 406 | 4.776 | 3.496 | 287.787 | 297.840 |
| `runtime_load` | `gate` | 30 | 85 | 403 | 4.741 | 3.719 | 285.584 | 295.551 |
| `runtime_load` | `down` | 20 | 85 | 476 | 5.600 | 3.715 | 285.352 | 295.402 |
| `runtime_load` | `down` | 18 | 85 | 439 | 5.165 | 3.528 | 284.957 | 294.781 |
| `runtime_load` | `gate` | 10 | 85 | 465 | 5.471 | 3.823 | 284.677 | 296.271 |
| `runtime_load` | `up` | 10 | 85 | 465 | 5.471 | 3.627 | 284.497 | 295.341 |
| `runtime_load` | `gate` | 31 | 84 | 394 | 4.690 | 3.702 | 279.818 | 289.938 |
| `runtime_load` | `up` | 33 | 85 | 403 | 4.741 | 3.409 | 279.412 | 289.344 |
| `runtime_load` | `down` | 19 | 85 | 454 | 5.341 | 3.743 | 278.972 | 288.660 |
| `runtime_load` | `gate` | 8 | 85 | 406 | 4.776 | 3.305 | 278.792 | 289.410 |
| `runtime_load` | `down` | 15 | 85 | 427 | 5.024 | 3.541 | 278.234 | 287.894 |

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
