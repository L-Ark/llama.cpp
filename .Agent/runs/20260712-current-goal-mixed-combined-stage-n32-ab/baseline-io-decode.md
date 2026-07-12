# Kimi IO batch-profile breakdown

This report summarizes `GGML_MOE_IO_BATCH_PROFILE_OUT` rows. It measures
how much independent read work the runtime actually exposes to io_uring.

## Filters

- min jobs: `None`
- max jobs: `8`

## Aggregate

- Batches: `5579`
- Read jobs: `25623`
- Avg read jobs/batch: `4.593`
- Weighted inflight avg: `3.479`
- Max inflight: `8`
- Wait: `15622.196 ms`
- Wall: `16633.431 ms`
- Wait/read job: `0.610 ms`
- Read batches <=4 jobs: `0.495`
- Read batches <=8 jobs: `1.000`

## By Prompt

| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 5579 | 25623 | 4.593 | 3.479 | 15622.196 | 16633.431 | 0.495 |

## By Op/Role

| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 1859 | 7933 | 4.267 | 3.507 | 5219.899 | 5691.274 | 0.572 |
| `runtime_load` | `up` | 1859 | 7931 | 4.266 | 3.177 | 5098.721 | 5401.630 | 0.571 |
| `runtime_load` | `down` | 962 | 5408 | 5.622 | 3.861 | 3041.425 | 3173.936 | 0.258 |
| `current_down_overlap` | `down` | 899 | 4351 | 4.840 | 3.504 | 2262.150 | 2366.591 | 0.432 |

## Top 20 Layer/Role Buckets By Wait

| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `gate` | 1 | 31 | 168 | 5.419 | 4.123 | 124.010 | 239.807 |
| `runtime_load` | `down` | 10 | 31 | 189 | 6.097 | 4.132 | 117.879 | 121.872 |
| `runtime_load` | `up` | 1 | 31 | 168 | 5.419 | 3.739 | 116.705 | 214.673 |
| `runtime_load` | `down` | 9 | 31 | 175 | 5.645 | 3.878 | 116.015 | 119.997 |
| `runtime_load` | `down` | 4 | 31 | 201 | 6.484 | 4.080 | 114.020 | 118.507 |
| `runtime_load` | `down` | 7 | 31 | 184 | 5.935 | 3.946 | 113.192 | 117.474 |
| `runtime_load` | `down` | 6 | 31 | 183 | 5.903 | 3.819 | 112.675 | 117.021 |
| `runtime_load` | `gate` | 28 | 31 | 160 | 5.161 | 4.200 | 111.747 | 115.291 |
| `runtime_load` | `gate` | 29 | 31 | 165 | 5.323 | 4.215 | 110.030 | 114.033 |
| `runtime_load` | `down` | 18 | 31 | 173 | 5.581 | 3.797 | 109.472 | 113.153 |
| `runtime_load` | `down` | 24 | 31 | 188 | 6.065 | 4.060 | 108.997 | 112.809 |
| `runtime_load` | `down` | 23 | 31 | 184 | 5.935 | 3.807 | 107.433 | 111.629 |
| `runtime_load` | `down` | 16 | 31 | 180 | 5.806 | 3.929 | 107.058 | 110.933 |
| `runtime_load` | `down` | 8 | 31 | 168 | 5.419 | 3.740 | 105.998 | 109.674 |
| `runtime_load` | `gate` | 51 | 31 | 142 | 4.581 | 3.686 | 105.525 | 108.927 |
| `runtime_load` | `up` | 29 | 31 | 165 | 5.323 | 3.498 | 105.081 | 108.629 |
| `runtime_load` | `up` | 28 | 31 | 160 | 5.161 | 3.331 | 104.254 | 107.828 |
| `runtime_load` | `down` | 25 | 31 | 172 | 5.548 | 3.783 | 104.048 | 107.849 |
| `runtime_load` | `gate` | 10 | 31 | 163 | 5.258 | 3.904 | 103.768 | 107.671 |
| `runtime_load` | `down` | 15 | 31 | 155 | 5.000 | 3.675 | 103.467 | 106.835 |

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
