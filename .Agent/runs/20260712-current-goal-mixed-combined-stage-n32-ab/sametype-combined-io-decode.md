# Kimi IO batch-profile breakdown

This report summarizes `GGML_MOE_IO_BATCH_PROFILE_OUT` rows. It measures
how much independent read work the runtime actually exposes to io_uring.

## Filters

- min jobs: `None`
- max jobs: `8`

## Aggregate

- Batches: `4753`
- Read jobs: `22431`
- Avg read jobs/batch: `4.719`
- Weighted inflight avg: `3.544`
- Max inflight: `8`
- Wait: `13160.784 ms`
- Wall: `13974.322 ms`
- Wait/read job: `0.587 ms`
- Read batches <=4 jobs: `0.473`
- Read batches <=8 jobs: `1.000`

## By Prompt

| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sametype-combined` | 4753 | 22431 | 4.719 | 3.544 | 13160.784 | 13974.322 | 0.473 |

## By Op/Role

| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `up` | 1591 | 7241 | 4.551 | 3.411 | 4221.447 | 4435.856 | 0.529 |
| `runtime_load` | `gate` | 1301 | 5431 | 4.174 | 3.457 | 3658.902 | 4015.177 | 0.594 |
| `runtime_load` | `down` | 962 | 5408 | 5.622 | 3.885 | 3029.710 | 3167.459 | 0.258 |
| `current_down_overlap` | `down` | 899 | 4351 | 4.840 | 3.451 | 2250.725 | 2355.830 | 0.432 |

## Top 20 Layer/Role Buckets By Wait

| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | `down` | 7 | 31 | 184 | 5.935 | 3.915 | 119.564 | 123.776 |
| `runtime_load` | `gate` | 29 | 31 | 165 | 5.323 | 4.320 | 119.047 | 122.921 |
| `runtime_load` | `down` | 9 | 31 | 175 | 5.645 | 3.960 | 116.108 | 120.206 |
| `runtime_load` | `gate` | 28 | 31 | 160 | 5.161 | 3.891 | 114.828 | 118.835 |
| `runtime_load` | `down` | 10 | 31 | 189 | 6.097 | 4.045 | 114.420 | 118.291 |
| `runtime_load` | `down` | 6 | 31 | 183 | 5.903 | 4.005 | 114.114 | 118.149 |
| `runtime_load` | `down` | 4 | 31 | 201 | 6.484 | 4.000 | 113.553 | 117.929 |
| `runtime_load` | `up` | 29 | 31 | 165 | 5.323 | 3.478 | 111.296 | 115.254 |
| `runtime_load` | `down` | 24 | 31 | 188 | 6.065 | 4.113 | 109.838 | 114.187 |
| `runtime_load` | `down` | 23 | 31 | 184 | 5.935 | 3.825 | 108.468 | 112.786 |
| `runtime_load` | `down` | 8 | 31 | 168 | 5.419 | 3.755 | 107.719 | 111.495 |
| `runtime_load` | `down` | 18 | 31 | 173 | 5.581 | 3.833 | 107.038 | 110.915 |
| `runtime_load` | `gate` | 32 | 31 | 145 | 4.677 | 3.632 | 106.383 | 110.241 |
| `runtime_load` | `up` | 28 | 31 | 160 | 5.161 | 3.311 | 105.289 | 109.659 |
| `runtime_load` | `up` | 9 | 31 | 143 | 4.613 | 3.514 | 105.130 | 108.553 |
| `runtime_load` | `gate` | 51 | 31 | 142 | 4.581 | 3.541 | 104.928 | 108.271 |
| `runtime_load` | `down` | 16 | 31 | 180 | 5.806 | 4.000 | 104.461 | 108.187 |
| `runtime_load` | `down` | 25 | 31 | 172 | 5.548 | 3.903 | 104.143 | 108.158 |
| `runtime_load` | `up` | 32 | 31 | 145 | 4.677 | 3.216 | 103.432 | 106.856 |
| `runtime_load` | `gate` | 33 | 31 | 144 | 4.645 | 3.980 | 103.409 | 106.894 |

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
