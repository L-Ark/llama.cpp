# Kimi batch-cover RAM oracle

This is a dev-only upper bound. It selects arbitrary complete IO batches
from profiled prompts and therefore can overfit the dev traces. It does
not change runtime behavior or claim SOTA.

## Inputs

- traces: `2`
- prompts: `2`
- roles: `down,gate,up`
- max jobs: `8`
- batches: `11134`
- unique expert keys: `24053`
- decode runs: `62`
- total wait: `31073.997 ms`
- total wait/token: `501.193 ms/token`

## Greedy Complete-Batch Cover Bound

| budget MiB | resident GiB | entries | covered batches | saved wait ms | saved ms/token | remaining ms/token | bounded tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.997 | 220 | 146 | 361.987 | 5.839 | 591.450 | 1.691 |
| 2048 | 1.999 | 442 | 291 | 667.810 | 10.771 | 586.517 | 1.705 |
| 4096 | 3.998 | 892 | 539 | 1337.495 | 21.573 | 575.716 | 1.737 |
| 6144 | 5.998 | 1340 | 817 | 2015.697 | 32.511 | 564.777 | 1.771 |
| 8192 | 7.999 | 1787 | 1052 | 2651.164 | 42.761 | 554.528 | 1.803 |
| 10240 | 9.999 | 2217 | 1286 | 3313.355 | 53.441 | 543.847 | 1.839 |

## Decision Notes

- If this oracle cannot reach the target, RAM batch-covering is not enough
  even with dev-trace overfitting.
- If it reaches the target only by selecting prompt-specific batches, the
  next step must be a general predictor or storage-format change, not a
  static dev hotset.
- Runtime A/B needs a prompt-general policy that reproduces this coverage
  without increasing TTFT by more than 20% or exceeding 16GB host RAM.
