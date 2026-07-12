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
- total wait: `30673.200 ms`
- total wait/token: `494.729 ms/token`

## Greedy Complete-Batch Cover Bound

| budget MiB | resident GiB | entries | covered batches | saved wait ms | saved ms/token | remaining ms/token | bounded tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2.000 | 439 | 267 | 725.012 | 11.694 | 584.950 | 1.710 |
| 4096 | 3.999 | 879 | 525 | 1382.611 | 22.300 | 574.344 | 1.741 |
| 6144 | 5.998 | 1327 | 782 | 2018.035 | 32.549 | 564.095 | 1.773 |
| 8192 | 8.000 | 1775 | 1033 | 2663.660 | 42.962 | 553.682 | 1.806 |
| 10240 | 9.996 | 2222 | 1267 | 3321.381 | 53.571 | 543.073 | 1.841 |
| 12288 | 11.998 | 2673 | 1493 | 3946.869 | 63.659 | 532.985 | 1.876 |

## Decision Notes

- If this oracle cannot reach the target, RAM batch-covering is not enough
  even with dev-trace overfitting.
- If it reaches the target only by selecting prompt-specific batches, the
  next step must be a general predictor or storage-format change, not a
  static dev hotset.
- Runtime A/B needs a prompt-general policy that reproduces this coverage
  without increasing TTFT by more than 20% or exceeding 16GB host RAM.
