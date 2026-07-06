# Kimi mradermacher IQ1_S budgeted hotset bound

- Scope: dev profiles only; held-out test prompts are not used.
- Asset: `mradermacher/Kimi-K2.7-Code-i1-GGUF` `i1-IQ1_S`.
- Runtime status: theoretical byte bound only, not directly runnable with the IQ3_S main GGUF.

| budget GiB | selected entries | pack GiB | hybrid byte ratio | current GiB | hybrid GiB | nbytes compatible |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 62.000 | 23151 | 61.999 | 0.6083 | 3506.568 | 2132.909 | False |
| 85.000 | 31692 | 84.998 | 0.5672 | 3506.568 | 1988.879 | False |
| 115.000 | 42746 | 114.998 | 0.5359 | 3506.568 | 1879.071 | False |

## All Dev Candidate Keys

- Entries: `56896`; pack: `153.753 GiB`; hybrid byte ratio: `0.5205`.
- Runtime nbytes mismatch count: `56896`.

## Worst Prompt Ratios

- `62.000 GiB`: worst `dev_python_reverse` `0.6503`, best `dev_france_regression` `0.5762`.
- `85.000 GiB`: worst `dev_linear_equation` `0.5919`, best `dev_france_regression` `0.5479`.
- `115.000 GiB`: worst `dev_python_reverse` `0.5455`, best `dev_france_regression` `0.5292`.

## Decision

- A selected IQ1_S hotset can fit in the current free-disk envelope for small budgets.
- It cannot be used directly by the current IQ3_S runtime because pack lookup keys include `nbytes`, and CUDA dispatch uses the main GGUF `src0_type`.
- Therefore this is useful as a byte-reduction bound, but not as a runtime experiment unless mixed-type expert override kernels are implemented or the full IQ1_S model is available.
