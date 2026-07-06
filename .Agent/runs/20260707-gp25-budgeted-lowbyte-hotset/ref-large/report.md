# Kimi budgeted lower-byte hotset bound

- Remote plan: `.Agent/runs/20260707-gp21-remote-range-pack/plan.tsv`
- Kinds: `down,gate,up`
- Profiles: `7`
- Candidate keys: `56896`

| budget GiB | selected entries | pack GiB | hybrid byte ratio | current GiB | hybrid GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 150.000 | 44045 | 150.000 | 0.6952 | 3506.568 | 2437.919 |
| 200.000 | 56360 | 199.999 | 0.6864 | 3506.568 | 2407.034 |
| 260.000 | 56896 | 202.804 | 0.6863 | 3506.568 | 2406.691 |

## Worst Prompt Ratios

- `150.000 GiB`: worst `dev_linear_equation` `0.7004`, best `dev_france_regression` `0.6917`.
- `200.000 GiB`: worst `dev_python_reverse` `0.6866`, best `dev_japan_factual` `0.6863`.
- `260.000 GiB`: worst `dev_linear_equation` `0.6863`, best `dev_python_reverse` `0.6863`.
