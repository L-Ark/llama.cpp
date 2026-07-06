# Kimi budgeted lower-byte hotset bound

- Remote plan: `.Agent/runs/20260707-gp21-remote-range-pack/plan.tsv`
- Kinds: `down,gate,up`
- Profiles: `7`
- Candidate keys: `56896`

| budget GiB | selected entries | pack GiB | hybrid byte ratio | current GiB | hybrid GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 62.000 | 19470 | 61.999 | 0.7626 | 3506.568 | 2674.038 |
| 85.000 | 26052 | 84.999 | 0.7348 | 3506.568 | 2576.459 |
| 115.000 | 34420 | 114.998 | 0.7110 | 3506.568 | 2493.161 |

## Worst Prompt Ratios

- `62.000 GiB`: worst `dev_python_reverse` `0.7955`, best `dev_france_regression` `0.7395`.
- `85.000 GiB`: worst `dev_linear_equation` `0.7574`, best `dev_france_regression` `0.7186`.
- `115.000 GiB`: worst `dev_python_reverse` `0.7237`, best `dev_france_regression` `0.7019`.

## Method

This is a dev-only bound. It uses
`.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct/*/route-profile.csv`
and does not inspect held-out test prompts.

For each `(tensor, expert)` key in the dev route profiles, the analysis maps the
current IQ3_S expert byte size to the AesSedai lower-byte tensor metadata from
GP21. It then selects keys greedily by saved runtime bytes per lower-byte pack
storage cost.

Reproduction:

```bash
.Agent/run-tools/kimi_lowbyte_budgeted_hotset.py \
  --remote-plan-tsv .Agent/runs/20260707-gp21-remote-range-pack/plan.tsv \
  $(find .Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct -path '*/route-profile.csv' | sort | sed 's#^#--profile #') \
  --kind up,gate,down \
  --budget-gib 62 \
  --budget-gib 85 \
  --budget-gib 115 \
  --out-dir .Agent/runs/20260707-gp25-budgeted-lowbyte-hotset
```

Large-reference run:

```bash
.Agent/run-tools/kimi_lowbyte_budgeted_hotset.py \
  --remote-plan-tsv .Agent/runs/20260707-gp21-remote-range-pack/plan.tsv \
  $(find .Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct -path '*/route-profile.csv' | sort | sed 's#^#--profile #') \
  --kind up,gate,down \
  --budget-gib 150 \
  --budget-gib 200 \
  --budget-gib 260 \
  --out-dir .Agent/runs/20260707-gp25-budgeted-lowbyte-hotset/ref-large
```

## Large Reference

| budget GiB | selected entries | pack GiB | hybrid byte ratio |
| ---: | ---: | ---: | ---: |
| 150 | 44045 | 150.000 | 0.6952 |
| 200 | 56360 | 199.999 | 0.6864 |
| 260 | 56896 | 202.804 | 0.6863 |

At `260 GiB`, every dev-profile candidate is selected. The remaining byte ratio
is still `0.6863`, so the AesSedai lower-byte representation itself is not
small enough to approach the GP10 target range of about `0.39-0.55x`.

## Manifest Dry-Run

The `85 GiB` and `115 GiB` selected plans were converted into remote-pack
manifests locally with:

```bash
python3 scripts/kimi-make-remote-pack-manifest.py ...
python3 scripts/kimi-build-remote-pack-from-manifest.py --dry-run ...
```

Results:

- `85 GiB`: `26052` entries, `84.999 GiB`, `invalid_remote_range_count=0`,
  `bad_pack_offset_count=0`.
- `115 GiB`: `34420` entries, `114.998 GiB`, `invalid_remote_range_count=0`,
  `bad_pack_offset_count=0`.

The local `fits_output_fs=0` in dry-run summaries is not a remote capacity
measurement; the output path points to `/root/...` while this check was run on
the local workstation. The layout/range validation is still valid.

## Decision

The prompt-agnostic selected hotset is more balanced than the France-derived
hotset, but the byte ratio is still far too high:

- `85 GiB` pack budget: `0.7348x`.
- `115 GiB` pack budget: `0.7110x`.
- full dev candidate set at `202.804 GiB`: `0.6863x`.

This rules out selected AesSedai lower-byte expert packs as the next primary
path to `5 tok/s`. The next primary branch should require either a substantially
smaller representation than AesSedai `IQ2_XXS`, a runtime method that avoids
reading most expert bytes, or a model-side/draft/speculative mechanism with
verified acceptance.
