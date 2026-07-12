# Kimi hidden-feature future-expert admission

This is a dev-only offline screen. It does not change runtime behavior or claim SOTA.

## Goal

Test whether activation-dump vectors provide a stronger prompt-general signal for future expert-ID prefetch than route history or router scores.

## Corpus

- root: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2`
- prompts: `3`
- horizons: `1,2,3`
- budgets: `8,16,32`

| prompt | groups | layers | token ids | complete upgate groups | down groups |
|---|---:|---:|---|---:|---:|
| `dev_france_regression` | `178` | `60` | `0` | `178` | `156` |
| `dev_japan_factual` | `178` | `60` | `0` | `178` | `156` |
| `dev_photosynthesis_factual` | `178` | `60` | `0` | `178` | `156` |

Important corpus limitation:

- the current activation dumps have `token_id=0` for all records, so this screen uses execution `call` order rather than true token/layer sequence labels;
- the dump was produced for representation/surrogate screening, not a full future-prefetch admission corpus;
- a passing row here would still require a fresh full N96 trace with explicit hidden/router labels before runtime reads.

## Admission Gate

- recall >= `0.65`
- predicted/actual bytes <= `1.35`
- full-step coverage >= `0.4`
- passing rows: `0`

## Top Rows

| H | policy | budget | recall | precision | pred/actual | full steps | samples |
|---:|---|---:|---:|---:|---:|---:|---:|
| `2` | `static_target_layer` | `32` | `0.631392` | `0.157848` | `4.000000` | `0.109848` | `528` |
| `2` | `hidden_knn1_same_layer` | `32` | `0.631392` | `0.157848` | `4.000000` | `0.109848` | `528` |
| `2` | `hidden_knn1_target_layer` | `32` | `0.631392` | `0.157848` | `4.000000` | `0.109848` | `528` |
| `2` | `hidden_knn3_same_layer` | `32` | `0.631392` | `0.157848` | `4.000000` | `0.109848` | `528` |
| `2` | `hidden_knn3_target_layer` | `32` | `0.631392` | `0.157848` | `4.000000` | `0.109848` | `528` |
| `2` | `hidden_knn5_same_layer` | `32` | `0.631392` | `0.157848` | `4.000000` | `0.109848` | `528` |
| `2` | `hidden_knn5_target_layer` | `32` | `0.631392` | `0.157848` | `4.000000` | `0.109848` | `528` |
| `1` | `static_target_layer` | `32` | `0.631356` | `0.157839` | `4.000000` | `0.109228` | `531` |
| `1` | `hidden_knn1_same_layer` | `32` | `0.631356` | `0.157839` | `4.000000` | `0.109228` | `531` |
| `1` | `hidden_knn1_target_layer` | `32` | `0.631356` | `0.157839` | `4.000000` | `0.109228` | `531` |
| `1` | `hidden_knn3_same_layer` | `32` | `0.631356` | `0.157839` | `4.000000` | `0.109228` | `531` |
| `1` | `hidden_knn3_target_layer` | `32` | `0.631356` | `0.157839` | `4.000000` | `0.109228` | `531` |
| `1` | `hidden_knn5_same_layer` | `32` | `0.631356` | `0.157839` | `4.000000` | `0.109228` | `531` |
| `1` | `hidden_knn5_target_layer` | `32` | `0.631356` | `0.157839` | `4.000000` | `0.109228` | `531` |
| `3` | `static_target_layer` | `32` | `0.629048` | `0.157262` | `4.000000` | `0.108571` | `525` |
| `3` | `hidden_knn1_same_layer` | `32` | `0.629048` | `0.157262` | `4.000000` | `0.108571` | `525` |
| `3` | `hidden_knn1_target_layer` | `32` | `0.629048` | `0.157262` | `4.000000` | `0.108571` | `525` |
| `3` | `hidden_knn3_same_layer` | `32` | `0.629048` | `0.157262` | `4.000000` | `0.108571` | `525` |
| `3` | `hidden_knn3_target_layer` | `32` | `0.629048` | `0.157262` | `4.000000` | `0.108571` | `525` |
| `3` | `hidden_knn5_same_layer` | `32` | `0.629048` | `0.157262` | `4.000000` | `0.108571` | `525` |

## Decision

Reject hidden-feature future-expert prefetch from the existing activation corpus: best row horizon `2` `static_target_layer` budget `32` reaches recall `0.631392` and full-step `0.109848` with pred/actual `4.000000`, so it does not pass admission.

Next predictor work must add new instrumentation or a real draft/router model, not reuse this limited activation corpus as proof.

## Reproduce

```bash
.Agent/run-tools/kimi_hidden_feature_future_expert_admission.py --root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2 --out-dir .Agent/runs/20260712-current-goal-hidden-feature-admission --horizons 1,2,3 --budgets 8,16,32 --neighbors 1,3,5
```
