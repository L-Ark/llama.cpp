# Kimi dev7 N96 filtered hidden admission

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Scope: dev-only offline admission; no held-out/test prompts; no SOTA claim.

## Goal

Refresh the future-expert predictor admission after fixing trace alignment.
The experiment tests whether route-score/history or filtered hidden vectors can
predict the next future layer's complete expert set well enough to justify
runtime prefetch reads.

Admission gate:

- useful expert recall >= `0.65`;
- predicted/actual bytes <= `1.35x`;
- complete-step coverage >= `0.40`;
- cold-start corpus collected under `MemoryMax=15900000000`,
  `MemorySwapMax=0`;
- all dev prompt outputs must pass semantic quality checks.

## Corpus

Root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-n96-filtered-hidden-corpus
```

Prompt file:

```text
.Agent/evals/kimi-general-dev-prompts.jsonl
```

Collection command:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-n96-filtered-hidden-corpus
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root "$ROOT" \
  --mode dev \
  --n 96 \
  --profile \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 1200 \
  --vram-mib 15000 \
  --threads 32 \
  --pinned-slots 12 \
  --upgate-pct 72 \
  --extra-runtime-env "GGML_MOE_ROUTE_SCORE_TRACE_OUT=\$RUN/route-score-trace.csv
GGML_MOE_ROUTE_DETAIL_OUT=\$RUN/route-detail.csv
GGML_MOE_ACTIVATION_DUMP_DIR=\$RUN
GGML_MOE_ACTIVATION_DUMP_MAX_RECORDS=20000
GGML_MOE_ACTIVATION_DUMP_CALL_STRIDE=1
GGML_MOE_ACTIVATION_DUMP_DECODE_ONLY=1
GGML_MOE_ACTIVATION_DUMP_ROLE_FILTER=up
GGML_MOE_ACTIVATION_DUMP_ACTIVE_SLOT=0"
```

Instrumentation note:

- activation dump is filtered to `role=up, active_slot=0`;
- complete expert labels come from `route-score-trace.csv`, not activation rows;
- this keeps the full dev7 N96 hidden corpus to about `1005 MiB`.

Corpus metrics:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB | route rows | activation rows | complete groups | f32 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.64` | `8200.50` | `51860.43/85` | `12.108` | `146496` | `5101` | `85` | `139.5` |
| `dev_japan_factual` | pass | `1.66` | `8800.24` | `47031.55/78` | `12.094` | `136416` | `4681` | `78` | `128.0` |
| `dev_linear_equation` | pass | `1.46` | `11143.78` | `23242.72/34` | `11.976` | `82968` | `2041` | `34` | `55.8` |
| `dev_mixed_summary` | pass | `1.60` | `11043.66` | `30678.48/49` | `12.003` | `104568` | `2941` | `49` | `80.4` |
| `dev_photosynthesis_factual` | pass | `1.61` | `7387.85` | `58214.73/94` | `11.954` | `155208` | `5641` | `94` | `154.2` |
| `dev_python_reverse` | pass | `1.53` | `9171.05` | `62118.78/95` | `12.136` | `162312` | `5701` | `95` | `155.9` |
| `dev_zh_france` | pass | `1.71` | `7892.55` | `28687.69/49` | `11.849` | `91824` | `2941` | `49` | `80.4` |

## Route-Score Baseline

Artifact:

```text
.Agent/runs/20260712-current-goal-dev7-n96-filtered-hidden-admission/route-score-analysis/analysis.md
```

Result:

- best simple score/history policy: `hybrid_layer8_token8`;
- recall `0.3496`;
- pred/actual `1.949x`;
- full-step `0.0008`;
- decision: reject; it fails recall, byte, and complete-step gates.

## Route-History Hybrid Baseline

Artifact:

```text
.Agent/runs/20260712-current-goal-dev7-n96-filtered-hidden-admission/hybrid-route-analysis/report.md
```

Best all-role row:

- `hybrid_recent`, H=`3`, budget=`32`;
- recall `0.5959`;
- pred/actual `4.0000x`;
- full-step `0.0357`.

Best admissible-byte budget-8 row:

- `hybrid_balanced`, H=`3`, budget=`8`;
- recall `0.3580`;
- pred/actual `1.0000x`;
- full-step `0.0005`.

Decision:

- reject route-history future prefetch;
- recall and full-step coverage are far below the gate even when byte ratio is
  admissible;
- larger budgets improve recall but exceed the byte gate by `2-4x`.

## Hidden-Vector H1 Admission

Tool:

```text
.Agent/run-tools/kimi_hidden_route_future_admission.py
```

Artifact:

```text
.Agent/runs/20260712-current-goal-dev7-n96-filtered-hidden-admission/hidden-route-h1-budget8-analysis/report.md
```

Command:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-dev7-n96-filtered-hidden-corpus
OUT=.Agent/runs/20260712-current-goal-dev7-n96-filtered-hidden-admission/hidden-route-h1-budget8-analysis
/usr/bin/time -v python3 .Agent/run-tools/kimi_hidden_route_future_admission.py \
  --root "$ROOT" \
  --out-dir "$OUT" \
  --layers 60 \
  --horizons 1 \
  --budgets 8 \
  --neighbors 1,3,5 \
  --feature-dim 128
```

Offline analysis cost:

- wall time `2:14.83`;
- max RSS `222680 KB`;
- this is offline analysis only, not runtime cost.

Result:

| policy | H | budget | recall | pred/actual | full-step |
|---|---:|---:|---:|---:|---:|
| `hidden_knn5` | `1` | `8` | `0.236955` | `1.000000` | `0.000981` |
| `hidden_knn3` | `1` | `8` | `0.234097` | `1.000000` | `0.002451` |
| `hidden_knn1` | `1` | `8` | `0.226393` | `1.000000` | `0.002907` |
| `static_target_layer` | `1` | `8` | `0.125801` | `1.000000` | `0.000000` |
| `source_route_plus_static` | `1` | `8` | `0.020153` | `1.000000` | `0.000000` |

Decision:

- reject hidden-vector future prefetch as the next runtime path;
- best hidden row recall is only `0.236955`, far below `0.65`;
- full-step coverage is only `0.000981`, far below `0.40`;
- it is also worse than the route-history budget-8 baseline recall `0.3580`;
- do not implement runtime prefetch reads from this filtered hidden KNN path.

## Next Action

Future-expert prediction remains unsupported by dev evidence. The next Kimi
optimization step should move back to byte/storage work:

1. reduce moved expert bytes with a quality-preserving representation, or
2. redesign RAM/VRAM storage so RAM holds batchable high-yield expert payloads,
   not uncontrolled GGUF page cache, or
3. only revisit prediction with a materially stronger signal, such as a real
   draft/router model with complete-batch admission.

Runtime prefetch should not be implemented from route-score, route-history, or
filtered hidden KNN predictors.
