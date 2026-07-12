# Kimi Current-Goal Route-Score N96 Refresh

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Source commit: `aefdd65de`

This is a dev-only offline admission. It does not change default runtime
behavior and does not claim SOTA.

## Goal

Refresh the router-score predictor admission on current `n96` traces. Earlier
route-history prefetch failed on current N96 evidence; this test checks whether
graph-side router scores, weights, entropy, or score margin provide a stronger
signal for complete-batch prefetch.

Admission gate:

- all-role recall `>=65%`;
- predicted/actual bytes `<=1.35x`;
- full-step coverage high enough to cover complete future batches, with
  `>=40%` as the exploratory gate;
- no held-out/test prompt traces.

## Trace Collection

Trace root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-route-score-n96-080423
```

Injected env:

```text
GGML_MOE_ROUTE_SCORE_TRACE_OUT=$RUN/route-score-trace.csv
```

Command shape:

```bash
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO=/root/lfz/llama.cpp-vendor-kimi \
      RUN=<trace-root>/<prompt-id> \
      PROMPT_ID=<prompt-id> \
      PROMPT_USER_TEXT=<prompt> \
      QUALITY_KEYWORDS=<keywords> \
      N=96 PROFILE=0 COPY_PROFILE=0 \
      EXTRA_RUNTIME_ENV="GGML_MOE_ROUTE_SCORE_TRACE_OUT=<run>/route-score-trace.csv" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Trace quality:

| prompt | quality | tok/s | decode ms | decode tokens | TTFT ms | RAM peak GiB | trace lines |
|---|---|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.85` | `46015.45` | `85` | `8720.75` | `11.853` | `5102` |
| `dev_intelligence_general` | pass | `1.86` | `51083.76` | `95` | `6862.30` | `11.716` | `5702` |

Quality samples:

- France: `France is a country in Western Europe known for its rich history,
  culture, and influence on art, fashion, and cuisine...`
- Intelligence: `Intelligence is a complex, multi-faceted capacity rather than
  a single thing...`

## Score-Aware Predictor Analysis

Artifact:

```text
.Agent/runs/20260712-current-goal-route-score-n96-refresh/score-predictor-analysis/analysis.md
```

Command:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-route-score-n96-080423
OUT=.Agent/runs/20260712-current-goal-route-score-n96-refresh/score-predictor-analysis
python3 .Agent/run-tools/kimi_route_score_predictor_analysis.py \
  --root "$ROOT" \
  --out-dir "$OUT"
```

Decode route-score rows: `10800`.

Best policy results:

| policy | recall | precision | pred/actual | full-step |
|---|---:|---:|---:|---:|
| `prev_token_top8` | `0.3505` | `0.3544` | `0.989x` | `0.0008` |
| `hybrid_layer4_token4` | `0.2369` | `0.2414` | `0.981x` | `0.0000` |
| `hybrid_layer8_token8` | `0.3637` | `0.1862` | `1.953x` | `0.0011` |

Score-margin bins:

| margin bin | avg margin | avg entropy | next-token same-layer recall |
|---:|---:|---:|---:|
| `1` | `0.026511` | `2.015458` | `0.3375` |
| `2` | `0.093941` | `1.996387` | `0.3453` |
| `3` | `0.194943` | `1.972152` | `0.3586` |
| `4` | `0.404537` | `1.912267` | `0.3763` |

## Decision

Reject simple score-aware route prefetch as the next runtime path.

Reasons:

- best recall is only `0.3637`, far below the `0.65` gate;
- best recall requires `1.953x` predicted bytes, above the `1.35x` gate;
- full-step coverage is only `0.0011`, effectively no complete future batches;
- score margin is weakly informative but not sufficient: the highest-margin
  quartile reaches only `0.3763` next-token same-layer recall.

This confirms the previous N32 score-trace result on current N96 traces. Router
scores alone do not solve the queue starvation / complete-batch prefetch problem.

## Next Action

Do not implement runtime prefetch from simple route score/history policies.

Future prediction work must use a stronger signal, such as:

- a real draft/router model trained to predict future expert IDs;
- hidden-state/router-logit features with complete-batch admission;
- or a predictor combined with lower-byte expert representation so wrong bytes
  are cheap enough to prefetch.

Without that, continue toward lower-byte expert representation or the approved
full `i1-IQ1_S` same-model smoke.
