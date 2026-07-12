# Kimi expert-ID predictor admission

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Head while running: `9330c27bc`

## Goal

After rejecting static RAM tiering, direct tiny IQ1_S/Q2_K replacement, and
hidden+route MoE-output surrogates, test whether expert-ID prediction can expose
future complete batches well enough to justify a runtime prefetch path.

This is dev-only offline analysis. It does not use held-out/test prompts, does
not change runtime behavior, and does not claim SOTA.

Input profile root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
```

Prompts:

- `dev_france_regression`
- `dev_intelligence_general`

## Admission Gates

A runtime prefetch/draft-router path is worth implementing only if the offline
signal can plausibly create useful complete future batches:

- byte recall: `>=65%`;
- predicted/actual bytes: `<=1.35x`;
- full-step or complete-batch coverage should be high enough to reduce exposed
  wait, not just row-level hit rate;
- no held-out/test prompt traces may be used for training or threshold choice.

## Static Prior

Tool:

```text
.Agent/run-tools/kimi_static_prior_predictability.py
```

Artifact:

```text
.Agent/runs/20260712-static-prior-expert-id-admission/static-prior.md
```

Method: leave-one-prompt-out; train per-tensor top-K experts on the other dev
prompt.

Result:

| top K | recall | precision | byte recall | predicted/actual bytes |
|---:|---:|---:|---:|---:|
| 8 | `0.0908` | `0.1118` | `0.0908` | `0.8117` |
| 16 | `0.1572` | `0.0969` | `0.1573` | `1.6233` |
| 32 | `0.2455` | `0.0756` | `0.2454` | `3.2467` |
| 64 | `0.3789` | `0.0584` | `0.3788` | `6.4933` |
| 128 | `0.5728` | `0.0462` | `0.5729` | `12.3965` |

Decision: reject static prior prefetch. It cannot reach the `>=65%` recall gate,
and the high-recall settings overfetch by more than `12x`.

## FineMoE-Style Route Prefix

Tool:

```text
.Agent/run-tools/kimi_finemoe_route_prefix_bound.py
```

Artifacts:

```text
.Agent/runs/20260712-finemoe-prefix-expert-id-admission/dev_france_regression.md
.Agent/runs/20260712-finemoe-prefix-expert-id-admission/dev_intelligence_general.md
```

Method: within a prompt, use earlier decode segments as history. For the current
token, use already-known prefix layers to choose the nearest historical route
prefix and predict future layers at distances 1-8.

Best observed rows:

| prompt | best distance | prefix byte recall | useful / false-positive | windows |
|---|---:|---:|---:|---:|
| `dev_france_regression` | 1 | `0.4230` | `0.501` | `1829` |
| `dev_intelligence_general` | 6-8 | `0.4340` | `0.552` | `1674-1612` |

Decision: reject FineMoE-style prefix prefetch for this Kimi trace. It is better
than static prior but still misses the recall gate by about 20 percentage
points, and useful bytes are lower than false-positive bytes.

## Overall Decision

Do not implement a runtime expert-ID prefetch path from these predictors.

The theoretical future-window bound is large, but the available prompt-general
signals do not expose useful complete future batches without excessive wrong
bytes. A future predictor attempt needs a substantially stronger signal, such as
a real draft/router model that predicts future expert IDs directly and passes
the same complete-batch admission before any runtime reads.

The next optimization direction should stay focused on:

- real lower-byte payloads that pass current-pack output-error gates;
- exact storage/layout changes that reduce moved bytes or exposed wait, not just
  reorder the same bytes;
- hardware/storage changes that raise the effective byte/sec ceiling.

## Reproduce

Static prior:

```bash
OUT=.Agent/runs/20260712-static-prior-expert-id-admission
python3 .Agent/run-tools/kimi_static_prior_predictability.py \
  --trace /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_france_regression/route-trace.csv \
  --trace /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/dev_intelligence_general/route-trace.csv \
  --top-k 8,16,32,64,128,192,256 \
  --out-json "$OUT/static-prior.json" \
  --out-csv "$OUT/static-prior.csv" \
  --out-md "$OUT/static-prior.md"
```

FineMoE prefix:

```bash
OUT=.Agent/runs/20260712-finemoe-prefix-expert-id-admission
for prompt in dev_france_regression dev_intelligence_general; do
  python3 .Agent/run-tools/kimi_finemoe_route_prefix_bound.py \
    --trace /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717/$prompt/route-trace.csv \
    --out-json "$OUT/$prompt.json" \
    --out-md "$OUT/$prompt.md" \
    --max-distance 8 \
    --min-segment-layers 50
done
```
