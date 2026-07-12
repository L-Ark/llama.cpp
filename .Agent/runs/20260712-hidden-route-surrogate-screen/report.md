# Kimi hidden+route MoE-output surrogate screen

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Head while running: `c440d510d`

## Goal

Test whether a stronger runtime-available signal can replace or predict enough
MoE work to justify future predictor/surrogate optimization.

The signal is:

- pre-MoE hidden vector from the activation dump;
- selected expert IDs / route stats.

The target is the summed active-expert MoE output for a layer/token group. This
is stricter than route-history prediction: if this small hidden+route surrogate
cannot approximate MoE output, it is not a good near-term path for replacing
expert bytes or compute.

This is dev-only offline analysis. It does not use held-out prompts, does not
change runtime behavior, and does not claim SOTA.

## Inputs

Dev activation corpus:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2
```

Prompts:

- `dev_france_regression`
- `dev_japan_factual`
- `dev_photosynthesis_factual`

The full-corpus run completed and produced:

- groups: `468`;
- layers evaluated: `53`;
- leave-one-prompt-out evaluation;
- complete top-8 groups only.

A bounded sanity screen with `--max-records-per-prompt 720` also ran:

- groups: `75`;
- layers evaluated: `25`.

## Result

Artifact:

- `.Agent/runs/20260712-hidden-route-surrogate-screen/report.json`
- `.Agent/runs/20260712-hidden-route-surrogate-screen/report-small.md`
- `.Agent/runs/20260712-hidden-route-surrogate-screen/report-small.json`
- `.Agent/runs/20260712-hidden-route-surrogate-screen/run-small.log`

Full-corpus best method:

- `nn_scaled:input_route_stats`
- mean rel-L2: `0.885045`
- max rel-L2: `1.298097`

Other full-corpus methods were worse:

- `nn_scaled:input`: mean rel-L2 `0.885192`;
- `nn_scaled:input_stats`: mean rel-L2 `0.885210`;
- best KRR family: mean rel-L2 `0.946113`.

Bounded sanity screen:

- best method: `krr:input_route_gated:lambda=10`;
- mean rel-L2: `1.064523`;
- max rel-L2: `1.492350`.

Gate:

- target mean rel-L2: `<=0.10`;
- observed full-corpus best mean rel-L2: `0.885045`;
- margin over gate: `8.85x`.

## Decision

Reject small hidden+route full-MoE-output surrogates as the next optimization
path.

This closes the near-term idea that a lightweight prototype/kernel surrogate can
replace active expert movement or compute under the current data and feature
set. It does not reject all future predictor work, but a future attempt must
predict expert IDs or future complete batches directly with a much stronger
signal, not try to approximate the full MoE output from this small feature set.

Next direction should return to exact byte movement and storage:

- stronger future-expert ID prediction only if a draft/router model can pass
  admission on complete-batch coverage and predicted/actual bytes;
- pack/layout or duplication only if it reduces moved bytes or exposed wait by
  more than the already rejected bounds;
- lower-byte only with a real payload that passes current-pack output-error
  gating by a wide margin.

## Reproduce

Full-corpus screen:

```bash
RUN=.Agent/runs/20260712-hidden-route-surrogate-screen
python3 .Agent/run-tools/kimi_input_route_moe_surrogate_oracle.py \
  --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_france_regression \
  --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_japan_factual \
  --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_photosynthesis_factual \
  --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
  --libggml-base build-cuda-batch/bin/libggml-base.so \
  --out-json "$RUN/report.json" \
  --out-md "$RUN/report-full.md" \
  --modes input,input_stats,input_route_stats,input_route_gated \
  --lambdas 0.1,1.0,10.0,100.0 \
  --complete-group-only \
  --target-rel-l2 0.10 \
  --torch-threads 8
```

Bounded sanity screen:

```bash
RUN=.Agent/runs/20260712-hidden-route-surrogate-screen
python3 .Agent/run-tools/kimi_input_route_moe_surrogate_oracle.py \
  --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_france_regression \
  --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_japan_factual \
  --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_photosynthesis_factual \
  --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
  --libggml-base build-cuda-batch/bin/libggml-base.so \
  --out-json "$RUN/report-small.json" \
  --out-md "$RUN/report-small.md" \
  --max-records-per-prompt 720 \
  --modes input,input_stats,input_route_stats,input_route_gated \
  --lambdas 1.0,10.0 \
  --complete-group-only \
  --target-rel-l2 0.10 \
  --torch-threads 8
```
