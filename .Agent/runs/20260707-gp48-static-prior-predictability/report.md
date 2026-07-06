# GP48 static expert-prior predictability

Timestamp: `2026-07-07T07:00:00+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit:

```text
01c33f542 tools: analyze kimi route predictability
```

Purpose:

- Evaluate whether prompt-agnostic static per-tensor expert priors can prefetch
  useful experts before routing.
- Use leave-one-prompt-out over dev traces to avoid prompt-specific tuning.

Inputs:

- Seven committed dev-only N96 `route-trace.csv` files from:
  `.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile`.
- No held-out test traces were used.

Method:

- For each held-out dev prompt:
  - train per-tensor expert counts from the other six dev prompts;
  - predict top-K experts for each tensor call in the held-out prompt;
  - measure recall, precision, byte recall, and predicted/actual byte ratio.

Command:

```bash
RUN=.Agent/runs/20260707-gp48-static-prior-predictability
mkdir -p "$RUN"
traces=()
while IFS= read -r p; do traces+=(--trace "$p"); done < <(
  find .Agent/runs/20260706-kimi-general-dev-baseline-n96-profile \
    -name route-trace.csv | sort
)
python3 .Agent/run-tools/kimi_static_prior_predictability.py \
  "${traces[@]}" \
  --top-k 8,16,32,64,128 \
  --out-json "$RUN/static-prior.json" \
  --out-csv "$RUN/static-prior.csv" \
  --out-md "$RUN/static-prior.md"
```

Headline result:

| top K | recall | precision | byte recall | predicted/actual bytes |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.123625 | 0.123625 | 0.123238 | 1.000000 |
| 16 | 0.192504 | 0.096252 | 0.192249 | 2.000000 |
| 32 | 0.284690 | 0.071172 | 0.284560 | 4.000000 |
| 64 | 0.416953 | 0.052119 | 0.416863 | 8.000000 |
| 128 | 0.601030 | 0.037564 | 0.600909 | 16.000000 |

Interpretation:

- Static per-tensor top-K priors do not provide a good prefetch tradeoff.
- Top-8 moves one active-set worth of bytes but recalls only `12.36%` of actual
  experts.
- Top-128 recalls only `60.10%` while requiring `16x` the actual expert bytes.
- Runtime integration would likely make IO/H2D pressure and TTFT worse under
  the 16GB RAM constraint.
- A viable predictor needs semantic/draft-model signal or a scheduler that
  increases independent known jobs without broad wrong-expert overfetch.

Validation:

- Local:
  - analyzer ran successfully;
  - `python3 -m py_compile` passed;
  - `git diff --check` passed.
- Remote:
  - analyzer ran successfully in
    `/root/lfz/tmp/vendor-kimi-speculative-gp33`;
  - headline metrics matched local;
  - `python3 -m py_compile` passed;
  - `git diff --check` passed.

Acceptance decision:

- Accepted as non-SOTA planning evidence.
- No runtime behavior changed.
- No held-out prompts were used.
- No token-rate or output-quality claim is made.
