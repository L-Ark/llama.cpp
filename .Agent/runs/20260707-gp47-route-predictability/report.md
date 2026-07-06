# GP47 expert route predictability

Timestamp: `2026-07-07T06:55:58+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit:

```text
8eb1091d2 tools: sweep kimi v2 shadow hotsets
```

Purpose:

- Test whether simple route-history prediction can prefetch experts early enough
  to improve IO queue continuity.
- Predictor tested: previous call for the same tensor predicts the current
  active expert set.

Inputs:

- Seven dev-only N96 `route-trace.csv` files from:
  `.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile`.
- No held-out test traces were used.

Method:

- Group consecutive route-trace rows with the same tensor into one MoE call.
- For each tensor call after the first occurrence of that tensor:
  - predicted experts = previous active expert set for that tensor;
  - actual experts = current active expert set;
  - compute recall, precision, byte recall, and predicted/actual byte ratio.

Command:

```bash
RUN=.Agent/runs/20260707-gp47-route-predictability
mkdir -p "$RUN"
traces=()
while IFS= read -r p; do traces+=(--trace "$p"); done < <(
  find .Agent/runs/20260706-kimi-general-dev-baseline-n96-profile \
    -name route-trace.csv | sort
)
python3 .Agent/run-tools/kimi_expert_predictability_from_trace.py \
  "${traces[@]}" \
  --out-json "$RUN/predictability.json" \
  --out-csv "$RUN/predictability.csv" \
  --out-md "$RUN/predictability.md"
```

Headline result:

```text
traces=7
calls=84445
predictable_call_ratio=0.985659
recall=0.337820
precision=0.342736
byte_recall=0.337250
predicted_to_actual_byte_ratio=0.985659
```

Aggregate by kind:

| kind | recall | precision | byte recall | predicted/actual bytes |
| --- | ---: | ---: | ---: | ---: |
| all | 0.3378 | 0.3427 | 0.3373 | 0.9857 |
| down | 0.3404 | 0.3454 | 0.3388 | 0.9857 |
| gate | 0.3367 | 0.3416 | 0.3380 | 0.9857 |
| up | 0.3367 | 0.3416 | 0.3345 | 0.9857 |

Interpretation:

- Simple previous-token same-tensor prediction is too weak for runtime prefetch:
  it predicts almost a full active set worth of bytes (`0.986x`) but only covers
  about one third of the needed experts.
- Enabling such a prefetch would likely waste SSD/H2D/VRAM bandwidth and could
  hurt token rate or TTFT under the 16GB host RAM constraint.
- A useful predictor must be stronger than previous-token repetition, for
  example:
  - a learned/draft-model route predictor with measured acceptance/recall;
  - a lower-cost online predictor that predicts more than one future layer with
    high recall and bounded overfetch;
  - a scheduler change that increases independent jobs without guessing many
    wrong experts.

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
