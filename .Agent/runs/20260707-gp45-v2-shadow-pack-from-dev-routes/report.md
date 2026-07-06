# GP45 v2 shadow pack from dev route profiles

Timestamp: `2026-07-07T06:46:00+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit:

```text
5d90765d5 tools: add kimi moepack v2 shadow profile
```

Purpose:

- Generate a metadata-only `GGMLMOEPACKv2` pack from dev route profiles.
- Use it to estimate whether selected low-byte expert payloads have enough
  general-prompt coverage to justify runtime v2 H2D/compute integration.

Inputs:

Only dev prompt route profiles were used:

```text
.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile/dev_france_regression/route-profile.csv
.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile/dev_japan_factual/route-profile.csv
.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile/dev_linear_equation/route-profile.csv
.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile/dev_mixed_summary/route-profile.csv
.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile/dev_photosynthesis_factual/route-profile.csv
.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile/dev_python_reverse/route-profile.csv
.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile/dev_zh_france/route-profile.csv
```

No held-out test route profiles were used.

Command:

```bash
RUN=.Agent/runs/20260707-gp45-v2-shadow-pack-from-dev-routes
mkdir -p "$RUN"
profiles=()
while IFS= read -r p; do profiles+=(--profile "$p"); done < <(
  find .Agent/runs/20260706-kimi-general-dev-baseline-n96-profile \
    -name route-profile.csv | sort
)
python3 .Agent/run-tools/kimi_make_v2_shadow_pack_from_routes.py \
  "${profiles[@]}" \
  --out-pack "$RUN/dev-hotset-iq1s-r055.shadow-v2.expert-pack" \
  --out-json "$RUN/dev-hotset-iq1s-r055.json" \
  --out-md "$RUN/dev-hotset-iq1s-r055.md" \
  --max-entries 4096 \
  --kind up,gate,down \
  --packed-type 24 \
  --packed-ratio 0.55
```

Result:

```text
candidate_entries=56896
selected_entries=4096
events=675560
event_coverage=0.359676
byte_coverage=0.366584
hybrid_byte_ratio=0.835037
```

Interpretation:

- A 4096-entry dev hotset at a hypothetical `0.55x` packed-byte ratio covers
  only `35.97%` of route events and `36.66%` of logical bytes.
- The whole workload hybrid byte ratio is still `0.835`.
- This is far too small to explain a jump from the current `~1.38 tok/s` level
  to `>5 tok/s` by selected low-byte payloads alone.
- The next useful measurement is either:
  - sweep larger metadata hotsets to find the coverage curve; or
  - run GP44 shadow profiling with this metadata pack on dev prompts to compare
    real decode observations against static route-profile estimates.

Validation:

- Local:
  - generator ran successfully;
  - `python3 -m py_compile` passed;
  - `git diff --check` passed.
- Remote:
  - generator ran successfully in
    `/root/lfz/tmp/vendor-kimi-speculative-gp33`;
  - output metrics matched local;
  - `git diff --check` passed.

Acceptance decision:

- Accepted as non-SOTA planning/instrumentation progress.
- No runtime behavior changed.
- No held-out prompts were used.
- No token-rate or output-quality claim is made.
