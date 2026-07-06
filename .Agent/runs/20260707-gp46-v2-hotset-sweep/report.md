# GP46 v2 low-byte hotset size sweep

Timestamp: `2026-07-07T06:52:10+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit:

```text
d52f36c8e tools: add kimi v2 shadow pack generator
```

Purpose:

- Determine whether selected low-byte v2 payloads can plausibly reach the
  `>5 tok/s` target by byte reduction alone.
- Sweep selected hotset size and hypothetical packed-byte ratios using only
  committed dev N96 route profiles.

Inputs:

- Seven committed dev route-profile files from:
  `.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile`.
- No held-out test prompt profiles were used.

Command:

```bash
RUN=.Agent/runs/20260707-gp46-v2-hotset-sweep
mkdir -p "$RUN"
profiles=()
while IFS= read -r p; do profiles+=(--profile "$p"); done < <(
  find .Agent/runs/20260706-kimi-general-dev-baseline-n96-profile \
    -name route-profile.csv | sort
)
python3 .Agent/run-tools/kimi_sweep_v2_shadow_hotset.py \
  "${profiles[@]}" \
  --out-csv "$RUN/sweep.csv" \
  --out-json "$RUN/sweep.json" \
  --out-md "$RUN/sweep.md" \
  --sizes 512,1024,2048,4096,8192,16384,32768,56896 \
  --ratios 0.55,0.35,0.276 \
  --baseline-tps 1.385 \
  --target-tps 5.0
```

Headline result:

```text
candidate_entries=56896
rows=24
best_packed_ratio=0.276000
best_selected_entries=56896
best_hybrid_byte_ratio=0.276000
best_ideal_transfer_only_tps=5.018114
best_target_met_transfer_only=1
```

Important rows:

| packed ratio | selected entries | hybrid byte ratio | ideal transfer-only tok/s | target met |
| ---: | ---: | ---: | ---: | --- |
| 0.55 | 4096 | 0.8350 | 1.659 | no |
| 0.55 | 32768 | 0.5921 | 2.339 | no |
| 0.55 | 56896 | 0.5500 | 2.518 | no |
| 0.35 | 56896 | 0.3500 | 3.957 | no |
| 0.276 | 32768 | 0.3438 | 4.028 | no |
| 0.276 | 56896 | 0.2760 | 5.018 | yes |

Interpretation:

- With the realistic selected-IQ1_S byte-ratio estimate around `0.52x-0.61x`,
  even covering every dev candidate entry only gives an ideal transfer-only
  bound near `2.5 tok/s`.
- A `5 tok/s` pure-transfer bound requires approximately `0.276x` total bytes
  and nearly complete coverage. Real decode would require even lower bytes
  because not all time is expert transfer.
- Therefore, selected low-byte v2 payloads alone are not a sufficient next
  optimization path for the `5 tok/s` target. They may still be useful as one
  component, but the next major direction should target scheduling/prediction,
  queue continuity, or reducing per-token expert demand rather than only
  converting selected payloads to IQ1_S.

Validation:

- Local:
  - sweep script ran successfully;
  - `python3 -m py_compile` passed;
  - `git diff --check` passed.
- Remote:
  - sweep script ran successfully in
    `/root/lfz/tmp/vendor-kimi-speculative-gp33`;
  - headline metrics matched local;
  - `python3 -m py_compile` passed;
  - `git diff --check` passed.

Acceptance decision:

- Accepted as non-SOTA planning evidence.
- No runtime behavior changed.
- No held-out prompts were used.
- No token-rate or output-quality claim is made.
