# GP26 speculative feasibility report

Timestamp: `2026-07-07T05:35:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Plan commit: `1c1cb74be`

## Scope

This is a dev-only feasibility pass. It does not inspect held-out test prompts
and does not claim a new SOTA.

The question is whether prediction/speculative decoding is plausible enough to
justify runtime work under the hard deployment constraints:

- 16 GB total host RAM including page cache;
- 32 GB RTX 5090;
- cold start;
- prompt-agnostic random user prompts;
- France semantic regression and general prompt quality gates;
- TTFT increase <= 20%;
- target above `5 tok/s`.

## Code paths

Detailed notes: `code-path-inspection.md`.

Summary:

- `examples/speculative` and `examples/speculative-simple` require a compatible
  draft model.
- No compatible draft model is currently present on the remote machine.
- `examples/lookup` and `common/ngram-*` can draft without an external model,
  but only from repeated ngram patterns in the prompt/context/history.

## Remote assets

Detailed inventory: `remote-asset-inventory.md`.

Summary:

- Present: Kimi IQ3_S GGUF shards.
- Missing: draft model.
- Missing: lower-byte full Kimi GGUF.

## Self-ngram lookup acceptance estimate

Analyzer:

```bash
python3 .Agent/run-tools/kimi_ngram_lookup_acceptance.py \
  --answers-glob '.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct/*/answer.txt' \
  --out-json .Agent/runs/20260707-gp26-speculative-feasibility/ngram-self-acceptance.json \
  --out-csv .Agent/runs/20260707-gp26-speculative-feasibility/ngram-self-acceptance.csv
```

Inputs:

- Dev answer files only from
  `.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct/*/answer.txt`.
- No held-out test files.
- Text-token approximation, not model-token runtime.

Best observed configuration:

- `ngram=1`
- `n_draft=8` or `16`
- prompts: `7`
- text tokens: `518`
- verify steps: `486`
- drafted tokens: `887` for `n_draft=8`
- accepted draft tokens: `26`
- accept rate: `2.93%`
- average accepted draft tokens per step: `0.0535`
- effective output tokens per step: `1.0658`
- max accepted run: `7`

Theoretical target gate:

- Current held-out SOTA is about `1.385 tok/s`.
- Reaching `5 tok/s` requires about `3.61x` end-to-end speedup.
- A zero-overhead speculative path would need roughly `3.6` accepted output
  tokens per expensive Kimi step.
- Real speculation has draft, verification, cache, prefetch, and rejection
  overhead, so the practical accepted-token requirement is higher.

Bounded implication:

- Self-ngram lookup at `1.0658` effective tokens/step would only raise
  `1.385 tok/s` to roughly `1.48 tok/s` before overhead.
- This is far below the `5 tok/s` target.

## Decision

- Reject no-extra-model self-ngram lookup as the primary path to `5 tok/s`.
- Do not implement a runtime lookup integration for Kimi SOTA yet.
- Model-based speculation remains potentially interesting, but it is currently
  blocked by missing draft-model assets and must first pass a separate
  RAM/VRAM/TTFT feasibility gate.
- The next useful optimization direction should target either:
  - a compatible, very small draft model with measured acceptance and resource
    impact; or
  - route/expert prediction that increases IO queue depth without needing high
    text-token acceptance; or
  - a more aggressive byte-reduction strategy than the AesSedai lower-byte
    selected hotsets measured in GP25.
