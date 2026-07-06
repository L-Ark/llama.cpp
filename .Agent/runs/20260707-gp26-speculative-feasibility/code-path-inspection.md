# GP26 code-path inspection

Timestamp: `2026-07-07T05:35:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Commit at inspection start: `1c1cb74be`

## Inspected paths

- `examples/speculative-simple/README.md`
- `examples/speculative-simple/speculative-simple.cpp`
- `examples/speculative/README.md`
- `examples/speculative/speculative.cpp`
- `examples/lookup/README.md`
- `examples/lookup/lookup.cpp`
- `examples/lookup/lookup-stats.cpp`
- `examples/lookahead/README.md`
- `common/ngram-cache.h`
- `common/ngram-cache.cpp`
- `common/ngram-map.cpp`

## Findings

- `examples/speculative-simple` requires `--model-draft` / `-md`.
- `examples/speculative` requires `--model-draft` / `-md` and validates that
  draft and target vocabularies match closely enough for speculation.
- `examples/lookup` does not require a draft model. It drafts from ngram caches
  built from prompt/context, optional dynamic cache, and optional static cache.
- `examples/lookup/lookup-stats.cpp` can estimate lookup acceptance, but still
  loads a llama model for tokenization. Running it directly against the full
  Kimi assets is heavier than needed for the first feasibility gate.
- `common/ngram-cache.*` uses ngram sizes `1..4` and drafts only when an
  empirical next-token choice satisfies sample-size and percentage thresholds.
- `common/ngram-map.cpp` has a simpler self-history ngram draft path. This is
  useful as a no-extra-model bound, but it only helps when the output repeats
  earlier token patterns.
- `examples/lookahead` is an independent example path. It is not currently wired
  into the Kimi vendor CLI/runtime path used for the SOTA measurements.

## Decision

- Model-based speculation cannot be evaluated until a compatible draft model is
  available and its RAM/VRAM/TTFT cost is bounded.
- No-extra-model lookup/self-ngram speculation should be screened with existing
  dev outputs before implementing a runtime integration.
