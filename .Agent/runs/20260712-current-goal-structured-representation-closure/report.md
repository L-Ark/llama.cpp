# Kimi Structured Representation Closure

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Source commit: `92d24ad42`

This is an evidence consolidation report for the active current goal. It does
not run new benchmarks, change runtime behavior, or claim SOTA.

## Goal

Close the remaining non-destructive structured-representation candidates that
could otherwise appear to be alternatives to the rejected blockwise lower-byte
family:

- shared down projector;
- slot/route-conditioned down surrogate;
- input+route full-MoE-output surrogate;
- output subspace / layer output subspace family.

The gate is the same as the current lower-byte admission:

- mean output rel-L2 `<=0.10` before runtime work;
- enough byte reduction to matter for `>2 tok/s`;
- generalized dev-only evidence before held-out testing.

## Evidence

| family | artifact | best result | size/cost note | decision |
|---|---|---:|---|---|
| shared down projector | `.Agent/runs/20260708-gp100-shared-down-projector-lambda-check/report.md` | mean rel-L2 `1.260756` | best mode `sum_h_abs_sq`, BF16 `4.92 GiB/60 layers` | reject |
| slot-concat down projector | `.Agent/runs/20260708-gp101-slot-concat-down-projector/report.md` | mean rel-L2 `1.252990` | BF16 `13.12 GiB/60 layers` | reject |
| route-conditioned down surrogate | `.Agent/runs/20260708-gp102-route-conditioned-down-surrogate/report.md` | mean rel-L2 `1.253345` | BF16 `4.92-8.20 GiB/60 layers` | reject |
| slot expert scalar down surrogate | `.Agent/runs/20260708-gp102-slot-expert-scalar-down-surrogate/report.md` | mean rel-L2 `1.252953` | BF16 `26.25 GiB/60 layers` | reject |
| input+route full-MoE-output surrogate | `.Agent/runs/20260708-gp105-input-route-moe-surrogate/report.md` | mean rel-L2 `0.885045` | tiny prototype memory, but error far above gate | reject |
| layer output subspace | `.Agent/runs/20260708-gp97-layer-output-subspace-oracle-smoke64/report.md` | LOO mean rel-L2 `1.194453` | small rank, poor generalization | reject |
| callstride output subspace | `.Agent/runs/20260708-gp88-callstride-output-subspace-oracle-512/report.md` | fused up/gate rank4 rel-L2 `0.636106`; down rank4 rel-L2 `0.555836` | ratio around `0.50x` but error too high | reject |

## Interpretation

These candidates are qualitatively different from simple blockwise quantization,
but they fail for the same practical reason: they do not preserve MoE output.

Important details:

- shared down projectors only attack down movement, while the current `>2 tok/s`
  byte bound requires all-role movement reduction;
- even the largest slot/route-conditioned down surrogates stay around
  `1.25` mean rel-L2, more than `12x` above the `0.10` gate;
- small full-MoE-output surrogates are memory-cheap but still around `0.885`
  mean rel-L2;
- output-subspace methods can approach useful byte ratios but do not generalize
  across prompts/layers well enough.

## Decision

Reject shared projector, small surrogate, and output-subspace families as the
next runtime implementation path.

Do not implement runtime kernels or pack layouts for these candidates unless a
new representation produces prompt-level output-error evidence near the
`<=0.10` gate.

## Next Action

The active decision tree is now narrower:

1. With explicit approval, run the guarded complete-model `i1-IQ1_S` smoke.
2. Without approval, design a genuinely new lower-byte representation whose
   first offline admission targets all-role effective bytes below `0.50x` and
   prompt-level output rel-L2 `<=0.10`.
3. Prediction/prefetch is closed for simple route IDs, route history, and route
   scores; a future predictor must use a real draft/router model or hidden/logit
   features and pass complete-batch admission.

## Reproduce

Each source artifact above already contains exact reproduction commands. No new
runtime command is required for this closure report.
