# Kimi lower-byte candidate gate

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`

## Purpose

Phase 5 established that `5 tok/s` requires structural byte reduction, not only
IO scheduling. This report gates the existing lower-byte candidates before any
runtime implementation.

Target from the accepted held-out SOTA bound:

- moved bytes: mean `4.39 GiB/token`;
- active expert footprint before cache: `8.10 GiB/token`;
- optimistic all-hit MoE floor: about `40 ms/token`;
- required moved-byte ratio after floor: median `0.39x`, worst `0.30x`.

Any primary candidate should therefore plausibly reach `0.30x-0.40x` total
moved bytes while preserving output quality.

## Candidate Results

| candidate | best byte ratio | error / risk | decision |
|---|---:|---|---|
| naive blockwise 1-bit re-encode | `0.367x-0.408x` on sampled gate tensors | rel L2 about `1.86-2.07`; too destructive | reject |
| naive blockwise 2-bit re-encode | `0.673x-0.878x` | rel L2 about `0.69-0.79`; too large and too many bytes | reject |
| D2MoE one-base residual | rank128 residual byte ratio can be `0.374x-0.502x` | residual norm still about `0.945`; base does not explain expert weights | reject |
| D2MoE clustered base, top32 sample | rank128 error down `0.500`, gate `0.306` at 16 clusters | one tensor already needs `448 MiB` bf16 bases; not scalable across layers/tensors | reject as primary |
| full external IQ1_S model | about `0.504x` full-model size | below current bytes but above `0.30x-0.40x`; quality risk; disk/runtime blocker | reject as direct 5 tok/s path |
| selected IQ1_S/v2 hotsets | selected hotsets can fit small budgets | coverage too low or prompt-specific; GP57 held-out overlay regressed badly | reject as prompt-general SOTA path |
| down activation block skipping | skip `19.8%` blocks at threshold `0.1`, `34.5%` at threshold `0.2` | mean relative error `0.210` / `0.368`; early layers are very inaccurate | reject as primary; possible late-layer down-only micro-optimization |

## Evidence

Naive re-encode:

- report: `.Agent/runs/20260707-gp11-quant-reencode-bound/report.md`
- sampled up/gate results:
  - `1-bit` can reach the target byte ratio, but rel L2 is about `1.8-2.1`;
  - `2-bit` has lower but still large rel L2 and cannot reach the required byte ratio.

D2MoE residual/base:

- reports:
  - `.Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-*.md`
  - `.Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-down-top32.md`
  - `.Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-gate-top32.md`
- one-base residual rank128 keeps residual norm around `0.945`;
- clustered-base improves some tensors, but at 16 clusters:
  - down rank128 error/weight is still `0.5002`;
  - gate rank128 error/weight is `0.3064`;
  - base memory for one tensor is `448 MiB`, which is not viable across all
    active layer/tensor groups under 32 GB VRAM.

Activation/down sparsity:

- reports:
  - `.Agent/runs/20260707-gp61-down-act-sparsity-multiprompt-n32/summary.md`
  - `.Agent/runs/20260707-gp62-shadow-error-multiprompt-n32/summary.md`
- exact zero ratio is effectively zero.
- threshold `0.1`:
  - skipped down blocks: `19.8%`;
  - mean relative output error: `0.2097`;
  - max relative error: `1.0`.
- threshold `0.2`:
  - skipped down blocks: `34.5%`;
  - mean relative output error: `0.3675`;
  - max relative error: `1.0`.
- late layers have low error but also low skip ratio. Early layers have high
  skip ratio but high error.
- Since up/gate is about `62%` of miss bytes in the held-out bound, down-only
  skipping cannot close the `5 tok/s` gap.

External lower-byte assets:

- reports:
  - `.Agent/runs/20260707-gp31-external-asset-refresh/report.md`
  - `.Agent/runs/20260707-gp34-iq1s-budgeted-hotset-bound/report.md`
  - `.Agent/runs/20260707-gp46-v2-hotset-sweep/report.md`
- IQ1_S full-model scale is about `0.504x`, above the target range.
- GP46 showed that selected lower-byte hotsets need near-complete coverage and
  about `0.276x` transfer-only ratio to reach `5 tok/s`, leaving no overhead
  margin.

## Decision

Do not implement runtime support for these rejected candidates as the next
primary token-rate optimization.

The next viable direction must reduce up/gate and down movement together, or
avoid moving full expert tensors entirely. Candidate classes that remain worth
investigating:

1. Activation-aware compressed compute, measured on real hidden states, not only
   weight reconstruction error.
2. Mixed late-layer down skipping as a local optimization only if it is gated by
   per-layer error and has no quality regression.
3. A stronger learned/predictive route mechanism only if it passes the
   cold-start audit and strict extra-read caps.

The immediate next experiment should be dev-only and should measure
activation-output error for compressed up/gate/down candidates before writing
any runtime path.
