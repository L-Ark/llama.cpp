# Kimi Non-Destructive Lower-Byte Closure

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `067f12271`

## Purpose

After refreshing complete-model lower-byte assets, close the remaining
non-destructive lower-byte candidates that can be evaluated without deleting old
packs or downloading a new full model.

This report does not change runtime behavior and does not claim SOTA.

## Artifacts

- `.Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/lower-byte-candidate-gate.json`
- `.Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/lower-byte-candidate-gate.md`
- `.Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/mixed-role-target050.json`
- `.Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/mixed-role-target050.md`

## Candidate Gate Refresh

The candidate gate aggregates previously committed, non-destructive evidence:

| candidate | evidence | decision |
|---|---|---|
| naive blockwise 1-bit re-encode | byte ratio `0.347x-0.488x`, rel-L2 `1.777-2.275` | reject |
| naive blockwise 2-bit re-encode | byte ratio `0.673x-0.878x`, rel-L2 `0.687-0.791` | reject |
| D2MoE clustered/base residual | 16-cluster rank128 error: down `0.5002`, gate `0.3064`; base `448 MiB` per tensor | reject as primary |
| selected IQ1_S hotsets | all-dev hybrid ratio `0.5205x`; small-budget worst ratio `0.6503x` | reject as direct primary path |
| selected v2 hotsets | best transfer-only row reports packed/hybrid `0.276x`, ideal `5.018 tok/s`, but lacks runtime/quality margin | reject as runtime path without stronger quality gate |
| down activation block skipping | threshold `0.1` skips `0.198` blocks with mean rel error `0.210`; threshold `0.2` skips `0.345` with mean rel error `0.368` | reject as primary |

The selected-v2 hotset line is not an acceptance result. It is an idealized
transfer-only row and is superseded by the real tiny v2 payload output-error
gate:

- `.Agent/runs/20260712-v2-compare-current-smoke/report.md`;
- real tiny IQ1_S/Q2_K overall mean rel-L2: `0.48436786`;
- up/gate IQ1_S mean rel-L2 range: about `0.52-0.56`;
- decision there: reject direct tiny IQ1_S/Q2_K runtime bridge.

## `0.50x` Mixed-Role Budget

The current short-term milestone needs roughly all-role `0.50x` effective
movement to approach `2 tok/s`. Therefore this refresh also reran the
mixed-role byte/error budget at:

- target global byte ratio: `<=0.50x`;
- target mean rel-L2 per component: `<=0.10`;
- source screen:
  `.Agent/runs/20260712-current-goal-lowbyte-freeze-screen/gp77-target050-summary.json`.

Result:

- combinations under byte target: `16`;
- passing combinations: `0`;
- best under-budget pair:
  - down: `down:aw_mse:bits1:block256`;
  - fused up/gate: `fused_up_gate:aw_mse_keep_input0p1:bits1:block256`;
  - global ratio: `0.3861x`;
  - down mean rel-L2: `0.499277`;
  - fused up/gate mean rel-L2: `0.598174`;
  - worst mean rel-L2: `0.598174`;
  - decision: reject.

This rejects the existing blockwise/activation-aware residual family even for
the `2 tok/s` byte target. The failure is still dominated by fused up/gate
output error, not storage size.

## Decision

Without explicit cleanup/download approval, no current non-destructive
lower-byte candidate should move to runtime implementation.

Rejected as next primary path:

- naive blockwise 1-bit or 2-bit re-encode;
- D2MoE clustered/base residual as currently screened;
- selected IQ1_S/v2 hotsets without a better output-error gate;
- down-only activation block skipping;
- existing GP68-GP77 blockwise/activation-aware residual family at `0.50x`.

Remaining actionable paths:

1. With explicit approval: run the guarded `i1-IQ1_S` complete-model dev smoke.
   This is still the only concrete same-model complete lower-byte candidate near
   the `0.50x` byte target.
2. Without approval: design a new representation family, not another runtime
   bridge for the rejected candidates. It must target all roles, especially
   fused up/gate, and must pass activation-output gates before runtime work.

## Reproduce

```bash
cd /root/lfz/llama.cpp-vendor-kimi
OUT=.Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure
mkdir -p "$OUT"

python3 .Agent/run-tools/kimi_lower_byte_candidate_gate.py \
  --out-json "$OUT/lower-byte-candidate-gate.json" \
  --out-md "$OUT/lower-byte-candidate-gate.md"

python3 .Agent/run-tools/kimi_mixed_role_byte_error_budget.py \
  --screen-json .Agent/runs/20260712-current-goal-lowbyte-freeze-screen/gp77-target050-summary.json \
  --out-json "$OUT/mixed-role-target050.json" \
  --out-md "$OUT/mixed-role-target050.md" \
  --target-global-ratio 0.50 \
  --target-mean-rel-l2 0.10
```
