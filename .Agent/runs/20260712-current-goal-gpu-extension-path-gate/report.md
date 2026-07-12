# Kimi CPU/defer GPU-extension Path Gate

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Current HEAD: `8daad70a1`

This report advances the active goal after writing the plan. It decides whether
the next implementation cycle should copy the DeepSeek gate-fallback SOTA
mechanism directly, or whether Kimi is already past that bottleneck.

## Evidence Scope

Current HEAD contains no runtime source changes versus the profiled CPU/defer
audit commit:

```bash
git diff --name-only 97d1c177f..HEAD -- \
  'ggml/**' 'src/**' 'common/**' 'examples/**' 'tools/**' 'CMakeLists.txt'
```

Result: no files.

Therefore the existing N96 profile at
`.Agent/runs/20260712-current-head-cpudefer-n96-audit/` is still valid for the
current runtime. Changes after that audit are docs/tools/run-artifact only.

## Current Runtime Gate

The DeepSeek lesson is applicable as an architecture pattern:

- CPU/defer MoE remains the scheduler;
- GPU is used as an expert-cache extension;
- true CPU fallback is only the residual path when the extension does not accept
  a call.

For current Kimi, the broad fallback part is already solved in the profiled path:

| item | result |
|---|---:|
| true CPU fallback rows | `0` |
| runs with extension miss | `0` |
| up/gate accept rate | `1.000` |
| down accept rate | `1.000` |
| weighted decode | `651.099 ms/token` |
| weighted token rate | `1.536 tok/s` |
| up/gate wall | `441.220 ms/token` |
| up/gate exposed wait proxy | `252.286 ms/token` |
| down wall | `171.086 ms/token` |
| down stage | `~160 ms/token` |

Conclusion: do not spend the next cycle on another broad CPU fallback hook or a
gate-only port. Kimi's remaining problem is not "CPU compute instead of GPU";
it is demand expert movement on the GPU-extension path.

## Source Audit

Current runner defaults already enable the Kimi GPU-extension path:

- `.Agent/run-tools/kimi-general-prompt-repro.sh:66-87`
  enables stream batch, down batch, fused up/gate, mixed up/gate parallel stage,
  current-down overlap, Q4 down batch, prompt down batch, and prompt matmul ID
  batch.

The v2 lower-byte override state is not a general runtime replacement path yet:

- `ggml/src/ggml-cuda/moe_stream_batch.cu:4738-4748` enables v2 override
  preflight, partial split planning, control profiles, and shadow staging.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:4910-4913` prints
  `dispatch=disabled` for the v2 override manifest path.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:15185-15223` records up/gate v2
  coverage/preflight/partial-split/shadow data, but the actual up/gate staging
  at `15444-15567` still uses the current logical payload size and v1
  `expert_pack_lookup(...)`.

Down has a narrower real v2 path:

- `ggml/src/ggml-cuda/moe_stream_batch.cu:17634-17704` validates a full-cover,
  homogeneous, smaller v2 down payload.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:17818-17862` inserts v2 down slots and
  reads v2 entries.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:17940-17958` can read those entries
  through v2 io_uring.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:18094-18137` switches the actual down
  compute to `run_src0_type` and `run_nb01`.

This means lower-byte runtime work is asymmetric today:

- down-only full-cover v2 can be A/B tested if a quality-safe payload exists;
- up/gate lower-byte replacement would require new runtime dispatch, not just
  setting `GGML_MOE_EXPERT_PACK_V2_OVERRIDE=1`;
- down-only byte reduction is already bounded below the `>2 tok/s` target, so
  up/gate must eventually be addressed.

## Bottleneck Priority

The current N96 audit shows the priority order:

1. Up/gate demand-read exposure.
   - `252.286 ms/token` wait proxy is the largest removable pool.
   - Top layers by wall/wait include `14`, `28`, `29`, `54`, `30`, `32`.

2. Down staging.
   - About `171.086 ms/token` wall, mostly staging.
   - Top layers include `4`, `6`, `9`, `7`, `10`, `8`.
   - Current-down overlap already covers part of it, so new down work must prove
     it reduces endpoint decode time, not just role-local stage time.

3. RAM/page-cache replacement.
   - Earlier evidence showed decode can reclaim about `9 GiB` low-value file
     cache, but mid-decode GB-scale loading is not reproducible.
   - Any RAM tier must be prompt-agnostic, cold-start valid, and batch-friendly.

4. Lower-byte movement.
   - `>2 tok/s` requires roughly all-role `0.50x` effective movement.
   - Existing IQ1_S/Q2_K direct replacement failed output-error gates, especially
     up/gate.

## Next Implementation Gate

The next runtime A/B should be default-off and should not claim SOTA until it
passes generalized prompts, held-out validation, and the hard RAM/TTFT/quality
gates.

Recommended order:

1. Prompt-agnostic RAM/VRAM layer-role A/B.
   - Build candidate profiles from dev aggregate wait, not from one prompt.
   - First candidates:
     - full `blk.14 gate` or full `blk.14 up+gate`;
     - full `blk.4 down`;
     - one combined small candidate only if individual role tests pass.
   - Use pageable RAM tier first (`GGML_MOE_RAM_TIER_PIN=0`) because large pinned
     ranges caused first-batch spikes in earlier tests.
   - Acceptance is endpoint decode token rate, not hit rate.

2. Unified up/gate/down read scheduling smoke.
   - Only schedule after route is known.
   - Combine same-layer up/gate and known down jobs into one scheduler decision
     when it increases batch size without delaying compute.
   - Reject if it increases H2D/staging or reduces overlap.

3. v2/lower-byte preflight only.
   - Use existing v2 preflight/shadow tools to quantify full-cover calls and
     saved bytes.
   - Do not implement up/gate runtime dispatch until a candidate passes output
     error screening.
   - Down full-cover v2 may be tested separately, but cannot be declared a
     `>2 tok/s` route by itself.

4. Complete lower-bit model smoke only with explicit storage approval.
   - The `i1-IQ1_S` full model remains the only identified complete-model
     candidate near `0.50x`, but it requires cleanup/download approval.

## Stop Rules

- If a proposed change only increases hit rate but token rate or TTFT regresses,
  reject it.
- If it uses prompt-specific hotsets or layouts, reject it for generalized SOTA.
- If it increases TTFT by more than `20%` or host RAM reaches the cgroup limit,
  reject it.
- If France quality fails, reject it immediately.
- If a SOTA improvement is not reproduced from the pushed commit with exact
  commands, do not call it SOTA.

## Decision

Next work should target prompt-agnostic RAM/VRAM layer-role storage and
up/gate-demand-read exposure. The DeepSeek gate fallback idea has already been
absorbed into Kimi's current GPU-extension path; copying it again would not
address the current bottleneck.
