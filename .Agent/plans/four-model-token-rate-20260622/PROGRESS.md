# DeepSeek V4 Flash Token-Rate Optimization Progress

Last updated: 2026-06-24 Asia/Shanghai

This file summarizes the Thinkless/WiCi run for `/root/lfz/ik_llama`, centered on the remote plan:

- Remote: `ssh -p 57990 root@111.237.107.89 -L 8080:localhost:8080`
- Repository: `/root/lfz/ik_llama`
- Plan: `/root/lfz/ik_llama/.Agent/plans/four-model-token-rate-20260622/deepseek-v4-plan.md`
- Model: `/root/lfz/models/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-00001-of-00001.gguf`
- Hard safety: no `git push`
- Validation discipline: `MemoryMax=16G`, `MemorySwapMax=0`, deterministic checks before promotion, and plan/log updates for every accepted or rejected route.

## Current State

The current quality-valid promoted frontier is A91 unless the active A92 audit finds instability.

- Accepted env:
  `GGML_DEEPSEEK4_ENABLE_CUDA_F8_DENSE=1 GGML_DEEPSEEK4_CUDA_F8_DENSE_ATTN_SAFE=1 GGML_DEEPSEEK4_CUDA_F8_DENSE_ALLOW_CLASSES=attn,ffn_up`
- Accepted flags:
  `--defer-experts --fit -ngl 999 -c 512 -ub 1 -t 20 -tb 20 -no-fa`
- A91 full-repeat result:
  - A80/A88 same-session baseline p50: `5.19 tok/s`
  - A91 `attn,ffn_up` p50: `6.38 tok/s`
  - A91 worst repeat: `6.34 tok/s`
  - Delta: `+1.19 tok/s`
  - Graph splits: `291 -> 248`
- Active step:
  - S33/A92 is running a post-A91 stability, correctness, and perf audit.
- Next planned step:
  - S34/A93 will only run after A92 confirms A91 stability. It will test narrow F8 placement beyond `attn,ffn_up` with output/logit correctness gates.

The earlier A64/A66 path reached `9.x tok/s`, but it is invalid for quality because deterministic output checks showed corruption. It remains useful as diagnostic evidence that broad F8 dense CUDA placement is fast, but it cannot be counted as a correct model result.

## Goal History

### v1: Continue Remote Plan

The initial requirement was to log into the remote server, read `deepseek-v4-plan.md`, and continue the existing task rather than restart. The target repository and plan were fixed by the planner:

- Repo: `/root/lfz/ik_llama`
- Plan: `.Agent/plans/four-model-token-rate-20260622/deepseek-v4-plan.md`
- Branch: `deepseek-v4-flash`

Validation required 16 GB cgroup benchmarking, log paths, metrics, plan updates, and local commits for confirmed progress. `git push` was forbidden by WiCi constraints.

### v2: Correctness Audit Added

After the broad A64/A66 F8 dense CUDA route produced `9.x tok/s`, Chat inspected output logs and found visible corruption on simple prompts such as `The capital of France is`. The goal was updated to require correctness auditing before accepting those results.

Effect:

- A64/A66 broad dense F8 throughput was downgraded to invalid-for-quality.
- A80/A88 became the accepted quality-valid frontier.
- Future work had to preserve deterministic output correctness, not just exit cleanly.

### v3-v5: Reopen Optimization Under Correctness Constraints

The user asked to continue improving token rate under correctness and at least 4-bit effective quantization quality. This reopened work from the A90 terminal frontier.

New constraints:

- Use A80/A88, then A91/A92 if stable, as the correctness baseline.
- Do not use broad unsafe dense F8 placement.
- Prefer narrow class-by-class or per-tensor F8 placement.
- Use deterministic output and preferably first-token/top-logit comparison before promotion.
- Keep the 16 GB RAM cap.
- Keep rollback available and record every attempt in `deepseek-v4-plan.md`.

## Plan Update History

The local `PLAN.md` evolved as a sequence of executor steps. The high-level phases were:

| Phase | Steps | Purpose | Outcome |
| --- | --- | --- | --- |
| Remote setup and baseline | S1-S2 | Establish server/repo context and record prior A63 failure | Completed |
| Broad F8 dense CUDA probe | S3-S7 | Add gated CUDA support for `F8_E4M3_B128` dense conversion and validate A64 | Fast but later invalidated for quality |
| Post-A64 profiling | S8-S15 | Stability, perf, MXFP4/MoE instrumentation, direct IQK probes | Found CPU MXFP4/OpenMP bottleneck, but no promoted route |
| Correctness audit | S16-S20 | Audit A64 output and localize unsafe F8 dense placement | Broad placement invalidated |
| Quality-valid F8 repair | S21-S23 | Validate attention-only F8 dense placement and profile remaining bottleneck | A80/A88 accepted, about `5.3-5.5 tok/s` |
| MoE/OpenMP route exploration | S24-S28 | Runtime sweep, source scheduling, helper partitioning, region granularity, persistent/coalesced helper team | Diagnostic or unpromoted |
| Terminal-frontier record | S29-S31 | Consolidate A88, audit residual routes, record A90 terminal receipt | Blocked until new steering |
| Correctness-preserving reopen | S32-S34 | A91 narrow placement expansion, A92 hardening, A93 planned next probe | A91 promoted; A92 active; A93 pending |

## Performance Trajectory

### Early Valid Baseline

The first reliable 16 GB baseline family reached about `1.8-1.9 tok/s`.

- A31 `-no-fa -t 20 -tb 20` full `n256`:
  - `1.90 / 1.91 / 1.92 tok/s`
  - p50: `1.91 tok/s`
  - worst: `1.90 tok/s`
  - graph splits: `1237`

This stage mainly established a stable run shape:

- `--fit`
- `--defer-experts`
- `-ub 1`
- `-t 20 -tb 20`
- `-no-fa`
- 16 GB cgroup discipline

### A64/A66 Broad Dense F8 Jump

A64 added a gated CUDA dense conversion path for `F8_E4M3_B128`, allowing broad dense `GGML_OP_MUL_MAT` work to stay on CUDA. It reduced graph splits sharply.

- A31/A59 graph splits: `1237`
- A64/A66 graph splits: `76`
- A64 full repeats:
  - `9.22 tok/s`
  - `9.84 tok/s`
- A66 stability run:
  - `9.92 tok/s`

However, later correctness audit found corrupted output. This means the speedup was real as a throughput measurement but not valid as a model-quality result.

Decision:

- Keep A64/A66 as diagnostic evidence.
- Do not count it as accepted.
- Do not use broad unsafe dense F8 placement for future promotion.

### A80/A88 Quality-Valid Frontier

The repair path narrowed F8 CUDA placement to attention-safe classes only.

- Accepted env:
  `GGML_DEEPSEEK4_ENABLE_CUDA_F8_DENSE=1 GGML_DEEPSEEK4_CUDA_F8_DENSE_ATTN_SAFE=1 GGML_DEEPSEEK4_CUDA_F8_DENSE_ALLOW_CLASS=attn`
- A88 smoke prompts:
  - baseline graph splits: `1237`, eval `1.82-1.87 tok/s`
  - attention-safe graph splits: `291`, eval `5.00-5.13 tok/s`
- A88 accepted n256 repeats:
  - `5.34 tok/s`
  - `5.50 tok/s`

This became the accepted quality-valid frontier before R2/R3 reopened optimization.

### A91 Quality-Safe Expansion

A91 added a default-off combined class allowlist:

`GGML_DEEPSEEK4_CUDA_F8_DENSE_ALLOW_CLASSES=attn,<extra>`

Candidates tested:

- `attn`
- `attn,attn_out`
- `attn,attn_qkv`
- `attn,ffn_gate`
- `attn,ffn_up`

Correctness smoke:

- 15 candidate/prompt runs exited `0`.
- No CUDA/assert/shape/NaN/Inf failures were recorded.
- Visible output stayed coherent relative to the accepted attention-only baseline.

Throughput:

- Smoke n64:
  - accepted_attn: `5.11 / 4.95 / 4.94`
  - attn_plus_attn_out: `5.19 / 5.10 / 5.30`
  - attn_plus_attn_qkv: `5.11 / 5.22 / 5.07`
  - attn_plus_ffn_gate: `6.27 / 5.83 / 6.02`
  - attn_plus_ffn_up: `5.97 / 6.04 / 6.12`
- Quick n128:
  - accepted_attn: `5.11 tok/s`
  - attn_plus_ffn_gate: `6.10 tok/s`
  - attn_plus_ffn_up: `6.12 tok/s`
- Full n256 paired repeats:
  - accepted_attn: `5.02 / 5.36 tok/s`
  - attn_plus_ffn_up: `6.42 / 6.34 tok/s`

Decision:

- Promote `attn,ffn_up`.
- A91 p50: `6.38 tok/s`
- A91 worst: `6.34 tok/s`
- Result commit: `9e2d2c0a`
- Push: not run, blocked by no-push constraint.

## Correctness And Quality Decisions

Correctness became the central acceptance gate after A64/A66.

Important decisions:

- Exiting `0` and having no NaN/OOM is not sufficient.
- Visible deterministic output must not show broad-F8 corruption patterns.
- Future candidates should include output checks and preferably logit/top-token checks.
- Broad dense F8 placement is forbidden for promotion.
- Quality floor is at least 4-bit effective quantization. No sub-4-bit route, expert skipping, top-k reduction, or math-changing shortcut is acceptable unless explicitly re-scoped and validated.

The current model remains mixed FP4/FP8:

- Dense F8 tensors: `GGML_TYPE_F8_E4M3_B128`
- Expert/MoE hot path: MXFP4/Q8 helper route
- Accepted work keeps this quality profile and does not reduce quantization below the stated floor.

## Bottleneck Findings

After the safe dense F8 placements, the remaining bottleneck shifted to MoE/expert paths.

Key findings:

- A67 identified CPU MXFP4 expert/MoE matmul plus OpenMP/libgomp overhead after broad A64.
- A83-A87 explored thread/runtime knobs and source-level OpenMP/helper routes under the accepted A80 path.
- A86 measured many tiny helper regions:
  - representative helper p50 around `123-132 us`
  - callsite granularity and graph/barrier overhead were likely important
- A87 reduced direct IQK MoE calls from six active-expert calls to one coalesced call at the instrumented site, but did not improve throughput enough to promote.

Retired routes:

- Broad OpenMP thread retuning.
- Lower-level OpenMP partitioning variations.
- Existing coalesced/persistent helper-team probe as implemented.

Remaining plausible routes:

- Narrow F8 placement expansion beyond `attn,ffn_up`.
- Per-tensor allowlists when whole classes are unsafe.
- Deeper MXFP4 helper optimization only if a concrete function target and correctness oracle are defined.
- Graph-level barrier/copy reduction with explicit validation.

## Continuation Decisions

### Initial Handoff

Chat handed the initial remote task to planner/executor because it required SSH, remote code inspection, benchmark execution, source edits, validation, and commits.

### Terminal Frontier At A90

After A88 and A89, the plan concluded there was no non-speculative route remaining under the then-current scope.

A90 recorded:

- Source was clean.
- A80/A88 was the accepted quality-valid frontier.
- A64/A66 was invalid-for-quality.
- A83-A87 routes were retired or unpromoted.
- Further work required new user steering or new execution evidence.

Continuation decision:

- Stop autonomous expansion until the user provided a new concrete direction.

### User Reopened Correctness-Preserving Optimization

The user asked to continue improving token rate under correctness constraints. This added R2.

Continuation decision:

- Reopen from A80/A88, not A64/A66.
- Require correctness-preserving candidates.
- Start with narrow F8 class expansion.

This led to A91, which promoted `attn,ffn_up`.

### User Added 4-Bit Quality Floor And Further Planning

The user asked to think further and continue planning while keeping correctness and at least 4-bit quantization quality. This added R3.

Continuation decision:

- Harden A91 first with A92.
- Then continue with A93 only if A92 confirms stability.
- Prefer class-by-class and per-tensor placement expansion.
- Defer MXFP4 CPU-path work unless perf gives a concrete target.

### Current Continuation Status

S33/A92 is active.

S34/A93 is planned but gated by A92:

- If A92 confirms A91 stability, A93 can test narrow expansions such as `attn,ffn_up,attn_out`, `attn,ffn_up,attn_qkv`, or `attn,ffn_up,ffn_gate`.
- If A92 finds instability, A93 should be skipped and the plan should fall back to A80/A88 or repair A91.

## Safety Notes

- WiCi forbids `git push`.
- One earlier incident during A86 accidentally executed `git push` because an unquoted here-document expanded markdown command text. The plan recorded this as a safety note. Later steps explicitly warn to avoid unquoted here-docs and shell expansion of plan prose.
- Subsequent steps record `pushed_commit: n/a` and keep changes local.
- Temporary source probes must save diffs, revert on failure, rebuild default `llama-cli`, and confirm clean tracked source state.

## Artifact Map

Primary remote plan:

- `/root/lfz/ik_llama/.Agent/plans/four-model-token-rate-20260622/deepseek-v4-plan.md`

Important run directories:

- A64: `/root/lfz/runs/ik_llama/deepseek-v4-a64-cuda-f8-b128-to-f16-convert`
- A66: `/root/lfz/runs/ik_llama/deepseek-v4-a66-post-a64-n256-r3`
- A86: `/root/lfz/runs/ik_llama/deepseek-v4-a86-openmp-region-granularity`
- A87: `/root/lfz/runs/ik_llama/deepseek-v4-a87-persistent-moe-helper-team`
- A88: `/root/lfz/runs/ik_llama/deepseek-v4-a88-accepted-frontier-consolidation`
- A89: `/root/lfz/runs/ik_llama/deepseek-v4-a89-residual-route-audit`
- A90: `/root/lfz/runs/ik_llama/deepseek-v4-a90-terminal-frontier-receipt`
- A91: `/root/lfz/runs/ik_llama/deepseek-v4-a91-quality-safe-f8-placement-expansion`
- A92: `/root/lfz/runs/ik_llama/deepseek-v4-a92-post-a91-stability-bottleneck-audit`
- A93 planned: `/root/lfz/runs/ik_llama/deepseek-v4-a93-narrow-f8-placement-beyond-a91`

Local Thinkless state:

- `.thinkless1/GOAL.md`
- `.thinkless1/PLAN.md`
- `.thinkless1/events.jsonl`
- `.thinkless1/goal-interrogations.jsonl`
- `.thinkless1/artifacts/`

## Current Recommended Reading Order

For a new agent or human reviewer:

1. Read this `PROGRESS.md`.
2. Read `.thinkless1/GOAL.md` for the current user-facing contract.
3. Read `.thinkless1/PLAN.md` for active S33/S34 execution details.
4. On the remote, read the tail of `deepseek-v4-plan.md` from A88 onward.
5. Inspect A91 and A92 run directories before changing any source.

## Open Questions

- Does A92 confirm A91 stability with expanded correctness prompts and paired full repeats?
- Does A92 perf identify a concrete post-A91 bottleneck target?
- Can A93 safely add another F8 class or per-tensor placement without reproducing A64-style corruption?
- Is a built-in logit/top-token export available in `llama-cli`, or does A93 need a temporary gated first-token logit diagnostic?

## Bottom Line

The run moved from a quality-valid `1.91 tok/s` baseline to a correctness-preserving A91 frontier of `6.38 tok/s` p50 under the 16 GB cgroup. A transient `9.x tok/s` path was found but rejected because it corrupted output. The active work is now A92 hardening of A91, followed by A93 narrow F8 placement expansion if A92 confirms stability.
