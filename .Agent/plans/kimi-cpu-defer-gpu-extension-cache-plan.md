# Kimi CPU/defer GPU-extension cache optimization plan

Date: 2026-07-10
Branch: `vendor/kimi-deepseek-41d205-additive`
Parent plan: `.Agent/plans/kimi-token-rate-16gb-optimization-plan.md`

## Goal

Optimize Kimi under the actual runtime architecture:

- CPU/defer MoE remains the main scheduling path.
- GPU is used as an expert-cache/compute extension for that CPU/defer path.
- The next gains should come from better gate/up/down residency and transfer scheduling, not from assuming a GPU-primary backend with CPU fallback.

Target environment:

- Host RAM hard limit: `<16 GB`, including page cache, pinned buffers, mmap pages, process memory, kernel cgroup memory, and helpers.
- GPU: single RTX 5090-class 32 GB card.
- Runs: cold start, `MemorySwapMax=0`.
- Workload: general/random user prompts, not prompt-specific France tuning.
- Quality: coherent semantic output; `Please introduce France in a short paragraph.` remains a mandatory regression prompt.
- TTFT: must not rise more than `20%` versus paired baseline.
- Reproducibility: accepted improvements must be committed and pushed with exact command/env/run artifacts/rollback.

Primary performance goal:

- Improve stable held-out general-prompt token rate, prioritizing minimum token rate across the held-out set.
- Long-term target remains `>5 tok/s`, but every step must be justified by current bottleneck evidence and accepted only if reproducible.

## Architecture hypothesis

DeepSeek's large gate-hotpool gain came from a specific shape:

- `n_cpu_moe=40` placed many MoE experts on the CPU/defer path.
- Gate was on a slow CPU/defer path before the GPU extension caught it.
- Moving gate hot experts to VRAM and computing gate on GPU converted a CPU-bound critical path into a GPU-resident path.

Kimi is different:

- Kimi decode already routes gate/up/down through CPU/defer orchestration into GPU extension paths.
- Existing Kimi SOTA uses expert pack, io_uring, pinned staging, VRAM cache, current-down overlap, Q4 down batch, and RAM tier.
- The dominant Kimi symptom is not raw gate CPU compute; it is expert movement and exposed `io_uring_wait`, especially when up/gate/down misses create critical-path stalls.

Therefore, the useful transferable idea is not "put all gate in VRAM" by itself. The useful idea is:

> Treat CPU/defer MoE as the scheduler and make the GPU extension more complete, better cached, and less IO-stalled for gate/up/down as a group.

## Current baseline evidence to carry forward

Recent merge branch evidence:

- Branch: `vendor/kimi-deepseek-41d205-additive`
- Commit: `4a20433748a6efa53f1d93d305906abfd5cb5d0e`
- Code merge commit: `9cb5e745468868a844d5abf4ce1d8ba46818d309`
- Kimi b4ef control with correct CUDA batch defines:
  - run 1: `/root/lfz/runs/vendor-kimi-token-rate/20260710-b4ef-control-batch-france-n96-112525`
  - token rate `1.91 tok/s`, `iouring_wait_us=46594173`
  - run 2: `/root/lfz/runs/vendor-kimi-token-rate/20260710-b4ef-control2-batch-france-n96-113502`
  - token rate `1.78 tok/s`, `iouring_wait_us=49347929`
- Merge branch run:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-merge-control2-france-n96-112932`
  - token rate `1.89 tok/s`, `iouring_wait_us=47146856`

Interpretation:

- France N96 single-run token rate is currently noisy because `io_uring_wait` varies materially.
- Merge branch preserves Kimi semantic output, slot counts, hit rates, pack entries, and batch histograms.
- Future SOTA claims must use paired baseline/candidate runs and report variance, not a single run.

## Non-negotiable acceptance gates

An optimization is accepted only if all are true:

- Source state is clean or only contains explicitly ignored/untracked measurement artifacts.
- Candidate is committed and pushed immediately once accepted.
- Reproduction command, env, prompt set, model path, pack paths, profiles, cgroup settings, run path, and rollback commit are recorded.
- Cold-start paired baseline and candidate are run close together on the same machine state.
- Host RAM peak stays under `15900000000` bytes with `MemorySwapMax=0`.
- TTFT ratio is `<=1.20` versus paired baseline.
- France prompt quality passes.
- Held-out general test-set quality passes.
- Token-rate improvement is visible on held-out test metrics, especially min token rate.
- CPU fallback/direct-read integrity checks do not regress.
- If performance, TTFT, RAM, or quality regresses, the change is reverted or left default-off and documented as rejected.

## Phase 0: Paired general baseline and bottleneck profile

Purpose: avoid optimizing France-specific or IO-noise artifacts.

Actions:

1. Re-run the existing dev/test prompt split from `.Agent/evals/` if present.
2. If the split is missing, create `.Agent/evals/kimi-general-dev-prompts.jsonl` and `.Agent/evals/kimi-general-test-prompts.jsonl`.
3. Do not use held-out test prompts for route/hotset/cache design.
4. Run cold-start N96 paired baseline on dev prompts and a final held-out check.
5. Record per-prompt answer text, quality verdict, TTFT, decode time, token rate, RAM peak, file/anon/kernel breakdown, VRAM slots/hit rates, expert-pack bytes, `iouring_wait_us`, pinned staging stats, current-down overlap stats, and CPU fallback by tensor type if counters are available.

Output:

- `.Agent/runs/<date>-kimi-cpu-defer-gpu-extension-baseline/report.md`
- A table with dev and held-out test mean/median/min token rate.

Acceptance:

- No code behavior changes in Phase 0 unless counters are default-off.
- Baseline must establish current variance band before comparing candidates.

## Phase 1: Determine whether Kimi still has a gate-specific bottleneck

Question:

- Does Kimi still have exposed gate miss/compute time comparable to the old DeepSeek gate CPU/defer bottleneck?

Method:

1. Add or reuse default-off counters for per-layer/role exposed wait: gate read wait, up read wait, down read wait, gate/up/down compute time, role-specific cache hit/miss, and whether a miss is on the current token critical path.
2. Run n32 dev profile first, then n96 only if overhead is acceptable.
3. Rank `(layer, role)` by exposed wait, not by hit rate alone.

Decision:

- If gate exposed wait is small, do not spend the next cycle on gate-only VRAM hotpool.
- If a few gate layers dominate exposed wait, test a small gate-only hotset A/B, default-off.
- If up/gate jointly dominate, proceed to Phase 2.

## Phase 2: Role-aware joint up/gate/down VRAM cache allocation

Hypothesis:

- Kimi stalls when any role needed by the current MoE layer misses and blocks the CPU/defer GPU extension.
- A cache allocation that maximizes joint completion probability can beat separate hotness-only role hit rates.

Experiments:

1. Baseline current split: down slots around `723`; upgate slots around `1735`.
2. Candidate A: shift more VRAM to up/gate while shrinking down.
3. Candidate B: align up/gate/down for the same expert IDs in high-impact layers.
4. Candidate C: layer-priority cache where low-coverage/high-wait layers receive complete or near-complete role coverage.

Metrics:

- Per-prompt min/median/mean token rate.
- Exposed `iouring_wait_us` by role.
- Cache hit rate by role and by layer.
- Number of tokens/layers where all required role experts are resident.
- TTFT and RAM/VRAM usage.

Acceptance:

- Held-out min token rate improves without TTFT/RAM/quality regression.
- Improvement must survive paired baseline rerun.

## Phase 3: Explicit RAM tier for second-hot experts

Hypothesis:

- Decode page cache is not a controlled expert cache.
- Replacing low-value file-backed pages with explicit RAM-resident expert slabs can reduce SSD wait, but only if it reduces exposed wait rather than just shifting bytes from SSD to H2D.

Experiments:

1. Identify decode-time low-value RAM: file-backed pages not touched after prompt, GGUF expert pages caused by fallback/refault, and dense/attention pages already resident in VRAM and not needed by decode CPU paths.
2. Define RAM candidate sets from dev prompts only: high exposed-wait layer gate+up full or partial layer, high exposed-wait up/gate/down grouped experts, and second-hot experts not worth VRAM but frequent enough to avoid SSD.
3. Test RAM layouts: 1.8 GiB current gate layer reference, 3-5 GiB structured RAM tier, and larger tier only if TTFT stays within `+20%` and RAM remains below cgroup.

Metrics:

- RAM hit count and RAM H2D bytes.
- SSD `iouring_wait_us` reduction.
- Decode time change.
- TTFT preload cost.
- Page-cache/file breakdown before prompt, after prompt, during decode.


Acceptance:

- Accept only if exposed wait and token rate improve on held-out prompts.
- Reject if hit rate improves but decode slows, TTFT exceeds gate, or RAM pressure/refault increases.

## Phase 4: Reduce io_uring queue starvation without unsafe prediction

Hypothesis:

- Pure IO bench can reach higher bandwidth, but runtime exposes small bursts.
- Larger batches help only if they do not add stale/unused reads or block demand.

Experiments:

1. Same-layer aggressive co-submit: enqueue known up/gate/down misses for the layer as soon as routing is available; demand priority remains first; no speculative future-layer reads initially.
2. Cross-layer safe scheduler shadow mode: track duplicate/in-flight opportunities without changing behavior.
3. If shadow counters show useful opportunities, enable default-off scheduler where demand can steal/wait on matching in-flight prefetch, stale prefetch is capped/pruned, and extra read ratio is bounded.

Acceptance:

- `iouring_wait_us` drops more than candidate overhead.
- Token rate improves on paired held-out runs.
- Extra bytes and TTFT remain bounded.

## Phase 5: Commit and push protocol

For every accepted improvement:

1. Update this plan before implementation with the exact hypothesis and expected upper bound.
2. Implement default-off if risk is nontrivial.
3. Run quick n32 dev check.
4. Run paired n96 dev check.
5. Run held-out test only after candidate survives dev.
6. Commit with full body: improvement size, env, commands, prompt split, output quality, TTFT/RAM/VRAM/IO metrics, and rollback point.
7. Push to the active `vendor/*kimi*` branch.

Rejected experiments remain documented with run paths and reason for rejection.
