# Kimi CPU/defer GPU-extension cache optimization plan

Date: 2026-07-10
Branch: `vendor/kimi-deepseek-41d205-additive`
Parent plan: `.Agent/plans/kimi-token-rate-16gb-optimization-plan.md`

## Current goal and next plan: CPU/defer GPU-extension for Kimi

Goal:

> On `vendor/kimi-deepseek-41d205-additive`, determine whether the DeepSeek-style idea of keeping CPU/defer MoE as the scheduler while expanding the GPU expert-cache/compute extension can improve Kimi's **general-prompt** decode speed. The immediate target is a reproducible held-out improvement toward `2 tok/s`; the long-term target remains stable `>5 tok/s` on random user prompts with 16 GB host RAM and one 32 GB RTX 5090-class GPU.

Why this is useful for Kimi:

- The transferable idea is **not** "copy DeepSeek gate-only hotpool".
- The useful part is that CPU/defer can remain the control path while GPU handles more expert residency, transfer, and compute.
- Current Kimi evidence shows the bottleneck is mostly exposed expert movement wait: `io_uring_wait`, staging/H2D, small runtime batches, and incomplete overlap.
- Recent profiling shows decode CPU fallback is already near zero, so the next gain must reduce exposed transfer/staging wait rather than only moving more work away from CPU.

Non-negotiable acceptance gates:

- Cold start only.
- Host RAM peak `<15900000000` bytes, including page cache, mmap pages, pinned memory, process memory, helpers, and kernel/cgroup accounting.
- TTFT `<=1.20x` paired baseline.
- France regression prompt must remain semantically correct: `Please introduce France in a short paragraph.`
- Held-out general prompts must pass quality and are not allowed for tuning.
- A result is SOTA only after paired baseline/candidate N96 dev plus N96 held-out validation.
- Accepted improvements must be committed and pushed immediately with exact env, commands, prompt set, run path, RAM/VRAM metrics, TTFT, token-rate delta, quality result, and rollback commit.
- If speed, quality, RAM, or TTFT regresses, revert or keep the code default-off and document the rejection.

Immediate execution plan:

1. Finish the current `blk.1 gate` profile-only VRAM-protection N96 paired A/B.
   - Candidate env must include `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`.
   - Parse per-prompt token rate, TTFT, decode time, RAM peak, `iouring_wait_us`, SSD bytes, upgate/down hit rates, and profile preload events.
   - Accept only if N96 dev improves without quality/RAM/TTFT regression; otherwise reject without held-out.

2. If the `blk.1 gate` profile-only A/B passes N96 dev, run the same config on N96 held-out.
   - Do not inspect held-out traces for tuning.
   - If held-out improves, promote the profile to tracked `.Agent/profiles/`, commit, and push as SOTA with full reproduction body.
   - If held-out does not improve, document rejection and keep only default-off infrastructure if useful.

3. If `blk.1 gate` fails, stop adding isolated slabs and move to the next higher-upside path:
   - same-layer aggressive co-submit of up/gate/down misses after routing;
   - layer/role-aware RAM cache replacing low-value decode-time file cache;
   - pack-layout changes that increase batchable contiguous reads instead of fragmenting SSD/RAM traffic.

4. Before every implementation step, update this plan with:
   - current bottleneck in seconds/token or aggregate N96 seconds;
   - theoretical upper bound from removable `iouring_wait`, staging/H2D bytes, or compute time;
   - exact experiment command and rollback point.

5. After every experiment, record:
   - per-prompt metrics and answer quality;
   - mean/median/min token rate;
   - TTFT ratio;
   - host RAM and decode page-cache/file/anon distribution;
   - VRAM expert-cache hit/miss/preload/pinned stats;
   - `iouring_wait_us`, SSD bytes, RAM->VRAM H2D bytes, staging wall, and CPU fallback counters;
   - accept/reject decision.

## 2026-07-10 goal and execution plan refresh

Active goal:

> On the current Kimi vendor branch, validate and implement the transferable part of the DeepSeek SOTA idea: CPU/defer MoE remains the scheduler, while GPU becomes a more complete expert-cache/compute extension for Kimi gate/up/down. The optimization target is a reproducible general-prompt token-rate gain under a strict 16 GB host RAM cap and a single 32 GB RTX 5090-class GPU, without losing semantic quality or increasing TTFT by more than 20%.

Current technical judgment:

- The DeepSeek gate-hotpool result is relevant to Kimi at the architecture level: the CPU/defer path can call a GPU expert-cache extension instead of falling through to slow CPU work.
- It should not be copied as a gate-only strategy without evidence. Kimi's current bottleneck is exposed expert movement wait, especially `io_uring_wait` caused by up/gate/down miss scheduling, small runtime batches, and incomplete overlap.
- The next accepted gain must reduce exposed wait on general prompts. Higher hit rate, lower SSD bytes, or a better single prompt is insufficient by itself.
- The short-term engineering target is to make stable general-prompt decode exceed `2 tok/s`. The long-term product target remains stable `>5 tok/s` for random user prompts on 16 GB host RAM + 32 GB VRAM.

Hard constraints for every experiment:

- Cold start only; no warm page cache or reused process state.
- Host RAM peak `<15900000000` bytes, including mmap/file cache, pinned staging, page cache, helper processes, and kernel/cgroup accounting.
- Use as much VRAM as safely possible, but do not trade VRAM hit rate for correctness or TTFT regressions.
- Quality gate must include `Please introduce France in a short paragraph.` and held-out general prompts; output must be coherent and semantically correct.
- TTFT must be `<=1.20x` the paired baseline.
- Dev prompts may guide optimization; held-out prompts are used only for final validation.
- Every accepted SOTA must be reproducible: commit and push immediately with env, command, prompt set, run path, RAM/VRAM metrics, TTFT, quality result, token-rate delta, and rollback commit in the commit body and plan.

Execution plan from here:

1. Re-establish the current reproducible baseline.
   - Use the current branch and pinned control profile.
   - Run N96 cold-start paired baseline on the dev prompt set.
   - Record per-prompt token rate, TTFT, decode time, quality answer, host RAM, file/anon breakdown, VRAM cache stats, expert-pack bytes, `iouring_wait_us`, H2D, pinned staging, and CPU fallback counters.
   - Treat the current `1.8-1.9 tok/s` France-like result as historical until reproduced by the exact command on this branch.

2. Locate the exposed critical path before changing code.
   - Break down each token into routing/top-k, up/gate expert read, up/gate H2D/staging, up/gate compute, down expert read, down H2D/staging, down compute, CPU fallback, and synchronization gaps.
   - For each layer/role, record whether the stall is caused by missing residency, queue starvation, small batch size, H2D serialization, compute, or fallback.
   - Rank optimization candidates by removable seconds, not by intuition or hit rate alone.

3. Test the DeepSeek-style gate extension hypothesis on Kimi.
   - Measure how much gate work still executes on CPU/defer slow paths.
   - Compare gate-only VRAM/RAM residency against up/gate paired residency and up/gate/down joint residency.
   - Accept gate-only work only if it reduces exposed wait and improves held-out token rate; otherwise keep the useful part as "GPU extension under CPU/defer scheduler" and optimize all roles together.

4. Optimize RAM/VRAM layout explicitly.
   - VRAM holds the hottest and most latency-critical experts.
   - RAM holds second-tier experts only when they are batchable and reduce exposed wait versus SSD.
   - Replace low-value decode-time file cache with explicit expert cache only after proving those file-backed pages are not needed during decode.
   - Prefer layer/role slabs for high-miss critical layers when random hot entries fragment IO or reduce batch size.

5. Reduce `io_uring` queue starvation.
   - Test more aggressive same-layer co-submit of up/gate/down misses after routing is known.
   - Test one-layer-ahead prefetch only when a predictor or route history gives enough accuracy to avoid wasting RAM/VRAM bandwidth.
   - Keep all speculative/predictive paths default-off until N32 dev A/B proves they reduce exposed wait without quality or TTFT regressions.

6. Validation ladder.
   - N32 dev: cheap screen for quality, RAM, and obvious regressions.
   - N96 dev: verify real token-rate and bottleneck changes.
   - N96 held-out: only final acceptance; no tuning based on held-out traces.
   - Accepted candidate: commit and push immediately.
   - Rejected candidate: revert or leave default-off, document measured reason and rollback point.

Immediate next action:

- `all-1200-minp4` RAM tier has been rejected after N96 held-out validation. It reduced SSD bytes but did not reduce exposed `io_uring_wait` on held-out prompts and slightly regressed median/mean token rate.
- Phase 4F control profiling shows no CPU fallback; the next work must target tensor staging, upgate kernel/wait, and cache-budget allocation rather than simply adding more RAM tier.
- Immediate next A/B candidates:
  - rebalance the existing 15 GiB VRAM expert-cache budget between upgate and down, with an explicit removable-wait bound before running;
  - test small layer/role slabs for top staging rows such as `blk.1 gate`, `blk.4 down`, and `blk.6 down`, only if they fit by replacing lower-yield cache entries.

## Current execution goal

Goal for the current optimization cycle:

> Verify whether the DeepSeek-style CPU/defer main path plus GPU expert-cache extension can produce another reproducible Kimi gain, and if so implement the smallest safe change that reduces exposed expert-transfer wait while preserving general-prompt quality under the 16 GB host RAM gate.

This is not a plan to copy DeepSeek's gate-only hotpool blindly. For Kimi, the current evidence says the bottleneck is mostly gate/up/down miss scheduling and exposed `io_uring_wait`, not standalone gate CPU compute.

Done criteria:

- Keep the active branch `vendor/kimi-deepseek-41d205-additive`.
- Keep every risky change default-off until it has paired baseline/candidate evidence.
- Use dev prompts for design and held-out prompts only for final validation.
- A candidate is accepted only when it improves reproducible general-prompt token rate, passes semantic quality, stays below `15900000000` bytes host RAM, keeps TTFT within `+20%`, and is committed/pushed with full reproduction details.
- A candidate that only improves one prompt, only improves hit rate, or only improves a noisy single run is rejected or left default-off.

## Immediate execution plan

1. Finish Phase 4B fused-path shadow documentation.
   - Record that the shadow code is diagnostic/default-off.
   - Record N32 dev3 quality, token rate, RAM, TTFT, and shadow cache-hit evidence.
   - Commit and push only as diagnostic infrastructure if build and smoke remain clean.

2. Decide whether actual fused up/gate/down co-submit is worth implementing.
   - Proceed only if the shadow shows a large same-layer miss group that current-down overlap does not already cover.
   - The expected benefit must be stated before implementation as a hard upper bound from `iouring_wait_us`, active expert bytes, and observed plannable rows.
   - If implemented, start with default-off env and N32 dev3 A/B before any N96 run.

3. If fused co-submit has weak upside, move to RAM/VRAM cache co-design.
   - Replace low-value decode-time file cache with explicit expert cache only when profiling proves those pages are not needed by decode.
   - Prefer layer/role-aware slabs for high-miss critical layers over random hot expert insertion.
   - Measure whether RAM resident experts reduce exposed wait rather than merely moving bytes from SSD to RAM.

4. Finalize only on general prompts.
   - Dev prompt gains can guide implementation.
   - SOTA claims must be based on held-out general prompts, not France-only or prompt-specific pack behavior.

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

## Phase 0 dev baseline result: 2026-07-10 12:55 CST

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase0-dev-baseline-125550`

Source state:

- Branch: `vendor/kimi-deepseek-41d205-additive`
- Commit before this result note: `5a694ee9dc13c90bc69e976bc63ee91fcd4f89a3`
- Mode: cold-start dev prompt sweep, `MemoryMax=15900000000`, `MemorySwapMax=0`, N96, 32 threads, pinned slots 12, current RAM tier profile enabled.

Command:

```bash
OUT=/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase0-dev-baseline-125550
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root "$OUT" \
  --mode dev \
  --n 96 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 900 \
  --extra-runtime-env "GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=$OUT/ram-batch-profile.csv"
```

Result table:

| prompt | quality | tok/s | TTFT ms | decode ms | decode tokens | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 1.91 | 8233.46 | 44544.32 | 85 | 13.56 | 46.45 | 455.1 |
| dev_japan_factual | pass | 1.95 | 8321.95 | 39910.16 | 78 | 13.56 | 42.01 | 413.6 |
| dev_photosynthesis_factual | pass | 1.80 | 7684.10 | 52276.97 | 94 | 13.41 | 54.37 | 501.2 |
| dev_linear_equation | pass | 1.60 | 11051.04 | 21207.11 | 34 | 13.56 | 25.15 | 280.1 |
| dev_python_reverse | pass | 1.72 | 8724.06 | 55242.75 | 95 | 13.56 | 60.08 | 549.7 |
| dev_zh_france | pass | 1.91 | 7714.09 | 25708.26 | 49 | 13.41 | 27.70 | 269.2 |
| dev_mixed_summary | pass | 1.79 | 11166.61 | 27421.46 | 49 | 13.56 | 31.17 | 330.8 |

Aggregate:

- Token rate: min `1.60`, median `1.80`, mean `1.81`, max `1.95` tok/s.
- TTFT: min `7684.10`, median `8321.95`, mean `8985.04`, max `11166.61` ms.
- RAM peak: min `13.41`, median `13.56`, max `13.56` GiB, under the 16 GB gate.
- `iouring_wait_us`: min `25.15s`, median `42.01s`, mean `40.99s`, max `60.08s`.
- Quality: all 7 dev prompts passed their automatic keyword gates and produced coherent outputs.

Key counters from the France regression run:

- `expert_pack_0`: `iouring_reads=85185`, `iouring_bytes=488667217920`, `iouring_wait_us=46450869`, `direct_reads=0`, `read_failures=0`.
- `expert_pack_iouring_0`: `inflight_avg=3.81`, `inflight_max=8`, `batch_hist=1:385,2-4:6594,5-8:8224,gt32:176`.
- `vram_upgate`: `slots=1735`, `hits=42389`, `misses=55003`, `hit_rate=43.5%`.
- `vram_down`: `slots=723`, `hits=29579`, `misses=18573`, `preloads=11676`, `hit_rate=61.4%`.
- `current_down_overlap`: `planned_jobs=11676`, `completed_jobs=11676`, `cache_hits=8044`, `worker_us=6418996`.

Interpretation:

- The current general dev baseline is stable enough for comparison and is not France-only: all 7 dev prompts pass quality, and token rate ranges from `1.60` to `1.95` tok/s.
- The next optimization should target exposed expert movement wait, especially up/gate misses and small runtime IO batches. Up/gate hit rate remains much lower than down hit rate on every dev prompt (`36.6%` to `46.5%` for up/gate versus `56.2%` to `62.4%` for down).
- RAM has headroom under the 16 GB hard cap in this profile, but any extra RAM tier must prove it reduces exposed wait, not just page cache or SSD bytes.
- Held-out test prompts must not be used for hotset/profile design. The current held-out file was inspected during setup, so before making a final SOTA acceptance claim, create or refresh a sealed held-out set and record that replacement.

Immediate next step:

- Run Phase 1 role/layer exposed-wait profiling on dev prompts only, then choose between gate-focused, up/gate-focused, or joint up/gate/down cache changes based on measured exposed wait rather than hit rate alone.

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

## Phase 1 result: n32 dev role/layer profile

Timestamp: 2026-07-10 13:08 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834`

Command shape:

- Same cold-start 16GB cgroup runner as Phase 0.
- N32, `PROFILE=1`, all 7 dev prompts, current RAM tier profile enabled.
- First 3 prompts were run first, then the same out root was resumed with `--skip-existing` for the remaining 4 prompts.

Quality note:

- 6/7 automatic quality gates passed.
- `dev_linear_equation` failed only because N32 truncated the answer after `**x =`; the N96 Phase 0 baseline for the same prompt passed with `x = 7`.
- Therefore this profile is valid for bottleneck localization only. It is not an acceptance-quality run and must not be used as SOTA evidence.

Aggregate profile metrics:

- Token rate under profiling: min `1.46`, median `1.68`, mean `1.65`, max `1.84` tok/s.
- TTFT under profiling: min `7722.46`, median `9463.29`, mean `9819.32`, max `12411.67` ms.
- RAM peak: min `13.44`, median `13.59`, max `13.59` GiB.
- `iouring_wait_s`: min `17.32`, median `19.15`, mean `19.78`, max `23.41`.
- `iouring_bytes_gib`: min `192.7`, median `207.6`, mean `222.5`, max `265.5`.
- `fallback-profile.csv` rows: `0`, so this profile did not expose CPU fallback.

Role/layer evidence:

- `up-gate-profile.csv`: `6083` decode rows, `28` layers, all `n_active=8`.
- `down-batch-profile.csv`: `13027` decode rows, all decode rows are `down`; prompt rows contain gate/up/down and are excluded from decode down ranking.
- Total n32 dev exposed up/gate wait: `18707.3 ms`; total up/gate wall: `35260.5 ms`.
- Total n32 dev decode down stage: `32703.7 ms`; total down wall: `34930.2 ms`.

Top up/gate exposed-wait layers:

| layer | calls | exposed wait ms | up/gate wall ms | misses | hit rate | type |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 217 | 1095.1 | 1142.1 | 2298 | 33.8% | 22 |
| 27 | 217 | 1083.1 | 1130.0 | 2170 | 37.5% | 22 |
| 12 | 217 | 1076.5 | 1123.6 | 2235 | 35.6% | 22 |
| 20 | 217 | 1076.5 | 1123.4 | 2234 | 35.7% | 22 |
| 16 | 217 | 1068.9 | 1115.6 | 2158 | 37.8% | 22 |
| 25 | 217 | 1064.9 | 1111.4 | 2246 | 35.3% | 22 |
| 18 | 217 | 1061.4 | 1108.7 | 2126 | 38.8% | 22 |
| 24 | 217 | 1057.5 | 1103.8 | 2192 | 36.9% | 22 |

Top down decode-stage layers:

| layer | calls | stage ms | wall ms | misses | hit rate |
|---:|---:|---:|---:|---:|---:|
| 10 | 217 | 1131.0 | 1162.2 | 1266 | 27.1% |
| 6 | 217 | 1126.8 | 1157.5 | 1252 | 27.9% |
| 4 | 217 | 1109.9 | 1141.4 | 1359 | 21.7% |
| 8 | 217 | 1109.2 | 1140.7 | 1248 | 28.1% |
| 7 | 217 | 1102.7 | 1133.7 | 1217 | 29.9% |
| 9 | 217 | 1098.1 | 1128.9 | 1201 | 30.8% |
| 18 | 217 | 1070.0 | 1100.9 | 1200 | 30.9% |
| 58 | 217 | 1054.5 | 1085.5 | 1214 | 30.1% |

Top combined `upgate_wait + down_stage` layers:

| layer | upgate wait ms | down stage ms | combined ms |
|---:|---:|---:|---:|
| 10 | 1095.1 | 1131.0 | 2226.1 |
| 18 | 1061.4 | 1070.0 | 2131.4 |
| 20 | 1076.5 | 1046.1 | 2122.6 |
| 16 | 1068.9 | 1045.4 | 2114.3 |
| 25 | 1064.9 | 1028.3 | 2093.2 |
| 24 | 1057.5 | 1030.8 | 2088.3 |
| 22 | 1055.8 | 1020.5 | 2076.3 |
| 23 | 1051.2 | 1016.8 | 2068.0 |

Decision:

- Kimi does not show a DeepSeek-style gate-only CPU bottleneck.
- The visible issue is paired up/gate staging wait for type `22` layers plus down staging; gate-only hotpool is not the right next primary move because up remains on the same critical path.
- The first optimization probe should be role-aware cache allocation, starting with a controlled upgate/down split sweep. If moving slots from down to upgate regresses, the next probe should be a joint layer-profile cache rather than more upgate-only space.

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

### Phase 2A exact next experiment: upgate/down split sweep

Timestamp: 2026-07-10 13:25 CST.

Hypothesis:

- Current split is `GGML_MOE_VRAM_CACHE_UPGATE_PCT=62`.
- Phase 1 shows upgate type `22` exposed wait is material, but down stage is larger. A small reallocation toward upgate may improve decode if added upgate hits reduce exposed wait more than the removed down slots increase down staging.
- This is a default-off env-only experiment; no source behavior changes.

Theoretical bound:

- N32 full-dev profile exposed upgate wait is `18.7s`; down decode stage is `32.7s`.
- A split-only change cannot remove all upgate wait because misses are broad across many layers and experts.
- Practical upside for N96 is expected to be modest, likely single-digit percentage unless the extra upgate slots cover high-repeat misses. Any down regression can erase the gain.

Experiment:

1. Run a quick cold-start N32 dev smoke using the first 3 dev prompts:
   - control: `UPGATE_PCT=62`;
   - candidate A1: `UPGATE_PCT=66`;
   - candidate A2: `UPGATE_PCT=70`.
2. Keep all other env exactly the same as Phase 0, including RAM tier.
3. Do not use held-out test prompts.
4. Compare quality, TTFT, token rate, RAM peak, upgate/down slots, hit rates, `iouring_wait_us`, and expert pack batch histograms.

Promotion:

- If neither candidate improves median and minimum token rate on the 3-prompt smoke, reject the split-only approach.
- If one candidate improves without quality/RAM/TTFT regression, run all 7 dev prompts at N96 with paired control and candidate.
- Only after paired N96 dev improvement, run a refreshed sealed held-out test set.

Rollback:

- Env-only rejected candidates require no source rollback.
- If a source patch is later needed for joint cache profile support, it must be default-off and reverted if it fails gates.

### Phase 2A result: split-only sweep rejected

Timestamp: 2026-07-10 13:24 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase2a-upgate-pct-n32-dev3-132403`

All runs:

- cold start, N32, first 3 dev prompts only;
- `PROFILE=0`;
- same 16GB cgroup and RAM tier as Phase 0;
- quality passed for all 9 prompt runs.

Result:

| split | upgate slots | down slots | token rate min | token rate median | token rate mean | TTFT median ms | iouring wait mean s | RAM peak GiB | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 62 | 1735 | 723 | 1.84 | 1.84 | 1.85 | 8537.31 | 18.99 | 13.55 | control |
| 66 | 1847 | 647 | 1.83 | 1.89 | 1.87 | 8454.36 | 18.71 | 13.56 | reject: min did not improve |
| 70 | 1959 | 571 | 1.73 | 1.86 | 1.82 | 8609.74 | 18.86 | 13.55 | reject: France regression |

Per-prompt observations:

- `pct66` improved France and Japan, but `dev_photosynthesis_factual` dropped from `1.84` to `1.83`, so the minimum-token-rate gate failed.
- `pct70` reduced France from `1.84` to `1.73`, confirming that stealing too many down slots hurts stability.
- `iouring_wait` moved only slightly; split-only does not fix runtime queue starvation.

Decision:

- Do not promote split-only VRAM reallocation to N96.
- Keep `GGML_MOE_VRAM_CACHE_UPGATE_PCT=62` for current baseline/SOTA reproduction.
- Move to a scheduling/cache-completeness experiment that can reduce exposed wait without just trading down misses for upgate misses.

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

### Phase 4A exact next experiment: same-layer gate/up/down cosubmit smoke

Timestamp: 2026-07-10 13:33 CST.

Existing default-off mechanism:

- `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`
- Optional filters:
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_DOWN_ONLY=1`
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_MIN_SEEN=<n>`
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_MIN_COUNT=<n>`
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT=<csv>`

Hypothesis:

- When the gate path sees a routed expert, the corresponding up/down expert ID is already known.
- Co-submitting the paired up/down read can create larger same-layer IO batches and reduce queue starvation.
- This is closer to the measured bottleneck than split-only cache reallocation.

Risks:

- Kimi's current fused up/gate path may not trigger the standalone gate cosubmit hook often enough; the first check must verify nonzero `gate/up/down cosubmit` jobs.
- Extra reads can evict useful cache lines or steal IO from demand reads, especially if unfiltered.
- Previous DeepSeek experiments had rejected variants, so this must stay default-off and pass Kimi-specific dev gates before any promotion.

Theoretical bound:

- The upper bound is a reduction in exposed `iouring_wait`, not total expert bytes.
- If cosubmit only changes batch shape but keeps the same critical demand order, gain will be small.
- If it lifts inflight depth from the current `~4` toward the pure IO bench regime without extra stale reads, the improvement could be material; the smoke should first prove `iouring_wait_us` drops and jobs are nonzero.

Experiment:

1. Run cold-start N32 dev3 control from Phase 2A as the paired baseline: `pct62`.
2. Run candidate C1:
   - `UPGATE_PCT=62`;
   - `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`;
   - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT=$OUT/gate-updown-cosubmit-profile.csv`.
3. If C1 has nonzero jobs but regresses from extra reads, test C2:
   - add `GGML_MOE_GATE_UPDOWN_COSUBMIT_DOWN_ONLY=1`;
   - add `GGML_MOE_GATE_UPDOWN_COSUBMIT_MIN_SEEN=1`.
4. Keep all other env identical to Phase 0/2A.
5. Do not use held-out test prompts.

Acceptance:

- All smoke prompts quality pass.
- `gate/up/down cosubmit` reports nonzero jobs, zero failures, and no read failures.
- N32 dev3 minimum and median token rate beat Phase 2A `pct62`.
- `iouring_wait_us` mean drops without TTFT or RAM regression.
- Only then run paired N96 dev validation.

### Phase 4A result: standalone gate cosubmit rejected

Timestamp: 2026-07-10 13:34 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4a-cosubmit-c1-n32-dev3-133415`

Candidate env:

```bash
GGML_MOE_GATE_UPDOWN_COSUBMIT=1
GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT=$OUT/gate-updown-cosubmit-profile.csv
```

Paired control:

- Phase 2A `pct62`: `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase2a-upgate-pct-n32-dev3-132403/pct62`

Result:

| run | quality | token rate min | token rate median | token rate mean | TTFT median ms | iouring wait mean s | RAM peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control pct62 | 3/3 pass | 1.84 | 1.84 | 1.85 | 8537.31 | 18.99 | 13.55 |
| C1 cosubmit | 3/3 pass | 1.78 | 1.79 | 1.81 | 8878.44 | 19.00 | 13.56 |

Activation check:

- No `[moe_stream_batch] gate/up/down cosubmit: ...` atexit counter appeared in `stderr.txt`.
- The profile CSV existed and had `6116` data rows, but it was pair-observation output:
  - `up_predict_down`: `177` rows;
  - `actual_up`: `177` rows;
  - `actual_down`: `5760` rows.
- Therefore the current Kimi path did not execute actual standalone cosubmit jobs.

Interpretation:

- The standalone hook lives on the one-stream gate path. Current Kimi SOTA uses fused up/gate batch paths for the relevant work, so this hook is not the right integration point.
- `up_predict_down` showed `pack_hits=0` because it used the up/gate expert byte size when predicting down; down has a different packed size. Any future fused-path implementation must use exact tensor sizes and per-size cache groups.
- Since jobs were zero and token rate regressed, do not run C2.

Decision:

- Reject `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` for current Kimi SOTA.
- Keep it default-off.
- Next source-level work should be stats-only first: add a fused up/gate path shadow counter that records potential same-layer up/gate/down co-submit opportunities with exact per-role sizes, without issuing extra reads. Only if the shadow proves useful should an actual prefetch path be implemented.

### Phase 4B next plan: fused-path co-submit shadow, stats-only

Hypothesis:

- The useful scheduling point for Kimi is inside the fused up/gate batch path, after routing has produced active expert IDs and before up/gate/down staging completes.
- Current-down overlap already handles some down preloading, but Phase 1 still shows down stage and up/gate wait on the critical path.
- A stats-only fused-path shadow can identify whether there are same-layer jobs that could be co-submitted earlier or grouped better without risking correctness.

Implementation plan:

1. Add a default-off env such as `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1`.
2. In the fused up/gate path, record for each layer/token:
   - active experts;
   - up, gate, and down tensor names;
   - exact expert bytes per role;
   - cache hit/miss by role;
   - pack hit/miss by role;
   - whether current-down overlap already submitted each down expert;
   - potential grouped job counts by expert byte size.
3. Write CSV to `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW_OUT`.
4. Run N32 dev3 with shadow only and verify:
   - quality unchanged;
   - token rate and TTFT not materially changed;
   - shadow rows are nonzero;
   - overhead is small enough to run full dev profiling.
5. Use the shadow report to decide between:
   - no-op if current-down overlap already covers the opportunities;
   - exact-size fused down co-submit;
   - larger grouped IO scheduling for up/gate/down role batches.

Acceptance for shadow:

- No behavior change by default.
- Shadow run quality passes.
- Runtime overhead is low enough for diagnostic use.
- The CSV provides enough evidence to calculate an upper bound before any actual prefetch implementation.

### Phase 4B result: fused-path co-submit shadow implemented, diagnostic only

Timestamp: 2026-07-10 13:45 CST.

Source status:

- Default-off diagnostic code added in `ggml/src/ggml-cuda/moe_stream_batch.cu`.
- Env gates:
  - `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1`
  - `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW_OUT=<csv>`
- No behavior change when the env is unset.
- Build passed with the existing warning set:

```bash
cmake --build build-cuda-batch -j$(nproc)
```

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4b-fused-shadow-n32-dev3-134523`

Command:

```bash
OUT=/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4b-fused-shadow-n32-dev3-134523
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root "$OUT" \
  --mode dev \
  --n 32 \
  --max-prompts 3 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 600 \
  --upgate-pct 62 \
  --extra-runtime-env "GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=$OUT/ram-batch-profile.csv
GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1
GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW_OUT=$OUT/fused-upgate-down-shadow.csv"
```

Paired control:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase2a-upgate-pct-n32-dev3-132403/pct62`

Result:

| run | quality | token rate min | token rate median | token rate mean | TTFT median ms | iouring wait mean s | RAM peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control pct62 | 3/3 pass | 1.84 | 1.84 | 1.85 | 8537.31 | 18.99 | 13.55 |
| fused shadow | 3/3 pass | 1.82 | 1.87 | 1.87 | 8662.96 | 18.52 | 13.55 |

Shadow aggregate after filtering duplicate per-process CSV headers:

- Rows: `5583`, all decode.
- Active expert rows: `44664`.
- Up cache hit rate: `44.08%`.
- Gate cache hit rate: `44.17%`.
- Down cache hit rate: `32.10%`.
- All-role cache-hit rows: `32.10%`.
- Down-overlap plannable rows: `25094 / 44664 = 56.18%`.
- Pack lookup:
  - up pack hits `44664`, misses `0`;
  - gate pack hits `44664`, misses `0`;
  - down pack hits `39432`, misses `0` for rows with a matching down tensor.

Top same-layer all-role miss layers:

| layer | active | all-role miss | any-role miss | down plannable | up miss | gate miss | down miss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 744 | 528 | 603 | 603 | 528 | 528 | 603 |
| 1 | 744 | 518 | 604 | 604 | 519 | 518 | 604 |
| 4 | 744 | 516 | 588 | 588 | 517 | 516 | 588 |
| 60 | 768 | 511 | 562 | 538 | 511 | 511 | 562 |
| 3 | 744 | 496 | 563 | 563 | 497 | 496 | 563 |
| 10 | 744 | 494 | 744 | 0 | 494 | 494 | 744 |
| 29 | 744 | 489 | 560 | 560 | 489 | 489 | 560 |
| 28 | 744 | 481 | 549 | 549 | 482 | 481 | 549 |

Interpretation:

- The DeepSeek gate-only lesson partially transfers, but the Kimi bottleneck is not a standalone gate CPU path.
- Up and gate miss together almost exactly, and down miss is also high. Therefore a gate-only cache increase is unlikely to be enough.
- The pack path is present for the measured up/gate/down rows; the remaining problem is residency and scheduling, not missing pack coverage.
- Current-down overlap already has many plannable rows, so an actual fused co-submit must prove that it reduces exposed wait beyond the existing overlap rather than duplicating work.
- Since the shadow run is diagnostic and min token rate did not improve, it is not a SOTA claim.

Decision:

- Keep the shadow path default-off.
- Commit/push it only as diagnostic infrastructure with the above run path and rollback point.
- Next candidate must be one of:
  - exact fused-path up/gate/down co-submit only if it uses the shadow data to avoid duplicate current-down work;
  - layer/role-aware RAM/VRAM cache reallocation for layers with high all-role miss;
  - no-op if the expected upper bound is too small after subtracting current-down overlap.

### Phase 4C plan: early current-down overlap A/B

Reason:

- The current Kimi path already supports current-layer down overlap, but the Phase 4B env did not enable `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY`.
- In the mixed up/gate path, the non-early mode starts current-down overlap only after up/gate staging and compute. The early mode starts it immediately after routing and before up/gate staging/compute.
- This is the narrowest way to test the user's desired behavior: once routing has produced active expert IDs, begin moving down experts without waiting for up/gate compute.

Hypothesis:

- If exposed down wait is still on the critical path, `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` can hide part of the current-down worker time behind up/gate staging/compute.
- Phase 4B measured current-down worker time around `2.34-2.48s` per N32 dev prompt, so the hard upper bound is roughly that worker time. A realistic bound is smaller because part of the work is already overlapped and because early down reads can compete with up/gate reads.
- If IO contention dominates, early mode may reduce down wait but increase up/gate staging wait or total `iouring_wait_us`; in that case reject it.

Experiment:

1. Run a fresh cold-start N32 dev3 control on current HEAD with shadow disabled.
2. Run candidate with the same env plus:

```bash
GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1
```

3. Keep:
   - `UPGATE_PCT=62`;
   - 1800 MiB RAM tier with `blk1_gate_full384.csv`;
   - `MemoryMax=15900000000`, `MemorySwapMax=0`;
   - no held-out test prompts.
4. Compare:
   - token-rate min/median/mean;
   - TTFT;
   - decode ms;
   - `expert_pack iouring_wait_us`;
   - current-down `worker_us`, planned jobs, completed jobs;
   - up/gate and down cache hit/miss;
   - iouring batch hist/inflight;
   - RAM peak and quality.

Acceptance:

- Quality 3/3 pass.
- Host RAM remains under the hard gate.
- TTFT ratio `<=1.20`.
- N32 dev3 min and median token rate beat the paired fresh control.
- The gain must be explained by lower exposed wait, not by noise or shorter output.
- If accepted on N32, run N96 dev before any SOTA claim.

### Phase 4C result: early current-down overlap rejected

Timestamp: 2026-07-10 14:02 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4c-early-down-n32-dev3-215843`

Paired fresh control command shape:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4c-early-down-n32-dev3-215843/control \
  --mode dev \
  --n 32 \
  --max-prompts 3 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 600 \
  --upgate-pct 62 \
  --extra-runtime-env "<1800 MiB blk1 gate RAM tier env>"
```

Candidate adds:

```bash
GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1
```

Control result:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB |
|---|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 1.86 | 8104.82 | 16649.94/31 | 13.55 |
| dev_japan_factual | pass | 1.89 | 7382.51 | 16428.44/31 | 13.56 |
| dev_photosynthesis_factual | pass | 1.88 | 7071.07 | 16527.53/31 | 13.41 |

Candidate result:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB | failure |
|---|---:|---:|---:|---:|---:|---|
| dev_france_regression | fail | n/a | n/a | n/a | 3.51 | exit `134`, CUDA OOM |
| dev_japan_factual | fail | n/a | n/a | n/a | 7.19 | exit `134`, CUDA OOM |
| dev_photosynthesis_factual | pass | 0.91 | 7772.54 | 34146.30/31 | 7.97 | severe slowdown |

Failure evidence:

- France/Japan stderr:
  - `ggml_cuda_compute_forward: MUL failed`
  - `CUDA error: out of memory`
  - abort signal `6`, exit `134`.
- The third run survived only after the earlier failures, but free VRAM was much lower:
  - `VRAM cache budget: requested=15000 MiB actual=3729 MiB free=4241 MiB`;
  - upgate slots dropped from the control `1735` to `431`;
  - down slots dropped from `723` to `180`;
  - upgate hit rate collapsed to `12.8%`;
  - decode regressed to `0.91 tok/s`.
- After the run, `nvidia-smi` showed no persistent process and memory returned to normal, so this is not accepted as a stable runtime state.

Historical cross-check:

- The parent plan already rejected `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` in Phase 7GG:
  - first n32 decode `28927.32 ms / 31`;
  - repeat n32 decode `29231.83 ms / 31`;
  - repeat fell inside baseline noise;
  - no n96 was run.
- Phase 7NW also closed the same-layer IO-fill direction because early overlap, aux ring, combined up/gate IO, and dual-fence variants either failed reproducibility or lost endpoint overlap.

Decision:

- Reject `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` for current Kimi SOTA.
- Do not run N96.
- Do not enable this flag in reproduction scripts.
- Keep the default-off code only as diagnostic/historical infrastructure.
- The next direction should not be "submit more same-layer IO" unless it first proves how it avoids the 7GG/7GH/7KW/7LE endpoint-overlap loss.

### Phase 4D plan: RAM/VRAM cache co-design trace and candidate screen

Reason:

- Phase 4B shadow shows large up/gate/down miss traffic, but Phase 4C and older 7NW evidence show that simply exposing more same-layer IO concurrency is not enough.
- Current N32 control still spends most host memory on file-backed pages, while the explicit 1800 MiB `blk1_gate_full384.csv` RAM tier has only about `0.6%` total RAM-tier hit rate in the Phase 4B/4C dev3 runs.
- The next practical question is whether the same or slightly larger RAM budget can be moved from low-value page cache / low-hit RAM tier entries into higher-yield expert entries selected from actual foreground IO across dev prompts.

Hypothesis:

- A prompt-agnostic RAM tier generated from multi-prompt foreground IO traces can outperform the current static `blk1_gate_full384.csv` tier.
- It must improve exposed wait or token rate, not only RAM hit rate.
- Candidate budgets must stay inside the 16 GB cgroup gate; based on the fresh control peak around `13.56 GiB`, `1800 MiB` is safe and `2400-3000 MiB` is only a guarded experiment.

Experiment sequence:

1. Run cold-start N32 dev3 trace with current accepted runtime, no early overlap, and tracing only:

```bash
GGML_MOE_IO_READ_TRACE_OUT=$RUN/io-read-trace.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv
```

2. Use only dev traces, never held-out test prompts, to generate candidate RAM profiles with:

```bash
.Agent/run-tools/kimi_ram_candidate_multidev_screen.py
.Agent/run-tools/kimi_make_ram_slab_profile_from_io_trace.py
```

3. Screen at least:
   - current control profile: `blk1_gate_full384.csv`, 1800 MiB;
   - best dev-trace `down` profile at 1800 MiB;
   - best dev-trace `up,gate` profile at 1800 MiB;
   - if the reports justify it, guarded 2400/3000 MiB profiles.
4. A/B only the strongest one or two candidates on N32 dev3.
5. Run N96 dev and held-out test only if N32 dev3 improves min and median token rate with quality pass.

Acceptance:

- Quality passes for all dev smoke prompts.
- RAM peak `<15900000000` bytes with `MemorySwapMax=0`.
- TTFT ratio `<=1.20`.
- N32 dev3 min and median token rate beat paired fresh control.
- RAM-tier hits must replace SSD waits on the critical path; if hit rate rises but token rate falls, reject.
- Any accepted improvement must be committed and pushed with run roots, exact profiles, commands, RAM/TTFT/quality, and rollback point.

### Phase 4D progress: trace screen and N32 candidate A/B

Timestamp: 2026-07-10 14:16 CST.

Trace run:

- Correct trace root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127`
- Earlier mistaken trace root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-220835`
  - rejected as a data source because `$RUN` was expanded too early and trace files were written under `/`;
  - generated root files were removed before the corrected run.

Correct trace command shape:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127 \
  --mode dev \
  --n 32 \
  --max-prompts 3 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 600 \
  --upgate-pct 62 \
  --extra-runtime-env "GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=\$RUN/ram-batch-profile.csv
GGML_MOE_IO_READ_TRACE_OUT=\$RUN/io-read-trace.csv
GGML_MOE_IO_WAIT_TRACE_OUT=\$RUN/io-wait-trace.csv"
```

Trace result:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB | io-read rows |
|---|---:|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 1.84 | 8758.53 | 16852.32/31 | 13.56 | 38849 |
| dev_japan_factual | pass | 1.75 | 9128.08 | 17672.96/31 | 13.56 | 38844 |
| dev_photosynthesis_factual | pass | 1.81 | 7607.67 | 17132.35/31 | 13.41 | 38035 |

Trace note:

- The trace run is diagnostic only; tracing overhead means its token rate is not used as the paired performance baseline.
- Total trace rows: `115722`, total traced IO bytes: `617.14 GiB`.
- Current `blk1_gate_full384.csv` RAM tier does not appear in `io-read-trace.csv` because RAM-tier hits bypass the IO trace.
- Current tier measured from `ram-batch-profile.csv`:
  - total RAM hits: `718`;
  - total RAM H2D bytes: `3.144 GiB`;
  - total RAM batch wall: `2068.418 ms`.

Candidate screening output:

- Candidate directory:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127/ram-candidates`

Top screened candidates:

| candidate | budget MiB | selected entries | trace hit rows | trace hit GiB | prompts | batches | dominant batches >=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| down-1800 | 1800 | 269 | 4437 | 29.02 | 3 | 2513 | 223 |
| upgate-1800 | 1800 | 375 | 5425 | 25.31 | 3 | 3186 | 210 |
| all-1800 | 1800 | 310 | 5519 | 31.45 | 3 | 3843 | 55 |
| all-3000 | 3000 | 529 | 8463 | 47.19 | 3 | 5062 | 310 |

N32 A/B run:

- Candidate root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-ram-candidate-n32-dev3-141626`
- Paired fresh control:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4c-early-down-n32-dev3-215843/control`

Result:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control blk1_gate-1800 | 3/3 | 1.86 | 1.88 | 1.88 | 7382.51 | 13.56 | 56.16 | 617.1 | 718 | 3.14 |
| all-1800 | 3/3 | 1.91 | 1.95 | 1.99 | 7579.92 | 13.61 | 52.90 | 588.8 | 5519 | 31.45 |
| all-3000 | 3/3 | 1.92 | 1.98 | 1.96 | 8726.13 | 14.78 | 53.17 | 573.0 | 8463 | 47.19 |

Per-prompt `all-1800`:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB | iouring wait s | RAM hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 2.11 | 7579.92 | 14672.06/31 | 13.61 | 16.53 | 1884 |
| dev_japan_factual | pass | 1.95 | 8151.71 | 15914.49/31 | 13.61 | 18.03 | 1911 |
| dev_photosynthesis_factual | pass | 1.91 | 7382.18 | 16234.74/31 | 13.47 | 18.34 | 1724 |

Interpretation:

- `all-1800` is the better next candidate despite `all-3000` having a slightly higher N32 minimum:
  - `all-1800` has better mean token rate;
  - lower TTFT;
  - far lower RAM peak;
  - less risk against the 16 GB hard gate.
- The gain is consistent with the intended mechanism:
  - RAM hits increase from `718` to `5519`;
  - explicit RAM H2D increases from `3.14 GiB` to `31.45 GiB`;
  - expert-pack iouring bytes drop from `617.1 GiB` to `588.8 GiB`;
  - iouring wait drops from `56.16s` to `52.90s`;
  - token-rate min/median/mean improve.
- This is still a dev N32 result, not a SOTA claim.

Decision:

- Promote only `all-1800` to N96 dev validation.
- Do not promote `all-3000` yet because TTFT and RAM are too close to the limit for only marginal min-token-rate gain.
- If N96 passes, copy the `all-1800.profile.csv` into a tracked `.Agent/profiles/kimi/ram-tier/phase4d-*` path, record exact reproduction commands, run held-out validation, then commit and push as a candidate improvement.

### Phase 4D result: all-1800 rejected after held-out test

Timestamp: 2026-07-10 14:59 CST.

N96 dev candidate:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-all1800-n96-dev3-142351`
- Fresh paired N96 dev control:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-fresh-control-n96-dev3-142832`

N96 dev paired result:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control blk1_gate-1800 | 3/3 | 1.82 | 1.83 | 1.86 | 8449.15 | 13.56 | 144.08 | 1369.9 | 1694 | 7.42 |
| all-1800 | 3/3 | 1.88 | 1.90 | 1.91 | 8436.26 | 13.62 | 140.23 | 1311.5 | 11562 | 65.82 |

N96 dev interpretation:

- Dev3 passed all gates.
- Improvement mechanism matched the hypothesis:
  - RAM hits increased by `9868`;
  - iouring bytes dropped by `58.4 GiB`;
  - iouring wait dropped by `3.85s`;
  - token-rate min/median/mean improved.
- This justified held-out validation but was still not a SOTA claim.

Held-out candidate:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-all1800-n96-test-143404`
- Profile used:
  `.Agent/profiles/kimi/ram-tier/phase4d-dev-trace/all-1800.profile.csv`
  - copied temporarily for the test;
  - removed from the worktree after rejection;
  - canonical artifact remains in the trace run:
    `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127/ram-candidates/all-1800.profile.csv`.
- Fresh paired held-out control:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-control-n96-test-144730`

Held-out result:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control blk1_gate-1800 | 6/6 | 1.54 | 1.82 | 1.788 | 9621.60 | 232735.82 | 14.81 | 311.95 | 2920.0 | 3455 | 15.13 |
| all-1800 | 6/6 | 1.54 | 1.82 | 1.785 | 9993.85 | 230457.74 | 14.81 | 315.54 | 2849.2 | 15244 | 85.87 |

Held-out per-prompt deltas:

| prompt | token-rate delta | decode delta ms | TTFT delta ms | quality |
|---|---:|---:|---:|---:|
| test_chinese_01 | +0.01 | -356.03 | -704.22 | pass/pass |
| test_coding_01 | +0.00 | +61.48 | +160.86 | pass/pass |
| test_english_factual_01 | +0.00 | -114.97 | -211.29 | pass/pass |
| test_english_factual_02 | +0.02 | -492.32 | +182.39 | pass/pass |
| test_mixed_instruction_01 | -0.02 | +531.74 | +742.01 | pass/pass |
| test_reasoning_math_01 | -0.03 | +680.83 | -2278.08 | pass/pass |

Held-out quality notes:

- All six held-out answers were semantically coherent.
- The math prompt produced the correct answer `5:15 PM` in both candidate and control.
- The very high math TTFT was not introduced by all-1800:
  - control TTFT `232735.82 ms`;
  - all-1800 TTFT `230457.74 ms`.

Decision:

- Reject `all-1800` as a SOTA/performance improvement.
- Reason: it improves dev3 but does not improve held-out min/median/mean token rate; mean regresses slightly and `iouring_wait` increases by `3.59s` despite lower iouring bytes.
- Do not commit the `phase4d-dev-trace/all-1800.profile.csv` runtime profile.
- Keep the result as evidence that dev3 static RAM hotsets can overfit and may only move bytes from SSD to RAM without reducing held-out endpoint wait.

Next plan:

1. Build the next RAM/VRAM candidate from the full 7-prompt dev set, not dev3.
2. Require candidate screening to enforce prompt/category coverage, e.g. `min_prompts >= 4` and no single category dominating selected traffic.
3. Prefer dynamic or online RAM admission rules over static prompt-trace hotsets:
   - promote an expert to RAM only after repeated misses across prompts or sustained per-layer pressure;
   - preserve the current explicit 16 GB RAM accounting;
   - record whether RAM hits reduce endpoint wait, not just SSD bytes.
4. Do not use held-out test prompts for candidate construction.
5. Any future RAM profile must pass:
   - N32 full-dev;
   - N96 full-dev;
   - held-out N96;
   - and only then be committed as an accepted profile.

### Phase 4E plan: full-dev7 prompt-agnostic RAM tier screen

Timestamp: 2026-07-10 15:12 CST.

Purpose:

- Re-run RAM-tier design with all seven dev prompts instead of the dev3 subset that overfit in Phase 4D.
- Keep held-out test prompts unused for construction.
- Require each selected expert to be observed in at least four dev prompts.

Trace run:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343`
- Env:

```bash
GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=$RUN/ram-batch-profile.csv
GGML_MOE_IO_READ_TRACE_OUT=$RUN/io-read-trace.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv
```

Trace quality:

- `6/7` auto-quality pass at N32.
- `dev_linear_equation` failed only because N32 truncated the answer after `x =`; this N32 trace remains valid for route/IO observation, but no SOTA claim can be made without N96 quality passing.

Trace aggregate:

- Total io-read rows: `292570`.
- Total traced IO: `1557.45 GiB`.
- Host RAM peak: `13.57 GiB`.

Current tier actual RAM hits during trace:

- `1770` RAM hits;
- `7.75 GiB` RAM H2D;
- `5380.88 ms` total RAM batch wall across seven prompts.

Candidate screen command:

```bash
python3 .Agent/run-tools/kimi_ram_candidate_multidev_screen.py \
  --input-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343 \
  --out-profile <candidate>.profile.csv \
  --out-report <candidate>.report.json \
  --out-csv <candidate>.candidates.csv \
  --budget-mib <1200|1800|2400> \
  --roles <down|up,gate|up,gate,down> \
  --min-count 2 \
  --min-prompts 4 \
  --max-jobs 8
```

Candidate summary:

| candidate | budget MiB | entries | trace hit rows | trace hit GiB | prompts hit | dominant batches >=4 | role mix |
|---|---:|---:|---:|---:|---:|---:|---|
| all-1200-minp4 | 1200 | 214 | 8940 | 49.31 | 7 | 77 | up/down/gate |
| all-1800-minp4 | 1800 | 320 | 11984 | 66.12 | 7 | 323 | up/down/gate |
| all-2400-minp4 | 2400 | 427 | 14751 | 81.19 | 7 | 681 | up/down/gate |
| down-1800-minp4 | 1800 | 267 | 8349 | 55.16 | 7 | 546 | down only |
| upgate-1800-minp4 | 1800 | 378 | 11307 | 52.34 | 7 | 695 | up/gate only |

Decision before A/B:

- First test `all-1200-minp4` and `all-1800-minp4`.
- Skip `all-2400-minp4` until a smaller candidate proves endpoint value because Phase 4D showed larger RAM tiers can improve hit bytes but fail held-out endpoint speed.
- Skip role-only candidates initially because Phase 4B showed up/gate/down misses are coupled; mixed all-role candidates are a better first test.

N32 full-dev A/B plan:

1. Run fresh N32 full-dev control with current `blk1_gate_full384.csv`.
2. Run `all-1200-minp4` with:
   - `GGML_MOE_RAM_TIER_MIB=1200`;
   - `GGML_MOE_RAM_TIER_PIN_MIB=1200`;
   - profile from the Phase 4E trace candidate directory.
3. Run `all-1800-minp4` with:
   - `GGML_MOE_RAM_TIER_MIB=1800`;
   - `GGML_MOE_RAM_TIER_PIN_MIB=1800`;
   - profile from the Phase 4E trace candidate directory.
4. Compare:
   - token-rate min/median/mean;
   - per-prompt decode delta;
   - TTFT ratio;
   - RAM peak;
   - iouring wait and bytes;
   - RAM hits and RAM H2D;
   - output quality, with the known N32 truncation caveat for `dev_linear_equation`.

N32 promotion rule:

- Candidate must improve full-dev min and median token rate over fresh control.
- Candidate must not materially worsen coding, mixed, or reasoning prompts.
- Host RAM must remain under `15900000000`.
- TTFT median must not exceed `1.20x` control.
- If N32 passes, run N96 full-dev before any held-out validation.

### Phase 4E N32 full-dev A/B result

Timestamp: 2026-07-10 15:27 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-minp4-n32-fulldev-ab-151309`

Runs:

- `control`: current `blk1_gate_full384.csv`, 1800 MiB.
- `all-1200-minp4`: full-dev7 `min_prompts>=4` all-role candidate, 1200 MiB.
- `all-1800-minp4`: full-dev7 `min_prompts>=4` all-role candidate, 1800 MiB.

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/7 | 1.62 | 1.85 | 1.793 | 8824.89 | 10729.33 | 13.56 | 139.08 | 1557.15 | 1770 | 7.75 |
| all-1200-minp4 | 6/7 | 1.63 | 1.89 | 1.839 | 8471.97 | 11157.23 | 13.03 | 137.68 | 1515.60 | 8940 | 49.31 |
| all-1800-minp4 | 6/7 | 1.56 | 1.91 | 1.813 | 8360.51 | 11407.33 | 13.62 | 139.42 | 1498.79 | 11984 | 66.12 |

N32 quality note:

- `dev_linear_equation` is `fail` for all three runs due N32 truncation, not a candidate-specific semantic failure.
- N96 full-dev remains mandatory before any quality claim.

Per-prompt deltas versus control:

| prompt | all-1200 tok delta | all-1200 decode delta ms | all-1800 tok delta | all-1800 decode delta ms |
|---|---:|---:|---:|---:|
| dev_france_regression | +0.06 | -466.73 | +0.05 | -405.37 |
| dev_japan_factual | +0.09 | -736.54 | +0.06 | -498.80 |
| dev_linear_equation | +0.01 | -195.52 | +0.04 | -454.69 |
| dev_mixed_summary | +0.02 | -149.10 | -0.04 | +519.48 |
| dev_photosynthesis_factual | +0.04 | -392.96 | +0.06 | -534.82 |
| dev_python_reverse | +0.06 | -647.63 | -0.11 | +1317.29 |
| dev_zh_france | +0.04 | -268.18 | +0.08 | -611.89 |

Decision:

- Promote `all-1200-minp4` to N96 full-dev.
  - It improves min/median/mean;
  - improves every per-prompt decode time;
  - reduces iouring wait by `1.40s`;
  - reduces iouring bytes by `41.55 GiB`;
  - lowers RAM peak from `13.56 GiB` to `13.03 GiB`;
  - keeps TTFT well inside the `+20%` gate.
- Reject `all-1800-minp4` at N32.
  - It regresses min token rate from `1.62` to `1.56`;
  - regresses Python and mixed prompts;
  - increases iouring wait slightly despite reducing bytes;
  - therefore it has the same "more RAM hits but worse endpoint" warning pattern as Phase 4D held-out.

Next:

- Run fresh paired N96 full-dev:
  - control with `blk1_gate_full384.csv`;
  - candidate with `all-1200-minp4`.
- Only if N96 full-dev improves min/median/mean and quality passes, copy the profile into a tracked path and run held-out N96.

### Phase 4E N96 full-dev and held-out result

Timestamp: 2026-07-10 23:55 CST.

Runs:

- N96 full-dev run root:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-all1200-n96-fulldev-153209`
- N96 held-out run root:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-all1200-n96-heldout-235502`

N96 full-dev aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 7/7 | 1.63 | 1.81 | 1.816 | 9024.04 | 11973.20 | 13.56 | 284.84 | 2799.68 | 3363 | 14.73 |
| all-1200-minp4 | 7/7 | 1.68 | 1.83 | 1.836 | 8622.40 | 11075.65 | 13.03 | 282.15 | 2729.91 | 15315 | 84.49 |

Full-dev interpretation:

- `all-1200-minp4` was a small positive dev signal:
  - token-rate mean `+0.020 tok/s`;
  - min `+0.05 tok/s`;
  - median `+0.02 tok/s`;
  - `iouring_wait` `-2.69s`;
  - SSD IO `-69.77 GiB`.
- However, per-prompt results already showed regressions on `dev_mixed_summary` and `dev_python_reverse`.
- Therefore it was allowed to proceed to held-out validation but was not accepted.

N96 held-out aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/6 | 1.62 | 1.85 | 1.810 | 9830.87 | 248966.26 | 14.81 | 307.34 | 2919.97 | 3455 | 15.13 |
| all-1200-minp4 | 6/6 | 1.63 | 1.82 | 1.802 | 9466.49 | 228378.04 | 14.81 | 312.94 | 2862.60 | 13174 | 72.50 |

Held-out per-prompt deltas versus control:

| prompt | tok delta | decode delta ms | TTFT delta ms | TTFT ratio |
|---|---:|---:|---:|---:|
| test_chinese_01 | -0.01 | +136.16 | +553.55 | 1.064 |
| test_coding_01 | +0.01 | -245.85 | -669.13 | 0.936 |
| test_english_factual_01 | +0.05 | -1280.16 | -1129.60 | 0.874 |
| test_english_factual_02 | -0.01 | +323.09 | -350.96 | 0.962 |
| test_mixed_instruction_01 | -0.05 | +1660.17 | -1423.75 | 0.873 |
| test_reasoning_math_01 | -0.04 | +845.74 | -20588.22 | 0.917 |

Decision:

- Reject `all-1200-minp4`; do not copy it into tracked `.Agent/profiles/`.
- Reason:
  - held-out median token rate regressed from `1.85` to `1.82`;
  - held-out mean regressed from `1.810` to `1.802`;
  - total held-out decode time increased by `1.44s`;
  - `iouring_wait` increased by `5.61s` even though SSD IO decreased by `57.37 GiB`;
  - RAM H2D increased from `15.13 GiB` to `72.50 GiB`, so the profile moved bytes from SSD to RAM but did not shorten the critical path.
- Quality passed and TTFT did not violate the `1.20x` gate, but performance was not a general-prompt improvement.

Additional observation:

- `test_reasoning_math_01` shows a severe prompt-dependent TTFT long tail:
  - control TTFT `248966.26 ms`;
  - candidate TTFT `228378.04 ms`;
  - both runs hit the `15899996160` byte cgroup peak.
- The next profiling pass must separate decode token-rate work from cold-start/prompt TTFT long-tail causes. A candidate that only improves decode bytes but leaves TTFT near the memory limit is not sufficient.

Next:

- Build an exposed-wait profile for N96 dev and held-out control runs:
  - per-layer and per-role `io_uring_wait`;
  - queue depth and batch histogram around stalls;
  - RAM tier H2D timing versus SSD wait;
  - prompt/prefill TTFT memory peak and major fault source for the reasoning/math long tail.
- Screen layer/role slab candidates only if they have a measurable upper bound in removable exposed wait and keep RAM below the 16 GB gate with margin.

### Phase 4F N96 dev5 profile result

Timestamp: 2026-07-11 00:23 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase5-control-n96-profile-dev5-002343/control-profile`

Command shape:

- prompt file: `.Agent/evals/kimi-general-dev-prompts.jsonl`
- first 5 dev prompts only: France, Japan, photosynthesis, linear equation, Python reverse
- N96, cold start, `PROFILE=1`
- control RAM tier: `.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv`
- host memory gate: `MemoryMax=15900000000`, `MemorySwapMax=0`

Important caveat:

- `PROFILE=1` slows the run and should not be compared as a token-rate SOTA result.
- This run is for bottleneck attribution only.

Profile summary:

| metric | value |
|---|---:|
| prompts | 5 |
| quality | 5/5 |
| weighted token rate under profiling | 1.624 tok/s |
| aggregate iouring throughput | 9.256 GiB/s |
| peak IO utilization reference | 0.899 of 10.3 GiB/s |
| direct read ratio | 0.000 |
| weighted iouring inflight avg | 3.909 |
| iouring wait / decode fraction | 0.971 |

Prompt-level profile:

| prompt | tok/s | decode ms | down wall ms | upgate wall ms | fallback ms |
|---|---:|---:|---:|---:|---:|
| dev_france_regression | 1.64 | 51685 | 22500 | 14053 | 0 |
| dev_japan_factual | 1.71 | 45601 | 21104 | 12599 | 0 |
| dev_linear_equation | 1.46 | 23331 | 17604 | 6182 | 0 |
| dev_photosynthesis_factual | 1.67 | 56272 | 22539 | 15350 | 0 |
| dev_python_reverse | 1.56 | 60760 | 25264 | 16535 | 0 |

Key result:

- CPU fallback is not the current decode bottleneck in this control profile:
  - `fallback-profile.csv` is empty;
  - CPU/MOE profile reports batch accept for upgate and down with no single/fallback path.
- The main bottlenecks are inside the GPU extension path:
  - tensor staging / expert movement for down, up, and gate tensors;
  - upgate CUDA batch kernel/wait time;
  - cache-budget allocation under a nearly full 32 GB VRAM budget.

Tensor staging totals by prompt:

| prompt | down stage ms | gate stage ms | up stage ms | upgate-call wall ms | upgate up_wait ms | upgate gate_wait ms |
|---|---:|---:|---:|---:|---:|---:|
| dev_france_regression | 15725.5 | 3187.9 | 2215.4 | 14052.9 | 7139.1 | 7672.4 |
| dev_japan_factual | 14436.9 | 3276.8 | 2067.8 | 12599.1 | 6316.7 | 6733.1 |
| dev_linear_equation | 9418.2 | 4291.3 | 2856.4 | 6181.9 | 3129.4 | 3321.3 |
| dev_photosynthesis_factual | 16582.7 | 2707.8 | 1838.9 | 15350.4 | 7734.6 | 8252.9 |
| dev_python_reverse | 18018.3 | 3385.1 | 2347.6 | 16535.1 | 8465.5 | 8903.5 |

Top removable tensor-stage rows from the profile:

| row | stage ms | wall ms | misses | hit rate |
|---|---:|---:|---:|---:|
| `blk.1 ffn_gate_exps` type 22 | 5175.4 | 5377.9 | 369 | 48.7% |
| `blk.4 ffn_down_exps` type 23 | 3043.5 | 3228.3 | 2863 | 24.8% |
| `blk.6 ffn_down_exps` type 2 | 2995.9 | 3192.7 | 2530 | 33.6% |
| `blk.1 ffn_down_exps` type 11 | 2451.6 | 2646.0 | 2877 | 24.4% |
| `blk.3 ffn_down_exps` type 11 | 2339.6 | 2422.3 | 2724 | 28.5% |
| `blk.10 ffn_down_exps` type 2 | 2203.0 | 2269.3 | 2535 | 33.4% |
| `blk.58 ffn_down_exps` type 23 | 2200.3 | 2266.4 | 2571 | 32.5% |

Top layer combined pressure:

| layer | tensor stage ms | tensor wall ms | upgate wall ms | up_wait ms | gate_wait ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 8246.9 | 8653.0 | 1936.6 | 1809.0 | 1847.4 |
| 25 | 2470.4 | 2556.6 | 2109.7 | 1904.4 | 2024.0 |
| 10 | 2461.4 | 2546.6 | 2107.9 | 1897.8 | 2022.4 |
| 18 | 2434.0 | 2519.3 | 2100.5 | 1892.9 | 2016.0 |
| 20 | 2429.2 | 2516.5 | 2089.9 | 1880.7 | 2005.4 |
| 26 | 2467.5 | 2553.6 | 2055.5 | 1827.2 | 1943.8 |

VRAM/RAM constraint from the same run:

- `moe_stream_batch` requested `15000 MiB` expert VRAM cache.
- End-of-run CUDA free memory for France profile was only `872 MiB`.
- Therefore a new full-layer VRAM residency experiment cannot simply add entries. It must replace lower-yield cache entries or rebalance the upgate/down split.
- Host memory peak for France profile was `14627450880` bytes, but held-out reasoning/math previously reached the `15899996160` byte cgroup peak. RAM-tier expansions require extra margin, not just average fit.

Interpretation:

- The rejected `all-1200-minp4` result is consistent with this profile:
  - RAM tier can reduce SSD bytes;
  - but if it increases RAM H2D and does not reduce the exposed tensor-stage or upgate wait rows, token rate does not improve.
- Current bulk SSD throughput is already near the pure IO upper bound in normal runs:
  - held-out control: `10.039 GiB/s`;
  - full-dev control: `10.554 GiB/s`;
  - profile run: `9.256 GiB/s` because profiling adds overhead.
- The next optimization should not start from "load more random experts into RAM".

Next candidate design rules:

1. VRAM cache rebalance before RAM expansion.
   - Compute the current marginal value of upgate versus down slots from misses, stage/wait rows, and hit rates.
   - Try a default-off cache split that moves a small amount of VRAM from low-yield down entries to high-pressure upgate or targeted gate/down slab rows.
   - Expected upper bound must be stated as removable milliseconds from the Phase 4F table before testing.

2. Layer/role slab only for top rows.
   - Candidate rows: `blk.1 gate`, `blk.4 down`, `blk.6 down`, `blk.1 down`, `blk.3 down`.
   - Slab must replace lower-yield cache entries; do not exceed current VRAM cache budget.
   - If implemented in RAM rather than VRAM, it must prove lower exposed stage/wait, not just lower SSD bytes.

3. Upgate kernel/wait analysis.
   - `up22_gate22` and `up18_gate18` upgate-call rows contribute large kernel/wait time.
   - Before changing cache policy again, inspect whether `parallel_up_gate`, `parallel_stage`, and CUDA stream waits are actually overlapping for the dominant type pairs.
   - If a type pair is compute/stream-bound rather than transfer-bound, RAM/VRAM residency will have limited upside.

4. TTFT long-tail analysis remains separate.
   - The held-out reasoning/math prompt hit a severe TTFT long tail and cgroup memory peak.
   - Do not accept decode-only improvements if they make TTFT or host memory margin worse.

### Phase 4G planned A/B: protected `blk.1 gate` VRAM slab

Purpose:

- Test the smallest targeted VRAM slab suggested by Phase 4F before writing new runtime code.
- Use existing `GGML_MOE_VRAM_PROFILE*` support only; keep runtime behavior default-off.

Candidate:

- Tensor: `blk.1.ffn_gate_exps.weight`.
- Experts: all 384 experts.
- Expert size from route profile: `4.48 MiB`.
- Full slab size: `1.68 GiB`.
- Runtime env:
  - `GGML_MOE_VRAM_PROFILE=<candidate-profile.csv>`
  - `GGML_MOE_VRAM_PROFILE_PROTECT=1`
  - `GGML_MOE_VRAM_PROFILE_PRELOAD=1`
  - `GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1`
  - keep `GGML_MOE_VRAM_CACHE_MIB=15000`
  - keep `GGML_MOE_VRAM_CACHE_UPGATE_PCT=62`

Theory and upper bound:

- Phase 4F dev5 profile measured `blk.1 gate` tensor-stage:
  - stage `5175.4 ms`;
  - wall `5377.9 ms`;
  - misses `369`;
  - hit rate `48.7%`.
- Ignoring eviction losses, eliminating this stage from the normal dev5 control window would save at most about `5.18s`.
- Using the normal N96 dev5 control decode window from Phase 4E (`~212.3s`, `386` decode tokens), the ideal ceiling is roughly:
  - baseline `386 / 212.3 = 1.82 tok/s`;
  - ideal no-overhead `386 / (212.3 - 5.18) = 1.86 tok/s`;
  - maximum gain about `+2.5%`.
- Because the slab consumes `1.68 GiB` inside a nearly full upgate VRAM cache, real gains may be lower or negative if protected entries evict higher-value dynamic entries.

Execution:

1. Generate a temporary dev-only profile under the run directory with rows:
   - CSV format: `rank,count,expert_bytes,cumulative_bytes,tensor_base,expert_idx,tensor`;
   - `expert_idx=0..383`;
   - `tensor=blk.1.ffn_gate_exps.weight`;
   - `count` set high enough to protect entries when profile policy is active.
2. Run paired N32 full-dev cold-start A/B:
   - control: current `blk1_gate_full384.csv` RAM tier only;
   - candidate: same control plus protected `blk.1 gate` VRAM profile.
3. Acceptance for promotion to N96:
   - quality must pass the same N32 caveat as previous full-dev screens;
   - mean/median/min token rate must not regress;
   - TTFT must stay within `1.20x`;
   - host RAM peak `<15900000000`;
   - logs must show the profile loaded and protected/preloaded entries.
4. If N32 fails or improvement is within noise with worse TTFT/RAM, reject and do not run N96.

### Phase 4G N32 result: rejected

Timestamp: 2026-07-11 00:40 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4g-blk1gate-vram-n32-004035`

Candidate profile:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4g-blk1gate-vram-n32-004035/blk1_gate_full384_vram_profile.csv`

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/7 | 1.52 | 1.79 | 1.720 | 126.851 | 8932.37 | 10833.34 | 13.56 | 145.73 | 1557.15 |
| blk1-gate-vram | 6/7 | 1.45 | 1.63 | 1.599 | 136.095 | 9305.66 | 11199.78 | 13.56 | 149.34 | 1656.58 |

Quality note:

- `dev_linear_equation` failed in both runs due the known N32 truncation caveat.
- Other prompts passed.

Per-prompt deltas:

| prompt | tok delta | decode delta ms | TTFT delta ms | TTFT ratio |
|---|---:|---:|---:|---:|
| dev_france_regression | -0.13 | +1319.88 | +24.46 | 1.003 |
| dev_japan_factual | -0.14 | +1385.85 | +994.18 | 1.114 |
| dev_linear_equation | -0.10 | +1482.86 | -278.11 | 0.974 |
| dev_mixed_summary | -0.11 | +1255.74 | +517.86 | 1.048 |
| dev_photosynthesis_factual | -0.16 | +1690.21 | +205.14 | 1.027 |
| dev_python_reverse | -0.01 | +122.36 | -201.20 | 0.979 |
| dev_zh_france | -0.20 | +1987.95 | +373.34 | 1.046 |

Decision:

- Reject `blk1-gate-vram`; do not run N96.
- Reason:
  - mean token rate regressed by `0.121 tok/s`;
  - median regressed by `0.16 tok/s`;
  - decode time increased by `9.245s`;
  - `iouring_wait` increased by `3.617s`;
  - SSD IO increased by `99.43 GiB`;
  - TTFT remained within the 20% gate, but endpoint performance clearly regressed.

Mechanism diagnosis:

- Candidate logs confirm the intended profile loaded:
  - `profile preload: loaded 384 entries`;
  - `profile preload: blk.1.ffn_gate_exps.weight loaded=384`.
- But `GGML_MOE_VRAM_PROFILE_PROTECT=1` has broader semantics than intended:
  - it pins generic preload slots, not only the explicit profile rows;
  - in the France run, down cache changed from `preloads=4324 pinned=0 hit_rate=57.7%` to `preloads=578 pinned=578 hit_rate=31.5%`;
  - this destroyed down cache behavior and outweighed any possible `blk.1 gate` benefit.
- Therefore existing profile protection cannot be used directly for targeted slabs in the current SOTA configuration.

Next plan:

- Add a default-off runtime option that restricts protection to explicit profile-count rows only, for example:
  - `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`.
- Intended behavior:
  - profile preload rows with `profile_count > 0` may be pinned;
  - ordinary down/current prefetch preloads with `profile_count == 0` must not be pinned just because profile protection is enabled;
  - existing default behavior must remain unchanged unless the new env is set.
- Re-run the same N32 A/B only after this code change.
- Expected bound remains small: if profile-only protection works perfectly, the best case is still only the `blk.1 gate` stage bound (`~5.18s` on profiled dev5), so this remains a screening experiment, not a likely SOTA jump.

### Phase 4H N32 result: profile-only protection fixed the mechanism

Timestamp: 2026-07-11 00:58 CST.

Code change:

- Added default-off env:
  - `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`
- Behavior:
  - default behavior is unchanged when the env is unset;
  - when set, `GGML_MOE_VRAM_PROFILE_PROTECT=1` only pins preload inserts that have explicit `profile_count > 0` or were already recorded as profile-pinned keys;
  - ordinary down/current prefetch preloads with `profile_count == 0` are no longer pinned solely because profile protection is enabled.

Build:

```bash
cmake --build build-cuda-batch -j $(nproc)
```

Build result:

- passed;
- only pre-existing warnings were emitted.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4h-profile-only-blk1gate-n32-005854`

Candidate env delta:

- same as Phase 4G plus:
  - `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/7 | 1.52 | 1.77 | 1.737 | 125.495 | 9091.74 | 11602.34 | 13.56 | 145.26 | 1557.15 |
| profile-only | 6/7 | 1.54 | 1.80 | 1.753 | 124.457 | 8858.62 | 10994.27 | 13.55 | 143.53 | 1557.15 |

Quality note:

- `dev_linear_equation` failed in both runs due the known N32 truncation caveat.
- Other prompts passed.

Per-prompt deltas:

| prompt | tok delta | decode delta ms | TTFT delta ms | TTFT ratio |
|---|---:|---:|---:|---:|
| dev_france_regression | +0.02 | -228.51 | -396.69 | 0.956 |
| dev_japan_factual | +0.01 | -124.52 | +262.21 | 1.031 |
| dev_linear_equation | +0.02 | -177.49 | -1066.66 | 0.908 |
| dev_mixed_summary | +0.00 | +5.20 | +92.65 | 1.008 |
| dev_photosynthesis_factual | +0.03 | -203.04 | +613.63 | 1.079 |
| dev_python_reverse | +0.01 | -115.19 | -535.21 | 0.946 |
| dev_zh_france | +0.02 | -194.34 | -225.98 | 0.972 |

Mechanism check:

- Candidate logs confirm profile preload happened:
  - `profile preload: blk.1.ffn_gate_exps.weight loaded=384`.
- Candidate logs also confirm the Phase 4G regression mechanism was fixed:
  - down cache stayed `preloads=4324 pinned=0 hit_rate=57.7%` on France, matching control;
  - upgate/down hit rates and IO bytes were unchanged in aggregate;
  - `iouring_wait` decreased by `1.73s`, not because bulk bytes fell but because the protected profile-only behavior avoided the previous down-cache damage.

Decision:

- Keep the default-off code change for N96 dev validation.
- Do not call this SOTA:
  - N32 signal is small;
  - held-out was not run;
  - profile-only slab must pass N96 dev before any held-out run.

Next:

- Run paired N96 full-dev with:
  - control: current RAM tier only;
  - candidate: same profile-only `blk.1 gate` VRAM slab.
- Promote to held-out only if N96 full-dev min/median/mean improve, quality passes, TTFT remains within `1.20x`, and RAM remains under the 16 GB gate.

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
