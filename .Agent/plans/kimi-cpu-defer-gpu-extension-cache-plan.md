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
