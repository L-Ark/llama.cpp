# Kimi token-rate optimization plan under 16GB host RAM

## Goal

Continue optimizing Kimi IQ3_S decode throughput in the ik_llama-compatible
vendor path while preserving stable, semantically correct output.

Hard target:

- Maximize token rate.
- Keep total host RAM below 16 GB, including process RSS, allocator overhead,
  page cache, pinned host buffers, mmap cache, and any helper process memory.
- Use as much VRAM as possible without causing graph/cache instability, and
  prefer GPU compute/cache over host RAM tiers whenever both are viable.
- Keep TTFT increase within 20% of the current cold-start baseline.
- Every accepted step must pass the fixed semantic quality gate:

```text
Please introduce France in a short paragraph.
```

The answer must be coherent, about France, and semantically correct.

## Current correctness base

The correctness base is the vendor Kimi path after the DeepSeek2 YaRN kq-scale
fix documented in `.Agent/plans/kimi-vendor-port-plan.md`.

Known accepted quality output shape:

```text
France is a country in Western Europe known for its rich history, art, and culture...
```

The previous quality target, stable `-n 96` output, has been met. Future speed
work must not regress this.

Recent vendor-side DeepSeek optimization branch
`wici/vendor/deepseek-token-rate-16gb` has been merged by ancestry into
`vendor/kimi-moe-stream-on-vendor`. The current tree keeps the Kimi/vendor code
as the source of truth and imports the missing route cache simulator:

```text
scripts/moe-route-cache-sim.py
```

Use that tool for profile/cache planning, but do not assume DeepSeek-specific
heuristics are correct for Kimi until measured.

## Non-negotiable acceptance gates

Every optimization, benchmark, and commit must satisfy all gates below. A result
that violates any one gate is rejected even if token rate improves.

1. Commit/push or rollback gate.
   - When a change produces a reproducible token-rate improvement and satisfies
     all gates below, commit and push it immediately.
   - When a change reduces token rate, lowers answer quality, increases TTFT by
     more than 20%, exceeds the host RAM limit, depends on warm cache, or cannot
     be reproduced, revert to the previous accepted version before continuing.
   - Do not stack another optimization on top of a rejected change.
2. Reproducibility gate.
   - Every accepted improvement must include enough reproduction detail for a
     clean machine to rerun it: commit, branch, command, env, model path, expert
     pack path, cgroup/memory setup, cold-start method, GPU model, driver/CUDA
     version, exact prompt, seed, and output.
   - A single lucky run is not accepted. For small changes, run at least one
     cold-start validation plus one repeat. For large token-rate gains, run the
     three-run `-n 96` semantic gate.
3. Host RAM gate.
   - Host RAM must stay below 16 GB.
   - Include page cache, process RSS, pinned memory, mmap-resident pages, helper
     process memory, and cgroup/accounted kernel memory where visible.
   - Do not accept a run that only passes because the page cache was already
     warmed. All promotion runs must be cold start.
   - If previous measurements did not include page cache or non-process memory,
     Phase 0 baseline must be rerun before any speed patch.
4. VRAM/GPU-first gate.
   - VRAM should be filled deliberately.
   - Prefer VRAM expert cache, route-profile preload, and graph sizing changes
     over host hot tiers when both are viable.
   - Prefer GPU compute paths over CPU/host staging paths unless profiling proves
     the GPU path is slower or unsafe.
   - Leave explicit graph reserve and safety margins in the run record.
5. Output quality gate.
   - Output quality is checked at every step.
   - The France prompt must produce a coherent short paragraph.
   - The answer must be semantically correct, about France, and free of repeated
     malformed clauses, tokenizer artifacts, role/template leakage, and unrelated
     drift.
   - Large token-rate jumps require the same three-run `-n 96` gate as the
     correctness plan.
6. TTFT gate.
   - TTFT may not increase by more than 20%.
   - Compare against the current cold-start baseline collected in Phase 0.
   - If token rate improves but TTFT exceeds the limit, the patch is rejected
     unless the next patch in the same experiment restores TTFT before commit.
7. Cold-start gate.
   - Every benchmark must be cold start.
   - Restart the process.
   - Drop filesystem cache where permitted, or run in a fresh cgroup/VM state
     that proves cache is cold.
   - Record the method used to prove cold start.

## Required run record

Every experiment must create a run directory under:

```text
/root/lfz/runs/vendor-kimi-token-rate/YYYYMMDD-HHMMSSZ-<short-name>
```

Each run directory must contain:

- `README.md`: short reproduction recipe from a clean checkout.
- `command.txt`: exact command, git commit, branch, model path, expert pack path,
  prompt, seed, and sampling parameters.
- `env.txt`: all `GGML_*`, CUDA, cgroup, and memory-related env vars.
- `stdout.txt` and `stderr.txt`.
- `answer.txt`: exact generated answer.
- `metrics.json`: structured measurements.
- `memory.txt`: RSS, cgroup `memory.current`, `memory.peak`, page cache, swap,
  pinned memory if observable, and OOM status.
- `vram.txt`: `nvidia-smi` before load, after load, first token, mid-decode, and
  after exit.
- `system.txt`: GPU model, driver version, CUDA runtime, CPU, kernel, disk model,
  filesystem, and effective cgroup limits.
- `cold-start.txt`: exact cold-start procedure and evidence that the run did not
  reuse warm page cache.
- `moe.txt`: MoE counters, read counts, bytes, cache hit/miss, prefetch timing,
  staging timing, H2D timing, compute timing, and `read_failures`.
- `quality.txt`: pass/fail decision for the France prompt with the exact reason.

`metrics.json` must include at least:

```json
{
  "commit": "",
  "branch": "",
  "reproduction_readme": "",
  "cold_start": true,
  "cold_start_method": "",
  "host_ram_peak_gib": 0,
  "page_cache_peak_gib": 0,
  "process_rss_peak_gib": 0,
  "cgroup_memory_peak_gib": 0,
  "host_ram_gate_pass": false,
  "vram_peak_mib": 0,
  "vram_unused_min_mib": 0,
  "gpu_compute_path": "",
  "ttft_ms": 0,
  "ttft_baseline_delta_pct": 0,
  "ttft_gate_pass": false,
  "decode_tokens": 0,
  "decode_seconds": 0,
  "token_rate_tps": 0,
  "token_rate_baseline_delta_pct": 0,
  "seconds_per_token": 0,
  "prompt_eval_ms": 0,
  "moe_read_bytes": 0,
  "moe_h2d_ms_per_token": 0,
  "moe_compute_ms_per_token": 0,
  "moe_wait_ms_per_token": 0,
  "moe_cache_hit_pct": 0,
  "read_failures": 0,
  "quality_pass": false,
  "accepted": false,
  "commit_pushed": false,
  "rollback_required": false
}
```

## Development loop

Optimization must alternate between design and execution.

Before every implementation patch:

1. Update this plan with the current bottleneck and the selected next change.
2. Run one profiling experiment to break down per-token time.
3. Rank candidate optimizations by expected token-rate gain.
4. Theorize why the top candidate should help.
5. Compute an upper bound from hard limits such as SSD bandwidth, PCIe/H2D
   bandwidth, expert tensor size, active expert count, CUDA kernel occupancy, or
   measured up/down compute time.
6. Define the acceptance and rollback criteria.

After every implementation patch:

1. Build and run the exact benchmark.
2. Compare measured gain with the theoretical bound.
3. If the result diverges, explain the gap with evidence before trying another
   patch.
4. Run the quality gate.
5. Write the reproduction method and full metrics into the run directory.
6. If the run improves token rate and satisfies host RAM, VRAM/GPU-first, TTFT,
   cold-start, reproducibility, and quality gates, commit and push immediately.
7. If any hard gate fails, revert the patch before continuing and record the
   failed run as rejected.

## Phase 0: cold 16GB baseline

Goal: establish the real baseline under the final deployment constraint.

This must be redone because earlier measurements did not strictly prove total
host RAM below 16 GB including page cache.

Baseline command rules:

- Use the accepted Kimi IQ3_S correctness config.
- Use `-n 96`.
- Use the fixed France prompt.
- Use cold start.
- Run inside a cgroup or equivalent host-memory guard with an effective limit
  below 16 GB. If this guard cannot be enabled, the run is diagnostic only and
  cannot be accepted as baseline.
- Disable or bound host hot tiers initially:
  - `GGML_MOE_RAM_TIER_MIB=0` unless explicitly testing RAM tier.
  - Avoid settings that intentionally keep large expert pages in host page cache.
- Use current best VRAM cache settings, but record exact graph reserve and cache
  budget.

Required baseline measurements:

- End-to-end TTFT.
- Decode token rate.
- Per-token wall time.
- Per-token breakdown:
  - routing/top-k
  - up/gate read
  - up/gate H2D
  - up/gate compute
  - activation/SwiGLU
  - down read
  - down H2D
  - down compute
  - synchronization/wait
  - sampling
- Host RAM peak including page cache.
- VRAM peak and unused VRAM.
- Cache hit rate by layer and expert tensor kind.
- Top recurring experts from the route trace.

Acceptance:

- `host_ram_peak_gib < 16`.
- The recorded host RAM includes page cache and non-process memory.
- The run is cold start and the cold-start proof is recorded.
- France `-n 96` output passes.
- TTFT becomes the baseline for the 20% gate.
- `read_failures = 0`.
- Baseline is recorded in this plan with timestamp, run directory, and
  reproduction method.

## Phase 1: bottleneck classification

Goal: decide whether the next gain should come from IO, H2D transfer, up/gate
compute, down compute, cache policy, graph memory, or scheduling overlap.

Run one instrumented decode and classify each token into:

- IO-bound: read wait dominates and cache hit rate is low.
- H2D-bound: transfer time dominates after reads are available.
- Compute-bound: up/gate or down CUDA kernels dominate.
- Sync-bound: large gaps exist between IO completion, H2D, and kernel launch.
- Cache-capacity-bound: recurring experts miss because VRAM cache budget is too
  small or incorrectly partitioned.
- Quality-sensitive: faster math path changes logits enough to harm output.

Prioritization rule:

1. Fix any correctness-sensitive path before speed.
2. Optimize the largest measured per-token bucket first.
3. Prefer changes that increase VRAM reuse and reduce host RAM pressure.
4. Avoid large math rewrites until IO/cache/scheduling evidence shows compute is
   the bottleneck.

## Phase 2: VRAM-first cache and profile preload

Hypothesis:

Kimi token rate should improve if recurring routed experts are already in VRAM
at decode time, reducing SSD reads and H2D stalls while keeping host RAM below
16 GB.

Design work:

- Export or collect route profile for the fixed prompt and representative Kimi
  prompts.
- Use `scripts/moe-route-cache-sim.py` to compare cache policies.
- Compute theoretical token-rate ceiling from:
  - active experts per token
  - expert tensor bytes by type
  - measured SSD bandwidth
  - measured H2D bandwidth
  - available VRAM after graph reserve
- Decide VRAM partition by tensor kind:
  - up/gate
  - down
  - late-layer vs full-layer
  - CID/profile-preloaded cache

Candidate env/features:

- `GGML_MOE_VRAM_CACHE_MIB`
- `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB`
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP`
- profile-guided preload from the merged DeepSeek work where compatible
- `GGML_MOE_TRACE_PREFETCH`
- route cache simulator output

Acceptance:

- Token rate improves.
- Host RAM remains below 16 GB cold.
- VRAM usage increases or unused VRAM is justified by graph reserve/safety.
- GPU-side cache/compute is preferred over host RAM growth.
- TTFT increase <= 20%.
- France `-n 96` passes three times if gain is large.
- Reproduction method is complete.
- Commit and push immediately after passing.

Rollback:

- Revert if cache preload increases TTFT over 20%, causes host RAM/page cache
  growth above 16 GB, or changes output quality.

## Phase 3: reduce up/gate compute time

Hypothesis:

Earlier Kimi work showed up/gate path is likely a major speed lever. A true
fused/id-MMQ up-gate path for the observed Kimi IQ3_S tensor types can reduce
per-token compute and launch overhead without changing routing.

Required theory before coding:

- Identify exact Kimi up/gate tensor quant types in the loaded model.
- Document the math:
  - gate projection
  - up projection
  - SwiGLU activation
  - elementwise multiply
- Show why the proposed kernel is numerically equivalent or bounded relative to
  the accepted path.
- Estimate upper-bound token rate from current measured up/gate ms/token.

Implementation candidates:

- Vendor-native pruned id-MMQ path for Kimi observed types only.
- Keep opt-in behind an env var.
- Do not reuse a broad ik whole-file port that pulls unrelated quant kernels.

Acceptance:

- Compare output against baseline for short deterministic runs.
- France `-n 96` passes.
- For any large token-rate gain, run three `-n 96` passes.
- Host RAM including page cache remains below 16 GB.
- The accepted path uses GPU compute; any CPU fallback must be justified by
  measured speed and memory evidence.
- TTFT increase <= 20%.
- Reproduction method is complete.
- Commit and push only after acceptance.

Rollback:

- Revert if semantic quality changes, if logits drift causes unstable tails, or
  if measured speed does not match the theoretical model and the gap cannot be
  explained.

## Phase 4: down path and Q8_K/reference-cache probes

Hypothesis:

Down projection can be accelerated by a scoped reference path or late-layer cache
if it is currently a material per-token bucket.

Current caution:

Earlier down Q8_K probes were semantically unsafe when enabled broadly. Late-layer
range gating improved output but still failed `-n 96`. Treat down changes as
quality-sensitive.

Design requirements:

- Re-measure down ms/token after Phase 2 and Phase 3.
- Only proceed if down remains a top bottleneck.
- Compute expected gain from down ms/token, not from intuition.
- Restrict first tests to narrow layer ranges.

Acceptance:

- No semantic regression.
- Three `-n 96` runs for any accepted down math/cache change.
- Host RAM including page cache remains below 16 GB.
- TTFT increase <= 20%.
- Clear explanation of any numeric approximation.
- Reproduction method is complete.
- Commit and push only after acceptance.

Rollback:

- Revert immediately on repeated malformed clauses, grammar collapse, or
  unrelated tail drift.

## Phase 5: overlap IO, H2D, and compute

Hypothesis:

If profiling shows idle gaps between expert read, H2D transfer, and CUDA compute,
stream scheduling can improve token rate without changing math.

Design requirements:

- Timeline one decode token with CUDA events and MoE trace logs.
- Identify which stage waits on which dependency.
- Prove whether the current implementation actually overlaps work.
- Estimate upper bound from the max of overlapped stage times:

```text
ideal_token_ms = max(read_ms, h2d_ms, compute_ms) + unavoidable_sync_ms
```

Candidate work:

- staged pinned slots tuning
- split up/gate staging
- prefetch distance tuning
- async H2D stream separation
- route-aware next-token prefetch where correctness permits

Acceptance:

- Token rate approaches the overlap bound or the gap is explained.
- Host RAM remains below 16 GB; pinned memory counts against the limit.
- VRAM remains deliberately used; do not trade VRAM reuse for host RAM pressure.
- TTFT increase <= 20%.
- Quality gate passes.
- Reproduction method is complete.
- Commit and push only after acceptance.

## Phase 6: host RAM pressure audit

Goal: make sure the accepted fast config will run on real 16GB RAM machines.

For every accepted configuration:

- Run cold with a 16GB cgroup.
- Record page cache before and after.
- Verify no large mmap pages remain resident beyond the intended working set.
- Verify pinned slots and RAM tier do not silently push total memory above limit.
- Prefer `madvise`/`dontneed` style page release only after confirming it does
  not destroy token rate.

Acceptance:

- `memory.peak < 16GB` with margin.
- Page cache and pinned memory are included in the accounting.
- No swap/OOM.
- Token rate and TTFT still meet accepted numbers.
- The reproduction method shows how to enforce the same memory limit.

## Commit and push policy

Commit and push immediately only when all are true:

- Build passes.
- Cold-start benchmark passes under a strict host RAM guard below 16GB.
- Host RAM accounting includes page cache, RSS, pinned memory, mmap-resident
  pages, and helper processes.
- VRAM is used as fully as the graph/cache reserve allows, or the unused VRAM is
  explicitly justified.
- The accepted path maximizes GPU compute/cache and does not rely on an
  unbounded host RAM tier.
- France quality gate passes.
- `read_failures = 0`.
- TTFT increase <= 20%.
- Token rate improves or the patch is required instrumentation.
- The run directory, metrics, and reproduction method are recorded in this plan.

Do not commit:

- speculative speed patches that fail quality,
- patches that only work with warm page cache,
- patches that require more than 16GB host RAM,
- patches that improve average token rate but increase TTFT above the limit,
- patches that are not reproducible from the recorded command/env/model/cgroup
  setup,
- patches that shift work to host RAM when VRAM/GPU capacity is available.

## Baseline table

Fill this before the next code optimization.

| Time UTC | Commit | Run dir | Host RAM peak incl. page cache | VRAM peak / unused | TTFT | Token rate | Quality | Cold proof | Repro method | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-01 12:50:35 | `2c69cf836` | `/root/lfz/runs/vendor-kimi-token-rate/20260701-125035Z-n96-cold-16gb-baseline` | 14.901 GiB cgroup peak, includes page cache by cgroup accounting; `memory.events max=245278`, OOM=0 | 16224 MiB peak / 15886 MiB minimum free | 88609.77 ms | 0.20 tok/s, 5.01989 s/token | PASS: coherent France paragraph ending with `<|im_end|>` | `sync; echo 3 > /proc/sys/vm/drop_caches`; child shell moved into dedicated cgroup before `exec`; `cgroup.procs` sampled with timeout and model PID | `README.md` and `command.txt` in run dir | Accepted as the first cold 16GB correctness/token-rate baseline. Missing explicit anon/file page-cache split and MoE per-stage timing; collect in the next profiling run before any code optimization. |

## Experiment log

Every implementation step must append one row before and after execution.

| Time UTC | Step | Hypothesis | Theoretical upper bound | Result | Gates | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-01 12:47:29 | Phase 0 cgroup smoke v2 | Verify the actual inference process can run inside a strict 16GB host-memory cgroup after fixing the runner to move `$BASHPID`, not the outer shell. | N/A | `/root/lfz/runs/vendor-kimi-token-rate/20260701-124729Z-n2-cgroup-smoke-v2`: `rc=0`, output `France is`, cgroup peak 14.901 GiB. | 16GB/cold/smoke quality pass; not a baseline because `-n 2`. | Proceed to full `-n 96` baseline. |
| 2026-07-01 12:50:35 | Phase 0 full baseline | The current accepted Kimi correctness config should produce stable `-n 96` output under a 16GB host-memory guard, but token rate may drop because page cache is constrained. | Previous unconstrained reference was about 0.51-0.52 tok/s; under strict 16GB, lower bound unknown before measurement. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-125035Z-n96-cold-16gb-baseline`: `rc=0`, prompt eval/TTFT 88.61s, eval 401.59s / 80 decode runs, 0.20 tok/s, total 490.26s. | 16GB pass by cgroup peak 14.901 GiB; cold pass; quality pass; `read_failures` not reported; TTFT becomes baseline; reproducibility recorded. VRAM/GPU-first not optimized: 15.9 GiB VRAM unused. | Baseline established. Next required step is profiling with `memory.stat` anon/file sampling and MoE per-stage timing, then prioritize VRAM expert cache/profile preload. |
| 2026-07-01 13:03:05 | Phase 2A n32 expert pack + VRAM cache profile | Move repeated routed expert tensors into VRAM cache to reduce 16GB page-cache/reclaim stalls without changing model math. | Removing strict-cgroup reclaim stalls could recover the old unconstrained 0.51-0.52 tok/s ceiling; route replay with 15GB cache estimated up to 70.46% trace hit rate. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-130305Z-n32-phase2a-vram-cache-profile`: VRAM peak 31278 MiB, minimum free 832 MiB, cache hit 60.0%, route replay hit estimate 70.46%, but TTFT 110.33s (+24.5%), decode 0.16 tok/s, output started `The user asks: ...`. | Host RAM/cold/read failures pass; VRAM/GPU-first pass; TTFT gate fail; token-rate fail; quality fail. | Rejected. Do not promote and do not stack optimizations on this env. The gap is that enabling `GGML_MOE_STREAM`/batched up-gate changes output semantics and is slower under the 16GB cap. |
| 2026-07-01 13:11:49 | Phase 2B-1 stream-only isolation | If `GGML_MOE_STREAM=1` alone preserves quality, the Phase 2A regression is likely in expert pack/cache/fused up-gate. If it fails, stream path itself is the minimum bad switch. | N/A diagnostic. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-131149Z-n16-phase2b-stream-only`: cgroup peak 14.901 GiB, TTFT 109.13s, decode 0.26 tok/s, output `The:ayt老爷 grave!!!!!!!!!!!`. Log shows `[moe_stream] enabled` and `VRAM cache: cudaMalloc 16.0 GiB FAILED`. | Host RAM/cold pass; TTFT fail; quality fail. | Rejected. Minimum semantic regression switch is `GGML_MOE_STREAM=1`; stop env sweep and debug stream one-path correctness before any VRAM cache promotion. |
| 2026-07-01 13:17:22 | Phase 2C stream cache disabled isolation | Check whether stream-only corruption was caused by the default one-cache 16GiB cudaMalloc failure. | N/A diagnostic. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-131722Z-n16-phase2c-stream-cache0`: `GGML_MOE_STREAM=1`, `GGML_MOE_STREAM_ONE_CACHE_MIB=0`, cgroup peak 14.901 GiB, TTFT 99.18s, decode 0.25 tok/s, output `The:ayt老爷 grave!!!!!!!!!!!`. | Host RAM/cold pass; quality fail. | Rejected. one-cache allocation failure is not the root cause; stream compute/copy/scatter remains unsafe. |
| 2026-07-01 13:23:17 | Phase 1A n32 `-t 40 -tb 40` | Increasing CPU threads on the accepted non-stream path may improve token rate without changing math. | Perfect CPU scaling from 24 to 40 would be 1.67x on the CPU portion, but IO/reclaim can dominate. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-132317Z-n32-phase1a-threads40`: cgroup peak 14.901 GiB, quality pass, TTFT 65.28s, decode 155.51s / 31 runs, 0.20 tok/s. | Host RAM/cold/quality/TTFT pass; token-rate improvement absent. | Rejected for token-rate promotion. Useful observation: `-tb 40` may reduce prompt/TTFT, but decode remains page-cache/IO bound. |
| 2026-07-01 13:34:48 | Phase 2C src1 row fix compare | Stream one-path likely reads the wrong `src1` row. Compare stream output against CPU reference and then fix row selection. | Correctness-only. A fixed stream path should reduce compare max_abs from O(1e-1) to quantization-level O(1e-3), then restore `France is` prefix. | Before fix, `/root/lfz/runs/vendor-kimi-token-rate/20260701-133121Z-n2-phase2c-stream-compare` showed `max_abs` up to 0.527928 and no output yet. After fix, `/root/lfz/runs/vendor-kimi-token-rate/20260701-133448Z-n2-phase2c-stream-src1fix-compare` showed `max_abs <= 0.00151799` and output `France is`. | Host RAM/cold pass; compare pass; n2 prefix pass. | Keep the src1 row-selection fix. It is required before any VRAM/cache stream work. |
| 2026-07-01 13:37:39 | Phase 2C src1 fix stream-only n16 | Verify stream-only semantic smoke after fixing `src1` row selection. | Correctness-only; not a performance promotion. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-133739Z-n16-phase2c-src1fix-stream-only`: output `France is a country in Western Europe known for its rich history, culture, and`, TTFT 101.88s, decode 0.14 tok/s. | Host RAM/cold/quality pass; TTFT pass; token-rate fail. | Accept as stream correctness progress only. Do not promote as performance. |
| 2026-07-01 13:42:33 | Phase 2C src1 fix stream-only n32 | Verify longer stream-only semantic smoke after fixing `src1` row selection. | Correctness-only; not a performance promotion. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-134233Z-n32-phase2c-src1fix-stream-only`: output remained coherent about France, TTFT 111.49s, decode 0.14 tok/s. | Host RAM/cold/quality pass; TTFT fail; token-rate fail. | Commit the opt-in stream correctness fix because it restores semantic output and is default-off. Next work must address TTFT/decode speed before any stream/cache promotion. |
| 2026-07-01 13:51:07 | Phase 2A rerun after src1 fix | Re-test expert pack + 15GB VRAM cache after fixing stream `src1` row selection. | If quality is restored, token rate should improve over 0.20 tok/s; TTFT may still fail because stream prompt eval was slow. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-135107Z-n32-phase2a-src1fix-vram-cache-profile`: quality pass, VRAM peak 31278 MiB / 832 MiB free, read_failures=0, TTFT 114.29s, decode 0.22 tok/s, cache hit 62.3%, up/gate total 13.046 ms/call. | Host RAM/cold/VRAM/read/quality pass; token-rate pass for n32; TTFT fail. | Rejected for promotion due TTFT. Next candidate combines this path with `-t 40 -tb 40`, which previously reduced prompt/TTFT on non-stream baseline. |
| 2026-07-01 13:57:47 | Phase 2D src1 fix + VRAM cache + `-t 40 -tb 40` | Keep Phase 2A decode gains while reducing TTFT with more CPU threads. | Applying half of the earlier TTFT reduction should bring 114.29s near 99.4s; decode should ideally stay above 0.20 tok/s. | `/root/lfz/runs/vendor-kimi-token-rate/20260701-135747Z-n32-phase2d-src1fix-vram-cache-t40`: quality pass, VRAM peak 31278 MiB, read_failures=0, TTFT 102.37s, decode 0.20 tok/s. | Host RAM/cold/VRAM/read/quality/TTFT pass; token-rate improvement absent. | Rejected for promotion. Need a thread-count balance between `t24` (0.22 tok/s, TTFT fail) and `t40` (TTFT pass, 0.20 tok/s). |

## Current bottleneck after Phase 0

The first strict 16GB cold baseline shows the current path is not using the
available GPU memory effectively:

- Decode token rate is only 0.20 tok/s.
- TTFT/prompt eval is 88.61s.
- VRAM peak is 16224 MiB, with at least 15886 MiB still free.
- The cgroup is constantly at its 16,000,000,000 byte memory limit during
  decode and reports `memory.events max=245278`, with no OOM. This means the
  kernel is reclaiming aggressively inside the memory cap.
- The observed process state reached `D` during decode, consistent with IO wait
  or reclaim stalls.

Priority order before code changes:

1. Rerun one profiling baseline with cgroup `memory.stat` sampling to split
   anon/file/kernel memory and with MoE timing/counter envs enabled if available.
2. Use the route cache simulator and MoE trace data to estimate the maximum gain
   from filling the unused ~15.9 GiB VRAM with expert cache/profile preload.
3. Only after the IO/cache bound is quantified, consider math/kernel changes
   such as up/gate id-MMQ.

## Next candidate: Phase 2A expert pack plus VRAM cache

Design timestamp: 2026-07-01 13:02 UTC.

Current bottleneck:

- The accepted strict 16GB baseline decodes at 0.20 tok/s.
- cgroup memory is continuously capped at 16,000,000,000 bytes and reports
  `memory.events max=245278`.
- VRAM has about 15.9 GiB unused.
- The process entered `D` state during decode, consistent with IO wait or memory
  reclaim stalls.

Hypothesis:

Using the existing Kimi IQ3_S expert pack with MoE streaming and allocating a
large VRAM expert cache should reduce page-cache pressure, reduce mmap-backed
expert faults, and move repeated routed expert tensors into GPU memory. This
matches the GPU/VRAM-first gate and should improve token rate without changing
model math.

Candidate configuration:

```sh
GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
GGML_MOE_IO_BACKEND=iouring
GGML_MOE_STREAM=1
GGML_MOE_PARALLEL_EXPERTS=1
GGML_MOE_STAGE_PINNED_SLOTS=8
GGML_MOE_STREAM_FUSED_UP_GATE=1
GGML_MOE_STREAM_BATCH_ONLY=1
GGML_MOE_VRAM_CACHE_MIB=15000
GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
GGML_MOE_MMAP_DONTNEED=1
GGML_MOE_IO_BYTES=8388608
GGML_MOE_BATCH_PROFILE=1
GGML_MOE_BATCH_PROFILE_OUT=<run>/route-profile.csv
GGML_MOE_ROUTE_TRACE_OUT=<run>/route-trace.csv
GGML_MOE_TTFT_TRACE_OUT=<run>/ttft-trace.csv
GGML_MOE_TTFT_TRACE_MAX_EVENTS=200000
```

The first execution is a cold `-n 32` profiling run, not an accepted performance
promotion. It must still pass the France semantic smoke. If it improves speed
and quality holds, rerun the same config as full `-n 96` under the strict 16GB
gate before committing any code changes.

Theoretical upper bound:

- Strict 16GB baseline decode time is 5.01989 s/token.
- Previous same-correctness unconstrained reference was about 1.92-1.96 s/token.
  Removing cgroup page-cache/reclaim stalls alone therefore gives an expected
  ceiling around 0.51-0.52 tok/s.
- Additional VRAM cache hits can only improve beyond that if repeated expert
  tensors avoid SSD read and H2D transfer. The profiling run must report route
  frequency and cache counters before claiming a higher bound.

Acceptance for this candidate:

- `-n 32` profiling run: host RAM including page cache below 16GB, cold start,
  quality pass, and no read failures.
- Promotion run: full `-n 96`, same gates, TTFT <= Phase 0 TTFT * 1.20
  (<= 106331.72 ms), reproducible output, and token rate > 0.20 tok/s.
- If promoted, record exact reproduction and commit/push immediately.

Rollback:

- If output quality regresses, read failures appear, TTFT exceeds the limit, or
  host RAM exceeds the cap, reject the candidate and return to the Phase 0
  baseline config.

Result:

- Rejected by the `-n 32` profiling run at
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-130305Z-n32-phase2a-vram-cache-profile`.
- It did use VRAM as intended: peak 31278 MiB, minimum free 832 MiB.
- It enabled the stream/batched path:
  `[moe_stream] batched up/gate decode path active`.
- It produced route/profile artifacts:
  `route-profile.csv`, `route-trace.csv`, `ttft-trace.csv`, and `cache-sim.txt`.
- Runtime counters:
  - expert pack hits=3406, misses=2158, read_failures=0
  - VRAM cache hits=8340, misses=5564, hit_rate=60.0%
  - pinned staging copies=3406, host_stage=4623.159 ms, H2D=645.748 ms
  - up/gate profile total=31.246 ms/call
- Route simulator:
  - 15GB single cache estimated hit 85.8% from profile counts.
  - Trace replay with protected preload recommended upgate_pct=75, estimated
    hit 70.46%, miss 19.44 GiB.
- Failure:
  - TTFT was 110325.85 ms, exceeding the 106331.72 ms gate.
  - Decode was 0.16 tok/s, slower than the 0.20 tok/s baseline.
  - Output quality failed: `The user asks: "Please introduce France ...`.

Conclusion:

The limiting issue is not just cache capacity. The available VRAM can be filled,
but the current `GGML_MOE_STREAM` / batched up-gate path is not semantically
equivalent to the accepted non-stream path for Kimi. No code or env promotion is
allowed from Phase 2A.

## Next candidate: Phase 2B stream switch isolation

Design timestamp: 2026-07-01 13:12 UTC.

Current bottleneck:

- The correctness baseline is slow because it leaves about 15.9 GiB VRAM unused
  and runs at the cgroup memory limit.
- The first VRAM/cache attempt filled VRAM but failed quality and TTFT after
  activating the stream/batched up-gate path.

Hypothesis:

The quality regression is caused by one of the stream math switches, not by the
VRAM cache allocation itself. Before any cache optimization can be accepted, the
minimum stream switch that changes Kimi output must be isolated.

Isolation order:

1. `GGML_MOE_STREAM=1` only, no expert pack, no VRAM cache, no fused up/gate.
2. `GGML_MOE_STREAM=1` + expert pack, no VRAM cache, no fused up/gate.
3. `GGML_MOE_STREAM=1` + expert pack + VRAM cache, no fused up/gate.
4. Only if the above pass, add `GGML_MOE_STREAM_FUSED_UP_GATE=1`.

Use cold `-n 16` or `-n 32` smoke runs under the same 16GB cgroup. Stop the
isolation sweep as soon as the output starts with meta text, malformed grammar,
or any semantic drift. These are diagnostic runs only; promotion still requires
full `-n 96`.

Theoretical upper bound:

- If `GGML_MOE_STREAM=1` alone already fails quality, cache work cannot be
  accepted until the stream path is fixed.
- If stream-only passes and cache-only fails, the bug is in cache copy/staging.
- If cache-only passes and fused up/gate fails, the bug is in the batched/fused
  up-gate math path.

Acceptance for isolation:

- Host RAM below 16GB, cold start, `read_failures=0`.
- France output must start as a direct answer, not prompt analysis.
- TTFT should be recorded but does not promote the candidate.

Rollback:

- No code changes are made during isolation. Failed env combinations are recorded
  and not reused for promotion.

Result:

- `GGML_MOE_STREAM=1` alone failed the cold `-n 16` smoke:
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-131149Z-n16-phase2b-stream-only`.
- Output was corrupted:
  `The:ayt老爷 grave!!!!!!!!!!!`.
- Disabling the default stream one-cache with `GGML_MOE_STREAM_ONE_CACHE_MIB=0`
  produced the same corrupted output:
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-131722Z-n16-phase2c-stream-cache0`.
- Because the first isolation switch failed, later combinations with expert
  pack, VRAM cache, or fused up/gate are invalid until the base stream path is
  corrected.

Conclusion:

The next real implementation work is correctness debugging in the stream path,
not cache tuning. The stream path must become numerically equivalent enough to
the accepted non-stream path before any VRAM/cache optimization can be accepted.

## Next candidate: Phase 2C stream correctness debug

Design timestamp: 2026-07-01 13:16 UTC.

Current bottleneck:

- Non-stream path is semantically correct but slow under 16GB because it leaves
  VRAM unused and relies on mmap/page-cache behavior.
- Stream path is required to use existing GPU expert cache machinery, but
  `GGML_MOE_STREAM=1` alone corrupts output.

Hypothesis:

The stream one-path has a tensor layout, row mapping, quant type, or
copy/scatter mismatch for Kimi's observed expert types. Fixing the stream path
should unlock the VRAM-cache candidate. The upper bound is not accepted token
rate yet; the first target is matching the non-stream prefix and then recovering
at least the baseline 0.20 tok/s under 16GB.

Debug order:

1. Identify the first expert tensor type and route where stream output diverges.
2. Compare stream one-path output against the normal non-stream reference for a
   single routed expert and a single token.
3. Verify:
   - source weight pointer and byte count,
   - quant type dispatch,
   - `src1` row stride and token selection,
   - destination scatter offset,
   - synchronization before graph consumers read the output,
   - whether the unexpected default 16GiB stream VRAM cache allocation changes
     behavior or just logs a failed allocation.
4. Add the smallest debug instrumentation needed to prove the mismatch.
5. Only after the mismatch is fixed, rerun Phase 2B-1 stream-only smoke.

Theoretical upper bound:

- Correct stream-only decode measured 0.26 tok/s in the failed run, but this
  number is not acceptable because quality failed and TTFT exceeded the gate.
- If correctness is fixed without adding overhead, stream-only could at least
  exceed the 0.20 tok/s baseline. The useful upper bound still comes from
  Phase 2A cache replay: a corrected stream+cache path could approach the
  0.51-0.52 tok/s reclaim-free reference, with further gains only if cache hits
  reduce SSD/H2D waits.

Acceptance:

- `GGML_MOE_STREAM=1` only, cold `-n 16`, starts with a coherent France answer.
- Then cold `-n 32` passes the same quality check.
- Host RAM remains below 16GB, `read_failures=0`, and TTFT is recorded.
- Commit/push only if a code fix is made and these diagnostics pass.

Rollback:

- Revert any debug/fix patch that does not restore stream-only semantic quality
  or that worsens the accepted non-stream baseline.

Result:

- Root cause found: stream one-path used `rows[k].i1` directly as the `src1`
  row index. For the Kimi decode shape, `src1_nb2 == src1_nb1`, so the effective
  `src1.ne[1]` is 1 and the CPU reference uses `i11 = id % ne11 = 0`. Directly
  using route slot values read the wrong activation rows.
- Added default-off CPU compare instrumentation under
  `GGML_MOE_STREAM_COMPARE_CPU`.
- Before fix:
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-133121Z-n2-phase2c-stream-compare`
  showed stream-vs-CPU `max_abs` up to 0.527928 on
  `blk.2.ffn_gate_exps.weight`.
- Fix: stream one-path now derives `src1_ne1` from `src1_nb2 / src1_nb1` and
  uses `rows[k].i1 % src1_ne1`, matching the CPU `mul_mat_id` row selection.
- After fix:
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-133448Z-n2-phase2c-stream-src1fix-compare`
  showed `max_abs <= 0.00151799` and output `France is`.
- `n16` stream-only semantic smoke passed:
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-133739Z-n16-phase2c-src1fix-stream-only`.
- `n32` stream-only semantic smoke passed:
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-134233Z-n32-phase2c-src1fix-stream-only`.
- Not a performance promotion:
  n32 stream-only TTFT was 111.49s and decode was 0.14 tok/s, both worse than
  the Phase 0 performance baseline. The fix is accepted only as default-off
  correctness progress needed before future VRAM/cache work.

Next:

Re-test Phase 2A-style expert pack + VRAM cache with the src1 fix, starting with
cold `-n 32`. Promotion still requires full `-n 96`, host RAM <16GB, quality
pass, TTFT <=106331.72 ms, and token rate >0.20 tok/s.

Result:

- Rerun completed at
  `/root/lfz/runs/vendor-kimi-token-rate/20260701-135107Z-n32-phase2a-src1fix-vram-cache-profile`.
- Quality recovered and VRAM was used as intended.
- Decode improved to 0.22 tok/s versus 0.20 tok/s baseline.
- TTFT failed at 114285.53 ms, above the 106331.72 ms gate.
- Not promoted.

## Next candidate: Phase 2D src1 fix + VRAM cache + 40 CPU threads

Design timestamp: 2026-07-01 13:58 UTC.

Current bottleneck:

- Phase 2A after src1 fix restored quality and improved decode rate to
  0.22 tok/s, but TTFT failed.
- Phase 1A showed `-t 40 -tb 40` reduced TTFT on the accepted non-stream path
  from 88.61s to 65.28s while preserving quality.

Hypothesis:

Combining the corrected VRAM cache path with `-t 40 -tb 40` may keep the
0.22 tok/s decode improvement while reducing prompt eval/TTFT below the
106331.72 ms gate. This changes scheduling only; model math remains the same
as the src1-fixed stream/cache path.

Theoretical upper bound:

- Decode upper bound is the measured Phase 2A+src1fix value, about 0.22 tok/s,
  unless extra CPU threads also reduce staging/reclaim overhead.
- TTFT could improve by up to the Phase 1A observed prompt reduction
  (88.61s -> 65.28s, about 26%). Applying even half of that to the 114.29s
  Phase 2A+src1fix TTFT would bring TTFT near 99.4s, under the gate.

Acceptance:

- Cold `-n 32`: host RAM <16GB, VRAM near full, quality pass, read_failures=0,
  TTFT <=106331.72 ms, token rate >0.20 tok/s.
- If `-n 32` passes, run full cold `-n 96` with the same config.
- Commit/push only after full `-n 96` passes the complete gate.

Rollback:

- If TTFT still fails, quality regresses, or token rate falls to <=0.20 tok/s,
  reject the candidate and do not stack further env changes on it.

Result:

- `/root/lfz/runs/vendor-kimi-token-rate/20260701-135747Z-n32-phase2d-src1fix-vram-cache-t40`
  passed quality, host RAM, VRAM, read failure, and TTFT gates.
- TTFT recovered to 102374.15 ms.
- Decode token rate was 0.20 tok/s, so this is not a performance improvement
  over the Phase 0 baseline and cannot be promoted.

## Next candidate: Phase 2E thread balance sweep

Design timestamp: 2026-07-01 14:03 UTC.

Current bottleneck:

- `-t 24 -tb 24` with src1fix+VRAM cache improves decode to 0.22 tok/s but
  fails TTFT at 114.29s.
- `-t 40 -tb 40` passes TTFT at 102.37s but decode drops to 0.20 tok/s.

Hypothesis:

An intermediate thread count may keep enough CPU parallelism to satisfy TTFT
while avoiding the decode contention/regression seen at 40 threads.

Sweep order:

1. `-t 32 -tb 32`
2. If needed, `-t 28 -tb 28`
3. If needed, `-t 36 -tb 36`

Theoretical upper bound:

- The best observed decode from this path is 0.22 tok/s.
- The required TTFT is <=106331.72 ms.
- A successful intermediate setting only needs to keep decode >0.20 tok/s while
  keeping TTFT below the gate; the expected gain is modest, about 10% on n32.

Acceptance:

- Cold `-n 32`, host RAM <16GB, VRAM near full, quality pass, read_failures=0.
- TTFT <=106331.72 ms.
- Token rate >0.20 tok/s.
- If a sweep point passes, run full cold `-n 96` with that exact setting before
  promotion.

Rollback:

- Reject any sweep point that fails quality, TTFT, or token-rate gate. Do not
  commit performance config until full `-n 96` passes.

Result for `-t 32 -tb 32` smoke:

- `/root/lfz/runs/vendor-kimi-token-rate/20260701-140437Z-n32-phase2e-src1fix-vram-cache-t32`
- Cold `-n 32` under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches` before launch.
- Host RAM peak: 14.901 GiB, including page cache inside the cgroup.
- VRAM peak: 31278 MiB used, 832 MiB free.
- Quality: PASS.
- France answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, philosophy, and cuisine. Its capital, Paris, is famous`
- TTFT: 104061.14 ms, within the 106331.72 ms gate.
- Decode: 31 runs, 4.53227 s/token, 0.22 tok/s.
- MoE cache/read status: read_failures=0, VRAM cache hit_rate=62.3%.
- Decision: pass smoke gate. Next practice is a full cold `-n 96` with the exact
  same configuration before any performance promotion.

Full `-n 96` validation result:

- Result timestamp: 2026-07-01 14:18 UTC.
- `/root/lfz/runs/vendor-kimi-token-rate/20260701-141053Z-n96-phase2e-src1fix-vram-cache-t32`
- Commit/config baseline: `051956009` plan state, code includes
  `9b12713e2 cuda: fix Kimi MoE stream src1 row selection`.
- Cold start: `sync; echo 3 > /proc/sys/vm/drop_caches` before launch.
- Host RAM cgroup: `memory.max=16000000000`, `memory.swap.max=0`.
- Host RAM peak: 14.901 GiB, including page cache inside the cgroup.
- Page cache final: 13.783 GiB.
- VRAM peak: 31278 MiB used, 832 MiB free.
- Quality: PASS.
- France answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, philosophy, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also recognized for its diverse landscapes, from the vineyards of Bordeaux to the beaches of the Riviera, and plays a major role in European and global affairs.<|im_end|> [end of text]`
- TTFT: 105685.10 ms, within the 106331.72 ms gate.
- Decode: 362575.94 ms / 77 runs, 4.70878 s/token, 0.21 tok/s.
- MoE cache/read status: read_failures=0, VRAM cache hit_rate=63.1%.
- Up/gate profile: calls=2157, avg_active=8.00, total=13.489 ms/call.
- Decision: accept Phase 2E as a valid constrained improvement over the Phase 0
  `-n 96` baseline of 0.20 tok/s / 5.01989 s/token. The gain is modest but
  satisfies host RAM, VRAM use, quality, TTFT, and cold-start requirements.

Reproduction:

```sh
RUN=/root/lfz/runs/vendor-kimi-token-rate/$(date -u +%Y%m%d-%H%M%SZ)-n96-phase2e-src1fix-vram-cache-t32
mkdir -p "$RUN"
CG=/sys/fs/cgroup/kimi_phase2e_n96_t32_$$
mkdir "$CG"
echo 16000000000 > "$CG/memory.max"
echo 0 > "$CG/memory.swap.max"
sync
echo 3 > /proc/sys/vm/drop_caches

export GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
export GGML_MOE_IO_BACKEND=iouring
export GGML_MOE_STREAM=1
export GGML_MOE_PARALLEL_EXPERTS=1
export GGML_MOE_STAGE_PINNED_SLOTS=8
export GGML_MOE_STREAM_FUSED_UP_GATE=1
export GGML_MOE_STREAM_BATCH_ONLY=1
export GGML_MOE_VRAM_CACHE_MIB=15000
export GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
export GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
export GGML_MOE_MMAP_DONTNEED=1
export GGML_MOE_IO_BYTES=8388608
export GGML_MOE_BATCH_PROFILE=1
export GGML_MOE_BATCH_PROFILE_OUT="$RUN/route-profile.csv"
export GGML_MOE_ROUTE_TRACE_OUT="$RUN/route-trace.csv"
export GGML_MOE_TTFT_TRACE_OUT="$RUN/ttft-trace.csv"
export GGML_MOE_TTFT_TRACE_MAX_EVENTS=300000

PROMPT='<|im_user|>user<|im_middle|>Please introduce France in a short paragraph.<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>'
( echo $BASHPID > "$CG/cgroup.procs"
  /root/lfz/llama.cpp-vendor-kimi/build-cuda-batch/bin/llama-completion \
    --defer-experts --fit off -ngl 99 --special \
    -m /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf \
    -c 512 -n 96 --temp 0 --top-p 1.0 --top-k 1 --seed 1 \
    --no-display-prompt -no-cnv -t 32 -tb 32 -p "$PROMPT" \
    > "$RUN/stdout.txt" 2> "$RUN/stderr.txt" )
```

## Parallel low-risk candidate: Phase 1A non-stream CPU thread tuning

Design timestamp: 2026-07-01 13:24 UTC.

Current bottleneck:

- GPU stream path is unsafe for Kimi, so the accepted path remains non-stream
  CPU fallback for deferred experts.
- The baseline uses `-t 24 -tb 24` on a 40-thread host.
- The baseline spends 401.59s in decode for 80 runs under the 16GB cgroup.

Hypothesis:

Increasing CPU worker threads to match the available 40 logical CPUs may improve
the accepted non-stream fallback token rate without changing model math, output
semantics, VRAM allocation, or page-cache policy. This does not satisfy the
long-term GPU/VRAM-first goal, but it is allowed as a low-risk tuning step while
the GPU stream path is known unsafe.

Candidate configuration:

```sh
# Same as Phase 0 baseline, only changing:
-t 40 -tb 40
```

Theoretical upper bound:

- If CPU vec-dot work scales perfectly from 24 to 40 threads, the CPU portion
  could improve by up to 1.67x.
- Because the run is also constrained by page-cache reclaim and IO wait, the
  realistic upper bound is lower. A cold `-n 32` smoke should first estimate
  whether the gain is material before spending a full `-n 96` run.

Acceptance:

- Cold `-n 32` smoke must pass quality, host RAM < 16GB, and TTFT <= 106331.72 ms.
- If token rate improves over the baseline trend, run full cold `-n 96`.
- Promote only if full `-n 96` improves over 0.20 tok/s, keeps the France output
  correct, keeps TTFT within +20%, and records exact reproduction.

Rollback:

- If token rate is flat/slower, TTFT fails, or quality changes, reject the
  tuning and keep `-t 24 -tb 24`.

Result:

- Rejected for token-rate promotion.
- `/root/lfz/runs/vendor-kimi-token-rate/20260701-132317Z-n32-phase1a-threads40`
  passed quality and host RAM gates.
- TTFT improved to 65280.14 ms, but decode remained 0.20 tok/s
  (5016.38 ms/token), effectively the same as the Phase 0 baseline.
- Conclusion: decode is not materially improved by more CPU threads; the
  limiting bucket is page-cache/IO/reclaim and the unsafe stream path must be
  fixed for VRAM/cache gains.

## Immediate next action

Implement Phase 2H: make GPU down batch Kimi-correct on a tiny decode-only
validation before any performance run.

## Next candidate: Phase 2F parallel up/gate streams

Design timestamp: 2026-07-01 14:29 UTC.

Current bottleneck:

- Accepted Phase 2E full `-n 96` spends 362575.94 ms in decode for 77 runs:
  4.70878 s/token, 0.21 tok/s.
- Up/gate profile: 2157 calls, total=13.489 ms/call, with up=6.581 ms and
  gate=6.828 ms.
- This is about 28.0 up/gate calls per decoded token, or about 378 ms/token in
  the measured up/gate bucket.
- VRAM cache is near full but not complete: 31278 MiB used, 832 MiB free,
  hit_rate=63.1%, misses=12734.
- Miss staging/H2D is smaller than compute but still measurable:
  host_stage=13760.938 ms and h2d=2143.731 ms across the `-n 96` run.

Hypothesis:

Enable the existing parallel up/gate path:

```sh
GGML_MOE_STREAM_UP_GATE_PARALLEL=1
GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE=1
```

This should keep the same numerical path but launch up and gate work on separate
CUDA streams, while staging up/gate misses from separate CPU threads/rings. The
change targets the largest directly compressible bucket that remains after the
VRAM cache improvement.

Theoretical upper bound:

- If only the up/gate kernels parallelize, the per-call bucket can drop from
  about `6.581 + 6.828 = 13.409 ms` to `max(6.581, 6.828) = 6.828 ms`.
- With 28.0 calls/token, the hard compute-only saving is about
  `(13.409 - 6.828) * 28.0 = 184 ms/token`.
- Starting from 4.70878 s/token, the compute-only upper-bound token rate is
  `1 / (4.70878 - 0.184) = 0.221 tok/s`.
- If parallel staging also halves the measured staging/H2D miss bucket, the
  additional upper-bound saving is about
  `(13760.938 + 2143.731) ms / 77 / 2 = 103 ms/token`, giving a best case near
  `1 / (4.70878 - 0.184 - 0.103) = 0.226 tok/s`.
- Therefore this candidate is expected to be modest, roughly 5-8%, but it is
  mathematically safe and aligned with the measured bottleneck.

Acceptance:

- Cold `-n 32` smoke under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches`.
- Host RAM must remain below the 16GB cgroup cap, including page cache.
- VRAM should remain near full without OOM; expected free VRAM may be slightly
  lower due additional stream/ring buffers.
- France answer must be semantically correct and coherent.
- TTFT <= 106331.72 ms.
- Decode token rate must exceed the accepted Phase 2E smoke trend of 0.22 tok/s
  or at minimum show a clear per-token reduction without TTFT/quality risk.
- If smoke passes, run full cold `-n 96` before promotion.

Rollback:

- Reject if output quality changes, TTFT exceeds the gate, read_failures become
  nonzero, cgroup OOM occurs, or full `-n 96` does not improve over accepted
  Phase 2E `0.21 tok/s / 4.70878 s/token`.

Result:

- Result timestamp: 2026-07-01 14:28 UTC.
- `/root/lfz/runs/vendor-kimi-token-rate/20260701-142246Z-n32-phase2f-parallel-upgate`
- Cold `-n 32` under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches` before launch.
- Host RAM peak: 14.901 GiB, including page cache inside the cgroup.
- VRAM peak: 31278 MiB used, 832 MiB free.
- Quality: PASS.
- France answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, philosophy, and cuisine. Its capital, Paris, is famous`
- TTFT: 103570.58 ms, within the 106331.72 ms gate.
- Decode: 141731.47 ms / 31 runs, 4.57198 s/token, 0.22 tok/s.
- Comparison: accepted Phase 2E smoke was 4.53227 s/token, so this is slower.
- Read/cache status: read_failures=0, VRAM cache hit_rate=62.3%.
- Parallel path did activate:
  `IQ2_S parallel up/gate streams active` and
  `up/gate parallel CPU staging active`.
- Gap analysis:
  - Phase 2E smoke up/gate total was 13.652 ms/call.
  - Phase 2F parallel up/gate total was 13.642 ms/call, effectively unchanged.
  - The split profile shows `up=8.390 ms`, `gate=3.745 ms`,
    `up_wait=5.158 ms`, and `gate_wait=7.112 ms`; stream/event wait and
    staging synchronization consume the expected overlap.
  - iouring became active (`iouring_reads=2766`) but did not improve end-to-end
    decode because average inflight was only about 2.72 jobs and the up/gate
    bucket remained flat.
- Decision: reject Phase 2F; do not run full `-n 96` and do not promote these
  env variables.

## Next candidate: Phase 2G GPU down batch + GPU handoff

Design timestamp: 2026-07-01 14:36 UTC.

Current bottleneck:

- Accepted Phase 2E full `-n 96` decode is 362575.94 ms / 77 runs:
  4.70878 s/token, 0.21 tok/s.
- TTFT trace accounts for only about 57.0s of the run:
  - `runtime_load`: 12734 events, 27839.388 ms.
  - `call_upgate`: 2157 events, 29143.436 ms.
- The remaining about 305.6s over 77 decoded tokens is about 3.97 s/token and
  is not explained by up/gate or expert-pack staging.
- Code inspection shows the CPU bridge only calls `ggml_cuda_moe_stream_batch`
  for down tensors when `GGML_MOE_STREAM_DOWN_BATCH` is set. Phase 2E did not
  set it, so down expert matmul remained on the CPU fallback path.

Hypothesis:

Enable:

```sh
GGML_MOE_STREAM_DOWN_BATCH=1
GGML_MOE_GPU_HANDOFF=1
```

`GGML_MOE_STREAM_DOWN_BATCH=1` sends `ffn_down_exps` MoE matmuls through the CUDA
batch path. `GGML_MOE_GPU_HANDOFF=1` allows the up/gate fused activation already
on GPU to be consumed by the down batch path without an unnecessary host round
trip when the graph adjacency matches.

Theoretical upper bound:

- If down batch only removes half of the unexplained 3.97 s/token CPU bucket,
  token time could drop from 4.70878 s/token to about 2.72 s/token, or
  0.37 tok/s.
- If it removes most of that bucket and the remaining cost is the traced
  up/gate+load bucket of about `57.0s / 77 = 0.74 s/token` plus graph overhead,
  the optimistic bound is around 1 tok/s.
- It cannot be assumed correct or fast without measurement because down batch
  changes the largest numerical path and relies on route indexing, handoff
  shape matching, cache slot lifetime, and output scatter semantics.

Acceptance:

- Cold `-n 32` smoke under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches`.
- Host RAM must remain below the 16GB cgroup cap, including page cache.
- VRAM should remain near full without OOM.
- France answer must be semantically correct and coherent.
- TTFT <= 106331.72 ms.
- `read_failures=0`.
- Logs must show the down CUDA path was actually active, either via
  `GPU handoff consumed`, `profile: calls=...`, or down batch route/cache events.
- If smoke passes and improves decode materially, run full cold `-n 96` before
  promotion.

Rollback:

- Reject if quality changes, TTFT exceeds the gate, down batch silently declines,
  cgroup OOM occurs, or decode does not improve over accepted Phase 2E.

Result:

- Result timestamp: 2026-07-01 14:36 UTC.
- `/root/lfz/runs/vendor-kimi-token-rate/20260701-143039Z-n32-phase2g-down-batch-handoff`
- Cold `-n 32` under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches` before launch.
- Run was terminated intentionally after quality failure; exit code 143.
- Host RAM peak: 14.901 GiB, including page cache inside the cgroup.
- VRAM peak: 31336 MiB used, 774 MiB free.
- Quality: FAIL.
- Partial answer before termination:
  `France the. with myol:,ing andn of of`
- TTFT/decode: no valid final `common_perf_print` because the run was aborted
  after quality corruption.
- Read status before abort: read_failures=0.
- Down path diagnostics:
  - `down batch declined` occurred 727 times.
  - Prompt/multi-token phase declined with `reason=multirow_not_supported`,
    e.g. active routes larger than one row for the same expert.
  - Decode phase declined with `reason=launch_moe_mmvq_compact_batch` for many
    `ffn_down_exps` tensors.
- Decision: reject Phase 2G. Do not promote
  `GGML_MOE_STREAM_DOWN_BATCH=1` or `GGML_MOE_GPU_HANDOFF=1`.
- Gap analysis:
  - The high-level bottleneck finding remains valid: accepted Phase 2E leaves
    most decode time outside traced up/gate/load buckets.
  - The existing vendor down batch implementation is not Kimi-safe yet because
    prompt multirow routing is unsupported and the decode launch path fails for
    active=8 Kimi down tensors.
  - A future down optimization must first fix down batch correctness on a tiny
    comparison run before any performance measurement.

## Next candidate: Phase 2H decode-only down batch correctness

Design timestamp: 2026-07-01 15:02 UTC.

Current bottleneck:

- Phase 2G showed that down batch is the right target, but the existing path is
  not correct for Kimi.
- Failure modes from the aborted smoke:
  - Prompt/multi-token phase: `reason=multirow_not_supported`.
  - Decode phase: `reason=launch_moe_mmvq_compact_batch` with active=8 on many
    `ffn_down_exps` tensors.
  - Output corrupted to `France the. with myol:,ing andn of of`.
- Code inspection:
  - CPU bridge calls down batch whenever `GGML_MOE_STREAM_DOWN_BATCH` is set.
  - `ggml_cuda_moe_stream_batch` rejects any expert with more than one routed
    row, so prompt-phase down must be excluded until a true multirow GPU down
    implementation exists.
  - `launch_moe_mmvq_compact_batch` only allows `IQ3_XXS` and `IQ2_S`, while
    Kimi down tensors include `IQ3_S` (`type=23`) as well as `IQ2_S`
    (`type=11`).

Hypothesis:

First make the down batch path decode-only and type-complete:

1. In `ggml_cuda_moe_stream_batch`, if any expert has `matrix_row_counts[e] > 1`,
   decline without changing output. This keeps prompt/multi-token phase on the
   known-correct CPU fallback.
2. Add `IQ3_S` to `launch_moe_mmvq_compact_batch`, matching CUDA MMVQ support
   already present in `ggml/src/ggml-cuda/mmvq.cu`.
3. Run a tiny cold correctness smoke with `-n 2` or `-n 4`,
   `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_GPU_HANDOFF=0` initially, and
   `GGML_MOE_STREAM_DECLINE_DEBUG=1`. The expected behavior is prompt down
   declines cleanly for multirow, decode down activates for single-token rows,
   and the answer prefix remains semantically correct.
4. Only after non-handoff down correctness is demonstrated should handoff be
   reintroduced. Handoff can change src1 row addressing and must be verified
   separately.

Theoretical upper bound:

- This phase is a correctness gate, not a performance promotion.
- If decode-only down batch becomes correct, the later performance bound remains
  the Phase 2G estimate: removing half of the unexplained 3.97 s/token CPU
  bucket would reach about 0.37 tok/s; removing most of it could approach
  about 1 tok/s before other graph overhead dominates.

Acceptance:

- Build succeeds on the remote CUDA build.
- Tiny cold smoke under the 16GB cgroup starts from dropped page cache.
- Host RAM remains below the 16GB cap.
- Output for the France prompt is not corrupted; for a short smoke it must at
  least start coherently, e.g. `France is...`.
- Prompt-phase down declines may occur only because of multirow routing; decode
  phase must show either down profile calls or no `launch_moe_mmvq_compact_batch`
  failures.
- No performance promotion, commit, or full `n96` run until correctness is
  stable.

Rollback:

- If the tiny smoke corrupts output, fails to build, or still reports decode
  `launch_moe_mmvq_compact_batch` failures, revert the code changes before
  trying new performance settings.

Intermediate result:

- `/root/lfz/runs/vendor-kimi-token-rate/20260701-144022Z-n4-phase2h-down-decode-iq3s`
  built and ran with an initial `IQ3_S` whitelist expansion.
- Output prefix was coherent: `France is a country`.
- The attempt is not acceptable: decode still reported
  `launch_moe_mmvq_compact_batch` failures and no down profile calls.
- Follow-up diagnostics:
  - Entry logging showed up/gate compact calls use `type=18` (`IQ3_XXS`).
  - Down compact calls use `type=11` and `type=23`, which map to `Q3_K` and
    `IQ4_XS`, not `IQ2_S`/`IQ3_S`.
  - `ggml/src/ggml-cuda/mmvq.cu` already has CUDA MMVQ switch cases for both
    `Q3_K` and `IQ4_XS`.
- Revised implementation target:
  - Keep prompt multirow declines on CPU fallback.
  - Extend `launch_moe_mmvq_compact_batch` to accept `Q3_K` and `IQ4_XS`.
  - Remove temporary unconditional debug before committing.
  - Re-run tiny cold `-n 4`; acceptance requires coherent output and zero
    decode `launch_moe_mmvq_compact_batch` failures.

Correctness result:

- Result timestamp: 2026-07-01 15:09 UTC.
- Code change: `launch_moe_mmvq_compact_batch` now accepts `Q3_K` and
  `IQ4_XS` in addition to the existing compact MMVQ types.
- `/root/lfz/runs/vendor-kimi-token-rate/20260701-150614Z-n4-phase2h-down-q3-iq4`
- Cold `-n 4` under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches` before launch.
- Host RAM peak: 14.901 GiB, including page cache inside the cgroup.
- VRAM peak: 31338 MiB used, 772 MiB free.
- Quality: PASS.
- France prefix: `France is a country`.
- TTFT: 101830.07 ms, within the 106331.72 ms gate.
- Decode: 14378.79 ms / 3 runs, 4.79293 s/token, 0.21 tok/s.
- Diagnostics:
  - `launch_moe_mmvq_compact_batch` failures: 0.
  - `multirow_not_supported` declines: 52, all from prompt/multi-token routing
    that intentionally remains on CPU fallback.
  - Down profile active: true.
- Decision: accept Phase 2H as a correctness fix only. Next practice is a cold
  `-n 32` performance smoke with `GGML_MOE_STREAM_DOWN_BATCH=1` but still
  without `GGML_MOE_GPU_HANDOFF`; handoff remains a separate correctness risk.

Performance smoke result:

- Result timestamp: 2026-07-01 15:15 UTC.
- `/root/lfz/runs/vendor-kimi-token-rate/20260701-151003Z-n32-phase2h-down-batch-nohandoff`
- Commit: `a1702641b`.
- Cold `-n 32` under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches` before launch.
- Enabled `GGML_MOE_STREAM_DOWN_BATCH=1`; did not enable
  `GGML_MOE_GPU_HANDOFF`.
- Host RAM peak: 14.901 GiB, including page cache inside the cgroup.
- VRAM peak: 31338 MiB used, 772 MiB free.
- Quality: PASS.
- France answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- TTFT: 102309.52 ms, within the 106331.72 ms gate.
- Decode: 95613.73 ms / 31 runs, 3.08431 s/token, 0.32 tok/s.
- Read/correctness status: read_failures=0, launch_failures=0, down_profile=true.
- Down profile: calls=1644, avg_active=8.00, total=9.228 ms/call.
- Up/gate profile: calls=869, total=18.938 ms/call.
- Decision: pass smoke gate. Next practice is full cold `-n 96` with the exact
  same config before performance promotion.

Full `-n 96` validation result:

- Result timestamp: 2026-07-01 15:23 UTC.
- `/root/lfz/runs/vendor-kimi-token-rate/20260701-151516Z-n96-phase2h-down-batch-nohandoff`
- Commit/config: `83ab7ce63`, code includes
  `a1702641b cuda: allow Kimi down batch compact quant types`.
- Cold `-n 96` under `memory.max=16000000000`, `memory.swap.max=0`, and
  `drop_caches` before launch.
- Enabled `GGML_MOE_STREAM_DOWN_BATCH=1`; did not enable
  `GGML_MOE_GPU_HANDOFF`.
- Host RAM peak: 14.901 GiB, including page cache inside the cgroup.
- Page cache final: 13.815 GiB.
- VRAM peak: 31338 MiB used, 772 MiB free.
- Quality: PASS.
- France answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- TTFT: 98846.25 ms, within the 106331.72 ms gate.
- Decode: 295113.58 ms / 85 runs, 3.47192 s/token, 0.29 tok/s.
- Read/correctness status: read_failures=0, launch_failures=0, down_profile=true.
- Down profile: calls=4506, avg_active=8.00, stage=11.616 ms,
  kernel=0.115 ms, d2h=0.015 ms, scatter=0.034 ms, total=11.781 ms/call.
- Up/gate profile: calls=2381, total=22.785 ms/call.
- Cache/staging: VRAM cache hit_rate=45.5%, pinned staging
  host_stage=50153.894 ms and h2d=7758.831 ms.
- Decision: accept Phase 2H as a valid constrained performance improvement.
  Compared with accepted Phase 2E full `-n 96` (0.21 tok/s, 4.70878 s/token),
  this improves to 0.29 tok/s, 3.47192 s/token while satisfying host RAM, VRAM,
  quality, TTFT, read failure, launch failure, and cold-start gates.

Reproduction:

```sh
RUN=/root/lfz/runs/vendor-kimi-token-rate/$(date -u +%Y%m%d-%H%M%SZ)-n96-phase2h-down-batch-nohandoff
mkdir -p "$RUN"
CG=/sys/fs/cgroup/kimi_phase2h_n96_$$
mkdir "$CG"
echo 16000000000 > "$CG/memory.max"
echo 0 > "$CG/memory.swap.max"
sync
echo 3 > /proc/sys/vm/drop_caches

export GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
export GGML_MOE_IO_BACKEND=iouring
export GGML_MOE_STREAM=1
export GGML_MOE_PARALLEL_EXPERTS=1
export GGML_MOE_STAGE_PINNED_SLOTS=8
export GGML_MOE_STREAM_FUSED_UP_GATE=1
export GGML_MOE_STREAM_BATCH_ONLY=1
export GGML_MOE_STREAM_DOWN_BATCH=1
export GGML_MOE_VRAM_CACHE_MIB=15000
export GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
export GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
export GGML_MOE_MMAP_DONTNEED=1
export GGML_MOE_IO_BYTES=8388608
export GGML_MOE_BATCH_PROFILE=1
export GGML_MOE_BATCH_PROFILE_OUT="$RUN/route-profile.csv"
export GGML_MOE_ROUTE_TRACE_OUT="$RUN/route-trace.csv"
export GGML_MOE_TTFT_TRACE_OUT="$RUN/ttft-trace.csv"
export GGML_MOE_TTFT_TRACE_MAX_EVENTS=300000

PROMPT='<|im_user|>user<|im_middle|>Please introduce France in a short paragraph.<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>'
( echo $BASHPID > "$CG/cgroup.procs"
  /root/lfz/llama.cpp-vendor-kimi/build-cuda-batch/bin/llama-completion \
    --defer-experts --fit off -ngl 99 --special \
    -m /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf \
    -c 512 -n 96 --temp 0 --top-p 1.0 --top-k 1 --seed 1 \
    --no-display-prompt -no-cnv -t 32 -tb 32 -p "$PROMPT" \
    > "$RUN/stdout.txt" 2> "$RUN/stderr.txt" )
```

Next bottleneck:

- Down compute is no longer the bottleneck: kernel is only 0.115 ms/call.
- The next largest directly visible bucket is down/upgate staging and cache
  misses: VRAM cache hit_rate=45.5%, host_stage=50153.894 ms,
  h2d=7758.831 ms, plus up/gate total=22.785 ms/call.
- The next candidate should target cache partitioning/policy or safe handoff
  only after a separate handoff correctness test.

## Next candidate: Phase 2I GPU handoff correctness after down fix

Design timestamp: 2026-07-01 15:27 UTC.

Current bottleneck:

- Phase 2H made down batch correct and improved full `-n 96` to 0.29 tok/s.
- Down kernel itself is tiny: 0.115 ms/call.
- Remaining visible costs are staging/cache misses and up/gate work:
  host_stage=50153.894 ms, h2d=7758.831 ms, VRAM cache hit_rate=45.5%,
  up/gate total=22.785 ms/call.
- `GGML_MOE_GPU_HANDOFF=1` previously corrupted output in Phase 2G, but that
  experiment also had broken down batch. It must be re-tested after the down
  compact type fix.

Hypothesis:

Enable `GGML_MOE_GPU_HANDOFF=1` together with the accepted Phase 2H down batch.
This should let the up/gate fused GPU buffer feed down batch directly when
shape and pointer matching work, avoiding a host round trip for that adjacency.

Theoretical upper bound:

- Handoff does not address the large expert weight staging bucket, so it cannot
  plausibly get near the larger cache-policy gains.
- It can only save small activation transfers and synchronization around the
  up/gate to down handoff. The expected gain is modest, likely below 5%, unless
  it also removes hidden graph synchronization.
- Because Phase 2G showed severe corruption with handoff, this phase is first a
  correctness gate, not a performance promotion.

Acceptance:

- Tiny cold `-n 4` smoke under `memory.max=16000000000`, `memory.swap.max=0`,
  and `drop_caches`.
- Host RAM remains below the 16GB cap; VRAM remains near full without OOM.
- France output starts coherently, e.g. `France is...`.
- `launch_failures=0`, `read_failures=0`, and `down_profile=true`.
- Logs should show `GPU handoff consumed`.
- If and only if tiny smoke passes, run cold `-n 32` smoke with the same config.
- Full `-n 96` promotion requires quality pass, TTFT <=106331.72 ms, and decode
  faster than accepted Phase 2H 0.29 tok/s / 3.47192 s/token.

Rollback:

- Reject immediately if output corruption returns, handoff does not activate,
  launch failures appear, TTFT fails, or token rate is not better than Phase 2H.

Result timestamp: 2026-07-01 15:24 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-152433Z-n4-phase2i-handoff-correctness`

Measured result:

- Commit/config: `9498d1452`, accepted Phase 2H config plus
  `GGML_MOE_GPU_HANDOFF=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 103601.35 ms, inside the 106331.72 ms gate.
- Decode: 26878.70 ms / 3 runs, 8.95957 s/token, 0.11 tok/s.
- `launch_failures=0`, `down_profile=true`.
- Handoff activated: log contains
  `GPU handoff consumed: ne00=2048 dst_cols=8`.
- Automated quality script printed PASS, but the actual France answer was:
  `France isneedator`

Decision:

- Reject Phase 2I immediately on semantic quality.
- Do not run the planned `-n 32` or `-n 96` handoff tests.
- This fails the hard correctness gate because the answer is not a coherent
  short paragraph and is clearly corrupted, despite the prefix-based quality
  script returning PASS.
- Future gates must treat the automated quality flag as a first filter only.
  The recorded answer text itself must be checked for semantic correctness and
  coherence before accepting any speedup.

## Next candidate: Phase 2J profile-guided VRAM cache protection

Design timestamp: 2026-07-01 15:39 UTC.

Current bottleneck:

- Accepted Phase 2H full `-n 96` spends most remaining visible time in two
  buckets:
  - Down batch staging: 4506 calls, 11.616 ms average stage time, about 52.3s
    total visible stage time.
  - Up/gate compute: 2381 calls, 22.785 ms average total, about 54.2s total.
- Down batch kernel itself is only 0.115 ms/call, so optimizing down math is no
  longer the next highest-return path.
- VRAM cache hit rate is only 45.5% under a 15000 MiB cache:
  hits=33716, misses=40316.
- Route profile size distribution from the accepted Phase 2H run:
  - 4702208-byte entries: 24480 uses, 5040 unique, 107.20 GiB logical traffic.
  - 5619712-byte entries: 13616 uses, 3256 unique, 71.26 GiB logical traffic.
  - 6307840-byte entries: 27888 uses, 5827 unique, 163.83 GiB logical traffic.
  - 7798784-byte entries: 8160 uses, 1716 unique, 59.27 GiB logical traffic.
- The down cache slot size is 7.44 MiB with 2016 slots, while the observed down
  unique set is much larger than the cache. Plain LRU is likely evicting hot
  experts during the long decode sequence.

Hypothesis:

Use the accepted Phase 2H route profile as a cold-start profile input and enable
profile-aware cache protection/eviction:

- `GGML_MOE_VRAM_PROFILE=<Phase 2H route-profile.csv>`
- `GGML_MOE_VRAM_PROFILE_PROTECT=1`
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=20`
- `GGML_MOE_VRAM_CACHE_POLICY=profile_lfu_lru`

This should pin the highest-frequency experts while reserving 20% of slots for
new traffic, and should prevent the hottest cached experts from being evicted by
one-off experts. This is a low-code-risk config experiment because it uses
existing cache policy code.

Theoretical upper bound:

- Phase 2H down visible stage time is about 52.3s inside a 295.1s decode.
- If profile protection improves down cache hit rate from 45.5% to 70%, misses
  drop by about 45% and the visible down stage bucket could shrink by about
  23.5s.
- The full `-n 96` decode lower bound would then be roughly
  295.1s - 23.5s = 271.6s for 85 decode runs, or 3.20 s/token / 0.31 tok/s.
- This cannot reach 5 tok/s by itself; it is a cache/staging step before the
  larger up/gate compute work.

Acceptance:

- First run a cold `-n 4` smoke under `memory.max=16000000000`,
  `memory.swap.max=0`, and `drop_caches`.
- Host RAM must remain under the 16GB cgroup cap including page cache.
- VRAM should remain near full without OOM.
- TTFT must remain <=106331.72 ms.
- The actual France answer text must be semantically correct and coherent; the
  automated quality flag alone is not sufficient after Phase 2I.
- Logs must show profile preload/cache policy activity and no read failures or
  launch failures.
- If `-n 4` passes, run cold `-n 32`; promote to `-n 96` only if token rate is
  better than accepted Phase 2H while satisfying all gates.

Rollback:

- Reject if profile preload increases TTFT above the gate, output quality fails,
  host RAM exceeds the cgroup cap, VRAM OOMs, launch/read failures appear, or
  token rate is not better than Phase 2H at the same token count.

Result timestamp: 2026-07-01 15:31 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-153115Z-n4-phase2j-profile-cache`

Measured result:

- Commit/config: `7941c8d40`, accepted Phase 2H config plus profile-guided
  cache:
  - `GGML_MOE_VRAM_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/20260701-151516Z-n96-phase2h-down-batch-nohandoff/route-profile.csv`
  - `GGML_MOE_VRAM_PROFILE_PROTECT=1`
  - `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=20`
  - `GGML_MOE_VRAM_CACHE_POLICY=profile_lfu_lru`
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 103969.80 ms, inside the 106331.72 ms gate but close to the limit.
- Decode: 16782.27 ms / 3 runs, 5.59409 s/token, 0.18 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`.
- Profile activity: enabled.
- Cache policy diag: enabled.
- Cache result: hits=1338, misses=1190, preloads=1430, pinned=1430,
  hit_rate=52.9%.
- Pinned staging: copies=2673, host_stage=4087.613 ms, h2d=582.502 ms.
- Cache policy diag:
  profile_count_lookups=1275, hits=1263, inserted_nonzero=2917,
  inserted_avg=14.55, evictions=604, victim_nonzero=592,
  victim_avg=2.34.

Decision:

- The `-n 4` smoke satisfies the hard gates and proves the profile policy is
  active.
- Continue to cold `-n 32` with the same config before any promotion.
- Watch TTFT carefully because profile preload has only about 2.36s headroom
  under the gate.

Result timestamp: 2026-07-01 15:35 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-153506Z-n32-phase2j-profile-cache`

Measured result:

- Commit/config: `a96d7d0b0`, same profile-guided cache config as the passing
  `-n 4` smoke.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 102740.62 ms, inside the 106331.72 ms gate.
- Decode: 131577.12 ms / 31 runs, 4.24442 s/token, 0.23560 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, art, and culture. It is the largest nation in the EU by area and power, with`
- `launch_failures=0`, `read_failures=0`.
- Cache: hits=13487, misses=13457, preloads=1430, pinned=1430,
  hit_rate=50.1%.
- Pinned staging: copies=11887, host_stage=17641.260 ms, h2d=2583.955 ms.
- Cache policy diag:
  profile_count_lookups=13542, hits=10716, inserted_nonzero=12370,
  inserted_avg=6.35, evictions=12871, victim_nonzero=10045,
  victim_avg=3.46.
- Up/gate profile: calls=869, total=26.286 ms/call.
- Down profile: calls=1644, stage=14.475 ms/call, total=14.645 ms/call.

Comparison against accepted Phase 2H `-n 32`:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token, 0.32 tok/s.
- Phase 2J `-n 32`: 131577.12 ms / 31 runs, 4.24442 s/token, 0.23560 tok/s.
- Profile-guided cache is slower despite a modest hit-rate increase.
- The extra preload/protection overhead and slower per-call stage/upgate timings
  outweigh the reduced miss rate.

Decision:

- Reject Phase 2J.
- Do not run `-n 96` for this config.
- Keep accepted Phase 2H as the current best valid configuration.
- Next design should target a direct per-call overhead reduction instead of
  broad profile preload. The top visible candidates after this rejection are:
  reducing cache lookup/eviction overhead in the 2016-slot linear scans, or
  reducing up/gate compute cost, since up/gate remains about 54s of the Phase 2H
  full `-n 96` decode.

## Next candidate: Phase 2K vendor MMQ up/gate path

Design timestamp: 2026-07-01 15:46 UTC.

Current bottleneck:

- Accepted Phase 2H full `-n 96`:
  - Up/gate profile: 2381 calls, 22.785 ms/call, about 54.2s total.
  - Down profile: 4506 calls, 11.781 ms/call, about 53.1s total, but the down
    kernel itself is only 0.115 ms/call and most of this bucket is staging.
- Phase 2J showed that broad profile preloading increased per-call overhead and
  was slower at `-n 32`, so the next attempt should not add more preload work.
- Existing code already has an up/gate vendor MMQ path guarded by
  `GGML_MOE_STREAM_UP_GATE_FUSED_MMQ=1` for `IQ2_S` / `IQ3_XXS`. Phase 2H logs
  show the current up/gate path is `IQ2_S batched MMVQ up/gate path active`,
  so this is applicable to Kimi's up/gate tensors.

Hypothesis:

Enable `GGML_MOE_STREAM_UP_GATE_FUSED_MMQ=1` on top of the accepted Phase 2H
config. This changes only the up/gate compute path while keeping the accepted
down batch, 16GB host-RAM cap, expert pack, and VRAM cache settings.

Theoretical upper bound:

- If vendor MMQ halves the up/gate compute bucket, the full `-n 96` decode could
  save about 27s from the 295.1s Phase 2H decode, reaching roughly
  268.1s / 85 runs = 3.15 s/token, or 0.32 tok/s.
- If vendor MMQ is 3x faster for up/gate, the up/gate bucket drops from about
  54.2s to 18.1s, saving 36.1s and reaching about 259.0s / 85 runs =
  3.05 s/token, or 0.33 tok/s.
- This still cannot reach 5 tok/s alone; it is one of the visible high-cost
  buckets that must be reduced before larger architecture changes.

Correctness risk:

- This path changes quantized matmul implementation for up/gate. If it selects
  the wrong row, stride, slot, or quantized source layout, it can silently
  corrupt generation like the rejected handoff path.
- Therefore start with a cold `-n 4` smoke and require actual semantic output,
  not just a prefix match.

Acceptance:

- First run cold `-n 4` under `memory.max=16000000000`, `memory.swap.max=0`,
  and `drop_caches`.
- Host RAM must remain under the 16GB cgroup cap including page cache.
- VRAM should remain near full without OOM.
- TTFT must remain <=106331.72 ms.
- The actual answer for `Please introduce France in a short paragraph.` must be
  semantically correct and coherent.
- Logs must show `vendor MMQ up/gate path active`.
- `launch_failures=0`, `read_failures=0`, and down batch profile remains active.
- If `-n 4` passes, run cold `-n 32`; promote to `-n 96` only if token rate is
  better than accepted Phase 2H at the same token count while all gates pass.

Rollback:

- Reject immediately if output corruption appears, vendor MMQ does not activate,
  TTFT exceeds the gate, RAM/VRAM gates fail, launch/read failures appear, or
  the `-n 32` rate is not better than accepted Phase 2H.

Result timestamp: 2026-07-01 15:41 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-154139Z-n4-phase2k-vendor-mmq-upgate`

Measured result:

- Commit/config: `6280c4f6d`, accepted Phase 2H config plus
  `GGML_MOE_STREAM_UP_GATE_FUSED_MMQ=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 103384.78 ms, inside the 106331.72 ms gate.
- Decode: 18614.36 ms / 3 runs, 6.20479 s/token, 0.16117 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- Vendor MMQ activated: log contains `vendor MMQ up/gate path active: type=18`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: hits=733, misses=1795, hit_rate=29.0%.
- Pinned staging: copies=1782, host_stage=2636.471 ms, h2d=388.503 ms.
- Up/gate profile: calls=1, total=121.721 ms/call.
- Down profile: calls=160, stage=13.286 ms/call, total=13.465 ms/call.

Decision:

- The `-n 4` smoke passes the hard correctness/TTFT/memory gates and proves
  vendor MMQ activation.
- The single sampled up/gate call is much slower than Phase 2H, but `-n 4` has
  only one up/gate profile call, so run the planned cold `-n 32` before final
  rejection.

Result timestamp: 2026-07-01 15:45 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-154553Z-n32-phase2k-vendor-mmq-upgate`

Measured result:

- Commit/config: `a03085b02`, accepted Phase 2H config plus
  `GGML_MOE_STREAM_UP_GATE_FUSED_MMQ=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 101097.26 ms, inside the 106331.72 ms gate.
- Decode: 157639.97 ms / 31 runs, 5.08516 s/token, 0.19665 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, art, and culture. It is famous for landmarks like the Eiffel Tower, the Louvre`
- Vendor MMQ activated: true.
- `launch_failures=0`, `read_failures=0`.
- Cache: hits=12126, misses=14818, hit_rate=45.0%.
- Pinned staging: copies=13458, host_stage=17950.601 ms, h2d=2973.772 ms.
- Up/gate profile: calls=1, total=128.587 ms/call.
- Down profile: calls=1644, stage=10.642 ms/call, total=10.808 ms/call.

Comparison against accepted Phase 2H `-n 32`:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token, 0.32 tok/s.
- Phase 2K `-n 32`: 157639.97 ms / 31 runs, 5.08516 s/token, 0.19665 tok/s.
- Vendor MMQ is substantially slower at the same token count, even though
  correctness, RAM, VRAM, and TTFT gates pass.

Decision:

- Reject Phase 2K.
- Do not run `-n 96` for this config.
- Keep accepted Phase 2H as the current best valid configuration.
- The next design should inspect why up/gate profile only reports one call under
  vendor MMQ and where the missing decode time is spent before trying another
  up/gate kernel change.

## Next candidate: Phase 2L VRAM cache key index

Design timestamp: 2026-07-01 16:03 UTC.

Current bottleneck:

- Phase 2K explains the vendor MMQ gap enough for prioritization: CUDA graph
  replay hides most per-call profiling, but total `-n 32` eval is still much
  slower, so vendor MMQ is not a useful next path.
- Accepted Phase 2H `-n 32` visible costs:
  - Eval: 95613.73 ms / 31 runs, 3.08431 s/token, 0.32 tok/s.
  - Up/gate profile: 869 calls, 18.938 ms/call, about 16.5s.
  - Down profile: 1644 calls, 9.228 ms/call, about 15.2s.
  - Pinned staging: 13135 copies, host_stage=18199.846 ms,
    h2d=2844.363 ms.
  - VRAM cache: hits=12892, misses=14052, hit_rate=47.8%.
- The cache lookup path currently linearly scans up to the cache slot count:
  2798 slots for 5.36 MiB up, 2493 slots for 6.02 MiB gate, and 2016 slots for
  7.44 MiB down. This scan happens in `batch_cache_lookup_slot`,
  `batch_cache_find_slot`, and `batch_cache_contains_slot`.
- Phase 2J proved that broad profile preloading/protection increases overhead.
  A key index targets fixed per-call CPU overhead without changing math,
  routing, cache size, or eviction policy.

Hypothesis:

Add an in-memory `key -> slot` index to each `batch_vram_cache` and keep it in
sync on insert, clear, and cache reinitialization. Use it for lookup/find/
contains while preserving the existing LRU/LFU victim scan for misses and
evictions.

Theoretical upper bound:

- Phase 2H `-n 32` has 26944 cache events (hits+misses) and thousands of
  additional find/contains checks. A linear scan over about 2000-2800 slots can
  easily add millions of key comparisons per short run.
- If this removes 5-10 ms from each down batch call, the upper bound on
  `-n 32` is roughly 8-16s saved from the 95.6s decode, or 2.57-2.83 s/token
  (0.35-0.39 tok/s).
- If lookup overhead is mostly hidden by IO/kernel time, expected gain may be
  only a few percent. This still has low correctness risk because it should not
  change selected experts, math kernels, or cache capacity.

Correctness risk:

- A stale key index could return the wrong slot after eviction or failed copy,
  causing silent semantic corruption.
- The implementation must erase the previous key before overwriting a slot,
  erase on clear/failure, and clear the index on cache reset.

Acceptance:

- First run cold `-n 4` under `memory.max=16000000000`, `memory.swap.max=0`,
  and `drop_caches`.
- Host RAM must remain under the 16GB cgroup cap including page cache.
- VRAM should remain near full without OOM.
- TTFT must remain <=106331.72 ms.
- The actual France answer must be semantically correct and coherent.
- `launch_failures=0`, `read_failures=0`, and down batch profile remains active.
- If `-n 4` passes, run cold `-n 32`.
- Accept and push the code only if `-n 32` is faster than accepted Phase 2H
  `-n 32` (0.32 tok/s / 3.08431 s/token) with all gates passing; otherwise
  revert the code and record rejection.

Rollback:

- Revert the code if output quality fails, TTFT exceeds the gate, RAM/VRAM
  gates fail, launch/read failures appear, or `-n 32` does not improve over
  Phase 2H.

Result timestamp: 2026-07-01 15:56 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-155605Z-n4-phase2l-cache-index`

Measured result:

- Commit/config: `a8eb2778-dirty-cache-index`, accepted Phase 2H config plus a
  local dirty `batch_vram_cache` key-index implementation.
- Build: remote `build-cuda-batch` compiled successfully.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 100320.04 ms, inside the 106331.72 ms gate.
- Decode: 14031.68 ms / 3 runs, 4.67723 s/token, 0.21380 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: hits=731, misses=1797, hit_rate=28.9%.
- Pinned staging: copies=1783, host_stage=2704.985 ms, h2d=386.983 ms.
- Up/gate profile: calls=85, total=27.173 ms/call.
- Down profile: calls=160, stage=13.295 ms/call, total=13.461 ms/call.

Decision:

- The `-n 4` smoke passes correctness, memory, VRAM, TTFT, and launch/read gates.
- Continue to cold `-n 32` before deciding whether to keep or revert the code.
- Watch up/gate timing: the tiny smoke has a faster overall decode than recent
  rejected configs, but up/gate is slower than Phase 2H n32, so this may still
  fail at longer length.

Result timestamp: 2026-07-01 15:59 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-155938Z-n32-phase2l-cache-index`

Measured result:

- Commit/config: `ff2ec840-dirty-cache-index`, accepted Phase 2H config plus
  the local dirty cache key-index implementation.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 102493.78 ms, inside the 106331.72 ms gate.
- Decode: 97355.97 ms / 31 runs, 3.14052 s/token, 0.31842 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: hits=12892, misses=14052, hit_rate=47.8%.
- Pinned staging: copies=13135, host_stage=17689.156 ms, h2d=2846.699 ms.
- Up/gate profile: calls=869, total=18.410 ms/call.
- Down profile: calls=1644, stage=9.078 ms/call, total=9.247 ms/call.

Comparison against accepted Phase 2H `-n 32`:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token, 0.32 tok/s.
- Phase 2L `-n 32`: 97355.97 ms / 31 runs, 3.14052 s/token, 0.31842 tok/s.
- The key index slightly reduced pinned staging host time
  (18199.846 ms -> 17689.156 ms) and up/gate total
  (18.938 ms/call -> 18.410 ms/call), but down total was slightly slower
  (9.228 ms/call -> 9.247 ms/call) and the full decode was slower.

Decision:

- Reject Phase 2L.
- Reverted the local cache key-index code and did not commit it.
- Keep accepted Phase 2H as the current best valid configuration.
- Next design should target a larger bucket than cache lookup overhead. The
  remaining visible Phase 2H costs are still up/gate compute and expert staging;
  a useful next step should either reduce actual up/gate graph replay time or
  reduce the number/size of expert loads rather than just lookup overhead.

## Next candidate: Phase 2M split VRAM cache by expert size

Design timestamp: 2026-07-01 16:17 UTC.

Current bottleneck:

- Accepted Phase 2H `-n 32` still spends a large visible bucket in expert
  staging:
  - Pinned staging: 13135 copies, host_stage=18199.846 ms,
    h2d=2844.363 ms.
  - VRAM cache hit_rate=47.8%.
- Phase 2L showed lookup overhead is not the main limiter: a key index slightly
  reduced host_stage but did not improve end-to-end token rate.
- Current cache initialization logs show one final 7.44 MiB slot size:
  2016 slots. All smaller experts can fit in that pool, but 4.48/5.36/6.02 MiB
  experts waste space when stored in 7.44 MiB slots.
- Phase 2H `-n 32` route profile by tensor kind:
  - up: 4.48/5.36 MiB experts, 6952 uses, 2356 unique, 32.57 GiB traffic.
  - gate: 4.48/5.36 MiB experts, 6952 uses, 2356 unique, 32.57 GiB traffic.
  - down: 6.02/7.44 MiB experts, 13152 uses, 4294 unique, 81.40 GiB traffic.
- Existing split-cache code only sends `expert_sz <= 4 MiB` to the second cache,
  which is ineffective for Kimi. Kimi's up/gate experts are larger than 4 MiB.

Hypothesis:

Add a default-off environment threshold,
`GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB`, used only when
`GGML_MOE_VRAM_CACHE_SPLIT=1`. Test with:

- `GGML_MOE_VRAM_CACHE_SPLIT=1`
- `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=45`

This should create a small-expert pool for 4.48/5.36 MiB up/gate experts and a
large-expert pool for 6.02/7.44 MiB down experts. It reduces slot waste and
prevents up/gate and down experts from evicting each other.

Theoretical upper bound:

- With a 15000 MiB cache and `UPGATE_PCT=45`, the small pool gets about
  6750 MiB and can store about 1259 5.36 MiB slots. The large pool gets about
  8250 MiB and can store about 1108 7.44 MiB slots.
- Total resident expert entries can rise from 2016 unified slots to about 2367
  split slots, a roughly 17% increase in resident entries.
- If staging misses fall proportionally, host_stage could drop by up to about
  3.1s on Phase 2H `-n 32`, improving from 95.6s to about 92.5s
  (2.98 s/token, 0.34 tok/s). If partitioning hurts locality, it may be slower.

Correctness risk:

- Low math risk: selected experts and kernels do not change.
- Cache-policy risk: wrong cache ID or budget could OOM or reduce hit rate.
- The code change must be default-off unless `GGML_MOE_VRAM_CACHE_SPLIT=1`, so
  accepted Phase 2H behavior remains unchanged without the new env.

Acceptance:

- First run cold `-n 4` with split cache under `memory.max=16000000000`,
  `memory.swap.max=0`, and `drop_caches`.
- Host RAM must remain under the 16GB cgroup cap including page cache.
- VRAM should remain near full without OOM.
- TTFT must remain <=106331.72 ms.
- France answer must be semantically correct and coherent.
- Logs must show two active cache pools and no allocation retry/failure.
- `launch_failures=0`, `read_failures=0`, and down batch profile remains active.
- If `-n 4` passes, run cold `-n 32`.
- Accept and push the code/config only if `-n 32` is faster than accepted Phase
  2H `-n 32` (0.32 tok/s / 3.08431 s/token) with all gates passing; otherwise
  revert the code and record rejection.

Rollback:

- Revert the code if output quality fails, TTFT exceeds the gate, RAM/VRAM
  gates fail, cache allocation fails, launch/read failures appear, or `-n 32`
  does not improve over Phase 2H.

Result timestamp: 2026-07-01 16:08 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-160808Z-n4-phase2m-split-cache`

Measured result:

- Commit/config: `1a635914-dirty-split-cache`, accepted Phase 2H config plus
  local default-off split threshold code and:
  - `GGML_MOE_VRAM_CACHE_SPLIT=1`
  - `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`
  - `GGML_MOE_VRAM_CACHE_UPGATE_PCT=45`
- Build: remote `build-cuda-batch` compiled successfully.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31342 MiB used, 768 MiB free.
- TTFT: 103089.41 ms, inside the 106331.72 ms gate.
- Decode: 13886.92 ms / 3 runs, 4.62897 s/token, 0.21603 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache pools confirmed:
  - down: 1109 slots, 7.44 MiB slot, hits=383, misses=865,
    hit_rate=30.7%.
  - upgate: 1259 slots, 5.36 MiB slot, hits=380, misses=980,
    hit_rate=27.9%.
- Pinned staging: copies=1759, host_stage=2668.331 ms, h2d=381.974 ms.
- Up/gate profile: calls=85, total=26.065 ms/call.
- Down profile: calls=160, stage=13.649 ms/call, total=13.818 ms/call.

Decision:

- The `-n 4` smoke passes correctness, memory, VRAM, TTFT, cache allocation, and
  launch/read gates.
- Continue to cold `-n 32` before deciding whether to keep or revert the code.

Result timestamp: 2026-07-01 16:11 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-161150Z-n32-phase2m-split-cache`

Measured result:

- Commit/config: `d7d0fa42-dirty-split-cache`, accepted Phase 2H config plus
  local default-off split threshold code and:
  - `GGML_MOE_VRAM_CACHE_SPLIT=1`
  - `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`
  - `GGML_MOE_VRAM_CACHE_UPGATE_PCT=45`
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31342 MiB used, 768 MiB free.
- TTFT: 97293.10 ms, inside the 106331.72 ms gate.
- Decode: 92447.06 ms / 31 runs, 2.98216 s/token, 0.33533 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache pools:
  - down: 1109 slots, 7.44 MiB slot, hits=6809, misses=6311,
    hit_rate=51.9%.
  - upgate: 1259 slots, 5.36 MiB slot, hits=6836, misses=7068,
    hit_rate=49.2%.
- Pinned staging: copies=12466, host_stage=16966.087 ms, h2d=2701.115 ms.
- Up/gate profile: calls=869, total=16.697 ms/call.
- Down profile: calls=1644, stage=8.608 ms/call, total=8.770 ms/call.

Comparison against accepted Phase 2H `-n 32`:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token, 0.32 tok/s.
- Phase 2M `-n 32`: 92447.06 ms / 31 runs, 2.98216 s/token, 0.33533 tok/s.
- Improvement: about 3.3% lower seconds/token at the same token count while
  passing host RAM, VRAM, TTFT, quality, launch, and read gates.
- The improvement matches the hypothesis: split pools reduce staging copies
  (13135 -> 12466), host_stage (18199.846 ms -> 16966.087 ms), up/gate total
  (18.938 ms/call -> 16.697 ms/call), and down total
  (9.228 ms/call -> 8.770 ms/call).

Decision:

- Accept Phase 2M as a valid constrained `-n 32` improvement.
- Commit and push the default-off code plus this plan record immediately.
- Next practice: run cold `-n 96` with the same split-cache config before
  promoting it as the new full-length best configuration.

Reproduction:

```sh
RUN=/root/lfz/runs/vendor-kimi-token-rate/$(date -u +%Y%m%d-%H%M%SZ)-n32-phase2m-split-cache
mkdir -p "$RUN"
CG=/sys/fs/cgroup/kimi_phase2m_n32_$$
mkdir "$CG"
echo 16000000000 > "$CG/memory.max"
echo 0 > "$CG/memory.swap.max"
sync
echo 3 > /proc/sys/vm/drop_caches

export GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
export GGML_MOE_IO_BACKEND=iouring
export GGML_MOE_STREAM=1
export GGML_MOE_PARALLEL_EXPERTS=1
export GGML_MOE_STAGE_PINNED_SLOTS=8
export GGML_MOE_STREAM_FUSED_UP_GATE=1
export GGML_MOE_STREAM_BATCH_ONLY=1
export GGML_MOE_STREAM_DOWN_BATCH=1
export GGML_MOE_VRAM_CACHE_SPLIT=1
export GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6
export GGML_MOE_VRAM_CACHE_UPGATE_PCT=45
export GGML_MOE_VRAM_CACHE_MIB=15000
export GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
export GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
export GGML_MOE_MMAP_DONTNEED=1
export GGML_MOE_IO_BYTES=8388608
export GGML_MOE_BATCH_PROFILE=1
export GGML_MOE_BATCH_PROFILE_OUT="$RUN/route-profile.csv"
export GGML_MOE_ROUTE_TRACE_OUT="$RUN/route-trace.csv"
export GGML_MOE_TTFT_TRACE_OUT="$RUN/ttft-trace.csv"
export GGML_MOE_TTFT_TRACE_MAX_EVENTS=180000

PROMPT='<|im_user|>user<|im_middle|>Please introduce France in a short paragraph.<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>'
( echo $BASHPID > "$CG/cgroup.procs"
  /root/lfz/llama.cpp-vendor-kimi/build-cuda-batch/bin/llama-completion \
    --defer-experts --fit off -ngl 99 --special \
    -m /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf \
    -c 512 -n 32 --temp 0 --top-p 1.0 --top-k 1 --seed 1 \
    --no-display-prompt -no-cnv -t 32 -tb 32 -p "$PROMPT" \
    > "$RUN/stdout.txt" 2> "$RUN/stderr.txt" )
```

Full-length result timestamp: 2026-07-01 16:18 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-161815Z-n96-phase2m-split-cache`

Measured result:

- Commit/config: `adf621b20`, same split-cache config as the accepted `-n 32`
  run.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31342 MiB used, 768 MiB free.
- TTFT: 104536.59 ms, inside the 106331.72 ms gate.
- Decode: 299594.70 ms / 85 runs, 3.52464 s/token, 0.28372 tok/s.
- Quality flag: PASS; full answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache pools:
  - down: 1109 slots, 7.44 MiB slot, hits=18230, misses=17786,
    hit_rate=50.6%.
  - upgate: 1259 slots, 5.36 MiB slot, hits=17991, misses=20105,
    hit_rate=47.2%.
- Pinned staging: copies=33497, host_stage=45305.989 ms, h2d=7268.052 ms.
- Up/gate profile: calls=2381, total=22.493 ms/call.
- Down profile: calls=4506, stage=11.130 ms/call, total=11.300 ms/call.

Comparison against accepted Phase 2H `-n 96`:

- Phase 2H `-n 96`: 295113.58 ms / 85 runs, 3.47192 s/token, 0.29 tok/s.
- Phase 2M `-n 96`: 299594.70 ms / 85 runs, 3.52464 s/token, 0.28372 tok/s.
- Phase 2M improves some visible buckets but does not improve full-length
  end-to-end decode.

Decision:

- Do not promote Phase 2M as the full-length `-n 96` best.
- Keep accepted Phase 2H as the current best full `-n 96` configuration.
- Keep the default-off `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB` code because Phase
  2M is a valid `-n 32` constrained improvement and does not change default
  behavior.
- Next design should target the gap between `-n 32` and `-n 96`: split-cache
  helps shorter runs but loses at longer output. The next experiment should
  tune split budget for `-n 96` specifically, or target graph replay/upgate
  cost rather than fixed cache partitioning.

## Next candidate: Phase 2N n96 split-cache budget sweep

Design timestamp: 2026-07-01 16:31 UTC.

Current bottleneck:

- Phase 2M split cache improved `-n 32` but missed the full `-n 96` target.
- Phase 2M `-n 96` still passed correctness, TTFT, RAM, and VRAM gates, so the
  split-cache mechanism is safe enough for config tuning.
- Phase 2H/2M `-n 96` route profile by tensor kind:
  - down: 223.10 GiB logical traffic, 7543 unique experts.
  - up+gate: 178.46 GiB logical traffic, 8296 unique experts.
- Phase 2M used `GGML_MOE_VRAM_CACHE_UPGATE_PCT=45`, producing:
  - down pool: 1109 slots, hit_rate=50.6%.
  - upgate pool: 1259 slots, hit_rate=47.2%.
- Since down traffic is larger and down misses feed the visible staging bucket,
  45% may over-allocate the upgate pool for full-length n96.

Hypothesis:

Keep the accepted default-off split threshold code and test a more down-heavy
budget:

- `GGML_MOE_VRAM_CACHE_SPLIT=1`
- `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=35`

This should reduce the small upgate pool from about 1259 slots to about 979
slots and increase the large down pool from about 1109 slots to about 1310
slots. If the full-length regression is caused by down pool pressure, the
larger down pool should reduce down misses and staging enough to beat Phase 2H
`-n 96`.

Theoretical upper bound:

- Phase 2M n96 down misses were 17786 and down stage was 11.130 ms/call.
- A 18% larger down pool could plausibly save 5-10% of down misses if locality
  is pool-capacity limited. That would save roughly 2-5s from full decode.
- The maximum practical target for this budget-only step is therefore modest:
  improve from Phase 2M's 3.52464 s/token to below Phase 2H's 3.47192 s/token,
  likely around 3.43-3.47 s/token if successful.

Acceptance:

- Run cold `-n 4` first under `memory.max=16000000000`, `memory.swap.max=0`,
  and `drop_caches`.
- Host RAM must remain under the 16GB cgroup cap including page cache.
- VRAM should remain near full without OOM.
- TTFT must remain <=106331.72 ms.
- France answer must be semantically correct and coherent.
- Logs must show two active cache pools and no allocation retry/failure.
- `launch_failures=0`, `read_failures=0`, and down batch profile remains active.
- If `-n 4` passes, run cold `-n 96` directly because this is a full-length
  budget tuning step.
- Promote only if `-n 96` is faster than accepted Phase 2H `-n 96`
  (0.29 tok/s / 3.47192 s/token) with all gates passing.

Rollback:

- Reject this config if quality fails, TTFT exceeds the gate, RAM/VRAM gates
  fail, cache allocation fails, launch/read failures appear, or `-n 96` does
  not improve over Phase 2H.

## Phase 2P full result: reject down prefetch depth 8 for n96

Result timestamp: 2026-07-02 17:01 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-170118Z-n96-phase2p-down-prefetch`

Reproduction:

- Commit: `adf621b20`.
- Start from accepted Phase 2H full configuration:
  - expert pack enabled.
  - VRAM cache budget: 15000 MiB.
  - down batch enabled.
  - GPU handoff disabled.
  - no split cache.
  - `-t 32 -tb 32`.
- Add:
  - `GGML_MOE_PREFETCH_DOWN=1`
  - `GGML_MOE_PREFETCH_DOWN_DEPTH=8`
- Run as a cold start:
  - `sync; echo 3 > /proc/sys/vm/drop_caches`
  - cgroup `memory.max=16000000000`
  - cgroup `memory.swap.max=0`
- Prompt:
  `Please introduce France in a short paragraph.`

Measured result:

- Host RAM peak: 14.901 GiB, inside the strict 16GB cgroup cap including page
  cache pressure.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 92848.35 ms, inside the 106331.72 ms gate.
- Decode: 301072.16 ms / 85 runs, 3.54203 s/token, 0.28232 tok/s.
- Quality flag: PASS; full answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Down prefetch: loads=9157, hits=9157, evicted_unused=0,
  useful_rate=100.0%.
- Cache: slots=2016, slot=7.44 MiB, hits=42866, misses=31182,
  preloads=9157, hit_rate=57.9%.
- Pinned staging: copies=35790, host_stage=51604.703 ms, h2d=7772.206 ms.
- Up/gate profile: calls=2381, total=24.696 ms/call.
- Down profile: calls=4506, stage=4.771 ms/call, total=4.939 ms/call.

Comparison:

- Accepted Phase 2H `-n 96`: 295113.58 ms / 85 runs,
  3.47192 s/token, 0.29 tok/s.
- Phase 2P depth 8 `-n 96`: 301072.16 ms / 85 runs,
  3.54203 s/token, 0.28232 tok/s.
- The prefetch mechanism behaves correctly and is useful for every recorded
  prefetched down load, and it cuts down visible down stage time from
  11.616 ms/call to 4.771 ms/call.
- The full decode still regresses because host staging and up/gate compute both
  get worse:
  - host_stage rises from 50153.894 ms to 51604.703 ms.
  - up/gate total rises from 22.785 ms/call to 24.696 ms/call.
- The likely bottleneck is H2D/cache contention from prefetching too far ahead:
  depth 8 helps the down layer but steals bandwidth/cache locality from
  up/gate at full output length.

Decision:

- Reject Phase 2P depth 8 for full `-n 96` promotion.
- No code rollback is needed because this path is fully default-off and was
  enabled only by env vars.
- Keep accepted Phase 2H as the current best full `-n 96` configuration.
- Keep Phase 2M as an accepted short-run `-n 32` improvement only.
- Next candidate should reduce down prefetch pressure instead of abandoning the
  mechanism: test shallower depths, starting with depth 2, because Phase 2P
  proves the mechanism can remove down stage time but depth 8 over-contends at
  full length.

## Next candidate: Phase 2Q shallow down prefetch depth 2

Design timestamp: 2026-07-02 17:18 CST.

Current bottleneck:

- Phase 2P depth 8 cut down visible staging from 11.616 ms/call to
  4.771 ms/call at full `-n 96`, and down prefetch useful_rate was 100.0%.
- Full decode still regressed from 295113.58 ms to 301072.16 ms because:
  - host_stage rose from 50153.894 ms to 51604.703 ms.
  - up/gate total rose from 22.785 ms/call to 24.696 ms/call.
- This points to prefetch pressure rather than a correctness issue: depth 8
  starts too much down H2D/cache activity ahead of demand and interferes with
  the up/gate path.

Hypothesis:

Run the same accepted Phase 2H configuration with shallower prefetch:

- `GGML_MOE_PREFETCH_DOWN=1`
- `GGML_MOE_PREFETCH_DOWN_DEPTH=2`

Depth 2 should preserve some down-stage overlap while reducing the amount of
early H2D/cache traffic. If contention was the reason depth 8 lost at full
length, depth 2 should show lower up/gate regression and lower host_stage than
Phase 2P, while still improving down profile against Phase 2H.

Theoretical upper bound:

- Phase 2H down visible stage is about 52.3s total.
- Phase 2P depth 8 reduced this bucket by about 30.8s, but added enough
  contention to lose about 6.0s overall.
- Depth 2 cannot hide as much down stage as depth 8. A realistic target is
  hiding 10-18s while limiting contention to under 5s, producing a possible
  full decode of 282-290s.
- Promotion requires beating Phase 2H full `-n 96`: 295113.58 ms / 85 runs,
  3.47192 s/token, 0.29 tok/s, with TTFT <=106331.72 ms and quality intact.

Execution:

- Cold `-n 4` first under:
  - `memory.max=16000000000`
  - `memory.swap.max=0`
  - `sync; echo 3 > /proc/sys/vm/drop_caches`
- If `-n 4` passes, run cold `-n 32`.
- Promote to cold `-n 96` only if `-n 32` is not slower than Phase 2H `-n 32`
  and logs show prefetch useful activity without high evicted-unused count.

Acceptance:

- Host RAM must remain under 16GB including page cache.
- VRAM should remain close to full without OOM.
- TTFT must remain <=106331.72 ms.
- Prompt must be:
  `Please introduce France in a short paragraph.`
- Output must be semantically correct and coherent.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Logs must include down prefetch activity and its useful-rate counters.
- Any full `-n 96` improvement that passes all gates must be committed and
  pushed immediately with reproduction details.

Rollback:

- Reject this config if quality fails, TTFT exceeds the gate, RAM/VRAM gates
  fail, prefetch does not activate, evicted-unused is high, launch/read
  failures appear, or full `-n 96` does not beat Phase 2H.
- No code rollback should be needed because the experiment is env-only and the
  prefetch path is default-off.

Result timestamp: 2026-07-02 17:11 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-171134Z-n4-phase2q-down-prefetch-depth2`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus:
  - `GGML_MOE_PREFETCH_DOWN=1`
  - `GGML_MOE_PREFETCH_DOWN_DEPTH=2`
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 102188.15 ms, inside the 106331.72 ms gate.
- Decode: 14379.67 ms / 3 runs, 4.79322 s/token, 0.20863 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Down prefetch: loads=137, hits=137, evicted_unused=0, useful_rate=100.0%.
- Cache: slots=2016, slot=7.44 MiB, hits=878, misses=1666,
  preloads=137, hit_rate=34.5%.
- Pinned staging: copies=1773, host_stage=2778.373 ms, h2d=384.665 ms.
- Up/gate profile: calls=85, total=28.221 ms/call.
- Down profile: calls=160, stage=10.585 ms/call, total=10.750 ms/call.

Decision:

- The `-n 4` smoke passes the hard gates: strict RAM, VRAM, TTFT, launch/read,
  prefetch activation, and semantic quality.
- Depth 2 does not yet show the intended down-stage benefit on the tiny sample:
  it has far fewer preloads than depth 8 and down stage remains near the
  non-prefetch path.
- Continue to `-n 32` because the plan requires the medium-length check after
  a passing smoke, but promotion to `-n 96` will require `-n 32` to beat Phase
  2H `-n 32` or show a clear bottleneck reason worth retuning.

Result timestamp: 2026-07-02 17:14 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-171437Z-n32-phase2q-down-prefetch-depth2`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus depth 2 down
  prefetch.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 101161.66 ms, inside the 106331.72 ms gate.
- Decode: 93449.43 ms / 31 runs, 3.01450 s/token, 0.33173 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Down prefetch: loads=1442, hits=1442, evicted_unused=0,
  useful_rate=100.0%.
- Cache: slots=2016, slot=7.44 MiB, hits=14345, misses=12615,
  preloads=1442, hit_rate=53.2%.
- Pinned staging: copies=13122, host_stage=18176.985 ms, h2d=2844.610 ms.
- Up/gate profile: calls=869, total=17.730 ms/call.
- Down profile: calls=1644, stage=6.558 ms/call, total=6.717 ms/call.

Comparison:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token,
  about 0.32 tok/s.
- Phase 2P depth 8 `-n 32`: 94356.18 ms / 31 runs, 3.04375 s/token,
  0.32854 tok/s.
- Phase 2Q depth 2 `-n 32`: 93449.43 ms / 31 runs, 3.01450 s/token,
  0.33173 tok/s.

Decision:

- The `-n 32` run passes all gates and beats both Phase 2H and Phase 2P on the
  matching token count.
- The mechanism now has a better balance: down stage improves over Phase 2H,
  while up/gate total is also lower than the Phase 2H `-n 32` profile.
- Continue to cold full `-n 96`.
- Full promotion still requires beating accepted Phase 2H full `-n 96`
  (295113.58 ms / 85 runs, 3.47192 s/token, 0.29 tok/s) with strict RAM,
  VRAM, TTFT, launch/read, and semantic quality gates passing.

Full result timestamp: 2026-07-02 17:18 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-171854Z-n96-phase2q-down-prefetch-depth2`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus depth 2 down
  prefetch.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 103081.24 ms, inside the 106331.72 ms gate.
- Decode: 304478.65 ms / 85 runs, 3.58210 s/token, 0.27917 tok/s.
- Quality flag: PASS; full answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Down prefetch: loads=3992, hits=3992, evicted_unused=0,
  useful_rate=100.0%.
- Cache: slots=2016, slot=7.44 MiB, hits=37713, misses=36335,
  preloads=3992, hit_rate=50.9%.
- Pinned staging: copies=35780, host_stage=49805.170 ms, h2d=7758.946 ms.
- Up/gate profile: calls=2381, total=23.671 ms/call.
- Down profile: calls=4506, stage=8.748 ms/call, total=8.915 ms/call.

Comparison:

- Accepted Phase 2H `-n 96`: 295113.58 ms / 85 runs,
  3.47192 s/token, 0.29 tok/s.
- Phase 2P depth 8 `-n 96`: 301072.16 ms / 85 runs,
  3.54203 s/token, 0.28232 tok/s.
- Phase 2Q depth 2 `-n 96`: 304478.65 ms / 85 runs,
  3.58210 s/token, 0.27917 tok/s.

Analysis:

- Depth 2 reduced prefetch pressure compared with depth 8:
  - host_stage improved from 51604.703 ms to 49805.170 ms.
  - up/gate total improved from 24.696 ms/call to 23.671 ms/call.
- It still regressed versus Phase 2H:
  - up/gate total is worse than 22.785 ms/call.
  - cache hit_rate falls to 50.9%, worse than depth 8 and only modestly useful
    for down.
  - down total improves from 11.781 ms/call to 8.915 ms/call, but the measured
    full decode does not realize that apparent local saving. The prefetch path
    likely shifts work into hidden synchronization/cache refill points outside
    the down-profile bucket.
- The `-n 32` win does not generalize to full length, so this cannot be used as
  a stable optimization under the current cold-start acceptance rule.

Decision:

- Reject Phase 2Q for full `-n 96` promotion.
- No code rollback is needed because this path is env-only and default-off.
- Stop pursuing blind prefetch-depth sweeps as the next main path. The next
  optimization should instrument where the missing full-length time moves
  before changing prefetch policy again.

## Next candidate: Phase 2R raise VRAM cache budget to 15400 MiB

Design timestamp: 2026-07-02 17:29 CST.

Current bottleneck:

- Accepted Phase 2H full `-n 96` leaves 772 MiB VRAM free while the down-cache
  path still has 40316 misses and 50153.894 ms host staging time.
- The strict host RAM limit is already the main model-size gate, but VRAM is
  not completely filled. The requirement says to use VRAM as much as possible,
  so the next low-risk test is to convert part of the remaining VRAM headroom
  into additional expert cache capacity.
- Phase 2H cache:
  - budget=15000 MiB.
  - down cache slot=7.44 MiB.
  - slots=2016.
  - hit_rate=45.5%.
- Increasing to 15400 MiB should add about 400 MiB / 7.44 MiB = 53 additional
  down-cache slots, while leaving roughly 300-400 MiB free if memory behavior
  stays close to Phase 2H.

Hypothesis:

Run the accepted Phase 2H config with:

- `GGML_MOE_VRAM_CACHE_MIB=15400`
- no split cache.
- no down prefetch.

This should not affect model math or routing. It can only change which expert
weights remain resident in VRAM. If the extra 53 slots catch frequently reused
experts, host staging and misses should fall with no quality change.

Theoretical upper bound:

- The absolute upper bound is limited because 400 MiB is only about 2.6% more
  cache than 15000 MiB.
- If the marginal slots are high-value, avoiding 1000-2000 misses could save
  roughly 1-3s of host staging/H2D work on the full run.
- A realistic successful result is therefore modest: full `-n 96` must drop
  below 295113.58 ms, but a 1-2s improvement is still valid if all gates pass
  and is worth committing because it uses more VRAM without code risk.

Execution:

- Cold `-n 4` first under:
  - `memory.max=16000000000`
  - `memory.swap.max=0`
  - `sync; echo 3 > /proc/sys/vm/drop_caches`
- If `-n 4` passes without OOM and leaves nonzero VRAM headroom, run cold
  `-n 32`.
- Promote to full `-n 96` only if `-n 32` is not slower than Phase 2H `-n 32`
  or if the cache-hit/staging metrics clearly improve without TTFT pressure.

Acceptance:

- Host RAM must remain under 16GB including page cache.
- VRAM should be fuller than Phase 2H without CUDA OOM or allocation retries.
- TTFT must remain <=106331.72 ms.
- Prompt must be:
  `Please introduce France in a short paragraph.`
- Output must be semantically correct and coherent.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Full `-n 96` must beat accepted Phase 2H full `-n 96` to be promoted.
- If promoted, commit and push immediately with the exact env vars and run
  paths.

Rollback:

- Reject if VRAM OOMs, free VRAM becomes unstable, TTFT exceeds the gate,
  host RAM exceeds the 16GB cap, output quality fails, launch/read failures
  appear, or full `-n 96` does not beat Phase 2H.
- No code rollback should be needed because this is env-only.

Result timestamp: 2026-07-02 17:28 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-172837Z-n4-phase2r-vram-cache15400`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config with
  `GGML_MOE_VRAM_CACHE_MIB=15400`, down prefetch disabled.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31666 MiB used, 444 MiB free.
- TTFT: 106568.74 ms, which exceeds the 106331.72 ms gate.
- Decode: 14927.76 ms / 3 runs, 4.97592 s/token, 0.20097 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2060, slot=7.44 MiB, hits=731, misses=1797,
  preloads=0, hit_rate=28.9%.
- Pinned staging: copies=1783, host_stage=2838.856 ms, h2d=387.078 ms.
- Up/gate profile: calls=85, total=28.555 ms/call.
- Down profile: calls=160, stage=14.235 ms/call, total=14.407 ms/call.

Decision:

- Reject Phase 2R immediately at `-n 4`.
- It uses VRAM more fully, but violates the TTFT rule by 237.02 ms.
- Do not run `-n 32` or `-n 96`.
- No code rollback is needed because the experiment is env-only.
- Larger cache budgets near 15400 MiB are too close to the TTFT/VRAM margin for
  the current cold-start acceptance rule. If cache budget is revisited, test a
  smaller step such as 15200 MiB and require TTFT headroom on smoke before
  continuing.

## Next candidate: Phase 2S raise VRAM cache budget to 15200 MiB

Design timestamp: 2026-07-02 17:43 CST.

Current bottleneck:

- Accepted Phase 2H uses `GGML_MOE_VRAM_CACHE_MIB=15000` and leaves 772 MiB
  VRAM free at full `-n 96`.
- Phase 2R with 15400 MiB used VRAM more fully, leaving 444 MiB free, but its
  `-n 4` TTFT was 106568.74 ms, which exceeded the 106331.72 ms gate by
  237.02 ms.
- The failure was a tight TTFT margin rather than OOM, quality failure, or RAM
  violation. A smaller cache increase may still satisfy the "use VRAM as much
  as possible" requirement while keeping cold-start TTFT inside the gate.

Hypothesis:

Run the accepted Phase 2H config with:

- `GGML_MOE_VRAM_CACHE_MIB=15200`
- no split cache.
- no down prefetch.

This should add about 200 MiB / 7.44 MiB = 26 additional down-cache slots over
Phase 2H, while preserving roughly 550-600 MiB free VRAM if memory behavior
tracks linearly. The extra slots may reduce misses and host staging without the
TTFT violation seen at 15400 MiB.

Theoretical upper bound:

- A 200 MiB cache increase is only about 1.3% over the accepted 15000 MiB cache.
- The best realistic gain is modest: if the marginal 26 slots avoid a few
  hundred to roughly 1000 full-run misses, expected full `-n 96` improvement is
  likely 0.5-2.0s.
- Promotion still requires full `-n 96` decode below 295113.58 ms / 85 runs
  with all gates passing.

Execution:

- Cold `-n 4` first under:
  - `memory.max=16000000000`
  - `memory.swap.max=0`
  - `sync; echo 3 > /proc/sys/vm/drop_caches`
- Continue to cold `-n 32` only if:
  - TTFT <=106331.72 ms.
  - VRAM has nonzero stable headroom.
  - semantic quality passes.
  - launch/read failures remain zero.
- Continue to cold `-n 96` only if `-n 32` is not slower than Phase 2H or shows
  better cache/staging metrics with enough TTFT headroom to justify full run.

Acceptance:

- Host RAM must stay under 16GB including page cache.
- VRAM should be fuller than Phase 2H without OOM or allocation retries.
- TTFT must stay <=106331.72 ms.
- Prompt must be:
  `Please introduce France in a short paragraph.`
- Output must be semantically correct and coherent.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Full `-n 96` must beat accepted Phase 2H full `-n 96`.
- If accepted, commit and push immediately with reproduction details.

Rollback:

- Reject if TTFT exceeds the gate, VRAM/RAM gates fail, output quality fails,
  launch/read failures appear, or full `-n 96` does not improve over Phase 2H.
- No code rollback should be needed because this is env-only.

Result timestamp: 2026-07-02 17:41 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-174105Z-n4-phase2t-lfu-cache-policy`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus
  `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`, no profile preload/protect, no split
  cache, no down prefetch.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 106329.47 ms, inside the 106331.72 ms gate by only 2.25 ms.
- Decode: 14340.15 ms / 3 runs, 4.78005 s/token, 0.20920 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2016, slot=7.44 MiB, hits=731, misses=1797,
  preloads=0, hit_rate=28.9%.
- Pinned staging: copies=1783, host_stage=2746.339 ms, h2d=387.230 ms.
- Up/gate profile: calls=85, total=28.919 ms/call.
- Down profile: calls=160, stage=14.094 ms/call, total=14.275 ms/call.

Decision:

- The smoke technically passes all hard gates, but TTFT headroom is only
  2.25 ms, which is too narrow to call robust at this stage.
- Continue to `-n 32` only as a diagnostic gate because the config is env-only
  and quality/RAM/VRAM are valid.
- Require a clear `-n 32` speed or cache/staging improvement before considering
  full `-n 96`; otherwise reject without full run because the TTFT margin is
  not reproducibility-friendly.

Result timestamp: 2026-07-02 17:44 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-174416Z-n32-phase2t-lfu-cache-policy`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus
  `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 99912.80 ms, inside the 106331.72 ms gate.
- Decode: 95284.36 ms / 31 runs, 3.07369 s/token, 0.32534 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2016, slot=7.44 MiB, hits=10785, misses=16159,
  preloads=0, hit_rate=40.0%.
- Pinned staging: copies=15211, host_stage=21256.248 ms, h2d=3303.428 ms.
- Up/gate profile: calls=869, total=18.744 ms/call.
- Down profile: calls=1644, stage=9.910 ms/call, total=10.068 ms/call.

Comparison:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token,
  about 0.32 tok/s.
- Phase 2T `-n 32`: 95284.36 ms / 31 runs, 3.07369 s/token,
  0.32534 tok/s.

Decision:

- The medium run passes all gates and is slightly faster than Phase 2H on the
  matching token count.
- Continue to full cold `-n 96`, but treat this as a marginal candidate:
  cache hit-rate and staging metrics are not clearly better than Phase 2H, so
  full promotion requires the actual `-n 96` decode to beat Phase 2H with
  semantic quality and TTFT gates intact.

Full result timestamp: 2026-07-02 17:48 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-174833Z-n96-phase2t-lfu-cache-policy`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus
  `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 104074.28 ms, inside the 106331.72 ms gate.
- Decode: 310487.68 ms / 85 runs, 3.65280 s/token, 0.27376 tok/s.
- Quality flag: PASS; full answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2016, slot=7.44 MiB, hits=26154, misses=47878,
  preloads=0, hit_rate=35.3%.
- Pinned staging: copies=42984, host_stage=59313.938 ms, h2d=9334.894 ms.
- Up/gate profile: calls=2381, total=24.455 ms/call.
- Down profile: calls=4506, stage=12.445 ms/call, total=12.610 ms/call.

Comparison:

- Accepted Phase 2H `-n 96`: 295113.58 ms / 85 runs,
  3.47192 s/token, 0.29 tok/s.
- Phase 2T `-n 96`: 310487.68 ms / 85 runs,
  3.65280 s/token, 0.27376 tok/s.

Analysis:

- LFU/LRU does not generalize from the small `-n 32` win to full length.
- It actively worsens the full cache behavior:
  - misses increase from 40316 to 47878.
  - hit_rate falls from 45.5% to 35.3%.
  - host_stage rises from 50153.894 ms to 59313.938 ms.
  - up/gate and down profiles both regress.
- The likely cause is that per-slot lifetime hit count over-protects experts
  that were hot early, then evicts experts needed later in the long decode.
  For this prompt, recency remains more important than raw hit count.

Decision:

- Reject Phase 2T for full `-n 96`.
- Do not use `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru` for the accepted config.
- No code rollback is needed because this was env-only.
- Keep accepted Phase 2H as the current best full `-n 96` configuration.

## Next candidate: Phase 2U down-only parallel CPU staging

Design timestamp: 2026-07-02 18:00 CST.

Current bottleneck:

- Accepted Phase 2H full `-n 96` down profile:
  - calls=4506.
  - stage=11.616 ms/call.
  - total=11.781 ms/call.
  - visible down staging bucket is about 52.3s.
- Down prefetch reduced down stage but hurt full decode through cache/H2D
  contention.
- Up/gate parallel staging was already rejected in Phase 2F because stream waits
  consumed the expected overlap.
- The code has a separate down-only staging path:
  `GGML_MOE_DOWN_PARALLEL_STAGE=1`. It splits down miss copy jobs across two
  CPU threads and two CUDA streams, then joins before launching the down batch
  kernel. This targets only the down staging bucket and does not prefetch ahead
  or change cache policy.

Hypothesis:

Run the accepted Phase 2H config with:

- `GGML_MOE_DOWN_PARALLEL_STAGE=1`
- no up/gate parallel mode.
- no down prefetch.
- no split cache.
- no cache policy override.
- cache budget remains 15000 MiB.

The path should preserve semantics because it only changes how missing down
experts are copied into the same cache slots before the same down batch kernel
uses them.

Theoretical upper bound:

- If copy jobs split evenly and H2D/host reads can overlap, down stage could
  approach roughly half of 11.616 ms/call, similar to the best local effect
  seen from prefetch but without loading future experts early.
- Full-run upper bound: saving up to about 26s from Phase 2H, giving
  269s / 85 runs = 3.16 s/token.
- More realistic: if only host-side staging overlaps and H2D bandwidth remains
  serialized, saving may be 3-8s full run, still enough to beat Phase 2H if it
  does not add synchronization overhead.

Execution:

- Cold `-n 4` first under:
  - `memory.max=16000000000`
  - `memory.swap.max=0`
  - `sync; echo 3 > /proc/sys/vm/drop_caches`
- If `-n 4` passes, run cold `-n 32`.
- Continue to full `-n 96` only if:
  - `-n 32` is not slower than Phase 2H `-n 32`, or
  - down stage is materially lower without TTFT/quality/RAM/VRAM risk.

Acceptance:

- Host RAM must stay under 16GB including page cache.
- VRAM should remain close to Phase 2H utilization without OOM.
- TTFT must stay <=106331.72 ms.
- Prompt must be:
  `Please introduce France in a short paragraph.`
- Output must be semantically correct and coherent.
- Logs must show `down parallel CPU staging active`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Full `-n 96` must beat accepted Phase 2H full `-n 96`.
- If accepted, commit and push immediately with reproduction details.

Rollback:

- Reject if TTFT exceeds the gate, RAM/VRAM gates fail, output quality fails,
  `down parallel CPU staging active` does not appear, launch/read failures
  appear, or full `-n 96` does not improve over Phase 2H.
- No code rollback should be needed because this is env-only.

Result timestamp: 2026-07-02 17:57 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-175735Z-n4-phase2u-down-parallel-stage`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus
  `GGML_MOE_DOWN_PARALLEL_STAGE=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 103118.25 ms, inside the 106331.72 ms gate.
- Decode: 14280.11 ms / 3 runs, 4.76004 s/token, 0.21008 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- Activation: log contains `down parallel CPU staging active`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2016, slot=7.44 MiB, hits=731, misses=1797,
  preloads=0, hit_rate=28.9%.
- Pinned staging: copies=1382, host_stage=1416.480 ms, h2d=297.596 ms.
- Up/gate profile: calls=85, total=29.033 ms/call.
- Down profile: calls=160, stage=12.211 ms/call, total=12.388 ms/call.

Decision:

- The smoke passes correctness, RAM, VRAM, TTFT, activation, and launch/read
  gates.
- It does not show the intended local down-stage improvement on the tiny run:
  down stage is 12.211 ms/call, which is not better than the accepted full
  Phase 2H down stage of 11.616 ms/call.
- Continue to `-n 32` only to rule out tiny-sample noise. If `-n 32` does not
  beat Phase 2H or materially reduce down stage, reject without full `-n 96`.

Result timestamp: 2026-07-02 18:00 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-180052Z-n32-phase2u-down-parallel-stage`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus
  `GGML_MOE_DOWN_PARALLEL_STAGE=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 99103.46 ms, inside the 106331.72 ms gate.
- Decode: 94223.24 ms / 31 runs, 3.03946 s/token, 0.32901 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2016, slot=7.44 MiB, hits=12892, misses=14052,
  preloads=0, hit_rate=47.8%.
- Pinned staging: copies=10390, host_stage=9152.341 ms, h2d=2213.267 ms.
- Up/gate profile: calls=869, total=20.299 ms/call.
- Down profile: calls=1644, stage=8.366 ms/call, total=8.528 ms/call.

Comparison:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token,
  about 0.32 tok/s.
- Phase 2U `-n 32`: 94223.24 ms / 31 runs, 3.03946 s/token,
  0.32901 tok/s.

Decision:

- The medium run passes all gates and shows the intended local effect:
  down stage is materially lower than the Phase 2H `-n 32` down profile
  while the total decode is faster.
- Continue to full cold `-n 96`.
- Full promotion still requires beating accepted Phase 2H full `-n 96`
  (295113.58 ms / 85 runs, 3.47192 s/token, 0.29 tok/s) with full semantic
  quality and all hard gates passing.

Full result timestamp: 2026-07-02 18:05 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-180510Z-n96-phase2u-down-parallel-stage`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus
  `GGML_MOE_DOWN_PARALLEL_STAGE=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 102802.72 ms, inside the 106331.72 ms gate.
- Decode: 304623.67 ms / 85 runs, 3.58381 s/token, 0.27903 tok/s.
- Quality flag: PASS; full answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2016, slot=7.44 MiB, hits=33716, misses=40316,
  preloads=0, hit_rate=45.5%.
- Pinned staging: copies=28304, host_stage=27237.971 ms, h2d=5985.814 ms.
- Up/gate profile: calls=2381, total=25.491 ms/call.
- Down profile: calls=4506, stage=10.890 ms/call, total=11.056 ms/call.

Comparison:

- Accepted Phase 2H `-n 96`: 295113.58 ms / 85 runs,
  3.47192 s/token, 0.29 tok/s.
- Phase 2U `-n 96`: 304623.67 ms / 85 runs,
  3.58381 s/token, 0.27903 tok/s.

Analysis:

- The full run does not reproduce the `-n 32` benefit.
- Down parallel staging lowers the pinned staging counters, but the end-to-end
  decode regresses:
  - up/gate total worsens from 22.785 ms/call to 25.491 ms/call.
  - down total only improves slightly from 11.781 ms/call to 11.056 ms/call.
  - full decode is 9510.09 ms slower than Phase 2H.
- This suggests the parallel copy threads/streams are shifting work into
  synchronization or GPU scheduling contention that is not represented by the
  local down-stage event timing.

Decision:

- Reject Phase 2U for full `-n 96`.
- Do not use `GGML_MOE_DOWN_PARALLEL_STAGE=1` in the accepted config.
- No code rollback is needed because this was env-only.
- Keep accepted Phase 2H as the current best full `-n 96` configuration.
- Next work should add wall-time instrumentation around up/gate, down, cache
  insert/copy, and stream synchronization because repeated env-only attempts
  improve local CUDA-event buckets without improving end-to-end decode.

## Next candidate: Phase 3A wall-time MoE profile instrumentation

Design timestamp: 2026-07-02 18:11 CST.

Current bottleneck:

- Accepted Phase 2H full `-n 96` eval time is 295113.58 ms, but the visible
  MoE CUDA-event buckets account for far less:
  - up/gate: 2381 calls * 22.785 ms/call = about 54.2s.
  - down: 4506 calls * 11.781 ms/call = about 53.1s.
  - pinned host staging: 50153.894 ms, partly included in stage buckets.
- Env-only optimizations repeatedly improved one local bucket but worsened full
  eval:
  - down prefetch improved down stage but slowed n96.
  - down parallel staging reduced staging counters but slowed n96.
  - LFU/LRU improved n32 slightly but worsened n96 cache behavior.
- Therefore the next bottleneck is not sufficiently visible in current logs.

Hypothesis:

Add default-off instrumentation under the existing `GGML_MOE_BATCH_PROFILE=1`
path:

- `wall_ms` for each up/gate call from function entry after buffer setup to
  return.
- `wall_ms` for each down batch call over the same scope.
- Report per-call `wall` and `wall_gap`, where:
  - up/gate `wall_gap = wall - stage - quant - kernel - d2h - scatter`.
  - down `wall_gap = wall - stage - quant - kernel - d2h - scatter`.

This does not change math, cache policy, allocation, or scheduling. It only
adds `std::chrono::steady_clock` timing when profiling is already enabled.

Theoretical value:

- If wall time closely matches CUDA buckets, then the unexplained decode time
  is outside these MoE stream functions and the next target should move upward
  in the graph/runtime.
- If wall_gap is large in up/gate or down, then hidden CPU work, lock waits,
  cache insertion, stream synchronization, or staging thread joins are the real
  bottleneck. That gives a concrete next optimization target instead of
  continuing env sweeps.

Execution:

- Implement the instrumentation locally.
- Build on the remote CUDA build directory.
- Run a cold `-n 4` Phase 3A baseline with accepted Phase 2H runtime env and
  `GGML_MOE_BATCH_PROFILE=1`.
- Host RAM, VRAM, TTFT, and quality gates still apply even though this is an
  instrumentation run.
- If `-n 4` passes and output is unchanged, run `-n 32` to get stable wall-gap
  attribution. Full `-n 96` is optional for attribution and should be run only
  if n32 still leaves ambiguity.

Acceptance:

- Instrumentation code is acceptable only if it is default-off outside
  `GGML_MOE_BATCH_PROFILE`.
- France output must remain semantically correct and coherent.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- TTFT must remain <=106331.72 ms for any cold run used as evidence.
- The run must report `wall` and `wall_gap` for both up/gate and down profile
  lines.

Rollback:

- If instrumentation changes output quality, causes launch/read failures,
  breaks build, or adds overhead outside profiling mode, revert the code.
- This phase is diagnostic; do not promote token-rate changes from it unless a
  separate optimized run without extra instrumentation passes all Phase 2H
  acceptance gates and improves full `-n 96`.

Implementation result timestamp: 2026-07-02 18:17 CST.

Code:

- Commit: `6536b772f cuda: add Kimi MoE wall-time profiling`.
- Remote build: `/root/lfz/llama.cpp-vendor-kimi/build-cuda-batch`.
- Build result: PASS; `llama-completion` linked successfully.

Smoke run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-181709Z-n4-phase3a-wall-profile`

Measured result:

- Commit/config: `6536b772f`, accepted Phase 2H runtime env, with
  `GGML_MOE_BATCH_PROFILE=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 104723.63 ms, inside the 106331.72 ms gate.
- Decode: 14167.59 ms / 3 runs, 4.72253 s/token, 0.21175 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Wall profile printed correctly:
  - up/gate: total=27.290 ms/call, wall=27.420 ms/call,
    wall_gap=0.130 ms/call.
  - down: total=13.593 ms/call, wall=14.300 ms/call,
    wall_gap=0.707 ms/call.

Decision:

- Instrumentation builds and the `-n 4` smoke passes quality, RAM, VRAM, TTFT,
  launch/read, and profile-output gates.
- Continue to cold `-n 32` to get stable wall-gap attribution. The `-n 4`
  sample suggests that the hidden gap is small inside the MoE functions, but
  `-n 32` is needed before moving the bottleneck search up the graph.

Attribution run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-182027Z-n32-phase3a-wall-profile`

Measured result:

- Commit/config: `6536b772f`, accepted Phase 2H runtime env, with
  `GGML_MOE_BATCH_PROFILE=1`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 105745.41 ms, inside the 106331.72 ms gate.
- Decode: 97526.35 ms / 31 runs, 3.14601 s/token, 0.31786 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2016, slot=7.44 MiB, hits=12892, misses=14052,
  preloads=0, hit_rate=47.8%.
- Pinned staging: copies=13135, host_stage=17934.216 ms, h2d=2843.707 ms.
- Wall profile:
  - up/gate: calls=869, total=18.848 ms/call, wall=18.876 ms/call,
    wall_gap=0.028 ms/call.
  - down: calls=1644, total=9.186 ms/call, wall=9.271 ms/call,
    wall_gap=0.084 ms/call.

Analysis:

- The MoE function wall gaps are tiny:
  - up/gate hidden gap is about 24 ms total.
  - down hidden gap is about 138 ms total.
- MoE function wall time accounts for about:
  - up/gate: 869 * 18.876 ms = 16.4s.
  - down: 1644 * 9.271 ms = 15.2s.
  - combined: about 31.6s of the 97.5s `-n 32` eval time.
- Therefore the remaining bottleneck is mostly outside the MoE stream functions
  themselves. Continuing to tune cache/preload/down-stage policy is unlikely to
  reach large gains without first profiling graph-level or non-MoE CUDA work.

Decision:

- Phase 3A instrumentation is accepted as diagnostic code: it is default-off
  outside `GGML_MOE_BATCH_PROFILE`, builds successfully, and preserves output
  quality under the cold-start gates.
- Do not claim a token-rate improvement from Phase 3A.
- Next candidate should add or use graph-level per-op/per-split timing to find
  the non-MoE decode bucket before making another optimization attempt.

## Next candidate: Phase 3B graph submit/synchronize wall-time profile

Design timestamp: 2026-07-02 18:29 CST.

Current bottleneck:

- Phase 3A shows MoE function wall gaps are tiny:
  - up/gate wall_gap=0.028 ms/call on `-n 32`.
  - down wall_gap=0.084 ms/call on `-n 32`.
- MoE wall time accounts for about 31.6s of the 97.5s `-n 32` eval time.
- `llama_context::graph_compute()` submits the graph asynchronously via
  `ggml_backend_sched_graph_compute_async`, while `llama_context::synchronize()`
  waits for completion and updates the perf counters. Current logs do not split
  graph submit time from synchronize/wait time.

Hypothesis:

Add default-off graph wall profiling controlled by:

- `LLAMA_KIMI_GRAPH_PROFILE=1`

Record aggregate wall time for:

- `graph_compute_async` submit calls.
- `ggml_backend_sched_synchronize` wait calls.
- queued-token class at synchronization time: prompt (`n_queued_tokens > 1`) vs
  decode (`n_queued_tokens == 1`).

This will determine whether the remaining time is mostly backend graph
execution/synchronization, CPU submission overhead, or outside the graph path.

Theoretical value:

- If synchronize decode wall time roughly matches eval time, the next
  optimization target is the scheduled graph/non-MoE CUDA kernels.
- If graph submit time is large, the target is graph scheduling/reuse or graph
  construction.
- If neither is large, the target is outside context graph execution, such as
  sampling/output transfer or application loop overhead.

Execution:

- Implement the instrumentation in `src/llama-context.cpp` with no effect
  unless `LLAMA_KIMI_GRAPH_PROFILE=1`.
- Build remotely.
- Run cold `-n 4` with accepted Phase 2H runtime env plus:
  - `GGML_MOE_BATCH_PROFILE=1`
  - `LLAMA_KIMI_GRAPH_PROFILE=1`
- If `-n 4` passes, run cold `-n 32`.

Acceptance:

- Code must build.
- Default path must not change when `LLAMA_KIMI_GRAPH_PROFILE` is unset.
- Runs must keep host RAM under 16GB, TTFT <=106331.72 ms, and semantically
  correct France output.
- Logs must print graph profile totals for submit and synchronize.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.

Rollback:

- Revert if build fails, default execution path changes, output quality fails,
  or graph profile adds unacceptable overhead to the attribution runs.

Result timestamp: 2026-07-02 17:32 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-173240Z-n4-phase2s-vram-cache15200`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config with
  `GGML_MOE_VRAM_CACHE_MIB=15200`, no split cache, no down prefetch.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31540 MiB used, 570 MiB free.
- TTFT: 106300.84 ms, inside the 106331.72 ms gate by only 30.88 ms.
- Decode: 14654.49 ms / 3 runs, 4.88483 s/token, 0.20472 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2043, slot=7.44 MiB, hits=731, misses=1797,
  preloads=0, hit_rate=28.9%.
- Pinned staging: copies=1783, host_stage=2748.542 ms, h2d=387.967 ms.
- Up/gate profile: calls=85, total=29.518 ms/call.
- Down profile: calls=160, stage=14.049 ms/call, total=14.227 ms/call.

Decision:

- The smoke passes all hard gates, but TTFT headroom is only 30.88 ms. This is
  a very narrow margin for cold-start reproducibility.
- Continue to `-n 32` because the plan allows it after a passing smoke, but
  Phase 2S should be promoted only if the speed/cache gain is clear and the
  longer run still leaves TTFT below the gate.
- If `-n 32` is slower than Phase 2H or has similarly tight TTFT without clear
  cache/staging improvement, reject without running `-n 96`.

Result timestamp: 2026-07-02 17:35 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-173544Z-n32-phase2s-vram-cache15200`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config with
  `GGML_MOE_VRAM_CACHE_MIB=15200`, no split cache, no down prefetch.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31540 MiB used, 570 MiB free.
- TTFT: 100855.49 ms, inside the 106331.72 ms gate.
- Decode: 98327.66 ms / 31 runs, 3.17186 s/token, 0.31527 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache: slots=2043, slot=7.44 MiB, hits=12961, misses=13983,
  preloads=0, hit_rate=48.1%.
- Pinned staging: copies=13077, host_stage=18034.264 ms, h2d=2835.196 ms.
- Up/gate profile: calls=869, total=18.110 ms/call.
- Down profile: calls=1644, stage=8.915 ms/call, total=9.082 ms/call.

Comparison:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token,
  about 0.32 tok/s.
- Phase 2S `-n 32`: 98327.66 ms / 31 runs, 3.17186 s/token,
  0.31527 tok/s.

Decision:

- Reject Phase 2S without running full `-n 96`.
- Although this config uses more VRAM and passes RAM/TTFT/quality at `-n 32`,
  it is slower than Phase 2H on the matching medium-length run.
- The 15200 MiB and 15400 MiB results together show that simply enlarging the
  single down-cache budget does not improve the current full-run bottleneck:
  15400 MiB violates TTFT at smoke, and 15200 MiB slows `-n 32`.
- No code rollback is needed because this was env-only.

## Next candidate: Phase 2T LFU/LRU VRAM cache eviction policy

Design timestamp: 2026-07-02 17:51 CST.

Current bottleneck:

- Accepted Phase 2H full `-n 96` still has 40316 cache misses and
  50153.894 ms host staging time.
- Larger single-cache budgets did not help:
  - 15400 MiB violated TTFT at `-n 4`.
  - 15200 MiB passed `-n 32` gates but slowed decode versus Phase 2H.
- Down prefetch reduced local down-stage time but did not improve full `-n 96`.
- Phase 2J rejected broad profile preload/protection because it added large
  overhead. That does not rule out a policy-only eviction change without
  preload/protect.

Hypothesis:

Run the accepted Phase 2H config with:

- `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`
- no `GGML_MOE_VRAM_PROFILE`.
- no `GGML_MOE_VRAM_PROFILE_PROTECT`.
- no split cache.
- no down prefetch.
- cache budget remains 15000 MiB.

Default policy is effectively LRU. The LFU/LRU policy evicts the slot with the
lowest observed hit count, using LRU only as a tie-break. Kimi routing reuses a
small subset of experts repeatedly in the France prompt, so preserving high-hit
experts may reduce misses and host staging without increasing VRAM or changing
model math.

Theoretical upper bound:

- Full Phase 2H host staging is about 50.15s over 35787 staged copies.
- If LFU/LRU avoids even 2-5% of misses after warmup, expected full-run savings
  could be about 1-3s.
- The policy scan cost is similar to current eviction scanning because both
  inspect cache slots. Extra `slot_hits` comparisons should be much smaller
  than the staging bucket.
- Promotion requires full `-n 96` below 295113.58 ms / 85 runs while preserving
  all hard gates.

Execution:

- Cold `-n 4` first under:
  - `memory.max=16000000000`
  - `memory.swap.max=0`
  - `sync; echo 3 > /proc/sys/vm/drop_caches`
- If `-n 4` passes, run cold `-n 32`.
- Continue to full `-n 96` only if `-n 32` is not slower than Phase 2H `-n 32`
  or shows a clear hit-rate/staging improvement with acceptable TTFT.

Acceptance:

- Host RAM must stay under 16GB including page cache.
- VRAM should remain close to Phase 2H utilization without OOM.
- TTFT must stay <=106331.72 ms.
- Prompt must be:
  `Please introduce France in a short paragraph.`
- Output must be semantically correct and coherent.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Full `-n 96` must beat accepted Phase 2H full `-n 96`.
- If accepted, commit and push immediately with reproduction details.

Rollback:

- Reject if TTFT exceeds the gate, VRAM/RAM gates fail, output quality fails,
  launch/read failures appear, or full `-n 96` does not improve over Phase 2H.
- No code rollback should be needed because this is env-only.

Result timestamp: 2026-07-02 16:40 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-164058Z-n4-phase2o-split-cache`

Measured result:

- Config: Phase 2M split cache with `GGML_MOE_VRAM_CACHE_UPGATE_PCT=40`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31342 MiB used, 768 MiB free.
- TTFT: 102663.93 ms, inside the 106331.72 ms gate.
- Decode: 14358.45 ms / 3 runs, 4.78615 s/token, 0.20894 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache pools confirmed:
  - down: 1210 slots, 7.44 MiB slot, hits=383, misses=865,
    hit_rate=30.7%.
  - upgate: 1119 slots, 5.36 MiB slot, hits=380, misses=980,
    hit_rate=27.9%.

Decision:

- The `-n 4` smoke passes correctness, memory, VRAM, TTFT, cache allocation, and
  launch/read gates.
- Continue to cold `-n 96`.

Full-length result timestamp: 2026-07-02 16:44 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-164405Z-n96-phase2o-split-cache`

Measured result:

- Commit/config: `adf621b20`, Phase 2M split-cache code with
  `GGML_MOE_VRAM_CACHE_UPGATE_PCT=40`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31342 MiB used, 768 MiB free.
- TTFT: 103968.03 ms, inside the 106331.72 ms gate.
- Decode: 302382.85 ms / 85 runs, 3.55745 s/token, 0.28110 tok/s.
- Quality flag: PASS; full answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache pools:
  - down: 1210 slots, 7.44 MiB slot, hits=18825, misses=17191,
    hit_rate=52.3%.
  - upgate: 1119 slots, 5.36 MiB slot, hits=17370, misses=20726,
    hit_rate=45.6%.
- Pinned staging: copies=33515, host_stage=46970.468 ms, h2d=7236.745 ms.
- Up/gate profile: calls=2381, total=23.427 ms/call.
- Down profile: calls=4506, stage=10.954 ms/call, total=11.121 ms/call.

Comparison:

- Phase 2H `-n 96`: 3.47192 s/token, 0.29 tok/s.
- Phase 2M pct45 `-n 96`: 3.52464 s/token, 0.28372 tok/s.
- Phase 2N pct35 `-n 96`: 3.49838 s/token, 0.28585 tok/s.
- Phase 2O pct40 `-n 96`: 3.55745 s/token, 0.28110 tok/s.
- pct40 balances hit rates but does not improve end-to-end decode; up/gate
  remains slower than Phase 2H and total staging stays high.

Decision:

- Reject pct40 for full-length promotion.
- Keep Phase 2H as the current best full `-n 96` configuration.
- Keep Phase 2M split-cache code only as default-off functionality and as the
  current best `-n 32` constrained config.
- Do not continue fixed split-budget sweeps for `-n 96` without a new
  bottleneck explanation. The next n96-focused design should target a different
  mechanism: reducing graph replay/upgate cost, reducing the number of expert
  loads, or making cache partitioning adaptive rather than fixed.

## Next candidate: Phase 2P down prefetch overlap

Design timestamp: 2026-07-02 17:02 CST.

Current bottleneck:

- Accepted Phase 2H full `-n 96` visible buckets:
  - Up/gate profile: 2381 calls, 22.785 ms/call, about 54.2s total.
  - Down profile: 4506 calls, stage=11.616 ms/call, about 52.3s visible down
    staging time.
  - Pinned staging: host_stage=50153.894 ms, h2d=7758.831 ms.
- Split-cache budget tuning improved some cache hit-rate buckets, but all
  tested fixed budgets missed Phase 2H at full `-n 96`.
- The code already has a default-off down prefetch hook:
  `GGML_MOE_PREFETCH_DOWN=1`, called from the up/gate path via
  `preload_registered_down_for_active`. It can enqueue down expert loads on the
  prefetch stream before the down layer uses them.

Hypothesis:

Enable down prefetch on top of the accepted Phase 2H full configuration:

- `GGML_MOE_PREFETCH_DOWN=1`
- `GGML_MOE_PREFETCH_DOWN_DEPTH=8`

Do not combine with split cache initially. This isolates whether overlapping
down expert staging with up/gate compute reduces full-length decode. The
mechanism should not change math or routing; it only changes when down experts
enter the VRAM cache.

Theoretical upper bound:

- If down prefetch can hide half of the Phase 2H down staging bucket, it could
  save about 26s on the 295.1s decode, improving to about 269s / 85 runs =
  3.16 s/token.
- A more realistic bound is lower because prefetch can evict useful cache lines
  and still contends for H2D bandwidth. If it hides 10-20% of down staging, the
  expected gain is about 5-10s, enough to beat Phase 2H if quality and TTFT hold.

Correctness and performance risks:

- Correctness risk is low if cache keys remain correct, but stale or wrongly
  prefetched down weights could corrupt output.
- Performance risk is moderate: prefetch may evict useful up/gate or down
  entries, increasing misses. The logs must show `down prefetch` useful rate,
  evicted-unused count, and cache hit rates.

Acceptance:

- Run cold `-n 4` first under `memory.max=16000000000`, `memory.swap.max=0`,
  and `drop_caches`.
- Host RAM must remain under the 16GB cgroup cap including page cache.
- VRAM should remain near full without OOM.
- TTFT must remain <=106331.72 ms.
- France answer must be semantically correct and coherent.
- Logs must show down prefetch activity.
- `launch_failures=0`, `read_failures=0`, and down batch profile remains active.
- If `-n 4` passes, run cold `-n 32` to check miss/eviction behavior.
- Promote to full `-n 96` only if `-n 32` is not slower than Phase 2H `-n 32`
  and all gates pass.
- Full `-n 96` promotion requires beating Phase 2H `-n 96`
  (0.29 tok/s / 3.47192 s/token).

Rollback:

- Reject this config if quality fails, TTFT exceeds the gate, RAM/VRAM gates
  fail, prefetch does not activate, down prefetch useful rate is poor with high
  evicted-unused count, launch/read failures appear, or token rate does not
  improve over the matching Phase 2H token count.

Result timestamp: 2026-07-02 17:03 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-165331Z-n4-phase2p-down-prefetch`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus:
  - `GGML_MOE_PREFETCH_DOWN=1`
  - `GGML_MOE_PREFETCH_DOWN_DEPTH=8`
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 105222.66 ms, inside the 106331.72 ms gate but with only about
  1.1s headroom.
- Decode: 14183.67 ms / 3 runs, 4.72789 s/token, 0.21151 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Down prefetch: loads=404, hits=404, evicted_unused=0, useful_rate=100.0%.
- Cache: slots=2016, slot=7.44 MiB, hits=1145, misses=1399, preloads=404,
  hit_rate=45.0%.
- Pinned staging: copies=1773, host_stage=2759.033 ms, h2d=386.877 ms.
- Up/gate profile: calls=85, total=27.275 ms/call.
- Down profile: calls=160, stage=5.341 ms/call, total=5.518 ms/call.

Decision:

- The `-n 4` smoke passes correctness, memory, VRAM, TTFT, prefetch activation,
  useful-rate, and launch/read gates.
- Continue to cold `-n 32`.
- Watch TTFT closely because this path has less headroom than Phase 2H.

Result timestamp: 2026-07-02 17:06 CST.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-165639Z-n32-phase2p-down-prefetch`

Measured result:

- Commit/config: `adf621b20`, accepted Phase 2H config plus down prefetch.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31338 MiB used, 772 MiB free.
- TTFT: 101970.04 ms, inside the 106331.72 ms gate.
- Decode: 94356.18 ms / 31 runs, 3.04375 s/token, 0.32854 tok/s.
- Quality flag: PASS; answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Down prefetch: loads=3196, hits=3196, evicted_unused=0,
  useful_rate=100.0%.
- Cache: slots=2016, slot=7.44 MiB, hits=16096, misses=10864,
  preloads=3196, hit_rate=59.7%.
- Pinned staging: copies=13123, host_stage=17959.990 ms, h2d=2848.131 ms.
- Up/gate profile: calls=869, total=18.573 ms/call.
- Down profile: calls=1644, stage=3.609 ms/call, total=3.776 ms/call.

Comparison against accepted Phase 2H `-n 32`:

- Phase 2H `-n 32`: 95613.73 ms / 31 runs, 3.08431 s/token, 0.32 tok/s.
- Phase 2P `-n 32`: 94356.18 ms / 31 runs, 3.04375 s/token, 0.32854 tok/s.
- Down prefetch works as intended for `-n 32`: down stage falls from
  9.065 ms/call to 3.609 ms/call with 100% useful prefetch hits.

Decision:

- Continue to cold full `-n 96`.
- Full promotion still requires beating Phase 2H `-n 96` while preserving
  quality, TTFT, RAM, VRAM, prefetch usefulness, and launch/read gates.

Result timestamp: 2026-07-01 16:28 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-162848Z-n4-phase2n-split-cache`

Measured result:

- Config: Phase 2M split cache with `GGML_MOE_VRAM_CACHE_UPGATE_PCT=35`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31336 MiB used, 774 MiB free.
- TTFT: 99467.20 ms, inside the 106331.72 ms gate.
- Decode: 14323.00 ms / 3 runs, 4.77433 s/token, 0.20945 tok/s.
- Quality: PASS for the tiny smoke; answer was `France is a country`.
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache pools confirmed:
  - down: 1310 slots, 7.44 MiB slot, hits=383, misses=865,
    hit_rate=30.7%.
  - upgate: 979 slots, 5.36 MiB slot, hits=380, misses=980,
    hit_rate=27.9%.

Decision:

- The `-n 4` smoke passes correctness, memory, VRAM, TTFT, cache allocation, and
  launch/read gates.
- Continue directly to cold `-n 96` because Phase 2N is a full-length budget
  tuning step.

Full-length result timestamp: 2026-07-01 16:31 UTC.

Run:
`/root/lfz/runs/vendor-kimi-token-rate/20260701-163150Z-n96-phase2n-split-cache`

Measured result:

- Commit/config: `adf621b20`, Phase 2M split-cache code with
  `GGML_MOE_VRAM_CACHE_UPGATE_PCT=35`.
- Host RAM peak: 14.901 GiB, inside the 16GB cgroup cap.
- VRAM peak: 31336 MiB used, 774 MiB free.
- TTFT: 103692.99 ms, inside the 106331.72 ms gate.
- Decode: 297362.72 ms / 85 runs, 3.49838 s/token, 0.28585 tok/s.
- Quality flag: PASS; full answer:
  `France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its beautiful countryside, wine regions, and historic cities such as Lyon and Marseille. It plays a major role in European and global politics as a founding member of the European Union.<|im_end|> [end of text]`
- `launch_failures=0`, `read_failures=0`, `down_profile=true`.
- Cache pools:
  - down: 1310 slots, 7.44 MiB slot, hits=19429, misses=16587,
    hit_rate=53.9%.
  - upgate: 979 slots, 5.36 MiB slot, hits=15631, misses=22465,
    hit_rate=41.0%.
- Pinned staging: copies=34488, host_stage=47046.256 ms, h2d=7380.708 ms.
- Up/gate profile: calls=2381, total=23.552 ms/call.
- Down profile: calls=4506, stage=10.415 ms/call, total=10.582 ms/call.

Comparison:

- Phase 2H `-n 96`: 3.47192 s/token, 0.29 tok/s.
- Phase 2M pct45 `-n 96`: 3.52464 s/token, 0.28372 tok/s.
- Phase 2N pct35 `-n 96`: 3.49838 s/token, 0.28585 tok/s.
- Moving from pct45 to pct35 improves down hit rate and total speed, but it
  over-constrains the upgate pool: upgate hit_rate drops from 47.2% to 41.0%,
  and up/gate total regresses from 22.493 ms/call to 23.552 ms/call.

Decision:

- Reject pct35 for full-length promotion.
- Keep Phase 2H as the current best full `-n 96` configuration.
- Next split-cache budget attempt, if pursued, should test an intermediate
  upgate pct around 40 rather than moving further toward down.

## Next candidate: Phase 2O intermediate split-cache budget

Design timestamp: 2026-07-02 16:41 CST.

Current bottleneck:

- Phase 2M pct45 and Phase 2N pct35 both passed quality, TTFT, host RAM, and
  VRAM gates, but neither beat Phase 2H at full `-n 96`.
- pct45 over-allocated upgate relative to down:
  - down hit_rate=50.6%, upgate hit_rate=47.2%.
  - Decode: 3.52464 s/token.
- pct35 over-corrected toward down:
  - down hit_rate=53.9%, upgate hit_rate=41.0%.
  - Decode: 3.49838 s/token.
- Phase 2H full target remains 3.47192 s/token / 0.29 tok/s.

Hypothesis:

Test the midpoint:

- `GGML_MOE_VRAM_CACHE_SPLIT=1`
- `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=40`

This should give the upgate pool more capacity than pct35 while preserving more
down capacity than pct45. If the optimal balance is between the prior two
points, pct40 should improve both total staging pressure and up/gate regression.

Theoretical upper bound:

- pct35 improved decode over pct45 by 2.23198s total but stayed 2.24914s slower
  than Phase 2H.
- pct40 can at best recover the upgate regression from pct35 while retaining
  some down-hit improvement. The likely gain range is small, about 1-4s on full
  `-n 96`.
- Promotion requires crossing below 295113.58 ms / 85 runs.

Acceptance:

- Run cold `-n 4` first under `memory.max=16000000000`, `memory.swap.max=0`,
  and `drop_caches`.
- Host RAM must remain under the 16GB cgroup cap including page cache.
- VRAM should remain near full without OOM.
- TTFT must remain <=106331.72 ms.
- France answer must be semantically correct and coherent.
- Logs must show two active cache pools and no allocation retry/failure.
- `launch_failures=0`, `read_failures=0`, and down batch profile remains active.
- If `-n 4` passes, run cold `-n 96`.
- Promote only if `-n 96` is faster than accepted Phase 2H `-n 96`
  (0.29 tok/s / 3.47192 s/token) with all gates passing.

Rollback:

- Reject this config if quality fails, TTFT exceeds the gate, RAM/VRAM gates
  fail, cache allocation fails, launch/read failures appear, or `-n 96` does
  not improve over Phase 2H.
