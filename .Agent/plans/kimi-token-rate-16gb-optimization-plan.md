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

Implement Phase 2C debug instrumentation to compare the first stream one-path
expert output against the accepted non-stream/CPU reference. The next code patch
must be diagnostic or correctness-only; it is not a performance promotion until
stream-only output passes semantic smoke.
