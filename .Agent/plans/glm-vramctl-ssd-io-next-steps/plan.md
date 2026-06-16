# GLM SSD I/O Next Steps

## Goal

Further improve GLM MoE inference speed by reducing expert-pack read latency
and CPU/GPU staging overhead, building on the current vramctl-style
`GGML_MOE_IO_BACKEND=iouring` implementation.

Primary metrics:

- total runtime
- prompt eval time
- eval time
- eval tokens/s
- expert-pack read wait/submit time
- H2D staging wait time

## Constraints

- Work only inside `/home/wici/lfz`.
- Do not modify or delete anything outside `/home/wici/lfz`.
- Keep all plans, logs, scripts, summaries, and generated artifacts under this
  task directory.
- Preserve the strict validation line:
  `MemoryMax=2G`, `MemorySwapMax=0`, `GGML_MOE_RAM_TIER_MIB=0`.
- Do not mix SSD I/O changes with routing, SER, VRAM cache policy, or profile
  changes unless explicitly starting a separate task.
- Keep current `direct` behavior as the stable fallback.

## Current Baseline

From `.Agent/plans/glm-vramctl-ssd-io-pipeline/plan.md`:

Important comparability note: this SSD I/O task uses a fixed single prompt as
an I/O microbenchmark:
`Answer with one short sentence: why does NVMe latency matter for MoE inference?`.
It is not the same workload as the dynamic-k `smoke4` dataset benchmark, whose
current-machine baseline was `0.881 tok/s` on
`.Agent/plans/repro-dynamic-k-ser-current-machine/smoke.jsonl`.

| backend | n | total_ms | prompt_eval_ms | eval_ms | eval tok/s | VRAM hit | expert reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| direct | 84 | 103564.18 | 11449.87 | 81328.70 | 1.02 | 7.1% | `direct=140550` |
| iouring | 84 | 100881.00 | 11184.83 | 78904.21 | 1.05 | 7.1% | `direct=1800`, `iouring=138750` |

Current `iouring` win: `2.6%` total, `3.0%` eval. This is real but modest.
The next changes should target the remaining serialized wait and staging costs.

Observed bottleneck hints:

- `iouring_wait_us=95044145` on the long run, close to total runtime. This
  means the current code still spends most time waiting for SSD completions.
- Queue depth is effectively capped by `GGML_MOE_STAGE_PINNED_SLOTS=16`, even
  though `GGML_MOE_IO_DEPTH=32`.
- `GGML_MOE_IO_BYTES=2097152` is parsed and reported, but current batched reads
  still issue one request per expert entry. It does not yet split reads into
  2 MiB chunks.
- Profile preload still uses the existing direct read path (`direct_reads=1800`)
  even under `GGML_MOE_IO_BACKEND=iouring`.
- The current per-batch helper submits an initial window, then often submits
  one replacement request after each completion. This keeps the code simple but
  may leave queue depth underutilized for small per-layer batches.

## Plan

### Phase 1: Instrument Before Changing Behavior

1. Add per-run counters for:
   - maximum effective io_uring inflight depth;
   - average effective inflight depth;
   - number of `io_uring_submit()` calls;
   - number of CQEs reaped per wait loop;
   - batch job count histogram, at least buckets: `1`, `2-4`, `5-8`, `9-16`,
     `17-32`, `>32`;
   - H2D enqueue time and staging-slot wait time split by up/gate/down rings.
2. Add optional trace CSV under an env gate, for example
   `GGML_MOE_IO_TRACE_OUT`, with fields:
   `seq,tensor,expert_idx,bytes,backend,batch_jobs,inflight,submit_us,wait_us,h2d_us`.
3. Re-run the current direct/iouring `-n 84` baseline to confirm the new
   counters do not materially perturb runtime.

Acceptance:

- build passes;
- `direct` and `iouring` outputs still complete under 2 GB;
- instrumentation overhead under `1%` when trace CSV is disabled.

### Phase 2: Increase Real Queue Depth

1. Sweep `GGML_MOE_STAGE_PINNED_SLOTS` with the current code:
   - `16` baseline;
   - `24`;
   - `32`;
   - `48`, only if 32 is stable under the 2 GB cgroup.
2. Keep `GGML_MOE_IO_DEPTH` equal to or below staging slots for each run.
3. Select a new default only if both total runtime and eval time improve
   without increasing read failures or OOM risk.

Rationale:

- Current depth is clipped by staging slots, so `IO_DEPTH=32` does not fully
  apply.
- Each slot is roughly `3.84-4.59 MiB`; increasing slots from 16 to 32 costs
  tens of MiB, which should be acceptable inside the existing command because
  `GGML_CUDA_NO_PINNED=1` avoids the huge full-model pinned allocation.

Acceptance:

- best result improves long-run eval time by at least `2%` over current
  iouring baseline, or the sweep documents that slot depth is not the limiting
  factor.

### Phase 3: Batch Submission and Completion Reaping

1. Modify `expert_pack_iouring_copy_jobs()` to submit replacement SQEs in
   batches instead of one submit after each completion.
2. Use `io_uring_peek_batch_cqe()` or equivalent CQE draining when completions
   are already available, falling back to `io_uring_wait_cqe()` only when none
   are ready.
3. Track one `submit_us` per flush, not per whole helper call only.
4. Keep fallback behavior identical to current direct path.

Rationale:

- The current wait loop is simple but can pay too many submit/wait transitions.
- More aggressive CQE draining should reduce syscall pressure and improve SSD
  queue utilization.

Acceptance:

- no increase in `iouring_fallbacks`;
- total runtime improves by at least `1%` over Phase 2 best, or the trace shows
  there are not enough ready CQEs to benefit.

### Phase 4: Use `GGML_MOE_IO_BYTES` for Chunked Large Reads

1. Implement chunked io_uring reads for expert entries larger than
   `GGML_MOE_IO_BYTES`.
2. For each expert job, submit `ceil(expert_bytes / io_bytes)` linked or
   tracked chunks into the same staging slot.
3. Enqueue H2D only after all chunks for that expert complete.
4. Start with `2 MiB` chunks, then sweep:
   - `1 MiB`;
   - `2 MiB`;
   - `4 MiB`;
   - full-entry read, as current behavior.

Rationale:

- vramctl's SSD path is designed around fixed-size queued I/O.
- Current expert entries are about `3.84 MiB` and `4.59 MiB`, so fixed chunking
  could improve device-level parallelism, but it can also add CQE bookkeeping
  overhead. This must be measured.

Acceptance:

- choose chunking only if it improves long-run total or eval time by at least
  `1.5%`;
- otherwise keep full-entry reads and document that this SSD/model combination
  prefers fewer larger reads.

### Phase 5: io_uring for Profile Preload

1. Convert profile preload reads from direct `pread` to batched io_uring under
   `GGML_MOE_IO_BACKEND=iouring`.
2. Keep a direct fallback for preload.
3. Measure load time and first-token latency separately.

Rationale:

- Long iouring run still shows `direct_reads=1800` from preload.
- This probably affects load/prompt time more than steady-state eval, but it is
  low risk if isolated behind the existing backend gate.

Acceptance:

- load time or prompt eval improves without increasing total runtime;
- no change to runtime miss behavior.

### Phase 6: Optional Direct-to-GPU Read Investigation

Only after Phases 1-5:

1. Check whether this kernel, filesystem, NVIDIA driver, and CUDA stack support
   practical GPU-direct storage style reads for this workflow.
2. Prototype only behind a separate env gate.
3. Keep the current pinned host staging path as the stable fallback.

Rationale:

- The largest remaining cost may be SSD-to-host plus host-to-device staging.
- Direct-to-GPU is high risk and platform-sensitive, so it should not be mixed
  into the main io_uring path until simpler changes are exhausted.

## Validation Matrix

Use the known-good GLM command shape from prior runs:

- `GGML_CUDA_NO_PINNED=1`
- `GGML_MOE_EXPERT_PACK=.../glm51-iq3xxs.expert-pack`
- `GGML_MOE_RAM_TIER_MIB=0`
- `GGML_MOE_VRAM_CACHE_MIB=12288`
- `GGML_MOE_VRAM_PROFILE=.../wici-glm51-interactive-n84-expert-top8.runtime.csv`
- `-c 2048 -ngl 79 -fa on -mla 3 -cmoe --defer-experts --no-warmup`
- `MemoryMax=2G`, `MemorySwapMax=0`

For every candidate:

1. Build `llama-cli`.
2. Run `-n 4` smoke for direct and candidate backend.
3. Run `-n 84` direct baseline and candidate backend.
4. Record:
   - stdout;
   - stderr;
   - exit status;
   - parsed summary JSON or table;
   - exact command.
5. Accept only if the candidate beats direct and the current iouring baseline
   on long-run total/eval time without correctness, failure, or OOM regression.

## Initial Priority

Start with Phase 1 and Phase 2. They are lowest risk and directly test whether
the current implementation is queue-depth limited. Do not implement chunking or
preload conversion until the new counters show whether the ring is actually
underfilled.

## Phase 1/2 Results

Implemented:

- Added low-overhead aggregate io_uring counters in
  `ggml/src/ggml-cuda/moe_stream_batch.cu`:
  - total io_uring batches;
  - submit calls;
  - wait calls;
  - CQEs;
  - average and max effective inflight depth;
  - batch-size histogram: `1`, `2-4`, `5-8`, `9-16`, `17-32`, `gt32`.
- Added the same ring-local counters to pinned staging reports so main/gate
  rings can be separated.
- Added task-local runner:
  `.Agent/plans/glm-vramctl-ssd-io-next-steps/run_io_sweep.py`.
- Did not add CSV trace in this phase. The aggregate counters were enough to
  diagnose queue depth without adding per-read file I/O overhead.

Validation:

- Build passed:
  `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`.
- Whitespace check passed:
  `git diff --check`.
- No `llama-cli` validation process remains running after the sweep.

Valid `-n 84` runs, all under `MemoryMax=2G`, `MemorySwapMax=0`,
`GGML_MOE_RAM_TIER_MIB=0`:

| run | backend | slots | depth | total_ms | prompt_eval_ms | eval_ms | eval tok/s | VRAM hit | inflight avg/max | read failures | fallbacks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `baseline-direct-slots16-depth16-n84` | direct | 16 | 16 | 101096.39 | 11085.13 | 79360.13 | 1.05 | 7.1% | n/a | 0 | 0 |
| `baseline-iouring-slots16-depth16-n84` | iouring | 16 | 16 | 99688.27 | 11008.68 | 78081.25 | 1.06 | 7.1% | 3.63 / 8 | 0 | 0 |
| `sweep-iouring-slots24-depth24-n84` | iouring | 24 | 24 | 100388.37 | 11103.06 | 78887.12 | 1.05 | 7.1% | 3.63 / 8 | 0 | 0 |
| `sweep-iouring-slots32-depth32-n84` | iouring | 32 | 32 | 100032.04 | 11120.32 | 78569.10 | 1.06 | 7.1% | 3.63 / 8 | 0 | 0 |
| `sweep-iouring-slots16-depth8-n84` | iouring | 16 | 8 | 100713.38 | 10947.17 | 79074.26 | 1.05 | 7.1% | 3.63 / 8 | 0 | 0 |
| `sweep-iouring-slots8-depth8-n84` | iouring | 8 | 8 | 101141.10 | 10818.35 | 79755.74 | 1.04 | 7.1% | 3.63 / 8 | 0 | 0 |

Key findings:

- Current `iouring` remains faster than `direct` in this rerun:
  - total runtime: `101096.39 ms` direct vs `99688.27 ms` iouring;
  - eval time: `79360.13 ms` direct vs `78081.25 ms` iouring.
- Phase 1 counters show this workload is not queue-depth limited by
  `GGML_MOE_STAGE_PINNED_SLOTS=16`:
  - all valid iouring runs have `inflight_max=8`;
  - average inflight is only `3.63`;
  - batch histogram is only `2-4:12478` and `5-8:12426`; no batches reach
    `9-16`, `17-32`, or `gt32`.
- Increasing slots/depth to `24` or `32` does not improve runtime because the
  model/runtime never presents more than eight read jobs to a ring batch.
- Reducing to slots/depth `8` is worse, so keep `slots=16` as the stable
  default/working configuration.

Conclusion:

- Phase 1 achieved the requested observability and confirmed the current
  `iouring` implementation still provides a measured improvement over direct.
- Phase 2 did not find a better slot/depth configuration. The best measured
  setting remains `GGML_MOE_STAGE_PINNED_SLOTS=16` with effective
  `GGML_MOE_IO_DEPTH=16` or higher.
- Further speedup should not come from larger staging rings. The next aligned
  optimization is Phase 3: reduce per-job wait/submit overhead by draining
  ready CQEs in batches and submitting replacement SQEs in groups.

## Dynamic-K Smoke4 I/O Impact

Dataset and command shape:

- Dataset: `.Agent/plans/repro-dynamic-k-ser-current-machine/smoke.jsonl`.
- Four samples, `-n 4`, same scoring as `smoke4-baseline`.
- Same 2 GB cgroup constraints:
  `MemoryMax=2G`, `MemorySwapMax=0`, `GGML_MOE_RAM_TIER_MIB=0`.
- Task-local runner:
  `.Agent/plans/glm-vramctl-ssd-io-next-steps/run_smoke_io_compare.py`.

Important implementation note:

- The original dynamic-k smoke command does not enable the parallel staging
  branches where the current io_uring hook is implemented.
- With only `GGML_MOE_IO_BACKEND=iouring`, the smoke run still reports
  `iouring_reads=0` and all expert reads remain direct.
- To measure the current io_uring method, the comparable pair must enable:
  `GGML_MOE_STREAM_UP_GATE_PARALLEL=1`,
  `GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE=1`,
  `GGML_MOE_STREAM_UP_GATE_SPLIT_STAGE=1`, and
  `GGML_MOE_DOWN_PARALLEL_STAGE=1`.

Results:

| run | staging flags | backend | accuracy | total_ms | wall_s | prompt eval tok/s | eval_ms | eval tok/s | direct reads | io_uring reads | VRAM hit |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke4-direct` | off | direct | 3/4 | 180642.90 | 184.01 | 3.505 | 13707.86 | 0.875 | 25962 | 0 | 13.5% |
| `smoke4-iouring` | off | iouring requested, not used | 3/4 | 183077.00 | 186.40 | 3.474 | 14536.37 | 0.826 | 25962 | 0 | 13.5% |
| `smoke4-stage-direct` | on | direct | 3/4 | 179478.76 | 182.83 | 3.502 | 11188.20 | 1.073 | 25962 | 0 | 13.5% |
| `smoke4-stage-iouring` | on | iouring | 3/4 | 180775.20 | 184.28 | 3.459 | 10841.54 | 1.107 | 7200 | 18762 | 13.5% |

Impact vs the current-machine dynamic-k baseline:

- Historical current-machine `smoke4-baseline`: `0.881 tok/s`.
- Fresh original direct rerun: `0.875 tok/s`; this is a small `-0.6%` rerun
  difference and confirms the baseline is reproducible.
- Enabling parallel staging with direct reads improves eval throughput to
  `1.073 tok/s`, `+22.5%` vs fresh original direct.
- Switching that staging path from direct to io_uring improves eval throughput
  further to `1.107 tok/s`, `+3.2%` vs staging-direct and `+26.4%` vs fresh
  original direct.
- Total wall time does not materially improve because this smoke benchmark
  generates only 12 eval tokens total and is dominated by repeated model load
  plus prompt eval. The best eval run, `smoke4-stage-iouring`, is `184.28 s`
  wall vs `184.01 s` for fresh original direct.
- Quality is unchanged in all runs: `3/4`, outputs `D,D,B,D`.

Conclusion:

- On dynamic-k `smoke4`, the current method helps the decode/eval phase, but
  the end-to-end smoke wall time is essentially flat.
- Most of the measured decode gain comes from enabling parallel staging; the
  io_uring replacement adds a smaller positive increment once that path is
  active.
- For end-to-end speed, the next work should target prompt/load-dominated cost
  or run a longer decode benchmark where eval throughput is the dominant metric.
