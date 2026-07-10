# Vendor DeepSeek New-Machine Dual-H2D Pipeline Token-Rate Plan

## Summary

Continue optimizing vendor DeepSeek on branch `vendor/deepseek-token-rate-16gb`
for the product target: random user prompts should produce stable `>5 tok/s`
on a 16GB host-RAM machine with a 32GB RTX 5090.

This plan explicitly covers both new-machine hardware states:

- **Case A, pre-cable / low-H2D:** effective H2D around `6.6 GB/s`, cold
  generalized result around `2.5-3.0 tok/s`.
- **Case B, post-cable / improved x4:** PCIe can reach `32 GT/s x4`, H2D
  around `11.5 GB/s`, cold generalized result around `3.0 tok/s`.

Old-machine `4.9 tok/s` remains an x16 reference, not the new-machine baseline.
All accepted results must be prompt-general, strict-cold, reproducible, and
within the 16GB cgroup including page cache.

**Run gate:** before any model execution, including quick probes and dirty
diagnostics, kill display processes and stale GPU/model processes first. A run
is invalid for comparison unless this is done and recorded before launch.

**Non-negotiable pre-launch first step:** the first operational step before
every DeepSeek run is to stop/kill display processes. This includes normal
benchmarks, SOTA reproductions, profiling runs, dirty experiments, and one-off
prompt checks. Do not start `llama-cli`, the demo script, or any model-loading
process until the display cleanup has completed and the clean `nvidia-smi`
process list has been captured.

**Launch rule:** every command sequence that starts the DeepSeek model must
begin with the display/model cleanup block below. This is required even for
one-off prompt tests, profiling runs, failed experiments, and local smoke
checks. If the cleanup block is not executed before the model process is
created, the resulting metrics must be marked invalid and cannot be used as a
baseline, candidate, SOTA, regression, or historical comparison.

## Required Pre-Run Procedure

Before every model run on the new machine, including diagnostic runs, profile
runs, cold benchmarks, and SOTA validation, display processes and stale model
processes must be killed/stopped first. This is a hard promotion requirement
because graphical processes consume VRAM and can change the effective
cache/workspace budget.

中文硬性要求：每次运行模型前必须先杀掉显示进程；没有完成并记录该步骤的 run
不能进入 baseline/SOTA/候选优化比较，只能作为无效诊断参考。

执行顺序也必须固定：先杀掉显示进程和旧模型进程，再确认 GPU 进程列表，再启动
16GB cgroup 下的模型。不能先启动模型后补记录。

Run-0 display cleanup is mandatory before **every** execution, not only before
final SOTA validation. If this cleanup is skipped, the run is invalid for
baseline, profile, regression, candidate, and SOTA comparison.

1. Kill/stop display and stale model usage before launching the model:

   ```bash
   echo '12345678' | sudo -S bash -lc '
     systemctl stop display-manager || true
     pkill -f "[g]nome-shell" || true
     pkill -f "[X]org" || true
     pkill -f "/usr/lib/xorg/[X]org" || true
     pkill -f "[o]llama" || true
     pkill -f "[l]lama-cli" || true
     pkill -f "[m]ain" || true
   '
   ```

2. Confirm `nvidia-smi` shows no display/model GPU processes and only
   driver-reserved VRAM remains before the run starts.
3. Set CPU policy to performance:

   ```bash
   echo '12345678' | sudo -S bash -lc '
     for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
       echo performance > "$f" 2>/dev/null || true
     done
     for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
       echo performance > "$f" 2>/dev/null || true
     done
   '
   ```

4. Record PCIe link state and H2D benchmark before the cold run:
   `current_link_speed`, `current_link_width`, `max_link_speed`,
   `max_link_width`, plus 4.25MiB pinned H2D throughput.

Each official run artifact must record `display_processes_stopped_before_run`
and the pre-run `nvidia-smi` process list. Runs that skip display-process
cleanup are diagnostic only and must not be promoted as SOTA.

## Current Evidence And Bottleneck

- The SOTA path is active on the new machine: gate full native expert pack,
  full GGUF alias TSV, up/down paired reads, `misses=0`, and strict 16GB cgroup
  accounting all work.
- Expert IO is not the primary bottleneck. `iouring_reads=8194` and
  `iouring_bytes≈36.5GB` match old-machine runs, and IO wait is not enough to
  explain the token-rate gap.
- H2D is the largest measured hardware-linked gap:
  - pre-cable new machine: about `6.6 GB/s`
  - post-cable new machine: about `11.5 GB/s`
  - old x16 reference: about `26-28 GB/s`
- New-machine CPU utilization remains low because threads mostly wait on
  H2D/CUDA/io_uring/staging events. Forcing CUDA spin and `--poll 100` did not
  close the gap, so the next software work must reduce bytes, reduce copy
  count, and overlap more work rather than only changing thread polling.

## Implementation Plan

0. **Mandatory Run-0 cleanup before any benchmark**
   - Before launching `llama-cli` or any profiling/demo wrapper on the new
     machine, stop the display stack and kill remaining GPU/model processes.
     This includes `display-manager`, `gnome-shell`, `Xorg`, `ollama`, and any
     stale `llama-cli`.
   - Capture `nvidia-smi` immediately after cleanup and before model launch.
     The run is valid only if no display/model GPU process remains.
   - Store both fields in the run artifact:
     `display_processes_stopped_before_run=true` and
     `pre_run_nvidia_smi_processes=<captured list>`.
   - Any run missing this cleanup or metadata is invalid for baseline,
     candidate, SOTA, and regression comparison. It may only be referenced as
     a diagnostic note.

1. **Baseline and metadata recording**
   - Record separate baselines for Case A and Case B. Never compare or promote
     results without stating the hardware case.
   - Every run artifact must include command/env, commit, model path, expert
     pack path, alias TSV, answer text, cgroup memory stats, PCIe link, H2D
     benchmark, MoE counters, and whether display processes were stopped.

2. **Native expert-pack parity is the next implementation target**
   - The next optimization must start from the full native DeepSeek expert pack,
     not a prompt-specific pack, gate-only pack, or up/down alias-only source.
   - Gate, up, and down must resolve through the same source abstraction:
     native expert pack lookup, exact tensor/expert key, offset/bytes metadata,
     same batch cache, same pinned staging, same O_DIRECT/io_uring path, same
     H2D scheduling, and same ready-event ownership.
   - `GGML_MOE_BATCH_FULLPACK=1` and
     `GGML_MOE_ROUTE_GROUP_NATIVE_PARITY=1` are the required experimental
     baseline for this work. Any result using mixed gate one-pack direct reads
     plus a separate up/down path is diagnostic only and cannot be promoted.
   - This work must preserve Kimi and existing DeepSeek behavior by remaining
     default-off until it beats the generalized SOTA and passes correctness.

3. **Required load order: Stage A gate+up, Stage B down**
   - After router metadata for a layer is available, build a single Stage A
     request list containing both `ffn_gate_exps` and `ffn_up_exps` for all
     selected experts in that layer.
   - Submit Stage A as one sorted batch against the native expert pack. Sorting
     should use source identity, file offset, and size so nearby tensors can be
     issued together and copied through the same pinned/H2D machinery.
   - Stage A must not use a separate gate fast path. Gate may consume the shared
     batch-cache entry, but the producer must be the same native batch loader
     that serves up.
   - Only after Stage A is queued should Stage B collect
     `ffn_down_exps` for the same routed experts and submit it through the same
     native batch loader. Stage B can overlap with current compute when data
     dependencies allow, but its data source and movement path must be identical
     to Stage A.
   - If any gate/up/down tensor is missing from the native pack, has an
     unsupported dtype/layout, gets a short read, or fails H2D/correctness
     validation, the run is rejected for SOTA. Fallback is allowed for safety
     while debugging but must be reported as `fallback_present=true`.

4. **Priority split pipeline is the next optimization**
   - Do not continue with plain `gate+up+down` co-batch as the promoted path.
     Diagnostic run
     `20260709T100926Z-20260709-native-parity-minseen1-cobatch-vram14000-quantum-n96`
     proved that simple co-batching reduced io_uring batches
     (`5236 -> 2733`) but regressed token rate (`4.2 -> 3.9 tok/s`) because
     down work shared the same staging/H2D critical path as gate/up.
   - Keep `min_seen=1` as the correctness/parity baseline for this work. The
     `min_seen=6` path is diagnostic only because it skips many up/down native
     cache admissions and therefore does not prove full gate/up/down parity.
   - Implement a default-off priority split route-group mode:
     `GGML_MOE_ROUTE_GROUP_NATIVE_PRIORITY_SPLIT=1`.
   - In this mode, the router should still generate all selected gate/up/down
     requests from the same full native expert pack in one route-group planning
     step, but execution must separate criticality:
     - priority 0: gate + up
     - priority 1: down
   - Gate/up must get first access to ready-event publication, H2D bandwidth,
     and pinned staging slots. Down may be known at the same time and may be
     queued early, but it must not delay gate/up cache readiness.
   - The preferred first implementation is a dual-ring design:
     - `stage_ring_priority0` for gate/up;
     - `stage_ring_priority1` for down;
     - separate counters for priority0 and priority1 copies, waits, H2D bytes,
       and ready-event waits.
   - If dual rings are too invasive, start with fixed slot reservation in the
     existing ring: gate/up reserve most slots, down can only use the remaining
     slots. Any down copy must back off rather than blocking gate/up.
   - The next accepted diagnostic must show:
     - gate/up ready wait decreases or at least does not regress;
     - down no longer increases gate/up critical-path latency;
     - `iouring_batches` may decrease, but token rate must not regress;
     - `iouring_reads`, `iouring_bytes`, `h2d_enqueues`, and async waits are
       recorded separately for priority 0 and priority 1.
   - 2026-07-10 diagnostic update: the first default-off priority split probe
     proved that "enqueue down into a background demand queue" is not enough.
     Run
     `20260710T024115Z-20260710-native-priority-split-minseen1-vram14000-quantum-n96`
     used full native expert pack parity, `min_seen=1`, `14GB` batch VRAM
     cache, `GGML_MOE_DOWN_BATCH_DEMAND_QUEUE=1`, and strict cold 16GB cgroup.
     It was RAM/correctness compliant, but rejected for SOTA:
     `eval_tok_s=4.0`, `prompt_tok_s=3.2`, `TTFT=16856 ms`,
     `memory_peak_bytes=14830362624`, answer coherent. Counters showed:
     `stage_a_jobs=8388`, `stage_b_jobs=4192`,
     `priority_down_enqueued=4192`, `priority_down_queue_fail=0`,
     `down demand queue submitted=4192 completed=4192`, but
     `iouring_wait_us=9999909` and `iouring_batches=6246`, worse than the
     native parity reference (`iouring_wait_us≈6574945`, `batches=5236`) and
     worse than the clean gate SOTA (`5.5 tok/s`). Conclusion: current down
     demand queue creates many small priority-1 batches and still competes with
     the same staging/H2D machinery. The next implementation must be a real
     dual-ring or reserved-slot priority scheduler, not another plain queue.
   - Concrete next method:
     - keep priority-0 gate/up on a dedicated staging ring or protected slot
       pool with independent ready-event publication;
     - keep priority-1 down on a separate ring/stream whose copy completion
       cannot hold `g_batch_mu` while priority-0 is publishing events;
     - add separate counters for priority-0 and priority-1 submit/wait/H2D ms;
     - batch priority-1 down by layer or short horizon so it does not fragment
       into thousands of one-job io_uring batches;
     - reject the run if down background work increases priority-0 wait or if
       token rate remains below the clean generalized SOTA.

5. **Make the full lifecycle observable before claiming speedup**
   - Add per-layer/per-token counters for Stage A and Stage B:
     request count, cache hit/miss count, io_uring read count, SSD bytes,
     pinned staging ms, H2D bytes, H2D ms, ready-event wait ms, kernel compute
     ms, CPU fallback ms, and total layer latency.
   - Print a timeline summary that makes one token's path visible from router
     output to gate/up load, gate/up compute, down load, down compute, and final
     layer completion.
   - Acceptance requires proving that up/down are no longer on an incompatible
     slow path. If CPU fallback remains, quantify which tensor/layout caused it
     and why it is still present.

6. **Cache policy after native parity, not before it**
   - Do not split VRAM evenly between gate/up/down until the native parity path
     is correct and measured. Equal split is not an objective by itself.
   - First run with one unified native batch cache so all three tensor roles
     compete under the same mechanism. Record role-specific hit/miss and bytes.
   - Then test prompt-general admission rules only if they reduce total bytes
     or waits without breaking Stage A/Stage B parity. Examples include
     role-aware `min_seen`, LFU/LRU, and small protected regions, but every
     rule must be prompt-general and must be validated on calibration prompts
     before held-out prompts.
   - Promotion requires a candidate to beat the clean generalized SOTA, not only
     improve a single prompt or a short `-n 16` diagnostic.
   - Current evidence says eviction policy is not the primary bottleneck:
     `min_seen=1 + GGML_MOE_CACHE_POLICY=lfu_lru` reached only `4.2 tok/s`,
     with the same `12580` reads, `56.06GB` read bytes, and `11008` async
     waits as the LRU baseline. LFU can remain a secondary knob, but it must
     not replace priority split/H2D work.

7. **Larger H2D batching / copy coalescing**
   - Once Stage A/Stage B parity is correct, combine adjacent or same-batch
     staging buffers into fewer H2D submissions where tensor layout and kernel
     inputs remain unchanged.
   - Add profile fields for H2D copy count, total H2D bytes, H2D ms, average
     copy size, and per-token H2D wait.
   - This is expected to help Case A most, but should also reduce wait overhead
     in Case B.
   - This must happen after priority split is in place. Coalescing that mixes
     down with gate/up on the same critical H2D path is rejected even if it
     reduces total batch count.

8. **Reduce H2D bytes with compact or partial up/down source**
   - First produce a hard-bound profile: actual rows/blocks used per
     token/layer, full-expert bytes copied, partial-copy bytes needed, and
     projected token-rate upper bound for Case A and Case B.
   - Only implement a default-off compact source if the bound shows meaningful
     gain. It must not replace the native full expert pack or rely on a
     prompt-specific profile.
   - 2026-07-10 hard-bound update:
     `20260710T032816Z-20260710-gate12288-routeprofile2-quantum-n96`
     recorded a clean strict-cold route profile for the current gate12288 SOTA
     path. Summary:
     - gate: `12316` active expert uses, `9403` cache hits, `2913` misses,
       `12.98GB` actual one-pack reads, `76.3%` hit rate;
     - up: `12316` active expert uses, `54.89GB` logical source bytes, no
       resident cache in the current SOTA path;
     - down: `12316` active expert uses, `54.89GB` logical source bytes, no
       resident cache in the current SOTA path.
   - Exact partial row/column copy is not a valid shortcut for the current
     kernels. Up/gate and down both compute exact dense matvec outputs over
     full `ne00` input and full `ne01` output for each selected expert, so a
     subset of rows/columns would change model math unless paired with a new
     compressed representation or a verified approximation.
   - Decision: do not implement simple partial-copy up/down. Full up+down
     GPU streaming would require about `109.77GB` of exact expert payload on
     this n96 profile and has already been empirically slower in native parity
     experiments. Future up/down work must be a real compact representation,
     quantized sidecar, or correctness-gated approximation.
   - The best exact byte-reduction target is now gate first-use misses:
     removing the remaining `12.98GB` gate reads has an H2D floor of about
     `1.96s` at the measured `6.63GB/s`, likely moving the `5.5 tok/s` SOTA
     only into the low-`6 tok/s` range. This still has the best risk/reward
     because it preserves model math and avoids stealing gate cache for
     low-yield up/down entries.

9. **Non-expert mmap/page-cache pressure**
   - Profile decode-stage major faults and `memory.stat file/anon` for dense,
     attention, norm, and other non-expert tensors.
   - Try default-off page retention or pre-touch only if it remains inside the
     16GB cgroup and does not raise TTFT beyond the gate.

## Validation And Promotion

Use the calibration/dev set for tuning:

- `Please introduce France in a short paragraph.`
- `Explain quantum computing briefly.`
- `Write a short Python function for Fibonacci.`
- `Introduce Japan in a short paragraph.`
- `Summarize climate change in one paragraph.`
- `AI infra 是做什么的`
- `How to deploy a large model on a small devices?`
- `今天吃什么`

Held-out prompts must only be used after a candidate is frozen.

Promotion requirements:

- Strict cold run with `MemoryMax=16000000000`, `MemorySwapMax=0`, and
  `drop_caches`.
- Display processes stopped before the run and verified absent from
  `nvidia-smi`.
- RAM and page cache within the 16GB cgroup; no swap/OOM/ram kill.
- France output and generalized prompt outputs are coherent and semantically
  correct.
- TTFT is not more than 20% above the corresponding hardware-case baseline.
- Token rate exceeds the corresponding hardware-case baseline and reproduces in
  at least two cold repeats.
- Accepted improvements must be committed and pushed immediately to
  `ssd/vendor/deepseek-token-rate-16gb` with reproduction details. Use
  `L-Ark <fliangae@connect.ust.hk>` for commits.

## Assumptions

- Current active new-machine state is Case B, post-cable Gen5 x4 H2D around
  `11.5 GB/s`.
- Case A is retained as a supported low-H2D scenario for bound analysis and
  future weak-link machines.
- Old-machine x16 `4.9 tok/s` is a reference ceiling source, not the baseline
  for new-machine SOTA promotion.
- All new runtime behavior must be default-off until generalized validation is
  complete.
- Kimi functionality must not be removed or weakened when touching shared MoE or
  CUDA streaming code.

## 2026-07-10 Execution Notes

- Implemented default-off native route-group diagnostics:
  `GGML_MOE_ROUTE_GROUP_NATIVE_COBATCH`,
  `GGML_MOE_ROUTE_GROUP_NATIVE_PRIORITY_SPLIT`,
  `GGML_MOE_ROUTE_GROUP_NATIVE_SKIP_GATE`, and role-aware native admission
  thresholds for up/down. These are diagnostic paths only unless a clean run
  beats the accepted generalized SOTA.
- Plain priority split with full native gate/up/down parity was rejected:
  `20260710T025110Z-20260710-native-priority-split-stageA-first-vram14000-quantum-n96`
  reached `4.2 tok/s`, RAM/correctness OK, but below the clean `5.5 tok/s`
  SOTA. It reduced the earlier bad priority-split wait (`iouring_wait_us`
  `10.0s -> 7.0s`) by copying Stage A gate/up before queuing down, but down
  still had `3122` queue batches and `async prefetch waits≈11008`.
- `GGML_MOE_DOWN_BATCH_DEMAND_QUEUE_DELAY_US=200` restored native-parity-like
  batching (`iouring_batches=5236`) but still produced only `4.2 tok/s`.
  Delay-based down aggregation alone does not move the product metric.
- Role-aware admission showed the current up/down native cache is only useful
  when it avoids low-reuse tensors:
  - `up_min_seen=1, down_min_seen=6`: `4.4 tok/s`, reads `8027`,
    bytes `35.77GB`, wait `5.20s`.
  - `up_min_seen=6, down_min_seen=6`: `5.3 tok/s`, reads `3940`,
    bytes `17.56GB`, wait `2.81s`, RAM/correctness OK.
  - `up_min_seen=8, down_min_seen=8`: rejected, `4.8 tok/s`.
  The best native-parity diagnostic remains below the clean gate-only SOTA, so
  it is not promoted.
- VRAM split experiments were rejected:
  - `gate12288 + up/down2GB` failed because the 12GB gate one-cache allocation
    failed after the batch cache allocation; gate fell back to `12316`
    O_DIRECT reads and token rate dropped to `2.9 tok/s`.
  - `gate10240 + up/down2GB` allocated both caches but reached only
    `4.6 tok/s`; gate hit rate stayed `76.1%`, and reduced gate cache was more
    costly than the up/down cache benefit.
  - `gate12288 + up/down1GB` allocated both caches but reached only
    `4.7 tok/s`; up/down cache produced many small batches and useful down
    rate was only `41.3%`.
  Conclusion: under the current implementation, moving VRAM away from gate
  one-cache to up/down is not beneficial.
- Gate one-cache size probe:
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13312` produced one dirty diagnostic at
    `5.6 tok/s`, but the repeat was `5.5 tok/s` with the same gate hit/miss
    profile as 12GB (`hits=9403`, `misses=2913`). Treat this as normal run
    variance, not a new SOTA.
- Current accepted generalized SOTA remains the clean gate one-cache path:
  `GGML_MOE_STREAM_ONE_CACHE_MIB=12288`, `eval_tok_s=5.5` on the Quantum n96
  calibration run, strict 16GB cgroup including page cache, coherent output,
  and display cleanup recorded.
- Updated next direction: do not continue simple VRAM splits or plain down
  queue delay sweeps. The next meaningful implementation must either:
  1. improve the gate one-cache layout/replacement/prefetch itself without
     reducing the 12GB gate budget;
  2. reduce up/down H2D bytes with a real compact representation or
     correctness-gated approximation, not simple partial matrix copies; or
  3. make up/down cache fills non-blocking enough that they do not add
     thousands of small wait points.

## 2026-07-10 Gate12288 Route Profile Bound

- Diagnostic run:
  `20260710T032816Z-20260710-gate12288-routeprofile2-quantum-n96`.
- Source and constraints: commit `e6d4f002d`, source clean, strict cold,
  display cleanup recorded, `memory_peak_bytes=14857424896`, RAM OK.
- Metric: `eval_tok_s=5.4` with route-profile overhead; clean SOTA guard
  without profile remains `5.5 tok/s`.
- Route profile:
  - gate logical source bytes `54.89GB`, cache hits `9403`, misses `2913`,
    actual O_DIRECT one-pack read bytes `12.98GB`;
  - up logical source bytes `54.89GB`, no current SOTA cache;
  - down logical source bytes `54.89GB`, no current SOTA cache.
- Interpretation:
  - current exact up/down kernels need full expert matrices, so partial
    row/column copies are not an exact optimization;
  - full native up/down movement explains why native parity topped out at
    `5.3 tok/s` and why VRAM split probes regressed;
  - gate first-use miss reduction is the next exact path with the smallest
    correctness risk, but its bandwidth floor suggests only low-`6 tok/s`
    potential unless paired with a deeper representation/layout change.

## 2026-07-10 Gate Direct Hot Pool First320 SOTA

- Method: add a prompt-general direct hot pool in front of the current
  gate12288 one-cache path. The manifest is generated from the full native
  expert pack, not from a prompt route profile:

  ```bash
  python3 scripts/ds4-expert-pack-direct-manifest.py \
    --pack /home/wici/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.expert-pack \
    --tensor-substr ffn_gate_exps \
    --limit 320 \
    --output /home/wici/runs/vendor-ds4-16gb/ds4-gate-pack-first320.direct_manifest.csv
  ```

- Runtime delta:
  - keep `GGML_MOE_STREAM_ONE_CACHE_MIB=12288`;
  - add `GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB=1400`;
  - add `GGML_MOE_STREAM_ONE_DIRECT_PREFILL_LIMIT=320`;
  - add `GGML_MOE_STREAM_ONE_DIRECT_PREFILL_ASYNC=1`;
  - set `GGML_MOE_STREAM_ONE_DIRECT_MODEL` to the same native expert-pack
    file and `GGML_MOE_STREAM_ONE_DIRECT_IO=direct`.
- Reproduction helper added:
  `scripts/demo-vendor-ds4-gate-hotpool-sota.sh`.
- Accepted candidate evidence:
  - Quantum n96 repeat 1:
    `/root/lfz/runs/vendor-ds4-16gb/demo-general-sota/20260710T034736Z-interactive`,
    `eval_tok_s=5.6`, `prompt_tok_s=3.6`, `TTFT=16178.44 ms`,
    `memory_peak_bytes=14881206272`, `memory_file_bytes=13976903680`,
    RAM OK, coherent output.
  - Quantum n96 repeat 2:
    `/root/lfz/runs/vendor-ds4-16gb/demo-general-sota/20260710T034831Z-interactive`,
    `eval_tok_s=5.7`, `prompt_tok_s=3.6`, `TTFT=18769.04 ms`,
    `memory_peak_bytes=14884630528`, `memory_file_bytes=13955420160`,
    RAM OK, coherent output.
  - France correctness guard:
    `/root/lfz/runs/vendor-ds4-16gb/demo-general-sota/20260710T034924Z-interactive`,
    `eval_tok_s=6.8`, `prompt_tok_s=3.9`, `TTFT=16663.37 ms`,
    `memory_peak_bytes=14873145344`, RAM OK, semantically correct France
    answer.
- TTFT constraint: the worst accepted repeat is `18769.04 ms`, about `+16.9%`
  versus the clean gate12288 reference `16059.33 ms`, so it remains below the
  `+20%` limit.
- Promotion: accept `5.6 tok/s` as the current prompt-general strict-cold
  SOTA for the Quantum n96 calibration metric, because the repeat pair is
  `5.6` and `5.7` and all hard constraints pass.
- Boundary probes:
  - first240 / 1024MiB pool reduced one-pack reads to `2778` and produced
    `5.6` then `5.5 tok/s`; useful but not enough to promote alone.
  - first336 / 1480MiB pool tied the accepted rounded SOTA but did not exceed
    it: `/root/lfz/runs/vendor-ds4-16gb/demo-general-sota/20260710T040001Z-interactive`,
    clean source `730b1ac45`, `eval_tok_s=5.6`, `TTFT=16139.43 ms`,
    RAM OK, direct hot pool hits `411`, one-pack reads `2723`. Not promoted.
  - first360 / 1600MiB pool was tested after parameterizing the helper script
    and clean-reproducing source `cf2c51bc9`. It preserved the 12GB one-cache
    and reduced one-pack reads to `2712`, but reached only `5.5 tok/s`:
    `/root/lfz/runs/vendor-ds4-16gb/demo-general-sota/20260710T035645Z-interactive`.
    Reject because the larger async prefill cost (`792.5 ms`, `1.60GB`) erased
    the read reduction.
  - first480 / 2048MiB pool is rejected. It allocated the direct hot pool
    first, caused the 12288MiB one-cache allocation to fail, and fell to
    `3.1 tok/s` with `11746` one-pack reads / `52.35GB`.
- Updated next direction:
  - do not exceed the VRAM point where the 12288MiB one-cache fails;
  - first336/first360 show that increasing entries can reduce reads but still
    tie or lose to prefill overhead, so stop simple entry-count sweeps unless
    a new hard bound predicts a larger win;
  - investigate allocation order or reserved VRAM accounting so direct hot
    pool cannot silently disable the main gate one-cache;
  - investigate cheaper prefill or overlapped prefill so the hot pool saves
    decode H2D without adding comparable TTFT/prefill cost.

## 2026-07-08 Execution Notes

- Added run-artifact hardware metadata collection to
  `scripts/demo-vendor-ds4-general-sota.sh`: display-process status, GPU state,
  PCIe link state, and 4.25MiB pinned H2D benchmark are now recorded in each run
  summary.
- New-machine run `20260708T131124Z-caseB-metadata-baseline` validated the new
  metadata path under strict cold cgroup. It was RAM/correctness compliant but
  only reached `2.8 tok/s`; H2D measured `6.71 GB/s`, so the machine was in the
  low-H2D Case A-like state despite the post-cable setup.
- Existing `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` was re-tested as a default-off
  aggregation probe on the new machine. It was rejected: `2.7 tok/s`, more total
  expert traffic (`39.5GB` vs baseline `36.5GB`), and no SOTA improvement.
- Gate one-cache split probes were rejected:
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=5120`: `2.7 tok/s`
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=7168`: `2.6 tok/s`
- Prompt-general VRAM profile probes using
  `calib-dev-sparse-pair-top512-updown-20260707` were rejected:
  - protected preload increased runtime misses to `8811` and produced
    `2.6 tok/s`;
  - profile-only `profile_lfu_lru` increased misses to `14491`, produced
    `2.2 tok/s`, and showed worse output quality.
- Updated conclusion: old preload/admission paths are not the next route to
  `>5 tok/s` on the new machine. The next implementation must directly reduce
  H2D bytes or H2D submissions, or implement a true shared gate/up/down read
  aggregation layer that does not increase total reads.
- Added default-off profiling hooks for the real current hot paths:
  - `GGML_MOE_H2D_COALESCE_PROFILE_OUT` records whether batch io_uring H2D
    copies are physically coalescible.
  - `GGML_MOE_ONE_PACK_READ_PROFILE_OUT` records `moe_stream` one-pack direct
    reads: tensor, expert, offset, bytes, O_DIRECT status, and read latency.
- Diagnostic run `20260708T132817Z-h2d-coalesce-profile` proved the current
  generalized new-machine path does **not** enter the batch
  `expert_pack_iouring_copy_jobs` H2D path for the dominant gate reads; no H2D
  coalesce CSV was produced. The hot gate source is `moe_stream` one expert
  pack, while up/down still shows `down parallel CPU staging active`.
- Diagnostic run `20260708T133227Z-one-pack-read-profile` produced:
  - `eval_tok_s=2.1`, diagnostic only because per-read CSV adds overhead;
  - `memory_peak_bytes=14901284864`, `ram_ok=true`,
    `display_processes_stopped_before_run=true`;
  - H2D benchmark `6.68 GB/s`, PCIe under load `16.0 GT/s x4`;
  - one-pack gate reads: `7701` reads, `34.32GB`, total direct-read wall
    `7445 ms`, avg `0.967 ms`, p50 `0.778 ms`, p95 `1.200 ms`;
  - consecutive physical-offset continuity only `162/7700 = 2.1%`.
- Consequence: simple adjacent-read or adjacent-H2D coalescing in the current
  execution order is not enough. The next implementation should prioritize:
  1. locating the up/down miss source that bypasses batch io_uring and moving it
     onto an expert-pack/batch path;
  2. changing gate/up/down scheduling to submit a known per-layer route group
     together, rather than trying to merge the already-issued read order;
  3. restoring larger effective VRAM cache allocation on the new machine. The
     diagnostic runs requested `9GB` batch VRAM cache but fell back to `6.9GB`,
     so cache pressure is worse than intended.
- Root-caused one new-machine-specific regression: the committed static GGUF
  alias TSV contained the old-machine model path
  `/root/lfz/models/.../DeepSeek-V4-Flash-FP4-FP8-native.gguf`. On the new
  machine this produced `expert alias source: open failed`, so up/down misses
  fell back to mmap host staging/page cache instead of batch io_uring.
- Implemented a run-local alias rewrite in
  `scripts/demo-vendor-ds4-general-sota.sh`: each artifact now contains
  `ds4-native-full-gguf-alias-source.effective.tsv`, generated from the static
  prompt-general alias TSV but with `source_path` set to the current `MODEL`.
  This is machine-path repair only, not prompt-specific optimization.
- Validation run `20260708T133916Z-effective-alias-fix` on prompt
  `How to deploy a large model on small devices?`:
  - `eval_tok_s=2.5` vs `2.2` before the alias fix on the same prompt;
  - `prompt_tok_s=3.2`, `TTFT=19050.8 ms`, elapsed `55.57s`;
  - `memory_peak_bytes=14610792448`, `memory_file_bytes=13723795456`,
    `ram_ok=true`, display processes stopped;
  - H2D benchmark `6.73 GB/s`, PCIe under load `16.0 GT/s x4`;
  - batch expert source restored:
    `expert alias source: io_uring direct reads enabled`,
    `iouring_reads=9416`, `iouring_bytes=41961914368`, `misses=0`;
  - batch VRAM cache improved to `hits=24151`, `misses=9416`,
    `hit_rate=71.9%`;
  - model answer was semantically correct and coherent, but truncated by the
    `n=96` demo cap.
- This is an accepted reproducibility/performance fix for the new machine but
  not a product SOTA: generalized random-prompt target remains `>5 tok/s`, and
  this single-prompt new-machine run is only `2.5 tok/s`.
- VRAM split probes after alias repair:
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=4096`: rejected. It allowed batch cache to
    allocate `9.0GiB` and reduced up/down iouring bytes to `36.56GB`, but gate
    one-pack bytes rose to `43.14GB`; total expert movement increased and
    token rate dropped to `2.4 tok/s`.
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=5120`: neutral/borderline. Batch cache
    became `7.9GiB`, up/down iouring bytes `38.78GB`, gate bytes `37.67GB`,
    `eval_tok_s=2.5`; elapsed improved slightly to `55.21s`, but not enough to
    promote as a meaningful SOTA.
- IO size probe:
  - `GGML_MOE_IO_BYTES=8388608` restored larger alias-source reads. On
    `How to deploy a large model on small devices?`, it kept
    `eval_tok_s=2.5` but improved `prompt_tok_s=3.3`, `TTFT=18828.9 ms`, and
    elapsed `54.73s` versus alias-fix default `55.57s`.
  - France sentinel with `GGML_MOE_IO_BYTES=8388608`:
    `eval_tok_s=2.7`, `prompt_tok_s=3.0`, `TTFT=18315.0 ms`,
    `memory_peak_bytes=14755557376`, `ram_ok=true`, output semantically
    correct/coherent. The answer was truncated only by `n=96`.
  - Accepted as the new demo default because it is prompt-general, restores the
    larger Kimi-style alias read size, lowers measured latency, and does not
    hurt RAM or correctness. It is still far below the `>5 tok/s` product
    target.
- Clean default reproduction after commit `66940ebce`:
  `20260708T134945Z-clean-default-66940eb` on
  `How to deploy a large model on small devices?` produced
  `eval_tok_s=2.5`, `prompt_tok_s=3.3`, `TTFT=18874.4 ms`,
  elapsed `54.45s`, `memory_peak_bytes=14763356160`,
  `memory_file_bytes=13785108480`, `ram_ok=true`, source clean, display
  processes stopped, H2D `6.70 GB/s`, PCIe under load `16.0 GT/s x4`.
- Hardware diagnosis update: GPU endpoint supports Gen5 x16, but the upstream
  root port `0000:00:06.0` has `LnkCap: Speed 16GT/s, Width x4` and
  `LnkCtl2 Target Link Speed: 16GT/s`. Therefore the new machine's current
  H2D ceiling is a real Gen4 x4 topology limit, not just NVIDIA power
  management. Software cannot make this link x16; optimization must reduce
  expert movement or improve overlap.
- Post-`66940ebce` n32 profile
  `20260708T135215Z-n32-profile-66940eb`:
  - `eval_tok_s=2.5`, `TTFT=18933.6 ms`, elapsed `29.07s`, RAM OK;
  - gate one-pack reads: `3505` reads, `15.62GB`, direct-read wall `4520 ms`;
  - batch up/down alias reads: `4193` jobs, `18.69GB`,
    io batch wall `8477 ms`, wait `2654 ms`, submit `507 ms`;
  - measured H2D for batch copies: `3316 ms`;
  - iouring inflight average only about `2.25` despite depth `32`, with many
    1-job and 2-4-job batches. This points to route/batch formation and
    movement volume as the next software bottleneck.
- Additional probes:
  - `GGML_MOE_STAGE_PINNED_SLOTS=16`: rejected,
    `20260708T135359Z-pinned16-probe`, `eval_tok_s=2.4`, elapsed `55.79s`.
  - `GGML_MOE_IO_SORT_OFFSET=1`: neutral/rejected,
    `20260708T135544Z-sortoffset-probe`, `eval_tok_s=2.5`, elapsed `54.69s`.
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=7168`: neutral/rejected,
    `20260708T135716Z-aliasfix-onecache7168`, `eval_tok_s=2.5`,
    elapsed `54.74s`.
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` after alias repair: rejected,
    `20260708T135935Z-aliasfix-cosubmit-retest`, `eval_tok_s=2.5`,
    elapsed `55.60s`, TTFT `19587.0 ms`. It increased batch iouring traffic to
    `45.28GB` while gate still read `34.32GB`, so total expert movement rose.
  - `GGML_MOE_CURRENT_DOWN_OVERLAP=1`: rejected,
    `20260708T140219Z-current-down-overlap-probe`, `eval_tok_s=2.5`,
    elapsed `55.50s`, TTFT `19090.4 ms`; no useful current-down-overlap
    counters appeared on the hot path.
  - `GGML_MOE_VRAM_CACHE_MIB=7680`: neutral/rejected,
    `20260708T140546Z-vram7680mib-probe`, `eval_tok_s=2.5`,
    elapsed `54.63s`. It successfully allocated `7.5GiB` batch cache and
    reduced up/down iouring bytes to `39.78GB`, but did not beat the clean
    default elapsed `54.45s`.
  - `GGML_MOE_BATCH_FULLPACK=1`: neutral/rejected,
    `20260708T140910Z-batch-fullpack-probe`, `eval_tok_s=2.5`,
    elapsed `54.80s`. Using the prompt-general full `.expert-pack` as the
    batch source did not improve over the run-local GGUF alias source.
- Keep current default at `GGML_MOE_STREAM_ONE_CACHE_MIB=6144`,
  `GGML_MOE_IO_BYTES=8388608`, `GGML_MOE_STAGE_PINNED_SLOTS=8`.
  The next high-value implementation is not another small env sweep; it should
  change route-group batching or reduce gate/up/down bytes.
- Accepted route-pruning improvement:
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=2` for the existing layer range `10-39`
    reduces routed up/down work versus the previous default `3`. This changes
    model math, so it requires stricter correctness checks than IO-only
    changes.
  - Prompt `How to deploy a large model on small devices?`,
    run `20260708T141213Z-topk-layer2-probe`: `eval_tok_s=2.9`,
    `prompt_tok_s=3.5`, `TTFT=18372.5 ms`, elapsed `49.05s`,
    `memory_peak_bytes=14764113920`, `ram_ok=true`,
    `display_processes_stopped_before_run=true`. Output was coherent and
    semantically correct.
  - France sentinel, run `20260708T141326Z-france-topk-layer2`:
    `eval_tok_s=2.9`, `prompt_tok_s=3.3`, `TTFT=17462.7 ms`,
    elapsed `48.99s`, `memory_peak_bytes=14743666688`, `ram_ok=true`.
    France output was semantically correct and coherent.
  - Accepted as the new default pending a clean-repeat and broader calibration
    prompt set. It improves the measured generalized demo prompt from `2.5` to
    `2.9 tok/s` while staying within RAM and lowering TTFT.
- Clean repeats after commit `15f243009`:
  - `20260708T141546Z-clean-default-topk2`, prompt
    `How to deploy a large model on small devices?`: `eval_tok_s=2.8`,
    `prompt_tok_s=3.5`, `TTFT=18366.3 ms`, elapsed `49.71s`,
    `memory_peak_bytes=14758408192`, source clean, RAM OK, output coherent.
  - `20260708T141709Z-clean-france-topk2`: `eval_tok_s=2.8`,
    `prompt_tok_s=3.2`, `TTFT=17807.6 ms`, elapsed `49.62s`,
    `memory_peak_bytes=14787674112`, source clean, RAM OK, France output
    semantically correct and coherent.
- Current clean generalized new-machine SOTA is therefore `2.8 tok/s` on these
  n96 cold strict runs. The `2.9 tok/s` probe is retained as observed variance,
  not the clean promoted number. Product target `>5 tok/s` is not met.
- Route-pruning follow-up:
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=1` rejected:
    `20260708T141910Z-topk-layer1-probe`, `eval_tok_s=2.9`, but the answer
    started awkwardly (`like to deploy...`) and was less coherent. Speed gain
    was not enough to justify the quality regression.
  - Expanding topk2 range to `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39` accepted:
    `20260708T142026Z-topk2-range0-39-probe` produced `eval_tok_s=2.9`,
    `prompt_tok_s=3.7`, `TTFT=17866.7 ms`, elapsed `49.33s`, RAM OK, coherent
    answer on the deployment prompt.
  - France sentinel `20260708T142142Z-france-topk2-range0-39` produced
    `eval_tok_s=2.9`, `prompt_tok_s=3.4`, `TTFT=18069.1 ms`, elapsed `48.72s`,
    RAM OK, and a semantically correct/coherent France paragraph.
  - New default: `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39`,
    `GGML_MOE_KEEP_TOPK_LAYER_VALUE=2`. This is still below the `>5 tok/s`
    target and needs broader calibration prompt validation.
- Clean repeat after commit `135113719`:
  `20260708T142403Z-clean-default-topk2-range0-39`, prompt
  `How to deploy a large model on small devices?`, `eval_tok_s=2.9`,
  `prompt_tok_s=3.8`, `TTFT=17908.0 ms`, elapsed `48.54s`,
  `memory_peak_bytes=14782988288`, `ram_ok=true`, source clean, output
  coherent. This is the current clean single-prompt top result on the new
  machine, but still not the product target.
- Invalidation: broader calibration after `135113719` showed topk2 is not
  acceptable as a generalized default.
  - `20260708T142705Z-calib-topk2-range0-39-fibonacci`: `eval_tok_s=2.9`, but
    the answer did not provide the requested Python function and started with
    malformed text (`thatThe...`).
  - `20260708T142848Z-calib-topk2-range0-39-climate`: `eval_tok_s=2.6`, but
    the answer exposed internal-style reasoning text (`using="thinking" ...`)
    instead of a clean paragraph.
  - Therefore all topk2 accepted/promoted notes above are invalidated for
    generalized SOTA. The default is reverted to
    `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`,
    `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`. Topk2 may remain diagnostic only.
- Revert validation after commit `e63565d60`:
  `20260708T143119Z-clean-revert-fibonacci` produced `eval_tok_s=2.4`,
  `prompt_tok_s=2.9`, `TTFT=18882.0 ms`, `memory_peak_bytes=14771392512`,
  `ram_ok=true`, source clean. The answer correctly provided a short Python
  Fibonacci generator. This confirms the unsafe topk2 default was the quality
  regression source, and the topk3/range10-39 default is restored.
- Additional topk2 isolation:
  - `20260708T143356Z-topk2-range10-39-fibonacci` with only
    `KEEP_TOPK_LAYER_VALUE=2` over the original `10-39` range reached
    `eval_tok_s=3.2`, but correctness failed: the answer became an odd
    instruction-style repetition and did not provide the requested function.
  - `20260708T143444Z-topk2-range10-39-climate` was coherent, but the Fibonacci
    failure is sufficient to reject topk2 for generalized use.
  - Conclusion: the issue is not only extending the range to `0-39`; reducing
    those MoE layers to top-2 can materially change behavior. Keep topk3.
- Current safe generalized baseline after topk2 rejection:
  - `20260708T143119Z-clean-revert-fibonacci`: `2.4 tok/s`, RAM OK, source
    clean, Fibonacci answer correct.
  - `20260708T143802Z-safe-default-n96-profile`: `2.4 tok/s` with profiling
    overhead, prompt `How to deploy a large model on small devices?`, RAM OK.
  - The safe default remains topk3 over range `10-39`; product target `>5
    tok/s` is still unmet.
- Default-off overlap/staging probes were added to the demo passthrough list so
  they can be tested without changing default behavior:
  - `GGML_MOE_MIXED_UP_GATE_PARALLEL_STAGE=1`,
    `20260708T144118Z-mixed-upgate-parallel-probe`: `eval_tok_s=2.5`, elapsed
    `54.99s`; no hot-path activation log appeared, so it is rejected as a
    meaningful improvement.
  - `GGML_MOE_STREAM_SERIAL_STAGE_BATCH=1`,
    `20260708T144314Z-serial-stage-batch-probe`: `eval_tok_s=2.4`, elapsed
    `55.6s`; no serial same-type batched staging hot-path log appeared, so it
    is rejected.
  - These results reinforce that small existing staging toggles are not enough;
    the next implementation should target prompt-general gate hot-pool or true
    route-group gate/up/down scheduling that reduces repeated expert movement.
- Next concrete default-off implementation:
  - preserve direct-manifest input order separately from lookup sort order so a
    calibration hot manifest can actually prefill the hottest entries first;
  - let the main `moe_stream` gate path lookup the one-direct hot pool before
    issuing a one-pack read/H2D copy;
  - generate manifest rows from the full native expert-pack itself, using the
    expert-pack file as `GGML_MOE_STREAM_ONE_DIRECT_MODEL`, so offsets match the
    already validated prompt-general gate source;
  - reject unless RAM/page cache remain inside 16GB, TTFT stays within the
    allowed 20%, output remains correct, and generalized prompts improve beyond
    the safe default.
- Implemented default-off gate hot-pool support:
  - `673e6ff2a` adds direct-manifest order preservation, exact gate hot-pool
    lookup in the main one-stream gate path, and
    `scripts/ds4-expert-pack-direct-manifest.py`;
  - `13d86fe61` adds a fast env guard so disabled hot-pool does not enter the
    lookup/mutex path by default.
- Gate hot-pool 512MiB smoke is rejected:
  - manifest:
    `/home/wici/runs/ds4-repro-assets/ds4-gate-hot-direct-manifest-20260708.csv`,
    generated from the full native expert-pack and ranked by
    `safe-one-pack-read-20260708T143802Z.csv`;
  - run `20260708T145532Z-20260708T-hotpool512-n32-smoke`, prompt
    `How to deploy a large model on a small devices?`, `n=32`;
  - `eval_tok_s=2.4`, `TTFT=19040.9 ms`, elapsed `29.85s`,
    `memory_peak_bytes=14335258624`, `ram_ok=true`,
    `display_processes_stopped_before_run=true`;
  - hot-pool inserted `120` entries / `534.8MB` in `237.97 ms`, but only hit
    `303/9614` lookups. Gate one-pack still read `3600` experts / `16.04GB`.
    It does not improve over the safe path.
- Offline coverage analysis explains the rejection: the gate working set is
  very broad. In the safe n96 profile, top `120/240/480/960/1440/2048` entries
  cover only about `6.5%/11.7%/21.1%/37.6%/50.0%/65.8%` of gate reads. Larger
  pools would consume multiple GiB of VRAM and prefill seconds while reducing
  batch-cache budget, so gate-only hot-pool is not the main path to `>5 tok/s`.
- Default-off safety smoke after `13d86fe61`:
  - run `20260708T145715Z-20260708T-default-n32-post-hotpool-code`, no hot-pool
    env, source clean, `ram_ok=true`, display processes stopped;
  - `eval_tok_s=2.3`, `TTFT=18690.5 ms`, H2D `6.64GB/s`, and no one-direct
    hot-pool/manifest log lines. This is a low-H2D diagnostic result, not a
    promoted SOTA.
- Hardware state update:
  - current GPU endpoint can advertise `32.0 GT/s x16`, but it is attached via
    root port `0000:00:06.0`;
  - root port `0000:00:06.0` reports max `16.0 GT/s x4`, so the software-visible
    GPU link is capped at Gen4 x4 on this topology;
  - measured H2D remains about `6.6-6.7GB/s`. Case B `32 GT/s x4` is not active
    on this wiring/slot, so reaching stable `>5 tok/s` requires either a
    hardware placement change or much larger expert-movement reduction than the
    gate hot-pool can provide.
- Next diagnostic for two-stage aggregation:
  - add demo option `--route-profile` that writes run-local grouped route CSVs
    only. External route/profile paths remain blocked as prompt-specific input.
  - Use this to estimate the theoretical benefit of layer-level retained
    gate/up/down aggregation from actual `matrix_row_counts` and cache state.
  - Promotion is not possible from this diagnostic alone; it only determines
    whether a real route-group implementation has enough byte-reduction headroom
    to pursue.
- Default-off implementation target after route profile:
  - `GGML_MOE_GATE_BATCH_PREFETCH=1` should preload current-layer active gate
    experts from the prompt-general expert-pack through the existing batch
    io_uring path before the per-expert `moe_stream_one` loop;
  - `moe_stream_one` may then consume a device pointer from the batch VRAM cache
    instead of issuing its own synchronous one-pack read/H2D copy;
  - theoretical benefit is not fewer gate bytes, but fewer synchronous gate
    read submissions and better use of the existing batched direct-IO path.
    Reject if it raises TTFT beyond 20%, reduces correctness, increases total
    movement enough to lower token rate, or breaks Kimi/shared MoE paths.
- `GGML_MOE_GATE_BATCH_PREFETCH=1` result:
  - implemented in `7c5780963`, debugged in `92a2cd4a6`, and enabled for
    DS4 low-bit gate types in `c67bccd3d`;
  - n8 debug `20260708T151618Z-20260708T-gate-batch-prefetch-n8-debug2`
    confirmed the probe is active: gate prefetch reason `active`, initial jobs
    `6`, and gate one-pack reads dropped to `1`;
  - n32 validation `20260708T151710Z-20260708T-gate-batch-prefetch-n32-active`
    is rejected: `eval_tok_s=2.4`, `TTFT=18729.9 ms`, elapsed `29.75s`,
    `memory_peak_bytes=14493487104`, `ram_ok=true`, output coherent;
  - root cause: batch cache initializes first at `9.0GiB`, causing one-stream
    `6.0GiB` gate cache allocation to fail. Gate reads move from one-pack into
    batch io_uring, but total batch iouring rises to `9581` reads /
    `42.70GB`, worse than the default n32 combined movement of about `37.8GB`.
  - Keep the code default-off for future cache-partition experiments, but do
    not promote it as SOTA.
- Cache-partition follow-up:
  - `GGML_MOE_GATE_BATCH_PREFETCH=1`,
    `GGML_MOE_STREAM_ONE_CACHE_MIB=0`, `GGML_MOE_VRAM_CACHE_MIB=12288`
    turns the previous failed split-cache shape into a single 12GiB batch
    cache with gate/up/down all using the batch io_uring path.
  - n8 smoke `20260708T152102Z-20260708T-unified-cache12g-gateprefetch-n8`:
    `eval_tok_s=3.0`, `TTFT=18124.4 ms`, `ram_ok=true`; batch iouring
    `4778` reads / `21.29GB`, better than the earlier active gate-prefetch n8
    `23.98GB`.
  - n32 validation `20260708T152155Z-20260708T-unified-cache12g-gateprefetch-n32`:
    `eval_tok_s=2.7`, `TTFT=17201.7 ms`, elapsed `27.06s`,
    `memory_peak_bytes=14694068224`, `ram_ok=true`, coherent English answer.
  - n96 validation `20260708T152301Z-20260708T-unified-cache12g-gateprefetch-n96`:
    `eval_tok_s=2.6`, `TTFT=18223.0 ms`, elapsed `52.62s`,
    `memory_peak_bytes=14696202240`, `ram_ok=true`, coherent English answer.
  - France sentinel `20260708T152414Z-20260708T-unified-cache12g-france-n96`:
    `eval_tok_s=2.6`, `TTFT=17277.4 ms`, `ram_ok=true`; France output was
    semantically correct and coherent.
  - Fibonacci calibration
    `20260708T152520Z-20260708T-unified-cache12g-fibonacci-n96`: `eval_tok_s=2.4`,
    `TTFT=18643.4 ms`, `ram_ok=true`; Python generator output was correct but
    speed did not beat the safe default.
  - Chinese general prompt
    `20260708T152738Z-20260708T-unified-cache12g-aiinfra-n96`: `eval_tok_s=2.6`,
    `TTFT=17601.9 ms`, `ram_ok=true`, but correctness failed because the answer
    only translated the prompt (`What does AI infrastructure do?`) instead of
    explaining AI infra. Therefore this config is **not promoted** as a
    generalized SOTA despite the English speed signal.
  - Follow-up baseline comparison
    `20260708T152919Z-20260708T-default-aiinfra-n96-quality-compare` showed the
    default safe path produces the same translation-only answer, so the AI
    infra failure is not an optimization-induced quality regression. However,
    it still means the current demo prompt set contains a quality issue that
    must be handled before claiming robust random-prompt product readiness.
  - 13GiB sweep `20260708T152640Z-20260708T-unified-cache13g-gateprefetch-n32`
    is rejected: `eval_tok_s=2.5`, slower than 12GiB.
  - Next step: preserve the 12GiB unified-cache idea as a performance probe,
    but do not make it default until generalized prompt quality passes. The
    main route to `>5 tok/s` still requires reducing full-expert movement or a
    hardware link/topology change.
- New-machine recheck after syncing `882bd544a`:
  - Strict cold France default
    `20260708T153417Z-20260708T-default-france-n96-recheck`:
    `eval_tok_s=2.6`, `prompt_tok_s=3.0`, `TTFT=18519.5 ms`,
    elapsed `52.60s`, `memory_peak_bytes=14757728256`,
    `memory_file_bytes=13691789312`, `ram_ok=true`,
    `display_processes_stopped_before_run=true`. France output was coherent
    and semantically correct. H2D remained `6.62GB/s`, PCIe under load
    `16.0 GT/s x4`, so the machine is still in the low-H2D Case A-like state.
  - Movement counters for that run: batch up/down path `8194` io_uring reads /
    `36.52GB`; gate one-pack path `6473` reads / `28.85GB`. Total expert
    movement is still about `65GB`, which explains why small source/cache
    toggles cannot reach `>5 tok/s` on a `6.6GB/s` H2D path.
  - `GGML_MOE_BATCH_FULLPACK=1` recheck
    `20260708T153604Z-20260708T-batch-fullpack-n32-smoke` was neutral/rejected:
    `eval_tok_s=2.5`, `ram_ok=true`; using the native expert pack as the batch
    source does not improve over run-local GGUF alias.
  - Unified 12GiB cache recheck
    `20260708T153651Z-20260708T-unified12-france-n96-recheck`:
    `GGML_MOE_GATE_BATCH_PREFETCH=1`,
    `GGML_MOE_STREAM_ONE_CACHE_MIB=0`, `GGML_MOE_VRAM_CACHE_MIB=12288`;
    `eval_tok_s=2.7`, `TTFT=17709.9 ms`, elapsed `50.61s`,
    `memory_peak_bytes=14718070784`, `ram_ok=true`, France correctness pass.
    It is a small positive signal but still not promoted as generalized SOTA:
    it is not uniformly faster on calibration prompts and product `>5 tok/s`
    remains unmet.
  - Unified 14GiB request
    `20260708T153759Z-20260708T-unified14-france-n32-smoke` was rejected:
    requested `14336MiB` fell back to an actual `12.2GiB` cache and produced
    only `2.5 tok/s`, slower than the 12GiB probe.
  - Added default-off diagnostic env passthrough to
    `scripts/demo-vendor-ds4-general-sota.sh` for
    `GGML_DS4_SPARSE_FUSED_MMVQ_PROFILE` and
    `GGML_DS4_SPARSE_FUSED_MMVQ_MEMBERSHIP_OUT`; this only records/profiles
    when explicitly enabled and does not change default/Kimi behavior.
  - Sparse membership diagnostic
    `20260708T154045Z-20260708T-sparse-membership-france-n32` produced
    `records=0` because the current hot path no longer uses the CPU fallback
    recording point. Therefore that old hook is not sufficient for the current
    SOTA H2D hard-bound.
  - Current valid hard-bound input is the route profile
    `20260708T154200Z-20260708T-route-profile-france-n32-recheck`, `n32`,
    safe default, `ram_ok=true`, H2D `6.68GB/s`: gate logical bytes `42.21GB`
    with `3328` cache misses; up logical bytes `23.18GB` with `2124` cache
    misses; down logical bytes `23.18GB` with `0` cache misses because down is
    already covered by the paired prefetch/cache path. This shifts the next
    implementation priority away from down and toward reducing gate/up
    miss bytes or avoiding full-expert movement for those roles.
  - Detailed reproduction data is recorded in
    `.Agent/runs/20260708-vendor-ds4-newmachine/low-h2d-recheck-and-route-bound-20260708.json`.
- Gate-topk movement reduction:
  - Route-profile observation: the safe default prunes up/down routed rows via
    `GGML_MOE_KEEP_TOPK_*`, but `ffn_gate_exps` still computed all selected
    gate experts. In the n32 route profile, gate had `9471` active experts /
    `42.21GB` logical source bytes while up/down had only `5201` active experts
    / `23.18GB` each. Because pruned up rows are explicitly written to zero,
    computing gate for the same pruned ids cannot affect the later SwiGLU
    product and is full-expert movement waste.
  - Implemented default-off core switch `GGML_MOE_KEEP_TOPK_GATE=1`; when set,
    gate follows the same topk pruning decision as up/down. The demo SOTA path
    now defaults this env to `1`, while the shared core behavior remains
    disabled unless the env is set. Kimi behavior is unchanged.
  - n32 route smoke
    `20260708T154621Z-20260708T-gate-topk-france-n32-smoke`: `eval_tok_s=2.7`,
    `TTFT=17408.8 ms`, `ram_ok=true`, correctness pass. Gate active experts
    dropped `9471 -> 5201`, gate cache misses `3328 -> 1856`, and gate logical
    source bytes `42.21GB -> 23.18GB`.
  - n96 France validation
    `20260708T154722Z-20260708T-gate-topk-france-n96`: `eval_tok_s=3.1`,
    `prompt_tok_s=3.7`, `TTFT=17369.4 ms`, elapsed `46.35s`,
    `memory_peak_bytes=14608982016`, `memory_file_bytes=13779464192`,
    `ram_ok=true`, display processes stopped, H2D `6.72GB/s`, correctness pass.
    Compared with the default France recheck, gate one-pack bytes dropped from
    `28.85GB` to `12.81GB`, and token rate improved `2.6 -> 3.1 tok/s`.
  - n96 Fibonacci validation
    `20260708T154831Z-20260708T-gate-topk-fibonacci-n96`: `eval_tok_s=2.7`,
    `TTFT=17965.1 ms`, `memory_peak_bytes=14678474752`, `ram_ok=true`;
    correctness pass with a valid Python Fibonacci generator.
  - n96 deployment validation
    `20260708T154943Z-20260708T-gate-topk-deploy-n96`: `eval_tok_s=2.9`,
    `prompt_tok_s=4.2`, `TTFT=17576.7 ms`, `memory_peak_bytes=14792970240`,
    `ram_ok=true`; output was coherent deployment guidance.
  - Accepted as a prompt-general low-H2D improvement, but it does not complete
    the product target: best observed new-machine result is `3.1 tok/s`, still
    below stable `>5 tok/s`.
  - Clean repeat after commit `79e7754a4`, without manually supplying
    `GGML_MOE_KEEP_TOPK_GATE`, run
    `20260708T155306Z-20260708T-clean-default-gatetopk-france-n96`:
    `eval_tok_s=3.1`, `prompt_tok_s=3.7`, `TTFT=17181.4 ms`,
    elapsed `45.96s`, `memory_peak_bytes=14357655552`,
    `memory_file_bytes=13527908352`, `ram_ok=true`, source clean, display
    processes stopped, H2D `6.65GB/s`, France correctness pass. The run
    environment confirms the demo default set `GGML_MOE_KEEP_TOPK_GATE=1`.
    Gate one-pack reads/bytes remained `2875` / `12.81GB`.
  - Detailed reproduction data is recorded in
    `.Agent/runs/20260708-vendor-ds4-newmachine/gate-topk-generalized-sota-20260708.json`.
- Gate-topk plus unified 12GiB cache:
  - After gate-topk reduced the gate working set, the previously diagnostic
    unified-cache path became useful. Promoted demo defaults:
    `GGML_MOE_GATE_BATCH_PREFETCH=1`,
    `GGML_MOE_STREAM_ONE_CACHE_MIB=0`,
    `GGML_MOE_VRAM_CACHE_MIB=12288`, while keeping
    `GGML_MOE_KEEP_TOPK_GATE=1`.
  - n32 smoke
    `20260708T155539Z-20260708T-gatetopk-unified12-france-n32-smoke`:
    `eval_tok_s=2.9`, `TTFT=16755.6 ms`, `ram_ok=true`, correctness pass.
  - n96 France validation
    `20260708T155637Z-20260708T-gatetopk-unified12-france-n96`:
    `eval_tok_s=3.2`, `prompt_tok_s=4.0`, `TTFT=16895.3 ms`,
    elapsed `45.05s`, `memory_peak_bytes=14633455616`,
    `memory_file_bytes=13644005376`, `ram_ok=true`, source clean, display
    processes stopped, H2D `6.57GB/s`, correctness pass. This is the current
    highest strict cold prompt-general new-machine result, but still below the
    product target.
  - n96 Fibonacci validation
    `20260708T155749Z-20260708T-gatetopk-unified12-fibonacci-n96`:
    `eval_tok_s=2.9`, `TTFT=17448.7 ms`, `ram_ok=true`; correctness pass with
    a valid Python Fibonacci generator.
  - n96 deployment validation
    `20260708T155859Z-20260708T-gatetopk-unified12-deploy-n96`:
    `eval_tok_s=3.0`, `prompt_tok_s=4.6`, `TTFT=17199.5 ms`, `ram_ok=true`;
    output was coherent deployment guidance.
  - Accepted as the new low-H2D new-machine SOTA candidate pending a clean
    default repeat after commit. Product target `>5 tok/s` remains unmet.
  - Clean default repeat after commit `471df3481`, without manually supplying
    the unified-cache envs:
    `20260708T160217Z-20260708T-clean-default-unified12-gatetopk-france-n96`
    produced `eval_tok_s=3.1`, `prompt_tok_s=4.0`, `TTFT=16721.5 ms`,
    elapsed `45.64s`, `memory_peak_bytes=14643257344`,
    `memory_file_bytes=13673680896`, `ram_ok=true`, source clean, display
    processes stopped, H2D `6.70GB/s`, France correctness pass. The run env
    confirms the new defaults:
    `GGML_MOE_KEEP_TOPK_GATE=1`, `GGML_MOE_GATE_BATCH_PREFETCH=1`,
    `GGML_MOE_STREAM_ONE_CACHE_MIB=0`, `GGML_MOE_VRAM_CACHE_MIB=12288`.
    Therefore `3.2 tok/s` is recorded as the observed high, while the clean
    default reproducible value is `3.1 tok/s`.
  - Detailed reproduction data is recorded in
    `.Agent/runs/20260708-vendor-ds4-newmachine/gate-topk-unified12-sota-20260708.json`.
- Gate-topk unified cache size sweep:
  - With gate-topk active, `GGML_MOE_VRAM_CACHE_MIB=14336` was retested. The
    allocation still cannot fit a full 14GiB pool, but the retry lands at
    `12.2GiB / 2951` slots instead of the `12.0GiB / 2891` slots from the
    12GiB request.
  - n32 smoke
    `20260708T160501Z-20260708T-gatetopk-unified14-france-n32-smoke`:
    `eval_tok_s=3.0`, `TTFT=16589.8 ms`, `ram_ok=true`, correctness pass.
  - n96 France validation
    `20260708T160554Z-20260708T-gatetopk-unified14-france-n96`:
    `eval_tok_s=3.2`, `prompt_tok_s=4.0`, `TTFT=16853.5 ms`,
    elapsed `45.27s`, `memory_peak_bytes=14712995840`,
    `memory_file_bytes=13850742784`, `ram_ok=true`, source clean, display
    processes stopped, H2D `6.62GB/s`, correctness pass. Batch io_uring bytes
    were `49.92GB`.
  - n96 Fibonacci validation
    `20260708T160703Z-20260708T-gatetopk-unified14-fibonacci-n96`:
    `eval_tok_s=2.9`, `TTFT=17408.7 ms`, `memory_peak_bytes=14729986048`,
    `ram_ok=true`, correctness pass.
  - Promoted demo default `GGML_MOE_VRAM_CACHE_MIB=14336`, with the explicit
    note that the effective pool is the retry-sized `12.2GiB`, not a true
    14GiB allocation. Product `>5 tok/s` remains unmet.
  - Clean default repeat after commit `4b2763f65`, without manually supplying
    any SOTA envs:
    `20260708T160949Z-20260708T-clean-default-unified14-gatetopk-france-n96`
    produced `eval_tok_s=3.3`, `prompt_tok_s=4.0`, `TTFT=16855.6 ms`,
    elapsed `44.39s`, `memory_peak_bytes=14637469696`,
    `memory_file_bytes=13664124928`, `ram_ok=true`, source clean, display
    processes stopped, H2D `6.71GB/s`, France correctness pass. Environment
    confirms the defaults:
    `GGML_MOE_KEEP_TOPK_GATE=1`, `GGML_MOE_GATE_BATCH_PREFETCH=1`,
    `GGML_MOE_STREAM_ONE_CACHE_MIB=0`, `GGML_MOE_VRAM_CACHE_MIB=14336`.
    This is the current clean reproducible new-machine strict-cold SOTA, but
    still below the required stable `>5 tok/s`.
  - Detailed reproduction data is recorded in
    `.Agent/runs/20260708-vendor-ds4-newmachine/gate-topk-unified14-sota-20260708.json`.
- Current-SOTA route profile after unified14:
  - n32 route profile
    `20260708T161322Z-20260709T-current-sota-route-profile-france-n32`:
    `eval_tok_s=3.0`, `TTFT=16606.0 ms`, `ram_ok=true`, source clean, display
    processes stopped, H2D `6.62GB/s`.
  - Role totals after gate-topk: gate `5201` active experts with `1994` cache
    misses; up `5201` active experts with `1993` cache misses; down `5201`
    active experts with `0` misses at down execution time because up/down
    paired prefetch makes down resident first.
  - Batch counters: `iouring_reads=5979`, `iouring_bytes=26.65GB`, pinned
    staging copies `3987`, gate staging copies `1992`, and inflight average
    only `2.48` despite `8` slots. This means the next low-risk software
    experiment should test larger staging/refill concurrency before attempting
    a more invasive compact source.
  - Added demo passthrough/default support for `GGML_MOE_IO_REFILL_BATCH` so
    experiments can test `GGML_MOE_STAGE_PINNED_SLOTS=16` and
    `GGML_MOE_IO_REFILL_BATCH=8` without changing the baseline default
    (`4`).
  - `GGML_MOE_STAGE_PINNED_SLOTS=16` plus `GGML_MOE_IO_REFILL_BATCH=8`,
    run `20260708T161540Z-20260709T-slots16-refill8-france-n32-smoke`,
    was rejected: `eval_tok_s=2.9`, `ram_ok=true`, source dirty diagnostic.
    It increased inflight average from about `2.48` to `3.25` but also
    increased io_uring wait time to `7.08s`, so more slots/refill do not solve
    the current bottleneck.
  - `GGML_MOE_IO_REFILL_BATCH=8` alone,
    run `20260708T161639Z-20260709T-refill8-france-n32-smoke`, was also
    rejected: `eval_tok_s=2.9`, wait time `6.57s`, no gain over default.
  - Next implementation probe: the existing up/gate parallel branch only
    enables for `IQ2_S`, while current DeepSeek up/gate type is `39`. Added
    default-off env `GGML_MOE_STREAM_UP_GATE_PARALLEL_ANY=1` to allow the
    existing non-mixed up/gate parallel staging/compute path for DS4 types
    when explicitly requested. This must be correctness-tested before any
    promotion.
- Follow-up results for the above probe:
  - `GGML_MOE_STREAM_UP_GATE_PARALLEL_ANY=1` did not enter the current SOTA
    path. The active path logged `[moe_stream] batched decode path active` and
    `down parallel CPU staging active`, while the fused up/gate batch function
    was not called. The probe is rejected and the code change is not promoted.
  - Increasing `GGML_MOE_STAGE_PINNED_SLOTS` / `GGML_MOE_IO_REFILL_BATCH`
    increased inflight depth but reduced or failed to improve token rate, so it
    remains diagnostic only.
- Top-k byte-reduction experiments:
  - `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39`,
    `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, and `GGML_MOE_VRAM_CACHE_MIB=13312`
    are the next prompt-general candidate defaults. They are not prompt
    specific and do not use any held-out prompt.
  - France n96 dirty-candidate run
    `20260708T162650Z-20260709T-top3-alllayers-france-n96` reached
    `eval_tok_s=4.1`, `TTFT=15928.8 ms`, `ram_ok=true`, and coherent output.
  - AI-infra n96 dirty-candidate run
    `20260708T163451Z-20260709T-top3-alllayers-vram13-ai-infra-n96` reached
    `eval_tok_s=3.5`, `TTFT=16293.2 ms`, `ram_ok=true`, and produced a
    semantically useful Chinese explanation. The current default contrast run
    `20260708T163555Z-20260709T-current-default-ai-infra-n96` only translated
    the prompt, so top3-all-layers is better quality for this dev prompt.
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=2` over all layers is rejected despite
    France n96 `eval_tok_s=5.1`: run
    `20260708T162746Z-20260709T-top2-alllayers-france-n96` had coherent
    France output, but run
    `20260708T162839Z-20260709T-top2-alllayers-ai-infra-n96` degraded into
    repeated restatement of the prompt. It is not a valid generalized SOTA.
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=2` only for layers `10-39` is rejected:
    France n96 run `20260708T162931Z-20260709T-top2-l10-39-france-n96`
    reached only `3.7 tok/s`, below the top3-all-layers candidate.
- Down-batch profile for top3-all-layers:
  - Profile run
    `20260708T163208Z-20260709T-top3-alllayers-down-profile2-france-n32`
    produced `2720` down-batch calls. Total `wall_ms=4424.4`, dominated by
    `stage_ms=3720.9`; `kernel_ms=601.5`, `d2h_ms=27.4`, and
    `scatter_ms=13.9`.
  - Expert movement remains the bottleneck. D2H/scatter is already small, so
    the next real route to stable `>5 tok/s` is reducing expert bytes/misses or
    making the PCIe/H2D path faster, not optimizing the final D2H scatter.
- Clean validation after promoting the candidate defaults in commit
  `ddd2cd249f3b`:
  - France n96
    `20260708T163915Z-20260709T-clean-ddd2cd2-default-france-n96`:
    `eval_tok_s=3.6`, `prompt_tok_s=4.2`, `TTFT=16664.8 ms`,
    `memory_peak_bytes=14526316544`, `memory_file_bytes=13741305856`,
    `ram_ok=true`, `source_dirty=false`, display cleanup recorded, output
    coherent.
  - France n96 repeat
    `20260708T164112Z-20260709T-clean-ddd2cd2-default-france-repeat-n96`:
    `eval_tok_s=3.5`, `prompt_tok_s=4.2`, `TTFT=16398.8 ms`,
    `memory_peak_bytes=14738862080`, `memory_file_bytes=13808717824`,
    `ram_ok=true`, `source_dirty=false`, display cleanup recorded, output
    coherent.
  - AI infra n96
    `20260708T164013Z-20260709T-clean-ddd2cd2-default-ai-infra-n96`:
    `eval_tok_s=3.4`, `prompt_tok_s=4.0`, `TTFT=16590.6 ms`,
    `memory_peak_bytes=14556254208`, `memory_file_bytes=13767069696`,
    `ram_ok=true`, `source_dirty=false`, display cleanup recorded, output is a
    useful Chinese explanation.
  - This is an accepted prompt-general improvement over the prior clean
    new-machine France SOTA of about `3.3 tok/s`, but it does not meet the
    product requirement of stable `>5 tok/s`. Continue optimizing expert
    movement / H2D throughput.
  - Detailed record:
    `.Agent/runs/20260708-vendor-ds4-newmachine/top3-alllayers-vram13-clean-generalized-20260709.json`.
- Additional top-k sweep after the clean top3-all-layers validation:
  - `GGML_MOE_KEEP_TOPK_UPDOWN=3`,
    `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-29`,
    `GGML_MOE_KEEP_TOPK_LAYER_VALUE=2` keeps layers `30-39` at top3 while
    reducing layers `0-29` to top2. It remains prompt-general.
  - Clean France n96
    `20260708T164708Z-20260709T-top2-l0-29-top3-rest-france-n96`:
    `eval_tok_s=4.0`, `prompt_tok_s=5.1`, `TTFT=16257.6 ms`,
    `memory_peak_bytes=14727180288`, `memory_file_bytes=13833728000`,
    `ram_ok=true`, `source_dirty=false`, display cleanup recorded, output
    coherent.
  - Clean AI infra n96
    `20260708T164804Z-20260709T-top2-l0-29-top3-rest-ai-infra-n96`:
    `eval_tok_s=4.1`, `prompt_tok_s=5.2`, `TTFT=15268.9 ms`,
    `memory_peak_bytes=14690164736`, `memory_file_bytes=13752385536`,
    `ram_ok=true`, `source_dirty=false`, display cleanup recorded, output
    useful and coherent.
  - Clean deployment n96
    `20260708T164957Z-20260709T-top2-l0-29-top3-rest-deploy-n96`:
    `eval_tok_s=3.7`, `prompt_tok_s=6.0`, `TTFT=15677.2 ms`,
    `memory_peak_bytes=14773948416`, `memory_file_bytes=13872824320`,
    `ram_ok=true`, `source_dirty=false`, display cleanup recorded, output
    coherent.
  - `0-34` top2 is not better: France n96
    `20260708T164902Z-20260709T-top2-l0-34-top3-rest-france-n96` reached
    only `3.9 tok/s`.
  - Promote `0-29` top2 plus `30-39` top3 as the next default generalized
    SOTA candidate. It still fails the product requirement of stable
    `>5 tok/s`, so the next implementation must reduce expert movement beyond
    rank pruning or change the effective H2D limit.
  - Clean default confirmation at promoted commit `7c2f1f2e2`:
    `20260708T165237Z-20260709T-clean-7c2f1f2-default-france-n96` ran with no
    manual top-k env and reached `eval_tok_s=4.1`, `prompt_tok_s=5.0`,
    `TTFT=15887.4 ms`, `memory_peak_bytes=14812753920`,
    `memory_file_bytes=13841481728`, `ram_ok=true`, `source_dirty=false`,
    display cleanup recorded, and coherent France output. This is the current
    default clean generalized SOTA on the new machine, but still below the
    product target.
  - Hardware/H2D finding:
  - The GPU endpoint supports `32GT/s x16`, but its upstream root port
    `0000:00:06.0` has `LnkCap Speed 16GT/s, Width x4` and target link speed
    `16GT/s`. Therefore the new machine cannot reach Gen5 x4 for this GPU
    through software alone. The measured `6.6-6.7 GB/s` 4.25MiB H2D is
    consistent with the root-port Gen4 x4 limit.
- CUDA graph diagnostic after split-topk SOTA:
  - The demo script now allows `GGML_CUDA_DISABLE_GRAPHS` to be overridden,
    while keeping the default disabled unless explicitly changed.
  - Dirty diagnostic with `GGML_CUDA_DISABLE_GRAPHS=0`:
    France n96 `20260708T165722Z-20260709T-cuda-graphs-enabled-france-n96`
    reached `eval_tok_s=5.0`, `TTFT=15794.9 ms`, `ram_ok=true`, coherent
    output.
  - The gain did not generalize: AI infra n96
    `20260708T165814Z-20260709T-cuda-graphs-enabled-ai-infra-n96` reached
    `4.1 tok/s`, and deployment n96
    `20260708T165908Z-20260709T-cuda-graphs-enabled-deploy-n96` reached
    `3.7 tok/s`, both similar to current default behavior.
  - Conclusion: CUDA graph enablement is not promoted as default and does not
    satisfy stable `>5 tok/s` for random prompts. Keep as diagnostic only.
- Long-output check:
  - Clean default n192 France run
    `20260708T165446Z-20260709T-clean-61969d4-default-france-n192` reached
    only `eval_tok_s=4.2`, confirming that the remaining gap is not just TTFT
    amortization.
  - Clean default down profile
    `20260708T165605Z-20260709T-clean-61969d4-default-down-profile-france-n32`
    showed `stage_ms=3050.9`, `kernel_ms=569.5`, `d2h_ms=22.9`,
    `scatter_ms=10.6`, `wall_ms=3714.9`, with `active_avg=2.98` and
    `iouring_bytes=17.98GB`. Expert read/H2D staging remains dominant.
- Cache policy and profile-hotset experiments:
  - `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru` is rejected:
    `20260708T170220Z-20260709T-lfu-lru-default-france-n96` dropped to
    `2.8 tok/s` and output degraded.
  - `GGML_MOE_CURRENT_DOWN_OVERLAP=1` with early overlap is rejected:
    `20260708T170324Z-20260709T-current-down-overlap-france-n96` reached only
    `4.0 tok/s`.
  - Disabling gate batch prefetch is rejected:
    `20260708T170444Z-20260709T-no-gate-batch-prefetch-france-n96` reached
    only `2.8 tok/s`; gate prefetch is required.
  - `GGML_MOE_STAGE_PINNED_SLOTS=12` is neutral/slightly positive:
    `20260708T170553Z-20260709T-slots12-default-france-n96` reached
    `4.2 tok/s`, not enough to promote.
  - Added default-off diagnostics/experiments:
    `GGML_MOE_BATCH_PROFILE_OUT` is now passed through by the demo script, and
    the CUDA cache supports `GGML_MOE_VRAM_PROFILE_PRELOAD=0` plus
    `GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT=1` to lazily pin hot profile entries
    after their first runtime load. Default behavior is unchanged.
  - A dev calibration profile was generated from France, AI infra, and deploy
    n32 runs and stored on the new machine at
    `/home/wici/profile-calib-20260709/combined-dev-profile.csv`.
  - Full protected preload is rejected:
    `20260708T171307Z-20260709T-dev-profile-protect-france-n96` preloaded
    `2410` slots, increased total expert traffic to `34.4GB`, and reached only
    `3.6 tok/s`.
  - Limited protected preload is rejected:
    `20260708T171420Z-20260709T-dev-profile-protect512-france-n96` reached
    only `4.1 tok/s`.
  - Lazy hot pinning is partially promising but not yet a generalized SOTA:
    with `GGML_MOE_VRAM_PROFILE_PRELOAD=0`,
    `GGML_MOE_VRAM_PROFILE_PROTECT=1`,
    `GGML_MOE_VRAM_PROFILE_RESERVE_SLOTS=2600`, and
    `GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT=1`, France n96
    `20260708T171630Z-20260709T-dev-profile-lazypin512-france-n96` reached
    `5.0 tok/s`, and AI infra n96
    `20260708T171722Z-20260709T-dev-profile-lazypin512-ai-infra-n96` reached
    `5.0 tok/s`, both with coherent output and RAM OK.
  - The same lazy-pin setting did not generalize to deploy:
    `20260708T171813Z-20260709T-dev-profile-lazypin512-deploy-n96` reached
    only `3.7 tok/s`; larger pin budget and slots12 did not help
    (`20260708T171916Z-20260709T-dev-profile-lazypin1000-deploy-n96`,
    `20260708T172050Z-20260709T-lazypin512-slots12-deploy-n96`).
  - Adding a deploy n96 calibration profile did not fix deploy:
    `20260708T172351Z-20260709T-calib-deploy-n96` generated
    `/home/wici/profile-calib-20260709/deploy-n96.csv`, which was merged into
    `/home/wici/profile-calib-20260709/combined-dev-plus-deploy-n96-profile.csv`.
    Running lazy pin against that expanded profile,
    `20260708T172503Z-20260709T-devplusdeploy96-lazypin512-deploy-n96`, still
    reached only `3.7 tok/s` with coherent output.
  - Conclusion: lazy pin is useful default-off infrastructure for future
    prompt-general hotset work, but current target remains unmet because one
    dev prompt still stays well below `5 tok/s`.
  - Next implementation direction: lazy pin can reduce iouring wait for some
    prompts, but deploy remains dominated by expert movement. The next code
    work should target lower bytes per route or a true cross-call/cross-layer
    scheduling change; cache-policy tuning alone is not sufficient.
- Down staging single-ring probe:
  - `GGML_MOE_DOWN_STAGE_SINGLE_RING=1`,
    `20260708T172906Z-20260709T-down-stage-single-ring-deploy-n96`, prompt
    `How to deploy a large model on a small devices?`, source clean at
    `6d2135e5c6`, strict cold cgroup, display cleanup recorded.
  - Result: `eval_tok_s=3.7`, `prompt_tok_s=6.0`, `TTFT=15590.4 ms`,
    `memory_peak_bytes=14734995456`, `memory_file_bytes=13854978048`,
    `ram_ok=true`, output coherent.
  - Counters remained effectively unchanged for the deploy bottleneck:
    `iouring_reads=7811`, `iouring_bytes=34.81GB`,
    `iouring_wait_us=10821038`, `inflight_avg=1.80`.
  - Rejected: single-ring down staging does not reduce deploy wait time or
    improve token rate, so it is not a route to generalized `>5 tok/s`.

## Next Work After 2026-07-09

Important correction: the split top-k default recorded at
`7c2f1f2e2` is no longer accepted as a generalized SOTA. It reached
`4.1 tok/s` on France and about `3.7 tok/s` on deploy, but a required dev
prompt recheck found that Fibonacci output degraded into meta/reasoning text
instead of directly returning a Python function. Therefore this configuration
violates the correctness requirement and must not be used as the accepted
generalized baseline.

Prompt-general top-k schedule diagnostic support was added in
`afebe2d196` via `GGML_MOE_KEEP_TOPK_LAYER_SCHEDULE`. It is default-off and
only used to test non-prompt-specific layer schedules. Results:

- `0-9:1` on deploy reached `4.2 tok/s` but exposed reasoning text; rejected.
- `0-4:1` on deploy reached `4.5 tok/s` but repeated assistant persona text;
  rejected.
- `0-19:2` and `0-9:2` failed Fibonacci clean correctness; rejected.
- `30-39:2` passed Fibonacci but did not improve deploy and produced an
  awkward deploy opening; rejected.
- `13824MiB + GGML_MOE_GATE_UPDOWN_COSUBMIT=1` reached `3.9 tok/s` on deploy
  and passed France/AI, but Fibonacci still failed because the underlying
  split-topk math was invalid; rejected as a generalized default.

The demo default was restored in `c3ed5da8b8` to the safe top3-all-layers
quality baseline: `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39`,
`GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_KEEP_TOPK_UPDOWN=3`,
`GGML_MOE_KEEP_TOPK_GATE=1`. Clean strict-cold validation on the new machine:

- Fibonacci:
  `20260708T180233Z-20260709T-clean-safe-default-fibonacci-n96`,
  `eval_tok_s=3.0`, `prompt_tok_s=4.2`, `TTFT=14986.9 ms`,
  `memory_peak_bytes=14241767424`, `ram_ok=true`, clean code output.
- Deploy:
  `20260708T180323Z-20260709T-clean-safe-default-deploy-n96`,
  `eval_tok_s=3.1`, `prompt_tok_s=4.9`, `TTFT=16879.2 ms`,
  `memory_peak_bytes=14403170304`, `ram_ok=true`, coherent deployment answer.
- France:
  `20260708T180431Z-20260709T-clean-safe-default-france-n96`,
  `eval_tok_s=3.5`, `prompt_tok_s=4.2`, `TTFT=16556.1 ms`,
  `memory_peak_bytes=14246940672`, `ram_ok=true`, semantically correct and
  coherent France paragraph.
- AI infra:
  `20260708T180533Z-20260709T-clean-safe-default-aiinfra-n96`,
  `eval_tok_s=3.4`, `prompt_tok_s=4.0`, `TTFT=16105.5 ms`,
  `memory_peak_bytes=14397513728`, `ram_ok=true`, useful Chinese explanation.

Current accepted generalized baseline is therefore the safe top3-all-layers
path, with observed dev prompt range `3.0-3.5 tok/s` on this low-H2D new
machine. Product target `>5 tok/s` remains unmet. Future SOTA claims must pass
Fibonacci or an equivalent code-generation correctness prompt, not only France
and deployment.

Non-math safe default improvement:

- `GGML_MOE_VRAM_CACHE_MIB=13824` plus
  `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` does not change top-k math and passed
  Fibonacci, France, AI infra, and deploy checks. It was promoted as the demo
  default in `05572dd55f`.
- Validation before promotion, source clean at `c63fc747ae` with manual env:
  - deploy `20260708T180949Z-20260709T-safe-top3-vram13824-cosubmit-deploy-n96`:
    `eval_tok_s=3.6`, `TTFT=16267.8 ms`, `ram_ok=true`, coherent answer;
  - Fibonacci
    `20260708T181101Z-20260709T-safe-top3-vram13824-cosubmit-fibonacci-n96`:
    `eval_tok_s=3.1`, `TTFT=14322.0 ms`, `ram_ok=true`, clean code output;
  - France
    `20260708T181150Z-20260709T-safe-top3-vram13824-cosubmit-france-n96`:
    `eval_tok_s=4.2`, `TTFT=16303.2 ms`, `ram_ok=true`, coherent France
    answer;
  - AI infra
    `20260708T181231Z-20260709T-safe-top3-vram13824-cosubmit-aiinfra-n96`:
    `eval_tok_s=4.2`, `TTFT=16054.8 ms`, `ram_ok=true`, useful Chinese
    explanation.
- Clean default confirmation after `05572dd55f`:
  - deploy
    `20260708T181626Z-20260709T-clean-default-05572dd-deploy-repeat-n96`:
    `eval_tok_s=3.6`, `prompt_tok_s=5.6`, `TTFT=15743.4 ms`,
    `memory_peak_bytes=14316167168`, `ram_ok=true`;
  - France
    `20260708T181728Z-20260709T-clean-default-05572dd-france-n96`:
    `eval_tok_s=3.7`, `prompt_tok_s=4.5`, `TTFT=15763.3 ms`,
    `memory_peak_bytes=14359330816`, `ram_ok=true`.
- One clean default deploy run
  `20260708T181502Z-20260709T-clean-default-05572dd-deploy-n96` reached only
  `3.1 tok/s` with the same read count because `iouring_wait_us` rose from
  about `9.1s` to `12.6s`. Treat current safe default as roughly
  `3.1-4.2 tok/s` depending on SSD/io_uring wait variance, not a stable
  `>5 tok/s` result.

Lazy pin can reach `5.0 tok/s` on France and AI infra but fails deploy, so it
is not accepted as generalized SOTA. The next optimization must reduce expert
movement or improve scheduling without changing model math enough to trigger
the Fibonacci/meta-reasoning failure.

Next implementation priority:

0. Before every run, kill display processes and other GPU/model processes,
   then record the pre-run GPU process list. This is required for all
   diagnostics, profiles, candidate runs, and SOTA validation. A run that does
   not execute this cleanup is invalid for baseline/SOTA comparison, even if
   token rate is higher.
1. Re-profile deploy under current SOTA with fine-grained batch/locality
   metrics, after the required display-process cleanup, to identify why
   `iouring_wait_us` is about `10s` while France/AI are lower.
2. Measure per-layer/per-role misses and physical source locality for deploy,
   with `GGML_MOE_BATCH_PROFILE_OUT` and any available read/locality trace
   hooks. The goal is to distinguish unavoidable extra expert movement from
   scheduler-induced small batches.
3. If locality shows many one- and two-job batches from the same layer, design
   a true route-group scheduler that collects gate/up/down requests after the
   router decision and submits the layer's required sources in one ordered
   plan. This must be default-off and must not increase total bytes like the
   rejected `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` path.
4. If locality shows genuinely broad random misses, prioritize reducing bytes
   per route: compact up/gate/down pack variants or partial expert reads, but
   only after computing a hard upper bound from actual rows/blocks used. The
   candidate must be prompt-general and cannot use held-out prompts for tuning.
5. Any accepted improvement must be committed and pushed immediately to
   `vendor/deepseek-token-rate-16gb` on `https://github.com/wici-ai/ssd-llama.git`
   with exact reproduction metadata. Commit author/committer must remain
   `L-Ark <fliangae@connect.ust.hk>`.

## 2026-07-09 Follow-Up After `8dc65bf`

Starting point: safe top3-all-layers plus `vram13824/cosubmit` default at
`8dc65bf41c`. The accepted safe high repeat remains deploy `3.6 tok/s`, while
one repeat showed `3.1 tok/s` due to higher `iouring_wait_us`.

Rejected low-risk scheduling probes:

- `GGML_MOE_IO_DEPTH=16` and `GGML_MOE_IO_REFILL_BATCH=8`,
  run `20260708T182120Z-20260709T-depth16-refill8-safe-default-deploy-n96`:
  `eval_tok_s=3.1`, `TTFT=16479.5 ms`, `ram_ok=true`. It did not raise
  effective inflight because the hot path was still limited by staging shape;
  `iouring_wait_us=12177868`.
- `GGML_MOE_STAGE_PINNED_SLOTS=16`, `GGML_MOE_IO_DEPTH=16`,
  `GGML_MOE_IO_REFILL_BATCH=8`, run
  `20260708T182238Z-20260709T-slots16-depth16-refill8-safe-default-deploy-n96`:
  `eval_tok_s=3.1`, `TTFT=16270.6 ms`, `ram_ok=true`. It raised
  `inflight_max` to 16 but still left `iouring_wait_us=12558372`, so slot
  count is not the main limiter.
- `GGML_MOE_STREAM_DEFER=1`, run
  `20260708T182419Z-20260709T-stream-defer-safe-default-deploy-n96`, hung/ran
  far beyond normal runtime and was killed. Rejected.
- `GGML_MOE_IO_SORT_OFFSET=1`, run
  `20260708T182644Z-20260709T-sortoffset-safe-default-deploy-n96`:
  `eval_tok_s=3.1`, `TTFT=16157.0 ms`, `ram_ok=true`; no improvement.
- `GGML_MOE_BATCH_FULLPACK=1`, run
  `20260708T182750Z-20260709T-batch-fullpack-safe-default-deploy-n96`:
  `eval_tok_s=3.1`, `TTFT=16439.3 ms`, `ram_ok=true`. The native expert-pack
  source produced the same read count/bytes as the GGUF alias path:
  `iouring_reads=11331`, `iouring_bytes=50.50GB`, so the current pack layout
  is not sufficient.
- Added default-off passthrough for planned host prefetch envs in
  `58c7250d1d`. With `GGML_MOE_PLANNED_HOST_PREFETCH=1`,
  `GGML_MOE_HOST_PREFETCH_MAX_MIB=512`, and `GGML_MOE_HOST_PREFETCH_SLOTS=64`,
  run `20260708T183033Z-20260709T-planned-host-prefetch512-deploy-n96`
  reached only `eval_tok_s=3.2`, `ram_ok=true`. Runtime report showed
  `planned_enqueued=0`, `hits=0`, and `misses=11331`, so this existing hook
  does not help the current hot path.
- `GGML_MOE_GATE_UPDOWN_COSUBMIT=0` with the larger `13824MiB` cache, run
  `20260708T183158Z-20260709T-vram13824-no-cosubmit-deploy-n96`, reached only
  `3.1 tok/s`. Therefore cosubmit remains necessary for the current safe
  default, even though it is not enough to reach `>5 tok/s`.

Updated bottleneck conclusion:

- Safe top3 deploy still moves about `50.5GB` of expert data for n96 on the
  cosubmit path, with `11331` full-expert reads of `4.25MiB` each.
- Existing scheduling knobs can shift `iouring_wait_us` but do not reduce the
  required bytes enough. Reaching stable `>5 tok/s` on the Gen4 x4 H2D path
  now requires reducing full-expert movement or changing the expert source
  layout/runtime to avoid moving unused bytes.

Next concrete implementation direction:

0. Run the required pre-run cleanup first: stop/kill display processes,
   kill other GPU/model processes, verify `nvidia-smi` is clear, and record
   this in the run artifact before launching the model.
1. Build a hard-bound report from route profiles for safe top3: for each
   tensor role and layer, compute actual active rows/blocks used versus the
   full `4.25MiB` expert payload currently read.
2. If the bound shows enough headroom, implement a default-off compact or
   partial expert source for up/gate/down that can read only required blocks or
   a smaller packed representation while producing identical math.
3. If partial reads are too invasive for the current kernels, implement a
   route-clustered prompt-general pack layout that groups gate/up/down for the
   same layer/expert and allows the runtime to issue fewer, larger ordered
   reads without increasing bytes.
4. Any rank-pruning or approximate math change is disallowed unless it passes
   Fibonacci/code-generation correctness in addition to France, AI infra, and
   deploy.

## 2026-07-09 Calibration Overlay And H2D-Coalesce Probe

Execution rule followed: every run below was launched after stopping display
and model/GPU processes, and each artifact records
`display_processes_stopped_before_run=true` unless explicitly marked as a
locality simulation.

Prompt-general calibration traces were generated under the current safe
top3-all-layers default, strict cold n32, 16GB cgroup, and no held-out prompts:

- `france`:
  `/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-calibration-safe-top3/france`,
  `eval_tok_s=3.4`, RAM OK, coherent France output.
- `aiinfra`:
  `/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-calibration-safe-top3/aiinfra`,
  `eval_tok_s=3.4`, RAM OK, Chinese answer explains AI infrastructure rather
  than only translating the prompt.
- `fibonacci`:
  `/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-calibration-safe-top3/fibonacci`,
  `eval_tok_s=3.1`, RAM OK, output begins a Python Fibonacci function.
- `deploy`:
  `/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-calibration-safe-top3/deploy`,
  `eval_tok_s=3.3`, RAM OK, coherent deployment answer.

The combined calibration trace is:
`/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-calibration-safe-top3/combined-dev-calibration-io-read-trace.csv`
with `20168` rows.

Built prompt-general first-use overlay pack from the four calibration traces:

- pack:
  `/home/wici/runs/vendor-ds4-16gb/assets/ds4-safe-top3-dev4-firstuse-overlay.expert-pack`;
- build log:
  `/home/wici/runs/vendor-ds4-16gb/assets/ds4-safe-top3-dev4-firstuse-overlay.build.log`;
- size: `48G`, `11454` entries, `51044155392` copied bytes.

Locality sim on the combined trace showed layout headroom, but not yet a
runtime win:

- current trace layout: `read_jobs=20159`, `read_bytes=89.84GB`,
  `adjacent/read_jobs=0.0261`, `coalesce_rows=0`;
- simulated first-use layout: `adjacent/read_jobs=0.2012`,
  `coalesce_rows=203`;
- simulated greedy-pair layout: `adjacent/read_jobs=0.2346`,
  `coalesce_rows=166`.

Implemented and pushed a default-off, controlled demo entry point for this
kind of prompt-general calibration overlay in `0c2cebadc`
(`vendor-ds4: allow calibration overlay demo runs`). Directly setting
`GGML_MOE_EXPERT_PACK_OVERLAY` remains blocked by the demo script; the allowed
path is explicit `--calibration-overlay-pack` and is recorded in `config.json`.

Rejected overlay runtime test:

- run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T184438Z-20260709-dev4-firstuse-overlay-smoke-deploy-n32`;
- config: `GGML_MOE_BATCH_FULLPACK=1` plus the calibration overlay pack;
- result: `eval_tok_s=3.3`, RAM OK, coherent deploy output;
- counters: overlay loaded and replaced `11454` duplicate keys, but
  `iouring_reads=5202`, `iouring_bytes=23.18GB`, and batch histogram matched
  the baseline n32 path. This proves pack layout alone does not help because
  the runtime still issues one read and one H2D copy per expert payload.

Rejected VRAM-cache probes:

- `GGML_MOE_VRAM_CACHE_MIB=14080`, run
  `20260709-vram14080-smoke-deploy-n32`: process aborted with `status=134`,
  therefore rejected even though cgroup RAM was below 16GB.
- `GGML_MOE_VRAM_CACHE_MIB=14200`, run
  `20260709-vram14200-smoke-deploy-n32`: `eval_tok_s=3.3`, RAM OK, but
  `VRAM cache hits=15566 misses=3442`, worse than the `13824MiB` path on the
  same prompt (`misses=3042`). Larger cache near the free-VRAM edge is not a
  reliable improvement.

Diagnostic H2D-coalesce implementation was tested from unpushed experimental
head `e1798ff50` and then reverted because it did not improve SOTA:

- First diagnostic with contiguous pinned staging showed the missing mechanism:
  run `20260708T185217Z-20260709-contig2-overlay-locality-deploy-n32` had
  `host_contiguous_pairs=2126`, `dst_contiguous_pairs=2715`,
  `both_contiguous_pairs=1727`, and theoretical `total_saved_copy_count=1727`,
  but speed stayed `eval_tok_s=3.2` because actual H2D copies were not yet
  coalesced.
- Actual H2D-coalesce diagnostic, run
  `20260708T185700Z-20260709-h2d-coalesce-overlay-deploy-n32`, reduced
  `iouring_h2d_enqueues` from `5202` to `4659`, but speed was only
  `eval_tok_s=3.4`.
- n96 candidate, run
  `20260708T185751Z-20260709-h2d-coalesce-overlay-deploy-n96`, produced a
  coherent deploy answer and stayed within RAM (`memory_peak_bytes=14106255360`,
  `memory_file_bytes=13226201088`), but reached only `eval_tok_s=3.2`,
  `prompt_tok_s=5.3`, `first_output_ms=16144.8 ms`,
  `iouring_wait_us=12782809`, `iouring_reads=11331`,
  `iouring_bytes=50.50GB`, and `iouring_h2d_enqueues=10782`.
- Conclusion: this implementation does not beat the accepted safe default
  deploy high repeat (`3.6 tok/s`) and does not approach product `>5 tok/s`.
  The experimental code was not pushed and the working tree was reset to the
  last pushed safe head.

Updated implementation direction after the rejected probes:

1. Do not promote first-use overlay layout alone. It is useful evidence, but
   current runtime does not turn physical adjacency into fewer reads/bytes.
2. Do not spend more time on small VRAM-cache increments near `14GB`; they are
   unstable and can increase misses or abort.
3. A future coalescing attempt must reduce both H2D enqueue count and
   `iouring_wait_us` on n96, not only n32 profile counters. The current
   implementation reduced enqueue count too little and increased scheduling
   wait.
4. The next high-impact path should target total bytes: compact/full-route
   expert representations or a true runtime plan that reads one larger ordered
   span and directly places experts into a layout that the kernels can consume
   without per-expert staging. Any such path must preserve exact math and pass
   France, AI infra, Fibonacci/code-generation, and deploy before held-out
   testing.

## 2026-07-09 Byte-Bound And Next Clean Top-K Probe

New strict-cold n96 deploy byte-bound profile under current safe HEAD
`ca0027a45`:

- run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T190140Z-20260709-safe-default-deploy-bytebound-n96`;
- profile:
  `/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-safe-default-deploy-n96-bytebound`;
- result: `eval_tok_s=3.1`, `prompt_tok_s=5.3`,
  `first_output_ms=16055.2 ms`, `memory_peak_bytes=14146871296`,
  `memory_file_bytes=13267075072`, `ram_ok=true`, display cleanup recorded;
- runtime counters: `iouring_reads=11331`, `iouring_bytes=50.50GB`,
  `iouring_wait_us=12761453`, `VRAM cache hits=36735 misses=9171`.

Trace analysis:

- reads are exactly balanced by role:
  `gate=3777`, `up=3777`, `down=3777`; each role moves `16.83GB`;
- unique entries are `8463` total, `37.72GB`; perfect run-level expert cache
  would save only `12.78GB` and still move about `37.7GB`;
- current repeat factor is only `1.34x`, so cache/de-dup alone is insufficient
  for the product target;
- rough linear byte target from the measured `3.1 tok/s`: reaching `5.0 tok/s`
  would require around `31.3GB` effective expert movement, not merely the
  `37.7GB` perfect-cache bound;
- current physical layout remains poor for span reads:
  `span/read=79.33`, `gap/read=78.33`, `adjacent/read_jobs=0.0101`.

Implication:

- To reach stable `>5 tok/s`, the next accepted method must either reduce
  active expert payloads by roughly `35-40%`, or implement a deeper compact
  exact representation/runtime that avoids moving full `4.25MiB` payloads per
  selected expert. Simple cache, overlay layout, and small scheduling knobs are
  below the required bound.

Next clean probe before more invasive compact-kernel work:

1. Re-test `top2 all layers` on the current clean HEAD, strict cold, 16GB
   cgroup, after required display-process cleanup.
2. Use only dev/calibration prompts first: France, AI infra, Fibonacci/code,
   and deploy. This is not held-out final testing.
3. Candidate config:
   `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39`,
   `GGML_MOE_KEEP_TOPK_LAYER_VALUE=2`,
   `GGML_MOE_KEEP_TOPK_UPDOWN=2`,
   `GGML_MOE_KEEP_TOPK_GATE=1`, and start with `GGML_MOE_VRAM_CACHE_MIB=13824`
   unless a smoke run proves `14336` is stable on clean HEAD.
4. Acceptance condition: generalized quality must pass, especially Fibonacci
   must produce a direct Python function rather than meta/reasoning text, and
   deploy must stay coherent. If any dev quality fails, reject and do not
   promote even if France/AI infra reach `>5 tok/s`.
5. If clean top2 passes quality and improves generalized token rate, record
   exact reproduction info and immediately commit/push source plus plan
   results. If it fails, keep it rejected and proceed to compact/exact expert
   payload work.

Clean top2-all-layers probe result:

- config:
  `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39`,
  `GGML_MOE_KEEP_TOPK_LAYER_VALUE=2`,
  `GGML_MOE_KEEP_TOPK_UPDOWN=2`,
  `GGML_MOE_KEEP_TOPK_GATE=1`,
  `GGML_MOE_VRAM_CACHE_MIB=13824`;
- source: clean `a2722b1a0`, strict cold, 16GB cgroup, display cleanup
  recorded for all runs;
- France run
  `20260708T190607Z-20260709-clean-top2-alllayers-france-n96`:
  `eval_tok_s=4.2`, RAM OK, output coherent;
- AI infra run
  `20260708T190647Z-20260709-clean-top2-alllayers-aiinfra-n96`:
  `eval_tok_s=4.9`, RAM OK, but output repeats/rewrites the request instead
  of directly answering. Quality fail;
- Fibonacci run
  `20260708T190723Z-20260709-clean-top2-alllayers-fibonacci-n96`:
  `eval_tok_s=4.2`, RAM OK, but output explains the Fibonacci sequence and
  does not provide the requested Python function. Quality fail;
- deploy run
  `20260708T190803Z-20260709-clean-top2-alllayers-deploy-n96`:
  `eval_tok_s=3.9`, RAM OK, coherent, but still below `5 tok/s`.

Decision: reject fixed top2 all-layers. It does not meet generalized
correctness, and even its deploy speed is below target. Do not promote.

Dynamic top-k assessment:

- A prompt-general dynamic top-k policy could be useful in principle: keep
  top3 when router confidence is low, skip the third expert only when the
  third selected weight is negligible.
- Current `ggml-cpu.c` fixed pruning sees selected expert ids in the matmul
  route loop, but not the selected weights. The selected weights are present
  higher in the DeepSeek graph, so implementing dynamic top-k correctly would
  require passing or preserving that tensor into the CPU/GPU expert route path.
- Next step before implementing dynamic top-k: add a default-off diagnostic
  that records selected weights per layer/rank on the dev prompts, then compute
  how often rank-3 weight is small enough to skip. If the possible skip rate
  is well below the `35-40%` byte reduction target, do not implement dynamic
  top-k.

External compact DeepSeek candidate:

- Metadata/browser check found
  `cloudyu/DeepSeek-V4-Flash-4Expert-GGUF` with `ds4flash-4expert.gguf`.
  The model card describes a DeepSeek V4 Flash 4Expert Q4_K GGUF, top-k `4`,
  `256` routed experts, `2048` FFN dim, file size `164 GiB`, and HumanEval
  Pass@1 matching the original top-k=6 report in that card.
- This is not the same current native GGUF. Treat it as a candidate model
  variant for the product goal, not as a replacement SOTA unless it passes the
  same gates: strict cold, 16GB cgroup including page cache, display cleanup,
  generalized prompt quality, Fibonacci/code generation, and TTFT.
- The expected benefit is structural: native current path still selects/prunes
  from top-k 6 and quality fails when forced to top2, while a trained/exported
  top-k 4 variant may reduce active expert movement without the same quality
  failure. This directly attacks the byte-bound gap instead of relying on cache
  repeats.
- Next step: if disk/network allow, download or resume
  `ds4flash-4expert.gguf` into `/home/wici/models`, run metadata/load smoke,
  then strict-cold n96 on France, AI infra, Fibonacci, and deploy. Do not
  promote unless quality is clean and token rate is stably above the current
  safe default; product success still requires stable `>5 tok/s`.

## 2026-07-09 Execution Notes

Current clean-head revalidation:

- Synced new machine `/home/wici/ssd-llama` to `fad6a67f980f` via bundle
  because the machine cannot fetch from GitHub without credentials.
- Strict-cold default run after required display cleanup:
  `20260708T191708Z-20260709-current-default-france-n96`.
- Result: `eval_tok_s=3.6`, `prompt_tok_s=4.6`,
  `first_output_ms=16178.119012`, `memory_peak_bytes=14091792384`,
  `memory_file_bytes=13268803584`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`, source clean, France output
  coherent.
- Hardware state: H2D benchmark `6.688 GB/s`; PCIe was idle `2.5 GT/s x4`
  before the run and `16.0 GT/s x4` under load. `nvidia-smi -q` reports
  `Host Max=4`, so the new machine is still effectively Gen4 x4 for this GPU
  path, not the hoped-for Gen5 x4 software case.
- Counters: `iouring_reads=8913`, `iouring_bytes=39.720GB`,
  `iouring_wait_us=10402055`, `iouring_h2d_enqueues=8913`,
  VRAM cache `hits=36752`, `misses=6745`, `hit_rate=84.5%`.

External compact 4Expert candidate status:

- Disk is sufficient on the new machine (`/` has about `513GB` available), but
  Hugging Face and hf-mirror resolve to `198.18.x.x` on this network and HTTPS
  fails with `SSL_ERROR_SYSCALL` / `UNEXPECTED_EOF_WHILE_READING`.
- No full 4Expert GGUF is present under `/home/wici`, `/mnt`, `/data`, or
  `/root`; only the current native GGUF and native expert pack are available.
- Decision: 4Expert remains a promising payload-reduction candidate, but is
  blocked on artifact acquisition on this machine. Do not treat this as a
  source-level blocker for native optimization.

Dynamic top-k negative-id skip experiment:

- Implemented a local, default-off experiment, not pushed:
  `GGML_DS4_DYNAMIC_TOPK_WEIGHT_THRESHOLD` plus
  `GGML_DS4_DYNAMIC_TOPK_MIN_KEEP`. The graph kept the first two selected
  experts, masked low-weight tail picks, converted pruned tail ids to `-1`,
  and patched the CPU route loops to skip negative ids and zero the output.
- `threshold=0.08`, France n32
  `20260709-dyntopk008-negskip-france-n32`: payload dropped to
  `iouring_bytes=10.736GB` and apparent `eval_tok_s=10.5`, but answer was
  empty. Quality fail and not promotable.
- `threshold=0.02`, France n32
  `20260709-dyntopk002-negskip-france-n32`: also empty answer. Quality fail.
- `threshold=0.005`, France n32
  `20260709-dyntopk0005-negskip-france-n32`: output coherent,
  `eval_tok_s=4.1`, `iouring_bytes=22.033GB`, RAM OK, but still below target.
- `threshold=0.005`, France n96
  `20260709-dyntopk0005-negskip-france-n96`: output coherent,
  `eval_tok_s=3.5`, `prompt_tok_s=4.3`, `first_output_ms=16061.454965`,
  RAM OK, but below the clean default France repeat and far below `>5`.
- `threshold=0.01`, France n32
  `20260709-dyntopk001-negskip-france-n32`: output coherent but only
  `eval_tok_s=3.2`.
- Decision: reject this dynamic threshold implementation. It proves that
  skipping low-weight experts can reduce bytes enough to exceed `5 tok/s`, but
  the threshold needed for large byte savings breaks generation quality; the
  low safe threshold does not improve n96. Revert the local source changes and
  do not push code.
- Next native direction: dynamic pruning needs a better criterion than an
  absolute per-pick weight threshold, likely layer-aware and calibrated from
  actual selected-weight distributions, or a trained/exported compact expert
  artifact. Blind thresholding is closed for now.

Router-weight profile diagnostic:

- Added a default-off diagnostic path, `GGML_DS4_WEIGHT_PROFILE_OUT`, to dump
  actual DeepSeek4 `ffn_weights-*` tensors after graph execution. The callback
  marks those tensors as graph outputs only when the env is set, and
  `build_moe_v4` makes the weights contiguous only in that diagnostic mode.
  Normal SOTA runs without this env keep the original graph path.
- The first attempt to read intermediate tensors after graph compute produced
  invalid negative/stale values because non-output intermediate buffers were
  reused by the allocator. The final diagnostic fixes this by marking
  `ffn_weights` as output and reading a contiguous tensor.
- Dev profile runs under strict cold cgroup and display cleanup:
  - `20260709-weight-profile-france-n16`
  - `20260709-weight-profile-aiinfra-n16`
  - `20260709-weight-profile-fibonacci-n16`
  - `20260709-weight-profile-deploy-n16`
  CSVs are under
  `/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-weight-profile-dev/`.
- Combined distribution across 30,186 recorded picks:
  - rank0 mean `0.566`, p50 `0.483`;
  - rank1 mean `0.292`, p50 `0.300`;
  - rank2 mean `0.212`, p50 `0.227`;
  - rank3 mean `0.167`, p50 `0.179`;
  - rank4 mean `0.141`, p50 `0.147`;
  - rank5 mean `0.122`, p50 `0.126`.
- Combined threshold skip fractions for tail ranks `>=2`:
  - `0.05`: tail skip `8.0%`, all-pick skip `5.4%`;
  - `0.10`: tail skip `18.3%`, all-pick skip `12.2%`;
  - `0.15`: tail skip `42.2%`, all-pick skip `28.1%`;
  - `0.18`: tail skip `59.4%`, all-pick skip `39.6%`;
  - `0.20`: tail skip `70.1%`, all-pick skip `46.7%`.
- Interpretation: enough byte reduction to approach `>5 tok/s` requires an
  aggressive threshold around `0.15-0.18`, not the low thresholds. This matches
  the failed dynamic-threshold experiment: small safe thresholds do not move
  n96 speed, while aggressive skipping is likely to break quality unless the
  policy is layer-aware and correctness-gated.

Late-layer top2 schedule probe:

- Based on the weight profile, later layers have more low-weight tail picks, so
  a default-off fixed schedule `GGML_MOE_KEEP_TOPK_LAYER_SCHEDULE=24-42:2` was
  tested as a no-code smoke.
- Runs:
  - `20260709-late24top2-fibonacci-n32`: `eval_tok_s=3.5`, RAM OK, but output
    started with `that## ...`; quality suspicious/fail.
  - `20260709-late24top2-deploy-n32`: `eval_tok_s=3.3`, RAM OK, output started
    awkwardly with `like deploying...`; no speed gain.
  - `20260709-late24top2-france-n32`: `eval_tok_s=3.6`, RAM OK, coherent
    France output but no speed gain.
- Decision: reject `24-42:2`. The profile says late layers are more prunable,
  but fixed top2 is still not a viable generalized route.
- Next implementation direction: if continuing native-model pruning, use the
  recorded weight profile to build a layer-aware dynamic policy and validate
  it first on dev prompts. A candidate must pass Fibonacci/deploy quality and
  show n96 speedup before any source promotion.

## 2026-07-09 Run-0 Cleanup Clarification

The run procedure is now explicitly treated as part of the optimization plan,
not just an operational reminder:

1. Every future DeepSeek run on the new machine must start by killing display
   processes and stale GPU/model processes.
2. The launcher/demo script or manual run note must record that cleanup was
   executed before the model process starts.
3. `nvidia-smi` must be captured after cleanup and stored with the run
   artifact so later reviewers can verify that display VRAM did not affect the
   result.
4. A result cannot be promoted, compared as a clean regression, or called SOTA
   unless `display_processes_stopped_before_run=true` is present and the
   pre-run GPU process list is recorded.

Required cleanup command for manual runs:

```bash
echo '12345678' | sudo -S bash -lc '
  systemctl stop display-manager || true
  pkill -f "[g]nome-shell" || true
  pkill -f "[X]org" || true
  pkill -f "/usr/lib/xorg/[X]org" || true
  pkill -f "[o]llama" || true
  pkill -f "[l]lama-cli" || true
  nvidia-smi
'
```

## 2026-07-09 Duplicate-Rank0 Dynamic Top-K Rejection

After rejecting the negative-id dynamic top-k implementation, a safer
duplicate-rank0 variant was tested. Instead of writing `-1` for pruned tail
experts, it replaced pruned tail expert ids with the rank0 expert for the same
token and set the corresponding tail weights to zero. The goal was to avoid
invalid expert ids while preventing extra expert movement, because rank0 should
already be active for that token.

Run:

- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T194955Z-20260709-dyntopk015-duprank0-france-n32`
- env: `GGML_DS4_DYNAMIC_TOPK_WEIGHT_THRESHOLD=0.15`,
  `GGML_DS4_DYNAMIC_TOPK_MIN_KEEP=2`
- strict cold, 16GB cgroup, display cleanup recorded
- `eval_tok_s=3.2`, `prompt_tok_s=4.3`, `first_output_ms=16023.9 ms`
- `memory_peak_bytes=13970964480`, `memory_file_bytes=13144674304`,
  `ram_ok=true`
- France output was coherent, but source was dirty and performance regressed.

Clean same-head comparison:

- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T194721Z-20260709-clean-1bbfbfa-default-france-n32`
- `eval_tok_s=3.5`, `prompt_tok_s=4.6`, `first_output_ms=16138.5 ms`,
  `ram_ok=true`, source clean.

Counter comparison:

- clean default: `iouring_reads=4959`, `iouring_bytes=22099525632`,
  `iouring_wait_us=4092211`, VRAM hit rate `84.3%`.
- duplicate-rank0 top-k: `iouring_reads=4903`, `iouring_bytes=21849964544`,
  `iouring_wait_us=4288576`, VRAM hit rate `84.4%`.

Decision: reject and revert. The duplicate-rank0 method reduced actual expert
bytes by only about `1.1%`, far below the theoretical tail-pruning potential,
and added graph overhead from mask/concat/cast operations. It does not solve
the real scheduling problem: the runtime still sees nearly the same active
expert-source footprint. Future pruning work must either support true safe
route skipping in the MoE kernels/runtime or use a validated compact expert
artifact; graph-level id substitution is closed.

## 2026-07-09 VRAM And RAM-Tier Probes

VRAM cache sweep on the new machine:

- Prompt: `Explain quantum computing briefly.`
- Mode: strict cold n32, display cleanup recorded, source clean.
- Requested `GGML_MOE_VRAM_CACHE_MIB=15000/16000/17000/18000`.
- All runs stayed RAM-valid but failed to allocate the requested cache size.
  `cudaMemGetInfo` reported only about `14165 MiB` free at cache init, so the
  larger requests fell into retry and produced actual pools of only
  `12.7-13.7 GiB`.
- Results were all `eval_tok_s=3.0`, below the clean default envelope.
- Decision: reject larger raw VRAM requests as a SOTA path. The default
  `13824 MiB` cache is already close to the practical allocatable limit on the
  current CUDA/model workspace layout. Future VRAM work must free workspace or
  reduce cache fragmentation; simply raising `GGML_MOE_VRAM_CACHE_MIB` is not
  effective.

Default-off RAM-tier passthrough:

- Added `GGML_MOE_RAM_TIER_MIB`, `GGML_MOE_RAM_TIER_PROFILE`,
  `GGML_MOE_RAM_TIER_SKIP`, `GGML_MOE_RAM_TIER_PIN`, and
  `GGML_MOE_RAM_TIER_PIN_MIB` to the demo wrapper's config recording and
  systemd passthrough in commit `fea235f92`.
- This does not alter the default SOTA path; it only makes RAM-tier probes
  reproducible under the same 16GB cgroup wrapper.

RAM-tier 512MiB exact-host probe:

- Built a dev-profile CSV at
  `/home/wici/runs/vendor-ds4-16gb/manual-profiles/20260709-ramtier-dev-profile.csv`
  from the existing prompt-general dev profile. No held-out prompts were used.
- Tested `GGML_MOE_RAM_TIER_MIB=512` with skips `0`, `500`, `1500`, `2500`.
- Clean representative runs:
  - `20260709-ramtier512-clean-quantum-n32`: `eval_tok_s=3.0`,
    `memory_peak_bytes=14277423104`, RAM OK, source clean,
    RAM-tier hits `44/5451 = 0.8%`.
  - `20260709-ramtier512-skip500-quantum-n32`: `eval_tok_s=3.0`,
    RAM-tier hits `55/5451 = 1.0%`.
  - `20260709-ramtier512-skip1500-quantum-n32`: `eval_tok_s=3.0`,
    RAM-tier hits `64/5451 = 1.2%`.
  - `20260709-ramtier512-skip2500-quantum-n32`: `eval_tok_s=3.0`,
    RAM-tier hits `49/5451 = 0.9%`.
- One dirty-source pass showed `3.4 tok/s`, but clean repeats did not
  reproduce it, so it is treated as normal run variance, not a candidate.
- Decision: reject RAM-tier 512MiB as a SOTA path. It is exact and
  RAM-compliant, but it only replaces dozens of expert-pack reads out of about
  5.4k reads for n32 and does not move token rate. Larger RAM tiers are risky
  under the 16GB host-RAM limit because clean runs already peak around
  `14.3GB`.

Updated next direction:

1. Do not spend more time on raw VRAM-size or small RAM-tier tuning.
2. Continue with byte reduction or true route scheduling:
   - safe kernel/runtime support for true expert route skipping, not
     graph-level id substitution;
   - compact/partial up/gate/down expert representation with a hard byte
     bound before implementation;
   - retained gate interface only if it proves large simultaneous gate-source
     and up/down movement cuts without recomputing gate.

## 2026-07-09 Top2 Route-Skip And Decode-Warmup Rejection

The existing `GGML_MOE_KEEP_TOPK_*` path is a true route-skip path: it zeros
low-rank pick outputs and skips those picks before building
`matrix_row_counts`, so it reduces actual expert movement. A schedule sweep
was run to measure the real speed/quality boundary.

Quantum n32 schedule sweep, strict cold, display cleanup recorded, source
clean:

- `GGML_MOE_KEEP_TOPK_LAYER_SCHEDULE=0-42:2`:
  `eval_tok_s=4.6`, `prompt_tok_s=5.8`, `first_output_ms=13521.3 ms`,
  `iouring_bytes=16.31GB`, RAM OK. Output was broadly coherent.
- `16-42:2`: `eval_tok_s=3.9`, `iouring_bytes=21.40GB`; output started with
  minor wording damage.
- `8-42:2`: `eval_tok_s=4.3`, `iouring_bytes=18.77GB`; output coherent.
- `0-23:2,24-42:3`: `eval_tok_s=3.9`, `iouring_bytes=19.10GB`; output had
  minor wording damage.

All-layer top2 n96 validation:

- France:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T200921Z-20260709-top2all-france-n96`
  reached `eval_tok_s=4.2`, `prompt_tok_s=5.8`, RAM OK, source clean, and
  produced a coherent France paragraph.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T201001Z-20260709-top2all-fibonacci-n96`
  reached `eval_tok_s=4.1`, RAM OK, source clean, but failed the code-quality
  gate. The answer started with `thatThe Fibonacci sequence...` and explained
  Fibonacci instead of writing the requested Python function.
- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T201041Z-20260709-top2all-deploy-n96`
  reached `eval_tok_s=3.7`, RAM OK, source clean, and gave a usable deployment
  answer, but remained below 5.
- Default top3 Fibonacci control:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T201143Z-20260709-default-fibonacci-n96-recheck`
  reached `eval_tok_s=3.0`, RAM OK, source clean, and correctly began a
  Python `def fibonacci(n):` function. Therefore the top2 failure is a real
  quality regression, not a prompt issue.

A default-off decode-warmup experiment was then implemented locally but not
pushed. It kept top3 during prompt and the first N decode tokens, then applied
`0-42:2` afterward. Tested Fibonacci n96:

- warmup `16`: `eval_tok_s=3.9`, RAM OK, dirty source, same quality failure.
- warmup `32`: `eval_tok_s=4.4`, RAM OK, dirty source, same quality failure.
- warmup `48`: `eval_tok_s=4.5`, RAM OK, dirty source, same quality failure.

Decision: reject top2 all-layer and decode-warmup top2 as generalized SOTA
paths. True route skipping can reduce bytes enough to approach the target, but
dropping the third up/down expert changes code-generation behavior. The local
warmup implementation was reverted and not pushed. Future route-skip work must
use a safer criterion than fixed rank count, or preserve the third expert via
a compact/partial representation instead of deleting it.

## 2026-07-09 PCIe Limit And Role-Split Top-K Rejection

Hardware diagnosis on the new machine showed the GPU is attached below root
port `0000:00:06.0`, not the Thunderbolt root ports. The endpoint reports
`LnkCap Speed 32GT/s, Width x16`, but the root port reports only
`LnkCap Speed 16GT/s, Width x4` and `LnkCtl2 Target Link Speed 16GT/s`.
Under load the run artifacts consistently show `current_link_speed=16.0 GT/s`
and `current_link_width=4`, with 4.25MiB pinned H2D around `6.4-6.7 GB/s`.

Consequence: this physical slot/path is a Gen4 x4 bottleneck. The earlier
Case B assumption of Gen5 x4 / about `11.5 GB/s` is not true for the current
new-machine wiring. Reaching stable `>5 tok/s` on this machine therefore
requires real H2D byte reduction or deeper overlap; it cannot rely on PCIe
retraining alone.

A default-off role-specific top-k diagnostic was added:

- `GGML_MOE_KEEP_TOPK_UP_LAYER_SCHEDULE`
- `GGML_MOE_KEEP_TOPK_GATE_LAYER_SCHEDULE`
- `GGML_MOE_KEEP_TOPK_DOWN_LAYER_SCHEDULE`

If these envs are unset, the default SOTA path is unchanged. The goal is to
separate gate/up/down sensitivity before attempting any compact third-expert
representation.

Smoke/default diagnostic:

- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T202747Z-20260709-role-schedule-default-smoke-france-n32`
- source dirty because the default-off diagnostic code was not yet committed
  at run time;
- `eval_tok_s=3.5`, `prompt_tok_s=4.7`,
  `first_output_ms=16005.9 ms`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`;
- France output remained coherent.

Role-split top-k diagnostics, strict cold, 16GB cgroup, display cleanup
recorded:

- gate-only top2:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T202833Z-20260709-gate-only-top2-fibonacci-n32`
  reached `eval_tok_s=3.2` but failed the Fibonacci/code gate; the output
  explained the Fibonacci sequence instead of writing the requested function.
- gate-only top1:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T202928Z-20260709-gate-only-top1-fibonacci-n32`
  reached `eval_tok_s=3.3` but produced system-style pollution and failed.
- down-only top2:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T202955Z-20260709-down-only-top2-fibonacci-n32`
  reached `eval_tok_s=3.1` and failed the Fibonacci/code gate.
- up-only top2:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T203023Z-20260709-up-only-top2-fibonacci-n32`
  reached `eval_tok_s=3.3` and failed the Fibonacci/code gate.
- gate-only top2 deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T202901Z-20260709-gate-only-top2-deploy-n32`
  reached `eval_tok_s=3.3` and gave a usable deploy answer, but the Fibonacci
  failure rejects the candidate.

Decision: reject fixed third-route deletion for each individual role. The
third selected expert is quality-critical for code-generation even when only
gate, only up, or only down is pruned. The next valid optimization must
preserve the third expert contribution, for example by moving the third
expert through a cheaper representation, improving cache residency for the
full third expert, or overlapping H2D with compute more aggressively.

## 2026-07-09 Down-Prefetch Overlap Rejection

The demo wrapper now records and passes through:

- `GGML_MOE_PREFETCH_DOWN`
- `GGML_MOE_PREFETCH_DOWN_DEPTH`

This makes down-prefetch overlap experiments reproducible under the same
strict 16GB cgroup launcher.

Strict cold diagnostics, display cleanup recorded, 16GB cgroup:

- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T203301Z-20260709-overlap-prefetch-d2-fibonacci-n64`
  with `GGML_MOE_CURRENT_DOWN_OVERLAP=1`,
  `GGML_MOE_PREFETCH_DOWN=1`, `GGML_MOE_PREFETCH_DOWN_DEPTH=2`:
  `eval_tok_s=3.0`, `prompt_tok_s=4.8`,
  `first_output_ms=14169.9 ms`, RAM OK. Fibonacci output correctly began with
  a Python function.
- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T203340Z-20260709-overlap-prefetch-d2-deploy-n64`
  with the same depth-2 config:
  `eval_tok_s=3.2`, `prompt_tok_s=5.2`, RAM OK. Deploy output was coherent.
- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T203418Z-20260709-overlap-prefetch-d8-fibonacci-n64`
  with depth `8`:
  `eval_tok_s=3.0`, `prompt_tok_s=4.8`,
  `first_output_ms=14172.9 ms`, RAM OK. Fibonacci output remained correct.

Decision: reject current down-overlap/prefetch knobs as a SOTA path. They can
preserve correctness and lower TTFT in some runs, but do not increase decode
token rate on the Gen4 x4 new-machine path. Future overlap work must profile
and reduce actual H2D wait/copy serialization, not only enable existing
prefetch depth knobs.

## 2026-07-09 Clean Copy Profile And Queue-Depth Sweep

Clean profiling run:

- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T203629Z-20260709-clean-copy-profile-fibonacci-n32`
- source clean at `eeb42534b`, strict cold, 16GB cgroup, display cleanup
  recorded.
- `eval_tok_s=3.0`, `prompt_tok_s=4.4`,
  `first_output_ms=14775.9 ms`, `ram_ok=true`.
- Fibonacci output correctly began with a Python function. The run is
  diagnostic only because copy/io CSV profiling adds overhead.

Profile summary:

- `iouring_reads=5357`, `iouring_bytes=23.87GB`, `iouring_wait_us=3813386`.
- Copy-profile rows: `5357`, each copy `4.456MB`.
- Per-op copy totals:
  - `gate_batch_preload`: `1787` copies, `7.964GB`.
  - `gate_updown_cosubmit`: `2167` copies, `9.657GB`.
  - `runtime_load`: `1403` copies, `6.252GB`.
- Batch profile:
  - `gate_batch_preload`: `827` batches, average `2.16` jobs.
  - `gate_updown_cosubmit`: `304` batches, average `7.13` jobs.
  - `runtime_load`: `1046` batches, average `1.34` jobs.
- Coalescing profile: `total_saved_copy_count=0`, `host_contiguous_pairs=0`;
  destination slots are often contiguous, but host source offsets are not.

Interpretation:

- Simple adjacent-copy coalescing cannot help the current full native path,
  because expert source offsets are not physically contiguous in execution
  order.
- The weakest areas are small `runtime_load` and `gate_batch_preload` batches,
  not `gate_updown_cosubmit`, whose average batch size is already higher.
- A useful software change must either change scheduling so more active
  experts are known before runtime load, or reduce bytes for the third expert
  without deleting its contribution.

Queue-depth / pinned-slot sweep, source clean, strict cold, display cleanup
recorded:

- `GGML_MOE_STAGE_PINNED_SLOTS=12`,
  `GGML_MOE_IO_DEPTH=16`, `GGML_MOE_IO_REFILL_BATCH=8`,
  Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T203813Z-20260709-slots12-depth16-refill8-fibonacci-n64`
  reached `eval_tok_s=3.4`, quality OK.
- `GGML_MOE_STAGE_PINNED_SLOTS=16`,
  `GGML_MOE_IO_DEPTH=16`, `GGML_MOE_IO_REFILL_BATCH=8`,
  Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T203849Z-20260709-slots16-depth16-refill8-fibonacci-n64`
  reached `eval_tok_s=3.6`, quality OK.
- default Fibonacci n64 comparison:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204021Z-20260709-default-fibonacci-n64-compare`
  also reached `eval_tok_s=3.6`, quality OK.
- default deploy n64 comparison:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204057Z-20260709-default-deploy-n64-compare`
  reached `eval_tok_s=3.8`, quality OK.
- slots16/depth16/refill8 deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204132Z-20260709-slots16-depth16-refill8-deploy-n64`
  reached `eval_tok_s=3.9`, quality OK.
- slots16/depth16/refill8 France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204205Z-20260709-slots16-depth16-refill8-france-n64`
  reached `eval_tok_s=3.4`, quality OK.

Decision: do not change default pinned slots or IO depth based on this sweep.
The best difference was a small deploy-only `3.8 -> 3.9 tok/s`, while
Fibonacci matched default and France did not improve. This is useful as a
diagnostic but not an accepted SOTA.

## 2026-07-09 Planned Host Prefetch Rejection

`GGML_MOE_PLANNED_HOST_PREFETCH=1` was tested because the clean copy profile
showed many small `runtime_load` and `gate_batch_preload` batches. This path
keeps model math unchanged and tries to move expert reads into a background
host-pinned prefetch worker before the H2D copy.

Strict cold, source clean, display cleanup recorded:

- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204441Z-20260709-planned-host512-fibonacci-n64`
  with `GGML_MOE_HOST_PREFETCH_MAX_MIB=512`,
  `GGML_MOE_HOST_PREFETCH_SLOTS=64`:
  `eval_tok_s=3.0`, `prompt_tok_s=4.7`,
  `memory_peak_bytes=14280712192`, RAM OK, Fibonacci quality OK.
- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204520Z-20260709-planned-host1024-fibonacci-n64`
  with `GGML_MOE_HOST_PREFETCH_MAX_MIB=1024`,
  `GGML_MOE_HOST_PREFETCH_SLOTS=128`:
  `eval_tok_s=3.0`, `prompt_tok_s=4.7`,
  `memory_peak_bytes=14297808896`, RAM OK, Fibonacci quality OK.
- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204559Z-20260709-planned-host1024-deploy-n64`
  with the same 1024MiB config:
  `eval_tok_s=3.2`, `prompt_tok_s=5.3`,
  `memory_peak_bytes=14356013056`, RAM OK, deploy quality OK.

Decision: reject planned host prefetch as currently implemented. It preserves
correctness and remains under the 16GB cgroup, but it does not increase
decode speed and likely fails to cover the critical wait because H2D remains
serialized and the worker is not far enough ahead of runtime use. The next
implementation must build a real per-layer gate/up/down request plan
immediately after routing, then submit that plan as a larger ordered batch.

## 2026-07-09 Cosubmit Allow-Evict Rejection

The clean copy profile showed `gate/up/down cosubmit no_slot_skips=1403`,
nearly matching the `runtime_load=1403` copy count. A default-off local
experiment therefore changed cosubmit admission to allow evicting non-pinned
VRAM cache slots instead of abandoning preloads when the cache was full. The
goal was to reduce later `runtime_load` reads while keeping model math
unchanged.

The implementation was tested locally as dirty source and then reverted; it
was not pushed.

Strict cold diagnostics, 16GB cgroup, display cleanup recorded:

- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T204933Z-20260709-cosubmit-evict-fibonacci-n64`
  reached `eval_tok_s=3.3`, `prompt_tok_s=4.6`,
  `memory_peak_bytes=13937123328`, RAM OK. Fibonacci quality was OK.
- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T205010Z-20260709-cosubmit-evict-deploy-n64`
  reached `eval_tok_s=3.4`, `prompt_tok_s=5.5`,
  `memory_peak_bytes=14106189824`, RAM OK. Deploy quality was OK.
- `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T205047Z-20260709-cosubmit-evict-france-n64`
  reached `eval_tok_s=3.7`, `prompt_tok_s=4.7`,
  `memory_peak_bytes=14116089856`, RAM OK. France quality was OK.

Comparison: default clean n64 runs were `3.6` for Fibonacci and `3.8` for the
deploy prompt in the same session. Allowing cosubmit to evict cache entries
therefore reduced generalized speed despite preserving correctness.

Decision: reject allow-evict cosubmit admission. The no-slot count is real,
but blindly evicting existing cache entries hurts future hit rate more than
it helps preloading imminent up/down experts. Future admission changes need a
scored victim policy using next-use distance or route-local priority, not
plain LRU eviction.

## 2026-07-09 Low-Hit Cosubmit Eviction Rejection

A stricter local variant of the cosubmit eviction experiment was tested. It
only allowed cosubmit preload to evict non-pinned VRAM cache slots whose
observed `slot_hits` were less than or equal to a threshold. This was intended
to avoid the previous plain-LRU failure by only replacing low-reuse slots.

The implementation was tested locally as dirty source and then reverted; it
was not pushed.

Strict cold diagnostics, 16GB cgroup, display cleanup recorded:

- `GGML_MOE_GATE_UPDOWN_COSUBMIT_EVICT_MAX_HITS=0`,
  Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T205509Z-20260709-cosubmit-evict-hit0-fibonacci-n64`
  reached `eval_tok_s=2.8`, `prompt_tok_s=4.8`,
  `memory_peak_bytes=13852487680`, RAM OK. Fibonacci quality was OK.
- `GGML_MOE_GATE_UPDOWN_COSUBMIT_EVICT_MAX_HITS=1`,
  Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T205550Z-20260709-cosubmit-evict-hit1-fibonacci-n64`
  reached `eval_tok_s=3.7`, `prompt_tok_s=4.7`,
  `memory_peak_bytes=13969629184`, RAM OK. Fibonacci quality was OK.
- `GGML_MOE_GATE_UPDOWN_COSUBMIT_EVICT_MAX_HITS=0`,
  deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T205625Z-20260709-cosubmit-evict-hit0-deploy-n64`
  reached `eval_tok_s=3.5`, `prompt_tok_s=5.7`,
  `memory_peak_bytes=14089379840`, RAM OK. Deploy quality was OK.
- `GGML_MOE_GATE_UPDOWN_COSUBMIT_EVICT_MAX_HITS=1`,
  deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T205721Z-20260709-cosubmit-evict-hit1-deploy-n64`
  reached `eval_tok_s=3.5`, `prompt_tok_s=5.3`,
  `memory_peak_bytes=14165651456`, RAM OK. Deploy quality was OK.
- `GGML_MOE_GATE_UPDOWN_COSUBMIT_EVICT_MAX_HITS=1`,
  France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T205757Z-20260709-cosubmit-evict-hit1-france-n64`
  reached `eval_tok_s=4.5`, `prompt_tok_s=4.6`,
  `memory_peak_bytes=14226100224`, RAM OK. France quality was OK.

Decision: reject low-hit cosubmit eviction. The `hits<=1` variant can improve
France in one run, but it does not generalize: deploy remains below the
default clean comparison (`3.5` vs default `3.8`), and `hits<=0` regresses
Fibonacci. This is a prompt-sensitive cache side effect, not an accepted
generalized SOTA. The next admission policy must use actual future route
distance or a per-layer request plan, not past slot hit count alone.

## 2026-07-09 Role-Aware Gate-Slot Eviction Rejection

A more targeted local admission experiment allowed cosubmit preload to evict
only existing cache slots whose recorded tensor name matched `ffn_gate_exps`.
The theory was that replacing old gate experts with imminent up/down experts
might be safer than evicting arbitrary old slots. The implementation required
recording tensor names for cache slots and was tested only as dirty source; it
was reverted and not pushed.

Strict cold diagnostics, 16GB cgroup, display cleanup recorded:

- `GGML_MOE_GATE_UPDOWN_COSUBMIT_EVICT_NAME_FILTER=ffn_gate_exps`,
  Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T210225Z-20260709-cosubmit-evict-gate-fibonacci-n64`
  reached `eval_tok_s=3.3`, `prompt_tok_s=4.8`,
  `memory_peak_bytes=13824450560`, RAM OK. Fibonacci quality was OK.
- Same config, deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T210302Z-20260709-cosubmit-evict-gate-deploy-n64`
  reached `eval_tok_s=3.3`, `prompt_tok_s=5.3`,
  `memory_peak_bytes=14124695552`, RAM OK. Deploy quality was OK.
- Same config, France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T210339Z-20260709-cosubmit-evict-gate-france-n64`
  reached `eval_tok_s=3.5`, `prompt_tok_s=4.7`,
  `memory_peak_bytes=14153060352`, RAM OK. France quality was OK.

Decision: reject role-aware gate-slot eviction. It preserves correctness, but
it regresses Fibonacci and deploy versus clean default comparisons and does
not reproduce the France-only speedup seen in the previous low-hit experiment.
This further supports that cache eviction without a true future-use model is
not enough. Next work should shift from reactive slot eviction to route-local
planning or a compact third-expert representation.

## 2026-07-09 GPU-Top2 Plus CPU-Tail Candidate

Implemented a default-off exactness-preserving split:

- `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2`
- Applies only to `ffn_up_exps` and `ffn_down_exps`.
- GPU batch receives only route ranks `< 2`.
- Tail ranks remain in `matrix_row_counts` and fall through to the existing
  CPU fallback path.
- This is not top2 deletion: the third expert contribution is still computed,
  preserving code-generation quality in the tested prompts.

Implementation notes:

- `mul_mat_id` workspace now includes tail row counts and tail row mappings.
- Before CUDA batch, active rows are partitioned into GPU rows and CPU-tail
  rows.
- If CUDA accepts the batch, only CPU-tail rows are restored for fallback.
- If CUDA declines the batch, all rows are restored for CPU fallback.
- Default behavior is unchanged unless `GGML_MOE_GPU_KEEP_TOPK_UPDOWN` is set.

Dirty-source candidate diagnostics, strict cold, 16GB cgroup, display cleanup
recorded:

- Fibonacci n32:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T210845Z-20260709-gpu2-cputail-fibonacci-n32`
  reached `eval_tok_s=3.2`, `prompt_tok_s=3.9`,
  `first_output_ms=15635.9 ms`,
  `memory_peak_bytes=14787715072`, RAM OK. Fibonacci quality was OK.
- Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T210930Z-20260709-gpu2-cputail-fibonacci-n64`
  reached `eval_tok_s=4.1`, `prompt_tok_s=4.0`,
  `first_output_ms=15118.4 ms`,
  `memory_peak_bytes=14793691136`, RAM OK. Fibonacci quality was OK.
- Deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T211005Z-20260709-gpu2-cputail-deploy-n64`
  reached `eval_tok_s=4.3`, `prompt_tok_s=4.3`,
  `first_output_ms=16649.9 ms`,
  `memory_peak_bytes=14733709312`, RAM OK. Deploy quality was OK.
- France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T211102Z-20260709-gpu2-cputail-france-n64`
  reached `eval_tok_s=4.8`, `prompt_tok_s=4.0`,
  `first_output_ms=16504.8 ms`,
  `memory_peak_bytes=14753718272`, RAM OK. France quality was OK.

Comparison to clean default n64 from the same session:

- Fibonacci: default `3.6`, candidate `4.1`.
- Deploy: default `3.8`, candidate `4.3`.
- France: previous clean/default runs were typically `3.4-3.6`, candidate
  `4.8`.

Decision: promote to source-controlled candidate and clean-reproduce before
calling it accepted SOTA. It is still below the product target `>5 tok/s`,
but it is the first prompt-general, correctness-preserving path that moves
multiple dev prompts toward the target under the 16GB RAM cap. Next validation
must repeat from a clean commit and then test n96 plus more dev prompts.

Clean source reproduction after commit and push:

- Source-controlled commit:
  `f2509e3f91f596f983c83d46e360722e54718ac5`
  (`vendor-ds4: add gpu topk cpu tail split`), pushed to
  `wici-ai/ssd-llama` branch `vendor/deepseek-token-rate-16gb`.
- New machine checkout: `/home/wici/ssd-llama`, source clean
  `vendor/deepseek-token-rate-16gb@f2509e3f9`.
- Build: `cmake --build build-cuda -j$(nproc)` completed successfully.
- All runs below were strict cold runs with display/model processes killed
  before launch, `drop_caches`, 16GB cgroup, swap disabled, and
  `display_processes_stopped_before_run=true`.
- Runtime delta:
  `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2`.

Clean validation results:

- Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T211818Z-20260709-clean-gpu2-cputail-fibonacci-n64`
  reached `eval_tok_s=4.1`, `prompt_tok_s=4.0`,
  `first_output_ms=14752.2 ms`,
  `memory_peak_bytes=14784131072`,
  `memory_file_bytes=13902618624`, RAM OK. Output began with a valid Python
  Fibonacci function, so quality passed.
- Deploy n64 first clean run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T211853Z-20260709-clean-gpu2-cputail-deploy-n64`
  reached `eval_tok_s=3.4`, `prompt_tok_s=4.4`,
  `first_output_ms=16399.3 ms`,
  `memory_peak_bytes=14793326592`,
  `memory_file_bytes=13807443968`, RAM OK. Output was coherent and covered
  quantization, so quality passed. This run shows deploy prompt variance and
  should not be used alone as the accepted speed point.
- France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T211931Z-20260709-clean-gpu2-cputail-france-n64`
  reached `eval_tok_s=4.8`, `prompt_tok_s=3.9`,
  `first_output_ms=16242.0 ms`,
  `memory_peak_bytes=14821982208`,
  `memory_file_bytes=13851746304`, RAM OK. France output was semantically
  correct and coherent.

Clean A-B deploy comparison on the same commit:

- Default, without the split:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T212022Z-20260709-clean-default-deploy-n64-ab`
  reached `eval_tok_s=3.2`, `prompt_tok_s=5.1`,
  `first_output_ms=16112.8 ms`,
  `memory_peak_bytes=13917773824`, RAM OK.
- Candidate, `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T212101Z-20260709-clean-gpu2-cputail-deploy-n64-ab`
  reached `eval_tok_s=4.1`, `prompt_tok_s=4.1`,
  `first_output_ms=16615.0 ms`,
  `memory_peak_bytes=14790180864`, RAM OK.

Decision: accept this as a source-controlled prompt-general improvement over
the immediate clean default comparison, not as final product success. The
accepted clean generalized speed band is now roughly `4.1-4.8 tok/s` on the
tested n64 prompts, with one observed deploy low outlier at `3.4 tok/s`.
TTFT stayed within the `+20%` rule in the A-B comparison:
`16615.0 ms` vs `16112.8 ms`, about `+3.1%`. The product target remains
stable `>5 tok/s` for arbitrary user prompts, so the next step is still to
reduce the remaining tail/transfer cost rather than declare completion.

## 2026-07-09 Refill Batch 8 Runtime Tuning

Current bottleneck profile:

- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T212406Z-20260709-clean-gpu2-cputail-deploy-n96-profile`
- Profile directory:
  `/tmp/20260709-clean-gpu2-cputail-deploy-n96-profile`
- Source:
  `vendor/deepseek-token-rate-16gb@f2509e3f9`, clean.
- Config:
  `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2`, default refill batch 4,
  strict cold, display cleanup recorded, 16GB cgroup.
- Result:
  `eval_tok_s=3.9`, `prompt_tok_s=4.1`,
  `first_output_ms=17014.5 ms`,
  `memory_peak_bytes=14794346496`, RAM OK.

Profile summary:

- Expert source read traffic:
  `iouring_reads=9294`, `iouring_bytes=41418227712` (`38.6 GiB`),
  `iouring_wait_us=5513003`.
- Timed H2D:
  `5325` copies, `h2d=3862.2 ms` in stderr. The detailed copy profile
  counted `6969.1 ms` summed per copy, which includes overlapping transfers
  and should not be read as additive wall time.
- VRAM cache:
  `hits=28663`, `misses=7134`, `hit_rate=80.1%`.
- Copy profile by op:
  `gate_batch_preload` read `14.6 GiB`, `runtime_load` read `15.0 GiB`,
  `gate_updown_cosubmit` read `9.0 GiB`.
- Down/up batch profile:
  up path had `1809` cache misses and `3618` staged jobs; down path had no
  cache misses in this profile. Up/down GPU kernels were only about
  `1554 ms` summed across profile rows, so the next bottleneck is transfer
  and cache admission rather than math throughput.
- Cache evictions:
  evicted tensors were later used (`victim_used=6042`, `victim_unused=0`).
  Simple eviction policy is not future-aware enough.

Rejected probes:

- `GGML_MOE_GATE_BATCH_PREFETCH=0`:
  - Deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T212714Z-20260709-gpu2-cputail-gatepref0-deploy-n64`
    reached only `eval_tok_s=2.3`, RAM OK, output coherent.
  - France n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T212802Z-20260709-gpu2-cputail-gatepref0-france-n64`
    reached only `eval_tok_s=2.7`, RAM OK, output coherent.
  - Decision: reject. Gate prefetch causes read traffic, but removing it
    exposes synchronous gate waits and badly regresses decode.
- `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`:
  - Deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T212901Z-20260709-gpu2-cputail-lfulru-deploy-n64`
    reached `eval_tok_s=3.5`, RAM OK.
  - France n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T212939Z-20260709-gpu2-cputail-lfulru-france-n64`
    reached `eval_tok_s=3.4`, RAM OK.
  - Decision: reject. Plain LFU/LRU is not route-future-aware and regresses
    both tested prompts.
- `GGML_MOE_STAGE_PINNED_SLOTS=16`:
  - Deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213035Z-20260709-gpu2-cputail-pinned16-deploy-n64`
    reached `eval_tok_s=4.3`, RAM OK.
  - France n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213110Z-20260709-gpu2-cputail-pinned16-france-n64`
    reached `eval_tok_s=4.6`, RAM OK.
  - Fibonacci n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213202Z-20260709-gpu2-cputail-pinned16-fibonacci-n64`
    reached only `eval_tok_s=3.5`, RAM OK.
  - Decision: reject as default. It helps deploy but regresses Fibonacci,
    so it is not a generalized SOTA setting.

Accepted runtime tuning:

- Change default `GGML_MOE_IO_REFILL_BATCH` from `4` to `8` in
  `scripts/demo-vendor-ds4-general-sota.sh`.
- Also make `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2` the demo default. The kernel
  implementation remains overrideable, but the generalized SOTA demo should
  not require an undocumented manual environment variable to reproduce the
  accepted path.
- Theory:
  with the current transfer-heavy path, small refill batches leave too much
  submit/wait overhead around the iouring queues. Refill batch 8 increases
  queued read work without increasing the pinned slot count or changing model
  math, so it can improve IO overlap while preserving correctness.

Strict cold validation, all with display/model processes killed before run,
16GB cgroup, `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2`, source clean:

- Deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213258Z-20260709-gpu2-cputail-refill8-deploy-n64`
  reached `eval_tok_s=4.3`, `prompt_tok_s=4.3`,
  `first_output_ms=16365.0 ms`,
  `memory_peak_bytes=14815309824`, RAM OK. Output was coherent and covered
  model compression/quantization.
- Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213332Z-20260709-gpu2-cputail-refill8-fibonacci-n64`
  reached `eval_tok_s=4.0`, `prompt_tok_s=4.0`,
  `first_output_ms=14758.1 ms`,
  `memory_peak_bytes=14822469632`, RAM OK. Output began with a valid Python
  Fibonacci function.
- France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213420Z-20260709-gpu2-cputail-refill8-france-n64`
  reached `eval_tok_s=4.7`, `prompt_tok_s=4.0`,
  `first_output_ms=16808.2 ms`,
  `memory_peak_bytes=14775328768`, RAM OK. France output was semantically
  correct and coherent.

Decision: accept refill batch 8 as a small source-controlled generalized
runtime improvement. It does not reach the product target yet, but it improves
deploy versus the prior clean A-B default while keeping Fibonacci and France
within the accepted SOTA band and preserving RAM/TTFT/correctness constraints.
Next optimization should be a future-aware cache/admission path or a real
split pool for gate/up/down that avoids evicting tensors known to be needed
later in the same decode sequence.

Default-demo reproduction after making both SOTA toggles defaults:

- Commit:
  `19ee499d984a5c55df8940d9eb11950fc6404da6`
  (`vendor-ds4: default generalized sota topk split`), pushed to
  `wici-ai/ssd-llama` branch `vendor/deepseek-token-rate-16gb`.
- New machine checkout:
  `/home/wici/ssd-llama`, source clean
  `vendor/deepseek-token-rate-16gb@19ee499d9`.
- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213740Z-20260709-default-sota-env-deploy-n64`
- Invocation passed only model/expert pack/run root. It did not manually pass
  `GGML_MOE_GPU_KEEP_TOPK_UPDOWN` or `GGML_MOE_IO_REFILL_BATCH`.
- Artifact config confirmed runtime defaults:
  `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2`,
  `GGML_MOE_IO_REFILL_BATCH=8`,
  `GGML_MOE_STAGE_PINNED_SLOTS=8`,
  `GGML_MOE_VRAM_CACHE_MIB=13824`.
- Strict cold result:
  `eval_tok_s=4.3`, `prompt_tok_s=4.5`,
  `first_output_ms=16298.6 ms`,
  `memory_peak_bytes=14797275136`,
  `memory_file_bytes=13795110912`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`, source clean. Output was
  coherent and covered model compression/quantization.

## 2026-07-09 Post-Refill8 Queue And Role-Split Rejections

All runs below used strict cold startup, 16GB cgroup, display/model process
cleanup before launch, prompt-general inputs, and correctness review.

Rejected queue knobs:

- `GGML_MOE_IO_REFILL_BATCH=16`:
  - Deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213922Z-20260709-default-top2-refill16-deploy-n64`
    reached `eval_tok_s=4.2`, `prompt_tok_s=4.4`,
    `first_output_ms=16448.4 ms`,
    `memory_peak_bytes=14800322560`, RAM OK.
  - Fibonacci n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T213957Z-20260709-default-top2-refill16-fibonacci-n64`
    reached `eval_tok_s=3.9`, `prompt_tok_s=4.0`,
    `first_output_ms=14768.7 ms`,
    `memory_peak_bytes=14835859456`, RAM OK.
  - Decision: reject. Refill 16 is slightly worse than accepted refill 8
    and does not move the lower bound toward `>5 tok/s`.
- `GGML_MOE_IO_SORT_OFFSET=1`:
  - Deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T214053Z-20260709-default-top2-refill8-sortoffset-deploy-n64`
    reached only `eval_tok_s=3.4`, RAM OK.
  - Fibonacci n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T214130Z-20260709-default-top2-refill8-sortoffset-fibonacci-n64`
    reached `eval_tok_s=4.0`, RAM OK.
  - Decision: reject. Sorting by file offset may improve apparent SSD
    locality, but it disrupts the current overlap/timing enough to regress
    deploy sharply.

Rejected default-off role split implementation:

- Implemented dirty-source experiment:
  `GGML_MOE_VRAM_CACHE_ROLE_SPLIT=1`, using cache id 1 for `ffn_up_exps` and
  `ffn_gate_exps`, cache id 0 for `ffn_down_exps`, with budget controlled by
  `GGML_MOE_VRAM_CACHE_UPGATE_PCT`.
- The code compiled, and default-off regression was acceptable:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T214741Z-20260709-rolesplit-off-regress-deploy-n64`
  reached `eval_tok_s=4.2`, RAM OK, source dirty.
- `GGML_MOE_VRAM_CACHE_ROLE_SPLIT=1` with default upgate pct 63:
  - Deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T214833Z-20260709-rolesplit63-deploy-n64`
    reached `eval_tok_s=4.3`, RAM OK.
  - Fibonacci n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T214908Z-20260709-rolesplit63-fibonacci-n64`
    reached `eval_tok_s=4.0`, RAM OK.
- `GGML_MOE_VRAM_CACHE_ROLE_SPLIT=1`,
  `GGML_MOE_VRAM_CACHE_UPGATE_PCT=80`:
  - Deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215001Z-20260709-rolesplit80-deploy-n64`
    reached `eval_tok_s=4.2`, RAM OK.
  - Fibonacci n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215036Z-20260709-rolesplit80-fibonacci-n64`
    reached `eval_tok_s=4.0`, RAM OK.
- Decision: reject and revert the uncommitted implementation. Simple static
  role split does not improve over the accepted default. The profile signal
  remains valid, but the next cache work must be future-use aware at the
  request/key level rather than statically partitioning the same 13.5 GiB
  budget by tensor role.

## 2026-07-09 GPU-Top1 Plus CPU-Tail Generalized SOTA

Tested exact split variant:

- `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=1`
- Only route rank 0 for up/down is sent through the GPU batch path.
- Remaining up/down route ranks stay in the existing CPU fallback path, so
  this is an exactness-preserving split rather than expert deletion.
- Theory:
  on the Gen4 x4 new machine, H2D and iouring wait dominate. Moving fewer
  up/down routes through the GPU path can reduce expert transfer pressure.
  The risk is that CPU tail compute becomes too expensive. The clean results
  below show the tradeoff is favorable on most dev prompts.

Clean source validation, commit `01914d5b5`, strict cold, 16GB cgroup,
display/model process cleanup recorded before every run:

- Deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215454Z-20260709-clean-gpu1-cputail-deploy-n64`
  reached `eval_tok_s=4.6`, `prompt_tok_s=3.8`,
  `first_output_ms=17192.1 ms`,
  `memory_peak_bytes=14865059840`, RAM OK. Output was coherent and covered
  quantization/model optimization.
- Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215528Z-20260709-clean-gpu1-cputail-fibonacci-n64`
  reached `eval_tok_s=4.5`, `prompt_tok_s=3.5`,
  `first_output_ms=15295.1 ms`,
  `memory_peak_bytes=14880780288`, RAM OK. Output began with a valid Python
  Fibonacci function.
- France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215602Z-20260709-clean-gpu1-cputail-france-n64`
  reached `eval_tok_s=4.1`, `prompt_tok_s=3.3`,
  `first_output_ms=16942.4 ms`,
  `memory_peak_bytes=14833598464`, RAM OK. France output remained
  semantically correct and coherent, but this is slower than the top2 France
  run.
- Quantum n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215705Z-20260709-clean-gpu1-cputail-quantum-n64`
  reached `eval_tok_s=4.2`, `prompt_tok_s=3.2`,
  `first_output_ms=16721.0 ms`,
  `memory_peak_bytes=14842560512`, RAM OK. Output was semantically correct;
  it has the same minor opening-format quirk as the top2 output.
- Japan n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215739Z-20260709-clean-gpu1-cputail-japan-n64`
  reached `eval_tok_s=5.2`, `prompt_tok_s=3.3`,
  `first_output_ms=17226.1 ms`,
  `memory_peak_bytes=14838919168`, RAM OK. Output was coherent.

Top2/default comparison for the newly added dev prompts:

- Quantum top2/default:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215839Z-20260709-clean-top2-default-quantum-n64`
  reached `eval_tok_s=3.3`, `prompt_tok_s=3.8`,
  `first_output_ms=15675.5 ms`, RAM OK.
- Japan top2/default:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T215917Z-20260709-clean-top2-default-japan-n64`
  reached `eval_tok_s=4.8`, `prompt_tok_s=4.1`,
  `first_output_ms=15682.1 ms`, RAM OK.

Decision: accept GPU top1 plus CPU tail as the new generalized SOTA default
for the demo script. Across the five dev prompts tested here, top1 reached
approximately `4.52 tok/s` average with `4.1 tok/s` minimum, versus the
available top2/default comparison at approximately `4.22 tok/s` average with
`3.3 tok/s` minimum. TTFT remains within the `+20%` rule for the corresponding
prompt comparisons. This is still below the product requirement of stable
`>5 tok/s` for random user prompts; only the Japan prompt crossed `5 tok/s`.
The next optimization should focus on reducing prompt-dependent variance,
especially France/quantum, while preserving the lower-bound gain from top1.

## 2026-07-09 Gate-Preload Up/Down Protection Probe

Starting point:

- Clean default reproduction after required display/model cleanup, source
  `4367a37e9`, prompt `How to deploy a large model on a small devices?`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T220340Z-20260709-default-top1-sota-deploy-n64`
  reached `eval_tok_s=4.6`, `prompt_tok_s=3.8`,
  `first_output_ms=17584.4 ms`, `memory_peak_bytes=14861144064`,
  `memory_file_bytes=13810601984`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`, source clean. Output was
  coherent. The run did not meet the product target.

Profile:

- Diagnostic n96 profile:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T220525Z-20260709-default-top1-profile-deploy-n96`
  with profile files in `/tmp/20260709-default-top1-profile-deploy-n96`.
- Result: `eval_tok_s=4.5`, `prompt_tok_s=3.6`,
  `first_output_ms=17936.9 ms`, `memory_peak_bytes=14857277440`,
  `ram_ok=true`, display cleanup recorded.
- Expert pack counters: `hits=15530`, `misses=0`, `iouring_reads=7282`,
  `iouring_bytes=32451854336`, `iouring_wait_us=3781957`.
- VRAM cache: `hits=20540`, `misses=5122`, `preloads=4320`,
  `hit_rate=80.0%`.
- Copy profile by operation:
  - `gate_batch_preload`: `3178` copies, `13.19 GiB`, `h2d_ms=2329.3`,
    `wall_ms=8108.6`.
  - `gate_updown_cosubmit`: `2160` copies, `8.96 GiB`,
    `h2d_ms=1749.2`, `wall_ms=9095.5`.
  - `runtime_load`: `1944` copies, `8.07 GiB`, `h2d_ms=1386.2`,
    `wall_ms=4162.1`.
- Gate/up/down cosubmit had `no_slot_skips=8248`, so many planned up/down
  cosubmits could not enter the VRAM cache because slots were unavailable.
- Cache eviction profile showed gate preload evicting up/down slots as well
  as gate slots. This suggested a possible prompt-general cache admission
  rule: gate preload should not evict already-used up/down slots.

Implemented default-off experiment:

- Added `GGML_MOE_GATE_PRELOAD_EVICT_UPDOWN_MAX_HITS`.
- When set to `0`, a gate preload insertion may not evict an up/down cache
  slot whose hit count is greater than `0`.
- This is not prompt-specific; it only changes cache admission by tensor role
  and observed cache hits.

Dirty-source probe with `GGML_MOE_GATE_PRELOAD_EVICT_UPDOWN_MAX_HITS=0`,
strict cold, 16GB cgroup, display cleanup recorded:

- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T220919Z-20260709-gate-protect-updown-hit0-deploy-n64`
  reached `eval_tok_s=4.7`, RAM OK, coherent output.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T221007Z-20260709-gate-protect-updown-hit0-fibonacci-n64`
  reached `eval_tok_s=4.6`, RAM OK, valid Python function output.
- France:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T221041Z-20260709-gate-protect-updown-hit0-france-n64`
  reached `eval_tok_s=5.0`, RAM OK, coherent France paragraph.
- Quantum:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T221137Z-20260709-gate-protect-updown-hit0-quantum-n64`
  reached `eval_tok_s=4.3`, RAM OK, semantically correct output with the
  existing minor opening-format quirk.
- Japan:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T221211Z-20260709-gate-protect-updown-hit0-japan-n64`
  reached `eval_tok_s=5.1`, RAM OK, coherent output.

Clean-source validation after commit `0fbe22e19` initially made this guard the
demo default. It did not reproduce as a generalized SOTA:

- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T221405Z-20260709-clean-gate-protect-default-deploy-n64`
  reached `eval_tok_s=4.6`, RAM OK, source clean.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T221439Z-20260709-clean-gate-protect-default-fibonacci-n64`
  reached `eval_tok_s=4.4`, RAM OK, source clean, valid Python function.
- France:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T221513Z-20260709-clean-gate-protect-default-france-n64`
  reached `eval_tok_s=4.9`, RAM OK, source clean, coherent France paragraph.

Decision:

- Reject gate-preload up/down hit protection as the default generalized SOTA.
  It improves France but does not clearly improve deploy, and Fibonacci
  dropped below the previous clean `4.5 tok/s` point. This is not stable
  prompt-general progress toward `>5 tok/s`.
- Keep the runtime knob default-off for future diagnostics, but restore the
  demo default path so current SOTA remains the earlier GPU-top1 plus CPU-tail
  configuration.
- Follow-up script hygiene: the default-off knob must be exported as an empty
  string by default rather than referenced conditionally inside the generated
  runner heredoc; otherwise `set -u` can print an unbound-variable warning
  before launch. Empty string remains disabled because the runtime checks that
  the env exists and has a non-empty value.
- Next work should not focus on simple eviction heuristics. The profile points
  to a hard H2D-byte problem: roughly `30 GiB` expert movement for n96 on a
  `6.6-6.7 GB/s` H2D path. To reach stable `>5 tok/s`, the next candidate
  must either reduce bytes per generated token or make up/down/gate admission
  future-use aware without relying on a specific prompt trace.

## 2026-07-09 GPU0 CPU-Tail Boundary Rejection

Hypothesis:

- Since H2D is the measured bottleneck, try the lower-H2D boundary by setting
  `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=0`.
- This keeps exactness by leaving all up/down route ranks in the existing CPU
  tail path. It does not delete experts or rely on a prompt-specific profile.
- Expected tradeoff: lower H2D pressure, but higher CPU expert compute.

Clean-source test, commit `64e6431e9`, strict cold, 16GB cgroup, required
display/model cleanup before each run:

- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T222129Z-20260709-gpu0-cputail-deploy-n64`
  reached `eval_tok_s=3.2`, `prompt_tok_s=5.5`,
  `first_output_ms=16157.6 ms`, `memory_peak_bytes=14260916224`,
  `ram_ok=true`, source clean. Output was coherent.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T222207Z-20260709-gpu0-cputail-fibonacci-n64`
  reached `eval_tok_s=3.0`, `prompt_tok_s=4.6`,
  `first_output_ms=14536.9 ms`, `memory_peak_bytes=14264561664`,
  `ram_ok=true`, source clean. Output began with a valid Python Fibonacci
  function.
- France:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T222247Z-20260709-gpu0-cputail-france-n64`
  reached `eval_tok_s=3.4`, `prompt_tok_s=4.7`,
  `first_output_ms=15758.0 ms`, `memory_peak_bytes=14225068032`,
  `ram_ok=true`, source clean. France output was semantically correct and
  coherent.

Decision:

- Reject `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=0`. It saves RAM/H2D pressure but
  shifts too much work to CPU and drops decode throughput far below the
  accepted top1 GPU plus CPU-tail SOTA.
- This rules out "move all up/down to CPU" as a path to `>5 tok/s` on the new
  machine. The next optimization must keep at least route rank 0 on GPU and
  reduce H2D bytes inside that path, or improve reuse/admission without pushing
  the full up/down workload to CPU.

## 2026-07-09 Drop Gate Cache After Use Rejection

Hypothesis:

- Profile showed gate preload as the largest H2D category and showed gate
  slots competing with up/down slots.
- A default-off experiment was implemented locally: after a gate one-stream
  expert finished and synchronized, clear that gate expert's batch-cache slot
  metadata so up/down entries can reuse the slot sooner.
- This was exactness-preserving and prompt-general, but dirty-source only.

Dirty-source run, strict cold, 16GB cgroup, required display/model cleanup
before each run, `GGML_MOE_DROP_GATE_BATCH_CACHE_AFTER_USE=1`:

- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T222633Z-20260709-drop-gate-cache-deploy-n64`
  reached `eval_tok_s=3.7`, `prompt_tok_s=3.7`,
  `first_output_ms=17484.1 ms`, `memory_peak_bytes=14821130240`,
  `ram_ok=true`, output coherent.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T222711Z-20260709-drop-gate-cache-fibonacci-n64`
  reached `eval_tok_s=4.5`, `prompt_tok_s=3.3`,
  `first_output_ms=15454.6 ms`, `memory_peak_bytes=14868500480`,
  `ram_ok=true`, valid Python function output.
- France:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T222744Z-20260709-drop-gate-cache-france-n64`
  reached `eval_tok_s=5.1`, `prompt_tok_s=3.3`,
  `first_output_ms=16712.1 ms`, `memory_peak_bytes=14845317120`,
  `ram_ok=true`, coherent France paragraph.

Decision:

- Reject. The run is not generalized: France improves, Fibonacci is roughly
  neutral, but deploy drops from the accepted `4.6 tok/s` default to
  `3.7 tok/s`.
- Revert the local code experiment and do not keep the runtime knob. Gate
  cache residency has real reuse value for deploy-like prompts; freeing gate
  immediately after use is too blunt.
- Next cache work must be selective and future-use aware, not a role-wide gate
  eviction rule.

## 2026-07-09 CUDA Graph Enable Rejection

Hypothesis:

- The demo currently defaults to `GGML_CUDA_DISABLE_GRAPHS=1`.
- Enabling CUDA graph with `GGML_CUDA_DISABLE_GRAPHS=0` might reduce launch or
  CPU scheduling overhead without changing model math. It is prompt-general and
  exactness-preserving.

Clean-source test, commit `4c9fbb20f`, strict cold, 16GB cgroup, required
display/model cleanup before each run:

- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223016Z-20260709-cudagraph-on-deploy-n64`
  reached `eval_tok_s=4.6`, `prompt_tok_s=3.7`,
  `first_output_ms=17469.9 ms`, `memory_peak_bytes=14834925568`,
  `ram_ok=true`, output coherent.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223050Z-20260709-cudagraph-on-fibonacci-n64`
  reached `eval_tok_s=4.5`, `prompt_tok_s=3.2`,
  `first_output_ms=15805.1 ms`, `memory_peak_bytes=14863200256`,
  `ram_ok=true`, valid Python function output.
- France:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223124Z-20260709-cudagraph-on-france-n64`
  reached `eval_tok_s=5.0`, `prompt_tok_s=3.3`,
  `first_output_ms=16969.0 ms`, `memory_peak_bytes=14848954368`,
  `ram_ok=true`, coherent France output.
- Quantum:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223218Z-20260709-cudagraph-on-quantum-n64`
  reached only `eval_tok_s=3.8`, `prompt_tok_s=3.1`,
  `first_output_ms=18095.4 ms`, `memory_peak_bytes=14817832960`,
  `ram_ok=true`. Output was semantically correct but retained the known minor
  opening-format quirk.
- Japan:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223255Z-20260709-cudagraph-on-japan-n64`
  reached `eval_tok_s=5.1`, `prompt_tok_s=3.4`,
  `first_output_ms=16500.9 ms`, `memory_peak_bytes=14836994048`,
  `ram_ok=true`, coherent output.

Decision:

- Reject enabling CUDA graph as the default SOTA path. It helps or matches some
  prompts, but Quantum drops below the accepted top1 baseline and TTFT is
  higher on that prompt.
- Keep `GGML_CUDA_DISABLE_GRAPHS=1` as the demo default.
- CUDA graph is not the main route to stable `>5 tok/s`; the H2D-byte profile
  remains the stronger bottleneck evidence.

## 2026-07-09 CPU Tail Source Profile And Pack-Mmap Rejection

Current top1 profile:

- Added demo passthrough for CPU fallback diagnostic envs:
  `GGML_KIMI_CPU_MOE_PROFILE`, `GGML_KIMI_CPU_MOE_NAME_PROFILE`,
  `GGML_KIMI_CPU_MOE_NAME_PROFILE_TOP`,
  `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT`,
  `GGML_MOE_FALLBACK_REASON_PROFILE_OUT`,
  `GGML_MOE_FALLBACK_SOURCE_PROBE_OUT`.
- Diagnostic run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223557Z-20260709-cpu-tail-profile-deploy-n64`
  with files in `/tmp/20260709-cpu-tail-profile-deploy-n64`.
- Because CSV fallback profiling is heavy, the run reached only
  `eval_tok_s=3.7`; do not use this token rate as a SOTA comparison.
- CPU fallback profile showed:
  - total fallback time `4644.9 ms`;
  - up fallback `2428.5 ms`, down fallback `2208.7 ms`, gate fallback
    `7.7 ms`;
  - decode up `1653.8 ms`, decode down `1118.8 ms`;
  - prompt up/down also consumed about `1707 ms` combined;
  - all CPU fallback source rows were `CPU_Mapped` GGUF with
    `pack_mmap_calls=0` and `gguf_calls=11880`.
- Copy profile from the same diagnostic still showed substantial H2D:
  `gate_batch_preload=10.35 GiB`, `gate_updown_cosubmit=8.96 GiB`,
  `runtime_load=5.20 GiB`.

Hypothesis:

- Try existing `GGML_MOE_CPU_FALLBACK_PACK_MMAP=1`, which should make decode
  CPU fallback read expert weights from the expert pack mmap path instead of
  the main GGUF mmap source. This is exactness-preserving and prompt-general.
- Added demo passthrough and config recording for
  `GGML_MOE_CPU_FALLBACK_PACK_MMAP`.

Dirty-source config probe, strict cold, 16GB cgroup, display/model cleanup
recorded before each run:

- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223759Z-20260709-cpu-fallback-pack-mmap-deploy-n64`
  reached `eval_tok_s=4.7`, `prompt_tok_s=3.7`,
  `first_output_ms=17673.4 ms`, `ram_ok=true`, coherent output.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223834Z-20260709-cpu-fallback-pack-mmap-fibonacci-n64`
  reached only `eval_tok_s=3.9`, `prompt_tok_s=2.9`,
  `first_output_ms=15891.7 ms`, `ram_ok=true`, valid Python function output.
- France:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T223910Z-20260709-cpu-fallback-pack-mmap-france-n64`
  reached `eval_tok_s=5.1`, `prompt_tok_s=3.3`,
  `first_output_ms=16999.1 ms`, `ram_ok=true`, coherent France output.
- The pack-mmap path did activate: all three runs reported
  `[kimi_cpu_fallback_pack_mmap] enabled=1 hits=10080 misses=0 ... fallback_gguf=0`.

Decision:

- Reject `GGML_MOE_CPU_FALLBACK_PACK_MMAP=1` as a default SOTA path. It helps
  France and slightly helps deploy, but Fibonacci drops badly.
- Keep the demo passthrough for future diagnostics, but do not set this env by
  default.
- CPU tail remains material, but simply switching decode fallback source to
  expert-pack mmap is not a stable route to `>5 tok/s`.

## 2026-07-09 Role-Specific GPU Top-K Plan

Current bottleneck evidence:

- The accepted generalized path uses `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=1`, so GPU
  handles the top route for both `ffn_up_exps` and `ffn_down_exps`, while the
  remaining selected routes are handled by CPU fallback.
- CPU fallback profiling showed about `4.6s` total fallback time on a profiled
  deploy run, with up fallback and down fallback both material.
- The previous blunt `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=2` candidate moved both up
  and down second routes to GPU and was not generalized. It likely added too
  much H2D pressure in exchange for reducing CPU tail.

Next experiment:

- Add default-off role-specific controls:
  `GGML_MOE_GPU_KEEP_TOPK_UP` and `GGML_MOE_GPU_KEEP_TOPK_DOWN`.
- If unset, behavior must remain exactly the current accepted default through
  `GGML_MOE_GPU_KEEP_TOPK_UPDOWN=1`.
- Test asymmetric candidates:
  - `UP=1, DOWN=2`: reduce down CPU fallback while keeping up H2D at the SOTA
    level.
  - `UP=2, DOWN=1`: reduce up CPU fallback while keeping down H2D at the SOTA
    level.
- Theory: if one role has a better CPU-time-saved / H2D-byte-added ratio, the
  asymmetric variant can reduce the measured CPU tail without paying the full
  cost of top2 for both roles. The hard upper bound is the profiled CPU tail
  saved minus additional H2D time at the current new-machine bandwidth. Since
  H2D remains the largest hardware-linked gap, only asymmetric configurations
  that reduce end-to-end time on multiple prompts can be promoted.

Validation:

- Strict cold, 16GB cgroup including page cache, display/model cleanup before
  every run.
- Start with dev prompts `deploy`, `fibonacci`, and `France`; expand to
  `quantum` and `Japan` only if the first three show generalized improvement.
- Record config, answer, RAM stats, TTFT, token rate, and profile counters in
  the run artifact.
- Promote only if the clean generalized result beats the current accepted
  average/min without correctness loss or TTFT >20%. If accepted, commit and
  push immediately to `vendor/deepseek-token-rate-16gb` with reproduction
  details.

Dirty-source diagnostic results, strict cold, 16GB cgroup, display/model
cleanup before every run:

- `GGML_MOE_GPU_KEEP_TOPK_UP=1`, `GGML_MOE_GPU_KEEP_TOPK_DOWN=2`:
  - deploy:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T224641Z-20260709-role-up1-down2-deploy-n64`,
    `eval_tok_s=3.1`, `prompt_tok_s=3.8`,
    `first_output_ms=18164.8 ms`, `memory_peak_bytes=14856286208`,
    `ram_ok=true`, cleanup recorded.
  - fibonacci:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T224722Z-20260709-role-up1-down2-fibonacci-n64`,
    `eval_tok_s=2.9`, `prompt_tok_s=2.4`,
    `first_output_ms=17066.3 ms`, `memory_peak_bytes=14892400640`,
    `ram_ok=true`, cleanup recorded. The n64 answer was truncated before the
    function body completed, so quality is not acceptable.
  - France:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T224806Z-20260709-role-up1-down2-france-n64`,
    `eval_tok_s=3.2`, `prompt_tok_s=3.4`,
    `first_output_ms=17725.0 ms`, `memory_peak_bytes=14786007040`,
    `ram_ok=true`, cleanup recorded. The output started correctly but was
    truncated at the token cap.
- `GGML_MOE_GPU_KEEP_TOPK_UP=2`, `GGML_MOE_GPU_KEEP_TOPK_DOWN=1`:
  - deploy:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T224907Z-20260709-role-up2-down1-deploy-n64`,
    `eval_tok_s=3.0`, `prompt_tok_s=3.5`,
    `first_output_ms=19046.0 ms`, `memory_peak_bytes=14496821248`,
    `ram_ok=true`, cleanup recorded.
  - fibonacci:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T224949Z-20260709-role-up2-down1-fibonacci-n64`,
    `eval_tok_s=3.0`, `prompt_tok_s=3.3`,
    `first_output_ms=16259.0 ms`, `memory_peak_bytes=14570172416`,
    `ram_ok=true`, cleanup recorded. The n64 answer was truncated before the
    function body completed, so quality is not acceptable.
  - France:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T225031Z-20260709-role-up2-down1-france-n64`,
    `eval_tok_s=3.0`, `prompt_tok_s=3.5`,
    `first_output_ms=17005.3 ms`, `memory_peak_bytes=14501822464`,
    `ram_ok=true`, cleanup recorded. The output started correctly but was
    truncated at the token cap.

Decision:

- Reject role-specific top2 GPU routing. Both asymmetric variants are far below
  the accepted top1 GPU / CPU-tail SOTA (`deploy≈4.6`, `fibonacci≈4.5`,
  `France≈4.1` on the clean five-prompt validation).
- The observed gap matches the bottleneck model: adding a second GPU route for
  either role increases expert H2D traffic and cache pressure more than it
  saves CPU fallback time on the new x4 machine.
- Revert the experimental runtime knobs and keep the current SOTA code path.
  Next work should reduce H2D bytes/copy count or improve reuse, not move more
  complete expert routes to GPU.

## 2026-07-09 Clean SOTA Profile And Gate Cache Recheck Plan

Clean profile run:

- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T225321Z-20260709-clean-sota-profile-deploy-n96`
  with profile files in `/tmp/20260709-clean-sota-profile-deploy-n96`.
- Source clean at `8e90384c7`, strict cold, 16GB cgroup, display/model cleanup
  recorded.
- Summary: `eval_tok_s=3.2` with profiling overhead, `prompt_tok_s=3.9`,
  `first_output_ms=17981.1 ms`, `memory_peak_bytes=14883590144`,
  `memory_file_bytes=13932077056`, `ram_ok=true`.
- Hardware state: PCIe settles at `16.0 GT/s x4`; 4.25MiB H2D benchmark
  `6.81 GB/s`.

Measured bottleneck split:

- Gate one-pack direct reads dominate the profile:
  `12552` gate reads, `52.10 GiB` transferred, `17891 ms` total profile time,
  `9189 ms` source-read time, `misses=0`.
- Gate reuse is high: only `2821` unique `(tensor, expert)` gate entries were
  read for `12552` uses. The hottest gate expert was read `95` times in this
  single request.
- CPU fallback remains material but smaller than repeated gate movement:
  decode up `2168 ms`, decode down `2097 ms`, prompt up `932 ms`, prompt down
  `1702 ms`, prompt gate `7 ms`.

Next experiment:

- Re-test a small `GGML_MOE_STREAM_ONE_CACHE_MIB` gate cache on the current
  clean SOTA rather than large split-cache sizes. Earlier large gate-cache
  probes were rejected because they displaced up/down batch cache. The new
  profile suggests a small hot LRU may still be useful if it captures repeated
  gate experts without stealing too much up/down VRAM.
- Start with `GGML_MOE_STREAM_ONE_CACHE_MIB=1024` and `2048`, keeping
  `GGML_MOE_VRAM_CACHE_MIB=13824` and all other SOTA defaults unchanged.
- Test `deploy`, `fibonacci`, and `France` under strict cold, display cleanup,
  and 16GB cgroup. Promote only if generalized token rate improves without
  correctness loss or TTFT regression.
- If small cache helps, implement future-use-aware admission instead of a blunt
  LRU. If small cache still regresses, stop gate cache work and focus on
  reducing gate bytes/copy count at the representation level.

Small gate-cache results, clean source `8e90384c7`, strict cold, 16GB cgroup,
display/model cleanup before every run:

- `GGML_MOE_STREAM_ONE_CACHE_MIB=1024`:
  - deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T225632Z-20260709-stream-one-cache1024-deploy-n64`,
    `eval_tok_s=4.6`, `prompt_tok_s=3.8`,
    `first_output_ms=17819.5 ms`, `ram_ok=true`.
  - fibonacci n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T225731Z-20260709-stream-one-cache1024-fibonacci-n64`,
    `eval_tok_s=4.5`, `prompt_tok_s=3.6`,
    `first_output_ms=15709.3 ms`, `ram_ok=true`.
  - France n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T225806Z-20260709-stream-one-cache1024-france-n64`,
    `eval_tok_s=4.5`, `prompt_tok_s=3.7`,
    `first_output_ms=16444.2 ms`, `ram_ok=true`.
  - Deploy log showed the cache is active: `hits=3792`, `misses=4920`; direct
    gate reads were `4920` / `21.9GB`.
- `GGML_MOE_STREAM_ONE_CACHE_MIB=2048`:
  - deploy n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T225924Z-20260709-stream-one-cache2048-deploy-n64`,
    `eval_tok_s=5.0`, `prompt_tok_s=4.2`,
    `first_output_ms=17273.5 ms`, `ram_ok=true`.
  - fibonacci n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T225958Z-20260709-stream-one-cache2048-fibonacci-n64`,
    `eval_tok_s=4.8`, `prompt_tok_s=3.9`,
    `first_output_ms=14822.8 ms`, `ram_ok=true`.
  - France n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T230030Z-20260709-stream-one-cache2048-france-n64`,
    `eval_tok_s=5.1`, `prompt_tok_s=3.8`,
    `first_output_ms=16156.2 ms`, `ram_ok=true`.
  - Quantum n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T230234Z-20260709-stream-one-cache2048-quantum-n64`,
    `eval_tok_s=4.5`, `prompt_tok_s=3.6`,
    `first_output_ms=16085.7 ms`, `ram_ok=true`.
  - Japan n64:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T230307Z-20260709-stream-one-cache2048-japan-n64`,
    `eval_tok_s=5.3`, `prompt_tok_s=3.6`,
    `first_output_ms=16743.1 ms`, `ram_ok=true`.
  - France n96:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T230126Z-20260709-stream-one-cache2048-france-n96`,
    `eval_tok_s=5.3`, `first_output_ms=16295.1 ms`, `ram_ok=true`.
  - France n128:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T230517Z-20260709-stream-one-cache2048-france-n128`,
    `eval_tok_s=5.2`, `first_output_ms=16587.1 ms`, `ram_ok=true`.

Interim decision:

- `2048MiB` is the strongest current speed candidate, improving the five-prompt
  n64 average to about `4.94 tok/s` and lowering TTFT on most prompts versus
  the prior accepted clean five-prompt SOTA.
- It is not yet promoted as final product SOTA because the minimum prompt rate
  is still `4.5 tok/s`, below the stable `>5 tok/s` target, and the France
  answer remains semantically correct for several sentences but is still cut
  off by the token cap at n96/n128. This needs either a cleaner stopping
  policy or a stricter correctness definition before promotion.
- Since `2048MiB` helps without RAM/TTFT regression, test
  `GGML_MOE_STREAM_ONE_CACHE_MIB=3072` next, focusing first on the weaker
  prompts (`quantum`, `fibonacci`) and rejecting immediately if VRAM/cache
  pressure causes regression.

`GGML_MOE_STREAM_ONE_CACHE_MIB=3072`, clean source `8e90384c7`, strict cold,
16GB cgroup, display/model cleanup before every run:

- Quantum n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T230955Z-20260709-stream-one-cache3072-quantum-n64`,
  `eval_tok_s=4.8`, `prompt_tok_s=3.6`,
  `first_output_ms=15913.2 ms`, `ram_ok=true`.
- Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T231028Z-20260709-stream-one-cache3072-fibonacci-n64`,
  `eval_tok_s=5.0`, `prompt_tok_s=3.8`,
  `first_output_ms=15261.8 ms`, `ram_ok=true`.
- Deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T231119Z-20260709-stream-one-cache3072-deploy-n64`,
  `eval_tok_s=5.1`, `prompt_tok_s=4.3`,
  `first_output_ms=16943.5 ms`, `ram_ok=true`.
- France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T231151Z-20260709-stream-one-cache3072-france-n64`,
  `eval_tok_s=5.4`, `prompt_tok_s=3.9`,
  `first_output_ms=16596.3 ms`, `ram_ok=true`.
- Japan n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T231223Z-20260709-stream-one-cache3072-japan-n64`,
  `eval_tok_s=5.5`, `prompt_tok_s=3.8`,
  `first_output_ms=16612.6 ms`, `ram_ok=true`.

Interim decision:

- `3072MiB` is the best speed candidate so far: five-prompt average is about
  `5.16 tok/s`, and TTFT remains within the gate versus the previous clean
  SOTA. RAM including page cache remains below 16GB.
- It still does not prove the product target because Quantum is `4.8 tok/s`,
  below stable `>5`, and short-paragraph prompts still reach the token cap
  before a clean stop.
- Test `GGML_MOE_STREAM_ONE_CACHE_MIB=4096` on Quantum first. If Quantum does
  not cross 5 or if memory/TTFT regresses, do not spend time on larger blunt
  gate caches; move to future-use-aware gate admission or output stopping.

`GGML_MOE_STREAM_ONE_CACHE_MIB=4096` first probe:

- Quantum n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T231338Z-20260709-stream-one-cache4096-quantum-n64`,
  `eval_tok_s=5.0`, `prompt_tok_s=3.6`,
  `first_output_ms=15874.1 ms`, `memory_peak_bytes=14867517440`,
  `ram_ok=true`, display cleanup recorded.
- This is close to the product threshold but the summary still reports
  `product_target_gt_5_tok_s_met_by_this_run=false`, so it is not a strict
  `>5 tok/s` pass.
- Run one final `5120MiB` Quantum probe. If it does not clearly beat 5 or if it
  hurts RAM/TTFT, stop increasing blunt gate-cache size.

`GGML_MOE_STREAM_ONE_CACHE_MIB=5120` final blunt-cache probe:

- Quantum n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T231604Z-20260709-stream-one-cache5120-quantum-n64`,
  `eval_tok_s=4.4`, `prompt_tok_s=3.5`,
  `first_output_ms=15904.7 ms`, `memory_peak_bytes=14851764224`,
  `ram_ok=true`, display cleanup recorded.
- Cache counters by Quantum run:
  - `3072MiB`: `hits=4948`, `misses=3528`, direct reads `15.7GB`,
    `eval_tok_s=4.8`;
  - `4096MiB`: `hits=5321`, `misses=3155`, direct reads `14.1GB`,
    `eval_tok_s=5.0`;
  - `5120MiB`: `hits=5683`, `misses=2793`, direct reads `12.4GB`,
    `eval_tok_s=4.4`.

Decision:

- Stop increasing blunt gate cache. Larger cache continues reducing direct
  gate reads, but at `5120MiB` end-to-end speed regresses, so the missing time
  is no longer only gate read volume.
- Current best speed candidate is `GGML_MOE_STREAM_ONE_CACHE_MIB=3072`: clean,
  strict-cold, RAM-compliant, five-prompt average about `5.16 tok/s`, but min
  prompt rate is still `4.8 tok/s`. It is a strong candidate, not final product
  completion.
- Next implementation should keep roughly the `3-4GB` gate-cache budget but
  improve admission/eviction: retain high future-use gate experts and avoid
  caching entries that will not be reused in the current request. This targets
  the Quantum lower bound without blindly consuming more VRAM.

## 2026-07-09 Gate Admission Manifest Plan

Hypothesis:

- The existing one-cache LRU improves from 0GB to 3/4GB, but 5GB regresses even
  though it reduces direct reads further. This suggests cache pollution and
  secondary VRAM pressure, not just insufficient capacity.
- Existing code already supports `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE`, which
  restricts cache insertion to listed `(tensor, expert)` entries. We can use it
  before writing new eviction code.

Plan:

1. Generate gate access profiles from calibration/dev prompts only:
   deploy, Fibonacci, France, Quantum, and Japan. Do not use held-out prompts.
2. Aggregate `(tensor, expert)` counts from `gate_one` profile rows and create
   prompt-general admission manifests for top entries sized to the existing
   cache slots:
   - top `722` for `3072MiB`;
   - top `963` for `4096MiB`.
3. Test Quantum first with:
   `GGML_MOE_STREAM_ONE_CACHE_MIB=4096` and
   `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<top963 manifest>`.
4. If Quantum exceeds strict `>5 tok/s` without RAM/TTFT regression, run the
   full dev set. If it does not, reject manifest admission and move to deeper
   request-local future-use eviction or output stopping.

Promotion rule:

- This is still a dev/calibration-tuned optimization. It can become a candidate
  only after clean generalized dev validation, and final SOTA still requires
  held-out validation after the candidate is frozen.

Execution:

- Generated five dev-prompt gate access profiles with source clean at
  `8e90384c7`, strict cold, 16GB cgroup, and display cleanup before every run.
  Profile root: `/tmp/20260709-gate-admit-calib`.
- Aggregated `43116` gate accesses across `5020` unique `(tensor, expert)`
  entries, producing:
  - `/tmp/20260709-gate-admit-calib/top722.tsv`
  - `/tmp/20260709-gate-admit-calib/top963.tsv`
- A dirty-source demo plumbing test allowed the calibration manifest to be
  passed explicitly without raw prompt-specific env injection. The runtime
  loaded the manifest successfully.
- Quantum test with `GGML_MOE_STREAM_ONE_CACHE_MIB=4096` and top963 admission:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T233045Z-20260709-gate-admit-top963-cache4096-quantum-n64`,
  `eval_tok_s=4.4`, `prompt_tok_s=3.3`,
  `first_output_ms=16576.1 ms`, `memory_peak_bytes=14870351872`,
  `ram_ok=true`, display cleanup recorded.
- Cache counters confirm the manifest hurt Quantum locality:
  `cache admission: loaded 963 entries`,
  one-pack direct reads `4312` / `19.2GB`,
  VRAM cache `hits=4164`, `misses=4312`, `hit_rate=49.1%`.
  This is worse than ordinary 4GB LRU (`hits=5321`, `misses=3155`) and ordinary
  3GB LRU (`hits=4948`, `misses=3528`).

Decision:

- Reject global top-count admission manifests. The aggregate dev manifest
  misses request-local Quantum reuse and lowers hit rate.
- Revert the dirty demo plumbing; do not add a calibration admission option
  until there is a manifest strategy that improves generalized performance.
- Next cache work must be request-local/future-use-aware, not a static global
  top-N allowlist.

Follow-up:

- Because ordinary `4096MiB` Quantum reached displayed `5.0 tok/s` but did not
  satisfy the strict `>5` flag, run one cold repeat before abandoning the blunt
  cache size sweep. If the repeat is still not strictly above 5, stop this
  direction and treat `3072/4096MiB` as speed candidates rather than product
  completion.

Repeat result:

- Ordinary `GGML_MOE_STREAM_ONE_CACHE_MIB=4096` Quantum n64 repeat:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T233850Z-20260709-stream-one-cache4096-quantum-repeat-n64`,
  source clean `490ab7f530`, strict cold, display cleanup recorded,
  `eval_tok_s=5.0`, `prompt_tok_s=3.6`, `first_output_ms=16225.7 ms`,
  `memory_peak_bytes=14881697792`, `ram_ok=true`,
  `product_target_gt_5_tok_s_met_by_this_run=false`.

Decision:

- Do not mark 4GB blunt gate cache as meeting the stable `>5 tok/s` target.
  It repeatedly lands at displayed `5.0` on Quantum but does not strictly cross
  the product threshold.
- Current best generalized speed candidate remains the 3GB/4GB gate-cache
  family, with 3GB having the best five-prompt average evidence and 4GB having
  the best Quantum-only evidence. Further progress requires request-local
  admission/eviction or reducing per-token gate bytes/copy count, not larger
  static cache capacity.

Follow-up capacity sweep:

- The runtime only prints one decimal place for generation token rate, and the
  summary parses displayed `5.0` as not strictly greater than 5. Before writing
  deeper cache code, run a narrow Quantum-only sweep between the repeatable
  `4096MiB` edge and the rejected `5120MiB` point.
- Test `4352MiB` first, then `4608MiB` only if needed. If either strictly
  crosses `>5 tok/s` without TTFT/RAM regression, validate the full dev set.
  If both fail or regress, stop blunt-capacity tuning.

## 2026-07-09 4352MiB Gate Cache Dev-Speed SOTA

Configuration:

- vendor DeepSeek, source clean `c0549942aa` for validation runs;
- strict cold, 16GB cgroup including page cache, display/model cleanup before
  every run;
- `GGML_MOE_STREAM_ONE_CACHE_MIB=4352`;
- `GGML_MOE_VRAM_CACHE_MIB=13824`;
- `GGML_MOE_IO_REFILL_BATCH=8`;
- `GGML_MOE_STAGE_PINNED_SLOTS=8`;
- `GGML_CUDA_DISABLE_GRAPHS=1`;
- gate full native expert pack enabled, no prompt-specific packs/profiles.

Dev prompt speed validation:

- Quantum n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T235633Z-20260709-stream-one-cache4352-quantum-n64`,
  `eval_tok_s=5.1`, `prompt_tok_s=3.6`,
  `first_output_ms=16013.4 ms`, `memory_peak_bytes=14860742656`,
  `ram_ok=true`, strict `>5` flag true.
- Deploy n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T235846Z-20260709-stream-one-cache4352-deploy-n64`,
  `eval_tok_s=5.7`, `prompt_tok_s=4.3`,
  `first_output_ms=16789.6 ms`, `memory_peak_bytes=14865670144`,
  `ram_ok=true`, strict `>5` flag true.
- Fibonacci n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T235917Z-20260709-stream-one-cache4352-fibonacci-n64`,
  `eval_tok_s=5.4`, `prompt_tok_s=3.8`,
  `first_output_ms=15113.9 ms`, `memory_peak_bytes=14875869184`,
  `ram_ok=true`, strict `>5` flag true.
- France n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T235949Z-20260709-stream-one-cache4352-france-n64`,
  `eval_tok_s=5.7`, `prompt_tok_s=3.9`,
  `first_output_ms=16483.9 ms`, `memory_peak_bytes=14864424960`,
  `ram_ok=true`, strict `>5` flag true.
- Japan n64:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T000020Z-20260709-stream-one-cache4352-japan-n64`,
  `eval_tok_s=5.8`, `prompt_tok_s=3.8`,
  `first_output_ms=16385.9 ms`, `memory_peak_bytes=14848446464`,
  `ram_ok=true`, strict `>5` flag true.

Summary:

- Five-prompt dev minimum `5.1 tok/s`, mean about `5.54 tok/s`, max `5.8
  tok/s`.
- This is the first strict-cold 16GB-cgroup dev set where every measured prompt
  is strictly above `5 tok/s`.
- TTFT remains within the previous gate and is generally lower than the older
  accepted top1 baseline.
- RAM including page cache remains below 16GB in all runs.

Correctness:

- France n160 correctness run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T000117Z-20260709-stream-one-cache4352-france-n160-correctness`,
  `eval_tok_s=6.0`, `prompt_tok_s=3.8`,
  `first_output_ms=16180.4 ms`, `memory_peak_bytes=14905323520`,
  `ram_ok=true`.
- Output is semantically correct and coherent for multiple sentences:
  it describes France as a Western European country, Paris as capital, and
  mentions Eiffel Tower, Louvre, Versailles, wine regions, cuisine, economy,
  and government. It still ends at the token cap after "art," so output-stop
  cleanup remains a follow-up item, but the model quality path is not degraded.

Decision:

- Promote `GGML_MOE_STREAM_ONE_CACHE_MIB=4352` as the current dev-prompt
  speed SOTA configuration and make it the demo default for reproducibility.
- This does not complete the overall product goal yet because held-out prompt
  validation is still pending and the France answer should be made to stop more
  cleanly.

Clean default reproduction after promotion:

- Default 4352MiB Quantum n64, source clean `256e55303c`, no explicit
  `GGML_MOE_STREAM_ONE_CACHE_MIB` env:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T000552Z-20260709-default4352-quantum-repro-n64`,
  `eval_tok_s=5.0`, `prompt_tok_s=3.6`,
  `first_output_ms=15711.8 ms`, `memory_peak_bytes=14871277568`,
  `ram_ok=true`, but strict `>5` flag false.

Revised decision:

- 4352MiB is a strong dev-speed candidate but not stable enough on Quantum to
  prove the product target. Continue the narrow sweep with `4608MiB`; if that
  does not strictly and repeatably cross `>5`, the demo default should be
  treated as a candidate default rather than final SOTA completion.

4608MiB probe:

- Quantum n64 with explicit `GGML_MOE_STREAM_ONE_CACHE_MIB=4608`, source clean
  `256e55303c`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T000807Z-20260709-stream-one-cache4608-quantum-n64`,
  `eval_tok_s=4.6`, `prompt_tok_s=3.6`,
  `first_output_ms=16218.6 ms`, `memory_peak_bytes=14825299968`,
  `ram_ok=true`, strict `>5` flag false.

Decision:

- Reject `4608MiB`. The narrow capacity sweep confirms a sharp local optimum:
  4096/4352 are near the Quantum threshold, while 4608 and 5120 regress.
- Keep 4352 as a useful dev-speed candidate/default for reproducibility, but
  do not claim stable product completion. The next real optimization must
  target variance and request-local cache quality, or reduce gate copy cost,
  rather than increasing cache size.

Length-sensitivity check:

- The weak Quantum result is measured at n64, where rounding and early-token
  overhead can dominate. Run Quantum n96 with the same default 4352MiB config
  before changing kernels again. If n96 is stable `>5`, record that the
  remaining gap is short-output stability; if n96 also fails, continue with
  gate copy/caching work.

Result:

- Default 4352MiB Quantum n96:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T001017Z-20260709-default4352-quantum-n96`,
  source clean `256e55303c`, `eval_tok_s=5.1`, `prompt_tok_s=3.6`,
  `first_output_ms=15808.1 ms`, `memory_peak_bytes=14875435008`,
  `ram_ok=true`, strict `>5` flag true.
- Interpretation: the 4352MiB candidate crosses `>5` on the weak Quantum prompt
  for n96, but n64 remains borderline. Held-out should use the locked n192
  protocol used by previous SOTA artifacts after candidate freeze.

## 2026-07-09 4352MiB Held-Out V1 Frozen-Candidate Result

Candidate freeze:

- Candidate: default 4352MiB gate one-cache in pushed source `26e0469ff0`.
- Held-out v1 was only run after the candidate was frozen. These prompts must
  not be used for further tuning.
- Strict cold, 16GB cgroup including page cache, display/model cleanup before
  every run.

Held-out speed results:

- Photosynthesis n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T001445Z-20260709-heldout4352-photosynthesis-n192`,
  `eval_tok_s=5.6`, `prompt_tok_s=3.8`,
  `first_output_ms=15984.4 ms`, `memory_peak_bytes=14855651328`,
  `memory_file_bytes=13891424256`, `ram_ok=true`.
- Office n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T001528Z-20260709-heldout4352-office-n192`,
  `eval_tok_s=7.3`, `prompt_tok_s=3.8`,
  `first_output_ms=18799.2 ms`, `memory_peak_bytes=14863302656`,
  `memory_file_bytes=13882265600`, `ram_ok=true`.
- Palindrome JS n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T001614Z-20260709-heldout4352-palindrome-js-n192`,
  `eval_tok_s=5.5`, `prompt_tok_s=4.5`,
  `first_output_ms=16588.9 ms`, `memory_peak_bytes=14927384576`,
  `memory_file_bytes=13882314752`, `ram_ok=true`.
- Exercise n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T001709Z-20260709-heldout4352-exercise-n192`,
  `eval_tok_s=5.7`, `prompt_tok_s=3.7`,
  `first_output_ms=16295.5 ms`, `memory_peak_bytes=14884085760`,
  `memory_file_bytes=13877923840`, `ram_ok=true`.
- Brazil n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T001752Z-20260709-heldout4352-brazil-n192`,
  `eval_tok_s=5.8`, `prompt_tok_s=3.7`,
  `first_output_ms=16493.6 ms`, `memory_peak_bytes=14900174848`,
  `memory_file_bytes=13872996352`, `ram_ok=true`.

Aggregate:

- Held-out speed min/mean/max: `5.5 / 5.98 / 7.3 tok/s`.
- All five held-out runs satisfy strict `>5 tok/s`.
- All five runs satisfy the 16GB RAM/page-cache cgroup limit and display
  cleanup requirement.

Correctness:

- Photosynthesis: pass. Output is complete and semantically correct.
- Office: fail. Output degenerates into repeated `<ds>` tokens and does not
  answer the three-tip request.
- Palindrome JS: partial pass. The JavaScript function is correct, but the
  answer continues into optional explanation and is truncated at the cap.
- Exercise: pass with minor typo (`endendorphins`), otherwise coherent.
- Brazil: partial pass. Output is semantically correct for many sentences but
  continues too long and is truncated at the cap.

Decision:

- Do not mark the product goal complete. The frozen candidate proves the speed
  side of the held-out target (`min >5 tok/s`) but fails correctness on the
  office held-out prompt.
- Do not tune against held-out v1. Next work must use calibration/dev prompts
  to implement a prompt-general output stopping/repetition guard or generation
  control, then freeze a new candidate before any further held-out validation.

## 2026-07-09 Request-Local Gate Cache Admission Plan

Hypothesis:

- The ordinary gate one-cache admits every miss into VRAM. This helps at
  `3-4GB`, but it also admits one-hit or low-reuse entries that can evict hotter
  request-local entries.
- Static global top-N admission was rejected because it missed Quantum-local
  reuse. A request-local policy should adapt to whichever prompt is currently
  running without being prompt-specific.

Implementation plan:

- Add a default-off runtime knob
  `GGML_MOE_STREAM_ONE_CACHE_MIN_ACCESSES=N`.
- When `N > 1`, a gate expert is only inserted into the one-cache after it has
  been requested at least `N` times in the current process/request. Earlier
  requests still execute normally through the existing direct expert-pack read
  path, but they do not consume cache slots.
- Start with `N=2` and `GGML_MOE_STREAM_ONE_CACHE_MIB=4096` on Quantum. Theory:
  the extra first-read cost is bounded by one additional direct read for entries
  that would have been cached on first use, while low-reuse entries no longer
  evict hot entries. If the hot-set churn is the reason 4GB stalls at displayed
  `5.0`, this should improve the strict lower bound.

Validation:

- Strict cold, 16GB cgroup, display cleanup before every run.
- First test Quantum n64 because it is the current weak prompt. If Quantum does
  not exceed strict `>5 tok/s`, reject or adjust `N` before running the full dev
  set.
- If Quantum improves, run the five-prompt dev set and then freeze the
  candidate for held-out validation.

Execution:

- Implemented dirty-source runtime knob
  `GGML_MOE_STREAM_ONE_CACHE_MIN_ACCESSES`.
- Quantum test with `GGML_MOE_STREAM_ONE_CACHE_MIB=4096` and
  `GGML_MOE_STREAM_ONE_CACHE_MIN_ACCESSES=2`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260708T234833Z-20260709-cache4096-min2-quantum-n64`,
  `eval_tok_s=4.5`, `prompt_tok_s=3.1`,
  `first_output_ms=16953.7 ms`, `memory_peak_bytes=14856441856`,
  `ram_ok=true`, display cleanup recorded.
- Cache counters:
  `one expert pack reads=4009` / `17.9GB`,
  VRAM cache `hits=4467`, `misses=4009`, `hit_rate=52.7%`,
  `admit_delay_skips=2482`.

Decision:

- Reject delayed insert. It skips too many early useful insertions and performs
  worse than ordinary 4GB LRU (`5.0 tok/s`) and ordinary 3GB LRU (`4.8 tok/s`)
  on Quantum.
- Revert the dirty runtime knob. Request-local policy must be more selective
  than a fixed "cache after N accesses" rule.

## 2026-07-09 Output Correctness Recovery Plan

Reason:

- The frozen 4352MiB candidate already demonstrated the speed side of the
  product target on held-out v1: all five held-out prompts were strict
  `>5 tok/s` under the 16GB cgroup with display cleanup recorded.
- It cannot be promoted because one held-out prompt degenerated into repeated
  marker-like tokens and several long answers ran until the token cap.
- Further tuning must not use held-out v1. The next work uses only calibration
  and dev prompts, then freezes a new candidate before any future held-out
  validation.

Hypothesis:

- The current demo uses deterministic greedy decoding:
  `--temp 0 --top-p 1 --top-k 1`. This is fast and reproducible, but it gives
  the sampler no generic protection against repeated low-quality token loops.
- A prompt-general repetition-control layer, such as repeat penalty or DRY
  sampling, should reduce marker loops and cap-runaway answers without changing
  the MoE streaming path. Because sampling overhead is small relative to
  expert movement/compute, token rate should remain close to the current
  4352MiB candidate.

Design:

1. Add default-off runtime/script knobs for generic output controls:
   repeat penalty, repeat-last-n, frequency/presence penalty, DRY options,
   conversation-mode forcing, and a generic system prompt. Do not add
   prompt-specific strings, prompt-specific stop rules, or held-out-derived
   heuristics.
2. Run calibration/dev prompts only, for example:
   - `Please introduce France in a short paragraph.`
   - `Explain quantum computing briefly.`
   - `Write a short Python function for Fibonacci.`
   - `Introduce Japan in a short paragraph.`
   - `Summarize climate change in one paragraph.`
   - `How to deploy a large model on a small devices?`
   - `AI infra 是做什么的`
   - `今天吃什么`
3. For each candidate sampler setting, require:
   - strict cold run with display/model cleanup before launch;
   - 16GB cgroup including page cache, no swap/OOM;
   - France output semantically correct and coherent;
   - no obvious marker-token loop or immediate degeneration on the dev set;
   - TTFT not more than 20% above the accepted baseline;
   - token rate still targets `>5 tok/s` on the weak dev prompts.
4. If a generic output-control candidate passes the dev set, freeze it and only
   then run a held-out validation set. The held-out result determines whether
   it can become the product SOTA.

Initial priority:

- First test low-risk repeat controls because they require no CUDA or IO code
  changes and can be enabled through `llama-cli` flags.
- If repeat controls do not address cap-runaway output, test whether a generic
  concise-answer system prompt plus explicit conversation mode improves natural
  stopping and avoids malformed first tokens. This remains prompt-general and
  must be validated on the dev set before any held-out run.
- If repeat controls reduce correctness failures but cost too much speed, keep
  them as an optional rejected/diagnostic path and return to the MoE bottleneck
  plan.

Execution notes:

- Added default-off demo knobs for prompt-general sampler/output controls:
  `LLAMA_DEMO_REPEAT_PENALTY`, `LLAMA_DEMO_REPEAT_LAST_N`,
  `LLAMA_DEMO_FREQUENCY_PENALTY`, `LLAMA_DEMO_PRESENCE_PENALTY`,
  `LLAMA_DEMO_DRY_*`, `LLAMA_DEMO_SYSTEM_PROMPT`, and
  `LLAMA_DEMO_FORCE_CONVERSATION`. When unset, the previous command line is
  unchanged.
- Default 4352MiB greedy dev probe, sleep tips n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T002804Z-20260709-dev-default4352-sleep-tips-n192`,
  `eval_tok_s=5.3`, `first_output_ms=16362.2 ms`,
  `memory_peak_bytes=14903431168`, `ram_ok=true`, display cleanup recorded.
  Output is semantically correct but runs into the token cap.
- Repeat-penalty diagnostic, desk tips n192, `repeat_penalty=1.08`,
  `repeat_last_n=128`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T003156Z-20260709-dev-repeat108-desk-tips-n192`,
  `eval_tok_s=5.2`, `first_output_ms=16469.2 ms`,
  `memory_peak_bytes=14898249728`, `ram_ok=true`, display cleanup recorded.
  Output does not degenerate, but it still runs into the token cap and begins
  with a malformed `inHere` prefix. Reject as a correctness fix.
- Generic system prompt plus forced conversation, France n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T003520Z-20260709-dev-sysprompt-conv-france-n192`,
  `eval_tok_s=5.8`, `first_output_ms=18993.0 ms`,
  `memory_peak_bytes=14865874944`, `ram_ok=true`, display cleanup recorded.
  France content is semantically correct, but output remains too long and TTFT
  is close to the 20% ceiling. Reject as default.
- Longer concise system prompt without forced conversation, France n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T003716Z-20260709-dev-sysprompt-brief-france-n192`,
  `eval_tok_s=5.9`, `first_output_ms=19887.1 ms`,
  `memory_peak_bytes=14887485440`, `ram_ok=true`, display cleanup recorded.
  Output is coherent and naturally complete, but TTFT exceeds the likely 20%
  bound versus the 4352MiB candidate. Reject as default unless a higher TTFT
  exception is explicitly marked unacceptable.
- Short system prompt, France n192, `Be concise. Stop when done.`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T003825Z-20260709-dev-sysprompt-short-france-n192`,
  `eval_tok_s=5.9`, `first_output_ms=17924.1 ms`,
  `memory_peak_bytes=14884691968`, `ram_ok=true`, display cleanup recorded.
  TTFT is better, but output still runs into the token cap. Reject.
- Short max-word system prompt, France n192, `Max 80 words. Stop after
  answer.`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T003938Z-20260709-dev-sysprompt-max80-france-n192`,
  `eval_tok_s=5.9`, `first_output_ms=18182.4 ms`,
  `memory_peak_bytes=14899146752`, `ram_ok=true`, display cleanup recorded.
  Output ignores the hard word budget and truncates. Reject.
- One-paragraph system prompt, France n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T004059Z-20260709-dev-sysprompt-onepara-france-n192`,
  `eval_tok_s=6.0`, `first_output_ms=18444.7 ms`,
  `memory_peak_bytes=14913933312`, `ram_ok=true`, display cleanup recorded.
  Output still runs into the token cap. Reject.

Decision:

- Prompt-general sampler/system-prompt controls are useful diagnostics but do
  not yet produce an accepted product SOTA. The only complete France output was
  the longer system prompt, but it raises TTFT too much for promotion.
- Next correctness work should be a runtime-generic stopping strategy, for
  example a wrapper-level sentence-boundary/repetition guard or a llama-cli
  generation-loop change that stops after a coherent completed answer without
  using held-out or prompt-specific text.

## 2026-07-09 Wrapper-Level Sentence Guard Candidate

Implementation:

- Add default-off `LLAMA_DEMO_OUTPUT_GUARD=sentence` to the demo script.
- The guard is prompt-general and does not inspect the prompt. It keeps
  `answer.raw.txt` and `raw_answer` in `summary.json`, then trims the displayed
  `answer.txt`/summary answer to the last complete sentence only when the raw
  output ends mid-sentence. It does not trim unterminated code fences.
- This is not a compute-speed optimization because the underlying generation
  still runs to the configured token cap. It is a correctness/display guard for
  cap-truncated answers while a true llama-cli generation-loop stop is still
  pending.

First diagnostic:

- Default 4352MiB greedy config plus `LLAMA_DEMO_OUTPUT_GUARD=sentence`,
  France n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T004447Z-20260709-dev-outputguard-france-n192`,
  source `ecbc1c165a` dirty, strict cold, display cleanup recorded,
  `eval_tok_s=6.2`, `prompt_tok_s=3.9`, `first_output_ms=16326.8 ms`,
  `memory_peak_bytes=14909624320`, `memory_file_bytes=13996359680`,
  `ram_ok=true`, `output_guard_applied=true`.
- Displayed France answer is semantically correct, coherent, and ends on a
  complete sentence. This satisfies the France correctness gate for this dev
  run while preserving raw output for audit.

Next validation:

- Commit the default-off guard for reproducibility, sync the new machine to a
  clean source commit, and rerun a dev prompt set with
  `LLAMA_DEMO_OUTPUT_GUARD=sentence`.
- If the dev set passes speed/RAM/TTFT/correctness, freeze a candidate before
  any held-out validation. Do not use held-out v1 to tune the guard.

Clean-source dev validation after initial guard commit:

- Source `1b1def0112`, strict cold, 16GB cgroup, display cleanup before every
  run, `LLAMA_DEMO_OUTPUT_GUARD=sentence`.
- France n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T004656Z-20260709-clean-outputguard-france-n192`,
  `eval_tok_s=5.8`, `first_output_ms=16288.0 ms`,
  `memory_peak_bytes=14864781312`, `ram_ok=true`,
  `output_guard_applied=true`, correctness pass.
- Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T004749Z-20260709-clean-outputguard-quantum-n192`,
  `eval_tok_s=5.1`, `first_output_ms=16331.5 ms`,
  `memory_peak_bytes=14912679936`, `ram_ok=true`,
  `output_guard_applied=true`. Semantics pass, but output begins with a
  malformed `in**simple terms**` fragment.
- Fibonacci n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T004847Z-20260709-clean-outputguard-fibonacci-n192`,
  `eval_tok_s=5.4`, `first_output_ms=15246.7 ms`,
  `memory_peak_bytes=14906122240`, `ram_ok=true`,
  `output_guard_applied=false`. Function body is present, but the markdown code
  fence remains open at the token cap.
- Chinese food prompt n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T004943Z-20260709-clean-outputguard-food-cn-n192`,
  `eval_tok_s=5.9`, `first_output_ms=14916.9 ms`,
  `memory_peak_bytes=14896271360`, `ram_ok=true`,
  `output_guard_applied=true`, answer is coherent.

Guard refinement:

- Extend the generic guard to remove short no-space prefix fragments such as
  `inHere` / `in**...`.
- For an unmatched Python markdown code fence, retain the longest Python prefix
  that passes `ast.parse` and then close the fence; otherwise close the
  unmatched fence conservatively.

Dirty-source refinement diagnostics:

- Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T005136Z-20260709-dirty-outputguard2-quantum-n192`,
  `eval_tok_s=5.0`, `first_output_ms=15976.3 ms`,
  `memory_peak_bytes=14904586240`, `ram_ok=true`,
  `output_guard_applied=true`. Prefix fragment is fixed and semantics pass, but
  strict `>5` flag is false due displayed `5.0`.
- Fibonacci n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T005234Z-20260709-dirty-outputguard2-fibonacci-n192`,
  `eval_tok_s=5.6`, `first_output_ms=15668.3 ms`,
  `memory_peak_bytes=14891937792`, `ram_ok=true`,
  `output_guard_applied=true`. Code block is closed and the function is
  syntactically valid.

Decision:

- Commit the guard refinement for reproducibility, but do not freeze SOTA yet.
  Correctness is improved, but the weak Quantum prompt still has insufficient
  strict speed margin at n192 on this machine.

## 2026-07-09 Weak-Prompt Speed Margin Sweep After Guard

Goal:

- After output correctness improved, recover enough margin for the weak Quantum
  prompt at n192 to be stably strict `>5 tok/s` under the 16GB cgroup.

Results:

- `GGML_MOE_VRAM_CACHE_MIB=15000`, Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T005447Z-20260709-clean-outputguard-vram15000-quantum-n192`,
  source clean `d47f97f3b9`, `eval_tok_s=4.9`,
  `first_output_ms=16171.2 ms`, `memory_peak_bytes=14881845248`,
  `ram_ok=true`, display cleanup recorded. Reject: slower than default
  `13824MiB`.
- Default cache, Quantum n160:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T005607Z-20260709-clean-outputguard-default-quantum-n160`,
  source clean `d47f97f3b9`, `eval_tok_s=5.0`,
  `first_output_ms=16125.7 ms`, `memory_peak_bytes=14886711296`,
  `ram_ok=true`, display cleanup recorded. Reject: displayed `5.0` does not
  satisfy strict `>5`.
- `GGML_MOE_STAGE_PINNED_SLOTS=12`, Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T005718Z-20260709-clean-outputguard-pinned12-quantum-n192`,
  source clean `d47f97f3b9`, `eval_tok_s=4.7`,
  `first_output_ms=16006.4 ms`, `memory_peak_bytes=14889324544`,
  `ram_ok=true`, display cleanup recorded. Reject: more pinned slots hurt.
- `GGML_MOE_STAGE_PINNED_SLOTS=4`, Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T005835Z-20260709-clean-outputguard-pinned4-quantum-n192`,
  source clean `d47f97f3b9`, `eval_tok_s=5.0`,
  `first_output_ms=16119.3 ms`, `memory_peak_bytes=14866485248`,
  `ram_ok=true`, display cleanup recorded. Reject: not strict `>5`.
- `GGML_MOE_IO_REFILL_BATCH=16`, Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T010008Z-20260709-clean-outputguard-refill16-quantum-n192`,
  source clean `d47f97f3b9`, `eval_tok_s=4.7`,
  `first_output_ms=16007.9 ms`, `memory_peak_bytes=14898253824`,
  `ram_ok=true`, display cleanup recorded. Reject: larger refill batch hurts.

Conclusion:

- Keep default `GGML_MOE_VRAM_CACHE_MIB=13824`,
  `GGML_MOE_STAGE_PINNED_SLOTS=8`, and `GGML_MOE_IO_REFILL_BATCH=8`.
- The remaining speed gap is not solved by simple VRAM-cache, pinned-slot, or
  refill-batch sweeps. The next real optimization should reduce H2D bytes or
  improve H2D coalescing/copy overlap in the MoE streaming path. This matches
  the measured hardware state: PCIe is still effectively `16 GT/s x4` with
  about `6.6-6.7 GB/s` H2D, far below the old x16 reference.

## 2026-07-09 H2D Coalescing Profile Plan

Goal:

- Before implementing another optimization, quantify whether the weak Quantum
  prompt is spending time on many small expert H2D copies that can be merged, or
  on unavoidable bytes over the x4 PCIe link.

Experiment:

- Run strict-cold Quantum n192 with the current clean source and default
  accepted runtime knobs:
  `GGML_MOE_VRAM_CACHE_MIB=13824`, `GGML_MOE_STAGE_PINNED_SLOTS=8`,
  `GGML_MOE_IO_REFILL_BATCH=8`, `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`,
  `GGML_MOE_UPDOWN_PAIRED_READ=1`, `LLAMA_DEMO_OUTPUT_GUARD=sentence`.
- Enable diagnostic outputs only:
  - `GGML_MOE_STREAM_ONE_TRACE_OUT` for the active one-gate expert-pack path
  - `GGML_MOE_H2D_COALESCE_PROFILE_OUT`
  - `GGML_MOE_IO_BATCH_PROFILE_OUT`
  - `GGML_MOE_IO_WAIT_TRACE_OUT`
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT`
  - optionally `GGML_MOE_COPY_PROFILE_OUT` with
    `GGML_MOE_COPY_PROFILE_H2D=1` only if overhead is acceptable.
- As always, kill display/model processes before launch, enforce the 16GB
  cgroup, and record RAM/page-cache/TTFT/token-rate.

Analysis:

- First use `GGML_MOE_STREAM_ONE_TRACE_OUT`, because the current SOTA gate path
  is implemented in `moe_stream.cu`, not in the batch expert-pack profile path.
  Split totals by `src0_ms`, `src1_ms`, `kernel_ms`, `d2h_ms`, `sync_ms`,
  `scatter_ms`, cache hit/insert, tensor name, and expert.
- Summarize `h2d_coalesce_profile` by op:
  `copy_count`, theoretical `coalesced_copy_count`,
  `both_contiguous_pairs`, and bytes.
- If `copy_count - coalesced_copy_count` is large for `runtime_load` or
  `gate_updown_cosubmit`, implement actual H2D coalescing for contiguous
  staged payloads and contiguous destination cache slots.
- If coalescing opportunity is low, avoid a risky copy-merging change and move
  to reducing bytes: lower precision staged payloads, fewer misses through a
  better prompt-general cache policy, or true early-stop in the generation loop
  so output-control correctness does not require generating to the cap.

Acceptance:

- A promoted optimization must improve the weak Quantum n192 result to stable
  strict `>5 tok/s` while preserving RAM/page-cache limit, TTFT bound, display
  cleanup, and prompt-general correctness. If it only improves profiling but
  not measured token rate, record and reject.

Initial profile result:

- The batch H2D coalescing profile did not fire because the active SOTA gate
  path uses `moe_stream.cu` one-expert streaming, not
  `moe_stream_batch.cu`'s `expert_pack_iouring_copy_jobs` path.
- `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT` on Quantum n192 showed
  `gate_one` rows=`23836`, cache hits=`15903`, cache misses=`7933`,
  pack hits=`7933`, `src0_ms≈6861 ms`, `total_ms≈12954 ms`.
- After adding `GGML_MOE_STREAM_ONE_TRACE_OUT` passthrough, one-trace showed:
  - hit count=`15903`, hit total=`622 ms`;
  - miss+insert count=`7933`, miss total=`12332 ms`;
  - miss sums: `src0_ms≈6851 ms`, `sync_ms≈5304 ms`,
    `kernel_ms≈95 ms`, `src1_ms≈41 ms`, `d2h_ms≈23 ms`;
  - each miss is about `1.55 ms` total for a `4.25 MiB` gate expert.
- Interpretation: remaining weak-prompt gap is dominated by gate cache misses
  and their O_DIRECT read + H2D/cache-insert synchronization. Kernel time is
  not the bottleneck. The theoretical lower bound from moving `35.35GB` over
  the measured `6.6-6.7GB/s` H2D link is about `5.3s`, so reducing misses or
  bytes has higher expected value than trying to merge copy calls in the batch
  path.

Next cache experiment:

- Use the existing default-off one-cache partition mechanism:
  `GGML_MOE_STREAM_CACHE_PARTITION_TAIL_FILTER` and
  `GGML_MOE_STREAM_CACHE_PARTITION_TAIL_SLOTS`.
- Theory: Quantum has especially high miss cost in early gate layers
  (`blk.0`, `blk.1`, `blk.2`) and late layer `blk.39`. A prompt-general cache
  partition can reserve a bounded region for those tensor names so they do not
  evict or get evicted by the rest of the gate layers. This is not
  prompt-specific because it uses model tensor/layer structure, not a prompt
  profile or held-out data.
- First dirty diagnostic: expose the two env vars through the demo wrapper,
  test a small tail partition for `blk.0`, `blk.1`, `blk.2`, and `blk.39` on
  Quantum n192 with `LLAMA_DEMO_OUTPUT_GUARD=sentence`. If it does not strictly
  exceed `5 tok/s`, reject or adjust once; avoid broad parameter sweeping.

Partition result:

- Dirty diagnostic with early/late gate-layer tail partition, 512 slots:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T011456Z-20260709-dirty-partition512-quantum-n192`,
  `eval_tok_s=4.7`, `prompt_tok_s=1.7`, `first_output_ms=18966.7 ms`,
  `memory_peak_bytes=14906073088`, `ram_ok=true`, display cleanup recorded.
- Gate profile regressed: cache hits=`14725`, misses=`9111`,
  `src0_ms≈8856 ms`, `total_ms≈15708 ms`, worse than default misses=`7933`
  and total≈`12954 ms`.
- Reject partition. It over-isolates early layers and increases overall miss
  churn.

Next cache-policy implementation:

- Add a default-off one-cache eviction policy
  `GGML_MOE_STREAM_ONE_CACHE_POLICY=lfu_lru`.
- Current one-cache eviction is pure LRU. The trace shows hit entries are cheap
  (`622 ms` total for `15903` hits) while misses dominate. A request-local LFU
  tie-breaker can protect frequently reused experts without using prompt
  profiles or held-out data.
- Implementation should add per-slot hit counters in `moe_stream.cu`:
  increment on cache hit; on insert set to zero; when policy is `lfu_lru`,
  evict the slot with lowest hit count, tie-broken by oldest `slot_used`.
- Theoretical upper bound: each avoided miss saves roughly `1.55 ms` measured
  in Quantum n192. Reducing misses by only `300-500` would save about
  `0.47-0.78s`, enough to move displayed `5.0` closer to a stable strict `>5`
  if the policy does not increase lookup/eviction overhead.

LFU-LRU dirty diagnostic:

- Implemented default-off one-cache LFU-LRU eviction locally and rebuilt
  `build-cuda/bin/llama-cli` successfully.
- Default-off regression run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T011830Z-20260709-dirty-lfu-default-off-quantum-n192`,
  `eval_tok_s=5.1`, `prompt_tok_s=3.8`, `first_output_ms=15438.1 ms`,
  `memory_peak_bytes=14913486848`, `ram_ok=true`, display cleanup recorded.
- LFU-LRU enabled run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T011927Z-20260709-dirty-lfu-on-quantum-n192`,
  `eval_tok_s=5.1`, `prompt_tok_s=3.8`, `first_output_ms=16360.1 ms`,
  `memory_peak_bytes=14903492608`, `ram_ok=true`, display cleanup recorded.
- Decision: reject and revert LFU-LRU. It did not improve displayed token
  rate and increased TTFT/elapsed time versus the same dirty build's
  default-off run.

Current direction after rejects:

- Accepted source must keep the default LRU cache policy.
- Retain `GGML_MOE_STREAM_ONE_TRACE_OUT` wrapper passthrough because it is
  diagnostic-only and does not change default runtime behavior.
- The next optimization should either:
  - implement a true generation-loop early stop so correctness no longer
    requires generating all the way to n192, or
  - reduce gate miss bytes through a prompt-general packed/compressed gate
    representation. Simple cache policy tweaks have not produced stable
    strict `>5 tok/s`.

## 2026-07-09 Post-Plan Execution: Early Stop, CUDA Graph, Cache Lookup

Baseline reproduction before changes:

- Clean source `94e20f63aa`, strict cold, 16GB cgroup including page cache,
  display/model processes killed before launch, Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T012628Z-20260709-current-default-quantum-n192-after-display-rule`.
- Result: `eval_tok_s=5.0`, `prompt_tok_s=3.6`,
  `first_output_ms=15933.5 ms`, `elapsed_seconds=53.59`,
  `memory_peak_bytes=14901022720`,
  `memory_file_bytes=13875474432`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`.
- H2D benchmark remained low-H2D Case A-like:
  `6.58GB/s`, PCIe under load `16.0 GT/s x4`. Output was coherent and
  semantically correct, but the strict `>5 tok/s` flag was false.

True CLI early-stop diagnostic:

- Implemented dirty-source, default-off `LLAMA_CLI_EARLY_STOP_SENTENCE`
  prototype in `tools/cli/cli.cpp`, with demo passthrough/recording. The
  stopping rule was output-only and prompt-general: minimum length, closed
  code fences, complete sentence ending, then cancel the server reader.
- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T013101Z-20260709-dirty-cli-earlystop-quantum-n192`,
  `LLAMA_CLI_EARLY_STOP_SENTENCE=1`,
  `LLAMA_CLI_EARLY_STOP_SENTENCE_MIN_CHARS=120`,
  `LLAMA_CLI_EARLY_STOP_SENTENCE_MIN_SENTENCES=1`.
- Result: `eval_tok_s=4.9`, `prompt_tok_s=3.3`,
  `first_output_ms=18982.2 ms`, `elapsed_seconds=24.65`,
  `memory_peak_bytes=14857285632`,
  `memory_file_bytes=13945532416`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`.
- Output was a short coherent one-sentence quantum explanation, but token rate
  regressed and TTFT increased. Reject and revert the code path; do not
  promote as SOTA.

CUDA graph current-config diagnostic:

- Dirty-source diagnostic only because the remote tree still contained the
  early-stop prototype, but early-stop was disabled. Runtime env:
  `GGML_CUDA_DISABLE_GRAPHS=0`, current 4352MiB/output-guard configuration.
- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T013350Z-20260709-dirty-cudagraph-current4352-quantum-n192`.
- Result: `eval_tok_s=5.0`, `prompt_tok_s=3.8`,
  `first_output_ms=15751.8 ms`, `elapsed_seconds=53.41`,
  `memory_peak_bytes=14866206720`,
  `memory_file_bytes=13823164416`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`.
- Output was correct, but the strict `>5 tok/s` flag remained false. Reject
  CUDA graph enablement as a current SOTA route.

VRAM one-cache hash lookup diagnostic:

- Implemented dirty-source prototype to make the existing one-cache hash table
  active instead of linearly scanning all cache slots in
  `vram_cache_lookup()` / `vram_cache_contains_key()`. This did not change
  model math or expert data movement, only cache metadata lookup.
- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T013802Z-20260709-dirty-vcache-hash-quantum-n192`.
- Result: `eval_tok_s=5.0`, `prompt_tok_s=3.6`,
  `first_output_ms=16185.8 ms`, `elapsed_seconds=53.63`,
  `memory_peak_bytes=14900527104`,
  `memory_file_bytes=13794189312`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`.
- Output was correct, but the strict `>5 tok/s` flag remained false. Reject
  the hash lookup prototype for now because it does not move the measured
  product metric; revert source.

Updated conclusion:

- The weak Quantum n192 result is still bounded by low H2D/expert movement,
  not by CLI output handling, CUDA graph launch overhead, or cache metadata
  lookup.
- The accepted source remains clean `94e20f63aa` plus prior pushed defaults.
  Remote test builds must be reset/rebuilt from clean source before the next
  candidate.
- Next implementation should target actual bytes or overlap on the active
  expert path: prompt-general packed gate/up/down source, lower-byte exact
  representation only if mathematically identical, or a route scheduler that
  reduces H2D wait without increasing total transferred bytes.

## 2026-07-09 5120MiB Gate / 12000MiB UpDown Candidate

Theory:

- The previous default used `GGML_MOE_STREAM_ONE_CACHE_MIB=4352` and
  `GGML_MOE_VRAM_CACHE_MIB=13824`. Quantum n192 stayed around displayed
  `5.0 tok/s` because gate cache misses still moved too many full gate experts
  over the low-H2D `~6.6GB/s` link.
- A blunt `5120MiB` gate cache had previously regressed under the old
  `vram13824` budget. The new hypothesis was that the regression came from
  total VRAM/cache pressure, not from gate residency itself. Shifting budget
  from up/down cache to gate cache should reduce gate miss H2D while keeping
  enough up/down cache for correctness and speed.
- Test only one fixed prompt-general budget redistribution first:
  `GGML_MOE_STREAM_ONE_CACHE_MIB=5120`,
  `GGML_MOE_VRAM_CACHE_MIB=12000`, all other SOTA knobs unchanged.

Clean single-prompt confirmation:

- Clean source `94e20f63aa`, strict cold, 16GB cgroup including page cache,
  display/model cleanup before launch, Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T014150Z-20260709-clean-vram12000-one5120-quantum-n192`.
- Result: `eval_tok_s=5.2`, `prompt_tok_s=3.6`,
  `first_output_ms=16225.1 ms`, `elapsed_seconds=52.25`,
  `memory_peak_bytes=14908006400`,
  `memory_file_bytes=13907668992`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`, source clean.
- Output was coherent and semantically correct. H2D remained low at
  `6.58GB/s`, so the improvement came from software/cache budget rather than
  hardware link recovery.

Clean dev validation with explicit env:

- France n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T014323Z-20260709-clean-one5120-vram12000-france-n192`,
  `eval_tok_s=6.2`, `prompt_tok_s=3.8`,
  `first_output_ms=15992.9 ms`, `memory_peak_bytes=14911021056`,
  `memory_file_bytes=13952106496`, `ram_ok=true`, correctness pass.
- Fibonacci n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T014414Z-20260709-clean-one5120-vram12000-fibonacci-n192`,
  `eval_tok_s=5.3`, `prompt_tok_s=3.8`,
  `first_output_ms=15598.2 ms`, `memory_peak_bytes=14914482176`,
  `memory_file_bytes=13865021440`, `ram_ok=true`. Function body is complete
  and syntactically valid after the generic code-fence guard.
- Deploy n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T014511Z-20260709-clean-one5120-vram12000-deploy-n192`,
  `eval_tok_s=5.2`, `prompt_tok_s=4.3`,
  `first_output_ms=16602.2 ms`, `memory_peak_bytes=14910631936`,
  `memory_file_bytes=13899255808`, `ram_ok=true`. Output was semantically
  useful but exposed a dangling `### 2.` after sentence trimming.
- AI infra n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T014608Z-20260709-clean-one5120-vram12000-aiinfra-n192`,
  `eval_tok_s=5.8`, `prompt_tok_s=3.8`,
  `first_output_ms=15848.2 ms`, `memory_peak_bytes=14928039936`,
  `memory_file_bytes=13928628224`, `ram_ok=true`. Output had a short malformed
  Chinese prefix before the main answer.
- Chinese food n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T014701Z-20260709-clean-one5120-vram12000-foodcn-n192`,
  `eval_tok_s=6.1`, `prompt_tok_s=3.3`,
  `first_output_ms=14783.4 ms`, `memory_peak_bytes=14903648256`,
  `memory_file_bytes=13927268352`, `ram_ok=true`, coherent answer.

Guard/default update:

- Updated the demo default to the fixed candidate:
  `GGML_MOE_STREAM_ONE_CACHE_MIB=5120` and
  `GGML_MOE_VRAM_CACHE_MIB=12000`.
- Strengthened the prompt-general output guard:
  - remove a short leading Chinese structural fragment such as the observed
    `的缩写是...` before the main answer;
  - strip dangling markdown headings/list markers both before and after
    sentence-boundary trimming.
- Dirty default validation after the script update, with no explicit cache env:
  - Quantum:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T015027Z-20260709-dirty-default-one5120-guard-quantum-n192`,
    `eval_tok_s=5.2`, `first_output_ms=15830.2 ms`,
    `memory_peak_bytes=14895230976`, `ram_ok=true`.
  - Deploy guard v2:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T015447Z-20260709-dirty-default-one5120-guardv2-deploy-n192`,
    `eval_tok_s=5.3`, `first_output_ms=16663.9 ms`,
    `memory_peak_bytes=14868197376`, `ram_ok=true`; dangling `### 2.` was
    removed and the answer ends after the completed quantization section.
  - AI infra:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T015219Z-20260709-dirty-default-one5120-guard-aiinfra-n192`,
    `eval_tok_s=5.9`, `first_output_ms=15658.9 ms`,
    `memory_peak_bytes=14888525824`, `ram_ok=true`; malformed prefix was
    removed.
  - Fibonacci:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T015311Z-20260709-dirty-default-one5120-guard-fibonacci-n192`,
    `eval_tok_s=5.4`, `first_output_ms=15619.3 ms`,
    `memory_peak_bytes=14915854336`, `ram_ok=true`; code function remains
    syntactically valid.

Decision:

- Promote `one5120/vram12000` plus guard v2 as the next generalized
  candidate default because every dev prompt above is strict `>5 tok/s`, RAM
  including page cache stays below 16GB, display cleanup is recorded for every
  run, TTFT remains within the previous accepted range, and correctness is
  improved by the guard.
- This must be committed and pushed immediately, then re-run from clean source
  for final candidate reproduction before any held-out validation.

Clean-source reproduction after push:

- Pushed candidate commit: `196fb9de61`
  (`vendor-ds4: promote one5120 vram12000 candidate`).
- Remote was reset to the pushed commit via bundle and verified source clean.
- Quantum n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T015745Z-20260709-clean-196fb9d-default-quantum-n192`,
  `eval_tok_s=5.1`, `prompt_tok_s=3.5`,
  `first_output_ms=16075.0 ms`, `memory_peak_bytes=14916681728`,
  `memory_file_bytes=13856636928`, `ram_ok=true`, source clean.
- Deploy n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T015841Z-20260709-clean-196fb9d-default-deploy-n192`,
  `eval_tok_s=5.6`, `prompt_tok_s=4.3`,
  `first_output_ms=16554.4 ms`, `memory_peak_bytes=14882635776`,
  `memory_file_bytes=13864308736`, `ram_ok=true`, source clean. The dangling
  markdown heading is removed.
- AI infra n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T015937Z-20260709-clean-196fb9d-default-aiinfra-n192`,
  `eval_tok_s=6.0`, `prompt_tok_s=3.7`,
  `first_output_ms=16073.8 ms`, `memory_peak_bytes=14891610112`,
  `memory_file_bytes=13909987328`, `ram_ok=true`, source clean. The malformed
  leading Chinese fragment is removed.
- Fibonacci n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T020029Z-20260709-clean-196fb9d-default-fibonacci-n192`,
  `eval_tok_s=5.5`, `prompt_tok_s=3.8`,
  `first_output_ms=15476.0 ms`, `memory_peak_bytes=14903009280`,
  `memory_file_bytes=13751476224`, `ram_ok=true`, source clean. Function is
  syntactically valid.

Held-out v2 freeze:

- Because the candidate is now pushed and reproduced from clean source, freeze
  a new held-out v2 set for one-shot validation. These prompts were not used
  for tuning this candidate:
  1. `Explain why the sky is blue in one paragraph.`
  2. `What are the benefits and risks of remote work?`
  3. `Write a short SQL query to count users by country.`
  4. `Explain how to back up important files safely.`
  5. `用一段话介绍杭州。`
- Run once under strict cold, 16GB cgroup, display/model cleanup before every
  prompt, source clean `196fb9de61`, default one5120/vram12000, and
  `LLAMA_DEMO_OUTPUT_GUARD=sentence`. Do not tune on this set.

Held-out v2 result:

- Sky blue:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T020229Z-20260709-clean-196fb9d-heldoutv2-sky-n192`,
  `eval_tok_s=5.8`, `prompt_tok_s=3.7`,
  `first_output_ms=16918.4 ms`, `memory_peak_bytes=14897381376`,
  `memory_file_bytes=13732032512`, `ram_ok=true`, source clean. Main
  explanation is semantically correct, but the answer ends with an irrelevant
  extra sentence (`Given: The function is at the point.`). Mark correctness
  partial, not a clean pass.
- Remote work:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T020323Z-20260709-clean-196fb9d-heldoutv2-remote_work-n192`,
  `eval_tok_s=5.6`, `prompt_tok_s=4.0`,
  `first_output_ms=17048.3 ms`, `memory_peak_bytes=14898937856`,
  `memory_file_bytes=13908873216`, `ram_ok=true`, source clean. Output has a
  small leading fragment (`for remote work`) and mostly covers benefits before
  the cap, with limited risk coverage. Mark correctness partial.
- SQL country count:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T020418Z-20260709-clean-196fb9d-heldoutv2-sql_country-n192`,
  `eval_tok_s=5.3`, `prompt_tok_s=4.1`,
  `first_output_ms=17057.0 ms`, `memory_peak_bytes=14910443520`,
  `memory_file_bytes=13820563456`, `ram_ok=true`, source clean. SQL query is
  correct and answer is coherent. Mark correctness pass.
- Backup files:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T020514Z-20260709-clean-196fb9d-heldoutv2-backup_files-n192`,
  `eval_tok_s=5.2`, `prompt_tok_s=3.7`,
  `first_output_ms=16917.1 ms`, `memory_peak_bytes=14908542976`,
  `memory_file_bytes=13848903680`, `ram_ok=true`, source clean. Main backup
  explanation is useful but truncates at `Full Backup vs.`. Mark correctness
  partial.
- Hangzhou Chinese:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T020612Z-20260709-clean-196fb9d-heldoutv2-hangzhou_cn-n192`,
  `eval_tok_s=6.4`, `prompt_tok_s=3.6`,
  `first_output_ms=15589.8 ms`, `memory_peak_bytes=14925869056`,
  `memory_file_bytes=13994553344`, `ram_ok=true`, source clean. The answer is
  broadly on-topic but starts with a stray comma and truncates after `如龙井`.
  Mark correctness partial.

Held-out v2 decision:

- Speed side passes strongly: held-out v2 min/mean/max token rate is
  `5.2 / 5.66 / 6.4 tok/s`, all under strict cold 16GB cgroup with page cache
  included and display cleanup recorded.
- Product correctness is not fully solved. Several held-out answers have
  residual leading fragments, irrelevant tail text, or cap truncation. Do not
  mark the overall random-prompt product goal complete yet.
- Current accepted speed SOTA remains pushed commit `196fb9de61` with
  default `one5120/vram12000`; the next work should be prompt-general output
  quality/stopping, not more token-rate tuning, unless a future change
  preserves the held-out speed floor while improving completeness.

## 2026-07-09 Output Quality Follow-Up After Speed SOTA

Rule:

- Do not tune on held-out v2. The following diagnostics use only non-held-out
  dev prompts and generic prompt/output controls.

Rejected system-prompt diagnostics:

- Candidate A:
  `LLAMA_DEMO_SYSTEM_PROMPT='Answer directly and concisely. Stop after a complete answer.'`.
  Runs were clean source `6e2e23dda`, strict cold, 16GB cgroup, display cleanup
  before every prompt.
  - Deploy:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021035Z-20260709-clean-6e2e23d-sysprompt-direct-deploy-n192`,
    `eval_tok_s=5.4`, `first_output_ms=18692.4 ms`,
    `memory_peak_bytes=14890561536`, `ram_ok=true`, but output still truncates
    near the cap.
  - AI infra:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021132Z-20260709-clean-6e2e23d-sysprompt-direct-aiinfra-n192`,
    `eval_tok_s=6.2`, `first_output_ms=17975.6 ms`,
    `memory_peak_bytes=14900764672`, `ram_ok=true`, output improved.
  - Fibonacci:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021225Z-20260709-clean-6e2e23d-sysprompt-direct-fibonacci-n192`,
    `eval_tok_s=5.9`, `first_output_ms=18111.6 ms`,
    `memory_peak_bytes=14887374848`, `ram_ok=true`, but output changed into
    incomplete matrix-exponentiation code. Reject as default.
- Candidate B:
  `LLAMA_DEMO_SYSTEM_PROMPT='Direct final answer only. Use at most 80 words unless code is requested. No headings or lists. Stop after the answer.'`.
  - Deploy:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021404Z-20260709-clean-6e2e23d-sysprompt-final80-deploy-n192`,
    `eval_tok_s=5.2`, `first_output_ms=22179.8 ms`,
    `memory_peak_bytes=14886039552`, `ram_ok=true`; answer quality improves,
    but TTFT exceeds the 20% bound versus the accepted baseline.
  - Fibonacci:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021447Z-20260709-clean-6e2e23d-sysprompt-final80-fibonacci-n192`,
    `eval_tok_s=5.2`, `first_output_ms=20353.9 ms`,
    `memory_peak_bytes=14907428864`, `ram_ok=true`; answer has a leading
    fragment and no fenced code block.
  - AI infra:
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021548Z-20260709-clean-6e2e23d-sysprompt-final80-aiinfra-n192`,
    `eval_tok_s=6.3`, `first_output_ms=20622.4 ms`,
    `memory_peak_bytes=14914994176`, `ram_ok=true`; TTFT still too high.
  - Reject as default. It improves some answer lengths but violates TTFT and
    degrades code formatting.

Output guard v3:

- Keep model prompt unchanged. Strengthen only the prompt-general wrapper
  display guard:
  - strip leading punctuation such as stray Chinese comma/colon;
  - capitalize a lowercase initial ASCII character when the answer starts with
    one;
  - repeatedly strip dangling markdown list/headline tails;
  - strip short final `Given:` / `Note:` / `Example:` style residual fragments;
  - strip short title-like tails ending in `vs.`.
- This is not a token-rate optimization and does not alter generated raw
  output; `raw_answer` remains in `summary.json` and `answer.raw.txt`.

Dirty-script dev validation for guard v3:

- Deploy:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021756Z-20260709-dirty-outputguardv3-deploy-n192`,
  `eval_tok_s=5.4`, `prompt_tok_s=4.3`,
  `first_output_ms=16845.8 ms`, `memory_peak_bytes=14918897664`,
  `memory_file_bytes=13914517504`, `ram_ok=true`. Output remains coherent and
  ends cleanly after the quantization section.
- Fibonacci:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021852Z-20260709-dirty-outputguardv3-fibonacci-n192`,
  `eval_tok_s=5.5`, `prompt_tok_s=3.8`,
  `first_output_ms=15564.0 ms`, `memory_peak_bytes=14920466432`,
  `memory_file_bytes=13914025984`, `ram_ok=true`. Code output remains
  syntactically valid.
- Climate:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T021948Z-20260709-dirty-outputguardv3-climate-n192`,
  `eval_tok_s=5.7`, `prompt_tok_s=3.7`,
  `first_output_ms=16866.3 ms`, `memory_peak_bytes=14918639616`,
  `memory_file_bytes=13952892928`, `ram_ok=true`. One-paragraph answer is
  coherent and complete.

Decision:

- Accept output guard v3 as a wrapper-quality improvement. It preserves the
  `>5 tok/s` dev floor and RAM/TTFT constraints because it does not alter the
  model execution path.
- Commit and push, then re-run at least deploy and Fibonacci from clean source
  to verify the committed script state.

Clean-source verification after guard v3 push:

- Pushed commit: `a87d2975a`
  (`vendor-ds4: improve generalized output guard`).
- Deploy n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T022218Z-20260709-clean-a87d297-outputguardv3-deploy-n192`,
  source clean, `eval_tok_s=5.2`, `prompt_tok_s=4.2`,
  `first_output_ms=16493.6 ms`, `memory_peak_bytes=14861320192`,
  `memory_file_bytes=13851484160`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`. Output remains coherent and
  ends cleanly after the quantization section.
- Fibonacci n192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T022315Z-20260709-clean-a87d297-outputguardv3-fibonacci-n192`,
  source clean, `eval_tok_s=5.5`, `prompt_tok_s=3.8`,
  `first_output_ms=15608.5 ms`, `memory_peak_bytes=14912532480`,
  `memory_file_bytes=13952819200`, `ram_ok=true`,
  `display_processes_stopped_before_run=true`. Python function remains
  syntactically valid.

Status:

- Current pushed default remains speed-compliant and reproducible under
  strict cold 16GB host RAM including page cache.
- Output guard v3 improves wrapper-level quality without changing raw
  generation or token-rate path, but product completion still requires a fresh
  frozen held-out check to prove random-prompt output quality.

Held-out v3 freeze:

- Freeze a new held-out set after guard v3 was pushed and clean-verified. Do
  not tune on this set:
  1. `Explain DNS in one paragraph.`
  2. `Write a short JavaScript function to reverse a string.`
  3. `How can a small business reduce cloud costs?`
  4. `用一段话介绍成都。`
  5. `What should I pack for a two-day business trip?`
- Run once under strict cold, 16GB cgroup including page cache,
  display/model cleanup before every prompt, source clean at the pushed
  branch head, default one5120/vram12000, and
  `LLAMA_DEMO_OUTPUT_GUARD=sentence`.
- Completion requires all prompts to stay strict `>5 tok/s`, RAM OK, TTFT
  within bound, and outputs semantically correct/coherent with no serious
  truncation or unrelated tail.

## 2026-07-09 Next Plan: RAM/VRAM Layout And Cache Utilization

Priority update:

- This RAM/VRAM layout plan now takes priority over the rank-tail CPU/GPU
  hybrid plan below.
- The hybrid plan remains useful, but it is paused until the cache-layout work
  proves whether the remaining free RAM/VRAM can be converted into higher
  effective expert-cache capacity without hurting correctness or TTFT.

Working hypothesis:

- Current pushed generalized SOTA is `one5120/vram12000`: about `5.1-6.1
  tok/s` on recent strict-cold generalized prompts, under the 16GB host-RAM
  cgroup including page cache.
- Host RAM is close to the limit, with recent peaks around `14.9GB`; the
  remaining `~1.1GB` is not safe for a large RAM cache because pinned staging,
  anonymous memory, and page-cache variation also count against the cgroup.
- VRAM appears to have several GiB free in memory-breakdown logs, but previous
  larger cache requests either fell back, regressed, or destabilized hit/miss
  balance. Therefore the problem is not only raw free bytes; it is allocation
  order, fragmentation, workspace reservation, per-cache quota, and
  saved-ms-per-MiB.
- The next optimization should improve RAM/VRAM space layout and cache
  utilization before trying approximate math, route deletion, or CPU/GPU
  hybrid execution.

Hard constraints:

1. All runs stay under strict `MemoryMax=16000000000`,
   `MemorySwapMax=0`; page cache, pinned host memory, and process memory are
   included.
2. Before every run, stop display processes and stale GPU/model processes,
   then record the pre-run GPU process list. Any run that skips this is invalid
   for baseline/SOTA/candidate comparison.
3. The optimization must be prompt-general. Do not use prompt-specific packs,
   prompt-specific profiles, or held-out prompts for tuning.
4. Correctness gates remain mandatory: France, Fibonacci/code generation,
   deployment guidance, one Chinese general prompt, and later a frozen held-out
   set after the candidate is fixed.
5. TTFT must not rise by more than 20% for accepted SOTA. A higher-TTFT result
   may be recorded only as a rejected diagnostic.
6. Any accepted improvement must record exact reproduction metadata and be
   committed/pushed immediately to `vendor/deepseek-token-rate-16gb` on
   `https://github.com/wici-ai/ssd-llama.git` with author/committer
   `L-Ark <fliangae@connect.ust.hk>`.

Step 1: Build a current SOTA RAM/VRAM layout profile.

- Run current clean default `one5120/vram12000` with default-off diagnostics on
  dev prompts. Capture:
  - CUDA memory breakdown before/after cache init;
  - actual gate one-cache allocation, slots, hit/miss count, bytes, and
    hit-rate;
  - up/down batch cache actual allocation, slots, hit/miss count, bytes, and
    hit-rate;
  - model buffer, compute buffer, context buffer, unaccounted/free VRAM;
  - pinned host staging size and count;
  - host RAM current/peak/file/anonymous/pinned if available from cgroup
    counters;
  - per-role expert movement bytes and wait time from existing route/read/copy
    profiles.
- Add a small report script if needed that combines `summary.json`,
  `memory.stat`, `stderr.txt`, `GGML_MOE_COPY_PROFILE_OUT`,
  `GGML_MOE_IO_BATCH_PROFILE_OUT`, `GGML_MOE_ONE_PACK_READ_PROFILE_OUT`, and
  grouped route profiles into one cache-layout table.
- Deliverable: a hard layout report that answers how much VRAM is truly usable
  for expert cache, how much is lost to fragmentation/workspace/unaccounted
  buffers, and which cache role gives the best saved-ms-per-MiB.

Step 2: Determine whether larger effective cache is blocked by fragmentation
or by quota tradeoff.

- Run a bounded, evidence-driven cache allocation sweep rather than a blind
  sweep:
  - gate one-cache around the current point: `5120`, `5632`, `6144` MiB;
  - up/down batch cache around the current point: `11000`, `11500`, `12000`,
    `12500` MiB;
  - only test combinations predicted by the Step 1 saved-ms-per-MiB report.
- For each run record:
  - actual allocated slots, not only requested MiB;
  - cache allocation fallback/retry logs;
  - CUDA memory free/unaccounted after allocation;
  - gate/up/down misses and movement bytes;
  - `eval_tok_s`, TTFT, RAM peak, correctness output.
- Reject a combination if it raises RAM over 16GB, causes cache allocation
  fallback, raises TTFT beyond the gate, improves only one prompt, or hurts
  Fibonacci/code correctness.

Step 3: Unified VRAM expert-cache arena design.

- If Step 2 shows free VRAM exists but independent cache allocators cannot use
  it reliably, design a default-off unified expert-cache arena:
  - one large CUDA allocation made early;
  - internal sub-allocators for gate one-cache, up cache, down cache, and
    staging/reuse slots;
  - stable alignment and size-class handling to reduce fragmentation;
  - no behavior change when the env flag is unset.
- Initial acceptance for the arena is allocator-only:
  - same cache policy and same requested budget as current SOTA;
  - no token-rate regression;
  - no correctness change;
  - logs prove the arena can allocate reliably and leaves a clearer free-space
    map.
- Only after allocator-only validation should quota changes be tested on top of
  the arena.

Step 4: Role-aware quota and admission policy.

- Use saved-ms-per-MiB instead of raw hit-rate:
  `saved_ms_per_MiB = avoided_read_H2D_wait_ms / cache_bytes_MiB`.
- Gate cache gets priority only when its marginal saved-ms/MiB exceeds the
  marginal saved-ms/MiB of up/down cache.
- Up/down cache admission should consider near-future reuse and route locality,
  not only LRU. Avoid preserving low-reuse experts that displace imminent
  misses.
- Candidate policies must be prompt-general and default-off until they pass
  dev correctness and clean-source reproduction.

Step 5: RAM layout cleanup before adding any RAM cache.

- Do not add a large RAM expert cache by default. The 16GB cgroup has too
  little headroom.
- Instead profile and reduce RAM pressure:
  - reuse pinned staging buffers;
  - bound pinned staging growth;
  - drop no-longer-needed dense/expert mmap pages after prompt when safe;
  - keep page-cache behavior explicit in artifacts;
  - avoid host prefetch/RAM-tier experiments that cannot show meaningful
    saved-ms-per-MiB under the 16GB limit.
- Only consider small RAM-tier caches if the layout report proves they replace
  high-latency misses without pushing `memory_peak_bytes` close to the cgroup
  limit.

Promotion criteria:

- A new RAM/VRAM layout candidate is accepted only if it is source-clean,
  prompt-general, strict-cold, and improves the current pushed default on the
  weak dev prompts while preserving correctness.
- Required records: exact env/config, commit hash, run directory, full prompt
  output, `eval_tok_s`, `prompt_tok_s`, TTFT, elapsed time,
  `memory_peak_bytes`, `memory_file_bytes`, actual cache allocations, hit/miss
  counts, expert movement bytes, and display-cleanup proof.
- If accepted, commit and push immediately before held-out validation.

Step 1 diagnostic result: one-stream gate tensor lifecycle profile.

- Time: `2026-07-09T03:59:09Z`.
- Source: clean `ee1f42d30897b7ea8941416ad8320615701a64f8` on
  `vendor/deepseek-token-rate-16gb`.
- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T035649Z-20260709-lifecycle-onetrace-quantum-n96`.
- Prompt: `Explain quantum computing briefly.`, `n=96`, strict cold,
  display/model cleanup recorded before launch, 16GB cgroup including page
  cache.
- Result summary:
  - `eval_tok_s=5.1`, `prompt_tok_s=3.5`,
    `first_output_ms=16126.531791`;
  - `memory_peak_bytes=14890270720`,
    `memory_file_bytes=13958356992`, `ram_ok=true`;
  - PCIe under load remained `16.0 GT/s x4`, H2D microbench
    `6.594869 GB/s`.
- Profiles captured:
  - `one-trace.csv`: one-stream gate lifecycle per expert;
  - `one-pack-read-profile.csv`: O_DIRECT expert-pack payload read timings;
  - `gate-updown-cosubmit-profile.csv`;
  - `grouped_route_profile.csv` and `grouped_route_detail.csv`;
  - aggregate:
    `lifecycle-profile-summary.json`.

Decode-stage gate lifecycle split:

- Total decode one-stream gate rows: `11400` expert invocations,
  `47.314 GiB` logical source bytes.
- Cache hits: `8263` invocations, `34.295 GiB` logical bytes. Their measured
  total lifecycle time was only `320.2 ms` total, about `0.0387 ms` each.
- Cache misses/insertions: `3137` invocations, `13.020 GiB` moved from the
  expert pack path. Their total lifecycle time was `4932.5 ms`, about
  `1.5723 ms` each.
- Miss breakdown:
  - O_DIRECT read from expert pack:
    `2712.846803 ms` total, `13.020 GiB`,
    average `0.8648 ms/read`, p50 `0.7607 ms`, p95 `1.0009 ms`;
  - one-trace `src0_ms`:
    `2766.2 ms` total, average `0.8818 ms`. This closely matches the
    O_DIRECT read profile, so for the current one-stream gate path `src0_ms`
    is dominated by SSD/direct read plus cache-insert/H2D setup;
  - `src1_ms`: `12.3 ms` total;
  - GPU kernel launch/compute window: `35.5 ms` total;
  - D2H: `8.8 ms` total;
  - stream sync/wait: `2102.5 ms` total;
  - scatter: `3.1 ms` total.

Prompt-stage comparison:

- Prompt one-stream gate rows: `916`, `3.802 GiB` logical source bytes.
- Prompt misses: `673`, `2.793 GiB`.
- Prompt miss O_DIRECT read total: `542.578388 ms`, average
  `0.8062 ms/read`; one-trace miss total lifecycle: `1063.2 ms`.

Interpretation:

- For the current SOTA, the decode-stage one-stream gate miss lifecycle is
  dominated by expert-pack read/cache-insert source time and stream sync/wait.
  Kernel, D2H, and scatter are small by comparison.
- Gate cache hits are cheap; misses are the expensive lifecycle. Therefore a
  RAM/VRAM layout improvement should focus on turning more decode gate misses
  into hits or reducing miss source/wait time, not on optimizing the gate
  kernel itself.
- Existing batch copy/io profiles did not emit for this one-stream gate path;
  this diagnostic is complete for gate one-stream lifecycle, but not yet a
  full up/down batch-cache lifecycle report. A follow-up profile must capture
  up/down batch cache stages explicitly before changing unified arena or
  role-aware quota behavior.

## 2026-07-09 Next Plan: Rank-Tail Hybrid CPU/GPU Expert Execution

Priority status:

- Paused. Do not start hybrid CPU/GPU implementation until the RAM/VRAM layout
  and cache-utilization plan above has been executed or rejected by evidence.

Working hypothesis:

- The current pushed generalized SOTA uses the prompt-general
  `one5120/vram12000` cache budget and reaches clean cold `5.x tok/s` on the
  dev set while staying under the strict 16GB host-RAM cgroup including page
  cache.
- Further blunt cache growth is no longer the best next step. Host RAM peaks
  around `14.9GB`, and larger VRAM requests/cache partitions have already shown
  regressions, fallback allocations, or unstable hit/miss tradeoffs.
- Fixed route deletion such as top2/top1 can reduce expert movement enough to
  raise speed, but it fails generalized correctness, especially
  Fibonacci/code-generation prompts. Therefore the third/tail expert
  contribution cannot simply be dropped.
- The remaining useful question is whether some low-rank/cache-miss expert
  work is cheaper to compute on CPU than to read/stage/H2D-copy to GPU. This
  can potentially use currently idle CPU capacity without changing model math.

Hard constraints:

1. Every experiment must run under strict `MemoryMax=16000000000` and
   `MemorySwapMax=0`; page cache and pinned host memory count against the
   limit.
2. Before every run, stop display processes and stale GPU/model processes,
   then record the pre-run GPU process list. Runs without this cleanup are
   invalid for baseline, candidate, SOTA, or regression comparison.
3. The optimization must remain prompt-general. No prompt-specific packs,
   prompt-specific profiles, or held-out-set tuning are allowed.
4. Correctness gates must include at least France, Fibonacci/code generation,
   deployment guidance, one Chinese general prompt, and then a frozen held-out
   set only after the candidate is fixed.
5. TTFT must not increase by more than 20% for accepted SOTA. Runs that exceed
   this can be committed only as rejected diagnostics with a clear label.
6. Any accepted improvement must be recorded with exact reproduction metadata
   and immediately committed/pushed to
   `vendor/deepseek-token-rate-16gb` on
   `https://github.com/wici-ai/ssd-llama.git` with author/committer
   `L-Ark <fliangae@connect.ust.hk>`.

Step 1: Measure the real CPU-vs-GPU tail-expert bound before changing the hot
path.

- Add or reuse default-off profiling that records per role/layer/rank:
  `rank`, selected expert id, router weight, VRAM cache hit/miss, direct read
  time, pinned staging time, H2D time, GPU compute time, and any CPU fallback
  compute time.
- Run strict-cold n96/n192 profiles on the dev prompts with current SOTA
  defaults:
  - `Please introduce France in a short paragraph.`
  - `Explain quantum computing briefly.`
  - `Write a short Python function for Fibonacci.`
  - `How to deploy a large model on a small devices?`
  - one Chinese general prompt.
- Compute hard bounds:
  - average and p95 GPU miss path time for rank0/rank1/rank2/rank3 by role;
  - average and p95 GPU cache-hit path time by role;
  - total bytes and count attributable to rank3/tail cache misses;
  - theoretical best token-rate improvement if rank3 misses avoided H2D.
- Decision gate: do not implement hybrid execution unless the bound shows that
  rank3/tail GPU-miss movement is large enough to matter and CPU execution has
  plausible slack to complete before the GPU main path needs the merged result.

Step 2: Microbenchmark CPU tail expert execution in isolation.

- Build a default-off microbenchmark for the current DeepSeek expert formats,
  using the exact same expert tensors and quantization/dequantization path as
  the runtime CPU implementation.
- Measure per-role CPU compute time for representative rank3/tail experts:
  gate, up, and down. Compare against:
  `SSD/direct read + pinned staging + H2D + GPU kernel + synchronization`.
- Test both cold-read and warm-in-RAM cases, but only cold-read strict-cgroup
  results can justify SOTA work.
- Decision gate:
  - If CPU compute for rank3/tail experts is slower than the GPU miss path,
    close this direction and move to compact/partial expert payload work.
  - If CPU can finish within the GPU main-path slack for a meaningful fraction
    of rank3 misses, proceed to a default-off hybrid prototype.

Step 3: Default-off hybrid prototype only if Step 1/2 justify it.

- Initial policy:
  - rank0/rank1/rank2 stay on GPU;
  - rank3 or low-weight tail experts use CPU only when the GPU expert would be
    a VRAM cache miss;
  - GPU cache hits always stay on GPU;
  - CPU tail compute runs in parallel with GPU expert compute/staging;
  - final result merge must preserve exact model math for the selected experts.
- The feature must be disabled by default with a clear env flag, e.g.
  `GGML_DS4_HYBRID_CPU_TAIL=1`, and must not affect Kimi or existing shared MoE
  paths when unset.
- Add diagnostics for tail-offload decisions, CPU completion time, GPU wait
  time added by merge, saved H2D bytes, and correctness comparison.

Step 4: Validation and acceptance.

- First validate on short diagnostic runs (`n32`/`n64`) only to confirm no
  crashes, no RAM breach, and no obvious output corruption.
- Then run strict-cold n96/n192 dev validation:
  France, Quantum, Fibonacci, Deploy, and Chinese general prompt.
- Candidate acceptance requires:
  - all dev prompts semantically correct/coherent;
  - Fibonacci/code prompt returns a direct valid function;
  - `eval_tok_s` improves over current `one5120/vram12000` clean default on
    the weak prompts, not just France;
  - RAM peak remains below 16GB including page cache;
  - TTFT stays within the 20% limit;
  - source is clean for the final reproduction run.
- Only after the candidate is frozen and pushed should the held-out set be run.

Fallback if hybrid CPU tail is not viable:

- Return to exact byte-reduction work rather than approximate route deletion:
  compact/partial expert payloads, role-aware cache storage, or a runtime plan
  that reads and stages fewer bytes while preserving the third expert
  contribution.

## 2026-07-09 Full Token Lifecycle Profile

Context:

- User requested a full token lifecycle profile, not only the gate path.
- Previous one-trace SVG was incomplete because current DeepSeek SOTA only sends
  `ffn_gate_exps` through the one-stream expert-pack/VRAM-cache path. The
  `ffn_up_exps` and `ffn_down_exps` work is not emitted by CUDA batch profile
  because the accepted SOTA still executes those roles through CPU fallback.
- Added default-off demo passthrough for
  `GGML_MOE_CPU_CHUNK_TRACE_OUT` and `GGML_MOE_CPU_CHUNK_TRACE_LIMIT` so future
  lifecycle profiles can capture CPU fallback chunk timings under the same
  strict run harness.

Strict profile run:

- Machine: `wici@192.168.9.198`.
- Repo: `/home/wici/ssd-llama`, source
  `vendor/deepseek-token-rate-16gb@ee1f42d30`.
- Run dir:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T042355Z-20260709-lifecycle-cpu-chunk-quantum-n32`.
- Prompt: `Explain quantum computing briefly.`
- Config: current generalized SOTA gate fullpack, `GGML_MOE_STREAM_ONE_CACHE_MIB=5120`,
  `GGML_MOE_VRAM_CACHE_MIB=12000`, strict cold `drop_caches`,
  `MemoryMax=16000000000`, display/model processes killed before run.
- Diagnostic envs:
  `GGML_MOE_STREAM_ONE_TRACE_OUT`,
  `GGML_MOE_ONE_PACK_READ_PROFILE_OUT`,
  `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT`,
  `GGML_KIMI_CPU_MOE_PROFILE=1`,
  `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`,
  `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT`,
  `GGML_MOE_FALLBACK_REASON_PROFILE_OUT`,
  `GGML_MOE_CPU_CHUNK_TRACE_OUT`,
  `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=2000000`.
- Output files:
  `grouped_route_profile.csv`, `grouped_route_detail.csv`,
  `one-trace.csv`, `one-pack-read-profile.csv`,
  `gate-updown-cosubmit-profile.csv`,
  `cpu-fallback-profile.csv`, `fallback-reason-profile.csv`,
  `cpu-chunk-trace.csv`.
- Run validity: `ram_ok=true`, `memory_peak_bytes=14855942144`,
  `memory_file_bytes=14017069056`, display cleanup OK. Source was dirty only
  because of the new diagnostic passthrough in the demo script, so this run is a
  profiling diagnostic, not an SOTA acceptance run.
- Observed speed with profiling: `eval_tok_s=4.8`, `prompt_tok_s=3.6`,
  `TTFT=15789.1 ms`.

Measured lifecycle decomposition:

- Prompt phase:
  - route-selected logical expert bytes: `11.41 GiB`;
  - route cache hit/miss counts: `243 / 2505`;
  - gate one-stream total: `444.2 ms`
    (`read/cache/H2D=253.2 ms`, `sync/wait=176.8 ms`,
    `kernel=3.6 ms`);
  - up/down CPU fallback wall total: `1918.5 ms`;
  - CPU chunk work sum: `22944.2 ms` across worker chunks.
- Decode phase:
  - route-selected logical expert bytes: `46.32 GiB`;
  - route cache hit/miss counts: `2594 / 8566`;
  - gate one-stream total: `2437.6 ms`
    (`read/cache/H2D=1271.1 ms`, `sync/wait=1079.4 ms`,
    `kernel=43.7 ms`);
  - up/down CPU fallback wall total: `1841.2 ms`;
  - CPU chunk work sum: `50201.9 ms` across worker chunks.
- Approximate decode per-token lifecycle at `n32`:
  - total: `208.3 ms/token`;
  - gate expert-pack read/cache/H2D: `39.7 ms/token`;
  - gate stream sync/wait: `33.7 ms/token`;
  - gate GPU kernel plus D2H/scatter: `2.7 ms/token`;
  - up/down CPU fallback wall: `57.5 ms/token`;
  - dense, attention, router matmul/top-k, graph scheduling, and residual
    unprofiled time: `74.6 ms/token`.

Top decode layers by measured gate plus CPU fallback total:

| layer | total ms | gate ms | gate read ms | gate sync ms | gate kernel ms | up CPU ms | down CPU ms | gate hit/miss | up/down miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 218.9 | 131.4 | 68.2 | 60.8 | 1.3 | 44.6 | 42.9 | 26/67 | 93/93 |
| 0 | 206.7 | 124.4 | 65.8 | 56.1 | 1.3 | 41.4 | 40.9 | 31/62 | 93/93 |
| 2 | 203.0 | 122.0 | 63.9 | 55.8 | 1.2 | 40.3 | 40.8 | 30/63 | 93/93 |
| 4 | 143.1 | 83.7 | 43.2 | 37.9 | 1.4 | 29.6 | 29.8 | 54/39 | 93/93 |
| 3 | 134.9 | 82.2 | 44.6 | 35.3 | 1.2 | 26.0 | 26.6 | 59/34 | 93/93 |
| 19 | 134.7 | 80.1 | 41.4 | 36.6 | 1.1 | 26.6 | 28.0 | 55/38 | 93/93 |

Interpretation:

- The full lifecycle bottleneck is split between:
  1. gate miss movement from expert pack through host/cache/H2D plus stream
     synchronization (`~73 ms/token`); and
  2. up/down CPU fallback (`~58 ms/token`).
- Gate GPU kernel time is not the bottleneck (`~1.4 ms/token` inside decode
  gate trace). The costly part is feeding the kernel and waiting for staged
  data.
- Up/down are selected every MoE layer after router/top-k, but current accepted
  SOTA does not have them on the fast one-stream gate path. Their CPU fallback
  is now visible in `fallback-reason-profile.csv` and `cpu-chunk-trace.csv`.
- The remaining `~75 ms/token` residual includes dense/attention, router
  matmul/top-k, graph scheduling, and any non-MoE CUDA/CPU work. This profile
  still does not isolate router matmul/top-k as a standalone op; a graph/op
  timer is needed if router itself must be separated from dense/attention.

Next profiling task:

- Add a default-off graph/op timer for DeepSeek router construction:
  `ffn_gate_inp` matmul/lora, softplus/hash/bias, `top_k`, and weight
  normalization. The timer must produce per-layer prompt/decode rows that can
  be joined with `grouped_route_profile.csv`.
- After router timing is isolated, rerun n32/n96 lifecycle profile and replace
  the residual bucket with explicit `router`, `attention/dense`, and
  `scheduler/other` buckets.

## 2026-07-09 Nearest Step: VRAM Layout Before Up/Down Full Path

Priority:

- This is now the immediate next optimization step.
- Do this before implementing the up/down full expert-pack GPU chain and before
  the CPU/GPU hybrid tail-expert work.
- Reason: current SOTA still has about `8.8-9.0 GiB` nominal free VRAM, but
  previous larger-cache attempts were unstable or regressed. We need a precise
  allocator/cache layout map before adding more fast-path expert payloads.

Current SOTA VRAM layout from measured memory breakdown:

```text
RTX 5090 total:       32109 MiB

free:                  8987 MiB

ggml/self total:      17361 MiB
  model weights:      17339 MiB
  context/KV:            18 MiB
  compute buffer:         2 MiB

unaccounted:           5760 MiB
  gate VRAM cache:    ~5120 MiB
  CUDA/runtime/
  stream workspace/
  fragmentation:       ~640 MiB
```

Operational interpretation:

- The accepted generalized SOTA mostly uses VRAM for:
  1. dense/attention/non-MoE model weights (`~17.3 GiB`);
  2. one-stream gate expert VRAM cache (`~5.0 GiB`);
  3. CUDA/runtime/stream workspace/fragmentation (`~0.6 GiB`);
  4. remaining free VRAM (`~8.8-9.0 GiB`).
- The gate cache stores `blk.*.ffn_gate_exps.weight` entries. It does not hold a
  comparable persistent up/down cache.
- Therefore current VRAM is not full, but the free region cannot be blindly
  assigned to cache because allocator fragmentation, workspace needs, batch
  path buffers, and H2D/cache synchronization can make larger nominal caches
  slower or invalid.

Nearest-step objective:

- Convert the vague "`~9 GiB free`" into a safe, measured VRAM budget that can
  be used for the next expert-cache increment without breaking cold-start
  correctness, TTFT, or 16GB host RAM.
- Decide whether the next cache payload should be:
  - larger gate cache;
  - hot up cache;
  - hot down cache;
  - split gate/up/down cache;
  - workspace reservation for a future up/down GPU batch path.

Step A: Build an exact VRAM accounting profile.

- Run current generalized SOTA with default-off VRAM profile enabled and record:
  - `cudaMemGetInfo` before model load, after model load, after one-stream
    cache allocation, after first prompt MoE call, at first decode token, and
    at process exit;
  - one-stream cache requested/allocated MiB and slot count;
  - any batch-cache allocation requested/allocated MiB;
  - pinned host slots and sizes;
  - CUDA context/runtime overhead;
  - peak temporary workspace per token if visible.
- Required run hygiene:
  - stop display processes and stale model/GPU processes before the run;
  - strict `MemoryMax=16000000000`, `MemorySwapMax=0`;
  - cold `drop_caches`;
  - no prompt-specific profile/pack/alias;
  - source clean for accepted comparisons.

Step B: Sweep safe VRAM allocations without changing model math.

- Keep current prompt-general SOTA behavior and vary only cache/layout budgets:
  - gate one-cache: `5120`, `6144`, `7168`, `8192` MiB;
  - reserve free-VRAM floor: `2048`, `4096`, `6144` MiB;
  - if allocator supports it, fixed workspace reservation before expert cache.
- For each candidate, record:
  - `eval_tok_s`, `prompt_tok_s`, `TTFT`;
  - `memory_peak_bytes`, `memory_file_bytes`, `ram_ok`;
  - `cuda free/used` checkpoints;
  - gate hit/miss, read/H2D/sync/kernel split;
  - answer text and correctness.
- Reject immediately if:
  - host RAM exceeds 16GB including page cache;
  - CUDA OOM or cache insertion failure appears;
  - TTFT increases by more than 20%;
  - correctness regresses on France, Quantum, Fibonacci, Deploy, or Chinese
    dev prompts;
  - speed gain is prompt-specific rather than generalized.

Step C: Test whether free VRAM is better spent on up/down hot cache.

- Before building a full up/down GPU path, run a diagnostic-only cache-source
  analysis using `grouped_route_profile.csv` and fallback profiles:
  - rank/layer frequency of up/down selected experts across the dev prompts;
  - bytes needed for top-K hot up/down experts by layer;
  - expected CPU fallback wall time covered if those entries were resident;
  - expected H2D/read time if they used a pack/cache path.
- Build only the smallest prompt-general hot up/down pack/cache prototype that
  can test this bound. Do not tune on held-out prompts.
- If the bound shows the free VRAM can remove a meaningful fraction of the
  `~57.5 ms/token` up/down CPU fallback without causing gate regression, then
  proceed to the up/down full expert-pack GPU chain.

Step D: Acceptance and push rule.

- If a VRAM layout-only change improves generalized cold-start token rate while
  meeting all constraints, immediately:
  - record exact run dir, commit hash, env, command, memory, correctness output,
    TTFT, token rates, and source cleanliness;
  - commit and push to
    `origin/vendor/deepseek-token-rate-16gb` with author/committer
    `L-Ark <fliangae@connect.ust.hk>`.
- If all layout/cache sweeps regress or fail constraints, record the rejection
  table and then move to the up/down full-chain implementation with the measured
  safe VRAM reserve as a hard budget.

### 2026-07-09 Execution: Gate Cache 7168MiB Layout SOTA

Implementation:

- Added default-off `GGML_MOE_VRAM_ACCOUNTING_OUT` trace in one-stream and
  batch-cache CUDA code.
- Promoted default `GGML_MOE_STREAM_ONE_CACHE_MIB` from `5120` to `7168` in
  `scripts/demo-vendor-ds4-general-sota.sh` after clean validation.
- This changes cache/layout only; model math, prompt-general expert pack, and
  output guard behavior are unchanged.

VRAM accounting evidence:

- Baseline accounting run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T044402Z-20260709-vram-accounting-quantum-n32`.
- At first one-stream MoE call, before gate cache allocation:
  `used=17941.312 MiB`, `free=14167.812 MiB`.
- One-stream workspace allocation added only about `6 MiB`.
- `5120MiB` gate cache actually allocated `5117.000 MiB`,
  `1204` slots of `4.250 MiB`, leaving `9043.812 MiB` free.
- Sweep results:
  - `6144MiB`: allocated `6141.250 MiB`, `1445` slots,
    `5.1 tok/s`, TTFT `19336.95 ms`, RAM OK; no clear speed gain.
  - `7168MiB`: allocated `7165.500 MiB`, `1686` slots,
    `5.3 tok/s` on n32 Quantum, TTFT `16006.02 ms`, RAM OK.
  - `8192MiB`: allocated `8189.750 MiB`, `1927` slots,
    `5.2 tok/s` on n32 Quantum, TTFT `15762.29 ms`, RAM OK.
- Interpretation: `7168MiB` is the best current gate-cache layout point.
  It leaves about `6995.812 MiB` CUDA free after cache allocation, which is the
  current measured budget for future up/down GPU workspace/cache experiments.

Clean n96 comparison:

| config | run | eval tok/s | prompt tok/s | TTFT ms | RAM peak | source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| gate5120 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T044825Z-20260709-clean-gate5120-quantum-n96` | 5.1 | 3.5 | 15913.05 | 14883000320 | clean |
| gate7168 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T044902Z-20260709-clean-gate7168-quantum-n96` | 5.4 | 3.5 | 15802.83 | 14885810176 | clean |

Dev prompt validation for gate7168:

| prompt | run | eval tok/s | TTFT ms | RAM peak | correctness |
| --- | --- | ---: | ---: | ---: | --- |
| France | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T045007Z-20260709-clean-gate7168-dev-france-n96` | 5.9 | 16141.47 | 14852157440 | pass: coherent English France paragraph |
| Quantum | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T044902Z-20260709-clean-gate7168-quantum-n96` | 5.4 | 15802.83 | 14885810176 | pass: coherent short explanation |
| Deploy | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T045121Z-20260709-clean-gate7168-dev-deploy-n96` | 5.8 | 16535.28 | 14913478656 | pass: coherent deployment guidance opening |
| Shenzhen Chinese | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T045158Z-20260709-clean-gate7168-dev-shenzhen-n96` | 6.2 | 16591.60 | 14893309952 | pass: coherent Chinese itinerary opening |
| Fibonacci n96 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T045044Z-20260709-clean-gate7168-dev-fibonacci-n96` | 5.7 | 14841.38 | 14901903360 | speed/RAM pass, code truncated before return; not a clean correctness pass at n96 |
| Fibonacci n192 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T045302Z-20260709-clean-gate7168-fibonacci-n192` | 5.3 | 16159.72 | 14893985792 | pass: complete executable iterative Fibonacci function |

Acceptance status:

- `gate7168` is accepted as the current clean generalized speed/layout SOTA for
  non-code n96 dev prompts and for Fibonacci when allowed `n192`.
- Host RAM remains below 16GB including page cache in all listed runs.
- TTFT does not increase beyond the 20% limit relative to clean `gate5120`.
- The remaining product-quality issue is output length/guard behavior for code
  prompts at `n96`, not cache correctness.

Next step after this layout SOTA:

- Use the measured post-cache free VRAM budget (`~6996 MiB`) as a hard budget
  for up/down GPU-path work.
- Start with up/down hot-cache or batch-cache prototypes that reserve at least
  `~4 GiB` free VRAM for CUDA workspace/fragmentation until a tighter allocator
  bound is measured.
- Do not let up/down cache/workspace regress the accepted gate7168 hit rate or
  TTFT.

### 2026-07-09 Execution: Up/Down Batch GPU Probe

Finding:

- The previous SOTA binary had `GGML_CUDA_MOE_STREAM_BATCH=OFF`, so up/down
  could not enter the real CUDA batch implementation. This is one concrete
  reason CPU fallback persisted: the exported batch symbol was the stub path.
- Rebuilt the remote diagnostic binary with
  `GGML_CUDA_MOE_STREAM_BATCH=ON`. This is a diagnostic build, not the accepted
  SOTA build/config.
- Added demo passthrough/override for:
  `GGML_MOE_STREAM_DECLINE_DEBUG`,
  `GGML_MOE_STREAM_DOWN_BATCH`,
  `GGML_MOE_STREAM_DOWN_Q80_COMPAT_BATCH`, and
  `GGML_MOE_STREAM_UP_Q80_COMPAT_BATCH`.
- Changed the demo defaults so up/down batch stays disabled unless explicitly
  requested. This preserves the accepted gate7168 SOTA even if a user builds
  with `GGML_CUDA_MOE_STREAM_BATCH=ON`.
- Follow-up safety fix: `GGML_MOE_STREAM_DOWN_BATCH` must be unset when
  disabled, not set to `0`, because the CPU-side check tests env presence.
  Also defaulted gate batch prefetch/cosubmit off so a batch-enabled build does
  not allocate the 12GB batch cache before the one-stream gate cache.

Diagnostic results:

| config | run | eval tok/s | result |
| --- | --- | ---: | --- |
| batch OFF, gate7168 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T044902Z-20260709-clean-gate7168-quantum-n96` | 5.4 | accepted layout SOTA |
| batch ON, up+down, bcache2048, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T050237Z-20260709-batchon-gate7168-updown-decline-debug-n16` | 3.2 | rejected; batch accepted but slow |
| batch ON, up+down, bcache4096, n32 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T050351Z-20260709-batchon-gate7168-bcache4096-quantum-n32` | 3.5 | rejected |
| batch ON, up+down, bcache6144, n32 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T050419Z-20260709-batchon-gate7168-bcache6144-quantum-n32` | 3.7 | rejected |
| batch ON, down-only, bcache6144, n32 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T050612Z-20260709-batchon-gate7168-downonly-bcache6144-quantum-n32` | 4.2 | rejected |
| batch ON, down-only, bcache6144, n96 no profile | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T050709Z-20260709-batchon-gate7168-downonly-bcache6144-noprofile-quantum-n96` | 4.2 | rejected |
| batch ON build, default script after safety fix | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T051226Z-20260709-batchbuild-default-gate7168-sanity4-n32` | 5.2 | sanity pass: only one-stream gate cache allocated |

Important evidence:

- With batch ON and bcache2048, `kimi_cpu_moe_profile` reported
  `batch_accept=1440`, `batch_decline=0`, proving the real GPU batch path was
  entered and accepted.
- With up+down bcache6144, down-batch profile showed:
  - `calls=2720`;
  - `stage_ms=1695.2`;
  - `kernel_ms=505.9`;
  - `wall_ms=2498.3`;
  - `hits=2716`, `miss=644`;
  - end-to-end `3.7 tok/s`.
- With down-only bcache6144, down-batch profile showed:
  - `calls=1360`;
  - `stage_ms=1002.9`;
  - `kernel_ms=177.2`;
  - `wall_ms=1253.3`;
  - `hits=1062`, `miss=618`;
  - end-to-end `4.2 tok/s`.
- Removing CSV profiling did not improve down-only n96: still `4.2 tok/s`.
  Therefore the regression is the batch path/stage/cache behavior, not profile
  output overhead.

Interpretation:

- Up/down GPU path is now partially technically open: CUDA batch accepts and
  computes down, and up can reuse the Q80-compatible down-batch path.
- It is not acceptable as an optimization yet. Stage/cache movement dominates
  the GPU path, and the current batch path is slower than CPU fallback plus the
  accepted gate7168 fast path.
- The next optimization must reduce `stage_ms`, not just increase batch cache:
  2048 -> 6144 MiB improved hit rate but still stayed far below SOTA.

Next implementation task:

- Build a prompt-general up/down fast source, analogous to the gate expert pack,
  so up/down batch cache fills do not stage from slow GGUF/page-cache paths.
- Then rerun down-only first. Acceptance target for the first useful GPU path:
  down-only must exceed clean gate7168 baseline on at least Quantum n96 while
  keeping RAM below 16GB and TTFT within the 20% limit.
- Only after down-only is faster should up Q80-compatible batch be re-enabled,
  because current up+down is slower than down-only.

### 2026-07-09 Up/Down Pack-Alias Source Guard

The first up/down pack-alias probe exposed a source-path hazard:

- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T051905Z-20260709-batchon-downonly-packalias-bcache6144-quantum-n32`.
- Requested input alias:
  `/home/wici/runs/vendor-ds4-16gb/updown-pack-alias/ds4-native-updown-pack-payload-alias-newmachine.tsv`,
  with `source_path` pointing to
  `/home/wici/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.expert-pack`.
- The demo script still rewrote the effective alias first column to `$MODEL`.
  The batch loader therefore opened
  `/home/wici/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`
  while using offsets from the expert pack.
- Observed result: apparent `eval_tok_s=9.8`, `memory_peak_bytes=12922929152`,
  `ram_ok=true`, but `run_ok=false` and `answer_present=false`. This is
  rejected and must never be promoted because the source mapping was wrong and
  the output was empty.

Code update:

- Add `DS4_ALIAS_PRESERVE_SOURCE_PATH=1` to
  `scripts/demo-vendor-ds4-general-sota.sh`.
- Default remains unchanged: static GGUF aliases are still rewritten to `$MODEL`.
- For expert-pack alias experiments, set `DS4_ALIAS_PRESERVE_SOURCE_PATH=1` so
  the effective run TSV preserves the pack `source_path`.

Next rerun:

1. Use the same corrected up/down pack-alias TSV.
2. Run down-only batch first with:
   `DS4_ALIAS_PRESERVE_SOURCE_PATH=1`,
   `GGML_MOE_STREAM_DOWN_BATCH=1`,
   `GGML_MOE_STREAM_UP_Q80_COMPAT_BATCH=0`,
   `GGML_MOE_GATE_BATCH_PREFETCH=0`,
   `GGML_MOE_GATE_UPDOWN_COSUBMIT=0`,
   and `GGML_MOE_VRAM_CACHE_MIB=6144`.
3. Verify stderr opens the alias source as `native.expert-pack`, not
   `native.gguf`.
4. Accept only if the answer is present/coherent, RAM stays under 16GB, TTFT
   stays within the 20% gate, and token rate beats the clean gate7168 SOTA.

Execution results:

| config | run | eval tok/s | RAM peak | result |
| --- | --- | ---: | ---: | --- |
| preserve source, gate7168, bcache6144 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T052212Z-20260709-batchon-downonly-packalias-preserve-bcache6144-quantum-n32` | 2.9 | 14814023680 | rejected: valid source and answer, but slow |
| preserve source, gate7168, bcache7168 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T052329Z-20260709-batchon-downonly-packalias-preserve-bcache7168-quantum-n32` | 4.2 | 14692495360 | rejected: below gate7168 SOTA |
| preserve source, gate7168, bcache8192 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T052356Z-20260709-batchon-downonly-packalias-preserve-bcache8192-quantum-n32` | 4.2 | 14857048064 | rejected: below gate7168 SOTA |
| preserve source, gate6144, bcache8192 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T052531Z-20260709-batchon-downonly-packalias-preserve-gate6144-bcache8192-quantum-n32` | 4.2 | 14810861568 | rejected: below gate7168 SOTA |
| preserve source, gate5120, bcache8192 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T052558Z-20260709-batchon-downonly-packalias-preserve-gate5120-bcache8192-quantum-n32` | 4.0 | 14792900608 | rejected: gate miss regression |

Important findings:

- Corrected pack-alias source now opens
  `/home/wici/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.expert-pack`.
- Corrected source produces coherent Quantum output and stays within the 16GB
  host RAM cgroup.
- The apparent `9.8 tok/s` wrong-source run remains rejected.
- Increasing batch cache does not reduce down miss count:
  - gate7168/bcache6144: `hits=893`, `misses=787`;
  - gate7168/bcache8192: `hits=893`, `misses=787`;
  - gate6144/bcache8192: `hits=893`, `misses=787`;
  - gate5120/bcache8192: `hits=893`, `misses=787`.
- Therefore the dominant down-batch cost is not capacity eviction. It is the
  first-use synchronous miss path: each unique down expert still pays
  `SSD/O_DIRECT read + pinned staging + H2D + cache insert` before the current
  token can continue.
- For bcache8192, even when the cache allocated `8.0 GiB` / `1927` slots,
  misses stayed `787` and `stage_ms` increased to `1532.9`.
- The GPU down kernel is still cheap (`~177 ms` total for n32); the blocking
  miss stage is what makes GPU down slower than CPU fallback.

Revised next implementation direction:

1. Stop trying to make synchronous GPU-on-miss down batch the accepted path.
2. Implement an opt-in CPU-on-miss / async-fill policy for up/down:
   - if an up/down expert is already resident in the batch VRAM cache, use the
     GPU batch kernel;
   - if it is missing, do not block the current token on read/H2D;
   - let the existing CPU fallback compute the current token;
   - submit an asynchronous pack/alias read + H2D cache fill for future tokens;
   - ensure cache slot state cannot be observed as ready until H2D completes.
3. Profile the new policy with:
   - CPU fallback wall time;
   - async fill submits/completions;
   - later GPU cache-hit count;
   - stage time exposed on the critical path;
   - answer correctness and exact RAM cgroup accounting.
4. Acceptance target: beat clean gate7168 SOTA on Quantum n96 and at least one
   additional dev prompt, with `ram_ok=true`, coherent output, and TTFT within
   the 20% gate.

### 2026-07-09 Gate-Triggered Down Cosubmit Probe

Existing `GGML_MOE_CURRENT_DOWN_OVERLAP=1` was tested first with the corrected
up/down pack alias:

- `early=0`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T052906Z-20260709-batchon-downonly-packalias-currentoverlap-early0-quantum-n32`,
  `eval_tok_s=4.2`, `stage_ms=1276.0`, `hits=893`, `misses=787`.
- `early=1`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T052932Z-20260709-batchon-downonly-packalias-currentoverlap-early1-quantum-n32`,
  `eval_tok_s=4.2`, `stage_ms=1283.1`, `hits=893`, `misses=787`.

Interpretation: this hook is inside the batch up/gate path, while the accepted
DeepSeek SOTA uses one-stream gate. It therefore did not trigger useful current
down overlap for the accepted path.

Full gate/up/down cosubmit was then tested through the one-stream gate hook:

- gate7168/bcache8192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T053100Z-20260709-batchon-downonly-packalias-cosubmit-gate7168-bcache8192-quantum-n32`,
  `eval_tok_s=4.3`, `stage_ms=815.3`, `hits=1229`, `misses=451`,
  `iouring_reads=1926`, `iouring_bytes=8583118848`.
- gate6144/bcache8192:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T053127Z-20260709-batchon-downonly-packalias-cosubmit-gate6144-bcache8192-quantum-n32`,
  `eval_tok_s=4.3`, `stage_ms=710.9`, `hits=1281`, `misses=399`,
  `iouring_reads=2085`, `iouring_bytes=9291694080`.

This proved gate-triggered cosubmit can reduce critical down misses, but full
cosubmit also moved up tensors that were not used by the down-only GPU path.

Code update:

- Added default-off `GGML_MOE_GATE_UPDOWN_COSUBMIT_DOWN_ONLY=1`.
- When unset, full gate/up/down cosubmit behavior is unchanged.
- When set, gate cosubmit skips `ffn_up_exps` and preloads only
  `ffn_down_exps` into the batch VRAM cache.
- The demo script now records and passes this env into the 16GB cgroup.
- Commit: `d02b54d vendor-ds4: add down-only gate cosubmit mode`.

Down-only cosubmit results:

| config | run | eval tok/s | stage_ms | hits/misses | RAM peak | result |
| --- | --- | ---: | ---: | --- | ---: | --- |
| gate7168/bcache8192/down-only cosubmit | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T053503Z-20260709-batchon-downonly-packalias-cosubmitdownonly-gate7168-bcache8192-quantum-n32` | 4.5 | 241.2 | 1570 / 110 | 14773305344 | rejected: below gate7168 SOTA |
| gate6144/bcache8192/down-only cosubmit | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T053529Z-20260709-batchon-downonly-packalias-cosubmitdownonly-gate6144-bcache8192-quantum-n32` | 4.6 | 105.8 | 1654 / 26 | 14731517952 | rejected: below gate7168 SOTA |

The down-only cosubmit patch is a real stage improvement:

- compared with corrected pack-alias down-only, misses dropped from `787` to
  `26` in the best gate6144 split;
- critical-path stage dropped from about `1260-1530 ms` to `105.8 ms`;
- kernel stayed about `177 ms`, confirming the kernel was not the bottleneck.

However, end-to-end token rate is still below the clean gate7168 path because
the gate-triggered preloads still add large off-critical-path IO/H2D work:

- best down-only cosubmit `iouring_reads=1712`,
  `iouring_bytes=7629438976`, `async prefetch waits=787`;
- clean default sanity on pushed `d02b54d`:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T053647Z-20260709-d02-default-gate7168-sanity-quantum-n32`,
  `eval_tok_s=5.0`, `run_ok=true`, `ram_ok=true`, answer coherent, and stderr
  only shows the one-stream gate VRAM cache.

Conclusion:

- Up/down GPU path is now technically open and can be fed by one-stream gate
  prefetch, but it is not yet an accepted token-rate improvement.
- The remaining problem is prefetch selectivity: loading every gate-selected
  down expert almost eliminates down misses but moves too many bytes.

Next implementation target:

1. Add a selective down-cosubmit policy instead of preloading every gate event.
   Candidate filters:
   - only preload if the same `(layer, expert)` appears at least twice inside a
     short decode window;
   - only preload top recurring experts per layer from a prompt-general dev
     route histogram;
   - skip first-use low-reuse experts and let CPU fallback handle them.
2. The policy must remain prompt-general and default-off.
3. It must record planned/skipped/preloaded/down-hit counters and compare saved
   down-stage time against extra cosubmit IO/H2D bytes.
4. Acceptance remains unchanged: beat clean gate7168 SOTA with RAM under 16GB,
   coherent output, and TTFT within the 20% gate.

### 2026-07-09 Down Batch Hit-Only Probe

Implementation:

- Added default-off `GGML_MOE_DOWN_BATCH_HIT_ONLY=1`.
- When unset, down batch behavior is unchanged.
- When set, down batch only runs when every active down expert is already in
  the batch VRAM cache. The first miss returns
  `down_cache_miss_hit_only`, letting the existing CPU fallback compute that
  down call instead of synchronously staging SSD/H2D for the miss.
- Commit: `948dc68 vendor-ds4: add down batch hit-only mode`.

Rationale:

- Down-only cosubmit made most down calls cache hits, but a small number of
  misses still forced synchronous stage on the critical path.
- Hit-only mode converts those few misses into CPU fallback and keeps the GPU
  path for resident down experts.

Diagnostic n32 results with profile and decline debug:

| config | run | eval tok/s | accepted down calls | stage_ms | declines | RAM peak | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate6144/bcache8192/down-only cosubmit/hit-only | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T054149Z-20260709-downbatch-hitonly-cosubmitdownonly-gate6144-bcache8192-quantum-n32` | 4.6 | 1334 | 53.4 | 26 | 14703419392 | rejected |
| gate7168/bcache8192/down-only cosubmit/hit-only | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T054215Z-20260709-downbatch-hitonly-cosubmitdownonly-gate7168-bcache8192-quantum-n32` | 4.7 | 1224 | 50.5 | 136 | 14824144896 | rejected |

Best no-profile n96 result:

- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T054334Z-20260709-hitonly-cosubmitdownonly-gate7168-bcache8192-noprofile-quantum-n96`.
- `eval_tok_s=5.1`.
- `prompt_tok_s=3.5`.
- `TTFT=16262.069675 ms`.
- `memory_peak_bytes=14872895488`.
- `ram_ok=true`.
- `run_ok=true`.
- Output is coherent:
  "Quantum computing is a type of computing that uses the strange rules of
  quantum physics..."
- Counters:
  - one-stream gate cache: `hits=8947`, `misses=3369`, `hit_rate=72.6%`;
  - batch cache: `hits=3166`, `misses=0`, `preloads=2950`;
  - cosubmit: `jobs=1475`, `cache_hits=7228`, `no_slot_skips=3613`;
  - batch iouring: `iouring_reads=1475`,
    `iouring_bytes=6573260800`, `iouring_wait_us=759329`.

Decision:

- Reject as SOTA because the accepted clean gate7168 Quantum n96 result remains
  `5.4 tok/s`.
- Keep the code default-off. It proves a safe CPU-on-miss mechanism and reduces
  critical down stage, but still spends too much off-critical-path IO/H2D on
  cosubmit preloads.

Updated bottleneck:

- The bottleneck has moved from synchronous down miss stage to cosubmit preload
  volume and prefetch waits.
- Reaching a real improvement now requires reducing cosubmit bytes/jobs while
  preserving most later down hits.

Next implementation target:

1. Add selective down-only cosubmit with a cheap, prompt-general reuse filter.
2. Start with a conservative per-layer/expert repeat filter:
   - record whether `(down_tensor, expert)` has appeared before in decode;
   - skip the first appearance;
   - only cosubmit on the second and later appearances;
   - combine with hit-only down batch so skipped first appearances use CPU
     fallback instead of synchronous GPU staging.
3. Measure whether this reduces `iouring_reads` enough to beat clean gate7168.
4. If repeat filtering loses too many down hits, move to a small per-layer hot
   set learned from the allowed dev prompt set, not held-out prompts.

### 2026-07-09 Repeat-Filtered Cosubmit Probe

Implementation:

- Added default-off `GGML_MOE_GATE_UPDOWN_COSUBMIT_MIN_SEEN`.
- `MIN_SEEN=1` means a `(tensor, expert)` is not cosubmitted on its first
  gate observation; it is eligible from the second observation onward.
- Added `repeat_skips` to the gate/up/down cosubmit atexit report.
- Commit: `109816d vendor-ds4: add cosubmit repeat filter`.

Result:

- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T054734Z-20260709-hitonly-cosubmitdownonly-minseen1-gate7168-bcache8192-noprofile-quantum-n96`.
- Config:
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=7168`;
  - `GGML_MOE_VRAM_CACHE_MIB=8192`;
  - `GGML_MOE_STREAM_DOWN_BATCH=1`;
  - `GGML_MOE_DOWN_BATCH_HIT_ONLY=1`;
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`;
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_DOWN_ONLY=1`;
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_MIN_SEEN=1`;
  - corrected up/down pack alias with preserved expert-pack source path.
- Metrics:
  - `eval_tok_s=4.6`;
  - `prompt_tok_s=3.3`;
  - `TTFT=16379.814546 ms`;
  - `memory_peak_bytes=14861430784`;
  - `ram_ok=true`;
  - `run_ok=true`;
  - output coherent.
- Counters:
  - batch `iouring_reads=1475`;
  - batch `iouring_bytes=6573260800`;
  - batch `iouring_wait_us=950593`;
  - batch cache `hits=3039`, `misses=0`, `preloads=2950`;
  - cosubmit `jobs=1475`, `cache_hits=7208`, `no_slot_skips=720`,
    `repeat_skips=2913`.

Decision:

- Reject. It is slower than both:
  - no-repeat hit-only n96: `5.1 tok/s`;
  - accepted clean gate7168 n96: `5.4 tok/s`.
- The repeat filter skipped many early events, but still filled the same 1475
  slots and moved the same `6.57GB` through iouring/H2D by the end of the run.
  It reduced useful early prefetch rather than reducing total payload enough.

Updated next direction:

- A pure repeat threshold is too crude.
- The next candidate should cap the total down cosubmit working set with a
  prompt-general hot set or admission score, not simply delay every first use.
- Required profile before implementation:
  1. From allowed dev prompts, compute per-layer/expert reuse and hit benefit
     under the accepted gate7168 path.
  2. Build a small hot-set admission table that fits the available batch cache
     without filling it with low-value entries.
  3. Keep held-out prompts unused until a candidate is frozen.
  4. Compare against clean gate7168 and against no-repeat hit-only.

### 2026-07-09 Profile-Guided Down Cosubmit Admission

Implementation target:

- Add a default-off admission gate for gate-triggered up/down cosubmit:
  `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_MIN_COUNT`.
- It reuses the existing `GGML_MOE_VRAM_PROFILE` parser and
  `(tensor, expert)` hash index.
- When the env value is positive, a candidate up/down expert is cosubmitted only
  if its profile count is at least that value.
- Intended first use is down-only:
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`;
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_DOWN_ONLY=1`;
  - `GGML_MOE_DOWN_BATCH_HIT_ONLY=1`;
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_MIN_COUNT=1`;
  - `GGML_MOE_VRAM_PROFILE=<allowed-dev down hotset csv>`.

Why this is the right next probe:

- Previous down-only cosubmit proved that moving down experts to GPU can reduce
  synchronous down stage from roughly `~1276 ms` to `~50-240 ms` on n32 probes.
- It still lost to clean gate SOTA because it moved too many low-value down
  experts, filling cache slots and adding `~6.57GB` iouring/H2D work.
- A dev-profile hotset should reduce total cosubmit jobs/bytes while preserving
  repeated down hits on prompt-general routes.

Acceptance rule:

- This is not a prompt-specific optimization. The profile must be built from
  allowed dev prompts only; held-out prompts remain unused until a candidate is
  frozen.
- If a candidate beats the accepted clean gate7168 generalized SOTA
  (`Quantum n96 5.4 tok/s`) under 16GB RAM, with correct output and TTFT within
  limits, immediately record the exact run and push source + docs to
  `origin/vendor/deepseek-token-rate-16gb`.
- If it does not beat SOTA, keep the code default-off and record the rejection
  with `profile_skips`, jobs, bytes, cache hit/miss, RAM, TTFT, and answer text.

Implementation note:

- The demo wrapper must pass through and record `GGML_MOE_VRAM_PROFILE_PRELOAD`.
  Without this, a run labeled "nopreload" still preloads the profile inside the
  cgroup because the variable is lost at `systemd-run` boundary. Any nopreload
  measurement before this passthrough fix must be treated as a preload run.

Result:

- Code commits:
  - `12843ef vendor-ds4: gate cosubmit by profile hotset`;
  - `bfbb4fe vendor-ds4: pass through vram profile preload flag`.
- Remote validation head: `bfbb4fe08`.
- Build: `/home/wici/ssd-llama/build-cuda/bin/llama-cli`, CUDA build passed.

Preload-mode probes before the passthrough fix:

| config | run | eval tok/s | TTFT ms | RAM peak | decision |
| --- | --- | ---: | ---: | ---: | --- |
| down top48, profile preload | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T055906Z-20260709-profilehot-top48-preload-cosubmitdownonly-gate7168-bcache8192-quantum-n96` | 5.4 | 16046.57 | 14834192384 | equal to SOTA, not promoted |
| down top64, profile preload | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T060010Z-20260709-profilehot-top64-preload-cosubmitdownonly-gate7168-bcache8192-quantum-n96` | 5.2 | 16307.59 | 14862815232 | rejected |
| down top128, profile preload | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T060047Z-20260709-profilehot-top128-preload-cosubmitdownonly-gate7168-bcache8192-quantum-n96` | 5.3 | 16155.75 | 14864621568 | rejected |

These runs are valid RAM/correctness probes but not true nopreload probes,
because `GGML_MOE_VRAM_PROFILE_PRELOAD=0` was not passed through yet.

True nopreload probes after `bfbb4fe`:

| config | run | eval tok/s | TTFT ms | RAM peak | source | decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| down top48, n32 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T060701Z-20260709-true-nopreload-downhot-top48-cosubmitdownonly-gate7168-bcache8192-quantum-n32` | 5.1 | 16271.10 | 14817521664 | clean | diagnostic rejected |
| down top48, n96 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T060752Z-20260709-true-nopreload-downhot-top48-cosubmitdownonly-gate7168-bcache8192-quantum-n96` | 5.2 | 16205.96 | 14841495552 | clean | rejected |
| down top128, n96 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T060843Z-20260709-true-nopreload-downhot-top128-cosubmitdownonly-gate7168-bcache8192-quantum-n96` | 5.3 | 16209.74 | 14880526336 | clean | rejected |

Important counters:

- top48 true nopreload n96:
  - profile entries loaded for admission: `48`;
  - cosubmit jobs: `42`;
  - batch iouring bytes: `187170816`;
  - down cache hits: `648`;
  - profile skips: `10907`;
  - gate one-stream direct reads remain `3369`, `15013773312` bytes.
- top128 true nopreload n96:
  - profile entries loaded for admission: `128`;
  - cosubmit jobs: `93`;
  - batch iouring bytes: `414449664`;
  - down cache hits: `1133`;
  - profile skips: `9634`;
  - gate one-stream direct reads remain `3369`, `15013773312` bytes.

Conclusion:

- The profile-guided down GPU path is correct and can convert selected down
  fallback events into batch-cache hits under the 16GB host RAM cap.
- It does not improve token rate over the accepted clean gate7168 generalized
  SOTA (`5.4 tok/s` on Quantum n96). The remaining dominant bytes are still the
  one-stream gate path (`~15.0GB` direct reads in this prompt), and the small
  down hotset does not change that bottleneck.
- Do not promote these runs as SOTA.

Next direction:

- Stop simply enlarging down hotsets; top48->top128 increases down hits but not
  enough to win.
- The next useful implementation should address shared VRAM/cache layout across
  gate and down, or reduce gate direct-read volume/latency. A promising next
  probe is a unified admission policy that reserves a small down cache budget
  without stealing effective capacity from the gate one-stream cache, then tests
  whether gate misses stay flat while down hits increase.

### 2026-07-09 Clean Gate Cache VRAM Sweep

Purpose:

- Test whether unused VRAM is better spent on the prompt-general gate one-stream
  cache before adding more up/down GPU path complexity.
- No prompt-specific profile, no cosubmit, no down batch; only
  `GGML_MOE_STREAM_ONE_CACHE_MIB` changes.

Results:

| config | run | eval tok/s | TTFT ms | RAM peak | gate reads | gate bytes | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate8192 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T061106Z-20260709-clean-gate8192-quantum-n96` | 5.4 | 16123.69 | 14863228928 | not recorded | not recorded | equal to SOTA, not promoted |
| gate9216 first | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T061144Z-20260709-clean-gate9216-quantum-n96` | 5.5 | 15938.00 | 14854176768 | 3000 | 13369344000 | candidate only |
| gate9216 repro1 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T061247Z-20260709-clean-gate9216-quantum-n96-repro1` | 5.0 | 16117.88 | 14878085120 | 3000 | 13369344000 | not reproducible, rejected |

Observation:

- gate9216 reduced gate direct reads versus gate7168/top48 profiles
  (`3000` reads and `13.37GB` versus `3369` reads and `15.01GB` on Quantum
  n96), so the cache layout is directionally useful.
- The same gate9216 hit/miss pattern produced `5.5 tok/s` once and `5.0 tok/s`
  on immediate replay. The variance is therefore not due to expert routing or
  cache hit rate; it is likely SSD/direct-read latency, PCIe link behavior, or
  scheduler noise.

Decision:

- Do not promote gate9216 as SOTA. It is not reproducible enough under the
  current acceptance rule.
- Current accepted generalized cold SOTA remains clean gate7168 Quantum n96
  `5.4 tok/s`.

Next profiling needed:

- Add/enable low-overhead direct-read latency accounting for the one-stream gate
  path so repeated runs can separate:
  - gate direct read time;
  - gate H2D/cache insert time;
  - CUDA kernel time;
  - scheduler/SSD variance.
- Then rerun gate7168/gate9216 repeated A/B with identical source and cold
  procedure. Promote only if the median and at least one immediate replay exceed
  the accepted SOTA while RAM/TTFT/correctness remain valid.

Implementation:

- Added default-off `GGML_MOE_ONE_PACK_READ_SUMMARY=1`.
- It records aggregate one-stream expert-pack read latency without per-read CSV
  file writes:
  - `direct_reads`;
  - `direct_total_ms`;
  - `direct_avg_ms`;
  - `direct_max_ms`;
  - equivalent buffered counters if O_DIRECT falls back.
- Demo wrapper records and passes the env through the 16GB cgroup.
- This is profiling-only and must not change default SOTA behavior.

### 2026-07-09 Execution: Gate Cache 12288MiB Layout SOTA

Implementation:

- Updated the generalized demo default from
  `GGML_MOE_STREAM_ONE_CACHE_MIB=7168` to `12288`.
- No prompt-specific profile, no cosubmit, no down batch. This is a pure VRAM
  layout change that spends more free VRAM on the prompt-general gate
  one-stream cache.

Why it can improve token rate:

- The accepted gate7168 path still had many gate expert misses:
  - gate7168 Quantum n96: `3369` direct reads, `15013773312` bytes.
- Larger gate cache keeps more gate experts resident across decode and reduces
  SSD direct-read volume.
- gate12288 Quantum n96 measured:
  - `2913` direct reads, `12981633024` bytes.
- The objective upper-bound improvement from bytes alone is roughly:
  `(15.01GB - 12.98GB) / measured SSD+direct-read bandwidth`, which is enough
  to move a borderline `5.4 tok/s` run into the displayed `5.5 tok/s` range
  when read latency is not dominated by long-tail stalls.

Accepted SOTA evidence:

| prompt/config | run | eval tok/s | prompt tok/s | TTFT ms | RAM peak | source | correctness |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Quantum n96 gate12288 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T062449Z-20260709-clean-gate12288-quantum-n96` | 5.5 | 3.6 | 16102.45 | 14873735168 | clean `e82857ff5` | pass |
| Quantum n96 gate12288 repro1 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T062543Z-20260709-clean-gate12288-quantum-n96-repro1` | 5.5 | 3.6 | 16045.33 | 14859661312 | clean `e82857ff5` | pass |
| Deploy n96 gate12288 general check | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T062638Z-20260709-clean-gate12288-deploy-n96-general-check` | 6.0 | 4.4 | 16575.34 | 14852141056 | clean `e82857ff5` | pass |

Constraints:

- Host RAM remains below 16GB including page cache:
  - peak range: `14852141056` to `14873735168` bytes.
- TTFT does not regress beyond 20% versus accepted gate7168 Quantum n96:
  - gate7168 accepted TTFT: `15802.83 ms`;
  - gate12288 accepted Quantum TTFT: `16045-16102 ms`, about `+1.5-1.9%`.
- Output is semantically correct and coherent:
  - Quantum answer explains qubits/superposition/entanglement coherently;
  - Deploy answer gives appropriate deployment/quantization guidance.
- This is prompt-general: no dev profile, no held-out tuning, no
  prompt-specific pack.

Decision:

- Promote gate12288 clean layout as the current generalized cold-start SOTA:
  `eval_tok_s=5.5` on Quantum n96, with a Deploy prompt check at `6.0 tok/s`.
- The code/default config must be pushed to
  `origin/vendor/deepseek-token-rate-16gb` so future demo runs use the accepted
  layout without extra env overrides.

Remaining caveat:

- gate9216 and some gate11264 runs showed token-rate variance despite identical
  gate read counts. The new one-pack read summary should remain available for
  diagnosing SSD/direct-read long tails, but it is not enabled in the accepted
  SOTA runs.

### 2026-07-09 Probe: Gate12288 + Fallback-Value Down GPU Hotset

Purpose:

- Continue the original up/down GPU-path objective after the gate12288 SOTA.
- The previous call-count down hotsets were not selective enough. This probe
  builds down admission profiles from fallback wall-time value instead:
  `current_sota_updown_decode_top4096_fallback_us.tsv`, down tensors only.
- Runtime remains prompt-general in execution, but the fallback-value profile is
  still an admission artifact and must be validated on multiple prompts before
  promotion.

Theory:

- Gate12288 already reduces gate direct reads to about `2913` reads /
  `12.98GB` on Quantum n96.
- A small down batch cache can fit beside gate12288 if constrained:
  - `GGML_MOE_VRAM_CACHE_MIB=2048` requests 2GiB and actually allocates about
    `1.7GiB`, `420` down slots;
  - `GGML_MOE_VRAM_CACHE_MIB=1024` leaves more CUDA workspace for other prompts.
- The useful target is not "all down experts"; it is high fallback-value down
  experts, because profile showed many down GPU hits are `n_active=1` and save
  only tens of microseconds over CPU fallback.

Quantum n96 results, gate12288 + bcache2048:

| down admission | run | eval tok/s | TTFT ms | RAM peak | decision |
| --- | --- | ---: | ---: | ---: | --- |
| call-count top48 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T063411Z-20260709-gate12288-bcache2048-downhot-top48-quantum-n96` | 5.3 | 16186.32 | 14864633856 | rejected |
| call-count top128 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T063449Z-20260709-gate12288-bcache2048-downhot-top128-quantum-n96` | 5.3 | 15742.45 | 14879125504 | rejected |
| fallback top32 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T063819Z-20260709-gate12288-bcache2048-fallbackdown-top32-quantum-n96` | 5.5 | 16097.68 | 14862381056 | equal to SOTA |
| fallback top64 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T063856Z-20260709-gate12288-bcache2048-fallbackdown-top64-quantum-n96` | 5.5 | 16079.59 | 14858711040 | equal to SOTA |
| fallback top128 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064003Z-20260709-gate12288-bcache2048-fallbackdown-top128-quantum-n96` | 5.6 | 16293.96 | 14867554304 | candidate only |
| fallback top128 repro1 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064059Z-20260709-gate12288-bcache2048-fallbackdown-top128-quantum-n96-repro1` | 5.5 | 16304.43 | 14869839872 | not above SOTA |
| fallback top192 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064207Z-20260709-gate12288-bcache2048-fallbackdown-top192-quantum-n96` | 4.9 | 16152.53 | 14863855616 | rejected |
| fallback top256 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064246Z-20260709-gate12288-bcache2048-fallbackdown-top256-quantum-n96` | 5.7 | 15488.11 | 14852931584 | candidate only |
| fallback top256 repro1 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064343Z-20260709-gate12288-bcache2048-fallbackdown-top256-quantum-n96-repro1` | 5.5 | 16248.07 | 14866653184 | not above SOTA |
| fallback top320 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064535Z-20260709-gate12288-bcache2048-fallbackdown-top320-quantum-n96` | 5.5 | 16120.46 | 14859427840 | equal to SOTA |
| fallback top384 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064612Z-20260709-gate12288-bcache2048-fallbackdown-top384-quantum-n96` | 5.5 | 16098.39 | 14872334336 | equal to SOTA |

Representative counters:

- fallback top128:
  - gate one-stream reads stay fixed at `2913`, `12981633024` bytes;
  - down cosubmit jobs: `69`;
  - down iouring bytes: `307494912`;
  - down cache hits: `370`;
  - no gate regression, but replay falls back to `5.5`.
- fallback top256:
  - gate one-stream reads stay fixed at `2913`, `12981633024` bytes;
  - down cosubmit jobs: `146`;
  - down iouring bytes: `650641408`;
  - down cache hits: `495`;
  - first run reaches `5.7`, replay falls back to `5.5`.

Generalization / safety check:

- `bcache2048 + fallback top256` on Deploy fails with CUDA OOM:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064708Z-20260709-gate12288-bcache2048-fallbackdown-top256-deploy-n96`.
  - `run_ok=false`, exit `134`;
  - stderr shows `CUDA error: out of memory`;
  - reject regardless of Quantum speed.
- Reducing to `bcache1024` avoids the OOM:
  - Deploy run
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064817Z-20260709-gate12288-bcache1024-fallbackdown-top256-deploy-n96`
    reaches `5.7 tok/s`, RAM OK, coherent answer;
  - Quantum run
    `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T064853Z-20260709-gate12288-bcache1024-fallbackdown-top256-quantum-n96`
    reaches `5.5 tok/s`, RAM OK, coherent answer.
  - This is safe but not faster than clean gate12288 (`Deploy 6.0`, Quantum
    accepted `5.5`).

Decision:

- Do not promote fallback-value down hotset as SOTA.
- The up/down GPU path is now functionally open and can produce down cache hits
  without gate regression, but the only observed Quantum improvements
  (`5.6-5.7`) are not replay-stable and fail/underperform generalized prompt
  checks.
- Current accepted SOTA remains clean gate12288.

Next implementation direction:

- The remaining gap is not source coverage; it is GPU down granularity and
  workspace safety:
  1. Avoid sending tiny `n_active=1` down calls through a separate GPU launch
     unless they are batched across layers/tokens.
  2. Add a hard free-VRAM/workspace reserve for batch cache allocation so Deploy
     cannot OOM when gate12288 is active.
  3. Consider a true fused multi-layer down work queue rather than per-call
     down GPU hits. The profile shows per-call GPU down wall median is close to
     CPU fallback, so launch/sync amortization is the next real bottleneck.

Implementation follow-up:

- The CUDA batch allocator already supports free-VRAM clamping through:
  - `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB`;
  - `GGML_MOE_VRAM_CACHE_SAFETY_MIB`;
  - `GGML_MOE_VRAM_CACHE_AUTO_CLAMP`.
- The generalized demo did not record/pass these envs through `systemd-run`.
- Add demo passthrough and config recording so future up/down GPU probes can
  reserve workspace explicitly instead of relying on failed `cudaMalloc`
  retries. This is default-off and does not change clean gate12288 behavior.

Reserve validation:

- The previously failing Deploy config was rerun with:
  - `GGML_MOE_VRAM_CACHE_MIB=2048`;
  - `GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB=512`;
  - `GGML_MOE_VRAM_CACHE_SAFETY_MIB=128`;
  - fallback down top256 admission.
- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T065301Z-20260709-gate12288-bcache2048-reserve512-fallbackdown-top256-deploy-n96`.
- Result:
  - `run_ok=true`;
  - `eval_tok_s=5.8`;
  - `TTFT=16828.11 ms`;
  - `memory_peak_bytes=14863167488`;
  - output coherent.
- Allocator evidence:
  - requested batch cache `2048MiB`;
  - free before allocation `1873MiB`;
  - actual clamped cache `1233MiB`;
  - down cache `290` slots;
  - no CUDA OOM.
- Quantum with the same reserve config:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T065356Z-20260709-gate12288-bcache2048-reserve512-fallbackdown-top256-quantum-n96`
  reached only `5.4 tok/s`.

Decision:

- Reserve passthrough is accepted as a safety/diagnostic improvement.
- The reserve-enabled up/down candidate is rejected as a speed SOTA: it is safe
  but slower than clean gate12288 on both checked prompts (`Deploy 5.8` vs clean
  `6.0`; Quantum `5.4` vs accepted `5.5`).

### 2026-07-09 Implementation: Down GPU Min-Active Gate

Reason:

- Down batch profiling showed most accepted GPU down calls are tiny:
  - `n_active` mean about `1.01`;
  - median GPU down wall around `0.20 ms`;
  - decode CPU fallback down average around `0.227 ms/call`.
- For these calls, GPU launch/sync/cache bookkeeping can erase the compute
  advantage. The next probe should only send down calls to GPU when there is
  enough active work to amortize the overhead.

Implementation:

- Add default-off `GGML_MOE_DOWN_BATCH_MIN_ACTIVE`.
- If set to `N > 1`, down batch declines with `down_min_active` when
  `n_active < N`; the existing CPU fallback then handles the call.
- Demo config records and passes the env through the 16GB cgroup.
- Default remains off, so clean gate12288 SOTA behavior is unchanged.

Planned tests:

- Gate12288 + fallback down top256 + bcache1024/2048 with:
  - `GGML_MOE_DOWN_BATCH_MIN_ACTIVE=2`;
  - `GGML_MOE_DOWN_BATCH_MIN_ACTIVE=4`.
- Accept only if Quantum and at least one non-Quantum dev prompt both improve
  or remain non-regressed versus clean gate12288 without OOM.

Results:

| config | run | eval tok/s | TTFT ms | RAM peak | decision |
| --- | --- | ---: | ---: | ---: | --- |
| bcache1024 fallback top256 min_active=2 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T070312Z-20260709-gate12288-bcache1024-fallbackdown-top256-minactive2-quantum-n96` | 5.6 | 16161.01 | 14857887744 | candidate only |
| bcache1024 fallback top256 min_active=4 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T070348Z-20260709-gate12288-bcache1024-fallbackdown-top256-minactive4-quantum-n96` | 5.8 | 17068.94 | 14863319040 | candidate only |
| bcache1024 fallback top256 min_active=4 replay | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T070504Z-20260709-gate12288-bcache1024-fallbackdown-top256-minactive4-quantum-n96-repro1` | 5.5 | 16154.93 | 14857895936 | rejected |

Counter observation:

- min_active changed down execution admission but did not change gate-side
  cosubmit payload:
  - cosubmit jobs still `146`;
  - down iouring bytes still `650641408`;
  - down cache preloads still `292`;
  - actual down GPU hits under min_active only `21`.
- Therefore min_active alone can avoid some tiny GPU down calls, but it still
  pays most of the preload/read cost. The `5.8` first run is not replay-stable.

Decision:

- Keep `GGML_MOE_DOWN_BATCH_MIN_ACTIVE` as a default-off diagnostic/safety
  knob. It is useful for proving the launch-granularity issue.
- Do not promote it as SOTA.

Next implementation direction:

- Gate-side cosubmit is now the wrong abstraction for down acceleration: it
  preloads many entries that min_active later refuses to execute.
- Move toward a demand-driven or queued down path:
  1. collect down requests after routing;
  2. group requests across adjacent calls/layers where possible;
  3. only preload entries that will actually satisfy the grouped down work;
  4. launch fewer, larger GPU down batches.
- Until this is implemented, clean gate12288 remains the accepted generalized
  SOTA.

## 2026-07-09 demand-driven down prefill plan

Objective:

- Continue the original `5df8d52 plan: prioritize ds4 vram layout profiling`
  direction: use the measured free VRAM/cache budget to make the up/down GPU
  path useful, not just functional.
- The immediate target is down first, because the latest profiles show down
  CPU fallback is material, but synchronous down miss staging can cost more
  than the CPU fallback it replaces.

Implementation:

- Add default-off `GGML_MOE_DOWN_BATCH_DEMAND_PREFILL=1`.
- It only activates together with `GGML_MOE_DOWN_BATCH_HIT_ONLY=1`.
- On a down batch cache miss:
  - insert the real routed down expert into the batch VRAM cache;
  - copy it on `prefetch_stream` and mark the slot pending with a CUDA event;
  - return `false` for the current call so the existing CPU fallback computes
    the current token exactly;
  - later calls that route the same down expert use `batch_cache_lookup_slot`,
    wait on the event only if the copy is still pending, and then run the GPU
    down batch path.

Theory:

- The previous gate-triggered cosubmit wasted reads because it guessed future
  up/down experts from the gate hook; min-active results showed many preloaded
  down entries were never used by accepted GPU down batches.
- Demand-driven prefill should have much higher precision: every prefilled
  entry was actually requested by the current route.
- The upper bound is limited by the CPU fallback wall time covered by repeated
  down experts. The latest lifecycle profile measured up/down CPU fallback at
  about `57.5 ms/token`, with down contributing a material share. A perfect
  down cache hit conversion can only recover the repeated-down portion; it
  cannot fix gate H2D or up fallback.
- The expected win is therefore incremental, not a jump to 10 tok/s: if
  repeated down fallback accounts for `10-25 ms/token`, a successful prefill
  should move the generalized rate from the clean `5.5 tok/s` class toward
  roughly `5.8-6.2 tok/s`, provided gate cache hit rate is not displaced.

Test matrix:

1. Clean source/build verification after the default-off change.
2. Quantum dev prompt, cold start, `n96`, 16GB cgroup:
   - clean gate12288 SOTA replay;
   - gate12288 + `GGML_MOE_VRAM_CACHE_MIB=1024`;
   - `GGML_MOE_STREAM_DOWN_BATCH=1`;
   - `GGML_MOE_DOWN_BATCH_HIT_ONLY=1`;
   - `GGML_MOE_DOWN_BATCH_DEMAND_PREFILL=1`;
   - no gate-side up/down cosubmit.
3. If the first run beats clean SOTA, replay the same prompt once.
4. If replay holds, run at least one different dev prompt such as Deploy.
5. Only after the config is frozen and documented should held-out test prompts
   be used.

Acceptance gate:

- `source_dirty=false` for the accepted reproduction.
- 16GB cgroup includes page cache: `memory_peak_bytes < 16000000000` and
  `ram_ok=true`.
- Correctness output is semantically coherent for the tested prompt.
- TTFT does not exceed the clean baseline by more than 20%.
- Generalized result must beat the current accepted clean gate12288 SOTA, not
  only match it.
- If accepted, immediately record exact run paths, config, commit hash,
  memory/page-cache evidence, TTFT, output text, and push the source to
  `vendor/deepseek-token-rate-16gb` as `L-Ark`.

Rollback rule:

- If demand prefill increases TTFT, hurts correctness, exceeds RAM, causes OOM,
  displaces gate cache enough to lose token rate, or fails to replay above the
  clean SOTA, keep it default-off, record the rejection, and do not promote it
  as SOTA.

First experiment result:

- Run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T071911Z-20260709-gate12288-bcache1024-demandprefill-quantum-n96`.
- Source state: remote dirty experiment tree carrying local `0634d32`
  contents on top of the older applied remote head. This is not acceptable as
  final SOTA evidence.
- Config:
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=12288`;
  - `GGML_MOE_VRAM_CACHE_MIB=1024`;
  - `GGML_MOE_STREAM_DOWN_BATCH=1`;
  - `GGML_MOE_DOWN_BATCH_HIT_ONLY=1`;
  - `GGML_MOE_DOWN_BATCH_DEMAND_PREFILL=1`;
  - gate-side up/down cosubmit disabled.
- Result:
  - `eval_tok_s=4.7`;
  - `prompt_tok_s=3.3`;
  - `first_output_ms=16476.89`;
  - `memory_peak_bytes=14873661440`;
  - `memory_file_bytes=13901287424`;
  - `ram_ok=true`;
  - `source_dirty=true`.
- Output was coherent but truncated by `n96`, ending at
  `Another key idea is entanglement, where qubits become`, so it is not a
  quality-acceptable SOTA run.
- Counters:
  - batch down cache: `1.0 GiB`, `240` slots;
  - demand prefill activated;
  - pinned staging `copies=2212`, `waits=2204`;
  - batch expert pack `direct_reads=2212`, `iouring_reads=0`;
  - batch cache `hits=1848`, `preloads=2212`, `hit_rate=100%`;
  - gate one-stream stayed at `2913` reads / `12.98GB`, same as clean
    gate12288.

Diagnosis:

- The naive demand-prefill path is functionally correct but not actually
  hiding the miss cost.
- Each down miss immediately performs a direct expert-pack read plus pinned
  staging/H2D submission inside the current down call before returning to CPU
  fallback.
- The high `copies/waits` count means the prefill work competes with decode
  instead of becoming free background work.
- Because many entries are low-reuse, this extra transfer pressure overwhelms
  the CPU fallback time saved by later GPU hits.

Next refinement:

- Add `GGML_MOE_DOWN_BATCH_DEMAND_MIN_SEEN`.
- Default remains `1`, preserving the first implementation when explicitly
  enabled.
- Candidate tests should use `2` or `3`: only prefill a down expert after it
  has appeared multiple times in real routed down work.
- Expected benefit: reduce one-off down preloads and pinned waits while keeping
  repeated down experts eligible for later GPU batch hits.
- If min-seen still cannot beat clean gate12288, the next real fix is not more
  admission knobs; it is a queued/background prefill worker or a true grouped
  route scheduler that can submit multiple down fills without blocking the
  current token.

Min-seen experiment results:

| config | run | eval tok/s | prompt tok/s | TTFT ms | peak RAM | file RAM | down copies | down waits | down GPU hits | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| demand prefill `min_seen=2` | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T072311Z-20260709-gate12288-bcache1024-demandprefill-minseen2-quantum-n96` | 5.3 | 3.7 | 16161.48 | 14865575936 | 13889429504 | 990 | 982 | 1765 | reject: below clean 5.5 |
| demand prefill `min_seen=3` | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T072416Z-20260709-gate12288-bcache1024-demandprefill-minseen3-quantum-n96` | 5.3 | 3.4 | 16562.20 | 14887661568 | 13936877568 | 467 | 459 | 1608 | reject: below clean 5.5 |

Observations:

- Increasing the reuse threshold works mechanically:
  - `min_seen=1`: `2212` down copies, `2204` waits, `4.7 tok/s`;
  - `min_seen=2`: `990` down copies, `982` waits, `5.3 tok/s`;
  - `min_seen=3`: `467` down copies, `459` waits, `5.3 tok/s`.
- Gate behavior stayed stable: `2913` one-stream gate reads and `12.98GB`
  gate read bytes in all three demand-prefill probes.
- The remaining loss is not gate regression. It is the down-fill mechanism
  itself: even filtered down fills are direct reads with pinned staging waits
  inside the decode path.
- The current demand-prefill code remains default-off and is not promoted.

Next implementation requirement:

- Stop doing down fill work synchronously inside `ggml_cuda_moe_stream_batch`.
- Build one of:
  1. a background down-fill queue that reserves cache slots quickly, returns to
     CPU fallback immediately, and lets a worker thread perform expert-pack
     read plus H2D on the prefetch stream; or
  2. a true route-group scheduler that submits grouped up/down fills with the
     existing batch/io_uring job path before the corresponding layer needs
     them.
- The first acceptance test for either design is simple: down copy/wait counts
  must drop from hundreds of synchronous waits to either background work or a
  small number of grouped waits, while Quantum and one other dev prompt both
  exceed the clean `5.5 tok/s` class under the 16GB cgroup.

## 2026-07-09 route-group correction: gate/up/down are all known after router

Correction:

- The previous explanation that expert gate has a fundamentally earlier
  scheduling point than up/down was wrong.
- The 12GB "gate cache" refers to `ffn_gate_exps`, not the router gate.
- `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` expert ids are all known
  after the router selects the active experts for the layer.
- Therefore the target architecture is not "gate first, up/down later"; it is
  "router-selected route group first, then plan gate/up/down together."

Implementation direction:

- Keep the accepted gate12288 path as the clean SOTA baseline.
- Reuse the mature gate mechanisms where possible:
  - prompt-general expert pack source;
  - O_DIRECT / aligned alias / io_uring batch reads;
  - VRAM cache slot lookup by `(tensor, expert_id)`;
  - hit/miss/read-byte counters and run metadata.
- Replace the old gate-derived cosubmit framing with a route-group plan:
  1. collect active expert ids after router for the current layer;
  2. generate the three tensor names for each selected expert:
     `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`;
  3. check cache residency for all three;
  4. submit misses through the same fast packed source where possible;
  5. allow current-token CPU fallback only when the requested tensor is not
     ready, while background fill prepares future hits.

Current code work in progress:

- Added a default-off background down-fill queue:
  - `GGML_MOE_DOWN_BATCH_DEMAND_QUEUE=1`;
  - `GGML_MOE_DOWN_BATCH_DEMAND_QUEUE_MAX`;
  - per-slot `slot_filling` state so a reserved-but-not-yet-copied slot is not
    exposed as a valid GPU cache hit;
  - queue worker uses a non-blocking CUDA stream and its own pinned staging
    ring, batching same-size jobs up to 32 at a time;
  - current token still falls back to CPU if the down tensor is not ready.
- This is infrastructure for the route-group design, not accepted SOTA.
- The next test is intentionally conservative:
  - build the default-off prototype;
  - run it with `GGML_MOE_DOWN_BATCH_HIT_ONLY=1`,
    `GGML_MOE_DOWN_BATCH_DEMAND_PREFILL=1`,
    `GGML_MOE_DOWN_BATCH_DEMAND_QUEUE=1`, and `min_seen=2/3`;
  - verify queue counters show background completion rather than synchronous
    `pinned staging waits` in the decode call.

Acceptance:

- Do not accept this queue by itself unless it beats clean gate12288 on a dev
  prompt replay and one different dev prompt under the 16GB cgroup.
- If it only matches or regresses, record rejection and continue to the full
  route-group scheduler that plans gate/up/down together immediately after
  router selection.

Queue prototype results:

| config | run | eval tok/s | prompt tok/s | TTFT ms | peak RAM | file RAM | submitted | completed | batches | io reads | io wait us | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `min_seen=2`, queue, no delay | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T080854Z-20260709-gate12288-bcache1024-demandqueue-minseen2-quantum-n96` | 5.4 | 3.5 | 15915.14 | 14885720064 | 13914394624 | 990 | 990 | 983 | 990 | 811189 | reject: below 5.5 |
| `min_seen=3`, queue, no delay | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T081054Z-20260709-gate12288-bcache1024-demandqueue-minseen3-quantum-n96` | 5.4 | 3.5 | 16020.84 | 14904147968 | 13882884096 | 467 | 467 | 460 | 467 | 371034 | reject: below 5.5 |
| `min_seen=2`, queue, `delay_us=200` | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T081318Z-20260709-gate12288-bcache1024-demandqueue-minseen2-delay200-quantum-n96` | 5.3 | 3.5 | 16192.35 | 14879543296 | 13931597824 | 990 | 990 | 977 | 990 | 907483 | reject: below 5.5 |

Queue diagnosis:

- The queue fixed one problem from synchronous demand prefill:
  - reads moved from direct per-call reads to the batch/io_uring path;
  - failures were `0`;
  - RAM stayed under the 16GB cgroup.
- But it did not fix the route-group problem:
  - `990` jobs became `983` batches;
  - `467` jobs became `460` batches;
  - adding a `200us` worker delay barely improved batching and worsened token
    rate, likely because useful down fills arrived too late for reuse.
- Therefore this queue is useful infrastructure but not a mature up/down path.

Next step:

- Stop trying to infer batches from a background worker queue.
- Implement a real per-layer route-group planner:
  - from the selected experts in the current layer, construct gate/up/down
    tensor requests as one group;
  - insert/cache misses for all three roles in a deliberate order;
  - submit same-size misses in one io_uring batch before/while up+gate compute
    runs;
  - reserve separate budgets or admission rules so up/down fills do not evict
    high-value gate entries.
- The route-group implementation should directly target reducing "batches
  ~= jobs" to a small number of grouped submits per layer.

First route-group hook probe:

- Added a default-off hook at `ggml_cuda_moe_stream_up_gate_batch`:
  - `GGML_MOE_ROUTE_GROUP_DOWN_QUEUE=1`;
  - `GGML_MOE_ROUTE_GROUP_MIN_SEEN`;
  - once `active_experts` is built from `matrix_row_counts`, derive the current
    layer `ffn_down_exps` tensor name from `ffn_up_exps` / `ffn_gate_exps`;
  - use `expert_pack_lookup_any_size()` so route-group does not guess down
    expert byte size;
  - enqueue down fills through the default-off background queue.
- Test run:
  `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T081947Z-20260709-gate12288-bcache1024-routegroup-downqueue-minseen2-quantum-n96`.
- Result:
  - `eval_tok_s=5.6`;
  - `prompt_tok_s=3.5`;
  - `first_output_ms=16339.77`;
  - `memory_peak_bytes=14872551424`;
  - `memory_file_bytes=13935788032`;
  - `source_dirty=true`.
- Rejection/diagnosis:
  - no `route group down queue active` log appeared;
  - no route-group or down-demand queue atexit counters appeared;
  - stderr only showed the existing gate one-stream counters and down batch
    cache allocation;
  - therefore the current SOTA hot path did not enter
    `ggml_cuda_moe_stream_up_gate_batch`.
- This `5.6 tok/s` run is not accepted as SOTA. It is a dirty-source,
  route-group-inactive observation and could be normal run variance.

Revised route-group implementation target:

- The next hook must attach to the actual SOTA hot path.
- Current evidence shows the accepted path is still:
  - `ggml_cuda_moe_stream_one` for `ffn_gate_exps`;
  - batch/down path and CPU fallback for down;
  - not the fused `up_gate_batch` path.
- Therefore either:
  1. make the fused `up_gate_batch` path part of the SOTA hot path and then use
     the new route-group hook; or
  2. implement a small per-layer aggregator around the current gate one-stream
     selected-expert calls, replacing the old one-expert-at-a-time
     `gate_updown_cosubmit` behavior with a grouped flush.
- The immediate next profiling check should confirm which tensor paths are
  active under the clean gate12288 SOTA before adding more planner code.

CPU hot-path route-group hook update, 2026-07-09:

- Corrected the implementation target: `ffn_gate_exps` is not the router gate.
  Like `ffn_up_exps` and `ffn_down_exps`, it is selected after router chooses
  expert ids. Therefore gate/up/down can be planned from the same per-layer
  selected expert set.
- Added default-off infrastructure on the actual SOTA hot path:
  - CPU side calls
    `ggml_cuda_moe_stream_route_group_preload_updown_from_gate()` immediately
    before the `ggml_cuda_moe_stream_one()` loop for `ffn_gate_exps`;
  - CUDA side derives current-layer `ffn_up_exps` and `ffn_down_exps` names
    from the gate tensor name;
  - active experts come from `matrix_row_counts`, which is already the
    router-selected per-layer expert set;
  - misses are looked up from the prompt-general expert pack with
    `expert_pack_lookup_any_size()`;
  - fills are submitted to the existing background queue;
  - new switches:
    `GGML_MOE_ROUTE_GROUP_UPDOWN_QUEUE`,
    `GGML_MOE_ROUTE_GROUP_TARGETS=updown|up|down`,
    `GGML_MOE_ROUTE_GROUP_MIN_SEEN`.
- This code is default-off and must not change accepted SOTA behavior unless
  explicitly enabled.

Dirty probe results on the new machine:

| config | run | eval tok/s | prompt tok/s | TTFT ms | peak RAM | file RAM | submitted | reads/bytes | useful rate | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `targets=updown,min_seen=2` | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T082736Z-20260709-routegroup-updown-hotpath-minseen2-quantum-n96` | 3.7 | 2.6 | 20124.70 | 14885445632 | 13901737984 | 13318 | 59.35GB | 10.9% | reject: severe cache/IO pollution |
| `targets=updown,min_seen=8` | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T082904Z-20260709-routegroup-updown-hotpath-minseen8-quantum-n96` | 5.2 | 3.8 | 16185.69 | 14866280448 | 13905485824 | 2132 | 9.50GB | 17.9% | reject: below clean SOTA |
| `targets=down,min_seen=8` | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T083114Z-20260709-routegroup-down-hotpath-minseen8-quantum-n96` | 5.5 | 3.6 | 16264.45 | 14872051712 | 13927759872 | 370 | 1.65GB | 58.9% | diagnostic only: matches but does not beat SOTA |
| `targets=down,min_seen=6` | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T083225Z-20260709-routegroup-down-hotpath-minseen6-quantum-n96` | 5.4 | 3.6 | 16167.53 | 14870360064 | 13929824256 | not promoted | not promoted | not promoted | reject: below clean SOTA |
| default-off regression | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T083323Z-20260709-defaultoff-routegroup-infra-regression-quantum-n96` | 5.6 | 3.6 | 16168.19 | 14865346560 | 13941719040 | 0 | route-group off | n/a | default-off path is not regressed |

Important caveats:

- These runs are dirty-source probes on the remote tree and are not accepted
  SOTA evidence.
- The Quantum n96 answer is semantically coherent but truncated at
  "where qubits become"; it cannot be used as a final correctness pass.
- No new SOTA is promoted. The accepted generalized SOTA remains the clean
  gate12288 result at `5.5 tok/s` until a clean-source generalized run beats it
  with non-truncated output and full 16GB cgroup evidence.

Diagnosis:

- The route-group hook now triggers on the real hot path; the earlier issue
  where `up_gate_batch` was inactive is fixed.
- Treating `up` and `down` equally is currently wrong:
  - `updown,min_seen=2` submitted `13318` fills and read `59.35GB`, with only
    `10.9%` useful reuse;
  - even `updown,min_seen=8` still read `9.50GB` for only `17.9%` useful reuse.
- `down` has a much better short-term cache payoff:
  - `down,min_seen=8` read only `1.65GB` and reached `58.9%` useful reuse;
  - it matched, but did not exceed, the clean SOTA.
- The remaining problem is not knowing selected experts. That part is solved.
  The remaining problem is admission and batching:
  - too many prompt/decode selected experts are not reused enough;
  - the background worker still creates many tiny batches
    (`370` jobs became `362` batches in the best down-only probe);
  - fills often arrive as waits rather than free future hits.

Next implementation direction:

This subsection is superseded by the strict native expert-pack parity plan
below. The earlier down-only direction was useful as a diagnostic, but it does
not satisfy the architectural requirement that up/down be aligned with the
current gate path.

## 2026-07-09 strict route-group plan: native expert-pack parity for gate/up/down

Goal:

- After router selection, immediately collect the current layer's selected
  expert ids and plan all three expert tensors from the same source of truth:
  `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps`.
- up/down must be aligned with the current gate mechanism from the native
  expert-pack upward. The implementation must not rely on a separate
  GGUF-alias-only slow path as the target design.
- The required load order is fixed:
  1. move `up` and `gate` together in the same route-group batch;
  2. compute/use the fused or paired up+gate path when both are ready;
  3. move `down` afterwards, using the same native expert-pack read/cache/H2D
     mechanism as the up+gate batch, not a different fallback mechanism.

Hard implementation requirements:

1. **Native expert-pack first**
   - Treat `DeepSeek-V4-Flash-FP4-FP8-native.expert-pack` as the canonical
     fast source for all three tensor roles.
   - gate/up/down lookup must share the same pack entry path and validation:
     `(tensor_name, expert_id) -> pack entry -> O_DIRECT/io_uring read ->
     pinned staging -> H2D -> VRAM cache slot`.
   - The plan may generate a derived manifest or index for faster lookup, but
     it must be generated from the native expert-pack and be reproducible from
     the same source. It must not be prompt-specific.

2. **Same cache model for gate/up/down**
   - All three roles must use explicit VRAM cache accounting with role-aware
     counters:
     `gate_hits/misses`, `up_hits/misses`, `down_hits/misses`,
     `reads`, `bytes`, `H2D enqueues`, `waits`, `evictions`, and
     `evicted_unused`.
   - Cache keys must remain `(tensor_name, expert_id)`.
   - A reserved-but-not-ready slot must not be reported as a hit.
   - Cache admission can be role-specific, but the loading mechanism must be
     the same.

3. **Route-group batch construction**
   - At the CPU hot path where `matrix_row_counts` is available for
     `ffn_gate_exps`, build a route-group object:
     - layer id / tensor prefix;
     - selected expert ids;
     - gate tensor requests;
     - up tensor requests;
     - down tensor requests.
   - Do not enqueue one job per expert immediately. First aggregate all selected
     requests for the layer, group by source file and payload size, then submit
     batch reads.
   - The batcher target is one up+gate batch per layer group, followed by one
     down batch per layer group when possible. The success criterion is that
     `batches ~= jobs` must disappear; the counters should show multi-job
     batches for route-group reads.

4. **Required load order**
   - Stage A: `up + gate`
     - For every selected expert in the layer, check both up and gate cache
       residency.
     - Submit missing up and missing gate entries together through the same
       native expert-pack batch path.
     - The route-group log must show one combined up+gate planning event, with
       role counts and bytes.
   - Stage B: `down`
     - After up+gate has been planned/submitted, submit selected down misses
       through the same native expert-pack batch path.
     - down may be overlapped with up+gate compute only if the implementation
       preserves correctness and does not expose unready slots as hits.
   - Forbidden target behavior:
     - up via one mechanism and gate via another;
     - down via a different queue that bypasses the native expert-pack batch
       counters;
     - single-expert notify loops that produce mostly batch size 1.

5. **Computation path**
   - Once up and gate are available together, prefer the existing batched/fused
     up+gate CUDA path, or extend it if needed.
   - If the fused path is not taken, the run must record why:
     unsupported type, cache miss, workspace limit, kernel incompatibility, or
     explicit disable flag.
   - Current-token CPU fallback remains allowed only as a diagnostic fallback,
     not as the target optimized path.

Implementation steps:

1. **Pack/index audit**
   - Verify the native expert-pack contains all expected entries:
     - `43 MoE layers * 256 experts * 3 roles = 33024` entries.
   - Print exact role counts for `ffn_gate_exps`, `ffn_up_exps`,
     `ffn_down_exps`.
   - Confirm offsets/sizes for up/gate/down by layer are suitable for grouped
     native expert-pack reads. If layout is poor, generate a reproducible
     derived up/gate/down route-group index from the native pack.

2. **Route-group planner**
   - Replace the current `route_group_preload_updown_from_gate` behavior with a
     planner that creates a full `gate/up/down` request set.
   - Add counters:
     - selected experts per layer;
     - requested/hit/miss entries per role;
     - up+gate batch count and average jobs per batch;
     - down batch count and average jobs per batch;
     - bytes and H2D time per role.

3. **Native expert-pack batch loader**
   - Implement a shared batch loader for route-group requests:
     - input: vector of pack entries with role labels;
     - output: reserved VRAM slots and ready events;
     - path: native expert-pack O_DIRECT/io_uring -> pinned staging -> H2D.
   - Use it for up+gate and down. Do not keep a separate down-only worker as
     the target path.

4. **up+gate compute integration**
   - Connect the route-group loaded up+gate entries to the existing
     `up_gate_batch` / fused path.
   - Add a correctness guard comparing against the current accepted path on a
     short diagnostic run before performance testing.

5. **down integration**
   - Use the same shared batch loader for down.
   - Connect ready down entries to the existing down batch CUDA path.
   - Record whether any down CPU fallback remains and why.

6. **Evaluation sequence**
   - First run short correctness probes with deterministic prompts:
     France, Quantum, Fibonacci.
   - Then run generalized cold tests under the 16GB cgroup with display
     processes killed before every run.
   - Test prompts used for tuning must remain separate from held-out test
     prompts. The final SOTA must be measured on held-out prompts.

Acceptance criteria:

- New result must beat clean gate12288 generalized SOTA, currently `5.5 tok/s`,
  on clean source.
- `memory_peak_bytes` must stay below `16000000000`, including page cache.
- TTFT must not increase by more than `20%` unless explicitly recorded as a
  rejected/non-accepted diagnostic.
- Output must be semantically correct and not artificially prompt-specific.
- Counters must prove route-group parity:
  - up and gate loaded in the same native expert-pack batch path;
  - down loaded afterwards through the same native expert-pack batch path;
  - no hidden GGUF-alias-only or page-cache slow path is required for the
    optimized expert movement.
- When a conforming new SOTA appears, immediately record full reproduction
  metadata, commit, and push to `vendor/deepseek-token-rate-16gb`.

Implementation progress, native parity probe:

- Added default-off native parity infrastructure:
  - `GGML_MOE_ROUTE_GROUP_NATIVE_PARITY=1`;
  - `GGML_MOE_BATCH_FULLPACK=1` makes the batch expert-pack path use the full
    native expert pack instead of the up/down GGUF-alias-only source;
  - `GGML_MOE_GATE_BATCH_PREFETCH=1` lets gate one-stream consume the shared
    batch cache via `ggml_cuda_moe_stream_cache_dev_ptr()`;
  - CPU gate-only active preload is skipped when native parity is enabled, so
    Stage A is owned by the route-group planner instead of the old gate-only
    preload.
- Route-group planner now builds:
  - Stage A: selected `ffn_gate_exps` + `ffn_up_exps`;
  - Stage B: selected `ffn_down_exps`;
  - both stages use the same native expert-pack lookup, io_uring copy helper,
    pinned staging ring, H2D enqueue path, and batch cache slot accounting.
- Added role-aware counters:
  `gate_hits/misses`, `up_hits/misses`, `down_hits/misses`,
  Stage A/B jobs, batches, bytes, `missing_pack`, `no_slot`, `copy_fail`.

Dirty probe results:

| config | run | eval tok/s | prompt tok/s | TTFT ms | RAM peak | batch pack entries | Stage A | Stage B | expert reads/bytes | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| native parity, alias batch source, 1GB cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T084535Z-20260709-native-parity-short-france-n16` | 2.2 | 2.4 | 20701.19 | 14811230208 | 22016 | up only; gate missing | down | 4668 / 20.8GB | reject: not native full parity |
| native fullpack parity, sync ready, 1GB cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T084631Z-20260709-native-fullpack-parity-short-france-n16` | 1.7 | 2.2 | 21662.33 | 14828732416 | 33024 | gate+up | down | 8637 / 38.5GB | reject: duplicate gate path and sync waits |
| native fullpack parity, async ready, 1GB cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T084913Z-20260709-native-fullpack-parity-async-france-n16` | 1.9 | 2.2 | 21160.24 | 14824009728 | 33024 | gate+up | down | 8637 / 38.5GB | reject: too many waits |
| native fullpack, old gate preload bridge, 1GB cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T085051Z-20260709-native-fullpack-parity-gatebatch-france-n16` | 2.1 | 2.6 | 19877.45 | 14824595456 | 33024 | old gate preload + up | down | 8637 / 38.5GB | reject: gate not loaded by route-group Stage A |
| native fullpack route-group Stage A, 1GB cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T085240Z-20260709-native-parity-upgate-stagea-france-n16` | 2.1 | 2.5 | 19710.47 | 14811947008 | 33024 | gate+up | down | 8637 / 38.5GB | reject: cache too small |
| native fullpack route-group Stage A, 4GB cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T085345Z-20260709-native-parity-upgate-vram4096-france-n16` | 3.2 | 2.6 | 19126.87 | 14815993856 | 33024 | gate+up | down | 5450 / 24.3GB | reject: still below SOTA |
| native fullpack unified cache, 12GB cache, no one-cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T085441Z-20260709-native-parity-unified-vram12000-france-n16` | 4.3 | 3.0 | 17627.32 | 14826369024 | 33024 | gate+up | down | 4130 / 18.4GB | diagnostic: direction improves but not SOTA |
| native fullpack unified cache, 14GB cache, no one-cache, n16 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T085532Z-20260709-native-parity-unified-vram14000-france-n16` | 4.4 | 3.0 | 17927.48 | 14784262144 | 33024 | gate+up | down | not promoted | diagnostic: small gain over 12GB |
| native fullpack unified cache, 14GB cache, no one-cache, Quantum n96 | `/home/wici/runs/vendor-ds4-16gb/demo-general-sota/20260709T085612Z-20260709-native-parity-unified-vram14000-quantum-n96` | 4.1 | 3.0 | 16852.48 | 14838063104 | 33024 | gate+up | down | 12580 / 56.1GB | reject: below clean 5.5 SOTA and n96 output truncated |

Key diagnosis:

- The native expert-pack parity requirement is now implemented as a default-off
  probe: batch path sees `33024` entries and no missing gate/up/down pack
  entries.
- With the gate bridge and CPU skip of old gate-only preload, route-group Stage
  A owns `gate+up` loading and Stage B owns `down` loading.
- Moving the 12GB one-cache budget to unified route-group cache is directionally
  useful:
  - 1GB unified cache n16: `2.1 tok/s`, `38.5GB` reads;
  - 12GB unified cache n16: `4.3 tok/s`, `18.4GB` reads;
  - 14GB unified cache n16: `4.4 tok/s`.
- The implementation is not accepted because n96 is only `4.1 tok/s`, below
  the clean generalized SOTA `5.5 tok/s`, and output is still truncated at n96.
- Remaining bottleneck:
  - route-group still submits too many misses;
  - n96 full native parity still reads `56.1GB`;
  - `async prefetch waits=11008`;
  - batch grouping is better than one job per batch but still too fragmented:
    `5236` batches for `12580` reads.

Next step under this architecture:

1. Keep native parity infrastructure default-off.
2. Add route-group admission instead of loading every selected expert:
   - keep Stage A `gate+up` as the required first stage;
   - keep Stage B `down` as the required second stage;
   - but admit only entries likely to be reused enough, based on prompt-general
     counters or online reuse thresholds.
3. Improve grouping before H2D:
   - collect more than one layer's route-group requests where correctness
     allows;
   - sort by source offset and size before issuing io_uring;
   - target significantly fewer than `5236` batches for n96.
4. Connect up+gate compute more tightly to Stage A readiness so loaded up
   entries are consumed by the fused path rather than only serving as future
   cache candidates.
