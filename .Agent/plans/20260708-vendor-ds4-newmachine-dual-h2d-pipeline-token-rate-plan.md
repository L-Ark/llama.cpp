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

## Required Pre-Run Procedure

Before every model run on the new machine, including diagnostic runs, profile
runs, cold benchmarks, and SOTA validation, display processes must be
killed/stopped first. This is a hard promotion requirement because graphical
processes consume VRAM and can change the effective cache/workspace budget.

中文硬性要求：每次运行模型前必须先杀掉显示进程；没有完成并记录该步骤的 run
不能进入 baseline/SOTA/候选优化比较，只能作为无效诊断参考。

1. Kill/stop display usage before launching the model:

   ```bash
   echo '12345678' | sudo -S systemctl stop display-manager || true
   echo '12345678' | sudo -S pkill -f gnome-shell || true
   echo '12345678' | sudo -S pkill -f Xorg || true
   echo '12345678' | sudo -S pkill -f '/usr/lib/xorg/Xorg' || true
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

2. **Two-stage gate/up/down aggregation**
   - After router selection, collect all current-layer gate/up/down expert read
     requests before submitting IO.
   - Sort requests by source offset and submit them as one aggregated plan.
   - Keep default behavior unchanged; gate with env such as
     `GGML_MOE_GATE_UPDOWN_AGGREGATE=1`.
   - Fallback to the current path on any unsupported dtype, pack miss, alias
     miss, short read, or correctness risk.

3. **Larger H2D batching / copy coalescing**
   - Combine adjacent or same-batch staging buffers into fewer H2D submissions
     where tensor layout and kernel inputs remain unchanged.
   - Add profile fields for H2D copy count, total H2D bytes, H2D ms, average
     copy size, and per-token H2D wait.
   - This is expected to help Case A most, but should also reduce wait overhead
     in Case B.

4. **Reduce H2D bytes with compact or partial up/down source**
   - First produce a hard-bound profile: actual rows/blocks used per
     token/layer, full-expert bytes copied, partial-copy bytes needed, and
     projected token-rate upper bound for Case A and Case B.
   - Only implement a default-off compact source if the bound shows meaningful
     gain. It must not replace the native full expert pack or rely on a
     prompt-specific profile.

5. **Prompt-general VRAM cache admission**
   - Use only calibration/dev prompts for tuning. Held-out prompts are final
     test only.
   - Improve gate/up/down split-pool and admission behavior to reduce misses
     without hurting the gate fullpack path.
   - Reject any candidate that improves France only or reduces generalized
     correctness.
   - Add a default-off gate hot-pool experiment using only prompt-general
     calibration manifests. It must preserve manifest hot-order, bypass
     per-token gate one-pack reads only on exact tensor/expert hits, and fall
     back to the current path on any miss. Expected upper bound is modest:
     every 1GB of effective gate residency saves at most about 1GB of repeated
     SSD/H2D movement, or roughly 0.15s on the current 6.6-6.8GB/s H2D path, so
     this is a validation step rather than the whole route to `>5 tok/s`.

6. **Non-expert mmap/page-cache pressure**
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
