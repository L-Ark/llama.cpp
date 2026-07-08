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

Before every official cold benchmark, profile run, or SOTA validation on the new
machine:

1. Stop display usage so VRAM is not consumed by graphical processes:

   ```bash
   echo '12345678' | sudo -S systemctl stop display-manager || true
   pkill -f gnome-shell || true
   pkill -f '/usr/lib/xorg/Xorg' || true
   ```

2. Confirm `nvidia-smi` shows no running GPU processes and only driver-reserved
   VRAM remains.
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

Runs that skip the display-process cleanup are diagnostic only and must not be
promoted as SOTA.

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
