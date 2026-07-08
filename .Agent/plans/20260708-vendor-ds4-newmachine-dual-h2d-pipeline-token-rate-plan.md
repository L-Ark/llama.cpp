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
