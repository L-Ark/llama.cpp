# GLM vramctl-style SSD I/O Pipeline

## Goal

Use the SSD I/O optimization ideas from `/home/wici/lfz/vramctl` to improve
GLM expert-pack backed runs in `ik_llama`, with end-to-end runtime metrics as
the primary target:

- TTFT
- first visible token time
- total runtime
- eval tok/s

## Constraints

- Work only inside `/home/wici/lfz`.
- Do not modify or delete anything outside `/home/wici/lfz`.
- Keep task plans, logs, summaries, and local runners inside this task
  directory.
- Preserve the strict 5090 2 GB host-RAM line for validation:
  `MemoryMax=2G`, `MemorySwapMax=0`, and `GGML_MOE_RAM_TIER_MIB=0`.
- Do not mix this SSD I/O change with routing, SER, VRAM cache, or profile
  strategy changes.
- Keep existing `GGML_MOE_IO_BACKEND=mmap` and `direct` behavior unchanged.

## Implementation Plan

1. Add env-gated `GGML_MOE_IO_BACKEND=iouring` to
   `ggml/src/ggml-cuda/moe_stream_batch.cu`.
2. Use vramctl's practical defaults for the first implementation:
   - `GGML_MOE_IO_BYTES=2097152`
   - `GGML_MOE_IO_DEPTH=32`
   - `GGML_MOE_IO_REQUIRE_DIRECT=0`
   - `GGML_MOE_IO_SQPOLL=0`
   - `GGML_MOE_IO_FIXED_BUFS=0`
3. Implement `io_uring` reads for batched expert-pack miss jobs:
   - keep O_DIRECT;
   - reuse existing pinned staging slots;
   - submit multiple read requests;
   - on completion, enqueue H2D copy to the target VRAM cache slot;
   - fallback per-job to the existing synchronous path when needed.
4. Add atexit counters for:
   - `iouring_reads`
   - `iouring_bytes`
   - `iouring_fallbacks`
   - `iouring_submit_us`
   - `iouring_wait_us`
   - `iouring_h2d_enqueues`
5. Build and run a smoke validation.

## Progress

- 2026-06-12: Read `/home/wici/lfz/Agent.md`.
- 2026-06-12: Read `.Agent/Agent.md`.
- 2026-06-12: Checked current branch and worktree. Existing untracked files
  are prior `.Agent` task artifacts and will be left untouched.
- 2026-06-12: Read relevant vramctl SSD code and notes. Key methods to adapt:
  O_DIRECT, io_uring, multiple pinned bounce/staging buffers, queue depth,
  completion-driven reads, and fallback handling.
- 2026-06-12: Located current GLM expert-pack read path in
  `ggml/src/ggml-cuda/moe_stream_batch.cu`.
- 2026-06-12: Implemented env-gated `GGML_MOE_IO_BACKEND=iouring` in
  `ggml/src/ggml-cuda/moe_stream_batch.cu`.
  - Existing `mmap` and `direct` modes remain default-compatible.
  - `iouring` uses O_DIRECT plus liburing for batched expert-pack miss jobs.
  - It falls back to the existing synchronous path if a batch is not suitable.
  - Added stderr counters for io_uring reads, bytes, fallbacks, submit/wait
    time, and H2D enqueues.
- 2026-06-12: Build passed:
  `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`.
- 2026-06-12: Ran short `-n 4` smoke with `GGML_MOE_IO_BACKEND=iouring`.
  Functional result passed: exit `0`, `read_failures=0`, `iouring_reads=6420`.
  However this run did not enable parallel stage job vectors, so it exercised
  the conservative single-read io_uring path:
  - direct short smoke: `total_ms=24999.55`, `eval=0.88 tok/s`;
  - iouring single-read smoke: `total_ms=25103.20`, `eval=0.85 tok/s`;
  - conclusion: single-read io_uring is not the target optimization path.
    Need validate the batched/parallel stage path, which is the vramctl-style
    design point.
- 2026-06-12: Changed single-read io_uring to opt-in only via
  `GGML_MOE_IO_URING_SINGLE=1`, because per-read ring setup is slower than the
  existing direct path on this machine.
- 2026-06-12: Added persistent `io_uring` queues to pinned staging rings and
  wired batched runtime-miss jobs through completion-driven direct reads plus
  async H2D copies. The preloaded profile entries still use the existing
  direct read path; runtime cache misses use io_uring when
  `GGML_MOE_IO_BACKEND=iouring`.
- 2026-06-12: Invalid longer probes were discarded:
  - missing `-ngl 79 --defer-experts` caused no GPU offload and 2 GB cgroup
    OOM;
  - `-ngl 99` attempted full expert GPU allocation and failed CUDA OOM;
  - missing `GGML_CUDA_NO_PINNED=1` attempted a 231.73 GiB pinned host
    allocation and failed before inference.
- 2026-06-12: Final build and whitespace checks passed:
  - `git diff --check`;
  - `cmake --build /home/wici/lfz/ik_llama/build-cuda --target llama-cli -j 8`.

## Results

All valid runs below used the strict host-RAM line:
`MemoryMax=2G`, `MemorySwapMax=0`, `GGML_MOE_RAM_TIER_MIB=0`, and the same
GLM expert-pack/profile/cache settings. Long runs also used the known-good GLM
runtime flags: `GGML_CUDA_NO_PINNED=1`, `-ngl 79`, `-fa on`, `-mla 3`,
`-cmoe`, and `--defer-experts`.

| run | backend | n | total_ms | prompt_eval_ms | eval_ms | eval tok/s | VRAM hit | expert reads | failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `smoke-direct-parallel-n4` | direct | 4 | 24030.34 | 10729.43 | 2743.87 | 1.09 | 14.8% | `direct=6420` | 0 |
| `smoke-iouring-parallel-n4-v4` | iouring | 4 | 24572.66 | 11195.71 | 2666.00 | 1.13 | 14.8% | `direct=1800`, `iouring=4620` | 0 |
| `long-direct-parallel-full-nopinned-n84` | direct | 84 | 103564.18 | 11449.87 | 81328.70 | 1.02 | 7.1% | `direct=140550` | 0 |
| `long-iouring-parallel-full-nopinned-n84` | iouring | 84 | 100881.00 | 11184.83 | 78904.21 | 1.05 | 7.1% | `direct=1800`, `iouring=138750` | 0 |

Long-run delta, iouring versus direct:

- total runtime improved by `2683.18 ms` (`2.6%`);
- prompt eval improved by `265.04 ms` (`2.3%`);
- eval improved by `2424.49 ms` (`3.0%`);
- eval throughput improved from `1.02` to `1.05 tok/s`;
- VRAM hit rate stayed identical at `7.1%`, so the measured gain is from the
  I/O path rather than a cache-policy change.

Conclusion: the vramctl-style batched `io_uring` path is functional and gives a
small but measurable improvement on the fast SSD machine for the `-n 84`
GLM run. The short `-n 4` end-to-end result is noisier and slightly worse in
total time because load/prompt overhead dominates, but runtime token evaluation
still improves there (`1.09` to `1.13 tok/s`).
