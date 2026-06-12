# 5090 diffusiongemma-26B-A4B-it-GGUF Reproduction Plan

Goal:

Reproduce `unsloth/diffusiongemma-26B-A4B-it-GGUF` on RTX 5090 with system RAM
limited to 2 GB, targeting output throughput close to or reaching the reported
~700 token/s result, and report all metrics required by `.Agent/Agent.md`.

## Requirements

- Hardware: RTX 5090, 32 GB VRAM.
- Host RAM limit: `MemoryMax=2G`.
- Model: `unsloth/diffusiongemma-26B-A4B-it-GGUF` from Hugging Face.
- Target: close to or reach ~700 output token/s if the model/runtime supports
  the same reproduction path.
- Report:
  - `direct_reads`
  - `VRAM hit`
  - `RAM hit`
  - `eval tok/s`
  - `prompt_eval tok/s`
  - `total_ms`
  - `TTFT`
  - `time_to_type_s`
  - `first_visible_s`
  - `accuracy` if using an accuracy sample
  - `read_failures`
  - full config and log paths

## Initial Plan

1. Read `.Agent/Agent.md` and follow its plan/progress rules.
2. Inspect current repo state and GPU state.
3. Research the Hugging Face model card and files to identify:
   - exact GGUF filename/quantization used for the 700 tok/s claim;
   - required llama.cpp flags;
   - context length, batch/ubatch, diffusion-specific flags, and chat template.
4. Check whether current `ik_llama` build supports diffusiongemma/Gemma 3n style
   model execution and the needed GGUF metadata.
5. Locate or download the needed model into a non-committed model/data path.
6. Build a 2 GB host-RAM constrained run command using `systemd-run`.
7. Run a short smoke test first, then a throughput benchmark.
8. Collect all required metrics from stdout/stderr/logs.
9. Compare result against ~700 token/s and document gaps or blockers.

## Progress Log

- 2026-06-12: Read `.Agent/Agent.md`.
- 2026-06-12: Confirmed repo branch
  `feat/glm51-5090-2gb-vram-standalone`; only `.Agent/` is untracked.
- 2026-06-12: Confirmed GPU:
  `NVIDIA GeForce RTX 5090`, total VRAM `32607 MiB`, used `221 MiB`, free
  `31889 MiB`, utilization `0%`.
- 2026-06-12: Researched Hugging Face model card. Important finding:
  `unsloth/diffusiongemma-26B-A4B-it-GGUF` requires the DiffusionGemma
  llama.cpp PR/branch and the dedicated `llama-diffusion-cli`; standard
  `llama-cli` / `llama-server` cannot generate from it yet.
- 2026-06-12: Model card recommended GGUF options:
  - Q8_0: about 25-26.9 GB, recommended near-lossless.
  - Q4_K_M: about 16-16.8 GB, smallest and fits a single 24 GB GPU.
  - Example command:
    `llama-diffusion-cli -m diffusiongemma-26B-A4B-it-Q8_0.gguf -ngl 99 -cnv -n 2048`.
- 2026-06-12: Local `ik_llama` build does not contain
  `llama-diffusion-cli`. Available binaries include `llama-cli`,
  `llama-bench`, `llama-gemma3-cli`, and others, but not the required
  diffusion runner.
- 2026-06-12: Checked local filesystem; no existing diffusiongemma GGUF or
  `llama-diffusion-cli` binary was found.
- 2026-06-12: Disk free under `/root` is about `72G`. This is enough for one
  Q8_0 or Q4_K_M download plus a runner checkout, but not enough for careless
  duplicate large downloads.
- 2026-06-12: Cloned `ggml-org/llama.cpp` into
  `/root/lfz/llama.cpp-diffusiongemma`, fetched PR `24423` into local branch
  `diffusiongemma`, current commit `10a2613aa`.
- 2026-06-12: First CUDA CMake configure failed because CMake could not find
  a CUDA compiler (`CMAKE_CUDA_COMPILER-NOTFOUND`) even though CUDA toolkit
  headers were detected under `/usr/local/cuda/include`. Next step is to
  compare the working `ik_llama/build-cuda` compiler configuration and reuse it
  if possible.
- 2026-06-12: Reconfigured with explicit 5090 CUDA settings:
  `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc`
  and `-DCMAKE_CUDA_ARCHITECTURES=120`. Build completed successfully and
  produced:
  `/root/lfz/llama.cpp-diffusiongemma/build-cuda/bin/llama-diffusion-cli`.
- 2026-06-12: `ldd` confirmed the runner loads local PR branch libraries and
  CUDA libraries, including `libggml-cuda.so.0`, `libcudart.so.13`,
  `libcublas.so.13`, and `libcublasLt.so.13`.
- 2026-06-12: `llama-diffusion-cli --help` confirmed diffusion-specific
  runtime flags are available, including `--diffusion-eb`,
  `--diffusion-eb-max-steps`, `--diffusion-kv-cache`,
  `--diffusion-gpu-sampling`, and `--diffusion-gpu-sample-reduce`.
- 2026-06-12: Decided to download only Q8_0 first because the 5090 has 32 GB
  VRAM and the model card recommends Q8_0. If Q8_0 cannot run within the 2 GB
  host RAM constraint or VRAM budget, fall back to Q4_K_M.
- 2026-06-12: Downloaded Q8_0 to
  `/root/lfz/models/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q8_0.gguf`.
  File size is about `26G`; disk free after download is about `46G`.
- 2026-06-12: Confirmed no residual model process and GPU is idle before smoke:
  VRAM used `223 MiB`, free `31887 MiB`, utilization `0%`.
- 2026-06-12: Ran first 2 GB RAM smoke with Q8_0:
  `-ngl 99 -p 'Why is the sky blue?...' -n 256 --perf` under
  `systemd-run MemoryMax=2G MemorySwapMax=0`.
  Result: failed with `status=1`, `Finished with result: oom-kill`.
  No output was produced. Stderr showed context auto-sized to
  `n_ubatch=2304 n_batch=2304 n_ctx=2304`, then systemd killed the process
  after about `20.658s`. This indicates Q8_0 cannot be used with the current
  runner under a strict 2 GB host-RAM cap without further host-memory reduction.
  Next step: download and test Q4_K_M.
- 2026-06-12: Downloaded Q4_K_M to
  `/root/lfz/models/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`.
  File size is about `16G`. Disk free after keeping both Q8_0 and Q4_K_M is
  about `30G`; avoid additional large downloads.
- 2026-06-12: Ran Q4_K_M 2 GB RAM smoke with
  `-ngl 99 -p 'Why is the sky blue?...' -n 256 --perf` under
  `systemd-run MemoryMax=2G MemorySwapMax=0`.
  Result: failed with `status=1`, `Finished with result: oom-kill`.
  No output was produced. Stderr again showed context auto-sized to
  `n_ubatch=2304 n_batch=2304 n_ctx=2304`, then systemd killed the process
  after about `8.510s`.
- 2026-06-12: Since both Q8_0 and Q4_K_M fail under 2 GB host RAM before
  output, the next attempts focus on reducing host-side buffers/staging rather
  than changing quantization: try `--no-host`, `-ngl all`, and single-GPU
  split mode `-sm none`.
- 2026-06-12: Tried Q4_K_M with
  `-ngl all -sm none --no-host -n 256 --perf` under `MemoryMax=2G`.
  Result: still failed with `oom-kill` after about `8.046s`, with the same
  diffusion context sizing (`n_ubatch=2304 n_batch=2304 n_ctx=2304`) and no
  generated output. Since `--no-host` did not help, the likely remaining cause
  is file-backed mmap/page-cache or mandatory load-time staging counted against
  the cgroup. Next attempt: add `--direct-io`.
- 2026-06-12: Tried Q4_K_M with
  `-ngl all -sm none --no-host --direct-io -n 256 --perf` under
  `MemoryMax=2G`. Stderr confirmed direct I/O disabled mmap, but the run still
  failed with `oom-kill` after about `8.217s`. This indicates the current
  DiffusionGemma PR runner needs more than 2 GB host RSS for mandatory
  load/context setup even when mmap is disabled. Next diagnostic: run a small
  RAM limit ladder to find the minimum host RAM required, without redefining
  the final 2 GB target.
- 2026-06-12: Ran Q4_K_M with the same direct-io/no-host command under
  `MemoryMax=4G`. Result: success. Output was produced. Diffusion settings:
  `max_steps=48`, entropy-bound stopped after `27` actual steps over `1`
  block.
  Reported generation timing (`max_steps=48`, stopped after `27` actual
  steps):
  `total time: 1893.36ms`, `time per step: 70.12ms`,
  `throughput: 135.2 tok/s`, `in-step parallel 3651 tok/s`.
  This proves the runner/model path works on the 5090, but the strict 2 GB
  host-RAM requirement is not yet satisfied and throughput is far below the
  ~700 tok/s target.
- 2026-06-12: Inspected
  `/root/lfz/llama.cpp-diffusiongemma/examples/diffusion/diffusion-cli.cpp`.
  The runner hardcodes `needed = blocks * canvas_length + 2048`, so even
  `-n 256` forces `n_ubatch=2304 n_batch=2304 n_ctx=2304`. Command-line
  `-ub/-b/-c` cannot reduce this because the code uses `std::max`. Next
  attempt: patch the external PR runner only, preserving default behavior, to
  allow an environment variable to reduce prompt headroom for short smoke runs.
- 2026-06-12: Patched the external PR runner only (not `ik_llama`) to support
  `LLAMA_DIFFUSION_PROMPT_HEADROOM`, defaulting to `2048` so upstream behavior
  is unchanged. Rebuilt `llama-diffusion-cli` successfully.
- 2026-06-12: Verified the patch under `MemoryMax=4G` with
  `LLAMA_DIFFUSION_PROMPT_HEADROOM=128`. Log showed
  `n_ubatch=512 n_batch=2048 n_ctx=384`, so the environment variable reduced
  `n_ctx` but the default `n_batch=2048` remained high. The run succeeded with
  `total time=1765.65ms`, `max_steps=48`, `27` actual steps,
  `throughput=145.0 tok/s`. Next attempt:
  under the strict 2 GB cap, explicitly set `-b`, `-ub`, and `-c` near the
  reduced prompt+canvas requirement.
- 2026-06-12: Tried strict 2 GB with reduced headroom and explicit reduced
  batch/context:
  `LLAMA_DIFFUSION_PROMPT_HEADROOM=128 -b 384 -ub 384 -c 384`.
  Log confirmed `n_ubatch=384 n_batch=384 n_ctx=384`, but the run still
  failed with `oom-kill` after about `8.604s`. Therefore the 2 GB failure is
  not primarily caused by the auto `2304` context/batch sizing. Next diagnostic:
  find the minimum MemoryMax with the reduced config.
- 2026-06-12: Same reduced config succeeded at `MemoryMax=3G` with
  `total time=1746.38ms`, `max_steps=48`, `27` actual steps,
  `throughput=146.6 tok/s`. It failed at
  `MemoryMax=2500M`, so the current minimum host RAM is between 2.5 GB and
  3 GB.
- 2026-06-12: Tried reducing output to `-n 64` with
  `LLAMA_DIFFUSION_PROMPT_HEADROOM=64 -b 320 -ub 320 -c 320` under
  `MemoryMax=2G`. Log confirmed `-n 64 -> 1 blocks` because the model canvas is
  fixed at 256 tokens, then the run still failed with `oom-kill`. Thus reducing
  requested output below 256 does not bypass the fixed one-canvas memory floor.
  Next diagnostic: inspect CUDA backend host staging/pinned-buffer controls.
- 2026-06-12: Tried strict 2 GB with `GGML_CUDA_NO_PINNED=1`, direct I/O,
  no host buffers, reduced prompt headroom, and explicit
  `-b 384 -ub 384 -c 384`. Log path:
  `runs/smoke-q4-2g-nopinned-head128-b384-n256/`. Result: still failed with
  `oom-kill` after about `20.054s`, no generated output. Log confirmed
  `n_ubatch=384 n_batch=384 n_ctx=384`, so disabling pinned CUDA host buffers
  did not reduce mandatory host RSS enough to satisfy `MemoryMax=2G`.
- 2026-06-12: Next plan before further attempts:
  - inspect `llama-diffusion-cli` help/source for host-memory, offload, repack,
    and diffusion early-stop controls;
  - test whether 2 GB can run by moving unavoidable host pressure to swap/SSD
    or by disabling additional staging/repack behavior;
  - separately test diffusion step controls on a known-working memory limit to
    determine whether the advertised ~700 tok/s corresponds to fewer diffusion
    steps rather than a missing cache mechanism;
  - report ik_llama-specific MoE cache metrics as unavailable for this external
    DiffusionGemma runner unless the log contains equivalent counters.
- 2026-06-12: `llama-diffusion-cli --help` confirmed relevant controls:
  `--no-repack`, `--no-host`, `--direct-io`, `--mmap/--no-mmap`,
  `--diffusion-eb-max-steps`, `--diffusion-eb-entropy-bound`,
  `--diffusion-eb-confidence`, `--diffusion-kv-cache`,
  `--diffusion-gpu-sampling`, and `--diffusion-gpu-sample-reduce`.
  System swap is currently absent (`Swap: 0B`), so `MemorySwapMax=0` forbids
  any SSD-backed spill. Next 2 GB attempt: add `--no-repack` to test whether
  weight repacking is the extra host-memory source.
- 2026-06-12: Tried strict 2 GB with `--no-repack` added:
  `runs/smoke-q4-2g-norepack-head128-b384-n256/`. Result: still failed with
  `oom-kill` after about `23.107s`, no generated output. This indicates weight
  repacking is not the main mandatory host-RAM source. Next attempt: keep
  physical RAM capped via `MemoryMax=2G`, but add an SSD-backed swap file and
  allow cgroup swap, following the accepted fallback path of placing data on
  SSD when it cannot fit in RAM.
- 2026-06-12: Created and enabled temporary SSD-backed swap file
  `/root/lfz/tmp/diffusiongemma-8g.swap` (8 GB). Ran Q4_K_M with
  `MemoryMax=2G MemorySwapMax=8G`, `--direct-io --no-host --no-repack`,
  `GGML_CUDA_NO_PINNED=1`, and reduced headroom/batch/context. Log path:
  `runs/smoke-q4-2g-swap8g-head128-b384-n256/`. Result: success. Output was
  produced, but generation timing fell to `total time=4654.06ms`,
  `max_steps=48`, `27` actual diffusion steps, `throughput=55.0 tok/s`,
  `in-step parallel=1485 tok/s`.
  This satisfies "system RAM capped at 2 GB with SSD fallback" but is far from
  the ~700 tok/s target; SSD/swap is a correctness fallback, not a performance
  path. Next: sweep diffusion early-stop/max-step controls on a known-working
  physical-RAM limit to test whether ~700 tok/s is achieved by fewer diffusion
  steps.
- 2026-06-12: Swept `--diffusion-eb-max-steps` on Q4_K_M with
  `MemoryMax=3G MemorySwapMax=0`, reduced headroom/batch/context, direct I/O,
  no host buffers, no repack, and no pinned CUDA host buffers:
  - max_steps=4: `740.70ms`, `345.6 tok/s`, service runtime `24.007s`.
  - max_steps=6: `847.69ms`, `302.0 tok/s`, service runtime `21.507s`.
  - max_steps=8: `956.45ms`, `267.7 tok/s`, service runtime `21.674s`.
  - max_steps=12: `1174.54ms`, `218.0 tok/s`, service runtime `21.832s`.
  Even forcing only 4 diffusion steps did not approach 700 tok/s under this
  constrained host-memory configuration. Next: compare 2 GB + SSD swap and
  less-constrained memory at `max_steps=4` to separate swap overhead from
  runner/GPU throughput.
- 2026-06-12: Compared `max_steps=4` under 2 GB RAM + 8 GB SSD swap vs 4 GB
  physical RAM:
  - `MemoryMax=2G MemorySwapMax=8G` (`max_steps=4`): success, `2449.10ms`,
    `104.5 tok/s`, service runtime `32.814s`.
  - `MemoryMax=4G MemorySwapMax=0` (`max_steps=4`): success, `720.66ms`,
    `355.2 tok/s`, service runtime `22.231s`.
  The SSD fallback makes the 2 GB run possible but materially reduces
  throughput. Next: test a longer `-n 2048` run, closer to the model card's
  example command, to verify whether the 256-token smoke underestimates
  steady-state output rate.
- 2026-06-12: Tested Q4_K_M `-n 2048` with 4 GB physical RAM and
  `max_steps=4`. Context was sized as `8 blocks`:
  `n_ubatch=2176 n_batch=2176 n_ctx=2176`, but the actual reported generation
  still completed only `256 tok` over `1 blocks`: `total time=702.98ms`,
  `364.2 tok/s`. The output quality was visibly degraded/repetitive at
  `max_steps=4` / only
  4 diffusion steps. This means the 2048-token command did not provide a
  higher steady-state throughput measurement for this prompt/run; it stopped
  after the first canvas. Next: compare memory-saving flags against
  performance-priority flags to see whether `--no-repack` or
  `GGML_CUDA_NO_PINNED=1` are reducing single-step throughput.
- 2026-06-12: Tested a 4 GB performance-priority comparison without
  `--no-repack` and without `GGML_CUDA_NO_PINNED=1`. Result:
  `765.61ms`, `334.4 tok/s`, service runtime `10.086s`, all at
  `max_steps=4`, slightly slower than the memory-saving 4 GB run
  (`355.2 tok/s`, `max_steps=4`). Therefore those two
  memory-saving flags are not the main reason throughput is below 700 tok/s.
  Next: measure a warm second turn in one process to determine whether the
  low-step results are dominated by cold-start/kernel initialization overhead.
- 2026-06-12: Ran a same-process warm-up test in conversation mode with
  `MemoryMax=4G MemorySwapMax=0`, Q4_K_M, `max_steps=4`, reduced context, no
  host buffers, no repack, and no pinned CUDA host buffers. First turn:
  `715.77ms`, `357.7 tok/s`. Second turn in the same process:
  `247.55ms`, `1034.1 tok/s`. Both are `max_steps=4`. This exceeds the
  ~700 tok/s target and shows the
  low cold-run 4-step throughput was dominated by warm-up/initialization
  overhead. The generated text at 4 steps is visibly degraded/repetitive, so it
  is a speed-oriented configuration, not a quality-preserving one. Next:
  repeat the same warm second-turn test under `MemoryMax=2G` with SSD swap
  fallback.
- 2026-06-12: Repeated the same-process warm-up test under
  `MemoryMax=2G MemorySwapMax=8G` with SSD swap fallback. First turn:
  `1786.81ms`, `143.3 tok/s`. Second warm turn:
  `409.96ms`, `624.5 tok/s`, service runtime `30.336s`. Both are
  `max_steps=4`. This is close to
  ~700 tok/s under the 2 GB RAM cap, but does not reach it. The output remains
  visibly degraded/repetitive at 4 steps. Next: try a more aggressive
  `max_steps=3` speed-only configuration under the same 2 GB + SSD fallback to
  see whether it crosses 700 tok/s, while recording the quality caveat.
- 2026-06-12: Tried the more aggressive `max_steps=3` speed-only configuration
  under `MemoryMax=2G MemorySwapMax=8G`. First turn: `1808.33ms`,
  `141.6 tok/s`. Second warm turn: `380.37ms`, `673.0 tok/s`, service runtime
  `29.499s`. Both are `max_steps=3`. This is closer to 700 but still below
  it, and output quality is
  worse than the 4-step run. Next: test `max_steps=2` as a numeric boundary
  only, not as a quality-preserving configuration.
- 2026-06-12: User updated the allowed experimental scope: RAM gradient tests
  are now permitted up to 16 GB. New plan:
  - disable the temporary SSD swap to avoid swap interference;
  - run Q4_K_M same-process warm-up tests under pure physical `MemoryMax`
    limits from 3 GB to 16 GB;
  - keep the speed-oriented `max_steps=4` configuration as the primary
  comparison because it already exceeded 700 tok/s at 4 GB without swap, but
  record the quality caveat;
  - report metrics for the best pure-RAM result and clearly distinguish it from
    the earlier strict 2 GB + SSD fallback result.
- 2026-06-12: Disabled temporary SSD swap and ran pure physical RAM gradient
  tests with Q4_K_M, same-process warm-up, `max_steps=4`,
  `LLAMA_DIFFUSION_PROMPT_HEADROOM=128`, `GGML_CUDA_NO_PINNED=1`,
  `--direct-io --no-host --no-repack`, `-b 384 -ub 384 -c 384`, `-n 256`:
  - 3 GB (`max_steps=4`): first turn `273.6 tok/s`, warm second turn
    `729.5 tok/s`, service runtime `26.174s`.
  - 4 GB (`max_steps=4`): first turn `363.4 tok/s`, warm second turn
    `1060.2 tok/s`, service runtime `23.467s`.
  - 6 GB (`max_steps=4`): first turn `355.0 tok/s`, warm second turn
    `1054.4 tok/s`, service runtime `21.738s`.
  - 8 GB (`max_steps=4`): first turn `344.2 tok/s`, warm second turn
    `1063.0 tok/s`, service runtime `21.981s`.
  - 12 GB (`max_steps=4`): first turn `352.8 tok/s`, warm second turn
    `1049.5 tok/s`, service runtime `24.248s`.
  - 16 GB (`max_steps=4`): first turn `267.3 tok/s`, warm second turn
    `645.9 tok/s`, service runtime `27.621s`.
  Pure RAM removes the large SSD/swap penalty. The useful threshold for
  reaching the 700 tok/s target is around 3 GB after warm-up; 4-12 GB reached
  about 1.05k tok/s. The 16 GB result is anomalously lower and should be
  repeated before interpreting it as a trend.
- 2026-06-12: Repeated 4 GB and 16 GB to check stability:
  - repeat 4 GB (`max_steps=4`): first turn `351.5 tok/s`, warm second turn
    `1052.4 tok/s`, `total_ms=243.25` for the measured warm turn, service
    runtime `22.425s`.
  - repeat 16 GB (`max_steps=4`): first turn `260.7 tok/s`, warm second turn
    `763.9 tok/s`, `total_ms=335.14` for the measured warm turn, service
    runtime `27.325s`.
  4 GB is the best reproducible pure-RAM point in this sweep. 16 GB is still
  above 700 on repeat, but slower than 4-12 GB and not the recommended setting.

## Current Best Result

Best speed-oriented pure-RAM reproduction:

- GPU: RTX 5090, 32 GB VRAM.
- Host RAM limit: `MemoryMax=4G`, `MemorySwapMax=0`.
- Model: `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`.
- Runner: external DiffusionGemma llama.cpp PR checkout,
  `/root/lfz/llama.cpp-diffusiongemma/build-cuda/bin/llama-diffusion-cli`.
- Important flags:
  `LLAMA_DIFFUSION_PROMPT_HEADROOM=128`,
  `GGML_CUDA_NO_PINNED=1`,
  `-ngl all -sm none --no-host --direct-io -nr -cnv`,
  `-b 384 -ub 384 -c 384`,
  `--diffusion-eb on --diffusion-eb-max-steps 4`,
  `-n 256 --perf`.
- Measurement method: same-process two-turn conversation; first turn warms the
  runner/GPU kernels, second turn is the measured target prompt.
- Best repeated measured warm turn (`max_steps=4`):
  `total_ms=243.25`, `eval tok/s=1052.4`, `4` diffusion steps over one
  256-token canvas.
- Cold first-turn result in the same repeated run (`max_steps=4`):
  `total_ms=728.23`, `eval tok/s=351.5`.
- `time_to_type_s`: about `20.95s`, inferred from the runner timestamp when
  conversation mode becomes ready.
- `TTFT`: about `0.243s` for the measured warm target turn. This runner prints
  a completed denoised canvas rather than streaming token-by-token, so TTFT is
  effectively the turn generation time.
- `first_visible_s`: about `21.93-22.01s` from process start for the measured
  target turn, inferred from stderr step timestamps plus the warm-turn total.
- `prompt_eval tok/s`: N/A; the diffusion runner does not report a separate
  llama prompt-eval timing.
- `direct_reads`, `VRAM hit`, `RAM hit`, `read_failures`: N/A for this runner
  because it does not use ik_llama's GLM expert-pack / VRAM-cache path.
- Accuracy: N/A; this was a throughput smoke prompt, not an accuracy dataset
  evaluation.
- Quality caveat: `max_steps=4` is speed-oriented and the output is visibly
  repetitive/degraded. The default adaptive 27-step run produced better text
  but only about `135-147 tok/s` without swap and `55 tok/s` with 2 GB + SSD
  swap.
- Logs:
  `runs/ramgrad-repeat-q4-4g-noswap-warm2-maxsteps4-head128-b384-n256/`.

Strict 2 GB RAM status:

- Without swap: Q4_K_M still OOMs before generation even after reducing
  context/batch/headroom and disabling repack/pinned host buffers.
- With SSD swap fallback: the run succeeds, but measured warm-turn throughput
  is below the 4 GB pure-RAM result unless steps are reduced aggressively:
  - `max_steps=4`: `624.5 tok/s`.
  - `max_steps=3`: `673.0 tok/s`, with worse output quality.
  - `max_steps=2`: `817.0 tok/s`, reaches the numeric target but output is
    poor/incomplete.
- 2026-06-12 continuation plan for the original 2 GB target:
  - re-enable the temporary SSD swap file only for the strict
    `MemoryMax=2G` fallback path;
  - run same-process warm-up with `max_steps=2` as a numeric boundary test;
  - if it crosses 700 tok/s, report it as a speed-only 2 GB RAM + SSD fallback
    reproduction with explicit quality caveat;
  - keep the pure-RAM 4 GB result separate because it is faster and cleaner,
    but it does not satisfy the original 2 GB RAM constraint.
- 2026-06-12: Ran the original 2 GB RAM target with SSD fallback and
  `max_steps=2` as a numeric boundary test. Log path:
  `runs/warm2-q4-2g-swap8g-maxsteps2-head128-b384-n256/`. Configuration:
  Q4_K_M, `MemoryMax=2G`, `MemorySwapMax=8G`,
  `LLAMA_DIFFUSION_PROMPT_HEADROOM=128`, `GGML_CUDA_NO_PINNED=1`,
  `--direct-io --no-host --no-repack`, `-b 384 -ub 384 -c 384`, `-n 256`,
  conversation warm-up. Result: success. First turn `1626.51ms`,
  `157.4 tok/s`; measured warm second turn `313.34ms`, `817.0 tok/s`;
  service runtime `31.371s`. Both are `max_steps=2`. This reaches the
  requested ~700 tok/s class under
  a 2 GB RAM cap only by allowing SSD swap fallback and using an extremely
  aggressive 2-step speed setting. Output quality is poor and incomplete, so it
  should be reported as a speed-only reproduction, not a quality-preserving
  inference configuration. Temporary swap was disabled after the run.
- 2026-06-12: Updated this plan to explicitly annotate summarized throughput
  results with their corresponding `max_steps` values.

## Max Steps Summary

| Result group | RAM / fallback | max_steps | actual steps | Throughput |
| --- | --- | ---: | ---: | ---: |
| Default adaptive Q4 smoke | 4 GB RAM | 48 | 27 | 135.2 tok/s |
| Reduced context Q4 check | 4 GB RAM | 48 | 27 | 145.0 tok/s |
| Reduced context Q4 check | 3 GB RAM | 48 | 27 | 146.6 tok/s |
| Strict 2 GB + SSD fallback | 2 GB RAM + 8 GB swap | 48 | 27 | 55.0 tok/s |
| EB sweep | 3 GB RAM | 4 | 4 | 345.6 tok/s |
| EB sweep | 3 GB RAM | 6 | 6 | 302.0 tok/s |
| EB sweep | 3 GB RAM | 8 | 8 | 267.7 tok/s |
| EB sweep | 3 GB RAM | 12 | 12 | 218.0 tok/s |
| max_steps=4 comparison | 2 GB RAM + 8 GB swap | 4 | 4 | 104.5 tok/s |
| max_steps=4 comparison | 4 GB RAM | 4 | 4 | 355.2 tok/s |
| Same-process warm-up | 4 GB RAM | 4 | 4 | 1034.1 tok/s |
| Same-process warm-up | 2 GB RAM + 8 GB swap | 4 | 4 | 624.5 tok/s |
| Same-process warm-up | 2 GB RAM + 8 GB swap | 3 | 3 | 673.0 tok/s |
| Same-process warm-up | 2 GB RAM + 8 GB swap | 2 | 2 | 817.0 tok/s |
| RAM gradient | 3 GB RAM | 4 | 4 | 729.5 tok/s |
| RAM gradient | 4 GB RAM | 4 | 4 | 1060.2 tok/s |
| RAM gradient | 6 GB RAM | 4 | 4 | 1054.4 tok/s |
| RAM gradient | 8 GB RAM | 4 | 4 | 1063.0 tok/s |
| RAM gradient | 12 GB RAM | 4 | 4 | 1049.5 tok/s |
| RAM gradient | 16 GB RAM | 4 | 4 | 645.9 tok/s |
| RAM gradient repeat | 4 GB RAM | 4 | 4 | 1052.4 tok/s |
| RAM gradient repeat | 16 GB RAM | 4 | 4 | 763.9 tok/s |
| Quality-priority tuned EB | 4 GB RAM | 20 | 11 | 393.6 tok/s |
| Quality-priority tuned EB | 2 GB RAM + 8 GB swap | 20 | 11 | 229.8 tok/s |

## Open Questions

- Which exact GGUF file did Unsloth use for the ~700 token/s number? The model
  card recommends Q8_0 but advertises Q4_K_M for smaller single-GPU use.
- The ~700 token/s number likely refers to diffusion/block decoding, not
  standard autoregressive token-by-token decoding.
- Does this `ik_llama` branch contain the necessary runtime support for
  diffusiongemma/Gemma 3n GGUF metadata?
- Can the model fit under 2 GB host RAM with direct mmap/offload on this
  machine, or does it require full VRAM placement?

## External Results Research

Plan for official / public reproduction comparison:

- Search official model cards and docs for reported throughput, hardware,
  quantization, and exact command-line flags.
- Search llama.cpp PR / issue discussion for DiffusionGemma performance and
  recommended runtime flags.
- Search public reproduction reports for 5090, H100, RTX 4090/5090, and
  consumer GPU results.
- Compare public configs against our experiments:
  - model quantization: Q8_0 vs Q4_K_M;
  - `max_steps` / actual steps;
  - warm vs cold measurement;
  - RAM and swap usage;
  - `--diffusion-kv-cache`, `--diffusion-gpu-sampling`, and
    `--diffusion-gpu-sample-reduce`;
  - `-n`, `-b`, `-ub`, `-c`, `--direct-io`, `--no-host`, and `--no-repack`.
- Record whether any actionable optimization remains for this machine.

Progress:

- 2026-06-12: Started external official/public results research after reading
  `.Agent/Agent.md`.
- 2026-06-12: Initial external findings:
  - Google reports 700+ tok/s on RTX 5090 and 1000+ tok/s on H100, but describes
    this as a speed-oriented DiffusionGemma result rather than a llama.cpp GGUF
    command.
  - Official Google/NVIDIA material emphasizes 256-token parallel denoising,
    quantized deployment within about 18 GB VRAM, and lower output quality than
    standard Gemma 4.
  - Unsloth/HF GGUF card confirms 256-token canvas, 8 active / 128 total
    experts, and EB sampler use.
  - Public 5090 llama.cpp PR #24423 reproduction reports Flash Attention
    auto-disabled on SM120, Q4_K_M max stable `-n 8192`, and recommends
    `--diffusion-eb-max-steps 20 --diffusion-eb-t-max 0.3
    --diffusion-eb-t-min 0.05` for Q4_K_M.
  - That public Q4_K_M tuning reports about `545 tok/s` for short 256-token
    single-block generation at about 6 steps/block, and `244-252 tok/s` for
    multi-block long generation. It explicitly notes that colder/single-block
    settings can inflate speed while degrading quality.
- 2026-06-12: Sources used:
  - Google launch post:
    `https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/`
  - NVIDIA technical blog:
    `https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/`
  - Unsloth GGUF model card / guide:
    `https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF`
  - llama.cpp DiffusionGemma PR and public reproduction comments:
    `https://github.com/ggml-org/llama.cpp/pull/24423`
- 2026-06-12: Next local comparison: run the public Q4_K_M tuned EB parameters
  on our 5090 with `MemoryMax=4G`, no swap, same-process warm-up, to compare
  against our aggressive `max_steps=4` / `max_steps=2` speed-only settings.
- 2026-06-12: Ran the public Q4_K_M tuned EB parameters locally:
  `--diffusion-eb-max-steps 20 --diffusion-eb-t-min 0.05
  --diffusion-eb-t-max 0.3`, with Q4_K_M, `MemoryMax=4G`, no swap, same-process
  warm-up. Log path:
  `runs/public-eb-q4-4g-noswap-warm2-t005-03-max20-head128-b384-n256/`.
  Result: first turn `1460.52ms`, `16` actual steps, `175.3 tok/s`; warm
  second turn `639.20ms`, `11` actual steps, `400.5 tok/s`. Output quality was
  noticeably better than the extreme `max_steps=2/4` speed-only settings, but
  throughput was far below 700. This supports the conclusion that our
  `817 tok/s` 2 GB result is a numeric speed reproduction only, while
  quality-preserving GGUF/llama.cpp settings remain closer to `400 tok/s` on
  this short prompt and around `135-147 tok/s` with default hotter EB.
- 2026-06-12: User requested trying the quality-priority path explicitly:
  `--diffusion-eb-max-steps 20 --diffusion-eb-t-min 0.05
  --diffusion-eb-t-max 0.3`, accepting about `400 tok/s`. Plan:
  - rerun / verify this path under 4 GB pure RAM with same-process warm-up;
  - run the same quality-priority path under strict 2 GB RAM + SSD fallback;
  - compare output quality, actual steps, throughput, and startup/runtime
    metrics against the speed-only `max_steps=2/4` results.
- 2026-06-12: Quality-priority path results:
  - 4 GB pure RAM, no swap:
    `runs/quality-eb-q4-4g-noswap-warm2-t005-03-max20-head128-b384-n256-repeat/`.
    First turn: `1460.52ms`, `16` actual steps, `177.0 tok/s`.
    Warm target turn: `650.41ms`, `11` actual steps, `393.6 tok/s`.
    Service runtime: `23.214s`. Output is much more coherent than
    `max_steps=2/4`, though still has minor repetition/artifacts.
  - 2 GB RAM + 8 GB SSD swap fallback:
    `runs/quality-eb-q4-2g-swap8g-warm2-t005-03-max20-head128-b384-n256/`.
    First turn: `3972.04ms`, `16` actual steps, `64.5 tok/s`.
    Warm target turn: `1114.04ms`, `11` actual steps, `229.8 tok/s`.
    Service runtime: `35.281s`. Output quality is comparable to the 4 GB
    quality-priority run, but SSD fallback roughly halves warm throughput.
  Recommendation: for quality-priority GGUF/llama.cpp use, prefer 4 GB pure RAM
  with `max_steps=20, t_min=0.05, t_max=0.3`. The strict 2 GB + SSD fallback is
  viable for correctness but not a good performance path.
