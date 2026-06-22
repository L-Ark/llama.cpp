# 5090 Fast SSD First Three Reproduction

## Goal

Reproduce the first three historical GLM-5.1 IQ3_XXS 5090 configurations on
this machine, which has a faster SSD than the original experiment host, using
the same configuration intent and reporting updated metrics.

## Constraints

- Work only under `/home/wici/lfz`.
- Read `/home/wici/lfz/Agent.md` and this repo's `.Agent/Agent.md` before
  acting.
- Use `MemoryMax=2G` and `MemorySwapMax=0` for target runs.
- Use `RAM=0` / no RAM tier for the strict 2 GB line.
- Keep logs and generated outputs in this task directory unless a command's
  harness writes a standard repo-local cache/log.
- Report:
  - `direct_reads`
  - `VRAM hit`
  - `RAM hit`
  - `eval tok/s`
  - `prompt_eval tok/s` when available
  - `total_ms`
  - `TTFT`
  - `time_to_type_s`
  - `first_visible_s`
  - `accuracy` if applicable
  - `read_failures`
  - config details and log paths

## Historical Configurations To Reproduce

1. `pre-opt baseline`
   - profile: historical baseline/top4 profile equivalent
   - `GGML_MOE_VRAM_CACHE_MIB=1536`
   - `GGML_MOE_RAM_TIER_MIB=0`
   - historical: `direct_reads=118824`, `VRAM hit=6.0%`,
     `RAM hit=0.0%`, `eval tok/s=1.02`, `total_ms=105695.33`,
     `TTFT=17.84`

2. `optimized candidate1`
   - profile: top8 runtime profile
   - `GGML_MOE_VRAM_CACHE_MIB=8192`
   - `GGML_MOE_RAM_TIER_MIB=0`
   - historical: `direct_reads=91145`, `VRAM hit=29.1%`,
     `RAM hit=0.0%`, `eval tok/s=1.25`, `total_ms=88138.02`,
     `TTFT=21.68`

3. `optimized candidate2`
   - profile: top8 runtime profile
   - `GGML_MOE_VRAM_CACHE_MIB=12288`
   - `GGML_MOE_RAM_TIER_MIB=0`
   - historical: `direct_reads=87729`, `VRAM hit=31.8%`,
     `RAM hit=0.0%`, `eval tok/s=1.01`, `total_ms=109228.67`,
     `TTFT=21.19`

## Plan

1. Check repo, build, model, expert pack, GPU, storage, and no leftover runs.
2. Identify the exact repo-local profiles available for top4/top8 equivalents.
3. Build `llama-cli` if needed.
4. Run each target configuration under `systemd-run` with `MemoryMax=2G` and
   `MemorySwapMax=0`.
5. Store stdout/stderr/status/env in `.Agent/plans/5090-fast-ssd-first-three-repro/runs/`.
6. Parse logs for the required metrics.
7. Compare new fast-SSD results against the historical slow-SSD results.

## Progress

- 2026-06-12: Read `/home/wici/lfz/Agent.md`.
- 2026-06-12: Read `.Agent/Agent.md`.
- 2026-06-12: Created task directory
  `.Agent/plans/5090-fast-ssd-first-three-repro/`.
- 2026-06-12: Initial environment check:
  - repo branch is `feat/glm51-5090-2gb-vram-standalone`;
  - GPU is `NVIDIA GeForce RTX 5090`, `32607 MiB` total VRAM,
    `41 MiB` used, `32071 MiB` free, `0%` utilization;
  - repo filesystem is `/dev/nvme0n1p2` on `ext4`;
  - no leftover `llama-cli` / `moe-run.py` target process was found;
  - `build-cuda/bin/llama-cli` is not present in this checkout;
  - `models/GLM-5.1-UD-IQ3_XXS` is not present in this checkout.
- 2026-06-12: Corrected task placement after user clarification:
  task directory is under repo-local `.Agent/plans/`, not top-level
  `/home/wici/lfz/.Agent/`. The mistaken top-level `.Agent` directory was
  removed after confirming it contained no experiment logs.
- 2026-06-12: Inspected `scripts/create-moe-expert-pack.py`; it can create a
  repo-local `glm51-iq3xxs.expert-pack` from GLM GGUF shards if the pack is
  missing.
- 2026-06-12: Configured CUDA build in `build-cuda` with
  `GGML_NATIVE=ON`, `GGML_CUDA=ON`,
  `CMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc`, and
  `CMAKE_CUDA_ARCHITECTURES=120`.
- 2026-06-12: Built `build-cuda/bin/llama-cli` successfully.
- 2026-06-12: Verified `build-cuda/bin/llama-cli --help` runs.
- 2026-06-12: No GLM-5.1 GGUF or expert pack exists under
  `/home/wici/lfz`. Found GLM-5.1 UD-IQ3_XXS GGUF shards outside the work
  directory at `/home/wici/models/glm-5.1/UD-IQ3_XXS/`; these are the only
  outside-workdir files needed and are GLM GGUF files.
- 2026-06-12: Disk free on `/home/wici/lfz/ik_llama` filesystem:
  about `513G`.
- 2026-06-12: Created repo-local model directory
  `models/GLM-5.1-UD-IQ3_XXS/` with symlinks to the seven external GLM GGUF
  shards under `/home/wici/models/glm-5.1/UD-IQ3_XXS/`. No outside-workdir
  files were modified.
- 2026-06-12: Generated repo-local expert pack
  `models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack`.
  Inspect result:
  - version `1`;
  - entries `58368`;
  - payload `250752270336` bytes / `233.53 GiB`;
  - unique expert tensors `228`;
  - file size shown by `ls -lh`: `234G`.
- 2026-06-12: Current storage after pack generation:
  `/dev/nvme0n1p2`, `ext4`, `ZHITAI TiPro9000 2TB`, `280G` free.
- 2026-06-12: GPU idle before target runs:
  `NVIDIA GeForce RTX 5090`, total `32607 MiB`, used `41 MiB`,
  free `32071 MiB`, utilization `0%`.
- 2026-06-12: Added local PTY harness
  `.Agent/plans/5090-fast-ssd-first-three-repro/run_local_interactive.py`
  to run `moe-run.py --chat` through `systemd-run --pty --wait --collect`
  and record ready time, TTFT, first visible token time, stdout, env, route
  trace, TTFT trace, and result JSON.
- 2026-06-12: `run_local_interactive.py --help` passed.
- 2026-06-12: `moe-run.py --chat --dry-run` with the repo-local preset
  produced the expected `llama-cli` command. The model path resolves through
  the repo symlink to the external GLM GGUF shard, which is allowed by the
  top-level workspace rule. The expert pack path will be overridden to the
  repo-local generated pack for actual runs.
- 2026-06-12: First local smoke attempt with system-level `systemd-run --pty`
  failed before model startup because polkit required interactive
  authentication and the API session has no controlling `/dev/tty`.
  `systemd-run --user` was verified to support `MemoryMax=2G` and
  `MemorySwapMax=0` without polkit, so the local harness was changed to use
  `systemd-run --user --pty --wait --collect`.
- 2026-06-12: Smoke run
  `runs/smoke-baseline-vram1536-n4-user-quiet/` succeeded under
  `systemd-run --user` and `MemoryMax=2G MemorySwapMax=0`; it loaded the
  repo-local expert pack with `read_failures=0`. The short `-n 4` smoke did
  not produce visible assistant text and the TTFT trace had `mark_submit`
  without `mark_first_token`, so it is only a harness/load smoke, not a valid
  latency measurement.
- 2026-06-12: Completed first full target run:
  `runs/optimized-candidate1-vram8192-top8-n84/`, with top8 runtime profile,
  `GGML_MOE_VRAM_CACHE_MIB=8192`, `GGML_MOE_RAM_TIER_MIB=0`,
  `MemoryMax=2G`, `MemorySwapMax=0`, `-n 84`, seed `42`.
  Initial parsed metrics:
  - `time_to_type_s=11.263`
  - `interactive_ttft_s=10.613`
  - `first_visible_s=21.876`
  - `prompt_eval tok/s=1.89`
  - `eval tok/s=1.06`
  - `total_ms=81767.08`
  - `direct_reads=110518`
  - `read_failures=0`
  - `VRAM hit=5.6%`
  - `RAM hit=0.0%`
  - stderr:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate1-vram8192-top8-n84/llama-cache/moe-chat/20260612-141654-29500.stderr.log`
  - route trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate1-vram8192-top8-n84/route.trace.csv`
  - TTFT trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate1-vram8192-top8-n84/ttft.trace.csv`
- 2026-06-12: Completed full baseline run:
  `runs/pre-opt-baseline-vram1536-route-n84/`, with route profile,
  `GGML_MOE_VRAM_CACHE_MIB=1536`, `GGML_MOE_RAM_TIER_MIB=0`,
  `MemoryMax=2G`, `MemorySwapMax=0`, `-n 84`, seed `42`.
  Initial parsed metrics:
  - `time_to_type_s=11.404`
  - `interactive_ttft_s=9.339`
  - `first_visible_s=20.744`
  - `prompt_eval tok/s=2.14`
  - `eval tok/s=1.12`
  - `total_ms=77129.30`
  - `direct_reads=101591`
  - `read_failures=0`
  - `VRAM hit=12.1%`
  - `RAM hit=0.0%`
  - stderr:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/pre-opt-baseline-vram1536-route-n84/llama-cache/moe-chat/20260612-141905-58607.stderr.log`
  - route trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/pre-opt-baseline-vram1536-route-n84/route.trace.csv`
  - TTFT trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/pre-opt-baseline-vram1536-route-n84/ttft.trace.csv`
- 2026-06-12: Discovered that the repo does not contain the historical top4
  profile file. The route-profile baseline above is useful as a diagnostic
  but is not the strict historical `top4 profile` reproduction. Created a
  task-local top4 runtime profile by taking the first 4 expert rows per tensor
  from `presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv`:
  `.Agent/plans/5090-fast-ssd-first-three-repro/wici-glm51-interactive-n84-expert-top4.runtime.csv`.
  It has `900` data rows, `225` unique tensors, and exactly `4` rows per
  tensor.
- 2026-06-12: Completed strict baseline run:
  `runs/pre-opt-baseline-vram1536-top4-n84/`, with reconstructed top4 runtime
  profile, `GGML_MOE_VRAM_CACHE_MIB=1536`, `GGML_MOE_RAM_TIER_MIB=0`,
  `MemoryMax=2G`, `MemorySwapMax=0`, `-n 84`, seed `42`.
- 2026-06-12: Completed optimized candidate2 run:
  `runs/optimized-candidate2-vram12288-top8-n84/`, with top8 runtime profile,
  `GGML_MOE_VRAM_CACHE_MIB=12288`, `GGML_MOE_RAM_TIER_MIB=0`,
  `MemoryMax=2G`, `MemorySwapMax=0`, `-n 84`, seed `42`.
- 2026-06-12: Final check after target runs: no leftover target model process;
  GPU idle at `41 MiB` used, `32071 MiB` free, `0%` utilization.

## Results

All target runs used:

- branch: `feat/glm51-5090-2gb-vram-standalone`
- binary: `build-cuda/bin/llama-cli`
- model shards: repo-local symlinks under `models/GLM-5.1-UD-IQ3_XXS/`
  pointing to GLM GGUF shards at `/home/wici/models/glm-5.1/UD-IQ3_XXS/`
- expert pack: repo-local
  `models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack`
- GPU: `NVIDIA GeForce RTX 5090`
- storage: `/dev/nvme0n1p2`, `ext4`, `ZHITAI TiPro9000 2TB`
- memory cap: `systemd-run --user`, `MemoryMax=2G`, `MemorySwapMax=0`
- generation shape: `-n 84`, seed `42`, `-b 2048`, `-t 8`, `-tb 24`,
  `N_GPU_LAYERS=79`
- RAM tier disabled: `GGML_MOE_RAM_TIER_MIB=0`
- direct I/O enabled; `read_failures=0` for all final runs

The historical top4 profile file was not present in this checkout. For the
strict baseline, a task-local top4 equivalent was reconstructed from the
repo-local top8 runtime profile by keeping the first 4 expert rows per tensor.
The earlier `pre-opt-baseline-vram1536-route-n84` run used the full route
profile and is diagnostic only.

| Experiment | Profile | VRAM MiB | Historical TTFT s | New TTFT s | Historical total ms | New total ms | Historical eval tok/s | New eval tok/s | New prompt eval tok/s | Historical direct_reads | New direct_reads | Historical VRAM hit | New VRAM hit | RAM hit | read_failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-opt baseline | reconstructed top4 | 1536 | 17.84 | 9.26 | 105695.33 | 83188.60 | 1.02 | 1.01 | 2.16 | 118824 | 115224 | 6.0% | 0.0% | 0.0% | 0 |
| optimized candidate1 | top8 runtime | 8192 | 21.68 | 10.61 | 88138.02 | 81767.08 | 1.25 | 1.06 | 1.89 | 91145 | 110518 | 29.1% | 5.6% | 0.0% | 0 |
| optimized candidate2 | top8 runtime | 12288 | 21.19 | 10.60 | 109228.67 | 81843.82 | 1.01 | 1.06 | 1.89 | 87729 | 110520 | 31.8% | 5.6% | 0.0% | 0 |

Additional interactive timings:

| Experiment | time_to_type_s | first_visible_s | visible_token_estimate | assistant_sha256 |
| --- | ---: | ---: | ---: | --- |
| pre-opt baseline | 11.096 | 20.354 | 68 | `347a8103d81157433b38628c3b5674e664d509ef4ce863261fb4040578c76831` |
| optimized candidate1 | 11.263 | 21.876 | 68 | `347a8103d81157433b38628c3b5674e664d509ef4ce863261fb4040578c76831` |
| optimized candidate2 | 11.413 | 22.017 | 68 | `347a8103d81157433b38628c3b5674e664d509ef4ce863261fb4040578c76831` |

Log paths:

- pre-opt baseline:
  - run dir:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/pre-opt-baseline-vram1536-top4-n84/`
  - stderr:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/pre-opt-baseline-vram1536-top4-n84/llama-cache/moe-chat/20260612-142411-117614.stderr.log`
  - route trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/pre-opt-baseline-vram1536-top4-n84/route.trace.csv`
  - TTFT trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/pre-opt-baseline-vram1536-top4-n84/ttft.trace.csv`
- optimized candidate1:
  - run dir:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate1-vram8192-top8-n84/`
  - stderr:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate1-vram8192-top8-n84/llama-cache/moe-chat/20260612-141654-29500.stderr.log`
  - route trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate1-vram8192-top8-n84/route.trace.csv`
  - TTFT trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate1-vram8192-top8-n84/ttft.trace.csv`
- optimized candidate2:
  - run dir:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate2-vram12288-top8-n84/`
  - stderr:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate2-vram12288-top8-n84/llama-cache/moe-chat/20260612-142100-87824.stderr.log`
  - route trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate2-vram12288-top8-n84/route.trace.csv`
  - TTFT trace:
    `.Agent/plans/5090-fast-ssd-first-three-repro/runs/optimized-candidate2-vram12288-top8-n84/ttft.trace.csv`

Notes:

- Fast SSD substantially reduced TTFT for all three reproduced configs.
- The optimized top8 runs did not reproduce the historical high VRAM hit rates:
  current top8 runs showed `5.6%` VRAM hit instead of historical `29.1%` /
  `31.8%`. The run logs show successful profile preload counts, no pack read
  failures, and no RAM tier usage. This points to profile/cache behavior
  mismatch rather than storage read failure.
- All three final runs produced the same assistant output hash, so the
  generation result was stable across the reproduced configurations.
