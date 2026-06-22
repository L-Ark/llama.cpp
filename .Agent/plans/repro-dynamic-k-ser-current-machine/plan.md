# Reproduce Dynamic-k And SER Sweeps On Current Machine

## Goal

Reproduce the experiments recorded in:

- `.Agent/plans/glm-dynamic-k-cache-aware-routing/`
- `.Agent/plans/glm-ser-midrange-sweep/`

on this current `/home/wici/lfz` machine, using the same configuration intent
under the strict 5090 2 GB host-RAM line, and record the new results.

## Constraints

- Work only under `/home/wici/lfz`.
- Do not delete or modify anything outside `/home/wici/lfz`.
- Outside the workspace, use only GLM GGUF model files if needed.
- Target model runs must use `MemoryMax=2G` and `MemorySwapMax=0`.
- RAM tier remains disabled: `GGML_MOE_RAM_TIER_MIB=0`.
- Keep all new run outputs, logs, cache dirs, traces, and summaries under this
  task directory.
- Historical task folders are read-only references for this reproduction; do
  not edit their recorded results.

## Historical Experiments To Reproduce

### Dynamic-k / Cache-aware Routing Plan

Key historical experiments:

1. E1 free-form interactive SER screen:
   - baseline SER off
   - `SER=4,0.05`
   - `SER=2,0.2`
   - `SER=1,0.8`
   - `SER=1,0.95`
   - `SER=1,1.0`
   - `SER=1,1.1`
2. E2 single-item scored smoke:
   - baseline SER off
   - `SER=1,0.95`
3. E4 4-item scored smoke:
   - baseline SER off
   - `SER=1,0.95`
4. E4 interactive TTFT follow-up:
   - `SER=1,0.95`
5. E5 trace/simulation check:
   - baseline single-item trace
   - cache simulator and route predictability analysis

### SER Midrange Sweep Plan

Key historical experiments:

1. Phase 1 single-item scored screen:
   - `SER=1,0.96`
   - `SER=1,0.97`
   - `SER=1,0.975`
   - `SER=1,0.98`
   - `SER=1,0.99`
2. Phase 2 4-item scored smoke:
   - `SER=1,0.96`
   - `SER=1,0.975`

## Plan

1. Check branch, build, model links, expert pack, datasets, GPU idle state, and
   no leftover model runs.
2. Create task-local runners adapted to `/home/wici/lfz` and task-local
   output/cache paths.
3. Run a lightweight dry-run/syntax check for the runners.
4. Reproduce scored direct-completion runs first because they give accuracy.
5. Reproduce interactive SER/TTFT runs next.
6. Reproduce the trace/simulator run.
7. Parse and compare new results against the historical results from the two
   source plans.
8. Record exact commands, configs, logs, failures, and final conclusions here.

## Progress

- 2026-06-12: Read `/home/wici/lfz/Agent.md`.
- 2026-06-12: Read `.Agent/Agent.md`.
- 2026-06-12: Created this task directory and plan.
- 2026-06-12: Checked local dependencies:
  - `build-cuda/bin/llama-cli` exists and reports version
    `4971 (5aa52594)`;
  - repo-local GLM expert pack exists at
    `models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack`;
  - top8 runtime profile exists at
    `presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv`;
  - GPU idle: `NVIDIA GeForce RTX 5090`, `41 MiB` used,
    `32071 MiB` free, `0%` utilization;
  - no leftover model process was found.
- 2026-06-12: Historical tasks use `/root/lfz/data/...` datasets. To keep all
  work within `/home/wici/lfz`, reconstructed task-local `smoke_one.jsonl` and
  `smoke.jsonl` from the committed historical result stdout/prompt echoes.
- 2026-06-12: Added task-local runners:
  - `run_smoke_accuracy_local.py` for direct-completion scored runs;
  - `run_interactive_local.py` for interactive TTFT/free-form runs.
  Both use `/home/wici/lfz/ik_llama`, `systemd-run --user`,
  `MemoryMax=2G`, `MemorySwapMax=0`, task-local `LLAMA_CACHE`, and task-local
  output directories.
- 2026-06-12: Syntax checks passed for both task-local runners.
- 2026-06-12: Completed first smoke-one baseline runner validation:
  `runs/accuracy/smoke-one-baseline/`.
  Result: `accuracy=1/1`, output `B`, `direct_reads=6678`,
  `VRAM hit=10.1%`, `RAM hit=0.0%`, `prompt_eval=2.09 tok/s`,
  `eval=0.86 tok/s`, `total_ms=40339.95`, `wall_s=41.28`,
  `read_failures=0`.
- 2026-06-12: Completed smoke-one SER screen:
  - baseline: `1/1`, output `B`, `direct_reads=6678`,
    `VRAM hit=10.1%`, `prompt_eval=2.093 tok/s`, `eval=0.861 tok/s`,
    `total_ms=40339.95`, `wall_s=41.28`, `read_failures=0`;
  - `SER=1,0.95`: `1/1`, output `B`, `direct_reads=5180`,
    `VRAM hit=21.7%`, `prompt_eval=2.256 tok/s`, `eval=1.169 tok/s`,
    `total_ms=37489.38`, `wall_s=38.43`, `read_failures=0`;
  - `SER=1,0.96`: `1/1`, output `B`, `direct_reads=4390`,
    `VRAM hit=30.7%`, `prompt_eval=2.316 tok/s`, `eval=1.460 tok/s`,
    `total_ms=36494.28`, `wall_s=37.22`, `read_failures=0`;
  - `SER=1,0.97`: `1/1`, output `B`, `direct_reads=4060`,
    `VRAM hit=32.8%`, `prompt_eval=2.437 tok/s`, `eval=1.587 tok/s`,
    `total_ms=34819.93`, `wall_s=35.54`, `read_failures=0`;
  - `SER=1,0.975`: `0/1`, output `C`, `direct_reads=3505`,
    `VRAM hit=36.5%`, `prompt_eval=2.445 tok/s`, `eval=1.999 tok/s`,
    `total_ms=34599.30`, `wall_s=35.32`, `read_failures=0`;
  - `SER=1,0.98`: `0/1`, output `A`, `direct_reads=3729`,
    `VRAM hit=30.3%`, `prompt_eval=2.529 tok/s`, `eval=1.796 tok/s`,
    `total_ms=33849.22`, `wall_s=34.58`, `read_failures=0`;
  - `SER=1,0.99`: `0/1`, output `A`, `direct_reads=3315`,
    `VRAM hit=25.0%`, `prompt_eval=2.789 tok/s`, `eval=1.464 tok/s`,
    `total_ms=32206.13`, `wall_s=32.92`, `read_failures=0`.
- 2026-06-12: Completed 4-item smoke scored reproduction for the key
  dynamic-k and SER midrange configs:
  - baseline: `3/4`, outputs `D,D,B,D`, `direct_reads=25962`,
    `VRAM hit=13.5%`, `prompt_eval=3.525 tok/s`, `eval=0.881 tok/s`,
    `total_ms=179778.30`, `wall_s=183.38`, `read_failures=0`;
  - `SER=1,0.95`: `3/4`, outputs `D,D,B,D`, `direct_reads=19740`,
    `VRAM hit=25.4%`, `prompt_eval=3.741 tok/s`, `eval=1.226 tok/s`,
    `total_ms=167435.74`, `wall_s=170.57`, `read_failures=0`;
  - `SER=1,0.96`: `2/4`, outputs `C,D,B,D`, `direct_reads=17316`,
    `VRAM hit=31.6%`, `prompt_eval=3.835 tok/s`, `eval=1.465 tok/s`,
    `total_ms=162708.43`, `wall_s=166.13`, `read_failures=0`;
  - `SER=1,0.975`: `1/4`, outputs `A,D,C,D`, `direct_reads=14590`,
    `VRAM hit=35.8%`, `prompt_eval=3.988 tok/s`, `eval=1.784 tok/s`,
    `total_ms=158401.97`, `wall_s=161.93`, `read_failures=0`.
  Compared with historical `glm-ser-midrange-sweep`, this machine reproduces
  the speed trend but not the `SER=1,0.96` quality result: historical `1,0.96`
  was `3/4`, current machine is `2/4`. Current quality-preserving smoke
  candidate is `SER=1,0.95`.
- 2026-06-12: First attempt to start interactive baseline/`SER=1,0.95`
  batch failed before launching the model because the shell redirection target
  directory `runs/interactive/` did not exist. Created the directory and
  restarted the batch.
- 2026-06-12: Completed interactive free-form SER screen:
  - baseline: `time_to_type=10.54s`, `TTFT=10.82s`,
    `first_visible=21.36s`, `prompt_eval=1.85 tok/s`,
    `eval=1.04 tok/s`, `direct_reads=28302`, `VRAM hit=8.1%`,
    `read_failures=0`;
  - `SER=4,0.05`: `time_to_type=11.21s`, `TTFT=10.52s`,
    `first_visible=21.73s`, `prompt_eval=1.90 tok/s`,
    `eval=1.03 tok/s`, `direct_reads=28302`, `VRAM hit=8.1%`,
    `read_failures=0`;
  - `SER=2,0.2`: `time_to_type=11.37s`, `TTFT=10.67s`,
    `first_visible=22.04s`, `prompt_eval=1.87 tok/s`,
    `eval=1.04 tok/s`, `direct_reads=28302`, `VRAM hit=8.1%`,
    `read_failures=0`;
  - `SER=1,0.8`: `time_to_type=9.40s`, `TTFT=10.62s`,
    `first_visible=20.02s`, `prompt_eval=1.88 tok/s`,
    `eval=1.03 tok/s`, `direct_reads=28302`, `VRAM hit=8.1%`,
    `read_failures=0`;
  - `SER=1,0.95`: `time_to_type=11.01s`, `TTFT=10.20s`,
    `first_visible=21.21s`, `prompt_eval=2.13 tok/s`,
    `eval=1.23 tok/s`, `direct_reads=22917`, `VRAM hit=11.8%`,
    `read_failures=0`;
  - `SER=1,1.0`: `time_to_type=11.34s`, `TTFT=4.39s`,
    `first_visible=15.73s`, `prompt_eval=4.56 tok/s`,
    `eval=4.81 tok/s`, `direct_reads=4364`, `VRAM hit=34.9%`,
    `read_failures=0`, output visibly corrupted;
  - `SER=1,1.1`: `time_to_type=11.26s`, `TTFT=4.31s`,
    `first_visible=15.57s`, `prompt_eval=4.64 tok/s`,
    `eval=4.86 tok/s`, `direct_reads=4347`, `VRAM hit=29.3%`,
    `read_failures=0`, output visibly corrupted.
- 2026-06-12: Completed E5 trace/simulation reproduction:
  - trace run: `runs/trace/smoke-one-baseline-trace/`;
  - accuracy `1/1`, output `B`;
  - `direct_reads=6678`, `VRAM hit=10.1%`, `RAM hit=0.0%`,
    `prompt_eval=2.100 tok/s`, `eval=0.848 tok/s`,
    `total_ms=40286.70`, `wall_s=41.16`, `read_failures=0`;
  - route trace:
    `runs/trace/smoke-one-baseline-trace/smoke-one-baseline-trace.ceval_computer_network_0010.route.csv`;
  - trace length: `5424` events plus header;
  - cache simulator output:
    `runs/trace/smoke-one-baseline-trace.cache-sim.txt`;
  - predictability output:
    `runs/trace/smoke-one-baseline-trace.predictability.json`;
  - simulator replay at measured split (`upgate_pct=60`) estimated
    `hit_pct=10.07%`, matching measured `VRAM hit=10.1%`;
  - simulator's single-trace sweep recommended `upgate_pct=75`, estimated
    `hit_pct=25.61%`, `miss_gib=16.34`;
  - route predictability analyzer recommended
    `collect_more_traces_before_runtime_prefetch`.
- 2026-06-12: Wrote machine-readable summary:
  `summary.json`.

## Results

All target model runs used `systemd-run --user` with `MemoryMax=2G` and
`MemorySwapMax=0`, RAM tier disabled, repo-local GLM expert pack, top8 runtime
profile, direct I/O, seed `42`, and task-local output/cache directories.

### Smoke-One Screen

| config | accuracy | output | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s | read_failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1/1 | B | 6678 | 10.1% | 2.093 | 0.861 | 40339.95 | 41.28 | 0 |
| SER 1,0.95 | 1/1 | B | 5180 | 21.7% | 2.256 | 1.169 | 37489.38 | 38.43 | 0 |
| SER 1,0.96 | 1/1 | B | 4390 | 30.7% | 2.316 | 1.460 | 36494.28 | 37.22 | 0 |
| SER 1,0.97 | 1/1 | B | 4060 | 32.8% | 2.437 | 1.587 | 34819.93 | 35.54 | 0 |
| SER 1,0.975 | 0/1 | C | 3505 | 36.5% | 2.445 | 1.999 | 34599.30 | 35.32 | 0 |
| SER 1,0.98 | 0/1 | A | 3729 | 30.3% | 2.529 | 1.796 | 33849.22 | 34.58 | 0 |
| SER 1,0.99 | 0/1 | A | 3315 | 25.0% | 2.789 | 1.464 | 32206.13 | 32.92 | 0 |

### Smoke4 Scored Runs

| config | accuracy | outputs | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s | read_failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3/4 | D,D,B,D | 25962 | 13.5% | 3.525 | 0.881 | 179778.30 | 183.38 | 0 |
| SER 1,0.95 | 3/4 | D,D,B,D | 19740 | 25.4% | 3.741 | 1.226 | 167435.74 | 170.57 | 0 |
| SER 1,0.96 | 2/4 | C,D,B,D | 17316 | 31.6% | 3.835 | 1.465 | 162708.43 | 166.13 | 0 |
| SER 1,0.975 | 1/4 | A,D,C,D | 14590 | 35.8% | 3.988 | 1.784 | 158401.97 | 161.93 | 0 |

### Interactive Free-Form SER Screen

| config | time_to_type_s | TTFT s | first_visible_s | prompt tok/s | eval tok/s | direct_reads | VRAM hit | read_failures | quality note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 10.54 | 10.82 | 21.36 | 1.85 | 1.04 | 28302 | 8.1% | 0 | normal prefix |
| SER 4,0.05 | 11.21 | 10.52 | 21.73 | 1.90 | 1.03 | 28302 | 8.1% | 0 | no traffic effect |
| SER 2,0.2 | 11.37 | 10.67 | 22.04 | 1.87 | 1.04 | 28302 | 8.1% | 0 | no traffic effect |
| SER 1,0.8 | 9.40 | 10.62 | 20.02 | 1.88 | 1.03 | 28302 | 8.1% | 0 | no traffic effect |
| SER 1,0.95 | 11.01 | 10.20 | 21.21 | 2.13 | 1.23 | 22917 | 11.8% | 0 | speedup, output quality questionable |
| SER 1,1.0 | 11.34 | 4.39 | 15.73 | 4.56 | 4.81 | 4364 | 34.9% | 0 | corrupted |
| SER 1,1.1 | 11.26 | 4.31 | 15.57 | 4.64 | 4.86 | 4347 | 29.3% | 0 | corrupted |

### Trace And Simulation

Trace run:

- `accuracy=1/1`, output `B`
- `route_events=5424`
- measured `VRAM hit=10.1%`
- simulator replay at `upgate_pct=60`: `hit_pct=10.07%`
- simulator single-trace recommendation: `upgate_pct=75`, `hit_pct=25.61%`
- analyzer recommendation: `collect_more_traces_before_runtime_prefetch`

### Current-Machine Conclusion

- The fast SSD/current machine substantially improves wall time and prompt
  throughput versus the historical stored results.
- The dynamic-k trend reproduces: increasing SER threshold reduces expert
  traffic and improves speed, but quality drops sharply.
- The `glm-dynamic-k-cache-aware-routing` conclusion still holds: low
  thresholds (`4,0.05`, `2,0.2`, `1,0.8`) have no traffic effect; `1,0.95`
  gives a measured speedup; `>=1.0` is fast but corrupts output.
- The `glm-ser-midrange-sweep` conclusion changes on this machine: historical
  `SER=1,0.96` preserved `3/4` smoke accuracy, but current machine/run gives
  `2/4`. The current quality-preserving smoke candidate is `SER=1,0.95`, not
  `SER=1,0.96`.
- Cache simulator behavior reproduces: replayed trace hit rate matches measured
  hit rate closely, so simulator remains useful for offline cache-policy
  exploration. A runtime cache-aware routing patch is still not justified from
  one trace.
