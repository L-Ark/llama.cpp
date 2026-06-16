# DeepSeek-V4-Flash SSD 5 tok/s Optimization Progress

## 2026-06-15

### Task Created

- Task directory: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization`
- Plan file: `plan.md`
- Progress file: `progress.md`

### Current Baseline

Source:

- Previous task record: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd/proceed.md`

Baseline metrics:

| Metric | Value |
| --- | ---: |
| TTFT p50 | 15.98 s |
| TTFT hard cap | 19.98 s |
| Decode token rate p50 | 1.34 tok/s |
| Worst observed decode token rate | 1.29 tok/s |
| Prefill p50 | 8.01 tok/s |
| Stable CUDA/Disk placement | `{'cuda':16,'disk':84}` |

### Hard Gates

- Decode p50 target: `>=5.0 tok/s`
- Worst repeat decode target: `>=4.7 tok/s`
- TTFT p50: `<=19.98 s`, no exceptions
- Smoke pass rate: `>50%`
- Smoke pass rate drop from no-accuracy-loss baseline: `<=15 percentage points`
- CPU RAM expert tier budget: `2 GiB`
- Do not stop other users' GPU or RAM-heavy processes

### Next Action

Start Route A:

1. Add benchmark wrapper and smoke baseline.
2. Add disk-MoE telemetry.
3. Build and run baseline repeat.
4. Commit instrumentation only if it does not change model output semantics.

### Repository And Remote Rule Update

- User clarified that fastLLM must use a newly created private GitHub repository, not a fork.
- fastLLM local git identity is configured as:
  - name: `L-Ark`
  - email: `fliangae@connect.ust.hk`
- Added fastLLM remote:

```bash
git -C /home/wici/lfz/fastllm remote add private https://github.com/L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git
```

- Current remotes:
  - `origin`: `https://github.com/ztxz16/fastllm.git`
  - `private`: `https://github.com/L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git`
- No GitHub CLI or token is currently available in the environment, so the private GitHub repo still needs to be created/authenticated before the first push.
- vramctl must not receive task commits. Useful vramctl code should be copied or ported into fastLLM and committed/pushed with the fastLLM changes.

## 2026-06-15 16:09:32 CST

### Private Repo Initialization Attempt

- fastLLM private remote is configured as `private`: `https://github.com/L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git`.
- Current fastLLM HEAD: `1a3cbb94b74dcf20b9d81d02c9e460d03f741b85`.
- Current branch: `master`.
- Working tree note: only untracked `.venv-ftllm/` is present and must not be committed.
- Attempted non-interactive HTTPS ls-remote / push. Both timed out because no usable GitHub authentication is available in the environment.
- Checked SSH auth with `ssh -T git@github.com`; GitHub rejected the local key with `Permission denied (publickey)`.
- Initialization status: local remote is ready, but first push is blocked until GitHub authentication is provided.

Next valid initialization command after auth is available:

```bash
git -C /home/wici/lfz/fastllm push -u private master
```

## 2026-06-15 16:17:06 CST

### Private Repo Initialization Completed

- SSH authentication confirmed with GitHub account `L-Ark`.
- Updated fastLLM `private` remote to SSH:

```bash
git@github.com:L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git
```

- Pushed `/home/wici/lfz/fastllm` branch `master` to `private/master`.
- Remote `HEAD` and `refs/heads/master` now point to:

```text
1a3cbb94b74dcf20b9d81d02c9e460d03f741b85
```

- Local branch `master` now tracks `private/master`.
- Working tree still has only untracked `.venv-ftllm/`; it was not pushed or committed.

Next commit/push rule:

```bash
git -C /home/wici/lfz/fastllm push private master
```

## 2026-06-15 16:28:46 CST

### Route A Instrumentation Implementation

Implemented in `/home/wici/lfz/fastllm`:

- Added env-gated disk-MoE telemetry in `src/devices/disk/diskdevice.cpp`.
  - Enable with `FASTLLM_DISK_MOE_STATS=1`.
  - Reports merge runs, selected expert refs, disk load requests, loaded disk weights, read calls, read bytes, read GiB, read seconds, effective read GiB/s, load seconds, and placeholder zero RAM cache fields.
  - Default behavior remains unchanged when the env var is unset.
- Added output token SHA256 to `ftllm bench` so benchmark runs can record an output hash.
- Added `tools/scripts/deepseek_v4_flash_gate.py` to run repeated DeepSeek V4-Flash benchmarks, wait for GPU idleness, record git/resource state, parse TTFT/decode/prefill/total throughput, parse disk-MoE telemetry, and emit `summary.json`.
- Created fixed smoke suite: `smoke_eval.jsonl` with 20 cases, 4 each for factual, math, coding, JSON, and multilingual categories.

Validation so far:

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python -m py_compile \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  /home/wici/lfz/fastllm/tools/fastllm_pytools/benchmark.py
```

- Result: passed.

```bash
cmake -S /home/wici/lfz/fastllm -B /home/wici/lfz/fastllm/build-route-a-nonuma \
  -DUSE_CUDA=ON -DCUDA_ARCH=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DUSE_NUMAS=OFF
```

- Result: configured successfully with CUDA 13.2 and effective architecture `120f`.

```bash
CPATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/include \
LIBRARY_PATH=/home/wici/lfz/fastllm/build-route-a-nonuma/nccl-lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

- Result: passed after adding a build-local `libnccl.so -> libnccl.so.2` symlink.
- Initial `USE_NUMAS=ON` build failed because the system lacks `numa.h`; current DeepSeek SOTA path uses CUDA+disk, not NUMA.

Runtime load check:

```bash
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python -c "import ftllm, ftllm.llm as llm; print(llm.fastllm_lib._name)"
```

- Result: loaded build-local `libfastllm_tools.so`.

Telemetry smoke:

```bash
FASTLLM_DISK_MOE_STATS=1 PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python -c "import ftllm.llm; print('loaded')"
```

- Result: emitted `[FASTLLM_DISK_MOE_STATS]` with zero counters on import-only run, confirming the reporter path works.

Not yet run:

- Full 3-repeat DeepSeek V4-Flash benchmark with the new library. It requires loading the 149 GiB model and previously needed temporary swap to avoid host-RAM kill.
- Baseline smoke accuracy run. The smoke dataset is now fixed; a scorer/runner still needs to be added or run manually against a serving/chat path.

Decision:

- This is non-semantic Route A instrumentation. It does not claim token-rate improvement or SOTA.
- Safe to commit as `baseline-instrumentation` once staged diff is confirmed.

### Route A Instrumentation Commit

- Commit: `450779885b7bd0656b9f70278303b0c727c4b1db`
- Subject: `feat: add disk MoE benchmark telemetry`
- Pushed to: `private/master` (`git@github.com:L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git`)
- Remote verification:

```text
450779885b7bd0656b9f70278303b0c727c4b1db refs/heads/master
```

Committed files:

- `src/devices/disk/diskdevice.cpp`
- `tools/fastllm_pytools/benchmark.py`
- `tools/scripts/deepseek_v4_flash_gate.py`

This commit is instrumentation only. It is not recorded as a token-rate SOTA improvement.

Current fastLLM working tree after commit/push:

```text
## master...private/master
?? .venv-ftllm/
```

Next Route A gap:

- Add a smoke scorer/runner so the fixed `smoke_eval.jsonl` can produce baseline pass rate and optimized pass-rate comparisons.
- Then run at least a small validation pass, and plan the full DeepSeek baseline repeat with sufficient swap/RAM handling.

### Route A Smoke Runner Commit

- Commit: `9482f64fc89401fadcc399444e7800c8a7b8dfb4`
- Subject: `feat: add DeepSeek smoke gate runner`
- Pushed to: `private/master`.
- Remote verification:

```text
9482f64fc89401fadcc399444e7800c8a7b8dfb4 refs/heads/master
```

Committed file:

- `tools/scripts/deepseek_v4_flash_smoke.py`

Validation:

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python -m py_compile \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_smoke.py
```

- Result: passed.

Score-only scorer validation:

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_smoke.py \
  --smoke-file /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/smoke_eval.jsonl \
  --out-dir /tmp/deepseek_v4_smoke_score_test \
  --score-only /tmp/deepseek_v4_smoke_canned.jsonl
```

- Result: `20/20` pass, category pass rates all `4/4`.

Current fastLLM working tree:

```text
## master...private/master
?? .venv-ftllm/
```

Next step:

- Install or point runtime at the build-local `ftllm` package and run a small DeepSeek V4-Flash benchmark with `FASTLLM_DISK_MOE_STATS=1` to verify real disk-MoE telemetry.
- Because prior full-model runs were killed without extra swap, use temporary swap if available and remove it after the run.

## 2026-06-15 16:46:34 CST

### Route A Real Telemetry Smoke

Purpose:

- Verify that the committed disk-MoE telemetry works on a real DeepSeek-V4-Flash run before starting long baseline repeats.
- This is a short validation run, not a SOTA claim.

Git status before and after:

```text
## master...private/master
?? .venv-ftllm/
```

fastLLM commit:

```text
9482f64fc89401fadcc399444e7800c8a7b8dfb4 feat: add DeepSeek smoke gate runner
```

Resource handling:

- GPU was idle at run start: `41 MiB` used, `32071 MiB` free, `0%` GPU utilization.
- A temporary 32 GiB swap file was used for model loading and removed after the run:
  - `/home/wici/lfz/deepseek-v4-route-a.swap`
- No other user process was stopped.

Command:

```bash
RUN_DIR=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-telemetry-smoke-20260615-163915
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  --out-dir "$RUN_DIR" \
  --repeat 1 \
  --input-tokens 16 \
  --output-tokens 8 \
  --timeout-s 1200 \
  --max-wait-s 10 \
  --max-gpu-used-mib 1024 \
  --poll-s 2
```

Artifacts:

- Summary: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-telemetry-smoke-20260615-163915/summary.json`
- Log: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-telemetry-smoke-20260615-163915/bench_repeat_1.log`

Metrics:

| Metric | Value |
| --- | ---: |
| Input tokens | 16 |
| Target output tokens | 8 |
| Actual output tokens | 8 |
| TTFT | 7.40317 s |
| Prefill | 2.16 tok/s |
| Decode after TTFT | 1.26 tok/s |
| Batch total | 0.62 tok/s |
| Total time | 12.9529 s |
| Output token SHA256 | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` |
| Selected expert refs | 3154 |
| Disk load requests | 6308 |
| Loaded disk weights | 6308 |
| Disk read calls | 18924 |
| Disk read bytes | 42,166,910,976 |
| Disk read GiB | 39.270996 |
| Disk read seconds | 38.186563 |
| Effective expert read bandwidth | 1.028398 GiB/s |
| Disk load seconds | 38.272252 |
| RAM cache hit rate | not enabled |
| VRAM hit estimate | not available |

Decision against gates:

- Telemetry path: pass.
- Full benchmark gate: not evaluated, because this was a 1-repeat 16/8 smoke run.
- Smoke accuracy gate: not evaluated in this run.
- SOTA token-rate claim: no.
- Commit: no new commit; the tested code was already committed and pushed as `9482f64fc89401fadcc399444e7800c8a7b8dfb4`.

Next action:

- Run the plan-defined benchmark shape with 3 repeats and telemetry enabled to establish an instrumented no-accuracy-loss performance baseline.
- Run the fixed smoke suite once on the same no-accuracy-loss path to establish the baseline pass rate.

## 2026-06-15 17:02:25 CST

### Route A Instrumented Full Baseline

Purpose:

- Establish the no-accuracy-loss instrumented performance baseline using the plan-defined benchmark shape.
- This run uses the current stable placement `{'cuda':16,'disk':84}` and does not change routing, top-k, active expert count, or model math.

Git status before and after:

```text
## master...private/master
?? .venv-ftllm/
```

fastLLM commit:

```text
9482f64fc89401fadcc399444e7800c8a7b8dfb4 feat: add DeepSeek smoke gate runner
```

Resource handling:

- GPU was idle at run start: `41 MiB` used, `32071 MiB` free, `0%` GPU utilization.
- A temporary 32 GiB swap file was used during the run and removed after completion:
  - `/home/wici/lfz/deepseek-v4-route-a.swap`
- After cleanup, only the system default `/swap.img` remains.
- No other user process was stopped.

Command:

```bash
RUN_DIR=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-baseline-full-20260615-164803
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  --out-dir "$RUN_DIR" \
  --repeat 3 \
  --input-tokens 128 \
  --output-tokens 256 \
  --timeout-s 2400 \
  --max-wait-s 3600 \
  --max-gpu-used-mib 1024 \
  --poll-s 30
```

Artifacts:

- Summary: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-baseline-full-20260615-164803/summary.json`
- Repeat 1 log: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-baseline-full-20260615-164803/bench_repeat_1.log`
- Repeat 2 log: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-baseline-full-20260615-164803/bench_repeat_2.log`
- Repeat 3 log: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-baseline-full-20260615-164803/bench_repeat_3.log`

Benchmark metrics:

| Repeat | TTFT (s) | Decode tok/s | Prefill tok/s | Total tok/s | Expert read GiB | Expert read GiB/s | Read calls | Output hash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16.53037 | 1.21 | 7.74 | 1.13 | 749.498291 | 1.022457 | 361170 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |
| 2 | 16.00929 | 1.35 | 8.00 | 1.25 | 749.498291 | 1.246818 | 361170 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |
| 3 | 15.98457 | 1.29 | 8.01 | 1.20 | 749.498291 | 1.105631 | 361170 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |

Summary:

| Metric | Value |
| --- | ---: |
| Decode p50 | 1.29 tok/s |
| Decode worst | 1.21 tok/s |
| TTFT p50 | 16.00929 s |
| TTFT worst | 16.53037 s |
| Selected expert refs per repeat | 60195 |
| Load requests per repeat | 120390 |
| Read bytes per repeat | 804,767,662,080 |
| RAM cache hit rate | not enabled |
| VRAM hit estimate | not available |

Decision against hard gates:

- Decode p50 `1.29 tok/s`: fail target `>=5.0 tok/s`.
- Worst repeat `1.21 tok/s`: fail target `>=4.7 tok/s`.
- TTFT p50 `16.00929 s`: pass cap `<=19.98 s`.
- Smoke accuracy: not evaluated in this run; must be run before optimized routes can claim final pass.
- SOTA claim: no. This is below the previous record p50 `1.34 tok/s`, so no SOTA-improving commit is made.

Observation:

- The full baseline reads about `749.5 GiB` of expert data per 256-token run through `361170` read calls.
- Effective expert-path read bandwidth is only `1.02-1.25 GiB/s`, far below the storage fio ceiling reported earlier. This keeps Route B/C/F high priority: reduce fragmented reads, pack expert payloads, and then pipeline/batch reads.

Next action:

- Run the fixed no-accuracy-loss smoke suite to establish the baseline pass rate.
- Then start Route B/C implementation work, with the first practical target being fewer read calls and higher effective expert read bandwidth.

## 2026-06-15 17:24:21 CST

### Route A No-Accuracy-Loss Smoke Baseline

Purpose:

- Establish the no-accuracy-loss smoke pass-rate baseline required by the plan.
- This run uses the same stable placement and generation semantics as the benchmark baseline.

Git status before and after:

```text
## master...private/master
?? .venv-ftllm/
```

fastLLM commit:

```text
9482f64fc89401fadcc399444e7800c8a7b8dfb4 feat: add DeepSeek smoke gate runner
```

Resource handling:

- GPU was idle before starting the smoke run.
- A temporary 32 GiB swap file was used during model loading and removed after completion:
  - `/home/wici/lfz/deepseek-v4-smoke.swap`
- After cleanup, only the system default `/swap.img` remains.
- No other user process was stopped.

Command:

```bash
RUN_DIR=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-smoke-baseline-20260615-170602
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
FASTLLM_DISK_MOE_STATS=1 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_smoke.py \
  /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low \
  --device cuda \
  --moe_device "{'cuda':16,'disk':84}" \
  --moe_device_layers -1 \
  --cuda_shared_expert true \
  --gpu_mem_ratio 0.98 \
  --cuda_slab 256 \
  --threads 8 \
  --smoke-file /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/smoke_eval.jsonl \
  --out-dir "$RUN_DIR" \
  --max-new-tokens 64
```

Artifacts:

- Summary: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-smoke-baseline-20260615-170602/smoke_summary.json`
- Results JSONL: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-smoke-baseline-20260615-170602/smoke_results.jsonl`

Smoke result:

| Metric | Value |
| --- | ---: |
| Total cases | 20 |
| Passed | 19 |
| Failed | 1 |
| Pass rate | 95% |
| factual | 4/4 |
| math | 3/4 |
| coding | 4/4 |
| JSON | 4/4 |
| multilingual | 4/4 |

Failure:

| Case | Category | Expected | Reason | Output note |
| --- | --- | --- | --- | --- |
| `math_004` | math | `32` | missing substring `32` | Output was truncated at `2^5=` under `--max-new-tokens 64`. |

Disk-MoE telemetry printed at process exit:

| Metric | Value |
| --- | ---: |
| merge runs | 38221 |
| selected expert refs | 255631 |
| load requests | 511262 |
| loaded disk weights | 511262 |
| read calls | 1533786 |
| read bytes | 3,417,618,776,064 |
| read GiB | 3182.905518 |
| read seconds | 3208.422778 |
| effective expert read bandwidth | 0.992047 GiB/s |
| load seconds | 3214.034829 |

Decision against smoke gates:

- Baseline pass rate is `95%`.
- Future optimized routes must pass `>50%` and must not drop by more than `15 percentage points`, so the effective minimum relative to this baseline is `80%`.
- This baseline has no intentional accuracy-loss optimization and is the smoke reference for later routes.
- The smoke runner returned exit code `1` because it currently requires `20/20` to return zero. The measured pass rate is still valid for the plan gate.

Observation:

- The runner only writes `smoke_summary.json` after all cases finish. This made the long smoke run hard to monitor and would lose partial evidence if interrupted.

Next action:

- Improve the smoke runner so each case is written incrementally and the process can report progress without changing model semantics.
- Then start Route B/C implementation work aimed at reducing read calls and increasing actual expert read bandwidth.

## 2026-06-15 17:29:50 CST

### Route A Smoke Runner Reliability Commit

Purpose:

- Make future smoke runs observable and partially recoverable.
- This is instrumentation only. It does not change prompts, model loading, generation parameters, scoring logic, routing, top-k, active expert count, or model math.

Git status before commit:

```text
## master...private/master
 M tools/scripts/deepseek_v4_flash_smoke.py
?? .venv-ftllm/
```

Change:

- `tools/scripts/deepseek_v4_flash_smoke.py` now writes:
  - `smoke_results.partial.jsonl` after each completed case
  - `smoke_partial_summary.json` after each completed case
  - `[SMOKE_PROGRESS] ...` lines on stdout after each completed case

Validation:

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python -m py_compile \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_smoke.py
```

- Result: passed.

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_smoke.py \
  --smoke-file /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/smoke_eval.jsonl \
  --out-dir /tmp/deepseek_v4_smoke_score_test_incremental \
  --score-only /tmp/deepseek_v4_smoke_canned.jsonl
```

- Result: `20/20` score-only pass.

Commit:

```text
f49e035d7dcd283a909b00e9ea0693aac5159cfb feat: stream DeepSeek smoke progress
```

Push:

```text
f49e035d7dcd283a909b00e9ea0693aac5159cfb refs/heads/master
```

Git status after push:

```text
## master...private/master
?? .venv-ftllm/
```

Decision:

- Commit accepted as Route A instrumentation/reliability.
- No token-rate SOTA claim.

Next action:

- Start Route B/C code inspection and implement the first optimization that can reduce fragmented expert reads or improve expert read overlap.

## 2026-06-15 17:42:29 CST

### Route D Prototype: 2 GiB RAM Expert Cache Rejected

Purpose:

- Test whether a conservative 2 GiB RAM cache for loaded disk expert weights can improve short-run decode by replacing repeated SSD reads with memory copies.
- This prototype did not change routing, top-k, active expert count, or model math.
- The implementation was not committed.

Git status before run:

```text
## master...private/master
 M src/devices/disk/diskdevice.cpp
?? .venv-ftllm/
```

fastLLM base commit:

```text
f49e035d7dcd283a909b00e9ea0693aac5159cfb feat: stream DeepSeek smoke progress
```

Resource handling:

- GPU was idle at run start.
- A temporary 32 GiB swap file was used and removed after completion:
  - `/home/wici/lfz/deepseek-v4-ramcache-smoke.swap`
- No other user process was stopped.

Command:

```bash
RUN_DIR=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-ramcache-smoke-20260615-174045
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  --out-dir "$RUN_DIR" \
  --repeat 1 \
  --input-tokens 16 \
  --output-tokens 8 \
  --timeout-s 1200 \
  --max-wait-s 10 \
  --max-gpu-used-mib 1024 \
  --poll-s 2 \
  --extra-env FASTLLM_DISK_MOE_RAM_CACHE=1
```

Artifacts:

- Summary: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-ramcache-smoke-20260615-174045/summary.json`
- Log: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-ramcache-smoke-20260615-174045/bench_repeat_1.log`

Metrics:

| Metric | Short baseline | RAM cache prototype |
| --- | ---: | ---: |
| Input/output tokens | 16/8 | 16/8 |
| Decode tok/s | 1.26 | 0.88 |
| TTFT | 7.40317 s | 8.95722 s |
| Prefill | 2.16 tok/s | 1.79 tok/s |
| Batch total | 0.62 tok/s | 0.47 tok/s |
| Output hash | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` |
| Read GiB | 39.270996 | 39.270996 |
| Effective expert read GiB/s | 1.028398 | 0.914241 |
| RAM cache hits | 0 | 0 |
| RAM cache misses | 0 | 6308 |
| RAM cache hit rate | n/a | 0% |
| RAM cache bytes | 0 | 2,139,095,040 |
| RAM cache evictions | 0 | 5989 |

Decision:

- Rejected. Decode regressed by about `30%` on the short validation run.
- The cache produced `0%` hit rate and heavy eviction churn under the 2 GiB cap.
- The output hash matched baseline, so the prototype was semantically safe, but it does not improve performance.
- Per plan rollback rule, do not commit this route and restore fastLLM to the previous committed state.

Cause analysis:

- The current active expert working set exceeds the 2 GiB cache window even in the 16/8 smoke.
- Simple LRU over individual loaded weights churns before reuse, so it adds allocation/copy/cache-bookkeeping overhead without reducing SSD reads.
- A useful RAM tier likely needs route-profile-aware admission, not blind LRU, and should probably cache packed logical expert payloads after Route B rather than current fragmented tensor loads.

Rollback:

- Restore `/home/wici/lfz/fastllm/src/devices/disk/diskdevice.cpp` to `f49e035d7dcd283a909b00e9ea0693aac5159cfb`.

Next action:

- Continue with Route B/C instead of committing the rejected cache.
- Prioritize reducing read fragmentation and increasing effective expert-path read bandwidth before retrying any RAM cache.

## 2026-06-15 17:58:21 CST

### Route B/C Exploration: Load Threads And Expert Pack Infrastructure

Purpose:

- Check whether higher synchronous disk load thread count improves the current pread path.
- Confirm whether DeepSeek-V4-Flash disk experts can be repacked within available disk space.
- Add and validate default-off runtime support for a disk expert pack manifest.

Git status during implementation:

```text
## master...private/master
 M src/model.cpp
?? .venv-ftllm/
?? tools/scripts/deepseek_v4_flash_pack_experts.py
```

#### Load Thread Test

Command:

```bash
RUN_DIR=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-loadthreads16-smoke-20260615-174621
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  --out-dir "$RUN_DIR" \
  --repeat 1 \
  --input-tokens 16 \
  --output-tokens 8 \
  --timeout-s 1200 \
  --max-wait-s 10 \
  --max-gpu-used-mib 1024 \
  --poll-s 2 \
  --extra-env FASTLLM_DISK_MOE_LOAD_THREADS=16
```

Result:

| Metric | 8-thread short baseline | 16-thread short test |
| --- | ---: | ---: |
| Decode tok/s | 1.26 | 1.26 |
| TTFT | 7.40317 s | 7.72916 s |
| Read GiB/s | 1.028398 | 0.965024 |
| Output hash | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` |

Decision:

- Rejected as an optimization route. Increasing load threads to 16 does not improve token rate and slightly worsens TTFT/read bandwidth in this short test.
- Do not test 32 threads before improving layout/pipeline.

#### Expert Layout Finding

Safetensors layout inspection:

- Full routed expert tensors: `140.25 GiB`.
- Current free disk under `/home/wici/lfz`: about `135.78 GiB`.
- Full duplicate pack of every routed expert does not fit safely.
- Current placement `{'cuda':16,'disk':84}` selects disk layers `6-42`, so disk-tier expert pack estimate is `117.9375 GiB`.
- This is feasible but leaves limited free space.

Physical layout observation:

- For each expert, `w1/w2/w3.weight` are contiguous in the shard.
- `w1/w2/w3.scale` are also contiguous.
- Runtime `gateup` needs `w1+w3`, with `w2` physically between them.
- Current runtime reads exact parts, so it jumps between weight and scale regions separated by hundreds of MB.
- A pack ordered by runtime read order can reduce seek distance without changing tensor bytes.

#### Pack Infrastructure

Implemented in `/home/wici/lfz/fastllm`:

- `tools/scripts/deepseek_v4_flash_pack_experts.py`
  - Builds disk-layer expert pack files.
  - Writes per-tensor manifest entries with file, offset, byte size, dtype, shape, source offset, layer, expert, part, and SHA256.
  - Default pack order per expert:
    - `w1.weight`
    - `w1.scale`
    - `w3.weight`
    - `w3.scale`
    - `w2.weight`
    - `w2.scale`
- `src/model.cpp`
  - Adds default-off runtime support for `FASTLLM_DISK_MOE_PACK_MANIFEST`.
  - Overrides `DiskWeightPart.fileName/fileOffset` only for tensors present in the manifest.
  - Keeps original safetensors path as fallback for missing tensors.
  - Checks manifest byte size against original tensor byte size before overriding.

Validation:

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python -m py_compile \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_pack_experts.py
```

- Result: passed.

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_pack_experts.py \
  --out-dir /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-dryrun \
  --dry-run
```

- Result: disk layers `6-42`, `56832` tensors, `117.9375 GiB`.

```bash
CPATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/include \
LIBRARY_PATH=/home/wici/lfz/fastllm/build-route-a-nonuma/nccl-lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

- Result: passed.

Single-layer pack command:

```bash
/home/wici/lfz/fastllm/.venv-ftllm/bin/python \
  /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_pack_experts.py \
  --out-dir /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-layer6 \
  --layers 6
```

- Result: layer `6`, `1536` tensors, `3.1875 GiB`.

Single-layer pack 3-repeat short validation:

```bash
RUN_DIR=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-layer6-repeat-20260615-175424
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  --out-dir "$RUN_DIR" \
  --repeat 3 \
  --input-tokens 16 \
  --output-tokens 8 \
  --timeout-s 1200 \
  --max-wait-s 10 \
  --max-gpu-used-mib 1024 \
  --poll-s 2 \
  --extra-env FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-layer6/manifest.json
```

Artifacts:

- Summary: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-layer6-repeat-20260615-175424/summary.json`
- Logs: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-layer6-repeat-20260615-175424/bench_repeat_*.log`

Metrics:

| Repeat | Decode tok/s | TTFT (s) | Read GiB/s | Output hash |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.27 | 7.30978 | 1.052834 | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` |
| 2 | 1.28 | 7.30332 | 1.057591 | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` |
| 3 | 1.26 | 7.33109 | 1.040840 | `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0` |

Summary:

| Metric | Value |
| --- | ---: |
| Decode p50 | 1.27 tok/s |
| Decode worst | 1.26 tok/s |
| TTFT p50 | 7.30978 s |
| Output hash stable | yes |

Decision:

- Single-layer pack is semantically safe and runtime manifest fallback works.
- The short 3-repeat improvement is too small to claim SOTA.
- Commit the default-off Route B infrastructure, but do not record it as a token-rate SOTA improvement.

Next action:

- Generate full disk-layer pack for layers `6-42` if disk space remains sufficient.
- Run 3-repeat plan-shaped benchmark with full pack manifest.
- If full pack improves token rate reproducibly, record it as SOTA and keep the commit; otherwise revert or adjust pack order/IO backend.

## 2026-06-15 18:02:34 CST - Route B infrastructure pushed

Commit:

- fastLLM private/master: `12deab4a5e91145c4d0b0d3f34cf870f9a215836`
- Message: `feat: add DeepSeek disk expert pack manifest`
- Remote verification: `refs/heads/master` on `git@github.com:L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git`

Scope:

- Added `tools/scripts/deepseek_v4_flash_pack_experts.py` for DeepSeek-V4-Flash disk expert layer packing.
- Added default-off runtime manifest override via `FASTLLM_DISK_MOE_PACK_MANIFEST`.
- Did not claim token-rate SOTA from this commit; layer-6-only validation was reproducible but too small to be meaningful.

Updated constraint from user:

- Continue to prefer the original `2 GiB` CPU RAM expert tier.
- If evidence shows the `5 tok/s` target is practically unreachable under `2 GiB`, RAM expert cache may be relaxed up to `8 GiB`.
- Any result using the relaxed RAM limit must be explicitly labeled as `8 GiB RAM cache` and compared separately from strict-plan results.

## 2026-06-15 18:22:00 CST - Route B full pack benchmark

Artifacts:

- Full pack: `/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full`
- Manifest: `/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json`
- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-full-swap12-20260615-180857`
- Summary: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-full-swap12-20260615-180857/summary.json`

Resource notes:

- Initial full-pack run without extra swap was killed during model load around `Loading 89`.
- Deleted reproducible layer-6 pack and dry-run pack to free space.
- Re-ran with temporary `12 GiB` swap at `/home/wici/lfz/deepseek-v4-routeb-full.swap`.
- Temporary swap was removed automatically after the run.

Metrics:

| Repeat | Decode tok/s | TTFT (s) | Read GiB/s | Output hash |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.34 | 16.06744 | 1.161730 | `14947c5d527c9bacc3b4bc596865b060e6ba2dc2d05aa1f033bf56a72e5b85de` |
| 2 | 1.44 | 15.64041 | 1.358608 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |
| 3 | 1.34 | 15.69783 | 1.157793 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |

Summary:

| Metric | Value |
| --- | ---: |
| Decode p50 | 1.34 tok/s |
| Decode worst | 1.34 tok/s |
| TTFT p50 | 15.69783 s |
| TTFT worst | 16.06744 s |
| Read calls / repeat | 361170 |
| Read GiB / repeat | 749.498291 |

Decision:

- Route B full pack is a reproducible improvement over the instrumented full baseline (`1.29` p50, `1.21` worst), but it only matches the previously reported older SOTA p50 (`1.34`) and remains far below the `5 tok/s` target.
- The root bottleneck remains disk expert streaming shape: `361170` synchronous reads per repeat and only `1.16-1.36 GiB/s` measured read throughput despite the SSD being capable of much higher sequential fio bandwidth.
- Keep the default-off pack manifest infrastructure, but do not treat Route B alone as sufficient.

Next action:

- Implement Route C/F: copy the useful vramctl `io_uring` / pipeline ideas into fastLLM and add an expert prefetch backend.
- Goal for next route: increase effective read queue depth and move from synchronous per-part blocking reads toward batched/asynchronous expert reads.

## 2026-06-15 18:49:37 CST - Route C preadv scatter probe

Purpose:

- Test whether reducing synchronous read call count is enough to improve expert streaming bandwidth.
- This is a Route C probe, not the final `io_uring` / pipeline backend.

Implementation:

- Added an env-gated `FASTLLM_DISK_MOE_COALESCE_READ=1` path in `src/devices/disk/diskdevice.cpp`.
- Uses `preadv` scatter reads when adjacent `DiskWeightPart` entries are contiguous in the packed expert file.
- Avoids the earlier bounce-buffer coalesce prototype, which reduced `read_calls` but slowed short decode due to extra memcpy.

Short probe:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-preadv-smoke-swap8-20260615-183445`
- Decode: `1.33 tok/s`
- TTFT: `7.35064 s`
- Read calls: `7072` vs previous short packed path about `18924`
- Read bandwidth: `1.071426 GiB/s`
- Output hash: `0ce5b6b70ee0ac9bed8ae008ad69b12abd41f65e2b04109efc29158b98a6e4d0`

Full 3-repeat benchmark:

```bash
RUN_DIR=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-preadv-full-swap12-20260615-183627
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  --out-dir "$RUN_DIR" \
  --repeat 3 \
  --input-tokens 128 \
  --output-tokens 256 \
  --timeout-s 2400 \
  --max-wait-s 10 \
  --max-gpu-used-mib 1024 \
  --poll-s 2 \
  --extra-env FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --extra-env FASTLLM_DISK_MOE_COALESCE_READ=1
```

Artifacts:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-preadv-full-swap12-20260615-183627`
- Summary: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-preadv-full-swap12-20260615-183627/summary.json`

Metrics:

| Repeat | Decode tok/s | TTFT (s) | Read calls | Read GiB/s | Output hash |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.34 | 15.66695 | 135410 | 1.140864 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |
| 2 | 1.36 | 15.58567 | 134374 | 1.152893 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |
| 3 | 1.35 | 15.51767 | 135410 | 1.149422 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |

Summary:

| Metric | Value |
| --- | ---: |
| Decode p50 | 1.35 tok/s |
| Decode worst | 1.34 tok/s |
| TTFT p50 | 15.58567 s |
| TTFT worst | 15.66695 s |
| Read calls / repeat | about 135k |
| Read GiB / repeat | 749.498291 |
| Read bandwidth | 1.14-1.15 GiB/s |

Decision:

- `preadv` scatter reduces read calls by about `62%` versus Route B full pack (`361170` to about `135k`).
- It does not materially improve actual expert read bandwidth; the path remains near `1.15 GiB/s`.
- Decode p50 improves only marginally (`1.34` to `1.35 tok/s`), which is not enough to claim Route C success and has not been smoke-gated.
- Do not commit this as a SOTA result yet.

Conclusion:

- The main bottleneck is not only syscall count or file contiguity.
- The critical missing piece is outstanding IO depth: fastLLM still loads selected expert tensors synchronously on the decode critical path.
- Next Route C step must implement true queue-depth based async expert reads using `io_uring` / pipeline ideas from vramctl.

## 2026-06-15 19:13:00 CST - Route C io_uring probe

Purpose:

- Move from synchronous `pread` / `preadv` probes toward a queue-depth based `io_uring` read backend.
- Keep all new read paths default-off and do not claim SOTA unless token rate improves reproducibly and output/smoke gates pass.

Implementation status in `/home/wici/lfz/fastllm`:

- Added optional `liburing` detection in `CMakeLists.txt`.
- Added env-gated `FASTLLM_DISK_MOE_IO_URING=1` read backend with `FASTLLM_DISK_MOE_IO_URING_QD`.
- Added experimental cross-expert batching behind `FASTLLM_DISK_MOE_IO_URING_BATCH=1`.
- Added a destination-overlap guard so async reads fall back if a task batch writes overlapping memory ranges.

Short 64/64 control, packed path without `io_uring`:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-control-64x64-20260615-190445`
- Decode: `1.37 tok/s`
- TTFT: `12.69842 s`
- Read bandwidth: `1.184695 GiB/s`
- Output hash: `845647cc63461f068cd63ce5af73985a101d332a3a276469a42eb49c3f4585bc`

Basic per-weight `io_uring` probe:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-default-safe-20260615-190235`
- Decode: `1.53 tok/s`
- TTFT: `12.30143 s`
- Read bandwidth: `1.297419 GiB/s`
- `io_uring_batches`: `33892`
- `io_uring_submits`: `101676`
- Output hash: `77f8acd06814dd9c2e7f6450d78c1edd027a1e1e5e76411951ee0949eb45dbff`

Basic per-weight `io_uring` after destination-overlap guard:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-overlapguard-20260615-190848`
- Decode: `1.43 tok/s`
- TTFT: `12.40422 s`
- Read bandwidth: `1.134164 GiB/s`
- `io_uring_batches`: `33892`
- `io_uring_submits`: `101676`
- Output hash: `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Experimental cross-expert batch result:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-batched-smoke-swap8-20260615-185836`
- Read bandwidth improved to `3.539655 GiB/s`.
- `io_uring_batches` fell to `333`, showing that queue-depth/batching can materially raise SSD utilization.
- Decode regressed to `0.95 tok/s`.
- Output hash changed to `f116...`.
- This route is invalid as a performance result because it fails the semantic gate and slows decode.

Decision:

- Do not commit the current Route C code as an effective optimization.
- The `io_uring` direction is still the correct next route, but it must be redesigned around a correctness-preserving pipeline.
- Per-weight `io_uring` does not provide enough outstanding IO and is not a reproducible SOTA improvement.
- Cross-expert batching proves that higher IO depth can raise read bandwidth, but the current prototype is not valid.

Next action:

- Route C remains the next route.
- Replace the current cross-expert batch prototype with a safer pipeline:
  - prepare read plans without mutating execution order;
  - issue async reads only for direct expert payload buffers first;
  - keep conversion/scale/repack-sensitive parts on the synchronous path until validated;
  - validate each step with fixed benchmark hashes and smoke pass rate before recording any SOTA.

## 2026-06-15 19:25:30 CST - Route C worker/global batch probes

Purpose:

- Determine whether the earlier cross-expert `io_uring` bandwidth gain can be recovered without the severe decode regression.
- Compare per-worker batching, which preserves fastLLM's load-thread structure, against global batching, which maximizes queue depth.

Implementation status:

- `FASTLLM_DISK_MOE_IO_URING_BATCH=1` now uses per-worker batch by default.
- Added `FASTLLM_DISK_MOE_IO_URING_GLOBAL_BATCH=1` to explicitly enable the global read batch experiment.
- Global batch now does read globally but finishes loaded weights in parallel.
- All of this remains uncommitted and default-off.

Per-worker batch short probe:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-workerbatch-20260615-191545`
- Decode: `1.53 tok/s`
- TTFT: `12.70390 s`
- Read bandwidth: `1.196927 GiB/s`
- `io_uring_batches`: `19240`
- Result: decode is not worse than the short control, but read bandwidth is still near the synchronous path because each worker has too few read tasks to build useful SSD queue depth.

Global read batch with parallel finish:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-globalbatch-parfinish-20260615-191931`
- Decode: `1.05 tok/s`
- TTFT: `17.60480 s`
- Read bandwidth: `4.135939 GiB/s`
- `io_uring_batches`: `2405`
- Result: effective expert read bandwidth improves materially, but token rate regresses badly. This does not satisfy Route C gates and must not be committed as an optimization.

Conclusion:

- Higher SSD queue depth is achievable in fastLLM: global batch raises measured expert read bandwidth from about `1.18 GiB/s` to `4.14 GiB/s`.
- That bandwidth does not translate into decode rate because the global batch shape loses useful overlap and increases critical-path latency.
- Per-worker batching preserves decode but does not increase queue depth enough to materially improve read bandwidth.
- Route C alone is not sufficient in the current shape. Continuing to chase raw read bandwidth without reducing read volume is unlikely to reach `5 tok/s`.

Next action:

- Keep Route C code default-off and uncommitted.
- Move to a cache/prefetch route that reduces actual bytes read per token:
  - first retry RAM expert cache using the user's relaxed `8 GiB` cap as a separate experiment;
  - then combine cache hits with the safer per-worker `io_uring` path only if it improves token rate;
  - do not commit until repeat token rate improves and smoke/TTFT gates pass.

## 2026-06-15 19:29:30 CST - Route D 8 GiB RAM cache probe

Purpose:

- Test the user's allowed RAM relaxation from `2 GiB` to `8 GiB` after Route C failed to convert raw SSD bandwidth into token-rate gains.
- Goal was to reduce actual expert bytes read per token without changing routing or expert math.

Implementation status:

- Added default-off env-gated CPU RAM payload cache:
  - `FASTLLM_DISK_MOE_RAM_CACHE_BYTES=<bytes>`
  - caches finished CPU payload bytes for disk weights;
  - hit path allocates the same temporary `Data` metadata and copies payload bytes from RAM instead of reading from SSD;
  - no routing, top-k, or active expert count changes.
- This code remains uncommitted.

Short 64/64 probe with `8 GiB` cap:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-ramcache8g-smoke-20260615-192502`
- Decode: `1.20 tok/s`
- TTFT: `24.09916 s`
- RAM cache bytes: `8587575296`
- RAM cache hits: `17946`
- RAM cache misses: `15946`
- RAM cache evictions: `14661`
- Read bytes: `99.273193 GiB`
- Read bandwidth: `1.310138 GiB/s`

Decision:

- The cache successfully reduces SSD read volume by about half versus the 64/64 packed control (`210.997559 GiB` to `99.273193 GiB`).
- It still regresses token rate and violates the hard TTFT cap (`24.10 s > 19.98 s`).
- The cause is likely host-memory pressure plus large RAM memcpy/eviction overhead on this 15 GiB RAM host.
- Do not commit this route as an optimization.

Next action:

- Stop increasing RAM cache size in this form.
- Test VRAM placement / hot expert route next, because avoiding host copies entirely is more likely to reduce both read bytes and critical-path latency.

## 2026-06-15 19:31:30 CST - Route E placement probe

Purpose:

- Check whether the RTX 5090 can place more routed experts in VRAM using the existing runtime knobs before implementing a routing-profile hot expert selector.

Probe:

```text
--moe_device "{'cuda':24,'disk':76}"
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json
```

Result:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-placement-cuda24-20260615-192829`
- Process exited with return code `-11`.
- Log ended with:

```text
FastLLM Error: Disk MoE pack manifest byte mismatch for tensor: layers.25.ffn.experts.219.w2.scale
```

Decision:

- This is not a valid token-rate measurement.
- The current full pack manifest is tied to the existing `cuda:16,disk:84` placement and cannot be reused blindly for `cuda:24,disk:76`.
- Do not continue placement experiments with this manifest.

Next action:

- Either regenerate a placement-specific pack for any new VRAM hot set, or add manifest compatibility checks so incompatible entries fall back to the original safetensors path.
- Since disk space is already tight, the next low-risk implementation step is manifest-safe fallback rather than generating another full pack immediately.

### Placement fallback follow-up

Implementation:

- Added default-off manifest fallback:
  - `FASTLLM_DISK_MOE_PACK_FALLBACK_ON_MISMATCH=1`
  - byte-mismatched pack entries fall back to the original safetensors offset instead of aborting.
  - default behavior remains strict abort on mismatch.

`cuda:24,disk:76` with fallback:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-placement-cuda24-fallback-20260615-193108`
- No benchmark metrics were produced.
- Log shows repeated CUDA illegal address errors while releasing model weight slab.
- Not a valid result.

`cuda:20,disk:80` with fallback:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-placement-cuda20-fallback-20260615-193254`
- No benchmark metrics were produced.
- MoE stats before failure: `merge_runs=20`, `read_gib=1.494141`.
- Log again shows CUDA illegal address errors while releasing model weight slab.
- Not a valid result.

Decision:

- Naively increasing `cuda:N` above the current `16` is not a safe Route E implementation.
- The current fastLLM placement path appears unstable for these ratios, independent of the pack fallback.
- Do not report or commit these placement probes as improvements.

Next action:

- Revert or keep default-off experimental Route C/D/E support only after deciding the next implementation route.
- A valid Route E needs explicit routing-profile hot expert selection and a matching pack/manifest, not just a global `cuda` ratio.

## 2026-06-15 20:45 CST - Route E routing-profile VRAM cache committed

Implementation:

- Added routing profile collection:
  - `FASTLLM_DISK_MOE_ROUTE_PROFILE=/path/to/route_profile.tsv`
  - records `(layer, expert)` hit count and estimated bytes per routed expert pair.
- Added profile-guided VRAM hot expert cache:
  - `FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/path/to/route_profile.tsv`
  - `FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=<bytes>`
  - keeps selected hot expert weights resident on GPU after first load.
- Added CUDA-side reuse for cached weights:
  - if a temporary expert weight already has `cudaData`, skip the H2D copy in `DoCudaMergeMOEFromCPU`.
- Cleaned the commit before pushing:
  - removed failed/default-off Route C io_uring/coalesce leftovers from the submitted diff.
  - removed failed RAM-cache and placement fallback leftovers from the submitted diff.

Commit:

```text
7b5f0b8f423a2ac69b29e3fb6c7792800f978c44 feat: add routing-profile VRAM expert cache
```

Pushed to:

```text
git@github.com:L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git master
```

Profile run:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-64x64-20260615-194456`
- Profile file: `route_profile.tsv`
- Theoretical profile coverage:
  - `1 GiB`: 80 expert pairs, about `17.40%` active hit potential on the profile prompt.
  - `2 GiB`: 160 expert pairs, about `27.85%`.
  - `4 GiB`: 321 expert pairs, about `42.17%`.
  - `8 GiB`: 642 expert pairs, about `59.83%`.

Short A/B check at `64/64`, `gpu_mem_ratio=0.90`:

| Route | Decode tok/s | TTFT | Read GiB | Output hash |
| --- | ---: | ---: | ---: | --- |
| control | `1.33` | `12.68005 s` | `210.997559` | same |
| 1 GiB VRAM cache | `1.39` | `12.81826 s` | `180.566895` | same |

Full reproducibility run:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1g-keepcpu-gpumem90-full-20260615-200625`
- Command shape:
  - `--input_tokens 128`
  - `--output_tokens 256`
  - repeat `3`
  - `--moe_device "{'cuda':16,'disk':84}"`
  - `--gpu_mem_ratio 0.90`
  - `FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json`
  - `FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824`
  - `FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-64x64-20260615-194456/route_profile.tsv`

Metrics:

| Metric | Previous SOTA | Route E 1 GiB VRAM cache |
| --- | ---: | ---: |
| Decode p50 | `1.34 tok/s` | `1.45 tok/s` |
| Decode worst | `1.34 tok/s` | `1.44 tok/s` |
| TTFT p50 | `15.70 s` | `15.81617 s` |
| TTFT worst | n/a | `15.86857 s` |
| Read GiB / repeat | `749.498291` | `659.687988` |
| Output hash | baseline stable | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` |

Per-repeat decode:

```text
1.57, 1.44, 1.45 tok/s
```

Per-repeat TTFT:

```text
15.86857, 15.72681, 15.81617 s
```

Telemetry:

```text
vram_cache_bytes=1069547520
vram_cache_stores=160
vram_cache_hits=14426
vram_cache_misses=176
selected_expert_refs=60195
read_calls=317892
read_gib=659.687988
```

Smoke gate:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1g-smoke-20260615-202037`
- Result: `20/20`, pass rate `100%`.
- Category pass rates:
  - factual `4/4`
  - math `4/4`
  - coding `4/4`
  - JSON `4/4`
  - multilingual `4/4`

Gate decision:

- This is a valid SOTA improvement and was committed/pushed.
- TTFT remains under the hard cap: `15.81617 s <= 19.98 s`.
- Smoke pass rate is above the required baseline floor: `100% >= 80%`.
- Target is not reached: `1.45 tok/s < 5.0 tok/s`.

Failed adjacent probes:

- `2 GiB` VRAM cache with the current placement/gpu-mem settings OOMed or hit CUDA illegal address.
- Naively increasing `cuda:N` above `16` remains unsafe with the current fastLLM placement path.
- Route C-style global io_uring/pread batching previously increased apparent read bandwidth in some runs but reduced decode token rate, so it is not the next mainline route.

Next action:

- Do not start by returning to Route C as-is.
- Next mainline should be Route E2: redesign VRAM capacity/placement so a `2-4 GiB` profile-guided hot set can fit safely.
- Route C can be revisited only as a local backend for remaining cache misses after Route E2, not as the primary next step.

## 2026-06-15 21:09 CST - Route E2 VRAM cache expansion/backoff probes

Goal:

- Increase active hot expert cache above the Route E `1 GiB` SOTA without changing routing semantics.
- Keep output hash stable and TTFT within the hard cap.
- Determine whether a safer runtime free-memory backoff lets `1.25-2 GiB` profile-guided VRAM cache fit on the 5090.

Implementation under test:

- Local uncommitted experiment in `src/devices/disk/diskdevice.cpp`.
- Added `FASTLLM_DISK_MOE_VRAM_CACHE_RESERVE_BYTES`.
- Added `vram_cache_backoffs` telemetry.
- The cache stops adding new expert weights when CUDA free memory is below `next_weight_bytes + reserve`.

Important status:

- This code is not committed.
- It did not produce a reproducible SOTA improvement.
- Current committed SOTA remains:

```text
7b5f0b8f423a2ac69b29e3fb6c7792800f978c44 feat: add routing-profile VRAM expert cache
```

Probe results:

| Run | Shape | Result | Decode | TTFT | VRAM cache bytes | Read GiB | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `route-e2-vramcache2g-gpumem70-slab128-64x64-20260615-204823` | `64/64` | fail | n/a | n/a | n/a | n/a | CUDA OOM / illegal address |
| `route-e2-vramcache2g-backoff-gpumem70-slab128-64x64-20260615-205126` | `64/64` | fail | n/a | n/a | n/a | n/a | backoff did not prevent later temp allocation OOM |
| `route-e2-vramcache1536m-backoff2g-gpumem85-64x64-20260615-205300` | `64/64` | pass | `1.43` | `12.82924 s` | `213909504` | n/a | reserve too conservative; only about `204 MiB` cached |
| `route-e2-vramcache1536m-backoff1g-gpumem85-64x64-20260615-205528` | `64/64` | pass | `1.44` | `13.11988 s` | `762052608` | n/a | below SOTA |
| `route-e2-vramcache1536m-backoff512m-gpumem85-64x64-20260615-205748` | `64/64` | pass | `1.47` | `12.83375 s` | `975962112` | `186.095` | promising short run only |
| `route-e2-vramcache1536m-backoff512m-gpumem85-128x256-single-20260615-210029` | `128/256` | fail | n/a | n/a | n/a | n/a | long run CUDA OOM / illegal address |
| `route-e2-vramcache1280m-backoff512m-gpumem90-128x256-single-20260615-210427` | `128/256` | pass | `1.42` | `17.75851 s` | `735313920` | `693.729492` | stable but below SOTA |

Latest single long-run command:

```text
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_gate.py \
  --repo /home/wici/lfz/fastllm \
  --model /home/wici/lfz/models/DeepSeek-V4-Flash \
  --ftllm /home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm \
  --out-dir /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e2-vramcache1280m-backoff512m-gpumem90-128x256-single-20260615-210427 \
  --repeat 1 \
  --input-tokens 128 \
  --output-tokens 256 \
  --threads 8 \
  --moe-device "{'cuda':16,'disk':84}" \
  --gpu-mem-ratio 0.90 \
  --cuda-slab 256 \
  --timeout-s 1800 \
  --extra-env FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
  --extra-env FASTLLM_DISK_MOE_LOAD_THREADS=8 \
  --extra-env FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --extra-env FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-64x64-20260615-194456/route_profile.tsv \
  --extra-env FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1342177280 \
  --extra-env FASTLLM_DISK_MOE_VRAM_CACHE_RESERVE_BYTES=536870912
```

Latest single long-run metrics:

```text
decode_tok_s=1.42
ttft_s=17.75851
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
read_gib=693.729492
read_gib_per_s=1.204338
read_calls=334296
vram_cache_bytes=735313920
vram_cache_stores=110
vram_cache_hits=8958
vram_cache_misses=9310
vram_cache_backoffs=1
```

Decision:

- No commit, because the best long `128/256` E2 result is worse than the committed Route E SOTA:
  - SOTA decode p50: `1.45 tok/s`
  - E2 `1.25 GiB` single-run decode: `1.42 tok/s`
- The simple free-memory backoff protects some short runs but is not enough for larger hot sets. It either stores too little cache or still leaves later decode-time CUDA temporary allocations exposed.

Next action:

- Do not return to Route C as the immediate mainline.
- Keep Route E as the SOTA baseline.
- Next candidate should be a more deterministic VRAM budget allocator:
  - reserve activation/temp workspace before hot expert caching,
  - load hot experts after model placement stabilizes,
  - cap cache by measured steady-state free memory rather than first OOM/backoff,
  - then retest `1.125 GiB`, `1.25 GiB`, and `1.5 GiB` cache sizes on the full `128/256` gate.
- Only after VRAM hit rate stops improving should Route C be reintroduced as a miss-path read backend.

## 2026-06-15 21:32 CST - Route E2 dynamic VRAM cap and decode-only store probes

Goal:

- Make VRAM hot expert cache expansion more deterministic than the previous permanent backoff experiment.
- Preserve the committed Route E behavior unless new env flags are explicitly set.
- Test whether cache capacity above `1 GiB` can be made stable on the full `128/256` gate.

Implementation under test:

- Local uncommitted changes in `src/devices/disk/diskdevice.cpp`.
- Added explicit opt-in dynamic cap:
  - `FASTLLM_DISK_MOE_VRAM_CACHE_DYNAMIC_CAP=1`
  - uses current CUDA free memory plus an estimated CUDA merge workspace reserve before caching another hot expert weight.
- Added telemetry:
  - `vram_cache_min_free_bytes`
  - `vram_cache_reserve_bytes`
  - `vram_cache_store_skips`
- Added decode-only cache-store gate:
  - `FASTLLM_DISK_MOE_VRAM_CACHE_STORE_MAX_BATCH=1`
  - intended to skip cache writes during large-batch prefill and write only in batch-1 decode.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed.

Probe results:

| Run | Shape | Extra env | Result | Decode | TTFT | VRAM cache bytes | Read GiB | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `route-e2-dyncap1280m-gpumem90-64x64-20260615-211403` | `64/64` | dynamic cap, `1280 MiB` limit | pass | `1.54` | `12.97491 s` | `855638016` | `187.240723` | promising short run only |
| `route-e2-dyncap1280m-gpumem90-128x256-single-20260615-211622` | `128/256` | dynamic cap, `1280 MiB` limit | fail | n/a | n/a | n/a | n/a | CUDA OOM after warmup, `gpuFree: 6 MB` |
| `route-e2-dyncap1152m-gpumem90-128x256-single-20260615-211811` | `128/256` | dynamic cap, `1152 MiB` limit | fail | n/a | n/a | n/a | n/a | CUDA OOM after warmup |
| `route-e2-dyncap1280m-decodestore-gpumem90-128x256-single-20260615-212118` | `128/256` | dynamic cap, decode-only store | pass | `1.36` | `15.54426 s` | `0` | `749.498291` | no cache stores; batch-1 decode does not enter CUDA disk-MoE with default min tokens |
| `route-e2-decodecuda-dyncap1280m-64x64-20260615-212613` | `64/64` | dynamic cap, decode-only store, `FASTLLM_DISK_MOE_GPU_PREFILL_MIN_TOKENS=1` | pass | `1.22` | `13.00051 s` | `1149763584` | `187.091309` | cache stores work, but decode regresses and output hash changes |

Key evidence:

```text
route-e2-dyncap1280m-gpumem90-64x64:
decode_tok_s=1.54
vram_cache_bytes=855638016
vram_cache_stores=128
vram_cache_hits=3816
vram_cache_misses=2274
vram_cache_backoffs=72
vram_cache_min_free_bytes=161873920
vram_cache_reserve_bytes=297012224

route-e2-dyncap1280m-decodestore-gpumem90-128x256:
decode_tok_s=1.36
ttft_s=15.54426
vram_cache_bytes=0
vram_cache_stores=0
vram_cache_store_skips=200
read_gib=749.498291
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81

route-e2-decodecuda-dyncap1280m-64x64:
decode_tok_s=1.22
vram_cache_bytes=1149763584
vram_cache_stores=172
vram_cache_hits=3474
vram_cache_misses=886
output_hash=6a76123df4ac7f884fa57117a099a099c2a8b0e8a6d48dd16bfb6da0d908a635
```

Decision:

- No commit.
- The dynamic cap improves one short `64/64` run but does not survive the full `128/256` gate above `1 GiB`.
- Delaying cache writes to decode-only is not useful with the default CUDA disk-MoE threshold because batch-1 decode does not write the VRAM cache.
- Lowering `FASTLLM_DISK_MOE_GPU_PREFILL_MIN_TOKENS` to `1` enables cache writes in decode but regresses speed and changes the output hash, so it is rejected.
- Roll back the uncommitted E2 experiment to the committed Route E SOTA.

Next action:

- Current SOTA remains Route E commit `7b5f0b8f423a2ac69b29e3fb6c7792800f978c44`.
- Route E2 capacity expansion above `1 GiB` is blocked by post-warmup CUDA free memory, not by profile selection.
- Next viable route should target the remaining miss path rather than further VRAM cache expansion:
  - revisit Route C only as a miss-path backend,
  - avoid global batching that previously reduced decode rate,
  - focus on preserving current Route E cache hits while increasing effective read bandwidth for misses.

## 2026-06-15 21:35 CST - Route C miss-path `preadv` with Route E cache

Goal:

- Re-test the low-risk Route C `preadv` coalesced read path on top of the current Route E `1 GiB` VRAM hot expert cache.
- Keep expert order and math unchanged.
- Only optimize the remaining cache-miss read path.

Implementation under test:

- Local uncommitted `src/devices/disk/diskdevice.cpp` change.
- Added default-off env:
  - `FASTLLM_DISK_MOE_COALESCE_READ=1`
- When enabled, consecutive `DiskReadTask` entries that read contiguous offsets from the same packed expert file are loaded with one synchronous `preadv` call into separate destination buffers.
- No global batching, no async reorder, no `io_uring`.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed.

Short probe:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-coalesce-routee1g-64x64-20260615-213218`
- Shape: `64/64`
- Base route: Route E `1 GiB` VRAM profile cache
- Extra env:

```text
FASTLLM_DISK_MOE_COALESCE_READ=1
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-64x64-20260615-194456/route_profile.tsv
```

Metrics:

```text
decode_tok_s=1.37
ttft_s=13.33669
read_gib=202.132324
read_gib_per_s=1.069887
read_calls=35204
vram_cache_bytes=1069547520
vram_cache_stores=160
vram_cache_hits=1424
vram_cache_misses=176
output_hash=ae8a094bd3cd2f92f31901c75f59aa20d4451697f8b72dcae39f902c1195eadb
```

Decision:

- No commit.
- This does not improve over the earlier Route E short check:
  - Route E short decode: `1.39 tok/s`
  - Route C miss-path coalesce + Route E short decode: `1.37 tok/s`
- The read-call reduction is not enough; effective read bandwidth remains about `1.07 GiB/s`.
- Roll back this uncommitted Route C code.

Next action:

- Current SOTA remains Route E commit `7b5f0b8f423a2ac69b29e3fb6c7792800f978c44`.
- The next read-speed route needs real overlap or prefetch without changing output hash:
  - route-aware miss prefetch for the next layer/token,
  - or a correctness-preserving per-layer async read queue,
  - not simple syscall coalescing.

## 2026-06-15 21:40 CST - Route F pinned temporary expert payload probe

Goal:

- Reduce H2D copy overhead for disk expert miss weights.
- Keep routing, active expert count, and math unchanged.
- Test a default-off pinned host memory path for temporary disk expert payloads.

Implementation under test:

- Local uncommitted `src/devices/disk/diskdevice.cpp` change.
- Added default-off env:
  - `FASTLLM_DISK_MOE_PINNED_TEMP_WEIGHT=1`
- The disk temp weight load path registers the temporary CPU payload with CUDA host registration after allocation.
- `Data::ToCudaTemporary` then uses the existing fastLLM pinned H2D copy path.
- Host registration is released before deleting temp weights; for VRAM-cached weights it is released immediately after GPU caching succeeds.
- Added telemetry:
  - `pinned_temp_weights`
  - `pinned_temp_bytes`
  - `pinned_temp_register_failures`
  - `pinned_temp_enabled`

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed.

Short probe:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-pinnedtmp-routee1g-64x64-20260615-213832`
- Shape: `64/64`
- Base route: Route E `1 GiB` VRAM profile cache
- Extra env:

```text
FASTLLM_DISK_MOE_PINNED_TEMP_WEIGHT=1
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-64x64-20260615-194456/route_profile.tsv
```

Metrics:

```text
decode_tok_s=1.33
ttft_s=11.48638
read_gib=190.341064
read_gib_per_s=1.513722
load_seconds=168.645412
pinned_temp_weights=30574
pinned_temp_bytes=204377161728
pinned_temp_register_failures=0
vram_cache_bytes=1069547520
vram_cache_hits=3318
vram_cache_misses=176
output_hash=47a0b621cfe12354562f3f0e06cf7cbf06dc35ec9a015eff47f7d075030bd952
```

Decision:

- No commit.
- Although measured read bandwidth rises to `1.51 GiB/s`, decode regresses versus Route E short check:
  - Route E short decode: `1.39 tok/s`
  - pinned temp decode: `1.33 tok/s`
- Registering/unregistering about `30k` pinned regions per short run adds too much critical-path overhead.
- Roll back this uncommitted experiment.

Next action:

- Current SOTA remains Route E commit `7b5f0b8f423a2ac69b29e3fb6c7792800f978c44`.
- Do not pursue per-weight pinned registration further.
- Any future pinned path would need a reusable pinned buffer pool, not per-weight registration.

## 2026-06-15 21:45 CST - Route F reusable pinned pool probe

Goal:

- Avoid the high per-weight `cudaHostRegister` overhead seen in the pinned temp probe.
- Reuse pinned host buffers for disk expert temp payloads.
- Keep routing and math unchanged.

Implementation under test:

- Local uncommitted `src/devices/disk/diskdevice.cpp` change.
- Added default-off env:
  - `FASTLLM_DISK_MOE_PINNED_POOL_BYTES=<bytes>`
- The disk temp weight loader tries to allocate temp `cpuData` from a CUDA pinned host buffer pool.
- Released temp weights return pinned buffers to the pool before `Data` destruction.
- VRAM-cached weights release their CPU pinned buffer after the GPU cache copy succeeds.
- Added telemetry:
  - `pinned_pool_bytes`
  - `pinned_pool_hits`
  - `pinned_pool_misses`
  - `pinned_pool_fallbacks`
  - `pinned_pool_releases`

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed.

Short probe:

- Run dir: `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-pinnedpool2g-routee1g-64x64-20260615-214336`
- Shape: `64/64`
- Base route: Route E `1 GiB` VRAM profile cache
- Extra env:

```text
FASTLLM_DISK_MOE_PINNED_POOL_BYTES=2147483648
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-64x64-20260615-194456/route_profile.tsv
```

Result:

```text
returncode=-11
decode_tok_s=n/a
ttft_s=n/a
disk_moe_stats={}
```

Log:

- Process reached model loading `100`.
- It segfaulted before benchmark metrics or disk-MoE stats were emitted.

Decision:

- No commit.
- The reusable pinned pool implementation is not stable enough to benchmark.
- Roll back this uncommitted experiment.

Next action:

- Current SOTA remains Route E commit `7b5f0b8f423a2ac69b29e3fb6c7792800f978c44`.
- Avoid pinned temp/pool work unless there is time to redesign ownership deeply in `Data`.
- Continue with prefetch/async read designs that do not alter `Data` host allocation ownership.

## 2026-06-15 22:26 CST - Route E profile refresh becomes new SOTA

User question:

- "下一步是 route C 吗"

Answer:

- No. Route C as previously tested is not the next mainline because the
  miss-path read coalescing probe regressed token rate.
- The next mainline was to validate a new routing profile with the existing
  Route E `1 GiB` VRAM expert cache.

New profile:

- Generated profile run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-128x64-20260615-214823`
- Profile source:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-128x64-20260615-214823/route_profile.tsv`
- Profile copied into fastLLM for reproducibility:
  `/home/wici/lfz/fastllm/tools/profiles/deepseek_v4_flash_route_profile_128x64.tsv`
- Profile generation shape: `128` input tokens / `64` output tokens.

Profile generation metrics:

```text
decode_tok_s=1.34
ttft_s=15.6204
read_gib=218.779541
selected_expert_refs=17571
output_hash=d8dcad...
```

Full single probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1g-profile128x64-128x256-single-20260615-215444`
- Shape: `128/256`
- Env:

```text
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-profile-128x64-20260615-214823/route_profile.tsv
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824
FASTLLM_DISK_MOE_LOAD_THREADS=8
FASTLLM_DISK_MOE_STATS=1
```

Single metrics:

```text
decode_tok_s=1.50
ttft_s=15.93241
read_gib=629.979492
read_gib_per_s=1.208420
read_calls=303576
vram_cache_bytes=1069547520
vram_cache_stores=160
vram_cache_hits=19198
vram_cache_misses=174
selected_expert_refs=60195
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
```

Full repeat gate:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1g-profile128x64-128x256-full-20260615-215916`
- Shape: `128/256`
- Repeats: `3`
- Command placement:
  - `--moe-device "{'cuda':16,'disk':84}"`
  - `--gpu-mem-ratio 0.90`
  - `--cuda-slab 256`
  - `--threads 8`

Repeat metrics:

```text
decode_tok_s_values=[1.56, 1.50, 1.57]
decode_tok_s_p50=1.56
decode_tok_s_worst=1.50
ttft_s_values=[15.62350, 15.63186, 15.66776]
ttft_s_p50=15.63186
ttft_s_worst=15.66776
read_gib_each=629.979492
read_gib_per_s_values=[1.283066, 1.164093, 1.291200]
read_calls_each=303576
vram_cache_bytes=1069547520
vram_cache_stores=160
vram_cache_hits=19198
vram_cache_misses=174
selected_expert_refs=60195
output_hash_all=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
```

Smoke gate:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1g-profile128x64-smoke-20260615-221203`
- Result: `19/20`, pass rate `95%`.
- Category pass rates:
  - factual `4/4`
  - math `4/4`
  - coding `4/4`
  - JSON `3/4`
  - multilingual `4/4`
- Failed case:
  - `json_004`; output stopped before emitting a parseable JSON object.

Gate decision:

- This is a valid reproducible token-rate improvement.
- Previous SOTA:
  - commit `7b5f0b8f423a2ac69b29e3fb6c7792800f978c44`
  - decode p50 `1.45 tok/s`, worst `1.44 tok/s`
  - TTFT p50 `15.81617 s`
  - smoke `20/20`
- New SOTA candidate:
  - decode p50 `1.56 tok/s`, worst `1.50 tok/s`
  - TTFT p50 `15.63186 s`
  - smoke `19/20`
- TTFT remains under the hard cap: `15.63186 s <= 19.98 s`.
- Smoke pass rate remains at the no-loss baseline floor: `95% >= 80%`.
- Target is still not reached: `1.56 tok/s < 5.0 tok/s`.

Commit plan:

- fastLLM source code is unchanged.
- The effective reproducibility artifact is the refreshed routing profile,
  copied into fastLLM under `tools/profiles/`.
- Committed and pushed to `private/master`:
  - `92e4547e040a3067f60eff07ae1ac3ea7a8f3e27`
  - `perf: add DeepSeek V4 Flash hot expert profile`

Next action:

- Continue focusing on read speed for cache misses. The new profile improved
  token rate by increasing useful VRAM-cache hits, but measured disk read
  bandwidth remains only about `1.16-1.29 GiB/s`, far below the SSD fio ceiling.
- Do not resume old Route C globally as-is. Revisit read acceleration only as a
  miss-path backend under the current SOTA profile/cache configuration.

## 2026-06-15 22:48 CST - Route C/F `posix_fadvise(WILLNEED)` miss-path probe

Goal:

- Improve miss-path read overlap without changing routing, buffer ownership, or
  expert math.
- Test whether kernel readahead hints can reduce effective disk wait before the
  existing worker-thread reads.

Implementation under test:

- Local uncommitted `src/devices/disk/diskdevice.cpp` change.
- Added default-off env:
  - `FASTLLM_DISK_MOE_FADVISE_WILLNEED=1`
  - `FASTLLM_DISK_MOE_FADVISE_MIN_BYTES=<bytes>`
- After `loadIndices` are known for a `DiskMergeMOE` call, the code merged
  adjacent packed ranges for the missing weights and issued
  `posix_fadvise(..., POSIX_FADV_WILLNEED)` before entering the existing
  worker-thread load path.
- Added temporary telemetry:
  - `fadvise_calls`
  - `fadvise_bytes`
  - `fadvise_seconds`
  - `fadvise_failures`
  - `fadvise_enabled`

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed.

Short control:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-fadvise-control-64x64-20260615-223240`
- Shape: `64/64`
- Base route: current SOTA profile cache from commit
  `92e4547e040a3067f60eff07ae1ac3ea7a8f3e27`
- Result:

```text
decode_tok_s=1.35
ttft_s=13.43368
read_gib=197.276367
read_gib_per_s=1.077298
read_calls=95064
load_requests=31688
vram_cache_hits=2204
vram_cache_misses=174
output_hash=6bf00d4277c4dc997b9417c26687d348ee9d24c7ca3408c519282e822f5c8511
```

Short fadvise probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-fadvise-willneed-64x64-20260615-223515`
- Shape: `64/64`
- Extra env:

```text
FASTLLM_DISK_MOE_FADVISE_WILLNEED=1
```

Result:

```text
decode_tok_s=1.50
ttft_s=12.98226
read_gib=185.397949
read_gib_per_s=1.265846
read_calls=89340
load_requests=29780
fadvise_calls=17611
fadvise_bytes=199069532160
fadvise_seconds=1.208450
fadvise_failures=0
vram_cache_hits=4112
vram_cache_misses=174
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Notes:

- The short fadvise result was better than the immediately preceding short
  control, but the output hash and route/cache hit pattern differed.
- It was also below the earlier current-profile short check (`1.56 tok/s`),
  so it was not strong enough to claim an improvement.

Full single fadvise probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-fadvise-willneed-128x256-single-20260615-223752`
- Shape: `128/256`
- Extra env:

```text
FASTLLM_DISK_MOE_FADVISE_WILLNEED=1
```

Result:

```text
decode_tok_s=1.48
ttft_s=15.79574
read_gib=629.979492
read_gib_per_s=1.242660
read_calls=303576
load_requests=101192
fadvise_calls=62001
fadvise_bytes=676435329024
fadvise_seconds=4.543505
fadvise_failures=0
vram_cache_hits=19198
vram_cache_misses=174
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
```

Decision:

- No commit.
- Full single fadvise decode `1.48 tok/s` is below current SOTA:
  - SOTA full single: `1.50 tok/s`
  - SOTA 3-repeat p50: `1.56 tok/s`
  - SOTA 3-repeat worst: `1.50 tok/s`
- TTFT remains within gate, and output hash matches the full SOTA hash, but
  there is no token-rate improvement.
- The extra `posix_fadvise` syscalls add about `4.54 s` aggregate hint time in
  the full run and do not create enough useful overlap.

Rollback:

- Removed the uncommitted fadvise implementation and telemetry from
  `src/devices/disk/diskdevice.cpp`.
- Rebuilt `fastllm_tools` successfully after rollback.
- fastLLM git status after rollback:

```text
?? .venv-ftllm/
```

Next action:

- Do not pursue per-layer `posix_fadvise` hints further.
- The next read-speed attempt should either:
  - move real IO work earlier than the current `DiskMergeMOE` call, or
  - reduce miss read bytes with an additional cache tier.
- Simple same-call pre-read hints are too late in the critical path.

### 2026-06-15 23:20 CST - Route D relaxed-RAM profile cache

Goal:

- Reduce recurring SSD expert misses by adding a profile-guided CPU RAM cache
  after the existing VRAM hot expert set.
- This uses the user-approved relaxed RAM budget path. It is **not** a strict
  2 GiB RAM result.

Implementation:

- Added `FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES`.
- Added optional `FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_PROFILE`; if unset it
  reuses `FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE`.
- RAM hot-set selection skips the bytes already assigned to
  `FASTLLM_DISK_MOE_VRAM_CACHE_BYTES`, then fills RAM with the next hottest
  profile entries.
- Added RAM cache telemetry:
  `ram_cache_hits`, `ram_cache_misses`, `ram_cache_stores`,
  `ram_cache_bytes`, `ram_cache_enabled`.

Short 2 GiB probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram2g-profile-cache-64x64-20260615-224709`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.44
ttft_s=12.83692
read_gib=190.851562
ram_cache_bytes=2018770944
ram_cache_hits=1764
ram_cache_misses=302
ram_cache_stores=302
vram_cache_hits=1472
output_hash=52b99c74bf4aff9bb8739118f0d3cf588421a687db1599e9c3d67d32c2f56428
```

Decision:

- 2 GiB is not enough to improve over the current SOTA.
- Keep as a negative strict-RAM data point.

Short 4 GiB relaxed-RAM probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram4g-profile-cache-64x64-20260615-224940`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.63
ttft_s=13.08105
read_gib=136.975342
read_calls=66006
ram_cache_bytes=4184604672
ram_cache_hits=7778
ram_cache_misses=626
ram_cache_stores=626
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Full single 4 GiB relaxed-RAM probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram4g-profile-cache-128x256-single-20260615-225156`
- Shape: `128/256`
- Result:

```text
decode_tok_s=1.71
ttft_s=15.84299
read_gib=411.150146
read_calls=198126
ram_cache_bytes=4291559424
ram_cache_hits=35150
ram_cache_misses=642
ram_cache_stores=642
vram_cache_hits=19198
vram_cache_misses=174
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
```

Full repeat 4 GiB relaxed-RAM probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram4g-profile-cache-128x256-full-20260615-225618`
- Shape: `128/256`, 3 repeats
- Result:

```text
decode_tok_s=[1.62, 1.67, 1.70]
decode_p50_tok_s=1.67
decode_worst_tok_s=1.62
ttft_s=[15.88652, 15.86007, 15.89992]
ttft_p50_s=15.88652
ttft_worst_s=15.89992
read_gib=411.150146
read_calls=198126
ram_cache_bytes=4291559424
ram_cache_hits=35150
ram_cache_misses=642
ram_cache_stores=642
ram_cache_evictions=0
vram_cache_bytes=1069547520
vram_cache_hits=19198
vram_cache_misses=174
vram_cache_stores=160
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
```

Comparison to previous SOTA:

- Previous SOTA:
  - commit `92e4547e040a3067f60eff07ae1ac3ea7a8f3e27`
  - route E, 1 GiB VRAM profile cache
  - full p50 `1.56 tok/s`, worst `1.50 tok/s`
  - TTFT p50 `15.63186 s`, worst `15.66776 s`
- New relaxed-RAM result:
  - full p50 `1.67 tok/s`, worst `1.62 tok/s`
  - p50 improvement: `+0.11 tok/s`, about `+7.1%`
  - worst improvement: `+0.12 tok/s`, about `+8.0%`
  - TTFT p50 regression: about `+1.6%`, within the 25% gate.

Smoke gate:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram4g-profile-cache-smoke-20260615-230754`
- Result: `20/20`, pass rate `100%`.
- This is above the required 80% floor and does not regress below the previous
  no-loss smoke baseline.

Decision:

- Commit and push the 4 GiB relaxed-RAM profile cache implementation because
  token rate improves reproducibly and smoke passes.
- fastLLM commit:
  `36eb1934 perf: add profile-guided RAM expert cache`
- Pushed to:
  `private/master`
- Keep pursuing read-speed work. Even with RAM cache, reported SSD read
  throughput is still about `1.2 GiB/s`, far below the 8-12 GB/s target.

### 2026-06-15 23:58 CST - Route C io_uring backend probes, reverted

Goal:

- Port the useful vramctl io_uring/pipeline design into fastLLM as an
  expert-read backend with queue depth, aligned bounce buffers, batched
  submissions, and fallback to `pread`.
- Preserve output semantics while improving actual expert read bandwidth and
  token rate.

Git status before:

```text
?? .venv-ftllm/
```

Implementation attempted:

- Added optional `liburing` detection/linking in `CMakeLists.txt`.
- Added `FASTLLM_DISK_MOE_READ_BACKEND=iouring`.
- Added `FASTLLM_DISK_MOE_IORING_QD`, default `16`.
- Added `FASTLLM_DISK_MOE_IORING_UNIT_BYTES`, default `4194304`.
- Added `FASTLLM_DISK_MOE_IORING_DIRECT`, default-on for direct IO probes.
- Added fixed aligned bounce buffers, submit/completion polling, fallback to
  existing `pread`, and telemetry:
  `read_backend`, `iouring_batches`, `iouring_submits`,
  `iouring_completions`, `iouring_fallbacks`, `iouring_bytes`,
  `iouring_seconds`.
- Tried both merge-level batch loading and conservative per-weight loading.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probes. After rollback, build passed again.

Control run:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-pread-control-staticram4g-64x64-20260615-233707`
- Shape: `64/64`
- Env: 1 GiB VRAM profile cache + 4 GiB relaxed RAM profile cache, existing
  `pread` backend.

```text
decode_tok_s=1.54
ttft_s=13.96138
total_time_s=54.7461
read_gib=136.975342
read_seconds=115.565280
read_gib_per_s=1.185264
ram_cache_hits=7778
ram_cache_misses=626
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Probe matrix:

| Probe | Run dir | Result |
| --- | --- | --- |
| merge-level io_uring direct, QD16, 4 MiB | `route-c-iouring-qd16-4m-direct-staticram4g-64x64-20260615-233332` | `decode=1.37 tok/s`, `read_gib_per_s=5.616769`, `iouring_fallbacks=223`, hash mismatch `5ace6788...`; reject |
| merge-level io_uring buffered, QD16, 4 MiB | `route-c-iouring-qd16-4m-buffered-staticram4g-64x64-20260615-233939` | `decode=0.85 tok/s`, `read_gib_per_s=2.878680`, no fallbacks, hash mismatch `4998484e...`; reject |
| merge-level dedup fix | `route-c-iouring-dedup-qd16-4m-buffered-staticram4g-64x64-20260615-234352` | exit `139`, segfault during model loading around `Loading 4`; reject |
| per-weight io_uring direct, QD16, 4 MiB | `route-c-perweight-iouring-qd16-4m-direct-staticram4g-64x64-20260615-234536` | `decode=1.64 tok/s`, `read_gib_per_s=1.456622`, `iouring_fallbacks=1692`, hash mismatch `314dccdb...`; reject despite apparent speed |
| per-weight io_uring buffered, fresh ring per weight | `route-c-perweight-iouring-qd16-4m-buffered-staticram4g-64x64-20260615-234806` | hash matches control, but `decode=1.31 tok/s`, `read_gib_per_s=0.836744`; reject |
| per-weight io_uring buffered, thread-local persistent ring/bounce | `route-c-perweight-persistent-iouring-qd16-4m-buffered-staticram4g-64x64-20260615-235147` | `decode=1.39 tok/s`, `read_gib_per_s=0.887369`, hash mismatch `06993235...`; reject |

Decision:

- No commit.
- Reverted all uncommitted Route C source changes from:
  - `CMakeLists.txt`
  - `src/devices/disk/diskdevice.cpp`
- Rebuilt `fastllm_tools` successfully after rollback.
- fastLLM git status after rollback:

```text
?? .venv-ftllm/
```

Analysis:

- Generic io_uring submission can raise the measured read-bandwidth counter,
  but the tested variants do not satisfy the correctness and token-rate gates.
- The only hash-correct io_uring variant was slower than `pread`; ring setup,
  bounce-copy overhead, and loss of the existing effective multi-threaded
  behavior erased any benefit.
- Direct IO variants sometimes looked faster but changed output hashes and had
  fallback-heavy behavior, so they are not acceptable.
- Merge-level batching changed cache/hit behavior and output hashes, so Route C
  cannot batch by simply rearranging all miss weights after routing.

Next action:

- Do not recommit this generic io_uring backend.
- The next read-speed route should avoid changing load/cache ordering and
  instead reduce the number of logical reads before introducing async IO:
  - add expert-level coalescing for contiguous pack ranges with output-hash
    checks,
  - or implement Route F prefetch after route decisions while keeping the
    existing `LoadDiskWeightWithRamCache` semantics,
  - or add a small validation mode that compares direct-IO payload hashes
    against `pread` before using direct IO for reported runs.

### 2026-06-16 00:10 CST - Pack range coalescing probe, reverted

Goal:

- Reduce logical read syscall count without changing load/cache ordering.
- Merge only adjacent read tasks where:
  - `fileName` is identical,
  - `fileOffset + bytes == next.fileOffset`,
  - `dst + bytes == next.dst`.

Implementation attempted:

- Added `FASTLLM_DISK_MOE_COALESCE_READS` with default enabled.
- Changed `ReadDiskTasks` to combine adjacent contiguous pack ranges into one
  `ReadDiskRangeBytes` call.
- No change to routing, cache order, active experts, or math.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probes and after rollback.

Short probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-read-coalesce-staticram4g-64x64-20260615-235757`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.60
ttft_s=13.40950
total_time_s=52.8145
read_calls=55552
read_gib=136.975342
read_gib_per_s=1.107768
ram_cache_hits=7778
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Short control for same command shape:

```text
decode_tok_s=1.54
ttft_s=13.96138
read_calls=66006
read_gib=136.975342
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Full single probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-pack-read-coalesce-staticram4g-128x256-single-20260616-000035`
- Shape: `128/256`
- Result:

```text
decode_tok_s=1.67
ttft_s=16.59698
total_time_s=169.5226
read_calls=166700
read_gib=411.150146
read_gib_per_s=1.310014
ram_cache_hits=35150
vram_cache_hits=19198
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
```

Comparison:

- Previous full single relaxed-RAM SOTA:
  - `decode=1.71 tok/s`
  - `ttft=15.84299 s`
  - `read_calls=198126`
  - same output hash
- Coalescing reduced full read calls by about `15.9%`, from `198126` to
  `166700`, but did not improve token rate.
- Full single decode `1.67 tok/s` only matches the previous 3-repeat p50 and
  is below the previous full single `1.71 tok/s`.

Decision:

- No commit.
- Reverted the coalescing change from `src/devices/disk/diskdevice.cpp`.
- Rebuilt `fastllm_tools` successfully after rollback.
- fastLLM git status after rollback:

```text
?? .venv-ftllm/
```

Next action:

- Coalescing adjacent tensor parts is not enough by itself.
- The next useful route should target cache hit rate or earlier useful work:
  - Route F conservative prefetch using the previous token's route/profile,
  - larger relaxed RAM hot set up to the user-approved 8 GiB limit,
  - or a validated direct-IO path with byte-level comparison before enabling
    it for benchmark runs.

### 2026-06-16 00:20 CST - 6-8 GiB relaxed RAM profile cache sweep

Goal:

- Use the user-approved relaxed RAM limit up to 8 GiB to test whether a larger
  static profile cache improves token rate after the 4 GiB cache became the
  current relaxed-RAM SOTA.

Implementation:

- No code changes.
- Same fastLLM commit as current SOTA:
  `36eb1934 perf: add profile-guided RAM expert cache`
- Changed only `FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES`.

Short 8 GiB probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram8g-profile-cache-64x64-20260616-000608`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.45
ttft_s=14.68830
read_gib=113.741455
read_calls=54810
read_gib_per_s=1.194012
ram_cache_bytes=8061714432
ram_cache_hits=11510
ram_cache_misses=1206
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Decision:

- Reject 8 GiB for now. It reduces disk bytes and raises RAM hits, but short
  token rate drops from the 4 GiB short value `1.63 tok/s` to `1.45 tok/s`.

Short 6 GiB probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram6g-profile-cache-64x64-20260616-000840`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.69
ttft_s=13.11768
read_gib=123.540527
read_calls=59532
read_gib_per_s=1.227236
ram_cache_bytes=6136528896
ram_cache_hits=9936
ram_cache_misses=918
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Full single 6 GiB probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram6g-profile-cache-128x256-single-20260616-001115`
- Shape: `128/256`
- Result:

```text
decode_tok_s=1.61
ttft_s=17.04628
read_gib=351.297363
read_calls=169284
read_gib_per_s=1.327007
ram_cache_bytes=6430654464
ram_cache_hits=44764
ram_cache_misses=962
vram_cache_hits=19198
output_hash=109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81
```

Comparison to current 4 GiB relaxed-RAM SOTA full single:

```text
4 GiB decode_tok_s=1.71
4 GiB ttft_s=15.84299
4 GiB read_gib=411.150146
4 GiB ram_cache_hits=35150
```

Decision:

- No commit. There are no tracked code changes.
- 6 GiB improves the 64/64 short probe, but full `128/256` regresses token
  rate and TTFT. Larger static RAM cache reduces disk bytes but adds enough
  memory pressure / host-copy overhead to hurt the real full decode path.
- Keep current SOTA at 4 GiB relaxed RAM profile cache.

Next action:

- Do not use 6 GiB or 8 GiB static RAM cache for reported main results.
- A better RAM route needs admission/eviction or compression, not just a larger
  static hot set.
- Next implementation route should be Route F prefetch or a validated direct IO
  path that proves byte-identical payloads before measuring token rate.

### 2026-06-16 00:25 CST - Decode shared RAM cache probe, reverted

Goal:

- Avoid repeated CPU memcpy on RAM-cache hits during decode by returning the
  cached RAM `Data` pointer directly when the current call is not expected to
  run CUDA prefill preparation.
- Keep prefill / CUDA-preparation paths on the existing clone behavior to avoid
  in-place pack/reorder mutation of cached entries.

Implementation attempted:

- Added `FASTLLM_DISK_MOE_RAM_CACHE_SHARE_DECODE=1`.
- Added `ram_cache_shared_hits` telemetry.
- Added per-index shared RAM hit flags so `releaseOwnedWeights` would not
  delete shared cached pointers.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probe and after rollback.

Short probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-ramcache-share-decode-4g-64x64-20260616-001935`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.42
ttft_s=13.83316
read_gib=157.333008
read_calls=75816
read_gib_per_s=0.948232
ram_cache_hits=5532
ram_cache_shared_hits=5520
ram_cache_misses=604
vram_cache_hits=3088
output_hash=09caa7f5a22935614a552c2e210fc04bb8316edb46147bb8c2b209e50ad30794
```

Comparison:

- 4 GiB RAM-cache short baseline:
  - `decode=1.63 tok/s`
  - output hash `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted all shared RAM cache changes from `src/devices/disk/diskdevice.cpp`.
- Rebuilt `fastllm_tools` successfully after rollback.
- fastLLM git status after rollback:

```text
?? .venv-ftllm/
```

Analysis:

- Decode-time direct sharing of RAM cached `Data` is not semantics-preserving in
  the current disk-MoE path. The output hash changes and token rate regresses.
- Even when CUDA prefill preparation is avoided, downstream decode code likely
  expects per-call temporary weight ownership or mutates transient weight state.

Next action:

- Do not use shared RAM `Data` pointers.
- Continue with routes that preserve per-call temporary `Data` ownership:
  - compressed RAM cache to reduce memory pressure while still cloning,
  - route-history-aware eviction under 2 GiB/4 GiB,
  - or Route F prefetch that fills the existing RAM cache without changing the
    object ownership model.

### 2026-06-16 00:34 CST - Route E VRAM cache capacity probes, rejected

Goal:

- Test whether increasing the routing-profile VRAM hot expert cache above the
  current 1 GiB SOTA improves active expert hits enough to raise decode token
  rate.
- Keep RAM profile cache at the current relaxed 4 GiB SOTA configuration.

Environment:

```text
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/fastllm/tools/profiles/deepseek_v4_flash_route_profile_128x64.tsv
FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES=4294967296
FASTLLM_DISK_MOE_LOAD_THREADS=8
```

Probes:

- 2 GiB VRAM cache, `gpu_mem_ratio=0.98`
  - Run dir:
    `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache2g-profile128x64-ram4g-64x64-20260616-002323`
  - Result: rejected, process aborted with CUDA OOM / illegal-address cascade.
- 1.5 GiB VRAM cache, `gpu_mem_ratio=0.98`
  - Run dir:
    `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1536m-profile128x64-ram4g-64x64-20260616-002448`
  - Result: rejected, process aborted with CUDA OOM / illegal-address cascade.
- 1.5 GiB VRAM cache, `gpu_mem_ratio=0.95`
  - Run dir:
    `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1536m-gpumem095-ram4g-64x64-20260616-002631`
  - Result: rejected, still aborted with CUDA OOM / illegal-address cascade.
- 1.25 GiB VRAM cache, `gpu_mem_ratio=0.95`
  - Run dir:
    `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-vramcache1280m-gpumem095-ram4g-64x64-20260616-002757`
  - Result:

```text
decode_tok_s=1.51
ttft_s=13.00759
prefill_tok_s=4.92
batch_total_tok_s=1.17
read_gib=136.103760
read_calls=65586
read_gib_per_s=1.208337
ram_cache_bytes=4171235328
ram_cache_hits=7424
ram_cache_misses=624
vram_cache_bytes=1336934400
vram_cache_hits=4606
vram_cache_misses=214
vram_cache_stores=200
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Comparison to current short SOTA:

```text
1 GiB VRAM + 4 GiB RAM decode_tok_s=1.63
1 GiB VRAM + 4 GiB RAM ttft_s=13.08105
1 GiB VRAM + 4 GiB RAM read_gib=136.975342
1 GiB VRAM + 4 GiB RAM vram_cache_hits=4112
1 GiB VRAM + 4 GiB RAM ram_cache_hits=7778
```

Decision:

- No commit. There were no tracked source changes.
- Static VRAM cache sizes above 1 GiB are not currently useful:
  - 1.5-2 GiB is unsafe on the RTX 5090 with this model placement.
  - 1.25 GiB preserves output hash but lowers decode token rate despite higher
    VRAM hits, likely because the lower `gpu_mem_ratio` and reduced allocator
    headroom hurt the compute/H2D path more than the extra hot experts help.
- Keep current SOTA at 1 GiB VRAM profile cache + relaxed 4 GiB RAM profile
  cache.

Next action:

- Do not increase the static VRAM hot cache for reported main results.
- A future Route E retry should be reserve-aware and automatically back off
  based on actual free VRAM, rather than setting a larger static byte cap.
- Continue with read-rate work that preserves output semantics, especially
  Route F decode-time prefetch.

### 2026-06-16 00:43 CST - Route F same-layer `posix_fadvise` prefetch probe, reverted

Goal:

- Test a semantics-preserving read hint path before replacing the actual disk
  read implementation.
- After `loadIndices` are known, issue `posix_fadvise(POSIX_FADV_WILLNEED)` for
  disk weights that are not already present in the RAM profile cache.
- Keep actual payload reads on the existing `pread` path.

Implementation attempted:

- Added opt-in `FASTLLM_DISK_MOE_PREFETCH=1`.
- Added RAM-cache `Contains` check to avoid prefetching weights that would be
  served from RAM.
- Added prefetch telemetry:
  - `prefetch_calls`
  - `prefetch_bytes`
  - `prefetch_gib_per_s`
  - `prefetch_skipped_ram`

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probe.

Probe A:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-fadvise-prefetch-ram4g-vram1g-64x64-20260616-003437`
- Shape: `64/64`
- `gpu_mem_ratio=0.98`
- Result:

```text
decode_tok_s=1.57
ttft_s=13.32647
read_gib=136.975342
read_calls=66006
read_gib_per_s=1.320470
ram_cache_hits=7778
ram_cache_misses=626
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
prefetch_calls=66006
prefetch_gib=136.975342
prefetch_gib_per_s=39.024840
prefetch_skipped_ram=7778
```

Probe B:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-fadvise-prefetch-ram4g-vram1g-gpumem09-64x64-20260616-003739`
- Shape: `64/64`
- `gpu_mem_ratio=0.9`
- Result:

```text
decode_tok_s=1.36
ttft_s=14.29318
read_gib=171.340576
read_calls=82566
read_gib_per_s=1.365257
ram_cache_hits=3638
ram_cache_misses=600
vram_cache_hits=2732
output_hash=29874861985616e49385b2696407772ddbb33b2d79df3cb80d45238efdc8288b
prefetch_calls=82566
prefetch_gib=171.340576
prefetch_gib_per_s=47.681202
prefetch_skipped_ram=3638
```

Comparison:

- Current short SOTA:
  - Run dir:
    `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-staticram4g-profile-cache-64x64-20260615-224940`
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_gib_per_s=1.178972`
  - `output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the fadvise prefetch source changes from
  `src/devices/disk/diskdevice.cpp`.
- Same-layer `posix_fadvise` can raise measured `read_gib_per_s`, but it does
  not improve end-to-end decode token rate. It also produced a hash mismatch in
  the `gpu_mem_ratio=0.9` comparison run, so it is not a valid SOTA route.

Analysis:

- The hint is issued too late to overlap meaningful compute: it happens after
  routing and immediately before the blocking `pread` workers.
- It adds syscall overhead and can perturb page-cache pressure without reducing
  the number of blocking payload reads.
- The next read-rate route should affect the real blocking read schedule rather
  than only advising the kernel.

Next action:

- Try a byte-identical read scheduling route:
  - group disk load work by pack file offset,
  - assign contiguous chunks to worker threads instead of strided indices,
  - keep `ReadDiskRangeBytes` / `pread` unchanged,
  - require output hash equality before measuring repeats.

### 2026-06-16 00:49 CST - Route F offset-chunk load scheduling probe, reverted

Goal:

- Improve effective SSD read behavior without changing the byte read path.
- Sort each merge's disk load list by pack file and first offset, then assign
  contiguous chunks of that schedule to worker threads.
- Keep the original `loadIndices` order for later CUDA preparation and VRAM
  cache admission.

Implementation attempted:

- Added opt-in `FASTLLM_DISK_MOE_LOAD_SCHEDULE=offset_chunk`.
- Added `BuildDiskLoadSchedule` to stable-sort only the scheduled load list.
- Changed `LoadDiskWeightsOp` to support contiguous chunk assignment when the
  schedule is enabled.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probe.

Probe 1:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-offset-chunk-ram4g-vram1g-64x64-20260616-004216`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.67
ttft_s=12.61716
read_gib=139.901367
read_calls=67416
read_gib_per_s=1.158233
ram_cache_hits=7298
ram_cache_misses=622
vram_cache_hits=4122
output_hash=1961456d19690e0990b3a2a8d20875feccd1a9b2435baa448c9a069b1dc3e874
```

Probe 2:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-offset-chunk-ram4g-vram1g-64x64-repeat2-20260616-004450`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.45
ttft_s=12.67525
read_gib=136.975342
read_calls=66006
read_gib_per_s=1.165192
ram_cache_hits=7778
ram_cache_misses=626
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Comparison:

- Current short SOTA:
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_gib_per_s=1.178972`
  - `output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the offset-chunk scheduling source changes from
  `src/devices/disk/diskdevice.cpp`.
- The first probe showed a possible TTFT/decode improvement, but the output hash
  changed and the second probe did not reproduce the token-rate gain.
- This route is not a valid SOTA improvement.

Analysis:

- Sorting the load order can perturb which weights enter the RAM cache first
  under the fixed byte budget, because cache stores happen from parallel worker
  completion order.
- The real read bandwidth did not improve; both probes stayed around
  `1.16 GiB/s`, so changing assignment order alone does not move the SSD path
  toward the 8-12 GB/s target.

Next action:

- Avoid pure scheduling-only changes unless they also preserve cache admission
  determinism.
- The next read route should reduce blocking read volume or batch adjacent pack
  ranges while keeping tensor boundaries validated:
  - deterministic RAM-cache admission separate from worker completion order, or
  - pack-range coalescing with full-shape validation and a full benchmark gate.

### 2026-06-16 00:55 CST - Route D priority RAM-cache admission probe, reverted

Goal:

- Test whether cache instability near the RAM byte cap is limiting RAM hit rate.
- Make RAM-cache admission deterministic by profile priority when the byte cap
  is reached.
- Keep this behavior opt-in with
  `FASTLLM_DISK_MOE_RAM_CACHE_PRIORITY_EVICT=1`.

Implementation attempted:

- Added profile priority tracking to `DiskMoeRamProfileHotSet`.
- Changed RAM cache entries from bare `Data*` to `{Data*, bytes, priority}`.
- Added optional replacement of lower-priority cached entries when admitting a
  higher-priority weight would otherwise exceed the RAM cache cap.
- Added `ram_cache_evictions` telemetry.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probe.

Probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-priority-evict-ram4g-vram1g-64x64-20260616-005116`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.58
ttft_s=13.07939
read_gib=154.195312
read_calls=74304
read_gib_per_s=1.195563
ram_cache_hits=6004
ram_cache_misses=612
ram_cache_evictions=0
ram_cache_bytes=4091019264
ram_cache_stores=612
vram_cache_hits=3120
output_hash=7d60efb3c1221d690623e975ccdbc69b3865f81b8b59fce36cdad5fd4a0d58b4
```

Comparison:

- Current short SOTA:
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_gib=136.975342`
  - `ram_cache_hits=7778`
  - `output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the priority RAM-cache admission source changes.
- This was not the active bottleneck in the current configuration:
  `ram_cache_evictions=0`, so the new replacement path never triggered.
- The run increased disk read volume and changed output hash, so it fails the
  route gate.

Next action:

- Do not spend more time on RAM admission unless a run shows actual eviction or
  cap pressure.
- Move back to read-volume reduction: implement a safer pack-range coalescing
  variant or an expert-level packed read that validates tensor offsets and keeps
  cache admission order unchanged.

### 2026-06-16 01:00 CST - Route B contiguous-read coalescing probe, reverted

Goal:

- Reduce read syscall count without changing tensor layout.
- Only coalesce adjacent read tasks when both conditions are true:
  - same pack file and contiguous file offsets
  - contiguous destination memory
- Keep the feature opt-in with
  `FASTLLM_DISK_MOE_COALESCE_CONTIGUOUS_READS=1`.

Implementation attempted:

- Added `DiskMoeCoalesceContiguousReadsEnabled`.
- Changed `ReadDiskTasks` to merge only direct contiguous tasks into one
  `ReadDiskRangeBytes` call.
- No temporary scatter buffer, no reordering, no change to `pread` itself.

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probe.

Probe:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-b-contiguous-read-coalesce-ram4g-vram1g-64x64-20260616-005640`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.53
ttft_s=13.77460
read_gib=150.958008
read_calls=61298
read_gib_per_s=1.118352
ram_cache_hits=6480
ram_cache_misses=626
ram_cache_bytes=4184604672
vram_cache_hits=3164
output_hash=c29ebc91c5cc12bb73cb908023a87dadadb8ec6840734f77afe802f847ab0529
```

Comparison:

- Current short SOTA:
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_gib=136.975342`
  - `read_calls=66006`
  - `read_gib_per_s=1.178972`
  - `ram_cache_hits=7778`
  - `output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the contiguous-read coalescing source changes.
- Although `read_calls` fell from `66006` to `61298`, token rate regressed and
  output hash changed.
- The route does not improve actual SSD streaming or end-to-end decode.

Analysis:

- Syscall count is not the limiting factor in this configuration.
- The important metric is still blocking read/load time and cache hit stability;
  this probe reduced calls but increased total read bytes and lowered RAM/VRAM
  hits after routing diverged.

Next action:

- Stop treating syscall reduction alone as a success metric.
- The next viable read path needs to prefetch or cache whole future experts
  before the blocking load point, using route decisions from previous tokens or
  profile-guided next-layer predictions, while preserving output semantics.

### 2026-06-16 01:13 CST - Route F last-route cross-layer prefetch probe, reverted

Goal:

- Hide blocking expert reads by starting prefetch before the next layer reaches
  its disk load point.
- Use the previous token's selected experts for layer `L + 1` as the predictor
  when the current layer `L` starts.
- Keep main `LoadDiskWeightWithRamCache` semantics unchanged.

Implementation attempted:

- Added opt-in `FASTLLM_DISK_MOE_LAST_ROUTE_PREFETCH=1`.
- Added a background worker and per-layer registry of gate/down disk weights.
- Recorded `lastSelected[layer]` after routing.
- At layer `L`, scheduled prefetch for `lastSelected[L + 1]` when layer `L + 1`
  weights were already registered from a prior token.
- Tried two backend modes:
  - actual background `pread` into a temporary buffer
  - `posix_fadvise(POSIX_FADV_WILLNEED)` via
    `FASTLLM_DISK_MOE_LAST_ROUTE_PREFETCH_MODE=fadvise`
- Added telemetry:
  - `route_prefetch_tasks`
  - `route_prefetch_bytes`
  - `route_prefetch_gib_per_s`
  - `route_prefetch_dropped`

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probes.

Probe A: background `pread`, 64 MiB per layer

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-last-route-prefetch-64m-ram4g-vram1g-64x64-20260616-010341`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.45
ttft_s=13.24073
read_gib=136.975342
read_calls=66006
read_gib_per_s=1.067470
ram_cache_hits=7778
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
route_prefetch_tasks=2151
route_prefetch_gib=93.453857
route_prefetch_gib_per_s=2.060937
route_prefetch_dropped=105
```

Probe B: background `pread`, 16 MiB per layer

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-last-route-prefetch-16m-ram4g-vram1g-64x64-20260616-010619`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.41
ttft_s=13.01978
read_gib=172.112549
read_calls=82938
read_gib_per_s=1.125267
ram_cache_hits=4114
vram_cache_hits=2132
output_hash=fc3dbd1731f6d7f39309b84a3a7426a5a6ac76158bf4033cf7a79e80b66b92e0
route_prefetch_tasks=2304
route_prefetch_gib=35.990479
route_prefetch_gib_per_s=1.298661
route_prefetch_dropped=0
```

Probe C: `posix_fadvise`, 64 MiB per layer

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-last-route-prefetch-fadvise-64m-ram4g-vram1g-64x64-20260616-010927`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.58
ttft_s=13.36931
read_gib=141.246094
read_calls=68064
read_gib_per_s=1.120869
ram_cache_hits=7290
vram_cache_hits=3914
output_hash=cd51cb1025b6ae8ce5890e0c379a84e47a3359c09473ee435ba723cccdfb9888
route_prefetch_tasks=2270
route_prefetch_gib=102.969238
route_prefetch_gib_per_s=14.411821
route_prefetch_dropped=0
```

Comparison:

- Current short SOTA:
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_gib=136.975342`
  - `read_calls=66006`
  - `read_gib_per_s=1.178972`
  - `ram_cache_hits=7778`
  - `vram_cache_hits=4112`
  - `output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the last-route cross-layer prefetch source changes.
- Background `pread` competes with the main disk loader and lowers token rate.
- `fadvise` is cheaper but still fails to improve token rate and changes output
  hash.

Analysis:

- The predictor is too noisy and the prefetch window is too small to offset the
  added IO pressure.
- The effective read bandwidth remains around `1.07-1.13 GiB/s` on the main
  path, so this did not move toward the `8-12 GB/s` read target.
- Prefetch needs either a more accurate route oracle/profile for future layers
  or a separate cache tier that avoids competing with the blocking read path.

Next action:

- Do not continue with last-route cross-layer prefetch in this form.
- Next viable directions:
  - revisit model-level routing/profile information to predict future experts
    more accurately before scheduling IO;
  - implement a real bounded compressed RAM tier that reduces disk bytes instead
    of adding speculative reads;
  - or redesign the pack format/read path around whole-expert blobs that can be
    cached and copied with less per-weight overhead.

### 2026-06-16 01:19 CST - Route D whole-expert compression feasibility check

Goal:

- Before implementing a compressed RAM cache, verify whether DeepSeek-V4-Flash
  expert payloads are actually compressible enough to justify decompression
  overhead.

Method:

- Sampled 120 routed expert tensors from:
  `/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json`
- Used Python `zlib.compress(data, 1)` as a quick conservative proxy.
- No source changes.

Result:

```text
sample_tensors=120
raw_mib=255.0
zlib1_mib=237.7874
overall_ratio=0.9325
expert_count=9472
expert_payload_mib_each=12.75
expert_payload_total_gib=117.9375
```

Breakdown:

```text
w1.weight mean_ratio=0.9774
w2.weight mean_ratio=0.9769
w3.weight mean_ratio=0.9768
w1.scale  mean_ratio=0.2207
w2.scale  mean_ratio=0.2200
w3.scale  mean_ratio=0.2178
```

Decision:

- Do not implement whole-expert compression as the next main route.
- Whole-expert compression only saves about `6.75%` in this sample. A 4 GiB RAM
  cache would become only about `4.29 GiB` effective, while every RAM hit would
  add decompression overhead.
- This is unlikely to move token rate materially toward `5 tok/s`.

Next action:

- Evaluate a scale-only RAM cache instead:
  - scale tensors are highly compressible/small;
  - scale reads are frequent small reads;
  - caching scale parts separately might reduce read calls and some bytes
    without decompressing the main weight payload.

### 2026-06-16 01:24 CST - Route D scale-only RAM cache probe, reverted

Goal:

- Cache only NVFP4 scale payloads in RAM, because scale tensors compress well
  and are frequent small disk reads.
- Leave main weight payload reads and `Data` ownership unchanged.

Implementation attempted:

- Added opt-in `FASTLLM_DISK_MOE_SCALE_CACHE_BYTES`.
- Added a raw byte cache keyed by `(file, offset, bytes)` for
  `DiskWeightPart::isScalePart`.
- On scale cache hit, copied cached bytes into the existing destination buffer.
- Added telemetry:
  - `scale_cache_hits`
  - `scale_cache_misses`
  - `scale_cache_stores`
  - `scale_cache_bytes`

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probes.

Probe A: 2 GiB scale cache

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-scale-cache2g-ram4g-vram1g-64x64-20260616-011614`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.61
ttft_s=13.41655
read_gib=133.223633
read_calls=46679
read_gib_per_s=1.125563
ram_cache_hits=7634
ram_cache_misses=624
vram_cache_hits=4080
output_hash=fbd63a61c0a2a2ad8120902a83044b49be6607d7549befb5501c3b607a7ab53b
scale_cache_hits=19855
scale_cache_misses=13412
scale_cache_stores=8192
scale_cache_bytes=2147483648
```

Probe B: 4 GiB scale cache

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-scale-cache4g-ram4g-vram1g-64x64-20260616-011852`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.40
ttft_s=13.06644
read_gib=157.371826
read_calls=51180
read_gib_per_s=1.170118
ram_cache_hits=5170
ram_cache_misses=618
vram_cache_hits=2348
output_hash=3962993b2f00df0717ba42cabca20dc675034b3f844498b26095b3a78e22a40c
scale_cache_hits=27942
scale_cache_misses=11619
scale_cache_stores=11619
scale_cache_bytes=3045851136
```

Comparison:

- Current short SOTA:
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_gib=136.975342`
  - `read_calls=66006`
  - `ram_cache_hits=7778`
  - `vram_cache_hits=4112`
  - `output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the scale-only RAM cache source changes.
- 2 GiB scale cache reduced `read_calls` and `read_gib`, but did not improve
  decode token rate and changed output hash.
- 4 GiB scale cache was clearly worse, likely due to extra memory/cache pressure
  and changed RAM/VRAM hit behavior.

Analysis:

- Scale-only caching helps the measured read volume but not the end-to-end
  bottleneck.
- The main payload reads and CPU clone/copy path still dominate. Extra RAM
  structures can also lower whole-weight RAM/VRAM hit stability.

Next action:

- Do not continue with scale-only RAM cache as a SOTA route.
- More promising work needs to reduce main payload transfers or reduce per-hit
  CPU clone/copy overhead without changing semantics:
  - safe object-pool reuse for RAM-cache hits,
  - a whole-expert blob cache with deterministic admission,
  - or model-level changes that keep more complete experts in VRAM/RAM by
    routing value rather than by tensor part.

### 2026-06-16 01:31 CST - Route D temporary CPU buffer pool probe, reverted

Goal:

- Reduce per-hit RAM cache overhead by reusing temporary CPU buffers for cloned
  disk weights.
- Keep object ownership semantics unchanged:
  - RAM cache still returns a private temporary `Data` per request.
  - Each returned temporary still receives a full `memcpy`.
  - Only the `cpuData` allocation is recycled after the temporary is released.

Implementation attempted:

- Added opt-in `FASTLLM_DISK_MOE_TEMP_WEIGHT_BUFFER_POOL=1`.
- Added `FASTLLM_DISK_MOE_TEMP_WEIGHT_BUFFER_POOL_BYTES` with a 512 MiB probe
  limit.
- Added a size-bucketed `uint8_t*` pool.
- Changed `CloneLoadedDiskWeightCpu` to acquire buffers from the pool.
- Changed `releaseOwnedWeights` to detach and recycle `cpuData` only when:
  - the temporary is not fake,
  - `dataDevice == CPU`,
  - `cpuData != nullptr`,
  - `cudaData == nullptr`.
- Added telemetry:
  - `temp_buffer_pool_hits`
  - `temp_buffer_pool_misses`
  - `temp_buffer_pool_recycles`
  - `temp_buffer_pool_bytes`

Build:

```text
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma --target fastllm_tools -j 8
```

Result: build passed before probes.

Probe 1:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-temp-buffer-pool512m-ram4g-vram1g-64x64-20260616-012521`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.65
ttft_s=14.47085
read_gib=136.975342
read_calls=66006
read_gib_per_s=1.132772
ram_cache_hits=7778
vram_cache_hits=4112
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
temp_buffer_pool_hits=7886
temp_buffer_pool_misses=518
temp_buffer_pool_recycles=7966
temp_buffer_pool_bytes=534773760
```

Probe 2:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-temp-buffer-pool512m-ram4g-vram1g-64x64-repeat2-20260616-012800`
- Shape: `64/64`
- Result:

```text
decode_tok_s=1.31
ttft_s=13.08820
read_gib=145.018799
read_calls=69882
read_gib_per_s=1.134437
ram_cache_hits=7174
vram_cache_hits=3424
output_hash=308c6b41c542da33e8102fded2840c429e01f95d672b13e812591d3ab5efe432
temp_buffer_pool_hits=7276
temp_buffer_pool_misses=522
temp_buffer_pool_recycles=7356
temp_buffer_pool_bytes=534773760
```

Comparison:

- Current short SOTA:
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_gib=136.975342`
  - `read_calls=66006`
  - `ram_cache_hits=7778`
  - `vram_cache_hits=4112`
  - `output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the temporary CPU buffer pool source changes.
- Probe 1 had a small positive decode result and preserved hash, but the repeat
  regressed badly and changed hash. The improvement is not reproducible.

Analysis:

- Allocation reuse by itself is not enough to stabilize or materially improve
  the current path.
- The route is sensitive to memory layout/cache behavior and can perturb
  downstream routing/output, even though object ownership is preserved.

Next action:

- Do not continue with temporary buffer pooling in this form.
- A safer follow-up would be instrumentation-only timing of:
  - RAM-cache clone `memcpy`,
  - CUDA preparation,
  - actual CUDA MoE compute,
  before attempting another optimization.

## 2026-06-16 - Timing breakdown for current SOTA path

Purpose:

- Diagnose whether the next read-speed route should restart Route C
  `io_uring`, or focus on a narrower bottleneck in the current pack/cache path.

Temporary source change:

- Added instrumentation-only counters in `src/devices/disk/diskdevice.cpp` for:
  - RAM cache clone `memcpy`,
  - CUDA disk-weight preparation,
  - VRAM cache store,
  - CUDA MoE merge,
  - owned-weight release.

Run:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-a-disk-moe-timing-breakdown-64x64-20260616-013340`
- Shape: `64/64`
- Runtime: current SOTA env, 1GiB VRAM profile cache + 4GiB relaxed RAM profile cache.

Result:

```text
decode_tok_s=1.33
ttft_s=13.05826
read_gib=194.910645
read_calls=93924
read_seconds=169.776108
read_gib_per_s=1.148045
load_seconds=170.378570
ram_cache_hits=1930
ram_cache_misses=614
vram_cache_hits=654
vram_cache_misses=174
output_hash=e41c657e23e34901edc642961c27638786ef1ef28602120451ff401f5ceb3fc9
ram_clone_seconds=4.985290
cuda_prepare_seconds=3.327745
vram_store_seconds=0.112389
cuda_moe_seconds=3.649870
release_seconds=0.076419
```

Analysis:

- The run is diagnostic only and did not improve SOTA.
- `read_seconds` is almost the entire `load_seconds`, while RAM clone,
  CUDA preparation, VRAM store, CUDA merge, and release together account for
  only about 12.15 seconds.
- This confirms the current limiting factor is still effective expert read
  throughput, not CPU allocation/free or CUDA merge.
- The measured `read_gib_per_s=1.15` is far below the machine fio capability
  previously reported by the user, so the next useful route should reduce
  fragmented synchronous reads and improve queue depth / sequentiality.

Decision:

- No commit.
- Revert the instrumentation-only source changes after recording the result.
- Do not restart the old Route C implementation as-is, because that route was
  already reverted for instability/slower results. A new read-speed route should
  be scoped around the measured bottleneck: pack-layout locality, larger
  request coalescing, async prefetch correctness, and queue-depth utilization.

## 2026-06-16 - Route C manifest read microbenchmark

Route:

- Route C: `io_uring / Pipeline Expert Read Backend`

Purpose:

- Measure actual expert-pack read bandwidth independently from model compute.
- Test whether the packed DeepSeek-V4-Flash expert files can reach the user's
  reported fio-class bandwidth when requests are shaped as real manifest
  tensor/expert reads.

Source changes:

- Added a manifest-aware mode to `tools/ssd_read_bench.cpp`.
- Added CMake target `ssd_read_bench`.
- No inference semantics changed by this benchmark tool.

Git status before route runtime experiment:

```text
 M CMakeLists.txt
 M tools/ssd_read_bench.cpp
?? .venv-ftllm/
```

Build command:

```bash
cmake -S /home/wici/lfz/fastllm \
  -B /home/wici/lfz/fastllm/build-ssd-read-bench \
  -DUSE_CUDA=OFF -DUSE_NUMAS=OFF -DPY_API=OFF
cmake --build /home/wici/lfz/fastllm/build-ssd-read-bench \
  --target ssd_read_bench -j 8
```

Run dir:

- `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-manifest-read-bench-20260616-014115`

Commands:

```bash
/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload tensor --backend pread --threads 8 --seconds 10 --direct --open-per-read

/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload tensor --backend pread --threads 8 --seconds 10 --direct

/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload expert --backend pread --threads 8 --seconds 10 --direct

/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload expert --backend iouring --queue-depth 16 --threads 1 --seconds 10 --direct

/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload expert --backend iouring --queue-depth 32 --threads 1 --seconds 10 --direct

/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload tensor --backend iouring --queue-depth 32 --threads 1 --seconds 10 --direct
```

Metrics:

| workload | backend | threads/QD | direct | bandwidth |
| --- | --- | ---: | --- | ---: |
| tensor, open-per-read | pread | 8 threads | on | 9.918 GB/s |
| tensor, persistent fd | pread | 8 threads | on | 9.534 GB/s |
| expert span, persistent fd | pread | 8 threads | on | 10.970 GB/s |
| expert span | io_uring | QD16, 1 thread | on | 12.146 GB/s |
| expert span | io_uring | QD32, 1 thread | on | 10.713 GB/s |
| tensor | io_uring | QD32, 1 thread | on | 9.745 GB/s |

Analysis:

- The packed expert files can reach the 8-12 GB/s first-stage Route C bandwidth
  target when the request stream has enough queue depth.
- The manifest layout is not the hardware bottleneck.
- The current model-path `read_gib_per_s` counter is not directly comparable to
  fio-style wall bandwidth because it accumulates per-read latency across
  worker threads.
- The token-rate bottleneck is more likely that decode waits at each layer for
  the selected experts to be synchronously available, so the SSD is not kept
  busy across layers/tokens.

Decision:

- Keep the benchmark tool as diagnostic work-in-progress.
- Do not commit yet because no decode token-rate improvement has been produced.
- Next runtime experiment should overlap future-layer reads with current-layer
  compute or otherwise preserve queue depth without adding extra host copies.

## 2026-06-16 - Route C naive contiguous expert span load

Route:

- Route C follow-up: runtime expert read coalescing.

Temporary source change:

- Added `FASTLLM_DISK_MOE_COALESCE_EXPERT_READS=1`.
- When both gateup and down for a routed expert missed RAM/VRAM cache and their
  disk parts were contiguous in the pack file, read one full expert span into a
  temporary host buffer, then copied slices into the existing `Data` targets.
- Added temporary stats: `coalesced_read_spans`, `coalesced_read_gib`.

Run:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-coalesced-expert-read-64x64-20260616-014706`
- Shape: `64/64`
- Runtime env: current SOTA cache settings plus
  `FASTLLM_DISK_MOE_COALESCE_EXPERT_READS=1`.

Result:

```text
decode_tok_s=1.17
ttft_s=16.31016
read_gib=136.975342
read_calls=13736
read_gib_per_s=1.138599
load_seconds=371.137298
ram_cache_hits=7778
ram_cache_misses=626
vram_cache_hits=4112
coalesced_read_spans=10454
coalesced_read_gib=130.164551
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Comparison:

- Current short SOTA:
  - `decode_tok_s=1.63`
  - `ttft_s=13.08105`
  - `read_calls=66006`
  - `read_gib=136.975342`
  - hash `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted the `src/devices/disk/diskdevice.cpp` runtime coalescing changes.
- Rebuilt `fastllm_tools` after revert.

Analysis:

- The route preserved output hash but regressed token rate by about 28%.
- It greatly reduced `read_calls`, but `load_seconds` increased from the SOTA
  short-path range to `371.14s`.
- The likely cause is the extra 13MiB temporary buffer allocation and memcpy per
  expert span, plus reduced useful overlap despite fewer syscalls.
- Therefore, full-span coalescing with an intermediate host copy is the wrong
  runtime shape.

Next action:

- Keep Route C focused on queue-depth / overlap rather than naive coalescing.
- A better next attempt is decode-time prefetch of next-layer routed experts
  into the existing RAM cache or a bounded bounce-buffer pool, so actual reads
  happen earlier without changing the final `Data` layout or adding a full-span
  copy on the critical path.

## 2026-06-16 - Route C buffered vs O_DIRECT manifest control

Route:

- Route C diagnostic follow-up.

Purpose:

- Check whether runtime should focus on Direct IO/bounce buffers or whether
  buffered reads are already sufficient when isolated from model compute.

Run dir:

- `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-manifest-buffered-read-bench-20260616-015040`

Commands:

```bash
/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload tensor --backend pread --threads 8 --seconds 10

/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload expert --backend pread --threads 8 --seconds 10

/home/wici/lfz/fastllm/build-ssd-read-bench/ssd_read_bench \
  --manifest /home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
  --workload expert --backend iouring --queue-depth 16 --threads 1 --seconds 10
```

Metrics:

| workload | backend | direct | bandwidth |
| --- | --- | --- | ---: |
| tensor | pread | off | 7.301 GB/s |
| expert span | pread | off | 7.236 GB/s |
| expert span | io_uring | off | 3.893 GB/s |

Comparison with O_DIRECT:

- tensor pread direct: `9.918 GB/s`
- expert pread direct: `10.970 GB/s`
- expert io_uring direct QD16: `12.146 GB/s`

Decision:

- No commit.
- Direct IO remains a plausible Route C backend requirement.
- Buffered `io_uring` is not attractive for this workload.

Analysis:

- Buffered pread is still much faster than the observed model-path progress, so
  page cache alone is not the full bottleneck.
- Direct IO gives a real isolated-read uplift, but runtime token speed needs
  overlap/queue-depth preservation; simply replacing individual reads with a
  slower extra-copy path is not enough.

## 2026-06-16 - Route C Direct IO runtime SOTA

Route:

- Route C: `io_uring / Pipeline Expert Read Backend`
- Implemented first successful runtime read-speed subroute: direct aligned
  expert reads with a thread-local bounce buffer.

Commit:

- Repository: `/home/wici/lfz/fastllm`
- Commit: `20fe3dc38fa3f9f386998fb3a9a6569145e4e8ed`
- Subject: `perf: add direct disk moe reads`

Changed behavior:

- Added `FASTLLM_DISK_MOE_DIRECT_READS=1`.
- When enabled, disk-MoE reads use an `O_DIRECT` fd for 4KiB-aligned file
  offsets and byte sizes.
- Direct reads go through a thread-local 4KiB-aligned bounce buffer, then copy
  into the existing `Data` target.
- Unaligned reads automatically fall back to the existing buffered `pread`
  path.
- Added telemetry:
  - `direct_read_calls`
  - `direct_read_gib`
  - `direct_read_fallbacks`
- Added manifest-aware `ssd_read_bench` target for Route C diagnostics.
- Default behavior is unchanged unless `FASTLLM_DISK_MOE_DIRECT_READS=1` is set.

Git status before commit:

```text
 M CMakeLists.txt
 M src/devices/disk/diskdevice.cpp
 M tools/ssd_read_bench.cpp
?? .venv-ftllm/
```

Build verification:

```bash
cmake -S /home/wici/lfz/fastllm \
  -B /home/wici/lfz/fastllm/build-ssd-read-bench \
  -DUSE_CUDA=OFF -DUSE_NUMAS=OFF -DPY_API=OFF
cmake --build /home/wici/lfz/fastllm/build-ssd-read-bench \
  --target ssd_read_bench -j 8

CPATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/include \
LIBRARY_PATH=/home/wici/lfz/fastllm/build-route-a-nonuma/nccl-lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma \
  --target fastllm_tools -j 8
```

Build result:

- `ssd_read_bench`: passed.
- `fastllm_tools`: passed.

Short benchmark command shape:

```bash
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
FASTLLM_DISK_MOE_STATS=1 \
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/fastllm/tools/profiles/deepseek_v4_flash_route_profile_128x64.tsv \
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824 \
FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES=4294967296 \
FASTLLM_DISK_MOE_DIRECT_READS=1 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm bench /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low --device cuda --moe_device "{'cuda':16,'disk':84}" --moe_device_layers -1 \
  --cuda_shared_expert true --gpu_mem_ratio 0.98 --cuda_slab 256 \
  --input_tokens 64 --output_tokens 64 --batch 1 --warmup 0 --temperature 0 --threads 8
```

Short repeat metrics:

| run | decode tok/s | TTFT s | output hash | log |
| --- | ---: | ---: | --- | --- |
| initial | 1.74 | 12.30877 | `cd51cb1025b6ae8ce5890e0c379a84e47a3359c09473ee435ba723cccdfb9888` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-64x64-20260616-015330/bench.log` |
| repeat2 | 1.74 | 12.38464 | `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-64x64-repeat2-20260616-015554/bench.log` |
| repeat3 | 1.80 | 11.94044 | `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-64x64-repeat3-20260616-015730/bench.log` |

Short comparison:

- Previous short SOTA: `1.63 tok/s`, TTFT `13.08105s`.
- Direct-read short repeat p50: `1.74 tok/s`.
- Direct-read short repeat worst: `1.74 tok/s`.

Full benchmark command shape:

```bash
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
FASTLLM_DISK_MOE_STATS=1 \
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/fastllm/tools/profiles/deepseek_v4_flash_route_profile_128x64.tsv \
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824 \
FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES=4294967296 \
FASTLLM_DISK_MOE_DIRECT_READS=1 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm bench /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low --device cuda --moe_device "{'cuda':16,'disk':84}" --moe_device_layers -1 \
  --cuda_shared_expert true --gpu_mem_ratio 0.98 --cuda_slab 256 \
  --input_tokens 128 --output_tokens 256 --batch 1 --warmup 0 --temperature 0 --threads 8
```

Full repeat metrics:

| run | decode tok/s | TTFT s | read GiB/s counter | direct read GiB | direct fallbacks | output hash | log |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| single | 1.83 | 15.15125 | 1.427779 | 392.697510 | 8892 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-128x256-single-20260616-015939/bench.log` |
| repeat2 | 1.83 | 14.96627 | 1.441321 | 392.697510 | 8892 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-128x256-repeat2-20260616-020331/bench.log` |
| repeat3 | 1.86 | 14.45664 | 1.525950 | 397.354248 | 6648 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-128x256-repeat3-20260616-020659/bench.log` |

Full comparison:

- Previous full SOTA p50: `1.67 tok/s`.
- Previous full SOTA worst: `1.62 tok/s`.
- Previous full SOTA TTFT p50: `15.88652s`.
- Direct-read full p50: `1.83 tok/s`.
- Direct-read full worst: `1.83 tok/s`.
- Direct-read TTFT p50: `14.96627s`.
- Hash preserved against previous full SOTA:
  `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81`.

Smoke command:

```bash
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
FASTLLM_DISK_MOE_STATS=1 \
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/fastllm/tools/profiles/deepseek_v4_flash_route_profile_128x64.tsv \
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824 \
FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES=4294967296 \
FASTLLM_DISK_MOE_DIRECT_READS=1 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_smoke.py \
  /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low --device cuda --moe_device "{'cuda':16,'disk':84}" --moe_device_layers -1 \
  --cuda_shared_expert true --gpu_mem_ratio 0.98 --cuda_slab 256 --threads 8 \
  --smoke-file /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/smoke_eval.jsonl \
  --out-dir /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-smoke-20260616-021123 \
  --max-new-tokens 64
```

Smoke result:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-bounce-smoke-20260616-021123`
- Result: `20/20`, pass rate `100%`.
- Category pass rates:
  - factual: `4/4`
  - math: `4/4`
  - coding: `4/4`
  - json: `4/4`
  - multilingual: `4/4`
- Baseline no-loss smoke reference: `95%`.
- Gate floor: `>=80%`.

Gate decision:

- Decode token rate improved reproducibly over previous SOTA:
  - p50 `1.67 -> 1.83 tok/s`
  - worst `1.62 -> 1.83 tok/s`
- TTFT gate passed:
  - p50 `14.96627s <= 19.98s`
  - also better than previous SOTA p50 `15.88652s`
- Smoke gate passed:
  - `100% > 50%`
  - no drop from no-loss baseline; `100% >= 80%`
- Active expert count and routing semantics preserved.
- No top-k reduction.
- No active expert-count reduction.
- No prompt-specific oracle routing.
- CPU RAM expert-tier config unchanged from current SOTA runtime line.

Decision:

- Commit created because improvement is reproducible and gates pass.
- Push commit to `private`.
- Push result: `36eb1934..20fe3dc3  master -> master` on
  `git@github.com:L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git`.

Next action:

- Continue toward 5 tok/s. Direct IO is a real SOTA improvement but still far
  from the target. The remaining major gap is queue-depth preservation across
  decode layers/tokens, not raw SSD capability.

## 2026-06-16 - Direct IO stats-off control

Purpose:

- Check whether the remaining direct-read runtime bottleneck is telemetry
  atomic-counter overhead.

Run:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-stats-off-64x64-20260616-101649`
- Shape: `64/64`
- Runtime: Direct IO enabled, disk-MoE stats disabled.

Result:

```text
decode_tok_s=1.81
ttft_s=11.78030
output_hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d
```

Comparison:

- Direct IO stats-on short repeats:
  - `1.74`, `1.74`, `1.80 tok/s`
- Disabling stats does not produce a clear additional gain beyond run-to-run
  variance.

Decision:

- No commit.
- Do not spend the next route on telemetry aggregation.
- Continue with queue-depth / prefetch work.

## 2026-06-16 - Route F shared-expert overlap prefetch attempt

Route:

- Route F: Decode-time prefetch pipeline.

Purpose:

- Use a conservative real overlap window:
  - routing is already known after `BuildMoERoutingData`,
  - CUDA shared expert computation happens before disk routed expert merge,
  - try to prefetch routed disk experts into RAM cache while shared expert
    compute runs.

Temporary source change:

- Added default-off `FASTLLM_DISK_MOE_PREFETCH_SHARED_OVERLAP=1`.
- Added disk prefetch start/wait API.
- Started prefetch before shared expert compute and waited before `MergeMOE`.
- Prefetch used `LoadDiskWeightWithRamCache`, then deleted the returned
  temporary `Data*`, leaving only the RAM-cache copy.

Run:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-shared-overlap-prefetch-64x64-20260616-102253`
- Shape: `64/64`
- Runtime: Direct IO SOTA env plus
  `FASTLLM_DISK_MOE_PREFETCH_SHARED_OVERLAP=1`.

Result:

```text
decode_tok_s=1.01
ttft_s=16.37130
read_gib=375.925781
read_calls=181152
read_gib_per_s=1.373228
load_seconds=274.313287
ram_cache_hits=5378
ram_cache_misses=612
vram_cache_hits=994
direct_read_calls=172233
direct_read_gib=357.331055
direct_read_fallbacks=8919
prefetch_batches=2405
prefetch_weights=32864
prefetch_wait_seconds=26.586260
output_hash=e34f54625b94c5e3485bb217909388b12e76b365b7edcf8e7a9921553d91b832
```

Comparison:

- Direct IO short SOTA:
  - `decode_tok_s=1.74-1.80`
  - `read_gib=136.975342`
  - `read_calls=66006`
  - hash `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`

Decision:

- No commit.
- Reverted all shared-overlap prefetch source changes.
- Rebuilt `fastllm_tools` after revert; build passed.

Analysis:

- The route regressed badly and changed output hash.
- Prefetch caused about `2.74x` more read bytes than the short SOTA and sharply
  reduced VRAM cache hits.
- The overlap window is real, but this implementation did not coordinate with
  existing VRAM/RAM cache ownership and duplicated work.
- A future prefetch route must be request-aware:
  - skip weights that will be served by VRAM,
  - avoid cloning RAM-cache hits only to delete them,
  - track in-flight cache fills so `MergeMOE` waits on the same load instead of
    issuing duplicate loads,
  - preserve deterministic cache interaction before any full/smoke gate.

Next action:

- Keep Direct IO as the current committed SOTA.
- If continuing Route F, implement an in-flight RAM-cache fill table keyed by
  weight name before attempting overlap again.

## 2026-06-16 - Route F in-flight RAM-cache prefetch attempt

Route:

- Route F: Decode-time prefetch pipeline.

Purpose:

- Fix the duplicated-read failure from the previous shared-overlap prefetch
  attempt by adding a RAM-cache in-flight fill table keyed by weight name.
- Let prefetch and `MergeMOE` share the same cache fill when they race for a
  hot RAM-cache weight.
- Skip prefetch for weights already served by RAM cache or VRAM cache.

Temporary source change:

- Added default-off `FASTLLM_DISK_MOE_PREFETCH_SHARED_OVERLAP=1`.
- Added disk prefetch start/wait API.
- Added RAM-cache `CanCache` / `Contains` helpers.
- Added in-flight cache fill tracking with condition-variable waiters.
- Started prefetch only during the CUDA shared-expert overlap window.

Runs:

- `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-inflight-prefetch-64x64-20260616-103947`
- `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-inflight-prefetch-64x64-repeat2-20260616-104150`
- `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-f-inflight-prefetch-64x64-repeat3-20260616-104327`

Shape:

- `64/64`

Runtime:

- Direct IO SOTA env plus `FASTLLM_DISK_MOE_PREFETCH_SHARED_OVERLAP=1`.

Results:

```text
run1: decode_tok_s=1.81 ttft_s=12.05657 hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d read_gib=137.012695 prefetch_batches=57 prefetch_weights=244 prefetch_wait_s=0.346813
run2: decode_tok_s=1.72 ttft_s=12.50359 hash=0d9c0cc44fac81edc7157a8db33771cc149dba30d144473cd5b3de52a017b32b read_gib=160.607666 prefetch_batches=50 prefetch_weights=230 prefetch_wait_s=0.312281
run3: decode_tok_s=1.83 ttft_s=12.20985 hash=10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d read_gib=137.012695 prefetch_batches=57 prefetch_weights=244 prefetch_wait_s=0.344692
```

Comparison:

- Direct IO short SOTA repeats:
  - `1.74`, `1.74`, `1.80 tok/s`
  - expected hash `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d`
- This route produced one faster run, one slower run, and one matching/faster
  run, but repeat2 changed output hash and increased read bytes to
  `160.607666 GiB`.

Decision:

- No commit.
- Reverted all in-flight prefetch source changes.
- Rebuilt `fastllm_tools` after revert; build passed.

Analysis:

- The in-flight table did not actually remove wait contention:
  `ram_inflight_waits=0` on all runs.
- Prefetch coverage was tiny relative to total selected expert traffic:
  only `230-244` weights over `2405` MOE merge runs.
- The repeat2 hash change means this implementation still perturbed cache or
  scheduling behavior enough to alter output, so it fails the accuracy gate even
  before full-length testing.
- The useful next route should avoid decode-time speculative cache mutation and
  instead improve deterministic read throughput directly:
  - issue each selected expert as one ordered expert-span read into a local
    staging buffer,
  - split the span into tensors after read completion,
  - keep tensor order and cache updates identical to the committed Direct IO
    path,
  - use persistent aligned fds and fixed bounce buffers from the microbench
    path that already reached about `10.970 GB/s` for expert spans.

Next action:

- Keep commit `20fe3dc3` as current SOTA.
- Start a Route C deterministic expert-span backend rather than more
  decode-time prefetch.

## 2026-06-16 - Route C aligned direct target reads

Route:

- Route C: `io_uring / Pipeline Expert Read Backend`
- This is a direct-read miss-path improvement, not an `io_uring` route.

Purpose:

- Remove the extra O_DIRECT bounce-buffer memcpy for disk-MoE loaded weights.
- Keep the existing deterministic per-weight load order, RAM cache behavior,
  VRAM cache behavior, and MergeMOE ownership model unchanged.

Commit:

- Repository: `/home/wici/lfz/fastllm`
- Commit: `e08fb65e45a263c48af54d540b9260a2d7ca424c`
- Subject: `perf: add aligned disk moe direct reads`
- Pushed to: `private/master`

Changed behavior:

- Added `Data::AllocateCpuAligned` and aligned CPU-data ownership tracking.
- Added `FASTLLM_DISK_MOE_ALIGNED_CPU_WEIGHTS=1`.
- When both `FASTLLM_DISK_MOE_DIRECT_READS=1` and
  `FASTLLM_DISK_MOE_ALIGNED_CPU_WEIGHTS=1` are set, loaded disk-MoE weights use
  4 KiB aligned CPU buffers.
- O_DIRECT reads now read directly into the final tensor buffer when the target
  address is 4 KiB aligned.
- Added telemetry:
  - `direct_read_inplace_calls`
  - `direct_read_inplace_gib`
- Default behavior is unchanged unless the new env var is enabled.

Build verification:

```bash
CPATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/include \
LIBRARY_PATH=/home/wici/lfz/fastllm/build-route-a-nonuma/nccl-lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib \
cmake --build /home/wici/lfz/fastllm/build-route-a-nonuma \
  --target fastllm_tools -j 8
```

Build result:

- `fastllm_tools`: passed.

Benchmark env delta from previous Direct IO SOTA:

```bash
FASTLLM_DISK_MOE_ALIGNED_CPU_WEIGHTS=1
```

Full benchmark command shape:

```bash
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
FASTLLM_DISK_MOE_STATS=1 \
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/fastllm/tools/profiles/deepseek_v4_flash_route_profile_128x64.tsv \
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824 \
FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES=4294967296 \
FASTLLM_DISK_MOE_DIRECT_READS=1 \
FASTLLM_DISK_MOE_ALIGNED_CPU_WEIGHTS=1 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm bench /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low --device cuda --moe_device "{'cuda':16,'disk':84}" --moe_device_layers -1 \
  --cuda_shared_expert true --gpu_mem_ratio 0.98 --cuda_slab 256 \
  --input_tokens 128 --output_tokens 256 --batch 1 --warmup 0 --temperature 0 --threads 8
```

Short 64/64 repeat metrics:

| run | decode tok/s | TTFT s | read GiB | read GiB/s counter | direct inplace GiB | output hash | log |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| single | 2.02 | 11.26791 | 136.975342 | 1.723952 | 130.025391 | `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-64x64-20260616-105343/bench.log` |
| repeat2 | 1.93 | 10.91395 | 169.672119 | 1.775333 | 160.433350 | `3e8c4684c5c7255d50568011e242e26a42c9f20d2d6674e783b7ae3c52da1d1b` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-64x64-repeat2-20260616-105557/bench.log` |
| repeat3 | 2.02 | 10.79912 | 146.301270 | 1.880160 | 140.088135 | `281d4a8bdc6310b15f29ab4c7abbb1ae123d045adca5d5ebb4685b2b86f1a240` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-64x64-repeat3-20260616-105729/bench.log` |

Short comparison:

- Previous Direct IO short p50: `1.74 tok/s`.
- New short p50: `2.02 tok/s`.
- New short worst: `1.93 tok/s`.
- Short output hashes still vary in some runs, so the SOTA claim is based on
  full-length repeats plus smoke.

Full 128/256 repeat metrics:

| run | decode tok/s | TTFT s | read GiB | read GiB/s counter | load seconds | direct inplace GiB | output hash | log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| single | 2.09 | 13.21888 | 411.150146 | 1.974610 | 208.900133 | 392.697510 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-128x256-single-20260616-105931/bench.log` |
| repeat2 | 2.13 | 17.93766 | 411.150146 | 1.901524 | 216.943234 | 392.697510 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-128x256-repeat2-20260616-110306/bench.log` |
| repeat3 | 2.13 | 13.33493 | 411.150146 | 2.060583 | 200.189195 | 397.354248 | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-128x256-repeat3-20260616-110610/bench.log` |

Full comparison:

- Previous Direct IO full p50: `1.83 tok/s`.
- Previous Direct IO full worst: `1.83 tok/s`.
- Previous Direct IO TTFT p50: `14.96627s`.
- New full p50: `2.13 tok/s`.
- New full worst: `2.09 tok/s`.
- New full TTFT p50: `13.33493s`.
- Full output hash is stable and matches previous Direct IO full SOTA.
- TTFT remains within the hard gate `<=19.98s`.

Smoke command:

```bash
PYTHONPATH=/home/wici/lfz/fastllm/build-route-a-nonuma/tools \
LD_LIBRARY_PATH=/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/nccl/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/wici/lfz/fastllm/.venv-ftllm/lib/python3.12/site-packages/nvidia/cublas/lib \
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
FASTLLM_DISK_MOE_STATS=1 \
FASTLLM_DISK_MOE_PACK_MANIFEST=/home/wici/lfz/fastllm_cache/deepseek-v4-flash-disk-pack-full/manifest.json \
FASTLLM_DISK_MOE_VRAM_CACHE_PROFILE=/home/wici/lfz/fastllm/tools/profiles/deepseek_v4_flash_route_profile_128x64.tsv \
FASTLLM_DISK_MOE_VRAM_CACHE_BYTES=1073741824 \
FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES=4294967296 \
FASTLLM_DISK_MOE_DIRECT_READS=1 \
FASTLLM_DISK_MOE_ALIGNED_CPU_WEIGHTS=1 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/python /home/wici/lfz/fastllm/tools/scripts/deepseek_v4_flash_smoke.py \
  /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low --device cuda --moe_device "{'cuda':16,'disk':84}" --moe_device_layers -1 \
  --cuda_shared_expert true --gpu_mem_ratio 0.98 --cuda_slab 256 --threads 8 \
  --smoke-file /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/smoke_eval.jsonl \
  --out-dir /home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-smoke-20260616-111016 \
  --max-new-tokens 64
```

Smoke result:

- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-smoke-20260616-111016`
- Summary:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-aligned-direct-target-smoke-20260616-111016/smoke_summary.json`
- Result: `19/20`, pass rate `95%`.
- Failed case: `math_004`, missing substring `32`.
- Baseline no-loss smoke reference: `95%`.
- Smoke gate: passed.

Rejected side probe:

- `FASTLLM_DISK_MOE_LOAD_THREADS=12` with `--threads 12` aborted during model
  loading with `malloc(): unsorted double linked list corrupted`.
- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-direct-read-threads12-64x64-20260616-104740`
- Decision: do not use larger ftllm thread count as an optimization route.

Decision:

- Commit and push accepted as new SOTA.
- Current SOTA commit: `e08fb65e45a263c48af54d540b9260a2d7ca424c`.

Analysis:

- The previous Direct IO path used a thread-local aligned bounce buffer for all
  O_DIRECT reads, then copied into the final `Data` storage.
- With aligned disk-loaded CPU buffers, all direct reads in the full benchmark
  become inplace:
  - single/repeat2: `392.697510 GiB` inplace
  - repeat3: `397.354248 GiB` inplace
- This removes one large memcpy per direct-read payload while preserving the
  same final tensor layout.
- The full token-rate gain is about `16%` over the previous full SOTA:
  `1.83 -> 2.13 tok/s`.

Next action:

- Keep this aligned direct target path as the new committed SOTA.
- Remaining gap to `5 tok/s` is no longer explained by raw SSD fio alone:
  continue by reducing read volume or active expert miss count rather than
  adding more naive read coalescing.

## 2026-06-16 - Route D relaxed RAM capacity probes on aligned direct SOTA

Route:

- Route D: RAM expert cache.

Purpose:

- Re-test the user's allowed RAM relaxation after the aligned direct-read SOTA.
- Check whether larger RAM cache capacity can reduce read volume enough to
  improve token rate over commit `e08fb65e`.

Runtime:

- Current aligned direct SOTA env.
- Shape: `64/64`.
- Only changed `FASTLLM_DISK_MOE_RAM_PROFILE_CACHE_BYTES`.

Runs:

| RAM cap | decode tok/s | TTFT s | read GiB | load seconds | RAM hits | VRAM hits | output hash | log |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 8 GiB | 1.93 | 11.94096 | 113.741455 | 61.431456 | 11510 | 4112 | `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-ram8g-aligned-direct-64x64-20260616-112618/bench.log` |
| 6 GiB | 1.79 | 11.88833 | 187.639160 | 111.345202 | 2998 | 754 | `68f6783bd45e66cd19961613042400f3aa1d11eda27fac07b84e8c3d3429bbfe` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-ram6g-aligned-direct-64x64-20260616-112842/bench.log` |

Comparison:

- Current aligned direct short p50: `2.02 tok/s`.
- Current aligned direct short worst: `1.93 tok/s`.
- 8 GiB reduces disk reads but only matches the short worst and does not improve
  p50.
- 6 GiB is unstable and regresses token rate with changed output hash.

Decision:

- No commit.
- Do not raise the reported RAM cap beyond the current 4 GiB SOTA setting as a
  standalone route.

Analysis:

- 8 GiB confirms that additional RAM hits can reduce read volume, but the extra
  RAM-cache hit cloning/memcpy and altered cache admission pattern erase the
  benefit.
- 6 GiB perturbs RAM/VRAM hit behavior badly enough to increase reads and alter
  the output hash.
- The next RAM route should reduce per-hit copy overhead without sharing mutable
  `Data` objects across CUDA prepare.

## 2026-06-16 - Route D borrowed RAM payload probe rejected

Route:

- Route D: RAM expert cache copy-overhead reduction.

Purpose:

- Test whether RAM-cache hits for eligible down-projection weights can borrow
  cached CPU payloads instead of cloning into each transient `Data`.
- The probe was intentionally limited to down weights because gate/up weights
  are more likely to be reordered or mutated by CUDA preparation.

Run:

- Env: current aligned direct SOTA plus
  `FASTLLM_DISK_MOE_BORROW_RAM_DOWN_WEIGHTS=1`.
- Shape: `64/64`.
- Run dir:
  `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-d-borrow-down-ram4g-aligned-direct-64x64-20260616-113638`

Result:

- Build passed.
- Runtime process was killed with exit code `137` before benchmark metrics were
  produced.
- No valid token-rate, TTFT, smoke, or hash result exists for this route.

Decision:

- Reverted all uncommitted borrowed-payload code changes.
- Rebuilt `fastllm_tools` back to committed SOTA `e08fb65e`.
- No commit.

Analysis:

- The idea reduces one RAM-hit clone, but sharing CPU payload ownership across
  transient `Data` objects is unsafe in the current lifetime model.
- Do not continue this exact route without a bounded owner object or explicit
  reference-counted cache entry model.

## 2026-06-16 - Route C io_uring read-backend probes rejected

Route:

- Route C: io_uring / pipeline expert read backend.

Purpose:

- Raise actual model-path expert read bandwidth toward the plan's `8-12 GB/s`
  first-stage target without changing expert routing or math.
- Test three io_uring integration shapes:
  - one MoE-layer-wide batched queue
  - per-worker/per-weight io_uring reads
  - per-worker batched queues

Runs:

| Probe | Shape | decode tok/s | TTFT s | read GiB/s counter | output hash | log |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| layer-wide batched, invalid first attempt | `64/64` | n/a | n/a | n/a | n/a | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-batched-qd32-64x64-20260616-114755/bench.log` |
| layer-wide batched | `64/64` | `1.59` | `14.42814` | `6.343031` | `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-batched-qd32-64x64-20260616-115024/bench.log` |
| per-worker/per-weight | `64/64` | `2.09` | `10.77308` | `2.055069` | `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-worker-qd32-64x64-20260616-115331/bench.log` |
| per-worker/per-weight | `128/256` | `2.17` | `13.85266` | `2.235384` | `109e541e1ab08e6d9272ebe30ec01550aa510d303e0f9c1f1796e1887baead81` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-worker-qd32-128x256-single-20260616-115543/bench.log` |
| per-worker/per-weight repeat2 | `128/256` | `2.09` | `13.49780` | `2.037452` | same as above | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-worker-qd32-128x256-repeat2-20260616-115917/bench.log` |
| per-worker/per-weight repeat3 | `128/256` | `2.10` | `13.52379` | `2.054627` | same as above | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-worker-qd32-128x256-repeat3-20260616-120309/bench.log` |
| per-worker batched | `64/64` | `1.96` | `11.45291` | `1.796508` | `cd51cb1025b6ae8ce5890e0c379a84e47a3359c09473ee435ba723cccdfb9888` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-c-iouring-workerbatched-qd32-64x64-20260616-120744/bench.log` |

Invalid run note:

- The first `route-c-iouring-batched-qd32-64x64-20260616-114755` attempt
  aborted with CUDA OOM during model loading.
- Its `env.log` showed another concurrent benchmark process using about
  `15.8 GiB` VRAM and heavy RAM/swap, so this run is invalid and not counted
  against the route.

Comparison:

- Current committed full SOTA `128/256`: p50 `2.13 tok/s`, worst `2.09 tok/s`,
  TTFT p50 `13.33493s`.
- Per-worker/per-weight io_uring repeats: `2.17 / 2.09 / 2.10 tok/s`.
- New p50 would be `2.10 tok/s`, so it does not improve over SOTA.
- Layer-wide batched io_uring raised measured read bandwidth to `6.34 GiB/s`,
  but decode dropped to `1.59 tok/s`.

Decision:

- No commit.
- Reverted all Route C io_uring code and rebuilt `fastllm_tools` back to
  committed SOTA `e08fb65e`.

Analysis:

- The raw read backend can improve measured read bandwidth, but token rate is
  still dominated by the full expert-load pipeline, not just disk submission.
- A single layer-wide queue removes the original 8 load-worker parallelism and
  serializes too much CPU/H2D preparation.
- Creating one io_uring ring per loaded weight preserves correctness but adds
  too much per-weight setup overhead; full-run p50 regresses.
- Per-worker batched queues perturb cache/load ordering and short-run output
  hash, so that implementation is not acceptable.

Next route:

- Do not continue naive io_uring integration.
- Focus on reducing expert bytes loaded per token or increasing cache hit rate
  without changing output semantics: routing-profile quality, VRAM hot-set
  selection, or a safer RAM cache ownership model.

## 2026-06-16 - Route E eager expert-level VRAM prewarm probe rejected

Route:

- Route E: quick feasibility probe for expert-level placement.

Purpose:

- Test whether profile-guided `(layer, expert)` residency can help before
  implementing loader-level expert placement.
- The probe kept the existing layer-level base placement unchanged and added a
  default-off `FASTLLM_DISK_MOE_VRAM_CACHE_EAGER_LAYER=1`.
- When enabled, each disk MoE layer prewarmed profile-selected hot experts into
  the existing VRAM expert cache on first layer entry.

Runs:

| Probe | Shape | decode tok/s | TTFT s | read GiB | read GiB/s counter | load seconds | VRAM hits | VRAM misses | VRAM stores | output hash | log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| first attempt | `64/64` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-expert-eager-vram-64x64-20260616-122421/bench_repeat_1.log` |
| valid eager prewarm | `64/64` | `2.03` | `17.10058` | `141.831299` | `1.902656` | `74.795881` | `3840` | `0` | `160` | `10588b3040e9c4662abd09f10dedf158b9f3273c1ddc085b5063aa55b6f2d50d` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-expert-eager-vram-64x64-20260616-122558/bench_repeat_1.log` |

Invalid run note:

- The first attempt exited with return code `-9` during model loading before
  benchmark metrics were emitted.
- It was not counted as a route result.
- A temporary 8 GiB swap file was then created with sudo for the valid probe and
  removed immediately after the run.

Comparison:

- Current aligned direct short p50: `2.02 tok/s`.
- Current aligned direct short worst: `1.93 tok/s`.
- Eager prewarm reached `2.03 tok/s`, so it is not a meaningful improvement.
- TTFT rose to `17.10058s`, much worse than the short SOTA TTFT range around
  `10.8-11.3s` and close to the full-run SOTA p50 `13.33493s`.
- Output hash matched the valid aligned-direct 64/64 reference.

Decision:

- No commit.
- Revert the eager-prewarm implementation.
- Do not continue this exact shape: it only moves cache misses earlier, raises
  TTFT, and does not reduce the steady-state loaded expert working set.

Analysis:

- The probe confirms that the runtime can tolerate profile-guided expert-level
  VRAM residency without changing output for this prompt.
- It does not validate the desired base placement because the original
  layer-level CUDA placement remains in place and the hot expert cache is still
  an extra structure on top of it.
- The next Route E step should implement loader-level expert placement: decide
  disk-vs-CUDA per `(layer, expert)` during weight load using the route profile,
  so the CUDA expert budget replaces part of the layer-level allocation instead
  of adding an eager cache pass.

## 2026-06-16 - Route E loader-level expert placement probe rejected

Route:

- Route E: replace layer-level MoE placement with profile-guided
  `(layer, expert)` placement during model weight loading.

Purpose:

- Test whether base placement can be changed from layer granularity to expert
  granularity, so the CUDA-resident expert budget is selected by routing
  profile rather than by layer ratio.
- The probe used the existing aligned direct-read SOTA as the base and added
  env-gated expert placement:
  - `FASTLLM_DISK_MOE_EXPERT_PLACEMENT_PROFILE`
  - `FASTLLM_DISK_MOE_EXPERT_PLACEMENT_BYTES`
  - `FASTLLM_DISK_MOE_EXPERT_PLACEMENT_REPLACE_BASE`

Runs:

| Probe | Shape | Result | Key log |
| --- | ---: | --- | --- |
| loader placement, 8 GiB | `64/64` | abort `-6` during warmup | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-loader-expert-placement-8g-64x64-20260616-125015/bench_repeat_1.log` |
| loader placement, 1 GiB | `64/64` | abort `-6` during warmup | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-loader-expert-placement-1g-64x64-20260616-125211/bench_repeat_1.log` |
| force DiskMergeMOE dispatch | `64/64` | segfault `-11`; gdb showed CPU fallback path | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-loader-expert-placement-1g-64x64-20260616-125539/bench_repeat_1.log` |
| lower GPU prefill threshold | `64/64` | segfault `-11`; still fell back to CPU | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-loader-expert-placement-1g-gpumin1-64x64-20260616-125754/bench_repeat_1.log` |
| debug first rejection | `8/1` | segfault `-11`; diagnostic showed selected expert had no CUDA data | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-loader-expert-placement-debug-1g-8x1-20260616-135655/bench_repeat_1.log` |
| prepare CPU-resident hot expert for CUDA | `8/1` | abort `-6`; `std::system_error: Resource deadlock avoided` | `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/runs/route-e-loader-expert-placement-resident-1g-8x1-20260616-140001/bench_repeat_1.log` |

Decision:

- No commit.
- Revert the loader-level expert placement code before migration.
- Keep current committed SOTA `e08fb65e` as the reproducible state.

Analysis:

- Returning `"cuda"` from the loader-level placement selector is not enough in
  `--low` mode: a selected hot expert can remain a compact CPU `nvfp4` weight
  with `cudaData == nullptr`.
- Mixed expert-level placement also requires DiskMergeMOE to own all selected
  experts. If any selected CPU-resident hot expert reaches CPU fallback, the
  fallback path is not valid for this mixed CUDA/disk layout.
- Forcing CPU-resident hot experts through the disk-MoE CUDA preparation path
  changed the failure from GPU-path rejection to a runtime
  `std::system_error`, so this implementation shape is not stable enough to
  migrate.

Next route:

- If expert-level placement is resumed on the new machine, use a safer design:
  keep hot experts represented as disk weights plus explicit VRAM cache entries,
  or build a dedicated owned CUDA expert table for DiskMergeMOE, instead of
  changing normal model weight placement to `"cuda"` in `--low` mode.
