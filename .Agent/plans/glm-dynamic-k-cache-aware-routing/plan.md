# GLM Dynamic-k / Cache-aware Routing Plan

## Goal

Implement and evaluate training-free GLM MoE routing reductions that trade a
small/medium amount of quality for faster inference under strict 5090 2 GB
system-RAM constraints.

## Constraints

- Every real model run must use `MemoryMax=2G` and `MemorySwapMax=0` unless a
  run is explicitly marked as a non-target diagnostic.
- Do not enable SSD swap for target GLM runs.
- Keep `.Agent/` local and uncommitted unless explicitly requested.
- Record all progress, configs, logs, and results in this file.
- Report all `.Agent/Agent.md` metrics for meaningful real runs.

## Plan

1. Research dynamic-k, top-p, and cache-aware MoE routing papers/repos.
2. Inspect existing ik_llama routing and cache primitives.
3. Prefer existing `--smart-expert-reduction` as the first no-code experiment.
4. Use trace/simulation before adding new routing code.
5. Run target GLM experiments under strict `MemoryMax=2G MemorySwapMax=0`.
6. Only implement a cache-aware patch if trace/sim/no-code runs show a clear
   path to improvement.

## Progress

- 2026-06-12: Read `/root/lfz/ik_llama/.Agent/Agent.md`.
- 2026-06-12: Confirmed branch `feat/glm51-5090-2gb-vram-standalone` and only
  `.Agent/` is untracked.
- 2026-06-12: Confirmed no residual model process and GPU idle before work.
- 2026-06-12: Created task folder and subfolders.
- 2026-06-12T04:13:45Z: Resumed task after `.Agent/` was committed and
  pushed. Re-read `.Agent/Agent.md` and this plan. Current worktree is clean on
  `feat/glm51-5090-2gb-vram-standalone`.

## Current Execution Checklist

1. Research dynamic-k, top-p routing, cache-aware routing/admission, and MoE
   offload implementations with source links.
2. Inspect ik_llama's existing expert routing, smart expert reduction, and
   VRAM/RAM expert cache primitives.
3. Identify a no-code or minimally invasive experiment that can run under
   `MemoryMax=2G MemorySwapMax=0`, preferably using existing
   `--smart-expert-reduction` or env-gated knobs.
4. Run the smallest meaningful target experiment first. Record exact command,
   logs, metrics, and whether quality is scored.
5. Decide whether an implementation patch is justified. If yes, specify the
   smallest patch surface and a verification path before editing runtime code.

## Findings

### Research Summary

- D2DMoE / dynamic-k: per-token expert count can be reduced based on router
  confidence or cumulative probability. The key design is not fixed top-k, but
  "select enough experts for this token". This maps directly to ik_llama's
  existing `--smart-expert-reduction` path, which calls `ggml_top_k_thresh`.
  Source: https://arxiv.org/abs/2310.04361
- Dynamic_MoE reference repo: `top_p_sampling_batched_all_sequence()` sorts
  route probabilities, accumulates them, and selects experts until a top-p
  threshold is reached. Local copy:
  `.Agent/plans/glm-dynamic-k-cache-aware-routing/research/repos/Dynamic_MoE/`.
  Relevant file:
  `modeling/modeling_moe.py`.
- Mixture of Cache-Conditional Experts: cache-aware expert routing can trade
  small quality loss for large miss-rate/latency gains in memory-constrained
  generation. It reports over 50% cache-miss reduction and on-device speedups
  with negligible to small quality impact on language modeling, MMLU, and
  GSM8K. Source: https://arxiv.org/abs/2412.00099
- MoE-ERAS: Expert Residency Aware Selection factors expert location
  (accelerator-resident vs host) into selection, reporting latency reduction
  on top of LRU caching/quantization. Source:
  https://openreview.net/pdf?id=o43eHjPEMO
- Fate / ExpertFlow direction: cross-layer or adaptive expert prediction can
  guide prefetch. This is a better later-stage direction than blind cache-size
  sweeps, but it requires reliable route traces and async prefetch plumbing.
  Sources: https://arxiv.org/html/2502.12224v2 and
  https://arxiv.org/html/2510.26730v1
- MoE-Infinity reference repo uses LFU-style cache utilities for expert
  storage decisions. Local copy:
  `.Agent/plans/glm-dynamic-k-cache-aware-routing/research/repos/MoE-Infinity/`.
- mixtral-offloading reference repo caches whole experts with LRU per layer
  and separate main/offload storage. This supports the earlier conclusion that
  whole-expert packing/admission is cleaner than independent tensor admission.
  Local copy:
  `.Agent/plans/glm-dynamic-k-cache-aware-routing/research/repos/mixtral-offloading/`.

### ik_llama Code Findings

- Router selection is in `src/llama-build-context.cpp` inside
  `llm_build_context::llm_build_moe_ffn()`.
- Normal routing calls `ggml_top_k(selection_probs, n_expert_used)`.
- `--smart-expert-reduction` sets `cparams.min_experts` and
  `cparams.thresh_experts`; when enabled, routing calls
  `ggml_top_k_thresh(selection_probs, n_expert_used, min_experts, thresh)`.
- CUDA support is in `ggml/src/ggml-cuda/argsort.cu`; selected experts below
  `thresh * max_val` are written as `-1` while preserving at least
  `min_experts`.
- Expert cache/offload logic is already substantial and env-gated in
  `ggml/src/ggml.c`, `ggml/src/ggml-cuda/moe_stream.cu`, and
  `ggml/src/ggml-cuda/moe_stream_batch.cu`.
- Current cache policy is not router-residency-aware: routing is selected
  before the cache miss is known. Therefore cache-aware routing requires a new
  graph/runtime bridge or a fused op that sees both router scores and residency.

## Experiments

### E1: Existing SER Under Strict 2 GB RAM

Purpose: test the no-code dynamic-k path already present in ik_llama before
adding cache-aware runtime code.

Common config:

- runner:
  `.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_local_interactive.py`
- wrapper: `systemd-run --scope --pty --wait --collect -p MemoryMax=2G -p
  MemorySwapMax=0`
- model preset path: `--low-host-ram-profile glm51-2gb-vram12`
- VRAM cache: `GGML_MOE_VRAM_CACHE_MIB=12288`
- RAM tier: `GGML_MOE_RAM_TIER_MIB=0`, `GGML_MOE_RAM_TIER_SKIP=0`
- profile:
  `presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv`
- prompt: `Write a concise explanation of why NVMe latency matters for MoE inference.`
- seed: `42`
- predict tokens: `36` for the first short target test
- metrics to extract after each run: `time_to_type_s`, `interactive_ttft_s`,
  `first_visible_s`, `eval_tokens_per_s`, total ms, direct reads, VRAM hit,
  RAM hit, read failures, and stderr log path.

Runs:

1. `baseline-ser-off`: no `--smart-expert-reduction`.
2. `ser-4-005`: `--smart-expert-reduction 4,0.05`.
3. `ser-2-020`: `--smart-expert-reduction 2,0.2`.
4. `ser-1-080`: `--smart-expert-reduction 1,0.8`.
5. `ser-1-095`: `--smart-expert-reduction 1,0.95`.
6. `ser-1-100`: `--smart-expert-reduction 1,1.0`.
7. `ser-1-110`: `--smart-expert-reduction 1,1.1`.

Progress:

- `baseline-ser-off` first attempt failed before model start because systemd
  249 rejects `systemd-run --scope --pty` with
  `--pty/--pipe is not compatible in timer or --scope mode.` The local runner
  was adjusted to use a transient service with `--same-dir --pty --wait
  --collect`, keeping `MemoryMax=2G` and `MemorySwapMax=0`.
- `baseline-ser-off-rerun`, `ser-4-005`, `ser-2-020`, and `ser-1-080` all ran
  to completion under `MemoryMax=2G MemorySwapMax=0` with `RAM tier=0`.
  However, all measured `direct_reads=28476`, `VRAM hit=7.5%`, and
  `read_failures=0`. This means the tested SER thresholds did not reduce
  actual expert-pack/cache accesses for this GLM prompt/path. The next probe is
  an intentionally extreme `--smart-expert-reduction 1,1.1` run to separate
  "threshold too conservative" from "SER not affecting fused streamed access".
- `ser-1-110` confirmed that the mechanism works: direct reads fell to `4526`,
  prompt eval improved to `1.21 tok/s`, and eval improved to `3.36 tok/s`.
  The output was visibly corrupted (`Ph tray:S...`), so this is not usable.
- `ser-1-100` had the same quality failure pattern as `ser-1-110` and is also
  not usable despite high speed (`eval_tokens_per_s=2.62`).
- `ser-1-095` is the first candidate with measurable speed gain and no obvious
  corruption in the short output prefix. It reduced direct reads from `28476`
  to `22115` and improved eval from `0.41 tok/s` to `0.53 tok/s`. This is only
  a candidate for scored evaluation, not a recommended default.

E1 measured results:

| run | SER | direct_reads | VRAM hit | RAM hit | prompt tok/s | eval tok/s | total_ms | TTFT | time_to_type_s | quality note | stderr log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| baseline-ser-off-rerun | off | 28476 | 7.5% | 0.0% | 0.31 | 0.41 | 144338.34 | unreliable local PTY echo parse | 52.48 | normal short prefix | `/root/.cache/llama.cpp/moe-chat/20260612-042603-350752.stderr.log` |
| ser-4-005 | 4,0.05 | 28476 | 7.5% | 0.0% | 0.29 | 0.42 | 138884.43 | 68.34 | 45.17 | normal short prefix | `/root/.cache/llama.cpp/moe-chat/20260612-042238-348231.stderr.log` |
| ser-2-020 | 2,0.2 | 28476 | 7.5% | 0.0% | 0.29 | 0.42 | 140131.72 | 68.46 | 44.42 | normal short prefix | `/root/.cache/llama.cpp/moe-chat/20260612-042922-352687.stderr.log` |
| ser-1-080 | 1,0.8 | 28476 | 7.5% | 0.0% | 0.30 | 0.43 | 139664.73 | 67.25 | 46.19 | normal short prefix | `/root/.cache/llama.cpp/moe-chat/20260612-043250-354715.stderr.log` |
| ser-1-095 | 1,0.95 | 22115 | 11.1% | 0.0% | 0.35 | 0.53 | 122673.68 | 57.46 | 45.19 | normal short prefix, unscored | `/root/.cache/llama.cpp/moe-chat/20260612-043829-358049.stderr.log` |
| ser-1-100 | 1,1.0 | 4526 | 24.3% | 0.0% | 1.24 | 2.62 | 56254.10 | 16.10 | 44.33 | corrupted output | `/root/.cache/llama.cpp/moe-chat/20260612-044106-359573.stderr.log` |
| ser-1-110 | 1,1.1 | 4526 | 24.3% | 0.0% | 1.21 | 3.36 | 54112.49 | 16.48 | 44.39 | corrupted output | `/root/.cache/llama.cpp/moe-chat/20260612-043611-356651.stderr.log` |

## Results

### First Decision

- Existing `--smart-expert-reduction` is a real dynamic-k mechanism in this
  codebase and can reduce GLM expert traffic.
- The useful threshold range is narrow for this prompt. `<=0.8` does not change
  traffic; `>=1.0` produces large speedups but visibly breaks output quality.
- `SER=1,0.95` is the only observed no-code candidate worth scoring on the
  evaluation dataset. It gives modest speedup:
  - direct reads: `28476 -> 22115` (`-22.3%`);
  - eval rate: `0.41 -> 0.53 tok/s` (`+29.3%`);
  - total time: `144338.34 -> 122673.68 ms` (`-15.0%`).
- Do not implement or enable a cache-aware routing patch yet. The no-code
  experiment already shows the main risk: hard global score thresholds can
  cross from "no effect" to "quality collapse" quickly. A patch should be
  justified by scored accuracy traces, not only by latency.

### Recommended Next Step

Run scored smoke/daily evaluation for:

1. baseline SER off;
2. `--smart-expert-reduction 1,0.95`;
3. optionally `--smart-expert-reduction 1,0.975` as a boundary point.

Required scoring metrics: accuracy, exact output, direct reads, VRAM hit,
RAM hit, prompt/eval tok/s, total ms, TTFT, time-to-type, and read failures.
Only if `SER=1,0.95` preserves accuracy on the small set should it be promoted
to a documented optional speed/quality tradeoff. Cache-aware routing should be
designed after route traces show which dropped/missed experts are low-impact.

### E2: Scored Smoke Accuracy

Purpose: verify whether the only promising no-code candidate (`SER=1,0.95`)
preserves at least the single fastest automatically scored sample.

Dataset:

- `/root/lfz/data/glm_resource_eval/processed/smoke_one.jsonl`
- sample: `ceval_computer_network_0010`
- expected answer: `B`
- scoring: `choice_letter_exact`

Runner:

- `.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py`
- raw `llama-cli -p` completion, not chat mode, matching the earlier scoring
  finding that chat `--conversation` changes prompt semantics.
- `systemd-run --same-dir --wait --collect -p MemoryMax=2G -p MemorySwapMax=0`
- same GLM model, expert pack, top8 profile, `VRAM=12288`, `RAM=0`.

Runs:

1. `smoke-accuracy-baseline`: SER off.
2. `smoke-accuracy-ser-1-095`: `--smart-expert-reduction 1,0.95`.

E2 measured results:

| run | SER | accuracy | output | direct_reads | VRAM hit | RAM hit | prompt tok/s | eval tok/s | total_ms | wall_s | read_failures | summary |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke-accuracy-baseline-v2 | off | 1/1 | `B` | 6378 | 15.6% | 0.0% | 0.58 | 0.49 | 134378.69 | 137.64 | 0 | `.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/accuracy/smoke-accuracy-baseline-v2.summary.json` |
| smoke-accuracy-ser-1-095 | 1,0.95 | 1/1 | `B` | 5172 | 20.2% | 0.0% | 0.64 | 0.69 | 122190.12 | 124.84 | 0 | `.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/accuracy/smoke-accuracy-ser-1-095.summary.json` |

Notes:

- The first direct-completion baseline attempt produced empty captured stdout
  because `systemd-run --wait` did not relay service stdout/stderr to the
  parent `capture_output`. The runner was fixed to redirect stdout/stderr
  inside the service shell to explicit files.
- The raw completion includes prompt echo, so scoring must extract the first
  choice letter after `答案：`. The baseline-v2 score was recomputed from the
  saved stdout after fixing the extractor.
- On this one scored sample, `SER=1,0.95` preserved accuracy and improved all
  measured speed/cache metrics:
  - direct reads: `6378 -> 5172` (`-18.9%`);
  - prompt eval: `0.58 -> 0.64 tok/s` (`+10.3%`);
  - eval: `0.49 -> 0.69 tok/s` (`+40.8%`);
  - total: `134378.69 -> 122190.12 ms` (`-9.1%`).

### Updated Decision After E2

`SER=1,0.95` is now a plausible optional speed/quality tradeoff candidate, but
not ready as a default because it is only scored on one sample. The next
required validation is `smoke.jsonl` or `daily.jsonl` with the same metrics.
`SER>=1.0` should be rejected for now because it visibly corrupts free-form
output despite excellent speed.

### E3: Attempted `smoke.jsonl` Wider Scoring

Purpose: extend E2 from one scored item to the 4-item `smoke.jsonl` set.

Status: incomplete. The baseline run completed two samples, but the Python
wrapper stayed blocked after the second `systemd-run --wait` despite the
per-sample `llama-cli` process having exited and stdout/stderr/timings being
complete. I killed the wrapper and manually summarized the two completed rows.

Partial baseline evidence:

- summary:
  `.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/accuracy/smoke4-baseline-partial2.summary.json`
- rows completed: `2/4`
- accuracy: `2/2`
- direct reads total: `9792`
- read failures: `0`

| sample | expected | output | correct | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ceval_computer_network_0009 | D | D | 1 | 3384 | 13.2% | 0.73 | 0.42 | 148529.83 |
| cmmlu_chinese_civil_service_exam_0098 | D | D | 1 | 6408 | 15.0% | 1.31 | 0.43 | 185952.75 |

Follow-up fix before full smoke/daily:

- Replace `systemd-run --wait` per-sample execution with direct child process
  execution inside one outer `systemd-run` service, or add explicit timeout and
  `systemctl show` polling so the wrapper cannot hang after completed samples.
- Then rerun complete `smoke.jsonl` for baseline and `SER=1,0.95`.

### E4: Full `smoke.jsonl` With One Outer 2 GB Cgroup

Started: 2026-06-12T05:03:32Z.

Plan:

1. Modify `run_smoke_accuracy.py` so it can run as an inner worker without
   wrapping every sample in `systemd-run --wait`.
2. Launch the whole worker once under `systemd-run --same-dir --wait --collect
   -p MemoryMax=2G -p MemorySwapMax=0`.
3. Run full `/root/lfz/data/glm_resource_eval/processed/smoke.jsonl` for:
   - baseline SER off;
   - `--smart-expert-reduction 1,0.95`.
4. Record accuracy and all resource metrics. If `SER=1,0.95` preserves smoke
   accuracy and improves direct reads/token rate, keep it as a candidate for
   `daily.jsonl`; otherwise reject it.

Progress update before committing `.Agent/`:

- Added `--inner-no-systemd` to
  `.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py`
  so a whole dataset can run inside one outer `MemoryMax=2G` cgroup.
- Syntax check passed for the runner:
  `python3 -m py_compile .../run_smoke_accuracy.py`.
- Started `smoke4-baseline-v2` under one outer cgroup, but stopped it before
  completion to avoid blocking the requested `.Agent/` commit. Partial files
  are preserved under
  `.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/accuracy/`.
- The interrupted run had completed the first sample's stdout/stderr and was
  still in the second sample's model load/run path when stopped. Do not treat
  `smoke4-baseline-v2` as a scored result.
- Before commit, removed nested `.git` metadata from cloned research repos so
  `.Agent/` can be committed as a normal source/document snapshot rather than
  as unresolved gitlinks/submodules.
