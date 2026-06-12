# Commit Experiment Log

This file records what was done before commits and the important experiment
configurations/results that should remain reproducible over time.

Keep this file local unless the user explicitly asks to commit `.Agent/`.

## Reproducibility Rule Before Future Commits

Before each future commit:

- run the build check for `llama-cli`;
- confirm no unexpected dirty tracked files;
- rerun at least one important historical configuration or smoke equivalent;
- report all core metrics listed in `.Agent/Agent.md`;
- do not rerun every historical configuration unless the change touches shared
  runtime behavior or the user asks for full reproduction.

For the first three historical configs below, rerunning one representative
config is enough for a normal commit gate.

## Historical Starting Point: First Three 5090 Configs

Context:

- GPU: RTX 5090, 32 GB VRAM.
- Host RAM constraint: original line was constrained; current future gate should
  use `MemoryMax=2G` unless explicitly stated otherwise.
- Model: GLM-5.1 IQ3_XXS.
- RAM hit is `0.0%` for these rows because RAM tier was disabled/not used.

| config | direct_reads | VRAM hit | RAM hit | eval tok/s | total_ms | TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-opt baseline | 118824 | 6.0% | 0.0% | 1.02 | 105695.33 | 17.84 |
| optimized candidate1 | 91145 | 29.1% | 0.0% | 1.25 | 88138.02 | 21.68 |
| optimized candidate2 | 87729 | 31.8% | 0.0% | 1.01 | 109228.67 | 21.19 |

Interpretation:

- Candidate1 had the strongest decode throughput among the first three
  (`1.25 eval tok/s`) and reduced total runtime, but TTFT regressed.
- Candidate2 had the best VRAM hit rate and lowest direct reads, but decode
  throughput did not improve over baseline.
- These results show the recurring tradeoff: increasing expert cache hit rate
  can improve decode or direct reads, but may regress TTFT/startup/prompt
  latency.

## Commit 932347bf

Commit:

`932347bf feat: internalize glm51 5090 2gb-vram standalone presets`

Purpose:

- Make the 5090 2 GB host-RAM GLM setup more self-contained in `ik_llama`.
- Internalize presets/scripts/docs that were previously external.
- Keep the repository runnable without depending on moved baseline files.

Recommended reproduction gate after this point:

- build `llama-cli`;
- run one smoke GLM config with `MemoryMax=2G`;
- confirm expert-pack read failures are `0`.

## Commit 66fb66c0

Commit:

`66fb66c0 perf: add 5090 prompt profile preload hooks`

Pre-commit checks:

- `cmake --build /root/lfz/ik_llama/build-cuda --target llama-cli -j 8`
  passed.
- Only two tracked files were changed:
  - `examples/main/main.cpp`
  - `ggml/src/ggml-cuda/moe_stream_batch.cu`
- Push succeeded to:
  `origin/feat/glm51-5090-2gb-vram-standalone`

Changes:

- Added raw prompt startup profile preload:
  `LLAMA_PROMPT_STARTUP_PROFILE_PRELOAD=1`.
- Added raw prompt TTFT markers:
  `LLAMA_PROMPT_TTFT_MARKERS=1`.
- Added pack-backed expert-row preload for deferred experts.
- Preserved `mark_*` TTFT trace rows even after normal trace cap.

Key experiment: same-prompt profiling diagnostic.

Dataset/sample:

`/root/lfz/data/glm_resource_eval/processed/smoke_one.jsonl`

Sample:

`ceval_computer_network_0010`

Configuration:

- top8 profile line;
- `VRAM=12288`;
- `RAM=0`;
- `MemoryMax=2G`;
- 4 generated tokens;
- raw prompt TTFT markers enabled.

Generated prompt-specific profile files:

- aggregate route profile:
  `/root/lfz/project/exp/cache_profiles/smoke_ceval_computer_network_0010_prompt_aggregate.route.csv`
- top8 per-layer profile:
  `/root/lfz/project/exp/cache_profiles/smoke_ceval_computer_network_0010_expert_top8.runtime.csv`
- full observed route profile:
  `/root/lfz/project/exp/cache_profiles/smoke_ceval_computer_network_0010_expert_top32.runtime.csv`

Results:

| config | time_to_type_s | interactive_ttft_s | first_visible_s | prompt_eval tok/s | eval tok/s | VRAM hit | direct_reads | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original top8 profile, no startup preload | 16.445 | 65.666 | 82.111 | 0.8376 | 0.3871 | 15.6% | 6378 | 1/1 |
| original top8 profile, startup preload24 | 15.702 | 64.711 | 80.413 | 0.8500 | 0.4597 | 15.6% | 6378 | 1/1 |
| same-prompt top8 profile | 14.460 | 64.917 | 79.377 | 0.8473 | 0.8874 | 65.0% | 3696 | 1/1 |
| same-prompt top8 profile + extra startup preload24 | 17.858 | 64.971 | 82.829 | 0.8466 | 0.8667 | 65.0% | 3696 | 1/1 |
| same-prompt full observed profile | 18.523 | 66.739 | 85.262 | 0.8242 | 1.0544 | 70.3% | 4299 | 1/1 |

Interpretation:

- Same-prompt profiling greatly improves VRAM hit rate and decode token rate.
- The major gain comes from `GGML_MOE_VRAM_PROFILE` startup preload/pinning,
  not from the extra `main.cpp` startup preload hook.
- Extra startup preload24 was confirmed to run, but was redundant once the
  prompt-specific top8 profile had already pinned 1800 profile entries.
- Full observed profile improves decode speed further but regresses
  `time_to_type_s` and `first_visible_s`, because the full observed profile is
  larger than the 12 GB VRAM cache budget.
- This is useful as an upper-bound diagnostic, not as a general benchmark.

Future follow-up:

- Build profiles from a profiling split and test on a disjoint testing split.
- Track TTFT, time-to-type, token rate, hit rates, direct reads, and accuracy
  together.
- Consider making expert-row preload independent from
  `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS`; currently that tensor limit must
  be greater than zero before expert-row startup preload runs.
