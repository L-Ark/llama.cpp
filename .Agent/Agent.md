# ik_llama Agent Notes

This directory stores local agent notes about this repository, experiment
practice, and changes. Keep it out of commits unless explicitly requested.

## Required Agent Workflow

Every future instruction for this repository must start by reading this file:

`/root/lfz/ik_llama/.Agent/Agent.md`

This is a hard requirement, not a best-effort note. Before acting on any future
user instruction about this repository, first read this file and explicitly
carry forward these workflow rules.

Before any experiment, implementation attempt, performance run, profiling run,
or non-trivial repository change:

- write a plan first;
- create or update a markdown plan file before running commands;
- record progress in that same markdown file as the work proceeds;
- record final results, failed attempts, exact configs, log paths, and next
  steps in the plan file before giving the final answer.

For every large task, create a new task folder under:

`/root/lfz/ik_llama/.Agent/plans/`

Use a short descriptive folder name, for example:

`/root/lfz/ik_llama/.Agent/plans/5090-same-prompt-profile/`

All plans and progress for that task must live inside that task folder. Do not
scatter progress notes into unrelated docs. If a task continues across turns,
resume from the existing task folder and update its markdown files instead of
starting over.

All progress for experiments, attempts, and large tasks must be recorded in the
relevant plan markdown file as the work proceeds. Do not keep progress only in
chat or terminal output.

## Repository Understanding

`ik_llama` is a llama.cpp-derived repository with Wici/GLM MoE-specific runtime
work. The current active branch is:

`feat/glm51-5090-2gb-vram-standalone`

The branch focuses on making GLM-5.1 IQ3_XXS usable and reproducible on a
5090 32 GB GPU under a strict host RAM cap. The key runtime idea is MoE expert
streaming/tiering:

- model dense layers are mostly GPU-resident;
- MoE experts can be deferred and loaded through `glm51-iq3xxs.expert-pack`;
- VRAM cache stores hot expert rows;
- RAM tier may be disabled for the 2 GB host-RAM line;
- profile CSVs can pin likely-hot experts into VRAM cache at startup.

The model and pack expected by the current local setup are under:

`models/GLM-5.1-UD-IQ3_XXS/`

## Current Branch Changes

The latest pushed commit is:

`66fb66c0 perf: add 5090 prompt profile preload hooks`

It adds experimental hooks for:

- raw prompt startup profile preload via
  `LLAMA_PROMPT_STARTUP_PROFILE_PRELOAD=1`;
- raw prompt TTFT markers via `LLAMA_PROMPT_TTFT_MARKERS=1`;
- pack-backed expert-row preload into VRAM cache when deferred experts have
  `tensor->data == nullptr`;
- preserving TTFT `mark_*` rows even when the per-expert trace cap is reached.

These are env-gated measurement/preload hooks. They should not be described as
default performance wins.

## Metrics To Report

Every meaningful experiment should report:

- `direct_reads`: expert-pack direct reads.
- `VRAM hit`: VRAM expert cache hit rate.
- `RAM hit`: RAM-tier hit rate, or `0.0%` / disabled when RAM tier is off.
- `eval tok/s`: decode/eval token rate.
- `prompt_eval tok/s`: prompt evaluation token rate when available.
- `total_ms`: llama total runtime from stderr timings when available.
- `TTFT`: interactive submit-to-first-token time.
- `time_to_type_s`: startup/preparation time until user can type or submit.
- `first_visible_s`: end-to-end process start to first visible token.
- `accuracy`: scored accuracy, e.g. `1/1`, when using the evaluation dataset.
- `read_failures`: expert-pack read failures, expected to be `0`.
- config details: profile file, VRAM cache MiB, RAM tier MiB, GPU, host RAM
  limit, prompt/sample, token count, seed, and log paths.

## Important Constraints

- Host RAM must be limited to `2G` for the 5090 constrained line unless a test
  explicitly says otherwise.
- Do not count startup preload/model load in TTFT. Report `time_to_type_s`
  separately.
- The 5090 setup has 32 GB VRAM. Do not confuse it with the earlier 5060 Ti
  machine/NVMe notes.
- Use `RAM=0` for the strict 2 GB RAM line unless explicitly testing RAM tier.
- Keep generated experiment logs/profiles outside committed source unless the
  user asks to commit them.
- The current same-prompt profiling result is an upper-bound diagnostic. It is
  not a valid general benchmark because profiling and testing use the same
  prompt.
- Before each future commit, verify that at least one important historical
  configuration remains reproducible. It is not necessary to rerun every old
  configuration every time.

## Practical Commands

Build check:

```bash
cmake --build /root/lfz/ik_llama/build-cuda --target llama-cli -j 8
```

Check tree:

```bash
git -C /root/lfz/ik_llama status --short --branch
```

Check GPU idle before long runs:

```bash
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

Check for leftover runs:

```bash
pgrep -af 'run_hit_rate_baseline|/llama-cli|moe-run.py' || true
```
