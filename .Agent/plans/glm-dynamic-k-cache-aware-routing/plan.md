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

## Findings

Pending.

## Experiments

Pending.

## Results

Pending.
