# Pull Remote New Changes

## Goal

Pull the latest remote changes for
`feat/glm51-5090-2gb-vram-standalone`, merge them with the local checkout
without losing local experiment artifacts, and understand the new changes.

## Constraints

- Work only under `/home/wici/lfz`.
- Do not delete or modify anything outside `/home/wici/lfz`.
- Preserve existing local `.Agent/plans/5090-fast-ssd-first-three-repro/`
  experiment records unless the user explicitly asks otherwise.
- Read `/home/wici/lfz/Agent.md` and repo-local `.Agent/Agent.md` before
  acting.

## Plan

1. Record current branch and worktree state.
2. Fetch remote changes.
3. Inspect incoming commits and changed files.
4. Check whether incoming tracked paths conflict with local untracked
   experiment artifacts.
5. Pull/merge the remote branch using a non-destructive strategy.
6. Review the new code/docs and summarize what changed.
7. Run lightweight validation if appropriate and practical.

## Progress

- 2026-06-12: Read `/home/wici/lfz/Agent.md`.
- 2026-06-12: Read `.Agent/Agent.md`.
- 2026-06-12: Current branch is
  `feat/glm51-5090-2gb-vram-standalone`.
- 2026-06-12: Current worktree has one untracked local experiment/task
  directory: `.Agent/plans/5090-fast-ssd-first-three-repro/`.
- 2026-06-12: Fetched
  `origin/feat/glm51-5090-2gb-vram-standalone`; remote advanced from
  `5aa52594` to `0be7a52c`.
- 2026-06-12: Incoming commits:
  - `c53afa3f docs: track agent dynamic-k experiments`
  - `0be7a52c docs: record 5090 glm ser sweep results`
- 2026-06-12: Incoming tracked paths are all under `.Agent/`; no core source,
  build, script, preset, model, or test path outside `.Agent/` changed.
- 2026-06-12: Pulled with
  `git pull --ff-only origin feat/glm51-5090-2gb-vram-standalone`.
  The merge was a clean fast-forward with no conflicts.
- 2026-06-12: Local untracked experiment directory
  `.Agent/plans/5090-fast-ssd-first-three-repro/` was preserved.
- 2026-06-12: Read the new remote plans:
  - `.Agent/plans/glm-dynamic-k-cache-aware-routing/plan.md`
  - `.Agent/plans/glm-ser-midrange-sweep/plan.md`
  - `.Agent/plans/glm-ser-midrange-sweep/REPRODUCE.md`

## Results

Pulled and integrated successfully. Current branch:

`feat/glm51-5090-2gb-vram-standalone` at
`0be7a52c docs: record 5090 glm ser sweep results`.

Worktree after pull:

- branch is aligned with `origin/feat/glm51-5090-2gb-vram-standalone`;
- only local untracked task directories remain:
  - `.Agent/plans/5090-fast-ssd-first-three-repro/`
  - `.Agent/plans/pull-remote-new-changes/`

Understanding of new remote changes:

1. `c53afa3f docs: track agent dynamic-k experiments`
   - Adds/records the dynamic-k/cache-aware routing task under
     `.Agent/plans/glm-dynamic-k-cache-aware-routing/`.
   - Includes local research snapshots of Dynamic_MoE, MoE-Infinity,
     fiddler, and mixtral-offloading for reference.
   - Documents that existing `--smart-expert-reduction` is the first useful
     no-code dynamic-k mechanism.
   - Finds `SER=1,0.95` as a plausible optional speed/quality tradeoff:
     preserved 4-item smoke accuracy `3/4`, improved eval tok/s from `0.451`
     to `0.590`, and reduced wall time from `656.51s` to `619.24s`.
   - Rejects `SER>=1.0` because output corruption appears despite large speed
     gains.
   - Uses route/cache simulation to validate that the existing simulator can
     reproduce measured VRAM hit rate on a real trace. The one-trace result is
     not enough to justify runtime cache-aware routing.

2. `0be7a52c docs: record 5090 glm ser sweep results`
   - Adds `.Agent/plans/glm-ser-midrange-sweep/`.
   - Performs a midrange SER threshold sweep under the strict target line:
     `MemoryMax=2G`, `MemorySwapMax=0`, RAM tier disabled, top8 profile,
     `GGML_MOE_VRAM_CACHE_MIB=12288`.
   - Phase 1 single-item screen:
     - `SER=1,0.96`, `1,0.97`, and `1,0.975` answered correctly.
     - `SER=1,0.98` and `1,0.99` were faster but already wrong on the single
       smoke item.
   - Phase 2 4-item smoke:
     - `SER=1,0.96`: `3/4` accuracy, `direct_reads=18938`,
       `VRAM hit=25.8%`, `eval tok/s=0.641`, `wall_s=589.08`.
     - `SER=1,0.975`: `2/4` accuracy, faster but too much accuracy loss for
       default use.
   - Current recommendation: `--smart-expert-reduction 1,0.96` is the best
     conservative candidate from these measurements. It is an experimental
     recommendation, not a source-code default.

Engineering implication:

- The branch's runtime/source code did not change in this pull.
- No rebuild is required solely because of this pull.
- The newest evidence points toward using existing SER first, especially
  `--smart-expert-reduction 1,0.96`, before implementing any new
  cache/residency-aware routing code.
- A runtime cache-aware routing patch is still not justified by the committed
  evidence; the next credible step would be multi-trace simulation/holdout
  validation.
