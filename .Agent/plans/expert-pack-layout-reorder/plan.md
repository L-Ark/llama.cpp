# Expert-Pack Physical Layout Reorder

## Goal

Optimize GLM-5.1 MoE SSD read performance by rebuilding the expert-pack with a
better physical byte layout, while preserving model semantics and the existing
runtime lookup contract.

The first implementation must be conservative:

- do not overwrite the original `glm51-iq3xxs.expert-pack`;
- generate a separate reordered pack under this task directory;
- keep tensor/expert identity unchanged;
- update only pack offsets/index metadata;
- validate with strict `MemoryMax=2G`, `MemorySwapMax=0`, `RAM tier=0`.

## Hypothesis

Current runtime still performs about `62582` io_uring expert reads on the n84
diagnostic. Offset sorting within each io_uring batch did not materially change
performance because the underlying expert-pack layout remains route-agnostic.

If the pack physically groups rows that are accessed together, the SSD path may
see fewer expensive random seeks and better locality without changing selected
experts, cache policy, or output.

## Candidate Layouts

Start with low-risk static layouts:

1. `profile-hot-first`: order entries by the oracle/profile CSV frequency, then
   keep the original order for the rest.
2. `trace-order-first`: order first by first occurrence in the route trace,
   then by original order.
3. `layer-kind-expert`: deterministic structural layout by layer, tensor kind,
   expert id; use as a sanity baseline if current pack order is different.

The first benchmark should compare only one or two layouts against the original
pack to avoid wasting runs.

## Implementation Plan

1. Inspect the expert-pack file format and runtime index lookup.
2. Add a task-local reorder script that:
   - reads the original pack header/index;
   - preserves payload bytes exactly;
   - writes a new pack with entries in a requested physical order;
   - updates offsets in the index;
   - preserves alignment.
3. Add a task-local validation script that compares:
   - entry count and tensor/expert keys;
   - byte sizes;
   - payload hash per entry or sampled payload hashes;
   - original versus reordered offset distribution.
4. Generate one reordered pack from the current n84 oracle route/profile.
5. Run a quick n8 smoke with the reordered pack.
6. If safe, run n84 A/B under strict 2GB:
   - original pack;
   - reordered pack;
   - same profile/cache/runtime settings;
   - `GGML_MOE_IO_SORT_OFFSET=1` both on and off only if needed.

## Acceptance

A reordered pack is a real win only if:

- `read_failures=0`;
- output remains deterministic for the fixed prompt;
- `expert_pack_iouring_reads` does not increase materially;
- `eval tok/s` improves by at least `5%` or aggregate `iouring_wait_us`
  improves enough to justify more layouts;
- all logs and generated pack paths are recorded here.

## Stop Conditions

Stop before benchmarking if:

- pack format has embedded offsets/checksums that cannot be safely rewritten;
- runtime assumes a fixed physical order beyond the index;
- generated pack fails byte-level validation.

Stop after first A/B if:

- n8 shows read failures or output/runtime instability;
- n84 shows no measurable wait-time or token-rate improvement.

## Progress

- 2026-06-15: Task created. Required Agent files read. Existing worktree has
  unrelated modified CUDA/ggml files and many untracked `.Agent/plans` folders;
  do not revert them. Next step is format/index inspection.
- 2026-06-15: Format inspection:
  - pack header/index format is v1: `MAGIC`, version, header size, entry
    count, data start, followed by fixed index rows
    `(tensor[128], expert_idx, reserved, offset, nbytes)`;
  - runtime reads all entries and sorts them by `(tensor, expert_idx, nbytes)`
    before lookup, so physical payload order can change as long as index
    offsets are updated;
  - original full pack is `234G`, but filesystem free space is only `213G`.
    Therefore a full reordered duplicate is not safe in this workspace.
- 2026-06-15: Adjusted first implementation to a hot-subset reordered pack:
  build a separate pack containing the n84 oracle/profile rows
  (`19167` rows, `76.63 GiB` payload), enough to validate this workload without
  overwriting the original full pack.
- 2026-06-15: Existing `moe-read-schedule-analyze.py` was tried for locality
  analysis, but it also runs expensive predictor sweeps and did not finish
  promptly. Stopped that self-started process and moved to a task-local focused
  reorder/validation script.
- 2026-06-15: Added task-local script
  `reorder_expert_pack.py`. It reads the source pack index, selects rows from
  profile/trace, writes a v1-compatible output pack with updated offsets, and
  validates deterministic payload samples by SHA256.
- 2026-06-15: Smoke-generated
  `packs/profile-hot-first-16.expert-pack` with `16` rows. `create-moe-expert-pack.py
  --inspect` accepts the generated pack, and all sample hashes matched the
  original payload bytes.
- 2026-06-15: Proceeding with `trace-order-first` full hot-subset pack because
  it is the strongest diagnostic for physical locality on the fixed n84 trace.
- 2026-06-15: Generated
  `packs/trace-order-first-n84-hot.expert-pack`:
  - entries: `19167`;
  - payload: `76.626 GiB`;
  - file size: `82279333888` bytes;
  - missing rows from source: `0`;
  - sampled SHA256 checks: `8/8` matched;
  - inspect accepted the pack as v1-compatible.
- 2026-06-15: Added `run_layout_case.py`, a task-local wrapper around the
  existing 5090 runner that allows selecting a different `GGML_MOE_EXPERT_PACK`
  while preserving the current strict 2GB optimized config.
- 2026-06-15: n8 hot-pack smoke passed:

  | label | pack | eval tok/s | total ms | io_uring reads | io_uring wait us | VRAM hit | pack misses | read failures |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `trace-hotpack-n8` | `trace-order-first-n84-hot.expert-pack` | `1.48` | `25488.89` | `7382` | `5152678` | `41.5%` | `0` | `0` |

  The hot subset covers the fixed prompt path for n8. Proceed to n84.
- 2026-06-15: n84 hot-pack validation completed under strict 2GB, RAM tier 0,
  same oracle profile/cache settings, and `GGML_MOE_IO_SORT_OFFSET=1`.

  Individual n84 runs:

  | label | pack | eval tok/s | total ms | io_uring reads | io_uring wait us | VRAM hit | pack misses | read failures |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `offset-sort-n84` | original full pack | `1.74` | `68249.45` | `62582` | `51730280` | `58.1%` | `0` | `0` |
  | `offset-sort-n84-repeat1` | original full pack | `1.73` | `70732.79` | `62582` | `52366939` | `58.1%` | `0` | `0` |
  | `original-pack-n84-post` | original full pack | `1.69` | `69749.31` | `62582` | `53706587` | `58.1%` | `0` | `0` |
  | `trace-hotpack-n84` | trace-order hot pack | `1.85` | `65567.03` | `62582` | `46692953` | `58.1%` | `0` | `0` |
  | `trace-hotpack-n84-repeat1` | trace-order hot pack | `1.92` | `63787.69` | `62582` | `44634422` | `58.1%` | `0` | `0` |

  Averages saved in `n84-averages.json`:

  | group | runs | avg eval tok/s | avg total ms | avg io_uring reads | avg io_uring wait us |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | original full pack | `3` | `1.720` | `69577.18` | `62582` | `52601268.67` |
  | trace-order hot pack | `2` | `1.885` | `64677.36` | `62582` | `45663687.50` |

  Measured average delta:

  - eval token rate: `+9.59%`;
  - total runtime: `-7.04%`;
  - aggregate io_uring wait time: `-13.19%`;
  - io_uring reads: unchanged;
  - VRAM hit: unchanged;
  - pack lookup misses and read failures: `0`.

  Interpretation:

  - The physical-layout hypothesis is validated for this fixed n84 oracle
    workload: read count and cache behavior are unchanged, but read wait time
    drops materially.
  - This is the first strict-2GB result in this task that reaches the requested
    `5-10%` eval-token-rate improvement band.
  - Caveat: this pack is a trace/profile hot subset, not a general full pack.
    It is a strong diagnostic and a prototype layout, not yet a broad-prompt
    production artifact.

## Current Artifacts

- Reorder script:
  `reorder_expert_pack.py`
- Runner wrapper:
  `run_layout_case.py`
- Reordered hot pack:
  `packs/trace-order-first-n84-hot.expert-pack`
- Manifest:
  `packs/trace-order-first-n84-hot.manifest.json`
- Per-run summaries:
  `runs/trace-hotpack/*/summary.json`
- Aggregated n84 summary:
  `n84-summary.json`
- Aggregated n84 averages:
  `n84-averages.json`

## Next Work

To make this general rather than same-prompt diagnostic:

1. Build a multi-prompt profile/trace set and create a hot pack that covers the
   union of held-out-relevant rows within a storage budget.
2. Compare `profile-hot-first` and `trace-order-first` on held-out prompts.
3. Implement a two-pack runtime fallback if we want a small optimized hot pack
   plus the original full pack for arbitrary prompts. This avoids needing
   another `234G` full duplicate while preserving correctness outside the hot
   subset.
