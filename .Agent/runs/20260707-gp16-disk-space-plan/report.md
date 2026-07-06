# Kimi Disk Space Plan For IQ2_XXS Candidate

Generated at: `2026-07-07T02:48:42+0800`

This is a non-destructive inventory. No files were deleted.

## Current Disk

```text
/dev/root: 993G total, 907G used, 86G available
```

Top-level `/root/lfz` usage:

| path | size |
|---|---:|
| `/root/lfz/models` | `378G` |
| `/root/lfz/runs` | `438G` |
| `/root/lfz/runs/ik_llama` | `427G` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets` | `414G` |
| `/root/lfz/runs/vendor-kimi-token-rate` | `9.3G` |

## Current SOTA Assets To Preserve

The current vendor Kimi reproduction uses:

| asset | size | reason |
|---|---:|---|
| `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S` | `378G` | current IQ3_S model shards |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | `164G` | primary current expert pack |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | `4.6G` | current overlay expert pack |

Deleting any of these would make current SOTA runtime reproduction harder
without redownloading or rebuilding assets.

## Candidate Old Assets

These appear to be historical expert-pack artifacts, not current SOTA inputs:

| asset | size |
|---|---:|
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack` | `160G` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack` | `75G` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack` | `7.2G` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack` | `4.7G` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/tmp-hot-upgate-pair-smoke.expert-pack` | `328M` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-missing-down-overlay.expert-pack` | `190M` |

Estimated releasable space from these old expert packs: about `247G`.

## Feasibility

- Current free disk: `86G`.
- After deleting only the old expert-pack candidates above:
  - estimated free disk: about `333G`.
- Smallest external GGUF candidate from GP15:
  - AesSedai `IQ2_XXS`: `262.79 GiB`.

This is likely enough to download the IQ2_XXS GGUF shards and run a first
metadata/smoke check. It may not be enough to also build a full IQ2 expert pack
alongside the downloaded GGUF, depending on pack size and temporary scratch
needs.

## Decision

Do not delete anything automatically.

The next actionable step requires user approval for a specific cleanup command,
or an attached disk. The conservative cleanup target is the old expert-pack
list above, while preserving the current IQ3_S model and current SOTA expert
packs for rollback and reproduction.
