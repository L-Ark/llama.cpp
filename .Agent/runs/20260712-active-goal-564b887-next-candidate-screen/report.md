# Kimi next-candidate screen from current HEAD audit

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Input root:
`/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855`
Baseline audit:
`.Agent/runs/20260712-active-goal-564b887-cpudefer-n96-audit/report.md`

This is an offline screen only. It does not claim a SOTA result and does not
use held-out prompts for tuning.

## Baseline Constraint

Current France endpoint reference:

- `n96`, cold start, `COPY_PROFILE=0` with IO-batch trace;
- decode: `49512.13 ms / 85`, `1.72 tok/s`;
- TTFT: `9746.78 ms`;
- host RAM peak: `11.92 GiB`;
- quality: pass;
- CPU fallback rows: `0`;
- pack misses: `0`;
- direct reads: `0`.

To reach `2 tok/s` on this same endpoint:

- target decode time is `42500 ms`;
- required saving is `7012 ms`;
- required per-token saving is `82.5 ms/token`.

## Byte-Reduction Bound

Artifact:

- `.Agent/runs/20260712-active-goal-564b887-next-candidate-screen/byte-target-2tps.md`
- `.Agent/runs/20260712-active-goal-564b887-next-candidate-screen/byte-target-5tps.md`

For `2 tok/s`:

- median moved bytes: `5.32 GiB/token`;
- median optimistic all-hit MoE floor: `163.1 ms/token`;
- required byte ratio at `10.40 GiB/s` after floor: median `0.66x`,
  worst `0.59x`;
- role miss-byte shares are balanced:
  - gate: `34.6%`;
  - down: `33.2%`;
  - up: `32.2%`.

For `5 tok/s`:

- required byte ratio after floor is about `0.07x` median, worst `0.00x`;
- this is beyond queue tuning or RAM placement alone. It needs a much smaller
  expert representation, a highly accurate predictive path, or a different
  compute/storage form.

Decision:

- For the `2 tok/s` milestone, a lower-byte representation around `0.6x` can
  plausibly be enough if quality holds.
- For the `5 tok/s` product target, partial scheduling/cache changes are not
  enough by themselves.

## RAM/VRAM Layer-Role Screen

Artifact:

- `.Agent/runs/20260712-active-goal-564b887-next-candidate-screen/wait-weighted-layer-role.md`

Top wait-weighted layer/role buckets are mostly up/gate:

- `blk.14 upgate`: `9.122 ms/token`, observed `2303 MiB`;
- `blk.29 upgate`: `9.005 ms/token`, observed `2431 MiB`;
- `blk.28 upgate`: `8.933 ms/token`, observed `2392 MiB`;
- `blk.54 upgate`: `8.857 ms/token`, observed `2737 MiB`;
- `blk.30 upgate`: `8.699 ms/token`, observed `2372 MiB`.

Greedy ideal layer/role residency bound:

| budget | ideal saved | ideal tok/s | note |
|---:|---:|---:|---|
| `2048 MiB` | `7.3 ms/token` | `1.633` | one small upgate slab |
| `4096 MiB` | `14.2 ms/token` | `1.652` | two slabs at best |
| `6144 MiB` | `22.3 ms/token` | `1.674` | still far below 2 tok/s |
| `8192 MiB` | `27.1 ms/token` | `1.688` | not enough |
| `10240 MiB` | `35.9 ms/token` | `1.713` | not enough |
| `12288 MiB` | `44.6 ms/token` | `1.739` | still below 2 tok/s |

This bound is intentionally optimistic because it assumes selected layer/role
IO disappears completely. A real RAM tier still pays RAM-to-VRAM H2D and may
shrink SSD batch size, so the measured gain should be smaller.

Decision:

- RAM/VRAM layer-role residency alone should not be the next primary path to
  `2 tok/s`.
- It can still be useful as an auxiliary path after lower-byte movement or
  prediction reduces the required bytes.
- A proper RAM slab A/B still needs `io-read-trace.csv`; the current audit
  collected IO batch profiles but not per-read trace rows.

## Next Implementation Priority

1. Lower-byte movement screen:
   - target all-role moved-byte ratio `<=0.6x` for the `2 tok/s` milestone;
   - preserve France semantic quality and generalized dev quality;
   - do not download or replace large model assets without explicit approval.

2. If lower-byte remains blocked, collect `io-read-trace.csv` on generalized
   dev prompts and run RAM slab screening from actual per-read locality.

3. Prediction/prefetch is only worth runtime integration if it can cover
   future up/gate bytes with useful expert-byte recall `>=65%`,
   predicted/actual bytes `<=1.35x`, and complete-step coverage `>=40%`.

4. Scheduler-only changes remain secondary unless an offline trace bound shows
   endpoint-saving potential above the `82.5 ms/token` gap while preserving
   current overlap.
