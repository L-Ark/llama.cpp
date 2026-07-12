# Kimi current-goal baseline/profile refresh

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Runtime commit: `a4076b7a4`
Rollback point: `531eefa8e`

This report refreshes the current Kimi baseline after the goal/plan commit.
It does not claim a new SOTA. It identifies the next bottleneck under the
strict cold-start `16 GB` host RAM gate.

## Runs

Copy/H2D diagnostic run:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-n96
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" N=96 PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,europe|paris|country" \
      PROFILE=1 COPY_PROFILE=1 VRAM_MIB=15000 PINNED_SLOTS=12 \
      UPGATE_PCT=72 IQ2_UPGATE_PARALLEL=1 \
      MOE_IO_DEPTH=8 MOE_IO_REFILL_BATCH=4 MOE_PREFETCH_DOWN_DEPTH=2 \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

IO-batch diagnostic run:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-n96-iobatch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" N=96 PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,europe|paris|country" \
      PROFILE=1 COPY_PROFILE=0 VRAM_MIB=15000 PINNED_SLOTS=12 \
      UPGATE_PCT=72 IQ2_UPGATE_PARALLEL=1 \
      MOE_IO_DEPTH=8 MOE_IO_REFILL_BATCH=4 MOE_PREFETCH_DOWN_DEPTH=2 \
      EXTRA_RUNTIME_ENV="GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

## Endpoint Result

The IO-batch run is the cleaner endpoint reference because it does not enable
H2D synchronization for copy timing.

- prompt: `Please introduce France in a short paragraph.`
- quality: pass
- output:
  `France is a country in Western Europe known for its rich history, culture,
  and influence on art, fashion, and cuisine. Its capital, Paris, is famous
  for landmarks like the Eiffel Tower and the Louvre Museum. France is also
  known for its beautiful countryside, wine regions, and historic cities such
  as Lyon and Marseille. It plays a major role in European and global politics
  as a founding member of the European Union.`
- TTFT: `8498.12 ms`
- decode: `50692.07 ms / 85`
- token rate: `1.68 tok/s`
- host RAM peak: `12802908160 bytes`, about `11.92 GiB`
- swap max: `0`
- active file at exit: `12026712064 bytes`
- CPU fallback rows: `0`
- GPU extension miss: `0`

## CPU/defer GPU Extension

The CPU/defer GPU-extension path is active and complete for this profile.

- `up_gate` batch accept: `5101 / 5101`
- `down` batch accept: `5278 / 5278`
- true fallback rows: `0`
- weighted decode: `649.959 ms/token`, `1.539 tok/s` in the copy-profile run
- up/gate GPU-extension wall: `443.978 ms/token`
- up/gate exposed wait proxy: `255.369 ms/token`
- decode down GPU-extension wall: `170.954 ms/token`
- down stage: `159.927 ms/token`

Decision: do not spend the next implementation cycle on another broad CPU
fallback hook. The current failure mode is expert movement and wait, not CPU
math fallback.

## Copy-Path Split

The copy-profile run used `COPY_PROFILE_H2D=1`, so its endpoint token rate is
diagnostic only. It still proves where bytes come from.

- profiled copy wall: `276366 ms`
- expert-pack miss wall: `0 ms`
- expert-pack hit wall: `276366 ms`
- profiled expert-pack bytes: `452.62 GiB`
- non-io_uring wall: `0 ms`

By role:

| op | role | calls | GiB | io wait ms | H2D ms | wall ms |
|---|---|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 26908 | 132.52 | 90819 | 7417 | 90819 |
| `runtime_load` | up | 26905 | 123.20 | 82956 | 6825 | 82956 |
| `runtime_load` | down | 19053 | 127.85 | 68659 | 6794 | 68659 |
| `current_down_overlap` | down | 11754 | 69.05 | 33932 | 3707 | 33932 |

Interpretation:

- the current pack coverage is not the problem: `pack_hit=0` is absent;
- rebuilding another prompt-specific pack is not justified by this profile;
- the next win must reduce VRAM-cache misses, reduce bytes per miss, or make
  demand reads less exposed.

## IO Batch Exposure

All rows:

- batches: `15449`
- read jobs: `84620`
- avg read jobs/batch: `5.477`
- weighted inflight avg: `4.087`
- wait: `46154.564 ms`
- wall: `52282.207 ms`
- read batches with `<=4` jobs: `46.7%`

Decode-only proxy, filtering `jobs <= 8`:

- batches: `15272`
- read jobs: `71447`
- avg read jobs/batch: `4.678`
- weighted inflight avg: `3.470`
- wait: `43730.248 ms`
- wall: `45773.007 ms`
- read batches with `<=4` jobs: `47.3%`

Prompt proxy, filtering `jobs > 8`:

- batches: `177`
- read jobs: `13173`
- avg read jobs/batch: `74.424`
- weighted inflight avg: `7.429`
- wait: `2424.316 ms`
- wall: `6509.200 ms`

Decode-only by role:

| op | role | batches | read jobs | avg read jobs/batch | inflight | wait ms | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `runtime_load` | gate | 5087 | 22517 | 4.426 | 3.477 | 14706.035 | 15503.109 |
| `runtime_load` | up | 5087 | 22514 | 4.426 | 3.252 | 14442.326 | 15051.388 |
| `runtime_load` | down | 2635 | 14662 | 5.564 | 3.805 | 8392.745 | 8732.688 |
| `current_down_overlap` | down | 2463 | 11754 | 4.772 | 3.459 | 6189.143 | 6485.821 |

Interpretation:

- prompt can expose large independent read batches;
- decode cannot: almost all decode rows are `<=8` jobs and about half are
  `<=4` jobs;
- simply increasing `MOE_IO_DEPTH` above `8` is not expected to help decode
  unless the scheduler first exposes larger per-layer batches;
- gate and up remain the largest exposed runtime-load wait buckets.

## Decisions

1. Reject split-only retuning for the current IQ3 path.
   The dev7 offline sweep found only `0.41%` miss-byte improvement at current
   IQ3 and at most `1.07%` under a hypothetical `0.25x` representation.

2. Do not rerun broad global cache policies.
   Historical `lfu_lru`, `profile_lfu_lru` and hybrid profile policies
   regressed token rate and quality on dev prompts.

3. Do not treat more IO depth as the next primary experiment.
   Decode rows do not expose enough jobs for depth `16` to matter.

4. Next candidate should be structural:
   - verify whether gate+up role reads can be safely co-submitted or otherwise
     made more continuous without losing the current up/gate compute overlap;
   - only after that, test a default-off N32 A/B;
   - if the co-submit design would serialize up compute behind gate reads, it
     should be rejected before implementation.

## Artifacts

- copy profile run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-n96`
- IO-batch run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-a4076-baseline-france-n96-iobatch`
- copy profile report:
  `.Agent/runs/20260712-current-goal-a4076-baseline-france-profile/copy-profile-breakdown.md`
- CPU/defer audit:
  `.Agent/runs/20260712-current-goal-a4076-baseline-france-profile/cpu-defer-audit/report.md`
- IO-batch report:
  `.Agent/runs/20260712-current-goal-a4076-baseline-france-profile/io-batch-breakdown.md`
- decode-only IO-batch report:
  `.Agent/runs/20260712-current-goal-a4076-baseline-france-profile/io-batch-decode-only.md`
