# Kimi Current-Goal RAM Slab N96 Refresh

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Source commit: `232e45435`

This is a dev-only admission. It does not change runtime behavior and does not
claim SOTA.

## Goal

Refresh the RAM/VRAM storage admission on current HEAD with fresh `n96`
cold-start IO traces. The question is whether low-value decode-time file/page
cache can be replaced by explicit RAM-resident, batchable expert slabs with
enough critical-path saving to justify a runtime A/B.

Admission target:

- stay within the `16 GB` host-RAM model;
- use generalized dev prompts, not held-out/test prompts;
- preserve quality;
- prove a plausible `>2 tok/s` path before runtime implementation.

## Fresh Trace Inputs

Trace root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-slab-iotrace-n96-074132
```

Injected trace env:

```text
GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv
GGML_MOE_IO_READ_TRACE_OUT=$RUN/io-read-trace.csv
```

Run command shape:

```bash
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO=/root/lfz/llama.cpp-vendor-kimi \
      RUN=<trace-root>/<prompt-id> \
      PROMPT_ID=<prompt-id> \
      PROMPT_USER_TEXT=<prompt> \
      QUALITY_KEYWORDS=<keywords> \
      N=96 PROFILE=1 COPY_PROFILE=0 \
      EXTRA_RUNTIME_ENV='GGML_MOE_IO_WAIT_TRACE_OUT=<run>/io-wait-trace.csv
GGML_MOE_IO_READ_TRACE_OUT=<run>/io-read-trace.csv' \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

## Trace Quality And Baseline

| prompt | quality | tok/s | decode ms | decode tokens | TTFT ms | RAM peak GiB | iouring GiB | iouring wait ms | inflight avg/max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.68` | `50702.05` | `85` | `8939.01` | `11.933` | `452.59` | `44921.22` | `3.78/8` |
| `dev_intelligence_general` | pass | `1.68` | `56612.94` | `95` | `7304.90` | `11.773` | `482.86` | `50097.91` | `3.63/8` |

Combined baseline:

- decode: `107314.99 ms / 180 tokens`;
- mean: `596.194 ms/token`;
- rate: `1.677 tok/s`;
- `2 tok/s` requires `<=500 ms/token`, so the required saving is
  `96.194 ms/token`.

Quality samples:

- France: `France is a country in Western Europe known for its rich history,
  culture, and influence on art, fashion, and cuisine...`
- Intelligence: `Intelligence is a complex, multi-faceted capacity rather than
  a single thing...`

## Slab Screen

Artifact:

```text
.Agent/runs/20260712-current-goal-ram-slab-n96-refresh/slab-screen/report.md
```

Command:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-slab-iotrace-n96-074132
OUT=.Agent/runs/20260712-current-goal-ram-slab-n96-refresh/slab-screen
python3 .Agent/run-tools/kimi_phase5c_ram_slab_screen.py \
  --input-root "$ROOT" \
  --route-profile-root "$ROOT" \
  --out-dir "$OUT" \
  --roles up,gate,down \
  --max-jobs 8 \
  --contig-window-mib 512 1024 2048 \
  --contig-top-per-source 8
```

Inputs:

- prompts: `2`;
- decode-like read rows: `150482`;
- decode-like batches: `32264`;
- decode tokens: `180`;
- total batch wait: `90830.872 ms`;
- total batch wait per token: `504.616 ms/token`;
- simulated VRAM hotset entries from route profiles: `2458`.

## Results

Best whole layer/role candidates:

| candidate | resident MiB | wait ms/token | hit batches | RAM-only batches | mixed risk |
|---|---:|---:|---:|---:|---:|
| `blk.1.upgate` | `2233.96` | `8.451` | `360` | `360` | `0` |
| `blk.9.upgate` | `2569.97` | `6.867` | `360` | `360` | `0` |
| `blk.7.upgate` | `2284.31` | `6.690` | `358` | `358` | `0` |
| `blk.1.all` | `3731.85` | `11.530` | `540` | `540` | `0` |
| `blk.9.all` | `4625.72` | `10.452` | `540` | `540` | `0` |

Greedy layer-slab budget bounds:

| budget | slab family | selected | resident MiB | saved wait ms/token | implied tok/s if fully saved |
|---:|---|---:|---:|---:|---:|
| `4096 MiB` | `layer_upgate` | `2` | `4028.2` | `14.36` | `1.718` |
| `8192 MiB` | `layer_upgate` | `4` | `7338.6` | `25.00` | `1.751` |
| `10240 MiB` | `layer_upgate` | `5` | `9177.6` | `30.86` | `1.769` |
| `4096 MiB` | `layer_all` | `1` | `3731.8` | `11.53` | `1.711` |
| `8192 MiB` | `layer_all` | `2` | `6729.4` | `20.08` | `1.735` |
| `10240 MiB` | `layer_all` | `3` | `9732.5` | `28.37` | `1.761` |

Contiguous pack-window candidates had good score per GiB but tiny absolute
savings, mostly around `0.03-0.12 ms/token`, so they are not a route to `2 tok/s`
without a different storage representation.

## Decision

Reject static layer/role RAM slabs as the next primary runtime A/B on current
N96 traces.

Reasons:

- the `2 tok/s` target needs about `96.194 ms/token` saving;
- even a `10 GiB` dev-screen layer-upgate set only saves `30.86 ms/token`;
- even a `10 GiB` layer-all set only saves `28.37 ms/token`;
- those bounds ignore cold preload, TTFT, RAM pressure, reclaim, and scheduler
  overhead, so real runtime gains would be lower;
- single contiguous windows have too little absolute impact;
- previous runtime A/B already showed GB-scale static RAM can pass one prompt
  but fail generalized dev behavior.

This does not mean RAM is useless. It means RAM needs a mechanism that changes
whole-batch behavior:

- a predictor/prefetcher that makes future complete batches RAM-resident before
  demand;
- a lower-byte expert representation so each batch is smaller;
- a pack layout that preserves large sequential reads and reduces H2D/staging
  cost across all active experts;
- or a RAM tier used as a secondary multiplier after byte reduction, not as the
  primary `>2 tok/s` path.

## Next Action

Keep static RAM slab paths closed unless a new admission predicts at least
`96 ms/token` saving under generalized dev traces. The next higher-value work is
still:

1. approved full `i1-IQ1_S` same-model smoke; or
2. a stronger non-destructive lower-byte representation with prompt-level error
   evidence below `0.50x`; or
3. a predictor/prefetch admission that covers complete future batches rather
   than isolated rows.
