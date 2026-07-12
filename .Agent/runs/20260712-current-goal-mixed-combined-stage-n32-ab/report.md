# Kimi mixed/same-type up-gate combined-stage N32 A/B

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `7de0ccac1`
Rollback point: `7de0ccac1`

This is a default-off diagnostic for larger same-layer up/gate IO batches.
It is not a SOTA claim.

## Question

The current decode profile shows that gate/up runtime-load wait dominates and
that decode exposes only small IO batches:

- gate runtime-load wait: about `14.7 s` on France N96;
- up runtime-load wait: about `14.4 s` on France N96;
- decode-only avg read jobs/batch: about `4.7`;
- decode-only weighted inflight: about `3.5`.

The tested idea was to co-submit same-layer up/gate misses in larger batches.

## Implementation Attempt

A default-off mixed-path implementation was compiled and tested with:

- `GGML_MOE_MIXED_UP_GATE_COMBINED_STAGE=1`.

It used a safe guard:

- `up_expert_bytes == gate_expert_bytes`;
- non-prompt mode;
- existing CUDA copy event available.

The guard did not activate for Kimi mixed calls. Kimi's mixed path is primarily:

- `up_type=18, gate_type=22`;
- `up_type=22, gate_type=18`.

Those type pairs have different expert byte sizes, so a same-size combined
`expert_pack_iouring_copy_jobs` call is not valid. The runtime source change
was reverted before commit.

## Existing Same-Type Diagnostic

The branch already has a same-type default-off path:

- `GGML_MOE_UP_GATE_COMBINED_STAGE=1`.

This path did activate:

```text
[moe_stream] up/gate combined staging active
```

It only affects same-type parallel up/gate calls, such as `22/22`. It does not
cover the mixed `18/22` and `22/18` Kimi calls.

## Commands

Baseline:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-mixed-combined-stage-n32-ab
RUN="$ROOT/baseline"
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" N=32 PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,europe|paris|country" \
      PROFILE=1 COPY_PROFILE=0 VRAM_MIB=15000 PINNED_SLOTS=12 \
      UPGATE_PCT=72 IQ2_UPGATE_PARALLEL=1 \
      MOE_IO_DEPTH=8 MOE_IO_REFILL_BATCH=4 MOE_PREFETCH_DOWN_DEPTH=2 \
      EXTRA_RUNTIME_ENV="GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Same-type combined candidate:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-mixed-combined-stage-n32-ab
RUN="$ROOT/sametype-combined"
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" N=32 PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,europe|paris|country" \
      PROFILE=1 COPY_PROFILE=0 VRAM_MIB=15000 PINNED_SLOTS=12 \
      UPGATE_PCT=72 IQ2_UPGATE_PARALLEL=1 \
      MOE_IO_DEPTH=8 MOE_IO_REFILL_BATCH=4 MOE_PREFETCH_DOWN_DEPTH=2 \
      GGML_MOE_UP_GATE_COMBINED_STAGE=1 \
      EXTRA_RUNTIME_ENV="GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

## Endpoint Metrics

| run | active | quality | tok/s | decode ms / runs | TTFT ms | RAM peak bytes | CPU fallback | iouring wait |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | default | pass | `1.70` | `18206.89 / 31` | `9182.96` | `12765011968` | `0` | `18034153 us` |
| same-type combined | yes | pass | `1.70` | `18241.14 / 31` | `8250.78` | `12764528640` | `0` | `16124716 us` |

The same-type candidate reduced total `io_uring_wait_us` by about `1.91 s`,
but endpoint decode was slightly slower.

## Decode-Only IO Batch Metrics

| run | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | <=4 job batches |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | `5579` | `25623` | `4.593` | `3.479` | `15622.196` | `16633.431` | `49.5%` |
| same-type combined | `4753` | `22431` | `4.719` | `3.544` | `13160.784` | `13974.322` | `47.3%` |

The IO diagnostic moved in the expected direction, but not enough to improve
the endpoint.

## Gap Analysis

Same-type combined reduced IO wait and the number of IO batches, but it did
not improve endpoint decode because:

- it covers only the same-type subset, not the dominant mixed `18/22` and
  `22/18` Kimi calls;
- it moves more work onto the main stage ring:
  - baseline stage-ring H2D timed: `9833.680 ms`;
  - same-type combined stage-ring H2D timed: `10409.699 ms`;
  - baseline stage-ring slot wait: `3647.685 ms`;
  - same-type combined stage-ring slot wait: `3868.470 ms`;
- the gate ring becomes less loaded, but the main stage ring becomes more
  contended;
- the candidate can reduce IO wait while losing up/gate copy-compute overlap.

## Decision

Reject for SOTA and do not run N96.

The experiment confirms that larger batches can reduce io_uring wait, but the
current same-type combined design does not improve endpoint token rate. The
mixed Kimi path requires multi-size IO batch support before this direction is
worth another runtime A/B.

## Next Direction

If continuing the joint up/gate IO direction, the next implementation must
support multi-size mixed up/gate jobs in one scheduler while preserving:

- independent H2D streams where useful;
- up compute starting as soon as its own bytes are ready;
- large SSD batches despite different expert byte sizes;
- endpoint token-rate improvement, not only IO wait reduction.

Otherwise, priority should shift back to lower-byte expert representation or
more reliable future expert admission, because current exact-byte scheduling
changes have not moved endpoint tok/s enough.

## Artifacts

- baseline run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-mixed-combined-stage-n32-ab/baseline`
- same-type combined run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-mixed-combined-stage-n32-ab/sametype-combined`
- decode-only baseline report:
  `.Agent/runs/20260712-current-goal-mixed-combined-stage-n32-ab/baseline-io-decode.md`
- decode-only same-type combined report:
  `.Agent/runs/20260712-current-goal-mixed-combined-stage-n32-ab/sametype-combined-io-decode.md`
