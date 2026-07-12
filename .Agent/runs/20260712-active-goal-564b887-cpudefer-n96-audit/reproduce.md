# Kimi current-HEAD CPU/defer GPU-extension audit reproduction

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Commit: `564b88773`
Output report: `.Agent/runs/20260712-active-goal-564b887-cpudefer-n96-audit/report.md`
Source root:
`/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855`

This is an audit, not a SOTA claim. The copy-profile runs add H2D
synchronization for timing decomposition. The endpoint token rate should be
taken from the `COPY_PROFILE=0` IO-batch run.

## Run Commands

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855
mkdir -p "$ROOT"
```

France copy/H2D diagnostic:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855/dev_france_regression
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

Generalized dev copy/H2D diagnostic:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855/dev_intelligence_general
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" N=96 PROMPT_ID=dev_intelligence_general \
      PROMPT_USER_TEXT="What is intelligence?" \
      QUALITY_KEYWORDS="intelligence,learn|reason|adapt|solve" \
      PROFILE=1 COPY_PROFILE=1 VRAM_MIB=15000 PINNED_SLOTS=12 \
      UPGATE_PCT=72 IQ2_UPGATE_PARALLEL=1 \
      MOE_IO_DEPTH=8 MOE_IO_REFILL_BATCH=4 MOE_PREFETCH_DOWN_DEPTH=2 \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

France endpoint and IO-batch diagnostic:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855/dev_france_regression_iobatch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" N=96 PROMPT_ID=dev_france_regression_iobatch \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,europe|paris|country" \
      PROFILE=1 COPY_PROFILE=0 VRAM_MIB=15000 PINNED_SLOTS=12 \
      UPGATE_PCT=72 IQ2_UPGATE_PARALLEL=1 \
      MOE_IO_DEPTH=8 MOE_IO_REFILL_BATCH=4 MOE_PREFETCH_DOWN_DEPTH=2 \
      EXTRA_RUNTIME_ENV="GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

## Report Commands

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-564b887-cpudefer-n96-112855
OUT=.Agent/runs/20260712-active-goal-564b887-cpudefer-n96-audit
rm -rf "$OUT"
mkdir -p "$OUT"

python3 .Agent/run-tools/kimi_cpu_defer_gpu_extension_audit.py \
  --input "$ROOT" --out "$OUT" --top-n 24

python3 .Agent/run-tools/kimi_copy_profile_breakdown.py \
  --run-dir "$ROOT/dev_france_regression" \
  --out "$OUT/dev_france_regression-copy-profile-breakdown.md"

python3 .Agent/run-tools/kimi_copy_profile_breakdown.py \
  --run-dir "$ROOT/dev_intelligence_general" \
  --out "$OUT/dev_intelligence_general-copy-profile-breakdown.md"

python3 .Agent/run-tools/kimi_io_batch_profile_breakdown.py \
  --input "$ROOT/dev_france_regression_iobatch/io-batch-profile.csv" \
  --out-json "$OUT/france-iobatch-all.json" \
  --out-md "$OUT/france-iobatch-all.md" \
  --out-role-csv "$OUT/france-iobatch-all-role.csv" \
  --out-layer-csv "$OUT/france-iobatch-all-layer.csv" \
  --top-n 24

python3 .Agent/run-tools/kimi_io_batch_profile_breakdown.py \
  --input "$ROOT/dev_france_regression_iobatch/io-batch-profile.csv" \
  --out-json "$OUT/france-iobatch-decode-only.json" \
  --out-md "$OUT/france-iobatch-decode-only.md" \
  --out-role-csv "$OUT/france-iobatch-decode-only-role.csv" \
  --out-layer-csv "$OUT/france-iobatch-decode-only-layer.csv" \
  --top-n 24 --max-jobs 8

python3 .Agent/run-tools/kimi_io_queue_metrics_summary.py \
  --metrics "$ROOT/dev_france_regression/metrics.json" \
  --metrics "$ROOT/dev_intelligence_general/metrics.json" \
  --metrics "$ROOT/dev_france_regression_iobatch/metrics.json" \
  --out-json "$OUT/io-queue-summary.json" \
  --out-csv "$OUT/io-queue-summary.csv" \
  --out-md "$OUT/io-queue-summary.md"
```

## Result Summary

- France endpoint: `1.72 tok/s`, `49512.13 ms / 85`, TTFT `9746.78 ms`.
- Host RAM peak: `12802686976 bytes`, about `11.92 GiB`.
- Quality: pass; France output is coherent and semantic.
- CPU fallback rows: `0`.
- CPU/defer extension miss: `0`.
- Pack misses: `0`; direct reads: `0`; read failures: `0`.
- France decode-only IO wait: `43111.496 ms`.
- France decode-only up/gate runtime-load wait:
  `14687.123 + 14323.879 = 29011.002 ms`.
- To reach `2 tok/s` on this France `n96` endpoint, the next accepted change
  must save about `7012 ms` total, or `82.5 ms/token`.
