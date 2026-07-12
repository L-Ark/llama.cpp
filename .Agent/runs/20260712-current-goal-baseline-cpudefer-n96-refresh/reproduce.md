# Reproduce Kimi Current-Goal Baseline Profile

Branch:

```bash
git checkout vendor/kimi-deepseek-41d205-additive
git rev-parse --short HEAD
# expected for this run: 2d0487c94
```

Run root used for the recorded profile:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-baseline-cpudefer-n96-082136
```

France quality/profile run:

```bash
RUN=$ROOT/dev_france_regression
EXTRA_RUNTIME_ENV="GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv
GGML_MOE_RAM_BATCH_PROFILE_OUT=$RUN/ram-batch-profile.csv"

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT='Please introduce France in a short paragraph.' \
      QUALITY_KEYWORDS='france,europe' \
      N=96 PROFILE=1 COPY_PROFILE=1 \
      EXTRA_RUNTIME_ENV="$EXTRA_RUNTIME_ENV" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Generalized dev profile run:

```bash
RUN=$ROOT/dev_intelligence_general
EXTRA_RUNTIME_ENV="GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv
GGML_MOE_RAM_BATCH_PROFILE_OUT=$RUN/ram-batch-profile.csv"

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN="$RUN" \
      PROMPT_ID=dev_intelligence_general \
      PROMPT_USER_TEXT='What is intelligence?' \
      QUALITY_KEYWORDS='intelligence,reason|learn|adapt|understand' \
      N=96 PROFILE=1 COPY_PROFILE=1 \
      EXTRA_RUNTIME_ENV="$EXTRA_RUNTIME_ENV" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Generate committed summaries:

```bash
OUT=.Agent/runs/20260712-current-goal-baseline-cpudefer-n96-refresh

python3 .Agent/run-tools/kimi_cpu_defer_gpu_extension_audit.py \
  --input "$ROOT" \
  --out "$OUT" \
  --top-n 20

python3 .Agent/run-tools/kimi_io_queue_metrics_summary.py \
  --metrics "$ROOT/dev_france_regression/metrics.json" \
  --metrics "$ROOT/dev_intelligence_general/metrics.json" \
  --out-json "$OUT/io-queue-summary.json" \
  --out-csv "$OUT/io-queue-summary.csv" \
  --out-md "$OUT/io-queue-summary.md"

python3 .Agent/run-tools/kimi_copy_profile_breakdown.py \
  --run-dir "$ROOT/dev_france_regression" \
  --out "$OUT/dev_france_regression-copy-profile-breakdown.md"

python3 .Agent/run-tools/kimi_copy_profile_breakdown.py \
  --run-dir "$ROOT/dev_intelligence_general" \
  --out "$OUT/dev_intelligence_general-copy-profile-breakdown.md"

python3 .Agent/run-tools/kimi_io_batch_profile_breakdown.py \
  --input "$ROOT/dev_france_regression/io-batch-profile.csv" \
  --input "$ROOT/dev_intelligence_general/io-batch-profile.csv" \
  --out-json "$OUT/io-batch-breakdown.json" \
  --out-md "$OUT/io-batch-breakdown.md" \
  --out-role-csv "$OUT/io-batch-by-role.csv" \
  --out-layer-csv "$OUT/io-batch-by-layer.csv" \
  --top-n 20

python3 .Agent/run-tools/kimi_wait_weighted_admission_screen.py \
  --profile-root "$ROOT" \
  --out-csv "$OUT/wait-weighted-layer-role.csv" \
  --out-md "$OUT/wait-weighted-layer-role.md" \
  --top 30
```

Notes:

- `COPY_PROFILE=1` enables `GGML_MOE_COPY_PROFILE_H2D=1`; the resulting token
  rate is diagnostic and must not be claimed as SOTA.
- Cold-start behavior is enforced by `kimi-general-prompt-repro.sh`, which calls
  `sync` and `echo 3 > /proc/sys/vm/drop_caches` before launching the model.
- The run must remain below `MemoryMax=15900000000` with swap disabled.
