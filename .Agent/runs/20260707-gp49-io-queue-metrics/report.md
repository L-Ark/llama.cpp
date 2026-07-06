# GP49 IO queue and backend utilization summary

Timestamp: `2026-07-07T07:04:32+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit:

```text
3a2be20ad tools: analyze kimi static expert priors
```

Purpose:

- Use committed dev N96 `metrics.json` counters to identify whether the current
  general-prompt bottleneck is SSD peak bandwidth, iouring queue starvation, or
  backend fallback.
- Avoid adding runtime code or using held-out test prompts.

Inputs:

- Seven committed dev-only N96 `metrics.json` files from:
  `.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile`.
- No held-out test metrics were used.

Command:

```bash
RUN=.Agent/runs/20260707-gp49-io-queue-metrics
mkdir -p "$RUN"
metrics=()
while IFS= read -r p; do metrics+=(--metrics "$p"); done < <(
  find .Agent/runs/20260706-kimi-general-dev-baseline-n96-profile \
    -name metrics.json | sort
)
python3 .Agent/run-tools/kimi_io_queue_metrics_summary.py \
  "${metrics[@]}" \
  --out-json "$RUN/io-summary.json" \
  --out-csv "$RUN/io-summary.csv" \
  --out-md "$RUN/io-summary.md" \
  --peak-gib-s 10.3
```

Headline result:

```text
prompts=7
weighted_token_rate=0.265472
iouring_gib_s=0.342567
peak_utilization=0.033259
iouring_read_ratio=0.427672
direct_read_ratio=0.572328
iouring_inflight_avg=3.334995
iouring_wait_decode_fraction=0.065532
```

Per-prompt highlights:

| prompt | tok/s | iouring GiB/s | iouring read ratio | direct reads | inflight avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev_france_regression | 1.260 | 4.814 | 0.962 | 2229 | 3.390 |
| dev_japan_factual | 0.420 | 0.675 | 0.502 | 26007 | 3.290 |
| dev_linear_equation | 0.160 | 0.059 | 0.126 | 16872 | 3.800 |
| dev_mixed_summary | 0.200 | 0.120 | 0.203 | 23826 | 3.500 |
| dev_photosynthesis_factual | 0.240 | 0.159 | 0.246 | 36641 | 3.290 |
| dev_python_reverse | 0.170 | 0.059 | 0.133 | 40761 | 3.210 |
| dev_zh_france | 0.380 | 0.452 | 0.425 | 15188 | 3.090 |

Interpretation:

- This dev baseline is not primarily capped by raw SSD peak bandwidth. Aggregate
  iouring throughput is only `0.343 GiB/s`, about `3.3%` of the measured
  `~10.3 GiB/s` pure-IO peak.
- Most reads are not using iouring: direct reads account for `57.2%` of
  expert-pack reads across these general prompts.
- Even when iouring is used, average inflight is only `~3.33`, below the
  configured/available higher depths.
- The France prompt is the exception: it keeps `96.2%` of reads on iouring and
  reaches `4.814 GiB/s`, which is consistent with much better token rate.
- The next high-leverage runtime work should therefore focus on forcing more
  general-prompt runtime loads into known-size iouring batch jobs and reducing
  direct-read fallback, before spending more effort on v2 low-byte payloads.

Validation:

- Local:
  - summary script ran successfully;
  - `python3 -m py_compile` passed;
  - `git diff --check` passed.
- Remote:
  - summary script ran successfully in
    `/root/lfz/tmp/vendor-kimi-speculative-gp33`;
  - headline metrics matched local;
  - `python3 -m py_compile` passed;
  - `git diff --check` passed.

Acceptance decision:

- Accepted as non-SOTA planning evidence.
- No runtime behavior changed.
- No held-out prompts were used.
- No token-rate or output-quality claim is made.
