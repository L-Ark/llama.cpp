# Kimi Prompt-Agnostic RAM Tier A/B

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Commit: `d4d46be61`

This is a default-off dev A/B. It is not a SOTA claim.

## Goal

Test whether prompt-agnostic full layer/role RAM residency can reduce
CPU/defer GPU-extension demand-read exposure under the hard constraints:

- cold start;
- `MemoryMax=15900000000`, `MemorySwapMax=0`;
- France quality pass;
- TTFT `<= +20%` versus paired same-commit baseline;
- host RAM peak `<16 GB`;
- endpoint decode token rate improves on generalized dev prompts;
- no prompt-specific hotset or layout.

## Runs

### Baselines

France baseline:

```text
run=/root/lfz/runs/vendor-kimi-token-rate/20260712-ramtier-ab-blk14-gate-baseline-france-n96-062624
prompt=Please introduce France in a short paragraph.
quality=pass
token_rate=1.88 tok/s
TTFT=8150.46 ms
decode=45128.01 ms / 85
memory_peak=12727726080
iouring_wait_us=46316006
iouring_bytes=485993840640
```

Intelligence baseline:

```text
run=/root/lfz/runs/vendor-kimi-token-rate/20260712-ramtier-ab-blk14-gate-baseline-intelligence-n96-063037
prompt=What is intelligence?
quality=pass
token_rate=1.92 tok/s
TTFT=6284.77 ms
decode=49475.42 ms / 95
memory_peak=12578398208
iouring_wait_us=50064190
iouring_bytes=518490226688
```

## Candidate 1: `blk14_gate_full384`

Profile:

```text
profile=.Agent/profiles/kimi/ram-tier/20260712-current-goal-dev-aggregate-layer-role/blk14_gate_full384.csv
entries=384
resident=2200 MiB budget
payload=2157969408 bytes, 2.0098 GiB
pin=0
preload=O_DIRECT, 8 threads
source=dev aggregate top up/gate wait layer, not per-prompt hotset
```

France:

```text
run=/root/lfz/runs/vendor-kimi-token-rate/20260712-ramtier-ab-blk14-gate-candidate-france-n96-062820
quality=pass
token_rate=1.92 tok/s
baseline=1.88 tok/s
delta=+0.04 tok/s
TTFT=8543.84 ms
TTFT_ratio=1.0483
decode=44260.90 ms / 85
decode_delta=-867.11 ms
memory_peak=14941683712
iouring_wait_us=45025449
iouring_wait_delta=-1290557 us
iouring_bytes=483672899584
RAM tier hits=413/84620, hit_rate=0.5%
RAM tier H2D bytes=2320941056
```

Intelligence:

```text
run=/root/lfz/runs/vendor-kimi-token-rate/20260712-ramtier-ab-blk14-gate-candidate-intelligence-n96-063209
quality=pass
token_rate=1.84 tok/s
baseline=1.92 tok/s
delta=-0.08 tok/s
TTFT=6853.30 ms
TTFT_ratio=1.0905
decode=51507.34 ms / 95
decode_delta=+2031.92 ms
memory_peak=14788255744
iouring_wait_us=50641291
iouring_wait_delta=+577101 us
iouring_bytes=516011933696
RAM tier hits=441/90105, hit_rate=0.5%
RAM tier H2D bytes=2478292992
```

Decision:

- Reject as generalized improvement.
- It passes RAM/TTFT/quality gates, and improves France, but it regresses the
  second dev prompt by `-0.08 tok/s` and increases decode time by `2031.92 ms`.
- The hit rate is only `0.5%`, so a single full gate layer does not produce
  enough prompt-general coverage to justify the `~2.2GB` RAM peak increase.
- Do not run held-out tests and do not claim SOTA.

## Candidate 2: `blk4_down_full384`

Profile:

```text
profile=.Agent/profiles/kimi/ram-tier/20260712-current-goal-dev-aggregate-layer-role/blk4_down_full384.csv
entries=384
resident=3100 MiB budget
payload=2994733056 bytes, 2.7891 GiB
pin=0
preload=O_DIRECT, 12 threads
source=dev aggregate top down stage layer, not per-prompt hotset
```

France run:

```text
run=/root/lfz/runs/vendor-kimi-token-rate/20260712-ramtier-ab-blk4-down-candidate-france-n96-063436
status=stopped manually after hard gate failure
service_runtime=4min35.585s
stdout=France
metrics.json=missing
exit.txt=missing
RAM tier preload=384 entries, 2856.00 MiB anonymous mmap
systemd status at 4min18s: Memory=14.6G, max=14.8G, available=133.3M
first RAM batch=79 jobs, 616103936 bytes, 60.19 ms
```

Decision:

- Reject immediately.
- The run had not produced a complete answer after more than four minutes, so
  TTFT exceeded the `+20%` gate by a wide margin.
- It pushed the service close to the 16GB cgroup limit and left only about
  `133 MB` available at the sampled status point.
- The experiment was stopped to avoid wasting machine time. It is not a SOTA
  candidate and should not be rerun in the same GB-scale form.

## Summary

| candidate | France | Intelligence | RAM/TTFT | decision |
|---|---|---|---|---|
| `blk14_gate_full384` | `1.88 -> 1.92 tok/s` | `1.92 -> 1.84 tok/s` | pass | reject, not generalized |
| `blk4_down_full384` | no complete answer after `>4min` | not run | fail | reject, TTFT/RAM |

## Interpretation

Full layer/role RAM residency is too coarse for the current 16GB host-RAM
budget:

- a single full gate layer covers only about `0.5%` of runtime expert events;
- the small France gain does not generalize to another dev prompt;
- a full down layer has large first-batch and memory pressure, causing an
  immediate TTFT gate failure;
- hit rate is not a sufficient metric. Endpoint token rate and TTFT decide.

This confirms that replacing page cache with GB-scale static whole-layer RAM
tiers is not the next high-value path.

## Next Step

Do not continue whole-layer RAM-tier sweeps unless a new admission model predicts
materially higher multi-prompt coverage under a smaller memory budget.

Next acceptable RAM/cache work must be one of:

1. a small dev-aggregate wait-weighted RAM tier, capped around `512-1024 MiB`,
   with leave-one-prompt-out coverage before runtime;
2. a scheduler-only A/B that increases same-layer up/gate/down IO batch size
   without adding GB-scale resident memory;
3. a lower-byte/v2 path only after output-error screening passes.

Any new candidate must still be paired cold-start baseline vs experiment and
must improve generalized dev prompts before held-out tests.

## Reproduction

Baseline command shape:

```bash
RUN=/root/lfz/runs/vendor-kimi-token-rate/<baseline-run>
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO=/root/lfz/llama.cpp-vendor-kimi \
      RUN="$RUN" \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,paris|culture|europe" \
      N=96 PROFILE=0 COPY_PROFILE=0 RAM_AUDIT=1 \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Candidate env template:

```bash
EXTRA_RUNTIME_ENV="GGML_MOE_RAM_TIER_MIB=<mib>
GGML_MOE_RAM_TIER_PROFILE=<profile>
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=0
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=<threads>
GGML_MOE_RAM_BATCH_PROFILE_OUT=$RUN/ram-batch-profile.csv"
```
