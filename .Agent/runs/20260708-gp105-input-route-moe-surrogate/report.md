# Kimi input-route MoE-output surrogate oracle

This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.

- generated_at: `2026-07-07T21:09:06+0000`
- prompts: `3`
- groups: `468`
- layers evaluated: `53`
- max activation records per prompt: `4096`
- complete-group only: `True`

## Corpus

Source corpus:

- remote root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2`
- mode: dev-only, not held-out
- prompts: `dev_france_regression`, `dev_japan_factual`, `dev_photosynthesis_factual`
- runtime: strict cold-start `systemd-run` per prompt, `MemoryMax=15900000000`, `MemorySwapMax=0`
- generation: `n=16`, `GGML_MOE_ACTIVATION_DUMP_CALL_STRIDE=1`,
  `GGML_MOE_ACTIVATION_DUMP_MAX_RECORDS=4096`,
  `GGML_MOE_ACTIVATION_DUMP_DECODE_ONLY=1`

Corpus quality and resource gates:

| prompt | quality | tok/s | decode ms/runs | TTFT ms | RAM peak | activation records | complete paired groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | pass | `1.54` | `9717.18 / 15` | `85102.93` | `15899996160` | `4096` | `156` |
| `dev_japan_factual` | pass | `1.53` | `9785.88 / 15` | `86032.74` | `15899996160` | `4096` | `156` |
| `dev_photosynthesis_factual` | pass | `1.49` | `10045.85 / 15` | `74207.00` | `15899996160` | `4096` | `156` |

Note:

- The first GP105 corpus attempt used
  `GGML_MOE_ACTIVATION_DUMP_DIR=$RUN/act`; the runtime dump helper does not
  create directories, so no activation files were produced. That attempt is
  invalid for oracle use.
- The accepted r2 corpus writes dump files directly under `$RUN`, an existing
  directory. The oracle supports both `$RUN/act` and `$RUN` layouts.
- Current-down overlap shifts down dump call IDs relative to up/gate call IDs,
  so the oracle pairs same-layer/same-token up and down groups by sequence
  order rather than requiring identical call IDs.

## Summary

| method | rows | mean rel L2 | max rel L2 |
|---|---:|---:|---:|
| `nn_scaled:input_route_stats` | `468` | `0.885045` | `1.298097` |
| `nn_scaled:input` | `468` | `0.885192` | `1.294968` |
| `nn_scaled:input_route_gated` | `468` | `0.889611` | `1.315476` |
| `krr:input_route_stats:lambda=0.1` | `468` | `0.946113` | `1.634649` |
| `krr:input:lambda=0.1` | `468` | `0.946187` | `1.634866` |
| `krr:input_route_gated:lambda=0.1` | `468` | `0.950895` | `1.566818` |
| `krr:input_route_stats:lambda=1` | `468` | `0.956856` | `1.535904` |
| `krr:input:lambda=1` | `468` | `0.956872` | `1.536431` |
| `krr:input_route_gated:lambda=1` | `468` | `0.958382` | `1.541913` |
| `krr:input:lambda=10` | `468` | `1.013628` | `1.575504` |
| `krr:input_route_stats:lambda=10` | `468` | `1.013629` | `1.575486` |
| `krr:input_route_gated:lambda=10` | `468` | `1.013756` | `1.579193` |
| `krr:input:lambda=100` | `468` | `1.034084` | `1.592181` |
| `krr:input_route_stats:lambda=100` | `468` | `1.034085` | `1.592178` |
| `krr:input_route_gated:lambda=100` | `468` | `1.034095` | `1.592627` |
| `nn_raw:input` | `468` | `1.073053` | `2.239367` |
| `nn_raw:input_route_stats` | `468` | `1.073141` | `2.239367` |
| `nn_raw:input_route_gated` | `468` | `1.076084` | `2.239367` |

## Prototype Resident Estimate

| feature mode | feature dim | BF16 bytes/prototype with output | MiB/layer @ K=64 | GiB/60 layers @ K=64 |
|---|---:|---:|---:|---:|
| `input` | `2048` | `8192` | `0.500` | `0.029` |
| `input_route_stats` | `2073` | `8242` | `0.503` | `0.029` |
| `input_route_gated` | `12288` | `28672` | `1.750` | `0.103` |

## Worst Layer Rows

| layer | method | rows | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|
| `blk.46` | `nn_raw:input_route_gated` | `9` | `1.373654` | `1.885809` |
| `blk.48` | `nn_raw:input_route_gated` | `9` | `1.370692` | `2.239367` |
| `blk.48` | `nn_raw:input` | `9` | `1.361187` | `2.239367` |
| `blk.48` | `nn_raw:input_route_stats` | `9` | `1.361187` | `2.239367` |
| `blk.46` | `nn_raw:input` | `9` | `1.356306` | `1.858983` |
| `blk.46` | `nn_raw:input_route_stats` | `9` | `1.356306` | `1.858983` |
| `blk.47` | `nn_raw:input_route_gated` | `9` | `1.343991` | `1.665813` |
| `blk.49` | `nn_raw:input_route_gated` | `9` | `1.342746` | `1.717519` |
| `blk.54` | `nn_raw:input_route_gated` | `9` | `1.329215` | `1.630131` |
| `blk.49` | `nn_raw:input` | `9` | `1.325380` | `1.600511` |
| `blk.49` | `nn_raw:input_route_stats` | `9` | `1.325380` | `1.600511` |
| `blk.47` | `nn_raw:input` | `9` | `1.319970` | `1.665813` |
| `blk.47` | `nn_raw:input_route_stats` | `9` | `1.319970` | `1.665813` |
| `blk.54` | `nn_raw:input` | `9` | `1.312296` | `1.630131` |
| `blk.54` | `nn_raw:input_route_stats` | `9` | `1.312296` | `1.630131` |
| `blk.40` | `nn_raw:input` | `9` | `1.310852` | `1.548708` |
| `blk.40` | `nn_raw:input_route_stats` | `9` | `1.310852` | `1.548708` |
| `blk.51` | `nn_raw:input_route_gated` | `9` | `1.305766` | `1.382963` |
| `blk.51` | `nn_raw:input` | `9` | `1.303921` | `1.382963` |
| `blk.51` | `nn_raw:input_route_stats` | `9` | `1.303921` | `1.382963` |
| `blk.40` | `nn_raw:input_route_gated` | `9` | `1.303806` | `1.511780` |
| `blk.43` | `nn_raw:input` | `9` | `1.295107` | `1.511876` |
| `blk.43` | `nn_raw:input_route_stats` | `9` | `1.295107` | `1.511876` |
| `blk.58` | `nn_raw:input_route_gated` | `6` | `1.291448` | `1.456765` |
| `blk.44` | `nn_raw:input_route_gated` | `9` | `1.290516` | `1.476785` |
| `blk.55` | `nn_raw:input_route_gated` | `9` | `1.276513` | `1.356532` |
| `blk.44` | `nn_raw:input` | `9` | `1.276068` | `1.476785` |
| `blk.44` | `nn_raw:input_route_stats` | `9` | `1.276068` | `1.476785` |
| `blk.52` | `nn_raw:input` | `9` | `1.259956` | `1.529454` |
| `blk.52` | `nn_raw:input_route_stats` | `9` | `1.259956` | `1.529454` |
| `blk.55` | `nn_raw:input` | `9` | `1.258049` | `1.336340` |
| `blk.55` | `nn_raw:input_route_stats` | `9` | `1.258049` | `1.336340` |
| `blk.52` | `nn_raw:input_route_gated` | `9` | `1.254352` | `1.529454` |
| `blk.45` | `nn_raw:input` | `9` | `1.253819` | `1.439188` |
| `blk.45` | `nn_raw:input_route_stats` | `9` | `1.253819` | `1.439188` |
| `blk.53` | `nn_raw:input_route_gated` | `9` | `1.252403` | `1.488131` |
| `blk.50` | `nn_raw:input_route_gated` | `9` | `1.250135` | `1.383903` |
| `blk.45` | `nn_raw:input_route_gated` | `9` | `1.249354` | `1.439188` |
| `blk.43` | `nn_raw:input_route_gated` | `9` | `1.247088` | `1.456527` |
| `blk.41` | `nn_raw:input` | `9` | `1.246818` | `1.555132` |
| `blk.41` | `nn_raw:input_route_stats` | `9` | `1.246818` | `1.555132` |
| `blk.58` | `nn_raw:input` | `6` | `1.244115` | `1.436298` |
| `blk.58` | `nn_raw:input_route_stats` | `6` | `1.244115` | `1.436298` |
| `blk.50` | `nn_raw:input` | `9` | `1.243198` | `1.321471` |
| `blk.50` | `nn_raw:input_route_stats` | `9` | `1.243198` | `1.321471` |
| `blk.53` | `nn_raw:input` | `9` | `1.230302` | `1.398905` |
| `blk.53` | `nn_raw:input_route_stats` | `9` | `1.230302` | `1.398905` |
| `blk.39` | `nn_raw:input_route_gated` | `9` | `1.230070` | `1.371393` |
| `blk.41` | `nn_raw:input_route_gated` | `9` | `1.222180` | `1.445952` |
| `blk.56` | `nn_raw:input` | `9` | `1.216935` | `1.522854` |

## Decision

Reject as primary: best input-route surrogate nn_scaled:input_route_stats has mean rel L2 0.885045, above the 0.1 gate. This closes the small prototype/kernel full-MoE-output surrogate family for now.

## Reproduce

Corpus:

```bash
cd /root/lfz/tmp/kimi-stage2m-align
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/tmp/kimi-stage2m-align \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2 \
  --mode dev \
  --n 16 \
  --max-prompts 3 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 600 \
  --extra-runtime-env "GGML_MOE_ACTIVATION_DUMP_DIR=\$RUN
GGML_MOE_ACTIVATION_DUMP_MAX_RECORDS=4096
GGML_MOE_ACTIVATION_DUMP_CALL_STRIDE=1
GGML_MOE_ACTIVATION_DUMP_DECODE_ONLY=1"
```

Oracle:

```bash
.Agent/run-tools/kimi_input_route_moe_surrogate_oracle.py --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_france_regression --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_japan_factual --prompt-root /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-groupcomplete-activation-corpus-r2/dev_photosynthesis_factual --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --libggml-base build-cuda-batch/bin/libggml-base.so --out-json /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-input-route-moe-surrogate/report.json --out-md /root/lfz/runs/vendor-kimi-token-rate/20260708-gp105-input-route-moe-surrogate/report.md --max-records-per-prompt 4096 --complete-group-only --modes input,input_route_stats,input_route_gated --lambdas 0.1,1.0,10.0,100.0 --torch-threads 8
```
