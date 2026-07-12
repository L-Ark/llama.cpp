# Kimi Current-Goal Predictor N96 Refresh

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Source commit: `394fe292a`

This is a dev-only offline admission. It does not change runtime behavior and
does not claim SOTA.

## Goal

After the current N96 RAM slab refresh rejected static RAM-resident layer/role
slabs, test whether route-history prediction is strong enough to justify a
runtime prefetch path that covers complete future batches.

Admission gate:

- all-role byte recall `>=65%`;
- predicted/actual bytes `<=1.35x`;
- full-step coverage `>=40%`;
- no held-out/test prompt traces.

## Inputs

Trace root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-slab-iotrace-n96-074132
```

Prompts:

- `dev_france_regression`;
- `dev_intelligence_general`.

The same fresh N96 traces passed quality and stayed under the 16GB RAM cap:

| prompt | quality | tok/s | TTFT ms | RAM peak GiB |
|---|---|---:|---:|---:|
| `dev_france_regression` | pass | `1.68` | `8939.01` | `11.933` |
| `dev_intelligence_general` | pass | `1.68` | `7304.90` | `11.773` |

Wait reference used for the hybrid screen:

```text
504.616 ms/token
```

## Previous Same-Tensor Predictor

Artifact:

```text
.Agent/runs/20260712-current-goal-predictor-n96-refresh/previous-same-tensor.md
```

Command:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-slab-iotrace-n96-074132
OUT=.Agent/runs/20260712-current-goal-predictor-n96-refresh
python3 .Agent/run-tools/kimi_expert_predictability_from_trace.py \
  --trace "$ROOT/dev_france_regression/route-trace.csv" \
  --trace "$ROOT/dev_intelligence_general/route-trace.csv" \
  --out-json "$OUT/previous-same-tensor.json" \
  --out-csv "$OUT/previous-same-tensor.csv" \
  --out-md "$OUT/previous-same-tensor.md"
```

Result:

| scope | byte recall | precision | predicted/actual bytes |
|---|---:|---:|---:|
| all | `0.3260` | `0.3306` | `0.9898` |
| down | `0.3252` | `0.3306` | `0.9898` |
| gate | `0.3283` | `0.3306` | `0.9898` |
| up | `0.3245` | `0.3306` | `0.9898` |

Decision: reject. The byte recall is about half of the `65%` admission gate.

## Hybrid Route-History Predictor

Artifact:

```text
.Agent/runs/20260712-current-goal-predictor-n96-refresh/hybrid/report.md
```

Command:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-ram-slab-iotrace-n96-074132
OUT=.Agent/runs/20260712-current-goal-predictor-n96-refresh/hybrid
python3 .Agent/run-tools/kimi_future_hybrid_predictor_admission.py \
  --input-glob "$ROOT/dev_*/route-trace.csv" \
  --out "$OUT" \
  --layers 60 \
  --max-decode-experts 8 \
  --horizons 1 2 3 \
  --budgets 8 16 32 \
  --bundles all,upgate,down \
  --recent-window 4 \
  --io-wait-ms-per-token 504.616
```

Best all-role rows:

| predictor | horizon | budget | recall | precision | full steps | pred/actual bytes | predicted GiB/token | waste GiB/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_recent` | `3` | `32` | `0.6051` | `0.1513` | `3.48%` | `4.00x` | `84.029` | `71.345` |
| `hybrid_recent` | `2` | `32` | `0.6023` | `0.1506` | `3.46%` | `4.00x` | `85.547` | `72.696` |
| `hybrid_recent` | `1` | `32` | `0.6017` | `0.1504` | `3.31%` | `4.00x` | `87.066` | `73.999` |
| `hybrid_recent` | `3` | `16` | `0.4898` | `0.2449` | `0.58%` | `2.00x` | `42.015` | `31.754` |

Passing rows: `0`.

Interpretation:

- route-history can recover more useful bytes than the trivial previous-tensor
  predictor, but only by overfetching far too much;
- the best recall row still misses the `65%` recall gate and uses `4.00x`
  predicted bytes;
- full-step coverage is only `3.48%`, far below the `40%` complete-step gate;
- the linear wait-cover column in the hybrid report is not a runtime saving
  claim, because the false-positive bytes would add SSD/H2D work and fragment
  batches.

## Decision

Reject route-history predictor/prefetch as the next runtime path on current N96
evidence.

Do not implement a runtime prefetcher from route IDs alone. A future predictor
attempt needs a stronger signal and must pass the same complete-batch admission
before runtime reads:

- router logits or hidden-state features;
- a small draft/router model that predicts expert IDs directly;
- or a predictor combined with lower-byte expert representation so overfetch is
  cheap enough to be useful.

## Next Action

The only remaining non-approved routes with enough headroom are lower-byte
expert representation or a stronger predictor with real features. Without
explicit approval for the full `i1-IQ1_S` same-model smoke, the next
non-destructive step should screen a representation that reduces all active
expert bytes below `0.50x` with prompt-level output-error evidence.
