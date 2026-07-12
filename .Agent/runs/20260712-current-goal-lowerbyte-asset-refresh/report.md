# Kimi Lower-Byte Asset Refresh

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit: `d88337fbc`

## Purpose

After closing scattered RAM-tier and scheduler-only paths as primary routes,
refresh complete-model lower-byte candidate metadata and re-check whether a
destructive storage cleanup/download is required before the next runtime smoke.

This report is non-destructive:

- no model payload was downloaded;
- no pack/model file was deleted;
- no runtime SOTA benchmark was run;
- Hugging Face access was metadata/header only.

## Artifacts

- `.Agent/runs/20260712-current-goal-lowerbyte-asset-refresh/hf-candidate-size-probe.json`
- `.Agent/runs/20260712-current-goal-lowerbyte-asset-refresh/hf-candidate-size-probe.stdout`
- `.Agent/runs/20260712-current-goal-lowerbyte-asset-refresh/hf-candidate-size-summary.md`
- `.Agent/runs/20260712-current-goal-lowerbyte-asset-refresh/iq1s-guarded-execute0-refresh.log`
- `.Agent/runs/20260712-current-goal-lowerbyte-asset-refresh/iq1s-guarded-execute0-refresh.exit`

## Candidate Metadata

Current IQ3_S model directory size used for ratios:

- `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S`
- `405392165559 bytes`

Metadata refresh result:

| candidate | source | size GiB | ratio vs IQ3_S | status |
|---|---|---:|---:|---|
| `i1-IQ1_S` | `mradermacher/Kimi-K2.7-Code-i1-GGUF` | `190.39` | `0.504x` | best complete-model `2 tok/s` smoke candidate |
| `i1-IQ1_M` | `mradermacher/Kimi-K2.7-Code-i1-GGUF` | `212.28` | `0.562x` | weaker byte reduction |
| `i1-IQ2_XXS` | `mradermacher/Kimi-K2.7-Code-i1-GGUF` | `248.77` | `0.659x` | too large for current byte target |
| `unsloth UD-IQ1_M` | `unsloth/Kimi-K2.7-Code-GGUF` | `283.04` | `0.750x` | too large |
| `unsloth UD-IQ2_XXS` | `unsloth/Kimi-K2.7-Code-GGUF` | `296.00` | `0.784x` | too large |
| `NullVoider UD-IQ1_M` | `NullVoider/Kimi-K2.7-Code-GGUF` | `283.04` | `0.750x` | too large |
| `huihui UD-IQ1_M-MXFP4` | `huihui-ai/Huihui-Kimi-K2.7-Code-abliterated-GGUF` | `282.78` | `0.749x` | different/abliterated model and too large |
| `deep55 pruned` | `freakyskittle/kimi-k2.75-code-GGUF` | `188.74` | `0.500x` | model identity/quality risk, not first smoke |
| `pruned compact oxidize q4` | `freakyskittle/kimi-k2.75-code-GGUF` | `377.79` | `1.001x` | no byte gain |

The metadata refresh did not identify a new safer complete-model candidate. The
only same-model candidate still worth a runtime smoke is `i1-IQ1_S`.

## Storage Gate Refresh

Current free space:

- `/root/lfz` available: `112758599680 bytes` (`105.02 GiB`)

Guarded dry-run command:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
EXECUTE=0 \
DELETE_OLD_PACKS=1 \
CONFIRM_DELETE=DELETE_OLD_KIMI_NON_SOTA_PACKS \
DOWNLOAD=1 \
RUN_SMOKE=1 \
VALIDATE_PARTS=1 \
.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh
```

Dry-run exit: `0`.

Preserved paths:

| path | bytes |
|---|---:|
| `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S` | `405392165559` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | `175133036544` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | `4844539904` |

Delete candidates, only if explicitly confirmed:

| path | bytes |
|---|---:|
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack` | `0` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack` | `79544299520` |
| `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp146-dev-grouped-overlay-shadow/gp146-france-routefirst-overlay.expert-pack` | `81460133888` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack` | `7689551872` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack` | `5042724864` |

Projected totals from the dry-run:

- delete candidate total: `173736710144 bytes`;
- projected free after candidate deletion: `286495305728 bytes`;
- `i1-IQ1_S` required bytes: `204430872480 bytes`;
- projected leftover after download: `82064433248 bytes` (`76.43 GiB`);
- projected space gate: pass for the configured `50 GiB` reserve.

The log also prints `space_ready=0` later because `EXECUTE=0` means no deletion
was actually performed. This is expected and does not contradict the projected
space gate.

## Decision

Current state:

- same-model complete lower-byte runtime smoke still requires explicit approval
  for cleanup/download;
- `i1-IQ1_S` remains the only concrete complete-model `2 tok/s` smoke candidate;
- `deep55 pruned` is similar in size but changes model identity/quality risk,
  so it should not be the first task-aligned smoke;
- no candidate in this refresh is close to the `~0.25x` effective ratio needed
  for the long-term `5 tok/s` target.

Without explicit cleanup/download approval, do not execute the `i1-IQ1_S`
smoke. The next non-destructive work should be a new lower-byte representation
design/screen that targets:

- all-role effective moved-byte ratio near `0.50x` for the `2 tok/s` milestone;
- much closer than previous `IQ1_S/Q2_K` and blockwise residual attempts on
  activation-output error;
- up/gate support, not down-only.

## Reproduce

```bash
cd /root/lfz/llama.cpp-vendor-kimi
OUT=.Agent/runs/20260712-current-goal-lowerbyte-asset-refresh
mkdir -p "$OUT"

python3 .Agent/run-tools/kimi_hf_candidate_size_probe.py \
  --timeout-s 60 \
  --out-json "$OUT/hf-candidate-size-probe.json" \
  > "$OUT/hf-candidate-size-probe.stdout"

python3 /tmp/hf_candidate_summary.py \
  "$OUT/hf-candidate-size-probe.json" \
  "$OUT/hf-candidate-size-summary.md"

EXECUTE=0 \
DELETE_OLD_PACKS=1 \
CONFIRM_DELETE=DELETE_OLD_KIMI_NON_SOTA_PACKS \
DOWNLOAD=1 \
RUN_SMOKE=1 \
VALIDATE_PARTS=1 \
.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh \
  > "$OUT/iq1s-guarded-execute0-refresh.log" 2>&1
echo $? > "$OUT/iq1s-guarded-execute0-refresh.exit"
```
