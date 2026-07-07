# GP103 Complete Lower-Quant Candidate Refresh and Disk Gate

Timestamp: `2026-07-08T04:40:00+0800`.

Status: completed non-destructive candidate refresh; no deletion and no full
model download.

## Purpose

Refresh complete lower-quant Kimi-K2.7-Code GGUF candidates after GP100-GP102
closed the learned down-replacement family. This is a disk/compatibility gate
for the full-model-quant path. It does not change runtime behavior or claim
SOTA.

## Disk Gate

Remote disk check:

```bash
ssh -p 51056 root@92.180.27.82 \
  'df -B1 /root/lfz/models /root/lfz/tmp; \
   du -sb /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S 2>/dev/null || true; \
   du -sb /root/lfz/models 2>/dev/null || true'
```

Result:

```text
Filesystem         1B-blocks         Used    Available Use% Mounted on
/dev/root      1065418129408 821409554432 243991797760  78% /
/dev/root      1065418129408 821409554432 243991797760  78% /
405392141472    /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S
405839599243    /root/lfz/models
```

Current available bytes: `243991797760` (`227.235 GiB`).

Required safety margin for failed/resumable partials, logs, and runtime scratch:
`50 GiB` (`53687091200` bytes).

Maximum final model size that can pass without deleting current SOTA assets:

```text
243991797760 - 53687091200 = 190304706560 bytes
190304706560 bytes = 177.235 GiB
```

## Metadata Sources

HF API/search endpoints queried non-destructively from the remote host:

- `https://huggingface.co/api/models?search=Kimi-K2.7-Code%20GGUF&limit=50&full=true`
- `https://huggingface.co/api/models/mradermacher/Kimi-K2.7-Code-i1-GGUF?expand=siblings`
- `https://huggingface.co/api/models/unsloth/Kimi-K2.7-Code-GGUF?expand=siblings`
- `https://huggingface.co/api/models/NullVoider/Kimi-K2.7-Code-GGUF?expand=siblings`
- `https://huggingface.co/api/models/huihui-ai/Huihui-Kimi-K2.7-Code-abliterated-GGUF?expand=siblings`
- `https://huggingface.co/api/models/freakyskittle/kimi-k2.75-code-GGUF?expand=siblings`

Per-file sizes were taken from resolver headers using:

```bash
curl -sS -D - -o /dev/null --max-time 30 \
  -H 'Range: bytes=0-4095' \
  'https://huggingface.co/<repo>/resolve/main/<file>'
```

and summing `x-linked-size`.

## Candidate Size Table

| Candidate | Repo | Files | Size bytes | GiB | Remaining GiB | Disk gate |
|---|---|---:|---:|---:|---:|---|
| `i1-IQ1_S` | `mradermacher/Kimi-K2.7-Code-i1-GGUF` | 5 | `204430872480` | `190.391` | `36.844` | fail |
| `i1-IQ1_M` | `mradermacher/Kimi-K2.7-Code-i1-GGUF` | 5 | `227937931680` | `212.284` | `14.951` | fail |
| `i1-IQ2_XXS` | `mradermacher/Kimi-K2.7-Code-i1-GGUF` | 6 | `267116363680` | `248.771` | `-21.536` | fail |
| `UD-IQ1_M` | `unsloth/Kimi-K2.7-Code-GGUF` | 8 | `303909170464` | `283.037` | `-55.802` | fail |
| `UD-IQ2_XXS` | `unsloth/Kimi-K2.7-Code-GGUF` | 8 | `317825871168` | `295.998` | `-68.763` | fail |
| `UD-IQ1_M` | `NullVoider/Kimi-K2.7-Code-GGUF` | 8 | `303909170464` | `283.037` | `-55.802` | fail |
| `UD-IQ1_M-MXFP4` | `huihui-ai/Huihui-Kimi-K2.7-Code-abliterated-GGUF` | 8 | `303627382432` | `282.775` | `-55.540` | fail |
| `deep55 pruned` | `freakyskittle/kimi-k2.75-code-GGUF` | 1 | `202657355840` | `188.739` | `38.496` | fail |
| `pruned compact oxidize q4` | `freakyskittle/kimi-k2.75-code-GGUF` | 1 | `405650677344` | `377.792` | `-150.557` | fail |

Notes:

- `i1-IQ1_S` remains the closest complete original-model candidate, but it
  leaves only `36.844 GiB` free, below the `50 GiB` safety gate.
- `deep55 pruned` is slightly smaller than `i1-IQ1_S`, but it is a pruned
  `kimi-k2.75-code` derivative, not the current Kimi-K2.7-Code original model,
  and still leaves only `38.496 GiB`, below the safety gate.
- Unsloth/NullVoider `UD-IQ1_M` and Huihui MXFP4 candidates are much larger than
  `i1-IQ1_S`.

## Compatibility Observations

Current code has generic CUDA entries for `IQ1_M` and `MXFP4`, but this is not
enough to claim Kimi vendor stream compatibility:

- `i1-IQ1_S` already passed prior GP32 metadata checks and appears plausible for
  the current Kimi MoE stream path.
- `IQ1_M`/`MXFP4` would need header/tensor-type and Kimi MoE stream-path checks
  before any runtime smoke.
- All candidates fail the disk gate first, so no deeper compatibility work was
  performed in this step.

## Decision

No complete lower-quant candidate can be downloaded safely on the current disk
while preserving IQ3_S SOTA assets and a `50 GiB` safety margin.

The full-model-quant runtime path remains gated on one of:

- explicit approval to delete non-SOTA remote artifacts;
- a separate/external storage target;
- a newly discovered complete candidate below `177.235 GiB` with compatible
  tensor types and acceptable quality risk.

## Reproduce

Use branch `vendor/kimi-speculative-general-token-rate-16gb`, then run:

```bash
python3 .Agent/run-tools/kimi_hf_candidate_size_probe.py \
  --out-json /tmp/kimi-hf-candidate-size-probe.json
```

The tool queries the HF repos listed above, filters GGUF siblings by
quant/prefix, requests resolver headers with `Range: bytes=0-4095`, and sums
`x-linked-size`.
