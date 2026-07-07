# GP108 re-encode and v2 mixed-type path audit

Date: 2026-07-08

This is a planning audit based on existing committed experiments. It does not
change runtime behavior and does not use held-out prompts for tuning.

## Question

Can we avoid the full `i1-IQ1_S` download gate by either:

- re-encoding current IQ3/IQ2 expert payloads into a smaller representation; or
- using the existing default-off `GGMLMOEPACKv2` mixed-type machinery with a
  selected low-byte expert hotset?

## Evidence

### Existing IQ3/IQ2 re-encoding

Evidence: `.Agent/runs/20260707-gp11-quant-reencode-bound/report.md`.

The current tooling can dequantize selected current GGUF expert weights and
apply simple lower-bit blockwise re-encoding, but the error is far too high at
the required byte ratios:

- 1-bit blockwise re-encode:
  - ratio: `0.347x-0.488x`;
  - rel L2: `1.777-2.275`;
  - rejected.
- 2-bit blockwise re-encode:
  - ratio: `0.673x-0.878x`;
  - rel L2: `0.687-0.791`;
  - rejected.
- 4-bit can approach rel L2 around `0.09-0.13`, but its byte ratio is larger
  than the current payload for the sampled IQ2/IQ3 tensors and does not help
  the `5 tok/s` byte budget.

Conclusion: direct simple re-encoding from the existing IQ3/IQ2 expert weights
is not a viable primary route. A future re-encoding route would need a much
stronger quantizer with activation-output gates, not only weight rel L2.

### v2 mixed-type selected expert packs

Evidence:

- `.Agent/runs/20260707-gp41-mixed-type-pack-audit/report.md`
- `.Agent/runs/20260707-gp42-moepack-v2-parser-skeleton/report.md`
- `.Agent/runs/20260707-gp43-moepack-v2-read-debug/report.md`
- `.Agent/runs/20260707-gp44-v2-shadow-profile/report.md`
- `.Agent/runs/20260707-gp46-v2-hotset-sweep/report.md`

The v2 parser/debug/shadow stack exists in the current branch. It is
default-off and does not affect the accepted SOTA path.

However, GP46 showed that selected low-byte hotsets alone do not have enough
headroom:

| packed ratio | selected entries | hybrid byte ratio | ideal transfer-only tok/s |
|---:|---:|---:|---:|
| `0.55` | `56896` | `0.5500` | `2.518` |
| `0.35` | `56896` | `0.3500` | `3.957` |
| `0.276` | `56896` | `0.2760` | `5.018` |

The `0.276x` row is an optimistic transfer-only bound with nearly complete
coverage. It is lower than realistic selected `IQ1_S` payload ratios and still
does not include non-transfer decode time. Therefore v2 selected hotsets alone
are not a sufficient primary route to stable `5 tok/s`.

## Decision

Do not restart runtime work for:

- naive current-IQ3/IQ2 re-encoding; or
- selected v2 mixed-type hotset execution as a standalone path.

The remaining useful directions are:

1. Full-model lower-quant smoke, currently gated by disk/deletion approval.
2. A new compute/storage-form change that reduces active expert demand or
   preserves activation-output accuracy at about `0.30x-0.40x` moved bytes.
3. Prediction/scheduling only as a secondary multiplier after bytes are reduced,
   because prior bounds show scheduling alone cannot reach `5 tok/s`.

## Reproduce

```bash
sed -n '1,180p' .Agent/runs/20260707-gp11-quant-reencode-bound/report.md
sed -n '1,230p' .Agent/runs/20260707-gp41-mixed-type-pack-audit/report.md
sed -n '1,130p' .Agent/runs/20260707-gp46-v2-hotset-sweep/report.md
rg -n "GGMLMOEPACKv2|packed_type|packed_nbytes|expert_pack_v2" ggml/src/ggml-cuda/moe_stream_batch.cu .Agent/run-tools
```

