# GP35 guarded IQ1_S smoke prep dry-run

Timestamp: `2026-07-07T10:55:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Commit tested on remote:
`a891550541b1480a8174ba45d61af1c244414cc5`.

Tool:

`.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh`

## Default Dry-Run

Remote command:

```bash
cd /root/lfz/tmp/vendor-kimi-speculative-gp33
.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh
```

Result:

- Exit code: `0`.
- No deletion requested.
- No deletion executed.
- No download executed.
- No smoke run executed.
- Repo branch:
  `vendor/kimi-speculative-general-token-rate-16gb`.
- Repo head:
  `a891550541b1480a8174ba45d61af1c244414cc5`.
- Target model:
  `/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF/Kimi-K2.7-Code.i1-IQ1_S.gguf`.
- Required IQ1_S bytes:
  `204429739520`.
- Free bytes before action:
  `89653850112`.
- Required bytes including `20 GiB` post-download reserve:
  `225904576000`.
- Missing bytes:
  `136250725888`.
- `space_ready=0`.
- `smoke_ready=0`, because model size is `0`.

Preserved assets observed by the script:

| path | bytes |
| --- | ---: |
| `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S` | `405392165559` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | `175133036544` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | `4844539904` |

Deletion candidates observed by the script:

| path | bytes |
| --- | ---: |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack` | `171692638208` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack` | `79544299520` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack` | `7689551872` |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack` | `5042724864` |

## Delete Guard Negative Test

Remote command:

```bash
cd /root/lfz/tmp/vendor-kimi-speculative-gp33
DELETE_OLD_PACKS=1 .Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh
```

Result:

- Exit code: `1`.
- The script stopped with:
  `ERROR deletion requested without exact CONFIRM_DELETE token`.
- No deletion was executed.

## Decision

The guarded executor is ready for a future approved IQ1_S full-model smoke.
Current state remains gated by disk capacity or explicit approval to remove old
non-SOTA packs.

## Correction From GP36

GP36 added a Hugging Face metadata validation gate and found the authoritative
five-part total is `204430872480` bytes, not the earlier hardcoded
`204429739520` bytes used in this dry-run. The GP35 dry-run remains useful as a
deletion-guard validation record, but the final model-size constant was
corrected in the executor before any real download.
