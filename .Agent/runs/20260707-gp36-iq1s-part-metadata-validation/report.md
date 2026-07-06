# GP36 IQ1_S part metadata validation

Timestamp: `2026-07-07T11:15:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Commit tested on remote:
`9aa90af34d3c8add20f294ea6ca94dc647e46c5b`.

Tool:

`.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh`

## Purpose

Validate the current Hugging Face `mradermacher` IQ1_S multipart metadata before
any destructive cleanup or full download.

## Remote Command

```bash
cd /root/lfz/tmp/vendor-kimi-speculative-gp33
.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh
```

## Result

- Exit code: `0`.
- No deletion executed.
- No download executed.
- No smoke executed.
- Repo head:
  `9aa90af34d3c8add20f294ea6ca94dc647e46c5b`.
- `required_iq1s_bytes=204430872480`.
- `free_before=89649844224`.

Part metadata validation:

| part | bytes |
| --- | ---: |
| `Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5` | `41875931136` |
| `Kimi-K2.7-Code.i1-IQ1_S.gguf.part2of5` | `41875931136` |
| `Kimi-K2.7-Code.i1-IQ1_S.gguf.part3of5` | `41875931136` |
| `Kimi-K2.7-Code.i1-IQ1_S.gguf.part4of5` | `41875931136` |
| `Kimi-K2.7-Code.i1-IQ1_S.gguf.part5of5` | `36927147936` |

Total:

- `part_total_ok bytes=204430872480`.

Disk gate:

- Required with `20 GiB` reserve:
  `225905708960`.
- Missing:
  `136255864736`.
- `space_ready=0`.
- `smoke_ready=0`, because model size is `0`.

## Correction

The earlier hardcoded final size `204429739520` was incorrect. The authoritative
size is the five-part Hugging Face metadata sum:

`204430872480`

The executor was corrected before any real download or smoke run.

## Decision

The part metadata gate passes. The full IQ1_S n32 smoke remains gated only by
disk capacity or explicit approval to delete old non-SOTA packs.
