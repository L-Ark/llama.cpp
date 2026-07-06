# GP37 IQ1_S resumable download guard

Timestamp: `2026-07-07T11:25:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Commit tested on remote:
`2072a294c151fff1a5ae2da7014688b6b2d19ca0`.

Tool:

`.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh`

## Purpose

Make the future `204430872480` byte IQ1_S download resumable without storing all
five parts plus the final GGUF at the same time.

## Implementation Summary

- Added `RESUME_DOWNLOAD=1` default.
- Download target remains:
  `/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF/Kimi-K2.7-Code.i1-IQ1_S.gguf`.
- Temporary path:
  `/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF/Kimi-K2.7-Code.i1-IQ1_S.gguf.tmp`.
- If the temp file exists:
  - fail if it is larger than the expected final size;
  - fail if `RESUME_DOWNLOAD=0`;
  - otherwise map the temp size to a completed part prefix and current part
    offset.
- Completed parts are skipped.
- Current partial part resumes with HTTP range.
- If an appended range has the wrong byte count, the temp file is truncated
  back to the pre-request size.
- Final size is validated before renaming temp to final model path.

## Remote Dry-Run

Command:

```bash
cd /root/lfz/tmp/vendor-kimi-speculative-gp33
.Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh
```

Result:

- Exit code: `0`.
- No deletion executed.
- No download executed.
- No smoke executed.
- `resume_download=1`.
- `part_total_ok bytes=204430872480`.
- `space_ready=0`.
- `free_before=89647194112`.
- `required=225905708960`.
- `missing=136258514848`.
- Dry-run download line:
  `resumable stream 5 parts into ... via ...Kimi-K2.7-Code.i1-IQ1_S.gguf.tmp`.

## Decision

The resumable-download guard is ready for future approved execution. The full
IQ1_S n32 smoke remains gated by disk capacity or explicit deletion approval.
