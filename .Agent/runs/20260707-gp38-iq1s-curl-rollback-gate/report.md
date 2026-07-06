# GP38 IQ1_S resumable curl rollback gate

Timestamp: `2026-07-07T05:58:56+0800`.

Branch:
`vendor/kimi-speculative-general-token-rate-16gb`.

Commit tested on remote:
`24c9d0f1e61ce12cc6a6984a93cab45aa7e01afc`.

Remote worktree:
`/root/lfz/tmp/vendor-kimi-speculative-gp33`.

Command:

```bash
ssh -p 51056 root@92.180.27.82 \
  'cd /root/lfz/tmp/vendor-kimi-speculative-gp33 && \
   git fetch origin vendor/kimi-speculative-general-token-rate-16gb && \
   git checkout vendor/kimi-speculative-general-token-rate-16gb && \
   git reset --hard origin/vendor/kimi-speculative-general-token-rate-16gb && \
   .Agent/run-tools/kimi_iq1s_prepare_full_smoke.sh'
```

Result:

- Exit code: `0`.
- Script date: `2026-07-06T21:58:56+00:00`.
- `execute=0`.
- `delete_old_packs=0`.
- `download=1`.
- `run_smoke=1`.
- `validate_parts=1`.
- `resume_download=1`.
- No deletion executed.
- No download executed.
- No smoke executed.
- `repo_head=24c9d0f1e61ce12cc6a6984a93cab45aa7e01afc`.
- `required_iq1s_bytes=204430872480`.
- `free_before=89644806144`.
- `part_total_ok bytes=204430872480`.
- `space_ready=0`.
- `free=89644802048`.
- `required=225905708960`.
- `missing=136260906912`.
- `smoke_ready=0`.
- `model_size=0`.
- `expected=204430872480`.

Validated part metadata:

- `Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5`: `41875931136`.
- `Kimi-K2.7-Code.i1-IQ1_S.gguf.part2of5`: `41875931136`.
- `Kimi-K2.7-Code.i1-IQ1_S.gguf.part3of5`: `41875931136`.
- `Kimi-K2.7-Code.i1-IQ1_S.gguf.part4of5`: `41875931136`.
- `Kimi-K2.7-Code.i1-IQ1_S.gguf.part5of5`: `36927147936`.

Dry-run download shape:

```text
mkdir -p '/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF'
resumable stream 5 parts into '/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF/Kimi-K2.7-Code.i1-IQ1_S.gguf' via '/root/lfz/models/Kimi-K2.7-Code-i1-IQ1_S-GGUF/Kimi-K2.7-Code.i1-IQ1_S.gguf.tmp'
```

Decision:

- The default guarded dry-run remains reproducible after the curl rollback
  implementation.
- The rollback path itself is implemented but not destructively exercised by
  this default dry-run.
- Full IQ1_S n32 smoke remains gated by disk capacity or explicit approval to
  delete old non-SOTA Kimi expert packs.
- No SOTA or token-rate claim is made from this phase.
