# GP39 IQ1_S synthetic curl rollback test

Timestamp: `2026-07-07T06:02:14+0800`.

Branch:
`vendor/kimi-speculative-general-token-rate-16gb`.

Commit tested on remote:
`b1e43fbaed6e30439979ec6cb2aee8e6cc761219`.

Remote worktree:
`/root/lfz/tmp/vendor-kimi-speculative-gp33`.

Command:

```bash
ssh -p 51056 root@92.180.27.82 \
  'cd /root/lfz/tmp/vendor-kimi-speculative-gp33 && \
   git fetch origin vendor/kimi-speculative-general-token-rate-16gb && \
   git checkout vendor/kimi-speculative-general-token-rate-16gb && \
   git reset --hard origin/vendor/kimi-speculative-general-token-rate-16gb && \
   .Agent/run-tools/kimi_iq1s_test_curl_rollback.sh'
```

Result:

- Exit code: `0`.
- Synthetic prepare script date: `2026-07-06T22:02:14+00:00`.
- `execute=1`.
- `delete_old_packs=0`.
- `download=1`.
- `run_smoke=0`.
- `validate_parts=0`.
- `resume_download=1`.
- Synthetic `required_iq1s_bytes=16`.
- Synthetic `space_ready=1`.
- Fake curl wrote partial stdout and exited with rc `23`.
- Prepare script exited non-zero as expected.
- Rollback log:
  `ERROR part 1 curl failed rc=23; truncating temp back to 0`.
- Test assertion:
  `[kimi_iq1s_test_curl_rollback] pass tmp_size=0 rc=1`.

Decision:

- The resumable download failure path is now covered by a reproducible synthetic
  test.
- A failed append does not leave corrupt partial bytes in `MODEL_PATH.tmp`.
- This removes one safety gap before any future real IQ1_S `EXECUTE=1`
  download.
- Full IQ1_S n32 smoke remains gated by disk capacity or explicit approval to
  delete old non-SOTA Kimi expert packs.
- No SOTA or token-rate claim is made from this phase.
