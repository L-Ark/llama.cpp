# GP111 Q4_0 Down Batch Decode-Only Gate

This is a default-off runtime refinement for `GGML_MOE_Q4_DOWN_BATCH=1`. It does
not claim SOTA.

- Branch: `vendor/kimi-speculative-general-token-rate-16gb`
- Base implementation commit: `1f4729c3f`
- Remote worktree: `/root/lfz/tmp/kimi-stage2m-align`
- Remote run root: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp111-q4-decode-only-heldout-smoke`
- Cgroup: `MemoryMax=15900000000`, `MemorySwapMax=0`

## Change

`Q4_0` down batch is now decode-only:

- if `src0->type == GGML_TYPE_Q4_0` and `ids->ne[1] > 1`, CPU-side
  down-batch eligibility stays false;
- `ids->ne[1] == 1` decode calls still use the Q4 down batch path when
  `GGML_MOE_Q4_DOWN_BATCH=1`;
- default behavior is unchanged when the env is unset.

## Hypothesis

GP110 improved token rate but failed the TTFT gate. The suspected overhead was
prompt/prefill Q4 eligibility: prompt calls have multirow routing and the CUDA
down batch path declines them, so allowing Q4 prompt eligibility could add TTFT
cost without removing prompt fallback.

## Build

```bash
ssh -p 51056 root@92.180.27.82 \
  'cd /root/lfz/tmp/kimi-stage2m-align && cmake --build build-cuda-batch --target llama-completion -j 16'
```

Result: build passed.

## Smoke Results

Two held-out TTFT-worst prompts were rerun at n96.

| prompt | GP4 tok/s | GP110 Q4 tok/s | GP111 tok/s | GP4 TTFT | GP110 TTFT | GP111 TTFT | quality |
|---|---:|---:|---:|---:|---:|---:|---|
| `test_english_factual_01` | `1.48` | `1.91` | `1.94` | `53196 ms` | `85678 ms` | `84346 ms` | pass |
| `test_coding_01` | `1.14` | `1.59` | `1.60` | `84073 ms` | `115983 ms` | `106272 ms` | pass |

Both runs stayed under the 16GB cgroup limit and had `direct_reads=0`.

## Mechanism Check

The decode path still triggers:

- stderr contains `Q4_0 down batch candidate active`;
- Q4 tensors show decode `batch_eligible=95 batch_accept=95`;
- prompt calls still use fallback, e.g. `prompt_calls=1 prompt_fallback=...`.

## Decision

Reject GP111 as a sufficient TTFT fix. It reduces TTFT slightly while preserving
token-rate gains, but still violates the `+20%` TTFT gate:

- Brazil remains `+58.6%` vs GP4 TTFT;
- coding remains `+26.4%` vs GP4 TTFT.

Keep the code default-off as a diagnostic/refinement, but do not promote
`GGML_MOE_Q4_DOWN_BATCH=1` to the SOTA environment. The next step must profile
TTFT before decode starts and identify the actual source of the Q4-on prompt
overhead.

## Reproduce

```bash
cd /root/lfz/tmp/kimi-stage2m-align
cmake --build build-cuda-batch --target llama-completion -j 16

BASE=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp111-q4-decode-only-heldout-smoke

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
  env REPO=/root/lfz/tmp/kimi-stage2m-align \
      RUN=$BASE/test_english_factual_01-q4-decodeonly-n96 \
      N=96 PROFILE=1 PROMPT_ID=test_english_factual_01 \
      PROMPT_USER_TEXT='Give a short overview of Brazil.' \
      QUALITY_KEYWORDS='Brazil|Brazilian,South America|Portuguese|Brasilia|Brasília' \
      EXTRA_RUNTIME_ENV='GGML_MOE_Q4_DOWN_BATCH=1' \
      .Agent/run-tools/kimi-general-prompt-repro.sh

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 -p RuntimeMaxSec=1200 \
  env REPO=/root/lfz/tmp/kimi-stage2m-align \
      RUN=$BASE/test_coding_01-q4-decodeonly-n96 \
      N=96 PROFILE=1 PROMPT_ID=test_coding_01 \
      PROMPT_USER_TEXT='Write a JavaScript function that returns the largest number in an array.' \
      QUALITY_KEYWORDS='function|JavaScript|javascript,array,largest|max|Math.max' \
      EXTRA_RUNTIME_ENV='GGML_MOE_Q4_DOWN_BATCH=1' \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```
