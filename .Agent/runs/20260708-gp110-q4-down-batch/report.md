# GP110 Q4_0 Down Batch Probe

This is a default-off runtime probe. It does not claim a new SOTA.

- Branch: `vendor/kimi-speculative-general-token-rate-16gb`
- Local base commit: `30f5398da`
- Remote worktree: `/root/lfz/tmp/kimi-stage2m-align`
- Remote run root: `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp110-q4-down-batch`
- Gate: `GGML_MOE_Q4_DOWN_BATCH=1`
- Model: `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf`
- Cgroup: `MemoryMax=15900000000`, `MemorySwapMax=0`
- Prompt scope: dev smoke only, no held-out SOTA claim

## Implementation

- `ggml/src/ggml-cpu/ggml-cpu.c`
  - allows `Q4_0` `ffn_down_exps` tensors to reach the CUDA down batch path
    only when `GGML_MOE_Q4_DOWN_BATCH=1`;
  - optional `GGML_MOE_Q4_DOWN_BATCH_TENSOR=<tensor>` keeps single-tensor
    targeting available for debug.
- `ggml/src/ggml-cuda/moe_stream_batch.cu`
  - accepts the same gated `Q4_0` down candidate;
  - default behavior is unchanged when the env is unset;
  - up/gate type support is not broadened.

## Build

```bash
ssh -p 51056 root@92.180.27.82 \
  'cd /root/lfz/tmp/kimi-stage2m-align && cmake --build build-cuda-batch --target llama-completion -j 16'
```

Result: build passed.

## Smoke Results

Default-off baseline rows use the GP109 `VRAM_MIB=15000` run where available.
France also has a paired GP110 default-off run; it was slower than GP109
(`1.70 tok/s` vs `1.79 tok/s`), so the table uses GP109 for the less favorable
comparison against Q4.

| prompt | baseline tok/s | Q4 tok/s | delta | baseline TTFT | Q4 TTFT | quality | Q4 iouring bytes | Q4 down hit | Q4 upgate hit |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| France | `1.79` | `1.83` | `+2.2%` | `81917 ms` | `83002 ms` | pass | `157.1 GB` | `63.4%` | `41.5%` |
| Moon phases | `1.77` | `1.82` | `+2.8%` | `82571 ms` | `95717 ms` | pass | `156.2 GB` | `64.1%` | `41.0%` |
| AI infra | `1.77` | `1.88` | `+6.2%` | `98662 ms` | `96098 ms` | manual pass, strict keyword gate failed | `152.8 GB` | `62.9%` | `43.6%` |

Mean token rate over these three dev prompts:

- baseline: about `1.78 tok/s`;
- Q4 gated: about `1.84 tok/s`;
- mean lift: about `+3.7%`.

All runs stayed under the 16GB cgroup limit and had `direct_reads=0`.

## Mechanism Check

The fixed CPU-side gate is required. A CUDA-only gate did not trigger because
the CPU eligibility function rejected `Q4_0` before calling CUDA.

After the CPU gate fix:

- stderr contains:
  - `[moe_stream_batch] Q4_0 down batch candidate active tensor=blk.6.ffn_down_exps.weight ...`
- France `decode,type=2` fallback:
  - baseline paired default-off: `2338.648 ms`;
  - Q4 gated: removed from fallback CSV.
- Q4 down tensors become batch eligible, for example:
  - `blk.6.ffn_down_exps.weight batch_eligible=32 batch_accept=31`;
  - `blk.7.ffn_down_exps.weight batch_eligible=32 batch_accept=31`;
  - `blk.8.ffn_down_exps.weight batch_eligible=32 batch_accept=31`;
  - `blk.9.ffn_down_exps.weight batch_eligible=32 batch_accept=31`;
  - `blk.10.ffn_down_exps.weight batch_eligible=32 batch_accept=31`;
  - `blk.15.ffn_down_exps.weight batch_eligible=32 batch_accept=31`;
  - `blk.18.ffn_down_exps.weight batch_eligible=32 batch_accept=31`.

## Cost

The probe is not free:

- down slot size grows from `7.44 MiB` to `7.88 MiB`;
- Q4 gated France iouring bytes rise from `131.1 GB` baseline to `157.1 GB`;
- down hit rate falls from `73.4%` to `63.4%` on France;
- up/gate hit rate also falls on France (`45.2%` to `41.5%`), likely due
  changed cache pressure and route/output drift.

The speedup comes from removing CPU decode fallback despite moving more bytes.
This is useful cleanup, but it is not a path to `5 tok/s` by itself.

## Decision

Keep `GGML_MOE_Q4_DOWN_BATCH` default-off and commit the implementation plus
this reproducible report. Do not add it to the SOTA environment yet.

Next validation before promotion:

1. Run a full dev n96 A/B with `GGML_MOE_Q4_DOWN_BATCH=0/1`.
2. If dev n96 passes quality, TTFT, RAM, and token-rate gates, run held-out n96.
3. Promote only if held-out prompt-general mean improves without quality loss or
   TTFT > `+20%`.

## Reproduce

```bash
cd /root/lfz/tmp/kimi-stage2m-align
cmake --build build-cuda-batch --target llama-completion -j 16

BASE=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp110-q4-down-batch

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 -p RuntimeMaxSec=900 \
  env REPO=/root/lfz/tmp/kimi-stage2m-align \
      RUN=$BASE/france-n32-q4-on-cpugate \
      N=32 PROFILE=1 PROMPT_ID=dev_france_q4_on_cpugate \
      PROMPT_USER_TEXT='Please introduce France in a short paragraph.' \
      QUALITY_KEYWORDS='France|French,Europe|Paris|culture|country' \
      EXTRA_RUNTIME_ENV='GGML_MOE_Q4_DOWN_BATCH=1' \
      .Agent/run-tools/kimi-general-prompt-repro.sh

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 -p RuntimeMaxSec=900 \
  env REPO=/root/lfz/tmp/kimi-stage2m-align \
      RUN=$BASE/moon-n32-q4-on-cpugate \
      N=32 PROFILE=1 PROMPT_ID=moon \
      PROMPT_USER_TEXT='Briefly explain why the Moon has phases.' \
      QUALITY_KEYWORDS='Moon|moon,Sun|sun,Earth|earth,phase|phases' \
      EXTRA_RUNTIME_ENV='GGML_MOE_Q4_DOWN_BATCH=1' \
      .Agent/run-tools/kimi-general-prompt-repro.sh

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 -p RuntimeMaxSec=900 \
  env REPO=/root/lfz/tmp/kimi-stage2m-align \
      RUN=$BASE/aiinfra-n32-q4-on-cpugate \
      N=32 PROFILE=1 PROMPT_ID=aiinfra \
      PROMPT_USER_TEXT='What is AI infrastructure? Answer in one short paragraph.' \
      QUALITY_KEYWORDS='AI|infrastructure,compute|servers|cloud|network|data,models|systems|hardware' \
      EXTRA_RUNTIME_ENV='GGML_MOE_Q4_DOWN_BATCH=1' \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```
