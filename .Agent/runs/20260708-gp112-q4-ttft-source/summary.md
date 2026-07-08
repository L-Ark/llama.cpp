# GP112 Q4 TTFT Source Summary

Status: accepted as the next prompt-general SOTA env after full held-out paired
validation and France regression validation.

## Inputs

- Historical accepted GP4 held-out baseline:
  `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- GP110 full Q4 held-out n96 logs copied from:
  `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp110-q4-down-batch-heldout-n96`
- GP111 decode-only Q4 logs copied from:
  `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp111-q4-decode-only-heldout-smoke`
- New paired current-HEAD runs:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp112-q4-paired-brazil-n96`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp112-q4-paired-coding-n96`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp112-q4-paired-heldout-n96`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp112-q4-paired-france-n96`

## Historical GP4 Comparison

GP110/GP111 versus the historical GP4 accepted baseline still fails the TTFT
gate:

- GP110 full Q4:
  - mean token rate `1.367 -> 1.802 tok/s`;
  - mean TTFT `98156.1 -> 119898.4 ms`;
  - TTFT gate fails on `5/6` prompts.
- GP111 decode-only Q4:
  - mean token rate `1.310 -> 1.770 tok/s` over the two worst prompts;
  - mean TTFT `68634.4 -> 95308.6 ms`;
  - TTFT gate fails on `2/2` prompts.

The source differential does not support "decode IO got slower" as the cause:

- GP110 mean io_uring wait decreased `65394.7 -> 38532.2 ms`;
- GP111 mean io_uring wait decreased `74790.0 -> 44179.4 ms`;
- prompt fallback time increased at the same time, across up/gate/down and
  several tensor types, not only Q4_0 down.

## Full Held-Out Paired Comparison

The same current HEAD, same prompt, same cold-start script, and same 16 GB
cgroup were tested with only `GGML_MOE_Q4_DOWN_BATCH=1` changed.

Full held-out paired result:

- prompts: all six held-out prompts;
- quality: `pass` for all twelve q4-off/q4-on runs;
- Host RAM peak: `15899996160` for all runs;
- `direct_reads=0` for all runs;
- mean token rate: `1.713 -> 1.820 tok/s`;
- min token rate: `1.520 -> 1.600 tok/s`;
- mean TTFT: `108041.2 -> 105251.7 ms`;
- max TTFT ratio: `1.131`, within the `+20%` gate;
- mean decode time: `51468.0 -> 47041.3 ms`;
- mean decode fallback: `4991.6 -> 0.0 ms`;
- cold-start SOTA audit: `acceptance_safe=True`.

Per prompt:

| prompt | q4 off tok/s | q4 on tok/s | TTFT ratio | decode delta |
|---|---:|---:|---:|---:|
| `test_chinese_01` | `1.81` | `1.88` | `1.008` | `-2055.2 ms` |
| `test_coding_01` | `1.52` | `1.60` | `0.997` | `-3168.8 ms` |
| `test_english_factual_01` | `1.84` | `1.96` | `1.131` | `-3215.3 ms` |
| `test_english_factual_02` | `1.79` | `1.85` | `1.057` | `-14059.4 ms` |
| `test_mixed_instruction_01` | `1.72` | `1.86` | `0.927` | `-4276.4 ms` |
| `test_reasoning_math_01` | `1.60` | `1.77` | `0.894` | `+214.8 ms` |

France regression:

- remote root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp112-q4-paired-france-n96`;
- output quality: pass for q4 off and q4 on;
- Host RAM peak: `15899996160`;
- `direct_reads=0`;
- token rate: `1.82 -> 1.91 tok/s`;
- TTFT: `76005.02 -> 81816.45 ms`, ratio `1.076`;
- prompt:
  `Please introduce France in a short paragraph.`

## Decision

Accept Q4 decode-only as the next SOTA reproduction env.

Implementation:

- add `GGML_MOE_Q4_DOWN_BATCH=1` to
  `.Agent/run-tools/kimi-general-prompt-repro.sh`;
- keep the underlying CUDA Q4 path env-gated, so non-SOTA/default llama.cpp
  behavior remains unchanged.

This is a moderate improvement, not a path to `5 tok/s` by itself. It should be
kept while the next optimization phase continues to target structural byte
reduction or a compatible lower-byte model path.
