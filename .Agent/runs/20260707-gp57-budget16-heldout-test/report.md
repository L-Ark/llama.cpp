# GP57 budget16 overlay held-out test gate

Status: rejected for final SOTA.

This was a one-time held-out test-set gate for the frozen GP55/GP56 budget16
overlay. No test output was used to tune the candidate.

## Candidate

```text
GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack
GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1
```

## Result Summary

Baseline is `.Agent/runs/20260707-gp4-postcommit-test-n96-profile/summary.md`.

| prompt | quality | baseline tok/s | overlay tok/s | baseline TTFT ms | overlay TTFT ms | TTFT gate | RAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_english_factual_01` | pass | `1.48` | `0.43` | `53196.00` | `66832.04` | fail | `15899996160` |
| `test_english_factual_02` | pass | `1.49` | `0.26` | `69820.25` | `76615.78` | pass | `15899996160` |
| `test_reasoning_math_01` | pass | `1.32` | `0.22` | `231632.04` | `230011.77` | pass | `15899996160` |
| `test_coding_01` | pass | `1.14` | `0.17` | `84072.90` | `91520.20` | pass | `15899996160` |
| `test_chinese_01` | pass | `1.44` | `0.27` | `70343.09` | `74465.20` | pass | `15899996160` |
| `test_mixed_instruction_01` | pass | `1.33` | `0.25` | `79872.16` | `94611.12` | pass | `15899996160` |

Aggregate token rate:

- baseline arithmetic mean: `1.3667 tok/s`;
- overlay arithmetic mean: `0.2667 tok/s`.

Gate result:

- Quality: pass for all six prompts.
- RAM: under the configured `15900000000` cgroup limit for all prompts, but
  with no margin.
- TTFT: fail because `test_english_factual_01` rose from `53196.00 ms` to
  `66832.04 ms`, exceeding the `+20%` limit of `63835.20 ms`.
- Token rate: fail because every prompt regressed by far more than `5%`, and
  aggregate mean regressed from `1.3667` to `0.2667 tok/s`.

## Decision

- The budget16 overlay is rejected as a final prompt-agnostic SOTA candidate.
- Do not enable this overlay for random user prompts.
- Runtime rollback/disable path is simply to omit:

```text
GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack
```

- The builder and reports remain useful for analysis because no default runtime
  behavior was changed by the rejected candidate.
- Next investigation must return to dev-only evidence. Do not tune on these
  held-out outputs.

## Reproduction

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
out=/root/lfz/tmp/runs/20260707-gp57-budget16-heldout-test
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo "$repo" \
  --prompt-file .Agent/evals/kimi-general-test-prompts.jsonl \
  --out-root "$out" \
  --mode test \
  --n 96 \
  --profile \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 900 \
  --extra-runtime-env "GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1"
```
