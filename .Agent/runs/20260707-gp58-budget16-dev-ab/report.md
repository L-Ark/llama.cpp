# GP58 budget16 overlay dev-only A/B diagnosis

Status: diagnostic complete; no final SOTA claim.

GP57 rejected the budget16 overlay on the held-out test gate. GP58 returned to
dev-only evidence and tested whether the overlay has a uniform runtime overhead
on high or medium coverage dev prompts.

## Results

| prompt | mode | quality | tok/s | TTFT ms | decode ms/runs | misses | direct reads | iouring reads | iouring wait ms | RAM peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | default | pass | `1.27` | `81785.87` | `60514.89/77` | `633` | `348` | `58416` | `54594.17` | `15899996160` |
| `dev_france_regression` | budget16 | pass | `1.33` | `75849.10` | `57919.97/77` | `543` | `272` | `58582` | `53027.94` | `15899996160` |
| `dev_japan_factual` | default | pass | `0.44` | `77090.97` | `194889.58/85` | `13383` | `4095` | `48098` | `51664.48` | `15899996160` |
| `dev_japan_factual` | budget16 | pass | `0.53` | `76068.10` | `160610.34/85` | `10344` | `3392` | `51840` | `53259.14` | `15899996160` |

## Interpretation

- The budget16 overlay is not a uniform overhead:
  - it improved the high-coverage France dev prompt from `1.27` to `1.33 tok/s`;
  - it improved the medium-coverage Japan dev prompt from `0.44` to `0.53 tok/s`.
- The France improvement is small and comes from fewer misses/direct reads plus
  slightly lower iouring wait.
- The Japan improvement is larger and comes from fewer misses/direct reads,
  even though iouring reads and wait increased.
- This does not rescue GP57. The held-out gate still failed and the overlay must
  remain disabled for final/random-prompt use.
- The likely problem is not a simple runtime overhead from extra pack entries;
  it is that a hotset selected from all dev prompts can overfit dev routing and
  still fail on unseen prompt distributions.

## Decision

- Keep budget16 rejected as a final candidate.
- Do not run more held-out experiments for tuning.
- Next dev-only step should use leave-one-dev-out validation:
  build or simulate hotsets from `N-1` dev prompts and evaluate coverage on the
  held-out dev prompt, rotating across dev prompts. This creates a proxy for
  prompt-agnostic generalization without touching the true held-out test set.

## Reproduction

All runs used cold start, `N=96`, `PROFILE=1`, `MemoryMax=15900000000`,
`MemorySwapMax=0`, and current branch head `932083e69`.

Default France:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
run=/root/lfz/tmp/runs/20260707-gp58-budget16-dev-ab/default_dev_france_n96
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=96 PROFILE=1 COPY_PROFILE=0 \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france,europe|paris|eiffel|louvre|riviera|bordeaux" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Budget16 France:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
run=/root/lfz/tmp/runs/20260707-gp58-budget16-dev-ab/budget16_dev_france_n96
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=96 PROFILE=1 COPY_PROFILE=0 \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france,europe|paris|eiffel|louvre|riviera|bordeaux" \
      EXTRA_RUNTIME_ENV="GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Default Japan:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
run=/root/lfz/tmp/runs/20260707-gp58-budget16-dev-ab/default_dev_japan_n96
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=96 PROFILE=1 COPY_PROFILE=0 \
      PROMPT_ID=dev_japan_factual \
      PROMPT_USER_TEXT="Please introduce Japan in a short paragraph." \
      QUALITY_KEYWORDS="japan,asia|tokyo|island" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Budget16 Japan:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
run=/root/lfz/tmp/runs/20260707-gp58-budget16-dev-ab/budget16_dev_japan_n96
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=96 PROFILE=1 COPY_PROFILE=0 \
      PROMPT_ID=dev_japan_factual \
      PROMPT_USER_TEXT="Please introduce Japan in a short paragraph." \
      QUALITY_KEYWORDS="japan,asia|tokyo|island" \
      EXTRA_RUNTIME_ENV="GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```
