# GP56 budget16 overlay multi-dev validation

Status: accepted dev generalization check.

This validation used only dev prompts. No held-out test prompt was used.

## Runtime

Common runtime overlay:

```text
GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack
GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1
```

Both runs used:

- cold start via `systemd-run --wait --collect`;
- `MemoryMax=15900000000`;
- `MemorySwapMax=0`;
- `N=96`;
- `PROFILE=1`;
- `COPY_PROFILE=0`.

## Results

Baselines are from
`.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile`.

| prompt | quality | baseline tok/s | overlay tok/s | baseline TTFT ms | overlay TTFT ms | TTFT gate | baseline decode ms/runs | overlay decode ms/runs | RAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_python_reverse` | pass | `0.17` | `0.24` | `84157.32` | `74730.11` | pass | `566552.71/95` | `398190.38/95` | `15899996160` |
| `dev_mixed_summary` | pass | `0.20` | `0.30` | `91143.37` | `91176.80` | pass | `266891.43/54` | `178704.87/54` | `15899996160` |

Acceptance:

- `dev_python_reverse`: pass. Token rate improved by about `41%`, TTFT
  decreased, quality passed, and RAM stayed under the configured cgroup cap.
- `dev_mixed_summary`: pass. Token rate improved by about `50%`, TTFT stayed
  within `+20%`, quality passed, and RAM stayed under the configured cgroup cap.
- RAM again hit the cgroup cap exactly, so the next stage must keep monitoring
  page cache and should not enlarge the overlay before profiling memory
  distribution.

## Answers

### dev_python_reverse

```text
Here's a Python function to reverse a string: ```python def reverse_string(s): return s[::-1] ``` **Example usage:** ```python text = "Hello, World!" reversed_text = reverse_string(text) print(reversed_text) # Output: "!dlroW ,olleH" ``` **How it works:** `s[::-1]` uses Python's slice notation with a step of `-1`, which creates a new string containing all characters
```

### dev_mixed_summary

```text
Solar power offers predictable daytime generation and works well on rooftops, making it easy for small towns to scale gradually. Wind power can produce energy day and night but depends more on location and consistent breezes, so it suits towns with open landscapes more than densely built areas.
```

## Expert-Pack Counters

| prompt | hits | misses | direct reads | iouring reads | iouring GiB | iouring wait ms | down hit | upgate hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_python_reverse` | `58937` | `31035` | `7544` | `48347` | `248.07` | `60488.51` | `69.9%` | `35.8%` |
| `dev_mixed_summary` | `36265` | `14209` | `3778` | `30676` | `157.84` | `35376.44` | `69.5%` | `36.3%` |

## Reproduction

`dev_python_reverse`:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
run=/root/lfz/tmp/runs/20260707-gp56-budget16-multidev/dev_python_reverse_n96
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=96 PROFILE=1 COPY_PROFILE=0 \
      PROMPT_ID=dev_python_reverse \
      PROMPT_USER_TEXT="Write a Python function to reverse a string." \
      QUALITY_KEYWORDS="python|def|string,[::-1]|reverse" \
      EXTRA_RUNTIME_ENV="GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

`dev_mixed_summary`:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
run=/root/lfz/tmp/runs/20260707-gp56-budget16-multidev/dev_mixed_summary_n96
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=96 PROFILE=1 COPY_PROFILE=0 \
      PROMPT_ID=dev_mixed_summary \
      PROMPT_USER_TEXT="In two sentences, compare solar power and wind power for a small town." \
      QUALITY_KEYWORDS="solar|sun,wind,power|energy" \
      EXTRA_RUNTIME_ENV="GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

## Decision

The budget16 overlay now has dev evidence on three prompts:

- `dev_linear_equation N=48`: `0.16 -> 0.22 tok/s`;
- `dev_python_reverse N=96`: `0.17 -> 0.24 tok/s`;
- `dev_mixed_summary N=96`: `0.20 -> 0.30 tok/s`.

The next step can be a planned held-out test-set gate. Do not tune on the
held-out outputs; run once, record all metrics, and accept only if quality,
TTFT, RAM, and token-rate gates pass.
