# GP50 Partial Batched iouring Smoke Report

Timestamp: `2026-07-07T07:58:00+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit under test: `02c2f3c9b4eff7214eea8569f4ff1d94154a8233` plus local GP50 changes.

## Change

`expert_pack_iouring_copy_jobs()` was all-or-nothing: one invalid job in a copy
batch forced every valid job in that batch through the fallback copy loop.

GP50 keeps the existing full-batch fast path first. If that fails, it partitions
the batch into iouring-eligible jobs and fallback jobs, runs iouring on the valid
subset, and sends only the invalid subset through the existing fallback loop. If
the partial iouring attempt fails, the original full fallback behavior is used.

## Build

The first full runtime build failed in unrelated `lightning-indexer.cu`; the Kimi
runtime build was therefore configured with the same disabled lightning-indexer
setting used by the existing remote SOTA build.

Reproduction build command:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
build=$repo/build-gp50-runtime
cmake -S "$repo" -B "$build" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_LIGHTNING_INDEXER=OFF \
  -DGGML_CUDA_MOE_STREAM_BATCH=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build "$build" --target llama-completion -j$(nproc)
```

Verified build flags included:

```text
-DGGML_CUDA_MOE_STREAM_BATCH
-DGGML_CUDA_NO_LIGHTNING_INDEXER
```

## Runtime Reproduction

Both runs used:

- cold start with `echo 3 > /proc/sys/vm/drop_caches`;
- `MemoryMax=15900000000`, `MemorySwapMax=0`;
- `N=32`;
- prompt: `Please introduce France in a short paragraph.`;
- `PROFILE=1`, `COPY_PROFILE=1`;
- `GGML_MOE_IO_BACKEND=iouring`;
- `GGML_MOE_VRAM_CACHE_MIB=15000`;
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=62`;
- `GGML_MOE_STAGE_PINNED_SLOTS=12`;
- `GGML_MOE_IO_DEPTH=8`;
- `GGML_MOE_IO_REFILL_BATCH=4`;
- `GGML_MOE_PREFETCH_DOWN_DEPTH=2`.

Baseline command:

```bash
repo=/root/lfz/llama.cpp-vendor-kimi
run=/root/lfz/tmp/runs/20260707-gp50-partial-iouring/baseline_dev_france_n32
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=32 PROFILE=1 COPY_PROFILE=1 \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,paris|europe|culture" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

GP50 command:

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
run=/root/lfz/tmp/runs/20260707-gp50-partial-iouring/dev_france_n32_batch_on
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=32 PROFILE=1 COPY_PROFILE=1 \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,paris|europe|culture" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

## Results

| Run | Quality | TTFT ms | Decode ms | Runs | Token rate | Peak RAM | direct reads | iouring reads | iouring bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | pass | 69178.25 | 24749.49 | 31 | 1.25 | 15899996160 | 671 | 22647 | 126391910400 |
| GP50 | pass | 68671.81 | 24792.35 | 31 | 1.25 | 15899996160 | 133 | 23185 | 129268056064 |

Baseline output:

```text
France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous
```

GP50 output:

```text
France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous
```

## Conclusion

GP50 is accepted as a transport-path cleanup, not as a token-rate SOTA
improvement:

- semantic quality passed;
- TTFT did not regress and was slightly lower;
- token rate was neutral at `1.25 tok/s`;
- direct expert-pack reads dropped from `671` to `133`;
- iouring reads increased from `22647` to `23185`.

The lack of token-rate improvement indicates that on this France n32 run, the
remaining bottleneck is not the small number of mixed-batch direct reads removed
by GP50. The next optimization should target larger uncovered fallback sources
or increase sustained iouring queue depth rather than only reducing mixed-batch
direct-read leakage.
