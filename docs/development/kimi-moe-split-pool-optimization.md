# Kimi MoE split-pool VRAM cache optimization

This note records the accepted Kimi IQ3_S token-rate optimization, why it
worked, and how to evaluate the same idea on other MoE models.

## Summary

The accepted optimization enables the existing env-gated two-pool MoE VRAM
cache instead of adding a new compute path:

```sh
GGML_MOE_VRAM_CACHE_SPLIT=1
GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6
GGML_MOE_VRAM_CACHE_UPGATE_PCT=60
```

All previously accepted Kimi runtime settings stay unchanged. The split cache
keeps inference math unchanged; it only changes where routed expert tensors are
cached in VRAM.

Under strict cold start and a 16 GB host-RAM cgroup, the accepted n96 result was:

| metric | previous SOTA | split-pool accepted |
| --- | ---: | ---: |
| Token rate | `0.65 tok/s` | `0.714926 tok/s` average |
| n96 run range | N/A | `0.709216-0.723042 tok/s` |
| Max TTFT | `63209.39 ms` reference run | `69972.05 ms` |
| Host RAM peak | `14.808 GiB` | `14.808 GiB` |
| Quality | pass | pass |
| `read_failures` | `0` | `0` |

The required quality prompt was:

```text
Please introduce France in a short paragraph.
```

All three accepted n96 promotion runs produced the same coherent answer:

```text
France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also known for its diverse landscapes, from the vineyards of Bordeaux to the beaches of the Riviera, and plays a major role in European and global affairs.<|im_end|> [end of text]
```

## Implementation

The implementation is in `ggml/src/ggml-cuda/moe_stream_batch.cu`.

`batch_cache_id_for_size()` chooses the cache pool:

- when `GGML_MOE_STREAM_FUSED_UP_GATE=1`,
  `GGML_MOE_VRAM_CACHE_SPLIT=1`, and `expert_sz <=
  GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB`, the expert goes to cache id `1`;
- otherwise it goes to cache id `0`.

For Kimi IQ3_S with `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`:

| expert size | pool |
| ---: | --- |
| `4.484 MiB` | upgate/small pool, cache id `1` |
| `5.359 MiB` | upgate/small pool, cache id `1` |
| `6.016 MiB` | down/large pool, cache id `0` |
| `7.438 MiB` | down/large pool, cache id `0` |

`batch_cache_budget_mib_for_id()` splits the total VRAM cache budget between the
two pools. With:

```sh
GGML_MOE_VRAM_CACHE_MIB=15000
GGML_MOE_VRAM_CACHE_UPGATE_PCT=60
```

the upgate/small pool receives about 9000 MiB and the down/large pool receives
about 6000 MiB.

`batch_cache_get()` then creates one fixed-slot allocator per cache id. If a
pool later sees a larger expert in the same class, it recreates that pool with
the larger slot size. In the accepted n96 run, this produced:

- upgate/small pool: `8.8 GiB`, `1679` slots, `5.36 MiB` each;
- down pool first stage: `5.9 GiB`, `997` slots, `6.02 MiB` each;
- down pool final stage: `5.9 GiB`, `806` slots, `7.44 MiB` each.

The existing profile/preload paths already call `batch_cache_id_for_size()`, so
profiled preloads and runtime cache lookups use the same pool assignment.

## Why It Worked

The previous shared cache used one effective slot size for mixed expert tensors.
When smaller up/gate experts and larger down experts share a pool, small experts
consume slots sized for larger tensors. That wastes VRAM and reduces the number
of reusable experts that can stay resident.

The split-pool runtime fixes that specific waste:

- small up/gate experts use smaller slots and get more resident entries;
- down experts keep a separate larger-slot pool;
- the total cache still fills available VRAM under the existing safety reserve;
- no quantization, routing, or math path changes are introduced.

The measured counters match the expected mechanism:

| counter | previous SOTA | split-pool first n96 |
| --- | ---: | ---: |
| Pinned staging copies | `64145` | `56714` |
| Host staging time | `79083.729 ms` | `71092.196 ms` |
| H2D time | `13310.781 ms` | `11754.203 ms` |
| Down total | `18.954 ms/call` | `17.827 ms/call` |
| Up/gate total | `17.903 ms/call` | `16.896 ms/call` |

Representative final cache counters from the first accepted n96 run:

- global hit rate: `51.4%`;
- down pool: `22822` hits, `9802` misses, `70.0%` hit rate;
- upgate pool: `31967` hits, `41969` misses, `43.2%` hit rate;
- expert pack: `56714` hits, `3076` misses, `read_failures=0`.

The result is a real decode-rate gain because fewer routed experts have to be
staged from host memory and copied to GPU during decode.

## Acceptance Gates

Use these gates before accepting this optimization or any variant:

- cold start only: run `sync` and drop page cache before the benchmark;
- host RAM must stay below 16 GB, including page cache and pinned memory;
- VRAM should be deliberately used, with cache budget and safety reserve
  recorded;
- TTFT must not increase by more than 20% versus the accepted cold baseline;
- the France prompt must produce a semantically correct, coherent answer;
- `read_failures` and CUDA launch failures must stay at `0`;
- performance must be reproducible, not a single lucky run.

For the accepted Kimi run, the TTFT gate was `<= 106331.72 ms`; the worst split
n96 TTFT was `69972.05 ms`.

## Reproduction Shape

Use the accepted Kimi environment plus the split delta:

```sh
export GGML_KIMI_CPU_MOE_ELIGIBILITY_PROFILE=1
export GGML_KIMI_CPU_MOE_NAME_PROFILE=1
export GGML_KIMI_CPU_MOE_PROFILE=1
export GGML_MOE_BATCH_PROFILE=1
export GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
export GGML_MOE_IO_BACKEND=iouring
export GGML_MOE_IO_BYTES=8388608
export GGML_MOE_MMAP_DONTNEED=1
export GGML_MOE_PARALLEL_EXPERTS=1
export GGML_MOE_PREFETCH_DOWN=1
export GGML_MOE_PREFETCH_DOWN_DEPTH=2
export GGML_MOE_STAGE_PINNED_SLOTS=8
export GGML_MOE_STREAM=1
export GGML_MOE_STREAM_BATCH_ONLY=1
export GGML_MOE_STREAM_DECLINE_DEBUG=1
export GGML_MOE_STREAM_DOWN_BATCH=1
export GGML_MOE_STREAM_FUSED_UP_GATE=1
export GGML_MOE_STREAM_FUSED_UP_GATE_MIXED_TYPES=1
export GGML_MOE_TTFT_TRACE_MAX_EVENTS=120000
export GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
export GGML_MOE_VRAM_CACHE_MIB=15000
export GGML_MOE_VRAM_CACHE_SAFETY_MIB=512
export GGML_MOE_VRAM_CACHE_SPLIT=1
export GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6
export GGML_MOE_VRAM_CACHE_UPGATE_PCT=60
```

Then run a strict cold n96 benchmark:

```sh
cd /root/lfz/llama.cpp-vendor-kimi
sync
echo 3 > /proc/sys/vm/drop_caches

build-cuda-batch/bin/llama-completion --defer-experts --fit off -ngl 99 --special \
  -m /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf \
  -c 512 -n 96 --temp 0 --top-p 1.0 --top-k 1 --seed 1 \
  --no-display-prompt -no-cnv -t 32 -tb 32 \
  -p '<|im_user|>user<|im_middle|>Please introduce France in a short paragraph.<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>'
```

For the strict 16 GB host-RAM gate, run the process inside a cgroup with
`memory.max` below 16 GB and `memory.swap.max=0`, then record `memory.peak` and
`memory.stat`.

## Migration To Other Models

This optimization is portable as a method, not as fixed parameters.

It is likely worth testing when all of these are true:

- the target model is MoE;
- decode is limited by expert weight staging, H2D copies, or VRAM cache misses;
- expert tensor sizes differ enough that a shared fixed-slot cache wastes VRAM;
- routed expert reuse is stable enough for a cache to matter;
- not all experts already fit in VRAM.

It is unlikely to help when:

- the model is dense;
- all expert tensors have nearly identical sizes;
- all active experts are already resident in VRAM;
- the bottleneck is prompt TTFT, sampler overhead, or non-expert compute;
- the split reduces the hot pool's budget and lowers hit rate.

Use this migration procedure for each new model:

1. Establish a strict cold baseline.
   - Same prompt, seed, `-n`, context, thread count, and memory cgroup.
   - Record token rate, TTFT, host RAM, VRAM, output text, `read_failures`, and
     cache counters.

2. Measure expert tensor sizes.
   - Group by role if names expose up, gate, and down tensors.
   - If roles are not clear, group by byte size first.

3. Collect or replay route traces.
   - Count route frequency by tensor size and by expert id.
   - Estimate bytes saved and potential hit-rate changes for candidate pools.

4. Choose split thresholds and budgets.
   - Set `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB` just above the largest small class.
   - Sweep `GGML_MOE_VRAM_CACHE_UPGATE_PCT`; do not assume Kimi's `60` is right.
   - Keep `GGML_MOE_VRAM_CACHE_MIB`, auto-clamp, and safety reserve explicit.

5. Validate with a short cold smoke run.
   - Use the same quality prompt and deterministic sampling.
   - Reject immediately on semantic drift, TTFT failure, RAM failure, or read
     failures.

6. Promote only after full-length cold runs.
   - Run at least three n96-equivalent cold promotions for large gains.
   - Accept only if average and minimum token rate beat the baseline while all
     gates pass.

7. Commit and push only accepted improvements.
   - Include the exact env, command, run directories, output answer, and metrics.
   - Revert source changes when performance, quality, TTFT, or RAM gates fail.

For models such as DeepSeek MoE, Qwen MoE, Mixtral-style MoE, or other Kimi
quantizations, repeat the measurement and budget sweep from scratch. The Kimi
IQ3_S values `SPLIT_MAX_MIB=6` and `UPGATE_PCT=60` are a good starting
hypothesis only when the new model has comparable expert size classes and route
reuse.
