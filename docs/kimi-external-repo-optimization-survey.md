# Kimi External MoE Optimization Survey

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`

This document records a code-level survey of five external repositories and maps their techniques to the current Kimi/ik_llama optimization target:

- Target machine class: 16 GB host RAM including page cache, 32 GB RTX 5090 VRAM.
- Target workload: random user prompts, prompt-general performance, cold start.
- Target output: stable semantic correctness, including `Please introduce France in a short paragraph.`
- Target performance: eventually above 5 tok/s without TTFT increasing more than 20%.
- Current bottleneck: decode-time expert movement. The runtime can reach about 10.0-10.4 GiB/s in pure IO benches, but real decode sustains lower throughput because expert IDs are discovered layer by layer. The current expert-pack IO path is op-local: it builds the read plan for the current MoE operation, submits reads, waits, performs H2D, and returns. It does not yet have a global cross-layer producer that can keep the NVMe queue full.

## Sources Read

The repositories were cloned under `/Users/spark/external-repo-study`.

| Repository | Local commit | Local path | Main area inspected |
| --- | ---: | --- | --- |
| AirLLM | `696aba8` | `/Users/spark/external-repo-study/airllm` | Layer-wise offload, prefetch, compression |
| Unsloth | `296cacb` | `/Users/spark/external-repo-study/unsloth` | MoE grouped GEMM kernels and fusion |
| flash-moe | `3601d41` | `/Users/spark/external-repo-study/flash-moe` | Expert packing, parallel pread, Metal kernels, IO experiments |
| MoE-Infinity | `6285c09` | `/Users/spark/external-repo-study/MoE-Infinity` | Expert offload, prediction, priority prefetch, async IO scheduler |
| mllm | `c4fd487` | `/Users/spark/external-repo-study/mllm` | Mobile quantized kernels, W4A16/W8A8 decode benchmarks |

## Executive Summary

The most transferable idea is from MoE-Infinity: use a native global expert-transfer scheduler with priorities, deduplication, stale prefetch pruning, and demand stealing of in-flight prefetches. This is different from the previous Kimi planned host-prefetch attempt. The previous attempt issued many low-useful host-prefetch reads, generated about 40.5 GiB of extra read traffic per prompt, had almost no useful hits, and slowed decode. A MoE-Infinity-style scheduler should first be implemented as a bounded, default-off control plane that avoids duplicate reads and allows demand requests to consume or supersede low-priority prefetch work.

The second most transferable set of ideas is from flash-moe: keep each expert contiguous, reduce small reads, test large alignment for staging buffers, and consider reordering expert-pack layout by co-access patterns. We already have expert packs, but we have not fully tested 2 MiB-aligned pinned staging and co-occurrence-aware pack layout under the Kimi workload.

AirLLM is useful mostly as a contrast. Its next-layer prefetch works because dense transformer layers are deterministic. Kimi MoE expert IDs are not known until routing/topK, so AirLLM's direct one-layer-ahead loader does not solve the up/gate miss bottleneck by itself.

Unsloth's grouped GEMM is useful for resident experts, prefill, or larger batches, but it is not the first bottleneck for our single-request decode path. Also, Unsloth is AGPLv3, so code must not be copied into this codebase without a license decision. Only the ideas should be used.

mllm is mainly a warning against adding per-token activation quantization or sparsity kernels without careful measurement. Its own W4A16 vs W8A8 decode benchmark shows that activation quantization overhead can erase faster GEMM at M=1.

## Repository Findings

### AirLLM

Relevant files:

- `airllm/airllm_base.py`
- `airllm/utils.py`

AirLLM splits checkpoints into layer shards and streams one module at a time from disk to GPU. The execution model is:

1. Keep most model parameters off-device.
2. Load the current layer's shard to the GPU just before execution.
3. Optionally start loading the next deterministic layer on a worker thread.
4. Free the current layer after use.

Important implementation points:

- `AirLLMBaseModel` uses a `ThreadPoolExecutor(max_workers=1)` for prefetch.
- If the prefetched layer index matches the next layer, it consumes the prefetched result.
- If compression is enabled, prefetch is disabled in that path.
- Compression uses block-wise 4-bit/8-bit style mechanisms to reduce disk traffic.

Transferability to Kimi:

- Direct layer prefetch is not enough for Kimi decode because Kimi's per-layer active experts are dynamic. Dense next-layer identity is known, but expert IDs are not known until gate/topK.
- The offload lifecycle is useful as a design pattern for deterministic tensors, but our dense and attention tensors are already mostly resident or managed differently.
- Compression is only useful if it reduces transferred bytes while preserving Kimi output quality and while matching available CUDA kernels. It is not a short-path fix.

Recommended action:

- Do not port AirLLM directly.
- Keep its deterministic prefetch model as a baseline reference for non-MoE tensors only.

### MoE-Infinity

Relevant files:

- `moe_infinity/memory/expert_tracer.py`
- `moe_infinity/memory/expert_predictor.py`
- `moe_infinity/memory/expert_prefetcher.py`
- `moe_infinity/memory/expert_priority_score.py`
- `moe_infinity/memory/offloading_policy.py`
- `moe_infinity/serving/expert_prefetch_coordinator.py`
- `moe_infinity/serving/expert_batch.py`
- `moe_infinity/distributed/expert_prefetcher.py`
- `core/prefetch/archer_prefetch_handle.{h,cpp}`
- `core/prefetch/task_scheduler.cpp`
- `core/aio/archer_prio_aio_handle.cpp`
- `core/parallel/expert_dispatcher.{h,cpp}`

Key mechanisms:

- Expert tracing records a layer-by-expert activation matrix for each sequence.
- Prediction finds similar historical traces and uses them to score future experts.
- Prefetch tasks are low priority. Demand tasks are high priority.
- The scheduler deduplicates identical tasks and removes stale or lower-priority conflicting tasks.
- Demand execution can consume an already-prefetching or already-resident expert instead of issuing a duplicate read.
- Async IO uses a queue depth around 32, chunked reads, O_DIRECT alignment, and pinned host buffers.
- Profiling breaks down routing, cache lookup, prediction, scheduling, disk-to-CPU, CPU-to-GPU, sync wait, expert compute, eviction, and queue coordination.

Transferability to Kimi:

- The priority scheduler design is highly relevant.
- The Python/PyTorch implementation cannot be copied directly. Kimi needs a native C++/CUDA version integrated with expert-pack offsets and existing ggml CUDA MoE streaming.
- Trace-based prediction is risky under the prompt-general constraint. It may still be useful if trained only on dev prompts and validated only on held-out prompts, with strict extra-read caps.

Recommended action:

1. Build a native global expert-transfer scheduler behind a default-off env flag.
2. Add task keys such as `(tensor_kind, layer, expert, byte_offset, byte_size, quant_type)`.
3. Use priority 0 for demand, priority 1 for speculative/prefetch.
4. Deduplicate queued and in-flight tasks.
5. Let demand requests steal or wait for matching in-flight prefetch instead of issuing duplicate reads.
6. Prune stale prefetches when the layer window has passed.
7. Add counters for useful prefetch hits, wasted bytes, duplicate suppression, stale cancellation, demand wait time, and queue depth.

Acceptance criteria:

- Cold start only.
- Host RAM peak including page cache stays under 16 GB.
- TTFT does not increase more than 20%.
- Held-out prompt quality passes.
- Any accepted result must improve held-out token rate and be committed with exact env and reproduction commands.

### flash-moe

Relevant files:

- `repack_experts.py`
- `metal_infer/main.m`
- `metal_infer/infer.m`
- `metal_infer/shaders.metal`
- `docs/io-and-gpu-exploration.md`
- `docs/optimization-experiments-q4.md`
- `docs/plan-async-pread-pipeline.md`

Key mechanisms:

- Experts are repacked into contiguous per-layer files.
- Instead of many small component reads per expert, the optimized path reads one full expert block and then uses known offsets.
- Multiple active experts can be read in parallel.
- Their experiments found mmap cold expert files to be much slower due to page faults.
- 2 MiB-aligned DMA buffers improved isolated read throughput and gave a smaller but measurable full-pipeline gain on their platform.
- Temporal prediction and simple learned prediction had low hit rates and could waste bandwidth.
- Some dequant math was rearranged into FMA-friendly forms in Metal kernels.

Transferability to Kimi:

- We already use expert packs, so the core contiguous-pack idea is already adopted.
- The next useful checks are staging alignment and pack layout.
- Their mmap warning supports avoiding GGUF mmap fallback for dynamic experts.
- Their failed prediction results support the need for strict useful-hit accounting and extra-byte limits.
- The FMA dequant idea is worth auditing in CUDA kernels, but it is lower priority while movement dominates.

Recommended action:

1. Run a Kimi expert-pack microbench with current pinned slots vs 2 MiB-aligned pinned slots.
2. Measure pure IO and real n32/n96 decode.
3. Build a co-occurrence-aware pack-ordering experiment using dev route traces only.
4. Validate on held-out prompts only after the layout is frozen.
5. Audit CUDA dequant kernels for unnecessary multiply/add sequences where FMA can be used without changing numerical behavior.

### Unsloth

Relevant files:

- `unsloth/kernels/moe/README.md`
- `unsloth/kernels/moe/`
- Qwen3 MoE tests and expert rebinding helpers under the test tree.

Key mechanisms:

- Grouped GEMM computes all selected experts in a grouped operation instead of looping over experts.
- Token permutation is fused into the first grouped GEMM prologue.
- Unpermutation is fused into the second GEMM.
- TopK weight multiplication is fused into the second GEMM epilogue.
- Experts are rebound into shared contiguous buffers for kernel access.

Transferability to Kimi:

- This is most useful when many tokens or many resident experts are computed together.
- Our current decode workload is single request, single token at a time, and movement-bound. Grouped GEMM is not the highest-priority fix.
- Some fusion ideas can help resident expert compute and prefill fallback.
- License risk: Unsloth is AGPLv3. Do not copy code into this repository without explicit license approval.

Recommended action:

- Do not port code.
- Use the grouped-GEMM/fusion idea only as a future resident-expert compute optimization after transfer stalls are reduced.
- If compute becomes dominant, prototype a clean-room grouped resident expert kernel for the Kimi quant formats.

### mllm

Relevant files:

- `examples/qwen3_moe/config_30B_A3B_gguf.json`
- `examples/qwen3_moe/quant_cfg_30B_q4_k.json`
- `mllm-kernel/benchmarks/bench_w4a16_vs_w8a8.py`
- `mllm-kernel/mllm_kernel/cuda/csrc/gemm/int8/int8_scaled_mm_cutlass.cu`
- `mllm-kernel/mllm_kernel/cuda/jit/gptq_marlin*`
- `mllm-kernel/mllm_kernel/cuda/jit/awq_marlin*`

Key mechanisms:

- Mobile/edge runtime with CUDA, CPU, QNN, and Ascend paths.
- W4A16 and W8A8 quantized kernels are benchmarked.
- Their decode benchmark notes that W8A8 GEMM can be faster alone, but activation quantization overhead can make the total decode path slower at M=1.
- JIT kernel cache is used for CUDA kernels.

Transferability to Kimi:

- The main lesson is negative: do not add activation quantization, sparsity scanning, or extra per-token preprocessing unless the M=1 overhead is included in the profile.
- W4A16/Marlin-style kernels could be useful if we convert Kimi expert formats and keep quality, but this is a large format/kernel project.
- Qwen3 MoE GGUF examples are not directly applicable to Kimi's current expert-pack and IQ formats.

Recommended action:

- Use mllm as a benchmark methodology reference.
- Avoid W8A8 activation quantization as a near-term decode optimization unless fused into existing kernels.
- Consider W4A16/Marlin-style resident expert kernels only after IO scheduling is improved.

## Migration Priority

| Priority | Idea | Source | Expected benefit | Main risk | First validation |
| ---: | --- | --- | --- | --- | --- |
| 1 | Global priority expert-transfer scheduler | MoE-Infinity | Higher useful IO overlap, fewer queue gaps, no duplicate reads | Sync overhead, wasted prefetch bytes | Default-off scheduler counters, no-read probe, n32 dev |
| 2 | 2 MiB-aligned pinned staging | flash-moe | Better NVMe/H2D staging efficiency | No gain on this platform, pinned memory pressure | Pure IO bench, then n32/n96 held-out |
| 3 | Co-occurrence-aware expert-pack layout | flash-moe | Better locality and fewer command gaps | Prompt-specific overfit | Train layout on dev routes only, test held-out only |
| 4 | Trace/predictor-guided prefetch | MoE-Infinity | More future expert IDs known earlier | Poor generalization, wasted bandwidth | Shadow-only predictor recall on held-out |
| 5 | Resident grouped expert compute fusion | Unsloth | Faster compute when experts are resident | Not current bottleneck, license constraints | Clean-room microbench only |
| 6 | W4A16/Marlin-style expert kernels | mllm | Faster resident expert matmul | Large kernel/format work, quality risk | Isolated resident-expert benchmark |

## Concrete Next Plan

### Step 1: Add scheduler observability before changing behavior

Implement counters around the existing expert-pack path:

- Queue submit batch size.
- Queue depth over time.
- Demand wait by layer and tensor kind.
- Read bytes by up/gate/down.
- H2D time.
- Current-down overlap useful time.
- Duplicate demand reads for same expert within a short window.
- Time from gate/topK availability to read submission.

This answers whether the runtime is delayed by gate visibility, CPU scheduling, sync boundaries, or IO worker starvation.

### Step 2: Prototype a default-off global scheduler

Add an env-guarded scheduler, for example:

- `GGML_MOE_GLOBAL_EXPERT_SCHED=1`
- `GGML_MOE_GLOBAL_PREFETCH_MAX_MIB=<cap>`
- `GGML_MOE_GLOBAL_PREFETCH_WINDOW=<layers>`
- `GGML_MOE_GLOBAL_PREFETCH_MAX_EXTRA_READ_RATIO=<ratio>`

The first version should not try aggressive prediction. It should only manage tasks already known to be needed soon and should support:

- Priority 0 demand tasks.
- Priority 1 prefetch tasks.
- Deduplication across queued, in-flight, and ready tasks.
- Demand stealing of in-flight prefetch.
- Stale low-priority cancellation.
- Strict memory and byte caps.

Acceptance: it must not slow current held-out n96 SOTA. If there is no improvement, revert or keep disabled and document the measured blocker.

### Step 3: Re-test next-layer shadow prefetch only with useful-hit gating

Previous planned host-prefetch failed because useful hits were near zero and extra reads were large. Any retry must start in shadow mode:

- Use dev prompts only to tune thresholds.
- Use held-out prompts only for final validation.
- Log predicted bytes, actual bytes, useful hit bytes, false-positive bytes, and demand wait reduction.
- Reject if extra-read ratio is high or TTFT rises more than 20%.

### Step 4: Test 2 MiB-aligned pinned staging

Implement a microbench-only mode first:

- Same expert-pack offsets and read sizes.
- Current pinned staging allocation vs 2 MiB-aligned staging allocation.
- Pure IO throughput.
- End-to-end n32 and n96 decode.
- Host RAM peak including page cache and pinned memory.

Acceptance: real held-out decode must improve without violating 16 GB RAM.

### Step 5: Test co-occurrence-aware pack layout

Build an alternate expert-pack manifest from dev route traces:

- Do not use held-out route traces during layout generation.
- Keep alias/offset mapping exact.
- Reorder experts within layer by co-access/hotness clusters.
- Compare pure IO and real decode.

Acceptance: held-out prompts improve. Dev-only improvement is not enough.

## Explicit Non-Goals

- Do not rely on OS page cache as the primary expert cache under the 16 GB host RAM constraint.
- Do not copy AGPLv3 Unsloth code into this repository.
- Do not add W8A8 activation quantization or activation sparsity scanning to decode without proving total M=1 latency improves.
- Do not tune expert packs or predictors on held-out prompts.
- Do not accept prompt-specific SOTA as general SOTA.

## Reproducibility Rules For Any Follow-Up

Every accepted optimization must record:

- Git commit hash.
- Branch name.
- Exact env vars.
- Exact model, expert pack, and alias paths.
- Cold-start command.
- Prompt set name.
- n32/n96 token rate.
- TTFT.
- Decode time.
- Host RAM peak including page cache.
- `direct_reads` and expert-pack read counters.
- Full model answer for quality review.
- Whether the prompt was dev or held-out.

If an optimization improves only dev prompts or only one prompt-specific pack, it must not be labeled general SOTA.
