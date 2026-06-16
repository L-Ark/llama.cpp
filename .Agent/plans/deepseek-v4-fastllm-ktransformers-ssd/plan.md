# DeepSeek V4-Flash SSD MoE Runtime Plan

## 1. Goal

Evaluate and develop an SSD-backed MoE inference path for DeepSeek V4-Flash under a small CPU RAM tier.

Primary target:

- DeepSeek V4-Flash

Runtime roles:

- fastLLM: practical disk-MoE baseline
- KTransformers/SGLang: research mainline for MXFP4 SSD expert-cache extension

This plan should not depend on the current local machine or local disk layout.

## 2. Constraints

- CPU RAM tier budget: 2GB.
- VRAM: use as much as available.
- SSD: allowed as full cold expert tier.
- Quality: no intentional quality loss.
- No top-k reduction.
- No expert-count reduction.
- No prompt-specific oracle routing.
- Q4 may be a main result only if it passes quality validation against native behavior.
- Q2 or more aggressive quantization is exploratory only.

## 3. Runtime Split

### fastLLM Baseline

Use fastLLM first because it already has:

- DeepSeekV4 support
- Disk MoE support
- CUDA / NUMA / Disk mixed inference controls
- DeepSeekV4 quantization configs

fastLLM should answer:

- What is the practical disk-backed baseline?
- How expensive is SSD expert miss latency?
- How much does a 2GB RAM tier help?
- Can Q4 pass quality validation?
- What p50/p95 token latency should KTransformers beat?

### KTransformers / SGLang Research Mainline

Use KTransformers/SGLang for the research implementation because it already has:

- `deepseek_v4` model path
- V4-Flash MXFP4 routed experts
- NSA sparse MLA support
- GPU expert placement machinery
- SGLang batching/serving integration

The required missing piece is an SSD/RAM expert tier for MXFP4 routed experts.

Important caveat:

- The current V4-Flash MXFP4 path does not get usable dynamic expert update just by enabling a flag. MXFP4 expert slot replacement must be implemented explicitly.

## 4. Baseline Experiment Plan

Measure fastLLM first.

Configurations:

- native DeepSeek V4-Flash with disk-backed experts
- native with VRAM hot experts plus SSD cold tier
- native with VRAM hot experts plus 2GB RAM tier plus SSD cold tier
- Q4 with the same placement patterns

Required metrics:

- model id and revision
- runtime revision
- dtype / quantization
- VRAM expert cache size
- CPU RAM tier size
- SSD tier mode
- context length
- generated tokens
- TTFT
- decode tokens/sec
- p50/p95 token latency
- expert cache hit rate
- expert miss count
- SSD read bytes
- SSD effective GB/s
- RAM tier hit rate
- VRAM usage
- CPU RSS
- output hash
- quality verdict

Context matrix:

- 128 tokens
- 4K tokens
- 16K tokens
- 64K tokens
- 128K tokens if practical

Use at least 256 generated tokens and 3 repeats per configuration.

## 5. KTransformers SSD Extension Spike

The spike proves that one DeepSeek V4-Flash MXFP4 expert can move from SSD or RAM into a GPU expert slot and produce correct output.

Spike scope:

- one layer
- one or a few selected experts
- synchronous path first
- async path only after correctness
- metrics from the beginning

Required components:

1. Expert metadata index
   - map layer, expert id, projection, dtype, shape, file, offset, and byte size
   - cover MXFP4 weight tensors and scale tensors

2. SSD expert reader
   - read expert payloads by offset
   - expose synchronous API for correctness
   - optionally expose async API for scheduling
   - log read bytes, latency, and bandwidth

3. 2GB RAM tier
   - fixed budget
   - pinned staging where useful
   - candidate hot expert cache
   - explicit hit/miss/eviction metrics

4. GPU expert slot manager
   - track logical expert to GPU slot mapping
   - copy MXFP4 payloads into resident GPU tensors
   - update masks and maps in-place when CUDA graph compatibility requires stable addresses

5. MXFP4 dynamic replacement
   - implement replacement for the V4-Flash MXFP4 tensor layout
   - do not rely on int4/fp8/bf16 copy paths
   - verify with deterministic output tests

6. Telemetry
   - log every promotion and eviction
   - log SSD read, RAM hit/miss, H2D copy, and scheduler reason

## 6. Scheduler Design

The scheduler should minimize decode latency by maximizing expert hit rate under VRAM plus 2GB CPU RAM constraints.

Tiers:

- VRAM: hot resident experts
- CPU RAM: 2GB staging and small candidate tier
- SSD: full cold expert storage

Decision inputs:

- free VRAM
- RAM tier pressure
- recent expert activation frequency
- per-layer expert reuse
- active request count
- current context length
- measured SSD bandwidth
- measured SSD latency
- H2D bandwidth
- expert byte size
- GPU utilization
- KV reusable count

First policy:

- LFU/LRU hybrid
- separate protected and dynamic slots
- route-history driven promotion
- no speculative prefetch until baseline miss costs are measured

Advanced policy:

- async prefetch under GPU compute slack
- multi-request expert sharing
- score-based eviction using expected future miss penalty

## 7. KV Tradeoff

Do not start with KV offload for short contexts.

For DeepSeek V4-Flash, compressed KV is small enough at ordinary context lengths that expert cache is the first-order target.

Guidance:

- <=16K context: expert cache only
- 64K context: measure KV pressure
- >=128K context: evaluate KV offload/recompute
- very long context or multi-request: KV/expert tradeoff becomes central

KV offload/recompute should be activation-checkpoint or prefix-block based. Do not implement naive token-by-token SSD KV reload.

## 8. Prefill vs Decode

Prefill:

- expect limited scheduling upside if expert fetch is already pipelined
- focus on layout, compression, duplicate-fetch elimination, and bulk reads

Decode:

- primary target
- reduce expert miss count
- hide SSD reads behind compute when possible
- increase VRAM hit rate
- use RAM tier as staging, not as a large store

## 9. Quality Validation

Use native behavior as the reference.

Prompt suite:

- short factual QA
- coding
- math/reasoning
- JSON/tool-style output
- long-context retrieval
- multilingual response

Q4 is accepted as a main result only if:

- no systematic task failure versus native
- no obvious format degradation
- coding/math quality remains acceptable
- long-context retrieval remains correct

KTransformers SSD extension is accepted only if:

- deterministic small tests match the non-SSD path
- no routing semantics are changed
- no expert reduction or hidden approximation is introduced

## 10. Experiments

Ablations:

- no RAM tier
- 2GB RAM tier
- static GPU expert placement
- dynamic GPU expert placement
- sync SSD fetch
- async SSD prefetch
- Q4 off/on
- KV tradeoff off/on for long context
- single request
- multi request

Primary metric:

- p95 decode token latency

Secondary metrics:

- decode tokens/sec
- TTFT
- expert hit rate
- SSD GB/token
- RAM tier hit rate
- quality pass rate

## 11. Success Criteria

This plan succeeds if:

- fastLLM produces a stable DeepSeek V4-Flash disk baseline
- KTransformers spike proves correct MXFP4 expert replacement into GPU slots
- dynamic scheduling beats naive SSD miss handling
- 2GB RAM tier benefit is quantified
- accepted configurations pass quality validation

Stretch goal:

- KTransformers SSD extension beats fastLLM on p95 decode latency under equivalent quality constraints.

## 12. Stop Conditions

Stop or de-scope KTransformers SSD extension if:

- MXFP4 expert replacement cannot be made correct
- tensor layout support becomes a major runtime rewrite before a single-layer spike works
- CUDA graph or SGLang invariants prevent safe in-place updates
- Q4 or runtime changes fail quality validation

If stopped:

- continue with fastLLM as the main DeepSeek V4-Flash runtime
- keep KTransformers as a reference for attention and GPU expert placement

## 13. Relationship to ik_llama

Do not port DeepSeek V4-Flash to `ik_llama` as the first implementation path.

The transferable part from `ik_llama` is the scheduling idea:

- expert-pack style indexing
- VRAM hot cache
- 2GB RAM staging tier
- SSD cold tier
- route-history scoring
- hit/miss telemetry

The model/runtime implementation should stay with fastLLM or KTransformers/SGLang unless DeepSeek V4-Flash support in `ik_llama` becomes mature independently.

