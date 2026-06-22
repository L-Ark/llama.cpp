# DeepSeek-V4-Flash SSD Expert Streaming 5 tok/s Optimization Plan

## 1. Goal And Hard Gates

Target model:

- `/home/wici/lfz/models/DeepSeek-V4-Flash`

Target runtime line:

- fastLLM disk-MoE first, with ktransformers/vramctl ideas used only when they are implemented and measured in this task.
- vramctl is not a separate commit target for this task. Any useful vramctl io_uring / pipeline code must be copied or ported into fastLLM and committed together with the fastLLM changes.

Current SOTA baseline from the previous DeepSeek V4-Flash fastLLM task:

- decode token rate p50: `1.34 tok/s`
- worst observed decode token rate: `1.29 tok/s`
- TTFT p50: `15.98 s`
- current stable placement: `{'cuda':16,'disk':84}` with shared expert on CUDA

Hard success gates:

- Decode token rate p50 must reach at least `5.0 tok/s`.
- Worst repeat decode token rate must be at least `4.7 tok/s`.
- TTFT p50 must be no worse than `19.98 s`, which is `15.98 s * 1.25`. There is no exception to this TTFT gate.
- Smoke test pass rate must be greater than `50%`.
- Smoke test pass rate must not drop by more than `15 percentage points` from the no-accuracy-loss baseline.
- No top-k reduction.
- No active expert-count reduction.
- No prompt-specific oracle routing for reported main results.
- CPU RAM expert tier budget is `2 GiB`.
- Do not stop other users' GPU or RAM-heavy processes. If the RTX 5090 or host RAM is occupied, wait until the process exits naturally.

Commit and rollback rules:

- When token rate improves over the previous SOTA and the improvement is reproducible, commit all effective tracked changes.
- Commit fastLLM changes to `/home/wici/lfz/fastllm`.
- Push fastLLM commits to the private remote named `private`, not to upstream `origin`.
- The intended private remote is `https://github.com/L-Ark/fastllm-deepseek-v4-flash-ssd-5tps.git`.
- The private GitHub repository must be created before the first push; do not use a fork for this task.
- Each SOTA-improving commit must be recorded in `progress.md` with commit hash, changed behavior, exact benchmark command, repeat metrics, smoke result, and log paths.
- If a route regresses token rate by more than `5%`, exceeds the TTFT gate, or fails smoke accuracy gates, roll back to the previous committed SOTA, record the cause, and plan the next route in `progress.md`.

## 2. Implementation Routes

### Route A: Baseline Instrumentation And Gates

- Add a benchmark wrapper that records exact environment, command line, git revisions, GPU/RAM state, TTFT, decode tok/s, prefill tok/s, total tok/s, output hash, and log paths.
- Add disk-MoE telemetry in the fastLLM disk path:
  - selected expert count
  - disk read bytes
  - disk read latency
  - effective expert read bandwidth
  - RAM cache hit/miss/eviction if enabled
  - VRAM hot expert hit estimate if available
- Create a fixed smoke test suite with at least 20 examples covering factual QA, simple math, coding, JSON output, and multilingual responses.
- Run the no-accuracy-loss baseline smoke test first and store its pass rate as the reference.
- Commit only instrumentation if it builds and does not change model semantics.

Expected commit name:

- `baseline-instrumentation`

### Route B: Repack Experts Into Contiguous Files

- Build an expert pack tool for DeepSeek-V4-Flash routed experts.
- Pack each logical `(layer, expert)` into contiguous payload blocks, including all tensors needed by that expert.
- Generate a manifest with:
  - layer id
  - expert id
  - tensor names
  - file path
  - offset
  - byte size
  - dtype
  - shape
  - checksum
- Change the disk-MoE read path to use manifest offsets when the pack is enabled.
- Keep the original safetensors path as a fallback.
- Optimize for large aligned reads and fewer seeks.

Gate for this route:

- expert read bandwidth improves over the current measured path
- token rate does not regress by more than `5%`
- TTFT remains `<=19.98 s`
- smoke accuracy gates pass

Expected commit name after reproducible improvement:

- `expert-pack-layout`

### Route C: io_uring / Pipeline Expert Read Backend

- Use the vramctl `io_uring` and pipeline design as implementation reference, not as an unverified direct dependency.
- Copy or port only the useful parts into fastLLM. Do not commit this task's implementation to `/home/wici/lfz/vramctl`.
- Implement an expert-level read backend with:
  - fixed bounce buffer pool
  - queue depth of at least 16 when resources permit
  - 2 MiB or 4 MiB aligned IO units
  - batched submit and completion polling
  - fallback to existing `pread` backend
- Measure actual expert read bandwidth, not only fio bandwidth.
- Target first-stage bandwidth: `8-12 GB/s` effective expert reads.

Gate for this route:

- effective expert read bandwidth improves materially over Route B
- decode token rate improves reproducibly over previous SOTA
- TTFT and smoke gates pass

Expected commit name after reproducible improvement:

- `iouring-expert-streamer`

### Route D: 2 GiB RAM Compressed Expert Cache

- Add a CPU RAM cache for loaded routed expert payloads with a strict `2 GiB` cap.
- Cache whole logical expert payloads or reusable tensor payloads only when they fit the budget.
- Track exact byte usage and evict using a route-history-aware LFU/LRU policy.
- Optional compression is allowed only if decode token rate improves after decompression overhead is included.
- Cache must not change routing or expert math.

Gate for this route:

- RAM cache hit rate is recorded
- token rate improves reproducibly or the route is reverted
- process RSS remains compatible with the host and the 2 GiB expert-tier budget
- TTFT and smoke gates pass

Expected commit name after reproducible improvement:

- `compressed-ram-expert-cache`

### Route E: VRAM Hot Expert Cache By Routing Profile

- Replace layer-ratio placement with routing-profile-driven hot expert placement.
- Build a profile from a fixed profiling prompt split and test on a disjoint benchmark/smoke split.
- Select hot experts globally by benefit per byte, not by equal layer quota.
- Target active expert cache hit rate: at least `32%`.
- Automatically back off if the selected hot set causes CUDA OOM.

Gate for this route:

- active expert hit rate improves from about `16%` to at least `32%`
- decode token rate improves reproducibly over previous SOTA
- TTFT remains `<=19.98 s`
- smoke accuracy gates pass

Expected commit name after reproducible improvement:

- `routing-profile-vram-cache`

### Route F: Decode-Time Prefetch Pipeline

- Start prefetch as soon as route decisions are available.
- Pipeline SSD read, RAM cache lookup, decompression, and H2D copy where applicable.
- Record wasted prefetch bytes and useful prefetch hit rate.
- Keep prefetch conservative: a wrong prefetch may waste bandwidth, but must not alter routing or output semantics.

Gate for this route:

- decode token rate improves reproducibly over previous SOTA
- wasted prefetch does not erase bandwidth gains
- TTFT and smoke gates pass

Expected commit name after reproducible improvement:

- `expert-prefetch-pipeline`

### Final Commit

When all hard gates pass:

- commit name: `deepseek-v4-flash-ssd-5tps`
- record final metrics and reproduction instructions in `progress.md`

## 3. Benchmark Protocol

Before every long run:

- Check GPU and RAM occupancy.
- If other heavy processes are present, wait. Do not kill them.
- Record `nvidia-smi`, memory state, git revisions, and command line.

Benchmark command shape:

```bash
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm bench /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low \
  --device cuda \
  --moe_device "{'cuda':16,'disk':84}" \
  --moe_device_layers -1 \
  --cuda_shared_expert true \
  --gpu_mem_ratio 0.98 \
  --cuda_slab 256 \
  --input_tokens 128 \
  --output_tokens 256 \
  --batch 1 \
  --warmup 0 \
  --temperature 0 \
  --threads 8
```

Required repeats:

- at least 3 repeats for any SOTA claim
- record p50 and worst observed value

Required metrics:

- TTFT
- decode token rate after first token
- total token rate
- prefill token rate
- total time
- expert read bytes
- expert read bandwidth
- VRAM hit estimate
- RAM cache hit rate
- CPU RSS
- GPU VRAM usage
- output hash
- smoke pass rate

## 4. Smoke Test Protocol

Create a fixed `smoke_eval.jsonl` inside this task directory before reporting accuracy.

Minimum cases:

- 4 factual QA
- 4 simple math or reasoning
- 4 coding
- 4 JSON/tool-style formatting
- 4 multilingual

Evaluation:

- Run baseline first with no intentional quality-loss optimization.
- Record baseline pass rate in `progress.md`.
- Every optimized route must satisfy:
  - pass rate `>50%`
  - pass rate no more than `15 percentage points` below the baseline pass rate
  - no empty output
  - no gibberish
  - no systematic failure on JSON, math, or coding cases

## 5. Progress Recording Rules

All work for this task must be recorded in:

- `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-flash-ssd-5tps-optimization/progress.md`

Every progress entry must include:

- timestamp
- route name
- git status before and after
- exact commands
- logs and artifact paths
- metrics table
- pass/fail decision against hard gates
- commit hash if committed
- rollback hash or reverted files if rolled back
- next action

Do not modify the previous task plan at:

- `/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd/plan.md`

## 6. Initial Assumptions

- Full DeepSeek-V4-Flash model is already available at `/home/wici/lfz/models/DeepSeek-V4-Flash`.
- fastLLM repo is available at `/home/wici/lfz/fastllm`.
- vramctl repo is available at `/home/wici/lfz/vramctl`.
- fio reaching `12 GB/s` is treated as a hardware upper bound only; actual expert-path bandwidth must be measured separately.
- All reported main results must preserve active expert count and routing semantics.
