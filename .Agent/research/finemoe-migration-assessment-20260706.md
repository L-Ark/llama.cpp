# FineMoE Migration Assessment

Date: 2026-07-06

Source cloned locally:

- Repository: `https://github.com/IntelliSys-Lab/FineMoE-EuroSys26`
- Local path: `/Users/spark/FineMoE-EuroSys26`
- HEAD at read time: `5c584686da077676e3854a363832e9e7b973a054`
- Paper/demo title: `Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading`

## What FineMoE Implements

FineMoE is a PyTorch/HuggingFace MoE offloading system built on MoE-Infinity.
It targets serving models where inactive experts are offloaded from GPU memory
to CPU memory and prefetched back to GPU.

The main mechanism is not a new expert compute kernel. It is a predictor-driven
expert placement policy:

- collect per-request expert trajectories;
- store pairs of prompt/input embeddings and expert maps;
- at the start of a request, match current input embeddings against historical
  embeddings to predict the first `prefetch_distance` layers;
- during execution, match the observed partial expert trajectory against
  historical trajectories to predict later layers;
- turn predicted expert maps into prioritized prefetch tasks;
- protect predicted cache candidates during eviction;
- evict lower-value sparse nodes using a probability/visit based priority.

Important files:

- `/Users/spark/FineMoE-EuroSys26/README.md`
- `/Users/spark/FineMoE-EuroSys26/finemoe/runtime/model_offload.py`
- `/Users/spark/FineMoE-EuroSys26/finemoe/memory/expert_prefetcher.py`
- `/Users/spark/FineMoE-EuroSys26/finemoe/memory/expert_tracer.py`
- `/Users/spark/FineMoE-EuroSys26/finemoe/models/modeling_qwen/modeling_qwen2_moe.py`
- `/Users/spark/FineMoE-EuroSys26/core/prefetch/task_scheduler.cpp`
- `/Users/spark/FineMoE-EuroSys26/core/aio/archer_prio_aio_handle.cpp`

## Why It Is Relevant

Our current Kimi path is still sensitive to routed expert residency:

- Kimi IQ3_S split-pool SOTA improved token rate by using VRAM more efficiently
  for expert cache slots.
- The accepted split-pool optimization is exact: it changes cache placement,
  not model math.
- Remaining misses still trigger expert-pack reads, pinned staging, and H2D
  copies.
- Adding more hot experts blindly can hurt because it changes eviction pressure,
  staging pressure, and sometimes lowers effective throughput even when hit
  rate rises.

FineMoE's strongest transferable idea is to make the cache/prefetch policy
less static:

- static hotset: preload globally frequent experts;
- current Kimi overlap: prefetch selected down once routing for the current
  layer is known;
- FineMoE-style policy: predict future layer experts from the prompt and
  already-observed route prefix, then reserve/protect those experts before they
  are needed.

This is aligned with our constraints because it can remain exact. It does not
change quantization, routing, logits, or expert math.

## What Can Be Migrated

### Candidate 1: Offline Route-Trajectory Predictor

Build a small offline analysis tool over our existing Kimi route traces:

1. For each deterministic benchmark prompt, record a route trajectory:
   - token index;
   - layer;
   - selected experts;
   - tensor role if available: `gate`, `up`, `down`;
   - cache hit/miss and transfer time.
2. For a target run, predict layer `L + d` from the prefix `0..L`.
3. Evaluate prediction quality without changing runtime:
   - top-k expert recall;
   - byte-weighted recall;
   - miss-time-weighted recall;
   - false-positive bytes;
   - earliest layer where prediction becomes stable.
4. Only continue if prediction recall is materially better than the current
   global hotset and current-down prefetch.

This is the safest first step. It requires no runtime changes and directly
answers whether FineMoE's predictor is useful for Kimi.

### Candidate 2: Predicted Candidate Protection For VRAM Cache

If offline prediction is strong, add an env-gated protection score to the
existing MoE VRAM cache:

- keep current split-pool cache and exact expert math;
- when route-prefix prediction says an expert is likely needed soon, mark the
  corresponding cache entry as protected for a limited future layer window;
- evict unprotected or low predicted-value entries first;
- never exceed the existing VRAM budget or 16GB host-RAM cap.

This maps to FineMoE's `replace_cache_candidates()` idea but should be
implemented inside our existing cache metadata rather than by porting Archer.

Suggested score:

```text
evict_score = recent_hit_score + alpha * predicted_prob - beta * byte_cost
```

Use byte-weighting because Kimi expert sizes differ by role and quant type.

### Candidate 3: Probabilistic Prefetch Queue

FineMoE sorts predicted experts by priority and enqueues prefetch tasks. For
Kimi, this should be adapted to the current expert-pack/io_uring path:

- keep current deterministic current-layer down prefetch;
- add a speculative future-layer queue with a strict byte/time budget;
- submit only if SSD/H2D workers are not on the critical path;
- cancel or deprioritize predictions once real routing disagrees;
- track wasted bytes separately from useful prefetched bytes.

This must not compete with known-required current-layer transfers. The first
implementation should be a measurement-only dry run that logs what would have
been prefetched.

## What Should Not Be Migrated Directly

- Do not port FineMoE's PyTorch/HuggingFace monkey-patching layer. Our runtime
  is llama.cpp/ggml, not HF modules.
- Do not port Archer's AIO implementation. It uses a single AIO thread and
  1 MiB block scheduling; our current io_uring/expert-pack path is closer to the
  real bottleneck and already has Kimi-specific batching.
- Do not assume FineMoE's multi-GPU placement policy applies. Our active Kimi
  target is single-node, strict 16GB host RAM, and VRAM should be filled by the
  existing cache budget.
- Do not use prompt embedding matching as the first runtime feature. It requires
  storing embeddings and adding similarity compute. Start with route-prefix
  matching because our deterministic benchmark traces already contain the
  relevant signal.

## Expected Benefit

FineMoE is not a payload-compression method, so its upside is bounded by
avoidable cache misses and prefetch stalls.

For Kimi, a realistic target is modest unless route prediction is very strong:

- low-risk expected gain: `+3%` to `+8%` token rate by reducing harmful evictions
  and avoiding late prefetches;
- optimistic gain: `+10%` to `+15%` if future-layer route prediction sharply
  reduces miss transfers without increasing wasted prefetch bytes;
- unlikely to double token rate by itself because it does not reduce expert
  bytes, expert compute, or required H2D volume for true misses.

The likely best use is improving stability around the current SOTA rather than
creating a new large jump.

## Required Phase 0 Bound

Before runtime changes, build a prediction-bound artifact:

1. Use existing Kimi route traces or collect new n96/n32 traces under strict
   cold-start settings.
2. Implement an offline route-prefix predictor:
   - exact prefix match if enough examples exist;
   - fallback to layer-local transition table;
   - optionally compare with prompt-level/global hotset.
3. Report for each prefetch distance `d = 1..8`:
   - top-k recall;
   - byte-weighted recall;
   - miss-time-weighted recall;
   - false-positive bytes;
   - predicted cache residency pressure by split pool.
4. Compare against current accepted Kimi SOTA counters:
   - VRAM cache hit rate by pool;
   - expert-pack bytes;
   - pinned staging copies/time;
   - H2D time;
   - TTFT and token rate.

Promotion criterion for runtime work:

- predicted useful bytes must exceed false-positive bytes by at least `3x`;
- predicted misses must account for at least `5%` of current decode time;
- predicted protection must not require more VRAM than current split-pool budget;
- no increase in host RAM/page cache is allowed.

If this bound fails, FineMoE should be rejected for Kimi and kept only as a
research note.

Initial Phase 0 result:

- Tool: `.Agent/run-tools/kimi_finemoe_route_prefix_bound.py`
- Input trace: `.Agent/runs/20260706-kimi-finemoe-phase0/route_trace.csv`
- Source run: `/root/lfz/runs/ik_llama/kimi-iq3s-16gb-accepted-default-fulltrace-n64-001-20260627-011039Z/route_trace.csv`
- Artifact: `.Agent/runs/20260706-kimi-finemoe-phase0/route-prefix-bound.json`
- Human summary: `.Agent/runs/20260706-kimi-finemoe-phase0/route-prefix-bound.md`
- Reproduce:

```bash
python3 .Agent/run-tools/kimi_finemoe_route_prefix_bound.py --trace .Agent/runs/20260706-kimi-finemoe-phase0/route_trace.csv --out-json .Agent/runs/20260706-kimi-finemoe-phase0/route-prefix-bound.json --out-md .Agent/runs/20260706-kimi-finemoe-phase0/route-prefix-bound.md --max-distance 8
```

The accepted fulltrace contains `64` raw segments; after dropping the initial
single-layer warm segment, `63` decode-like segments are used for the bound.

Summary:

| distance | prefix predictor byte recall | prefix useful/false-positive | global hotset byte recall | global useful/false-positive |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0.4330` | `0.764` | `0.3471` | `0.532` |
| `2` | `0.4324` | `0.762` | `0.3486` | `0.536` |
| `4` | `0.4312` | `0.758` | `0.3507` | `0.541` |
| `8` | `0.4285` | `0.750` | `0.3541` | `0.549` |

Interpretation:

- Route-prefix nearest-neighbor prediction beats the global hotset baseline by
  roughly `7.4-8.6` byte-recall percentage points on this trace.
- However, useful/false-positive bytes are only about `0.75x`, far below the
  `3x` promotion threshold.
- A speculative prefetch queue based on this predictor would likely waste more
  bytes than it helps under the strict 16GB cold-start budget.
- Do not implement FineMoE-style runtime prefetch/protection from this bound.
  Keep it as a research artifact unless a richer multi-prompt trace set shows a
  much stronger predictor.

## Runtime Plan If Phase 0 Passes

1. Add route-trace export for exact selected experts per token/layer if the
   existing traces are insufficient.
2. Add a default-off predictor dry-run mode:
   - computes predicted candidates;
   - logs would-prefetch/would-protect entries;
   - performs no IO or cache changes.
3. Add predicted candidate protection:
   - env-gated;
   - cache metadata only;
   - no speculative IO yet.
4. Validate strict cold n96:
   - France prompt output must remain semantically correct;
   - TTFT increase <= 20%;
   - host RAM < 16GB including page cache;
   - token rate must beat same-host baseline.
5. Only after protection helps, test speculative prefetch:
   - strict byte budget;
   - never starve current-layer known-required transfers;
   - track wasted bytes and cancellation.

## Current Decision

FineMoE is applicable as a cache/prefetch prediction idea, not as directly
portable code.

Recommended next action:

- do not change runtime yet;
- implement the offline Phase 0 predictor bound first;
- proceed only if Kimi route trajectories are predictable enough to reduce
  real miss time under the 16GB cold-start constraints.

Compared with D2MoE, this path is safer because it is exact. It may improve
token rate modestly by better cache decisions, but it is unlikely to produce a
large jump unless route-prefix prediction is much stronger than the current
global hotset.
