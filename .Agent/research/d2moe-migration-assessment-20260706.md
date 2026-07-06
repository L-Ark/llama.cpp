# D2MoE Migration Assessment

Date: 2026-07-06

Source cloned locally:

- Repository: `https://github.com/lliai/D2MoE`
- Local path: `/Users/spark/D2MoE`
- HEAD at read time: `b3092f0 Update get_scale.py`
- Paper linked by the repo: `D^2-MoE: Delta Decompression for MoE-based LLMs Compression`, arXiv `2502.17298`

## What D2MoE Implements

D2MoE is an offline MoE compression/reparameterization method, not an online
expert-cache or SSD/VRAM scheduler.

For each MoE layer it replaces the original expert matrices with:

- one shared base expert per layer:
  - `Wmean_gate`
  - `Wmean_up`
  - `Wmean_down`
- per-expert low-rank delta matrices:
  - `delta_u1/delta_v1` for gate
  - `delta_u3/delta_v3` for up
  - `delta_u2/delta_v2` for down
- optional shared delta factors across experts:
  - `share_V`
  - `share_U`
- optional hidden-channel pruning of the shared base path through `pp_ratio`.

The merge stage builds the shared base using one of:

- frequency-weighted expert average
- plain expert mean
- Fisher-weighted average

Then each expert delta is compressed with SVD, optionally scaled by activation
statistics collected from calibration data.

Relevant files:

- `/Users/spark/D2MoE/README.md`
- `/Users/spark/D2MoE/D2-deepseek.py`
- `/Users/spark/D2MoE/model/merge_deepseek.py`
- `/Users/spark/D2MoE/preprocess/get_expert_freq.py`
- `/Users/spark/D2MoE/preprocess/get_fisher.py`
- `/Users/spark/D2MoE/preprocess/get_scale.py`
- `/Users/spark/D2MoE/cal_params.py`

## Why It Is Relevant To Our Current Bottleneck

Our current Kimi/DeepSeek work is dominated by expert movement and fallback
payload, especially up/down movement from SSD/GGUF/expert-pack into CPU/GPU
execution paths under a strict 16GB host RAM budget.

D2MoE attacks a different layer of the problem:

- It reduces per-expert payload by representing experts as base + low-rank delta.
- The base expert can be resident once per layer instead of duplicated per expert.
- The per-selected-expert payload can become low-rank delta only.
- In its `shared_infer` path, the shared up/gate base can be computed once for all
  tokens, and each selected expert only adds its delta contribution.

This maps directly to our long-term objective of reducing per-token expert
bytes and making more useful expert data fit in VRAM.

## What Can Be Migrated

### Candidate 1: Offline Base + Delta Expert Pack Format

Create a new experimental expert-pack variant for Kimi/DeepSeek:

- store per-layer base tensors:
  - `base_gate`
  - `base_up`
  - `base_down`
- store per-expert low-rank deltas:
  - `delta_gate_u/v`
  - `delta_up_u/v`
  - `delta_down_u/v`
- optionally store shared `V` factors once per layer.

Runtime shape:

1. Keep base tensors resident in VRAM if possible.
2. Route as usual.
3. For each selected expert, load only the low-rank delta payload.
4. Compute:
   - `gate = base_gate(x) + delta_gate_u(delta_gate_v(x))`
   - `up = base_up(x) + delta_up_u(delta_up_v(x))`
   - `down = base_down(act(gate) * up) + delta_down_u(delta_down_v(act(gate) * up))`

Expected benefit:

- large reduction in SSD-to-VRAM or SSD-to-CPU bytes for misses;
- more hot expert information can fit into the same VRAM cache;
- could reduce the CPU fallback page-cache pressure by avoiding full expert
  tensor faults.

Main risk:

- semantic quality may regress because this is lossy model compression, not an
  exact runtime optimization.
- Kimi K2 and DeepSeek V4 dimensions/quantization formats are much larger than
  the DeepSeek-MoE-16B code path in D2MoE; the preprocessing pipeline must be
  rewritten for GGUF/sharded expert packs.

### Candidate 2: Frequency/Fisher Weighted Base For Existing Hotset

Before implementing the full low-rank delta runtime, test only the offline
analysis part:

- collect route frequencies from our existing prompt set and n96/n32 profiles;
- compute a per-layer frequency-weighted base expert;
- measure residual norm `expert - base` per expert and tensor type;
- estimate low-rank residual compressibility.

This can be done without changing inference. It gives a hard bound on whether
D2MoE-style compression is worth implementing for Kimi/DeepSeek.

Promotion criterion:

- residual low-rank approximation must preserve fixed prompt outputs at top-1
  or pass the existing semantic gate;
- estimated selected-expert bytes must drop enough to move token rate materially
  under the current 16GB cold-start constraints.

### Candidate 3: Shared Base Up/Gate Compute

D2MoE's `shared_infer` idea is relevant to our current up/gate bottleneck:

- base up/gate can be computed once per layer for the token;
- selected experts add only delta up/gate;
- down remains selected-expert dependent.

If the base tensors are resident in VRAM, this can reduce repeated expert up/gate
movement and make current-down prefetch easier to hide.

This is not drop-in because our current kernels expect full quantized expert
rows. It requires new graph/tensor operators or a decompressed/repacked pack.

## What Should Not Be Migrated Directly

- The PyTorch/Hugging Face module replacement code is not suitable for direct
  integration into llama.cpp runtime.
- The pruning/probe path is not immediately aligned with our strict correctness
  constraints; it changes hidden dimensions dynamically and is likely high-risk
  for semantic stability.
- The current D2MoE repo does not provide SSD, io_uring, pinned staging, CUDA
  graph, expert-pack, or cold-start cgroup machinery.
- It does not solve TTFT/page-cache management directly.

## Recommended Next Experiment

Do not start by rewriting runtime kernels. First build a measurement artifact:

1. Pick one representative layer range from Kimi or DeepSeek where fallback bytes
   dominate.
2. Load full expert tensors offline.
3. Build frequency-weighted base experts using our existing route profiles.
4. Compute residual SVD curves for up/gate/down separately:
   - rank vs residual error;
   - rank vs packed byte size;
   - estimated selected-expert transfer bytes.
5. Use a tiny CPU/Python reconstruction harness to compare full expert output
   vs base+delta output on captured hidden states if available.
6. Only if this bound is strong, design a new default-off expert-pack format and
   runtime path.

This should be treated as a model-compression candidate, not a low-risk runtime
optimization. It must go through the same correctness gates as quantization:

- France prompt semantic correctness;
- existing five-prompt set;
- fixed-output/top-1 comparison where possible;
- 16GB strict cold-start;
- TTFT increase <= 20%;
- commit/push only after compliant improvement.

## Current Assessment

D2MoE has one potentially valuable idea for us:

> use per-layer shared base experts plus low-rank per-expert deltas to reduce
> selected-expert payload and make more effective expert information resident in
> VRAM.

It is not immediately mergeable code. The correct migration path is a
default-off offline feasibility study followed by a new expert-pack format if
the low-rank residual bound is strong.

## Kimi-Specific Optimization Plan

This plan adapts the D2MoE idea to the current Kimi 16GB cold-start runtime. The
goal is not to port D2MoE's PyTorch modules. The goal is to reduce selected
expert payload in our SSD/expert-pack/VRAM-cache stack by rewriting expert
storage as a hybrid full-weight plus base/delta format.

Current Kimi reference:

- Current merged-branch n96 cold-start line: about `1.35-1.36 tok/s`.
- Fastest recorded accepted reference: `1.38 tok/s`.
- Quality gate prompt: `Please introduce France in a short paragraph.`
- Constraints: strict cold start, host RAM below 16GB including page cache,
  semantic correctness, and TTFT increase no more than 20%.

Expected outcome if the residual bound is favorable:

- Realistic target: `1.8-2.3 tok/s`.
- Optimistic target: `2.5-3.0 tok/s`.
- Required condition: selected expert payload must fall to roughly `30%-50%` of
  current full-expert bytes without breaking semantic quality.

### Phase 0: Offline Residual SVD Bound

Do this before any runtime source work.

Initial implementation artifact:

- Tool: `.Agent/run-tools/kimi_d2moe_phase0_plan.py`
- Run directory: `.Agent/runs/20260706-kimi-d2moe-phase0/`
- Inventory: `.Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv`
- Bound input plan: `.Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json`
- Human summary: `.Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.md`

Initial result:

- Selected representative layers: `1, 13, 15, 26, 27, 29, 31, 56, 57, 60`.
- Selected tensors: `30` up/gate/down tensors.
- Estimated BF16 resident base footprint for the selected tensor set: `0.820 GiB`.
- Rank64 low-rank delta byte estimates are about `15%-25%` of current full
  quantized expert bytes depending on tensor type.
- Rank128 low-rank delta byte estimates are about `30%-50%` of current full
  quantized expert bytes.
- This is only a payload/input-selection bound. It does not yet dequantize
  tensors or prove residual quality.

Dequantization smoke result:

- Tool: `.Agent/run-tools/kimi_d2moe_dequant_sample.py`
- Artifact: `.Agent/runs/20260706-kimi-d2moe-phase0/dequant-sample-summary.json`
- Human summary: `.Agent/runs/20260706-kimi-d2moe-phase0/dequant-sample-summary.md`
- Server command:

```bash
python3 .Agent/run-tools/kimi_d2moe_dequant_sample.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260706-kimi-d2moe-phase0/dequant-sample-summary.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/dequant-sample-summary.md --max-per-type 1
```

Observed sample coverage:

- `Q3_K`: `blk.1.ffn_down_exps.weight`, expert `116`, finite ratio `1.0`.
- `IQ2_S`: `blk.1.ffn_gate_exps.weight`, expert `116`, finite ratio `1.0`.
- `Q4_0`: `blk.15.ffn_down_exps.weight`, expert `336`, finite ratio `1.0`.
- `IQ4_XS`: `blk.26.ffn_down_exps.weight`, expert `293`, finite ratio `1.0`.
- `IQ3_XXS`: `blk.29.ffn_gate_exps.weight`, expert `227`, finite ratio `1.0`.

Conclusion:

- The selected GGUF expert byte ranges can be addressed directly from the
  inventory offsets and decoded through this checkout's `libggml-base.so`.
- This removes the immediate quant-format blocker for real Phase 0 residual
  SVD work.
- It is still not a quality or token-rate claim; the next required step is
  building a weighted base and residual-rank curve on decoded experts.

First residual-rank sample:

- Tool: `.Agent/run-tools/kimi_d2moe_residual_rank.py`
- Artifact: `.Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-sample.json`
- Human summary: `.Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-sample.md`
- Tensor: `blk.1.ffn_down_exps.weight`
- Quant type: `Q3_K`
- Expert sample: top 4 route-profile experts `116, 23, 160, 137`
- Weighted base: route-count weighted mean over the sampled experts
- Command:

```bash
python3 .Agent/run-tools/kimi_d2moe_residual_rank.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --phase0-plan .Agent/runs/20260706-kimi-d2moe-phase0/phase0-bound-input-plan.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-sample.json --out-md .Agent/runs/20260706-kimi-d2moe-phase0/residual-rank-sample.md --preferred-kind down --max-experts 4 --ranks 16,32,64,128 --oversample 8 --niter 1 --torch-threads 8
```

Observed rank curve:

- Rank16 BF16 delta is `4.68%` of the current quantized full expert bytes, but
  leaves residual norm ratio around `0.992`.
- Rank64 BF16 delta is `18.70%` of the current quantized full expert bytes, but
  leaves residual norm ratio around `0.971`.
- Rank128 BF16 delta is `37.40%` of the current quantized full expert bytes, but
  leaves residual norm ratio around `0.946`.

Interpretation:

- For this sampled early-layer `down` tensor, residual energy is not
  concentrated in low rank. Rank128 explains only about `10.3%-10.4%` of
  residual energy.
- Do not promote a down-only D2MoE runtime path from this result.
- Before rejecting the whole D2MoE direction, run the same residual-rank bound
  for `gate` and `up`, plus middle/late layers. The D2MoE idea is only worth
  runtime work if some tensor classes or layer bands show much stronger
  residual compression than this first `down` sample.

1. Select a small but representative Kimi layer set:
   - at least one early sparse layer;
   - at least one middle high-traffic layer;
   - at least one late layer;
   - include layers where current fallback / expert-pack movement dominates.
2. Extract Kimi expert tensors for `up`, `gate`, and `down` from the current
   GGUF or expert-pack assets.
3. Build per-layer weighted base tensors:
   - primary weighting: Kimi n96/n32 route frequency;
   - fallback weighting: plain mean if route frequency is unavailable;
   - record the exact profile used.
4. For each tensor kind and expert, compute residual:
   - `residual = expert_weight - weighted_base_weight`.
5. Run SVD curves for residual ranks such as `16, 32, 64, 96, 128, 192, 256`.
6. Record:
   - residual norm ratio;
   - estimated packed bytes;
   - compression ratio vs current full expert tensor;
   - per-layer and per-tensor outliers;
   - expected selected-expert transfer-byte reduction.
7. Stop if useful ranks are too high:
   - reject the D2 route if rank needed for acceptable residual is near full
     matrix size;
   - continue only if low-rank deltas plausibly reduce selected-expert bytes by
     at least 2x.

Deliverable:

- A JSON/markdown artifact under `.Agent/runs/` with the residual curves,
  byte estimates, source tensor paths, and exact script command.

### Phase 1: Down-Only Delta Prototype

Start with `down` because it is most aligned with the current movement bottleneck
and with current-down prefetch/overlap.

Runtime shape:

- Keep `base_down` resident in VRAM where possible.
- Store selected experts' `down_delta_u/down_delta_v` in an experimental pack.
- On active expert miss, load only `down_delta`.
- Compute full output as:
  - `base_down(act(gate) * up) + delta_down_u(delta_down_v(act(gate) * up))`.
- Keep the existing full expert path as correctness fallback.
- Gate the entire path behind a default-off env flag.

Acceptance gate:

- France prompt semantic quality must pass.
- n96 token rate must exceed current same-host baseline.
- TTFT must stay within `baseline * 1.2`.
- 16GB memory cap must include page cache and remain clean.

Reject immediately if:

- output becomes repetitive, incoherent, or semantically wrong;
- TTFT rises more than 20%;
- memory peak exceeds the strict cap;
- token rate is equal or lower after repeated cold starts.

### Phase 2: Hybrid Full/Delta Expert Pack

Do not D2-compress every expert uniformly. Use a tiered format:

- Tier A: hottest experts remain full-weight and use the existing hot expert
  cache path.
- Tier B: medium-frequency experts use base plus medium-rank delta.
- Tier C: cold experts use base plus low-rank delta or fallback to the existing
  full expert path.

Pack metadata must include:

- layer id;
- expert id;
- tensor kind: `up`, `gate`, or `down`;
- rank;
- quant type;
- residual error summary;
- route frequency;
- tier;
- source tensor checksum or source path.

This keeps hot experts fast and exact while reducing bytes for the long tail.

### Phase 3: Base Up/Gate Resident Compute

After down-only succeeds, extend to `up` and `gate`.

Runtime schedule:

1. Routing identifies active experts.
2. Start prefetch for selected `down_delta`.
3. Compute resident `base_up(x)` and `base_gate(x)` once for the layer.
4. Load selected `up_delta/gate_delta`.
5. Add expert-specific delta contribution.
6. Compute activation.
7. Finish resident `base_down + down_delta`.

This is the D2MoE `shared_infer` idea adapted to our existing overlap model:
base compute is shared and resident, while selected deltas are small enough to
make transfer easier to hide.

### Phase 4: Adaptive Rank And Speculative Delta Rank

If the first delta pack works, make rank adaptive instead of global.

Rank policy:

- higher rank for hot or quality-sensitive experts;
- lower rank for cold experts;
- separate rank budgets for `up`, `gate`, and `down`;
- conservative ranks for layers that show high semantic sensitivity.

Optional speculative loading:

- load a low-rank prefix first, such as rank 32;
- compute partial delta as soon as it arrives;
- append rank 64/128 only for risky experts or layers;
- fallback to full expert for outliers.

This should remain experimental until there is a reliable acceptance signal.

### Testing And Promotion Rules

Every phase must record all metrics:

- token rate;
- TTFT;
- prompt/decode timing;
- memory peak and file page cache;
- expert-pack bytes;
- VRAM cache hit/miss;
- exact output text;
- quality decision and reason.

Minimum tests:

- France prompt n96 strict cold start.
- Existing five-prompt semantic set.
- Same-host baseline comparison against current pushed Kimi line.
- Repeated cold-start run for any claimed improvement.

Commit and push only if all of these are true:

- token rate improves over current same-host baseline;
- France output is coherent and semantically correct;
- five-prompt set passes;
- TTFT increase is no more than 20%;
- memory stays below 16GB including page cache;
- the exact reproduction command and run artifact are recorded.

Do not promote source changes that only improve synthetic residual metrics. The
runtime path must improve real Kimi cold-start token rate under the same gates.
