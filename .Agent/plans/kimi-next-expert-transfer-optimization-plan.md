# Kimi Next Expert Transfer Optimization Plan

Date: 2026-07-07
Branch: `vendor/kimi-speculative-general-token-rate-16gb`
Reference survey: `docs/kimi-external-repo-optimization-survey.md`

## Objective

Improve prompt-general Kimi decode token rate under the real deployment target:

- Host RAM must stay below 16 GB, including page cache and process memory.
- GPU is RTX 5090 class, about 32 GB VRAM. Use VRAM as fully as possible.
- Runs must be cold-start runs.
- TTFT must not increase by more than 20% versus the accepted baseline.
- Output quality must remain correct and coherent. `Please introduce France in a short paragraph.` remains a required regression prompt.
- Optimization must be prompt-general, not prompt-specific. Dev prompts may be used for tuning; held-out prompts are used only for final SOTA validation.
- Any accepted improvement must be committed and pushed with exact reproduction commands and env vars. Any regression must be reverted or left disabled by default.

Current prompt-general SOTA reference:

- Branch: `vendor/kimi-speculative-general-token-rate-16gb`
- Accepted GP4 held-out profile root: `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- Held-out quality: 6/6
- n96 token rate: min/median/mean `1.14 / 1.385 / 1.367 tok/s`
- Host RAM peak: `15899996160` bytes
- `direct_reads=0`

## Current Bottleneck Hypothesis

The primary bottleneck is not raw SSD bandwidth in isolation. Pure IO benches can reach about 10.0-10.4 GiB/s, but real decode is lower because the runtime does not expose enough independent expert read tasks at once.

The current expert-pack path is op-local:

1. Current layer computes routing/topK.
2. The MoE op builds a read plan for the current active experts.
3. It submits reads, waits for completions, stages/H2D copies, computes, then returns.
4. There is no global cross-layer producer that keeps IO work queued across layer boundaries.

Observed problem shape:

- Runtime IO batches are usually small, often about 2-8 experts.
- Average in-flight work is much lower than pure IO bench depth.
- up/gate miss is the main blocking point because current layer cannot continue until up/gate experts are available.
- down overlap exists, but it only covers part of the transfer time.
- A previous planned host-prefetch attempt was rejected because useful hits were almost zero and extra read traffic was high.

The next plan therefore starts with observability and low-risk transfer improvements before adding aggressive prediction.

## Phase 0: Reconfirm Baseline and Add Missing Observability

Goal: make the next optimization cycle measurable before changing behavior.

Implementation:

- Add or verify counters for:
  - gate/topK finish timestamp per layer;
  - time from gate/topK availability to first expert read submission;
  - submitted batch size;
  - io_uring queue depth and in-flight depth over time;
  - demand wait time split by up/gate/down;
  - disk read time, pinned staging time, H2D time, compute time;
  - current-down overlap useful time;
  - duplicate reads for the same expert key within a short decode window;
  - fallback reads by tensor type and layer;
  - host RAM peak, `active_file`, `inactive_file`, pinned bytes.

Experiments:

- Re-run n96 cold-start baseline on dev and held-out prompt sets.
- Include `Please introduce France in a short paragraph.`
- Include at least two held-out prompts not used for hotset/pack/predictor tuning.

Acceptance:

- No behavior change unless counters are enabled.
- Counter overhead must be measured; if enabled profiling changes token rate materially, keep it default-off.

Output:

- `.Agent/runs/<date>-kimi-transfer-baseline-profile/report.md`
- exact commands and env vars.

## Phase 1: 2 MiB-Aligned Pinned Staging Experiment

Source idea: flash-moe observed meaningful read-path gains from large aligned DMA buffers. This is lower risk than predictive prefetch because it does not change expert choice or cache policy.

Hypothesis:

If current pinned staging slots are not aligned or sized optimally for NVMe/direct IO and H2D, 2 MiB-aligned staging may raise real expert-pack throughput without adding wrong-path bytes.

Implementation:

- Add an env-guarded mode:
  - `GGML_MOE_STAGE_2M_ALIGN=1`
- Allocate expert-pack staging buffers with 2 MiB alignment where possible.
- Keep total pinned memory within the existing 16 GB host RAM budget.
- Do not change routing, hotsets, expert pack content, or output path.

Experiments:

1. Pure expert-pack IO bench:
   - current staging vs 2 MiB-aligned staging;
   - same offsets, same expert lists, same queue depth;
   - report GiB/s for up/gate and down separately.
2. Real decode:
   - n32 dev quick check;
   - n96 held-out check only if n32 improves or is neutral.

Acceptance:

- Held-out n96 token rate improves or remains neutral with no quality regression.
- TTFT increase <= 20%.
- Host RAM peak < 16 GB.
- If only pure IO improves but real decode regresses, do not accept as SOTA; document the gap.

## Phase 2: MoE-Infinity-Style Global Expert Transfer Scheduler

Source idea: MoE-Infinity uses a priority scheduler with prefetch/demand separation, task deduplication, stale prefetch pruning, and demand consumption of in-flight prefetches.

Hypothesis:

The current runtime loses bandwidth because each MoE op owns its own short IO burst. A global scheduler can preserve useful work across layer boundaries and avoid duplicate reads, making IO less bursty without requiring high-risk prediction.

Important distinction from the rejected host-prefetch attempt:

- Do not blindly read many predicted experts.
- Do not allow prefetch to evict useful demand data.
- Do not issue duplicate demand and prefetch reads for the same expert.
- Demand must always have priority.

Implementation:

- Add a default-off scheduler:
  - `GGML_MOE_GLOBAL_EXPERT_SCHED=1`
  - `GGML_MOE_GLOBAL_PREFETCH_MAX_MIB=<cap>`
  - `GGML_MOE_GLOBAL_PREFETCH_WINDOW=<layers>`
  - `GGML_MOE_GLOBAL_PREFETCH_MAX_EXTRA_READ_RATIO=<ratio>`
- Task key:
  - `(tensor_kind, layer, expert, byte_offset, byte_size, quant_type)`
- Task states:
  - queued;
  - in-flight;
  - ready in host staging;
  - copied to device;
  - canceled/stale;
  - consumed by demand.
- Priority:
  - priority 0: demand;
  - priority 1: safe prefetch.
- Required behaviors:
  - deduplicate queued/in-flight/ready tasks;
  - demand steals or waits for matching in-flight prefetch;
  - stale prefetch is canceled or ignored when the layer window passes;
  - prefetch bytes are capped;
  - demand path can bypass scheduler if the scheduler is not ready.

Phase 2A: no-prediction scheduler

- Only schedule tasks that are already known from current routing and current down overlap.
- Goal is to prove the scheduler itself does not add latency and can deduplicate/wait/steal correctly.

Phase 2A.0: shadow task table

- Add a default-off observability mode:
  - `GGML_MOE_GLOBAL_EXPERT_SCHED_SHADOW=1`
- This mode must not change read submission, wait order, H2D copies, cache policy, routing, or output.
- It maintains only an in-process active task table for expert-pack io_uring read jobs.
- Task key:
  - `(source_idx, read_offset, read_size)`
  - This is the exact physical read task after alias alignment, so it matches what the IO path would deduplicate.
- Counters:
  - total batches and tasks;
  - demand vs prefetch tasks;
  - active duplicate keys;
  - demand request overlapping an active prefetch task;
  - prefetch request overlapping an active demand task;
  - same-priority active duplicates;
  - max active tasks and max batch size.
- Theory:
  - If duplicate/steal candidates are near zero, a no-prediction global scheduler cannot improve much without adding future prediction.
  - If demand-over-active-prefetch is nonzero, a real scheduler can replace duplicate reads with waiting/stealing the existing task.
  - If max active task count remains low even with multiple current threads, the bottleneck is lack of known future work, not only local queue depth.
- Experiment:
  - n32 cold-start France with correct GP4 alias env;
  - compare no-shadow vs shadow to ensure overhead is negligible;
  - record shadow counters in stderr/metrics.
- Acceptance for keeping the probe:
  - default behavior unchanged;
  - n32 quality passes;
  - host RAM remains under 16 GB;
  - shadow overhead is small enough for diagnostic use.

Result on 2026-07-07:

- n32 cold-start France A/B run:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260707-global-sched-shadow-n32-ab`
  - baseline: `1.68 tok/s`, decode `18474.01 ms / 31`, TTFT `87776.30 ms`
  - shadow: `1.71 tok/s`, decode `18114.25 ms / 31`, TTFT `87553.93 ms`
  - quality passed in both runs;
  - host RAM peak stayed at `15899996160` bytes;
  - `direct_reads=0`, `missing_pack=0`, `entries=69120`.
- Shadow counters:
  - `batches=5361`
  - `tasks=23501`
  - `demand=19828`
  - `prefetch=3673`
  - `active_duplicates=0`
  - `demand_hit_prefetch=0`
  - `prefetch_hit_demand=0`
  - `max_active=16`
- Decision:
  - keep the default-off shadow probe as Phase 2 observability;
  - do not implement a dedup/steal-only scheduler yet, because this trace shows no duplicate active physical read tasks to steal;
  - next useful scheduler work needs either prediction/known-future enqueue or byte reduction/layout changes.

Phase 2A.1: down prefetch fine-wait probe

- Add a default-off runtime switch:
  - `GGML_MOE_DOWN_PREFETCH_FINE_WAIT=1`
- Current SOTA env enables `GGML_MOE_CURRENT_DOWN_OVERLAP=1`; in that mode the old `preload_registered_down_for_active()` path is skipped, while current-down overlap already records per-slot ready events and joins its worker before the up/gate call returns.
- The down batch entry still performs a coarse `cudaStreamSynchronize(prefetch_stream)` whenever `GGML_MOE_PREFETCH_DOWN=1` is set.
- Hypothesis: this stream-wide synchronization is redundant under current-down overlap and may add exposed latency or serialize unrelated prefetch-stream work. Skipping it while relying on existing per-slot `batch_cache_wait_slot_ready()` should preserve correctness and may reduce down-stage wall time.
- Theoretical upper bound:
  - If the stream sync is a pure no-op, expected gain is approximately zero.
  - If it waits on stale or unrelated prefetch-stream work, the maximum gain is the measured sync wait at down entry; this is expected to be small but is a low-risk test because it does not change expert IDs, bytes, cache policy, or kernels.
- Experiment:
  - n32 cold-start `Please introduce France in a short paragraph.`
  - compare current SOTA env vs current SOTA env plus `GGML_MOE_DOWN_PREFETCH_FINE_WAIT=1`;
  - record token rate, TTFT, decode time, host RAM peak, full output, iouring wait, current-down overlap counters, and down batch profile.
- Acceptance:
  - if n32 improves or is neutral and quality/RAM/TTFT hold, run held-out n96 before accepting;
  - if n32 regresses, keep the switch disabled and document the result as rejected.

Result on 2026-07-07:

- Correct GP4-alias n32 A/B run:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260707-fine-wait-n32-ab-gp4alias`
  - baseline: `1.74 tok/s`, decode `17836.48 ms / 31`, TTFT `78236.87 ms`
  - fine-wait: `1.74 tok/s`, decode `17809.93 ms / 31`, TTFT `85794.85 ms`
  - quality passed in both runs;
  - host RAM peak stayed at `15899996160` bytes;
  - `direct_reads=0`, `missing_pack=0`, `entries=69120`.
- Rejected as a performance optimization because decode gain was negligible and TTFT rose by about `9.7%`.
- Runtime code was reverted to the prior SOTA behavior. The separate reproduction-runner fix that adds the GP4 alias env is retained because it prevents invalid non-SOTA sweeps.
- Invalid run to ignore:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260707-fine-wait-heldout-n96-test`
  - It omitted `GGML_MOE_EXPERT_GGUF_ALIAS_TSV` and `GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1`, causing `direct_reads`/`missing_pack` and artificially low token rates. It is not comparable to the accepted GP4 SOTA.

Phase 2B: known-late prefetch

- Once current layer topK is known, enqueue down and any immediately known follow-up work as early as possible.
- Measure whether gate/topK-to-read-submit time drops.

Phase 2C: shadow next-layer topK only after overhead proof

- Use next-layer shadow gate only if per-layer sync/copy overhead is measured and acceptable.
- Start in shadow-only mode that records predicted experts but performs no extra reads.
- Enable reads only if held-out useful-hit rate and extra-read ratio pass thresholds.

Acceptance:

- No quality regression on held-out prompts.
- n96 held-out token rate improves.
- TTFT increase <= 20%.
- Host RAM peak < 16 GB.
- Extra-read ratio must be reported and bounded.
- If scheduler overhead cancels gains, keep disabled and document the measured blocker.

## Phase 3: Co-Occurrence-Aware Expert Pack Layout

Source idea: flash-moe suggests expert co-location can reduce small-command and locality overhead. We already have expert packs, but their internal order is not necessarily optimized for Kimi route co-occurrence.

Hypothesis:

If experts that are frequently requested together are physically close in the pack, the runtime may improve read submission locality and reduce command overhead. This will not solve unknown future routing, but it may improve current-layer burst efficiency.

Implementation:

- Generate route co-occurrence statistics from dev prompts only.
- Do not inspect held-out routes while designing the layout.
- Build an alternate expert-pack manifest with exact alias mapping preserved.
- Reorder within safe boundaries only; do not change tensor values or quantization.

Experiments:

- Pure IO pattern replay using dev and held-out traces.
- n32 dev decode.
- n96 held-out decode only after layout is frozen.

Acceptance:

- Held-out token rate improves.
- No held-out prompt is used to tune the layout.
- Host RAM and TTFT constraints hold.
- If dev improves but held-out does not, reject as prompt-specific.

Offline layout bound on 2026-07-07:

- Dev-only route-detail input:
  - `.Agent/runs/20260707-gp30-route-detail-dev-n32/*/route-detail.csv`
- Current physical layout metadata:
  - `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack`
  - `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv`
- Tool:
  - `.Agent/run-tools/kimi_route_detail_pack_layout_bound.py`
- Report:
  - `.Agent/runs/20260707-route-detail-pack-layout-bound/report.md`
- Result:
  - current layout: `gap/read ~= 30.0`, `adjacent/job ~= 0.023`, `coalesce rows = 0.0%`;
  - first-use layout: `gap/read ~= 18.0`, `adjacent/job ~= 0.179`, `coalesce rows = 0.6%`;
  - frequency layout: `gap/read ~= 19.5`, `adjacent/job ~= 0.084`, `coalesce rows = 0.0%`;
  - greedy-pair layout: `gap/read ~= 19.2`, `adjacent/job ~= 0.353`, `coalesce rows = 0.1%`.
- Decision:
  - do not build a new physical pack for layout-only gains yet;
  - layout improves locality counters but does not reduce read bytes and rarely creates batches that are cheap to coalesce under the current read path;
  - if this path is revisited, it must be paired with explicit contiguous/coalesced read support or a larger block layout that changes the number of read commands/wait waves.

## Phase 4: Conservative Trace Predictor Exploration

Source idea: MoE-Infinity uses historical traces to predict future expert activations. Prior Kimi planned-prefetch failed because the useful-hit rate was too low, so this phase is only allowed after scheduler counters exist.

Hypothesis:

A bounded predictor may help only if it predicts a small number of high-confidence up/gate misses early enough to reduce demand waits without flooding the IO queue.

Implementation:

- Shadow-only first:
  - collect predicted expert IDs;
  - record whether they would have been used;
  - record byte recall and false-positive bytes.
- Train/tune only on dev prompts.
- Final metrics only on held-out prompts.
- Enforce extra-read cap before enabling real reads.

Acceptance:

- Shadow held-out byte recall must be high enough to reduce demand waits.
- False-positive bytes must be low enough that real IO would not starve demand.
- Real-read mode must improve held-out n96 token rate and pass all quality/RAM/TTFT limits.

Route-detail predictor bound on 2026-07-07:

- Dev-only route-detail input:
  - `.Agent/runs/20260707-gp30-route-detail-dev-n32/*/route-detail.csv`
- Tool:
  - `.Agent/run-tools/kimi_route_detail_predictor_bound.py`
- Report:
  - `.Agent/runs/20260707-route-detail-predictor-bound/report.md`
- Result:
  - no predictor passed the gate;
  - best under `1.35x` predicted bytes was `previous_same_layer`, but recall was only:
    - upgate: `32.0%` byte recall, `0.05%` full-step coverage, `0.97x` predicted bytes;
    - down: `32.2%` byte recall, `0.05%` full-step coverage, `0.97x` predicted bytes;
  - highest recall was `hybrid_lfu k=64`:
    - upgate: `63.1%` byte recall but `8.0x` predicted bytes;
    - down: `63.1%` byte recall but `8.0x` predicted bytes.
- Decision:
  - do not implement a simple previous-token/LFU predictor as a real prefetcher;
  - any runtime predictor must use stronger information than recent route history, such as router-level signals or a much smaller high-confidence candidate set;
  - otherwise focus on reducing expert bytes or changing pack/layout rather than adding false IO.

## Phase 5: Structural Byte-Reduction Gate

Purpose:

Use the current prompt-general SOTA profile to quantify how much expert movement
must shrink before any optimization can plausibly reach `5 tok/s`. This phase is
a planning gate before investing in new runtime formats.

Important rule:

- Held-out SOTA profiles may be used only to measure the target gap after a
  candidate has already been frozen. They must not be used to select hot experts,
  tune prompt-specific packs, or train a predictor.
- Dev profiles may be used for candidate design, but final SOTA claims still
  require held-out validation.

Tool update on 2026-07-07:

- Tool:
  - `.Agent/run-tools/kimi_byte_reduction_target_bound.py`
- Added:
  - evidence-scope labeling;
  - active expert footprint estimate before cache;
  - transfer-only byte ratio bound;
  - optimistic all-hit MoE floor estimate from up/down profile CSV;
  - stricter byte ratio after reserving time for the all-hit MoE floor.

Accepted held-out SOTA bound:

- Report:
  - `.Agent/runs/20260707-kimi-5tps-byte-reduction-bound-heldout-sota/report.md`
- Evidence scope:
  - accepted held-out SOTA profile only;
  - no hotset, predictor, or pack selection is derived from held-out routes.
- Result:
  - current moved bytes: median `4.23 GiB/token`, mean `4.39 GiB/token`;
  - active expert footprint before cache: median/mean `8.10 GiB/token`;
  - optimistic all-hit MoE floor: median `40.5 ms/token`, mean `40.1 ms/token`;
  - at the measured pure IO upper bound of `10.4 GiB/s`, after reserving the
    all-hit MoE floor, median required byte ratio is `0.39x`;
  - worst prompt requires `0.30x`;
  - role split of miss bytes is broad: down `37.6%`, up `31.2%`, gate `31.2%`.

Current-SOTA dev bound:

- Report:
  - `.Agent/runs/20260707-kimi-5tps-byte-reduction-bound-dev-sota/report.md`
- Evidence scope:
  - current SOTA dev profile only.
- Limitation:
  - this dev profile has route profiles but not up/down profile CSV, so it can
    estimate moved bytes but not the all-hit MoE floor. Do not interpret its
    `floor=0` rows as a compute conclusion.
- Result:
  - moved bytes are still about `4.36 GiB/token` on average, matching the
    held-out SOTA scale.

Decision:

- Queue-depth tuning, pack layout-only changes, and simple route-history
  predictors are bounded below the `5 tok/s` target.
- A viable `5 tok/s` path needs a representation or execution change that
  reduces moved expert bytes to about `0.30x-0.40x` of current IQ3 movement
  while preserving quality.
- Selected IQ1_S/v2 hotsets are not sufficient by themselves:
  - earlier GP46 showed that realistic selected-IQ1_S scale near `0.52x-0.61x`
    does not reach `5 tok/s`;
  - even hypothetical `0.276x` only reaches a transfer-only `5.0 tok/s` bound
    and leaves no room for runtime overhead.

Next actions:

1. Use dev-only samples to evaluate lower-byte expert representations under
   quality-preserving error metrics:
   - residual/base expert clusters;
   - mixed lower-bit up/gate/down candidates;
   - direct compressed compute candidates that avoid reconstructing full expert
     tensors.
2. Reject any candidate whose theoretical byte ratio cannot reach `0.30x-0.40x`
   after accounting for runtime overhead.
3. Only after a candidate passes dev-only error and byte gates, implement a
   default-off runtime path and run cold-start n32 dev quality/perf.
4. Run held-out n96 only once the candidate is frozen.

## Phase 6: Lower-Priority Compute Work

These are not first because the current bottleneck is expert movement, not compute.

Potential work:

- Audit CUDA dequant kernels for FMA-friendly math.
- Clean-room grouped resident expert compute inspired by Unsloth, without copying AGPL code.
- W4A16/Marlin-style resident expert kernels inspired by mllm only if format conversion and quality are justified.

Acceptance:

- Only pursue after transfer stalls are reduced or profiles show compute dominates.
- Any activation quantization or activation sparsity scan must include total M=1 overhead, not just GEMM time.

## Run Discipline

For every experiment:

1. Update this plan or a linked run note before running the experiment.
2. Use cold start.
3. Record exact command and env vars.
4. Record full model answer for quality review.
5. Record token rate, TTFT, decode time, host RAM peak, page cache, pinned bytes, direct reads, iouring bytes, H2D time, demand wait, and scheduler counters.
6. Separate dev prompts from held-out prompts.
7. Commit and push immediately only when a change passes constraints and improves prompt-general held-out performance.
8. Revert or keep default-off any change that regresses token rate, quality, TTFT, or RAM.

## Immediate Next Task

Continue from Phase 5:

1. Build a dev-only lower-byte representation screen that samples real active
   up/gate/down experts and reports byte ratio plus reconstruction/activation
   error.
2. Prioritize candidates that can plausibly reach `0.30x-0.40x` total moved
   bytes; reject candidates above that bound unless they also remove exposed IO
   through direct compressed compute.
3. Do not build more prompt-specific hot expert overlays. GP57 showed dev
   overlay gains can regress held-out performance severely.
4. Keep scheduler/predictor work default-off unless a shadow predictor can show
   high useful byte recall under a strict extra-read cap.

Rationale:

- The current accepted SOTA already moves only about half of the active expert
  footprint, yet it remains at `~1.37 tok/s`.
- At `10.4 GiB/s`, the `5 tok/s` budget after an optimistic MoE floor allows
  only about `1.6-1.7 GiB/token` of expert movement on typical prompts.
- This requires structural byte reduction, not another small IO scheduling
  adjustment.
