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

Next-gate shadow confirmation:

- Earlier GP64 dev-only `N=16` shadow profiling showed that
  `GGML_MOE_NEXT_GATE_SHADOW_TOPK=8` has useful signal:
  - expert/byte recall about `77.79%`;
  - false/actual bytes about `0.2221`;
  - but `N=16` was too short for final quality gating and TTFT was noisy.
- GP65 `N=32` dev result on 2026-07-07:
  - report:
    `.Agent/runs/20260707-gp65-next-gate-shadow-dev-n32-top8/report.md`;
  - summary:
    `.Agent/runs/20260707-gp65-next-gate-shadow-dev-n32-top8/summary.json`;
  - remote root:
    `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-next-gate-shadow-dev-n32-top8`;
  - prompts with shadow CSV: `7/7`;
  - expert/byte recall: `77.35%`;
  - precision: `77.35%`;
  - false/actual bytes: `0.2265`;
  - max record overhead: `2633 us`;
  - average token rate: `1.659 tok/s`;
  - host RAM peak: `15899996160`;
  - quality: `6/7` pass.
- GP65 decision:
  - do not implement real prefetch yet;
  - the single quality failure was `dev_linear_equation`, whose `N=32` output
    was truncated before the expected final keyword, so the predictor signal is
    not rejected but still needs longer-output confirmation;
  - GP66 `N=96` TopK=8 dev shadow is the next required gate before planning
    bounded runtime prefetch.
- GP66 `N=96` dev result on 2026-07-07:
  - report:
    `.Agent/runs/20260707-gp66-next-gate-shadow-dev-n96-top8/report.md`;
  - summary:
    `.Agent/runs/20260707-gp66-next-gate-shadow-dev-n96-top8/summary.json`;
  - remote root:
    `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp66-next-gate-shadow-dev-n96-top8`;
  - first prompt `dev_france_regression` passed with `1.90 tok/s`, TTFT
    `88618.6 ms`, decode `40632.75 ms / 77`;
  - second prompt `dev_japan_factual` produced no useful output after
    `7min 36.857s` systemd runtime and was killed;
  - cleanup transiently started later prompts, which were killed and ignored.
- GP66 decision:
  - reject TopK=8 next-gate shadow as a direct prefetch implementation gate;
  - do not implement real prefetch from this signal yet;
  - future predictor work must first reduce shadow/predictor overhead or prove
    with a paired baseline that the long TTFT was unrelated to the predictor.

Correction from later GP65 rerun with correct GP4 env:

- The early GP66 rejection used an invalid or unstable command path and is not
  the current predictor gate.
- Corrected GP4 env:
  - `GGML_MOE_EXPERT_GGUF_ALIAS_TSV=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv`;
  - `GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1`.
- Corrected N96 TopK=8 shadow result from
  `.Agent/runs/20260707-gp65-correct-gp4env-n96-analysis/report.md`:
  - shadow CSV: `7/7` prompts;
  - dev quality pass in both baseline and shadow: `7/7`;
  - expert/byte recall: `77.25%`;
  - false/actual bytes: `0.2275`;
  - max TTFT ratio: `1.085`;
  - max decode ratio: `1.027`;
  - mean token-rate ratio shadow/baseline: `0.993`;
  - all shadow runs kept `direct_reads=0` and
    `memory.peak=15899996160`.
- Updated decision:
  - next-gate shadow has enough signal and overhead margin to justify a
    bounded runtime prefetch experiment;
  - the prior generic `GGML_MOE_PLANNED_HOST_PREFETCH=1` path is still rejected
    because useful hits were near zero and evictions were high;
  - the next implementation must prove predicted expert IDs become available
    early enough in the CUDA/MoE path before demand copy creation.

GP71 next-gate bounded policy and execution-order probe:

- Goal:
  - move from generic TopK=8 shadow toward a bounded policy that might be
    practical for runtime prefetch.
- Offline policy bound:
  - tool:
    `.Agent/run-tools/kimi_next_gate_prefetch_policy_bound.py`;
  - input:
    `.Agent/runs/20260707-gp65-correct-gp4env-n96-shadow-top8/*/next-gate-shadow.csv`;
  - method:
    leave-one-dev prompt layer selection, so each prompt is evaluated with
    layer choices derived from the other dev prompts;
  - result:
    `.Agent/runs/20260707-gp71-next-gate-prefetch-policy-bound/report.md`.
- Policy result:
  - `cap1_all`:
    - recall `0.1243`;
    - precision `0.9942`;
    - false/actual `0.0007`.
  - `cap2_all`:
    - recall `0.2457`;
    - precision `0.9829`;
    - false/actual `0.0043`.
  - `cap4_all`:
    - recall `0.4714`;
    - precision `0.9429`;
    - false/actual `0.0286`.
  - `cap8_all`:
    - recall `0.7725`;
    - precision `0.7725`;
    - false/actual `0.2275`.
- Interpretation:
  - TopK order is highly meaningful;
  - `cap4` gives a better implementation target than TopK=8 because false
    prefetch bytes are low while useful byte coverage is still material;
  - these are optimistic because they do not subtract experts already resident
    in VRAM, but they justify a runtime timing probe.
- Execution-order probe:
  - code:
    default-off `GGML_MOE_NEXT_GATE_PREFETCH_PROBE_OUT=<csv>` in
    `src/llama-context.cpp`;
  - requires next-gate shadow nodes to exist, so run with
    `GGML_MOE_NEXT_GATE_SHADOW_OUT=<csv>` and
    `GGML_MOE_NEXT_GATE_SHADOW_TOPK=4`;
  - uses `ggml_backend_sched_eval_callback` to record graph execution order and
    selected expert IDs for:
    - predicted `ffn_moe_next_shadow_topk`;
    - actual `ffn_moe_topk`.
- Why this probe is needed:
  - graph-end CSV recording proves predictor quality but is too late for
    same-token prefetch;
  - runtime prefetch is only worth implementing if the predicted target-layer
    topK node is computed before the actual target layer's demand routing node.
- Acceptance to proceed to real prefetch:
  - n4/n8 dev smoke shows predicted target-layer records appear before the
    corresponding actual target-layer records for most layers;
  - probe output quality remains correct;
  - host RAM remains below 16 GB;
  - probe overhead is recorded and kept default-off;
  - if predicted records appear only after actual demand, reject this graph
    callback path and do not implement runtime prefetch here.

GP71 probe smoke result on 2026-07-07:

- Run:
  - remote:
    `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp71-next-gate-probe-smoke-n8-france`;
  - local:
    `.Agent/runs/20260707-gp71-next-gate-probe-smoke-n8-france`.
- Prompt:
  - `Please introduce France in a short paragraph.`
- Env on top of SOTA reproduction env:
  - `GGML_MOE_NEXT_GATE_SHADOW_OUT=$RUN/next-gate-shadow.csv`;
  - `GGML_MOE_NEXT_GATE_SHADOW_TOPK=4`;
  - `GGML_MOE_NEXT_GATE_PREFETCH_PROBE_OUT=$RUN/next-gate-prefetch-probe.csv`.
- Runtime:
  - `N=8`;
  - quality pass;
  - output: `France is a country in Western Europe known`;
  - token rate `1.38 tok/s`;
  - TTFT `76521.97 ms`;
  - decode `5082.61 ms / 7`;
  - host RAM peak `15899996160`;
  - `direct_reads=0`.
- Probe order report:
  - `.Agent/runs/20260707-gp71-next-gate-probe-smoke-n8-france/probe-order-summary.md`.
- Result:
  - probe rows: `893`;
  - predicted rows: `413`;
  - actual rows: `480`;
  - predicted-to-actual pairs: `413`;
  - positive seq lead: `413/413`;
  - positive time lead: `413/413`;
  - seq lead min/median/max: `2 / 3 / 3`;
  - time lead min/median/mean/max:
    `1120 / 8070 / 12090.8 / 163145 us`;
  - copy overhead mean/max: `13.15 / 68 us`.
- Interpretation:
  - during decode, predicted target-layer topK becomes visible before the
    corresponding actual target-layer routing node;
  - the graph callback path has a measurable same-token prefetch window;
  - the first prompt/prefill block records actual routing before decode
    predictions, so runtime prefetch must be decode-only.
- Decision:
  - proceed to plan a default-off `cap4` runtime prefetch hook from the eval
    callback path;
  - the hook must submit only top4 predicted experts, use expert pack /
    io_uring only, keep demand priority, and record useful hits, misses,
    false/unused entries, extra bytes, and demand delay;
  - do not run held-out prompts until a dev N32/N96 prefetch run passes quality,
    TTFT, RAM, direct-read, and net token-rate gates.

GP72 planned cap4 runtime host-prefetch hook:

- Goal:
  - convert GP71's execution-order lead into a real default-off prefetch
    candidate without writing a new IO subsystem.
- Implementation:
  - env:
    - `GGML_MOE_NEXT_GATE_PREFETCH=1`;
    - `GGML_MOE_NEXT_GATE_PREFETCH_CAP=4`;
    - keep `GGML_MOE_NEXT_GATE_SHADOW_OUT=<csv>` and
      `GGML_MOE_NEXT_GATE_SHADOW_TOPK=4` so the predicted topK graph nodes are
      materialized;
    - optional diagnostic:
      `GGML_MOE_NEXT_GATE_PREFETCH_PROBE_OUT=<csv>`.
  - `src/llama-context.cpp`:
    - reuse the graph eval callback path from GP71;
    - when a predicted `ffn_moe_next_shadow_topk-<layer>` node completes, copy
      its expert IDs and submit the first `cap` IDs to CUDA MoE.
  - `ggml/src/ggml-cuda/moe_stream_batch.cu`:
    - expose a small `ggml_cuda_moe_next_gate_prefetch_submit()` API;
    - map target layer to `blk.<layer>.ffn_up_exps.weight` and
      `blk.<layer>.ffn_gate_exps.weight`;
    - resolve exact `expert_bytes` from expert-pack metadata;
    - enqueue into the existing planned host-prefetch queue;
    - reuse the existing `host_prefetch_copy_h2d()` consumer, so demand can
      consume a ready prefetched buffer.
- Scope:
  - decode-only in practice, because prompt/prefill does not have useful
    next-layer lead;
  - up/gate only; down remains handled by current-down overlap;
  - default-off and not a SOTA claim until dev N96 and held-out gates pass.
- Theory:
  - cap4 policy bound suggests about `47%` useful expert coverage with
    `~2.9%` false/actual bytes before accounting for VRAM residency;
  - GP71 measured a median `~8 ms` same-token lead, enough to hide part of
    host read latency if the worker is not starved by demand IO.
- Dev acceptance before held-out:
  - quality pass;
  - Host RAM `<16GB`;
  - `direct_reads=0`;
  - host-prefetch useful hits are nonzero and materially higher than the old
    planned-host-prefetch path;
  - no large evicted-unused explosion;
  - n32/n96 token rate improves or at least shows reduced demand iouring wait
    without TTFT > +20%.

GP72 cap4 runtime host-prefetch smoke result on 2026-07-07:

- Run:
  - remote: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp72-next-gate-prefetch-smoke-n8-france`;
  - local artifact:
    `.Agent/runs/20260707-gp72-next-gate-prefetch-smoke-n8-france`.
- Repro env delta on top of GP4 SOTA:
  - `GGML_MOE_NEXT_GATE_SHADOW_OUT=<run>/next-gate-shadow.csv`;
  - `GGML_MOE_NEXT_GATE_SHADOW_TOPK=4`;
  - `GGML_MOE_NEXT_GATE_PREFETCH=1`;
  - `GGML_MOE_NEXT_GATE_PREFETCH_CAP=4`;
  - `GGML_MOE_HOST_PREFETCH_SLOTS=32`;
  - `GGML_MOE_HOST_PREFETCH_MAX_MIB=512`;
  - `GGML_MOE_NEXT_GATE_PREFETCH_PROBE_OUT=<run>/next-gate-prefetch-probe.csv`.
- Result:
  - `N=8`, France quality passed;
  - token rate `1.27 tok/s`, decode `5531.38 ms / 7`, TTFT `81474.99 ms`;
  - Host RAM peak `15899996160`, `direct_reads=0`;
  - host prefetch counters:
    - `planned_enqueued=5400`, `planned_dequeued=4809`;
    - `hits=8`, `misses=5581`;
    - `evicted=3879`, `read_failures=890`;
    - `used=159.25 MiB`, `slots=32`.
- Decision:
  - reject cap4 as a performance candidate;
  - do not run held-out prompts from this configuration;
  - keep the code default-off only as a diagnostic until a smaller/safer
    policy proves useful hits.
- Diagnosis:
  - GP71 proved the shadow topK has execution-order lead, but the current
    host-prefetch queue is too small and too early for cap4 all-layer
    predictions: many tasks are read and then evicted before demand;
  - the worker also records `read_failures`, so the next implementation must
    avoid losing exact expert-pack identity between prediction submission and
    worker execution, or record the failing key to prove the mismatch.

GP73 planned low-pressure next-gate prefetch diagnostic:

- Goal:
  - determine whether next-gate prefetch can produce useful hits when queue
    pressure is reduced.
- Configuration:
  - same N8 France cold-start smoke;
  - `GGML_MOE_NEXT_GATE_PREFETCH=1`;
  - `GGML_MOE_NEXT_GATE_PREFETCH_CAP=1`;
  - `GGML_MOE_HOST_PREFETCH_SLOTS=64`;
  - `GGML_MOE_HOST_PREFETCH_MAX_MIB=1024`;
  - keep `GGML_MOE_NEXT_GATE_SHADOW_OUT` and
    `GGML_MOE_NEXT_GATE_SHADOW_TOPK=1`;
  - omit `GGML_MOE_NEXT_GATE_PREFETCH_PROBE_OUT` to measure runtime overhead
    without CSV writes.
- Acceptance for further work:
  - quality pass, Host RAM `<16GB`, `direct_reads=0`;
  - host prefetch hits materially exceed cap4's `8` while evictions and
    read-failures fall;
  - token rate is neutral or better than the GP71 smoke reference `1.38 tok/s`.

GP73 low-pressure next-gate prefetch result on 2026-07-07:

- Run:
  - remote:
    `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp73-next-gate-prefetch-cap1-smoke-n8-france`;
  - local artifact:
    `.Agent/runs/20260707-gp73-next-gate-prefetch-cap1-smoke-n8-france`.
- Repro env delta on top of GP4 SOTA:
  - `GGML_MOE_NEXT_GATE_SHADOW_OUT=<run>/next-gate-shadow.csv`;
  - `GGML_MOE_NEXT_GATE_SHADOW_TOPK=1`;
  - `GGML_MOE_NEXT_GATE_PREFETCH=1`;
  - `GGML_MOE_NEXT_GATE_PREFETCH_CAP=1`;
  - `GGML_MOE_HOST_PREFETCH_SLOTS=64`;
  - `GGML_MOE_HOST_PREFETCH_MAX_MIB=1024`;
  - no prefetch-probe CSV output.
- Result:
  - `N=8`, France quality passed;
  - token rate `1.21 tok/s`, decode `5780.30 ms / 7`, TTFT `90019.35 ms`;
  - Host RAM peak `15899996160`, `direct_reads=0`;
  - host prefetch counters:
    - `planned_enqueued=3813`, `planned_dequeued=3813`;
    - `hits=22`, `misses=5567`;
    - `evicted=2727`, `read_failures=1000`;
    - `used=311.50 MiB`, `slots=64`.
- Decision:
  - reject this runtime hook as a performance path;
  - source code was reverted to GP71/default SOTA behavior;
  - do not run held-out prompts for GP72/GP73.
- Root-cause conclusion:
  - lower cap and larger slots did not make useful-hit rate meaningful;
  - the extra callback/shadow work plus host-prefetch queue churn raises TTFT
    and decode time;
  - next-gate prediction should not be implemented by feeding every predicted
    up/gate key into the existing host-prefetch slot cache.
- Next direction:
  - if next-layer prediction is revisited, it needs a demand-integrated design:
    record the exact predicted `expert_pack_entry` identity, prioritize only
    immediately upcoming layers, cancel stale tasks, and write into a small
    per-layer pending table consumed before demand IO submission;
  - otherwise return to the Phase 5 structural byte-reduction path, because
    current measurements still show that byte movement must shrink to about
    `0.30x-0.40x` to make `5 tok/s` plausible.

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

Phase 5A low-risk VRAM cache margin check:

- Current SOTA uses `GGML_MOE_VRAM_CACHE_MIB=15000`.
- Because the deployment target says to use VRAM as fully as possible, run a
  small dev-only cold-start A/B with `VRAM_MIB=15500` before changing any code.
- Hypothesis:
  - a small cache increase may reduce misses without touching Host RAM;
  - if the runtime auto-clamps or CUDA allocation pressure increases, the result
    should be neutral or rejected.
- Experiment:
  - n32 dev `Please introduce France in a short paragraph.`;
  - baseline `VRAM_MIB=15000` vs candidate `VRAM_MIB=15500`;
  - record token rate, TTFT, RAM peak, answer, cache hit rates, iouring bytes,
    and direct reads.
- Acceptance:
  - quality pass;
  - Host RAM remains below 16 GB;
  - TTFT <= baseline * 1.20;
  - decode token rate improves;
  - no `direct_reads` or missing pack regression.

Phase 5A n32 dev result on 2026-07-07:

- Run root:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260707-vram15500-n32-ab`
- Prompt:
  - `Please introduce France in a short paragraph.`
- Baseline `VRAM_MIB=15000`:
  - quality pass;
  - token rate `1.78 tok/s`;
  - decode `17437.84 ms / 31`;
  - TTFT `93581.89 ms`;
  - RAM peak `15899996160`;
  - `direct_reads=0`;
  - iouring bytes `131119579136`;
  - down hit `73.4%`;
  - upgate hit `45.2%`.
- Candidate `VRAM_MIB=15500`:
  - quality pass;
  - token rate `1.82 tok/s`;
  - decode `17053.06 ms / 31`;
  - TTFT `93185.55 ms`;
  - RAM peak `15899996160`;
  - `direct_reads=0`;
  - iouring bytes `130773565440`;
  - down hit `73.6%`;
  - upgate hit `45.3%`.
- Decision:
  - small positive dev signal only;
  - run held-out n96 before changing the default or claiming SOTA.

Phase 5A held-out result on 2026-07-07:

- Report:
  - `.Agent/runs/20260707-vram-cache-margin-heldout/report.md`
- Candidate run:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260707-vram15500-heldout-n96-candidate`
- Paired baseline run:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260707-vram15000-paired-heldout-n96-baseline`
- Historical GP4 accepted baseline:
  - `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- Result:
  - historical GP4 mean: `1.3667 tok/s`;
  - paired `VRAM_MIB=15000` mean: `1.7000 tok/s`;
  - candidate `VRAM_MIB=15500` mean: `1.7083 tok/s`;
  - candidate vs paired mean: `+0.49%`;
  - candidate regressed on `test_coding_01` and `test_reasoning_math_01`
    versus the paired `15000` run;
  - candidate TTFT versus historical GP4 exceeded +20% on most prompts;
  - quality/RAM/direct-read gates passed.
- Decision:
  - reject `VRAM_MIB=15500` as a new default;
  - do not claim it as SOTA;
  - investigate why the paired `VRAM_MIB=15000` baseline now reproduces around
    `1.70 tok/s` before changing SOTA records.
- Audit:
  - historical and paired `15000` runs moved the same GiB and had the same hit
    rates;
  - paired iouring wait was much lower, e.g. `test_chinese_01`
    `66449.2 ms -> 38436.9 ms` with the same `378.0 GiB`;
  - this points to storage/device/runtime state rather than a model-path
    improvement;
  - keep the historical GP4 SOTA as accepted until a stronger cold-start
    protocol proves the faster baseline is reproducible from a cold device
    state.

Phase 5B cold-start SOTA audit gate:

- Tool:
  - `.Agent/run-tools/kimi_cold_start_sota_audit.py`
- Purpose:
  - prevent warmed-IO/device-state runs from being accepted as model/runtime
    improvements.
- Gate rule:
  - if token rate improves while iouring bytes and VRAM hit rates are unchanged,
    and the improvement is explained mainly by lower `iouring_wait`, the run is
    marked as wait-only suspicious;
  - wait-only suspicious runs must not replace the accepted SOTA without a
    stronger cold-device reproduction protocol.
- Reports:
  - `.Agent/runs/20260707-cold-start-sota-audit/report.md`
  - `.Agent/runs/20260707-cold-start-sota-audit/paired-vs-historical.md`
- Results:
  - `VRAM_MIB=15500` versus paired `15000`:
    - mean `1.7000 -> 1.7083 tok/s`;
    - token-rate gate failed because min rate regressed and two prompts
      regressed;
    - acceptance safe `False`.
  - paired `15000` versus historical GP4:
    - mean `1.3667 -> 1.7000 tok/s`;
    - all prompts had identical bytes and hit rates;
    - iouring wait dropped by about `37.8%-45.9%`;
    - TTFT gate failed with max `+56.18%`;
    - wait-only suspicion `True`;
    - acceptance safe `False`.
- Decision:
  - keep historical GP4 as accepted SOTA;
  - require this audit for future held-out SOTA claims;
  - any apparent gain with identical bytes/hits must be treated as measurement
    instability until proven by stronger cold-device controls.

Phase 5C lower-byte candidate gate:

- Tool:
  - `.Agent/run-tools/kimi_lower_byte_candidate_gate.py`
- Report:
  - `.Agent/runs/20260707-lower-byte-candidate-gate/report.md`
  - `.Agent/runs/20260707-lower-byte-candidate-gate/report.json`
- Purpose:
  - gate the existing lower-byte ideas before writing a runtime path;
  - require any primary path toward `5 tok/s` to plausibly reduce total moved
    bytes to about `0.30x-0.40x` while preserving output quality.
- Result:
  - naive blockwise `1-bit` re-encode can reach about `0.367x-0.408x` on
    sampled gate tensors, but rel L2 is about `1.86-2.07`, so it is too
    destructive;
  - naive blockwise `2-bit` re-encode has rel L2 about `0.69-0.79` and still
    moves too many bytes at about `0.673x-0.878x`;
  - D2MoE one-base residual rank128 can reach some byte ratios near the target,
    but residual norm remains about `0.945`, so the base does not explain the
    expert weights;
  - D2MoE clustered bases improve some tensors, but a top32 sample still has
    rank128 error/weight `0.500` for down and `0.306` for gate at 16 clusters,
    while one tensor's bf16 bases already cost about `448 MiB`;
  - full external IQ1_S scale is about `0.504x`, above the required
    `0.30x-0.40x` range and carries quality/runtime risk;
  - selected IQ1_S/v2 hotsets remain coverage-limited or prompt-specific, and
    prior GP57 held-out overlay testing regressed badly;
  - down activation block skipping can skip `19.8%` blocks at threshold `0.1`
    or `34.5%` at threshold `0.2`, but mean relative output error is about
    `0.210` / `0.368`; early layers are inaccurate and down-only skipping cannot
    close the gap because up/gate is about `62%` of miss bytes.
- Decision:
  - do not implement any of these rejected candidates as the next primary
    runtime optimization;
  - lower-byte work must next measure activation-output error on real hidden
    states for compressed up/gate/down candidates, not rely only on weight
    reconstruction error;
  - late-layer down skipping can remain a possible local micro-optimization only
    if it is gated by per-layer error and held-out quality.

Phase 5D activation-output compressed-compute screen:

- Goal:
  - measure compressed expert output error on real MoE activation vectors before
    writing any runtime compressed expert kernel.
- Implementation:
  - add a default-off runtime dump controlled by:
    - `GGML_MOE_ACTIVATION_DUMP_DIR=<dir>`;
    - `GGML_MOE_ACTIVATION_DUMP_MAX_RECORDS=<n>`;
    - `GGML_MOE_ACTIVATION_DUMP_DECODE_ONLY=<0|1>`;
  - dump metadata to `activations.csv` and raw f32 vectors to
    `activations.f32`;
  - record up, gate, and down active experts with tensor name, expert id,
    dimensions, original expert bytes, and vector offset;
  - analyze with `.Agent/run-tools/kimi_activation_output_compression_screen.py`.
- Hypothesis:
  - weight relative error alone is too pessimistic or too indirect;
  - activation-output error on real hidden states is the right gate for deciding
    whether a low-byte direct compute kernel is worth implementing.
- Experiment:
  - dev-only cold-start run, initially `N=16`;
  - include `Please introduce France in a short paragraph.`;
  - limit activation dump records so Host RAM and TTFT are not used for SOTA
    claims;
  - run offline screen on dumped activations for blockwise `1-bit` and `2-bit`
    candidates.
- Acceptance for further work:
  - this phase does not accept a SOTA improvement;
  - a candidate can advance only if it reaches the `0.30x-0.40x` byte-ratio
    target and has low activation-output error on up/gate fused output and down
    output;
  - if all target-byte candidates still have large output error, reject this
    low-bit blockwise path and move to a different structural representation.

GP67 dev result on 2026-07-07:

- Code:
  - runtime dump in `ggml/src/ggml-cuda/moe_stream_batch.cu`;
  - analysis tool:
    `.Agent/run-tools/kimi_activation_output_compression_screen.py`.
- Report:
  - `.Agent/runs/20260707-gp67-activation-output-screen-n16-france/report.md`;
  - `.Agent/runs/20260707-gp67-activation-output-screen-n16-france/screen.json`;
  - activation sample:
    `.Agent/runs/20260707-gp67-activation-output-screen-n16-france/act/activations.csv`;
    `.Agent/runs/20260707-gp67-activation-output-screen-n16-france/act/activations.f32`.
- Runtime:
  - dev prompt: `Please introduce France in a short paragraph.`;
  - `N=16`;
  - cold-start `systemd-run` with `MemoryMax=15900000000` and swap disabled;
  - `GGML_MOE_ACTIVATION_DUMP_MAX_RECORDS=72`;
  - quality passed;
  - token rate `1.59 tok/s`;
  - TTFT `74011.21 ms`;
  - decode `9430.36 ms / 15`;
  - Host RAM peak `15899996160`.
- Screen result:
  - records loaded: `72`;
  - up records: `24`, gate records: `24`, down records: `24`;
  - target gate: byte ratio `<= 0.40x`, mean rel L2 `<= 0.10`;
  - passing matvec candidates: `0`;
  - passing fused candidates: `0`;
  - down 1-bit target-range candidates:
    - byte ratio `0.3091x-0.3636x`;
    - mean rel L2 `1.5517-1.9329`;
  - up/gate 1-bit target-range candidates:
    - byte ratio `0.3695x-0.3912x` for block128/256;
    - mean rel L2 about `2.06-2.28`;
  - fused up/gate 1-bit target-range candidates:
    - byte ratio `0.3695x-0.3912x`;
    - mean rel L2 `7.6308-8.9759`;
  - 2-bit candidates have lower error but byte ratios `0.60x-0.78x`, above
    the required `0.30x-0.40x`, and fused mean rel L2 still exceeds `1.0`.
- Decision:
  - reject naive blockwise 1-bit/2-bit compressed expert compute as the next
    primary runtime path;
  - do not implement a direct low-bit blockwise kernel for this candidate;
  - next structural byte-reduction work must use a qualitatively different
    representation, such as activation-aware trained codebooks, selective
    residual correction, or a small calibrated low-byte model/pack, and must
    pass this activation-output gate before runtime implementation.

GP68 activation-aware scale result on 2026-07-07:

- Code:
  - extended `.Agent/run-tools/kimi_activation_output_compression_screen.py`
    with `--scale-modes maxabs,aw_mse`;
  - `aw_mse` uses dumped real activation weights `x_i^2` to choose each block's
    scale by weighted least-squares instead of max-abs.
- Report:
  - `.Agent/runs/20260707-gp68-activation-aware-scale-screen-n16-france/report.md`;
  - `.Agent/runs/20260707-gp68-activation-aware-scale-screen-n16-france/screen.json`.
- Input:
  - reused the GP67 dev France `N=16` activation dump;
  - no held-out prompts were used.
- Result:
  - automatic gate still failed:
    - passing matvec candidates: `0`;
    - passing fused candidates: `0`;
    - advance blockwise low-bit path: `False`.
  - `aw_mse` improved target-range 1-bit error substantially:
    - down 1-bit mean rel L2 improved from `1.55-1.93` to about
      `0.495-0.499`;
    - up/gate 1-bit mean rel L2 improved from about `2.06-2.28` to about
      `0.593-0.596`;
    - fused up/gate 1-bit mean rel L2 improved from `7.63-8.98` to about
      `0.767-0.769`.
  - The improvement is not enough for quality-preserving runtime work:
    - target byte-ratio candidates are still far above the mean rel L2 gate
      `<=0.10`;
    - 2-bit `aw_mse` is also too inaccurate and remains above the byte target,
      with fused mean rel L2 about `0.574-0.583` at `0.717x-0.783x`.
- Decision:
  - reject activation-aware scalar block scaling as the next runtime path;
  - keep the result as evidence that activation-aware methods help, but require
    a richer representation than one scale per block;
  - next byte-reduction screen should test residual correction or trained
    codebooks against the same activation-output gate.

GP69 planned activation-aware codebook upper-bound screen:

- Goal:
  - test whether a richer per-block low-byte representation can plausibly pass
    the activation-output gate before any runtime implementation.
- Candidate:
  - activation-weighted 1-bit / 2-bit codebooks per block;
  - each block stores packed code indices plus fp16 codebook centers;
  - centers are fit by weighted 1D k-means using `x_i^2` from the dumped real
    activation vectors.
- Scope:
  - dev-only offline upper-bound screen using the GP67 activation dump;
  - no held-out prompts;
  - no runtime path;
  - not a SOTA claim.
- Why this is only an upper bound:
  - the screen fits codebook centers using the same activation sample used for
    evaluation;
  - if it fails, this codebook family is not worth runtime work;
  - if it passes, a later phase must train/freeze codebooks from dev prompts and
    validate on held-out prompts before any SOTA claim.
- Acceptance to advance:
  - byte ratio in the `0.30x-0.40x` target range for up/gate and down;
  - fused up/gate and down mean rel L2 `<=0.10`;
  - 2-bit results may be kept only as diagnostic evidence if their byte ratio
    is above the target.

GP69 target-range result on 2026-07-07:

- Code:
  - extended `.Agent/run-tools/kimi_activation_output_compression_screen.py`
    with `--scale-modes aw_codebook`;
  - codebook mode uses activation-weighted 1D k-means per block;
  - runtime remains unchanged.
- Report:
  - `.Agent/runs/20260707-gp69-activation-codebook-screen-n16-france-target/report.md`;
  - `.Agent/runs/20260707-gp69-activation-codebook-screen-n16-france-target/screen.json`.
- Input:
  - reused the GP67 dev France `N=16` activation dump;
  - 72 real decode activation records;
  - target-range subset only: `bits=1`, `block=128,256`;
  - the earlier full matrix run was stopped because the CPU offline k-means
    path was too slow; the target-range subset is the relevant decision gate.
- Result:
  - automatic gate failed:
    - passing matvec candidates: `0`;
    - passing fused candidates: `0`;
    - advance blockwise low-bit path: `False`.
  - down 1-bit codebook:
    - byte ratio `0.327x-0.364x`;
    - mean rel L2 `0.568-0.571`.
  - up/gate 1-bit codebook:
    - byte ratio `0.391x-0.435x`;
    - mean rel L2 about `0.589-0.594`.
  - fused up/gate:
    - byte ratio `0.391x-0.435x`;
    - mean rel L2 about `0.764-0.768`.
- Interpretation:
  - 1-bit codebook does not beat the GP68 activation-aware scalar scale in the
    target byte range;
  - even this optimistic per-sample upper-bound screen is far above the
    `<=0.10` mean rel L2 gate.
- Decision:
  - reject per-block 1-bit activation-weighted codebook as the next runtime
    path;
  - do not spend time on direct codebook kernels for this representation;
  - next structural byte-reduction work should move to residual correction or
    mixed precision with a small preserved high-error subset, and must keep the
    same activation-output gate.

GP70 planned activation-channel residual upper-bound screen:

- Goal:
  - test whether a small mixed-precision residual can rescue 1-bit base
    accuracy while keeping total bytes near the `0.30x-0.40x` target.
- Candidate:
  - start from `aw_mse` 1-bit blockwise base;
  - for each real activation vector, preserve the original expert weights for
    the largest `|activation|` input channels;
  - compute output as low-bit base plus exact correction on those preserved
    channels.
- Why this is an optimistic upper bound:
  - preserved channels are selected with the same activation being evaluated;
  - byte ratio assumes the preserved channels replace their low-bit base bytes
    instead of being stored as an extra residual overlay;
  - real runtime would need a prompt-general policy, packed column layout, and
    direct mixed compute.
- Experiment:
  - dev-only, reuse GP67 activation dump;
  - test keep fractions around the target budget, initially `0.02,0.05,0.10`;
  - use `bits=1`, `block=256`, `scale_mode=aw_mse`, because GP68 showed this
    is the best target-byte scalar base.
- Acceptance to advance:
  - up/gate fused and down mean rel L2 `<=0.10`;
  - byte ratio `<=0.40x` for target candidates;
  - if only `>0.40x` candidates pass, this is not a direct `5 tok/s` path and
    must be rejected or redesigned.

GP70 result on 2026-07-07:

- Code:
  - extended `.Agent/run-tools/kimi_activation_output_compression_screen.py`
    with `--keep-input-fracs`;
  - this adds optimistic input-channel correction candidates to the same
    matvec/fused activation-output gate.
- Report:
  - `.Agent/runs/20260707-gp70-activation-input-residual-screen-n16-france/report.md`;
  - `.Agent/runs/20260707-gp70-activation-input-residual-screen-n16-france/screen.json`.
- Input:
  - reused the GP67 dev France `N=16` activation dump;
  - 72 real decode activation records;
  - command target:
    - `--bits 1`;
    - `--blocks 256`;
    - `--scale-modes aw_mse`;
    - `--keep-input-fracs 0.02,0.05,0.10`.
- Result:
  - automatic gate failed:
    - passing matvec candidates: `0`;
    - passing fused candidates: `0`;
    - advance blockwise low-bit path: `False`.
  - Down improves but remains too inaccurate:
    - base `0.309x`, rel L2 `0.499`;
    - keep 10% `0.378x`, rel L2 `0.234`.
  - Up/gate remains far above the error gate:
    - keep 2% stays within target bytes at `0.382x`, but rel L2 is about
      `0.545-0.546`;
    - keep 5% is already about `0.401x`, with rel L2 about `0.499-0.500`;
    - keep 10% is `0.433x`, with rel L2 about `0.438-0.439`.
  - Fused up/gate remains the blocking point:
    - base `0.369x`, rel L2 `0.769`;
    - keep 2% `0.382x`, rel L2 `0.719`;
    - keep 5% `0.401x`, rel L2 `0.669`;
    - keep 10% `0.433x`, rel L2 `0.594`.
- Decision:
  - reject small input-channel residual correction as the next runtime path;
  - do not build column-gather mixed compute for this candidate;
  - the result suggests target-byte `1-bit + small residual` cannot preserve
    up/gate semantics even under oracle channel selection;
  - next work should shift away from lossy 1-bit representations and toward:
    - better use of existing exact expert bytes, such as cross-layer / next-layer
      demand scheduling;
    - or a representation that keeps substantially more than 10% of up/gate
      information while reducing bytes elsewhere enough to stay under the global
      budget.

GP74 planned mixed-role byte/error budget screen:

- Goal:
  - determine whether any already-screened compressed representation can be
    combined asymmetrically across down and fused up/gate to meet the global
    `0.30x-0.40x` moved-byte target.
- Method:
  - no model run and no held-out prompt tuning;
  - use the accepted held-out SOTA role split from Phase 5:
    - down `37.6%`;
    - up `31.2%`;
    - gate `31.2%`;
  - use GP70 activation-output error rows:
    - down matvec candidates for down;
    - fused up/gate candidates for up+gate;
  - compute global byte ratio:
    `0.376 * down_ratio + 0.624 * fused_upgate_ratio`;
  - report only combinations under `<=0.40x`, sorted by worst error and byte
    ratio.
- Acceptance:
  - a candidate can advance only if global byte ratio is `<=0.40x`, down mean
    rel L2 `<=0.10`, and fused up/gate mean rel L2 `<=0.10`;
  - if no combination passes, the next structural work must use a new
    representation, not more tuning of GP68-GP70 blockwise residuals.

GP74 mixed-role byte/error budget result on 2026-07-07:

- Code:
  - `.Agent/run-tools/kimi_mixed_role_byte_error_budget.py`.
- Report:
  - `.Agent/runs/20260707-gp74-mixed-role-byte-error-budget/report.md`;
  - `.Agent/runs/20260707-gp74-mixed-role-byte-error-budget/report.json`.
- Input:
  - GP70 activation-output screen only;
  - held-out role split constants from the accepted SOTA bound, not used for
    hotset or predictor tuning:
    - down `0.376`;
    - fused up/gate `0.624`.
- Result:
  - combinations under global `<=0.40x` byte target: `15`;
  - passing combinations: `0`;
  - best under-budget pair:
    - down `aw_mse:bits1:block256`;
    - fused up/gate `aw_mse_keep_input0p1:bits1:block256`;
    - global ratio `0.3861x`;
    - down mean rel L2 `0.498586`;
    - fused up/gate mean rel L2 `0.594271`;
    - worst mean rel L2 `0.594271`.
- Decision:
  - stop tuning GP68-GP70 blockwise 1-bit residuals as a primary path to
    `5 tok/s`;
  - next representation must be qualitatively different, for example a
    prompt-general trained residual/codebook, a calibrated lower-bit pack with
    activation-output validation, or exact-byte demand-integrated scheduling
    that avoids the GP72/GP73 host-prefetch cache failure.

GP75 planned exact-byte scheduler ceiling:

- Goal:
  - quantify whether an exact-byte demand-integrated scheduler can be a primary
    `5 tok/s` path before writing runtime code.
- Scope:
  - offline analysis only;
  - no model run, no routing change, no held-out tuning;
  - use the accepted GP4 held-out SOTA profile only as a ceiling measurement.
- Method:
  - parse `metrics.json` for token rate, decode runs, decode wall, and
    `iouring_bytes`;
  - parse `ttft-trace.csv` `runtime_load`, `call_upgate`, and `call_down`
    events to estimate exposed MoE call wall and moved bytes per decode token;
  - compute optimistic ceilings at `10.4 GiB/s`:
    - transfer-only ceiling with current moved bytes;
    - transfer plus all-hit MoE floor ceiling using the accepted Phase 5 floor
      of about `40 ms/token`;
    - measured exposed MoE-call replacement ceiling, replacing call wall with
      `max(all-hit floor, moved_bytes / 10.4 GiB/s)`.
- Acceptance:
  - if the ceiling is below `5 tok/s`, do not implement a full exact-byte
    scheduler as the next primary path;
  - only revisit scheduler work if it is combined with byte reduction or if a
    future trace proves a much higher fraction of decode wall is non-byte
    scheduler overhead.

GP75 exact-byte scheduler ceiling result on 2026-07-07:

- Code:
  - `.Agent/run-tools/kimi_exact_byte_scheduler_ceiling.py`.
- Report:
  - `.Agent/runs/20260707-gp75-exact-byte-scheduler-ceiling/report.md`;
  - `.Agent/runs/20260707-gp75-exact-byte-scheduler-ceiling/report.json`.
- Input:
  - accepted GP4 held-out profile root only:
    `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`;
  - no model run, no routing change, no held-out tuning.
- Result:
  - measured held-out mean token rate: `1.367 tok/s`;
  - transfer-only mean ceiling at `10.4 GiB/s`: `2.396 tok/s`;
  - floor+transfer mean ceiling with `40.1 ms/token` all-hit floor:
    `2.184 tok/s`;
  - best prompt floor+transfer ceiling: `2.395 tok/s`;
  - worst prompt floor+transfer ceiling: `1.812 tok/s`;
  - mean byte ratio needed for `5 tok/s` after floor: `0.383x`.
- Decision:
  - exact-byte scheduling without byte reduction is not a primary `5 tok/s`
    path;
  - do not implement a full demand-integrated exact-byte scheduler as the next
    main runtime change;
  - only revisit scheduling once a byte-reduced representation exists, or as a
    secondary improvement to keep the reduced bytes moving near peak bandwidth.

GP76 planned multi-prompt activation sample for prompt-general representation:

- Goal:
  - replace the France-only activation-output screen with a dev-only
    multi-prompt sample before testing any trained residual/codebook or
    calibrated lower-bit expert pack.
- Scope:
  - dev prompts only;
  - held-out test prompts remain sealed and unused for candidate design;
  - no SOTA claim;
  - no runtime behavior change.
- Prompt sample:
  - `dev_japan_factual`;
  - `dev_python_reverse`;
  - `dev_mixed_summary`;
  - keep `dev_france_regression` only as a regression reference, not as the
    only design signal.
- Runtime:
  - cold-start `systemd-run` for each prompt;
  - `MemoryMax=15900000000`, `MemorySwapMax=0`;
  - `N=16`;
  - `GGML_MOE_ACTIVATION_DUMP_DIR=<run>/act`;
  - `GGML_MOE_ACTIVATION_DUMP_MAX_RECORDS=72`;
  - `GGML_MOE_ACTIVATION_DUMP_DECODE_ONLY=1`;
  - all usual GP4 SOTA env, including alias and direct-read checks.
- Output:
  - one run directory per prompt with full answer, metrics, memory stats, and
    activation dump;
  - a merged index/report that records record counts by prompt/role/tensor;
  - no compression candidate may advance unless it is evaluated on this
    multi-prompt dev sample and later frozen before held-out testing.
- Acceptance:
  - each prompt quality passes;
  - Host RAM peak remains `<16GB`;
  - `direct_reads=0`;
  - activation dump contains up/gate/down records from decode.

GP76b multi-prompt activation sample result on 2026-07-07:

- Note:
  - the first GP76 attempt is rejected because `<run>/act` was not created
    before launch, and the dump path does not auto-create directories;
  - `dev_mixed_summary` at `N=16` also produced too short an answer for the
    `wind` quality keyword.
- Corrected run:
  - remote:
    `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample`;
  - local:
    `.Agent/runs/20260707-gp76b-dev-activation-sample`.
- Summary:
  - `.Agent/runs/20260707-gp76b-dev-activation-sample/summary.md`;
  - `.Agent/runs/20260707-gp76b-dev-activation-sample/summary.json`.
- Runtime:
  - `dev_japan_factual`: `N=16`, quality pass, token rate `1.54`, TTFT
    `86056.69 ms`, decode `9727.81 ms / 15`;
  - `dev_python_reverse`: `N=16`, quality pass, token rate `1.53`, TTFT
    `94270.20 ms`, decode `9824.16 ms / 15`;
  - `dev_mixed_summary`: `N=32`, quality pass, token rate `1.59`, TTFT
    `111750.48 ms`, decode `19454.73 ms / 31`.
- Constraints:
  - Host RAM peak stayed at `15899996160` for all three runs;
  - `direct_reads=0` for all three runs;
  - each prompt has `72` decode activation records:
    - `24` up;
    - `24` gate;
    - `24` down.
- Decision:
  - accept GP76b as a dev-only prompt-general activation sample for future
    representation screening;
  - do not use it as a token-rate SOTA claim because activation dump adds
    diagnostic overhead;
  - next compression/representation screen must evaluate on this multi-prompt
    sample before any runtime kernel or pack conversion is built.

GP77 planned multi-prompt representation screen:

- Goal:
  - re-run the activation-output compression gate on GP76b's prompt-general dev
    sample instead of the old France-only sample.
- Scope:
  - dev-only offline screen;
  - no model run, no runtime behavior change, no SOTA claim;
  - held-out test prompts remain sealed and unused.
- Candidate family:
  - reuse the GP68-GP70 best available diagnostic family only as a baseline:
    `aw_mse`, `bits=1`, `block=256`, `keep_input_fracs=0.02,0.05,0.10`;
  - this candidate is already rejected on France-only GP70, so the expected
    outcome is confirmation that it should not be implemented;
  - if it unexpectedly improves on multi-prompt data, it still must pass the
    same byte/error gate before any runtime work.
- Method:
  - run `.Agent/run-tools/kimi_activation_output_compression_screen.py` for
    each GP76b prompt activation dump:
    - `dev_japan_factual`;
    - `dev_python_reverse`;
    - `dev_mixed_summary`;
  - aggregate per-prompt `aggregate` and `fused_up_gate` rows into a weighted
    multi-prompt report;
  - apply the same advancement gate:
    - global moved byte target `0.30x-0.40x`;
    - component mean rel L2 `<=0.10`;
    - fused up/gate must pass, because up/gate is the blocking role.
- Acceptance:
  - if passing candidates are `0`, stop this candidate family permanently and
    move to a new representation family;
  - any future trained residual/codebook must be evaluated against this same
    GP76b multi-prompt sample before runtime implementation.

GP77 multi-prompt representation screen result on 2026-07-08:

- Local report:
  - `.Agent/runs/20260707-gp77-multiprompt-representation-screen/summary.md`
  - `.Agent/runs/20260707-gp77-multiprompt-representation-screen/summary.json`
- Remote report:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen`
- Inputs:
  - GP76b dev-only activation dumps for:
    - `dev_japan_factual`;
    - `dev_python_reverse`;
    - `dev_mixed_summary`.
- Candidate family:
  - `aw_mse`, `bits=1`, `block=256`;
  - `keep_input_fracs=0.02,0.05,0.10`;
  - fused up/gate evaluated explicitly because up/gate misses are the blocking
    role in the real profile.
- Result:
  - aggregate matvec candidates: `12`;
  - aggregate fused up/gate candidates: `4`;
  - passing matvec candidates: `0`;
  - passing fused candidates: `0`;
  - passing mixed-role candidates: `0`.
- Best mixed-role under `<=0.40x` global moved-byte budget:
  - down candidate: `down:aw_mse:bits1:block256`;
  - fused up/gate candidate:
    `fused_up_gate:aw_mse_keep_input0p1:bits1:block256`;
  - global byte ratio: `0.3861x`;
  - down mean rel L2: `0.499277`;
  - fused up/gate mean rel L2: `0.598174`;
  - worst mean rel L2: `0.598174`;
  - decision: reject.
- Best fused up/gate by rel L2:
  - candidate: `fused_up_gate:aw_mse_keep_input0p1:bits1:block256`;
  - byte ratio: `0.4326x`;
  - mean rel L2: `0.598174`;
  - max rel L2: `0.659500`;
  - decision: reject because it misses both the `<=0.40x` byte budget and the
    `<=0.10` error gate.
- Decision:
  - stop this blockwise 1-bit residual family permanently as a primary runtime
    path;
  - the failure is not a scheduling issue: the compressed output error is too
    high on prompt-general activation samples;
  - move to a qualitatively different representation family before writing
    runtime kernels.
- Reproduce:

```bash
cd /root/lfz/tmp/kimi-stage2m-align
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260707-gp77-multiprompt-representation-screen
rm -rf "$ROOT"
mkdir -p "$ROOT"
for id in dev_japan_factual dev_python_reverse dev_mixed_summary; do
  mkdir -p "$ROOT/$id"
  python3 .Agent/run-tools/kimi_activation_output_compression_screen.py \
    --activation-csv /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/$id/act/activations.csv \
    --activation-bin /root/lfz/runs/vendor-kimi-token-rate/20260707-gp76b-dev-activation-sample/$id/act/activations.f32 \
    --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
    --libggml-base build-cuda-batch/bin/libggml-base.so \
    --out-json "$ROOT/$id/screen.json" \
    --out-md "$ROOT/$id/report.md" \
    --bits 1 \
    --blocks 256 \
    --scale-modes aw_mse \
    --keep-input-fracs 0.02,0.05,0.10 \
    --max-records 72 \
    --torch-threads 8
done
python3 .Agent/run-tools/kimi_multi_prompt_screen_summary.py \
  --screen-json "$ROOT/dev_japan_factual/screen.json" \
  --screen-json "$ROOT/dev_python_reverse/screen.json" \
  --screen-json "$ROOT/dev_mixed_summary/screen.json" \
  --out-json "$ROOT/summary.json" \
  --out-md "$ROOT/summary.md"
```

GP78 planned next representation family:

- Goal:
  - find a prompt-general expert representation that can plausibly meet the
    `0.30x-0.40x` moved-byte target while keeping activation-output error near
    the `<=0.10` gate.
- Direction:
  - train or derive a shared codebook/residual basis on the GP76b dev activation
    sample instead of using independent per-block 1-bit signs;
  - evaluate fused up/gate first, because GP77 shows it is the quality
    bottleneck;
  - no runtime implementation until the offline activation-output gate passes.
- Candidate families to screen before runtime work:
  - per-role/product-quantized residual codebook for output contribution
    blocks;
  - clustered base expert plus small residual for experts within the same
    layer/role cluster;
  - mixed precision expert pack where only the most activation-sensitive rows
    keep higher precision.
- Required evidence:
  - dev-only screen on GP76b;
  - held-out prompts remain sealed;
  - report byte ratio, fused up/gate mean/max rel L2, down mean/max rel L2, and
    estimated 5 tok/s ceiling;
  - only if the offline gate passes, write a runtime pack/kernel plan.

GP78a planned shared-codebook screen:

- Goal:
  - test whether a per-block activation-weighted codebook can reduce fused
    up/gate error enough to justify a runtime representation path.
- Why this is different from GP77:
  - GP77 used sign-times-scale (`aw_mse`) plus optional exact input-channel
    keep correction;
  - GP78a uses `aw_codebook`, which learns two activation-weighted centers per
    block and can represent asymmetric weight distributions that 1-bit scale
    signs cannot;
  - this is still an offline screen only and does not claim SOTA.
- Candidate family:
  - `scale_modes=aw_codebook`;
  - `bits=1`;
  - `blocks=128,256,512`;
  - no input-channel keep correction in the first pass, so the byte ratio is
    directly attributable to the codebook representation.
- Acceptance:
  - same GP77 gate:
    - global moved byte ratio around `0.30x-0.40x`;
    - fused up/gate mean rel L2 `<=0.10`;
    - down mean rel L2 `<=0.10`;
  - if no candidate is close, do not write runtime kernels for this family;
  - if a candidate is close but slightly above byte budget, run a second pass
    with larger blocks or shared centers before any runtime work.

GP78a shared-codebook smoke result on 2026-07-08:

- Local report:
  - `.Agent/runs/20260708-gp78a-aw-codebook-block256-smoke/summary.md`
  - `.Agent/runs/20260708-gp78a-aw-codebook-block256-smoke/dev_japan_factual/report.md`
- Remote report:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260708-gp78a-aw-codebook-block256-smoke`
- Execution note:
  - the initial full `blocks=128,256,512` run was stopped because
    `aw_codebook` center iteration was too slow for rapid screening;
  - a smaller `block=256` smoke was run on `dev_japan_factual`;
  - held-out prompts remained unused.
- Result for `aw_codebook:bits1:block256`:
  - down byte ratio `0.3273x`, mean rel L2 `0.571013`;
  - gate byte ratio `0.3912x`, mean rel L2 `0.592142`;
  - up byte ratio `0.3912x`, mean rel L2 `0.595450`;
  - fused up/gate byte ratio `0.3912x`, mean rel L2 `0.773949`,
    max rel L2 `0.814748`.
- Decision:
  - reject early;
  - the byte ratio is within the target, but fused up/gate error is worse than
    GP77's best fused candidate (`0.773949` vs `0.598174`);
  - do not implement runtime kernels for this codebook family;
  - next representation work must exploit activation/output subspace structure
    rather than only changing per-block scalar/codebook reconstruction.

GP79 planned layer/full-hit VRAM cache simulation:

- Goal:
  - switch from byte-only and global-hit thinking to an IO-wave objective;
  - identify whether prompt-general static cache allocation can reduce exposed
    demand waves before writing runtime cache policy code.
- Motivation:
  - GP77 and GP78a show `0.30x-0.40x` low-bit expert representations are too
    inaccurate for fused up/gate;
  - the next practical path is higher VRAM hit rate, but only hits that remove
    a full tensor-call miss wave should be expected to improve token rate;
  - at `MOE_IO_DEPTH=8`, a tensor-call with 8 active experts still needs one
    demand wave unless all 8 experts are resident.
- Data:
  - use dev route traces from
    `.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct`;
  - do not use held-out prompt route traces for strategy selection.
- Method:
  - reconstruct consecutive tensor calls from `route-trace.csv`;
  - group up and gate into the up/gate budget and down into the down budget;
  - compare strategies with the same approximate GP4 VRAM cache budget:
    - global traffic hotset;
    - front-layer hotsets for several layer cutoffs;
    - greedy full-hit utility that adds the missing experts needed to complete
      high-frequency tensor-call signatures.
- Metrics:
  - validation bytes saved;
  - validation full-hit tensor calls;
  - validation IO waves saved, using `ceil(misses / 8)`;
  - leave-one-dev-out mean and worst validation behavior.
- Acceptance to proceed to runtime:
  - a strategy must beat global traffic hotset in validation IO waves, not just
    bytes;
  - it must avoid severe down regression;
  - only then write an env-gated runtime policy or static profile experiment.

GP79 layer/full-hit simulation result on 2026-07-08:

- Local report:
  - `.Agent/runs/20260708-gp79-layer-fullhit-cache-sim/summary.md`
  - `.Agent/runs/20260708-gp79-layer-fullhit-cache-sim/summary.json`
- Tool:
  - `.Agent/run-tools/kimi_layer_fullhit_cache_sim.py`
- Dev-only route root:
  - `.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct`
- Result:
  - `front_layers_0_2` is best by IO-wave metric:
    - mean wave saved `2.54%`;
    - worst wave saved `0.97%`;
    - mean bytes saved only `3.29%`;
    - up/gate full-hit `2.44%`;
    - down full-hit `2.77%`.
  - `global_hotset` saves more bytes (`19.53%`) but saves essentially no
    full tensor-call waves at depth 8.
- Interpretation:
  - the simulator validates the earlier bottleneck model: partial cache hits can
    save bytes while still leaving the same demand IO wave exposed;
  - the best prompt-general front-layer candidate is weak, so runtime testing
    should be limited to dev smoke before held-out validation.

GP79 front-layer admission smoke result on 2026-07-08:

- Local report:
  - `.Agent/runs/20260708-gp79-frontlayer-admit-smoke-summary.md`
  - `.Agent/runs/20260708-gp79-frontlayer-admit-smoke-summary.json`
- Admission profile:
  - `.Agent/runs/20260708-gp79-layer-admit-profile/front-layers-0-2.tsv`
  - entries: `1974`
  - generated by `.Agent/run-tools/kimi_make_layer_admit_profile.py`
- Runtime A/B:
  - baseline N32 France:
    - quality pass;
    - token rate `1.72`;
    - TTFT `77778.06 ms`;
    - decode `18051.43 ms / 31`;
    - RAM peak `15899996160`;
    - upgate hit `45.2%`, down hit `73.4%`;
    - iouring wait `11929606 us`.
  - candidate with
    `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/gp79-front-layers-0-2-admit.tsv`:
    - quality pass;
    - token rate `1.69`;
    - TTFT `82742.77 ms`;
    - decode `18354.19 ms / 31`;
    - RAM peak `15899996160`;
    - upgate hit `45.2%`, down hit `73.4%`;
    - iouring wait `12001517 us`.
- Decision:
  - reject;
  - token rate regressed and TTFT increased;
  - identical VRAM hit counters show this existing admission env did not affect
    the active Kimi split-cache path;
  - do not run held-out validation for this candidate;
  - a future cache-policy attempt needs a Kimi-specific split upgate/down cache
    admission or protection hook, and must justify the small simulated upper
    bound before implementation.

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

Continue from Phase 5E:

1. Stop treating scheduling-only work as the primary path; GP75 caps it around
   `2.18 tok/s` mean on held-out.
2. GP77 confirms blockwise 1-bit residual candidates fail the prompt-general
   offline gate.
3. Run GP78 to screen qualitatively different representation families:
   - prompt-general trained residual/codebook with dev/test split;
   - clustered base expert plus residual;
   - mixed precision expert pack guided by activation sensitivity.
4. The next screen must target global moved bytes around `0.30x-0.40x` and
   fused up/gate mean rel L2 close to the quality gate before any runtime
   kernel is written.
5. Do not build prompt-specific hot expert overlays. GP57 showed dev overlay
   gains can regress held-out performance severely.

Rationale:

- The current accepted SOTA already moves only about half of the active expert
  footprint, yet it remains at `~1.37 tok/s`.
- At `10.4 GiB/s`, the `5 tok/s` budget after an optimistic MoE floor allows
  only about `1.6-1.7 GiB/token` of expert movement on typical prompts.
- This requires structural byte reduction, not another small IO scheduling
  adjustment.
