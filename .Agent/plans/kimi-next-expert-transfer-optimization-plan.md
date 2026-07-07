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

Continue from Phase 5C:

1. Build a dev-only activation-output error screen for compressed expert
   compute candidates. It must sample real active up/gate/down experts and real
   hidden states, then report byte ratio, output error, and estimated runtime
   overhead.
2. Prioritize candidates that reduce up/gate and down movement together. A
   candidate above `0.30x-0.40x` total moved bytes is not a direct `5 tok/s`
   path unless it also removes exposed IO through direct compressed compute.
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
