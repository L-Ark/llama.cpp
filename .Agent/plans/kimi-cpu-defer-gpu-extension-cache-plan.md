# Kimi CPU/defer GPU-extension cache optimization plan

Date: 2026-07-11
Branch: `vendor/kimi-deepseek-41d205-additive`
Parent plan: `.Agent/plans/kimi-token-rate-16gb-optimization-plan.md`

## Current goal and plan checkpoint: 2026-07-11 CST

This section is the current working contract. All later experiments, commits,
and SOTA claims for this branch must be checked against it before promotion.

Goal:

> Build a prompt-general Kimi optimization path for one 32 GB RTX 5090-class GPU
> and strict 16 GB host RAM, starting from the current stable code, with the
> next measurable milestone at reproducible held-out `>2 tok/s` and the product
> target at stable random-prompt `>5 tok/s`.

Scope for the next phase:

- Treat the DeepSeek result as an architecture pattern, not a direct patch:
  keep the CPU/defer MoE scheduler, but make the GPU expert-cache/streaming
  extension handle more useful `gate/up/down` work with less exposed wait.
- Kimi's current stable path already has decode CPU fallback and direct reads at
  zero in the measured runs, so the next gain must come from reducing
  transfer/staging wait, improving useful VRAM/RAM residency, or lowering bytes
  per active expert while preserving quality.
- The highest-priority bottleneck remains mixed up/gate critical-path time. On
  current profiles, up/gate dominates down and residual compute, so candidates
  must name which up/gate rows they reduce and by how many ms/token.

Hard constraints:

- Cold start only.
- Host RAM peak must stay below `15900000000` bytes, including page cache,
  pinned memory, mmap/file-backed pages, helper processes, and cgroup
  accounting.
- Use VRAM aggressively, but TTFT must not exceed baseline by more than `20%`.
- Output quality must pass on `Please introduce France in a short paragraph.`
  and on held-out prompts; the France answer must be coherent and semantically
  correct.
- Optimization must be prompt-general. Held-out prompts cannot be used to choose
  expert hotsets, thresholds, layer slabs, pack order, or replacement policy.
- Every accepted improvement must be committed and pushed immediately with a
  reproducible commit body: baseline/candidate SHAs, exact commands/env,
  prompts, run directories, token rate, TTFT, RAM/VRAM/page-cache metrics,
  IO/H2D/staging/compute/fallback metrics, quality result, and rollback point.

Immediate execution plan:

1. Reconfirm the current baseline on a dev prompt.
   - Run cold-start N32 and one N96/profile when needed.
   - Record token rate, TTFT, decode ms/token, RAM split, page cache, VRAM cache
     hit rates, `io_uring_wait`, staging wall, H2D wall, compute wall, direct
     reads, and CPU fallback.

2. Separate DeepSeek-transferable work from Kimi-specific bottlenecks.
   - Confirm whether any gate/up/down role still misses the GPU extension.
   - If fallback remains zero, do not spend time on "CPU fallback removal" as a
     claimed optimization.
   - Focus on exposed up/gate transfer/staging wait and queue continuity.

3. Test only bounded candidates.
   - Before implementation, calculate the best-case ms/token saving from expert
     bytes, expected hit/miss movement, SSD bandwidth, H2D bandwidth, and
     compute cost.
   - Reject candidates whose theoretical bound cannot plausibly move
     prompt-general throughput toward `>2 tok/s`.

4. Candidate order.
   - First: paired up/gate residency or scheduling improvements that reduce the
     current dominant mixed up/gate rows.
   - Second: RAM/VRAM storage-layout changes that replace low-value page cache
     with batch-friendly expert slabs without fragmenting SSD batches.
   - Third: lower-byte expert representation through a default-off v2 override
     bridge, starting with tiny quality smoke and only scaling after correctness
     is proven.
   - Later: prediction/prefetch only if it can create sustained queue depth
     without prompt-specific tuning.

5. Promotion rule.
   - Promote and push only if dev and held-out cold-start runs pass token-rate,
     TTFT, RAM, quality, and reproducibility gates.
   - If a candidate is slower, exceeds RAM, raises TTFT too much, or fails
     quality, revert it or leave it default-off and document the rejection. It
     must not be called SOTA.

## Progress update: v2 override preflight bridge

Timestamp: 2026-07-11 CST.

Commit scope prepared:

- Added a default-off, manifest-gated v2 override preflight path in
  `ggml/src/ggml-cuda/moe_stream_batch.cu`.
- New envs:
  - `GGML_MOE_EXPERT_PACK_V2_OVERRIDE_PREFLIGHT=1` enables preflight;
  - `GGML_MOE_EXPERT_PACK_V2_OVERRIDE=1` also enables the same preflight for
    compatibility with the planned override gate;
  - `GGML_MOE_EXPERT_PACK_V2_OVERRIDE_MANIFEST=<manifest.tsv>` is required for
    safe allowlisting;
  - `GGML_MOE_EXPERT_PACK_V2_OVERRIDE_PROFILE_OUT=<csv>` writes per-call
    coverage and rejection reasons.
- Dispatch remains disabled. This patch does not change the default Kimi SOTA
  runtime path and does not replace any expert payload. It only proves which
  active `gate/up/down` routes are safe candidates for later lower-byte
  consumption.

Safety checks implemented:

- Exact manifest allowlist by `(tensor, expert_idx)`.
- v2 entry must match manifest `packed_type`, `packed_nbytes`, `packed_ne00`,
  `packed_ne01`, and `packed_nb01`.
- Logical matrix shape must match v2 `ne00/ne01`.
- Packed type must be supported by the current MMVQ path.
- Packed payload must be smaller than the current logical payload.
- Rejections are split into: no manifest, not allowlisted, no v2 entry,
  unsupported type, manifest mismatch, shape mismatch, and not smaller.

Build:

- Command: `cmake --build build-cuda-batch --target ggml-cuda -j2`.
- Result: pass. Only existing warning classes were emitted.

Cold-start diagnostic run with preflight enabled:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-override-preflight-n32-france`.
- Prompt: `Please introduce France in a short paragraph.`
- Command shape:
  `systemd-run --wait --collect --same-dir -p MemoryMax=15900000000
  -p MemorySwapMax=0 env RUN=<run> N=32 PROFILE=1
  PROMPT_ID=dev_france_v2_override_preflight
  PROMPT_USER_TEXT='Please introduce France in a short paragraph.'
  QUALITY_KEYWORDS='france,paris|europe|western'
  EXTRA_RUNTIME_ENV='<v2 pack + preflight envs>'
  .Agent/run-tools/kimi-general-prompt-repro.sh`.
- v2 pack:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack`.
- Manifest:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv`.
- Quality: pass.
- Output:
  `France is a country in Western Europe known for its rich history, culture,
  and influence on art, fashion, and cuisine. Its capital, Paris, is famous`.
- Token rate: `1.53 tok/s`.
- TTFT: `9628.09 ms`.
- Decode: `20289.77 ms / 31`, or `654.51 ms/token`.
- Host RAM peak: `12775698432` bytes.
- CPU fallback rows: `0`.
- Direct reads: `0`.
- Expert-pack IO: `39090` io_uring reads, `224038649856` bytes,
  `19175835 us` wait.
- Preflight coverage:
  - calls: `5760`;
  - active entries: `68736`;
  - accepted entries: `168`;
  - full-cover calls: `0`;
  - reject not allowlisted: `68568`;
  - all other reject reasons: `0`.
- Interpretation: the tiny 8-entry overlay is shape/type/manifest valid where
  it applies, but it cannot drive runtime replacement because no active call is
  fully covered. The lower token rate is diagnostic CSV overhead and is not a
  SOTA candidate.

Cold-start default-off regression check:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-preflight-defaultoff-n32-france`.
- Same prompt, `N=32`, `PROFILE=1`, no v2/preflight env.
- Quality: pass.
- Token rate: `1.60 tok/s`.
- TTFT: `8727.09 ms`.
- Decode: `19385.92 ms / 31`, or `625.35 ms/token`.
- Host RAM peak: `12767203328` bytes.
- CPU fallback rows: `0`.
- Direct reads: `0`.
- No v2 preflight files or stderr report were produced.
- Interpretation: default-off behavior is preserved within expected run-to-run
  variance. This commit is infrastructure for the next lower-byte A/B, not a
  SOTA performance claim.

Next decision:

- Do not expand this tiny pack into a SOTA claim.
- Next lower-byte implementation must build a prompt-general, dev-derived pack
  with enough full-active coverage for at least one role/layer group, then add
  guarded dispatch only when preflight shows nonzero full-cover calls.
- If full-active coverage cannot be achieved without prompt-specific selection,
  pause lower-byte work and return to up/gate scheduling/layout or RAM/VRAM
  storage-policy work.

## Progress update: dev7 budget-120 v2 metadata-only coverage

Timestamp: 2026-07-11 CST.

Tooling fix:

- Updated `.Agent/run-tools/kimi_iq1s_selected_payload_pack.py` so
  `--metadata-only` still writes a valid `GGMLMOEPACKv2` header/index file.
- Metadata-only mode now skips payload range downloads but produces a loadable
  v2 sidecar for runtime preflight. This is required to test large lower-byte
  candidate coverage without downloading or storing a 96-120 GiB payload.
- The tool now records actual index-file `pack_bytes` instead of `0`.

Candidate:

- Selected plan:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-hotset-bound-dev7/budget-120p0-selected-plan.tsv`.
- Selection source: 7 dev prompt profiles, not held-out prompts.
- Theoretical bound from the existing budget report:
  - selected entries: `44642`;
  - selected tensors: `180`;
  - estimated payload bytes: `128839811072`;
  - hybrid byte ratio across dev prompts: `0.5223`;
  - dev prompt selected-byte coverage: about `96.2-98.5%`.
- Metadata-only candidate output:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-metadataonly-dev7-budget120`.
- Pack index:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-metadataonly-dev7-budget120/selected-iq1s-overlay-v2.expert-pack`.
- Manifest:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-metadataonly-dev7-budget120/selected-iq1s-overlay-manifest.tsv`.
- Build result:
  - selected entries: `44642`;
  - metadata-only pack index size: `8216576` bytes;
  - estimated payload bytes if materialized: `128839811072`;
  - builder stats: no duplicate, missing, regex-skipped, type-skipped, or
    invalid rows.

Dev quality/preflight run:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-dev7-budget120-preflight-n32-france`.
- Prompt: `Please introduce France in a short paragraph.`
- Quality: pass.
- Output:
  `France is a country in Western Europe known for its rich history, culture,
  and influence on art, fashion, and cuisine. Its capital, Paris, is famous`.
- Token rate with diagnostic preflight: `1.68 tok/s`.
- TTFT: `9066.32 ms`.
- Decode: `18434.39 ms / 31`, or `594.66 ms/token`.
- Host RAM peak: `12796219392` bytes.
- CPU fallback rows: `0`.
- Direct reads: `0`.
- Preflight totals:
  - calls: `5760`;
  - active entries: `68736`;
  - accepted entries: `67657` (`98.43%`);
  - full-cover calls: `4910` (`85.24%`);
  - reject not allowlisted: `1079`;
  - no manifest, no entry, unsupported, manifest mismatch, shape mismatch, and
    not-smaller rejects: all `0`;
  - accepted saved bytes: `188448669696`;
  - logical total bytes: `390196101120`.
- By phase:
  - decode down full-cover: `1672/1861` (`89.84%`);
  - decode gate full-cover: `1620/1861` (`87.05%`);
  - decode up full-cover: `1546/1861` (`83.07%`);
  - prompt full-cover is lower, especially up (`12/59`), but accepted-entry
    coverage remains high.

Held-out preflight run:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-dev7-budget120-preflight-n32-heldout-sky`.
- Prompt: `Explain why the sky appears blue in a short paragraph.`
- This prompt was not used to select the dev7 budget-120 hotset.
- Quality: pass.
- Output:
  `The sky appears blue because sunlight interacts with molecules and tiny
  particles in Earth's atmosphere. Sunlight is made up of different colors of
  light, each with different wavelengths.`
- Token rate with diagnostic preflight: `1.68 tok/s`.
- TTFT: `10008.28 ms`.
- Decode: `18410.85 ms / 31`, or `593.90 ms/token`.
- Host RAM peak: `12794122240` bytes.
- CPU fallback rows: `0`.
- Direct reads: `0`.
- Preflight totals:
  - calls: `5760`;
  - active entries: `72984`;
  - accepted entries: `63219` (`86.62%`);
  - full-cover calls: `1455` (`25.26%`);
  - reject not allowlisted: `9765`;
  - no manifest, no entry, unsupported, manifest mismatch, shape mismatch, and
    not-smaller rejects: all `0`;
  - accepted saved bytes: `177555718144`;
  - logical total bytes: `414302601216`.
- By decode phase:
  - decode down full-cover: `572/1861` (`30.74%`);
  - decode gate full-cover: `504/1861` (`27.08%`);
  - decode up full-cover: `378/1861` (`20.31%`).

Decision:

- The lower-byte metadata path is now proven at runtime preflight scale:
  manifest validation, v2 lookup, shape/type matching, and smaller-byte checks
  all pass for a large dev-derived candidate.
- A full-call-only dispatch can probably help dev prompts, but it is not yet a
  strong prompt-general SOTA mechanism. Held-out full-cover is only `25.26%`
  even though accepted-entry coverage is `86.62%`.
- The next lower-byte design should not require every active expert in a call to
  be v2-covered. The better direction is a partial-covered split dispatch:
  run covered experts through v2 lower-byte slots and uncovered experts through
  the current path, then combine compact rows into the same output buffer.
- Before implementing split dispatch, update the theoretical bound using
  held-out accepted-entry coverage rather than dev full-cover coverage. The
  candidate has enough accepted entries to be interesting, but the quality and
  scheduling risks are higher than the tiny preflight path.
- Do not materialize the 120 GiB payload or call this SOTA until split dispatch
  or a better prompt-general full-cover selection passes cold-start quality,
  RAM, TTFT, and held-out gates.

## Progress update: partial-covered v2 split dispatch bound

Timestamp: 2026-07-11 CST.

New analysis tool:

- Added `.Agent/run-tools/kimi_v2_partial_split_bound.py`.
- Inputs:
  - run directory with `metrics.txt`;
  - `v2-override-preflight.csv`.
- Outputs:
  - full-call-only byte/time bound;
  - partial-covered split byte/time bound;
  - per phase/role accepted coverage, full-cover rate, saved bytes, miss-weighted
    saved ratio, and token-rate ceilings.
- This is an offline bound only. It does not change runtime behavior or claim
  SOTA.

Report:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-partial-split-bound-dev-heldout/report.md`.
- JSON:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-partial-split-bound-dev-heldout/report.json`.

Command:

```bash
.Agent/run-tools/kimi_v2_partial_split_bound.py \
  --run /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-dev7-budget120-preflight-n32-france \
  --run /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-dev7-budget120-preflight-n32-heldout-sky \
  --out-json /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-partial-split-bound-dev-heldout/report.json \
  --out-md /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-partial-split-bound-dev-heldout/report.md
```

Key result:

| run | accepted entries | full-cover calls | partial saved | full saved | partial miss ratio | full miss ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| France dev | `98.43%` | `85.24%` | `175.507 GiB` | `132.767 GiB` | `48.04%` | `32.56%` |
| Sky held-out | `86.62%` | `25.26%` | `165.362 GiB` | `33.389 GiB` | `43.42%` | `5.89%` |

Token-rate ceiling from measured pools:

| run | mode | measured pool | saved ms/token | ceiling tok/s |
| --- | --- | ---: | ---: | ---: |
| France dev | full `io_uring_wait` only | `18430.43 ms` | `193.60` | `2.49` |
| France dev | partial `io_uring_wait` only | `18430.43 ms` | `285.62` | `3.24` |
| France dev | partial `io_uring_wait + H2D` | `29876.14 ms` | `462.99` | `7.59` |
| Sky held-out | full `io_uring_wait` only | `18647.35 ms` | `35.44` | `1.79` |
| Sky held-out | full `io_uring_wait + H2D` | `30216.36 ms` | `57.43` | `1.86` |
| Sky held-out | partial `io_uring_wait` only | `18647.35 ms` | `261.21` | `3.01` |
| Sky held-out | partial `io_uring_wait + H2D` | `30216.36 ms` | `423.27` | `5.86` |

Interpretation:

- Full-call-only v2 dispatch is not sufficient for prompt-general progress.
  Held-out full-call coverage is too low, and even the `io_uring_wait + H2D`
  ceiling is only `1.86 tok/s`.
- Partial-covered split dispatch is the first lower-byte path with enough
  held-out theoretical room:
  - to exceed `2 tok/s`, held-out partial split needs to save about `93.9
    ms/token`; the conservative `io_uring_wait`-only bound offers `261.2
    ms/token`, leaving about `167 ms/token` for split overhead and imperfect
    realization;
  - to reach `5 tok/s`, held-out partial split must approach the
    `io_uring_wait + H2D` bound and keep new overhead below about `29 ms/token`.
- Therefore the next implementation should not start with a large full-call-only
  v2 dispatch. It should either:
  - implement a default-off partial split dispatch smoke on the already
    materialized tiny v2 payload pack, to validate row mapping and output
    combination; or
  - improve pack selection for held-out full-cover before any payload
    materialization.
- Because the actual 120 GiB payload is not materialized and quality of IQ1_S
  replacement is still unproven, any split-dispatch implementation must remain
  default-off and begin with N32 quality smoke before performance claims.

## Progress update: tiny v2 partial-split row-placement smoke

Timestamp: 2026-07-11 CST.

New smoke tool:

- Added `.Agent/run-tools/kimi_moepack_v2_partial_split_smoke.cpp`.
- It dynamically loads:
  - `ggml_cuda_moe_expert_pack_v2_lookup_debug`;
  - `ggml_cuda_moe_expert_pack_v2_read_debug`;
  - `ggml_cuda_moe_stream_mmvq_dev`.
- It uses the already materialized 8-entry v2 payload pack, selects one tensor
  group with multiple same-shape covered experts, inserts uncovered fallback
  rows, runs MMVQ for covered rows, fills fallback rows with deterministic
  sentinel values, and verifies all rows scatter into the requested destination
  rows.
- This is a control-plane smoke only. It does not load a model, does not replace
  runtime outputs, does not measure token rate, and does not validate IQ1_S
  model quality.

Command:

```bash
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-partial-split-smoke
mkdir -p "$RUN"

g++ -std=c++17 -O2 \
  -I/usr/local/cuda/include \
  .Agent/run-tools/kimi_moepack_v2_partial_split_smoke.cpp \
  -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart -ldl \
  -o "$RUN/kimi_moepack_v2_partial_split_smoke"

LD_LIBRARY_PATH=build-cuda-batch/bin:/usr/local/cuda/targets/x86_64-linux/lib \
  "$RUN/kimi_moepack_v2_partial_split_smoke" \
  build-cuda-batch/bin/libggml-cuda.so \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv \
  > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
```

Result:

- Run directory:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-v2-partial-split-smoke`.
- Status: pass.
- Tensor selected by the smoke:
  `blk.1.ffn_up_exps.weight`.
- Routes: `5`.
- Covered v2 routes: `3` experts: `23`, `116`, `197`.
- Fallback routes: `2` synthetic uncovered experts: `0`, `1`.
- Destination rows were intentionally out of order:
  - route 0 -> dst 3;
  - route 1 fallback -> dst 0;
  - route 2 -> dst 4;
  - route 3 fallback -> dst 1;
  - route 4 -> dst 2.
- Output row sums:
  - v2 expert 23: `4.242831e+02`;
  - fallback row 0: `2.048528e+06`;
  - v2 expert 116: `4.617555e+02`;
  - fallback row 1: `2.049552e+06`;
  - v2 expert 197: `4.319570e+02`.
- Runtime loader confirmed:
  - v2 pack loaded `8` metadata entries;
  - CUDA device: `NVIDIA GeForce RTX 5090`, compute capability `12.0`,
    VRAM `32109 MiB`.

Decision:

- The minimal partial split row-placement mechanism is now validated outside the
  model runtime: covered v2 rows and fallback rows can be combined into one
  compact destination with nonzero finite outputs.
- This clears only the row-mapping/control-plane blocker. The hard blockers
  before any SOTA claim remain:
  - default-off runtime implementation;
  - device-side split staging without excessive extra kernels/synchronization;
  - real model quality with lower-byte replacement;
  - cold-start N32/N96 under 16 GB host RAM and TTFT gates;
  - held-out prompt validation.
- Next implementation, if pursued, should be a default-off runtime probe limited
  to tiny payload entries and a small max-call count, with automatic fallback to
  the current path on any mismatch.

## Active goal: verify DeepSeek-style CPU/defer GPU-extension on Kimi

Timestamp: 2026-07-11 CST.

This is the top-priority goal for the next optimization round. Historical
sections below remain as audit history; new work must follow this section unless
this section is explicitly superseded by a later dated goal.

Goal:

> Determine, with cold-start prompt-general measurements, whether the DeepSeek
> SOTA pattern transfers to Kimi: keep the CPU/defer MoE scheduler, but make the
> GPU expert-cache extension handle the critical `gate/up/down` work more
> completely. The immediate target is a pushed, reproducible improvement to
> stable held-out `>2 tok/s` under strict 16 GB host RAM. The product target
> remains stable `>5 tok/s` for random user prompts on one 32 GB RTX
> 5090-class GPU.

Important interpretation:

- DeepSeek's large jump came from a CPU/defer path where important gate work was
  effectively made GPU-resident and GPU-computed.
- Kimi currently reports decode CPU fallback at zero on the stable path, so this
  is not a literal "remove CPU fallback" task.
- For Kimi, the transferable idea is architectural: CPU/defer stays the
  scheduler, while GPU cache/streaming becomes a better extension for
  `gate/up/down` residency, transfer, and compute.
- The main measured Kimi gap is exposed transfer/staging wait, especially mixed
  up/gate miss wait. Any proposed implementation must show it reduces that
  wait, not only that it increases hit rate or moves bytes elsewhere.

Hard gates for this goal:

- Cold start only; no warm-cache SOTA claims.
- Host RAM peak must stay below `15900000000` bytes, including page cache,
  pinned memory, mmap/file-backed pages, helper processes, and cgroup accounting.
- VRAM should be used aggressively, but not by increasing TTFT beyond `1.20x`,
  causing reclaim/refault spikes, or weakening output quality.
- The prompt `Please introduce France in a short paragraph.` must remain
  coherent and semantically correct.
- Optimization must be prompt-general. Dev prompts can be used for design;
  held-out prompts must not be used to choose hot experts, thresholds, pack
  order, or layer/role slabs.
- Every accepted improvement must be committed and pushed immediately with a
  reproducible commit body: before/after metrics, exact env and command, prompt
  split, run directories, quality result, TTFT/RAM/VRAM/IO/H2D/fallback metrics,
  and rollback commit.

Execution plan:

1. Re-establish the baseline before changing behavior.
   - Use the current pushed branch and rollback SHA.
   - Run at least one cold-start N32 smoke and one N96 profile on dev prompts.
   - Record token rate, TTFT, RAM peak, page-cache split, VRAM/cache settings,
     expert-pack bytes, `io_uring_wait`, staging wall, H2D wall, compute wall,
     CPU fallback count, direct reads, and exact output.

2. Audit current CPU/defer GPU-extension coverage.
   - Confirm the effective `n_cpu_moe`/CPU-defer layer coverage.
   - For every role/layer, separate these paths: VRAM cache hit, RAM tier hit,
     SSD/io_uring pack miss, H2D copy, GPU compute, and true CPU fallback.
   - Verify whether gate, up, and down are already fully handled by the GPU
     extension or whether any role still uses a slower CPU/defer subpath.

3. Compute a savings bound before each implementation.
   - Start from the measured per-token profile, especially mixed up/gate wait.
   - Estimate the maximum possible token-rate gain from the candidate using
     bytes per expert, expected hit/miss reduction, SSD bandwidth, H2D bandwidth,
     staging cost, and compute cost.
   - Continue only if the bound can plausibly save enough critical-path time to
     move prompt-general throughput toward `>2 tok/s`.

4. Run default-off A/B candidates in priority order.
   - Gate residency/cache audit first: test whether more complete gate residency
     helps Kimi, but reject gate-only work if the profile shows mixed up/gate
     wait remains dominant.
   - Up+gate paired cache/streaming second: because the largest current Kimi row
     is mixed `up=22/gate=18`, prioritize paired admission and transfer.
   - Same-layer `gate/up/down` cosubmit or scheduler changes third: use shadow
     counters first, then runtime behavior only if it increases queue continuity
     without increasing exposed wait.
   - RAM tier experiments only after low-value page cache is identified and
     replaced with batch-friendly expert slabs. Avoid prompt-specific hotsets.

5. Promotion and rollback.
   - Promote only if paired baseline/candidate runs pass token-rate, TTFT, RAM,
     quality, and reproducibility gates on dev and held-out prompts.
   - If a candidate is slower, raises TTFT too much, exceeds RAM, or fails
     quality, revert the behavior or leave it default-off and document the
     rejection. It must not be called SOTA.
   - Update this plan before the next implementation attempt with the run root,
     decision, and rollback point.

Current-head audit progress:

- Timestamp: 2026-07-11 CST.
- Branch/head: `vendor/kimi-deepseek-41d205-additive`,
  `7122d959cc579ce36b840a4bfc23cdd1b4fe54f5`.
- New cold-start baseline run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-active-goal-current-head-n32-japan`.
- Prompt: `Please introduce Japan in a short paragraph.`
- Quality: pass.
- Token rate: `1.68 tok/s`.
- TTFT: `8405.00 ms`.
- Decode: `18439.31 ms / 31`, or `594.816 ms/token`.
- Host RAM peak: `12767760384` bytes.
- Final RAM shape: `file=12064870400`, `active_file=12024356864`,
  `anon=450560`, `kernel=29577216`.
- CPU fallback rows: `0`.
- Direct reads: `0`.
- Expert-pack IO: `39075` iouring reads, `223114035200` bytes,
  `18734269 us` reported wait, inflight average `4.30`, max `8`.
- VRAM cache:
  - up/gate: `1735` slots, hit rate `44.4%`;
  - down: `723` slots, hit rate `59.5%`.
- Current-down overlap: `4228` completed jobs, no failed batches.

Current-head bottleneck report:

- Report root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-cpu-defer-gpu-extension-audit-current`.
- N32 current-head Japan attribution:
  - up/gate wall: `392.096 ms/token`;
  - down wall: `156.367 ms/token`;
  - residual after CPU MoE: `23.429 ms/token`;
  - decode fallback: `0 ms/token`.
- N96 France reference attribution:
  - token rate: `1.50 tok/s`;
  - decode: `665.138 ms/token`;
  - up/gate wall: `455.555 ms/token`;
  - down wall: `173.197 ms/token`;
  - residual after CPU MoE: `23.132 ms/token`;
  - decode fallback: `0 ms/token`.
- Largest N32 current-head up/gate rows:
  - `up=22/gate=18`: `208.673 ms/token`, hit about `47.0%`,
    wait about `4.894 ms/call`;
  - `up=22/gate=22`: `93.878 ms/token`, hit about `42.5%`,
    wait about `4.997 ms/call`;
  - `up=18/gate=18`: `71.447 ms/token`;
  - `up=18/gate=22`: `18.098 ms/token`.
- Largest N96 France up/gate row:
  - `up=22/gate=18/parallel_stage=1`: `260.608 ms/token`,
    hit about `47.6%`, wait about `5.259 ms/call`.

Coverage audit:

- Coverage run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-gpu-extension-coverage-n32-france`.
- Coverage summary:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-cpu-defer-gpu-extension-audit-current/gpu-extension-coverage-summary.md`.
- Result:
  - quality pass, `1.66 tok/s`, TTFT `9437.82 ms`, RAM peak
    `12766507008` bytes;
  - fallback rows: `0`;
  - v2 full homogeneous active-call coverage: `0` calls;
  - role-level cache hit including prompt rows:
    down `41.52%`, gate `29.58%`, up `29.59%`;
  - decode rows show up/gate hit is still materially below full-hit
    (`~41-47%` depending on type), while down varies widely by type.

Decision from this audit:

- The DeepSeek idea is useful for Kimi only as an architecture pattern:
  CPU/defer remains the scheduler and GPU remains the expert-cache/compute
  extension. Kimi does not currently have a DeepSeek-like missing gate CPU path
  to fix; decode CPU fallback and direct reads are already zero in the current
  baseline.
- Do not prioritize a gate-only hotpool. Up and gate hit/miss rates move
  together in the fused path, and the largest exposed row is paired
  `up=22/gate=18`; making only gate faster would leave up misses on the same
  critical path.
- Do not retry standalone `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`; historical Phase
  4A showed it does not execute actual jobs on the current fused up/gate Kimi
  path.
- Down-only work is not the first lever for `>2 tok/s`. On N96, removing all
  measured down wall would only reach the boundary, and full removal is not
  realistic because some down work is useful compute/H2D rather than avoidable
  wait.

Next implementation boundary:

1. First candidate must name the exact up/gate rows it expects to reduce,
   especially `up=22/gate=18` and `up=22/gate=22`, and must compute a
   best-case saving from the current-head report before runtime changes.
2. Acceptable mechanisms:
   - paired up/gate cache admission or residency that improves both roles;
   - cache-compatible fused-path scheduling that increases ready jobs without
     stealing from demand reads or duplicating current-down overlap;
   - RAM/VRAM tiering only after proving which decode-time file-backed pages
     are low-value and replacing them with batch-friendly expert slabs;
   - lower-byte expert representation only if the runtime can consume the
     smaller format correctly and quality remains stable.
3. Rejected or low-priority mechanisms for the next step:
   - gate-only hotpool;
   - standalone gate/up/down cosubmit hook;
   - layout changes that reduce read count but keep bytes and exposed wait flat;
   - large pinned RAM slabs without a bound showing reduced critical-path wait.
4. The immediate next action is an offline bound/report for paired up/gate
   residency plus explicit RAM/VRAM tier candidates, using dev prompts only.
   Only if that report can plausibly save at least `~95 ms/token` on N32 and
   `~165 ms/token` on N96 should the next runtime A/B be implemented.

Paired up/gate and RAM/VRAM tier bound result:

- Timestamp: 2026-07-11 CST.
- Report root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-paired-upgate-ram-bound-current`.
- Consolidated report:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-paired-upgate-ram-bound-current/report.md`.
- Inputs:
  - current dev4 profile root:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-current-dev4-n32-profile`;
  - dev7 IO trace root:
    `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343`;
  - current-head N32 Japan baseline:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-active-goal-current-head-n32-japan`.
- Baseline for this bound:
  - token rate: `1.68 tok/s`;
  - decode: `18439.31 ms / 31`, or `594.816 ms/token`;
  - reaching `2 tok/s` requires about `95 ms/token` saving;
  - CPU fallback rows: `0`;
  - direct reads: `0`.

Findings:

1. Whole-layer up/gate residency is too expensive for the next A/B.
   - Largest dev-general up/gate layer buckets are only about `6-10 ms/token`
     each.
   - Examples:
     - layer 14 upgate: `10.038 ms/token`, observed `2126.25 MiB`;
     - layer 28 upgate: `9.209 ms/token`, observed `2539.69 MiB`;
     - layer 1 upgate: `9.090 ms/token`, observed `2242.19 MiB`;
     - layer 29 upgate: `8.479 ms/token`, observed `2618.44 MiB`.
   - Reaching the `~95 ms/token` N32 gap would need many such layers and would
     exceed practical RAM/VRAM residency budgets.

2. Fixed prompt-general hotset retuning is not promising.
   - Cache oracle report:
     `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-paired-upgate-ram-bound-current/cache-oracle-bound.md`.
   - Current online cache beats dev-wide static global LFU at the same slot
     counts.
   - Upgate current hit rates on dev prompts are `37.0-45.4%`; dev-wide global
     LFU only gives `25.3-35.1%`.
   - Therefore a simple fixed prompt-general hotset/profile should not be the
     next runtime A/B.

3. Ordinary explicit RAM tier does not clear the `>2 tok/s` bound.
   - RAM candidates exclude current VRAM entries and use dev IO traces only.
   - Results:
     - `4096 MiB`: selected `61.263 GiB` weighted traffic, but only
       `0.422 GiB` RAM-dominant; scheduler-realistic best `1.68 tok/s`;
     - `8192 MiB`: selected `106.031 GiB` weighted traffic, but only
       `2.514 GiB` RAM-dominant; scheduler-realistic best `1.70 tok/s`;
     - `10240 MiB`: selected `126.794 GiB` weighted traffic, but only
       `3.931 GiB` RAM-dominant; scheduler-realistic best `1.71 tok/s`.
   - Decision: reject a large conventional RAM tier until a scheduler/layout
     change can make RAM hits batch-dominant or avoid mixed RAM/SSD
     fragmentation.

4. Lower-byte expert representation is the only bound here that looks large
   enough.
   - Quant-scaled cache bound:
     `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-paired-upgate-ram-bound-current/quant-scaled-cache-bound.md`.
   - Mean prompt-general global LFU miss traffic:
     - ratio `1.0`: `7.858 GiB/token`;
     - ratio `0.5`: `3.076 GiB/token`;
     - ratio `0.35`: `1.792 GiB/token`;
     - ratio `0.25`: `1.015 GiB/token`;
     - ratio `0.125`: `0.216 GiB/token`.
   - Ratio around `0.35` gets close to the transfer budget for much higher
     token rate if overlap and compute remain controlled; ratio `0.25` gives a
     stronger bound but needs quality/runtime proof.

Decision:

- Do not implement the next runtime A/B as gate-only, whole-layer up/gate
  residency, fixed hotset retuning, or a large ordinary RAM tier.
- Promote lower-byte expert representation to the next main design path, but
  only after writing a concrete implementation plan with:
  - exact role/type coverage;
  - dequant or direct-compute kernel path;
  - quality gate before performance gate;
  - cold-start N32 smoke before N96;
  - fallback to the current IQ3 path for unsupported cases.
- Secondary fallback path:
  - design a scheduler/layout change whose explicit goal is to increase
    RAM-dominant or up/gate-dominant batches; it must show a new offline bound
    that clears `2 tok/s` before runtime A/B.

Lower-byte implementation plan after the bound:

- Timestamp: 2026-07-11 CST.
- Current-head smoke run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-lowerbyte-current-head-smoke`.
- Tiny selected v2 pack:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack`.
- Tiny selected v2 manifest:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv`.
- Result: current-head MMVQ smoke still passes for all `8` selected entries:
  - `Q2_K` down entries for `blk.1` experts `23` and `116`;
  - `IQ1_S` gate entries for `blk.1` experts `23`, `116`, and `197`;
  - `IQ1_S` up entries for `blk.1` experts `23`, `116`, and `197`.

Known cleared blockers:

- `GGMLMOEPACKv2` metadata parse/read works on the current CUDA batch library.
- Real selected `IQ1_S`/`Q2_K` payload bytes can be copied to GPU.
- The exported CUDA MMVQ path can consume those payloads and return finite,
  nonzero outputs on the RTX 5090.

Known remaining blockers:

- The model runtime still does not use v2 packed entries in place of the current
  IQ3/IQ2/Q4 expert tensors.
- Normal pack lookup is keyed by the current tensor `nbytes`, while v2 lower-byte
  entries intentionally have different `packed_nbytes`.
- The fused up/gate and down paths still receive the current GGUF tensor type as
  their effective compute type unless a guarded override is added.
- Tiny 8-entry coverage is not enough to affect real prompts; it is only a
  quality/dispatch smoke target.

Concrete next lower-byte steps:

1. Add a default-off v2 override dispatch gate.
   - Proposed env:
     `GGML_MOE_EXPERT_PACK_V2_OVERRIDE=1`.
   - Additional safety env for the first smoke:
     `GGML_MOE_EXPERT_PACK_V2_OVERRIDE_MANIFEST=<manifest.tsv>`.
   - Only allow exact `(tensor, expert)` keys listed in the manifest.
   - Reject any entry whose `ne00`, `ne01`, `nb01`, or packed type mismatches
     the manifest.
   - Fall back to the current IQ3 path for every unsupported or uncovered call.

2. Scope the first runtime bridge to tiny quality smoke, not performance.
   - Start with the existing 8-entry pack.
   - Dispatch v2 entries only when the runtime can keep the original output
     shape and use the v2 `packed_type` as the effective MMVQ input type.
   - Record counters:
     - v2 lookups;
     - v2 accepted entries;
     - v2 rejected entries by reason;
     - v2 H2D bytes;
     - normal-pack fallback count;
     - kernel type used;
     - output quality.
   - First validation prompt:
     `Please introduce France in a short paragraph.`
   - Promotion criterion for this step is only "quality remains coherent and
     no crash"; no token-rate SOTA claim is allowed.

3. Build a small full-active coverage quality pack only after tiny smoke passes.
   - Use dev traces only to select one or a few complete active calls where
     every active up/gate/down expert has a lower-byte source.
   - Prefer covering full active call groups over isolated experts; partial
     coverage is useful for dispatch testing but weak for quality/performance.
   - Run N32 quality prompts and compare exact output quality against the
     current baseline.
   - Reject immediately if the answer to the France prompt is semantically
     wrong or incoherent.

4. Only then consider a performance-sized lower-byte pack.
   - Based on the current bound, a meaningful pack likely needs effective byte
     ratio near `0.35` or better, and historical IQ1_S selected overlays need
     about `96-120 GiB` before they approach the `>2 tok/s` byte target.
   - Before building such a pack, verify disk headroom and rollback assets.
   - Run paired cold-start N32 first, then N96 only if N32 quality, TTFT, RAM,
     and counters pass.

5. Do not promote until all standard gates pass.
   - Cold start.
   - Host RAM peak below `15900000000` bytes.
   - TTFT within `1.20x`.
   - Prompt-general validation, with held-out prompts untouched until final.
   - Full commit body with env, command, run dirs, quality output, metrics, and
     rollback commit.

## Current execution goal: prompt-general Kimi `>2 tok/s` through storage layout

Timestamp: 2026-07-11 CST.

Goal:

> On the current `vendor/kimi-deepseek-41d205-additive` branch, improve
> prompt-general Kimi decode throughput by making expert storage and transfer
> more efficient, without changing model semantics. The immediate measurable
> target is stable cold-start `>2 tok/s` on held-out/general prompts under
> strict host RAM `<16 GB` and one 32 GB RTX 5090-class GPU. The product target
> remains stable `>5 tok/s` for random user prompts.

Non-negotiable constraints for this phase:

- Host RAM must remain below `15900000000` bytes, including page cache and all
  process memory.
- Every accepted run must be cold start under the memory cgroup; warm-cache
  results are not SOTA.
- The prompt `Please introduce France in a short paragraph.` must remain
  semantically correct and coherent, but optimization must be prompt-general,
  not France-specific.
- Dev prompts may be used for design and tuning; held-out prompts may only be
  used for final validation.
- TTFT must not exceed the paired baseline by more than `20%`.
- Any accepted performance improvement must be committed and pushed
  immediately with a reproducible commit body.
- If token rate drops, quality regresses, TTFT violates the gate, RAM exceeds
  the limit, or profiling shows time only moved elsewhere, reject or revert the
  behavior change. Default-off instrumentation can remain if it does not affect
  default behavior.

Reproducibility requirement for every future SOTA commit:

- commit SHA and rollback SHA;
- exact branch, build target, model/assets, expert pack paths, and alias TSV;
- exact command and environment variables;
- prompt split and which prompts were dev versus held-out;
- token rate, decode ms/token, TTFT, RAM peak, VRAM/expert-cache settings,
  page-cache split, `io_uring_wait`, read bytes/jobs, staging wall, H2D wall,
  compute wall, and CPU fallback count;
- exact quality output or a saved output path;
- run directory containing logs, profiles, and generated analysis.

Latest profile finding that drives this plan:

- Current N32 dev-general traces still spend most exposed time in expert
  movement, especially `io_uring_wait`.
- Pure IO can reach roughly `10.0-10.4 GiB/s`, but decode runtime does not keep
  a deep continuous queue because routing exposes expert IDs layer by layer.
- A fresh two-prompt layout screen showed current pack offsets almost never
  allow adjacent read coalescing:
  - current read jobs: `56943`;
  - current extents after physical adjacency: `56135`;
  - ideal per layer/role/tensor layout extents: `19242`;
  - static `first_use` layout extents: `36216`;
  - static `greedy_pair` layout extents: `31556`.
- Estimated upper bound from static duplicate clustered pack layout:
  - `first_use`: about `2.27-2.30 tok/s` on the profiled N32 prompts;
  - `greedy_pair`: about `2.43-2.45 tok/s`;
  - ideal dynamic layout bound: about `2.9 tok/s`.
- Therefore the next useful optimization should target physical pack layout and
  controlled RAM/VRAM residency, not another fixed prompt-specific hotset.

Execution plan:

1. Preserve the current reproducible baseline.
   - Re-run one cold-start baseline before each behavior A/B.
   - Record full metrics and quality output.
   - Treat the current pushed HEAD as a rollback point unless a later accepted
     SOTA commit supersedes it.

2. Make pack layout measurable and reproducible.
   - Commit the IO trace layout screen tool used to compute current, static,
     and ideal coalescing bounds.
   - Save the exact trace inputs and analysis reports for every layout claim.
   - Do not call a layout result SOTA unless it is validated by runtime A/B,
     not just an offline bound.

3. Build a default-off duplicate clustered overlay pack A/B.
   - Prefer a duplicate overlay rather than rewriting the main pack so rollback
     is immediate.
   - Start with static `greedy_pair` order because it has the best offline
     bound among simple prompt-general layouts.
   - Keep separate measurements for `up`, `gate`, `down`, and `up+gate` so we
     can see whether coalescing saves critical-path wait or only reduces read
     count.
   - Required pass condition: lower `io_uring_wait`, read jobs, and decode
     ms/token without increasing TTFT beyond `1.20x`, RAM beyond the cap, or
     quality failures.

4. Rework RAM usage only after low-value file cache is identified.
   - Profile decode-stage `active_file`/`inactive_file` by file and timestamp:
     GGUF dense pages, GGUF expert refault pages, alias/index pages, pack pages,
     and runtime buffers.
   - Release or avoid low-value decode file cache only when it is proven not to
     be used by decode.
   - Replace freed RAM with explicit expert storage only if it is batch-friendly
     and reduces exposed wait.

5. Design a controlled RAM tier.
   - VRAM stores the hottest experts that avoid H2D and SSD on the critical
     path.
   - RAM stores second-tier experts or complete low-hit critical layer/role
     slabs only when the slab can be served in large batches.
   - SSD stores cold experts in a layout optimized for coalesced reads.
   - Avoid large pinned slabs unless profiling proves they beat pageable or
     file-backed alternatives; large pinning can hurt scheduler/reclaim behavior
     even if it removes SSD reads.

6. Validate prompt generalization.
   - Tune only on dev prompts.
   - Validate candidates on held-out prompts and at least one newly written
     prompt that was not used in layout/profile construction.
   - A result is not global SOTA if it only wins on prompts used to select
     experts or pack order.

7. Promotion rule.
   - Commit and push immediately only after a candidate passes metrics,
     quality, TTFT, RAM, and reproducibility gates.
   - Commit message body must include the full SOTA reproduction details listed
     above.
   - Update this plan with the run root, metrics, decision, and rollback point
     before starting the next optimization.

## Storage-layout tooling progress

Timestamp: 2026-07-11 CST.

Status: tooling and offline sizing only; no runtime behavior has changed and no
new SOTA is claimed.

Implemented/validated tooling:

- Add `.Agent/run-tools/kimi_io_trace_pack_layout_screen.py` as the
  reproducible offline screen for `GGML_MOE_IO_READ_TRACE_OUT` traces.
- Extend `scripts/kimi-build-trace-overlay-pack.py` so a trace overlay can be
  planned before copying payloads:
  - `--dry-run` avoids writing pack payloads;
  - `--report-json` saves the selected entry/byte summary;
  - `--alias-tsv` can copy missing entries from the GGUF alias TSV source;
  - `--roles up,gate,down` can restrict the overlay to a role subset.
- Important implementation detail:
  - `io-read-trace.csv` records the actual read-plan `nbytes`, which can be
    aligned span size rather than the pack-index payload size;
  - the overlay builder therefore selects source entries by `(tensor,
    expert_idx)` and writes the source entry's own payload `nbytes`.

Fresh dev2 dry-run inputs:

- Trace root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-layout-trace-dev2-n32`.
- Dev traces:
  - `dev_japan_factual/io-read-trace.csv`;
  - `dev_python_reverse/io-read-trace.csv`.
- Source packs:
  - `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack`;
  - `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack`.
- Alias TSV:
  `/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv`.

Dry-run results:

- Full `up,gate,down` greedy-pair overlay:
  - report:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-layout-trace-dev2-n32/overlay-plan/greedypair-dev2-with-alias-dryrun.json`;
  - trace rows after `--max-jobs 8`: `56943`;
  - entries: `28874`;
  - payload bytes: `164122886144` (`152.85 GiB`);
  - decision: do not build now because the filesystem has only about `133 GB`
    free, and this leaves no safe margin.
- `up+gate` greedy-pair overlay:
  - report:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-layout-trace-dev2-n32/overlay-plan/greedypair-dev2-upgate-dryrun.json`;
  - trace rows after `--max-jobs 8`: `36883`;
  - entries: `19246`;
  - payload bytes: `98490155008` (`91.72 GiB`);
  - decision: build this first because `up/gate` misses are the main exposed
    critical-path wait and the size fits current disk.
- `down` greedy-pair overlay:
  - report:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-layout-trace-dev2-n32/overlay-plan/greedypair-dev2-down-dryrun.json`;
  - trace rows after `--max-jobs 8`: `20060`;
  - entries: `9628`;
  - payload bytes: `65632731136` (`61.13 GiB`);
  - decision: keep as fallback A/B if `up+gate` fails or if disk budget becomes
    too tight.

Next runtime A/B plan:

1. Build default-off duplicate `up+gate` greedy-pair overlay:
   `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-dev2-upgate-greedypair-overlay.expert-pack`.
2. Use it only with explicit env:
   - `GGML_MOE_EXPERT_PACK_OVERLAY_EXTRA=<new overlay>`;
   - `GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1`;
   - `GGML_MOE_IO_ADJACENT_COALESCE=1`;
   - keep current baseline pack, overlay, alias TSV, memory cgroup, and
     cold-start settings unchanged.
3. Paired validation:
   - baseline and candidate on `dev_japan_factual` and `dev_python_reverse`;
   - record read jobs, adjacent coalesce groups/extents saved,
     `io_uring_wait`, decode ms/token, TTFT, RAM peak, quality output, and
     CPU fallback.
4. Pass condition:
   - lower decode ms/token and exposed `io_uring_wait`;
   - lower read-job/extents count or clear adjacent-coalesce evidence;
   - quality pass;
   - TTFT within `1.20x`;
   - host RAM below `15900000000` bytes.
5. Failure handling:
   - if bytes move but decode does not improve, reject the behavior and keep
     only tooling/docs;
   - if the overlay hurts generalization, do not promote it and do not call it
     SOTA.

Runtime A/B result:

- Timestamp: 2026-07-11 CST.
- Candidate artifact built:
  `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-dev2-upgate-greedypair-overlay.expert-pack`.
- Build report:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-layout-trace-dev2-n32/overlay-plan/greedypair-dev2-upgate-build.json`.
- Build stats:
  - entries: `19246`;
  - payload bytes copied: `98490155008`;
  - final pack size: `98493083648`;
  - build wall time: `4:36.56`;
  - max RSS: `107148 KB`.
- A/B run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-upgate-greedypair-ab`.
- Baseline `dev_japan_factual`:
  - quality: pass;
  - TTFT: `9090.12 ms`;
  - decode: `18387.02 ms / 31`;
  - token rate: `1.69 tok/s`;
  - RAM peak: `12767883264`;
  - `iouring_wait_us=17714011`;
  - `cqes=39075`;
  - `inflight_avg=4.26`.
- Candidate `dev_japan_factual`, depth 8:
  - env:
    `GGML_MOE_EXPERT_PACK_OVERLAY_EXTRA=<candidate>`,
    `GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1`,
    `GGML_MOE_IO_ADJACENT_COALESCE=1`,
    `GGML_MOE_IO_ADJACENT_MAX_GAP=4096`;
  - quality: pass;
  - TTFT: `10301.29 ms`;
  - decode: `20210.10 ms / 31`;
  - token rate: `1.53 tok/s`;
  - RAM peak: `12992634880`;
  - `iouring_wait_us=20922293`;
  - `cqes=30009`;
  - `inflight_avg=3.91`;
  - adjacent coalesce:
    `groups=7751`, `slices=16817`, `extents_saved=9066`,
    `physical_bytes=90125615104`, `payload_bytes=90125615104`,
    `max_group_jobs=3`.
- Candidate `dev_japan_factual`, depth 16 probe:
  - quality: pass;
  - TTFT: `8946.23 ms`;
  - decode: `20005.85 ms / 31`;
  - token rate: `1.55 tok/s`;
  - RAM peak: `12996317184`;
  - `iouring_wait_us=20215581`;
  - `cqes=30009`;
  - `inflight_avg=4.35`;
  - still below baseline.
- Decision:
  - reject the duplicate clustered overlay + adjacent coalesce behavior;
  - stop the paired test after the first dev prompt because it already fails
    the performance gate;
  - delete the 92GB candidate artifact to restore disk space;
  - keep only the default-off tooling and docs.
- Root cause:
  - the clustered overlay reduced completion count (`39075 -> 30009`) but did
    not reduce transferred bytes (`223.1 GB` stayed the same);
  - coalesced groups were small (`max_group_jobs=3`) and turned many reads into
    larger staging slots (`up/gate` slot sizes around `13-16 MiB`);
  - larger reads reduced effective queue continuity and increased
    `io_uring_wait`, so per-read overhead was not the dominant bottleneck in
    this path.
- Updated implication:
  - pack physical layout alone is not enough unless the runtime can coalesce
    SSD reads without inflating exposed staging/H2D latency;
  - future work should either lower bytes per expert, raise true queue depth
    with independent work, or split "large SSD read" from "per-expert H2D
    staging" so coalesced disk reads can still feed many small GPU transfers.

## H2D coalescing opportunity check

Timestamp: 2026-07-11 CST.

Status: default-off instrumentation and one cold-start profile; no runtime
behavior change and no SOTA claim.

Implementation:

- Add a guarded `GGML_MOE_H2D_COALESCE_PROFILE_OUT` call inside the io_uring
  expert-pack copy path.
- The profiler records, per io_uring batch:
  - logical copy count;
  - theoretical coalesced copy count;
  - host-contiguous source pairs;
  - device-destination-contiguous pairs;
  - pairs where both are contiguous.
- It is profile-only. It does not change read order, H2D copies, cache
  placement, or default env.

Run:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-h2d-coalesce-profile/baseline_japan`.
- Prompt:
  `Please introduce Japan in a short paragraph.`
- Command shape:
  current `kimi-general-prompt-repro.sh`, cold-start systemd cgroup
  `MemoryMax=15900000000`, `N=32`, `PROFILE=1`, plus:
  `GGML_MOE_H2D_COALESCE_PROFILE_OUT=$RUN/h2d-coalesce-profile.csv`.

Result:

- Quality: pass.
- TTFT: `8375.86 ms`.
- Decode: `18812.17 ms / 31`.
- Token rate: `1.65 tok/s`.
- RAM peak: `12765462528`.
- Expert pack:
  - `iouring_reads=39075`;
  - `iouring_bytes=223114035200`;
  - `iouring_wait_us=18926235`;
  - `cqes=39075`;
  - `inflight_avg=4.28`.
- H2D coalesce profile:
  - rows: `5742`;
  - total copies: `39075`;
  - theoretical coalesced copies: `39075`;
  - saved copies: `0`;
  - host-contiguous pairs: `0`;
  - dst-contiguous pairs: `0`;
  - both-contiguous pairs: `0`;
  - runtime_load copies: `34847`, saved `0`;
  - current_down_overlap copies: `4228`, saved `0`.

Decision:

- Reject standalone H2D coalescing as the next primary optimization.
- In the current baseline, io_uring groups are single-expert groups, so there
  are no contiguous host slices to merge.
- The earlier clustered overlay test created host-contiguous groups, but it
  regressed by reducing IO parallelism while bytes stayed constant. Therefore
  H2D coalescing would only be worth revisiting together with a byte-reducing
  pack format or a scheduler that keeps IO depth high.
- Next optimization should focus on:
  - reducing bytes per demanded expert with a quality-validated representation;
  - or raising true independent IO queue depth through prediction/prefetch that
    is not already covered by VRAM hits.

## Immediate goal: Kimi CPU/defer GPU-extension usefulness check

Timestamp: 2026-07-11 CST.

This is the next concrete execution goal after comparing the DeepSeek SOTA
path with Kimi.

Goal:

> Determine whether the DeepSeek-style CPU/defer main scheduler plus GPU
> expert-cache extension can still produce a reproducible, prompt-general Kimi
> gain from the current codebase. The short target is to recover and exceed the
> stable prompt-general `>2 tok/s` line without regressing the current zero
> decode CPU-fallback behavior. The product target remains stable `>5 tok/s`
> for random user prompts on one 32 GB RTX 5090-class GPU with host RAM strictly
> below 16 GB.

Why this is not a direct DeepSeek copy:

- DeepSeek's large gain came from moving important CPU/defer gate work into a
  VRAM-resident GPU extension path.
- Current Kimi profiles already show decode CPU fallback at zero in the stable
  configuration, so the first question is not "which missing kernel is falling
  back to CPU".
- Kimi's current exposed loss is mainly stalled expert movement and staging:
  `up/gate` wait, `down` staging, queue drain, and imperfect VRAM/RAM/SSD
  residency.
- Therefore the Kimi version of the DeepSeek idea is: keep CPU/defer as the
  scheduler, but make its GPU extension cover the critical expert residency,
  transfer, and compute path with less exposed wait.

Plan:

1. Re-establish the exact baseline on this branch.
   - Run one cold-start N32 smoke and one N96 profile with the current stable
     config.
   - Record token rate, TTFT, RAM peak, VRAM use, page-cache split, expert-pack
     bytes, H2D bytes, direct reads, `io_uring_wait`, staging wall, compute
     wall, and CPU fallback count.
   - The baseline must include the quality prompt
     `Please introduce France in a short paragraph.` and at least one
     non-France dev prompt.

2. Audit CPU/defer GPU-extension coverage before changing behavior.
   - Use `GGML_MOE_GPU_EXTENSION_COVERAGE_OUT` and existing profile CSVs to
     split time by role: `gate`, `up`, `down`, prompt, and decode.
   - Confirm whether any remaining slow path is a real CPU fallback, a GPU
     extension miss, a staging wait, an H2D wait, or a scheduler gap.
   - Do not implement a gate-only hotpool unless the profile proves gate is
     still on the exposed critical path.

3. Test Kimi-specific GPU-extension improvements in priority order.
   - First priority: reduce exposed mixed `up/gate` wait without losing useful
     overlap between up compute and gate movement.
   - Second priority: improve `down` staging only where it is still exposed
     after `up/gate`.
   - Third priority: replace low-value decode-time file-backed RAM pages with a
     batchable explicit expert RAM tier, but only when it reduces critical-path
     wait and stays under the host RAM cap.
   - Fourth priority: lower-byte or duplicate-on-disk expert layouts only when
     the runtime can consume them correctly for full active calls.

4. Keep the optimization prompt-general.
   - Use dev prompts only for tuning cache, layer, threshold, pack, or RAM-tier
     decisions.
   - Keep held-out prompts untouched until final validation.
   - Reject any change that only improves one prompt or one hand-picked hotset.

5. Promotion and rollback rule.
   - Promote only after paired cold-start A/B results show higher token rate,
     coherent output, TTFT within `1.20x`, host RAM below
     `15900000000` bytes, and profile evidence that critical-path wait fell.
   - If performance drops, quality regresses, TTFT exceeds the gate, or time
     merely moves elsewhere, revert the behavior change and retain only useful
     default-off instrumentation or docs.
   - Every accepted SOTA commit must be pushed and must include exact env,
     command, prompt split, run directory, metrics, quality output, and rollback
     commit in the commit body.

Current baseline for this goal:

- Timestamp: 2026-07-11 CST.
- Commit: `ddaafd1f7 docs: define Kimi GPU extension usefulness goal`.
- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-current-dev4-n32-profile`.
- Prompt split:
  - dev tuning only:
    `dev_france_regression`, `dev_japan_factual`,
    `dev_photosynthesis_factual`, `dev_python_reverse`;
  - held-out prompts remain unused for selection.
- Command shape:
  `systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0 ... env RUN=<run> N=32 PROFILE=1 PROMPT_ID=<dev-id> PROMPT_USER_TEXT=<prompt> QUALITY_KEYWORDS=<keywords> .Agent/run-tools/kimi-general-prompt-repro.sh`.
- Results:
  - France: quality pass, `1.71 tok/s`, TTFT `8969.44 ms`,
    decode `18175.30 ms / 31`, RAM peak `12766498816`;
  - Japan: quality pass, `1.70 tok/s`, TTFT `8418.58 ms`,
    decode `18194.00 ms / 31`, RAM peak `12769886208`;
  - photosynthesis: quality pass, `1.67 tok/s`, TTFT `7396.14 ms`,
    decode `18562.81 ms / 31`, RAM peak `12605837312`;
  - Python reverse: quality pass, `1.54 tok/s`, TTFT `9135.70 ms`,
    decode `20155.26 ms / 31`, RAM peak `12759814144`.
- Aggregate:
  - decode tokens: `124`;
  - decode ms/token: `605.543`;
  - approximate average decode rate: `1.65 tok/s`;
  - expert-pack bytes per prompt: `218.8-255.9 GB`;
  - `io_uring_wait` per prompt: `18.4-21.4 s`;
  - upgate hit rate: `37.0-45.4%`;
  - down hit rate: `57.0-59.5%`;
  - decode CPU fallback: no fallback rows observed in this profile.
- Wait-weighted screen:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-current-dev4-n32-profile/analysis/wait-weighted-screen.md`.
- Interpretation:
  - exposed time is still dominated by distributed `up/gate` misses;
  - low-layer `down` staging is also visible but smaller than aggregate
    `up/gate`;
  - the next experiment should improve general VRAM residency, not use a
    France-only hotset or whole-layer RAM admission.

Next A/B experiment: dev4 budgeted VRAM profile pinning

- Timestamp: 2026-07-11 CST.
- Candidate profile:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-current-dev4-n32-profile/analysis/dev4-budgeted-upgate1024-down512.profile.csv`.
- Generation command:
  `python3 .Agent/run-tools/kimi_make_budgeted_hot_expert_profile.py --route-root <dev4-profile-root> --screen-csv <wait-weighted-screen.csv> --out-profile <profile.csv> --out-report <profile.md> --upgate-budget-mib 1024 --down-budget-mib 512 --min-prompts 3 --max-layers-per-role 12`.
- Profile contents:
  - entries: `277`;
  - total: `1527.20 MiB`;
  - upgate: `211` entries, `1019.70 MiB`, route count `7551`;
  - down: `66` entries, `507.50 MiB`, route count `3142`;
  - selection uses only dev prompts, weighted by layer/role ms/token and route
    count per byte.
- Runtime env under test:
  - `GGML_MOE_VRAM_PROFILE=<profile.csv>`;
  - `GGML_MOE_VRAM_PROFILE_PROTECT=1`;
  - `GGML_MOE_VRAM_PROFILE_PRELOAD=1`;
  - `GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT=1`;
  - `GGML_MOE_VRAM_PROFILE_PIN_MIN_COUNT=1`;
  - `GGML_MOE_VRAM_CACHE_POLICY=profile_lfu_lru`;
  - `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=20`;
  - keep `UPGATE_PCT=62`, `VRAM_MIB=15000`, `PINNED_SLOTS=12`.
- Theory:
  - the profile pins high-reuse, high-criticality experts in the existing VRAM
    cache so repeated dev-general requests avoid SSD/io_uring staging;
  - the hard upper bound is limited by selected repeated traffic, roughly a
    single-digit percent of total expert-pack bytes in N32, so a successful
    result is expected to be incremental rather than enough for `>2 tok/s`;
  - a valid win must lower exposed `io_uring_wait` or role wall time, not only
    increase hit rate.
- Risks:
  - cold preload can increase TTFT;
  - pinned profile slots can evict useful dynamic cache entries and hurt
    prompt-general performance;
  - profile lookup normalizes some up/gate names to down names, so the run must
    verify actual `preloads` and `pinned` counts in metrics.
- Required paired validation:
  - run N32 baseline and candidate on at least two dev prompts, including
    `dev_python_reverse` because it is currently the slowest;
  - compare token rate, TTFT, RAM peak, upgate/down hit rate, `preloads`,
    `pinned`, `io_uring_wait`, and quality;
  - if promising, expand to N96 dev and later held-out; if not, reject and keep
    only this documented negative result.

Result:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-profile-cache-dev2-ab`.
- Paired prompts:
  `dev_japan_factual`, `dev_python_reverse`.
- Baseline:
  - Japan: quality pass, `1.70 tok/s`, TTFT `9309.39 ms`,
    decode `18249.54 ms / 31`, RAM peak `12763987968`,
    IO bytes `223.11 GB`, `io_uring_wait=18.80 s`, down hit `59.5%`,
    upgate hit `44.4%`;
  - Python reverse: quality pass, `1.53 tok/s`, TTFT `8706.42 ms`,
    decode `20203.99 ms / 31`, RAM peak `12758818816`,
    IO bytes `255.88 GB`, `io_uring_wait=21.58 s`, down hit `57.0%`,
    upgate hit `37.0%`.
- Candidate with preload/pin/profile_lfu_lru:
  - Japan: quality fail (`dangling_comma`), `1.37 tok/s`, TTFT `9414.15 ms`,
    decode `22595.99 ms / 31`, RAM peak `12760543232`,
    IO bytes `274.66 GB`, `io_uring_wait=23.59 s`, down hit `30.8%`,
    upgate hit `30.1%`, direct reads `75`, down pinned `578`,
    upgate pinned `172`;
  - Python reverse: quality fail (`repeat_collapse`), `1.35 tok/s`,
    TTFT `9437.42 ms`, decode `22921.04 ms / 31`,
    RAM peak `12760342528`, IO bytes `287.74 GB`,
    `io_uring_wait=23.89 s`, down hit `33.0%`, upgate hit `25.7%`,
    direct reads `62`, down pinned `578`, upgate pinned `142`.
- Decision:
  - reject preload/pin profile admission;
  - it increased IO bytes and exposed wait, reduced both VRAM hit rates, and
    failed quality;
  - failure mode: a fixed dev hotset was not general enough, and pinning
    reserved too many cache slots for low-current-value entries. Preload also
    introduced direct reads from the GGUF/source path. This is the opposite of
    the desired prompt-general dynamic cache behavior.

Next A/B experiment: profile-guided eviction without preload or pinning

- Timestamp: 2026-07-11 CST.
- Motivation:
  - the rejected run shows that hard pinning/preloading harms dynamic cache
    adaptation;
  - however, profile counts may still be useful as a soft eviction hint after
    an expert has already been demanded by the current prompt.
- Runtime env:
  - `GGML_MOE_VRAM_PROFILE=<dev4 profile.csv>`;
  - `GGML_MOE_VRAM_PROFILE_PRELOAD=0`;
  - `GGML_MOE_VRAM_PROFILE_PROTECT=0`;
  - `GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT=0`;
  - `GGML_MOE_VRAM_CACHE_POLICY=profile_lfu_lru`;
  - keep all other current SOTA env unchanged.
- Theory:
  - no cold preload means TTFT and direct reads should remain close to baseline;
  - no pinning means dynamic cache can still adapt to each prompt;
  - `profile_lfu_lru` should mainly evict zero-profile entries before
    high-dev-reuse entries, if the profile generalizes.
- Required A/B:
  - run the same two N32 dev prompts as the rejected test;
  - promote only if quality passes and token rate improves on both or improves
    average without hurting the slower Python prompt;
  - reject if hit rate rises but token rate falls, because this means the soft
    profile is not aligned with the critical path.

Result:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-profile-lfu-dev2-ab`.
- Baseline:
  - Japan: quality pass, `1.69 tok/s`, TTFT `8614.26 ms`,
    decode `18299.25 ms / 31`, RAM peak `12770054144`,
    IO bytes `223.11 GB`, `io_uring_wait=18.87 s`, down hit `59.5%`,
    upgate hit `44.4%`;
  - Python reverse: quality pass, `1.54 tok/s`, TTFT `9115.20 ms`,
    decode `20172.33 ms / 31`, RAM peak `12758016000`,
    IO bytes `255.88 GB`, `io_uring_wait=21.74 s`, down hit `57.0%`,
    upgate hit `37.0%`.
- Candidate with `profile_lfu_lru` only:
  - Japan: quality fail (`missing_keyword:japan`), `1.27 tok/s`,
    TTFT `8906.75 ms`, decode `24428.35 ms / 31`,
    RAM peak `12761583616`, IO bytes `295.20 GB`,
    `io_uring_wait=24.75 s`, down hit `31.5%`, upgate hit `33.2%`;
  - Python reverse: quality fail (`dangling_comma`), `1.24 tok/s`,
    TTFT `9935.98 ms`, decode `25057.94 ms / 31`,
    RAM peak `12756221952`, IO bytes `310.25 GB`,
    `io_uring_wait=26.15 s`, down hit `29.5%`, upgate hit `31.0%`.
- Decision:
  - reject profile-guided eviction;
  - even without preload/pin/direct reads, the dev profile is not a safe
    prompt-general eviction signal. It appears to protect too few useful
    current-prompt entries and evicts many zero-profile entries that are
    actually needed by the current prompt, causing cache thrash and more IO.
- Lesson:
  - do not use offline prompt-derived hotsets as a hard or soft runtime
    residency policy unless validation is done on held-out prompts and the
    profile covers enough traffic to avoid zero-profile thrash;
  - for random-prompt generalization, prefer online prompt-local signals or
    model-structural policies over dev hotsets.

Next A/B experiment: prompt-agnostic online LFU/LRU cache policy

- Timestamp: 2026-07-11 CST.
- Motivation:
  - the current default behaves like recency-dominant eviction;
  - a pure online `lfu_lru` policy uses only current-run reuse and does not
    depend on prompt-specific offline hotsets;
  - if expert reuse within a random prompt is strong enough, LFU/LRU may keep
    repeated experts without freezing the cache.
- Runtime env:
  - `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`;
  - no `GGML_MOE_VRAM_PROFILE`;
  - no preload or pinning changes.
- Theory:
  - TTFT should stay near baseline because there is no preload;
  - IO bytes should fall if current-prompt reuse is being evicted too early by
    LRU;
  - if route distribution is too broad or phase-local, LFU can hurt by keeping
    stale early experts and evicting later layer-local entries.
- Required A/B:
  - run the same Japan/Python N32 paired test;
  - accept only if both quality gates pass and average token rate improves
    without a large regression on the slower Python prompt;
  - if it fails, stop cache-policy env-only experiments and move to a more
    structural approach: per-layer byte/layout changes, lower-byte full-active
    overlay, or earlier true next-layer prefetch.

Result:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-online-lfu-dev2-ab`.
- Baseline:
  - Japan: quality pass, `1.70 tok/s`, TTFT `8190.17 ms`,
    decode `18198.73 ms / 31`, RAM peak `12769767424`,
    IO bytes `223.11 GB`, `io_uring_wait=18.87 s`, down hit `59.5%`,
    upgate hit `44.4%`;
  - Python reverse: quality pass, `1.54 tok/s`, TTFT `8761.89 ms`,
    decode `20114.04 ms / 31`, RAM peak `12759830528`,
    IO bytes `255.88 GB`, `io_uring_wait=21.60 s`, down hit `57.0%`,
    upgate hit `37.0%`.
- Candidate with online `lfu_lru`:
  - Japan: quality fail (`missing_keyword:asia|tokyo|island`),
    `1.25 tok/s`, TTFT `8519.61 ms`, decode `24756.42 ms / 31`,
    RAM peak `12763054080`, IO bytes `306.64 GB`,
    `io_uring_wait=25.65 s`, down hit `28.8%`, upgate hit `30.7%`;
  - Python reverse: quality fail (`repeat_collapse`), `1.13 tok/s`,
    TTFT `9128.27 ms`, decode `27547.87 ms / 31`,
    RAM peak `12756643840`, IO bytes `335.20 GB`,
    `io_uring_wait=28.74 s`, down hit `24.2%`, upgate hit `26.0%`.
- Decision:
  - reject online `lfu_lru`;
  - current expert reuse is not well served by a global frequency policy. It
    likely retains stale early high-frequency experts and evicts later
    layer-local entries, which increases down prefetch demand, total IO, and
    exposed wait.
- Cache-policy conclusion:
  - stop env-only cache-policy experiments for this stage;
  - fixed offline hotsets, profile-guided LFU, and online LFU all reduce
    prompt-general quality/performance on the dev2 validation;
  - future cache changes must be structurally constrained by layer/role/phase
    or must prove lower-byte full-active consumption, not only alter global
    eviction priority.

Next structural direction: reduce exposed bytes before global eviction tuning

- Timestamp: 2026-07-11 CST.
- Bottleneck after dev4 current baseline:
  - `up/gate` remains the largest aggregate exposed term;
  - `io_uring` is not saturated like the pure IO bench because runtime submits
    small layer-local batches after routing is known;
  - cache-policy changes made queue work larger, not smaller.
- Priority 1: true next-layer prefetch feasibility.
  - Instrument when layer `L+1` gate/top-k becomes known relative to completion
    of layer `L` down compute.
  - If there is usable slack, enqueue next-layer up/gate/down miss reads
    earlier without changing compute order.
  - Required proof: `io_uring_wait` and role wall decrease while total IO bytes
    do not increase materially.
- Priority 2: role/layer pack layout for larger contiguous batches.
  - Use existing route traces to identify layer/role buckets with persistent
    low hit and high wait.
  - Build a duplicate pack layout for those buckets that stores adjacent
    same-layer active experts contiguously by role and size, so one miss group
    becomes fewer larger reads.
  - This trades disk space for read efficiency without relying on prompt-specific
    hotsets.
- Priority 3: lower-byte full-active runtime path.
  - The tiny v2 overlay was rejected because active-call coverage was too low.
  - Revisit only if a generated pack can cover complete active calls for the
    targeted layers/roles, or if split-compute is implemented correctly.
- Priority 4: explicit RAM tier only after page-cache value audit.
  - RAM should hold batchable, high-value expert data only if it replaces
    measured low-value decode file pages and does not fragment SSD batches.
  - Whole-layer or large pinned RAM admission must be re-tested only with
    prompt-general dev/held-out validation, because previous full-layer/pinned
    attempts were unstable.

Predictive prefetch acceptance screen:

- Timestamp: 2026-07-11 CST.
- Script:
  `.Agent/run-tools/kimi_predictive_prefetch_acceptance.py`.
- Run:
  `python3 .Agent/run-tools/kimi_predictive_prefetch_acceptance.py --route-root /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-current-dev4-n32-profile --out-csv <analysis>/predictive-prefetch-acceptance.csv --out-md <analysis>/predictive-prefetch-acceptance.md`.
- Report:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-current-dev4-n32-profile/analysis/predictive-prefetch-acceptance.md`.
- Inputs:
  - dev prompts: `4`;
  - layers: `60`;
  - layer/prompt sequences: `240`;
  - role: `gate`; up/down use the same routed expert IDs, so the acceptance
    applies to the active expert ID set.
- Results:
  - previous-token same-layer top-8 precision/recall: `34.41% / 34.41%`;
  - exact set match: `0.03%`;
  - sliding window frequency `W=4, top-1`: precision `65.160%`, recall
    `8.145%`;
  - `W=4, top-2`: precision `59.230%`, recall `14.808%`;
  - `W=8, top-1`: precision `67.469%`, recall `8.434%`;
  - best `W=4, top-1` layers include layer `48` at `86.11%`, layer `9` at
    `83.33%`, and layers `37/56` at `81.48%`.
- Interpretation:
  - full top-8 previous-token prefetch is too wasteful and should not be
    implemented;
  - a small, throttled, layer-whitelisted sliding-window predictor is plausible;
  - expected recall is low, so this cannot by itself reach `>5 tok/s`, but it
    may reduce exposed queue gaps if submitted only when spare host-prefetch
    capacity exists.
- Next implementation scope:
  - default-off env, e.g. `GGML_MOE_PREDICTIVE_HOST_PREFETCH=1`;
  - window `4`, top-n `1` initially;
  - layer whitelist generated from the acceptance report, starting only with
    layers above a precision threshold such as `70%`;
  - submit to existing `GGML_MOE_PLANNED_HOST_PREFETCH`/host-prefetch queue;
  - do not prefetch if expert is already in VRAM cache or already queued;
  - record submitted/hit/unused counters;
  - A/B must verify that total IO bytes do not rise enough to erase the wait
    reduction.

Current implementation step: default-off predictive planned host prefetch

- Timestamp: 2026-07-11 CST.
- Code behavior:
  - adds `GGML_MOE_PREDICTIVE_HOST_PREFETCH=1`;
  - requires `GGML_MOE_PREDICTIVE_HOST_PREFETCH_LAYERS`, otherwise no
    prediction is submitted;
  - default window/top-n: `W=4`, `topn=1`;
  - default roles: `up,gate`; down is supported by
    `GGML_MOE_PREDICTIVE_HOST_PREFETCH_ROLES=all|down` but remains off for the
    first A/B;
  - uses the existing host-prefetch worker and planned queue;
  - keeps existing `GGML_MOE_PLANNED_HOST_PREFETCH` semantics unchanged, so
    enabling predictive prefetch does not automatically enable current-demand
    planned prefetch;
  - submits predictions only after current mixed up/gate copy is complete, to
    avoid competing with the current layer's exposed up/gate miss load.
- Build:
  - command: `cmake --build build-cuda-batch --target ggml-cuda -j2`;
  - result: passed; warnings are existing unused/missing-declaration/truncation
    warnings.
- First A/B env:
  - `GGML_MOE_PREDICTIVE_HOST_PREFETCH=1`;
  - `GGML_MOE_PREDICTIVE_HOST_PREFETCH_LAYERS=48,9,37,56,46`;
  - `GGML_MOE_PREDICTIVE_HOST_PREFETCH_WINDOW=4`;
  - `GGML_MOE_PREDICTIVE_HOST_PREFETCH_TOPN=1`;
  - `GGML_MOE_PREDICTIVE_HOST_PREFETCH_ROLES=upgate`;
  - `GGML_MOE_HOST_PREFETCH_SLOTS=16`;
  - `GGML_MOE_HOST_PREFETCH_MAX_MIB=128`.
- A/B prompts:
  - `dev_japan_factual`;
  - `dev_python_reverse`.
- Promotion gate:
  - quality must pass on both;
  - average token rate must improve without a Python regression;
  - host RAM must stay below `15900000000`;
  - total expert-pack bytes and `io_uring_wait` must not rise enough to erase
    the host-prefetch hit benefit.

Result:

- Code build:
  - `cmake --build build-cuda-batch --target ggml-cuda -j2` passed;
  - added a `ready_count` fast path so enabling host-prefetch with an empty
    ready queue does not take the host-prefetch mutex for every demand miss.
- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-predictive-prefetch-dev2-ab2`.
- Candidate `precision5`:
  - layers: `48,9,37,56,46`;
  - Japan: quality pass, `1.65 tok/s`, TTFT `8955.81 ms`,
    decode `18750.35 ms / 31`, RAM peak `12762730496`,
    IO bytes `223.11 GB`, `io_uring_wait=19.07 s`;
  - Python reverse: quality pass, `1.43 tok/s`, TTFT `9406.11 ms`,
    decode `21680.66 ms / 31`, RAM peak `12759486464`,
    IO bytes `255.88 GB`, `io_uring_wait=23.14 s`;
  - predictive counters: `predicted=112`, `submitted=0`,
    `skipped_vram=224`, `hits=0`.
- Candidate `wait5`:
  - layers: `14,33,32,28,29`;
  - Japan: quality pass, `1.67 tok/s`, TTFT `9020.59 ms`,
    decode `18545.44 ms / 31`, RAM peak `12769058816`,
    IO bytes `223.11 GB`, `io_uring_wait=19.09 s`;
  - Python reverse: quality pass, `1.52 tok/s`, TTFT `9395.73 ms`,
    decode `20434.09 ms / 31`, RAM peak `12762267648`,
    IO bytes `255.88 GB`, `io_uring_wait=21.81 s`;
  - predictive counters: `predicted=140`, `submitted=0`,
    `skipped_vram=280`, `hits=0`.
- All-layer top-1 smoke:
  - run:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-predictive-prefetch-all-smoke/dev_japan_all`;
  - quality pass, `1.65 tok/s`, TTFT `9492.44 ms`,
    decode `18802.33 ms / 31`, RAM peak `12767559680`;
  - counters: `predicted=896`, `submitted=0`, `skipped_vram=1792`,
    `hits=0`.
- All-layer top-2 smoke:
  - run:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-predictive-prefetch-top2-all-smoke/dev_japan_top2_all`;
  - quality pass, `1.64 tok/s`, TTFT `8761.06 ms`,
    decode `18891.12 ms / 31`, RAM peak `12779397120`;
  - counters: `predicted=1792`, `submitted=2`, `skipped_vram=3582`,
    `host_prefetch_hits=0`, `ready=2`.
- Default-off smoke:
  - run:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-predictive-defaultoff-france-smoke`;
  - quality pass, `1.68 tok/s`, TTFT `9837.23 ms`,
    decode `18501.71 ms / 31`, RAM peak `12768358400`;
  - confirms default-off behavior remains in the normal N32 range.
- Decision:
  - do not promote predictive host prefetch as an optimization;
  - keep only as default-off probe/instrumentation if retained in code;
  - reason: high-confidence predicted experts are already resident in VRAM, so
    prediction mostly becomes `skipped_vram`; the few second-rank submissions
    did not produce host-prefetch hits.
- Next direction:
  - stop small-model/current-history expert prediction for now;
  - focus on lowering bytes for nonresident experts or changing pack/layout so
    existing demand reads become larger and more contiguous.

## Current execution goal: Kimi CPU/defer GPU-extension parity

Timestamp: 2026-07-11 CST.

This section remains the broader constraint set and audit context. The
immediate execution entry point is the section above; older sections below
remain as audit history.

Goal:

> Reuse the successful DeepSeek CPU/defer + GPU expert-cache extension pattern
> for Kimi without changing the user-visible model behavior. For Kimi this does
> not mean "remove CPU fallback" first, because the current stable profile already
> shows decode CPU fallback at zero. The real goal is to make the CPU/defer MoE
> scheduler hand off more `gate/up/down` expert residency, transfer, and compute
> work to the GPU extension with less exposed wait. The immediate target is a
> pushed, cold-start, prompt-general, reproducible improvement to stable
> `>2 tok/s`; the product target remains stable `>5 tok/s` for random prompts on
> a 16 GB host-RAM machine with one 32 GB RTX 5090-class GPU.

Hard gates:

- Cold start only. Drop caches before accepted measurements.
- Host RAM peak below `15900000000` bytes, including page cache, mmap pages,
  pinned/pageable expert buffers, helper processes, and cgroup accounting.
- Use VRAM aggressively, but do not exceed the TTFT gate or create host reclaim
  instability.
- Paired TTFT must stay within `1.20x` of the baseline used for that experiment.
- Mandatory quality gate: `Please introduce France in a short paragraph.` must
  be coherent and semantically correct.
- Optimization must be prompt-general. Held-out prompts must not be used for
  hotset, layer, threshold, pack, or cache-policy selection.
- Every accepted SOTA must be pushed and reproducible. The commit body must
  include improvement size, exact env, exact command, prompt split, run
  directory, quality result, TTFT/RAM/VRAM/IO/H2D/fallback metrics, and rollback
  commit.

Why the DeepSeek idea is relevant to Kimi:

- The transferable idea is architectural: keep CPU/defer as the MoE scheduler,
  but treat GPU as an expert-cache and expert-compute extension.
- Kimi should not blindly copy the DeepSeek gate-only optimization. Current Kimi
  stable runs already report decode CPU fallback at zero, so the main exposed
  loss is expert movement/staging wait rather than missing GPU math kernels.
- The next Kimi win must therefore come from higher GPU-extension coverage,
  better queue continuity, fewer bytes per active expert, or a more useful
  RAM/VRAM/SSD residency split.

Plan:

1. Reproduce the current stable baseline before new optimization.
   - Use the current pushed branch and record commit, rollback point, model,
     env, prompt split, command, and run directory.
   - Run at least one cold-start N32 smoke and one N96 profile on dev prompts.
   - Record token rate, TTFT, host RAM peak, VRAM use, page-cache split,
     expert-pack bytes, H2D bytes, `io_uring_wait`, staging wall time, compute
     time, direct reads, and CPU fallback count.

2. Add a default-off GPU-extension coverage audit before changing behavior.
   - For each MoE call, log layer, role, active experts, cache hits/misses,
     source type, packed type if available, bytes requested, bytes transferred,
     `io_uring_wait`, H2D time, compute time, and whether the call was served by
     current GPU extension or had to use a residual path.
   - Confirm whether Kimi's slow calls are `gate`, `up`, `down`, mixed
     `up/gate`, or scheduler gaps between calls.
   - This audit must be behavior-neutral and default-off.

3. Prioritize experiments by measured critical-path seconds.
   - If gate wait is still exposed on Kimi, test a DeepSeek-style gate hotpool
     parity path first.
   - If mixed `up/gate` remains the largest row, prioritize paired `up/gate`
     residency and transfer scheduling.
   - If down wait is exposed after up/gate work, extend the same mechanism to
     same-layer `up/gate/down` demand scheduling.
   - If byte volume is the largest bound, use the `GGMLMOEPACKv2` lower-byte
     overlay only where the runtime can consume the smaller representation
     correctly; otherwise keep it as a profiling tool, not an SOTA claim.
   - If host RAM is filled by low-value file-backed pages, replace only the
     measured low-value portion with explicit, batchable expert RAM residency.

4. Validate every candidate as an A/B, then either promote or revert.
   - First run small N32 quality and accounting checks.
   - Then run N96 dev profile and compare against the paired cold-start
     baseline.
   - Promote only if token rate improves, TTFT stays within gate, host RAM stays
     below cap, quality passes, and profiling shows the critical-path wait was
     actually reduced.
   - If token rate falls, quality degrades, TTFT exceeds gate, or the profile
     only moves time elsewhere, revert the behavior change and keep only useful
     default-off instrumentation.

5. Report and commit discipline.
   - Update this plan before each new implementation step.
   - Record timestamped results after each run.
   - Push immediately after any accepted improvement.
   - Never call a result SOTA unless the pushed commit plus documented command
     can reproduce it from cold start.

Current implementation step:

- Timestamp: 2026-07-11 CST.
- Add a default-off GPU-extension coverage audit under
  `GGML_MOE_GPU_EXTENSION_COVERAGE_OUT`.
- Scope:
  - record one CSV row per `up`, `gate`, or `down` batch call;
  - include phase, role, tensor, logical type/shape/bytes, active routes,
    current VRAM cache hits/misses, v2 overlay hits, v2 supported hits,
    homogeneous full-cover status, and theoretical byte savings for full-cover
    or split-overlay paths;
  - do not change cache state, scheduling, expert bytes, or compute dispatch.
- Purpose:
  - decide whether Kimi can benefit from a DeepSeek-style gate hotpool parity
    path, paired `up/gate` residency, same-layer `up/gate/down` scheduling, or
    lower-byte `GGMLMOEPACKv2` runtime consumption;
  - reject overlay/runtime work if the profile shows low full-active coverage or
    only non-critical byte savings.
- Promotion rule:
  - this step is instrumentation only and cannot be called SOTA;
  - behavior changes may start only after a cold-start N32/N96 profile shows
    where the exposed wait sits.
- Verification:
  - command:
    `cmake --build build-cuda-batch --target ggml-cuda -j2`;
  - result: passed, rebuilt `bin/libggml-cuda.so`;
  - warnings: existing unused/missing-declaration/truncation warnings only;
  - SOTA claim: none, because runtime behavior is unchanged unless
    `GGML_MOE_GPU_EXTENSION_COVERAGE_OUT` is set.

First coverage run:

- Timestamp: 2026-07-11 CST.
- Commit: `561e0bd90 prof: audit Kimi GPU extension coverage`.
- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-gpu-extension-coverage-n32-france`.
- Command shape:
  `systemd-run --wait --collect --same-dir -p MemoryMax=15900000000 -p MemorySwapMax=0 ... env RUN=<run> N=32 PROFILE=1 PROMPT_ID=dev_france_coverage PROMPT_USER_TEXT='Please introduce France in a short paragraph.' EXTRA_RUNTIME_ENV='GGML_MOE_GPU_EXTENSION_COVERAGE_OUT=<run>/gpu-extension-coverage.csv; GGML_MOE_EXPERT_PACK_V2=<tiny-pack>' .Agent/run-tools/kimi-general-prompt-repro.sh`.
- Result:
  - quality: pass;
  - output: `France is a country in Western Europe known for its rich history,
    culture, and influence on art, fashion, and cuisine. Its capital, Paris, is
    famous`;
  - token rate: `1.66 tok/s`;
  - TTFT: `9437.82 ms`;
  - decode: `18692.28 ms / 31 runs`;
  - RAM peak: `12766507008` bytes;
  - CPU fallback profile: empty;
  - expert-pack iouring: `39090` reads, `224038649856` bytes,
    `18910801 us` wait, inflight avg `4.32`, max `8`;
  - VRAM cache: down hit `57.7%`, upgate hit `45.4%`.
- Coverage CSV:
  - rows: `5760`;
  - tiny v2 overlay rows with any hit: `67`;
  - full homogeneous v2 cover calls: `0`;
  - full-cover saved bytes: `0`;
  - decode v2 hit rates are too low for a runtime bridge:
    - up type `22`: `0.003346`;
    - gate type `22`: `0.007488`;
    - down type `11`: `0.002555`.
- Decode attribution from profile CSV:
  - up/gate wall: `12251.642 ms` total, about `395 ms/token`;
  - up/gate wait: up `7206.880 ms`, gate `7682.755 ms`;
  - largest up/gate pair: `up=22/gate=18/parallel_stage=1`, `6536.4 ms`
    wall, up wait `4038.4 ms`, gate wait `4372.1 ms`;
  - decode down wall: `5064.805 ms`, about `163 ms/token`;
  - decode down stage: `4702.819 ms`, about `152 ms/token`.
- Interpretation:
  - this confirms that the tiny `GGMLMOEPACKv2` sidecar is useful as a reader
    and coverage probe, but not as a promotable runtime overlay;
  - the next behavior-changing optimization should target exposed mixed
    `up/gate` wait first, then decode down staging;
  - a v2/lower-byte bridge should wait until the pack/hotset can cover full
    active calls or until a correct split-compute path exists.

Next A/B experiment: mixed up/gate grouped combined stage

- Timestamp: 2026-07-11 CST.
- Motivation:
  - the first coverage run shows the largest exposed term is mixed up/gate,
    especially `up=22/gate=18/parallel_stage=1`;
  - current mixed parallel stage plans `up_jobs` and `gate_jobs` separately and
    copies them on two role-specific threads/rings;
  - profile still shows low effective inflight (`~4.32`) and high up/gate wait,
    so the queue is not behaving like the pure IO bench.
- Proposed default-off env:
  `GGML_MOE_MIXED_UP_GATE_COMBINED_STAGE=1`.
- Implementation rule:
  - only applies inside the existing mixed-type parallel-stage branch;
  - build one combined job list from `up_jobs + gate_jobs`;
  - sort/group by `nbytes`, because `expert_pack_iouring_copy_jobs` requires a
    uniform expert byte size per batch;
  - pre-grow the shared pinned ring to `max(up_expert_bytes, gate_expert_bytes)`
    before grouped copy, so switching from smaller `up` entries to larger
    `gate` entries cannot free a pinned ring while earlier H2D work is pending;
  - copy all grouped jobs on one stream, record an event, then let both up and
    gate compute streams wait/continue from that event.
- Theory:
  - possible win: fewer tiny role-local IO submissions, better offset sorting,
    simpler queue continuity, and less role-to-role SSD contention;
  - hard upper bound is limited by the measured mixed up/gate wait:
    `~7206.880 ms` up wait and `~7682.755 ms` gate wait over N32, or roughly
    `230-250 ms/token`;
  - practical win must beat the lost overlap between up compute and gate copy,
    which was only `~264.207 ms` up compute total in the N32 profile.
- Required A/B:
  - run cold-start N32 baseline and `GGML_MOE_MIXED_UP_GATE_COMBINED_STAGE=1`
    with the same prompt/env;
  - compare token rate, TTFT, RAM peak, quality, up/gate wait, down staging,
    iouring inflight, and iouring wait;
  - if token rate falls, TTFT exceeds gate, or wait just moves elsewhere, revert
    the behavior change or leave it default-off with rejection notes.

Result:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-mixed-combined-stage-ab`.
- Build state: dirty test build with the proposed default-off branch compiled;
  behavior branch was reverted after the A/B because it failed.
- Baseline:
  - quality: pass;
  - token rate: `1.66 tok/s`;
  - TTFT: `8725.62 ms`;
  - decode: `18726.22 ms / 31 runs`;
  - RAM peak: `12766937088` bytes;
  - iouring wait: `19026709 us`, inflight avg `4.38`;
  - up/gate wall: `12236.745 ms`;
  - up wait: `7045.304 ms`, gate wait: `7707.093 ms`;
  - decode down wall: `5088.172 ms`.
- Combined stage:
  - quality: pass;
  - token rate: `1.63 tok/s`;
  - TTFT: `8455.28 ms`;
  - decode: `19006.09 ms / 31 runs`;
  - RAM peak: `12753309696` bytes;
  - iouring wait: `16387706 us`, inflight avg `4.42`;
  - up/gate wall: `12549.103 ms`;
  - up wait: `7716.684 ms`, gate wait: `8006.678 ms`;
  - decode down wall: `4997.365 ms`.
- Decision:
  - reject this behavior change;
  - although total iouring wait decreased, mixed up/gate wall and wait increased
    and token rate dropped from `1.66` to `1.63 tok/s`;
  - likely cause: combined copy removed useful overlap between up compute and
    gate copy, and same-size grouping could not merge the dominant `up=22` and
    `gate=18` jobs into one uniform iouring batch;
  - code behavior was reverted; only this negative result is retained.
- Next direction:
  - do not pursue role-level combined staging for mixed up/gate;
  - focus on either increasing useful up/gate VRAM hit rate, reducing up/gate
    transferred bytes with a broader full-active low-byte pack, or scheduling
    future-layer prefetch before the layer reaches the blocking point.

Next A/B experiment: VRAM cache split upgate percentage

- Timestamp: 2026-07-11 CST.
- Motivation:
  - coverage/profile showed upgate hit around `45%` and down hit around
    `57-64%`;
  - hypothesis A: giving more VRAM to up/gate may reduce the dominant mixed
    upgate wait;
  - hypothesis B: giving less VRAM to up/gate may improve down/current-down
    overlap enough to reduce total exposed wait.
- Cold-start N32 France profile sweep:
  - run root:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-upgate-cache-pct-ab`;
  - all runs used `PROFILE=1`, 16 GB cgroup, same prompt and current runtime
    env; only `UPGATE_PCT` changed.

| pct | quality | tok/s | TTFT ms | decode ms | IO GiB | IO wait s | up hit | gate hit | upgate wall s | down hit | down wall s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | pass | 1.62 | 8390.12 | 19152.88 | 213.57 | 20.39 | 0.347 | 0.347 | 13.192 | 0.702 | 4.601 |
| 55 | pass | 1.77 | 7371.90 | 17494.35 | 209.67 | 18.23 | 0.376 | 0.376 | 11.813 | 0.701 | 4.377 |
| 62 | pass | 1.66 | 8725.62 | 18726.22 | 208.65 | 19.03 | 0.455 | 0.455 | 12.237 | 0.639 | 5.088 |
| 75 | pass | 1.53 | 8954.26 | 20324.25 | 237.57 | 20.37 | 0.482 | 0.481 | 12.612 | 0.483 | 6.123 |
| 80 | pass | 1.51 | 8516.07 | 20513.24 | 234.78 | 20.26 | 0.502 | 0.502 | 12.619 | 0.483 | 6.301 |

- France-only interpretation:
  - `UPGATE_PCT=55` is best on this prompt, because it preserves down hit around
    `70%` while not increasing upgate wall;
  - raising upgate to `75/80` is harmful: small upgate hit gains are outweighed
    by down cache loss, more IO bytes, and slower decode.
- Prompt-general check:
  - run root:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-upgate55-general-smoke`;
  - paired cold-start N32, `PROFILE=0`, same prompt/env except `UPGATE_PCT`.

| prompt | pct 62 tok/s | pct 55 tok/s | quality | decision |
| --- | ---: | ---: | --- | --- |
| `dev_japan_factual` | 1.89 | 1.89 | pass/pass | neutral |
| `dev_python_reverse` | 1.70 | 1.56 | pass/pass | worse |

- Decision:
  - do not promote `UPGATE_PCT=55` as a prompt-general default;
  - keep the current `UPGATE_PCT=62` for now;
  - avoid tuning global VRAM split from a single prompt.
- Next direction:
  - if revisiting cache split, learn a per-layer/per-role cache policy from a
    dev prompt set and validate on held-out prompts;
  - global pct alone is too coarse because it improves down-heavy prompts and
    hurts upgate-heavy/code prompts differently.

## Active goal: Kimi lower-byte GPU-extension path

Timestamp: 2026-07-11 CST.

Goal:

> Keep Kimi on the CPU/defer MoE scheduler, but make the GPU extension handle
> useful `gate/up/down` experts with fewer transferred bytes and less exposed
> `io_uring` wait. The near-term goal is a pushed, reproducible, prompt-general
> improvement toward stable `>2 tok/s` at N96. The product goal remains stable
> `>5 tok/s` for random user prompts on a 16 GB host-RAM machine with one
> 32 GB RTX 5090-class GPU.

Why this is the current goal:

- The DeepSeek gate-hotpool idea is useful for Kimi as an architecture pattern:
  CPU/defer remains the scheduler, while GPU becomes the expert-cache/compute
  extension.
- It should not be copied literally as a gate-only fix. Current Kimi profiling
  already shows `CPU fallback: 0`; the remaining bottleneck is exposed expert
  movement and staging wait, especially mixed `up/gate` and then `down`.
- RAM-only whole-layer/slab experiments are clean but too small or too costly
  under the 16 GB host-RAM cap. The stronger path is reducing bytes per useful
  expert while keeping demand loading and cache accounting explicit.
- Phase 5D bounds show that a selected lower-byte overlay can plausibly cross
  the `>2 tok/s` boundary only if the runtime can actually consume the smaller
  representation; metadata-only or payload-only tools are not SOTA.

Hard success gates:

- Cold start only, with cache state reset before accepted measurements.
- Host RAM peak below `15900000000` bytes, including page cache, pinned memory,
  mmap/file-backed pages, helper processes, and cgroup accounting.
- Paired TTFT no more than `1.20x` baseline.
- Mandatory semantic gate: `Please introduce France in a short paragraph.` must
  be coherent and correct.
- Optimization must be prompt-general. Dev prompts can guide design; held-out
  prompts are validation only and cannot be used for hotset, pack, threshold, or
  layer selection.
- Any accepted SOTA must be reproducible from a pushed commit. The commit body
  must include improvement size, exact env, commands, prompt split, run paths,
  quality output/summary, TTFT/RAM/VRAM/IO/H2D/fallback metrics, and rollback
  commit.

Execution plan:

1. Commit and push the current default-off Phase 5D tooling.
   - Include `.Agent/run-tools/kimi_iq1s_selected_payload_pack.py` and this plan
     update only.
   - Make no SOTA claim; the payload builder only proves selected IQ1_S/Q2_K
     bytes and source offsets can be materialized.

2. Add a default-off runtime reader for the tiny `GGMLMOEPACKv2` sidecar.
   - Parse the v2 index separately from the current main expert pack.
   - Allow lookup by tensor/expert/type instead of assuming current-pack nbytes.
   - Keep fallback to the existing IQ3_S path for every unsupported case.
   - Add counters for overlay hit/miss, overlay bytes, type mismatch, nbytes
     mismatch, and fallback reason.

3. Add guarded mixed-type dispatch only for the tiny smoke path.
   - Admit `IQ1_S`/`Q2_K` only behind an explicit env flag.
   - Start with N32 quality smoke using the 8-entry pack; do not build a large
     overlay until the tiny path proves correctness and accounting.
   - If kernels are missing or quality is unstable, stop and document rejection
     rather than expanding the pack.

4. Re-profile before any promotion.
   - Run paired cold-start N32, then N96 dev.
   - Attribute per-token time into expert read, pinned/pageable staging, H2D,
     up/gate compute, down compute, overlay dequant/compute, `io_uring_wait`,
     and residual CPU/defer overhead.
   - Continue only if the measured win reduces critical-path wait, not merely
     bytes or hit-rate counters.

5. Validate prompt-general behavior.
   - Promote to held-out only after dev passes quality, RAM, TTFT, and wait
     gates.
   - The final reported SOTA must be based on held-out general prompts, not a
     France-only prompt or prompt-specific pack.

6. If the lower-byte overlay path fails, fall back to scheduling/layout work.
   - Investigate exact-byte IO scheduling, pack layout for larger contiguous
     reads, and unified same-layer `up/gate/down` demand scheduling.
   - Keep RAM-tier work secondary unless it replaces low-value page cache with
     demonstrably batchable expert data and improves held-out wait.

## Current authoritative goal and plan: Kimi CPU/defer GPU-extension, prompt-general

Timestamp: 2026-07-11 CST.

This section is the current execution target. Historical sections below remain as
audit history, but any new optimization must follow this goal and protocol first.

Goal:

> On `vendor/kimi-deepseek-41d205-additive`, make the Kimi CPU/defer MoE path use
> the GPU as a more complete expert-residency, transfer, and compute extension for
> `gate/up/down`, while preserving prompt-general correctness under a strict 16 GB
> host-RAM cap. The immediate engineering target is a pushed, reproducible
> held-out improvement toward stable `>2 tok/s`; the product target remains stable
> `>5 tok/s` for random user prompts on one 32 GB RTX 5090-class GPU.

Hard gates:

- Cold start only. Drop caches before every accepted measurement.
- Total host RAM peak must stay below `15900000000` bytes, including page cache,
  pinned memory, mmap/file-backed pages, helpers, and cgroup accounting.
- Use VRAM aggressively, but not by increasing TTFT, host reclaim, or quality risk.
- Paired TTFT must stay within `1.20x` of the baseline.
- Mandatory quality gate: `Please introduce France in a short paragraph.` must be
  coherent and semantically correct.
- Optimization must be prompt-general. Dev prompts can guide design; held-out
  prompts are validation only and must not be used for hotset, pack, threshold, or
  layer selection.
- Every accepted SOTA must be reproducible from a pushed commit. The commit body
  must include improvement size, exact env, command, prompt split, run directory,
  quality result, TTFT/RAM/VRAM/IO metrics, and rollback commit.

Current measured state:

- Branch: `vendor/kimi-deepseek-41d205-additive`.
- Rollback before the current profiling-only change:
  `2d03059f3 prof: attribute Kimi CPU MoE decode residual`.
- Latest corrected N96 profile run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4x-mixed-upgate-profile-n96-france`.
- Quality: pass.
- RAM peak: `12809871360` bytes.
- TTFT: `11668.39 ms`.
- Decode: `56536.74 ms / 85 runs`, `1.50 tok/s` under profiling.
- CPU fallback: `0`; direct reads: `0`.
- Expert-pack IO: `85754` reads, `491342774272` bytes, `37155.156 ms`
  `io_uring_wait`.
- Corrected profile coverage: `up-gate-profile.csv` has `5101` data rows,
  matching `5101` CPU MoE up_gate decode calls and covering all 60 MoE layers.
- Corrected per-token attribution:
  - full up/gate CSV: `455.555 ms/token`;
  - CPU MoE upgate total: `464.191 ms/token`;
  - down CSV: `173.197 ms/token`;
  - CPU MoE down total: `177.815 ms/token`;
  - residual after CPU MoE: `23.132 ms/token`.
- Largest measured target: mixed `up=22/gate=18`, `260.608 ms/token`.
- New mixed-only analysis artifact:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4x-mixed-upgate-profile-n96-france-analysis-v2/report.md`.
- New `upgate_breakdown.csv` fields include per type/layer calls, ms/token,
  cache hit rates, miss count per call, stage jobs per call, wait per call, and
  compute per call.

Key interpretation:

- The DeepSeek gate-hotpool lesson transfers to Kimi as an architecture pattern,
  not as a literal gate-only fix.
- Kimi decode already has zero measured CPU fallback on the current path, so the
  next gain is not from merely moving CPU math to GPU.
- The real target is exposed wait inside the CPU/defer GPU-extension path,
  especially mixed-type up/gate transfer, staging, cache admission, and queue
  continuity.
- Down-only optimization cannot reach `>2 tok/s` by itself. To move from about
  `665 ms/token` to below `500 ms/token`, the first target must save roughly
  `165 ms/token`; only mixed up/gate is large enough as a single workstream.
- The corrected mixed up/gate detail shows the largest row is
  `up=22/gate=18/parallel_stage=1`:
  - `2465` calls, `260.608 ms/token`, `8.986 ms/call`;
  - up hit `0.476`, gate hit `0.476`;
  - up miss `4.190/call`, gate miss `4.190/call`;
  - wait `5.259 ms/call`, up compute `0.147 ms/call`, gate compute
    `0.099 ms/call`.
- Therefore the next optimization should target transfer/staging wait and cache
  placement for mixed up/gate, not CUDA math throughput.
- Highest mixed layer targets from the N96 France profile are:
  `blk.29`, `blk.14`, `blk.28`, `blk.54`, `blk.30`, `blk.33`,
  `blk.51`, `blk.31`, `blk.32`, and `blk.52`. These are dev-profile
  candidates only; held-out prompts must not be used to choose layers.

## Phase 5D goal: lower-byte expert residency to reach stable `>2 tok/s`

Timestamp: 2026-07-11 CST.

Goal:

> Move the next Kimi optimization round from "more host RAM residency" to
> "fewer bytes per useful expert" while keeping the current CPU/defer scheduler
> and GPU expert-cache extension. The immediate target is a reproducible
> prompt-general N96 improvement from the current `~1.5 tok/s` profile toward
> stable `>2 tok/s` under the 16 GB host-RAM cap. The strategic goal remains
> stable `>5 tok/s` on random prompts, but no candidate may be promoted unless
> it is cold-start, quality-safe, TTFT-safe, and pushed with full reproduction
> details.

Why this is now the main path:

- Current Kimi decode already has `CPU fallback: 0`, so the DeepSeek
  gate-hotpool lesson applies as a scheduling/cache architecture pattern, not as
  a literal CPU-fallback removal task.
- Phase 5B rejected route-trace future-layer prefetch as the next runtime
  change: `H=1/B=16` co-occurrence recall was only `0.3057`, and `B=32` would
  waste about `26.95 GiB/token`.
- Phase 5C showed that replacing page cache with batch-safe whole layer RAM
  slabs is structurally clean but too small: even `~10 GiB` of slab residency
  only covered about `19-25 ms/token`, while reaching `2 tok/s` needs roughly
  `165 ms/token` of saving from the current profile.
- The remaining large term is expert bytes and exposed transfer/staging wait:
  the corrected N96 profile moved `491342774272` expert-pack bytes and spent
  `37155.156 ms` in `io_uring_wait`, with up/gate at about
  `455.555 ms/token` and down at about `173.197 ms/token`.

Phase 5D execution plan:

1. Re-establish the reproducible baseline before changing runtime behavior.
   - Use the current pushed branch `vendor/kimi-deepseek-41d205-additive`.
   - Cold-start N32 smoke and N96 paired run.
   - Record prompt split, command, run directory, token rate, TTFT, host RAM
     peak, VRAM usage, page-cache split, expert-pack bytes, H2D bytes,
     `io_uring_wait`, CPU fallback count, direct read count, and exact output.
   - The France prompt remains a mandatory quality gate, but it must not be the
     only prompt used for promotion.

2. Compute a lower-byte upper bound before implementation.
   - Reuse dev-only route traces and tools such as
     `.Agent/run-tools/kimi_quant_scaled_cache_bound.py` and
     `.Agent/run-tools/kimi_byte_reduction_target_bound.py`.
   - Sweep byte ratios such as `1.0`, `0.827`, `0.696`, `0.563`, `0.505`,
     `0.4`, `0.3`, `0.25`, and `0.2`.
   - Estimate for each ratio:
     - effective VRAM expert-cache capacity;
     - prompt-general hit-rate change;
     - expert bytes per token;
     - required SSD bandwidth;
     - exposed wait lower bound;
     - projected token rate at N96.
   - Continue only if the bound can plausibly close at least the `~165 ms/token`
     gap to `2 tok/s` on dev prompts without using held-out prompts for tuning.

3. Select a role-specific lower-byte candidate.
   - Prioritize the role/layer groups that dominate exposed wait, especially
     mixed up/gate, before down-only work.
   - Consider role-specific byte ratios rather than one global compression level:
     gate/up may justify more aggressive compact representation if quality is
     stable; down may need a safer ratio if output quality is more sensitive.
   - Do not assume a format such as NVFP4 is usable until the code path,
     quantization source, dequant/compute kernel, and quality result are proven.
   - The first candidate should be default-off and limited to dev-selected
     profiles so it can be reverted without touching the stable SOTA path.

4. Implement only after the theoretical bound is strong enough.
   - Preserve CPU/defer as the owner of routing and execution order.
   - Keep GPU as the expert-cache/compute extension.
   - Keep existing expert-pack demand path for misses.
   - Add counters that separate:
     - compressed-cache hits;
     - normal VRAM hits;
     - RAM-tier hits, if any;
     - SSD/io_uring misses;
     - decompression/dequant time;
     - H2D bytes;
     - quality-risk fallback.
   - The implementation must show that lower bytes reduce critical-path wait,
     not merely report higher cache hit rate.

5. Validate promotion with prompt-general gates.
   - Dev prompts guide design.
   - Held-out prompts are used only after dev passes.
   - Required quality checks include `Please introduce France in a short
     paragraph.` and at least one non-France held-out prompt.
   - Promote only if:
     - N96 token rate improves on prompt-general validation;
     - minimum/median behavior does not collapse on new prompts;
     - TTFT ratio is `<= 1.20x`;
     - host RAM peak is `< 15900000000` bytes;
     - CPU fallback and direct reads stay at the intended values;
     - output remains coherent and semantically correct.

6. Commit/push discipline for every accepted result.
   - Update this plan before each implementation attempt.
   - If a candidate passes all gates, commit and push immediately.
   - The commit body must include:
     - before/after token rate and improvement size;
     - exact branch, commit, env, and command;
     - prompt split and run directories;
     - TTFT, RAM, VRAM, page-cache, IO, H2D, and fallback metrics;
     - exact quality output or quality summary;
     - rollback commit.
   - If performance, quality, RAM, or TTFT fails, revert the runtime change or
     leave it default-off and document the rejection. It must not be called SOTA.

Fallback if Phase 5D bound is weak:

- Do not implement a low-byte runtime path just to create a code change.
- Return to narrower transfer scheduling work:
  - exact-byte scheduler improvements;
  - up/gate+down unified demand scheduling;
  - pack layout that increases contiguous demand reads;
  - one small pageable RAM slab A/B only if it is clearly default-off and
    measured against TTFT/refault gates.

## Phase 5D result: byte reduction can reach `>2 tok/s`; split retune cannot

Timestamp: 2026-07-11 CST.

Runs:

- Quant-scaled cache bound:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-quant-scaled-cache-bound-dev7/report.md`
- Byte-ratio target bound:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-byte-reduction-target-dev7/target-2tps.md`
  and
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-byte-reduction-target-dev7/target-5tps.md`
- Split-pool sweep:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-quant-split-sweep-dev7/report.md`

Inputs:

- Dev-only route traces from
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834`.
- Seven dev prompts:
  `dev_france_regression`, `dev_japan_factual`, `dev_linear_equation`,
  `dev_mixed_summary`, `dev_photosynthesis_factual`, `dev_python_reverse`,
  and `dev_zh_france`.
- Held-out/test prompts were not used.
- The `farthest-next` oracle in `kimi_quant_scaled_cache_bound.py` was stopped
  because full dev7 replay was too slow for routine use. The promoted decision
  uses `global LFU`, which is the prompt-general static policy, not the oracle.

Commands:

```bash
.Agent/run-tools/kimi_quant_scaled_cache_bound.py \
  --runs-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834 \
  --ratios "1.0,0.827,0.696,0.563,0.505,0.4,0.3,0.25,0.2" \
  --out-json /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-quant-scaled-cache-bound-dev7/quant_scaled_cache_bound.json \
  --out-md /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-quant-scaled-cache-bound-dev7/report.md

.Agent/run-tools/kimi_byte_reduction_target_bound.py \
  --runs-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834 \
  --out /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-byte-reduction-target-dev7/target-2tps.md \
  --target-tps 2.0 \
  --peak-gib-s 10.4 \
  --evidence-scope dev7-route-trace-n32

.Agent/run-tools/kimi_byte_reduction_target_bound.py \
  --runs-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834 \
  --out /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-byte-reduction-target-dev7/target-5tps.md \
  --target-tps 5.0 \
  --peak-gib-s 10.4 \
  --evidence-scope dev7-route-trace-n32

.Agent/run-tools/kimi_quant_split_sweep_bound.py \
  --runs-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834 \
  --ratios "1.0,0.696,0.563,0.505,0.4,0.3,0.25,0.2" \
  --pcts "40,45,50,55,60,62,65,70,75,80,85,90" \
  --current-pct 62 \
  --out-json /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-quant-split-sweep-dev7/split_sweep.json \
  --out-md /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-quant-split-sweep-dev7/report.md
```

Quant-scaled global-LFU bound:

| byte ratio | global LFU hit | global LFU miss GiB/token | projected ms/token at 10.4 GiB/s plus 94.1 ms floor | projected tok/s |
|---:|---:|---:|---:|---:|
| 1.000 | 28.4% | 8.644 | 925.2 | 1.08 |
| 0.827 | 31.8% | 6.815 | 749.4 | 1.33 |
| 0.696 | 35.1% | 5.461 | 619.2 | 1.62 |
| 0.563 | 39.3% | 4.130 | 491.2 | 2.04 |
| 0.505 | 41.6% | 3.565 | 436.9 | 2.29 |
| 0.400 | 46.8% | 2.572 | 341.4 | 2.93 |
| 0.300 | 53.8% | 1.675 | 255.2 | 3.92 |
| 0.250 | 58.6% | 1.251 | 214.4 | 4.66 |
| 0.200 | 64.8% | 0.850 | 175.8 | 5.69 |

Target-bound result:

- For `2 tok/s`, median moved bytes are `6.70 GiB/token`, mean
  `7.18 GiB/token`.
- The median all-hit MoE floor estimate is `98.6 ms/token`, mean
  `94.1 ms/token`.
- The median required byte ratio at `10.4 GiB/s` after reserving floor time is
  `0.60x`; worst dev prompt is `0.47x`.
- Miss-byte share by role:
  - down: `464.15 GiB`, `43.7%`;
  - gate: `304.96 GiB`, `28.7%`;
  - up: `291.85 GiB`, `27.5%`.
- For `5 tok/s`, the median required byte ratio after floor is `0.16x` and
  worst dev prompt is `0.03x`. Therefore `5 tok/s` is not a simple byte-ratio
  problem; it also requires lowering the all-hit MoE floor and/or changing the
  compute/storage form.

Split-pool result:

- Total simulated VRAM cache capacity: `14.645 GiB`.
- Current split: `62%` upgate / `38%` down.
- Best split across `40-90%` upgate is effectively `60%` for the new dev7
  traces, but relative gain is tiny:
  - ratio `1.000`: `0.01%`;
  - ratio `0.563`: `0.01%`;
  - ratio `0.505`: `0.04%`;
  - ratio `0.200`: `0.11%`.
- Decision: do not spend runtime work on changing the upgate/down VRAM split
  first. The current split is already near the route-trace optimum.

Interpretation:

- Phase 5D passes the theoretical-benefit gate for the `>2 tok/s` target:
  a prompt-general byte ratio around `0.56x`, with the effective larger VRAM
  cache from smaller entries, is enough to project just above `2 tok/s`.
- A safer target is `0.50x` or lower because the bound assumes `10.4 GiB/s`
  sustained movement and an optimistic all-hit MoE floor.
- If only up/gate are compressed and down remains at current bytes, up/gate
  must be roughly `0.29x` to reach a global `0.60x` movement ratio:
  `(0.60 - 0.437) / (0.287 + 0.275) ~= 0.29`.
- Directly re-encoding the already-quantized current experts is not a good
  first candidate based on the older GP11 screen:
  - `~0.5x` 1-bit blockwise candidates had rel-L2 around `1.8`;
  - 2-bit candidates were closer but ratio was usually `0.7x-0.9x`, which is
    not enough for the `2 tok/s` bound on the current traces;
  - 3/4-bit candidates often expand versus the current IQ2/IQ3 entries.
- Therefore the next practical implementation path should not be "simple
  requantize current dequantized weights". It should evaluate one of:
  - an external lower-quant expert source, built into an expert-only overlay
    pack while preserving current dense/attention weights;
  - a lower-byte auxiliary expert representation with activation/quality
    validation before runtime promotion;
  - a structural residual/shared-base form only if offline activation tests
    prove much lower error than naive blockwise re-encoding.

Next Phase 5D implementation target:

1. Build a default-off lower-byte expert overlay feasibility test.
   - Candidate ratio gate for `>2 tok/s`: `<=0.505x` global bytes, or
     `<=0.563x` only if real runtime sustains the IO peak and quality is clean.
   - Prefer whole `gate/up/down` overlay first, because role-only upgate needs
     an aggressive `~0.29x` ratio to compensate for full-byte down.
   - Keep the existing current-IQ3 model and expert-pack path as rollback.

2. Validate quality before optimizing transfer.
   - First run a small offline or short-runtime quality smoke with the lower-byte
     expert representation.
   - Mandatory prompt: `Please introduce France in a short paragraph.`
   - Include at least one non-France dev prompt before any N96 profiling.
   - Reject immediately if output becomes incoherent or semantically wrong.

3. Only then implement runtime integration.
   - Add default-off expert-pack selection for lower-byte overlay entries.
   - Report lower-byte hits, normal hits, SSD misses, dequant/convert time, H2D
     bytes, and output quality separately.
   - Promote only if N96 prompt-general validation exceeds the current stable
     SOTA, TTFT remains `<=1.20x`, host RAM remains below
     `15900000000` bytes, and the commit body contains full reproduction
     details.

## Phase 5D overlay screen: selected lower-byte packs are not plug-compatible yet

Timestamp: 2026-07-11 CST.

Additional runs:

- IQ2_XXS selected overlay bound:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq2xxs-selected-overlay-bound-dev7/report.md`
- IQ1_S selected hotset bound:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-hotset-bound-dev7/report.md`

Asset and disk state:

- Current free disk on `/root/lfz`: about `133 GiB`.
- Current local model assets include the IQ3_S GGUF shards and the 0.6B draft
  model; no complete lower-quant Kimi model is present.
- Historical candidate refresh showed the closest complete original-model lower
  quant, mradermacher `i1-IQ1_S`, is about `190.391 GiB` and failed the disk
  safety gate even when more space was available. It is not safe to download as
  a complete model now while preserving rollback assets.

IQ2_XXS selected overlay result:

- Source metadata:
  `.Agent/runs/20260707-gp21-remote-range-pack/plan.tsv`.
- Source is AesSedai `IQ2_XXS` expert ranges, not `i1-IQ1_S`.
- Dev7 route-profile inputs only; held-out/test prompts were not used.

| overlay budget GiB | selected entries | estimated pack GiB | hybrid byte ratio |
|---:|---:|---:|---:|
| 8 | 2580 | 7.998 | 0.9131 |
| 16 | 5082 | 15.999 | 0.8733 |
| 32 | 10083 | 31.999 | 0.8225 |
| 64 | 19636 | 63.999 | 0.7617 |
| 96 | 28568 | 95.999 | 0.7257 |
| 120 | 35063 | 119.998 | 0.7089 |

Decision:

- Reject IQ2_XXS selected overlay as the next main path. Even a `120 GiB`
  overlay only reaches `0.7089x`, far above the `<=0.563x` boundary target and
  the safer `<=0.505x` target for `>2 tok/s`.

IQ1_S selected hotset result:

- Tool:
  `/root/lfz/llama.cpp-vendor-kimi-gp169-clean/.Agent/run-tools/kimi_iq1s_budgeted_hotset_bound.py`.
- Source asset:
  `mradermacher/Kimi-K2.7-Code-i1-GGUF`
  `Kimi-K2.7-Code.i1-IQ1_S.gguf`.
- This is a theoretical byte bound only. It is not directly runnable with the
  current IQ3_S main GGUF.

| overlay budget GiB | selected entries | estimated pack GiB | hybrid byte ratio | current runtime nbytes compatible |
|---:|---:|---:|---:|---|
| 8 | 2992 | 7.999 | 0.8458 | false |
| 16 | 5984 | 15.998 | 0.7774 | false |
| 32 | 11962 | 31.998 | 0.6923 | false |
| 64 | 23895 | 63.999 | 0.5951 | false |
| 96 | 35777 | 95.998 | 0.5430 | false |
| 120 | 44642 | 119.998 | 0.5223 | false |
| all dev candidate keys | 55515 | 149.898 | 0.5108 | false |

Interpretation:

- IQ1_S selected overlay is the first selected-overlay bound that gets close to
  the Phase 5D `>2 tok/s` byte target:
  - `96 GiB` reaches `0.5430x`, just under the optimistic `0.563x` boundary;
  - `120 GiB` reaches `0.5223x`, closer to the safer `0.505x` target;
  - all dev candidate keys reach `0.5108x` but need about `149.9 GiB`, which is
    outside the current disk envelope.
- This still does not prove a runtime path:
  - output quality for mixed IQ3_S/IQ1_S experts is unknown;
  - current code does not treat selected IQ1_S entries as plug-compatible with
    IQ3_S expert tensors.

Current runtime compatibility assessment:

- The normal one-pack path keys lookup by `(tensor, expert, nbytes)`:
  `one_pack_lookup(src0_name, expert_index, src0_bytes)`.
- The read path also requires `entry->nbytes == sz`.
- The compute path receives `src0_type_int` from the main GGUF tensor, not from
  the pack entry.
- `ggml_cuda_moe_stream_supports_type()` currently admits `IQ3_XXS`, `IQ3_S`,
  `IQ2_S`, `MXFP4`, and `F8_E4M3_B128`; `IQ1_S` and `Q2_K` are not admitted by
  that fast-path gate.
- Therefore a lower-byte selected IQ1_S overlay needs a default-off mixed-type
  expert override, not just a different pack file.

Next implementation plan after this screen:

1. Add a metadata-only mixed-type override preflight before runtime math.
   - Manifest fields must include tensor, expert, runtime type, runtime nbytes,
     original type, original nbytes, `ne00`, `ne01`, `nb01`, source offset, and
     source pack/path.
   - Lookup must be default-off and must not replace the stable IQ3_S pack path.
   - The preflight must reject shape/type mismatches and report exactly which
     entries would be eligible.

2. Add a tiny runtime smoke only after preflight passes.
   - Start with a very small IQ1_S selected overlay, not a `96-120 GiB` pack.
   - Add counters for mixed-type hits, rejected overrides, bytes saved, H2D
     bytes, kernel type, and fallback-to-normal-pack events.
   - Run N32 short quality prompts first; do not claim token-rate SOTA.

3. Promote to a larger overlay only if quality survives.
   - The first performance-relevant pack would need to be around `96-120 GiB`
     to have a chance at `>2 tok/s` based on the current bound.
   - This is near the current disk limit and must preserve rollback assets.
   - Any accepted result must still satisfy cold start, 16 GB host RAM, TTFT
     `<=1.20x`, prompt-general validation, and the full reproducibility commit
     protocol.

## Phase 5D preflight result: metadata passes, runtime override is still blocked

Timestamp: 2026-07-11 CST.

New tool:

- `.Agent/run-tools/kimi_iq1s_mixed_type_preflight.py`

Purpose:

- Validate selected IQ1_S/Q2_K overlay metadata against the current IQ3_S expert
  inventory before any C++ runtime change.
- Produce a manifest with tensor, expert, layer, role, current type/nbytes,
  current offsets, override type/nbytes, expected override row stride, and
  explicit blockers.
- This is metadata-only. It does not download weights, build a pack, run a
  model, or claim SOTA.

Runs:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-mixed-type-preflight-budget8/report.md
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-mixed-type-preflight-budget96/report.md
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-mixed-type-preflight-budget120/report.md
```

Commands:

```bash
.Agent/run-tools/kimi_iq1s_mixed_type_preflight.py \
  --inventory-tsv .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
  --selected-plan-tsv /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-hotset-bound-dev7/budget-8p0-selected-plan.tsv \
  --out-dir /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-mixed-type-preflight-budget8

.Agent/run-tools/kimi_iq1s_mixed_type_preflight.py \
  --inventory-tsv .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
  --selected-plan-tsv /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-hotset-bound-dev7/budget-96p0-selected-plan.tsv \
  --out-dir /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-mixed-type-preflight-budget96

.Agent/run-tools/kimi_iq1s_mixed_type_preflight.py \
  --inventory-tsv .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
  --selected-plan-tsv /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-hotset-bound-dev7/budget-120p0-selected-plan.tsv \
  --out-dir /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-mixed-type-preflight-budget120
```

Results:

| selected plan | entries | metadata OK | runtime-ready now | source-offset entries | selected-entry byte ratio |
|---|---:|---:|---:|---:|---:|
| `budget-8p0` | 2992 | 2992 | 0 | 0 | 0.4564 |
| `budget-96p0` | 35777 | 35777 | 0 | 0 | 0.4955 |
| `budget-120p0` | 44642 | 44642 | 0 | 0 | 0.5021 |

For the `budget-120p0` manifest:

- `IQ1_S`: `44210` entries, current `236.433 GiB`, override `118.053 GiB`,
  selected-entry ratio `0.4993`.
- `Q2_K`: `432` entries, current `2.538 GiB`, override `1.938 GiB`,
  selected-entry ratio `0.7636`.

Important distinction:

- The preflight selected-entry byte ratio is the ratio among entries actually
  selected for overlay.
- The earlier selected-hotset hybrid byte ratio includes unselected traffic that
  remains on current IQ3_S bytes. For example, the `120 GiB` plan is
  `0.5223x` hybrid traffic but `0.5021x` among selected entries.

Blockers found on every selected entry:

1. `missing_source_offset`
   - The selected-plan TSV has tensor/expert/type/nbytes but no source offset.
   - Runtime cannot read the override payload until a range builder adds source
     offsets or creates a sidecar payload pack.

2. `fast_path_type_not_admitted_now`
   - Current Kimi one-stream fast path does not admit `IQ1_S` or `Q2_K` through
     `ggml_cuda_moe_stream_supports_type()`.
   - CUDA kernels may have lower-level support, but this Kimi path currently
     gates them out.

3. `runtime_nbytes_differs_from_current`
   - The normal one-pack path is keyed by current `(tensor, expert, nbytes)`.
   - IQ1_S/Q2_K override entries have different bytes from the current IQ3_S
     expert tensors and therefore cannot be a drop-in replacement.

Decision:

- The metadata shape/type/bytes gate passes for all tested selected entries.
- Runtime integration is still blocked by source-offset/payload absence and
  mixed-type override support.
- The next code change should be a default-off source-offset/payload manifest
  builder for a tiny selected subset, not a large pack or a direct runtime type
  override.
- Only after a tiny payload exists should the runtime path add mixed-type lookup
  and `IQ1_S`/`Q2_K` dispatch guards for an N32 quality smoke.

## Phase 5D payload result: tiny selected IQ1_S/Q2_K payload pack builds

Timestamp: 2026-07-11 CST.

New tool:

- `.Agent/run-tools/kimi_iq1s_selected_payload_pack.py`

Purpose:

- Clear the `missing_source_offset` blocker for a tiny selected subset.
- Read a selected-plan TSV, parse the mradermacher multipart IQ1_S GGUF header,
  compute source absolute offsets, and optionally range-download payloads into
  a `GGMLMOEPACKv2` sidecar.
- This remains default-off and is not used by current runtime.

Runs:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-metadata8/report.md
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/report.md
```

Commands:

```bash
.Agent/run-tools/kimi_iq1s_selected_payload_pack.py \
  --selected-plan-tsv /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-hotset-bound-dev7/budget-8p0-selected-plan.tsv \
  --out-dir /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-metadata8 \
  --max-entries 8 \
  --include-types IQ1_S,Q2_K \
  --metadata-only

.Agent/run-tools/kimi_iq1s_selected_payload_pack.py \
  --selected-plan-tsv /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-hotset-bound-dev7/budget-8p0-selected-plan.tsv \
  --out-dir /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8 \
  --max-entries 8 \
  --include-types IQ1_S,Q2_K
```

Result:

- Metadata-only run:
  - selected entries: `8`;
  - payload bytes represented: `26836992`;
  - GGUF data start: `6984096`;
  - source offsets successfully generated.
- Payload run:
  - selected entries: `8`;
  - payload bytes: `26836992`;
  - pack bytes: `26841088`;
  - output pack:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack`;
  - manifest:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv`.
- Pack structure validation passed:
  - magic: `GGMLMOEPACKv2`;
  - version: `2`;
  - entries: `8`;
  - data start: `4096`;
  - manifest rows match pack index rows;
  - all payload ranges fit inside the pack file.

Selected entries:

| tensor | expert | type | nbytes | source offset | pack offset |
|---|---:|---|---:|---:|---:|
| `blk.1.ffn_down_exps.weight` | 23 | `Q2_K` | 4816896 | 1455543200 | 4096 |
| `blk.1.ffn_down_exps.weight` | 116 | `Q2_K` | 4816896 | 1903514528 | 4820992 |
| `blk.1.ffn_gate_exps.weight` | 23 | `IQ1_S` | 2867200 | 3265205152 | 9637888 |
| `blk.1.ffn_gate_exps.weight` | 116 | `IQ1_S` | 2867200 | 3531854752 | 12505088 |
| `blk.1.ffn_gate_exps.weight` | 197 | `IQ1_S` | 2867200 | 3764097952 | 15372288 |
| `blk.1.ffn_up_exps.weight` | 23 | `IQ1_S` | 2867200 | 4380115872 | 18239488 |
| `blk.1.ffn_up_exps.weight` | 116 | `IQ1_S` | 2867200 | 4646765472 | 21106688 |
| `blk.1.ffn_up_exps.weight` | 197 | `IQ1_S` | 2867200 | 4879008672 | 23973888 |

Decision:

- The source-offset/payload blocker is cleared for a tiny selected subset.
- This still does not make the overlay runnable:
  - current runtime does not parse `GGMLMOEPACKv2` in the one-pack path;
  - current lookup expects current expert nbytes;
  - current Kimi one-stream type gate rejects `IQ1_S`/`Q2_K`.
- The next runtime step, if pursued, must be a default-off reader/lookup path for
  this v2 sidecar plus explicit mixed-type dispatch guards. It must start with
  this tiny pack and N32 quality smoke, not a large pack.

## Phase 5D runtime reader smoke: v2 sidecar parse/read is verified

Timestamp: 2026-07-11 CST.

New tool:

- `.Agent/run-tools/kimi_moepack_v2_runtime_smoke.cpp`

Purpose:

- Validate the already-built CUDA batch runtime's `GGMLMOEPACKv2` debug reader
  without loading a model.
- The tool uses `dlopen`/`dlsym` against `build-cuda-batch/bin/libggml-cuda.so`
  and calls the real exported symbols:
  - `ggml_cuda_moe_expert_pack_v2_lookup_debug`;
  - `ggml_cuda_moe_expert_pack_v2_read_debug`.
- This remains default-off and does not change inference behavior.

Run directory:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-v2-runtime-smoke
```

Commands:

```bash
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-v2-runtime-smoke
mkdir -p "$RUN"

.Agent/run-tools/kimi_moepack_v2_synthetic_test.py --out-dir "$RUN"

g++ -std=c++17 -O2 \
  .Agent/run-tools/kimi_moepack_v2_runtime_smoke.cpp \
  -ldl \
  -o "$RUN/kimi_moepack_v2_runtime_smoke"

LD_LIBRARY_PATH=build-cuda-batch/bin \
  "$RUN/kimi_moepack_v2_runtime_smoke" \
  build-cuda-batch/bin/libggml-cuda.so \
  "$RUN/synthetic-v2.expert-pack"

LD_LIBRARY_PATH=build-cuda-batch/bin \
  "$RUN/kimi_moepack_v2_runtime_smoke" \
  build-cuda-batch/bin/libggml-cuda.so \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv
```

Results:

- Synthetic v2 pack:
  - generated `2` entries;
  - runtime reader loaded `2` metadata entries;
  - lookup/read passed;
  - synthetic payload byte-pattern verification passed.
- Tiny selected IQ1_S/Q2_K overlay pack:
  - runtime reader loaded `8` metadata entries;
  - manifest mode checked all `8` rows;
  - lookup metadata matched `packed_type`, `packed_nbytes`, `packed_ne00`,
    `packed_ne01`, and `packed_nb01`;
  - `read_debug` returned the expected byte count for every entry.

Decision:

- The `GGMLMOEPACKv2` parse/read blocker is cleared for the current
  `build-cuda-batch/bin/libggml-cuda.so` and the 8-entry selected overlay pack.
- This is still not a runnable lower-byte inference path:
  - the current batch v2 reader is debug/shadow only;
  - one-pack compute/cache lookup still uses current `(tensor, expert, nbytes)`;
  - the Kimi one-stream type gate still rejects `IQ1_S`/`Q2_K`;
  - no quality or token-rate run has been performed with the overlay.
- The next implementation step should be a guarded runtime bridge from v2
  lookup/read to a tiny smoke compute path, or an explicit rejection if
  `IQ1_S`/`Q2_K` kernels cannot support the selected roles.

## Phase 5D compute smoke: selected IQ1_S/Q2_K entries run through CUDA MMVQ

Timestamp: 2026-07-11 CST.

New tool:

- `.Agent/run-tools/kimi_moepack_v2_mmvq_smoke.cpp`

Purpose:

- Validate that real selected lower-byte `GGMLMOEPACKv2` payloads can be copied
  to GPU and consumed by the exported CUDA MoE MMVQ function:
  `ggml_cuda_moe_stream_mmvq_dev`.
- This uses the same 8-entry tiny selected pack as the reader smoke and does not
  load a model.
- This remains default-off and does not change inference behavior.

Run directory:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-v2-mmvq-smoke
```

Command:

```bash
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-v2-mmvq-smoke
mkdir -p "$RUN"

g++ -std=c++17 -O2 \
  -I/usr/local/cuda/include \
  .Agent/run-tools/kimi_moepack_v2_mmvq_smoke.cpp \
  -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart -ldl \
  -o "$RUN/kimi_moepack_v2_mmvq_smoke"

LD_LIBRARY_PATH=build-cuda-batch/bin:/usr/local/cuda/targets/x86_64-linux/lib \
  "$RUN/kimi_moepack_v2_mmvq_smoke" \
  build-cuda-batch/bin/libggml-cuda.so \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv
```

Result:

- Runtime detected one CUDA device:
  `NVIDIA GeForce RTX 5090`, compute capability `12.0`, VRAM `32109 MiB`.
- All 8 selected entries passed `lookup -> read_debug -> H2D -> mmvq -> D2H`.
- Each entry returned finite, nonzero output:
  - `blk.1.ffn_down_exps.weight` expert `23`, `Q2_K`, `4816896` bytes;
  - `blk.1.ffn_down_exps.weight` expert `116`, `Q2_K`, `4816896` bytes;
  - `blk.1.ffn_gate_exps.weight` expert `23`, `IQ1_S`, `2867200` bytes;
  - `blk.1.ffn_gate_exps.weight` expert `116`, `IQ1_S`, `2867200` bytes;
  - `blk.1.ffn_gate_exps.weight` expert `197`, `IQ1_S`, `2867200` bytes;
  - `blk.1.ffn_up_exps.weight` expert `23`, `IQ1_S`, `2867200` bytes;
  - `blk.1.ffn_up_exps.weight` expert `116`, `IQ1_S`, `2867200` bytes;
  - `blk.1.ffn_up_exps.weight` expert `197`, `IQ1_S`, `2867200` bytes.

Decision:

- The selected `IQ1_S` and `Q2_K` payloads are not merely readable; they can run
  through the exported CUDA MMVQ path on this machine.
- This clears the kernel-feasibility blocker for a tiny selected subset.
- This still does not prove model quality or token-rate improvement:
  - the smoke uses synthetic activations, not routed Kimi hidden states;
  - the model runtime still does not dispatch v2 packed entries in place of the
    current IQ3_S/Q4/IQ2 tensors;
  - the current CPU/defer type gates still do not treat v2 packed type as the
    effective compute type.
- Next implementation step:
  - add a default-off tiny v2 overlay bridge that, for explicitly allowed
    `(tensor, expert)` keys only, reads packed bytes, uses the packed type/shape
    as the effective MMVQ input, and falls back to the current path otherwise;
  - first validation must be N32 quality smoke, not N96/SOTA.

## Phase 5B goal: transfer the CPU/defer GPU-extension idea to Kimi safely

Timestamp: 2026-07-11 CST.

Goal:

> Verify whether the DeepSeek-style `CPU/defer scheduler + GPU expert-cache
> extension` can give Kimi a prompt-general speedup by reducing exposed expert
> transfer wait, not by chasing prompt-specific hotsets. The immediate target is
> a reproducible, pushed improvement toward stable `>2 tok/s` on N96 under the
> 16 GB host-RAM cap; the strategic target remains stable `>5 tok/s` for random
> user prompts.

Why this is relevant to Kimi:

- DeepSeek gained largely because work that nominally lived in the CPU/defer MoE
  path was intercepted by a GPU-resident expert cache and GPU compute extension.
- Kimi already has `CPU fallback: 0` on the current measured path, so a literal
  "move CPU fallback to GPU" port is not the next high-yield move.
- The transferable part is the control pattern: keep CPU/defer as the scheduler,
  but expose more future expert work to the GPU/IO subsystem early enough that
  the SSD, pinned staging, H2D, and CUDA streams stay busy.
- Current corrected profile shows the largest exposed cost is mixed up/gate
  transfer/staging wait:
  - current N96 profile is about `665 ms/token` under profiling;
  - reaching `2 tok/s` requires below `500 ms/token`, so the next accepted
    optimization must save about `165 ms/token`;
  - mixed up/gate alone accounts for `~455.6 ms/token`, with the largest
    `up=22/gate=18` row at `260.6 ms/token`;
  - future-layer prefetch bound from Phase 5A suggests optimistic savings of
    `~164 ms/token` at `K=2` and `~227 ms/token` at `K=3`, which is the first
    direction large enough to plausibly cross `2 tok/s`.

Phase 5B execution plan:

1. Build a prompt-general future-expert predictability study before runtime
   changes.
   - Inputs: dev-only `route-trace.csv` files from existing dev runs.
   - Do not use held-out/test prompts for predictor design, hotset construction,
     pack layout, thresholds, or layer selection.
   - Measure whether active experts at layer `L` predict future active experts
     at `L+1`, `L+2`, and `L+3`.
   - Report recall at fixed budgets such as 8/16/32 predicted experts per target
     layer, extra bytes, useful prefetched bytes, false-prefetch bytes, and the
     theoretical `io_uring_wait` saving upper bound.
   - Use leave-one-dev-prompt-out where enough traces exist; otherwise label the
     result as format/proof-of-method only.

2. Decide whether future-layer prefetch is worth implementing.
   - Continue only if dev traces show enough prompt-general recall to cover a
     meaningful fraction of the `~165 ms/token` gap to `2 tok/s`.
   - Reject if recall comes from one prompt, one handpicked layer, or a
     France-specific hotset pattern.
   - If predictor quality is weak, stop runtime work and move to explicit
     RAM/VRAM layout optimization instead.

3. If predictor quality is sufficient, implement default-off runtime prefetch.
   - CPU/defer remains the owner of routing and execution order.
   - When layer `L` routing finishes, enqueue predicted misses for future layers
     into a low-priority prefetch queue that never blocks current-layer demand
     loads.
   - Current-layer demand IO always has priority over speculative/future IO.
   - Prefetched experts may enter VRAM only if they pass role/layer admission
     and do not evict higher-value current hot experts.
   - Add counters for predicted jobs, useful hits, wasted reads, evictions,
     cancelled jobs, prefetch wait hidden, and demand wait regression.

4. Validate with the established gate sequence.
   - N32 dev smoke first, cold start.
   - N96 dev paired run next, cold start.
   - Held-out validation only after dev passes.
   - Required metrics: token rate, TTFT ratio, host RAM peak, file/page-cache
     split, VRAM cache hit/miss, `io_uring_wait`, read bytes, H2D bytes,
     current-demand wait, prefetch waste, output text, and quality result.

5. Promotion/rejection rule.
   - Promote only if N96 dev and held-out both improve mean/median token rate,
     do not regress minimum token rate materially, keep TTFT within `1.20x`,
     keep host RAM below `15900000000` bytes, and pass quality.
   - If any hard gate fails, keep the code default-off or revert it; document the
     rejection and do not call it SOTA.
   - Every accepted SOTA commit must include exact env, command, prompt split,
     run directories, before/after metrics, quality output, and rollback commit.

## Phase 5B result: future-layer prefetch is not the next runtime change

Timestamp: 2026-07-11 CST.

New tool:

- `.Agent/run-tools/kimi_future_expert_predictability.py`

Run:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5b-future-expert-predictability-dev7
```

Command:

```text
.Agent/run-tools/kimi_future_expert_predictability.py \
  --input-glob "/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834/*/route-trace.csv" \
  --out /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5b-future-expert-predictability-dev7 \
  --horizons 1 2 3 \
  --budgets 8 16 32 \
  --io-wait-ms-per-token 374.779
```

Inputs:

- Seven dev-only route traces:
  `dev_france_regression`, `dev_japan_factual`, `dev_linear_equation`,
  `dev_mixed_summary`, `dev_photosynthesis_factual`, `dev_python_reverse`,
  and `dev_zh_france`.
- Held-out/test prompts were not used.
- Parser result: each dev trace produced `31` decode tokens, one prompt-like
  group, and zero incomplete groups.

Method:

- Leave-one-dev-prompt-out.
- Predictors:
  - `global_hot`: target-layer hot experts learned from the other dev prompts;
  - `cooc`: current layer active experts vote for future layer active experts,
    with global-hot fallback.
- Horizons: `L+1`, `L+2`, `L+3`.
- Budgets: `8`, `16`, `32` predicted experts per target layer.
- Byte estimates include `all`, `upgate`, and `down` bundles; the headline below
  uses all roles because runtime future-prefetch must ultimately make the
  missing future expert usable, not just predict its id.

Key results:

| horizon | budget | predictor | recall | precision | pred GiB/token | useful GiB/token | waste GiB/token |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 8 | `cooc` | 0.2198 | 0.2198 | 7.4973 | 1.6449 | 5.8524 |
| 1 | 8 | `global_hot` | 0.1417 | 0.1417 | 7.4973 | 1.0614 | 6.4359 |
| 1 | 16 | `cooc` | 0.3057 | 0.1529 | 14.9946 | 2.2899 | 12.7047 |
| 1 | 16 | `global_hot` | 0.2093 | 0.1046 | 14.9946 | 1.5673 | 13.4273 |
| 1 | 32 | `cooc` | 0.4056 | 0.1014 | 29.9893 | 3.0388 | 26.9505 |
| 1 | 32 | `global_hot` | 0.2989 | 0.0747 | 29.9893 | 2.2386 | 27.7506 |

Layer distribution:

- Best `H=1/B=16/cooc` target layers are only moderate:
  - `blk.13`: recall `0.4240`;
  - `blk.21`: recall `0.4188`;
  - `blk.23`: recall `0.4130`.
- Late layers are weak:
  - `blk.54`: recall `0.1740`;
  - `blk.58`: recall `0.1809`;
  - `blk.57`: recall `0.1895`.

Interpretation:

- The co-occurrence predictor is better than global-hot, so there is real routing
  structure.
- The absolute recall is too low for a demand-safe runtime prefetch. At `B=16`,
  the optimistic linear wait coverage is only about `114.6 ms/token`, below the
  roughly `165 ms/token` needed to reach `2 tok/s` from the current profile.
- At `B=32`, recall reaches `0.4056`, but all-role prefetch volume is about
  `30.0 GiB/token` with about `27.0 GiB/token` waste. This would likely create
  more IO/H2D pressure than it hides.
- No layer band has strong enough recall to justify hardcoded future-layer
  runtime prefetch without a better predictor.

Decision:

- Do not implement runtime future-layer prefetch from route-trace co-occurrence
  now.
- Keep the tool for future studies. Revisit only if a stronger predictor is
  available, such as router logits, token-history features, previous-token
  same-layer transitions, or a cheap draft model with measured acceptance.
- Continue the next optimization round with explicit RAM/VRAM layout and
  page-cache replacement, because that targets known low-value host memory
  instead of adding speculative IO.

## Phase 5C plan: explicit RAM/VRAM layout before more speculation

Timestamp: 2026-07-11 CST.

Goal:

> Replace low-value decode-time host page cache with controlled expert residency
> that can feed GPU demand loads efficiently, while preserving the current zero
> CPU fallback path, cold-start requirement, 16 GB host-RAM gate, prompt-general
> behavior, and reproducible SOTA protocol.

Design constraints:

- Do not rely on Linux page cache as an implicit expert cache.
- Do not pin multi-GiB slabs by default; test pageable RAM first unless profiling
  proves pinning wins.
- RAM-resident experts must be organized so demand loads remain batchable:
  prefer whole layer/role slabs or pack-contiguous ranges over scattered hot
  singletons.
- Demand IO always has priority over background RAM fill or speculative movement.
- Any RAM tier must report:
  - bytes resident by layer/role;
  - SSD reads avoided;
  - RAM-to-VRAM H2D bytes;
  - demand wait saved;
  - reclaim/refault and page-cache displacement;
  - TTFT ratio and host RAM peak.

Execution plan:

1. Measure what decode-time host RAM is low-value.
   - Use current SOTA cold-start N32/N96 dev runs.
   - Record cgroup memory, `active_file`, `inactive_file`, `pgmajfault`,
     `workingset_refault_file`, per-file residency, expert-pack counters,
     and fallback counters after prompt, after page-cache drop, mid-decode, and
     after decode.
   - Classify file-backed pages as dense/non-expert, GGUF expert residue,
     prompt-only residue, or refaulted demand pages.

2. Build an offline RAM-slab selector from dev-only IO/wait profiles.
   - Inputs: `io-batch-profile.csv`, `io-wait-trace.csv`,
     `up-gate-profile.csv`, `down-profile.csv`, and route traces.
   - Candidate units:
     - whole layer `up+gate`;
     - whole layer `down`;
     - pack-contiguous adjacent expert ranges;
     - mixed role/layer slabs only if they preserve large transfer batches.
   - Score candidates by exposed wait saved per GiB, prompt-general recurrence,
     and expected batching quality.
   - Exclude held-out/test prompts from selection.

3. Implement default-off RAM tier candidate.
   - Prefer a pageable RAM slab first.
   - Use explicit admission and eviction accounting, not ambient page cache.
   - Keep current SSD/io_uring demand path unchanged for misses.
   - Add instrumentation to prove whether RAM hits reduce demand wait or merely
     fragment SSD batches.

4. Validate in gates.
   - N32 dev cold-start smoke.
   - N96 dev paired cold-start.
   - Held-out validation only after dev passes.
   - Promote only if token rate improves, TTFT stays within `1.20x`, host RAM is
     below `15900000000` bytes, quality passes, and results are reproducible from
     a pushed commit.

## Phase 5C baseline result: normal decode does not materially use page cache

Timestamp: 2026-07-11 CST.

New tool:

- `.Agent/run-tools/kimi_phase_ram_cache_report.py`

Run:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5c-ram-cache-baseline-report
```

Command:

```text
.Agent/run-tools/kimi_phase_ram_cache_report.py \
  --run /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4z-queue-evidence-n32-photosynthesis/dev_photosynthesis_factual \
  --run /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4t-dev-photosynthesis-n96-profile/dev_photosynthesis_factual \
  --run /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4x-cpu-moe-phase-profile-n96-france \
  --run /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4t-instrumentation-n96-heldout-math-diagnostic/test_reasoning_math_01 \
  --out-dir /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5c-ram-cache-baseline-report
```

Artifacts:

- `report.md`
- `ram_cache_summary.csv`
- `ram_cache_summary.json`

Key result:

| prompt | phase data | tok/s | TTFT ms | RAM peak GiB | final file GiB | clean unmapped file GiB | decode iouring GiB | decode file delta MiB | decode refault |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_photosynthesis_factual` N32 queue profile | yes | 1.50 | 10399.5 | 11.75 | 11.23 | 11.21 | 143.1 | 77.3 | 0 |
| `dev_photosynthesis_factual` N96 phase profile | yes | 1.53 | 7385.5 | 11.76 | 11.24 | 11.22 | 443.2 | 79.2 | 0 |
| `dev_france_regression` N96 attribution profile | no | 1.52 | 11913.2 | 11.93 | 11.26 | 11.24 | phase unavailable | phase unavailable | phase unavailable |
| `test_reasoning_math_01` memory-pressure diagnostic | yes | 1.58 | 214863.1 | 14.81 | 13.65 | 13.65 | 310.7 | -33.2 | 1279 |

Interpretation:

- In normal low-refault runs, decode still moves hundreds of GiB through
  expert-pack io_uring/H2D:
  - N32 photosynthesis: `143.1 GiB`;
  - N96 photosynthesis: `443.2 GiB`.
- The same normal decode windows grow file cache by only about `77-79 MiB`, with
  `decode_refault=0`, `direct_reads=0`, and `fallback_gguf=0`.
- Final `file_mapped` is only about `0.031 MiB`; most of the remaining
  `~11.2 GiB` file-backed memory is clean, unmapped file cache from model/load
  lifecycle rather than a controlled expert cache.
- Therefore normal decode is not materially served by Linux page cache. The
  page cache is a plausible host-RAM replacement pool for explicit expert
  residency, but this is not yet proof that it is safe to evict all of it.
- The held-out math diagnostic remains the warning case:
  - RAM hits the cap;
  - TTFT is `214863.1 ms`;
  - decode refaults increase by `1279`;
  - direct reclaim is high.
  Any RAM tier that recreates this pressure must be rejected even if token rate
  improves on one prompt.

Phase 5C next implementation target:

1. Build a dev-only RAM slab candidate screen from foreground IO/wait data.
   - Candidate units:
     - whole layer/role `up+gate`;
     - whole layer/role `down`;
     - pack-contiguous expert ranges that preserve large batch reads.
   - Ranking:
     - exposed wait saved per GiB;
     - recurrence across dev prompts;
     - low overlap with current VRAM hot entries;
     - low risk of fragmenting SSD batches when mixed with RAM hits.

2. Start with pageable RAM, not pinned multi-GiB slabs.
   - Prior pinned slab attempts risked host pressure and scheduling jitter.
   - Pageable RAM still avoids SSD wait and allows the kernel to reclaim if the
     candidate is too large, making the first A/B safer under the 16 GB gate.

3. Acceptance metrics for the first runtime candidate:
   - token rate improves on N32 dev and then N96 dev;
   - TTFT ratio <= `1.20x`;
   - host RAM peak < `15900000000` bytes;
   - `workingset_refault_file`, `pgscan_direct`, and `pgsteal_direct` do not
     materially increase;
   - aggregate demand `io_uring_wait` decreases, not merely source bytes;
   - output quality passes.

## Phase 5C screen result: whole RAM slabs are batchable but too small a lever

Timestamp: 2026-07-11 CST.

New tool:

- `.Agent/run-tools/kimi_phase5c_ram_slab_screen.py`

Run:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5c-ram-slab-screen-dev7
```

Command:

```text
.Agent/run-tools/kimi_phase5c_ram_slab_screen.py \
  --input-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343 \
  --route-profile-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834 \
  --out-dir /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5c-ram-slab-screen-dev7 \
  --max-jobs 8 \
  --contig-window-mib 512 1024 2048 \
  --contig-top-per-source 8
```

Inputs:

- Dev-only trace root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343`.
- Seven dev prompts; held-out/test prompts were not used.
- Decode-like filter: `jobs <= 8`.
- Route-profile root for simulated current VRAM overlap:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834`.

Aggregate:

- decode-like rows: `195895`;
- decode-like batches: `38806`;
- decode runs: `217`;
- total traced batch wait: `117855.875 ms`;
- traced batch wait per decode token: `543.115 ms/token`;
- simulated VRAM hotset entries: `2458`.

Important findings:

1. Tiny contiguous windows are not useful runtime targets.
   - They rank high by `wait/GiB`, but cover only about `0.03-0.13 ms/token`
     each.
   - Most top contiguous windows are mixed-risk batches, meaning they would
     remove a few reads from otherwise SSD-backed batches and could reduce SSD
     batch size rather than improve the critical path.
   - Many of the top tiny windows overlap current simulated VRAM hot entries.

2. Whole layer/role slabs are batch-friendly.
   - They convert whole per-layer role demand batches into RAM hits:
     `ram_only == hit_batches` and `mixed_risk == 0` for the top layer slabs.
   - This is the correct structure if testing RAM residency, because it avoids
     the SSD/RAM fragmentation problem seen with scattered hot entries.

3. The upper bound is too small to be the main path to `2 tok/s`.

Top layer/upgate slabs:

| candidate | resident MiB | wait ms/token | ram-only batches | VRAM overlap MiB |
|---|---:|---:|---:|---:|
| `blk.29.upgate` | 2845.91 | 7.456 | 434 | 109.16 |
| `blk.8.upgate` | 2757.16 | 7.312 | 432 | 153.90 |
| `blk.28.upgate` | 2698.05 | 7.295 | 434 | 110.92 |
| `blk.9.upgate` | 2757.13 | 7.091 | 434 | 174.46 |
| `blk.7.upgate` | 2737.42 | 7.080 | 434 | 170.86 |

Top full-layer slabs:

| candidate | resident MiB | wait ms/token | ram-only batches | VRAM overlap MiB |
|---|---:|---:|---:|---:|
| `blk.8.all` | 4962.62 | 10.995 | 649 | 287.78 |
| `blk.7.all` | 4927.10 | 10.696 | 651 | 336.24 |
| `blk.9.all` | 4962.57 | 10.672 | 651 | 331.96 |
| `blk.29.all` | 4584.96 | 10.415 | 651 | 151.27 |

Greedy budget bounds:

| candidate family | budget MiB | selected | resident MiB | wait ms/token |
|---|---:|---:|---:|---:|
| `layer_upgate` | 2048 | 1 | 1220.2 | 2.58 |
| `layer_upgate` | 4096 | 2 | 3454.3 | 8.93 |
| `layer_upgate` | 8192 | 4 | 7976.4 | 21.34 |
| `layer_upgate` | 10240 | 4 | 9115.8 | 25.18 |
| `layer_all` | 4096 | 1 | 4086.4 | 9.76 |
| `layer_all` | 8192 | 2 | 7908.9 | 18.78 |
| `layer_all` | 10240 | 2 | 7908.9 | 18.78 |

Interpretation:

- The clean file cache pool is large, but a batch-safe RAM slab consumes GiB per
  layer and only saves single-digit to low-double-digit ms/token.
- To move from about `1.5 tok/s` toward `2 tok/s`, the required saving is on the
  order of `165 ms/token`. Even a very large `~10 GiB` layer-slab plan only has
  a traced wait upper bound of about `19-25 ms/token`, before H2D, RAM copy,
  scheduler, and TTFT costs.
- Therefore RAM slabs are not the main path to `2 tok/s`. They may still be
  worth one small pageable A/B if we want a low-risk incremental improvement,
  but only with modest expectations and strict TTFT/refault gates.

Decision:

- Do not spend the next major runtime phase on multi-GiB whole-layer RAM slabs
  as the primary optimization.
- If testing RAM at all, use one default-off pageable candidate such as
  `blk.29.upgate` or a small greedy `layer_upgate` profile, and treat success as
  incremental rather than SOTA-defining unless measured N96 gains exceed the
  bound/noise expectation.
- For the main path toward `>2 tok/s` and eventually `>5 tok/s`, move to lower
  byte-per-expert methods or a higher-coverage second-tier cache that does not
  require whole layer residency:
  - lower-byte expert representation;
  - role-specific compressed RAM tier;
  - GPU-resident partial expert representation;
  - predictor only if a stronger signal than route-trace co-occurrence is
    measured.

Execution plan:

1. Commit and push the profiling-only coverage fix.
   - Scope: mixed-type `ggml_cuda_moe_stream_up_gate_batch` records profile and
     CSV rows before returning.
   - This is not a SOTA claim and must not change model semantics.
   - Repro record must cite build, guard, N32, and N96 profile runs.

2. Add a mixed up/gate bottleneck subprofile before optimizing. Status: partial
   analysis-tool coverage is done; no runtime behavior change.
   - Current tool split: layer, hit/miss, stage jobs, wait, CUDA compute,
     D2H/scatter, and wall time.
   - Remaining if needed: per-row staged bytes, per-row `io_uring_wait`, pinned
     copy, and H2D source attribution.
   - Run N32 first, then N96 on dev prompts.
   - Acceptance for instrumentation: same output, no behavior change, overhead in
     profiling-only runs only.

3. Candidate A: improve mixed up/gate queue continuity.
   - Hypothesis: routing produces the active experts layer by layer, but the
     runtime exposes too few independent IO jobs; the mixed path often cannot keep
     the SSD queue deep enough.
   - Method: once a layer's routing is known, submit all missing up/gate entries
     as a coherent scheduler batch and avoid per-role gaps where possible.
   - Theoretical upper bound: at most the exposed mixed up/gate wait that is not
     overlapped by existing compute; use the new subprofile to compute the bound.
   - Reject if total decode time, aggregate `io_uring_wait`, TTFT, RAM peak, or
     held-out quality regresses.

4. Candidate B: role-aware VRAM admission for mixed up/gate.
   - Hypothesis: current global split spends some VRAM on down entries that are
     less critical than mixed up/gate misses on the decode critical path.
   - Method: default-off layer/role admission profile that protects measured
     high-wait mixed up/gate entries instead of blindly increasing global hit rate.
   - Do not tune from held-out prompts. Use dev traces only, then validate on
     held-out prompts.

5. Candidate C: controlled RAM replacement only after the IO profile proves it.
   - Goal: replace low-value page cache with batchable expert slabs, not scattered
     prompt-specific caches.
   - Prefer whole layer/role slabs or adjacent pack ranges that can feed large
     RAM-to-VRAM batches.
   - Reject if pinned/pageable RAM cache increases reclaim, TTFT, H2D bubbles, or
     mixed SSD/RAM scheduling fragmentation.

6. SOTA promotion protocol.
   - Update this plan before each implementation.
   - Run quick N32 dev check.
   - Run paired N96 dev check.
   - Run held-out validation only after dev passes.
   - Commit and push immediately only when all hard gates pass.
   - If performance, quality, RAM, or TTFT fails, revert the candidate or leave it
     default-off with a rejection record and do not call it SOTA.

## Phase 4Y plan: dev-only upgate lazy pin smoke

Timestamp: 2026-07-11 CST.

Reason:

- Phase 4W rejected static budgeted hot profiles because preload/protect
  displaced useful dynamic cache entries and increased N96 wait on at least one
  general prompt.
- Current N96 France profiling shows a large mixed `up=22/gate=18` row, but
  general dev N32 and dev photosynthesis N96 profiles mainly show
  `up=22/gate=22` and `up=18/gate=18`. Therefore the next candidate must not
  hardcode France-specific mixed layers.
- Existing runtime supports default-off lazy pin:
  `GGML_MOE_VRAM_PROFILE_PRELOAD=0` plus
  `GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT=1`. This avoids cold preload and should
  not raise TTFT unless the profile lookup/pinning itself adds overhead.

Candidate:

- Generate an upgate-only `GGML_MOE_VRAM_PROFILE` from dev-only route profiles:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4v-default-profile-n32-dev3`.
- Use the wait-weighted dev screen:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4v-default-profile-n32-dev3-analysis-v3/admission-screen/layer-role-screen.csv`.
- Budget:
  - upgate: `512 MiB`;
  - down: `0 MiB`.
- Generated output:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4y-lazy-upgate-profile/upgate512-down0.csv`.
- Selected:
  - `99` upgate entries;
  - `511.33 MiB`;
  - `3909` dev route count.

Candidate env delta:

```text
GGML_MOE_VRAM_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4y-lazy-upgate-profile/upgate512-down0.csv
GGML_MOE_VRAM_PROFILE_PROTECT=1
GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1
GGML_MOE_VRAM_PROFILE_PRELOAD=0
GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT=1
GGML_MOE_VRAM_PROFILE_PIN_MIN_COUNT=1
GGML_MOE_VRAM_PROFILE_RESERVE_PCT=92
```

Hypothesis:

- Lazy pin should avoid the Phase 4W preload/TTFT penalty.
- If the dev hot upgate entries recur during decode, pin-on-insert should
  reduce later upgate misses and lower upgate wait without changing down cache.

Theoretical bound:

- On dev photosynthesis N96, upgate wall is `177.032 ms/token`.
- The largest dev-general upgate type is `up=22/gate=22`, `99.705 ms/token`,
  with wait `5.308 ms/call` and compute only about `0.342 ms/call`.
- A 512 MiB profile cannot remove all upgate wait. With `99` entries versus
  thousands of possible layer experts, the realistic N32 smoke bound is small:
  expect at most a few percent if the selected entries recur.
- The candidate is worth continuing only if it reduces measured decode wall and
  upgate wait, not just nominal cache hit rate.

Execution:

1. Run a paired N32 cold-start smoke on `dev_photosynthesis_factual` with
   `PROFILE=1` and `COPY_PROFILE=1`.
2. Compare control and candidate:
   - token rate;
   - TTFT ratio;
   - RAM peak;
   - quality;
   - upgate wall and upgate hit/miss;
   - `iouring_wait`;
   - down hit rate to catch collateral damage.
3. If N32 smoke regresses or is noise-level, reject and do not run N96.
4. If N32 smoke clearly improves without violating gates, run paired N96 dev
   before any held-out validation.

Phase 4Y N32 smoke result: rejected.

Timestamp: 2026-07-11 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4y-lazy-upgate-n32-photosynthesis`

Prompt:

- `dev_photosynthesis_factual`
- User text: `Explain photosynthesis briefly.`

Candidate env delta:

- `GGML_MOE_VRAM_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4y-lazy-upgate-profile/upgate512-down0.csv`
- `GGML_MOE_VRAM_PROFILE_PROTECT=1`
- `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`
- `GGML_MOE_VRAM_PROFILE_PRELOAD=0`
- `GGML_MOE_VRAM_PROFILE_PIN_ON_INSERT=1`
- `GGML_MOE_VRAM_PROFILE_PIN_MIN_COUNT=1`
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=92`

Result:

| run | quality | tok/s | TTFT ms | decode ms / runs | RAM peak bytes | upgate hit | down hit | upgate pinned |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control | pass | 1.50 | 9795.98 | 20733.50 / 31 | 12604973056 | 42.5% | 58.2% | 0 |
| lazy_upgate512 | pass | 1.49 | 10095.77 | 20789.45 / 31 | 12608712704 | 42.4% | 58.2% | 65 |

Component deltas:

- decode: `+55.95 ms` total regression;
- TTFT ratio: `1.031`, within the gate but worse;
- `iouring_wait`: `15959.039 -> 16143.942 ms`, `+184.903 ms`;
- upgate wall: `462.690 -> 465.560 ms/token`, `+2.870 ms/token`;
- down wall: `169.635 -> 168.535 ms/token`, `-1.100 ms/token`;
- SSD bytes: `218799931392 -> 218895925248`, `+95.99 MB`;
- upgate cache hit rate did not improve and slightly decreased.

Decision:

- Reject `upgate512-down0` lazy pin.
- Do not run N96.
- Do not use this profile in SOTA reproduction.

Interpretation:

- Lazy pin avoided the large preload/TTFT failure mode, but it still did not
  reduce the critical path.
- Only `65` of `99` candidate entries became pinned during this N32 decode, and
  they did not increase aggregate upgate hit rate.
- Static or lazy dev hotset admission is not the right next lever unless a
  later profile proves a much stronger recurrence pattern.

Next:

- Stop tuning static/lazy `GGML_MOE_VRAM_PROFILE` budgets for now.
- Move to runtime queue/scheduler evidence:
  - join `copy-profile.csv`, `io-batch-profile.csv`, and `up-gate-profile.csv`
    by tensor/role/time window where feasible;
  - identify whether current upgate stalls are due to small per-layer batch
    size, queue drain between layers, H2D slot pressure, or unavoidable routing
    dependency;
  - only then implement co-submit/queue-continuity changes.

## Phase 4Z plan: queue/scheduler evidence before co-submit

Timestamp: 2026-07-11 CST.

Reason:

- Static preload and lazy pin both failed to reduce the critical path.
- Upgate wait remains large, but cache hit rate tuning alone has not improved
  decode wall.
- Before changing scheduling, collect direct queue evidence to decide whether
  exposed wait comes from:
  - small per-layer batches;
  - IO queue drain between batches;
  - refill/wait policy;
  - H2D slot pressure or copy coalescing;
  - unavoidable routing dependency.

Instrumentation run:

- Prompt: `dev_photosynthesis_factual`.
- N: `32`.
- Cold start, 16GB cgroup.
- Keep runtime behavior unchanged except profiling env.
- Enable:
  - `PROFILE=1`;
  - `COPY_PROFILE=1`;
  - `GGML_MOE_IO_BATCH_PROFILE_OUT=$RUN/io-batch-profile.csv`;
  - `GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv`;
  - `GGML_MOE_H2D_COALESCE_PROFILE_OUT=$RUN/h2d-coalesce-profile.csv`;
  - `GGML_MOE_IO_LOCALITY_PROFILE_OUT=$RUN/io-locality-profile.csv`.

Analysis to add:

- Parse `copy-profile.csv` by role/layer/tensor:
  `io_wait_ms`, `h2d_ms`, `enqueue_ms`, `slot_wait_ms`, bytes, read count.
- Parse `io-batch-profile.csv` by op and tensor role:
  jobs, read jobs, initial submit jobs, wait calls, inflight average/max,
  wait wall, submit wall, slot wait, batch-size histogram.
- Parse `io-wait-trace.csv`:
  wait count, zero-drain waits, waits with low inflight, waits with queue
  exhaustion, and top wait records.
- Parse `h2d-coalesce-profile.csv`:
  copy count, coalesced copy count, contiguous-pair opportunity, bytes.
- Join at aggregate tensor/role level with `up-gate-profile.csv`.

Decision rules:

- If most wait records have low `inflight_before` and small `read_jobs`, the
  next candidate should be queue continuity/co-submit.
- If wait records have high inflight but long wait, the bottleneck is storage
  latency/throughput or refill policy, not simply batch construction.
- If H2D coalescing is poor and H2D is significant, target staging layout.
- If `up-gate-profile.csv` wait is high but IO batch wait is not, target CUDA
  stream synchronization or cache slot readiness.
- This phase produces evidence only; no SOTA claim.

Phase 4Z result: decode waits are not queue-empty; locality is poor.

Timestamp: 2026-07-11 CST.

Run:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4z-queue-evidence-n32-photosynthesis/dev_photosynthesis_factual`

Reports:

- Decode bottleneck:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4z-queue-evidence-n32-photosynthesis/decode-analysis/report.md`
- Queue/scheduler evidence:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4z-queue-evidence-n32-photosynthesis/queue-analysis-v3/report.md`

Run metrics:

- Quality: pass.
- Token rate: `1.50 tok/s`.
- TTFT: `10399.55 ms`.
- Decode: `20716.66 ms / 31 runs`.
- Expert-pack iouring:
  - reads: `38275`;
  - bytes: `218799931392`;
  - wait: `15765374 us`;
  - inflight avg/max: `4.47 / 8`.

Decode-like IO batch evidence (`read_jobs <= 8`):

| op | rows | read jobs | avg read jobs | avg initial submit | wait ms | wait/read job | inflight avg | inflight max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| runtime_load | 4678 | 22320 | 4.771 | 4.771 | 10059.096 | 0.451 | 4.100 | 8 |
| current_down_overlap | 899 | 4489 | 4.993 | 4.993 | 1559.044 | 0.347 | 4.669 | 8 |

Decode-like IO wait evidence:

- wait rows: `10534`;
- wait ms: `11618.140`;
- p50/p95/p99 wait: `1.189 / 2.372 / 2.983 ms`;
- low inflight ratio: `0.055`;
- queue empty ratio: `0.000`;
- zero drain ratio: `0.000`;
- next-job-done ratio: `1.000`.

Decode-like IO locality evidence:

| op | rows | read jobs | read GiB | span/read | gap/read | source switches/row | adjacent pairs/read job |
|---|---:|---:|---:|---:|---:|---:|---:|
| runtime_load | 4678 | 22320 | 116.855 | 32.580 | 31.580 | 1.519 | 0.013 |
| current_down_overlap | 899 | 4489 | 26.377 | 34.017 | 33.017 | 1.734 | 0.010 |

Up/gate evidence:

- Largest type is `up=22/gate=18/parallel=1`:
  - calls: `899`;
  - wall: `8442.332 ms`, `9.391 ms/call`;
  - up/gate misses: `4.402 / 4.406 per call`;
  - up/gate wait: `4.990 / 5.429 ms/call`;
  - up/gate compute: `0.157 / 0.097 ms/call`.

Interpretation:

- Decode wait is not caused by an empty queue. During decode-like waits, the
  queue-empty ratio is `0`, zero-drain ratio is `0`, and `next_job_done=1.0`
  means the currently exposed batch has already submitted all available jobs.
- A plain same-layer submit-loop rewrite is unlikely to recover the pure IO
  bench gap unless it exposes genuinely new future-layer work.
- `runtime_load` batches average only `4.77` read jobs and `current_down_overlap`
  batches average `4.99` read jobs. This is a dependency/visibility limit, not
  just a missed submit call.
- Locality is very poor even for same-tensor batches:
  `gap/read` is about `31-33x` and adjacent pairs per read job are near zero.
  This points toward expert pack layout / route-order locality / predictive
  future-layer prefetch as higher-leverage than more static VRAM hotsets.
- H2D coalesce profile did not emit rows in this run, so H2D coalescing still
  needs either different instrumentation or a specific H2D-focused experiment
  before making claims.

Decision:

- Do not implement a naive same-layer co-submit path as the next step.
- Do not retry static/lazy `GGML_MOE_VRAM_PROFILE` budgets.
- Next work should be a layout/prefetch bound study:
  1. estimate how much wait is removable if same-tensor active experts are laid
     out contiguously or near-contiguously in the expert pack;
  2. estimate how many future-layer expert IDs would need to be known to raise
     decode-like read jobs from `~5` toward `12-16`;
  3. only implement a runtime path if the bound can save at least
     `~100 ms/token` on dev N96 without prompt-specific tuning.

## Phase 5A plan: layout and future-prefetch bound study

Timestamp: 2026-07-11 CST.

Reason:

- Phase 4Z showed decode-like waits are not queue-empty. The currently visible
  jobs are already submitted, so a naive same-layer submit-loop rewrite is not
  enough.
- The same profile showed poor locality: decode-like same-tensor batches have
  `gap/read` around `31-33x` and almost no adjacent expert pairs.
- However, poor locality alone does not prove that repacking will save enough
  decode wall. We need a bound before changing pack format or runtime prefetch.

Bound study inputs:

- Primary input:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4z-queue-evidence-n32-photosynthesis/dev_photosynthesis_factual`.
- Use `io-batch-profile.csv`, `io-wait-trace.csv`, `io-locality-profile.csv`,
  `up-gate-profile.csv`, and `metrics.json`.
- Use decode-like rows only: `read_jobs <= 8`.

Questions to answer:

1. Layout-only bound:
   - If same-tensor active experts were arranged closer together, what is the
     best observed wait/read-job for comparable batch sizes?
   - Does low `gap/read` actually correlate with lower wait/read-job in the
     current profile?
   - If every decode-like batch achieved the best observed wait/read-job for
     its batch size, how many ms/token would be saved?

2. Future-layer prefetch bound:
   - If a predictor could expose the next `K` decode-like batches before they
     are needed, the optimistic wait bound is `sum(wait) - max(wait)` per
     window.
   - Evaluate `K = 2, 3, 4, 8, 16`.
   - Report average jobs per future window. The target is to understand how
     many future batches are needed to raise visible read jobs from about `5`
     toward `12-16`.

3. Implementation threshold:
   - Do not implement pack layout or future-layer prefetch unless the measured
     bound can plausibly save at least `~100 ms/token` on dev N96 after scaling
     from this N32 profile.
   - If layout-only bound is small or weakly correlated with wait, prioritize
     predictive prefetch or smaller expert representation instead of pack
     repacking.
   - If future-layer prefetch bound is large, the next experiment must still
     validate predictor accuracy on dev prompts and reserve held-out prompts for
     validation only.

Planned tool:

- Add `.Agent/run-tools/kimi_layout_prefetch_bound.py`.
- Outputs:
  - `layout_bins.csv`;
  - `batch_size_bound.csv`;
  - `future_prefetch_bound.csv`;
  - `report.md`.
- This tool is offline analysis only and must make no SOTA claim.

Phase 5A result: future-layer prefetch has a larger bound than layout-only.

Timestamp: 2026-07-11 CST.

Tool:

- `.Agent/run-tools/kimi_layout_prefetch_bound.py`

Input:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4z-queue-evidence-n32-photosynthesis/dev_photosynthesis_factual`

Output:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5a-layout-prefetch-bound-photosynthesis/report.md`

Baseline decode-like IO:

- decode-like batches: `5577`;
- read jobs: `26809`;
- average read jobs/batch: `4.807`;
- read GiB: `143.232`;
- wait: `11618.140 ms`;
- wait/token: `374.779 ms/token`;
- wait/read job: `0.433 ms`;
- Pearson correlation `log(gap/read)` vs wait/read-job: `-0.400`.

Layout-only bound:

- p10 same-size bound:
  - saving: `3207.412 ms`;
  - saving/token: `103.465 ms/token`.
- min same-size bound:
  - saving: `5184.799 ms`;
  - saving/token: `167.252 ms/token`.
- But the locality bins do not support a simple "lower gap/read => lower wait"
  causal story:
  - runtime_load `gap<=1`: wait/read `0.891 ms`;
  - runtime_load `16<gap<=64`: wait/read `0.432 ms`;
  - current_down_overlap `gap<=1`: wait/read `0.726 ms`;
  - current_down_overlap `16<gap<=64`: wait/read `0.335 ms`.
- Therefore a pack relayout may still help by enabling better future grouping,
  but the current profile does not justify a standalone pack-relayout
  implementation as the next step.

Future-layer prefetch optimistic bound:

| future window batches | avg jobs/window | saving ms | saving ms/token |
|---:|---:|---:|---:|
| 2 | 9.612 | 5081.526 | 163.920 |
| 3 | 14.421 | 7039.994 | 227.097 |
| 4 | 19.218 | 8006.597 | 258.277 |
| 8 | 38.408 | 9609.637 | 309.988 |
| 16 | 76.817 | 10491.626 | 338.440 |

Interpretation:

- The `K=3` bound reaches the desired visible batch scale:
  average jobs/window `14.421`, within the target `12-16` range.
- The optimistic `K=3` saving is `227.097 ms/token`, large enough to justify a
  targeted prefetch/prediction exploration.
- This is still only an upper bound. It assumes future batches can be known and
  submitted early, and that waits overlap perfectly. A real predictor must
  measure accepted/usable future expert IDs and must not tune on held-out
  prompts.

Decision:

- Do not start with standalone pack relayout.
- Next implementation plan should test prompt-general future-layer expert
  prediction/prefetch in a default-off way:
  1. collect route traces across dev prompts and measure whether layer `L` can
     predict layer `L+1..L+3` top experts well enough;
  2. prefetch only when predicted experts have high confidence and a bounded
     extra-byte budget;
  3. require a paired N32 dev A/B before any N96 run;
  4. reject if extra bytes, TTFT, RAM, or quality regress.

## 2026-07-11 goal: transfer the DeepSeek CPU/defer GPU-extension pattern to Kimi

This section is the immediate goal and plan for the next Kimi workstream.

Goal:

> Determine whether the DeepSeek SOTA architecture pattern can improve Kimi:
> keep CPU/defer as the MoE routing and scheduling owner, but make GPU a
> stronger extension for expert residency, expert movement, and `gate/up/down`
> compute. The short-term target is a reproducible general-prompt improvement
> above the current stable Kimi SOTA and toward stable `>2 tok/s`; the product
> target remains stable `>5 tok/s` for random prompts on one 32 GB RTX
> 5090-class GPU with total host RAM below 16 GB.

Expected applicability to Kimi:

- The DeepSeek lesson should transfer only as an architecture pattern, not as a
  literal gate-only optimization.
- Kimi already has near-zero decode CPU fallback on the current stable path, so
  the next Kimi gain is not expected to come from simply moving fallback CPU
  math to GPU.
- The likely Kimi benefit is broader coverage of the CPU/defer extension path:
  once routing knows active experts, `gate`, `up`, and `down` should be fetched,
  cached, overlapped, and computed through GPU-backed paths whenever possible.
- A useful candidate must reduce exposed decode wait, especially
  `io_uring_wait`, staging wait, and H2D bubbles. Higher cache hit rate alone is
  not enough.

Hard success gates:

- Cold start only.
- Total host RAM peak must stay below `15900000000` bytes, including page
  cache, pinned memory, mmap/file-backed pages, helper processes, and cgroup
  accounting.
- Use VRAM aggressively, but do not accept a result that buys VRAM hit rate by
  increasing TTFT, RAM pressure, or quality risk.
- TTFT must be no more than `1.20x` the paired baseline.
- Mandatory quality prompt: `Please introduce France in a short paragraph.`
  must remain coherent and semantically correct.
- Optimization must be prompt-general. Dev prompts can guide design; held-out
  prompts are validation only and must not be used to tune packs, hotsets,
  thresholds, or layer choices.
- Every accepted improvement must be reproducible from a pushed commit with
  exact env, command, prompt split, run directory, metrics, quality result, and
  rollback commit in the commit message body and this plan.

Execution plan:

1. Freeze and re-state the current baseline.
   - Use the current branch `vendor/kimi-deepseek-41d205-additive`.
   - Record the rollback commit before each candidate.
   - Run at least one N96 cold-start general-prompt baseline with
     `GGML_MOE_PHASE_REPORT=1`.
   - Record token rate, TTFT, decode time, prompt time, RAM peak, file cache,
     VRAM cache hit rates, expert-pack bytes, `io_uring_wait`, staging wait,
     H2D time, down compute, up/gate compute, and CPU fallback counters.

2. Audit Kimi's CPU/defer extension coverage.
   - Confirm which path owns Kimi MoE scheduling under the current run
     arguments, especially any `n_cpu_moe` or CPU/defer settings.
   - For every fused and non-fused branch, record whether `gate`, `up`, and
     `down` are handled by GPU extension, expert pack, VRAM cache, RAM tier,
     mmap fallback, or CPU fallback.
   - Treat any remaining CPU fallback as a correctness/performance bug only if
     it is on the measured critical path.

3. First candidate: close extension coverage gaps without changing model
   semantics.
   - Start from default-off switches only.
   - Prioritize all-path current-down overlap, because shadow profiling showed
     many plannable same-layer down misses that current overlap does not cover.
   - Theoretical upper bound: the maximum gain is the exposed down-read wait
     that can be hidden under already-required up/gate staging and compute; the
     measured bound must come from phase counters, not from hit-rate deltas.
   - Accept only if paired N96 dev and held-out runs show lower decode time
     without RAM, TTFT, or quality regression.

4. Second candidate: role-joint scheduling for active experts.
   - When routing for a layer is known, submit `gate/up/down` misses in a
     single scheduler view where this does not break existing compute
     dependencies.
   - This is read scheduling, not joint compute. `down` compute still depends
     on the `up/gate` activation result.
   - Measure whether larger per-layer batches increase sustained SSD throughput
     without creating staging or H2D tail latency.

5. Third candidate: explicit RAM/VRAM storage layout.
   - VRAM holds hottest and most latency-critical experts.
   - RAM holds second-tier experts only if they replace low-value decode-time
     file cache and can still be transferred to VRAM in batchable slabs.
   - Prefer whole layer/role slabs or compact adjacent pack ranges over
     scattered single-expert entries when scattered entries shrink SSD batch
     size or increase scheduler overhead.
   - Before accepting any RAM tier, prove with phase memory counters which page
     cache pages were evicted and that refault/TTFT did not regress.

6. Validation ladder.
   - N32 dev smoke: quality, RAM, parser, and obvious latency regression.
   - N96 dev paired A/B: bottleneck and token-rate validation.
   - N96 held-out paired A/B: final acceptance gate.
   - Commit and push immediately for accepted improvements.
   - Reject or leave default-off any candidate that regresses quality, TTFT,
     RAM, held-out token rate, or aggregate exposed wait.

## 2026-07-11 Phase 4U.1 implementation plan: all-path current-down overlap

Hypothesis:

- Current Kimi decode is still limited by exposed expert movement wait.
- Existing current-down overlap only fires on part of the fused up/gate path.
- Shadow profiling on N32 dev3 showed `25094` plannable same-layer down jobs,
  while actual current-down overlap completed only `13041` jobs.
- Therefore, extending current-down overlap to the same-type fused up/gate
  paths may hide additional down transfer under already-required up/gate
  staging, up/gate compute, fuse, D2H, and scatter.

Implementation constraints:

- Add a new default-off switch:
  `GGML_MOE_CURRENT_DOWN_OVERLAP_ALL_PATHS=1`.
- Preserve existing default behavior when the env flag is absent.
- Do not enable the previously rejected
  `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` path.
- Start overlap only after the same-type up/gate work has been planned or
  staged, so the candidate is less aggressive than the rejected early mode.
- Join the overlap worker before returning from the up/gate batch function so
  downstream down compute sees either a ready/pending cache slot or the normal
  fallback behavior.
- Keep failures non-fatal for model semantics: a failed overlap should clear
  speculative slots and let the normal down path recover.

Theoretical upper bound:

- The absolute upper bound is the exposed down-transfer time for the additional
  plannable jobs that are not currently overlapped.
- From the N32 dev3 shadow run, additional plannable jobs are roughly
  `25094 - 13041 = 12053` down jobs across three prompts.
- The realized gain must be lower than this bound because the extra overlap
  competes with up/gate expert reads for the same SSD/io_uring and H2D
  resources, and some down reads will still be consumed after up/gate has
  already finished.
- Accept only if measured paired runs show lower decode wall time and lower
  exposed wait, not merely more submitted overlap jobs.

Validation:

1. Build and guard test.
2. N32 dev smoke with:
   `GGML_MOE_PHASE_REPORT=1`,
   `GGML_MOE_CURRENT_DOWN_OVERLAP=1`,
   `GGML_MOE_CURRENT_DOWN_OVERLAP_ALL_PATHS=1`.
3. Paired N96 dev A/B against the same commit and baseline env.
4. Held-out N96 only if dev improves without RAM, TTFT, or quality regression.
5. Commit and push only if accepted; otherwise keep default-off and record the
   measured rejection reason here.

Result:

- Status: rejected as SOTA; keep the implementation default-off only.
- Build: `cmake --build build-cuda-batch -j 8` passed.
- Guard: `build-cuda-batch/bin/test-kimi-deepseek2-guards` passed.
- Env:
  - baseline: `GGML_MOE_PHASE_REPORT=1`;
  - candidate: `GGML_MOE_PHASE_REPORT=1`,
    `GGML_MOE_CURRENT_DOWN_OVERLAP=1`,
    `GGML_MOE_CURRENT_DOWN_OVERLAP_ALL_PATHS=1`.
- Run roots:
  - France:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4u1-allpath-n32-france`
  - Japan/photosynthesis:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4u1-allpath-n32-dev3`

N32 paired result:

| prompt | base tok/s | candidate tok/s | decode delta ms | iouring wait delta s | current-down jobs delta | down hit delta | TTFT ratio | quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | 1.88 | 1.97 | -775.1 | -1.274 | +4142 | +18.6 pp | 0.950 | pass/pass |
| `dev_japan_factual` | 1.94 | 1.93 | +111.4 | +0.120 | +3853 | +17.3 pp | 1.021 | pass/pass |
| `dev_photosynthesis_factual` | 1.85 | 1.75 | +915.6 | +0.099 | +4069 | +19.4 pp | 0.955 | pass/pass |

Decision:

- Do not run N96 promotion for this candidate.
- Do not enable `GGML_MOE_CURRENT_DOWN_OVERLAP_ALL_PATHS=1` in SOTA
  reproduction.
- Although the candidate nearly doubles current-down overlap coverage and
  raises down hit rate by `17-19 pp`, it does not reliably reduce exposed wait.
- Total expert-pack read count and bytes are essentially unchanged. The change
  mostly shifts down reads earlier, but the late start point only overlaps them
  with fuse/D2H/scatter, which is too short for the additional down jobs.
- The added overlap worker time is not fully hidden and can contend with the
  same SSD/io_uring and staging resources needed by up/gate or normal down
  reads. This explains why hit rate improves while `dev_photosynthesis_factual`
  slows down.

Next direction from this rejection:

1. Do not chase down-hit rate alone.
2. If revisiting overlap, it must be a scheduler-level design that starts only
   when it can hide under real up/gate staging/compute time and avoid stealing
   critical up/gate IO.
3. Prefer measuring per-layer exposed wait before adding more speculative down
   traffic.
4. The next candidate should either:
   - implement role-joint scheduling with explicit IO-budget fairness between
     up/gate and down; or
   - target RAM/VRAM layout so high-value down reads become real cache hits
     without adding same-token speculative IO.

## 2026-07-11 Phase 4V profile: decode-only bottleneck correction

Why this was needed:

- The previous bottleneck script treated every `down-batch-profile.csv` row as
  decode work.
- That was wrong because `down-batch-profile.csv` has no `mode` column and also
  contains prompt/prefill rows with large `n_active`.
- The script now filters down rows with `n_active > 8` out of decode totals by
  default and only treats directories with `metrics.txt` as run directories.
- This is a tooling fix only; it does not change runtime behavior.

Run:

- Root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4v-default-profile-n32-dev3`
- Analysis:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4v-default-profile-n32-dev3-analysis-v2`
- Env:
  - `N=32`
  - `PROFILE=1`
  - `GGML_MOE_PHASE_REPORT=1`
  - default runtime, no `GGML_MOE_CURRENT_DOWN_OVERLAP_ALL_PATHS`
- Prompts:
  - `dev_france_regression`
  - `dev_japan_factual`
  - `dev_photosynthesis_factual`
- Quality: all pass.

Corrected decode-only profile:

| prompt | tok/s | decode ms/token | up/gate ms/token | down ms/token | residual ms/token | direct read ratio | iouring GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | 1.680 | 595.630 | 164.590 | 161.737 | 269.303 | 0.000 | 11.300 |
| `dev_japan_factual` | 1.650 | 606.544 | 168.059 | 159.635 | 278.850 | 0.000 | 11.051 |
| `dev_photosynthesis_factual` | 1.640 | 611.012 | 164.855 | 160.383 | 285.774 | 0.000 | 10.758 |

Aggregate:

- weighted decode: `604.395 ms/token` (`1.655 tok/s`) with profiling overhead;
- residual unattributed: `277.976 ms/token`;
- upgate wall: `165.835 ms/token`;
- down wall: `160.585 ms/token`;
- down stage: `150.492 ms/token`;
- decode CPU fallback: `0`;
- direct reads: `0`;
- iouring read ratio: `1.000`.

Top corrected detail rows:

| group | item | ms/token |
|---|---|---:|
| `upgate_type` | `up=22/gate=22` | 94.479 |
| `upgate_type` | `up=18/gate=18` | 71.356 |
| `down_type` | `type=11` | 66.213 |
| `down_type` | `type=23` | 57.905 |
| `down_type` | `type=2` | 36.466 |
| `upgate_layer` | `blk.1` | 8.367 |
| `upgate_layer` | `blk.60` | 8.027 |
| `upgate_layer` | `blk.5` | 7.754 |
| `upgate_layer` | `blk.4` | 7.575 |
| `upgate_layer` | `blk.3` | 7.430 |
| `down_layer` | `blk.6` | 5.478 |
| `down_layer` | `blk.10` | 5.477 |
| `down_layer` | `blk.4` | 5.405 |

Implications:

- The current stable path has no decode CPU fallback and no direct-read path in
  this dev profile. CPU fallback is not the next bottleneck.
- Expert movement is still a pressure signal (`iouring_wait_reported` is high),
  but the additive decode buckets show that residual non-profiled work is also
  large.
- Removing only measured upgate+down wall would still leave roughly
  `278 ms/token`, which is only about `3.6 tok/s`; reaching `5 tok/s` requires
  reducing residual or proving that part of residual is hidden MoE/scheduler
  wait not currently attributed.
- Therefore the next candidate must be tied to a measured component:
  - for MoE: reduce `upgate_type up=22/gate=22`,
    `upgate_type up=18/gate=18`, or down stage for types `11/23/2`;
  - for the product target: add residual attribution before claiming a path to
    `5 tok/s`.

Next plan:

1. Build a default-off wait-weighted layer/role admission screen.
   - Use dev route/profile data only.
   - Rank candidates by corrected decode wall/stage, not by hit rate alone.
   - Candidate should prefer upgate hot layers and down hot layers that are
     stable across dev prompts.
   - Do not use held-out prompts for selecting layers or experts.
2. Separately add residual attribution before the next large implementation.
   - Split residual into dense/attention/shared compute, scheduler overhead,
     CUDA sync, D2H/scatter not covered by current profiles, and host-side
     bookkeeping if feasible.
   - If residual is mostly non-MoE compute, the `5 tok/s` plan cannot rely only
     on expert-cache policy.
3. Only after the screen shows a theoretical bound, run an N32 default-off A/B.
   - Acceptance requires lower decode wall, lower exposed wait or measured
     component wall, unchanged quality, RAM under limit, and TTFT within gate.

Phase 4V admission screen result:

- Tool added:
  `.Agent/run-tools/kimi_wait_weighted_admission_screen.py`
- Output:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4v-default-profile-n32-dev3-analysis-v3/admission-screen/layer-role-screen.md`
- The screen ranks layer/role buckets by corrected decode wall, not hit rate.

Top screen rows:

| layer | role | ms/token | hit rate | misses | observed footprint |
|---:|---|---:|---:|---:|---:|
| 1 | upgate | 8.367 | 30.24% | 1038 | 1892.41 MiB |
| 60 | upgate | 8.027 | 33.46% | 1022 | 2186.62 MiB |
| 5 | upgate | 7.754 | 29.03% | 1056 | 2368.84 MiB |
| 4 | upgate | 7.575 | 30.51% | 1034 | 2368.84 MiB |
| 3 | upgate | 7.430 | 33.00% | 997 | 2636.81 MiB |
| 6 | down | 5.478 | 27.96% | 536 | 1653.75 MiB |
| 10 | down | 5.477 | 26.08% | 550 | 1614.38 MiB |
| 4 | down | 5.405 | 20.97% | 588 | 1643.69 MiB |

Interpretation:

- Whole layer/role admission is too coarse for the 16 GB host RAM and 32 GB
  VRAM target. A single upgate layer can cost around `1.9-2.6 GiB` for only
  `7-8 ms/token` of ideal bound in this N32 dev profile.
- A full down layer is cheaper than upgate but still around `1.6 GiB` for only
  `5.4 ms/token` of ideal bound.
- Therefore the next runtime candidate should not preload whole layers by
  default.
- Better next candidates:
  1. top-expert-within-hot-layer admission, budgeted by MiB and weighted by
     measured decode wall/stage;
  2. residual attribution, because `~278 ms/token` remains outside the current
     upgate/down buckets;
  3. IO-budget scheduling only if it can reduce measured upgate/down wall
     without increasing residual or total iouring wait.

## 2026-07-11 Phase 4W result: budgeted hot expert profile rejected

Goal:

- Test a smaller alternative to whole-layer admission:
  top experts inside wait-weighted hot layer/role buckets.
- Keep the mechanism default-off via existing `GGML_MOE_VRAM_PROFILE*`.
- Use dev-only profile data; do not use held-out prompts.

Tool:

- Added `.Agent/run-tools/kimi_make_budgeted_hot_expert_profile.py`.
- Inputs:
  - corrected layer/role screen:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4v-default-profile-n32-dev3-analysis-v3/admission-screen/layer-role-screen.csv`
  - dev route profiles from:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4v-default-profile-n32-dev3`
- Candidate output:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4w-budgeted-hot-profile/budgeted-hot-upgate1024-down512.csv`
- Budget:
  - upgate: `1024 MiB`
  - down: `512 MiB`
- Selected:
  - `197` upgate entries, `1019.92 MiB`;
  - `66` down entries, `509.69 MiB`;
  - total `263` entries, `1529.61 MiB`.

Runtime env for candidate:

```text
GGML_MOE_PHASE_REPORT=1
GGML_MOE_VRAM_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4w-budgeted-hot-profile/budgeted-hot-upgate1024-down512.csv
GGML_MOE_VRAM_PROFILE_PROTECT=1
GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1
GGML_MOE_VRAM_PROFILE_PRELOAD=1
GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1
GGML_MOE_VRAM_PROFILE_RESERVE_PCT=20
```

N32 dev result:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4w-budgeted-hot-n32-dev3`

| prompt | base tok/s | candidate tok/s | decode delta ms | iouring wait delta s | TTFT ratio | quality |
|---|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | 1.95 | 1.94 | +79.5 | +0.448 | 1.052 | pass/pass |
| `dev_japan_factual` | 1.90 | 1.95 | -446.5 | -0.221 | 1.152 | pass/pass |
| `dev_photosynthesis_factual` | 1.84 | 1.95 | -921.3 | -0.474 | 1.151 | pass/pass |

N32 aggregate:

- decode: `527.19 -> 513.34 ms/token`;
- token rate: `1.897 -> 1.948 tok/s`;
- iouring wait: `-0.247 s`;
- read bytes: `-0.87 GiB`;
- quality: all pass.

N96 dev result:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4w-budgeted-hot-n96-dev3`

| prompt | base tok/s | candidate tok/s | decode delta ms | iouring wait delta s | read GiB delta | direct reads candidate | TTFT ratio | quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | 1.94 | 1.93 | +324.1 | +1.186 | +0.53 | 263 | 1.105 | pass/pass |
| `dev_japan_factual` | 1.97 | 1.98 | -201.2 | -0.959 | -1.55 | 263 | 1.115 | pass/pass |
| `dev_photosynthesis_factual` | 1.92 | 1.81 | +3101.6 | +2.978 | +4.08 | 263 | 1.228 | pass/pass |

N96 aggregate:

- decode: `514.61 -> 527.15 ms/token`;
- token rate: `1.943 -> 1.897 tok/s`;
- iouring wait: `+3.206 s`;
- read bytes: `+3.07 GiB`;
- candidate introduced `263` direct reads;
- `dev_photosynthesis_factual` violates the TTFT gate: `1.228x > 1.20x`.

Decision:

- Reject this candidate as SOTA.
- Do not run held-out validation.
- Do not set this `GGML_MOE_VRAM_PROFILE` in reproduction.
- Keep the generator as offline tooling only.

Interpretation:

- N32 was a false positive. Longer N96 output exposed that the profile can
  increase read volume, direct reads, and iouring wait on at least one general
  prompt.
- Pinning a budgeted dev hotset can displace useful dynamic cache entries even
  when the selected profile is much smaller than whole-layer admission.
- Upgate hit rate did not improve reliably; down hit rate improved only modestly
  and was not enough to offset profile preload/protection side effects.
- The next optimization should not tune static VRAM profile budgets further
  until residual attribution is better understood.

Next plan:

1. Add residual attribution for decode.
   - Measure time outside current upgate/down profile buckets.
   - Split dense/attention/shared compute, CUDA sync, D2H/scatter, scheduler
     overhead, and host bookkeeping where feasible.
2. Only revisit cache admission after residual is explained.
   - If residual is mostly dynamic cache/scheduler wait, target scheduler.
   - If residual is dense/attention/shared compute, expert-cache policy alone
     cannot reach `5 tok/s`.

## 2026-07-11 Phase 4X goal and plan: Kimi CPU/defer GPU-extension audit

Goal:

- Verify whether the DeepSeek SOTA idea applies to Kimi in the current code:
  keep CPU/defer as the MoE control path, but make GPU the execution and
  storage extension for `gate/up/down` experts whenever this reduces measured
  critical-path time.
- The near-term target is a reproducible, prompt-general Kimi improvement over
  the current stable path and a stable `>2 tok/s` N96 result under the hard
  16 GB host RAM gate.
- The product target remains stable `>5 tok/s` for random user prompts on
  `16 GB host RAM + 32 GB RTX 5090`; this phase must state whether that target
  is blocked by expert movement, residual non-MoE time, or storage format.

Why this is not a direct DeepSeek gate-only port:

- In DeepSeek, a large gain came from the fact that the CPU/defer MoE path was
  still doing important gate work slowly until a GPU expert-cache extension
  caught it.
- In current Kimi profiles, decode CPU fallback is already `0`, direct reads
  are `0`, and the measured hot path is mostly GPU extension plus expert-pack
  movement.
- Therefore, Kimi should not assume a gate-only VRAM preload will reproduce the
  DeepSeek jump. The next useful work is to prove which part of Kimi still
  waits outside the current `upgate` and `down` profile buckets.

Current measured bound to respect:

- Corrected Phase 4V N32 profile:
  - decode: `604.395 ms/token`;
  - upgate wall: `165.835 ms/token`;
  - down wall: `160.585 ms/token`;
  - unattributed residual: `277.976 ms/token`;
  - decode CPU fallback: `0`.
- If all measured upgate and down wall vanished, the remaining residual alone
  would still cap rate at about `3.6 tok/s`.
- A stable `>2 tok/s` requires decode below `500 ms/token`, so Phase 4X needs
  at least about `105 ms/token` real saved time on N96, not just higher hit
  rate.
- A stable `>5 tok/s` requires decode near `200 ms/token`; expert-cache policy
  alone cannot credibly claim that until residual is attributed and reduced.

Hard gates for this phase:

- Cold start only.
- Total host RAM peak must remain below `15900000000` bytes, including page
  cache, pinned memory, mmap/file-backed pages, helper processes, and cgroup
  accounting.
- TTFT must stay within `1.20x` of the paired baseline.
- Quality must pass for `Please introduce France in a short paragraph.` and
  the general-prompt dev set before any held-out run.
- Optimizations must be prompt-general. Held-out prompts must not be used to
  tune packs, profiles, layer choices, cache budgets, thresholds, or admission
  policy.
- Accepted SOTA commits must include improvement size, env, exact commands,
  prompt split, RAM/TTFT/quality gate, run directories, and rollback commit in
  the commit body, then be pushed immediately.

Plan:

1. Reproduce the current stable baseline before changing runtime behavior.
   - Run one N96 cold-start dev prompt with `GGML_MOE_PHASE_REPORT=1`.
   - Record token rate, TTFT, decode time, prompt time, RAM peak, final
     `file/active_file/inactive_file`, VRAM hit rates, expert-pack bytes,
     `io_uring_wait`, staging wall, H2D wall, upgate wall, down wall, and CPU
     fallback counters.
   - Record the rollback commit and exact SOTA env.

2. Add residual attribution before any new cache policy.
   - Split decode wall into:
     - measured `upgate`;
     - measured `down`;
     - dense/attention/shared compute if available from graph/node timing;
     - CUDA synchronization gaps;
     - D2H/scatter not currently counted in MoE buckets;
     - scheduler/host bookkeeping;
     - sampler/output overhead.
   - Keep instrumentation default-off or low overhead under a profiling env.
   - Acceptance for instrumentation is no behavior change, same quality, and
     overhead within noise.

3. Audit CPU/defer GPU-extension coverage.
   - For each active expert operation, classify the actual path:
     `VRAM resident`, `dynamic VRAM cache hit`, `RAM tier`, `expert-pack
     iouring`, `pack mmap`, `GGUF mmap`, or `CPU fallback`.
   - Report the classification by layer, role, quant type, and prompt phase.
   - Any unclassified decode work must be treated as a profiling bug before
     optimization claims are made.

4. Only if residual is MoE scheduler or movement wait, test role-joint IO.
   - When routing for a layer is known, submit `gate/up/down` misses through a
     single scheduler view to improve queue depth and reduce stalls.
   - This is IO scheduling only. `down` compute still waits for the
     `up/gate` activation result.
   - Add fairness so down prefetch cannot starve current-layer `gate/up`.
   - The theoretical upper bound is the measured exposed `io_uring_wait` and
     staging wait that can be overlapped without increasing read bytes or TTFT.

5. Only if page cache is low-value during decode, test explicit RAM/VRAM tiers.
   - Replace low-value file cache with batchable expert storage, not scattered
     one-off entries.
   - Prefer compact layer/role slabs or adjacent pack ranges that can transfer
     to VRAM efficiently.
   - Measure evicted page-cache categories, refaults, direct reclaim, TTFT, and
     H2D wall. Reject any candidate that only shifts SSD wait into RAM pressure
     or H2D tail latency.

6. Validation ladder.
   - N32 dev smoke for parser, RAM, quality, and obvious regressions.
   - N96 dev paired A/B for the real decision.
   - N96 held-out paired A/B only after dev passes.
   - Commit and push immediately only for accepted improvements.
   - Default-off or revert rejected experiments and record the measured reason
     here.

Decision rules:

- If residual is mostly dense/attention/shared compute, the next plan should
  move away from expert-cache admission and toward non-MoE GPU graph/kernel
  optimization.
- If residual is mostly scheduler wait or uncovered MoE transfer, prioritize
  role-joint IO with fairness and better queue continuity.
- If residual is mostly RAM/page-cache churn, prioritize controlled RAM expert
  slabs and prompt-phase cache eviction.
- If all measured components are already near hardware limits, reaching
  `5 tok/s` requires a smaller expert representation, predictive prefetch that
  is validated on held-out prompts, or a different storage/compute format.

Phase 4X result: residual attribution corrected.

- Runtime change:
  - added prompt/decode phase totals to the existing
    `[kimi_cpu_moe_profile]` up_gate/down counters;
  - updated `.Agent/run-tools/kimi_dev_decode_bottleneck_summary.py` so reports
    include `cpu_moe_*` components and `residual_after_cpu_moe`.
- This is profiling-only and makes no SOTA claim. Default behavior is
  unchanged unless `GGML_KIMI_CPU_MOE_PROFILE` is enabled.
- Build:
  - `cmake --build build-cuda-batch -j 8` passed.
  - `build-cuda-batch/bin/test-kimi-deepseek2-guards` passed.

N96 cold-start attribution run:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4x-cpu-moe-phase-profile-n96-france`
- Report:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4x-cpu-moe-phase-profile-n96-france-analysis/report.md`
- Env additions:
  - `PROFILE=1`
  - `COPY_PROFILE=1`
  - `LLAMA_KIMI_GRAPH_PROFILE=1`
  - `GGML_KIMI_SPLIT_PROFILE=1`
  - `GGML_KIMI_SPLIT_PROFILE_TOP=80`
- Prompt:
  `Please introduce France in a short paragraph.`
- Quality: pass.
- Host RAM:
  - peak `12813234176` bytes, below the `15900000000` byte gate;
  - final file cache `12088602624` bytes;
  - `workingset_refault_file=0`.
- Timing:
  - TTFT `11913.25 ms`;
  - decode `55884.69 ms / 85 runs`;
  - token rate `1.52 tok/s`.
- Expert path:
  - CPU fallback `0`;
  - direct reads `0`;
  - iouring reads `85754`;
  - iouring bytes `491342774272`;
  - iouring wait `37717.144 ms`;
  - VRAM down hit `61.4%`;
  - VRAM upgate hit `43.5%`.

Corrected decode attribution:

| Component | ms/token | decode share |
|---|---:|---:|
| CPU MoE up_gate total | `459.570` | `69.9%` |
| CPU MoE up_gate CUDA extension | `454.829` | `69.2%` |
| iouring wait pressure signal | `443.731` | `67.5%` |
| CPU MoE down total | `175.414` | `26.7%` |
| CPU MoE down CUDA extension | `173.614` | `26.4%` |
| residual after CPU MoE | `22.482` | `3.4%` |

Interpretation:

- The old `residual_unattributed` value was a profiling artifact. The
  `up-gate-profile.csv` path covers only part of the up_gate work; it reported
  about `175.691 ms/token`, while full CPU/defer up_gate entry timing is
  `459.570 ms/token`.
- Dense/attention/shared CUDA graph work is not the main bottleneck in this
  profile. `GGML_KIMI_SPLIT_PROFILE` shows decode CUDA0 split wall is only
  `367.240 ms` total, about `4.3 ms/token`.
- The real bottleneck is the CPU/defer up_gate op using the GPU extension.
  Most of that time is inside its CUDA extension call and correlates with
  expert movement and low upgate hit rate.
- Therefore the next optimization should target complete up_gate movement and
  scheduling, not generic non-MoE graph kernels and not more down-only cache.

Next plan from Phase 4X:

1. Add complete up_gate subpath attribution.
   - Split the full CPU MoE up_gate CUDA extension time by layer, quant type,
     fused/unfused implementation path, cache hit/miss, iouring wait, H2D, and
     compute.
   - The current `up-gate-profile.csv` is insufficient because it misses about
     `459.570 - 175.691 = 283.879 ms/token` of full up_gate entry time.
   - Acceptance for instrumentation: no behavior change, quality pass, RAM
     below gate, and profiling overhead within noise.

2. Optimize only after the missing up_gate subpath is identified.
   - If the missing time is transfer wait, prioritize upgate-specific queue
     continuity and larger batch construction before adding down prefetch.
   - If the missing time is H2D tail latency, test staged slab layout or
     role-specific pinned reuse for upgate only.
   - If the missing time is compute/dequant, target the upgate CUDA kernel or
     quant representation.
   - If it is cache admission churn, change VRAM/RAM policy to favor upgate
     before compressing down cache.

3. The `>2 tok/s` bound is now clearer.
   - Current N96 decode is `657.467 ms/token`.
   - `>2 tok/s` requires below `500 ms/token`.
   - The first milestone must save at least `158 ms/token`, and the only
     measured component large enough is full up_gate.
   - Down-only optimization cannot reach `>2 tok/s` by itself.

## 2026-07-11 active goal and plan history

This section was the previous source of truth. It is retained as history and
experiment audit trail; the immediate goal and plan are now recorded in the
section above.

Active goal:

> On `vendor/kimi-deepseek-41d205-additive`, verify whether the DeepSeek SOTA
> architecture pattern transfers to Kimi: CPU/defer MoE remains the scheduler,
> while GPU becomes a stronger expert-cache, transfer, and compute extension for
> Kimi `gate/up/down`. The immediate target is a reproducible general-prompt gain
> toward stable `>2 tok/s`; the product target remains stable `>5 tok/s` for
> random user prompts on a 16 GB host-RAM machine with one 32 GB RTX 5090-class
> GPU.

Why this should be tested on Kimi:

- The useful DeepSeek lesson is not a gate-only hotpool. The useful lesson is
  the control split: CPU/defer owns routing/scheduling, while GPU handles hot
  expert residency, transfer, and compute whenever the extension can safely
  catch the request.
- Kimi already has near-zero decode CPU fallback in the current stable path, so
  the next gain is unlikely to come from simply moving more CPU math to GPU.
- Current Kimi bottleneck is exposed expert movement latency: `io_uring_wait`,
  small runtime batches, staging/H2D gaps, and incomplete overlap between
  routing, up/gate compute, down movement, and down compute.
- Therefore the Kimi-specific plan is to reduce exposed movement wait for
  `gate/up/down` together, not to optimize one role by hit rate alone.

Hard gates for every accepted result:

- Cold start only; no warm page cache or reused process state.
- Host RAM peak `<15900000000` bytes, including page cache, mmap/file-backed
  pages, pinned memory, process memory, helper processes, and cgroup/kernel
  accounting.
- Use VRAM aggressively, but do not trade VRAM hit rate for TTFT, RAM, or
  quality regressions.
- TTFT must be `<=1.20x` paired baseline.
- Mandatory quality prompt: `Please introduce France in a short paragraph.`
  must remain coherent and semantically correct.
- Optimization must be prompt-general. Dev prompts may guide design; held-out
  prompts are validation only and must not be used to tune packs, hotsets,
  thresholds, or layer selections.
- A result is SOTA only after paired baseline/candidate N96 dev plus N96
  held-out validation.
- Every accepted improvement must be committed and pushed immediately with exact
  env, command, prompt split, run path, RAM/VRAM metrics, TTFT, token-rate delta,
  quality result, and rollback commit.
- Failed candidates must be reverted or left default-off, with the measured
  rejection reason recorded here.

Current execution plan:

1. Finish Phase 4T observability before changing cache policy.
   - Add default-off phase reports gated by `GGML_MOE_PHASE_REPORT=1`.
   - Capture `before_prompt_eval`, `after_prompt_eval`, and `after_generation`
     memory state: `memory.current`, `memory.peak`, anon/file split,
     active/inactive file, major faults, refaults, direct reclaim, and direct
     steal.
   - Capture MoE deltas per phase: expert-pack hits/misses, direct/io_uring
     reads and bytes, submit/wait/H2D, staging ring waits, VRAM cache
     hits/misses, current-down overlap, and fallback counters.
   - Keep normal runtime behavior unchanged when the env flag is absent.

2. Re-profile the current stable SOTA with the new phase counters.
   - Run one N32 dev smoke to verify logging, quality, RAM, and parser output.
   - Run one targeted N96 diagnostic on the known TTFT long-tail prompt to
     separate prompt reclaim/refault cost from decode `io_uring_wait`.
   - Do not tune from held-out traces; use them only to identify whether the
     instrumentation explains a failure mode.

3. Rank the next optimization by removable seconds.
   - If phase counters show prompt/file-cache reclaim dominates, target prompt
     page-cache lifecycle and fallback elimination first.
   - If decode exposed `io_uring_wait` dominates, target same-layer miss
     co-submit, layer/role-aware RAM slabs, or pack-layout changes only where
     they can keep the queue fed asynchronously.
   - If staging/H2D dominates, target pinned/pageable strategy, larger coherent
     slabs, and H2D stream overlap before increasing SSD read volume.
   - If CPU fallback reappears, fix the exact unsupported type/role first and
     measure whether it is on the critical path.

4. Apply the DeepSeek-style GPU extension only where Kimi profiles justify it.
   - Test gate-only, up/gate paired, and up/gate/down joint residency as
     separate default-off A/B candidates.
   - Accept a candidate only when it reduces exposed decode time on general
     prompts, not merely SSD bytes or hit-rate counters.
   - Do not promote profile-specific packs, hotsets, or held-out-derived layer
     selections.

5. RAM/VRAM storage plan.
   - VRAM holds the hottest and most latency-critical experts.
   - RAM holds second-tier experts only when the data is batchable and replaces
     low-value decode-time file cache.
   - Prefer whole layer/role slabs or compact adjacent pack ranges over
     scattered single-expert RAM entries when fragmentation reduces batch size.
   - Before adding RAM-resident experts, prove which decode-time file-backed
     pages are low-value and safe to evict.

6. Validation and commit discipline.
   - N32 dev: quick screen for logging, quality, RAM, TTFT, and obvious
     regression.
   - N96 dev: real token-rate and bottleneck validation.
   - N96 held-out: final acceptance only.
   - Accepted candidate: commit and push immediately with full reproduction
     body.
   - Rejected candidate: revert or leave default-off, update this plan with the
     measured reason, and move to the next ranked bottleneck.

## 2026-07-11 Phase 4T observability result

Implementation status:

- Added default-off phase reporting gated by `GGML_MOE_PHASE_REPORT=1`.
- Normal runs without the env flag do not print phase counters and do not read
  cgroup memory stats from the hot path.
- Metrics parser now captures `kimi_phase_*`, `moe_phase_*`, and prompt-end
  `madvise(DONTNEED)` wall times.
- This is observability infrastructure only. It does not claim or promote a new
  SOTA.

Validation:

- Build: `cmake --build build-cuda-batch -j 8` passed.
- Guard: `build-cuda-batch/bin/test-kimi-deepseek2-guards` passed.
- Diff hygiene: `git diff --check` passed.
- Env-on France smoke:
  - Run:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4t-instrumentation-n48-france`
  - Command shape:
    `--mode dev --n 48 --runtime-max-sec 360 --extra-runtime-env GGML_MOE_PHASE_REPORT=1`
  - Result: quality `pass`, token rate `1.85 tok/s`, TTFT `7894.82 ms`,
    decode `25354.05 ms / 47`, memory peak `12722601984` bytes.
  - Parser captured `3` `kimi_phase` records and `24` `moe_phase` records.
- Default-off smoke:
  - Run:
    `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4t-default-off-n8-france`
  - Command shape: `--mode dev --n 8`, no `GGML_MOE_PHASE_REPORT`.
  - Result: quality `pass`, token rate `1.79 tok/s`, TTFT `7443.69 ms`,
    decode `3915.18 ms / 7`, memory peak `11.84 GiB`.
  - `rg "\[kimi_phase\]|\[moe_stream_batch_phase\]" stderr.txt` returned no
    matches.

Held-out diagnostic, not tuning input:

- Prompt: `test_reasoning_math_01`
  (`A train leaves at 3 PM and arrives 2 hours and 15 minutes later...`).
- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4t-instrumentation-n96-heldout-math-diagnostic`
- Command shape:
  `--mode test --n 96 --runtime-max-sec 360 --extra-runtime-env GGML_MOE_PHASE_REPORT=1`
- Result: quality `pass`, token rate `1.58 tok/s`, TTFT `214863.14 ms`,
  decode `38657.04 ms / 61`, memory peak `14.81 GiB`.
- Phase split:
  - Before prompt eval: `memory_current=11.345 GiB`, `file=11.121 GiB`,
    `pgmajfault=1196`.
  - After prompt eval: `memory_current=14.807 GiB`, `file=13.781 GiB`,
    `active_file=8.326 GiB`, `inactive_file=5.345 GiB`,
    `pgmajfault=15697`, `workingset_refault_file=18350`,
    `pgscan_direct=50933480`, `pgsteal_direct=28498889`.
  - Prompt MoE/expert-pack work was tiny: `24` iouring reads,
    `0.131 GiB`, `22.5 ms` iouring wait.
  - Prompt-end expert mmap `DONTNEED` took `852.524 ms`; dense mmap
    `DONTNEED` took `4.952 ms`.
  - Decode phase added `310.728 GiB` iouring bytes and `40.075 s` iouring
    wait, with only `79` additional major faults and `1279` additional file
    refaults between `after_prompt_eval` and `after_generation`.

Conclusion:

- The held-out TTFT long tail is not caused by expert-pack transfer during
  prompt eval. The phase counters show only `22.5 ms` expert-pack iouring wait
  in prompt, while cgroup memory reaches the 16 GB cap with large file-backed
  refault and direct reclaim counters.
- Decode token rate is still limited by exposed expert movement: the same run
  spends `40.075 s` in decode iouring wait for `61` decode steps.
- Next work must keep these two bottlenecks separate:
  - TTFT risk: identify and reduce prompt-phase GGUF/file-backed page-cache
    growth and refault/reclaim before accepting larger RAM tiers.
  - Token-rate risk: reduce decode exposed `io_uring_wait` via queue-fed
    gate/up/down scheduling, RAM/VRAM residency that is batchable, or pack
    layout changes.
- Do not use this held-out trace to choose hot experts, layer slabs, or cache
  thresholds. It only validates the failure mode and the new counters.

## 2026-07-11 Phase 4U next experiment: queue-fed decode IO

Dev profiling input:

- Prompt: `dev_photosynthesis_factual`
  (`Explain photosynthesis briefly.`).
- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4t-dev-photosynthesis-n96-profile`
- Command shape:
  `--mode dev --n 96 --profile --runtime-max-sec 600 --extra-runtime-env GGML_MOE_PHASE_REPORT=1`
- Result: quality `pass`, token rate `1.53 tok/s`, TTFT `7385.48 ms`,
  decode `61406.24 ms / 94`, memory peak `12630376448` bytes.

Bottleneck:

- Decode phase added `475905884160` bytes (`443.2 GiB`) of expert-pack
  iouring reads and `56.451 s` iouring wait.
- Decode total is `61.406 s`, so exposed expert movement is the dominant
  removable cost.
- Runtime IO queue is not saturated like the pure IO bench:
  `inflight_avg=3.77`, `inflight_max=8`, batch histogram mostly `2-8`.
- VRAM cache stats:
  - Down: `61.9%` hit rate, decode misses `15959`.
  - Upgate: `41.1%` hit rate, decode misses `53256`.
- CPU fallback is still zero in this run, so the next gain should not target
  CPU compute fallback.

Theory bound for the `>2 tok/s` short-term target:

- Current decode rate: `94 / 61.406 = 1.53 tok/s`.
- `2 tok/s` requires decode time `<=47.0 s`, so the experiment needs roughly
  `14.4 s` saved on this prompt.
- If the same `443.2 GiB` decode read volume reaches the previously measured
  high IO bench level of `10.4 GiB/s`, read wait lower bound is
  `443.2 / 10.4 = 42.6 s`.
- That saves about `13.8 s` versus the current `56.45 s` iouring wait, nearly
  enough by itself. A small additional reduction from better cache allocation
  or lower synchronization overhead could pass `2 tok/s`.

Immediate A/B, before writing runtime code:

1. Run an env-only queue-depth screen on this dev prompt:
   - Baseline: current default `GGML_MOE_IO_DEPTH=8`,
     `GGML_MOE_IO_REFILL_BATCH=4`.
   - Candidate:
     `GGML_MOE_IO_DEPTH=16`, `GGML_MOE_IO_REFILL_BATCH=8`,
     `GGML_MOE_PHASE_REPORT=1`.
   - Acceptance for this screen: quality pass, RAM `<15900000000`, TTFT not
     worse by `>20%`, lower decode time, higher `inflight_avg`, and lower
     phase `delta_iouring_wait_us`.

2. Interpret the result:
   - If depth/refill improves materially, test the same env on France + one
     more dev prompt, then N96 dev paired baseline/candidate.
   - If depth/refill does not improve, the queue is starved by scheduling
     readiness rather than the global depth cap. Move to default-off same-layer
     `up/gate/down` co-submit after routing.

3. Code path if env-only fails:
   - After routing/top-k for a layer is known, enumerate up, gate, and down
     misses together.
   - Submit ready expert-pack reads into one async scheduler without blocking
     earlier slices.
   - Preserve current-down overlap and per-slice readiness; do not wait for a
     whole coalesced group before allowing available up/gate work to run.
   - Add counters for submitted jobs, ready jobs, wait time, inflight depth,
     and role split.
   - Keep the path default-off until N32 dev A/B passes quality, RAM, TTFT, and
     decode-time gates.

Phase 4U.0 result: env-only `depth=16/refill=8` is rejected as SOTA.

- Candidate env:
  `GGML_MOE_IO_DEPTH=16`, `GGML_MOE_IO_REFILL_BATCH=8`,
  `GGML_MOE_PHASE_REPORT=1`.
- Paired baseline env:
  default `GGML_MOE_IO_DEPTH=8`, `GGML_MOE_IO_REFILL_BATCH=4`,
  `GGML_MOE_PHASE_REPORT=1`.
- Current commit: `811ac28dc`.
- Baseline run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4u-default-depth8-refill4-dev3-n96`
- Candidate runs:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4u-depth16-refill8-dev-france-japan-n96`
  and
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4u-depth16-refill8-dev-photosynthesis-n96`

Paired dev N96 results:

| prompt | baseline tok/s | candidate tok/s | baseline decode ms | candidate decode ms | baseline wait s | candidate wait s | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| dev_france_regression | 1.86 | 1.84 | 45728.20 | 46319.31 | 46.374 | 46.871 | regress |
| dev_japan_factual | 1.90 | 1.92 | 41024.81 | 40723.19 | 41.297 | 39.632 | small gain |
| dev_photosynthesis_factual | 1.82 | 1.84 | 51723.87 | 51016.66 | 52.728 | 51.970 | small gain |

Aggregate:

- Quality: `3/3` pass for both baseline and candidate.
- RAM: all runs stay around `11.7-11.9 GiB`, below the 16 GB gate.
- TTFT: candidate does not regress; it is slightly lower on these three
  prompts.
- Decode time: `138476.88 ms -> 138059.16 ms`, only `417.72 ms` saved
  (`0.30%`).
- Token rate: mean `1.86 -> 1.87 tok/s`, too small and prompt-mixed.
- Queue shape: `inflight_max` can rise from `8` to `12`, but `inflight_avg`
  stays around `3.8`, and batch histogram still has no `9-16` batch bucket.

Conclusion:

- The global io_uring depth/refill cap is not the main reason the runtime fails
  to reach pure IO bench throughput.
- The queue remains underfed because the scheduler often does not expose enough
  ready expert reads at once.
- Do not promote `depth=16/refill=8` as SOTA.
- Next implementation target is a default-off same-layer `up/gate/down`
  co-submit path that increases ready jobs after routing without blocking
  earlier slices or damaging current-down overlap.

## 2026-07-11 Phase 4U.1 plan: fused-path co-submit audit

Current code audit:

- Existing `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` is a single-gate expert path in
  `moe_stream.cu`. It does not cover the dominant Kimi SOTA path, which is
  `ggml_cuda_moe_stream_up_gate_batch`.
- Existing `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` was already rejected:
  it caused CUDA OOM on two N32 dev prompts and a severe slowdown on the third.
  Do not retry it as the next candidate.
- Existing `GGML_MOE_ROUTE_GROUP_NATIVE_PARITY=1` has useful route-group
  planning code, but its current entry point is gate-name based and is not
  called from the fused up/gate batch path.
- The route-group native planner also uses each pack entry's actual `nbytes`
  as its cache key. The fused up/gate batch often uses a shared
  `max(up_bytes, gate_bytes)` cache slot for both roles. If reused blindly,
  an early up/gate preload can land in a different cache and will not be hit by
  the later fused up/gate stage.

Design constraint for any new co-submit implementation:

- Hook must be inside `ggml_cuda_moe_stream_up_gate_batch` after active experts
  are known and before normal up/gate staging starts.
- The hook must use the exact same cache/key convention as fused up/gate:
  `up_key_name`, `gate_key_name`, and the fused batch's `cache_slot_bytes`
  cache for up/gate.
- Down jobs may use the down tensor's natural cache size, but must not duplicate
  current-down overlap work already submitted.
- The path must be default-off, e.g.
  `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT=1`.
- The first implementation may be a shadow/counter probe only. A real copy path
  is allowed only after counters prove it can create larger ready job groups
  without increasing exposed wait.
- It must not block on a whole co-submit group before available up/gate work can
  run.
- It must record enough evidence for acceptance/rejection:
  planned jobs by role, cache hits/misses, pack misses, submitted jobs, copied
  bytes, iouring batches, failures, and whether the later fused stage actually
  hit the preloaded slots.

Immediate implementation step:

1. Extend `.Agent/run-tools/kimi-general-prompt-repro.sh` metrics parsing for
   existing co-submit and route-group lines:
   - `gate/up/down cosubmit`;
   - `route group native parity`;
   - `route group down queue`;
   - `fused upgate/down shadow`;
   - `down demand queue`.
2. Run a default-off shadow probe on a dev prompt with:
   - `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1`;
   - `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW_OUT=<run>/fused-upgate-down-shadow.csv`;
   - `GGML_MOE_PHASE_REPORT=1`.
3. If the shadow shows large same-layer plannable jobs but current-down overlap
   already covers most down misses, implement only the fused up/gate-side
   cache-compatible hook first.
4. If the shadow shows no material plannable jobs beyond current-down overlap,
   stop this direction and move to RAM/VRAM cache layout work.

## 2026-07-11 goal: Kimi CPU/defer + GPU-extension path

Goal:

> Use the current Kimi CPU/defer MoE scheduler as the control path, and make the GPU expert-cache/compute extension more effective for Kimi `gate/up/down` experts. The near-term target is a reproducible, general-prompt SOTA above `2 tok/s`; the product target remains stable `>5 tok/s` for random user prompts on a 16 GB host-RAM machine with one 32 GB RTX 5090-class GPU.

How the DeepSeek result transfers to Kimi:

- The transferable part is the architecture: CPU/defer owns routing and scheduling, while GPU acts as a fast expert-residency, transfer, and compute extension.
- Do not assume Kimi will benefit from a DeepSeek-style gate-only hotpool. Current Kimi profiles show decode CPU fallback is already near zero in the stable path; the main remaining cost is exposed expert movement wait.
- For Kimi, the useful optimization target is `gate/up/down` together: reduce `io_uring_wait`, staging/H2D gaps, small read batches, and missed overlap on general prompts.
- A higher VRAM/RAM hit rate is not enough. It must lower exposed decode time and pass held-out prompt quality.

Hard gates for this goal:

- Cold start only; no warm page cache and no reused process state.
- Host RAM peak `<15900000000` bytes, including page cache, mmap/file-backed pages, pinned memory, helper processes, and cgroup/kernel accounting.
- TTFT `<=1.20x` the paired baseline.
- Mandatory quality prompt: `Please introduce France in a short paragraph.` must remain coherent and semantically correct.
- Optimization must be prompt-general. Dev prompts can guide design; held-out prompts are validation only and must not train/tune packs, hotsets, or thresholds.
- A result is SOTA only after paired baseline/candidate N96 dev plus N96 held-out validation.
- Every accepted improvement must be committed and pushed immediately with exact env, command, prompt split, run path, RAM/VRAM metrics, TTFT, token-rate delta, quality result, and rollback commit.
- Failed candidates must be reverted or left default-off and documented with the measured rejection reason.

Execution plan:

1. Reproduce and profile the current stable Kimi SOTA.
   - Run N96 cold-start dev and one held-out check from the exact branch/commit.
   - Record per-token decomposition: routing/top-k, expert read wait, pinned staging, H2D, up/gate compute, down compute, CPU fallback, and synchronization gaps.
   - Produce a layer/role wait table so changes target removable seconds instead of hit-rate intuition.

2. Audit the CPU/defer GPU-extension boundary.
   - Confirm which `gate/up/down` paths are GPU extension, which are CPU/defer control only, and whether any residual CPU compute/fallback remains during decode.
   - If a role still falls through to CPU on general prompts, fix that first only when its measured time is on the critical path.
   - If CPU fallback is already zero, skip CPU work and target transfer scheduling/residency.

3. Rebalance VRAM by exposed wait, not by equal hit rate.
   - Use the wait-weighted layer/role profile to decide whether `up/gate` needs more cache than `down`, or whether specific layers need protection.
   - Test env-only cache split changes first with N32 dev A/B.
   - Promote only if token rate, TTFT, RAM, and quality all pass; otherwise document and reject.

4. Replace low-value decode RAM with explicit expert data.
   - Identify decode-time file-backed pages that are not useful after prompt.
   - Free low-value cache only when it does not cause refaults or TTFT regressions.
   - Use the freed RAM for batchable second-tier expert storage: prefer whole layer/role slabs or compact adjacent packs over scattered single experts.
   - Measure whether RAM residency reduces exposed `io_uring_wait`; do not accept changes that only move bytes from SSD to RAM while increasing staging/H2D or synchronization cost.

5. Improve read scheduling only where the runtime can stay async.
   - Avoid the rejected blocking adjacent coalescer pattern.
   - If co-submit/coalesce is retried, it must preserve per-slice readiness and current overlap.
   - Test same-layer `up/gate/down` miss co-submit and one-layer-ahead prefetch only after a profile proves enough predictable work to keep queues fed.

6. Validation ladder and rollback.
   - N32 dev: quick screen for quality, RAM, TTFT, and obvious regressions.
   - N96 dev: real performance and bottleneck validation.
   - N96 held-out: final acceptance only.
   - Accepted candidate: commit and push immediately with full reproduction details.
   - Rejected candidate: revert or keep default-off, then update this plan with the failure reason and next ranked bottleneck.

## 2026-07-11 current goal and execution plan

Current goal:

> Improve Kimi's **general-prompt** decode speed on `vendor/kimi-deepseek-41d205-additive` by extending the CPU/defer MoE scheduler's GPU expert-cache path, while preserving cold-start correctness under the strict 16 GB host RAM gate. The immediate engineering target is a reproducible held-out gain toward stable `>2 tok/s`; the product target remains stable `>5 tok/s` on random user prompts with 16 GB host RAM and one 32 GB RTX 5090-class GPU.

Scope of this cycle:

- Do not copy the DeepSeek gate-hotpool result blindly. For Kimi, the useful transferable idea is CPU/defer orchestration plus a stronger GPU expert-cache/compute extension for gate/up/down as a group.
- Current evidence says decode CPU fallback is already near zero in the stable SOTA path; the dominant remaining bottleneck is exposed expert movement wait: `io_uring_wait`, small runtime read batches, staging/H2D synchronization, and incomplete overlap.
- RAM tier experiments with scattered hot experts and small single-role slabs have not produced stable gains. More host RAM must be used only when it replaces low-value file cache with batchable, latency-critical expert data.
- Pack-layout profiling shows a real read-extent bound, but layout alone cannot help unless runtime can merge adjacent/low-gap expert spans without recreating the old blocking coalescer failures.

Hard acceptance gates:

- Cold start only; no warm page cache and no reused process state.
- Host RAM peak `<15900000000` bytes, including page cache, mmap/file-backed pages, pinned memory, helper processes, and cgroup/kernel accounting.
- TTFT `<=1.20x` paired baseline.
- Mandatory quality prompt: `Please introduce France in a short paragraph.` must remain coherent and semantically correct.
- Optimization is for general prompts. Dev prompts may guide design; held-out prompts are only for validation and cannot be used to tune profiles or packs.
- A result is SOTA only after paired baseline/candidate N96 dev plus N96 held-out validation.
- Every accepted improvement must be committed and pushed immediately with exact env, command, prompt split, run path, RAM/VRAM metrics, TTFT, token-rate delta, quality result, and rollback commit.

Next plan:

1. Finish Phase 4O offline safe-adjacent bound before writing runtime code.
   - Use Phase 4E dev traces only.
   - Evaluate same-tensor adjacent/zero-gap groups with a conservative `16 MiB` span cap and `jobs <= 8`.
   - Compare current physical order with dev-trained per-tensor layouts: `expert_id`, `frequency`, and `greedy_pair`.
   - Record read jobs, safe extents, extents saved, grouped jobs, grouped GiB, and role/op distribution.

2. Decide whether a narrow async adjacent-span coalescer is worth implementing.
   - Proceed only if the offline bound shows either `>10%` read-plan reduction in small batches or a plausible `>=1s` N32 wait reduction.
   - Stop if the safe subset is too small, even if the ideal layout bound looks large.
   - If proceeding, write a source-level design first: grouped SQE structure, slice-to-H2D mapping, event lifetime, fallback path, counters, and rollback.

3. If Phase 4O passes the bound, implement default-off only.
   - No blocking `pread`.
   - No tiny coalesced-slot pool.
   - No waiting for every slice in a coalesced group before earlier slices can be consumed.
   - No broad cross-role merge that damages current up/gate/down overlap.
   - Validate with N32 dev A/B before any N96 run.

4. If Phase 4O fails, move to the next ranked bottleneck instead of forcing the idea.
   - Candidate fallback paths: VRAM cache budget rebalancing by exposed wait, layer/role RAM slabs only when full-batch coherent, or prediction/prefetch only after an acceptance-rate experiment proves usefulness on dev prompts.
   - Every fallback must start from a measured removable-seconds bound and update this plan before implementation.

## Current goal and next plan: CPU/defer GPU-extension for Kimi

Goal:

> On `vendor/kimi-deepseek-41d205-additive`, determine whether the DeepSeek-style idea of keeping CPU/defer MoE as the scheduler while expanding the GPU expert-cache/compute extension can improve Kimi's **general-prompt** decode speed. The immediate target is a reproducible held-out improvement toward `2 tok/s`; the long-term target remains stable `>5 tok/s` on random user prompts with 16 GB host RAM and one 32 GB RTX 5090-class GPU.

Why this is useful for Kimi:

- The transferable idea is **not** "copy DeepSeek gate-only hotpool".
- The useful part is that CPU/defer can remain the control path while GPU handles more expert residency, transfer, and compute.
- Current Kimi evidence shows the bottleneck is mostly exposed expert movement wait: `io_uring_wait`, staging/H2D, small runtime batches, and incomplete overlap.
- Recent profiling shows decode CPU fallback is already near zero, so the next gain must reduce exposed transfer/staging wait rather than only moving more work away from CPU.

Non-negotiable acceptance gates:

- Cold start only.
- Host RAM peak `<15900000000` bytes, including page cache, mmap pages, pinned memory, process memory, helpers, and kernel/cgroup accounting.
- TTFT `<=1.20x` paired baseline.
- France regression prompt must remain semantically correct: `Please introduce France in a short paragraph.`
- Held-out general prompts must pass quality and are not allowed for tuning.
- A result is SOTA only after paired baseline/candidate N96 dev plus N96 held-out validation.
- Accepted improvements must be committed and pushed immediately with exact env, commands, prompt set, run path, RAM/VRAM metrics, TTFT, token-rate delta, quality result, and rollback commit.
- If speed, quality, RAM, or TTFT regresses, revert or keep the code default-off and document the rejection.

Immediate execution plan:

1. Phase 4H `blk.1 gate` profile-only VRAM-protection is rejected.
   - N96 dev mean token rate regressed from `1.803` to `1.783`.
   - Decode sum increased by `2.151s`.
   - `iouring_wait` increased by `3.404s`.
   - SSD bytes and upgate/down hit rates were unchanged, so the slab did not reduce the real bottleneck.
   - Do not run held-out and do not promote the profile.

2. Stop adding isolated single-role slabs unless a new profile proves a much larger removable bound.
   - The measured `blk.1 gate` upper bound was too small and did not translate to exposed-wait reduction.
   - Future residency work must target cross-role critical-path stalls, not one role in isolation.

3. Move to the next higher-upside path:
   - same-layer aggressive co-submit of up/gate/down misses after routing;
   - layer/role-aware RAM cache replacing low-value decode-time file cache;
   - pack-layout changes that increase batchable contiguous reads instead of fragmenting SSD/RAM traffic.

4. Before every implementation step, update this plan with:
   - current bottleneck in seconds/token or aggregate N96 seconds;
   - theoretical upper bound from removable `iouring_wait`, staging/H2D bytes, or compute time;
   - exact experiment command and rollback point.

5. After every experiment, record:
   - per-prompt metrics and answer quality;
   - mean/median/min token rate;
   - TTFT ratio;
   - host RAM and decode page-cache/file/anon distribution;
   - VRAM expert-cache hit/miss/preload/pinned stats;
   - `iouring_wait_us`, SSD bytes, RAM->VRAM H2D bytes, staging wall, and CPU fallback counters;
   - accept/reject decision.

## 2026-07-10 goal and execution plan refresh

Active goal:

> On the current Kimi vendor branch, validate and implement the transferable part of the DeepSeek SOTA idea: CPU/defer MoE remains the scheduler, while GPU becomes a more complete expert-cache/compute extension for Kimi gate/up/down. The optimization target is a reproducible general-prompt token-rate gain under a strict 16 GB host RAM cap and a single 32 GB RTX 5090-class GPU, without losing semantic quality or increasing TTFT by more than 20%.

Current technical judgment:

- The DeepSeek gate-hotpool result is relevant to Kimi at the architecture level: the CPU/defer path can call a GPU expert-cache extension instead of falling through to slow CPU work.
- It should not be copied as a gate-only strategy without evidence. Kimi's current bottleneck is exposed expert movement wait, especially `io_uring_wait` caused by up/gate/down miss scheduling, small runtime batches, and incomplete overlap.
- The next accepted gain must reduce exposed wait on general prompts. Higher hit rate, lower SSD bytes, or a better single prompt is insufficient by itself.
- The short-term engineering target is to make stable general-prompt decode exceed `2 tok/s`. The long-term product target remains stable `>5 tok/s` for random user prompts on 16 GB host RAM + 32 GB VRAM.

Hard constraints for every experiment:

- Cold start only; no warm page cache or reused process state.
- Host RAM peak `<15900000000` bytes, including mmap/file cache, pinned staging, page cache, helper processes, and kernel/cgroup accounting.
- Use as much VRAM as safely possible, but do not trade VRAM hit rate for correctness or TTFT regressions.
- Quality gate must include `Please introduce France in a short paragraph.` and held-out general prompts; output must be coherent and semantically correct.
- TTFT must be `<=1.20x` the paired baseline.
- Dev prompts may guide optimization; held-out prompts are used only for final validation.
- Every accepted SOTA must be reproducible: commit and push immediately with env, command, prompt set, run path, RAM/VRAM metrics, TTFT, quality result, token-rate delta, and rollback commit in the commit body and plan.

Execution plan from here:

1. Re-establish the current reproducible baseline.
   - Use the current branch and pinned control profile.
   - Run N96 cold-start paired baseline on the dev prompt set.
   - Record per-prompt token rate, TTFT, decode time, quality answer, host RAM, file/anon breakdown, VRAM cache stats, expert-pack bytes, `iouring_wait_us`, H2D, pinned staging, and CPU fallback counters.
   - Treat the current `1.8-1.9 tok/s` France-like result as historical until reproduced by the exact command on this branch.

2. Locate the exposed critical path before changing code.
   - Break down each token into routing/top-k, up/gate expert read, up/gate H2D/staging, up/gate compute, down expert read, down H2D/staging, down compute, CPU fallback, and synchronization gaps.
   - For each layer/role, record whether the stall is caused by missing residency, queue starvation, small batch size, H2D serialization, compute, or fallback.
   - Rank optimization candidates by removable seconds, not by intuition or hit rate alone.

3. Test the DeepSeek-style gate extension hypothesis on Kimi.
   - Measure how much gate work still executes on CPU/defer slow paths.
   - Compare gate-only VRAM/RAM residency against up/gate paired residency and up/gate/down joint residency.
   - Accept gate-only work only if it reduces exposed wait and improves held-out token rate; otherwise keep the useful part as "GPU extension under CPU/defer scheduler" and optimize all roles together.

4. Optimize RAM/VRAM layout explicitly.
   - VRAM holds the hottest and most latency-critical experts.
   - RAM holds second-tier experts only when they are batchable and reduce exposed wait versus SSD.
   - Replace low-value decode-time file cache with explicit expert cache only after proving those file-backed pages are not needed during decode.
   - Prefer layer/role slabs for high-miss critical layers when random hot entries fragment IO or reduce batch size.

5. Reduce `io_uring` queue starvation.
   - Test more aggressive same-layer co-submit of up/gate/down misses after routing is known.
   - Test one-layer-ahead prefetch only when a predictor or route history gives enough accuracy to avoid wasting RAM/VRAM bandwidth.
   - Keep all speculative/predictive paths default-off until N32 dev A/B proves they reduce exposed wait without quality or TTFT regressions.

6. Validation ladder.
   - N32 dev: cheap screen for quality, RAM, and obvious regressions.
   - N96 dev: verify real token-rate and bottleneck changes.
   - N96 held-out: only final acceptance; no tuning based on held-out traces.
   - Accepted candidate: commit and push immediately.
   - Rejected candidate: revert or leave default-off, document measured reason and rollback point.

Immediate next action:

- `all-1200-minp4` RAM tier has been rejected after N96 held-out validation. It reduced SSD bytes but did not reduce exposed `io_uring_wait` on held-out prompts and slightly regressed median/mean token rate.
- Phase 4F control profiling shows no CPU fallback; the next work must target tensor staging, upgate kernel/wait, and cache-budget allocation rather than simply adding more RAM tier.
- Immediate next A/B candidates:
  - rebalance the existing 15 GiB VRAM expert-cache budget between upgate and down, with an explicit removable-wait bound before running;
  - test small layer/role slabs for top staging rows such as `blk.1 gate`, `blk.4 down`, and `blk.6 down`, only if they fit by replacing lower-yield cache entries.

## Current execution goal

Goal for the current optimization cycle:

> Verify whether the DeepSeek-style CPU/defer main path plus GPU expert-cache extension can produce another reproducible Kimi gain, and if so implement the smallest safe change that reduces exposed expert-transfer wait while preserving general-prompt quality under the 16 GB host RAM gate.

This is not a plan to copy DeepSeek's gate-only hotpool blindly. For Kimi, the current evidence says the bottleneck is mostly gate/up/down miss scheduling and exposed `io_uring_wait`, not standalone gate CPU compute.

Done criteria:

- Keep the active branch `vendor/kimi-deepseek-41d205-additive`.
- Keep every risky change default-off until it has paired baseline/candidate evidence.
- Use dev prompts for design and held-out prompts only for final validation.
- A candidate is accepted only when it improves reproducible general-prompt token rate, passes semantic quality, stays below `15900000000` bytes host RAM, keeps TTFT within `+20%`, and is committed/pushed with full reproduction details.
- A candidate that only improves one prompt, only improves hit rate, or only improves a noisy single run is rejected or left default-off.

## Immediate execution plan

1. Finish Phase 4B fused-path shadow documentation.
   - Record that the shadow code is diagnostic/default-off.
   - Record N32 dev3 quality, token rate, RAM, TTFT, and shadow cache-hit evidence.
   - Commit and push only as diagnostic infrastructure if build and smoke remain clean.

2. Decide whether actual fused up/gate/down co-submit is worth implementing.
   - Proceed only if the shadow shows a large same-layer miss group that current-down overlap does not already cover.
   - The expected benefit must be stated before implementation as a hard upper bound from `iouring_wait_us`, active expert bytes, and observed plannable rows.
   - If implemented, start with default-off env and N32 dev3 A/B before any N96 run.

3. If fused co-submit has weak upside, move to RAM/VRAM cache co-design.
   - Replace low-value decode-time file cache with explicit expert cache only when profiling proves those pages are not needed by decode.
   - Prefer layer/role-aware slabs for high-miss critical layers over random hot expert insertion.
   - Measure whether RAM resident experts reduce exposed wait rather than merely moving bytes from SSD to RAM.

4. Finalize only on general prompts.
   - Dev prompt gains can guide implementation.
   - SOTA claims must be based on held-out general prompts, not France-only or prompt-specific pack behavior.

## Goal

Optimize Kimi under the actual runtime architecture:

- CPU/defer MoE remains the main scheduling path.
- GPU is used as an expert-cache/compute extension for that CPU/defer path.
- The next gains should come from better gate/up/down residency and transfer scheduling, not from assuming a GPU-primary backend with CPU fallback.

Target environment:

- Host RAM hard limit: `<16 GB`, including page cache, pinned buffers, mmap pages, process memory, kernel cgroup memory, and helpers.
- GPU: single RTX 5090-class 32 GB card.
- Runs: cold start, `MemorySwapMax=0`.
- Workload: general/random user prompts, not prompt-specific France tuning.
- Quality: coherent semantic output; `Please introduce France in a short paragraph.` remains a mandatory regression prompt.
- TTFT: must not rise more than `20%` versus paired baseline.
- Reproducibility: accepted improvements must be committed and pushed with exact command/env/run artifacts/rollback.

Primary performance goal:

- Improve stable held-out general-prompt token rate, prioritizing minimum token rate across the held-out set.
- Long-term target remains `>5 tok/s`, but every step must be justified by current bottleneck evidence and accepted only if reproducible.

## Architecture hypothesis

DeepSeek's large gate-hotpool gain came from a specific shape:

- `n_cpu_moe=40` placed many MoE experts on the CPU/defer path.
- Gate was on a slow CPU/defer path before the GPU extension caught it.
- Moving gate hot experts to VRAM and computing gate on GPU converted a CPU-bound critical path into a GPU-resident path.

Kimi is different:

- Kimi decode already routes gate/up/down through CPU/defer orchestration into GPU extension paths.
- Existing Kimi SOTA uses expert pack, io_uring, pinned staging, VRAM cache, current-down overlap, Q4 down batch, and RAM tier.
- The dominant Kimi symptom is not raw gate CPU compute; it is expert movement and exposed `io_uring_wait`, especially when up/gate/down misses create critical-path stalls.

Therefore, the useful transferable idea is not "put all gate in VRAM" by itself. The useful idea is:

> Treat CPU/defer MoE as the scheduler and make the GPU extension more complete, better cached, and less IO-stalled for gate/up/down as a group.

## Current baseline evidence to carry forward

Recent merge branch evidence:

- Branch: `vendor/kimi-deepseek-41d205-additive`
- Commit: `4a20433748a6efa53f1d93d305906abfd5cb5d0e`
- Code merge commit: `9cb5e745468868a844d5abf4ce1d8ba46818d309`
- Kimi b4ef control with correct CUDA batch defines:
  - run 1: `/root/lfz/runs/vendor-kimi-token-rate/20260710-b4ef-control-batch-france-n96-112525`
  - token rate `1.91 tok/s`, `iouring_wait_us=46594173`
  - run 2: `/root/lfz/runs/vendor-kimi-token-rate/20260710-b4ef-control2-batch-france-n96-113502`
  - token rate `1.78 tok/s`, `iouring_wait_us=49347929`
- Merge branch run:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-merge-control2-france-n96-112932`
  - token rate `1.89 tok/s`, `iouring_wait_us=47146856`

Interpretation:

- France N96 single-run token rate is currently noisy because `io_uring_wait` varies materially.
- Merge branch preserves Kimi semantic output, slot counts, hit rates, pack entries, and batch histograms.
- Future SOTA claims must use paired baseline/candidate runs and report variance, not a single run.

## Non-negotiable acceptance gates

An optimization is accepted only if all are true:

- Source state is clean or only contains explicitly ignored/untracked measurement artifacts.
- Candidate is committed and pushed immediately once accepted.
- Reproduction command, env, prompt set, model path, pack paths, profiles, cgroup settings, run path, and rollback commit are recorded.
- Cold-start paired baseline and candidate are run close together on the same machine state.
- Host RAM peak stays under `15900000000` bytes with `MemorySwapMax=0`.
- TTFT ratio is `<=1.20` versus paired baseline.
- France prompt quality passes.
- Held-out general test-set quality passes.
- Token-rate improvement is visible on held-out test metrics, especially min token rate.
- CPU fallback/direct-read integrity checks do not regress.
- If performance, TTFT, RAM, or quality regresses, the change is reverted or left default-off and documented as rejected.

## Phase 0: Paired general baseline and bottleneck profile

Purpose: avoid optimizing France-specific or IO-noise artifacts.

Actions:

1. Re-run the existing dev/test prompt split from `.Agent/evals/` if present.
2. If the split is missing, create `.Agent/evals/kimi-general-dev-prompts.jsonl` and `.Agent/evals/kimi-general-test-prompts.jsonl`.
3. Do not use held-out test prompts for route/hotset/cache design.
4. Run cold-start N96 paired baseline on dev prompts and a final held-out check.
5. Record per-prompt answer text, quality verdict, TTFT, decode time, token rate, RAM peak, file/anon/kernel breakdown, VRAM slots/hit rates, expert-pack bytes, `iouring_wait_us`, pinned staging stats, current-down overlap stats, and CPU fallback by tensor type if counters are available.

Output:

- `.Agent/runs/<date>-kimi-cpu-defer-gpu-extension-baseline/report.md`
- A table with dev and held-out test mean/median/min token rate.

Acceptance:

- No code behavior changes in Phase 0 unless counters are default-off.
- Baseline must establish current variance band before comparing candidates.

## Phase 0 dev baseline result: 2026-07-10 12:55 CST

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase0-dev-baseline-125550`

Source state:

- Branch: `vendor/kimi-deepseek-41d205-additive`
- Commit before this result note: `5a694ee9dc13c90bc69e976bc63ee91fcd4f89a3`
- Mode: cold-start dev prompt sweep, `MemoryMax=15900000000`, `MemorySwapMax=0`, N96, 32 threads, pinned slots 12, current RAM tier profile enabled.

Command:

```bash
OUT=/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase0-dev-baseline-125550
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root "$OUT" \
  --mode dev \
  --n 96 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 900 \
  --extra-runtime-env "GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=$OUT/ram-batch-profile.csv"
```

Result table:

| prompt | quality | tok/s | TTFT ms | decode ms | decode tokens | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 1.91 | 8233.46 | 44544.32 | 85 | 13.56 | 46.45 | 455.1 |
| dev_japan_factual | pass | 1.95 | 8321.95 | 39910.16 | 78 | 13.56 | 42.01 | 413.6 |
| dev_photosynthesis_factual | pass | 1.80 | 7684.10 | 52276.97 | 94 | 13.41 | 54.37 | 501.2 |
| dev_linear_equation | pass | 1.60 | 11051.04 | 21207.11 | 34 | 13.56 | 25.15 | 280.1 |
| dev_python_reverse | pass | 1.72 | 8724.06 | 55242.75 | 95 | 13.56 | 60.08 | 549.7 |
| dev_zh_france | pass | 1.91 | 7714.09 | 25708.26 | 49 | 13.41 | 27.70 | 269.2 |
| dev_mixed_summary | pass | 1.79 | 11166.61 | 27421.46 | 49 | 13.56 | 31.17 | 330.8 |

Aggregate:

- Token rate: min `1.60`, median `1.80`, mean `1.81`, max `1.95` tok/s.
- TTFT: min `7684.10`, median `8321.95`, mean `8985.04`, max `11166.61` ms.
- RAM peak: min `13.41`, median `13.56`, max `13.56` GiB, under the 16 GB gate.
- `iouring_wait_us`: min `25.15s`, median `42.01s`, mean `40.99s`, max `60.08s`.
- Quality: all 7 dev prompts passed their automatic keyword gates and produced coherent outputs.

Key counters from the France regression run:

- `expert_pack_0`: `iouring_reads=85185`, `iouring_bytes=488667217920`, `iouring_wait_us=46450869`, `direct_reads=0`, `read_failures=0`.
- `expert_pack_iouring_0`: `inflight_avg=3.81`, `inflight_max=8`, `batch_hist=1:385,2-4:6594,5-8:8224,gt32:176`.
- `vram_upgate`: `slots=1735`, `hits=42389`, `misses=55003`, `hit_rate=43.5%`.
- `vram_down`: `slots=723`, `hits=29579`, `misses=18573`, `preloads=11676`, `hit_rate=61.4%`.
- `current_down_overlap`: `planned_jobs=11676`, `completed_jobs=11676`, `cache_hits=8044`, `worker_us=6418996`.

Interpretation:

- The current general dev baseline is stable enough for comparison and is not France-only: all 7 dev prompts pass quality, and token rate ranges from `1.60` to `1.95` tok/s.
- The next optimization should target exposed expert movement wait, especially up/gate misses and small runtime IO batches. Up/gate hit rate remains much lower than down hit rate on every dev prompt (`36.6%` to `46.5%` for up/gate versus `56.2%` to `62.4%` for down).
- RAM has headroom under the 16 GB hard cap in this profile, but any extra RAM tier must prove it reduces exposed wait, not just page cache or SSD bytes.
- Held-out test prompts must not be used for hotset/profile design. The current held-out file was inspected during setup, so before making a final SOTA acceptance claim, create or refresh a sealed held-out set and record that replacement.

Immediate next step:

- Run Phase 1 role/layer exposed-wait profiling on dev prompts only, then choose between gate-focused, up/gate-focused, or joint up/gate/down cache changes based on measured exposed wait rather than hit rate alone.

## Phase 1: Determine whether Kimi still has a gate-specific bottleneck

Question:

- Does Kimi still have exposed gate miss/compute time comparable to the old DeepSeek gate CPU/defer bottleneck?

Method:

1. Add or reuse default-off counters for per-layer/role exposed wait: gate read wait, up read wait, down read wait, gate/up/down compute time, role-specific cache hit/miss, and whether a miss is on the current token critical path.
2. Run n32 dev profile first, then n96 only if overhead is acceptable.
3. Rank `(layer, role)` by exposed wait, not by hit rate alone.

Decision:

- If gate exposed wait is small, do not spend the next cycle on gate-only VRAM hotpool.
- If a few gate layers dominate exposed wait, test a small gate-only hotset A/B, default-off.
- If up/gate jointly dominate, proceed to Phase 2.

## Phase 1 result: n32 dev role/layer profile

Timestamp: 2026-07-10 13:08 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-cpu-defer-gpu-ext-phase1-profile-n32-dev3-130834`

Command shape:

- Same cold-start 16GB cgroup runner as Phase 0.
- N32, `PROFILE=1`, all 7 dev prompts, current RAM tier profile enabled.
- First 3 prompts were run first, then the same out root was resumed with `--skip-existing` for the remaining 4 prompts.

Quality note:

- 6/7 automatic quality gates passed.
- `dev_linear_equation` failed only because N32 truncated the answer after `**x =`; the N96 Phase 0 baseline for the same prompt passed with `x = 7`.
- Therefore this profile is valid for bottleneck localization only. It is not an acceptance-quality run and must not be used as SOTA evidence.

Aggregate profile metrics:

- Token rate under profiling: min `1.46`, median `1.68`, mean `1.65`, max `1.84` tok/s.
- TTFT under profiling: min `7722.46`, median `9463.29`, mean `9819.32`, max `12411.67` ms.
- RAM peak: min `13.44`, median `13.59`, max `13.59` GiB.
- `iouring_wait_s`: min `17.32`, median `19.15`, mean `19.78`, max `23.41`.
- `iouring_bytes_gib`: min `192.7`, median `207.6`, mean `222.5`, max `265.5`.
- `fallback-profile.csv` rows: `0`, so this profile did not expose CPU fallback.

Role/layer evidence:

- `up-gate-profile.csv`: `6083` decode rows, `28` layers, all `n_active=8`.
- `down-batch-profile.csv`: `13027` decode rows, all decode rows are `down`; prompt rows contain gate/up/down and are excluded from decode down ranking.
- Total n32 dev exposed up/gate wait: `18707.3 ms`; total up/gate wall: `35260.5 ms`.
- Total n32 dev decode down stage: `32703.7 ms`; total down wall: `34930.2 ms`.

Top up/gate exposed-wait layers:

| layer | calls | exposed wait ms | up/gate wall ms | misses | hit rate | type |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 217 | 1095.1 | 1142.1 | 2298 | 33.8% | 22 |
| 27 | 217 | 1083.1 | 1130.0 | 2170 | 37.5% | 22 |
| 12 | 217 | 1076.5 | 1123.6 | 2235 | 35.6% | 22 |
| 20 | 217 | 1076.5 | 1123.4 | 2234 | 35.7% | 22 |
| 16 | 217 | 1068.9 | 1115.6 | 2158 | 37.8% | 22 |
| 25 | 217 | 1064.9 | 1111.4 | 2246 | 35.3% | 22 |
| 18 | 217 | 1061.4 | 1108.7 | 2126 | 38.8% | 22 |
| 24 | 217 | 1057.5 | 1103.8 | 2192 | 36.9% | 22 |

Top down decode-stage layers:

| layer | calls | stage ms | wall ms | misses | hit rate |
|---:|---:|---:|---:|---:|---:|
| 10 | 217 | 1131.0 | 1162.2 | 1266 | 27.1% |
| 6 | 217 | 1126.8 | 1157.5 | 1252 | 27.9% |
| 4 | 217 | 1109.9 | 1141.4 | 1359 | 21.7% |
| 8 | 217 | 1109.2 | 1140.7 | 1248 | 28.1% |
| 7 | 217 | 1102.7 | 1133.7 | 1217 | 29.9% |
| 9 | 217 | 1098.1 | 1128.9 | 1201 | 30.8% |
| 18 | 217 | 1070.0 | 1100.9 | 1200 | 30.9% |
| 58 | 217 | 1054.5 | 1085.5 | 1214 | 30.1% |

Top combined `upgate_wait + down_stage` layers:

| layer | upgate wait ms | down stage ms | combined ms |
|---:|---:|---:|---:|
| 10 | 1095.1 | 1131.0 | 2226.1 |
| 18 | 1061.4 | 1070.0 | 2131.4 |
| 20 | 1076.5 | 1046.1 | 2122.6 |
| 16 | 1068.9 | 1045.4 | 2114.3 |
| 25 | 1064.9 | 1028.3 | 2093.2 |
| 24 | 1057.5 | 1030.8 | 2088.3 |
| 22 | 1055.8 | 1020.5 | 2076.3 |
| 23 | 1051.2 | 1016.8 | 2068.0 |

Decision:

- Kimi does not show a DeepSeek-style gate-only CPU bottleneck.
- The visible issue is paired up/gate staging wait for type `22` layers plus down staging; gate-only hotpool is not the right next primary move because up remains on the same critical path.
- The first optimization probe should be role-aware cache allocation, starting with a controlled upgate/down split sweep. If moving slots from down to upgate regresses, the next probe should be a joint layer-profile cache rather than more upgate-only space.

## Phase 2: Role-aware joint up/gate/down VRAM cache allocation

Hypothesis:

- Kimi stalls when any role needed by the current MoE layer misses and blocks the CPU/defer GPU extension.
- A cache allocation that maximizes joint completion probability can beat separate hotness-only role hit rates.

Experiments:

1. Baseline current split: down slots around `723`; upgate slots around `1735`.
2. Candidate A: shift more VRAM to up/gate while shrinking down.
3. Candidate B: align up/gate/down for the same expert IDs in high-impact layers.
4. Candidate C: layer-priority cache where low-coverage/high-wait layers receive complete or near-complete role coverage.

Metrics:

- Per-prompt min/median/mean token rate.
- Exposed `iouring_wait_us` by role.
- Cache hit rate by role and by layer.
- Number of tokens/layers where all required role experts are resident.
- TTFT and RAM/VRAM usage.

Acceptance:

- Held-out min token rate improves without TTFT/RAM/quality regression.
- Improvement must survive paired baseline rerun.

### Phase 2A exact next experiment: upgate/down split sweep

Timestamp: 2026-07-10 13:25 CST.

Hypothesis:

- Current split is `GGML_MOE_VRAM_CACHE_UPGATE_PCT=62`.
- Phase 1 shows upgate type `22` exposed wait is material, but down stage is larger. A small reallocation toward upgate may improve decode if added upgate hits reduce exposed wait more than the removed down slots increase down staging.
- This is a default-off env-only experiment; no source behavior changes.

Theoretical bound:

- N32 full-dev profile exposed upgate wait is `18.7s`; down decode stage is `32.7s`.
- A split-only change cannot remove all upgate wait because misses are broad across many layers and experts.
- Practical upside for N96 is expected to be modest, likely single-digit percentage unless the extra upgate slots cover high-repeat misses. Any down regression can erase the gain.

Experiment:

1. Run a quick cold-start N32 dev smoke using the first 3 dev prompts:
   - control: `UPGATE_PCT=62`;
   - candidate A1: `UPGATE_PCT=66`;
   - candidate A2: `UPGATE_PCT=70`.
2. Keep all other env exactly the same as Phase 0, including RAM tier.
3. Do not use held-out test prompts.
4. Compare quality, TTFT, token rate, RAM peak, upgate/down slots, hit rates, `iouring_wait_us`, and expert pack batch histograms.

Promotion:

- If neither candidate improves median and minimum token rate on the 3-prompt smoke, reject the split-only approach.
- If one candidate improves without quality/RAM/TTFT regression, run all 7 dev prompts at N96 with paired control and candidate.
- Only after paired N96 dev improvement, run a refreshed sealed held-out test set.

Rollback:

- Env-only rejected candidates require no source rollback.
- If a source patch is later needed for joint cache profile support, it must be default-off and reverted if it fails gates.

### Phase 2A result: split-only sweep rejected

Timestamp: 2026-07-10 13:24 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase2a-upgate-pct-n32-dev3-132403`

All runs:

- cold start, N32, first 3 dev prompts only;
- `PROFILE=0`;
- same 16GB cgroup and RAM tier as Phase 0;
- quality passed for all 9 prompt runs.

Result:

| split | upgate slots | down slots | token rate min | token rate median | token rate mean | TTFT median ms | iouring wait mean s | RAM peak GiB | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 62 | 1735 | 723 | 1.84 | 1.84 | 1.85 | 8537.31 | 18.99 | 13.55 | control |
| 66 | 1847 | 647 | 1.83 | 1.89 | 1.87 | 8454.36 | 18.71 | 13.56 | reject: min did not improve |
| 70 | 1959 | 571 | 1.73 | 1.86 | 1.82 | 8609.74 | 18.86 | 13.55 | reject: France regression |

Per-prompt observations:

- `pct66` improved France and Japan, but `dev_photosynthesis_factual` dropped from `1.84` to `1.83`, so the minimum-token-rate gate failed.
- `pct70` reduced France from `1.84` to `1.73`, confirming that stealing too many down slots hurts stability.
- `iouring_wait` moved only slightly; split-only does not fix runtime queue starvation.

Decision:

- Do not promote split-only VRAM reallocation to N96.
- Keep `GGML_MOE_VRAM_CACHE_UPGATE_PCT=62` for current baseline/SOTA reproduction.
- Move to a scheduling/cache-completeness experiment that can reduce exposed wait without just trading down misses for upgate misses.

## Phase 3: Explicit RAM tier for second-hot experts

Hypothesis:

- Decode page cache is not a controlled expert cache.
- Replacing low-value file-backed pages with explicit RAM-resident expert slabs can reduce SSD wait, but only if it reduces exposed wait rather than just shifting bytes from SSD to H2D.

Experiments:

1. Identify decode-time low-value RAM: file-backed pages not touched after prompt, GGUF expert pages caused by fallback/refault, and dense/attention pages already resident in VRAM and not needed by decode CPU paths.
2. Define RAM candidate sets from dev prompts only: high exposed-wait layer gate+up full or partial layer, high exposed-wait up/gate/down grouped experts, and second-hot experts not worth VRAM but frequent enough to avoid SSD.
3. Test RAM layouts: 1.8 GiB current gate layer reference, 3-5 GiB structured RAM tier, and larger tier only if TTFT stays within `+20%` and RAM remains below cgroup.

Metrics:

- RAM hit count and RAM H2D bytes.
- SSD `iouring_wait_us` reduction.
- Decode time change.
- TTFT preload cost.
- Page-cache/file breakdown before prompt, after prompt, during decode.


Acceptance:

- Accept only if exposed wait and token rate improve on held-out prompts.
- Reject if hit rate improves but decode slows, TTFT exceeds gate, or RAM pressure/refault increases.

## Phase 4: Reduce io_uring queue starvation without unsafe prediction

Hypothesis:

- Pure IO bench can reach higher bandwidth, but runtime exposes small bursts.
- Larger batches help only if they do not add stale/unused reads or block demand.

Experiments:

1. Same-layer aggressive co-submit: enqueue known up/gate/down misses for the layer as soon as routing is available; demand priority remains first; no speculative future-layer reads initially.
2. Cross-layer safe scheduler shadow mode: track duplicate/in-flight opportunities without changing behavior.
3. If shadow counters show useful opportunities, enable default-off scheduler where demand can steal/wait on matching in-flight prefetch, stale prefetch is capped/pruned, and extra read ratio is bounded.

Acceptance:

- `iouring_wait_us` drops more than candidate overhead.
- Token rate improves on paired held-out runs.
- Extra bytes and TTFT remain bounded.

### Phase 4A exact next experiment: same-layer gate/up/down cosubmit smoke

Timestamp: 2026-07-10 13:33 CST.

Existing default-off mechanism:

- `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`
- Optional filters:
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_DOWN_ONLY=1`
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_MIN_SEEN=<n>`
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_MIN_COUNT=<n>`
  - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT=<csv>`

Hypothesis:

- When the gate path sees a routed expert, the corresponding up/down expert ID is already known.
- Co-submitting the paired up/down read can create larger same-layer IO batches and reduce queue starvation.
- This is closer to the measured bottleneck than split-only cache reallocation.

Risks:

- Kimi's current fused up/gate path may not trigger the standalone gate cosubmit hook often enough; the first check must verify nonzero `gate/up/down cosubmit` jobs.
- Extra reads can evict useful cache lines or steal IO from demand reads, especially if unfiltered.
- Previous DeepSeek experiments had rejected variants, so this must stay default-off and pass Kimi-specific dev gates before any promotion.

Theoretical bound:

- The upper bound is a reduction in exposed `iouring_wait`, not total expert bytes.
- If cosubmit only changes batch shape but keeps the same critical demand order, gain will be small.
- If it lifts inflight depth from the current `~4` toward the pure IO bench regime without extra stale reads, the improvement could be material; the smoke should first prove `iouring_wait_us` drops and jobs are nonzero.

Experiment:

1. Run cold-start N32 dev3 control from Phase 2A as the paired baseline: `pct62`.
2. Run candidate C1:
   - `UPGATE_PCT=62`;
   - `GGML_MOE_GATE_UPDOWN_COSUBMIT=1`;
   - `GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT=$OUT/gate-updown-cosubmit-profile.csv`.
3. If C1 has nonzero jobs but regresses from extra reads, test C2:
   - add `GGML_MOE_GATE_UPDOWN_COSUBMIT_DOWN_ONLY=1`;
   - add `GGML_MOE_GATE_UPDOWN_COSUBMIT_MIN_SEEN=1`.
4. Keep all other env identical to Phase 0/2A.
5. Do not use held-out test prompts.

Acceptance:

- All smoke prompts quality pass.
- `gate/up/down cosubmit` reports nonzero jobs, zero failures, and no read failures.
- N32 dev3 minimum and median token rate beat Phase 2A `pct62`.
- `iouring_wait_us` mean drops without TTFT or RAM regression.
- Only then run paired N96 dev validation.

### Phase 4A result: standalone gate cosubmit rejected

Timestamp: 2026-07-10 13:34 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4a-cosubmit-c1-n32-dev3-133415`

Candidate env:

```bash
GGML_MOE_GATE_UPDOWN_COSUBMIT=1
GGML_MOE_GATE_UPDOWN_COSUBMIT_PROFILE_OUT=$OUT/gate-updown-cosubmit-profile.csv
```

Paired control:

- Phase 2A `pct62`: `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase2a-upgate-pct-n32-dev3-132403/pct62`

Result:

| run | quality | token rate min | token rate median | token rate mean | TTFT median ms | iouring wait mean s | RAM peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control pct62 | 3/3 pass | 1.84 | 1.84 | 1.85 | 8537.31 | 18.99 | 13.55 |
| C1 cosubmit | 3/3 pass | 1.78 | 1.79 | 1.81 | 8878.44 | 19.00 | 13.56 |

Activation check:

- No `[moe_stream_batch] gate/up/down cosubmit: ...` atexit counter appeared in `stderr.txt`.
- The profile CSV existed and had `6116` data rows, but it was pair-observation output:
  - `up_predict_down`: `177` rows;
  - `actual_up`: `177` rows;
  - `actual_down`: `5760` rows.
- Therefore the current Kimi path did not execute actual standalone cosubmit jobs.

Interpretation:

- The standalone hook lives on the one-stream gate path. Current Kimi SOTA uses fused up/gate batch paths for the relevant work, so this hook is not the right integration point.
- `up_predict_down` showed `pack_hits=0` because it used the up/gate expert byte size when predicting down; down has a different packed size. Any future fused-path implementation must use exact tensor sizes and per-size cache groups.
- Since jobs were zero and token rate regressed, do not run C2.

Decision:

- Reject `GGML_MOE_GATE_UPDOWN_COSUBMIT=1` for current Kimi SOTA.
- Keep it default-off.
- Next source-level work should be stats-only first: add a fused up/gate path shadow counter that records potential same-layer up/gate/down co-submit opportunities with exact per-role sizes, without issuing extra reads. Only if the shadow proves useful should an actual prefetch path be implemented.

### Phase 4B next plan: fused-path co-submit shadow, stats-only

Hypothesis:

- The useful scheduling point for Kimi is inside the fused up/gate batch path, after routing has produced active expert IDs and before up/gate/down staging completes.
- Current-down overlap already handles some down preloading, but Phase 1 still shows down stage and up/gate wait on the critical path.
- A stats-only fused-path shadow can identify whether there are same-layer jobs that could be co-submitted earlier or grouped better without risking correctness.

Implementation plan:

1. Add a default-off env such as `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1`.
2. In the fused up/gate path, record for each layer/token:
   - active experts;
   - up, gate, and down tensor names;
   - exact expert bytes per role;
   - cache hit/miss by role;
   - pack hit/miss by role;
   - whether current-down overlap already submitted each down expert;
   - potential grouped job counts by expert byte size.
3. Write CSV to `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW_OUT`.
4. Run N32 dev3 with shadow only and verify:
   - quality unchanged;
   - token rate and TTFT not materially changed;
   - shadow rows are nonzero;
   - overhead is small enough to run full dev profiling.
5. Use the shadow report to decide between:
   - no-op if current-down overlap already covers the opportunities;
   - exact-size fused down co-submit;
   - larger grouped IO scheduling for up/gate/down role batches.

Acceptance for shadow:

- No behavior change by default.
- Shadow run quality passes.
- Runtime overhead is low enough for diagnostic use.
- The CSV provides enough evidence to calculate an upper bound before any actual prefetch implementation.

### Phase 4B result: fused-path co-submit shadow implemented, diagnostic only

Timestamp: 2026-07-10 13:45 CST.

Source status:

- Default-off diagnostic code added in `ggml/src/ggml-cuda/moe_stream_batch.cu`.
- Env gates:
  - `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1`
  - `GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW_OUT=<csv>`
- No behavior change when the env is unset.
- Build passed with the existing warning set:

```bash
cmake --build build-cuda-batch -j$(nproc)
```

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4b-fused-shadow-n32-dev3-134523`

Command:

```bash
OUT=/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4b-fused-shadow-n32-dev3-134523
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root "$OUT" \
  --mode dev \
  --n 32 \
  --max-prompts 3 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 600 \
  --upgate-pct 62 \
  --extra-runtime-env "GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=$OUT/ram-batch-profile.csv
GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW=1
GGML_MOE_FUSED_UPGATE_DOWN_COSUBMIT_SHADOW_OUT=$OUT/fused-upgate-down-shadow.csv"
```

Paired control:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase2a-upgate-pct-n32-dev3-132403/pct62`

Result:

| run | quality | token rate min | token rate median | token rate mean | TTFT median ms | iouring wait mean s | RAM peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control pct62 | 3/3 pass | 1.84 | 1.84 | 1.85 | 8537.31 | 18.99 | 13.55 |
| fused shadow | 3/3 pass | 1.82 | 1.87 | 1.87 | 8662.96 | 18.52 | 13.55 |

Shadow aggregate after filtering duplicate per-process CSV headers:

- Rows: `5583`, all decode.
- Active expert rows: `44664`.
- Up cache hit rate: `44.08%`.
- Gate cache hit rate: `44.17%`.
- Down cache hit rate: `32.10%`.
- All-role cache-hit rows: `32.10%`.
- Down-overlap plannable rows: `25094 / 44664 = 56.18%`.
- Pack lookup:
  - up pack hits `44664`, misses `0`;
  - gate pack hits `44664`, misses `0`;
  - down pack hits `39432`, misses `0` for rows with a matching down tensor.

Top same-layer all-role miss layers:

| layer | active | all-role miss | any-role miss | down plannable | up miss | gate miss | down miss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 744 | 528 | 603 | 603 | 528 | 528 | 603 |
| 1 | 744 | 518 | 604 | 604 | 519 | 518 | 604 |
| 4 | 744 | 516 | 588 | 588 | 517 | 516 | 588 |
| 60 | 768 | 511 | 562 | 538 | 511 | 511 | 562 |
| 3 | 744 | 496 | 563 | 563 | 497 | 496 | 563 |
| 10 | 744 | 494 | 744 | 0 | 494 | 494 | 744 |
| 29 | 744 | 489 | 560 | 560 | 489 | 489 | 560 |
| 28 | 744 | 481 | 549 | 549 | 482 | 481 | 549 |

Interpretation:

- The DeepSeek gate-only lesson partially transfers, but the Kimi bottleneck is not a standalone gate CPU path.
- Up and gate miss together almost exactly, and down miss is also high. Therefore a gate-only cache increase is unlikely to be enough.
- The pack path is present for the measured up/gate/down rows; the remaining problem is residency and scheduling, not missing pack coverage.
- Current-down overlap already has many plannable rows, so an actual fused co-submit must prove that it reduces exposed wait beyond the existing overlap rather than duplicating work.
- Since the shadow run is diagnostic and min token rate did not improve, it is not a SOTA claim.

Decision:

- Keep the shadow path default-off.
- Commit/push it only as diagnostic infrastructure with the above run path and rollback point.
- Next candidate must be one of:
  - exact fused-path up/gate/down co-submit only if it uses the shadow data to avoid duplicate current-down work;
  - layer/role-aware RAM/VRAM cache reallocation for layers with high all-role miss;
  - no-op if the expected upper bound is too small after subtracting current-down overlap.

### Phase 4C plan: early current-down overlap A/B

Reason:

- The current Kimi path already supports current-layer down overlap, but the Phase 4B env did not enable `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY`.
- In the mixed up/gate path, the non-early mode starts current-down overlap only after up/gate staging and compute. The early mode starts it immediately after routing and before up/gate staging/compute.
- This is the narrowest way to test the user's desired behavior: once routing has produced active expert IDs, begin moving down experts without waiting for up/gate compute.

Hypothesis:

- If exposed down wait is still on the critical path, `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` can hide part of the current-down worker time behind up/gate staging/compute.
- Phase 4B measured current-down worker time around `2.34-2.48s` per N32 dev prompt, so the hard upper bound is roughly that worker time. A realistic bound is smaller because part of the work is already overlapped and because early down reads can compete with up/gate reads.
- If IO contention dominates, early mode may reduce down wait but increase up/gate staging wait or total `iouring_wait_us`; in that case reject it.

Experiment:

1. Run a fresh cold-start N32 dev3 control on current HEAD with shadow disabled.
2. Run candidate with the same env plus:

```bash
GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1
```

3. Keep:
   - `UPGATE_PCT=62`;
   - 1800 MiB RAM tier with `blk1_gate_full384.csv`;
   - `MemoryMax=15900000000`, `MemorySwapMax=0`;
   - no held-out test prompts.
4. Compare:
   - token-rate min/median/mean;
   - TTFT;
   - decode ms;
   - `expert_pack iouring_wait_us`;
   - current-down `worker_us`, planned jobs, completed jobs;
   - up/gate and down cache hit/miss;
   - iouring batch hist/inflight;
   - RAM peak and quality.

Acceptance:

- Quality 3/3 pass.
- Host RAM remains under the hard gate.
- TTFT ratio `<=1.20`.
- N32 dev3 min and median token rate beat the paired fresh control.
- The gain must be explained by lower exposed wait, not by noise or shorter output.
- If accepted on N32, run N96 dev before any SOTA claim.

### Phase 4C result: early current-down overlap rejected

Timestamp: 2026-07-10 14:02 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4c-early-down-n32-dev3-215843`

Paired fresh control command shape:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4c-early-down-n32-dev3-215843/control \
  --mode dev \
  --n 32 \
  --max-prompts 3 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 600 \
  --upgate-pct 62 \
  --extra-runtime-env "<1800 MiB blk1 gate RAM tier env>"
```

Candidate adds:

```bash
GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1
```

Control result:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB |
|---|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 1.86 | 8104.82 | 16649.94/31 | 13.55 |
| dev_japan_factual | pass | 1.89 | 7382.51 | 16428.44/31 | 13.56 |
| dev_photosynthesis_factual | pass | 1.88 | 7071.07 | 16527.53/31 | 13.41 |

Candidate result:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB | failure |
|---|---:|---:|---:|---:|---:|---|
| dev_france_regression | fail | n/a | n/a | n/a | 3.51 | exit `134`, CUDA OOM |
| dev_japan_factual | fail | n/a | n/a | n/a | 7.19 | exit `134`, CUDA OOM |
| dev_photosynthesis_factual | pass | 0.91 | 7772.54 | 34146.30/31 | 7.97 | severe slowdown |

Failure evidence:

- France/Japan stderr:
  - `ggml_cuda_compute_forward: MUL failed`
  - `CUDA error: out of memory`
  - abort signal `6`, exit `134`.
- The third run survived only after the earlier failures, but free VRAM was much lower:
  - `VRAM cache budget: requested=15000 MiB actual=3729 MiB free=4241 MiB`;
  - upgate slots dropped from the control `1735` to `431`;
  - down slots dropped from `723` to `180`;
  - upgate hit rate collapsed to `12.8%`;
  - decode regressed to `0.91 tok/s`.
- After the run, `nvidia-smi` showed no persistent process and memory returned to normal, so this is not accepted as a stable runtime state.

Historical cross-check:

- The parent plan already rejected `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` in Phase 7GG:
  - first n32 decode `28927.32 ms / 31`;
  - repeat n32 decode `29231.83 ms / 31`;
  - repeat fell inside baseline noise;
  - no n96 was run.
- Phase 7NW also closed the same-layer IO-fill direction because early overlap, aux ring, combined up/gate IO, and dual-fence variants either failed reproducibility or lost endpoint overlap.

Decision:

- Reject `GGML_MOE_CURRENT_DOWN_OVERLAP_EARLY=1` for current Kimi SOTA.
- Do not run N96.
- Do not enable this flag in reproduction scripts.
- Keep the default-off code only as diagnostic/historical infrastructure.
- The next direction should not be "submit more same-layer IO" unless it first proves how it avoids the 7GG/7GH/7KW/7LE endpoint-overlap loss.

### Phase 4D plan: RAM/VRAM cache co-design trace and candidate screen

Reason:

- Phase 4B shadow shows large up/gate/down miss traffic, but Phase 4C and older 7NW evidence show that simply exposing more same-layer IO concurrency is not enough.
- Current N32 control still spends most host memory on file-backed pages, while the explicit 1800 MiB `blk1_gate_full384.csv` RAM tier has only about `0.6%` total RAM-tier hit rate in the Phase 4B/4C dev3 runs.
- The next practical question is whether the same or slightly larger RAM budget can be moved from low-value page cache / low-hit RAM tier entries into higher-yield expert entries selected from actual foreground IO across dev prompts.

Hypothesis:

- A prompt-agnostic RAM tier generated from multi-prompt foreground IO traces can outperform the current static `blk1_gate_full384.csv` tier.
- It must improve exposed wait or token rate, not only RAM hit rate.
- Candidate budgets must stay inside the 16 GB cgroup gate; based on the fresh control peak around `13.56 GiB`, `1800 MiB` is safe and `2400-3000 MiB` is only a guarded experiment.

Experiment sequence:

1. Run cold-start N32 dev3 trace with current accepted runtime, no early overlap, and tracing only:

```bash
GGML_MOE_IO_READ_TRACE_OUT=$RUN/io-read-trace.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv
```

2. Use only dev traces, never held-out test prompts, to generate candidate RAM profiles with:

```bash
.Agent/run-tools/kimi_ram_candidate_multidev_screen.py
.Agent/run-tools/kimi_make_ram_slab_profile_from_io_trace.py
```

3. Screen at least:
   - current control profile: `blk1_gate_full384.csv`, 1800 MiB;
   - best dev-trace `down` profile at 1800 MiB;
   - best dev-trace `up,gate` profile at 1800 MiB;
   - if the reports justify it, guarded 2400/3000 MiB profiles.
4. A/B only the strongest one or two candidates on N32 dev3.
5. Run N96 dev and held-out test only if N32 dev3 improves min and median token rate with quality pass.

Acceptance:

- Quality passes for all dev smoke prompts.
- RAM peak `<15900000000` bytes with `MemorySwapMax=0`.
- TTFT ratio `<=1.20`.
- N32 dev3 min and median token rate beat paired fresh control.
- RAM-tier hits must replace SSD waits on the critical path; if hit rate rises but token rate falls, reject.
- Any accepted improvement must be committed and pushed with run roots, exact profiles, commands, RAM/TTFT/quality, and rollback point.

### Phase 4D progress: trace screen and N32 candidate A/B

Timestamp: 2026-07-10 14:16 CST.

Trace run:

- Correct trace root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127`
- Earlier mistaken trace root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-220835`
  - rejected as a data source because `$RUN` was expanded too early and trace files were written under `/`;
  - generated root files were removed before the corrected run.

Correct trace command shape:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --repo /root/lfz/llama.cpp-vendor-kimi \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127 \
  --mode dev \
  --n 32 \
  --max-prompts 3 \
  --keep-going \
  --memory-max 15900000000 \
  --runtime-max-sec 600 \
  --upgate-pct 62 \
  --extra-runtime-env "GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=\$RUN/ram-batch-profile.csv
GGML_MOE_IO_READ_TRACE_OUT=\$RUN/io-read-trace.csv
GGML_MOE_IO_WAIT_TRACE_OUT=\$RUN/io-wait-trace.csv"
```

Trace result:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB | io-read rows |
|---|---:|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 1.84 | 8758.53 | 16852.32/31 | 13.56 | 38849 |
| dev_japan_factual | pass | 1.75 | 9128.08 | 17672.96/31 | 13.56 | 38844 |
| dev_photosynthesis_factual | pass | 1.81 | 7607.67 | 17132.35/31 | 13.41 | 38035 |

Trace note:

- The trace run is diagnostic only; tracing overhead means its token rate is not used as the paired performance baseline.
- Total trace rows: `115722`, total traced IO bytes: `617.14 GiB`.
- Current `blk1_gate_full384.csv` RAM tier does not appear in `io-read-trace.csv` because RAM-tier hits bypass the IO trace.
- Current tier measured from `ram-batch-profile.csv`:
  - total RAM hits: `718`;
  - total RAM H2D bytes: `3.144 GiB`;
  - total RAM batch wall: `2068.418 ms`.

Candidate screening output:

- Candidate directory:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127/ram-candidates`

Top screened candidates:

| candidate | budget MiB | selected entries | trace hit rows | trace hit GiB | prompts | batches | dominant batches >=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| down-1800 | 1800 | 269 | 4437 | 29.02 | 3 | 2513 | 223 |
| upgate-1800 | 1800 | 375 | 5425 | 25.31 | 3 | 3186 | 210 |
| all-1800 | 1800 | 310 | 5519 | 31.45 | 3 | 3843 | 55 |
| all-3000 | 3000 | 529 | 8463 | 47.19 | 3 | 5062 | 310 |

N32 A/B run:

- Candidate root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-ram-candidate-n32-dev3-141626`
- Paired fresh control:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4c-early-down-n32-dev3-215843/control`

Result:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control blk1_gate-1800 | 3/3 | 1.86 | 1.88 | 1.88 | 7382.51 | 13.56 | 56.16 | 617.1 | 718 | 3.14 |
| all-1800 | 3/3 | 1.91 | 1.95 | 1.99 | 7579.92 | 13.61 | 52.90 | 588.8 | 5519 | 31.45 |
| all-3000 | 3/3 | 1.92 | 1.98 | 1.96 | 8726.13 | 14.78 | 53.17 | 573.0 | 8463 | 47.19 |

Per-prompt `all-1800`:

| prompt | quality | tok/s | TTFT ms | decode ms/runs | RAM peak GiB | iouring wait s | RAM hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 2.11 | 7579.92 | 14672.06/31 | 13.61 | 16.53 | 1884 |
| dev_japan_factual | pass | 1.95 | 8151.71 | 15914.49/31 | 13.61 | 18.03 | 1911 |
| dev_photosynthesis_factual | pass | 1.91 | 7382.18 | 16234.74/31 | 13.47 | 18.34 | 1724 |

Interpretation:

- `all-1800` is the better next candidate despite `all-3000` having a slightly higher N32 minimum:
  - `all-1800` has better mean token rate;
  - lower TTFT;
  - far lower RAM peak;
  - less risk against the 16 GB hard gate.
- The gain is consistent with the intended mechanism:
  - RAM hits increase from `718` to `5519`;
  - explicit RAM H2D increases from `3.14 GiB` to `31.45 GiB`;
  - expert-pack iouring bytes drop from `617.1 GiB` to `588.8 GiB`;
  - iouring wait drops from `56.16s` to `52.90s`;
  - token-rate min/median/mean improve.
- This is still a dev N32 result, not a SOTA claim.

Decision:

- Promote only `all-1800` to N96 dev validation.
- Do not promote `all-3000` yet because TTFT and RAM are too close to the limit for only marginal min-token-rate gain.
- If N96 passes, copy the `all-1800.profile.csv` into a tracked `.Agent/profiles/kimi/ram-tier/phase4d-*` path, record exact reproduction commands, run held-out validation, then commit and push as a candidate improvement.

### Phase 4D result: all-1800 rejected after held-out test

Timestamp: 2026-07-10 14:59 CST.

N96 dev candidate:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-all1800-n96-dev3-142351`
- Fresh paired N96 dev control:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-fresh-control-n96-dev3-142832`

N96 dev paired result:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control blk1_gate-1800 | 3/3 | 1.82 | 1.83 | 1.86 | 8449.15 | 13.56 | 144.08 | 1369.9 | 1694 | 7.42 |
| all-1800 | 3/3 | 1.88 | 1.90 | 1.91 | 8436.26 | 13.62 | 140.23 | 1311.5 | 11562 | 65.82 |

N96 dev interpretation:

- Dev3 passed all gates.
- Improvement mechanism matched the hypothesis:
  - RAM hits increased by `9868`;
  - iouring bytes dropped by `58.4 GiB`;
  - iouring wait dropped by `3.85s`;
  - token-rate min/median/mean improved.
- This justified held-out validation but was still not a SOTA claim.

Held-out candidate:

- Run:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-all1800-n96-test-143404`
- Profile used:
  `.Agent/profiles/kimi/ram-tier/phase4d-dev-trace/all-1800.profile.csv`
  - copied temporarily for the test;
  - removed from the worktree after rejection;
  - canonical artifact remains in the trace run:
    `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-io-trace-n32-dev3-correct-141127/ram-candidates/all-1800.profile.csv`.
- Fresh paired held-out control:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4d-control-n96-test-144730`

Held-out result:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control blk1_gate-1800 | 6/6 | 1.54 | 1.82 | 1.788 | 9621.60 | 232735.82 | 14.81 | 311.95 | 2920.0 | 3455 | 15.13 |
| all-1800 | 6/6 | 1.54 | 1.82 | 1.785 | 9993.85 | 230457.74 | 14.81 | 315.54 | 2849.2 | 15244 | 85.87 |

Held-out per-prompt deltas:

| prompt | token-rate delta | decode delta ms | TTFT delta ms | quality |
|---|---:|---:|---:|---:|
| test_chinese_01 | +0.01 | -356.03 | -704.22 | pass/pass |
| test_coding_01 | +0.00 | +61.48 | +160.86 | pass/pass |
| test_english_factual_01 | +0.00 | -114.97 | -211.29 | pass/pass |
| test_english_factual_02 | +0.02 | -492.32 | +182.39 | pass/pass |
| test_mixed_instruction_01 | -0.02 | +531.74 | +742.01 | pass/pass |
| test_reasoning_math_01 | -0.03 | +680.83 | -2278.08 | pass/pass |

Held-out quality notes:

- All six held-out answers were semantically coherent.
- The math prompt produced the correct answer `5:15 PM` in both candidate and control.
- The very high math TTFT was not introduced by all-1800:
  - control TTFT `232735.82 ms`;
  - all-1800 TTFT `230457.74 ms`.

Decision:

- Reject `all-1800` as a SOTA/performance improvement.
- Reason: it improves dev3 but does not improve held-out min/median/mean token rate; mean regresses slightly and `iouring_wait` increases by `3.59s` despite lower iouring bytes.
- Do not commit the `phase4d-dev-trace/all-1800.profile.csv` runtime profile.
- Keep the result as evidence that dev3 static RAM hotsets can overfit and may only move bytes from SSD to RAM without reducing held-out endpoint wait.

Next plan:

1. Build the next RAM/VRAM candidate from the full 7-prompt dev set, not dev3.
2. Require candidate screening to enforce prompt/category coverage, e.g. `min_prompts >= 4` and no single category dominating selected traffic.
3. Prefer dynamic or online RAM admission rules over static prompt-trace hotsets:
   - promote an expert to RAM only after repeated misses across prompts or sustained per-layer pressure;
   - preserve the current explicit 16 GB RAM accounting;
   - record whether RAM hits reduce endpoint wait, not just SSD bytes.
4. Do not use held-out test prompts for candidate construction.
5. Any future RAM profile must pass:
   - N32 full-dev;
   - N96 full-dev;
   - held-out N96;
   - and only then be committed as an accepted profile.

### Phase 4E plan: full-dev7 prompt-agnostic RAM tier screen

Timestamp: 2026-07-10 15:12 CST.

Purpose:

- Re-run RAM-tier design with all seven dev prompts instead of the dev3 subset that overfit in Phase 4D.
- Keep held-out test prompts unused for construction.
- Require each selected expert to be observed in at least four dev prompts.

Trace run:

- Run root:
  `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343`
- Env:

```bash
GGML_MOE_RAM_TIER_MIB=1800
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=1800
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
GGML_MOE_RAM_BATCH_PROFILE_OUT=$RUN/ram-batch-profile.csv
GGML_MOE_IO_READ_TRACE_OUT=$RUN/io-read-trace.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$RUN/io-wait-trace.csv
```

Trace quality:

- `6/7` auto-quality pass at N32.
- `dev_linear_equation` failed only because N32 truncated the answer after `x =`; this N32 trace remains valid for route/IO observation, but no SOTA claim can be made without N96 quality passing.

Trace aggregate:

- Total io-read rows: `292570`.
- Total traced IO: `1557.45 GiB`.
- Host RAM peak: `13.57 GiB`.

Current tier actual RAM hits during trace:

- `1770` RAM hits;
- `7.75 GiB` RAM H2D;
- `5380.88 ms` total RAM batch wall across seven prompts.

Candidate screen command:

```bash
python3 .Agent/run-tools/kimi_ram_candidate_multidev_screen.py \
  --input-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343 \
  --out-profile <candidate>.profile.csv \
  --out-report <candidate>.report.json \
  --out-csv <candidate>.candidates.csv \
  --budget-mib <1200|1800|2400> \
  --roles <down|up,gate|up,gate,down> \
  --min-count 2 \
  --min-prompts 4 \
  --max-jobs 8
```

Candidate summary:

| candidate | budget MiB | entries | trace hit rows | trace hit GiB | prompts hit | dominant batches >=4 | role mix |
|---|---:|---:|---:|---:|---:|---:|---|
| all-1200-minp4 | 1200 | 214 | 8940 | 49.31 | 7 | 77 | up/down/gate |
| all-1800-minp4 | 1800 | 320 | 11984 | 66.12 | 7 | 323 | up/down/gate |
| all-2400-minp4 | 2400 | 427 | 14751 | 81.19 | 7 | 681 | up/down/gate |
| down-1800-minp4 | 1800 | 267 | 8349 | 55.16 | 7 | 546 | down only |
| upgate-1800-minp4 | 1800 | 378 | 11307 | 52.34 | 7 | 695 | up/gate only |

Decision before A/B:

- First test `all-1200-minp4` and `all-1800-minp4`.
- Skip `all-2400-minp4` until a smaller candidate proves endpoint value because Phase 4D showed larger RAM tiers can improve hit bytes but fail held-out endpoint speed.
- Skip role-only candidates initially because Phase 4B showed up/gate/down misses are coupled; mixed all-role candidates are a better first test.

N32 full-dev A/B plan:

1. Run fresh N32 full-dev control with current `blk1_gate_full384.csv`.
2. Run `all-1200-minp4` with:
   - `GGML_MOE_RAM_TIER_MIB=1200`;
   - `GGML_MOE_RAM_TIER_PIN_MIB=1200`;
   - profile from the Phase 4E trace candidate directory.
3. Run `all-1800-minp4` with:
   - `GGML_MOE_RAM_TIER_MIB=1800`;
   - `GGML_MOE_RAM_TIER_PIN_MIB=1800`;
   - profile from the Phase 4E trace candidate directory.
4. Compare:
   - token-rate min/median/mean;
   - per-prompt decode delta;
   - TTFT ratio;
   - RAM peak;
   - iouring wait and bytes;
   - RAM hits and RAM H2D;
   - output quality, with the known N32 truncation caveat for `dev_linear_equation`.

N32 promotion rule:

- Candidate must improve full-dev min and median token rate over fresh control.
- Candidate must not materially worsen coding, mixed, or reasoning prompts.
- Host RAM must remain under `15900000000`.
- TTFT median must not exceed `1.20x` control.
- If N32 passes, run N96 full-dev before any held-out validation.

### Phase 4E N32 full-dev A/B result

Timestamp: 2026-07-10 15:27 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-minp4-n32-fulldev-ab-151309`

Runs:

- `control`: current `blk1_gate_full384.csv`, 1800 MiB.
- `all-1200-minp4`: full-dev7 `min_prompts>=4` all-role candidate, 1200 MiB.
- `all-1800-minp4`: full-dev7 `min_prompts>=4` all-role candidate, 1800 MiB.

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/7 | 1.62 | 1.85 | 1.793 | 8824.89 | 10729.33 | 13.56 | 139.08 | 1557.15 | 1770 | 7.75 |
| all-1200-minp4 | 6/7 | 1.63 | 1.89 | 1.839 | 8471.97 | 11157.23 | 13.03 | 137.68 | 1515.60 | 8940 | 49.31 |
| all-1800-minp4 | 6/7 | 1.56 | 1.91 | 1.813 | 8360.51 | 11407.33 | 13.62 | 139.42 | 1498.79 | 11984 | 66.12 |

N32 quality note:

- `dev_linear_equation` is `fail` for all three runs due N32 truncation, not a candidate-specific semantic failure.
- N96 full-dev remains mandatory before any quality claim.

Per-prompt deltas versus control:

| prompt | all-1200 tok delta | all-1200 decode delta ms | all-1800 tok delta | all-1800 decode delta ms |
|---|---:|---:|---:|---:|
| dev_france_regression | +0.06 | -466.73 | +0.05 | -405.37 |
| dev_japan_factual | +0.09 | -736.54 | +0.06 | -498.80 |
| dev_linear_equation | +0.01 | -195.52 | +0.04 | -454.69 |
| dev_mixed_summary | +0.02 | -149.10 | -0.04 | +519.48 |
| dev_photosynthesis_factual | +0.04 | -392.96 | +0.06 | -534.82 |
| dev_python_reverse | +0.06 | -647.63 | -0.11 | +1317.29 |
| dev_zh_france | +0.04 | -268.18 | +0.08 | -611.89 |

Decision:

- Promote `all-1200-minp4` to N96 full-dev.
  - It improves min/median/mean;
  - improves every per-prompt decode time;
  - reduces iouring wait by `1.40s`;
  - reduces iouring bytes by `41.55 GiB`;
  - lowers RAM peak from `13.56 GiB` to `13.03 GiB`;
  - keeps TTFT well inside the `+20%` gate.
- Reject `all-1800-minp4` at N32.
  - It regresses min token rate from `1.62` to `1.56`;
  - regresses Python and mixed prompts;
  - increases iouring wait slightly despite reducing bytes;
  - therefore it has the same "more RAM hits but worse endpoint" warning pattern as Phase 4D held-out.

Next:

- Run fresh paired N96 full-dev:
  - control with `blk1_gate_full384.csv`;
  - candidate with `all-1200-minp4`.
- Only if N96 full-dev improves min/median/mean and quality passes, copy the profile into a tracked path and run held-out N96.

### Phase 4E N96 full-dev and held-out result

Timestamp: 2026-07-10 23:55 CST.

Runs:

- N96 full-dev run root:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-all1200-n96-fulldev-153209`
- N96 held-out run root:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-all1200-n96-heldout-235502`

N96 full-dev aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 7/7 | 1.63 | 1.81 | 1.816 | 9024.04 | 11973.20 | 13.56 | 284.84 | 2799.68 | 3363 | 14.73 |
| all-1200-minp4 | 7/7 | 1.68 | 1.83 | 1.836 | 8622.40 | 11075.65 | 13.03 | 282.15 | 2729.91 | 15315 | 84.49 |

Full-dev interpretation:

- `all-1200-minp4` was a small positive dev signal:
  - token-rate mean `+0.020 tok/s`;
  - min `+0.05 tok/s`;
  - median `+0.02 tok/s`;
  - `iouring_wait` `-2.69s`;
  - SSD IO `-69.77 GiB`.
- However, per-prompt results already showed regressions on `dev_mixed_summary` and `dev_python_reverse`.
- Therefore it was allowed to proceed to held-out validation but was not accepted.

N96 held-out aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | RAM hits | RAM H2D GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/6 | 1.62 | 1.85 | 1.810 | 9830.87 | 248966.26 | 14.81 | 307.34 | 2919.97 | 3455 | 15.13 |
| all-1200-minp4 | 6/6 | 1.63 | 1.82 | 1.802 | 9466.49 | 228378.04 | 14.81 | 312.94 | 2862.60 | 13174 | 72.50 |

Held-out per-prompt deltas versus control:

| prompt | tok delta | decode delta ms | TTFT delta ms | TTFT ratio |
|---|---:|---:|---:|---:|
| test_chinese_01 | -0.01 | +136.16 | +553.55 | 1.064 |
| test_coding_01 | +0.01 | -245.85 | -669.13 | 0.936 |
| test_english_factual_01 | +0.05 | -1280.16 | -1129.60 | 0.874 |
| test_english_factual_02 | -0.01 | +323.09 | -350.96 | 0.962 |
| test_mixed_instruction_01 | -0.05 | +1660.17 | -1423.75 | 0.873 |
| test_reasoning_math_01 | -0.04 | +845.74 | -20588.22 | 0.917 |

Decision:

- Reject `all-1200-minp4`; do not copy it into tracked `.Agent/profiles/`.
- Reason:
  - held-out median token rate regressed from `1.85` to `1.82`;
  - held-out mean regressed from `1.810` to `1.802`;
  - total held-out decode time increased by `1.44s`;
  - `iouring_wait` increased by `5.61s` even though SSD IO decreased by `57.37 GiB`;
  - RAM H2D increased from `15.13 GiB` to `72.50 GiB`, so the profile moved bytes from SSD to RAM but did not shorten the critical path.
- Quality passed and TTFT did not violate the `1.20x` gate, but performance was not a general-prompt improvement.

Additional observation:

- `test_reasoning_math_01` shows a severe prompt-dependent TTFT long tail:
  - control TTFT `248966.26 ms`;
  - candidate TTFT `228378.04 ms`;
  - both runs hit the `15899996160` byte cgroup peak.
- The next profiling pass must separate decode token-rate work from cold-start/prompt TTFT long-tail causes. A candidate that only improves decode bytes but leaves TTFT near the memory limit is not sufficient.

Next:

- Build an exposed-wait profile for N96 dev and held-out control runs:
  - per-layer and per-role `io_uring_wait`;
  - queue depth and batch histogram around stalls;
  - RAM tier H2D timing versus SSD wait;
  - prompt/prefill TTFT memory peak and major fault source for the reasoning/math long tail.
- Screen layer/role slab candidates only if they have a measurable upper bound in removable exposed wait and keep RAM below the 16 GB gate with margin.

### Phase 4F N96 dev5 profile result

Timestamp: 2026-07-11 00:23 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase5-control-n96-profile-dev5-002343/control-profile`

Command shape:

- prompt file: `.Agent/evals/kimi-general-dev-prompts.jsonl`
- first 5 dev prompts only: France, Japan, photosynthesis, linear equation, Python reverse
- N96, cold start, `PROFILE=1`
- control RAM tier: `.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv`
- host memory gate: `MemoryMax=15900000000`, `MemorySwapMax=0`

Important caveat:

- `PROFILE=1` slows the run and should not be compared as a token-rate SOTA result.
- This run is for bottleneck attribution only.

Profile summary:

| metric | value |
|---|---:|
| prompts | 5 |
| quality | 5/5 |
| weighted token rate under profiling | 1.624 tok/s |
| aggregate iouring throughput | 9.256 GiB/s |
| peak IO utilization reference | 0.899 of 10.3 GiB/s |
| direct read ratio | 0.000 |
| weighted iouring inflight avg | 3.909 |
| iouring wait / decode fraction | 0.971 |

Prompt-level profile:

| prompt | tok/s | decode ms | down wall ms | upgate wall ms | fallback ms |
|---|---:|---:|---:|---:|---:|
| dev_france_regression | 1.64 | 51685 | 22500 | 14053 | 0 |
| dev_japan_factual | 1.71 | 45601 | 21104 | 12599 | 0 |
| dev_linear_equation | 1.46 | 23331 | 17604 | 6182 | 0 |
| dev_photosynthesis_factual | 1.67 | 56272 | 22539 | 15350 | 0 |
| dev_python_reverse | 1.56 | 60760 | 25264 | 16535 | 0 |

Key result:

- CPU fallback is not the current decode bottleneck in this control profile:
  - `fallback-profile.csv` is empty;
  - CPU/MOE profile reports batch accept for upgate and down with no single/fallback path.
- The main bottlenecks are inside the GPU extension path:
  - tensor staging / expert movement for down, up, and gate tensors;
  - upgate CUDA batch kernel/wait time;
  - cache-budget allocation under a nearly full 32 GB VRAM budget.

Tensor staging totals by prompt:

| prompt | down stage ms | gate stage ms | up stage ms | upgate-call wall ms | upgate up_wait ms | upgate gate_wait ms |
|---|---:|---:|---:|---:|---:|---:|
| dev_france_regression | 15725.5 | 3187.9 | 2215.4 | 14052.9 | 7139.1 | 7672.4 |
| dev_japan_factual | 14436.9 | 3276.8 | 2067.8 | 12599.1 | 6316.7 | 6733.1 |
| dev_linear_equation | 9418.2 | 4291.3 | 2856.4 | 6181.9 | 3129.4 | 3321.3 |
| dev_photosynthesis_factual | 16582.7 | 2707.8 | 1838.9 | 15350.4 | 7734.6 | 8252.9 |
| dev_python_reverse | 18018.3 | 3385.1 | 2347.6 | 16535.1 | 8465.5 | 8903.5 |

Top removable tensor-stage rows from the profile:

| row | stage ms | wall ms | misses | hit rate |
|---|---:|---:|---:|---:|
| `blk.1 ffn_gate_exps` type 22 | 5175.4 | 5377.9 | 369 | 48.7% |
| `blk.4 ffn_down_exps` type 23 | 3043.5 | 3228.3 | 2863 | 24.8% |
| `blk.6 ffn_down_exps` type 2 | 2995.9 | 3192.7 | 2530 | 33.6% |
| `blk.1 ffn_down_exps` type 11 | 2451.6 | 2646.0 | 2877 | 24.4% |
| `blk.3 ffn_down_exps` type 11 | 2339.6 | 2422.3 | 2724 | 28.5% |
| `blk.10 ffn_down_exps` type 2 | 2203.0 | 2269.3 | 2535 | 33.4% |
| `blk.58 ffn_down_exps` type 23 | 2200.3 | 2266.4 | 2571 | 32.5% |

Top layer combined pressure:

| layer | tensor stage ms | tensor wall ms | upgate wall ms | up_wait ms | gate_wait ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 8246.9 | 8653.0 | 1936.6 | 1809.0 | 1847.4 |
| 25 | 2470.4 | 2556.6 | 2109.7 | 1904.4 | 2024.0 |
| 10 | 2461.4 | 2546.6 | 2107.9 | 1897.8 | 2022.4 |
| 18 | 2434.0 | 2519.3 | 2100.5 | 1892.9 | 2016.0 |
| 20 | 2429.2 | 2516.5 | 2089.9 | 1880.7 | 2005.4 |
| 26 | 2467.5 | 2553.6 | 2055.5 | 1827.2 | 1943.8 |

VRAM/RAM constraint from the same run:

- `moe_stream_batch` requested `15000 MiB` expert VRAM cache.
- End-of-run CUDA free memory for France profile was only `872 MiB`.
- Therefore a new full-layer VRAM residency experiment cannot simply add entries. It must replace lower-yield cache entries or rebalance the upgate/down split.
- Host memory peak for France profile was `14627450880` bytes, but held-out reasoning/math previously reached the `15899996160` byte cgroup peak. RAM-tier expansions require extra margin, not just average fit.

Interpretation:

- The rejected `all-1200-minp4` result is consistent with this profile:
  - RAM tier can reduce SSD bytes;
  - but if it increases RAM H2D and does not reduce the exposed tensor-stage or upgate wait rows, token rate does not improve.
- Current bulk SSD throughput is already near the pure IO upper bound in normal runs:
  - held-out control: `10.039 GiB/s`;
  - full-dev control: `10.554 GiB/s`;
  - profile run: `9.256 GiB/s` because profiling adds overhead.
- The next optimization should not start from "load more random experts into RAM".

Next candidate design rules:

1. VRAM cache rebalance before RAM expansion.
   - Compute the current marginal value of upgate versus down slots from misses, stage/wait rows, and hit rates.
   - Try a default-off cache split that moves a small amount of VRAM from low-yield down entries to high-pressure upgate or targeted gate/down slab rows.
   - Expected upper bound must be stated as removable milliseconds from the Phase 4F table before testing.

2. Layer/role slab only for top rows.
   - Candidate rows: `blk.1 gate`, `blk.4 down`, `blk.6 down`, `blk.1 down`, `blk.3 down`.
   - Slab must replace lower-yield cache entries; do not exceed current VRAM cache budget.
   - If implemented in RAM rather than VRAM, it must prove lower exposed stage/wait, not just lower SSD bytes.

3. Upgate kernel/wait analysis.
   - `up22_gate22` and `up18_gate18` upgate-call rows contribute large kernel/wait time.
   - Before changing cache policy again, inspect whether `parallel_up_gate`, `parallel_stage`, and CUDA stream waits are actually overlapping for the dominant type pairs.
   - If a type pair is compute/stream-bound rather than transfer-bound, RAM/VRAM residency will have limited upside.

4. TTFT long-tail analysis remains separate.
   - The held-out reasoning/math prompt hit a severe TTFT long tail and cgroup memory peak.
   - Do not accept decode-only improvements if they make TTFT or host memory margin worse.

### Phase 4G planned A/B: protected `blk.1 gate` VRAM slab

Purpose:

- Test the smallest targeted VRAM slab suggested by Phase 4F before writing new runtime code.
- Use existing `GGML_MOE_VRAM_PROFILE*` support only; keep runtime behavior default-off.

Candidate:

- Tensor: `blk.1.ffn_gate_exps.weight`.
- Experts: all 384 experts.
- Expert size from route profile: `4.48 MiB`.
- Full slab size: `1.68 GiB`.
- Runtime env:
  - `GGML_MOE_VRAM_PROFILE=<candidate-profile.csv>`
  - `GGML_MOE_VRAM_PROFILE_PROTECT=1`
  - `GGML_MOE_VRAM_PROFILE_PRELOAD=1`
  - `GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1`
  - keep `GGML_MOE_VRAM_CACHE_MIB=15000`
  - keep `GGML_MOE_VRAM_CACHE_UPGATE_PCT=62`

Theory and upper bound:

- Phase 4F dev5 profile measured `blk.1 gate` tensor-stage:
  - stage `5175.4 ms`;
  - wall `5377.9 ms`;
  - misses `369`;
  - hit rate `48.7%`.
- Ignoring eviction losses, eliminating this stage from the normal dev5 control window would save at most about `5.18s`.
- Using the normal N96 dev5 control decode window from Phase 4E (`~212.3s`, `386` decode tokens), the ideal ceiling is roughly:
  - baseline `386 / 212.3 = 1.82 tok/s`;
  - ideal no-overhead `386 / (212.3 - 5.18) = 1.86 tok/s`;
  - maximum gain about `+2.5%`.
- Because the slab consumes `1.68 GiB` inside a nearly full upgate VRAM cache, real gains may be lower or negative if protected entries evict higher-value dynamic entries.

Execution:

1. Generate a temporary dev-only profile under the run directory with rows:
   - CSV format: `rank,count,expert_bytes,cumulative_bytes,tensor_base,expert_idx,tensor`;
   - `expert_idx=0..383`;
   - `tensor=blk.1.ffn_gate_exps.weight`;
   - `count` set high enough to protect entries when profile policy is active.
2. Run paired N32 full-dev cold-start A/B:
   - control: current `blk1_gate_full384.csv` RAM tier only;
   - candidate: same control plus protected `blk.1 gate` VRAM profile.
3. Acceptance for promotion to N96:
   - quality must pass the same N32 caveat as previous full-dev screens;
   - mean/median/min token rate must not regress;
   - TTFT must stay within `1.20x`;
   - host RAM peak `<15900000000`;
   - logs must show the profile loaded and protected/preloaded entries.
4. If N32 fails or improvement is within noise with worse TTFT/RAM, reject and do not run N96.

### Phase 4G N32 result: rejected

Timestamp: 2026-07-11 00:40 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4g-blk1gate-vram-n32-004035`

Candidate profile:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4g-blk1gate-vram-n32-004035/blk1_gate_full384_vram_profile.csv`

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/7 | 1.52 | 1.79 | 1.720 | 126.851 | 8932.37 | 10833.34 | 13.56 | 145.73 | 1557.15 |
| blk1-gate-vram | 6/7 | 1.45 | 1.63 | 1.599 | 136.095 | 9305.66 | 11199.78 | 13.56 | 149.34 | 1656.58 |

Quality note:

- `dev_linear_equation` failed in both runs due the known N32 truncation caveat.
- Other prompts passed.

Per-prompt deltas:

| prompt | tok delta | decode delta ms | TTFT delta ms | TTFT ratio |
|---|---:|---:|---:|---:|
| dev_france_regression | -0.13 | +1319.88 | +24.46 | 1.003 |
| dev_japan_factual | -0.14 | +1385.85 | +994.18 | 1.114 |
| dev_linear_equation | -0.10 | +1482.86 | -278.11 | 0.974 |
| dev_mixed_summary | -0.11 | +1255.74 | +517.86 | 1.048 |
| dev_photosynthesis_factual | -0.16 | +1690.21 | +205.14 | 1.027 |
| dev_python_reverse | -0.01 | +122.36 | -201.20 | 0.979 |
| dev_zh_france | -0.20 | +1987.95 | +373.34 | 1.046 |

Decision:

- Reject `blk1-gate-vram`; do not run N96.
- Reason:
  - mean token rate regressed by `0.121 tok/s`;
  - median regressed by `0.16 tok/s`;
  - decode time increased by `9.245s`;
  - `iouring_wait` increased by `3.617s`;
  - SSD IO increased by `99.43 GiB`;
  - TTFT remained within the 20% gate, but endpoint performance clearly regressed.

Mechanism diagnosis:

- Candidate logs confirm the intended profile loaded:
  - `profile preload: loaded 384 entries`;
  - `profile preload: blk.1.ffn_gate_exps.weight loaded=384`.
- But `GGML_MOE_VRAM_PROFILE_PROTECT=1` has broader semantics than intended:
  - it pins generic preload slots, not only the explicit profile rows;
  - in the France run, down cache changed from `preloads=4324 pinned=0 hit_rate=57.7%` to `preloads=578 pinned=578 hit_rate=31.5%`;
  - this destroyed down cache behavior and outweighed any possible `blk.1 gate` benefit.
- Therefore existing profile protection cannot be used directly for targeted slabs in the current SOTA configuration.

Next plan:

- Add a default-off runtime option that restricts protection to explicit profile-count rows only, for example:
  - `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`.
- Intended behavior:
  - profile preload rows with `profile_count > 0` may be pinned;
  - ordinary down/current prefetch preloads with `profile_count == 0` must not be pinned just because profile protection is enabled;
  - existing default behavior must remain unchanged unless the new env is set.
- Re-run the same N32 A/B only after this code change.
- Expected bound remains small: if profile-only protection works perfectly, the best case is still only the `blk.1 gate` stage bound (`~5.18s` on profiled dev5), so this remains a screening experiment, not a likely SOTA jump.

### Phase 4H N32 result: profile-only protection fixed the mechanism

Timestamp: 2026-07-11 00:58 CST.

Code change:

- Added default-off env:
  - `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`
- Behavior:
  - default behavior is unchanged when the env is unset;
  - when set, `GGML_MOE_VRAM_PROFILE_PROTECT=1` only pins preload inserts that have explicit `profile_count > 0` or were already recorded as profile-pinned keys;
  - ordinary down/current prefetch preloads with `profile_count == 0` are no longer pinned solely because profile protection is enabled.

Build:

```bash
cmake --build build-cuda-batch -j $(nproc)
```

Build result:

- passed;
- only pre-existing warnings were emitted.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4h-profile-only-blk1gate-n32-005854`

Candidate env delta:

- same as Phase 4G plus:
  - `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6/7 | 1.52 | 1.77 | 1.737 | 125.495 | 9091.74 | 11602.34 | 13.56 | 145.26 | 1557.15 |
| profile-only | 6/7 | 1.54 | 1.80 | 1.753 | 124.457 | 8858.62 | 10994.27 | 13.55 | 143.53 | 1557.15 |

Quality note:

- `dev_linear_equation` failed in both runs due the known N32 truncation caveat.
- Other prompts passed.

Per-prompt deltas:

| prompt | tok delta | decode delta ms | TTFT delta ms | TTFT ratio |
|---|---:|---:|---:|---:|
| dev_france_regression | +0.02 | -228.51 | -396.69 | 0.956 |
| dev_japan_factual | +0.01 | -124.52 | +262.21 | 1.031 |
| dev_linear_equation | +0.02 | -177.49 | -1066.66 | 0.908 |
| dev_mixed_summary | +0.00 | +5.20 | +92.65 | 1.008 |
| dev_photosynthesis_factual | +0.03 | -203.04 | +613.63 | 1.079 |
| dev_python_reverse | +0.01 | -115.19 | -535.21 | 0.946 |
| dev_zh_france | +0.02 | -194.34 | -225.98 | 0.972 |

Mechanism check:

- Candidate logs confirm profile preload happened:
  - `profile preload: blk.1.ffn_gate_exps.weight loaded=384`.
- Candidate logs also confirm the Phase 4G regression mechanism was fixed:
  - down cache stayed `preloads=4324 pinned=0 hit_rate=57.7%` on France, matching control;
  - upgate/down hit rates and IO bytes were unchanged in aggregate;
  - `iouring_wait` decreased by `1.73s`, not because bulk bytes fell but because the protected profile-only behavior avoided the previous down-cache damage.

Decision:

- Keep the default-off code change for N96 dev validation.
- Do not call this SOTA:
  - N32 signal is small;
  - held-out was not run;
  - profile-only slab must pass N96 dev before any held-out run.

Next:

- Run paired N96 full-dev with:
  - control: current RAM tier only;
  - candidate: same profile-only `blk.1 gate` VRAM slab.
- Promote to held-out only if N96 full-dev min/median/mean improve, quality passes, TTFT remains within `1.20x`, and RAM remains under the 16 GB gate.

### Phase 4H N96 result: profile-only `blk.1 gate` slab rejected

Timestamp: 2026-07-11 01:33 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4h-profile-only-blk1gate-n96-011553`

Candidate env delta:

- `GGML_MOE_VRAM_PROFILE=/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4h-profile-only-blk1gate-n96-011553/blk1_gate_full384_vram_profile.csv`
- `GGML_MOE_VRAM_PROFILE_PROTECT=1`
- `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1`
- `GGML_MOE_VRAM_PROFILE_PRELOAD=1`
- `GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1`
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=20`

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT mean ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 7/7 | 1.64 | 1.81 | 1.803 | 266.989 | 9170.23 | 9146.69 | 11181.33 | 13.56 | 285.925 | 2799.676 |
| profile-only | 7/7 | 1.62 | 1.81 | 1.783 | 269.140 | 9341.36 | 9515.89 | 12177.74 | 13.56 | 289.329 | 2799.676 |

Aggregate delta:

- min token rate: `-0.02 tok/s`
- median token rate: `+0.00 tok/s`
- mean token rate: `-0.020 tok/s`
- decode sum: `+2.151s`
- median TTFT: `+171.13ms`
- mean TTFT: `+369.20ms`
- max TTFT: `+996.41ms`
- `iouring_wait`: `+3.404s`
- SSD expert bytes: `+0.000 GiB`
- upgate hit rate: unchanged at `0.415`
- down hit rate: unchanged at `0.608`
- host RAM peak: `+1.52 MiB`, still under the 16 GB gate

Per-prompt deltas:

| prompt | token-rate delta | decode delta ms | TTFT delta ms | TTFT ratio | quality |
|---|---:|---:|---:|---:|---:|
| dev_france_regression | -0.09 | +2101.39 | +84.17 | 1.009 | pass/pass |
| dev_japan_factual | -0.01 | +63.47 | +188.68 | 1.022 | pass/pass |
| dev_linear_equation | -0.02 | +263.54 | +33.22 | 1.003 | pass/pass |
| dev_mixed_summary | +0.04 | -667.09 | +1349.89 | 1.125 | pass/pass |
| dev_photosynthesis_factual | +0.04 | -1146.34 | +259.81 | 1.036 | pass/pass |
| dev_python_reverse | +0.00 | +172.68 | +496.10 | 1.054 | pass/pass |
| dev_zh_france | -0.10 | +1363.10 | +172.56 | 1.022 | pass/pass |

Mechanism check:

- Candidate did preload the intended profile row:
  - `profile preload: blk.1.ffn_gate_exps.weight loaded=384`
- Down cache damage from Phase 4G did not recur:
  - France down cache stayed `preloads=11676 pinned=0 hit_rate=61.4%`, matching control.
- Upgate/down cache aggregate hit rates did not improve.
- SSD expert bytes did not decrease.
- Exposed `iouring_wait` increased instead of decreasing.

Decision:

- Reject this candidate.
- Do not run held-out.
- Do not promote `blk1_gate_full384_vram_profile.csv` into tracked SOTA profiles.
- Keep `GGML_MOE_VRAM_PROFILE_PROTECT_PROFILE_ONLY=1` only as default-off infrastructure because it fixed the Phase 4G over-pinning mechanism and does not change default behavior.

Reason:

- The initial N32 gain was noise or prompt-length dependent.
- The theoretical bound for protecting a single `blk.1 gate` role was too small.
- The protected slab did not reduce the real critical path: aggregate `iouring_wait`, SSD bytes, and cache hit rates did not improve.

Next:

- Stop testing isolated single-role slabs.
- Prioritize cross-role scheduling and storage layout:
  - same-layer up/gate/down miss co-submit after routing;
  - layer/role-aware RAM cache using memory currently occupied by low-value decode file cache;
  - pack layout that makes mixed-role active expert reads more contiguous and batchable.

### Phase 4I plan: global scheduler shadow before real co-submit work

Reason:

- Phase 4A proved the standalone gate/up/down co-submit hook does not run on the current Kimi fused path.
- Phase 4B proved there are many same-layer up/gate/down miss opportunities, but current-down overlap already handles many down rows.
- Phase 4C proved simply starting down overlap earlier can lose VRAM/cache budget or create IO contention.
- Phase 4H proved isolated single-role residency does not reduce the critical path.
- Before changing scheduling again, we need to know whether demand reads frequently collide with already-active prefetch reads, or whether the runtime is mostly serialized by unavoidable per-layer dependency.

Existing default-off diagnostic:

- `GGML_MOE_GLOBAL_EXPERT_SCHED_SHADOW=1`
- It records every expert-pack io_uring batch by `(source_idx, offset, nbytes)`.
- It classifies a batch as prefetch when `trace_op` contains `prefetch` or `overlap`.
- It reports:
  - total batches and tasks;
  - demand versus prefetch tasks;
  - active duplicate reads;
  - demand reads that hit active prefetch;
  - prefetch reads that hit active demand;
  - demand-demand and prefetch-prefetch overlap;
  - max active set and batch histogram.

Hypothesis:

- If `demand_hit_prefetch` is material, a real scheduler could let demand wait on or steal the matching in-flight prefetch instead of issuing another read or stalling behind a separate batch.
- If `active_duplicates` is near zero, queue starvation is mostly caused by sequential layer dependency and small known-active sets; a global scheduler will not help much.
- If duplicates are mostly `prefetch_hit_prefetch` or `demand_hit_demand`, then the fix is deduplication/packing, not more aggressive prefetch.

Theoretical upper bound:

- Hard bound is the `iouring_wait_us` associated with duplicate active reads.
- The shadow does not measure exact saved milliseconds per duplicate, so the first bound will be conservative:
  - duplicate task ratio = `active_duplicates / tasks`;
  - useful duplicate ratio = `demand_hit_prefetch / demand_tasks`;
  - possible saved wait upper bound = current aggregate `iouring_wait_s * useful duplicate ratio`.
- If useful duplicate ratio is below a few percent, skip implementation and move to RAM/pack layout.

Experiment:

1. Run cold-start N32 dev3 paired against current control shape.
2. Candidate env adds only:

```bash
GGML_MOE_GLOBAL_EXPERT_SCHED_SHADOW=1
```

3. Keep the current reproducible control env:
   - `UPGATE_PCT=62`
   - `GGML_MOE_RAM_TIER_MIB=1800`
   - `GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv`
   - `GGML_MOE_RAM_TIER_PIN=1`
   - `GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1`
   - `GGML_MOE_RAM_TIER_PRELOAD_THREADS=4`
   - `MemoryMax=15900000000`
   - `MemorySwapMax=0`

Acceptance for diagnostic:

- Quality must match control.
- Host RAM must stay under the 16 GB gate.
- TTFT must remain within `1.20x`; shadow overhead should be small.
- The atexit global scheduler line must appear in stderr.
- Proceed to implementation only if useful demand-prefetch overlap is large enough to justify a real default-off scheduler.

Decision after diagnostic:

- If useful demand-prefetch overlap is high:
  - design a default-off scheduler that records active in-flight prefetch by `(source_idx, offset, nbytes)`;
  - demand can wait on or reuse the prefetch completion event;
  - duplicate reads must be suppressed;
  - stale prefetch must be capped.
- If useful overlap is low:
  - do not implement global scheduler;
  - move to storage layout: RAM tier replacement of low-value file cache and pack locality/role layout.

### Phase 4I result: global scheduler shadow shows no reusable in-flight overlap

Timestamp: 2026-07-11 01:38 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4i-global-sched-shadow-n32-dev3-0138`

Candidate env delta:

```bash
GGML_MOE_GLOBAL_EXPERT_SCHED_SHADOW=1
```

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB | up hit | down hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 3/3 | 1.78 | 1.80 | 1.797 | 51.739 | 8309.59 | 9506.20 | 13.55 | 58.851 | 617.072 | 0.441 | 0.585 |
| shadow | 3/3 | 1.79 | 1.81 | 1.817 | 51.204 | 9369.86 | 9483.67 | 13.56 | 57.989 | 617.072 | 0.441 | 0.585 |

Shadow aggregate:

| counter | value |
|---|---:|
| batches | 17160 |
| tasks | 115722 |
| demand tasks | 102681 |
| prefetch tasks | 13041 |
| active duplicates | 0 |
| demand hit active prefetch | 0 |
| prefetch hit active demand | 0 |
| demand hit active demand | 0 |
| prefetch hit active prefetch | 0 |
| batch hist 1 | 375 |
| batch hist 2-4 | 7586 |
| batch hist 5-8 | 8671 |
| batch hist gt32 | 528 |

Ratios:

- duplicate task ratio: `0.000000`
- useful demand-prefetch overlap ratio: `0.000000`
- saved-wait upper bound from demand-prefetch reuse: `0.000s`

Decision:

- Do not implement a real global in-flight prefetch reuse scheduler for the current Kimi path.
- The diagnostic proves that demand reads are not colliding with active prefetch reads under the current submit/wait scopes.
- Queue starvation is not caused by duplicate simultaneous reads that can be fixed by demand waiting on existing prefetch work.
- The small token-rate difference in the shadow run is normal N32 noise; it is not a SOTA claim and does not justify N96.

Next:

- Move to storage-layout/RAM-layout work.
- Do not repeat `GGML_MOE_IO_SORT_OFFSET=1` as a new idea:
  - it is already enabled in `.Agent/run-tools/kimi-general-prompt-repro.sh`;
  - historical sort-off ablation rejected disabling it.
- The next diagnostic must identify:
  - what decode-time file-backed memory is low value;
  - which expert reads dominate exposed wait under current general prompts;
  - whether a larger explicit RAM expert tier can replace low-value file cache without fragmenting SSD/RAM batches;
  - whether pack-layout/locality changes can make the existing sorted O_DIRECT reads more contiguous or reduce per-batch wait.

### Phase 4J plan: storage-layout and RAM-tier diagnostic

Reason:

- Current decode still spends most time in expert movement, and Phase 4I rules out useful in-flight duplicate reuse.
- The 16 GB host RAM is largely occupied by file-backed pages, but that cache is not a controlled expert cache and does not eliminate hundreds of GiB of expert-pack IO.
- Previous larger RAM tiers reduced SSD bytes but sometimes increased exposed wait, likely because RAM hits fragmented batches or added H2D/staging pressure.
- The next step is to profile storage layout at the batch level before designing another RAM tier or pack overlay.

Hypothesis:

- A RAM tier can help only if it stores entries that:
  - sit on the critical path;
  - are requested repeatedly across general prompts;
  - can be grouped by layer/role/size so RAM hits do not destroy SSD batch size;
  - replace low-value file cache without causing refault or TTFT regressions.
- A pack-layout change can help only if current active batches have enough physical locality that grouping/co-location reduces wait, not just bytes.

Diagnostic run:

1. Run cold-start N32 dev3 with current SOTA env and profiling only.
2. Add:

```bash
GGML_MOE_IO_BATCH_PROFILE_OUT=$OUT/io-batch-profile.csv
GGML_MOE_IO_LOCALITY_PROFILE_OUT=$OUT/io-locality-profile.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$OUT/io-wait-trace.csv
GGML_MOE_H2D_COALESCE_PROFILE_OUT=$OUT/h2d-coalesce-profile.csv
```

3. Keep:
   - `GGML_MOE_IO_SORT_OFFSET=1`, already default in the repro script;
   - current 1800 MiB RAM tier;
   - `MemoryMax=15900000000`;
   - cold start and quality gates.

Analysis after run:

- Join batch profile and locality profile by sequence.
- For each op/tensor/layer/role:
  - sum read jobs, read bytes, wait ms, wall ms;
  - compute jobs-per-batch, inflight avg/max, source switches, unique sources;
  - compute span/read and gap/read locality;
  - identify whether slow batches are small random reads or large contiguous groups.
- From `memory.stat.final.txt`, record file/active_file/inactive_file/anon/kernel and refault counters.
- Generate RAM-tier candidates ranked by removable wait, not by hit count alone:
  - role/layer slabs for high-wait low-hit layers;
  - compact next-hot expert entries grouped by tensor size and layer;
  - avoid mixing RAM and SSD inside batches when it reduces SSD read batch size.

Acceptance for moving past diagnostic:

- Produce a candidate list with estimated RAM cost, removable wait bound, expected TTFT cost, and fragmentation risk.
- Do not implement another RAM tier unless the estimated upper bound exceeds the observed noise band and explains why previous larger tiers regressed.
- If no RAM/pack-layout candidate has a credible bound, stop storage-layout tuning and move to byte-reduction/compression-style methods.

### Phase 4J result: storage profile points to wait-weighted RAM replacement

Timestamp: 2026-07-11 02:05 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4j-storage-layout-profile-n32-dev3-0205`

Profiling env:

```bash
GGML_MOE_IO_BATCH_PROFILE_OUT=$OUT/io-batch-profile.csv
GGML_MOE_IO_LOCALITY_PROFILE_OUT=$OUT/io-locality-profile.csv
GGML_MOE_IO_WAIT_TRACE_OUT=$OUT/io-wait-trace.csv
GGML_MOE_H2D_COALESCE_PROFILE_OUT=$OUT/h2d-coalesce-profile.csv
```

Quality and runtime:

| prompt | quality | tok/s | TTFT ms | decode ms | runs | RAM peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| dev_france_regression | pass | 1.79 | 8657.52 | 17358.49 | 31 | 13.56 |
| dev_japan_factual | pass | 1.81 | 8434.96 | 17145.48 | 31 | 13.56 |
| dev_photosynthesis_factual | pass | 1.83 | 7299.16 | 16935.96 | 31 | 13.41 |

Final cgroup memory shape:

- `file`: about `11.22 GiB`
- `active_file`: about `11.18-11.20 GiB`
- `inactive_file`: about `0.031 GiB`
- `workingset_refault_file`: `0`
- Interpretation: after process exit the remaining cgroup memory is mostly active file-backed pages; this confirms RAM is still not being used as a controlled expert cache, but this final sample alone does not prove which pages were useful during decode.

IO batch aggregate:

| op | calls | jobs | read jobs | RAM/prefetch jobs | wait ms | wall ms | slot wait ms | weighted inflight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| runtime_load | 14564 | 103399 | 102681 | 718 | 49148.14 | 67052.34 | 9276.71 | 4.906 |
| current_down_overlap | 2692 | 13041 | 13041 | 0 | 6983.06 | 7550.71 | 31.97 | 3.395 |

Current RAM tier behavior:

- `ram-batch-profile.csv` rows: `96`
- all hits were `runtime_load` on `blk.1.ffn_gate_exps.weight`
- RAM hit jobs: `718`
- RAM hit bytes: `3.144 GiB`
- RAM enqueue/wall: `2143.80 / 2156.79 ms`
- The first large `blk.1 gate` RAM-hit batch alone cost about `762 ms` wall.
- This reinforces the Phase 4H conclusion: the current `blk1_gate_full384.csv` RAM tier is not aligned with the top exposed wait rows.

Top runtime-load wait tensors:

| tensor | role | read jobs | wait ms | wall ms | slot wait ms |
|---|---:|---:|---:|---:|---:|
| `blk.4.ffn_down_exps.weight` | down | 819 | 466.14 | 921.01 | 85.64 |
| `blk.6.ffn_down_exps.weight` | down | 740 | 443.91 | 916.91 | 74.12 |
| `blk.29.ffn_gate_exps.weight` | gate | 699 | 407.95 | 490.42 | 55.40 |
| `blk.7.ffn_down_exps.weight` | down | 719 | 398.76 | 492.16 | 65.67 |
| `blk.10.ffn_down_exps.weight` | down | 719 | 396.38 | 484.04 | 63.20 |
| `blk.9.ffn_down_exps.weight` | down | 699 | 386.33 | 486.94 | 75.42 |
| `blk.28.ffn_gate_exps.weight` | gate | 710 | 384.06 | 460.84 | 49.45 |
| `blk.1.ffn_down_exps.weight` | down | 803 | 380.46 | 677.12 | 48.07 |

Locality diagnosis:

- The slowest individual batches have `60-75` read jobs, so the problem is not only tiny batch size.
- Physical locality is poor even with `GGML_MOE_IO_SORT_OFFSET=1`:
  - top slow batch `blk.4 down`: `520.7 MiB` read bytes but `span/read=7.77`;
  - top slow batch `blk.6 down`: `543.4 MiB` read bytes but `span/read=7.43`;
  - several top batches have `span/read` between `5x` and `9x`.
- `h2d-coalesce-profile.csv` was not generated, so this run did not hit the H2D coalesce profile path.

Offline wait-weighted RAM candidate screen:

- Input traces:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343/*/io-read-trace.csv`
- Method:
  - focus on `runtime_load`;
  - distribute each batch's `io-wait-trace` wait across read rows by byte share;
  - rank only actual SSD-read misses, so current RAM-tier hits are naturally excluded.

Top layer/role wait-weighted rows:

| layer/role | entries | trace MiB | wait score ms | prompts |
|---|---:|---:|---:|---:|
| blk4 down | 308 | 14512.2 | 1059.99 | 7 |
| blk6 down | 307 | 13964.1 | 1031.17 | 7 |
| blk29 gate | 324 | 9375.4 | 941.71 | 7 |
| blk28 gate | 305 | 9342.9 | 914.03 | 7 |
| blk7 down | 299 | 13814.2 | 911.46 | 7 |
| blk10 down | 304 | 13499.2 | 906.85 | 7 |
| blk9 down | 308 | 13680.4 | 905.64 | 7 |
| blk8 down | 302 | 13334.0 | 902.63 | 7 |

Candidate budget screen:

| candidate | RAM MiB | entries | wait score ms | trace bytes GiB | rows |
|---|---:|---:|---:|---:|---:|
| down-only 1800 | 1794.4 | 254 | 3858.49 | 51.50 | 7452 |
| down-only 2400 | 2399.0 | 342 | 4664.07 | 61.97 | 9013 |
| down-only 3000 | 2994.2 | 427 | 5369.27 | 71.20 | 10360 |
| up/gate 1800 | 1795.7 | 386 | 6254.14 | 51.69 | 11360 |
| up/gate 2400 | 2397.7 | 514 | 7628.85 | 63.33 | 13882 |
| all roles 1800 | 1796.7 | 359 | 6633.38 | 61.40 | 12195 |
| all roles 2400 | 2398.9 | 481 | 8126.24 | 75.12 | 14949 |

Decision:

- The next RAM experiment should replace the current `blk1_gate_full384.csv` tier rather than add to it.
- Do not test full-layer down slabs first:
  - they are expensive (`blk4_down_full384.csv` is `2856 MiB`);
  - previous full-layer/pinned RAM attempts showed RAM-H2D and pressure side effects;
  - wait-weighted sparse entries have a better RAM-to-wait bound.
- First candidate should be `up/gate 1800`:
  - same RAM budget as current SOTA tier;
  - focuses on critical-path up/gate runtime misses;
  - avoids large down entries and current-down overlap competition;
  - estimated full-dev7 wait score `6254 ms`, enough to exceed N32 noise if the bound materializes.

### Phase 4K plan: wait-weighted up/gate RAM-tier replacement A/B

Hypothesis:

- Replacing the current single-layer `blk1_gate_full384.csv` RAM tier with a multi-layer wait-weighted up/gate tier should reduce exposed `runtime_load` wait on general prompts.
- Because RAM budget stays at about `1800 MiB`, TTFT and host RAM pressure should be comparable to the current SOTA tier.
- The main risk is fragmented RAM hits reducing SSD batch size or adding RAM->VRAM H2D overhead; therefore N32 must prove lower `iouring_wait`, not just more RAM hits.

Candidate generation:

- Use Phase 4E full-dev7 traces.
- Rank `runtime_load` up/gate misses by wait-weighted score.
- Require:
  - `min_prompts >= 2`;
  - `min_count >= 2`;
  - budget about `1800 MiB`;
  - output profile stored under the experiment run root until accepted.

N32 A/B:

1. Control:
   - current SOTA RAM tier: `.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv`
2. Candidate:
   - generated wait-weighted up/gate 1800 profile.
3. Keep all other env identical:
   - `UPGATE_PCT=62`;
   - `GGML_MOE_RAM_TIER_MIB=1800`;
   - `GGML_MOE_RAM_TIER_PIN=1`;
   - `GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1`;
   - `GGML_MOE_RAM_TIER_PRELOAD_THREADS=4`;
   - `MemoryMax=15900000000`;
   - cold start.

Acceptance for N96 dev:

- N32 dev3 quality matches control.
- Host RAM remains below the hard gate.
- TTFT ratio remains `<=1.20`.
- Minimum and median token rate improve versus paired control.
- Aggregate `iouring_wait` decreases by at least `1s` on N32 dev3 or the gain is not credible.
- RAM tier profile must show the candidate is not just increasing RAM-H2D wall time.

### Phase 4K result: wait-weighted up/gate RAM replacement rejected

Timestamp: 2026-07-11 02:25 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4k-wait-upgate-1800-n32-dev3-0225`

Generated candidate:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4k-wait-upgate-1800-n32-dev3-0225/wait-upgate-1800.profile.csv`
- entries: `386`
- size: `1795.7 MiB`
- full-dev7 wait score: `6254.14 ms`
- trace bytes represented: `51.69 GiB`
- trace rows represented: `11360`

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control `blk1_gate_full384` | 3/3 | 1.89 | 1.90 | 1.913 | 48.631 | 8285.77 | 9235.67 | 13.55 | 55.002 | 617.072 |
| wait-upgate-1800 | 3/3 | 1.83 | 1.90 | 1.903 | 48.894 | 7924.32 | 8458.33 | 13.61 | 54.550 | 596.538 |

Per-prompt deltas:

| prompt | token-rate delta | decode delta ms | TTFT ratio | quality |
|---|---:|---:|---:|---:|
| dev_france_regression | +0.08 | -677.72 | 0.858 | pass/pass |
| dev_japan_factual | -0.05 | +414.16 | 1.021 | pass/pass |
| dev_photosynthesis_factual | -0.06 | +526.09 | 0.963 | pass/pass |

RAM and IO deltas:

| metric | control | candidate | delta |
|---|---:|---:|---:|
| `iouring_wait` | 55.002s | 54.550s | -0.452s |
| SSD expert bytes | 617.072 GiB | 596.538 GiB | -20.535 GiB |
| RAM hit jobs | 718 | 5195 | +4477 |
| RAM hit bytes | 3.144 GiB | 23.679 GiB | +20.535 GiB |
| RAM wall | 2.426s | 14.351s | +11.925s |
| RAM enqueue | 2.426s | 0.048s | -2.377s |
| host RAM peak | 13.55 GiB | 13.61 GiB | +53.5 MiB |

Decision:

- Reject candidate.
- Do not run N96.
- Do not promote the generated profile.

Reason:

- It reduced SSD bytes but did not remove the critical wait wave:
  - `iouring_wait` fell by only `0.452s`, below the required `1s` N32 credibility gate.
  - decode sum regressed by `0.263s`.
  - min token rate regressed from `1.89` to `1.83`.
- The candidate created many partial RAM hits:
  - RAM jobs increased from `718` to `5195`;
  - RAM bytes increased by `20.535 GiB`;
  - RAM wall increased by `11.925s`;
  - SSD bytes fell by the same amount, but the remaining SSD jobs still determined the batch tail.
- This confirms the working theory: a scattered expert RAM tier can reduce bytes while leaving enough SSD reads in each batch to keep the same wait latency.

Next:

- Do not test broader scattered wait-weighted RAM tiers.
- The next RAM/storage candidate must be batch-coherent:
  - select entries only when they cover most jobs in a slow batch;
  - or use a layer/role slab that turns an entire high-wait batch into RAM hits;
  - compare expected saved wait against RAM-H2D wall before running.
- Use `kimi_ram_candidate_multidev_screen.py` with stricter batch coverage, for example:
  - `--min-batch-hits >= 4`;
  - `--min-batch-hit-pct >= 75`;
  - separate up/gate and down candidates;
  - reject candidates that mostly create partial hits across many batches.

### Phase 4L plan: batch-coherent RAM/slab screen before another A/B

Reason:

- Phase 4K proved that scattered RAM hits can reduce SSD bytes without reducing token time.
- The missing property is batch coherence: if a slow batch still has enough SSD reads, the batch tail and `iouring_wait` remain.
- The next candidate must show that it can eliminate or heavily shrink whole slow batches, not just replace isolated rows with RAM-H2D.

Inputs:

- Phase 4E full-dev7 per-expert traces:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343/*/io-read-trace.csv`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343/*/io-wait-trace.csv`
- Phase 4J storage profile:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4j-storage-layout-profile-n32-dev3-0205`

Offline method:

1. For each IO batch, join per-row expert reads with total batch wait.
2. Test candidate families without running the model:
   - existing scattered candidates (`upgate-1800`, `down-1800`, `all-1800`, and the Phase 4K wait-weighted profile);
   - layer/role slabs such as `blk4 down`, `blk6 down`, `blk29 gate`, `blk28 gate`, `blk7 down`, `blk10 down`;
   - paired role slabs for layers where up/gate miss together.
3. For each candidate family, compute:
   - RAM MiB;
   - selected entries;
   - total candidate row hits;
   - candidate bytes;
   - number of batches touched;
   - number of batches with `>=75%` rows covered;
   - number of fully covered batches;
   - wait upper bound from covered batches;
   - wait upper bound from fully covered batches only;
   - remaining SSD rows in touched batches.

Acceptance to run a real N32 A/B:

- Candidate RAM cost must fit the 16 GB host gate when replacing the current 1800 MiB tier or must justify a larger tier with explicit page-cache replacement.
- Candidate must cover enough slow batches to have a credible N32 wait upper bound:
  - preferred: fully covered wait upper bound `>=1s`;
  - minimum: `>=75%` covered wait upper bound `>=2s`;
  - scattered partial-hit candidates are rejected even if byte coverage is high.
- Candidate must explain why it should avoid Phase 4K's failure mode:
  - either entire batch becomes RAM-hit;
  - or remaining SSD rows per touched batch are small enough that the original wait wave should shrink.

If no candidate passes:

- Stop RAM-tier tuning for the current pack layout.
- Move to pack-layout work:
  - physically co-locate experts that co-occur in slow batches;
  - evaluate whether a layout overlay can reduce `span/read` from the current `5x-9x` range;
  - only then revisit RAM tier.

### Phase 4L result: only layer/role slabs are batch-coherent

Timestamp: 2026-07-11 02:45 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4l-batch-coherent-ram-screen`

Outputs:

- `batch-coherent-screen.csv`
- `batch-coherent-screen.report.json`

Key result:

- Scattered profiles touch many batches but fully cover almost none:
  - Phase 4K wait-upgate-1800 touched `6035` runtime batches but fully covered only `22`;
  - full-batch wait upper bound was only `35.5 ms`;
  - this explains why it reduced SSD bytes without lowering token time.
- Layer/role slabs fully cover their target batches:
  - they have much lower touched-batch count but convert entire slow batches to RAM hits.

Top runtime-load full-batch candidates:

| candidate | RAM MiB | entries | touched batches | dominant batches | full batches | touch wait ms | full wait ms | hit GiB | remaining rows in touched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| slab blk28+29 upgate | 7560.0 | 1536 | 896 | 790 | 896 | 3576.5 | 3576.5 | 33.58 | 0 |
| slab blk4+6+7 down | 8904.0 | 1152 | 672 | 629 | 672 | 3002.6 | 3002.6 | 41.30 | 0 |
| slab blk4+6 down | 5880.0 | 768 | 448 | 421 | 448 | 2091.2 | 2091.2 | 27.81 | 0 |
| slab blk28+29 gate | 4116.0 | 768 | 448 | 395 | 448 | 1855.7 | 1855.7 | 18.28 | 0 |
| slab blk29 upgate | 3780.0 | 768 | 448 | 390 | 448 | 1801.3 | 1801.3 | 16.82 | 0 |
| slab blk28 upgate | 3780.0 | 768 | 448 | 400 | 448 | 1775.3 | 1775.3 | 16.76 | 0 |
| tracked blk4 down | 2856.0 | 384 | 224 | 217 | 224 | 1060.0 | 1060.0 | 14.17 | 0 |
| slab blk6 down | 3024.0 | 384 | 224 | 204 | 224 | 1031.2 | 1031.2 | 13.64 | 0 |
| slab blk29 gate | 2058.0 | 384 | 224 | 195 | 224 | 941.7 | 941.7 | 9.16 | 0 |
| slab blk28 gate | 2058.0 | 384 | 224 | 200 | 224 | 914.0 | 914.0 | 9.12 | 0 |
| Phase 4K wait-upgate-1800 | 1795.7 | 386 | 6035 | 43 | 22 | 27053.1 | 35.5 | 51.69 | 64002 |
| Phase 4E upgate-1800 | 1799.2 | 378 | 5999 | 26 | 18 | 26677.5 | 27.2 | 52.34 | 64162 |

Decision:

- Do not run any more scattered RAM tiers.
- A real A/B is justified only for a layer/role slab with full-batch coverage.
- Start with `tracked_blk4_down` because:
  - it is already tracked;
  - it has the smallest cost among candidates whose full-batch wait upper bound exceeds `1s`;
  - it fully covers `224` runtime-load batches;
  - its added RAM over the current tier is about `1.1 GiB`, likely still under the 16 GB host gate.

Risk:

- Upper bound is only about `1.06s` on N32 dev7 traces.
- The slab will replace SSD reads with RAM->VRAM H2D/staging; if H2D or RAM copy wall is comparable to O_DIRECT wait, token rate can still tie or regress.
- If `blk4 down` helps but is too small, a second candidate can test `blk4+6 down`, but its `5880 MiB` tier risks RAM/TTFT pressure and must not be tried unless single-layer data is positive.

### Phase 4M plan: tracked `blk4 down` RAM slab N32 A/B

Hypothesis:

- Replacing `blk1_gate_full384.csv` with `blk4_down_full384.csv` converts full `blk.4 down` runtime-load batches to RAM hits.
- Unlike Phase 4K, this should remove whole batch wait waves rather than leaving residual SSD tails.

Candidate:

- RAM profile: `.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk4_down_full384.csv`
- Profile size: `2856 MiB`
- Runtime env:

```bash
GGML_MOE_RAM_TIER_MIB=3000
GGML_MOE_RAM_TIER_PROFILE=.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk4_down_full384.csv
GGML_MOE_RAM_TIER_PIN=1
GGML_MOE_RAM_TIER_PIN_MIB=3000
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=4
```

Paired control:

- current SOTA RAM profile: `.Agent/profiles/kimi/ram-tier/gp112-prompt0-layer-role/blk1_gate_full384.csv`
- `GGML_MOE_RAM_TIER_MIB=1800`
- `GGML_MOE_RAM_TIER_PIN_MIB=1800`

N32 acceptance:

- quality matches control;
- host RAM peak `<15900000000`;
- TTFT ratio `<=1.20`;
- min and median token rate improve, or at least decode sum and `iouring_wait` clearly improve without min regression;
- `iouring_wait` should fall by close to the `~1s` upper bound;
- RAM wall must not increase enough to erase the wait gain.

If rejected:

- Do not test larger multi-layer down slabs immediately.
- Move to pack-layout/locality overlay, because RAM replacement has failed both scattered and small slab modes.

### Phase 4M result: tracked `blk4 down` RAM slab rejected

Timestamp: 2026-07-11 02:55 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4m-blk4down-ram-slab-n32-dev3-0255`

Aggregate:

| run | quality | tok/s min | tok/s median | tok/s mean | decode sum s | TTFT median ms | TTFT max ms | RAM peak GiB | iouring wait s | iouring bytes GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control `blk1 gate` | 3/3 | 1.95 | 1.95 | 1.970 | 47.201 | 8165.57 | 8912.50 | 13.55 | 53.241 | 617.072 |
| `blk4 down` slab | 3/3 | 1.74 | 1.83 | 1.813 | 51.281 | 9310.48 | 10157.07 | 14.67 | 58.956 | 614.268 |

Per-prompt deltas:

| prompt | token-rate delta | decode delta ms | TTFT ratio | quality |
|---|---:|---:|---:|---:|
| dev_france_regression | -0.18 | +1535.71 | 0.995 | pass/pass |
| dev_japan_factual | -0.08 | +676.73 | 1.244 | pass/pass |
| dev_photosynthesis_factual | -0.21 | +1866.98 | 1.312 | pass/pass |

RAM and IO deltas:

| metric | control | `blk4 down` | delta |
|---|---:|---:|---:|
| `iouring_wait` | 53.241s | 58.956s | +5.715s |
| SSD expert bytes | 617.072 GiB | 614.268 GiB | -2.804 GiB |
| RAM hit jobs | 718 | 819 | +101 |
| RAM hit bytes | 3.144 GiB | 5.949 GiB | +2.804 GiB |
| RAM wall | 2.343s | 0.004s | -2.339s |
| host RAM peak | 13.55 GiB | 14.67 GiB | +1.12 GiB |

Decision:

- Reject candidate.
- Do not run N96.
- Do not test larger multi-layer down slabs.

Reason:

- The slab fully covered `blk.4 down` batches, but it replaced the existing `blk1 gate` RAM tier rather than adding to it.
- Net SSD byte reduction was only `2.804 GiB`, because `blk4 down` saved `5.949 GiB` while losing `blk1 gate` RAM hits worth `3.144 GiB`.
- `iouring_wait` increased by `5.715s`, so removing those `blk4 down` reads did not reduce the exposed critical path.
- TTFT failed the `<=1.20x` gate on two of three prompts.
- Host RAM peak remained under the hard cap but rose to `14.67 GiB`, leaving little room for a combined `blk1 gate + blk4 down` tier.

Conclusion:

- RAM tier has now failed in both forms:
  - scattered high-reuse entries reduce bytes but not batch tails;
  - a small batch-coherent slab loses the current gate tier and worsens exposed wait.
- Under the current pack layout and 16 GB host RAM limit, RAM-tier tuning is not the next best path.

### Phase 4N plan: pack-layout locality screen

Reason:

- Phase 4J showed slow batches often have `60-75` read jobs but poor physical locality:
  - `span/read` often falls in the `5x-9x` range.
- Phase 4L showed RAM can cover entire batches only with large layer/role slabs.
- Phase 4M showed replacing the existing tier with such a slab worsens wait and TTFT.
- Therefore the next storage direction is not more RAM, but making the existing SSD reads more locality-friendly.

Existing diagnostic tool:

- `.Agent/run-tools/kimi_io_trace_pack_layout_screen.py`

Purpose:

- Use actual `io-read-trace.csv` rows to estimate whether a different expert-pack physical layout could reduce extents/span/gap for active expert batches.
- This does not create a pack and does not change runtime behavior.

Experiment:

1. Use Phase 4E full-dev7 traces:

```bash
/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343/*/io-read-trace.csv
```

2. Run screen with:

```bash
python3 .Agent/run-tools/kimi_io_trace_pack_layout_screen.py \
  --input-root /root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343 \
  --roles up,gate,down \
  --max-jobs 0 \
  --max-gap-mib 1 \
  --out-json /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4n-pack-layout-screen/pack-layout-screen.json \
  --out-md /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4n-pack-layout-screen/pack-layout-screen.md
```

3. Evaluate:
   - current extents/read bytes/span bytes/gap bytes;
   - ideal tensor layout;
   - layer/role layout;
   - layer layout;
   - static per-tensor layouts such as frequency/first-use/greedy-pair if reported.

Acceptance to build a real overlay/repacked expert pack:

- A candidate layout must reduce extents or span/read enough to plausibly save more than N32 noise:
  - target `>=1s` N32 wait upper bound;
  - or a clear reduction in top slow batch span/read from `5x-9x` toward `~1x-2x`.
- It must not require prompt-specific held-out data.
- It must be implementable as a dev-trained general pack or overlay, not a France-only pack.
- Before any pack artifact is used in runtime A/B, record exact build command, source trace set, included tensors, artifact size, and rollback path.

### Phase 4N result: pack layout has a bound, but requires coalesced reads

Timestamp: 2026-07-11 03:05 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4n-pack-layout-screen`

Outputs:

- `pack-layout-screen.json`
- `pack-layout-screen.md`
- `pack-layout-screen-maxjobs8.json`
- `pack-layout-screen-maxjobs8.md`

Full trace screen (`--max-jobs 0`):

| layout | extents | read reduction | read GiB | span GiB | gap GiB |
|---|---:|---:|---:|---:|---:|
| current offsets | 260231 | 0 | 1557.450 | 1557.444 | 0.000 |
| ideal tensor | 66523 | 193708 | 1557.450 | 1557.450 | 0.000 |
| ideal layer_role | 66523 | 193708 | 1557.450 | 1557.450 | 0.000 |
| static expert_id | 264679 | -4448 | 1557.450 | 1557.450 | 0.000 |
| static frequency | 239283 | 20948 | 1557.450 | 1557.450 | 0.000 |
| static first_use | 210090 | 50141 | 1557.450 | 1557.450 | 0.000 |
| static greedy_pair | 163708 | 96523 | 1557.450 | 1557.450 | 0.000 |

Small-batch screen (`--max-jobs 8`):

| layout | extents | read reduction | read GiB | span GiB | gap GiB |
|---|---:|---:|---:|---:|---:|
| current offsets | 192928 | 0 | 1046.189 | 1046.188 | 0.000 |
| ideal tensor | 64115 | 128813 | 1046.189 | 1046.189 | 0.000 |
| ideal layer_role | 64115 | 128813 | 1046.189 | 1046.189 | 0.000 |
| static expert_id | 192424 | 504 | 1046.189 | 1046.189 | 0.000 |
| static frequency | 186416 | 6512 | 1046.189 | 1046.189 | 0.000 |
| static first_use | 160328 | 32600 | 1046.189 | 1046.189 | 0.000 |
| static greedy_pair | 135009 | 57919 | 1046.189 | 1046.189 | 0.000 |

Interpretation:

- The locality bound is real:
  - even for small batches, an ideal same-tensor/layer-role layout could cut extents by `128813`;
  - a dev-trained static `greedy_pair` per-tensor layout could cut extents by `57919`.
- This is not a bytes-reduction bound:
  - read GiB is unchanged;
  - span GiB is unchanged in this model because it treats coalesced adjacent entries as zero-gap extents.
- Current runtime does not automatically exploit this:
  - it sorts by offset (`GGML_MOE_IO_SORT_OFFSET=1`);
  - it still creates one read plan per expert row;
  - pack layout alone will not reduce read count unless the runtime merges adjacent/low-gap spans.
- Historical warning:
  - broad read coalescers and blocking adjacent coalesced staging were previously rejected because they moved waits to slot reuse / blocking read / synchronization paths.

Decision:

- Do not build a layout-only pack and expect speedup.
- Do not retry the old broad coalescer.
- The only plausible storage-layout path is a narrow adjacent-span async coalescer paired with a dev-trained per-tensor layout.

### Phase 4O plan: narrow async adjacent-span coalescer design

Reason:

- RAM tier is exhausted for current constraints.
- Pack layout has a meaningful extents bound, especially under `greedy_pair`, but the runtime needs to merge adjacent spans to realize it.
- Previous coalescers failed because they were too broad or introduced blocking/slot waits on the critical path.

Design constraints:

- Default-off only.
- Do not change default runtime behavior.
- Do not use held-out prompts for layout training.
- No prompt-specific France pack.
- Preserve existing CQE-to-H2D streaming shape as much as possible.
- Merge only adjacent or very-low-gap entries within the same source and same tensor/role batch.
- Do not wait for a whole coalesced group before making earlier slices available if the old path would have streamed them earlier.
- Coalesced slot pool must be large enough to avoid the previous slot-wait failure mode, or the experiment should not proceed.

First implementation step before code:

1. Use Phase 4N JSON to estimate a safe narrow subset:
   - same tensor only;
   - `gap=0` or `gap <= 4 KiB` first;
   - span bytes capped at `16 MiB`;
   - group size at least `2`;
   - report expected extents saved and affected read GiB.
2. If the safe subset bound is small, stop.
3. If the safe subset is material, write a detailed source-level plan before coding:
   - data structures;
   - io_uring SQE mapping for grouped spans;
   - H2D slice enqueue order;
   - event lifetime;
   - fallback path;
   - counters and acceptance gates.

Acceptance before runtime A/B:

- The offline safe-subset bound must exceed N32 noise:
  - target at least `1s` expected wait reduction;
  - or `>10%` read-plan count reduction in `jobs<=8` batches.
- The design must explicitly avoid the known Phase 7LT/7MM failures:
  - no blocking pread;
  - no tiny coalesce slot pool;
  - no waiting for all slices before compute can proceed;
  - no cross-role broad merge that damages up/gate/down overlap.

### Phase 4O result: safe adjacent-span bound passes the design gate

Timestamp: 2026-07-11 03:35 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4o-safe-adjacent-bound`

Inputs:

- Trace root: `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343`
- Rows: `195895`
- `max_jobs=8`
- `max_span=16777216` bytes
- gaps tested: `0` and `4096`
- dev traces only; no held-out prompt was used.

Output files:

- `safe-adjacent-bound.json`
- `safe-adjacent-bound.md`

Result:

| layout | max gap | read jobs | safe extents | extents saved | saved % | groups | grouped jobs | read GiB | grouped payload GiB | grouped span GiB | extra gap GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_actual | 0 | 195895 | 193277 | 2618 | 1.34% | 2593 | 5211 | 1046.189 | 27.981 | 27.981 | 0.000000 |
| expert_id | 0 | 195895 | 192462 | 3433 | 1.75% | 3412 | 6845 | 1046.189 | 36.724 | 36.724 | 0.000000 |
| frequency | 0 | 195895 | 187256 | 8639 | 4.41% | 8062 | 16701 | 1046.189 | 90.234 | 90.234 | 0.000000 |
| greedy_pair | 0 | 195895 | 145443 | 50452 | 25.75% | 45655 | 96107 | 1046.189 | 510.773 | 510.773 | 0.000000 |
| current_actual | 4096 | 195895 | 193277 | 2618 | 1.34% | 2593 | 5211 | 1046.189 | 27.981 | 27.981 | 0.000000 |
| expert_id | 4096 | 195895 | 192462 | 3433 | 1.75% | 3412 | 6845 | 1046.189 | 36.724 | 36.724 | 0.000000 |
| frequency | 4096 | 195895 | 187256 | 8639 | 4.41% | 8062 | 16701 | 1046.189 | 90.234 | 90.234 | 0.000000 |
| greedy_pair | 4096 | 195895 | 145443 | 50452 | 25.75% | 45655 | 96107 | 1046.189 | 510.773 | 510.773 | 0.000000 |

Role/op detail for `greedy_pair`, `max_gap=0`:

| op:role | jobs | extents | groups | grouped jobs | grouped payload GiB | grouped span GiB |
|---|---:|---:|---:|---:|---:|---:|
| runtime_load:up | 63414 | 45728 | 14405 | 32091 | 146.589 | 146.589 |
| runtime_load:gate | 62205 | 46141 | 14548 | 30612 | 150.622 | 150.622 |
| runtime_load:down | 37964 | 28767 | 9197 | 18394 | 125.364 | 125.364 |
| current_down_overlap:down | 32312 | 24807 | 7505 | 15010 | 88.198 | 88.198 |

Interpretation:

- Current physical layout alone is not worth a coalescer:
  - `current_actual` can save only `2618 / 195895 = 1.34%` read plans.
  - This is below the N32 noise threshold and would likely recreate coalescer complexity without measurable speed.
- A dev-trained per-tensor `greedy_pair` layout is materially different:
  - it can reduce small-batch read plans by `50452 / 195895 = 25.75%`;
  - grouped payload is `510.773 GiB`;
  - the bound covers up, gate, runtime down, and current-down-overlap down.
- `gap=4096` does not improve over `gap=0` in this trace set, so the first runtime design should be exact-adjacent only.

Decision:

- Proceed to a source-level Phase 4P plan.
- Do not implement a coalescer against the current pack layout only.
- The next runtime A/B must pair two default-off components:
  - a dev-trained `greedy_pair` expert pack/repack built only from dev traces;
  - an exact-adjacent async coalescer in the io_uring pinned-staging path.
- Held-out prompts must remain unused until after N32/N96 dev evidence shows an actual speedup.

### Phase 4P plan: greedy-pair pack plus exact-adjacent async coalescer

Goal:

> Convert the Phase 4O read-plan reduction bound into a default-off runtime experiment without reintroducing the previous blocking coalescer failure modes.

Required components:

1. Pack/repack artifact:
   - Extend the existing pack tooling to support a dev-trained per-tensor `greedy_pair` order.
   - Candidate scripts:
     - `scripts/kimi-reorder-expert-pack.py` for full-pack reorder;
     - `scripts/kimi-build-trace-overlay-pack.py` for selected overlay creation.
   - The trace input must be dev-only:
     - `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343/*/io-read-trace.csv`
   - The output artifact must record:
     - source packs;
     - trace roots;
     - layout algorithm;
     - included entries/tensors/bytes;
     - sha256;
     - exact command.
   - The artifact must not be trained on held-out prompts or on a single prompt such as France.

2. Runtime coalescer:
   - Implementation target: `ggml/src/ggml-cuda/moe_stream_batch.cu`, inside `expert_pack_iouring_copy_jobs`.
   - Existing flow:
     - build one `iouring_read_plan` per expert;
     - optionally sort by source/offset;
     - submit one SQE per plan;
     - on each CQE, enqueue one `cudaMemcpyAsync` from the pinned slot to that expert's `dst`.
   - New default-off envs:
     - `GGML_MOE_IO_ADJACENT_COALESCE=1`
     - `GGML_MOE_IO_ADJACENT_MAX_SPAN_MIB=16`
     - `GGML_MOE_IO_ADJACENT_MAX_GAP=0`
     - `GGML_MOE_IO_ADJACENT_MIN_GROUP=2`
   - Coalesce only when all are true:
     - same source index;
     - same tensor;
     - adjacent offset with `gap=0` for the first implementation;
     - grouped span `<=16 MiB`;
     - group size `>=2`;
     - direct/io_uring source only, no page-cache source.

3. Data structure sketch:
   - Keep the existing `iouring_read_plan` as the per-expert logical plan.
   - Add a physical `iouring_group_plan`:
     - `source_idx`;
     - `offset`;
     - `read_sz`;
     - `std::vector<slice>` where each slice has `plan_idx`, `payload_offset_in_group`, and `expert_bytes`.
   - `pinned_stage_ensure` must size slots by `max(group.read_sz)`, not by a single expert entry.
   - `io_uring_prep_read` reads the physical group span into one pinned slot.
   - CQE handling iterates group slices in offset order:
     - `slot_src = slot.host + payload_offset_in_group`;
     - enqueue `cudaMemcpyAsync(job.dst, slot_src, expert_bytes, cudaMemcpyHostToDevice, st)` for each slice;
     - record one `slot.done` event after the last slice so the pinned slot cannot be reused before all H2D copies finish.
   - The function still returns only after all group CQEs have been handled and all H2D copies have been enqueued on the same stream.

4. Counters and traces:
   - Preserve existing expert payload counters where possible:
     - `iouring_reads` and `iouring_bytes` should remain comparable to old payload-level metrics.
   - Add physical counters:
     - `adjacent_coalesce_groups`;
     - `adjacent_coalesce_slices`;
     - `adjacent_coalesce_extents_saved`;
     - `adjacent_coalesce_physical_bytes`;
     - `adjacent_coalesce_payload_bytes`;
     - `adjacent_coalesce_max_group_jobs`;
     - `adjacent_coalesce_max_span_bytes`;
     - fallback/reject counters by reason: source mismatch, tensor mismatch, gap, span cap, group size.
   - Keep `GGML_MOE_IO_READ_TRACE_OUT` compatible by writing one row per logical expert read.
   - Add a separate optional group trace if needed:
     - `GGML_MOE_IO_COALESCE_TRACE_OUT`.

5. Validation ladder:
   - Build smoke with coalescer default-off.
   - N32 dev A/B, current pack + coalescer enabled:
     - expected: near-zero or tiny gain, no regression.
     - purpose: validate safety and counters.
   - N32 dev A/B, greedy-pair pack + coalescer enabled:
     - expected: read-plan/cqe reduction should appear in counters.
     - must improve token rate or clearly reduce `iouring_wait_us` before N96.
   - N96 dev paired run only if N32 passes.
   - N96 held-out only after N96 dev passes.

6. Rejection criteria:
   - Any semantic-quality failure.
   - Host RAM peak `>=15900000000` bytes.
   - TTFT ratio `>1.20`.
   - `slot_wait_ms` increases enough to erase `iouring_wait` savings.
   - token rate improves on dev but regresses on held-out.
   - counters show fewer CQEs but no reduction in exposed wait.

Expected upper bound:

- Phase 4O shows `25.75%` small-batch read-plan reduction for `greedy_pair`, covering `510.773 GiB` of payload.
- This is not a bytes-reduction optimization; H2D payload remains approximately unchanged.
- The only expected speedup is lower per-read io_uring/CQE/wait overhead and fewer tiny exposed reads.
- Because current runtime wait is not purely per-SQE overhead, the realistic gain is likely much smaller than `25.75%`; require measurement before any SOTA claim.

### Phase 4P step A result: greedy-pair trace overlay pack tool

Timestamp: 2026-07-11 03:55 CST.

Implemented:

- Extended `scripts/kimi-build-trace-overlay-pack.py`.
- `--trace` can now be repeated so the layout can be trained on the full dev prompt set instead of one prompt.
- Added `--mode`:
  - `first-use` keeps the old behavior and remains the default;
  - `frequency` sorts selected experts by per-tensor trace frequency;
  - `greedy-pair` starts from the most frequent expert and then greedily places the expert with highest same-batch co-occurrence with the previous expert.
- Added `--max-jobs`, so Phase 4O can be reproduced with `--max-jobs 8`.

Validation:

- Syntax:

```bash
python3 -m py_compile scripts/kimi-build-trace-overlay-pack.py
```

- Help output confirms the new flags:

```bash
python3 scripts/kimi-build-trace-overlay-pack.py --help
```

- Synthetic GGMLMOEPACKv1 test:
  - input experts: `[0, 1, 2, 3]`;
  - trace makes expert `0` most frequent and expert `2` the strongest same-batch neighbor;
  - `--mode greedy-pair --max-jobs 8` output order: `[0, 2, 1, 3]`;
  - assertion passed.

Next step:

- Do not build the large dev overlay yet until runtime coalescer code exists, because a greedy-pair pack without adjacent-span merging should not materially reduce read plans.
- Implement the default-off exact-adjacent coalescer in `expert_pack_iouring_copy_jobs`, then run:
  1. current pack + coalescer enabled safety A/B;
  2. greedy-pair dev overlay + coalescer enabled performance A/B.

### Phase 4P step B result: default-off exact-adjacent io_uring coalescer

Timestamp: 2026-07-11 04:20 CST.

Implemented:

- File: `ggml/src/ggml-cuda/moe_stream_batch.cu`
- Function: `expert_pack_iouring_copy_jobs`
- Default behavior remains off unless `GGML_MOE_IO_ADJACENT_COALESCE=1`.

Runtime envs:

- `GGML_MOE_IO_ADJACENT_COALESCE=1`
- `GGML_MOE_IO_ADJACENT_MAX_SPAN_MIB=16`
- `GGML_MOE_IO_ADJACENT_MAX_GAP=0`
- `GGML_MOE_IO_ADJACENT_MIN_GROUP=2`

Implementation details:

- The code still builds one logical `iouring_read_plan` per expert.
- It then builds physical `iouring_group_plan` objects.
- With coalescing disabled, each group contains exactly one logical plan.
- With coalescing enabled, plans can merge only when:
  - same source;
  - same tensor;
  - adjacent offset under the configured gap cap;
  - grouped span under the configured span cap;
  - group size reaches `GGML_MOE_IO_ADJACENT_MIN_GROUP`.
- SQE submission now reads one physical group into one pinned staging slot.
- CQE handling iterates group slices and enqueues one `cudaMemcpyAsync` per logical expert into its original `dst`.
- The pinned slot records one `done` event after the last slice copy, preventing slot reuse before all H2D copies from that group are enqueued and ordered on the stream.
- `GGML_MOE_IO_READ_TRACE_OUT` remains logical/per-expert for compatibility with existing scripts.

Counters added:

- `adjacent_coalesce_groups`
- `adjacent_coalesce_slices`
- `adjacent_coalesce_extents_saved`
- `adjacent_coalesce_physical_bytes`
- `adjacent_coalesce_payload_bytes`
- `adjacent_coalesce_max_group_jobs`
- `adjacent_coalesce_max_span_bytes`
- reject counters for source/tensor/gap/span.

Validation:

```bash
cmake --build build-cuda-batch -j 8
build-cuda-batch/bin/test-kimi-deepseek2-guards
```

Result:

- Build passed.
- `test-kimi-deepseek2-guards` passed.
- This is not a SOTA claim. No token-rate result is accepted until the A/B ladder below passes.

Next validation ladder:

1. N32 dev current-pack safety A/B:
   - baseline: current SOTA env, coalescer disabled;
   - candidate: same env with `GGML_MOE_IO_ADJACENT_COALESCE=1`;
   - expected: tiny effect because current physical layout only has a `1.34%` bound;
   - reject immediately if quality, RAM, TTFT, or token rate regresses.
2. If safety passes, build a dev-only greedy-pair overlay using `scripts/kimi-build-trace-overlay-pack.py --mode greedy-pair --max-jobs 8`.
3. N32 dev greedy-pair overlay + coalescer A/B:
   - require visible `adjacent_coalesce_extents_saved`;
   - require lower exposed `iouring_wait_us` or better token rate before any N96 run.
4. N96 dev only after N32 passes.
5. N96 held-out only after N96 dev passes.

### Phase 4P step C result: current-pack coalescer safety A/B rejected

Timestamp: 2026-07-11 04:55 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4p-current-pack-coalesce-safety-n32-dev3`

Experiment:

- N32 cold start.
- First 3 dev prompts only:
  - `dev_france_regression`
  - `dev_japan_factual`
  - `dev_photosynthesis_factual`
- Control:
  - current SOTA env;
  - `GGML_MOE_IO_ADJACENT_COALESCE` unset.
- Candidate:
  - same env;
  - `GGML_MOE_IO_ADJACENT_COALESCE=1`;
  - `GGML_MOE_IO_ADJACENT_MAX_SPAN_MIB=16`;
  - `GGML_MOE_IO_ADJACENT_MAX_GAP=0`;
  - `GGML_MOE_IO_ADJACENT_MIN_GROUP=2`.

Aggregate result:

| run | quality | min tok/s | median tok/s | mean tok/s | decode sum ms | iouring wait sum ms | max RAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 3/3 | 1.810 | 1.870 | 1.853 | 50176.6 | 56878.5 | 11.85 |
| candidate | 3/3 | 1.670 | 1.730 | 1.730 | 53815.8 | 59231.5 | 12.04 |

Paired result:

| prompt | tok/s delta | TTFT ratio | decode delta ms | iouring wait delta ms | RAM delta MiB |
|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | -0.210 | 0.964 | +2151.0 | +976.9 | +194.0 |
| `dev_japan_factual` | -0.080 | 1.024 | +681.0 | +828.2 | +157.4 |
| `dev_photosynthesis_factual` | -0.080 | 0.996 | +807.2 | +547.9 | +194.9 |

Candidate coalescer counters:

| prompt | groups | slices | extents saved | physical/payload bytes | max group jobs | max span bytes |
|---|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | 4105 | 8713 | 4608 | 49369169920 / 49369169920 | 3 | 16515072 |
| `dev_japan_factual` | 3760 | 7946 | 4186 | 44900122624 / 44900122624 | 3 | 16515072 |
| `dev_photosynthesis_factual` | 2423 | 5100 | 2677 | 28813524992 / 28813524992 | 3 | 16515072 |

Important counters:

- France:
  - control CQEs: `39090`
  - candidate CQEs: `34482`
  - candidate saved extents: `4608`
  - token rate still fell from `1.88` to `1.67`.
- Japan:
  - control CQEs: `39075`
  - candidate CQEs: `34889`
  - token rate fell from `1.87` to `1.79`.
- Photosynthesis:
  - control CQEs: `38275`
  - candidate CQEs: `35598`
  - token rate fell from `1.81` to `1.73`.

Interpretation:

- The implementation did reduce physical CQEs/read extents.
- It did **not** reduce exposed wait; `iouring_wait` increased on all three prompts.
- The likely mechanism is critical-path latency:
  - old path can receive and H2D each expert as soon as its smaller read completes;
  - coalesced path waits for the entire 2-3 expert span before the first slice can be H2D-enqueued;
  - the staged slot is also held until all slice copies are enqueued and ordered;
  - fewer CQEs are not enough to offset the longer first-slice latency on runtime-load critical paths.
- This matches the previous historical warning that broad/blocking coalescing can move the bottleneck from SQE count to slot/wait latency.

Decision:

- Reject `GGML_MOE_IO_ADJACENT_COALESCE=1` with the current implementation for SOTA use.
- Keep the code default-off only as diagnostic infrastructure.
- Do not build or test the large greedy-pair overlay with this coalescer implementation; greedy-pair would create more groups and likely amplify the same first-slice latency problem.
- No N96 run and no held-out run.

Next plan after rejection:

1. Do not pursue synchronous adjacent-span coalescing on `runtime_load` up/gate/down.
2. If coalescing is revisited, restrict it to non-critical prefetch paths first:
   - `current_down_overlap` only;
   - or explicit prefetch where compute does not wait for the first slice.
3. Add per-op coalescer counters before any retry, because this result needs separation of:
   - `runtime_load:up`;
   - `runtime_load:gate`;
   - `runtime_load:down`;
   - `current_down_overlap:down`.
4. Return to the main bottleneck ranking:
   - exposed `io_uring_wait` comes from critical-path runtime misses, not just physical read count;
   - next candidates should reduce wait without delaying the earliest needed expert.

### Phase 4Q plan: op-filtered adjacent coalescing for overlap-only paths

Reason:

- Phase 4P proved that reducing physical CQEs is not enough when the first needed expert is delayed.
- `runtime_load:*` is on the decode critical path; coalescing there can make the earliest slice wait for the whole group.
- `current_down_overlap:down` is a better place to test the idea because it is a prefetch/overlap path, so a later first slice may be hidden by current up/gate compute.

Hypothesis:

> Exact-adjacent coalescing may be safe only when restricted to non-critical prefetch/overlap operations, especially `current_down_overlap`, because the grouped read can reduce CQE overhead without delaying an immediately required expert.

Implementation plan:

1. Add an op filter:
   - env: `GGML_MOE_IO_ADJACENT_OP_FILTER`;
   - empty/default means no filter beyond `GGML_MOE_IO_ADJACENT_COALESCE=1`;
   - for Phase 4Q use `GGML_MOE_IO_ADJACENT_OP_FILTER=current_down_overlap`.
2. Add per-op coalescer counters:
   - `runtime_load`
   - `current_down_overlap`
   - `other`
   - record groups, slices, extents saved, physical bytes, payload bytes, max group, max span.
3. Keep all behavior default-off.
4. Do not touch greedy-pair overlay in this phase.

Validation:

1. Build:

```bash
cmake --build build-cuda-batch -j 8
build-cuda-batch/bin/test-kimi-deepseek2-guards
```

2. N32 dev3 current-pack A/B:

Control:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4q-overlap-only-coalesce-n32-dev3/control \
  --mode dev --n 32 --max-prompts 3 --keep-going --runtime-max-sec 900
```

Candidate:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4q-overlap-only-coalesce-n32-dev3/candidate \
  --mode dev --n 32 --max-prompts 3 --keep-going --runtime-max-sec 900 \
  --extra-runtime-env $'GGML_MOE_IO_ADJACENT_COALESCE=1\nGGML_MOE_IO_ADJACENT_OP_FILTER=current_down_overlap\nGGML_MOE_IO_ADJACENT_MAX_SPAN_MIB=16\nGGML_MOE_IO_ADJACENT_MAX_GAP=0\nGGML_MOE_IO_ADJACENT_MIN_GROUP=2'
```

Acceptance:

- quality `3/3`;
- RAM peak `<15900000000`;
- TTFT ratio `<=1.20`;
- no token-rate regression on min/median/mean;
- `current_down_overlap` counters show actual extents saved;
- `runtime_load` counters must stay zero when the op filter is enabled.

Rejection:

- any quality failure;
- token-rate regression;
- `iouring_wait` increase without decode improvement;
- nonzero `runtime_load` coalescing under the filter;
- RAM/TTFT gate failure.

### Phase 4Q result: overlap-only adjacent coalescing rejected

Timestamp: 2026-07-11 05:35 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4q-overlap-only-coalesce-n32-dev3`

Experiment:

- N32 cold start.
- First 3 dev prompts only:
  - `dev_france_regression`
  - `dev_japan_factual`
  - `dev_photosynthesis_factual`
- Control:
  - current SOTA env;
  - adjacent coalescer disabled.
- Candidate:
  - `GGML_MOE_IO_ADJACENT_COALESCE=1`
  - `GGML_MOE_IO_ADJACENT_OP_FILTER=current_down_overlap`
  - `GGML_MOE_IO_ADJACENT_MAX_SPAN_MIB=16`
  - `GGML_MOE_IO_ADJACENT_MAX_GAP=0`
  - `GGML_MOE_IO_ADJACENT_MIN_GROUP=2`

Build validation before A/B:

- `cmake --build build-cuda-batch -j 8`: passed.
- `build-cuda-batch/bin/test-kimi-deepseek2-guards`: passed.

Aggregate result:

| run | quality | min tok/s | median tok/s | mean tok/s | decode sum ms | iouring wait sum ms | max RAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 3/3 | 1.850 | 1.960 | 1.923 | 48420.8 | 54511.8 | 11.85 |
| candidate | 3/3 | 1.750 | 1.830 | 1.807 | 51531.1 | 57907.7 | 11.90 |

Paired result:

| prompt | tok/s delta | TTFT ratio | decode delta ms | iouring wait delta ms | RAM delta MiB | runtime extents | overlap extents |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | -0.210 | 1.006 | +1908.9 | +1974.5 | +51.3 | 0 | 103 |
| `dev_japan_factual` | -0.120 | 1.125 | +1016.9 | +1111.3 | +52.9 | 0 | 62 |
| `dev_photosynthesis_factual` | -0.020 | 1.035 | +184.5 | +310.2 | +51.9 | 0 | 47 |

Candidate op-filter evidence:

- `runtime_load` coalesced extents: `0` on all three prompts.
- `current_down_overlap` coalesced extents:
  - France: `103`
  - Japan: `62`
  - Photosynthesis: `47`
- The op filter worked correctly, so the regression is not caused by accidentally coalescing critical `runtime_load`.

Interpretation:

- Even overlap-only adjacent coalescing is too small and still harmful on the current pack layout.
- It saves only tens to low hundreds of extents per N32 prompt:
  - much smaller than the full current-pack coalescer;
  - far below the scale needed to reduce exposed wait.
- It still increases `iouring_wait`, likely because the overlap worker holds larger staging slots longer and changes refill/slot timing.
- This confirms that adjacent-span coalescing is not the next high-value path unless the storage layout is changed and the read path becomes truly non-blocking for first needed slices.

Decision:

- Reject overlap-only adjacent coalescing for SOTA.
- Keep the op filter and per-op counters default-off as diagnostic infrastructure only.
- Do not run N96 or held-out.
- Do not build a greedy-pair overlay for this coalescer path.

Next plan after Phase 4Q:

1. Stop adjacent coalescing work for now.
2. Return to bottleneck classes that can reduce critical-path wait without delaying first expert availability:
   - VRAM cache budget rebalancing by exposed wait;
   - route-predicted prefetch that begins before the current layer reaches the miss;
   - RAM/VRAM explicit tiering only when it fully avoids SSD wait for critical up/gate misses;
   - lower-byte expert representation for second-tier experts.
3. The next design step should build a wait-weighted layer/role report from current N32/N96 traces:
   - group by `runtime_load:up`, `runtime_load:gate`, `runtime_load:down`, `current_down_overlap:down`;
   - rank by exposed wait, miss count, bytes, and whether wait is on critical path;
   - select one candidate that removes wait without increasing first-slice latency.

### Phase 4R result: wait-weighted layer/role bottleneck report

Timestamp: 2026-07-11 05:50 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4r-wait-weighted-layer-role-report`

Inputs:

- Trace root: `/root/lfz/runs/vendor-kimi-token-rate/20260710-kimi-phase4e-full-dev7-trace-n32-150343`
- Prompts: `7`
- Source files:
  - per-prompt `io-wait-trace.csv`
  - per-prompt `io-read-trace.csv`
  - per-prompt `metrics.json`

Outputs:

- `wait-weighted-layer-role-report.json`
- `wait-weighted-layer-role-report.md`

Aggregate:

- total `iouring_wait_ms`: `135530.035`
- token rate min/median/mean: `1.590/1.780/1.776`

Role summary:

| op | role | wait ms | wait % | read jobs | read GiB | tensors |
|---|---|---:|---:|---:|---:|---:|
| `runtime_load` | `gate` | 45628.1 | 33.7% | 94072 | 465.5 | 59 |
| `runtime_load` | `up` | 44842.6 | 33.1% | 95818 | 437.8 | 60 |
| `runtime_load` | `down` | 28528.9 | 21.0% | 70368 | 464.2 | 60 |
| `current_down_overlap` | `down` | 16530.4 | 12.2% | 32312 | 189.9 | 29 |

Interpretation:

- Critical `runtime_load` up+gate is `66.8%` of all measured `iouring_wait`.
- Runtime down is `21.0%`.
- Overlap down is only `12.2%`; optimizing it cannot move token rate enough by itself.
- Adjacent coalescing attacked read extent count, but not the dominant first-expert critical-path wait.
- The next low-risk candidate should rebalance existing VRAM cache toward up/gate, because:
  - up/gate wait is both critical and larger;
  - wait per GiB is higher for up/gate than runtime down;
  - this does not delay first-slice H2D and does not add new read scheduling.

### Phase 4S plan: VRAM cache split rebalancing toward up/gate

Hypothesis:

> Increasing `GGML_MOE_VRAM_CACHE_UPGATE_PCT` from the current `62` to `70` may reduce critical up/gate runtime misses enough to improve token rate, while the down side may be partly protected by existing `current_down_overlap`.

Why this is the next candidate:

- It targets the dominant wait class from Phase 4R: `runtime_load:gate` + `runtime_load:up`.
- It avoids the first-slice latency problem that killed adjacent coalescing.
- It is an env-only A/B, so rollback is immediate.
- It does not use held-out prompts and does not build prompt-specific artifacts.

Theoretical bound:

- Total measured wait in Phase 4R:
  - up+gate critical wait: `90470.7 ms`
  - runtime down wait: `28528.9 ms`
  - overlap down wait: `16530.4 ms`
- The best possible split-only gain is bounded by reducing some fraction of the `90470.7 ms` up/gate wait.
- The risk is increased down wait if down cache hit rate falls.
- A useful N32 dev3 result must show lower total decode time or lower `iouring_wait`, not just higher up/gate hit rate.

Experiment:

Control:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate70-vram-split-n32-dev3/control \
  --mode dev --n 32 --max-prompts 3 --keep-going --runtime-max-sec 900
```

Candidate:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate70-vram-split-n32-dev3/upgate70 \
  --mode dev --n 32 --max-prompts 3 --keep-going --runtime-max-sec 900 \
  --upgate-pct 70
```

Acceptance:

- quality `3/3`;
- RAM peak `<15900000000`;
- TTFT ratio `<=1.20`;
- min/median/mean token rate must not regress;
- decode sum and `iouring_wait` should improve, or at minimum one improves without hurting the other;
- if N32 passes, run N96 dev before held-out.

Rejection:

- any quality failure;
- TTFT ratio `>1.20`;
- any clear token-rate regression;
- down wait increase cancels up/gate gain;
- no movement in exposed wait.

### Phase 4S result A: `UPGATE_PCT=70` rejected

Timestamp: 2026-07-11 06:10 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate70-vram-split-n32-dev3`

Aggregate result:

| run | quality | min tok/s | median tok/s | mean tok/s | decode sum ms | iouring wait sum ms | max RAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control `UPGATE_PCT=62` | 3/3 | 1.750 | 1.870 | 1.843 | 50476.9 | 57858.8 | 11.85 |
| candidate `UPGATE_PCT=70` | 3/3 | 1.800 | 1.800 | 1.810 | 51344.9 | 57415.1 | 11.85 |

Paired result:

| prompt | tok/s delta | TTFT ratio | decode delta ms | wait delta ms | read delta | bytes delta GiB |
|---|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | -0.070 | 0.962 | +627.6 | +418.9 | -68 | -0.2 |
| `dev_japan_factual` | -0.110 | 1.014 | +1004.2 | -83.6 | -351 | -1.5 |
| `dev_photosynthesis_factual` | +0.080 | 1.050 | -763.8 | -779.1 | -624 | -2.9 |

Interpretation:

- `UPGATE_PCT=70` reduces reads/bytes and slightly reduces aggregate `iouring_wait`, but decode sum and mean token rate regress.
- The improvement on `dev_photosynthesis_factual` suggests the direction can help some prompts.
- The regressions on France/Japan suggest `70` is too aggressive or shifts pressure into down/cache scheduling in a prompt-dependent way.

Decision:

- Reject `UPGATE_PCT=70`; no N96 and no held-out.
- Run exactly one smaller step, `UPGATE_PCT=66`, before abandoning global split rebalancing.
- If `66` does not improve min/median/mean without wait/decode regression, stop global split tuning and move to layer/role-specific admission.

### Phase 4S plan B: `UPGATE_PCT=66` small-step screen

Hypothesis:

> A smaller shift from `62` to `66` may capture part of the up/gate wait reduction without the down-side/cache scheduling regression seen at `70`.

Experiment:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate70-vram-split-n32-dev3/upgate66 \
  --mode dev --n 32 --max-prompts 3 --keep-going --runtime-max-sec 900 \
  --upgate-pct 66
```

Acceptance:

- quality `3/3`;
- RAM/TTFT gates pass;
- min/median/mean token rate improve or at least do not regress;
- decode sum and `iouring_wait` do not regress.

### Phase 4S result B: `UPGATE_PCT=66` passes N32 screen, not SOTA yet

Timestamp: 2026-07-11 06:40 CST.

Run root:

- `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate70-vram-split-n32-dev3`

Aggregate result:

| run | quality | min tok/s | median tok/s | mean tok/s | decode sum ms | iouring wait sum ms | reads | read GiB | max RAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control `UPGATE_PCT=62` | 3/3 | 1.750 | 1.870 | 1.843 | 50476.9 | 57858.8 | 116440 | 620.2 | 11.85 |
| candidate `UPGATE_PCT=66` | 3/3 | 1.830 | 1.900 | 1.897 | 49151.4 | 54834.2 | 115859 | 617.8 | 11.85 |

Paired result:

| prompt | tok/s delta | TTFT ratio | decode delta ms | wait delta ms | read delta | bytes delta GiB |
|---|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | +0.090 | 0.951 | -710.9 | -1364.0 | +32 | +0.2 |
| `dev_japan_factual` | -0.010 | 0.940 | +69.2 | -684.5 | -171 | -0.6 |
| `dev_photosynthesis_factual` | +0.080 | 0.993 | -683.8 | -976.1 | -442 | -2.1 |

Interpretation:

- The smaller split is materially better than `70`.
- It reduces aggregate `iouring_wait` by `3024.6 ms` and decode sum by `1325.5 ms` on N32 dev3.
- It passes quality and RAM gates, and TTFT improves on all three prompts.
- The Japan prompt still has a tiny token-rate/decode regression, so this is only a screen pass, not an accepted SOTA.

Decision:

- Promote `UPGATE_PCT=66` to N96 dev A/B.
- Do not run held-out until N96 dev passes.
- Do not claim SOTA yet.
- If N96 dev confirms the gain, commit/push the SOTA reproduction with `--upgate-pct 66` in the exact command body.
- If N96 dev regresses or is noisy, reject global split tuning and move to layer/role-specific cache admission from the Phase 4R wait-weighted report.

Next command:

```bash
python3 .Agent/run-tools/kimi_general_prompt_sweep.py \
  --prompt-file .Agent/evals/kimi-general-dev-prompts.jsonl \
  --out-root /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-dev \
  --mode dev --n 96 --keep-going --runtime-max-sec 1200 \
  --upgate-pct 66
```

### Phase 4S result C: `UPGATE_PCT=66` rejected after N96 held-out

Timestamp: 2026-07-11 08:00 CST.

Run roots:

- N96 dev:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-dev/control`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-dev/control-tail`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-dev/upgate66`
- N96 held-out:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-heldout/control`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-heldout/upgate66`
- Targeted repeat/debug:
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-python-timeout-repro`
  - `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-linear-repeat-n96`

Important execution note:

- The first N96 dev control sweep hit a transient timeout on `dev_python_reverse`.
- Systemd unit: `run-u11147.service`
- Result: `timeout`, `RuntimeMaxSec=1200`, CPU time `20.489s`, disk read `14.2G`, stdout empty.
- The same prompt passed targeted reruns:
  - N32 control: `1.78 tok/s`, TTFT `8631.94 ms`, decode `17391.63 ms / 31`, quality pass.
  - N96 control with `RuntimeMaxSec=300`: `1.78 tok/s`, TTFT `8080.16 ms`, decode `53297.96 ms / 95`, quality pass.
- Interpretation: the timeout is a real cold-start long-tail/IO-wait stability risk, but it was not deterministic. For the N96 dev aggregate, completed control rows came from the original control root for prompts 1-4 and from `control-tail` for prompts 5-7.

N96 dev aggregate:

| run | quality | min tok/s | median tok/s | mean tok/s | decode sum ms | iouring wait sum ms | reads | read GiB | max RAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control `UPGATE_PCT=62` | 7/7 | 1.670 | 1.780 | 1.784 | 270667.6 | 292393.6 | 528672 | 2814.4 | 11.86 |
| candidate `UPGATE_PCT=66` | 7/7 | 1.600 | 1.840 | 1.827 | 262926.8 | 281028.9 | 524339 | 2795.4 | 11.86 |

N96 dev paired result:

| prompt | tok/s delta | TTFT ratio | decode delta ms | wait delta ms | read delta | GiB delta | quality |
|---|---:|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | +0.140 | 0.981 | -3231.5 | -3867.2 | -416 | -1.6 | pass->pass |
| `dev_japan_factual` | +0.000 | 0.926 | -61.7 | -2039.6 | -135 | -0.2 | pass->pass |
| `dev_photosynthesis_factual` | +0.060 | 1.049 | -1525.0 | -2622.5 | -1697 | -7.9 | pass->pass |
| `dev_linear_equation` | -0.070 | 0.975 | +931.0 | +1461.7 | -1194 | -5.6 | pass->pass |
| `dev_python_reverse` | +0.080 | 1.055 | -2459.9 | -2643.0 | -804 | -3.7 | pass->pass |
| `dev_zh_france` | +0.060 | 0.975 | -793.7 | -1148.3 | +106 | +0.9 | pass->pass |
| `dev_mixed_summary` | +0.030 | 1.004 | -600.0 | -505.7 | -193 | -0.9 | pass->pass |

Linear-repeat check:

- Run root: `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-linear-repeat-n96`
- Control: `1.61 tok/s`, TTFT `10056.42 ms`, decode `21059.05 ms / 34`, quality pass.
- Candidate `UPGATE_PCT=66`: `1.68 tok/s`, TTFT `9171.51 ms`, decode `20290.26 ms / 34`, quality pass.
- Interpretation: the single full-dev `dev_linear_equation` regression is likely short-output noise, so `UPGATE_PCT=66` was allowed to proceed to held-out validation.

N96 held-out aggregate:

| run | quality | min tok/s | median tok/s | mean tok/s | decode sum ms | iouring wait sum ms | reads | read GiB | max TTFT ms | max RAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control `UPGATE_PCT=62` | 6/6 | 1.610 | 1.805 | 1.782 | 295435.4 | 312667.1 | 551077 | 2935.1 | 211385.6 | 14.81 |
| candidate `UPGATE_PCT=66` | 6/6 | 1.630 | 1.810 | 1.787 | 294275.8 | 315182.8 | 544140 | 2903.9 | 212810.6 | 14.81 |

N96 held-out paired result:

| prompt | tok/s delta | TTFT ratio | decode delta ms | wait delta ms | RAM GiB control/candidate | quality |
|---|---:|---:|---:|---:|---:|---|
| `test_english_factual_01` | -0.040 | 1.029 | +1103.6 | +2589.0 | 11.72/11.71 | pass->pass |
| `test_english_factual_02` | +0.010 | 0.892 | -357.9 | -184.4 | 11.86/11.86 | pass->pass |
| `test_reasoning_math_01` | -0.030 | 1.007 | +691.8 | +572.5 | 14.81/14.81 | pass->pass |
| `test_coding_01` | +0.020 | 0.932 | -676.3 | -524.9 | 11.86/11.86 | pass->pass |
| `test_chinese_01` | +0.070 | 1.117 | -1838.8 | -1932.8 | 11.86/11.86 | pass->pass |
| `test_mixed_instruction_01` | +0.000 | 1.064 | -81.9 | +1996.5 | 11.86/11.86 | pass->pass |

Decision:

- Reject `UPGATE_PCT=66` as SOTA.
- Reason:
  - held-out mean improvement is only `+0.005 tok/s`, which is noise-level;
  - held-out aggregate `iouring_wait` regresses by `+2515.7 ms`;
  - two held-out prompts regress in token rate;
  - one held-out prompt has very high absolute TTFT around `212s` in both control and candidate, showing a baseline cold-start stability issue that global VRAM split does not solve.
- No code change is needed because this was env-only.
- Do not promote `--upgate-pct 66` to default.
- Stop global `UPGATE_PCT` tuning for now.

Next plan after Phase 4S:

1. Profile the TTFT long-tail prompt `test_reasoning_math_01`.
   - Goal: determine why cold-start TTFT is around `211-213s` while decode is only around `35-36s`.
   - Break down prompt/decode initialization, dense mmap drop, expert pack load, GGUF alias setup, prompt fallback, RAM/file-cache growth, and first-token expert IO.
   - This is required before claiming stable random-prompt behavior under 16 GB host RAM.

2. Move from global split to layer/role-specific admission.
   - Use Phase 4R wait-weighted rows and N96 dev/held-out per-prompt regressions.
   - Protect only the layer/role entries that reduce exposed critical wait across dev prompts without increasing held-out wait.
   - Prefer default-off profile inputs over hardcoded prompt-specific packs.

3. Revisit RAM/VRAM tiering with explicit rejection criteria.
   - Do not add scattered RAM experts unless they reduce exposed wait on held-out prompts.
   - If using RAM, prefer batchable whole layer/role slabs or compact adjacent layouts.
   - Any candidate must improve held-out mean/median without increasing aggregate `iouring_wait` or TTFT ratio.

### Phase 4T finding: held-out TTFT long tail is prompt-eval reclaim/refault

Timestamp: 2026-07-11 08:25 CST.

Source runs:

- Control: `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-heldout/control/test_reasoning_math_01`
- Candidate: `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase4s-upgate66-vram-split-n96-heldout/upgate66/test_reasoning_math_01`

Observed control metrics:

- prompt: `A train leaves at 3 PM and arrives 2 hours and 15 minutes later. What time does it arrive?`
- prompt tokens: `33`
- `prompt eval time`: `211385.62 ms / 33 tokens = 0.16 tok/s`
- decode time: `35725.05 ms / 61 runs = 1.71 tok/s`
- total time: `247147.82 ms`
- host RAM peak: `15899996160 bytes` exactly at the cgroup cap
- final file cache: `14.66 GB`
- active_file: `8.47 GB`
- inactive_file: `6.19 GB`
- major faults: `16857`
- workingset file refaults: `21228`
- direct reclaim scan/steal:
  - `pgscan_direct=50912572`
  - `pgsteal_direct=28531879`
- `expert_pack` total across the run:
  - `iouring_reads=58392`
  - `iouring_bytes=333.8 GB`
  - `iouring_wait_us=37051518`
- fallback counter:
  - `[kimi_cpu_fallback_pack_mmap] enabled=1 hits=0 misses=0 bytes=0 fallback_gguf=0`

Comparison with normal held-out prompts:

| prompt | prompt tokens | TTFT ms | decode ms | RAM peak GiB | file GiB | pgmajfault | refault_file | iouring GiB | iouring wait ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_english_factual_01` | 16 | 7384.1 | 50092.9 | 11.72 | 11.22 | 2427 | 0 | 485.3 | 51721.9 |
| `test_english_factual_02` | 18 | 9329.4 | 47226.7 | 11.86 | 11.23 | 2503 | 0 | 471.8 | 50084.2 |
| `test_reasoning_math_01` | 33 | 211385.6 | 35725.1 | 14.81 | 13.65 | 16857 | 21228 | 310.9 | 37051.5 |
| `test_coding_01` | 23 | 10275.5 | 58906.3 | 11.86 | 11.23 | 2467 | 153 | 643.3 | 65140.8 |
| `test_chinese_01` | 17 | 7674.6 | 51014.9 | 11.86 | 11.23 | 2464 | 0 | 489.7 | 53707.3 |
| `test_mixed_instruction_01` | 22 | 9115.8 | 52469.6 | 11.86 | 11.23 | 2512 | 0 | 534.2 | 54961.3 |

Interpretation:

- The 211s TTFT is not model load and not decode; it is accounted as `prompt eval time`.
- It is also not explained by total `iouring_wait`, because the math prompt has less total `iouring_wait` than several normal prompts.
- The distinctive signal is host-memory pressure during prompt eval:
  - RAM hits the exact cgroup cap;
  - file cache grows to `13.65-14.66 GB`;
  - direct reclaim scans/stolen pages explode;
  - major faults and file refaults are much higher than normal prompts.
- This suggests prompt eval is still touching file-backed GGUF/mmap pages or otherwise building low-value page cache under the 16 GB cap, even though decode fallback counters are zero.
- Global VRAM split cannot fix this class of TTFT long tail.

Phase 4T plan:

1. Add phase-level instrumentation before changing cache policy.
   - Capture cgroup memory snapshots at:
     - after model load;
     - after dense/expert mmap drop;
     - before prompt eval;
     - after prompt eval and before first decode token;
     - after `LLAMA_DROP_EXPERT_MMAP_AFTER_PROMPT`;
     - after decode.
   - Log `madvise(DONTNEED)` wall time separately for dense and expert ranges.
   - Split `expert_pack` counters by phase: prompt eval vs decode.
   - Split `iouring_wait`, H2D enqueue count, pinned staging copies, and VRAM cache misses by phase.
   - Log any GGUF mmap tensor access by phase and tensor/role if feasible.

2. Re-run only a diagnostic prompt pair after instrumentation.
   - Use a dev prompt first; do not tune from held-out.
   - If no dev prompt reproduces the reclaim/refault pattern, run `test_reasoning_math_01` only as diagnostic and do not use it to build hotsets/profiles.
   - Acceptance for instrumentation: no behavior change, same output, and overhead below noise.

3. Decide the optimization from measured phase data.
   - If prompt eval faults GGUF expert pages, route prompt expert reads through expert pack / alias iouring path instead of mmap.
   - If `madvise(DONTNEED)` after prompt is the long pole, make it async or reduce the range set.
   - If file cache is low-value prompt residue, explicitly drop it earlier and replace it with controlled RAM expert cache only after proving no refault regression.
   - If prompt-phase expert batches are too wide and create reclaim pressure, cap prompt staging/concurrency separately from decode.

## Phase 5: Commit and push protocol

For every accepted improvement:

1. Update this plan before implementation with the exact hypothesis and expected upper bound.
2. Implement default-off if risk is nontrivial.
3. Run quick n32 dev check.
4. Run paired n96 dev check.
5. Run held-out test only after candidate survives dev.
6. Commit with full body: improvement size, env, commands, prompt split, output quality, TTFT/RAM/VRAM/IO metrics, and rollback point.
7. Push to the active `vendor/*kimi*` branch.

Rejected experiments remain documented with run paths and reason for rejection.
