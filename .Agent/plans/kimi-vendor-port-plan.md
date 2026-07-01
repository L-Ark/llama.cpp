# Kimi vendor port plan: n96 stable semantic output

## Goal

Port the minimum Kimi IQ3_S compatibility from `ik_llama` to the vendor branch so
that Kimi-K2.7-Code can produce stable, semantically correct `-n 96` output on a
single GPU.

The first success criterion is output quality and stability, not maximum token
rate. Speed improvements are only accepted when the same `-n 96` quality gate
continues to pass.

Current branch state:

- Base repo: `https://github.com/L-Ark/llama.cpp.git`
- Base branch: `feat/ds4-moe-stream-on-vendor`
- Local path: `/Users/spark/llama.cpp-vendor-kimi`
- Working branch: `feat/kimi-moe-stream-on-vendor`
- Base HEAD: `ef19fbdbf cuda: optionally drop streamed expert mmap pages`

## Existing vendor shape

The vendor branch already has upstream Kimi support and uses the newer per-model
graph layout:

- `src/models/kimi-linear.cpp` owns the Kimi graph and calls `build_moe_ffn(...)`.
- `src/llama-graph.cpp` / `src/llama-graph.h` own the shared MoE graph builder.
- `src/llama-model.cpp` reads Kimi MLA/KDA/MoE hparams under `LLM_ARCH_KIMI_LINEAR`.
- `src/llama-vocab.*`, `src/unicode.cpp`, `src/llama-chat.cpp`, and `common/chat.cpp`
  already contain Kimi-K2 tokenizer/chat handling.
- DS4 streaming/hot allocation is implemented through `src/models/deepseek4.cpp`,
  `src/llama-deepseek4-hot.*`, and `src/llama-context.cpp`.

Do not directly cherry-pick old `ik_llama` changes to `src/llama-build-context.cpp`.
The vendor integration points are model files, shared graph code, and CUDA MoE
stream runtime code.

## n96 acceptance gate

The fixed smoke prompt is inherited from the accepted `ik_llama` Kimi runs:

```text
Please introduce France in a short paragraph.
```

Use `-n 96` as the promotion gate. `-n 64` is allowed only for fast iteration.

A candidate passes only if all of these hold:

- The process exits successfully and generates the full requested output length
  unless it emits a normal EOS after a complete, coherent answer.
- The answer is about France and includes at least two relevant anchors such as
  Paris, Europe, French, culture, history, cuisine, art, tourism, or geography.
- The answer remains coherent through the end. The last 32 generated tokens must
  not contain unrelated drift, repeated junk, tokenizer artifacts, garbled
  characters, or malformed role/template text.
- No repeated n-gram loop is visible in the full answer.
- `read_failures = 0`; no short read, mmap offset error, checksum/pack mismatch,
  CUDA error, NaN, or sampling fatal error appears in logs.
- Under the target memory guard, there is no OOM and no unstable near-limit run.

For promotion, run the same final configuration three times with fixed seed and
sampling parameters. All three runs must pass the `-n 96` semantic gate.

## Runtime defaults for validation

Record the exact model path and expert pack for each run. The expected IQ3_S
model family is:

```text
/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S
```

Use the Kimi chat template from model metadata when available. If the CLI path
requires an explicit prompt wrapper, use the known-good `ik_llama` wrapper:

```text
<|im_user|>user<|im_middle|>Please introduce France in a short paragraph.<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>
```

Keep these parameters fixed for all quality comparisons:

- `-n 96`
- same seed
- same context length
- same temperature/top-p/top-k/min-p settings
- same model shard path
- same expert pack path
- same Kimi template path or prompt wrapper

Each run must save:

- command and environment
- stdout/stderr
- full generated answer
- token/s, seconds/token, TTFT, prompt eval timing
- VRAM before load, after load, and during decode
- cgroup `memory.peak`, `memory.current`, page cache, RSS, swap/OOM status
- MoE stream counters, including reads, bytes, cache hit/miss, and read failures

## Phase 0: vendor n96 baseline

Goal: establish the current vendor branch's Kimi quality failure mode before
porting any `ik_llama` runtime behavior.

Steps:

1. Build vendor with CUDA and MoE stream support.
2. Run the fixed France prompt with `-n 96` and minimal Kimi stream settings.
3. Record whether the failure is model load, tokenizer/template, graph/runtime,
   memory/OOM, read path, or semantic quality.
4. If model loading fails, reduce the issue to the first missing tensor, hparam,
   unsupported op, or stream runtime feature.

Acceptance:

- Baseline result is recorded with full logs.
- The next missing compatibility item is known and assigned to a concrete
  subsystem.

## Phase 1: correctness-first runtime diff

Goal: port only the MoE stream runtime differences required for Kimi IQ3_S to
complete `-n 96` correctly.

Diff vendor `ggml/src/ggml-cuda/moe_stream_batch.cu` against the accepted
`ik_llama` Kimi branch by behavior, not by whole-file replacement. Classify each
item as present, missing, or present with different behavior:

- `GGML_MOE_STREAM_BATCH_SPLIT_UPGATE`
- `GGML_MOE_STREAM_BATCH_SPLIT_UPGATE_SRC1_TOKEN`
- `GGML_MOE_STREAM_UP_GATE_DYNAMIC_X`
- `GGML_MOE_STREAM_UP_GATE_FUSED_MMQ_DYNAMIC_X`
- `GGML_MOE_VRAM_CACHE_CID2_MIB`
- `GGML_MOE_VRAM_CACHE_SPLIT`
- `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB`
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT`
- `GGML_MOE_DOWN_UNIFIED_IO`
- `GGML_MOE_DOWN_PARALLEL_STAGE`
- `GGML_MOE_STREAM_DOWN_Q8K`
- `GGML_MOE_STREAM_DOWN_Q8K_TYPES`
- `GGML_MOE_STREAM_DOWN_Q8K_LAYER_RANGE`
- `GGML_MOE_STREAM_DOWN_Q8K_SEPARATE_CACHE`
- `GGML_MOE_MMAP_DONTNEED`

Port order:

1. Read path and pack safety: expert-pack index/read validation, `read_failures`,
   mmap/dontneed behavior.
2. Batch shape compatibility: split up/gate and source-token handling.
3. Dynamic width compatibility: up/gate dynamic X and fused MMQ dynamic X.
4. Cache partition compatibility: CID2 and split VRAM cache.
5. Down path compatibility: unified down IO and Q8_K down cache.

Acceptance for each group:

- Build succeeds.
- The fixed `-n 96` run does not introduce a new crash, read failure, or semantic
  regression compared with the previous accepted group.
- The run log states whether the feature changed output hash, timing, memory, or
  MoE counters.

## Phase 2: stable n96 configuration

Goal: find the smallest runtime env stack that passes three consecutive `-n 96`
quality runs.

Start from this known `ik_llama` Kimi IQ3_S stack, but enable features
incrementally rather than all at once:

```sh
GGML_MOE_IO_BACKEND=iouring
GGML_MOE_VRAM_CACHE_MIB=24576
GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1
GGML_MOE_VRAM_CACHE_SAFETY_MIB=256
GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
GGML_MOE_STREAM=1
GGML_MOE_PARALLEL_EXPERTS=1
GGML_MOE_STAGE_PINNED_SLOTS=8
GGML_MOE_STREAM_FUSED_UP_GATE=1
GGML_MOE_STREAM_UP_GATE_FUSED_MMQ=1
GGML_MOE_STREAM_BATCH_ONLY=1
GGML_CUDA_NO_PINNED=1
GGML_MOE_MMAP_DONTNEED=1
GGML_MOE_IO_BYTES=8388608
GGML_MOE_STREAM_BATCH_SPLIT_UPGATE=1
GGML_MOE_STREAM_BATCH_SPLIT_UPGATE_SRC1_TOKEN=1
GGML_MOE_STREAM_IQ2S_BATCH_MMVQ=0
GGML_MOE_PREFETCH=0
GGML_MOE_PREFETCH_MODE=fadvise
GGML_MOE_DOWN_PARALLEL_STAGE=1
GGML_MOE_DOWN_UNIFIED_IO=1
GGML_MOE_VRAM_CACHE_SPLIT=1
GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=5
GGML_MOE_VRAM_CACHE_UPGATE_PCT=60
GGML_MOE_STREAM_DOWN_Q8K=1
GGML_MOE_STREAM_DOWN_Q8K_TYPES=all
GGML_MOE_STREAM_DOWN_Q8K_LAYER_RANGE=53-60
GGML_MOE_STREAM_DOWN_Q8K_SEPARATE_CACHE=1
GGML_MOE_STREAM_UP_GATE_DYNAMIC_X=1
GGML_MOE_STREAM_UP_GATE_FUSED_MMQ_DYNAMIC_X=1
GGML_MOE_VRAM_CACHE_CID2_MIB=2048
```

Keep `GGML_MOE_IO_URING_SINGLE` unset for the accepted Kimi path.

Acceptance:

- Same final env/config passes three consecutive `-n 96` France smoke runs.
- All three outputs are semantically correct through the tail.
- `read_failures = 0` in all three runs.
- No run breaches the memory/OOM gate.
- Token rate and TTFT are recorded, but a slower run can still pass if quality
  and stability are correct.

## Phase 3: optional speed work after n96 passes

Goal: improve throughput only after the stable `-n 96` configuration exists.

Rules:

- Any speed patch must keep the Phase 2 three-run `-n 96` gate green.
- Feature toggles default off unless they match existing vendor behavior.
- If a patch improves counters but harms semantic quality, revert it before the
  next candidate.
- Hot/cold Kimi dispatch is not part of the primary path.

If Kimi hot allocation is later needed, copy the vendor DS4 pattern rather than
old `ik_llama` build-context wiring:

- Add `src/llama-kimi-hot.{h,cpp}` only after proving the stream path is
  insufficient.
- Register it in `src/CMakeLists.txt`.
- Allocate from `src/llama-context.cpp`, guarded by
  `model.arch == LLM_ARCH_KIMI_LINEAR`.
- Dispatch from `src/models/kimi-linear.cpp` or shared MoE graph code.
- Keep all `KIMI_HOT_*` behavior opt-in.

## Quality classifier

Use both scriptable checks and manual review.

Scriptable fail patterns:

- fatal log patterns: `nan`, `failed to sample`, `fatal error`, `cuda error`
- repeated symbols: one punctuation/symbol repeated more than 10 times
- repeated 2/3/4-gram appearing three or more times
- malformed coordination such as `and and`
- missing `france` plus missing relevant France anchors

Manual review must check:

- The paragraph introduces France, not another country/topic.
- Facts are plausible and not obviously contradicted.
- The answer does not switch roles or expose broken template markers.
- The final sentence is coherent and not a degeneration tail.

## Commit policy

Commit source changes only after a feature group passes its `-n 96` gate. Keep
run logs and large artifacts out of source commits unless explicitly requested.
The `.Agent` plan may remain uncommitted unless the user asks to commit it.

## Implementation notes 2026-07-01

Implemented and verified so far:

- Kimi IQ3_S loader compatibility: `blk.*.ffn_exp_probs_b.bias` is accepted with
  fallback to the older unsuffixed tensor name. This fixes the initial
  `expected 1096, got 1036` load failure.
- Deferred expert CPU overrides: when `--defer-experts` is active, routed expert
  tensors are forced to CPU buffer types before applying GPU layer offload. This
  avoids trying to allocate the full 365.49 GiB expert range on CUDA.
- CUDA MoE stream build compatibility: `GGML_CUDA_MOE_STREAM_BATCH` builds on the
  vendor branch, with liburing detected when available and disabled cleanly when
  unavailable.
- Correctness-first fused up/gate graph hook: added `GGML_OP_MOE_FUSED_UP_GATE`
  and a CPU dispatch path that can call the CUDA streaming up/gate batch helper
  for decode.
- Fixed a stream batch correctness bug in `ggml_cuda_moe_stream_mmvq_batch_dev`:
  the helper now uses `d_x_ids` and `src0_stride` to select the staged active
  expert slot. Before this fix, every active route used the first staged expert,
  causing immediate semantic degeneration.

Current validation status:

- Build passes remotely in `/root/lfz/llama.cpp-vendor-kimi/build-cuda-batch`.
- `llama-completion --defer-experts --fit on ... -n 32 -st` now produces a
  semantically related France answer, but it is not stable enough:
  `<think>1.5 ... France is a large country in Europe ... it is is located in the, and, and`.
- `read_failures = 0` in the tested expert-pack runs.
- Raw wrapper with `-no-cnv` no longer produces token garbage, but answers with
  meta text instead of the requested paragraph; use the model chat template path
  for further quality testing.

Remaining blockers for `-n 96` acceptance:

- The ik `GGML_MOE_STREAM_UP_GATE_FUSED_MMQ` path has not been ported. Directly
  copying ik `mmq_id_common.cuh` and its template instances into vendor fails
  because it depends on ik-only quant types and conflicting CUDA helper
  definitions. The next implementation should be a vendor-native minimal
  IQ3_XXS/IQ2_S fused MMQ up/gate kernel, or a heavily pruned and namespaced
  port of just the needed id-MMQ pieces.
- Vendor still reserves a much larger graph than ik for this workload:
  tested `-c 512` chat path showed `CUDA0 model buffer size = 25351.19 MiB` and
  `CUDA0 compute buffer size = 4127.01 MiB`, leaving only about 0.7 GiB of MoE
  VRAM cache with 0% cache hit rate. The accepted ik run had roughly
  `CUDA0 buffer size = 11153.44 MiB` and `CUDA0 compute buffer size = 334 MiB`.
- ik exposes and enables `ggml_backend_sched_set_only_active_experts()`. The
  vendor branch has partial active expert copy/cache code but not the same
  scheduler switch and graph-size behavior. This needs a deliberate port rather
  than a whole-file replacement.
- Forcing `-b 1 -ub 1` is not a solution: it reduced CUDA compute buffer to
  about 0.93 MiB, but CUDA model buffer grew to about 29.7 GiB and the test
  emitted EOS immediately.

## Main risks

- Whole-file cherry-picks from `ik_llama` can patch the wrong layer because vendor
  moved graph logic into `src/models/*.cpp`.
- Some env names exist in both trees but have different behavior. Verify counters
  and output, not just string presence.
- Kimi hot/cold dispatch was not the accepted stable speed path; porting it first
  can delay the n96 quality target.
- NVFP4/IQ3 comparisons are out of scope for this port unless the target GGUF and
  kernels are explicitly switched.

## Implementation notes 2026-07-01 later pass

Additional implemented fixes:

- Moved deferred expert CPU buffer overrides before tensor creation in
  `llama_model::load_tensors()`. This fixed `--fit off -ngl 99`: the validated
  run now reports `CUDA0 model buffer size = 11153.44 MiB` instead of attempting
  a `385415 MiB` CUDA model allocation.
- Preserved Kimi's `deepseek2.rope.scaling.yarn_log_multiplier = 0.1` instead
  of dividing it by `0.1`. Vendor now prints `rope_yarn_log_mul = 0.1000`,
  matching ik metadata handling. A follow-up experiment that also changed the
  runtime `yarn_attn_factor` to ik's newer build-context semantics regressed
  output and was reverted; keep only the metadata fix for this vendor graph.
- Switched DeepSeek2/Mistral4 MoE expert selection from
  `ggml_argsort_top_k()` to `ggml_top_k()` to match the ik Kimi path. This
  improved early route overlap with the ik reference.

Latest validation:

- Build still passes for `llama-completion` and `llama-cli`.
- `--defer-experts --fit off -ngl 99` with the France IQ3_S expert pack is now
  memory-stable: model buffer ~11.15 GiB, compute buffer ~4.13 GiB,
  `read_failures = 0`.
- Best current short run is chat template plus `--reasoning off`:
  `.France is a beautiful country in Europe. It are known for its famous cities
  like Paris, and it is also the main of the country. Paris, the`
  This is semantically related but still fails the n96 quality bar because of
  grammar/factual instability and leading punctuation.
- Raw wrapper still does not match ik quality under vendor `llama-completion`;
  with the current graph it tends toward meta text or malformed starts.

Updated blockers:

- The remaining quality gap is most likely in the unported ik CUDA MoE math path:
  ik activates `IQ3_XXS fused MMQ up/gate path active` and
  `down Q3_K/IQ4_XS Q8_K-reference batch path active`; vendor currently uses a
  correctness-first MMVQ up/gate path and leaves down batch disabled/fallback.
- Whole-file copying ik `mmq_id*` is still not viable. The next patch should
  port a pruned id-MMQ subset only for the Kimi IQ3_S tensor types actually seen
  here: up/gate `IQ3_XXS` / `IQ2_S`, down `Q3_K` / `IQ4_XS`, with vendor type
  guards and isolated CMake entries.
- Do not run the three-run `-n 96` acceptance gate until an n32 smoke produces a
  grammatically stable France paragraph without template/meta artifacts.

## Implementation notes 2026-07-01 down Q8_K / MMQ probe

Additional implemented fixes:

- Added optional `x_channel_ids` support to vendor CUDA MMQ launch plumbing so a
  staged active expert slot can be selected independently from the logical
  channel. Normal MMQ callers pass `nullptr`, preserving existing behavior.
- Added an opt-in vendor MMQ up/gate probe under
  `GGML_MOE_STREAM_UP_GATE_FUSED_MMQ=1`. It builds and no longer crashes after
  stride fixes, but it is only a wrapper around vendor MMQ launches, not the ik
  id-MMQ fused tile path. It did not improve Kimi output and must not be used as
  the accepted path.
- Added down `Q3_K` / `IQ4_XS` x `Q8_K` reference batch support to
  `moe_stream_batch.cu`, including CUDA Q8_K quantization and late-layer range
  gating via `GGML_MOE_STREAM_DOWN_Q8K_LAYER_RANGE`.
- Scoped generic CPU/CUDA down batch dispatch to `ffn_down_exps` tensors only.
  This avoids sending `ffn_gate_exps` / `ffn_up_exps` prompt matmuls into the
  down batch probe.
- Fixed the down Q8_K batch scatter bug: the Q8_K kernel must receive
  `bc.d_ids_dst`; passing `nullptr` writes active routes into compact columns and
  corrupts decode output.

Validation from this pass:

- Remote build passes in
  `/root/lfz/llama.cpp-vendor-kimi/build-cuda-batch`.
- Q8_K down path activation is confirmed:
  `[moe_stream_batch] down Q3_K/IQ4_XS Q8_K-reference batch path active`.
- `n2` with down Q8_K after the scatter fix returns `.France`, matching the
  current short baseline prefix.
- Full-layer down Q8_K is not acceptable:
  `/root/lfz/runs/vendor-kimi/20260701-085957Z-n32-down-q8k-scatterfix`
  produced
  `.France, the French people, and the French of the French, ...`.
- Late-layer down Q8_K with `GGML_MOE_STREAM_DOWN_Q8K_LAYER_RANGE=53-60` is
  better but still not accepted:
  `/root/lfz/runs/vendor-kimi/20260701-090303Z-n32-down-q8k-scatterfix-l53-60`
  produced
  `.France is a country in Europe, with its capital is Paris. It has a famous landmark, the Eiffel Tower. The country has a beautiful, with`.
- The same late-layer config fails the `n96` quality gate:
  `/root/lfz/runs/vendor-kimi/20260701-090629Z-n96-down-q8k-scatterfix-l53-60`
  degenerated into repeated malformed clauses:
  `... The country is a, with a, and a, for a, and a, and a ...`.
- All above runs had `read_failures=0`; the failure is semantic quality, not
  expert-pack IO.

Current conclusion:

- Down Q8_K is now a working, scoped probe, but it is not sufficient for the
  `n96` target.
- The next implementation should focus on the true ik id-MMQ up/gate path for
  Kimi's observed up/gate types (`IQ3_XXS` and `IQ2_S`). The existing vendor MMQ
  wrapper is not an adequate substitute.

## Implementation notes 2026-07-01 n96 quality fix

Root cause update:

- The target IQ3_S Kimi GGUF reports `general.architecture = deepseek2`, so the
  accepted quality path is the vendor DeepSeek2 graph, not the newer
  `kimi-linear` graph.
- MMQ/down-Q8K probes were not the root quality issue. The vendor MMQ up/gate
  compare path matched the existing MMVQ compact output closely
  (`rel_rmse ~= 1e-4`) but still produced bad text when enabled.
- The decisive quality gap was the DeepSeek2 YaRN attention scale. Vendor had
  already adjusted `yarn_attn_factor` for RoPE, then computed `kq_scale` with
  an extra generic `0.1 * yarn_log_multiplier` factor. For this Kimi GGUF,
  `yarn_log_multiplier` is already `0.1`, and ik uses it as the full multiplier
  in `1 + yarn_log_multiplier * log(scale)`.

Implemented fix:

- Updated `src/models/deepseek2.cpp` so DeepSeek2/Kimi `kq_scale` uses:
  `mscale = attn_factor_org * (1 + rope_yarn_log_mul * log(1 / freq_scale))`.
  The RoPE-side adjusted `attn_factor` is left unchanged.

Acceptance validation:

- Remote build passed in `/root/lfz/llama.cpp-vendor-kimi/build-cuda-batch`.
- `n2` raw ik-style wrapper improved from the previous `The user` failure to
  `France is`.
  Run: `/root/lfz/runs/vendor-kimi/20260701-094340Z-n2-ik-wrapper-nocnv-kqscale`
- `n32` produced a natural paragraph:
  `France is a country in Western Europe known for its rich history, art, and culture...`
  Run: `/root/lfz/runs/vendor-kimi/20260701-094600Z-n32-ik-wrapper-nocnv-kqscale`
- Three independent `n96` runs passed the semantic quality gate. The output was
  stable and ended naturally with `<|im_end|>`:
  `France is a country in Western Europe known for its rich history, art, and culture. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. France is also celebrated for its cuisine, wine, and picturesque countryside, including regions like Provence and the Loire Valley. It has played a major role in shaping European politics, philosophy, and the arts.`
  Runs:
  `/root/lfz/runs/vendor-kimi/20260701-094850Z-n96-ik-wrapper-nocnv-kqscale-seed1`
  `/root/lfz/runs/vendor-kimi/20260701-095313Z-n96-ik-wrapper-nocnv-kqscale-repeat/run1`
  `/root/lfz/runs/vendor-kimi/20260701-095313Z-n96-ik-wrapper-nocnv-kqscale-repeat/run2`

Current conclusion:

- The plan's primary goal, stable semantically correct `n96` output on the
  Kimi IQ3_S target, is met by the DeepSeek2 YaRN `kq_scale` fix plus the
  earlier loader/deferred-expert compatibility fixes.
- The opt-in MMQ/down-Q8K probes remain experimental speed work. They should
  stay disabled for the accepted quality path unless separately revalidated.
