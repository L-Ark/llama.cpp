# Wici GLM-5.1 Interactive MoE Plan

## Product Metric

The primary path is real interactive chat:

```sh
python3 scripts/moe-run.py --chat --chat-load-mode fast-prompt
```

Measure it through a pseudo-TTY over `ssh wici`; API/SSE benchmarks are regression checks only.

TTFT is now the first optimization target. Token rate remains a hard regression
gate, but candidates are ranked by `interactive_ttft_s` and prompt eval time
before time-to-type or total generation throughput.

## Current Accepted Baseline

Host: `wici` (`192.168.1.128` in local `~/.ssh/config`)

Model: GLM-5.1 UD-IQ3_XXS

Historical one-run baseline from `bench/wici-glm51-interactive-calibration/20260526-190518-summary.json`:

- `time_to_type_s`: 8.65
- `interactive_ttft_s`: 15.07
- `eval_tokens_per_s`: 0.98
- n36 canary sha: `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`

Current TTFT-focused promoted candidate under p95 watch:

- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=3`
- default-preset n84 validation with current Wici background VRAM and
  `GGML_MOE_VRAM_CACHE_SAFETY_MIB=256`:
  - `time_to_type_s`: 10.35
  - `interactive_ttft_s`: 12.96
  - `eval_tokens_per_s`: 0.95
  - `moe_vram_cache_failed`: false
- default-preset n36 canary:
  - self-contained preset run, without harness-injected cache/cublas env:
    - `time_to_type_s`: 10.23
    - `interactive_ttft_s`: 12.68
    - `eval_tokens_per_s`: 0.88 on the short 15-token canary sample
    - `moe_vram_cache_failed`: false
    - sha: `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`
  - earlier validation:
    - `time_to_type_s`: 10.82
    - `interactive_ttft_s`: 12.92
    - `moe_vram_cache_failed`: false
  - sha: `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`
- post-diagnostic default-preset n36 canary, after restoring the production
  decode source-row mapping and keeping dynamic MMQ/Q8_1 prompt code gated:
  - `time_to_type_s`: 10.77
  - `interactive_ttft_s`: 13.07
  - `eval_tokens_per_s`: 0.91
  - `moe_vram_cache_failed`: false
  - sha: `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`

This improves typical TTFT versus the 15.07s historical baseline while
preserving the n36 canary. The dynamic scheduler has good single-run and
two-run confirmation data, but later default-preset checks still show occasional
prompt-compute outliers, so the remaining work is variance reduction rather than
more parameter sweeping.

Promoted config:

- `N_GPU_LAYERS=60`
- `GGML_CUDA_EAGER_CUBLAS=1`
- `GGML_MOE_VRAM_CACHE_MIB=1536`
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1`
- `GGML_MOE_VRAM_CACHE_SAFETY_MIB=256`
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=60`
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=10`
- `GGML_MOE_RAM_TIER_MIB=3072`
- `GGML_MOE_RAM_TIER_SKIP=447`
- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=3`
- `GGML_MOE_PROMPT_DYNAMIC_EXPERTS=1`

Latest commit-check runs on 2026-05-28:

- n36 default preset: canary passed, `interactive_ttft_s=12.03`,
  `time_to_type_s=10.17`, short-run `eval_tokens_per_s=0.92`.
- n84 default preset, two runs: one run measured
  `interactive_ttft_s=12.18`, while the second hit a prompt-eval outlier at
  `15.46`; token rate was `0.91-0.92`.
- Interpretation: the promoted default is semantically valid and preserves cache
  allocation, but the observed p95 is not yet a closed TTFT win. Do not call the
  Wici latency goal complete until the prompt MoE variance is addressed or the
  p95 gate is revalidated.

The earlier 2048 MiB cache can OOM at first CUDA graph launch after the TTFT
runtime rebuild. The 1792 MiB cache can also partially allocate when runtime
free VRAM is about 1437 MiB, causing `moe_vram_cache_failed` and token-rate
collapse. On the current driver/background-load state, 128 MiB and 192 MiB
runtime safety margins also allocate a larger cache and then OOM at first CUDA
graph instantiation. Keep 1536 MiB plus runtime auto-clamp and a 256 MiB
safety margin as the stable interactive default unless a later change proves
more graph headroom or dynamic cache resizing with canary preserved.

## Low-Host-RAM Variant

There is one accepted constrained-memory variant for the same GLM-5.1
interactive path when the process must fit under `MemoryMax=2G` with
`MemorySwapMax=0` on an RTX 5090 32 GB host.

This is related to the main Wici interactive plan, but it is not the promoted
default TTFT preset above. The normal interactive default relies on a RAM tier
(`GGML_MOE_RAM_TIER_MIB=3072`), which is invalid under a strict 2 GiB process
cap. The accepted workaround removes RAM-tier residency and shifts the useful
working set into VRAM plus SSD-backed direct expert reads.

Accepted low-host-RAM profile:

- CLI: `python3 scripts/moe-run.py --chat --low-host-ram-profile glm51-2gb-vram8`
- `N_GPU_LAYERS=79`
- `GGML_MOE_RAM_TIER_MIB=0`
- `GGML_MOE_RAM_TIER_SKIP=0`
- `GGML_MOE_IO_BACKEND=direct`
- `GGML_MOE_VRAM_CACHE_MIB=8192`
- `GGML_MOE_VRAM_CACHE_POLICY=lfu_lru`
- `GGML_MOE_VRAM_CACHE_UPGATE_PCT=60`
- `GGML_MOE_VRAM_PROFILE=presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv`
- `GGML_MOE_VRAM_PROFILE_PROTECT=1`
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=10`
- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=3`

Measured n84 results under `MemoryMax=2G`, `MemorySwapMax=0`,
RTX 5090 32 GB, GLM-5.1 UD-IQ3_XXS, and `glm51-iq3xxs.expert-pack`:

- pre-optimization baseline, top4 profile, `VRAM=1536`, `RAM=0`:
  `direct_reads=118824`, `VRAM hit=6.0%`, `RAM hit=0.0%`,
  `total_ms=105695.33`, `read_failures=0`
- optimized candidate, top8 profile, `VRAM=8192`, `RAM=0`:
  `direct_reads=91145`, `VRAM hit=29.1%`, `RAM hit=0.0%`,
  `total_ms=88138.02`, `read_failures=0`
- optimized candidate, top8 profile, `VRAM=12288`, `RAM=0`:
  `direct_reads=87729`, `VRAM hit=31.8%`, `RAM hit=0.0%`,
  `total_ms=109228.67`, `read_failures=0`

Interpretation:

- `glm51-2gb-vram8` is the accepted operational default for strict 2 GiB host
  limits. It materially improves VRAM hit rate over the small-cache baseline
  without reintroducing RAM-tier pressure.
- `glm51-2gb-vram12` is available as a higher-hit-rate variant, but it pushes
  runtime VRAM to about 30.8 GiB and did not improve end-to-end latency in the
  measured runs on the 5090 32 GB card. Treat it as a narrower fit-check
  option, not the default.
- The win here comes from moving hot experts into a protected VRAM cache seeded
  by the top-8 runtime profile while leaving cold misses on direct SSD expert
  reads. It is not a RAM-cache optimization.

Current integrated reproduction check:

- preset:
  `presets/moe/glm51/rtx5090-interactive-n84-repro.json`
- host-RAM overlay:
  `--low-host-ram-profile glm51-2gb-vram8`
- result: exit status `0`, `direct_reads=91145`, `VRAM hit=29.1%`,
  `read_failures=0`
- parsed answer SHA:
  `a2331ca78c18fcf3b30ae07521fc8746f097e8220c29b1c28f61fff220fcd990`

Reference run:

```sh
systemd-run --pty --wait --collect \
  -p WorkingDirectory="$(pwd)" \
  -p MemoryMax=2G \
  -p MemorySwapMax=0 \
  python3 scripts/moe-run.py \
    --chat \
    --chat-load-mode fast-prompt \
    --preset presets/moe/glm51/rtx5090-interactive-n84-repro.json \
    --low-host-ram-profile glm51-2gb-vram8 \
    -- \
    -n 84 \
    -b 2048 \
    -tb 24 \
    -t 8 \
    --simple-io \
    --seed 42 \
    --ignore-eos
```

## Recent Mechanism Filter

Ignore papers older than two years for new mechanism choices. As of
2026-05-28, the cutoff is 2024-05-28. Do not use FlexGen/MoE-Infinity-era
offload heuristics as a plan basis unless a recent system independently
revalidates the same mechanism on modern MoE inference. The relevant recent
ideas are:

- Tutti (arXiv:2605.03375, submitted May 5 2026) targets SSD-backed KV cache,
  not this short GLM-5.1 single-user prompt directly. The transferable ideas
  are GPU/IO critical-path removal, bulk object movement, and slack-aware
  scheduling. In this repo that maps to post-ready profile preload and future
  bulk expert movement, not KV-cache SSD work.
- Speculating Experts (arXiv:2603.19289, submitted Mar 9 2026) predicts
  future experts from internal model representations to overlap CPU-to-GPU
  movement with compute. In this repo it is applicable only after the trace
  shows expert movement inside submit-to-first-token; the first version should
  be a route-predictor accuracy report, not a production prefetcher.
- Fast MoE Inference via Predictive Prefetching and Expert Replication
  (arXiv:2605.11537, submitted May 12 2026) reinforces the same condition:
  predictive prefetch or replication is useful only when expert wait and load
  imbalance are on the critical path. Current Wici traces put most TTFT in CPU
  prompt MoE arithmetic/page faults, so use this later for token-rate and
  generation-cache policy, not before row-level prompt correctness.
- FineMoE (arXiv:2502.05370, revised Oct 4 2025) supports the route-aware
  direction: use fine-grained expert selection patterns and prompt hints to
  guide prefetching, caching, and offloading. In this repo that maps to
  `scripts/calibrate-wici-interactive-config.py` producing one route/cost
  selected cache or movement plan, not manual cache/thread sweeps.
- HarMoEny (arXiv:2506.12417, June 2025) reports large multi-GPU MoE TTFT wins
  by addressing expert-parallel load imbalance. Wici is single-host and
  constrained by CPU prompt MoE/page faults, so borrow only the mechanism test:
  prove imbalance or movement is critical before adding scheduling complexity.
- MoESD (NeurIPS 2025, arXiv:2505.19645) supports revisiting speculative
  decoding for sparse MoE, but it targets generation throughput more than
  immediate TTFT. Treat it as token-rate work after TTFT is no longer dominated
  by first prompt CPU MoE.
- Recent speculation systems such as SpecMD-style masked/self speculation are
  relevant to generation tok/s, not the current first-prompt bottleneck, unless
  the n36 canary is preserved and the TTFT trace shows decode dominates.
- Recent serving runtime work in vLLM, SGLang, and DeepEP points at chunked
  prefill, prefix caching, CUDA graphs, optimized MoE kernels, analytical
  resource selection, and communication/compute overlap. For single-host Wici
  interactive chat, borrow the principles, not distributed all-to-all code:
  stable graph setup, exact prompt MoE kernels, route/layout reuse, and
  trace-derived prefetch scheduling.

Source links kept with this plan:

- https://arxiv.org/abs/2605.03375
- https://arxiv.org/abs/2603.19289
- https://arxiv.org/abs/2605.11537
- https://arxiv.org/abs/2502.05370
- https://arxiv.org/abs/2506.12417
- https://arxiv.org/abs/2505.19645
- https://github.com/vllm-project/vllm
- https://github.com/sgl-project/sglang
- https://github.com/deepseek-ai/DeepEP

Recent mechanism probes:

- After-ready non-blocking startup profile preload:
  - `time_to_type_s`: 8.48
  - immediate-submit `interactive_ttft_s`: 14.54
  - n36 canary: pass
  - Result: useful only for human typing slack mode; rejected as the default
    immediate-submit metric because TTFT worsens.
- DRAM RAM-tier expansion to 4096 MiB:
  - `time_to_type_s`: 10.98
  - `interactive_ttft_s`: 13.12
  - `eval_tokens_per_s`: 0.92
  - n36 canary: pass
  - Result: rejected; more DRAM residency did not beat the current 3072 MiB
    tier on TTFT and failed the token-rate gate.
- DRAM RAM-tier expansion to 6144 MiB:
  - Result: rejected; startup rose to 12.57s and first-prompt CUDA graph
    instantiation OOMed after pinning the larger RAM tier.
- Dense-layer-to-cache VRAM tradeoff with `-ngl 58`:
  - full 1536 MiB cache allocated successfully
  - `interactive_ttft_s`: 12.87
  - `eval_tokens_per_s`: 0.86
  - n36 canary: pass
  - Result: rejected; freeing VRAM for cache hurts dense-layer throughput more
    than it helps this run.
- On-the-fly CUDA emulation of the CPU `IQ2_S -> Q8_K_R8` prompt repack:
  - `GGML_MOE_STREAM_PROMPT_UP_GATE=exact-q8-k`
  - with normal 256 MiB graph safety: exact IQ2_S prompt path activates, then
    first-prompt graph/compute OOMs.
  - with 768 MiB graph safety: run completes but n36 canary fails, generating
    `3294949294932947`; `interactive_ttft_s`: 15.34; `eval_tokens_per_s`: 0.86.
  - CPU-vs-GPU validation still diverges on later prompt layers, e.g.
    `blk.75.ffn_up_exps.weight` max_abs about 1184.5 with GPU near zero and
    CPU large.
  - Result: rejected. Do not run more performance probes on this implementation.
    `IQ2_S` is now excluded from the `exact-q8-k` opt-in path so future runs
    cannot accidentally use this broken arithmetic. The next exact-path step
    must first build a minimal row-tile comparator for the repacked `IQ2_S`
    arithmetic and prove bit/epsilon agreement before reconnecting it to the
    full interactive path.
- `GGML_MOE_STREAM_PROMPT_UP_GATE=exact-q8-k` after the `IQ2_S` guard:
  - n36 canary: pass, sha
    `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`
  - `interactive_ttft_s`: 13.35
  - `eval_tokens_per_s`: 0.91 on the short 15-token canary sample
  - stderr showed the normal `IQ2_S batched MMVQ up/gate path active` line and
    no exact Q8_K prompt activation.
  - Result: accepted as a guardrail, not as a performance improvement.
- CPU repack oracle for the next exact prompt path:
  - added `iqk_convert_repack_q8_r8()` and
    `tests/test-iqk-iq2-repack.cpp`.
  - Wici CPU result:
    `IQ2_S repack comparator ok: rows=8 k=512 direct_vs_scalar=0.000610352 repacked_vs_direct=9.01288 max_repacked_abs=1617.65`.
  - Wici CUDA tile result after matching the CPU fp16 scale round:
    `IQ2_S CUDA comparator: ok rows=8 k=512 max_abs=1 tol=425.308`.
  - Result: accepted as a minimal tile oracle only. The full prompt path is
    still rejected: re-enabling `IQ2_S` for `exact-q8-k` with 768 MiB graph
    safety generated `19999999`, failed the n36 canary, regressed TTFT to
    15.32s, and dropped `eval_tokens_per_s` to 0.84. CPU-vs-GPU validation
    still reports real-model prompt mismatches, e.g.
    `blk.77.ffn_up_exps.weight max_abs=13.0652 rmse=0.725633`.
  - `IQ2_S` remains excluded from the full `exact-q8-k` prompt path. Two
    explicit diagnostic-only env values are available:
    `exact-q8-k-iq2-probe` for the repacked approximation and
    `exact-q8-k-iq2-direct-probe` for direct `IQ2_S x Q8_K`.
  - Route-aware validation now prints the flat row, compact active row,
    expert, route index, destination rank, token, and column for the max
    mismatch. The direct probe still fails on real prompt rows; for example,
    `blk.75.ffn_up_exps.weight` reports `max_abs=1184.52` at
    `flat_row=1 col=1899 active=16 expert=27 route_i=0 dst=1 token=0`.
  - The next step is a real routed-row comparator that replays this captured
    expert row plus prompt row through the CUDA dot and CPU IQK dot before any
    interactive exact-path probe.
- Real routed-row replay comparator:
  - Added `ggml_cuda_moe_iq2_prompt_replay()` and route-aware replay logging.
  - For `blk.75.ffn_up_exps.weight expert=27 token=0 col=1899`, direct CUDA
    replay matches the CPU fused value: `-1184.25`; the repacked replay is
    `-1184.27`.
  - With the larger safety256 cache, full exact prompt staging also matches the
    CPU point with `max_abs=0.000244141`, proving the arithmetic is not the
    current blocker.
  - With the safety768 cache, active prompt routes exceeded up/gate cache slots
    (`160 > 122`), so staged slots were reused before the kernel consumed them.
    The exact prompt path now declines in that case with
    `chunked staging required` instead of silently corrupting output.
  - Result: chunked exact Q8_K prompt staging was the next diagnostic target at
    that point, not parameter sweeping. It has since been implemented and
    rejected as a production TTFT win.
- Chunked exact Q8_K prompt staging:
  - Implemented bounded up/gate staging so oversized prompt route sets are split
    into cache-safe chunks instead of reusing slots before compute.
  - Fixed the prompt compact-row scatter bug and protected already-staged slots
    within each chunk.
  - Validation now clears the previous oversized prompt-layer corruption, but
    the accepted semantic path still failed promotion: `exact-q8-k-iq2-probe`
    measured `interactive_ttft_s=17.23`, changed the n36 output, and stopped
    early. Direct IQ2 remains diagnostic-only.
  - Result: rejected as a TTFT win. Keep the chunked path as a diagnostic
    correctness guard, not a production config.
- Trace-seeded page-cache prefetch probe:
  - Added `LLAMA_CHAT_STARTUP_PROFILE_PAGECACHE_PREFETCH=1`, which reads the
    TTFT route/profile CSV and issues bounded expert-row page-cache prefetches
    through the existing prefetch backend.
  - Probe:
    `LLAMA_CHAT_STARTUP_PROFILE_PAGECACHE_PREFETCH=1`,
    `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_AFTER_READY=1`,
    `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_JOIN_ON_SUBMIT=0`,
    `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_EXPERT_ROWS=512`,
    `GGML_MOE_PREFETCH_MODE=readahead`.
  - n36 canary passed and time-to-type improved to `8.11s`, but immediate-submit
    TTFT regressed to `16.13s`.
  - Result: rejected for the primary metric. Page-cache hints are not enough;
    the next recent mechanism should be a compact TTFT expert pack or equivalent
    bulk-read path for the traced critical prefix.
- TTFT expert pack probes:
  - Added `scripts/create-ttft-expert-pack.py` to build bounded first-prompt
    expert packs from the TTFT route trace and the full `.expert-pack`.
  - Added `LLAMA_CHAT_TTFT_EXPERT_PACK=<pack>` startup registration. Copying
    pack rows into anonymous hot-expert RAM worked but regressed TTFT because
    the extra startup copy and memory pressure outweighed the hit benefit.
  - Added `LLAMA_CHAT_TTFT_EXPERT_PACK_MMAP=1`, which maps the TTFT pack and
    registers slices as hot experts without copying them into anonymous RAM.
  - Trace-order mmap pack, 512 rows: canary passed, `time_to_type_s=10.53`,
    `interactive_ttft_s=12.82`, `eval_tokens_per_s=0.87`.
  - Initial compute-order mmap pack, 512 rows: canary passed,
    `time_to_type_s=10.70`, `interactive_ttft_s=12.60`,
    `eval_tokens_per_s=0.91`.
  - Compute-order mmap pack, 768 rows: canary passed but regressed versus 512
    rows: `interactive_ttft_s=12.74`.
  - Fixed `scripts/create-ttft-expert-pack.py` so `--max-expert-rows` is applied
    after compute-order sorting, not before. Corrected global compute-order
    mmap pack, 512 rows:
    - n36: canary passed, `time_to_type_s=10.63`,
      `interactive_ttft_s=12.54`, `prompt_eval_time_ms=12509.76`,
      `eval_tokens_per_s=0.90`.
    - n84: `time_to_type_s=10.78`, `interactive_ttft_s=12.71`,
      `eval_tokens_per_s=0.94`.
  - Rebuilding the same 512-row pack from the immediately following trace
    (`20260528-044618`) regressed to `interactive_ttft_s=12.87`, so pack
    construction should be scored against a canary probe instead of blindly
    regenerating from the newest trace.
  - Follow-up traced rerun of the corrected 512-row mmap pack:
    `time_to_type_s=10.75`, `interactive_ttft_s=13.15`,
    `eval_tokens_per_s=0.85`, n36 canary passed. Comparing against the
    `20260528-044618` baseline trace showed prompt compute regressed from
    `9599 ms` to `9885 ms` and major-fault counts remained about `112k-116k`.
  - Result: rejected as the next TTFT mechanism. The pack registers too few
    of the 17,091 unique first-prompt expert rows to reduce the dominant CPU
    prompt MoE bucket. Do not keep growing pack size by hand; move to a
    structural prompt-kernel mechanism.
- Recent prompt-kernel probes:
  - Same-layer down prefetch from up/gate routes:
    `GGML_MOE_PREFETCH_DOWN_FROM_UPGATE=1` preserved the canary but regressed
    n36 TTFT to `13.28s`. Page-cache hints issued inside the current layer are
    too late/weak for the immediate-submit metric.
  - Decode CUDA up/gate control:
    `GGML_MOE_STREAM_FUSED_UP_GATE=1 GGML_MOE_STREAM_PROMPT_UP_GATE=1`
    preserved the n36 canary. One n84 same-session check improved TTFT from
    `13.43s` to `12.64s` and eval from `0.93` to `0.96 tok/s`, but stderr
    showed the prompt CUDA path still declined. Treat this as a decode/control
    observation, not a prompt TTFT solution.
  - Unsafe Q8_1 prompt CUDA:
    `GGML_MOE_STREAM_PROMPT_UP_GATE=unsafe-q8-1` entered the prompt path
    (`rows=160`) but then OOMed when decode/down buffers initialized. It is
    not acceptable as a default because it is both quality-changing and
    headroom-unsafe on current Wici.
  - Exact `IQ2_S x Q8_K` prompt CUDA current recheck:
    - Added a multi-active CUDA comparator to `test-iqk-iq2-repack`; Wici
      passes both the original one-active tile and the new four-active tile.
    - Fixed chunked exact prompt staging so transient prompt rows are not
      inserted as protected profile-preload cache entries. The chunk size is
      now based on evictable, unpinned cache slots.
    - With normal 256 MiB safety, exact prompt staging now enters the chunked
      path instead of declining, but decode/down later OOMs because the cache
      leaves too little graph headroom.
    - With 768 MiB safety, the run completes but fails the n36 canary,
      regresses TTFT to `19.39s`, and drops `eval_tokens_per_s` to `0.84`.
    - Result: exact prompt CUDA remains rejected as a production path. The
      useful retained work is the guarded comparator and transient staging
      fix; next TTFT work should target CPU prompt MoE cost directly or a new
      exact GPU design that does not change semantics or consume decode
      headroom.
- Exact CPU IQK `up_gate_many` current recheck:
  - `GGML_MOE_PROMPT_UP_GATE_MANY=1 GGML_MOE_VRAM_CACHE_SAFETY_MIB=512`
    preserved the n36 canary and measured `interactive_ttft_s=12.70`,
    `time_to_type_s=10.57`, and `eval_tokens_per_s=0.90`.
  - Result: useful guarded diagnostic but not a significant TTFT improvement.
- CPU prompt hybrid many scheduler:
  - Added opt-in `GGML_MOE_PROMPT_UP_GATE_HYBRID=1` and
    `GGML_MOE_PROMPT_DOWN_HYBRID=1`. The scheduler preserves the existing IQK
    arithmetic and only repartitions CPU threads across active prompt experts.
    It falls back to the existing many path when active experts are not fewer
    than worker threads.
  - Wici build and focused correctness tests passed.
  - n36 probes preserved the canary but regressed TTFT:
    - up/gate hybrid: `time_to_type_s=10.23`,
      `interactive_ttft_s=17.41`, `eval_tokens_per_s=0.91`.
    - down hybrid: `time_to_type_s=10.45`,
      `interactive_ttft_s=17.07`, `eval_tokens_per_s=0.87`.
  - Trace result: hybrid intra-expert threading increases the dominant prompt
    path on this host, consistent with the TTFT path being memory/page-fault
    limited rather than under-threaded arithmetic.
  - Result: rejected as a TTFT mechanism. Keep the guarded code only as an
    experimental path; do not promote it.
- Chat-template prefix prefill:
  - Fixed the previous correctness bug where `LLAMA_CHAT_PREFIX_PREFILL=1`
    decoded the chat prefix at position zero without first evaluating the
    initial `embd_inp` tokens. The prefill path now evaluates initial tokens
    plus the reusable chat prefix in normal order and marks initial tokens
    consumed.
  - Recheck: immediate TTFT improved to `10.97s`, but `time_to_type_s`
    regressed to `17.13s` and the n36 canary still changed.
  - Follow-up logical-token fix keeps the full prompt token stream and only
    skips evaluation for the prefetched KV prefix. This avoids dropping prefix
    tokens from sampler/history state, but the n36 canary still changes:
    `time_to_type_s=16.21`, `interactive_ttft_s=12.89`,
    `eval_tokens_per_s=0.90`.
  - Result: rejected for the default path. Split-prefill changes first-token
    logits enough to change deterministic output on this model; do not promote
    until an exact-logit equivalence test passes.
- Exact CPU IQK up/gate + down many current recheck:
  - `GGML_MOE_PROMPT_UP_GATE_MANY=1` preserved the n36 canary and measured
    `time_to_type_s=10.55`, `interactive_ttft_s=12.77`,
    `eval_tokens_per_s=0.89`.
  - `GGML_MOE_PROMPT_UP_GATE_MANY=1 GGML_MOE_PROMPT_DOWN_MANY=1` preserved the
    n36 canary and measured `time_to_type_s=10.31`,
    `interactive_ttft_s=12.69`, `eval_tokens_per_s=0.89`, versus the same
    rebuild default at `time_to_type_s=10.56`, `interactive_ttft_s=13.13`,
    `eval_tokens_per_s=0.90`.
  - n84 validation measured `time_to_type_s=10.21`,
    `interactive_ttft_s=13.59`, `eval_tokens_per_s=0.92`.
  - Result: useful as a guarded diagnostic, but not a promoted improvement; the
    n84 run regressed TTFT versus the accepted default and missed the 5% token
    rate gate against the 0.98 baseline.
- CUDA prompt up/gate control recheck:
  - `GGML_MOE_STREAM_FUSED_UP_GATE=1 GGML_MOE_STREAM_PROMPT_UP_GATE=1`
    preserved the n36 canary and measured n84 `interactive_ttft_s=12.66`,
    `eval_tokens_per_s=0.95`, but this was not a prompt CUDA win.
  - Added `GGML_MOE_STREAM_DECLINE_DEBUG=1` to print machine-readable decline
    reasons from `ggml_cuda_moe_stream_up_gate_batch`.
  - The debug run showed `env=1` does not enable prompt mode:
    `reason=multirow_requires_prompt_mode rows_stride=20 type=22`. The run
    falls back to CPU prompt MoE and only exercises the decode CUDA path.
  - Result: keep `GGML_MOE_STREAM_PROMPT_UP_GATE=1` classified as a decode
    control, not as a prompt optimization. Use explicit `unsafe-q8-1` or
    `exact-q8-k*` values for prompt-mode experiments.
- Decode-control recheck:
  - Sequential n84 run with `GGML_MOE_STREAM_PROMPT_UP_GATE=1`:
    `time_to_type_s=10.63`, `interactive_ttft_s=12.40`,
    `eval_tokens_per_s=0.95`, cache allocation succeeded.
  - Sequential n36 canary run: canary passed, sha
    `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`,
    `interactive_ttft_s=12.78`.
  - Promotion check without harness-injected env reproduced the decode path but
    did not reproduce the win: n84 measured `interactive_ttft_s=13.24`, then
    `17.20` on rerun, with token rate still `0.95`. Trace comparison showed
    the regression came from CPU prompt MoE compute variance, not decode.
  - Result: not promoted. Keep `GGML_MOE_STREAM_PROMPT_UP_GATE=1` classified as
    a promising but unstable decode-control probe, not an accepted default.
- Exact IQ2 prompt CUDA current recheck:
  - `GGML_MOE_STREAM_PROMPT_UP_GATE=exact-q8-k-iq2-direct-probe` with
    `GGML_MOE_VRAM_CACHE_SAFETY_MIB=768` enters chunked exact prompt staging:
    `routes=160 cache_slots=122 chunk=53`.
  - The run completes but fails the n36 canary, regresses TTFT to `20.16s`,
    and drops `eval_tokens_per_s` to `0.85`.
  - Validation mode confirms full-tensor GPU/CPU mismatches remain in the
    prompt/decode up-gate path; examples include max_abs values above 10 on
    later layers. The single routed-row replay comparator is therefore
    insufficient as a promotion gate.
  - Result: exact IQ2 prompt CUDA remains rejected. The next exact-kernel work
    must first make validation compare the full prompt output within tolerance
    before any latency probe can be accepted.
- Exact IQ2 prompt CUDA after prompt-only isolation:
  - Added `GGML_MOE_STREAM_FUSED_UP_GATE_PROMPT_ONLY=1` so prompt experiments
    can force decode back to the CPU path, and
    `GGML_MOE_STREAM_FUSED_UP_GATE_VALIDATE_PROMPT_ONLY=1` so validation logs
    are not dominated by decode-side fused up/gate mismatches.
  - Direct IQ2 exact prompt, decode forced to CPU:
    `time_to_type_s=10.48`, `interactive_ttft_s=19.79`,
    `eval_tokens_per_s=0.42`, n36 canary failed.
  - Repacked IQ2 exact prompt, decode forced to CPU:
    `time_to_type_s=10.76`, `interactive_ttft_s=21.17`,
    `eval_tokens_per_s=0.42`, n36 canary failed.
  - Prompt-only validation confirms the full prompt tensor is close but not
    deterministic-output safe; later layers show drift such as
    `blk.72.ffn_up_exps.weight max_abs=0.0455503`.
  - Result: exact Q8_K prompt staging is no longer the next TTFT mechanism for
    this IQ2_S Wici target. Keep the prompt-only controls as diagnostics, but
    do not run further latency probes on this path unless an exact-logit
    equivalence test passes first.
- Default production path after adding the comparator and re-quarantining
  `IQ2_S exact-q8-k`:
  - n36 canary: pass, sha
    `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`
  - `time_to_type_s`: 10.09
  - `interactive_ttft_s`: 12.78
  - `prompt_eval_time_ms`: 12767.72
  - `eval_tokens_per_s`: 0.88 on the short 15-token canary sample
  - Result: production path preserved; no speed win accepted from exact CUDA
    prompt work yet.
- Default production path after the route-aware validation diagnostic:
  - n36 canary: pass, sha
    `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`
  - `time_to_type_s`: 10.39
  - `interactive_ttft_s`: 13.41
  - `prompt_eval_time_ms`: 13384.42
  - `eval_tokens_per_s`: 0.89 on the short 15-token canary sample
  - Result: production path still preserved; route diagnostics did not create
    an accepted speed change.
- Prompt-only controls rebuild check:
  - default n36 still preserves the canary after adding the prompt-only
    diagnostic envs: `time_to_type_s=10.58`, `interactive_ttft_s=12.96`,
    `eval_tokens_per_s=0.89`.
  - after-ready startup preload with 5s declared typing slack preserves the
    canary and reduces time-to-type to `8.56s`, but immediate-submission TTFT
    remains about `13.03s`. Keep this as a human-typing mode observation, not
    an accepted default TTFT win.
- Dynamic prompt expert scheduler:
  - Added opt-in `GGML_MOE_PROMPT_DYNAMIC_EXPERTS=1`.
  - Mechanism: prompt MoE workers now claim active experts from an atomic
    counter instead of static `ith, ith+nth, ...` striping. This preserves the
    existing IQK arithmetic and output layout while reducing per-thread
    imbalance in prompt up/gate and down.
  - n36 canary: pass, sha
    `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`,
    `interactive_ttft_s=11.58`.
  - n84 validation: `time_to_type_s=10.50`, `interactive_ttft_s=11.50`,
    `eval_tokens_per_s=0.95`.
  - two-run n84 confirmation: worst `interactive_ttft_s=11.78`, token rate
    `0.95`, cache allocation succeeded.
  - Post-promotion default-preset checks preserved the n36 canary and produced
    n84 runs at `11.29-12.18s`, but also saw outliers at `15.46s` and `19.36s`.
  - Result: promoted as the best validated default mechanism so far, but not
    considered a closed p95 win. The next work item is prompt MoE variance
    reduction using trace evidence, not another broad parameter sweep.

## Recent Rejected Candidate

Disabling prompt deferral with:

```sh
GGML_MOE_STREAM_DEFER=0
GGML_MOE_STREAM_BATCH_ONLY=0
```

was rejected. It reached the chat prompt but OOMed during CUDA graph launch on the first user prompt. This path is not accepted unless a later change creates enough VRAM headroom and passes the n84 validation plus n36 canary.

## TTFT Focus

The current TTFT is dominated by the first prompt eval after the user submits
input. Prioritize changes that reduce `prompt_eval_time_ms` and
`interactive_ttft_s`:

- Treat `cpu_prompt_upgate` and `cpu_prompt_down` as coarse buckets, not root
  causes. Before building another large optimization, split them into IQK
  compute time, mmap/page-fault time, hot-expert cache time, RAM bandwidth
  pressure, storage wait, and synchronization/barrier time.
- Try prompt GPU streaming only after the exact quant/dot path is identified.
  The Q8_1 prompt shortcut corrupted GLM-5.1 IQ3_XXS output, so prompt GPU work
  must preserve IQK semantics or be declared quality-changing before testing.
- A blunt `GGML_MOE_VRAM_PROFILE_PRELOAD_MAX_TENSORS` cap improves TTFT but
  currently drops n84 token rate too far; keep it as an experiment, not a default.
- Prefer one-run TTFT probes for candidates, then run n36 canary only for a
  candidate that improves TTFT materially.
- Keep n84 `eval_tokens_per_s` within 5% of the 0.98 historical baseline for
  latency candidates unless an experiment is explicitly marked quality-changing.
- Do not accept a TTFT win that regresses 64-token sustained token rate by more
  than 5%.

## Mechanism-First TTFT Plan

Stop broad parameter sweeping. The remaining TTFT is a first-prompt critical
path problem, not a CLI tuning problem. New work should change mechanisms only
when a trace explains which part of submit-to-first-token is being removed.

The workflow is:

1. Capture one interactive run with a full first-prompt trace:
   - route IDs by layer and tensor;
   - expert cache hit/miss and source tier;
   - expert read size, enqueue time, completion time, and sync wait;
   - CPU prompt MoE subspans: hot-expert lookup/copy, IQK kernel wall time,
     page faults, and barrier/synchronization waits;
   - CUDA graph capture/allocation events;
   - prompt eval wall time, first-token wall time, and generation token rate.
2. Build an offline critical-path model from that trace. It must first classify
   `cpu_prompt_*` as compute-bound, page/cache-bound, storage-bound, RAM
   bandwidth-bound, or synchronization-bound. Only then should it estimate saved
   TTFT per cached byte, per bulk-read group, or per moved compute kernel.
3. Validate only the model-selected plan with one n84 interactive run and one
   n36 canary. Do not test unrelated thread/batch/GPU-layer variants unless the
   trace points to that subsystem.
4. Promote the accepted preset into `presets/moe/groundtruth/<name>.*` so normal
   `moe-run.py --chat` can reuse it without recalibration.

Use:

```sh
python scripts/calibrate-wici-interactive-config.py --promote-name <host-model-profile-name>
```

For a new host/model, run the same script once. Do not run 10 samples unless investigating variance after a promising change.
Promotion refuses to overwrite an existing accepted preset unless `--replace-existing-promotion` is passed after reviewing the validation and canary results.
Startup-preload, cache-budget, and reserve tuning flags are disabled by default; enable them only as falsification probes after `mechanism_plan` names that subsystem.
Promotion now also has a small anti-outlier gate: `--promotion-confirm-runs`
defaults to 2 and reruns the selected validation path before writing a
groundtruth preset. Every confirmation run must pass the same TTFT, cache, and
token-rate gates. This is intentionally much cheaper than 10-run sampling but
prevents a single lucky TTFT sample from becoming the default.

The current calibrator remains useful for validation and promotion, but it must
not become the optimization strategy. Its tuning flags are diagnostic only and
are now opt-in. The default path captures one first-prompt trace, emits a
mechanism plan, validates the current mechanism-selected preset once, and runs
the canary. Use `--plan-only` on a new host/model when the immediate need is to
classify the next mechanism before spending time on validation. The next
calibrator upgrade should be a root-cause trace-to-plan optimizer:

- read the first-prompt trace and route CSV;
- compute per-layer/per-tensor CPU prompt MoE subspans;
- attach page-fault deltas and, where available, perf counters to
  `cpu_prompt_upgate` and `cpu_prompt_down`;
- score each expert row by first-use depth, reuse count, bytes, miss wait,
  source tier, and whether it lies before the first visible token;
- emit exactly one mechanism class: CPU RAM residency, exact GPU/IQK prompt
  kernel, cross-layer prefetch, graph/setup movement, or "trace still
  insufficient";
- output one concrete plan for that class, not a parameter sweep;
- run exactly that plan through validation and canary gates.

After the May 28 prompt-only exact-Q8_K recheck, the root-cause planner should
not pick exact chunked prompt CUDA for this IQ2_S target. If the trace again
shows CPU prompt MoE plus major faults over an oversized route set, the next
mechanism is `speculative_expert_prefetch_accuracy_probe`: first score whether
recent expert-prediction ideas can predict enough next-layer expert bytes before
first use to hide mmap faults. Only after that report passes should runtime
cross-layer page-cache or VRAM prefetch be implemented.

Initial analyzer result on the current TTFT trace:

```sh
python3 scripts/analyze-moe-route-predictability.py \
  bench/wici-glm51-interactive-latency/ttft/20260528-061439-trace-capture-n36-run01.ttft.csv \
  --budgets-mib 128,512,2048
```

- submit-to-first-token route events: 35,544
- unique routed expert rows: 17,091
- routed working set: 68.33 GiB
- perfect 2 GiB prefetch oracle byte recall: 2.9%
- adjacent expert-id byte recall: 30.3%
- recommendation: collect more traces before implementing runtime prefetch

This rules out larger static TTFT packs as the next mechanism. A real prefetch
win needs a learned or representation-derived predictor with holdout recall, not
another manually chosen cache budget.

Implemented trace artifacts:

- `GGML_MOE_TTFT_TRACE_OUT=<path>` writes per-expert cache/load events from the
  CUDA MoE cache path.
- `llama-cli` writes `mark_ready`, `mark_submit`, and `mark_first_token` into
  that trace when the symbol is available, so the exact submit-to-first-token
  window can be isolated.
- CPU prompt MoE emits `cpu_prompt_upgate` and `cpu_prompt_down` timing events
  into the same trace for prompt/prefill batches.
- Next required telemetry: split those CPU prompt MoE events into page/cache,
  hot-expert cache, IQK compute, and synchronization subevents. The current
  aggregate label is useful for finding the subsystem, but too coarse to justify
  a specific large optimization.
- `scripts/bench-wici-glm51-interactive-latency.py --ttft-trace-dir <remote-dir>`
  enables per-run trace files.
- The interactive harness now records `assistant_sha256` for every successful
  run and `n36_canary_ok` for the fixed seed-42 n36 prompt. Summaries report
  n36 canary pass/fail counts so mechanism probes cannot silently drift output.
- `scripts/analyze-wici-ttft-trace.py <trace.csv>` summarizes the
  submit-to-first-token window into JSON.
- `scripts/analyze-moe-route-predictability.py <trace.csv>` estimates whether
  bounded speculative expert prefetch is promising before runtime code is
  written. It reports routed working-set size, budgeted oracle recall, simple
  adjacent-layer expert-id recall, and whether more traces or model-based route
  prediction are required.
- `scripts/calibrate-wici-interactive-config.py` now captures a TTFT trace
  during the route-capture phase and writes `ttft_trace_summary` plus
  `mechanism_plan` into the calibration summary. This is the default path for a
  new host/model: classify the bottleneck, choose one mechanism, then validate
  once instead of sweeping parameters.
- The calibrator resolves repo-relative remote trace paths under `--remote-dir`
  and waits for the stderr `TTFT trace written` marker before reading the CSV.
  This prevents successful trace captures from being misclassified as
  `trace_missing` just because the JSONL stores a relative path.
- CPU prompt MoE subtrace events are now emitted with short CSV-safe op names:
  `cpu_up_group`, `cpu_up_prep`, `cpu_up_cuda_probe`, `cpu_up_compute`,
  `cpu_down_group`, `cpu_down_prep`, and `cpu_down_compute`. The analyzer and
  calibrator also tolerate the earlier truncated long names for old traces.
- CPU prompt compute spans also emit `cpu_up_minflt`, `cpu_up_majflt`,
  `cpu_down_minflt`, and `cpu_down_majflt`. These are fault counters, not
  milliseconds. If major faults appear in the compute span, the mechanism plan
  switches toward prompt-critical CPU residency or a TTFT pack instead of a pure
  grouped-compute kernel.
- CPU prompt grouping now emits exact `cpu_up_route`, `cpu_gate_route`, and
  `cpu_down_route` events into TTFT traces. The planner computes unique routed
  expert GiB before first token; if that route set is larger than a bounded CPU
  pack can hold on Wici, it rejects more CPU residency variants and now selects
  a speculative expert-prefetch accuracy probe instead of another parameter
  sweep or exact-Q8_K prompt latency probe.
- The current exact CUDA target is now explicit: add a row-batched prompt MoE
  path that quantizes prompt activations to Q8_K and computes the exact CPU
  repacked semantics for active prompt quant types. For `IQ2_S`, that means
  matching the CPU `IQ2_S -> Q8_K_R8` row-tile conversion first, not direct
  raw `IQ2_S x Q8_K`. The on-the-fly scalar emulation attempted on May 28
  still fails validation and is diagnostic-only. The existing CUDA prompt
  shortcut uses Q8_1 activations and remains diagnostic-only.
- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_AFTER_READY=1` can start profile preload
  after the readiness marker. `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_JOIN_ON_SUBMIT=0`
  makes that preload non-blocking so it uses human typing slack instead of
  moving preload time into immediate-submit TTFT. This is a probe only until
  n84 validation plus n36 canary pass.

The calibrator also measures a session baseline by default. This prevents
promoting a candidate that only looks good against stale historical numbers but
does not improve the current hardware/model/run state. Use
`--no-measure-session-baseline` only for a quick plumbing check.
When no tuned candidate beats the session baseline, the baseline run is reused
as the validation result instead of launching a duplicate validation run.
The GPU idle guard defaults to `memory.used <= 1536 MiB` and utilization <= 5%;
the previous 512 MiB memory threshold was too strict for the current Wici driver
baseline, which idles around 1071 MiB.
The interactive harness records runtime VRAM cache telemetry from stderr:
requested/actual cache MiB, free/total VRAM, clamp state, safety margin,
allocation count/slots, and `moe_vram_cache_failed`.
Pass `--summary <path>` to write a machine-readable p50/p95 JSON summary next
to the JSONL records; the summary includes latency, token-rate, and cache
telemetry aggregates.
The calibration driver now passes that summary path for every trace, baseline,
candidate, validation, and canary phase. The top-level calibration summary links
each phase JSONL and its aggregate summary so TTFT selection can be audited
without re-parsing raw runs.
For human-typing latency checks, pass `--post-ready-delay-s <seconds>` to the
harness or calibrator. Keep the default at zero for immediate-submit benchmark
gates; delayed-submit wins are accepted only for that explicitly declared user
typing mode.

Acceptance gates:

- Validation is required by default.
- Canary is required by default.
- `moe_vram_cache_failed` fails validation.
- Default calibration rejects candidates with `interactive_ttft_s > 13.56`,
  which is the 10% improvement threshold from the 15.07s historical baseline.
- Default calibration also rejects tuned latency candidates that improve
  `interactive_ttft_s` by less than 10% versus the same-session baseline.
- If the fastest validated latency candidate fails the n36 canary, calibration
  tries the next validated candidates in latency order before rejecting the run.
- Latency candidates must keep n84 `eval_tokens_per_s` within 5% of the accepted 0.98 baseline.
- Token-rate candidates must still beat the current 0.97 tok/s baseline unless a quality-changing experiment is declared.
- Latency changes should improve one-run `interactive_ttft_s` by at least 10%
  before canary. Re-run p95 sampling only after a candidate survives validation,
  canary, and token-rate gates.
- No accepted latency change may regress 64-token sustained tok/s by more than 5%.

## TTFT Probe Results

May 26 focused one-run probes on `wici`:

- `GGML_MOE_VRAM_CACHE_MIB=1792`: stable default, canary preserved, observed n84 TTFT about 14.7-15.4s and token rate about 0.96.
- `GGML_MOE_VRAM_CACHE_MIB=2048`: rejected, OOM at first CUDA graph launch.
- `--chat-active-prewarm`: TTFT about 12.1s, but time-to-type about 53s and n36 canary changed; rejected as a default.
- `GGML_MOE_STREAM_DEFER=0` with lower GPU layers/cache: rejected, TTFT and token rate both worse.
- `-b 1`: rejected, prompt submission hung past five minutes.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_MAX_TENSORS`: useful diagnostic; TTFT improved but n84 token rate dropped to about 0.90, so blunt preload caps are not acceptable.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_PROMPT=0`: TTFT improved to about 13.7-14.2s, but n84 token rate dropped to about 0.91-0.92; rejected as a default.
- `GGML_MOE_VRAM_PROFILE_SKIP_FIRST_PRELOADS`: preserving only later preloads did not improve TTFT enough and damaged cadence when set too low; keep diagnostic-only.
- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=3`: accepted TTFT-focused default; with the safer 1536 MiB cache, n84 TTFT 13.21s, n84 token rate 0.96, n36 canary preserved.
- `GGML_MOE_VRAM_CACHE_MIB=1536` with preload 3: accepted safer default, default n84 TTFT 12.82s, n84 token rate 0.96, cache allocation succeeded, n36 canary preserved.
- `GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1 GGML_MOE_VRAM_CACHE_SAFETY_MIB=128` with cache 1536: accepted fail-safe, default n84 TTFT 12.82s, n84 token rate 0.96, cache allocation succeeded, runtime log showed `clamp=1 safety=128`.
- `GGML_MOE_VRAM_CACHE_MIB=1792` with preload 3: rejected as a default after a May 27 run partially allocated the cache, reported `moe_vram_cache_failed`, and dropped n84 token rate to 0.44.
- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=6`: rejected, TTFT regressed to about 17.3s.
- Trace-forced generated preset with `N_GPU_LAYERS=61` and preload 4: rejected,
  n84 TTFT reached 12.75s and token rate 0.97, but n36 canary changed.
- Groundtruth preset with preload 4: rejected, n84 TTFT regressed to 13.51s.
- Session-baseline calibration check on May 27: baseline TTFT 12.95s, preload-3
  candidate TTFT 13.16s, improvement -1.56%; no candidate selected and the
  baseline run was reused as validation.
- Canary-aware calibration plumbing check on May 27: preload-3 n84 TTFT 12.89s,
  n84 token rate 0.97, n36 canary sha preserved. The summary records the canary
  attempt under `canary_attempts`.
- Opt-in cache-tuning plumbing check on May 27: `--tune-cache-budget
  --cache-budget-candidates 1536,1664` ran both n84 candidates and carried the
  selected cache into the summary. This was a mechanism check with relaxed TTFT
  gates, not an accepted config change; normal gates would reject the observed
  ~17s TTFT runs.
- Harness cache telemetry check on May 27: default n84 recorded
  `requested=1536`, `actual=1536`, `free=2343`, `safety=128`, `clamp=true`,
  `slots=372`, `allocated_gib=1.5`, and `moe_vram_cache_failed=false`.
- Harness summary artifact check on May 27: `--summary` wrote
  `wici_glm51_interactive_latency_summary_v1` JSON with p50/p95 latency,
  token-rate, cache clamp, cache failure count, and cache allocation stats.
- Calibration summary artifact check on May 27: a short n36 validation wrote
  `validate_summary` beside `validate_output`; TTFT was 12.72s with
  `moe_vram_cache_failed=false`. This was a relaxed plumbing check, not a new
  accepted token-rate result.
- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_BATCH_SYNC=1`: rejected. Enqueuing
  startup profile tensors and synchronizing once measured n84 TTFT 13.37s and
  n84 token rate 0.96, worse than the accepted default. The code remains
  opt-in diagnostic-only.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_PROMPT_ASYNC=1`: rejected. Async prompt-time
  profile preload measured n84 TTFT 17.23s, n84 token rate 0.95, and no cache
  failure. This suggests the naive overlap path increases I/O/CUDA contention
  before first generation.
- Default rebuild check after those gated diagnostics: n36 TTFT 12.74s,
  `moe_vram_cache_failed=false`, canary sha preserved.
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=20`: viable but not accepted as an
  improvement. n84 TTFT was 12.83s, token rate 0.94, and cache allocation
  succeeded. It stays inside token-rate gates but does not beat the accepted
  12.82s TTFT default.
- `GGML_MOE_VRAM_PROFILE_RESERVE_PCT=30`: rejected. n84 TTFT regressed to
  13.50s with token rate 0.95.
- Full route-first profile from the captured trace: rejected as a default. It
  improved n84 TTFT to 11.77s and time-to-type to 8.62s, but n84 token rate
  dropped to 0.81, well outside the regression gate.
- `LLAMA_CHAT_STARTUP_PROFILE` with the route-first profile, while keeping the
  accepted runtime profile: rejected. It preserved n84 token rate at 0.96 and
  reduced time-to-type to 8.67s, but TTFT regressed to 15.03s. The CUDA preload
  path still used runtime-profile expert rows for the chosen startup tensor
  names, so this was not a true exact-row startup preload.
- `LLAMA_CHAT_STARTUP_PROFILE_EXPERT_ROWS=1` with the route-first profile:
  rejected. Exact route-first startup rows preserved n84 token rate at 0.96 but
  regressed TTFT to 15.46s.
- `LLAMA_CHAT_STARTUP_PROFILE_EXPERT_ROWS=1` with the accepted profile:
  rejected. It preserved token rate at 0.97 but regressed n84 TTFT to 13.27s.
- Default rebuild check after the exact-row startup gate: n36 canary sha
  preserved and `moe_vram_cache_failed=false`.
- `GGML_MOE_VRAM_PROFILE_PROMPT=<route-first-profile>` with evictable prompt
  preloads: useful but not accepted. n84 TTFT improved to 12.50s, but token
  rate fell to 0.90 and failed the latency token-rate gate.
- Adding `GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1` lets the accepted protected
  runtime profile replace temporary prompt entries. This recovered n84 token
  rate to 0.95 with TTFT 12.76s, but the TTFT gain is too small to accept as a
  new default.
- Adding `GGML_MOE_VRAM_PROFILE_PROMPT_MAX_ENTRIES=24` with preload eviction
  measured n84 TTFT 12.81s and token rate 0.96, effectively neutral versus the
  accepted default.
- Default rebuild check after the prompt-profile gates: n36 canary sha
  preserved and `moe_vram_cache_failed=false`.
- `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_AFTER_READY=1`: rejected for
  immediate-submit TTFT. The join-before-processing version improves
  time-to-type to 8.77s and preserves n84 token rate at 0.97, but immediate n84
  TTFT regressed to 15.12s because the preload wait moves into submit-to-token
  time.
- Same after-ready preload with `--post-ready-delay-s 2`: rejected. The
  join-before-processing version fixed the earlier 19.48s TTFT outlier, but n84
  token rate was 0.91 and missed the gate.
- Same after-ready preload with `--post-ready-delay-s 3`: useful for explicitly
  declared human-typing mode, not accepted as the immediate-submit default. n84
  time-to-type was 8.85s, TTFT 13.05s, token rate 0.96, cache failure false;
  n36 canary sha was preserved.
- Same after-ready preload with `--post-ready-delay-s 5`: also useful for
  declared human-typing mode. n84 time-to-type was 8.76s, TTFT 12.68s, token
  rate 0.95, cache failure false; n36 canary sha was preserved.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_PROMPT=0` on the current accepted preset:
  rejected as a default. n84 TTFT improved to 12.49s, but token rate fell to
  0.90 and failed the latency gate.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_PROMPT_MAX_TENSORS=3`: rejected. n84 TTFT was
  13.14s and token rate 0.90.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_PROMPT_MAX_TENSORS=12`: rejected. n84 TTFT was
  12.64s but token rate 0.89.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_PROMPT_MAX_TENSORS=12` with
  `GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1`: rejected. Token rate recovered to
  0.94, but n84 TTFT regressed to 13.03s.
- `N_GPU_LAYERS=61` on the accepted preset: rejected. n84 token rate was 0.97,
  but TTFT regressed to 17.43s.
- `N_GPU_LAYERS=59` on the accepted preset: rejected. n84 token rate was 0.97,
  but TTFT regressed to 13.06s.
- `--chat-threads-batch 16`: rejected. n84 TTFT regressed to 14.23s and token
  rate was only 0.93.
- `--chat-threads-batch 32`: not accepted. One n84 run looked neutral, but a
  rerun was neutral and the n36 canary timing was poor. This is not a mechanism
  win and should not be extended into more thread sweeps.
- TTFT trace implementation smoke test, `bench/ttft-trace-call-scope-n36.jsonl`:
  n36 completed with `interactive_ttft_s=12.61`, cache allocation succeeded, and
  the trace included submit/first-token markers plus CUDA expert-load events.
- CPU prompt trace, `bench/ttft-trace-cpu-prompt-n36.jsonl`: this run was slow
  (`interactive_ttft_s=17.34`) but it exposed the useful bottleneck split. In
  the submit-to-first-token window, `cpu_prompt_upgate` accounted for about
  9.37s, `cpu_prompt_down` about 4.15s, CUDA preload/runtime expert loads about
  0.44s total, and decode CUDA calls about 0.13s. The next TTFT mechanism should
  therefore target prompt/prefill MoE execution, not decode cache parameters.
- `GGML_MOE_STREAM_PROMPT_UP_GATE=unsafe-q8-1`: rejected for production. The
  prototype fixed the prompt row mapping and larger-column MMQ dispatch, but the
  Q8_1 CUDA prompt path does not match the IQK prompt dot path closely enough
  for GLM-5.1 IQ3_XXS. Validation produced corrupt text and large per-layer
  mismatches, so prompt CUDA up/gate now requires the explicit
  `unsafe-q8-1` value and remains diagnostic-only. The exact target is an
  IQK-compatible row-batched prompt MoE path, not the current Q8_1 shortcut.
- `GGML_MOE_STREAM_FUSED_UP_GATE_PROMPT_ONLY=1` restricts the fused CUDA
  up/gate probe to prompt batches. `GGML_MOE_STREAM_FUSED_UP_GATE_VALIDATE_PROMPT_ONLY=1`
  restricts validation comparison to prompt batches. These are diagnostic
  controls for isolating prompt TTFT work from decode-side fused-path behavior.
- Default correctness restored after the diagnostic CUDA edits:
  `bench/default-canary-after-src1-restore-n36.jsonl` produced the expected
  n36 text, sha `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`,
  `interactive_ttft_s=13.07`, and `eval_tokens_per_s=0.91`.
- Mechanism-planner trace check on May 27:
  `20260527-201342-trace-capture-n36-run01.ttft.csv` showed
  `submit_to_first_token_ms=12982`, with `cpu_prompt_upgate=7374ms`,
  `cpu_prompt_down=2847ms`, expert preload/runtime load only `407ms`, and decode
  CUDA calls only `100ms`. The generated plan is `prompt_moe_compute` ->
  `exact_row_batched_prompt_moe`.
- A later run under Wici background VRAM pressure produced the same mechanism
  classification from `20260527-202203-trace-capture-n36-run01.ttft.csv`:
  `submit_to_first_token_ms=17520`, `prompt_moe_cpu=14773ms`,
  `cuda_expert_load=379ms`. Do not use that slow run as a baseline; use it only
  as confirmation that the root cause remains prompt MoE compute when cache
  movement is small.
- Passive subtrace check after the short-name fix:
  `bench/ttft-subtrace-shortnames-safety512-n36.jsonl` preserved the n36 sha
  `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`.
  The trace `ttft-subtrace-shortnames-safety512-n36-run01.ttft.csv` measured
  `submit_to_first_token_ms=13098`, `prompt_moe_cpu=9951ms`,
  `cpu_prompt_compute=9692ms`, `cpu_prompt_prep=224ms`,
  and `cuda_expert_load=357ms`. This proves the next mechanism is exact
  row-batched prompt MoE, not residency, preload, cache-size, thread, or
  GPU-layer tuning.
- Exact CPU IQK up/gate-many prototype:
  `GGML_MOE_PROMPT_UP_GATE_MANY=1` preserved the n36 sha under
  `GGML_MOE_VRAM_CACHE_SAFETY_MIB=512` and improved one n36 TTFT from 13.108s
  to 12.629s, but the subtrace still showed about 9.69s of prompt MoE compute.
  Keep it guarded until an n84 run proves a material same-session TTFT win.
- `GGML_CUDA_EAGER_CUBLAS=1` plus `GGML_MOE_VRAM_CACHE_SAFETY_MIB=256` and
  `GGML_MOE_PROMPT_UP_GATE_MANY=1` preserved the n36 sha under the current
  Wici background VRAM load, with TTFT 13.049s and token rate 0.89. Treat it as
  a resource-headroom workaround, not a performance win.
- Nonblocking after-ready preload probe under the same safety256/eager resource
  workaround preserved the n36 sha and improved time-to-type to 8.45s, but TTFT
  was 13.34s and token rate 0.89. The matching join-on-submit control measured
  time-to-type 8.63s, TTFT 13.48s, and token rate 0.88. This is useful only as a
  human-typing slack probe; it is not an accepted immediate-submit TTFT win.
- Plan-only workflow verification:
  `python scripts/calibrate-wici-interactive-config.py --plan-only` captured one
  n36 interactive trace under safety256/eager, preserved the canary text, and
  emitted `prompt_moe_compute -> exact_row_batched_prompt_moe`. The trace
  measured `submit_to_first_token_ms=12826`, `prompt_moe_cpu=10037ms`,
  `cpu_prompt_compute=9719ms`, and `cuda_expert_load=342ms`.
- Fault-counter plan-only verification after the rusage trace upgrade preserved
  the canary and changed the mechanism classification to
  `prompt_critical_residency_or_ttft_pack`: `submit_to_first_token_ms=12704`,
  `cpu_prompt_compute=9825ms`, `major_faults=112962`, and
  `minor_faults=1025725`. The compute label was hiding page-cache misses, so
  simple grouped arithmetic is no longer the first implementation target.
- `GGML_MOE_PREFETCH=1 GGML_MOE_PREFETCH_MODE=readahead`: rejected. It
  preserved the n36 sha, but TTFT regressed to 13.06s and token rate dropped to
  0.85; the trace showed compute decreased while prefetch/preload increased.
  Readahead hints shifted the fault cost instead of removing it.
- `LLAMA_CHAT_STARTUP_PROFILE_CPU_HOTEXP=1` with
  `GGML_MOE_PROMPT_UP_GATE_HOTEXP=1` and `GGML_HOTEXP_CACHE_GB=4`: rejected.
  It preserved the n36 sha but TTFT regressed to 20.46s. The startup profile
  loaded only 24 expert rows from 3 tensors, too little to cover the first
  prompt, while the extra RAM cache reduced VRAM headroom. A real TTFT pack must
  be route-critical and bounded, not a generic hot-expert cache preload.
- Exact TTFT-route CPU residency probe:
  `LLAMA_CHAT_STARTUP_PROFILE=<ttft.csv>`,
  `LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_EXPERT_ROWS=512`, and
  `GGML_HOTEXP_INSERT_ON_MISS=0` preserved the n36 sha but regressed TTFT to
  16.88s and token rate to 0.86. The trace still showed about 108k major faults.
  A later planner pass measured about 68.3 GiB of unique first-token CPU prompt
  expert rows, so CPU residency is not the right next mechanism on the 15 GiB
  Wici host.
- `GGML_MOE_VRAM_PROFILE_PRELOAD_PROMPT=0` under safety256/eager preserved the
  n36 sha and measured TTFT 12.73s, but token rate fell to 0.84. It is rejected
  as a latency candidate because it does not materially improve TTFT and worsens
  the already-low constrained-VRAM token rate.
- `LLAMA_CHAT_PREFIX_PREFILL=1`: rejected by canary. It exactly prefills 4
  first-user chat-template prefix tokens before the ready marker, reducing n36
  TTFT to 11.76s and prompt route volume from about 68.3 GiB to 55.0 GiB, but
  the assistant text changed and the n36 sha became
  `bc10dcd0bce27de77ef97ee5b101b7ebed38625698e5b1589f556c2f6c35f29f`.
  Splitting prompt evaluation changes enough numerics for GLM-5.1 IQ3_XXS that
  prefix prefill is not acceptable unless declared quality-changing.
- `GGML_MOE_PROMPT_DOWN_MANY=1`: rejected as a TTFT candidate. It preserved the
  n36 canary under the safety256/eager resource workaround, but measured
  `interactive_ttft_s=13.28` and `eval_tokens_per_s=0.89`; the trace still
  showed about 121k major faults. The paired
  `GGML_MOE_PROMPT_UP_GATE_MANY=1 GGML_MOE_PROMPT_DOWN_MANY=1` probe also
  preserved the canary but regressed to `interactive_ttft_s=13.37` and
  `eval_tokens_per_s=0.86`. Keep both as guarded diagnostics, not defaults.
- Short-prompt decode-as-prefill falsification with `-b 4`: rejected. It
  regressed n36 TTFT to `29.52s`, dropped prompt eval throughput to
  `0.68 tok/s`, and changed the canary text. Do not extend this into batch-size
  sweeping.
- Current default without the safety256/eager workaround still fails under the
  present Wici VRAM state: n36 reached `time_to_type_s=10.56` but exited with
  return code 250 before the second ready marker after a partial/clamped cache
  allocation. Use safety256/eager for constrained-VRAM mechanism probes until
  graph/CUBLAS headroom is fixed.
- Harness canary-field verification under safety256/eager preserved the n36 sha
  with `interactive_ttft_s=13.10`, `eval_tokens_per_s=0.91`, and
  `n36_canary_ok=true`.
- Fresh end-to-end calibration captures at 20:29 and 20:31 failed before the
  second ready marker because Wici's background node service occupied about
  1.2 GiB of VRAM and CUDA graph launch/CUBLAS initialization ran out of
  headroom. Those are machine-load failures, not accepted performance results.

Next TTFT work should be narrower than the earlier token-rate exploration:
do not start with another preload/cache scheduler. First split the CPU prompt
MoE bucket. If page faults or cold expert memory dominate, build prompt-critical
CPU RAM residency or a TTFT-specific expert pack. If IQK arithmetic dominates,
build an exact IQK-compatible row-batched prompt kernel. If synchronization
dominates, restructure prompt MoE scheduling. A candidate only graduates to
token-rate work after it improves same-session TTFT and preserves the canary.

## Recent Breakthroughs Only

Use 2025-2026 work as the research base. As of 2026-05-28, the hard cutoff for
new mechanism choices is 2024-05-28. Older systems are historical context only
and should not drive the implementation plan unless a newer paper or runtime
revalidates the same mechanism on modern MoE inference.

Relevant recent mechanisms:

- KTransformers SOSP 2025, `https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf`:
  recent CPU/GPU hybrid MoE work reinforces the same direction as the trace:
  route-aware expert placement, CPU/GPU overlap, and row-batched prompt MoE
  matter more than blind CLI tuning.
- DeepGEMM, `https://github.com/deepseek-ai/DeepGEMM`: grouped GEMM is the
  right shape for prompt rows once the exact quant/dot path is available. Do
  not use it as a shortcut that changes GLM-5.1 IQ3_XXS numerics.
- FlashInfer MLSys 2025, `https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html`:
  use modern fused/batched serving kernels where they preserve exact behavior,
  especially for grouped prefill work.
- FlexInfer MLSys 2025, `https://proceedings.mlsys.org/paper_files/paper/2025/file/698cfaf72a208aef2e78bcac55b74328-Paper-Conference.pdf`:
  phase-specific CPU/GPU execution policy matters on constrained GPUs.
  Translation: prefill and decode should be optimized separately; TTFT work can
  choose a different expert placement policy than steady-state token generation.
- Tutti, `https://arxiv.org/abs/2605.03375`: SSD-backed cache work points to
  objectized transfers, GPU-side/low-CPU-overhead I/O submission, and slack-aware
  scheduling. Translation for MoE: represent expert rows as prefetchable
  objects, batch them by physical locality, and avoid I/O that contends with
  prompt eval. Current Wici traces show prompt MoE compute dominates before KV
  movement, so this is secondary until exact prompt MoE batching lands.
- DualPath, `https://arxiv.org/abs/2602.21548`: storage bandwidth imbalance is
  now a first-class inference bottleneck. Translation: profile whether Wici is
  NVMe/PCIe/CPU-submit limited during first prompt before changing compute
  settings.
- SP-MoE, `https://arxiv.org/abs/2510.10302`: SD-aware MoE offloading combines
  speculative expert prefetch, cutoff-layer policies, async prefetch threads,
  batched I/O, and an analytical latency model. Translation: implement
  prediction/cutoff for expert prefetch before revisiting speculative decoding.
- MoE-SpeQ, `https://arxiv.org/abs/2511.14102`: uses cheap on-device prediction
  to foresee future experts and overlap host-memory expert movement with useful
  compute. Translation: add a small expert-route predictor or gate-derived
  predictor for prompt/decode, then prefetch only high-confidence experts.
- LayerScope/PreScope, `https://arxiv.org/abs/2509.23638`: combines
  layer-aware expert prediction, cross-layer scheduling, and async I/O to remove
  wait bubbles on commodity servers. Translation: schedule across layers, not
  tensor-by-tensor in route CSV order.
- HybriMoE, `https://huggingface.co/papers/2504.05897`: uses dynamic CPU/GPU
  scheduling, impact-driven inter-layer prefetching, and score-based caching.
  Translation: some low-impact or late experts may be cheaper to keep on CPU
  while high-impact early experts get scarce GPU cache slots.
- Local Routing Consistency, `https://huggingface.co/papers/2505.16056`: expert
  offloading effectiveness depends on segment-level routing locality.
  Translation: measure GLM-5.1 local routing consistency before assuming any
  cache policy can generalize across prompts.
- LMCache, `https://arxiv.org/abs/2510.09665`: production cache systems expose
  first-class control APIs for pinning, lookup, cleanup, movement, compression,
  and cross-tier orchestration. Translation: replace env-var-only cache control
  with explicit runtime APIs for protected generation entries, temporary
  prompt-critical entries, and demotion after first token.
Mechanism priority:

1. **Exact CUDA Q8_K prompt MoE.** Current traces show CPU prompt MoE plus page
   faults dominate submit-to-first-token, and the first-prompt route set is
   about 68 GiB, too large for Wici RAM residency. Build an IQK-compatible CUDA
   prompt up/gate and down path for `matrix_row_counts[e] > 1` using Q8_K
   prompt activations and every CPU-reference representation seen in the trace.
   `IQ2_S` is not a raw-dot target when prompt rows exceed the CPU threshold:
   the CPU fused path repacks it to Q8_K_R8 first, so direct `IQ2_S x Q8_K`
   is rejected for canary-preserving work. The current Q8_1 CUDA shortcut is
   diagnostic-only and rejected for production.
2. **CPU prompt MoE root-cause split.** While building the exact path, expand
   tracing enough to avoid optimizing the wrong subcomponent: measure page
   faults, hot-expert cache time, IQK compute time, RAM bandwidth pressure,
   storage wait, and barriers inside `cpu_prompt_upgate` and `cpu_prompt_down`.
3. **Prompt-critical CPU RAM residency / TTFT pack.** If page/cache or storage
   movement dominates, copy the first-token-critical expert slices into a
   contiguous CPU RAM pack and run the exact existing IQK CPU path from that
   pack. This preserves numerics and attacks mmap/page-fault latency without
   relying on unsafe Q8_1 CUDA prompt math.
4. **Short-prompt decode-as-prefill probe.** For small prompts, test whether
   forcing prompt tokens through the exact single-token streamed decode path
   beats CPU prompt MoE while preserving the n36 canary. This is a falsification
   probe, not a default policy and not the same as broad `-b 1` sweeping.
5. **Critical-path planner.** Keep the new `mechanism_plan` output as the
   standard new-hardware/new-model mechanism selector. It should choose prompt
   MoE, expert movement, graph setup, or mixed tracing before any validation run.
6. **Routing-locality profiler.** Compute segment routing consistency for
   GLM-5.1 on the target prompts. If locality is low, cache policy alone will
   not deliver a large TTFT win.
7. **Predictive expert planner.** Build a layer-aware predictor/cutoff policy
   from route traces and gate outputs only if the root-cause split shows hidden
   expert movement rather than pure arithmetic. It should pick only experts
   likely to be needed before first visible token.
8. **Cross-layer scheduler.** Schedule expert movement globally across layers
   using estimated compute slack and load cost. Avoid naive prompt-time preload,
   which already measured worse.
9. **Two-tier expert residency.** Separate protected generation entries from
   temporary prompt-critical entries. Demote prompt entries after first token.
10. **Batched async expert I/O.** Convert scattered expert reads into fewer
   object-style reads with completion telemetry. Use CPU async I/O first; only
   consider GPUDirect-style work after the scheduler proves I/O is the bottleneck.
11. **Phase-specific CPU/GPU policy.** Allow prefill and decode to make different
   CPU/GPU placement choices, preserving the n36 canary and n84 token-rate gate.
12. **Graph/warmup separation.** Move CUDA graph capture/allocation and cache
   slot initialization out of submit-to-first-token without generating text or
   changing canary output.

Immediate implementation target from the current traces:

1. Keep the restored default n36 canary as the correctness gate:
   `6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9`.
2. Do not promote the current exact prompt CUDA path: after the transient-cache
   fix it runs, but it still fails the n36 canary and regresses TTFT. The next
   implementation target is a CPU prompt MoE reduction that preserves the
   existing IQK path: reduce per-layer route grouping/copy overhead, or build a
   native CPU many-route prompt kernel that keeps current numerics.
3. Keep CPU prompt MoE subtracing for `cpu_prompt_upgate` and
   `cpu_prompt_down`: page faults, hot-expert cache, IQK kernel time, and
   barriers. Use it to prove the new prompt kernel reduces the traced
   `cpu_up_compute` bucket before any cache/thread tuning.
4. Keep the existing decode CUDA path and canary gates unchanged. Do not accept
   Q8_1 prompt approximations or prefix-prefill prompt splitting unless a
   quality-changing experiment is explicitly declared before testing.

Systematic configuration mechanism:

1. Run the TTFT mechanism planner first, not a parameter sweep. It classifies
   traces by prompt MoE compute, expert movement, graph/setup, and decode work.
2. For each selected mechanism, run one canary-bearing probe plus targeted
   validation. If the canary or CPU/GPU validator fails, reject the mechanism
   before any timing comparison.
3. Use `scripts/compare-wici-interactive-candidate.py` to compare the selected
   candidate against the session baseline. It records the acceptance gates,
   traced prompt-compute delta, token-rate delta, and the next mechanism when a
   candidate fails. This is the replacement for ad hoc parameter sweeps.
4. Apply the recent-mechanism cutoff in the planner output. New mechanism
   candidates must be based on work from 2024-05-28 or newer, or on a modern
   runtime that independently revalidates the idea.
5. Derive cache budgets from available VRAM and measured graph/cache allocation,
   then reserve enough headroom for the selected mechanism. Do not sweep cache
   safety values as a tuning surface.
6. Only after the mechanism is correct, compare against the current accepted
   baseline with one cold and one warm sample. Ten-run sampling is deferred until
   there is a plausible win.

## Next Performance Direction

The current token rate is bounded by per-token expert movement from NVMe/RAM through the streamed MoE path. Cheap prompt-stream knobs are exhausted or rejected. The next significant token-rate work should prioritize:

1. row-batched streaming MoE correctness;
2. runtime-cost-calibrated expert cache selection;
3. slack-aware expert read scheduling;
4. MTP or explicit layer-skipped self-speculation only after verifier correctness is in place.
