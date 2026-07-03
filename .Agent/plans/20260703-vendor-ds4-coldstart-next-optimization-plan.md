# Vendor DeepSeek 16GB Cold-Start 下一阶段优化计划

## Summary

本计划从当前已 push 的 vendor DeepSeek cold-start 复现状态继续推进。最终结果必须体现在 `vendor` 框架，`ik_llama` 只能作为参考。

当前事实：

- 历史最高观测：`4.2 tok/s`，run `/root/lfz/runs/vendor-ds4-16gb/20260702T133103Z-20260702_pushed_onepack_odirect_sota_rerun/france-cpu40-vram0gb`。
- 2026-07-03 post-rollback strict cold guard 已复现 `4.2 tok/s`，run `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`。
- 2026-07-03/2026-07-04 多次 strict cold rerun 达到 `4.1-4.2 tok/s`，均满足：16GB cgroup 含 page cache、France 正确率、TTFT gate、O_DIRECT expert pack、VRAM cache counters。
- 2026-07-04 all-output ngram-simple 和 server speculative partial-serial fallback 均已验证并拒绝：前者会卡在同一 verification position，后者可保证正确推进但只有 `3.7 tok/s`。相关临时源码均已回退，当前有效 SOTA 仍是 `4.2 tok/s`。
- 2026-07-04 top512 CPU blocking-touch prewarm 首跑观测到 `4.3 tok/s`，但从已 push source 清洁 rebuild 后只复现 `4.2 tok/s`，未超过当前 SOTA；该源码已回退，当前 head 为接受路径。
- 2026-07-04 no-drop diagnostic 可达 `7.2 tok/s`，但不是 accepted cold-start，因为没有执行全局 `drop_caches`；它只能说明冷启动主要损失来自 CPU up/down fallback 的 page/source stall，不可作为 SOTA。
- 2026-07-04 target-only `llama-lookahead` 已拒绝：两次 strict cold 机械诊断都在第一 token 后因 DeepSeek4 coupled-sequence/KV 路径失败，且 gate cache hit rate 从接受路径的 `86-87%` 塌到约 `11.5%`。
- 2026-07-04 no-draft `ngram-mod` 已拒绝：strict cold run 正确率/RAM/TTFT 通过，但 `decoded speed=2.879 tok/s`，低于当前 `4.2 tok/s` SOTA；probe source 已回退，accepted `llama-cli` hash 恢复。
- 2026-07-04 gate cache headroom audit 已完成：`GGML_MOE_STREAM_ONE_CACHE_MIB=13312` 在 strict cold 16GB cgroup 下达到 `4.1 tok/s`，gate hit rate 仍为 `86.8%`，CUDA free 从约 `238MiB` 增至约 `492MiB`。该结果不是新 SOTA，但说明可以继续做一个很小的 down-cache 组合短诊断。
- 2026-07-04 gate cache `13312MiB` + down cache `256MiB` + MXFP4 batch-probe 短诊断已拒绝：RAM 通过，但输出错误、`eval_tok_s=1.6`、down cache hit rate `0.0%`、gate hit rate 降到 `58.5%`、stage 增至 `5.922 ms/call`。probe source 已回退，不允许 full strict cold。
- 2026-07-04 corrected down-batch compare 已完成：恢复 `tmp_dst_rows=max(dst_cols,n_active)` 后，MXFP4 down batch 与 CPU compare 数值一致（`max_abs_max=1.1920929e-07`），但性能仍只有 `2.6 tok/s`，down-cache hit rate 约 `0.0%`，stage `4.740 ms/call`，因此性能方向拒绝，probe source 已回退。
- 2026-07-04 CPU fallback top128 O_DIRECT staging 已拒绝：正确率/RAM/TTFT 通过，但同步 direct staging 读取 `44.79GB`，`eval_tok_s=2.9`，说明同步 O_DIRECT 不是可用 movement model。
- 当前最新计划：回到 accepted SOTA runtime，尝试一个更高性价比的 gate one-stream VRAM cache prefill 诊断。该候选只移动已接受 gate path 的 source-load 时间，不改模型数学；如果不能超过 `4.2 tok/s`，立即回退。
- 下一阶段目标：稳定超过 `4.2 tok/s`；未超过 `4.2 tok/s` 的结果只能作为 diagnostic/rejected/tie，不得 promote。
- 所有符合要求的新 SOTA 必须立刻记录完整复现信息并 push 到 `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`。记录必须足以未来从 push 后源码、profile、pack、runner 参数和 run artifact 完整复现。

## Current Baseline

- `current_pushed_head_before_this_update`: `b1168731e` (`vendor-ds4: reject top128 direct staging probe`)，已 push 到 `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`。
- `accepted_runtime_source_head`: code path restored at `5484a1806` (`vendor-ds4: reject cpu prewarm touch repro`); later pushed commits are docs/artifact updates unless explicitly stated as promoted source.
- `runtime_binary_build`: accepted `llama-cli` hash remains `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`; source-level accepted runtime behavior is the post-rollback SOTA path.
- `runtime_source_note`: all-output/server-speculative probes, CPU prewarm touch candidate, down batch, compact mmap, and other rejected source probes were reverted before final accepted runtime state. Committed heads include records/rejected artifacts/plan updates; runtime source is back on the accepted SOTA path.
- `binary_sha256`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`。
- `libggml_cpu_sha256`: `a6a3ea2d52fd8001716b56bb2703b686438d485253779079eb7f728494541f2a` after rollback rebuild。
- `pack_sha256`: `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076` for `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`。
- `profile_sha256`: `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b` for `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`。
- `updown_decode_top512_profile_sha256`: `45c35b2cb00faa4b37ad793e5e63e6008b68235064f5fdeb23a7e843c437084e` for `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top512.tsv`。
- `model_file`: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`，`ls/du` both report `146G`。因此 cold-start 方案不得依赖整模型热 page cache；所有 page cache 都必须计入 16GB cgroup。
- `accepted_config`: `cpu_moe=40`, `--vram-cache-gb 0`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`, strict cold `drop_caches`, 16GB `systemd-run` cgroup with `MemorySwapMax=0`, CLI extra args `-c 256 -b 16 -ub 16 -t 20 -tb 20`。

Fresh strict cold reproduction evidence:

| Run | eval_tok_s | prompt_tok_s | TTFT ms | memory_peak | memory_file | correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T031923Z-20260703T031710Z-current-sota-strict-repro/france-cpu40-vram0gb` | 4.1 | 1.5 | 30470.663075 | 16000000000 | 15102509056 | true |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T032154Z-20260703T032138Z-current-sota-strict-repro-retry/france-cpu40-vram0gb` | 4.1 | 1.6 | 28857.563383 | 16000000000 | 15102316544 | true |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb` | 4.2 | 1.6 | 28014.740620 | 16000000000 | 15096049664 | true |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T172115Z-20260704_phaseA_clean_sota_guard_head4be08352/france-cpu40-vram0gb` | 4.1 | 1.5 | 29484.204486 | 16000000000 | 15101456384 | true |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T180612Z-20260704_cpu_prewarm_top512_touch_pushed_repro/france-cpu40-vram0gb` | 4.2 | 1.7 | 32212.962507 | 16000000000 | 15105830912 | true |

These runs had `oom=0`, `oom_kill=0`, `ram_limit_killed=false`, one expert pack `hits=4623 misses=0 direct_reads=4623 direct_failures=0`, and VRAM cache `hits=30528 misses=4623 hit_rate=86.8%` with `3192` slots (`13.2GiB`).

Post-rollback guard France answer:

> France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.

## Bottleneck And Optimization Direction

Current measured bottleneck after O_DIRECT:

- Gate O_DIRECT pack path is no longer dominant.
- 2026-07-04 current-SOTA profile: gate one-stream path total is about `7.0s`, including source load about `5.2s`; VRAM gate cache hit rate remains `86.8%`.
- CPU up/down fallback remains the main target: profile total about `25.7s`, decode about `18.1s`, with up/down totals roughly `12.4s/13.4s`.
- No-drop same-prompt diagnostic improves to `7.2 tok/s` and cuts fallback total by about `50%`, proving cold page/source stalls are important. It is not accepted because it relies on global warm page cache outside the strict cold-start rule.
- Hard upper bounds from the refreshed profile: top512 up/down residency only reaches about `5.05 tok/s`; eliminating all CPU up/down fallback reaches about `8.86 tok/s`; eliminating CPU fallback plus all one-stream source load reaches about `13.36 tok/s`. Therefore a path to `10 tok/s` likely needs either multi-token speculative/MTP or a combined fallback+source elimination, not a small hotset tweak alone.
- Large buffered down pack is rejected because it competes with model mmap/page cache and raises refault pressure under the 16GB cgroup.
- Broad gate cache, global thread sweeps, down batch, one-stream all-up/down, compact mmap, and CPU prewarm touch are deprioritized or rejected unless a new hard-bound analysis shows a materially better path.

Latest optimization direction after the 2026-07-04 rejected probes and rollback:

1. Treat `b1168731e` as the current pushed documentation/source baseline before this plan update, while treating the accepted runtime behavior as the post-rollback path restored at `5484a1806`. Before a new source change, verify the worktree state and whether any probe source is still unaccepted.
2. Do not promote CPU prewarm touch. It tied at `4.2 tok/s` after pushed-source reproducibility and increased TTFT versus the accepted SOTA, so it remains rejected diagnostic evidence.
3. External draft/internal MTP is currently not viable: DS4 GGUF has no `mtp`, `draft`, `eagle`, `spec`, or `next` tensors, and local model inventory has no tokenizer-compatible small DS4 draft model.
4. No speculative diagnostic is currently active. no-draft `ngram-mod`, ngram-simple, server partial fallback, and target-only lookahead are all rejected for this path.
5. The latest narrow gate-cache/down-cache composition test is rejected. The next active work is correctness-first diagnosis for down/MXFP4 offload: use CPU compare traces or an equivalent deterministic harness to locate the numerical/output divergence before any more down-batch performance runs.
6. Do not resume pure compact-mmap, broad up/down hotset, large down-batch staging, no-filter one-stream, ngram-simple, lookahead, ngram-mod, or page-touch prewarm sweeps unless the plan is updated with a new bottleneck measurement and a better theoretical upper bound.
7. Stop a candidate immediately if gate cache hit rate drops materially, expert pack direct fallbacks appear, RAM exceeds 16GB including page cache, TTFT rises more than 20% for an accepted result, or the France output is incomplete/incoherent.

## Execution Plan

### Phase 0: Source/Remote Guard

- Confirm local head is at least `4e525a91f` and `ssd/vendor/deepseek-token-rate-16gb` points to the same or newer committed plan state.
- If `examples/speculative-simple/speculative-simple.cpp` is dirty, repair it first; no speculative source probe is accepted or active.
- If `ggml/src/ggml-cuda/moe_stream_batch.cu` is dirty, it must be treated as the active down-cache/MXFP4 batch-probe only; do not commit it unless it produces a compliant new SOTA and passes pushed-source reproduction.
- Confirm the accepted SOTA runtime binary hash is still `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62` after rollback rebuild.
- If source or binary does not match the accepted state, stop and repair reproducibility before any optimization.

### Phase 1: Speculative/MTP Feasibility

- Inspect the DS4 GGUF tensor list for `mtp`, `draft`, `eagle`, `spec`, and related head tensors. Record counts and exact tensor names.
- Search local model directories for possible draft models with matching or compatible tokenizer/vocab.
- For each candidate draft path, calculate before running:
  - model size and host/VRAM footprint under the 16GB cgroup and current gate cache;
  - expected draft cost per token;
  - required acceptance rate to beat `4.2 tok/s` and to approach `10 tok/s`;
  - whether the implementation verifies target logits without changing final output correctness.
- If no compatible draft/MTP path exists, record that explicitly and do not spend more time on ngram-simple unless a new acceptance mechanism is designed.

### Phase 2: Short Speculative Diagnostic

- Only run if Phase 1 finds a compatible draft model or internal MTP path.
- Start with a short France diagnostic under strict 16GB cgroup and cold `drop_caches`.
- Record acceptance rate, target forward count, draft forward count, TTFT, RAM/page cache, exact output, and whether target verification preserves correctness.
- Reject immediately if output diverges, TTFT exceeds the 20% gate for an accepted result, or effective throughput is below the current SOTA class.

### Phase 3: Combined Fallback/Source Design If Speculative Is Not Viable

- Use existing 2026-07-04 trace artifacts to separate CPU fallback compute from page/source stalls more precisely before coding.
- Any new CPU/GPU fallback candidate must calculate a hard upper bound from measured removable time. A candidate whose bound is near `4.2 tok/s` is not worth implementation.
- Preserve the accepted gate cache (`13568MiB`, `3192` slots, `86-87%` hit rate) unless the plan explicitly proves that sacrificing slots is compensated by a larger measured saving.
- Do not repeat rejected page-touch/top-N sweeps; top512 touch prewarm has already tied after pushed-source repro.

### Phase 4: SOTA Promotion Protocol

- If a candidate exceeds `4.2 tok/s` and passes RAM, correctness, TTFT, pack, and VRAM-cache gates, stop exploration immediately.
- Record exact run path, command/env, source head, build metadata, binary hashes, model/profile/pack hashes, memory stats including page cache, counters, TTFT, token rates, and France output.
- Commit and push source, plan, profiles, and artifacts to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`.
- Clean rebuild from the pushed source and rerun strict cold. Promote only if the pushed-source rerun still exceeds `4.2 tok/s` and passes every gate.
- If pushed-source rerun ties/regresses or violates a gate, revert runtime source to the accepted path and keep only rejected artifacts/docs.

## Execution Log

### 2026-07-03 Phase 0 Baseline Guard

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T033918Z-20260703T033849Z-next-plan-phase0-baseline-guard/france-cpu40-vram0gb`
- Result: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=28719.624610 ms`
- RAM: `memory_peak_bytes=16000000000`, `memory_file_bytes=15103021056`, `pgmajfault=271001`, `workingset_refault_file=1663190`, `ram_ok=true`, `ram_limit_killed=false`
- Correctness: `correctness_ok=true`
- Counters: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`
- Conclusion: baseline reproducibility passed; proceed to trace/profile.

### 2026-07-03 Phase 1 Trace/Profile Refresh

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T034133Z-20260703T034116Z-next-plan-phase1-trace-profile/france-cpu40-vram0gb`
- Result: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=28721.512209 ms`
- RAM: `memory_peak_bytes=16000000000`, `memory_file_bytes=15097171968`, `pgmajfault=274312`, `workingset_refault_file=1741084`, `ram_ok=true`, `ram_limit_killed=false`
- Correctness: `correctness_ok=true`
- One-stream trace: rows `35151`, `src0_ms=5264.715`, `src1_ms=159.082`, `kernel_ms=333.396`, `d2h_ms=103.189`, `sync_ms=1033.283`, `scatter_ms=44.167`, `dontneed_ms=79.170`, `total_ms=7020.208`, `span_ms=44008.047`, hits `30528`, inserts `3337`, misses `4623`
- CPU fallback profile: entries `7030`, total `24838.624 ms`, prompt `7460.501 ms`, decode `17378.123 ms`, up `11740.949 ms`, down `13097.675 ms`, decode up `8625.536 ms`, decode down `8752.587 ms`
- Decode hotset coverage estimate:
  - top64: `2340.235 ms`, payload `0.266 GiB`, calls `6382`
  - top128: `3695.326 ms`, payload `0.531 GiB`, calls `10051`
  - top256: `5379.372 ms`, payload `1.062 GiB`, calls `13423`
- Conclusion: bottleneck is still CPU up/down fallback. Gate pack/O_DIRECT path is stable and should not be disturbed.

### 2026-07-03 Phase 2/3 CPU Fallback Pack Mmap Probe

Theoretical reason: top128/top256 up/down fallback entries cover about `3.7s/5.4s` of measured decode fallback time. If replacing scattered GGUF mmaps with a compact mmap pack eliminated enough page/refault overhead without displacing critical page cache, expected upper bound was roughly current eval time minus covered overhead. Because the covered time is not all IO and CPU math remains unchanged, this direction required strict measurement and could only be accepted if it exceeded `4.2 tok/s`.

Artifacts:

- top128 fake trace: `/root/lfz/runs/vendor-ds4-16gb/20260703T034116Z-next-plan-phase1-hotsets/decode_top128_updown_fake_trace.csv`, sha256 `cc45241abe534228f4efc3b552a5e9878b2bcc9795a5ff3bd21eb3e774b98c1b`
- top128 meta: `/root/lfz/runs/vendor-ds4-16gb/20260703T034116Z-next-plan-phase1-hotsets/decode_top128_updown_meta.tsv`, sha256 `10ed2d629249bae235f55c960aee5ba9323aa3a33a2cd6c9ea35e5be5ae07729`
- top128 pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top128-updown-20260703.pack`, sha256 `f57ff2426647514c0145bb4b367b750f7a1753837b2ead22df091e9645483894`, size `545M`
- top256 fake trace: `/root/lfz/runs/vendor-ds4-16gb/20260703T034116Z-next-plan-phase1-hotsets/decode_top256_updown_fake_trace.csv`, sha256 `4b1ec8bd32096a7457aaedc67f22791138f2bb33fcaf9189c93be5289e2be700`
- top256 meta: `/root/lfz/runs/vendor-ds4-16gb/20260703T034116Z-next-plan-phase1-hotsets/decode_top256_updown_meta.tsv`, sha256 `6cb3df88c2b174524a2ec660ca6e23bef3ac2afeaad5aa782b15443a84dd34d8`
- top256 pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top256-updown-20260703.pack`, sha256 `a6eb88346b2a4719f85db804b4c6bf71ad58ec6ca3719c4292d8c9a11767161d`, size `1.1G`

Initial existing-path test:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T034653Z-20260703T034637Z-top128-cpu-fallback-pack-mmap/france-cpu40-vram0gb`
- Result: `4.1 tok/s`, RAM/correctness passed, but `kimi_cpu_fallback_pack_mmap hits=0 misses=35880`
- Root cause: this DS4 build has `GGML_CUDA_MOE_STREAM_BATCH=OFF`, so CUDA helper `ggml_cuda_moe_expert_pack_mmap_ptr()` is a stub and returns `nullptr`.

Temporary source probe:

- Implemented a narrow default-off local CPU fallback pack mmap path in `ggml/src/ggml-cpu/ggml-cpu.c` so `GGML_MOE_CPU_FALLBACK_PACK_MMAP=1` can mmap `GGML_MOE_EXPERT_PACK` directly without depending on the batch TU helper.
- Rebuilt successfully with `cmake --build build-ds4-moe-stream -j 16`.
- Default-off guard run `/root/lfz/runs/vendor-ds4-16gb/20260703T035423Z-20260703T035406Z-local-cpu-pack-default-off-guard/france-cpu40-vram0gb`: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30113.714119 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15095259136`, `correctness_ok=true`.

Measured candidates:

| Candidate | Run | eval_tok_s | prompt_tok_s | TTFT ms | memory_peak | memory_file | correctness | mmap counters | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| top128 local mmap | `/root/lfz/runs/vendor-ds4-16gb/20260703T035604Z-20260703T035544Z-top128-local-cpu-fallback-pack-mmap/france-cpu40-vram0gb` | 4.1 | 1.6 | 28948.869196 | 16000000000 | 15097528320 | true | `hits=10051 misses=25829 bytes=44791758848 fallback_gguf=25829` | rejected/tie |
| top256 local mmap | `/root/lfz/runs/vendor-ds4-16gb/20260703T035839Z-20260703T035815Z-top256-local-cpu-fallback-pack-mmap/france-cpu40-vram0gb` | 4.0 | 1.5 | 28267.070684 | 16000000000 | 15103459328 | true | `hits=13423 misses=22457 bytes=59818901504 fallback_gguf=22457` | rejected/regression |

Gap analysis:

- The mmap path mechanically worked: top128/top256 showed nonzero hits and no correctness/RAM violations.
- It did not improve token rate because it only changes pointer/source layout for CPU fallback reads. The main fallback cost still includes CPU math and synchronization; reduced source locality alone was not enough.
- top256 likely increased page-cache/refault pressure enough to lose throughput despite higher coverage.
- This direction should be deprioritized unless paired with a real compute/offload change.

Rollback:

- Temporary source patch was reverted with `git restore ggml/src/ggml-cpu/ggml-cpu.c`.
- Rebuilt successfully with `cmake --build build-ds4-moe-stream -j 16`.
- Post-rollback guard run `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`: `eval_tok_s=4.2`, `prompt_tok_s=1.6`, `TTFT=28014.740620 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15096049664`, `pgmajfault=280239`, `workingset_refault_file=1674043`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- Counters: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.

### Next Step After Rejected Mmap Probe

Do not continue with pure CPU fallback compact mmap as the primary optimization. The next candidate must reduce actual fallback compute or move a high-coverage subset onto GPU without breaking the 16GB host RAM/page-cache budget:

1. Use the Phase 1 profile to design a small down-only or up/down CUDA-offload hotset with explicit VRAM accounting.
2. Before coding, calculate the upper bound from covered fallback time and required VRAM. Reject designs that cannot plausibly exceed `4.2 tok/s`.
3. Start with a narrow default-off path and a tiny top64/top128 subset; check MXFP4 correctness on France before expanding.
4. Keep current gate O_DIRECT pack untouched during the first compute/offload probe.

### 2026-07-03 Batch-On / MXFP4 Down Cache Probe

Design:

- Goal: test whether the existing `moe_stream_batch.cu` implementation can now be compiled and used for a real compute/offload experiment, then check whether a small down-batch VRAM cache can reduce CPU fallback enough to beat `4.2 tok/s`.
- Theoretical upper bound: Phase 1 measured down fallback `13097.675 ms` total (`8752.587 ms` decode + `4345.088 ms` prompt). If down batch were correct, cache-resident, and staging were cheap, this is the maximum removable component. In practice, any benefit must pay for GPU staging, quantization, D2H/scatter, reduced gate cache, and synchronization.
- VRAM budget test: reduce gate one-stream cache from `13568MiB` to `12288MiB` to free about `1.25GiB`, then request `GGML_MOE_VRAM_CACHE_MIB=1536` for batch/down cache. This is intentionally a short diagnostic first because reducing gate slots risks increasing direct pack reads.

Batch-on build probe:

- Build dir: `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream-batch-probe`
- Configure: same CUDA/MoE stream build with `GGML_CUDA_MOE_STREAM_BATCH=ON`
- Result: build now succeeds on source head `2723926b0`; old missing `iqk/iqk_mul_mat.h` failure is no longer present.
- Clean batch-probe hashes after rollback: `llama-cli=866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`, `libggml-cuda.so.0.10.0=0265f38feea12270583ebe2a5c633091650ef4b037776bfd44f92ad916656ec8`.

Batch-on down-only diagnostic without MXFP4 enablement:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T041435Z-20260703T041435Z-batch-on-down-short-diagnostic/france-cpu40-vram0gb`
- Config: batch-probe binary, current accepted gate O_DIRECT env, `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_MOE_STREAM_DECLINE_DEBUG=1`, short `-n 16`, strict cold 16GB cgroup.
- Result: `eval_tok_s=3.0`, `prompt_tok_s=1.5`, `TTFT=28704.939681 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15144034304`, `ram_ok=true`.
- Correctness: intentionally too short for the normal heuristic; output was a valid prefix: `France, officially the French Republic, ...`.
- Counters: down batch declined with `reason=unsupported_type type=39`; type `39` is MXFP4. Batch implementation is reachable, but DS4 down cannot use it without MXFP4 gates.

Temporary MXFP4 source probe:

- Temporary changes in `ggml/src/ggml-cuda/moe_stream_batch.cu`:
  - add `GGML_TYPE_MXFP4` to `moe_stream_type_supported()`;
  - add `GGML_TYPE_MXFP4` to the compact MMVQ launcher type switch;
  - restore the previous correctness fix by allocating compact dst workspace with `dst_tmp_rows=max(dst_cols,n_active)` in both up/gate and down paths.
- This patch was never committed and was reverted after the rejected probes.

Short compare before compact launcher gate fix:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T041815Z-20260703T041815Z-mxfp4-down-batch-cache1536-short-compare/france-cpu40-vram0gb`
- Result: `eval_tok_s=2.4`, `prompt_tok_s=1.5`, `TTFT=30765.751509 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15138635776`, `ram_ok=true`, `correctness_ok=true`.
- Compare: `rows=32`, `max_abs=9.53674316e-07`, `mean_abs_avg=2.1451e-08`, `max_rel=2.34157307e-05`.
- Counters: `batch_accept=0`, `batch_decline=1360`; batch cache allocated only `1.3GiB` after requested `1.5GiB` failed; gate one-stream hit rate dropped to `69.0%`.
- Diagnosis: MXFP4 passed the outer type gate but compact launcher still rejected it.

Short accepted-probe after compact launcher gate fix:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T042041Z-20260703T042041Z-mxfp4-down-batch-cache1536-short-accepted-probe/france-cpu40-vram0gb`
- Result: `eval_tok_s=2.4`, `prompt_tok_s=1.5`, `TTFT=31736.282097 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15155052544`, `ram_ok=true`, `correctness_ok=true`.
- Counters:
  - down batch accepted `1244` calls and declined `116`;
  - batch/down cache requested `1536MiB`, `cudaMalloc 1.5GiB FAILED`, fallback allocation `1.3GiB`, `315` slots, `hits=2130 misses=1960 hit_rate=52.1%`;
  - batch profile `calls=1244 avg_active=3.29 stage=4.538 ms quant=0.001 ms kernel=0.035 ms d2h=0.006 ms scatter=0.008 ms total=4.588 ms/call`;
  - one-stream gate cache fell to `hits=6527 misses=2941 hit_rate=68.9%`;
  - one expert pack `hits=2926 misses=15 direct_failures=0`.
- Gap analysis:
  - The actual CUDA kernel is cheap (`~0.035 ms/call`), but staging dominates (`~4.538 ms/call`).
  - Reducing gate cache to free down-cache VRAM destroys too much of the accepted gate cache behavior.
  - Even with down batch accepted, the short run is `2.4 tok/s`, far below the `4.2 tok/s` promotion line. A full SOTA run is not justified.
- Verdict: rejected; do not promote and do not commit source.
- Rollback: `git restore ggml/src/ggml-cuda/moe_stream_batch.cu`, rebuild `build-ds4-moe-stream-batch-probe`; `git status` clean after rollback.

Next direction after this rejection:

1. Do not trade large gate cache capacity for down batch unless staging is fixed first.
2. If revisiting down compute, focus on eliminating staging overhead, not just increasing cache slots. The measured kernel time is already small; the bottleneck is host-to-device staging/cache insertion.
3. A plausible future design needs either true async prefetch overlap ahead of the down op, or a compact hotset that is loaded before decode without displacing gate cache. It must first show `stage ms/call` dropping materially in a short diagnostic before running full France.

### 2026-07-03 Current Down Overlap / Top128 Pack Probe

Design:

- Goal: test the next proposed staging fix directly. The prior accepted-probe showed down batch kernel time was tiny (`0.035 ms/call`) but stage time was dominant (`4.538 ms/call`). This probe keeps the same short diagnostic shape, but adds current-token down overlap and a small top128 up/down pack.
- Hypothesis: `GGML_MOE_CURRENT_DOWN_OVERLAP=1` can prefetch current-token down experts during the up/gate CUDA window; `GGML_MOE_EXPERT_PACK=ds4-france-decode-top128-updown-20260703.pack` plus `GGML_MOE_IO_BACKEND=iouring` can make those prefetched copies cheaper than scattered GGUF mmap staging. A useful signal must show `current down overlap` counters and a material drop in down `stage ms/call`.
- Risk: the current overlap hook is inside the fused up/gate batch path. If this model path does not invoke `ggml_cuda_moe_stream_up_gate_batch()`, overlap will not run. The top128 pack may also cover too few down experts to move staging.

Temporary source probe:

- Same temporary MXFP4 patch as the previous down-cache probe:
  - add `GGML_TYPE_MXFP4` to `moe_stream_type_supported()`;
  - add `GGML_TYPE_MXFP4` to the compact MMVQ launcher type switch;
  - use `dst_tmp_rows=max(dst_cols,n_active)` for compact dst workspace.
- This patch was not committed and was reverted after the probe.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T042727Z-20260703T042727Z-mxfp4-current-down-overlap-top128-short/france-cpu40-vram0gb`
- Config delta from prior short accepted-probe: `GGML_MOE_CURRENT_DOWN_OVERLAP=1`, `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top128-updown-20260703.pack`, `GGML_MOE_IO_BACKEND=iouring`, `GGML_MOE_STAGE_PINNED=1`, `GGML_MOE_STAGE_PINNED_SLOTS=4`.
- Top128 pack sha256: `f57ff2426647514c0145bb4b367b750f7a1753837b2ead22df091e9645483894`, size `545M`.

Result:

- `eval_tok_s=2.4`, `prompt_tok_s=1.5`, `TTFT=30268.960184 ms`
- `memory_peak_bytes=16000000000`, `memory_file_bytes=15118184448`, `pgmajfault=152072`, `workingset_refault_file=40603`, `ram_ok=true`, `ram_limit_killed=false`
- `correctness_ok=true`; answer prefix remained semantic and coherent.

Counters and diagnosis:

- No `[moe_stream_batch] current down overlap ...` report appeared, so the overlap hook did not run. This confirms the current model execution path does not enter the fused up/gate batch trigger needed by `GGML_MOE_CURRENT_DOWN_OVERLAP`.
- Down batch still accepted `1244` calls and declined `116`, but stage did not improve: `stage=4.777 ms/call`, worse than the previous `4.538 ms/call`.
- Top128 pack coverage was too low for this run: expert pack `hits=126 misses=1834`, with `iouring_reads=0`; most copies still used GGUF mmap source.
- Pinned staging report: `copies=1960`, `waits=1956`, `slot_wait=4.205 ms`, `host_stage=5706.089 ms`, `h2d=334.868 ms`. The bottleneck is host staging/page source time, not GPU copy or kernel.
- Gate behavior remained degraded by the reduced one-stream cache: gate VRAM cache `hits=6527 misses=2941 hit_rate=68.9%`.

Verdict:

- Rejected. This did not meet the plan requirement of proving staging can be reduced before running a full SOTA candidate.
- Rollback completed with `git restore ggml/src/ggml-cuda/moe_stream_batch.cu`; `build-ds4-moe-stream-batch-probe` rebuilt clean. Clean hashes after rollback: `llama-cli=866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`, `libggml-cuda.so.0.10.0=eef7caaf00861bdae63de2f9629a8bd6bf8114b9819e4134b149399a8bd18716`.

Next direction after this rejection:

1. Do not rely on the current fused up/gate overlap hook for DeepSeek unless a separate diagnostic first proves `ggml_cuda_moe_stream_up_gate_batch()` is entered.
2. A useful staging fix now needs to target the actual down runtime load path directly, or provide a much more accurate down hotset/trace prefetch path with high pack coverage.
3. Before another full run, run a short diagnostic that either reduces `host_stage`/`stage ms/call` directly, or shows pack hit coverage high enough to justify the VRAM/page-cache tradeoff.

### 2026-07-03 Down-Only Top512 Pack Staging Probe

Design:

- Goal: target the actual down runtime load path directly. The top128 up/down pack only hit `126/1960` runtime load attempts, so it was too sparse to judge whether compact pack reads can reduce staging. Build a down-only hotset so the same 4.25MiB slots are used only for `ffn_down_exps`.
- Hotset source: Phase 1 `fallback-profile.csv`, filtered to `phase=decode` and `tensor contains ffn_down_exps`, sorted by `fallback_us`.
- Coverage calculation:
  - total decode down fallback: `8752.587 ms`, `2627` entries;
  - top512 selected: `5319.314 ms`, `10623` calls, `2.125 GiB` payload, `60.8%` of decode down fallback.
- Theory: if pack hits increase materially and direct pack reads avoid scattered GGUF source faults, down `stage ms/call` should fall. A useful candidate must still overcome the known cost of reduced gate cache, where one-stream gate hit rate falls from `86.8%` to about `68.9%`.

Artifacts:

- fake trace: `/root/lfz/runs/vendor-ds4-16gb/20260703T042727Z-next-down-only-hotsets/decode_top512_down_fake_trace.csv`, sha256 `ed823324723ffb17f5759cd206072d3bf66ace70c8ad96f7ae19d7feefc4b7bc`
- meta: `/root/lfz/runs/vendor-ds4-16gb/20260703T042727Z-next-down-only-hotsets/decode_top512_down_meta.tsv`, sha256 `4a0db23c4e2bf56bcc19cd4288435161d67508a54d1f2e263625818adc058131`
- pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top512-down-20260703.pack`, sha256 `a046bd214a543d721f6410c2adbbef642b705730c45aa48ac78c260df954ca70`, size `2.2G`

Temporary source probe:

- Same temporary MXFP4 patch as previous down batch probes; not committed and reverted afterward.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T043352Z-20260703T043352Z-mxfp4-down-top512-pack-short/france-cpu40-vram0gb`
- Config delta from previous short accepted-probe: `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top512-down-20260703.pack`, `GGML_MOE_IO_BACKEND=iouring`, `GGML_MOE_STAGE_PINNED=1`, `GGML_MOE_STAGE_PINNED_SLOTS=4`.

Result:

- `eval_tok_s=2.7`, `prompt_tok_s=1.5`, `TTFT=31010.975401 ms`
- `memory_peak_bytes=16000000000`, `memory_file_bytes=15140425728`, `pgmajfault=149271`, `workingset_refault_file=22156`, `ram_ok=true`, `ram_limit_killed=false`
- `correctness_ok=true`; answer prefix remained semantic and coherent.

Counters:

- expert pack `hits=673 misses=1287`, improving coverage from top128's `126/1960` but still leaving most loads on GGUF mmap source.
- down batch `batch_accept=1244 batch_decline=116`, same as prior short accepted-probe.
- batch/down cache: `315` slots, `hits=2130 misses=1960 hit_rate=52.1%`.
- down profile improved but not enough: `stage=3.705 ms/call`, `kernel=0.032 ms/call`, `total=3.751 ms/call`; previous short accepted-probe was `stage=4.538 ms/call`, `total=4.588 ms/call`.
- pinned staging: `copies=1960 waits=1956 slot_wait=3.978 ms host_stage=4374.053 ms h2d=331.064 ms`.
- gate one-stream remained degraded: `hits=6527 misses=2941 hit_rate=68.9%`.

Verdict:

- Rejected for SOTA. The better down-only pack proves pack coverage can reduce staging, but the measured short-run token rate is only `2.7 tok/s`, far below `4.2 tok/s`. The remaining stage cost plus degraded gate cache make a full SOTA run unjustified.
- Do not continue by merely increasing pack size unless the design also prevents gate cache loss or removes runtime host staging more aggressively.
- Rollback completed with `git restore ggml/src/ggml-cuda/moe_stream_batch.cu`; `build-ds4-moe-stream-batch-probe` rebuilt clean. Clean hashes after rollback: `llama-cli=866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`, `libggml-cuda.so.0.10.0=d20153144f5d0e19d6956086752a43f1f2d68719fb20af3ccab62c75a3be2801`.

Next direction after this rejection:

1. The main unsolved constraint is VRAM allocation: down batch needs cache/staging space, but taking that from one-stream gate cache destroys the accepted SOTA path.
2. A future candidate must either preserve the `13568MiB` gate cache and find additional VRAM elsewhere, or reduce gate cache loss by sharing/partitioning slots without lowering gate hit rate.
3. Another pack-size sweep alone is low priority; top512 already reduced stage but did not approach the promotion threshold.

### 2026-07-03 Context-Size VRAM Free Probe

Design:

- Goal: test whether reducing context from `-c 256` to `-c 128` frees enough VRAM to add a down batch cache without stealing capacity from the accepted `13568MiB` gate one-stream cache.
- Theory: if context/KV/graph buffers are a meaningful part of the remaining VRAM pressure, `-c 128` should increase free VRAM. If the accepted path's pressure is dominated by model weights plus the gate cache and allocator overhead, context reduction will not help.
- This probe uses the clean accepted O_DIRECT SOTA binary and no source changes.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T043852Z-20260703T043851Z-odirect-sota-c128-vram-free-probe/france-cpu40-vram0gb`
- Config delta from accepted SOTA: `-c 128`; gate O_DIRECT config unchanged, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, strict cold 16GB cgroup.

Result:

- `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=29441.246676 ms`
- `memory_peak_bytes=16000000000`, `memory_file_bytes=15089229824`, `ram_ok=true`, `ram_limit_killed=false`
- `correctness_ok=true`; France answer was semantic and coherent.
- Counters stayed aligned with accepted SOTA: one expert pack `hits=4623 misses=0`, gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- GPU samples showed max used/free about `31873/237 MiB`, effectively the same as `-c 256`; context reduction does not free useful VRAM for down cache.

Verdict:

- Rejected/tie. It preserves correctness and RAM behavior but does not exceed `4.2 tok/s` and does not create additional VRAM budget.
- Next VRAM work should focus on the large unaccounted allocation/cache footprint, not context-size trimming.

### 2026-07-03 Tiny Down Cache While Preserving Gate Cache Probe

Design:

- Goal: verify whether any down-batch improvement is possible while preserving the accepted `13568MiB` gate one-stream cache.
- Theory: if a small down cache can be allocated in the remaining free VRAM, it might capture a narrow hotset and reduce staging without lowering gate hit rate. The required signal is nonzero down-cache hits and lower `stage ms/call`; otherwise the remaining free VRAM is not enough for this direction.
- Config keeps gate cache at `13568MiB`, requests only `GGML_MOE_VRAM_CACHE_MIB=256`, and uses the down-only top512 pack. This is a short diagnostic, not a SOTA candidate.
- Temporary MXFP4 batch patch was used only for measurement and was not committed.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T044340Z-20260703T044340Z-tiny-down-cache256-keep-gate13568-short/france-cpu40-vram0gb`
- Config delta: batch-probe binary, `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_VRAM_CACHE_MIB=256`, `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top512-down-20260703.pack`, `GGML_MOE_IO_BACKEND=iouring`, `GGML_MOE_STAGE_PINNED=1`, `GGML_MOE_STAGE_PINNED_SLOTS=4`, short `-n 32`.

Result:

- `eval_tok_s=2.4`, `prompt_tok_s=1.5`, `TTFT=30438.814584 ms`
- `memory_peak_bytes=16000000000`, `memory_file_bytes=15137615872`, `pgmajfault=148222`, `workingset_refault_file=22637`, `ram_ok=true`, `ram_limit_killed=false`
- `correctness_ok=true` for a short prefix; output: `Here is a short paragraph introducing France: France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and`

Counters and diagnosis:

- Gate cache requested and initialized as accepted: `13.2GiB`, `3192` slots.
- Remaining GPU free was too small: down cache requested `256MiB`, but `cudaMalloc 0.2GiB FAILED`; fallback allocation was only `0.1GiB`, `34` slots.
- Down cache was ineffective: `hits=0 misses=4090 hit_rate=0.0%`.
- Down batch accepted `1244` calls, but profile regressed to `stage=5.243 ms/call`, `kernel=0.031 ms/call`, `total=5.288 ms/call`; pinned staging reported `host_stage=6225.021 ms`, `h2d=695.276 ms`.
- Down top512 pack hits were `2221 misses=1869`, but iouring counters stayed zero; the path did not convert pack coverage into a useful cache hit path.
- One-stream gate counters in this short run were `hits=6527 misses=2941 hit_rate=68.9%`; this matches the earlier short down-batch probes and remains far from a SOTA signal.

Verdict:

- Rejected. Preserving the accepted gate cache leaves too little usable VRAM for an independent down cache, and the tiny fallback allocation has zero hit rate.
- Rollback completed: `git restore ggml/src/ggml-cuda/moe_stream_batch.cu`; clean rebuild of `build-ds4-moe-stream-batch-probe` succeeded. Clean hashes after rollback: `build-ds4-moe-stream/bin/llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `build-ds4-moe-stream-batch-probe/bin/llama-cli=866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`.

Next direction after this rejection:

1. Do not spend more time on independent down-cache sizing until the VRAM budget problem is solved. With the accepted gate cache intact, a useful down cache cannot allocate.
2. First locate and classify the roughly `14.7GiB` unaccounted CUDA memory shown by `common_memory_breakdown_print`; determine how much is the one-stream gate cache, CUDA allocator reserve, libraries/context, pack/batch staging, or fragmentation.
3. If the unaccounted memory is mostly gate cache, design a shared/partitioned cache that keeps gate hit rate near `86.8%` while reserving a small down hotset. This needs a theoretical slot-level plan before source changes.
4. If the unaccounted memory includes avoidable allocator reserve or duplicate buffers, reclaim that first and rerun a strict cold SOTA guard before testing down batch again.
5. Any next source candidate must first show either nonzero down-cache hit rate without reducing gate hit rate, or direct reduction of `stage ms/call` below the top512 probe's `3.705 ms/call`; otherwise reject at short diagnostic stage.

### 2026-07-03 One-Stream No-Filter Up/Down Diagnostic

Design:

- Goal: test whether the existing one-stream path can improve up/down fallback without adding an independent down batch cache.
- Theory: removing `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps` lets the same one-stream GPU path accept gate, up, and down experts. If CPU up/down compute were the only dominant cost, this should reduce CPU fallback time. The risk is that one-stream single-expert GPU execution is staging/synchronization heavy and that up/down calls pollute the gate cache.
- Source state: clean accepted SOTA binary; no source change. The gate admission profile was kept, so only current gate profile entries are admitted to the long-lived cache, while up/down still execute through the GPU single path without cache insertion.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T044912Z-20260703T044912Z-one-stream-all-no-filter-gate-admit-short/france-cpu40-vram0gb`
- Config delta: no `GGML_MOE_STREAM_ONE_NAME_FILTER`, keep `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, keep gate admission profile and O_DIRECT gate pack, short `-n 32`, strict cold 16GB cgroup.

Result:

- `eval_tok_s=1.7`, `prompt_tok_s=1.0`, `TTFT=34927.553761 ms`
- `memory_peak_bytes=16000000000`, `memory_file_bytes=15148027904`, `pgmajfault=5568`, `workingset_refault_file=33746`, `ram_ok=true`, `ram_limit_killed=false`
- `correctness_ok=true` for a short prefix; output: `Here is a short paragraph introducing France: France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and`

Counters and diagnosis:

- One-stream accepted all single calls: `single_accept=19863 single_decline=0`, so the no-filter path did remove CPU fallback for those ops.
- It was much slower than CPU fallback: `cuda_single=7.556 ms/call`, `fallback_t0=0.001 ms/call`, down total `7.624 ms/call`.
- Cache behavior collapsed: one-stream cache `hits=6511 misses=13352 hit_rate=32.8%`; the accepted SOTA gate-only run has `hits=30528 misses=4623 hit_rate=86.8%`.
- Gate pack reads changed to `hits=2923 misses=10429`, confirming that unrestricted up/down traffic substantially disturbed the accepted gate path.
- VRAM shape stayed the same as accepted SOTA: `free=236MiB`, `model=17362MiB`, `unaccounted=14510MiB`.

Verdict:

- Rejected. Full one-stream GPU coverage of up/down is not viable; it replaces CPU fallback with a slower staging/sync path and damages gate cache locality.
- Do not pursue a broad multi-filter one-stream patch. A future source candidate would need a selective early-return mechanism that only sends an op to one-stream when the expected GPU path is cheaper than CPU fallback. Current evidence says down single is not cheaper.

Next direction after this rejection:

1. Deprioritize one-stream up/down compute. The existing one-stream kernel path is suitable for the accepted gate stream, not broad up/down replacement.
2. Focus next on CPU fallback itself: profile the CPU up/down fallback inner loop by type/layer and check whether thread scheduling, row routing, or repeated conversion/barrier work can be reduced without changing math.
3. Any CPU-side source candidate must be default-off, preserve exact France correctness, and show a measurable reduction in `fallback_t0` or name-profile totals before a full SOTA run.

### 2026-07-03 CPU Willneed Probe On 4.2 SOTA

Design:

- Goal: retest the existing no-source `GGML_MOE_CPU_WILLNEED=1` switch on the current `4.2 tok/s` SOTA path.
- Theory: current remaining bottleneck is CPU up/down fallback. `MADV_WILLNEED` on active CPU fallback expert pages may overlap cold page-in with compute and reduce page stalls. Risk is broad prefetch under the 16GB cgroup can increase reclaim/refault pressure.
- Source state: clean accepted SOTA binary; no source change.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T045821Z-20260703T045821Z-odirect-sota-cpu-willneed-probe/france-cpu40-vram0gb`
- Config delta from accepted SOTA: add `GGML_MOE_CPU_WILLNEED=1`; keep gate O_DIRECT config, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `cpu_moe=40`, strict cold 16GB cgroup, full France output.

Result:

- `eval_tok_s=4.0`, `prompt_tok_s=1.6`, `TTFT=29414.893158 ms`
- `memory_peak_bytes=16000000000`, `memory_file_bytes=15103877120`, `pgmajfault=181560`, `workingset_refault_file=1579212`, `ram_ok=true`, `ram_limit_killed=false`
- `correctness_ok=true`; France answer was semantic and coherent.

Counters and diagnosis:

- Gate behavior stayed aligned with SOTA: one expert pack `hits=4623 misses=0`, direct failures `0`; gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- CUDA memory shape stayed aligned with SOTA: `free=238MiB`, `model=17362MiB`, `unaccounted=14508MiB`.
- The willneed prefetch did not convert into higher token rate and likely added page-cache/reclaim pressure: rounded generation fell from `4.2` to `4.0`.

Verdict:

- Rejected. Do not enable `GGML_MOE_CPU_WILLNEED=1` for current SOTA.
- No rollback required because this was an env-only probe and source remained clean.

### 2026-07-03 Narrow Thread Sweep Design

Design:

- Goal: verify whether the current `-t 20 -tb 20` point is still the best thread count for the accepted 4.2 SOTA path.
- Existing evidence: historical no-source sweeps showed `24 -> 4.0 tok/s`, `28 -> 4.1 tok/s`, `32 -> 2.3 tok/s`, all below SOTA. Values just below and just above 20 were not checked on the current pushed 4.2 SOTA path.
- Theory: CPU up/down fallback is a mix of compute, mmap page stalls, and thread scheduling. Slightly fewer threads may reduce memory pressure/context overhead; slightly more may help compute. Because the observed curve worsens above 20, only test narrow values `18` and `22`.
- Source state: clean accepted SOTA binary; no source change.

Planned runs:

- `-t 18 -tb 18`, accepted gate O_DIRECT config, strict cold 16GB cgroup, full France output.
- `-t 22 -tb 22`, same config, only if the `18` run does not reveal a reproducibility issue.

Acceptance:

- Promote only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and exact run metadata is recorded and pushed.
- Reject/tie on `eval_tok_s <= 4.2` or any gate violation.

Results:

| Threads | Run | eval_tok_s | prompt_tok_s | TTFT ms | memory_peak | memory_file | correctness | Verdict |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 18 | `/root/lfz/runs/vendor-ds4-16gb/20260703T050107Z-20260703T050107Z-odirect-sota-thread18-probe/france-cpu40-vram0gb` | 4.1 | 1.5 | 29008.648567 | 16000000000 | 15078158336 | true | rejected |
| 22 | `/root/lfz/runs/vendor-ds4-16gb/20260703T050239Z-20260703T050239Z-odirect-sota-thread22-probe/france-cpu40-vram0gb` | 4.1 | 1.6 | 28949.464863 | 16000000000 | 15078350848 | true | rejected |

Counters and diagnosis:

- Both runs preserved the accepted gate path: one expert pack `hits=4623 misses=0 direct_failures=0`, gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Both stayed within the strict 16GB cgroup with `oom=0`, `oom_kill=0`, `ram_limit_killed=false`.
- Both France answers were semantic and coherent.
- Neither exceeded the accepted `4.2 tok/s`; keep `-t 20 -tb 20` as the current best-known thread setting.

### 2026-07-03 CPU-Side Skip Filtered One-Stream Design

Design:

- Goal: reduce overhead before CPU up/down fallback without changing model math, routing, cache policy, or accepted gate stream behavior.
- Bottleneck basis: the accepted profile reports `single_accept=35151 single_decline=38222` and `cuda_single=0.429 ms/call` inside the CPU MoE profile. With `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, many up/down `ggml_cuda_moe_stream_one()` calls are guaranteed to be rejected by the CUDA-side name filter. Those failed calls happen before CPU fallback and do not contribute useful compute.
- Theory: add a default-off CPU-side precheck that mirrors the existing one-stream name filter. When `GGML_MOE_CPU_SKIP_FILTERED_STREAM_ONE=1` and `GGML_MOE_STREAM_ONE_NAME_FILTER` is set, `ggml_compute_forward_mul_mat_id()` skips the per-expert one-stream single path entirely for names that do not contain the filter substring. Gate tensors still enter the existing one-stream path; up/down tensors go directly to existing CPU fallback. Arithmetic and selected experts are unchanged.
- Theoretical upper bound: the coarse profile's `cuda_single` component is about `0.429 ms/call * 16920 calls ~= 7.3s`. Only the guaranteed-decline part is removable, so the practical bound is lower, but even a few seconds could move the rounded `4.2 tok/s` metric. If `cuda_single` mostly measures accepted gate work, the probe will tie and be rejected.

Implementation plan:

- Add a small helper in `ggml/src/ggml-cpu/ggml-cpu.c`:
  - default-off env `GGML_MOE_CPU_SKIP_FILTERED_STREAM_ONE`;
  - read `GGML_MOE_STREAM_ONE_NAME_FILTER`;
  - return false for stream-one eligibility when the filter is set and `src0->name` does not contain it.
- Preserve default behavior exactly when the env is unset.
- Build `build-ds4-moe-stream`.
- Run strict cold France with accepted SOTA config plus `GGML_MOE_CPU_SKIP_FILTERED_STREAM_ONE=1` and `GGML_KIMI_CPU_MOE_PROFILE=1` for first measurement.

Acceptance:

- Accept only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and the profile confirms rejected single calls are reduced without damaging gate cache.
- If accepted, commit source and plan immediately, push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then rerun from pushed source.
- If `eval_tok_s <= 4.2` or any gate fails, revert source and rebuild clean.

Implementation:

- Temporary default-off patch added `GGML_MOE_CPU_SKIP_FILTERED_STREAM_ONE=1` in `ggml/src/ggml-cpu/ggml-cpu.c`.
- When enabled, CPU `mul_mat_id` skips the one-stream single path if `GGML_MOE_STREAM_ONE_NAME_FILTER` is set and the tensor name does not contain that substring.
- Default behavior remained unchanged when the env was unset.

Runs:

| Run | Profile | eval_tok_s | prompt_tok_s | TTFT ms | memory_peak | memory_file | correctness | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T050751Z-20260703T050751Z-skip-filtered-stream-one-profile-probe/france-cpu40-vram0gb` | on | 4.2 | 1.6 | 28825.378866 | 16000000000 | 15107457024 | true | rejected/tie |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T050943Z-20260703T050943Z-skip-filtered-stream-one-noprofile-candidate/france-cpu40-vram0gb` | off | 4.1 | 1.6 | 29021.357026 | 16000000000 | 15092740096 | true | rejected |

Counters and diagnosis:

- The patch worked mechanically: profile run changed `single_decline` from the accepted profile's `38222` to `0`.
- Gate path was preserved: one expert pack `hits=4623 misses=0`, direct failures `0`; gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- The expected time saving did not materialize. Profile run still reported `cuda_single=0.419 ms/call`, close to the accepted profile's `0.429 ms/call`, so the coarse `cuda_single` time is dominated by accepted gate stream work rather than rejected up/down calls.
- No-profile candidate regressed to `4.1 tok/s`, so this source patch cannot be promoted.

Rollback:

- Reverted `ggml/src/ggml-cpu/ggml-cpu.c` with `git restore`.
- Clean rebuild completed: `build-ds4-moe-stream/bin/llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`.
- Source worktree clean after rollback.

### 2026-07-03 Single-Row CPU Fallback Chunk Probe Design

Design:

- Goal: test a narrowly scoped CPU fallback scheduling change on the current `4.1-4.2 tok/s` vendor DeepSeek cold-start path without changing model math, routing, gate cache policy, O_DIRECT expert pack, or accepted VRAM layout.
- Bottleneck evidence: refreshed fallback profile reports `24838.624 ms` total CPU up/down fallback, including `17378.123 ms` decode fallback (`8625.536 ms` up, `8752.587 ms` down). Existing top128/top256 fallback mmap probes mechanically hit the compact pack but did not improve token rate, so the bottleneck is not just pointer/source locality.
- Chunk evidence: current CPU chunk trace (`/root/lfz/runs/vendor-ds4-16gb/20260702T131703Z-20260702_phase1_sota_bottleneck_trace_post_kimi/france-cpu40-vram0gb/cpu_chunk_trace.csv`) shows the remaining fallback is dominated by `cne1=1` decode work: `448763.6 ms` thread-sum, estimated `32904.8 ms` wall, `22438.2 ms` ideal-at-20-threads, and about `10466.7 ms` tail imbalance. `cne1=1` accounts for most traced thread-sum and imbalance.
- Current code behavior: for `nr1 == 1`, `mul_mat_id` sets `chunk_size=64`, then collapses to `nchunk0=nth` because `nchunk0*nchunk1 < nth*4`. With `nth=20`, this gives only 20 row chunks per active expert: roughly `103` rows/chunk for up and `205` rows/chunk for down. A single slow page/refault chunk can therefore dominate an expert op tail.
- Historical boundary: prior `GGML_MOE_CPU_CHUNK_SIZE=24/32/64` tests only targeted multi-row/general fallback while explicitly preserving the current `nr1 == 1` behavior. They do not answer whether finer single-row decode chunking can reduce the current tail.

Theory and upper bound:

- A default-off `GGML_MOE_CPU_SINGLE_ROW_CHUNKS=<N>` can override only the `nr1 == 1 && nr0 > 1` chunk count after routing and before CPU fallback. Arithmetic, selected experts, output layout, and cache behavior remain unchanged.
- Candidate `N=40` doubles single-row row chunks: up `~52` rows/chunk, down `~103` rows/chunk. This should reduce tail sensitivity while keeping enough work per chunk to avoid excessive atomic scheduling overhead.
- Hard upper bound is the measured `cne1=1` imbalance estimate (`~10.5s` wall). Realistic gain is much lower because page faults, CPU vector dot arithmetic, graph barriers, and gate stream time remain. If the candidate recovers even `2-4s`, it may move the rounded `eval_tok_s` above `4.2`; otherwise it should tie or regress.
- This probe must preserve O_DIRECT pack counters (`hits=4623 misses=0 direct_failures=0`), gate VRAM cache hit rate (`~86.8%`), 16GB cgroup including page cache, France correctness, and TTFT gate.

Implementation plan:

- Add a small default-off helper in `ggml/src/ggml-cpu/ggml-cpu.c` for `GGML_MOE_CPU_SINGLE_ROW_CHUNKS`.
- In `ggml_compute_forward_mul_mat_id`, when `nr1 == 1 && nr0 > 1 && !disable_chunking && env > 0`, set `nchunk0=min(env,nr0)` and `nchunk1=1` instead of collapsing to `nth` chunks.
- Default behavior must be bit-for-bit path compatible when the env is unset.
- First test `GGML_MOE_CPU_SINGLE_ROW_CHUNKS=40` under strict cold France with the accepted SOTA config. Do not test broader values unless `40` shows a clear positive signal without gate/RAM/TTFT/correctness regression.

Acceptance:

- Accept only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and counters confirm the accepted gate path is unchanged.
- If accepted, immediately commit source and plan, push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean-rebuild and strict-cold rerun from pushed source before declaring a new SOTA.
- If `eval_tok_s <= 4.2` or any gate fails, revert `ggml/src/ggml-cpu/ggml-cpu.c`, clean rebuild, record the rejection, and push only the plan/record.

Implementation and result:

- Temporary default-off patch added `GGML_MOE_CPU_SINGLE_ROW_CHUNKS` to `ggml/src/ggml-cpu/ggml-cpu.c` and changed only the `mul_mat_id` `nr1 == 1 && nr0 > 1` chunk split when the env is set.
- Build succeeded. Dirty experimental hashes: `build-ds4-moe-stream/bin/llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `build-ds4-moe-stream/bin/libggml-cpu.so.0.10.0=75bf2d565669ef2821804515678d45fb97c11c9fea5c10e16ddf5d020c5583a7`.
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T052405Z-20260703T052405Z-single-row-chunks40-candidate/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: `GGML_MOE_CPU_SINGLE_ROW_CHUNKS=40`; otherwise accepted O_DIRECT gate-pack config, `cpu_moe=40`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, strict cold 16GB cgroup.
- Result: `eval_tok_s=4.0`, `prompt_tok_s=1.6`, `TTFT=29651.842642 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15089168384`, `pgmajfault=295997`, `workingset_refault_file=589338`, `ram_ok=false`, `ram_limit_killed=true`, `correctness_ok=true` by heuristic but answer was truncated by the RAM kill.
- Counters: one expert pack `hits=4015 misses=0 reads=4015 bytes=17892638720 failures=0 direct_reads=4015 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=20336 misses=4015 hit_rate=83.5%`.
- Diagnosis: finer single-row chunks did not improve token rate and violated the RAM gate. It also changed generation length/termination under kill and reduced observed gate cache/pack activity versus accepted SOTA (`4623` pack hits and `86.8%` VRAM cache hit rate), so this cannot be promoted. The assumed tail-imbalance recovery was outweighed by extra scheduling/page-cache pressure.
- Verdict: rejected. Do not continue increasing single-row chunk count. Any future variant would need a much smaller value and a profile-only justification first, but this direction is deprioritized because it failed both speed and RAM gates.
- Rollback: reverted `ggml/src/ggml-cpu/ggml-cpu.c` with `git restore` and clean rebuilt. Clean hashes after rollback: `build-ds4-moe-stream/bin/llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `build-ds4-moe-stream/bin/libggml-cpu.so.0.10.0=a6a3ea2d52fd8001716b56bb2703b686438d485253779079eb7f728494541f2a`.

### 2026-07-03 MXFP4 Hotset Repack Microbench Design

Design:

- Goal: decide whether a future hotset-local CPU repack path is worth implementing for remaining up/down fallback. This is a theory/microbench step only, not a SOTA candidate.
- Bottleneck basis: current profile shows `fallback_t0=1.469 ms/call` dominates `mul_mat_id` up/down; route/convert/barrier are near zero. Top-k/layer pruning and page-advice directions have been rejected because they either lose correctness, increase refaults, or tie/regress.
- Source inspection: MXFP4 CPU fallback currently uses row-wise `ggml_vec_dot_mxfp4_q8_0` on mmap/default CPU buffers. The repo also has `CPU_REPACK` support with MXFP4 8x8 AVX2 `ggml_gemv_mxfp4_8x8_q8_0`, but current SOTA logs show no `CPU_REPACK` buffer and cgroup memory is mostly file page cache, so full repack is not active.
- Full CPU repack is not acceptable under the 16GB host RAM limit because it would materialize large expert tensors as anonymous memory and displace the page cache. Any useful repack must be hotset-local and bounded, e.g. top tens/hundreds of up/down experts, with exact reproduction metadata.

Theory and upper bound:

- The hot top128 decode up/down entries cover only about `3695 ms` of measured decode fallback; top256 covers about `5379 ms`. A repack path can only speed the compute portion of that covered time, not routing, graph barriers, gate stream, or page-cache effects.
- If 8x8 repack GEMV is `S` times faster than row-wise dot for DS4 dimensions, the hard upper bound for top128 is roughly `3695*(1-1/S) ms`. Even an ideal `2x` kernel speedup gives only about `1.85s`, likely below run variance unless page/refault behavior also improves.
- Therefore first run a standalone microbench comparing current row-wise MXFP4 dot versus existing MXFP4 8x8 repack GEMV on representative DS4 shapes (`ne00=4096`, up rows `2048`, down rows `4096`, `Q8_0` activations). If speedup is below about `1.5x`, do not implement hotset repack integration.

Practice plan:

- Build a temporary standalone microbench outside committed source or under an ignored scratch path. It may compile against existing ggml CPU objects/headers but must not change runtime source.
- Measure warm compute only for row-wise current layout and 8x8 repacked layout; include correctness max-abs/mean-abs against row-wise output.
- Record exact compile command, CPU flags, dimensions, iteration count, speedup, and numerical error in this plan.
- If the microbench shows strong speedup and exact numerical agreement, write a separate design for a bounded hotset-local repack path. If not, reject the direction without model runs.

Acceptance for follow-up implementation:

- No model SOTA claim can be made from this microbench.
- A later hotset repack source candidate must be default-off, bounded by explicit memory budget, preserve accepted gate O_DIRECT config, pass France correctness, and satisfy strict 16GB cgroup including page cache.

Microbench result:

- Scratch source: `/tmp/mxfp4_repack_microbench.cpp`; binary: `/tmp/mxfp4_repack_microbench`. No repository runtime source was modified.
- Compile command: `g++ -O3 -march=native -std=c++17 -Iggml/include -Iggml/src -Iggml/src/ggml-cpu /tmp/mxfp4_repack_microbench.cpp -Lbuild-ds4-moe-stream/bin -lggml-cpu -lggml-base -Wl,-rpath,/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin -pthread -ldl -lm -o /tmp/mxfp4_repack_microbench`.
- CPU: AMD EPYC 7B13 with AVX2/FMA, no AVX512/AMX.
- Up-like shape (`k=4096`, `rows=2048`, `iters=200`): `repack_ms=4.075`, row-wise `86.766 ms`, repack GEMV `61.689 ms`, speedup `1.407x`, `max_abs=0`, `mean_abs=0`.
- Down-like shape (`k=2048`, `rows=4096`, `iters=200`): `repack_ms=3.027`, row-wise `89.916 ms`, repack GEMV `61.430 ms`, speedup `1.464x`, `max_abs=0`, `mean_abs=0`.
- Earlier oversized down check (`k=4096`, `rows=4096`, `iters=100`) showed similar `1.418x` speedup and exact agreement.
- Theoretical implication: top128 decode up/down coverage is about `3695 ms`; with `S=1.41-1.46`, the compute-only upper bound is roughly `1070-1170 ms`. Top256 coverage gives roughly `1680-1990 ms`, before repack memory, hotset lookup, page-cache effects, and integration overhead. This is below the threshold needed to reliably beat the `4.2 tok/s` SOTA under cold-run variance.
- Verdict: reject hotset-local MXFP4 CPU repack integration for now. Existing repack kernels are mathematically exact and faster in warm compute, but not fast enough to justify adding a memory-consuming hotset path under the strict 16GB cold-start constraint.


### 2026-07-03 Current SOTA `--no-repack` Diagnostic Design

Design:

- Goal: close the remaining loader/buffer ambiguity on the current accepted gate O_DIRECT SOTA path by testing `--no-repack` without changing source.
- Source state: clean accepted SOTA binary; no runtime source change.
- Current evidence: the MXFP4 microbench showed full/hotset CPU repack is not worth implementing under the 16GB cold-start constraint. The current SOTA logs also show cgroup memory dominated by file page cache, not an obvious CPU_REPACK anonymous buffer. However, there has not yet been a current-SOTA strict run with `--no-repack` alone.
- Theory: if the accepted path is already effectively using mmap/default CPU buffers for the remaining CPU up/down fallback, `--no-repack` should tie and prove repack is not a hidden lever. If some implicit CPU_REPACK buffer is still active, `--no-repack` may reduce anonymous memory/page pressure but can also slow CPU GEMV. Because arithmetic and routing are unchanged, correctness should remain identical.
- Hard upper bound: this switch can only affect loader/buffer placement and CPU GEMV layout; it cannot reduce the measured gate stream cost or expert pack direct-read volume. A speedup large enough to beat `4.2 tok/s` is unlikely unless there is hidden repack memory pressure.

Planned run:

- Strict cold France with accepted SOTA config plus `--no-repack`.
- Preserve `cpu_moe=40`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate admission profile, gate O_DIRECT expert pack, `GGML_MOE_KEEP_TOPK_*`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, drop_caches, and 16GB cgroup.

Acceptance:

- Accept only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and counters show the accepted gate path is preserved.
- If it ties or regresses, reject and keep the current SOTA unchanged.
- No rollback is needed because this is a CLI-flag-only diagnostic.


Result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T054431Z-20260703T054431Z-sota-no-repack-diagnostic/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: add CLI flag `--no-repack`; otherwise accepted gate O_DIRECT config, `cpu_moe=40`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, strict cold 16GB cgroup.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=29617.281870 ms`, `elapsed_seconds=63.11`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15084883968`, `pgmajfault=263322`, `workingset_refault_file=1707591`, `ram_ok=true`, `ram_limit_killed=false`, `oom_seen=false`.
- Correctness: `correctness_ok=true`; France answer was semantic and coherent: it described France as a Western European country with rich history, culture, landmarks, cuisine, fashion, EU membership, economy, and global cultural/economic influence.
- Gate/O_DIRECT counters stayed aligned with accepted SOTA: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- VRAM shape stayed aligned with accepted SOTA: CUDA0 `free=238MiB`, `model=17362MiB`, `unaccounted=14508MiB`.

Diagnosis:

- `--no-repack` does not expose a hidden performance lever on the current path. Token rate tied the repeated strict-cold reproduction line (`4.1`) and did not beat the historical accepted `4.2` SOTA.
- The identical gate pack/cache counters show the accepted gate path was preserved; the remaining bottleneck is still CPU up/down fallback plus cold page-cache/refault behavior, not an avoidable full CPU_REPACK buffer.

Verdict:

- Rejected/tie. Keep current SOTA unchanged.
- No rollback required because this was a CLI-flag-only diagnostic and source remained clean.


### 2026-07-03 Profile-Gated Top64 Up/Down One-Stream Design

Design:

- Goal: test a compute/offload candidate that avoids the previous no-filter failure. Instead of letting every up/down expert enter the CUDA one-stream path, allow only admission-profile hits: existing gate hotset plus decode top64 up/down fallback experts. Non-profile up/down must continue directly to CPU fallback.
- Bottleneck basis: current profile still shows CPU up/down fallback as the dominant cost (`24838.624 ms` total, `17378.123 ms` decode). The top64 decode up/down hotset covers `2340.235 ms` across `6382` calls with `0.266 GiB` unique payload.
- Prior failure boundary: `one-stream-all-no-filter` regressed to `1.7 tok/s` because it removed the name filter and made non-admitted up/down tensors pay GPU single-path staging without useful cache residency. This candidate must not repeat that; the source change must be default-off and profile-gated before staging/allocation.
- VRAM accounting: keep `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, so total cache allocation remains the accepted `~13.2 GiB / 3192 slots`. The top64 up/down profile entries need about `64` slots (`~272 MiB`), which displaces at most `~2.0%` of gate slots instead of increasing VRAM allocation. The current SOTA has about `238 MiB` free VRAM, so increasing cache size is not allowed.
- Theoretical upper bound: if all top64 covered CPU fallback time were removed, the maximum wall saving is about `2.34s`; practical saving is lower because GPU kernel/D2H/sync and possible gate cache evictions remain. This is just above the plan's `~2s` threshold and is worth one narrow probe.

Artifacts:

- Admission profile: `.Agent/profiles/vendor-ds4/current_sota_gate_plus_decode_top64_updown.tsv`, sha256 `10c9fe0d77426f2118773f3cfd5eeb43a89b8b6212e52f075cc3b98d9ebb3bec`, entries `3377` = existing gate profile plus top64 decode up/down entries.
- Hotset source: `/root/lfz/runs/vendor-ds4-16gb/20260703T034116Z-next-plan-phase1-hotsets/decode_top128_updown_meta.tsv`, first `64` rows, covered fallback `2340.235 ms`, calls `6382`, unique payload about `0.266 GiB`.

Implementation plan:

- Add a default-off env `GGML_MOE_STREAM_ONE_ALLOW_ADMIT_PROFILE=1` in `ggml/src/ggml-cuda/moe_stream.cu`.
- When enabled, `moe_stream_one_type_allowed()` may allow a tensor/expert if either the existing name filter matches or the admission profile explicitly contains `(src0_name, expert_index)`.
- This check must happen before one-stream staging/allocation, so non-profile up/down tensors return `false` and fall back to the existing CPU path.
- Default behavior with env unset must remain identical to the current accepted SOTA.

Practice plan:

- Build `build-ds4-moe-stream` with the default-off patch.
- First run strict cold France with accepted SOTA config plus `GGML_MOE_STREAM_ONE_ALLOW_ADMIT_PROFILE=1` and `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=.Agent/profiles/vendor-ds4/current_sota_gate_plus_decode_top64_updown.tsv`.
- Keep gate O_DIRECT expert pack unchanged; it only contains gate experts, so top64 up/down cache inserts will read from the existing mapped model source.
- Do not change cache budget, thread count, context, top-k policy, or RAM cgroup.

Acceptance:

- Accept only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, gate pack counters remain direct-failure-free, and VRAM cache hit/miss counters show the profile-gated path did not collapse like no-filter.
- If accepted, immediately commit source/profile/plan and push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean-rebuild/rerun from pushed source.
- If `eval_tok_s <= 4.2` or any gate fails, revert runtime source, clean rebuild, record the rejection, and push only plan/profile/record as diagnostic artifacts.


Implementation and result:

- Temporary default-off source patch added `GGML_MOE_STREAM_ONE_ALLOW_ADMIT_PROFILE=1` in `ggml/src/ggml-cuda/moe_stream.cu`.
- The patch allowed one-stream execution when either the existing `GGML_MOE_STREAM_ONE_NAME_FILTER` matched or the strict admission profile explicitly contained `(src0_name, expert_index)`. Profile-miss up/down tensors returned before staging/allocation and fell back to CPU.
- Dirty candidate build completed. Candidate CUDA lib hash was `54546f2fe665deb0853946183135230aecd1d86841046736cb6c524e32a53a8f`; `llama-cli` hash remained `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`.
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T060115Z-20260703T060115Z-profile-gated-top64-updown-stream/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: `GGML_MOE_STREAM_ONE_ALLOW_ADMIT_PROFILE=1` and `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_plus_decode_top64_updown.tsv`; otherwise accepted gate O_DIRECT config, cache budget, top-k, threads, and strict 16GB cold cgroup.
- Metrics: `eval_tok_s=3.4`, `prompt_tok_s=1.5`, `TTFT=30582.986899 ms`, `elapsed_seconds=86.99`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15064596480`, `pgmajfault=334517`, `workingset_refault_file=4314651`, `ram_ok=true`, `ram_limit_killed=false`.
- Manual correctness: rejected. The answer was mostly semantic but ended incomplete at `from its language`, so it does not satisfy the coherent complete France-output requirement.
- Counters: admission profile loaded `3377` entries; one expert pack `hits=5234 misses=1760 reads=5234 bytes=23325048832 failures=0 direct_reads=5234 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=49772 misses=6994 hit_rate=87.7%`.

Gap analysis:

- The profile gate worked mechanically and avoided the previous no-filter collapse, but it still changed the generation/cache trajectory. Relative to accepted SOTA, pack reads increased (`4623 -> 5234`), VRAM cache misses increased (`4623 -> 6994`), and file refault pressure increased sharply (`~1.7M -> 4.31M`).
- The top64 hotset's optimistic `2.34s` covered fallback was outweighed by one-stream GPU kernel/D2H/sync overhead, extra up/down source page pressure, gate cache churn, and a longer/incomplete output trajectory.
- This closes small profile-gated up/down one-stream as a SOTA path under the current shared cache/staging design. Any future up/down offload would need a separate cache/staging path with stronger proof that it does not perturb gate cache and output trajectory.

Rollback:

- Reverted `ggml/src/ggml-cuda/moe_stream.cu` with `git restore` and clean rebuilt `build-ds4-moe-stream`.
- Clean hashes after rollback: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu=a6a3ea2d52fd8001716b56bb2703b686438d485253779079eb7f728494541f2a`, `libggml-cuda=bc5f8d943233bd73b46c6df8307f42f2399bac269d4c69e1179a7b80f10e492e`.
- Verdict: rejected. Current accepted SOTA remains unchanged at historical `4.2 tok/s`; repeated strict-cold line remains `4.1 tok/s`.


### 2026-07-03 CPU Affinity `taskset 0-19` Diagnostic Design

Design:

- Goal: test whether pinning the 20 CPU fallback threads to a fixed 20-vCPU set reduces scheduler migration noise and improves the accepted SOTA path without changing model math, routing, cache policy, or source code.
- Topology evidence: `lscpu` reports `61` online vCPUs, `Thread(s) per core=1`, and a single NUMA node (`0-60`). Therefore NUMA memory binding is not useful, but scheduler migration across 61 vCPUs can still add cache/TLB variability for the CPU up/down fallback loop.
- Bottleneck basis: remaining bottleneck is CPU up/down fallback plus cold file-page/refault behavior. Existing source-level chunking, mmap pack, repack, willneed, and up/down one-stream candidates either tied or regressed. A no-source affinity probe is lower risk and can identify whether OS scheduling is part of the `4.1-4.2` variance.
- Theory: with `-t 20 -tb 20`, pinning to exactly CPUs `0-19` may reduce thread migration and improve cache locality. It can also regress if the scheduler currently benefits from spreading work over more vCPUs or if CPUs `0-19` are noisier. It cannot change arithmetic correctness.
- Hard upper bound: this can only recover scheduler/cache locality overhead, not the full `~24.8s` fallback compute or gate direct-read cost. Expected benefit is at most a small rounded-boundary move; accept only above `4.2 tok/s` with all gates.

Artifact:

- Wrapper: `.Agent/run-tools/llama-cli-taskset-0-19.sh`, sha256 `a391003fd8b44104d3964319ecf7f083572a11dab8b180ab07104c971c63fe43`. It executes `/usr/bin/taskset -c 0-19 /root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin/llama-cli "$@"`.

Practice plan:

- Run strict cold France with `--binary .Agent/run-tools/llama-cli-taskset-0-19.sh` and otherwise accepted SOTA config.
- Preserve gate O_DIRECT pack, admission profile, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, top-k policy, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, drop_caches, and 16GB cgroup.
- Compare token rate, TTFT, cgroup file/refault counters, pack counters, and manual France correctness against current accepted SOTA and repeated `4.1` line.

Acceptance:

- Accept only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and exact wrapper/source/command metadata are recorded and pushed.
- If it ties or regresses, reject and keep current SOTA unchanged. No runtime source rollback is needed.


Result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T061052Z-20260703T061052Z-taskset0-19-sota-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: strict runner binary was `.Agent/run-tools/llama-cli-taskset-0-19.sh`, which pins `llama-cli` to CPUs `0-19`; otherwise accepted gate O_DIRECT config, cache budget, top-k, threads, and strict 16GB cold cgroup.
- Metrics: `eval_tok_s=3.2`, `prompt_tok_s=1.5`, `TTFT=31655.711601 ms`, `elapsed_seconds=73.79`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15099891712`, `pgmajfault=271107`, `workingset_refault_file=1653058`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: `correctness_ok=true`; France answer was semantic, coherent, and complete.
- Gate/O_DIRECT counters stayed exactly aligned with accepted SOTA: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.

Diagnosis:

- CPU affinity did not disturb the gate cache or page-cache shape, but it substantially reduced generation throughput. The regression is therefore CPU scheduling/throughput related, not a cache correctness issue.
- Pinning exactly 20 threads to exactly 20 vCPUs likely removes scheduler flexibility that the CPU fallback path benefits from on this 61-vCPU single-NUMA VM.
- Do not use fixed `taskset 0-19` for the SOTA path. Broader affinity masks are unlikely to be a high-priority path unless a future profile shows migration overhead explicitly.

Verdict:

- Rejected. Current accepted SOTA remains unchanged at historical `4.2 tok/s`; repeated strict-cold line remains `4.1 tok/s`.
- No runtime source rollback required.


### 2026-07-03 MXFP4 Dot Prefetch Microbench Design

Design:

- Goal: check whether the remaining CPU up/down fallback can benefit from a low-level x86 MXFP4 dot prefetch path before spending another strict cold model run.
- Bottleneck basis: current accepted path still spends most time in CPU up/down fallback. The active MXFP4 CPU path uses `ggml_vec_dot_mxfp4_q8_0` from `ggml/src/ggml-cpu/arch/x86/quants.c`; the surrounding q4 x86 code has explicit `_mm_prefetch`, while the MXFP4 loop currently relies on hardware prefetch only.
- Theory: each MXFP4 dot scans sequential `block_mxfp4` source blocks and Q8 activation blocks. Explicitly prefetching several blocks ahead may hide some memory latency when expert rows are read from mmap/page cache under cold cgroup pressure. It cannot reduce routing, barriers, gate stream, or model math. It must be bitwise/near-bitwise equivalent because it only changes prefetch hints.
- Risk: the loop is already small and hardware prefetch may be sufficient; explicit prefetch can add instruction overhead or pollute cache. Therefore this is a microbench-first diagnostic.
- Hard upper bound: if prefetch improved the dot kernel by `S`, only the compute/memory-scan portion of the `~24.8s` CPU fallback can benefit. A warm microbench speedup below about `3%` is not worth a model run; `>=5%` with exact output agreement is the minimum signal for one strict cold candidate.

Implementation plan:

- Add a temporary default-off env `GGML_MXFP4_DOT_PREFETCH_BLOCKS=<N>` to the x86 MXFP4 dot path. Unset or `0` must preserve current behavior.
- When enabled, prefetch `x[ib+N]` and `y[ib+N]` in the AVX2 loop before loading current blocks.
- Build `build-ds4-moe-stream` and run a standalone microbench that calls `ggml_vec_dot_mxfp4_q8_0` on DS4-like shapes (`n=4096`) for representative row counts/iterations.
- Test distances `4`, `8`, and `16` in separate processes because the env is cached once.

Acceptance for model candidate:

- Proceed to a strict cold France model run only if microbench shows `>=5%` speedup over baseline and `max_abs`/`mean_abs` are zero or explainably identical within floating accumulation order.
- If microbench ties/regresses, revert source, clean rebuild, record rejection, and do not run a full model candidate.
- Any later model candidate must still pass `eval_tok_s > 4.2`, 16GB cgroup including page cache, TTFT gate, gate O_DIRECT counters, and manual France correctness.


Microbench result:

- Temporary default-off patch added `GGML_MXFP4_DOT_PREFETCH_BLOCKS=<N>` to `ggml/src/ggml-cpu/arch/x86/quants.c` inside `ggml_vec_dot_mxfp4_q8_0` AVX2 loop.
- Dirty candidate CPU lib hash: `72ea8640e9b152bbe1f6653b4f1cda8c4f685e61e94f19cf33665e030a5c74a5`; `llama-cli` hash stayed `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`.
- Scratch source: `/tmp/mxfp4_dot_prefetch_bench.cpp`; binary: `/tmp/mxfp4_dot_prefetch_bench`.
- Compile command: `g++ -O3 -march=native -std=c++17 -Iggml/include -Iggml/src -Iggml/src/ggml-cpu /tmp/mxfp4_dot_prefetch_bench.cpp -Lbuild-ds4-moe-stream/bin -lggml-cpu -lggml-base -Wl,-rpath,/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin -pthread -ldl -lm -o /tmp/mxfp4_dot_prefetch_bench`.
- The bench calls `ggml_cpu_init()` so the E8M0 lookup table is initialized, then measures `n=4096`, `rows=4096`, `iters=300`; each prefetch distance ran in a separate process.

| Prefetch blocks | ms | ns/dot | dots/s | checksum | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 257.461 | 209.523 | 4772755.230 | 2343998.85 | baseline |
| 4 | 267.941 | 218.051 | 4586088.358 | 2343998.85 | slower by `~4.1%` |
| 8 | 263.111 | 214.120 | 4670270.858 | 2343998.85 | slower by `~2.2%` |
| 16 | 262.079 | 213.280 | 4688671.309 | 2343998.85 | slower by `~1.8%` |

Diagnosis:

- Checksums and first/last values matched, so prefetch did not change arithmetic.
- Every tested prefetch distance regressed warm dot throughput. The MXFP4 AVX2 loop appears compute/instruction-bound enough, or hardware prefetch is already sufficient for this access pattern; explicit prefetch adds overhead/pollution.
- This fails the `>=5%` microbench threshold and does not justify a strict cold model run.

Rollback:

- Reverted `ggml/src/ggml-cpu/arch/x86/quants.c` with `git restore` and clean rebuilt.
- Clean hashes after rollback: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu=a6a3ea2d52fd8001716b56bb2703b686438d485253779079eb7f728494541f2a`, `libggml-cuda=bc5f8d943233bd73b46c6df8307f42f2399bac269d4c69e1179a7b80f10e492e`.
- Verdict: rejected at microbench stage; no model run performed. Current accepted SOTA remains unchanged.


### 2026-07-03 Current SOTA CUDA Graph Probe Design

Design:

- Goal: retry CUDA graph on the current accepted vendor DeepSeek SOTA path, using the exact current gate O_DIRECT config and strict 16GB cold-start runner.
- Switch semantics: `ggml/src/ggml-cuda/common.cuh` disables CUDA graph when `getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr`, so setting `GGML_CUDA_DISABLE_GRAPHS=0` still disables graph. The actual `llama-cli` process must run with this env unset.
- Current runner behavior: `.Agent/run-tools/strict_ds4_runner.py` always exports `GGML_CUDA_DISABLE_GRAPHS=1`. Therefore this probe uses a wrapper binary that unsets the variable immediately before executing the real `llama-cli`.
- Theory: CUDA graph may reduce repeated CUDA launch overhead in the accepted gate one-stream path. It cannot reduce cold source page-in, O_DIRECT expert pack reads, or CPU up/down fallback. Prior older attempts did not improve earlier SOTA lines, but the current O_DIRECT/gate-cache path should be retested directly.
- Hard upper bound: only CUDA launch/scheduling overhead is affected. Accept only if rounded `eval_tok_s` exceeds `4.2` and all SOTA gates pass.

Artifact:

- Wrapper: `.Agent/run-tools/llama-cli-cuda-graphs-enabled.sh`, sha256 `3e08e54173e404646f7cdbf269abb8e9bfa425e201eb881363ba74d7808310ee`. It runs `unset GGML_CUDA_DISABLE_GRAPHS` then execs `build-ds4-moe-stream/bin/llama-cli`.

Practice plan:

- Run strict cold France with `--binary .Agent/run-tools/llama-cli-cuda-graphs-enabled.sh` and otherwise accepted SOTA config.
- Keep all normal SOTA env vars unchanged, including the runner-provided `GGML_CUDA_DISABLE_GRAPHS=1`; the wrapper is responsible for unsetting it only for the child `llama-cli`.
- Verify actual run evidence via `environment.txt` plus stderr/command; exact command should show wrapper as binary.

Acceptance:

- Accept only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and gate pack/cache counters remain aligned with accepted SOTA.
- If it ties or regresses, reject and keep current SOTA unchanged. No runtime source rollback is needed.


Result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T091200Z-20260703T091200Z-cuda-graphs-enabled-sota-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: strict runner binary was `.Agent/run-tools/llama-cli-cuda-graphs-enabled.sh`, which unsets `GGML_CUDA_DISABLE_GRAPHS` before execing the real `llama-cli`; otherwise accepted gate O_DIRECT config, cache budget, top-k, threads, and strict 16GB cold cgroup.
- Evidence note: `environment.txt` still records runner-level `GGML_CUDA_DISABLE_GRAPHS=1`, but `exact_command.txt` shows the wrapper binary. The wrapper sha256 was `3e08e54173e404646f7cdbf269abb8e9bfa425e201eb881363ba74d7808310ee` and its only behavior is `unset GGML_CUDA_DISABLE_GRAPHS` followed by exec of the accepted `llama-cli`.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30535.957676 ms`, `elapsed_seconds=64.12`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15042371584`, `pgmajfault=272088`, `workingset_refault_file=1716820`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: `correctness_ok=true`; France answer was semantic, coherent, and complete.
- Gate/O_DIRECT counters stayed aligned with accepted SOTA: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.

Diagnosis:

- CUDA graph did not improve the current O_DIRECT/gate-cache SOTA path. It tied the repeated strict-cold reproduction line (`4.1`) and did not exceed the accepted `4.2` threshold.
- The unchanged gate/cache counters show the run exercised the same SOTA path; remaining bottleneck is still cold page/source behavior plus CPU up/down fallback, not CUDA launch overhead.

Verdict:

- Rejected/tie. Keep current accepted SOTA unchanged.
- No runtime source rollback required.


### 2026-07-03 Latest State And Next Optimization Plan

Current accepted state:

- Accepted cold-start SOTA remains the historical `4.2 tok/s` line from `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`.
- Repeated strict-cold reproduction line is `4.1 tok/s`; this is expected run-to-run variance and does not replace the accepted `4.2 tok/s` record.
- Latest pushed source/record head: `17f769b54441b3d33e42d1995cb0ea4500b41a0c`, pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- Current demo script: `.Agent/examples/demo_current_sota.sh`, added in commit `f8524ddd0 vendor-ds4: add current sota demo script` and pushed to the same `ssd` branch. It runs the strict 16GB cold SOTA config and prints run directory, exact command, answer, RAM counters, pack counters, and cache counters.
- Latest demo verification run: `/root/lfz/runs/vendor-ds4-16gb/20260703T083454Z-demo-current-sota-file-check/france-cpu40-vram0gb`, `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=29384.906208 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15103238144`, `ram_ok=true`, `correctness_ok=true`, gate pack `hits=4623 misses=0 direct_failures=0`, gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.

Model and artifact sizes:

- Model: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, `156,148,189,760 bytes` (`156.1 GB` decimal, `145.4 GiB`, `ls -lh` shows `146G`).
- Gate expert pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, `20,495,904,768 bytes` (`20.5 GB` decimal, `19.1 GiB`, `ls -lh` shows `20G`), sha256 `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`.
- Admission profile: `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, sha256 `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`.

Current compute split:

- GPU: attention/main dense/norm/embedding/output as far as `-ngl all --fit on` allows, plus `ffn_gate_exps.weight` via the DS4 one-stream path using the `13568MiB` VRAM gate cache and O_DIRECT gate expert pack misses.
- CPU: `ffn_up_exps.weight` and `ffn_down_exps.weight` still run mostly through CPU fallback with `-t 20 -tb 20`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, and layers `10-39` reduced to top3.
- This is not a full layer split and not all experts per layer. Only routed/pruned experts are computed. Gate uses one-stream cache; broad up/down one-stream and down-batch cache have been rejected.
- Accepted SOTA disables CUDA graph with `GGML_CUDA_DISABLE_GRAPHS=1`. The wrapper-based graph-enabled probe tied at `4.1 tok/s` and is rejected, so CUDA launch overhead is not the current promotion bottleneck.

Latest bottleneck conclusion:

- The stable accepted path is gate-only one-stream plus CPU up/down fallback. Gate O_DIRECT pack and gate VRAM cache are working and must not be disturbed without a specific theory and guard run.
- Rejected down-batch probes show GPU down kernel time is small, but host staging dominates. Reducing gate cache to make room for down cache collapses gate hit rate and loses far more than it gains.
- Preserving the accepted gate cache leaves only about `238MiB` free VRAM in observed runs, too little for an independent useful down cache.
- Broad one-stream up/down is much slower than CPU fallback and damages gate locality.
- CPU fallback remains the only large unresolved component, but previous coarse fixes failed: compact mmap pack tied/regressed, willneed regressed, fixed CPU affinity regressed, single-row chunks40 failed RAM gate, and MXFP4 prefetch regressed in microbench.

Next optimization plan:

1. Baseline guard before any new source change:
   - Run `.Agent/examples/demo_current_sota.sh --run-name <timestamp>-pre-next-candidate-guard` or the equivalent strict runner command.
   - Required: `4.1 tok/s` class or better, `ram_ok=true`, `correctness_ok=true`, no OOM/kill, pack `direct_failures=0`, gate cache `hit_rate` near `86.8%`.
   - If the guard fails, stop and debug reproducibility instead of starting a new optimization.

2. Fine CPU fallback phase trace before the next optimization method:
   - Add only default-off diagnostic instrumentation first, or use existing profiles if they can provide the same split.
   - The trace must split CPU up/down fallback by tensor, layer, expert, prompt/decode, `nr1/cne1`, chunk count, wall time, thread-sum or tail proxy, and page/refault deltas where available.
   - Record whether time is dominated by MXFP4 dot compute, source page faults/refaults, chunk tail imbalance, synchronization/barrier, or repeated setup/conversion.
   - This is diagnostic only and cannot promote SOTA.

3. Design candidates only after the trace identifies a compressible component:
   - If dot compute dominates: run microbench-first CPU kernel candidates only. No model run unless microbench shows a clear speedup threshold and identical/acceptable numerical output.
   - If tail imbalance dominates: design a smaller and safer scheduling change than the rejected `single-row chunks40`, with a theory for why it will not raise cgroup/file pressure. Start with a short profile run, not a full SOTA run.
   - If page/refault dominates: design a bounded page-source strategy that does not add anonymous RAM and does not displace the accepted gate cache/page-cache shape. Large buffered packs and broad `MADV_WILLNEED` remain rejected.
   - If setup/barrier dominates: target that path directly with a default-off source patch and prove the counter falls before a full run.

4. Low-priority/deprioritized directions unless new evidence appears:
   - More independent down-cache sizing: blocked by VRAM budget and gate cache loss.
   - Broad one-stream up/down offload: rejected by `1.7 tok/s` no-filter diagnostic.
   - CUDA graph: rejected/tie on current SOTA.
   - CPU affinity to exactly 20 CPUs: rejected.
   - Context-size trimming: did not free useful VRAM.
   - Larger compact mmap packs alone: top128/top256 tied/regressed.

5. Promotion and push discipline:
   - Any result with `eval_tok_s > 4.2` and passing RAM/correctness/TTFT/O_DIRECT gates must immediately stop exploration.
   - Commit source, plan, and run metadata, then push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using the existing `L-Ark` identity.
   - Rebuild clean from the pushed source and rerun strict cold before declaring the new SOTA.
   - Rejected or tie runs may be committed only as records. Any temporary source patch must be reverted and clean-rebuilt before recording rejection.


Baseline guard execution:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T095237Z-20260703T095236Z-pre-next-candidate-guard/france-cpu40-vram0gb`.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=29406.374764 ms`, `elapsed_seconds=62.51`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15086735360`, `pgmajfault=266453`, `workingset_refault_file=1678871`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: `correctness_ok=true`; answer was semantic, coherent, and complete for `Please introduce France in a short paragraph.`.
- Gate/O_DIRECT counters: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Verdict: baseline guard passed. This confirms the current pushed path remains reproducible at the `4.1 tok/s` repeated strict-cold line and is safe to use as the pre-candidate baseline. Accepted historical SOTA remains `4.2 tok/s`; this guard is not a promotion.


CPU fallback fine trace design:

- Goal: identify the next compressible CPU fallback component before any new optimization source change.
- Source state: clean accepted SOTA binary; no source change.
- Existing instrumentation: `GGML_MOE_CPU_CHUNK_TRACE_OUT={case_dir}/cpu_chunk_trace.csv` records per CPU fallback chunk `seq,tensor,type,expert,ith,nth,cne1,ir0_start,ir0_end,ir1_start,ir1_end,src0_bytes,ms`. `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`, and `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback-profile.csv` provide operation/name/expert aggregate timing.
- Run shape: strict cold full France run with accepted gate O_DIRECT config, plus chunk trace/profile env vars. Use a high trace limit (`GGML_MOE_CPU_CHUNK_TRACE_LIMIT=800000`) to capture the full prompt+decode fallback path.
- Required analysis after run:
  - summarize fallback profile by prompt/decode and up/down;
  - summarize chunk trace by tensor, phase-inferred shape (`cne1`), expert, thread, and row chunk ranges;
  - estimate wall time, thread-sum time, max-thread tail, and imbalance for `cne1=1` decode-like work vs multi-row prompt-like work;
  - compare RAM, page/refault, pack counters, gate cache counters, and France correctness against the baseline guard;
  - decide whether the next candidate should target dot compute, tail imbalance, page/refault/source load, or setup/barrier.
- Acceptance: this is diagnostic only. It cannot promote SOTA even if token rate ties/exceeds historical `4.2`; any optimization must be designed from the trace and run separately under the normal acceptance gates.


Fine trace first attempt:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T095715Z-20260703T095715Z-cpu-fallback-fine-trace/france-cpu40-vram0gb`.
- Metrics: `eval_tok_s=3.9`, `prompt_tok_s=1.5`, `TTFT=30553.465559 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15103176704`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- Gate/O_DIRECT counters remained aligned with SOTA: one expert pack `hits=4623 misses=0 direct_failures=0`; gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Problem: `cpu_chunk_trace.csv` has exactly `800001` lines including header, so the `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=800000` limit was hit and the trace is incomplete.
- Action: rerun the same diagnostic with `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=2000000` before drawing bottleneck conclusions. The first attempt is useful only as a guard that profiling does not disturb RAM/correctness/gate counters; it is not sufficient for phase-trace analysis.


Fine trace completed and early-layer top3 candidate design:

- Complete trace run: `/root/lfz/runs/vendor-ds4-16gb/20260703T100011Z-20260703T100011Z-cpu-fallback-fine-trace-limit2m/france-cpu40-vram0gb`.
- Metrics: `eval_tok_s=4.0`, `prompt_tok_s=1.5`, `TTFT=29526.174987 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15104675840`, `pgmajfault=259065`, `workingset_refault_file=1642262`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- Trace completeness: `cpu_chunk_trace.csv` has `909953` lines including header, below the `2000000` limit. `fallback-profile.csv` has `7031` lines.
- Analysis artifacts: `cpu_fallback_fine_trace_analysis.json` and `cpu_fallback_fine_trace_analysis.txt` in the run directory.
- Fallback profile: total up/down fallback `26972.399 ms`; decode up `9725.822 ms`, decode down `9673.781 ms`, prompt up `3317.296 ms`, prompt down `4255.500 ms`.
- Chunk trace thread-sum: up/down `506468.184 ms`; divided by 20 threads gives about `25323.4 ms`, close to the fallback profile wall total. This points to MXFP4 dot/source scanning as the dominant cost, not a large recoverable scheduling-tail component.
- Shape split: cne1=1 decode-like thread-sum is `423997.611 ms` (`~21199.9 ms / 20`), while cne1>1 prompt-like thread-sum is `82470.573 ms` (`~4123.5 ms / 20`). Decode-like work is the main residual CPU fallback cost.
- Layer fallback concentration: layers `0-9` account for `9928.306 ms` (`36.8%`) of total up/down fallback; layers `0-2` account for `4304.726 ms` (`16.0%`). The current accepted config leaves layers `0-9` at top4 while layers `10-39` use top3.
- Candidate: run a no-source strict cold probe with `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39` and `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, keeping all other accepted SOTA config unchanged. This reduces early layers from top4 to top3 as well.
- Theory and upper bound: if selected experts have roughly similar CPU fallback cost, reducing layers `0-9` from top4 to top3 can save at most about `25% * 9928.306 ms ~= 2482 ms`; layers `0-2` alone bound about `1076 ms`. This is enough to test because the current SOTA boundary is close, but it is not guaranteed because routing distribution is uneven and output quality may degrade.
- Risk: top-k pruning changes model math and may reduce answer quality or truncate/alter the output. This probe is acceptable only if France remains semantic, coherent, and complete. It must preserve 16GB cgroup, TTFT gate, gate pack direct failures `0`, and gate cache behavior.
- Acceptance: promote only if `eval_tok_s > 4.2` with RAM/correctness/TTFT/O_DIRECT gates passing. If `eval_tok_s <= 4.2` or correctness degrades, reject as diagnostic and keep accepted SOTA unchanged. No source rollback is needed because this is env-only.


Early-layer top3 probe result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T100937Z-20260703T100937Z-early-layer-top3-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`; otherwise accepted gate O_DIRECT config, `cpu_moe=40`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, strict cold 16GB cgroup.
- Metrics: `eval_tok_s=3.6`, `prompt_tok_s=1.5`, `TTFT=29979.512524 ms`, `elapsed_seconds=82.72`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15069327360`, `pgmajfault=311622`, `workingset_refault_file=3570771`, `ram_ok=true`, `ram_limit_killed=false`.
- Manual correctness: rejected. The answer was semantic for most of the paragraph but ended incomplete at `The country is also known`, so it does not satisfy the complete coherent France-output requirement.
- Gate/cache counters changed materially: one expert pack `hits=5290 misses=2086 direct_failures=0`; gate VRAM cache `hits=40507 misses=7376 hit_rate=84.6%`. Accepted SOTA has pack `hits=4623 misses=0` and gate cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Diagnosis: reducing all layers to top3 changed output trajectory, length, and gate-access pattern enough to increase cache misses/refault pressure. It also regressed token rate well below the `4.2 tok/s` promotion threshold.
- Verdict: rejected. Do not use global `0-39` top3 for accepted SOTA.


Narrow layer 0-2 top3 candidate design:

- Goal: test the smallest top-k change suggested by the fine trace, after global `0-39` top3 failed correctness and speed.
- Mechanism: the code supports one layer override on top of the default `GGML_MOE_KEEP_TOPK_UPDOWN`. Use `GGML_MOE_KEEP_TOPK_UPDOWN=3`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=3-9`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=4`. This means layers `0-2` and `10-39` use top3, while layers `3-9` stay at top4. It preserves the accepted `10-39` top3 policy and only changes layers `0-2` relative to SOTA.
- Theory and upper bound: layers `0-2` account for `4304.726 ms` (`16.0%`) of up/down fallback. Reducing those layers from top4 to top3 has a rough compute upper bound of `~1076 ms`, enough for a boundary test but likely smaller than the global top3 candidate.
- Risk: even layers `0-2` can influence the output trajectory and gate profile. Reject if France answer is incomplete or incoherent, if gate misses/direct behavior changes materially, if TTFT violates gate, if RAM exceeds 16GB, or if `eval_tok_s <= 4.2`.
- Acceptance: same SOTA gates as usual; no source rollback needed because this is env-only.


Narrow layer 0-2 top3 probe result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T101357Z-20260703T101357Z-narrow-layer0-2-top3-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: `GGML_MOE_KEEP_TOPK_UPDOWN=3`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=3-9`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=4`; otherwise accepted gate O_DIRECT config, strict cold 16GB cgroup.
- Metrics: `eval_tok_s=3.6`, `prompt_tok_s=1.5`, `TTFT=29360.514084 ms`, `elapsed_seconds=81.14`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15060041728`, `pgmajfault=329907`, `workingset_refault_file=4023978`, `ram_ok=true`, `ram_limit_killed=false`.
- Manual correctness: rejected. The answer was semantic but incomplete, ending at `and the`, so it fails the complete coherent France-output requirement.
- Gate/cache counters changed materially: one expert pack `hits=5077 misses=1619 direct_failures=0`; gate VRAM cache `hits=41184 misses=6696 hit_rate=86.0%`. This remains significantly different from the accepted SOTA trace shape.
- Diagnosis: even pruning only layers `0-2` changes the output trajectory enough to lengthen/truncate the answer and increase gate misses/refault pressure. The rough compute saving did not convert into token-rate improvement.
- Verdict: rejected. Deprioritize further top-k pruning on early layers unless a future candidate includes a stronger correctness-preserving routing argument and a prompt-set validation plan.

Next direction after top-k rejection:

- The fine trace plus pruning probes indicate that the remaining CPU fallback cost is real MXFP4 dot/source scan work. Correctness-preserving math/kernel changes are preferred over additional routing/top-k changes.
- Already rejected kernel/layout options: MXFP4 prefetch, hotset repack integration, compact mmap hotsets, broad one-stream up/down, down-batch cache under current VRAM budget.
- Before the next source candidate, inspect the current MXFP4 dot and `mul_mat_id` inner loop for a very narrow default-off optimization that changes scheduling or memory access without changing selected experts. If no plausible >1s upper bound can be calculated, do not run a full strict cold candidate.


Native MXFP4 multi-row dot microbench design:

- Source inspection: `ggml_compute_forward_mul_mat_id_one_chunk()` currently calls `vec_dot()` once per output row. For cne1=1 decode-like work, the same quantized activation row (`block_q8_0 * y`) is reused for many MXFP4 expert rows, but `ggml_vec_dot_mxfp4_q8_0()` reloads `y` for every row.
- Fine trace basis: cne1=1 decode-like up/down thread-sum is `423997.611 ms`, or about `21199.9 ms` at 20 threads. This dominates the remaining fallback path.
- Candidate idea: benchmark a native-layout multi-row MXFP4 dot that computes 4 rows for one shared `y` at a time. It should not require repacking, extra persistent RAM, changed routing, or changed selected experts. It only attempts to reuse loaded Q8 activation blocks and reduce per-row loop overhead.
- Theoretical upper bound: if sharing `y` and loop overhead improves the cne1=1 dot path by `S`, the decode-like bound is `21199.9*(1-1/S) ms`. A modest `1.10x` speedup gives about `1.93s`; `1.20x` gives about `3.53s`. This is enough to justify a microbench-first check.
- Practice: write a scratch microbench outside committed runtime source comparing current `ggml_vec_dot_mxfp4_q8_0()` row-wise calls against a prototype native multi-row implementation on representative up/down row counts. Verify max/mean abs against baseline.
- Acceptance for source implementation: proceed to a default-off source candidate only if the scratch benchmark shows at least `>=8%` warm compute speedup with exact or negligible numerical difference and no persistent memory requirement. Otherwise reject at microbench stage and do not run a full model candidate.


Native MXFP4 multi-row dot microbench result:

- Scratch source: `/tmp/native_mxfp4_multirow_bench.cpp`; binary: `/tmp/native_mxfp4_multirow_bench`. No repository runtime source was modified.
- Compile command: `g++ -O3 -march=native -std=c++17 -Iggml/include -Iggml/src -Iggml/src/ggml-cpu /tmp/native_mxfp4_multirow_bench.cpp -Lbuild-ds4-moe-stream/bin -lggml-cpu -lggml-base -Wl,-rpath,/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin -pthread -ldl -lm -o /tmp/native_mxfp4_multirow_bench`.
- First compile attempt failed because `ggml-impl.h` was incorrectly included inside `extern "C"`; the scratch source was fixed by including `ggml-impl.h` as C++ and wrapping only `quants.h` for C linkage.
- Results:
  - up-like `rows=2048 k=4096 iters=200`: baseline `93.1654 ms`, prototype `103.351 ms`, speedup `0.901x`, `max_abs=11924.5`, `mean_abs=2534.74`.
  - down-like `rows=4096 k=2048 iters=200`: baseline `95.2566 ms`, prototype `106.318 ms`, speedup `0.896x`, `max_abs=9282.04`, `mean_abs=1756.87`.
  - large `rows=4096 k=4096 iters=100`: baseline `91.7078 ms`, prototype `105.784 ms`, speedup `0.867x`, `max_abs=14492.8`, `mean_abs=2539.25`.
- Diagnosis: the quick native multi-row prototype failed both gates: it was slower than row-wise baseline and numerically wrong. The likely issue is that duplicating the x86 MXFP4 dot path outside the exact compiled runtime is fragile across VNNI/AVX feature paths and accumulation details; regardless, it provides no positive performance signal.
- Verdict: rejected at microbench stage. Do not implement this prototype in runtime source and do not run a full model candidate.

Next direction after native multi-row rejection:

- Do not continue ad hoc MXFP4 kernel rewrites without first building a correctness-identical unit test against the exact runtime feature path.
- Remaining plausible work is now instrumentation/feasibility, not blind source changes: either identify an existing tested kernel path that can be reused without repack/RAM cost, or prove a page-source/refault lever with a hard upper bound. Current accepted SOTA remains unchanged.


Current SOTA `--no-warmup` diagnostic design:

- Goal: test the only remaining low-risk no-source page/TTFT lever not yet recorded on the current `4.2 tok/s` O_DIRECT/gate-pack SOTA path.
- Existing support: `llama-cli --help` shows `--warmup, --no-warmup` with warmup enabled by default.
- Theory: warmup can touch model pages and CUDA paths before measured generation. Disabling it may lower TTFT and reduce cold file-cache churn inside the 16GB cgroup. It does not change model arithmetic, selected experts, gate cache policy, or CPU fallback math.
- Hard upper bound: `--no-warmup` cannot remove the measured `~27s` up/down fallback cost or gate expert pack reads. The only plausible gain is reduced warmup/page-cache side effects and maybe TTFT. Expect at most a small rounded-boundary move; accept only if it exceeds `4.2 tok/s` and all gates pass.
- Practice: run strict cold full France with accepted SOTA config plus CLI `--no-warmup`. Preserve `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate admission profile, gate O_DIRECT pack, top-k policy, threads, context, drop_caches, and 16GB cgroup.
- Acceptance: promote only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and gate pack/cache counters remain aligned with accepted SOTA. If it ties/regresses or any gate fails, reject. No source rollback is needed.


Current SOTA `--no-warmup` diagnostic result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T103043Z-20260703T103043Z-current-sota-no-warmup-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: added CLI `--no-warmup`; otherwise accepted gate O_DIRECT config, top-k policy, threads, context, drop_caches, and strict 16GB cgroup.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30784.612204 ms`, `elapsed_seconds=63.84`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15094276096`, `pgmajfault=272537`, `workingset_refault_file=1658322`, `ram_ok=true`, `ram_limit_killed=false`, `oom_seen=false`.
- Correctness: passed. France answer was semantic, coherent, and complete.
- Gate/O_DIRECT counters stayed aligned with accepted SOTA: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Diagnosis: disabling warmup did not improve token rate and TTFT was higher than the pre-candidate baseline guard (`29406.374764 ms`). It does not reduce the current CPU fallback or page/refault bottleneck enough to matter.
- Verdict: rejected/tie. Keep default warmup behavior for the accepted SOTA path.


Page/refault lever audit after no-warmup:

- Current accepted and guard runs already sit at the 16GB cgroup ceiling with about `15.0 GiB` file cache. Page-cache pressure is therefore real, but most simple page policy changes move cost between traced source time, TTFT, reclaim, and refaults rather than removing work.
- Already rejected/tied current-path page/source levers:
  - CPU fallback compact mmap top128/top256: worked mechanically but tied/regressed (`4.1` / `4.0`) because CPU math and sync remained dominant.
  - CPU `MADV_WILLNEED`: reduced major faults but regressed to `4.0 tok/s`, consistent with extra reclaim/prefetch pressure.
  - `--no-repack`: tied at `4.1 tok/s`, proving no hidden CPU_REPACK memory lever on the accepted path.
  - `--no-warmup`: tied at `4.1 tok/s` and worsened TTFT.
  - Down pack/iouring/staging probes: reduced short-run down staging in some cases but required reducing gate cache or had too little VRAM; full SOTA was unjustified.
  - Early top-k pruning: changed output trajectory, increased gate misses/refaults, and failed correctness/completeness.
- Historical page-policy evidence also rejects broad stream `DONTNEED=0`, op-wide stream `WILLNEED`, CPU fallback `DONTNEED_AFTER_EXPERT`, RAM-tier hotsets, and larger read-ahead/cache-size combinations under the same 16GB cgroup constraint.
- Conclusion: there is no remaining no-source or simple page-advice candidate with a defensible `>1s` upper bound and acceptable risk. Running more variants of the same page policy class is unlikely to move beyond `4.2 tok/s` and risks consuming time without new information.

Next required design step:

- Do not run another full strict-cold SOTA candidate until a microbench or short diagnostic proves a new mechanism.
- The next useful artifact should be a correctness-identical MXFP4 CPU kernel/unit-test harness that can compare any proposed dot/mul_mat_id variant against the exact runtime `ggml_vec_dot_mxfp4_q8_0()` path before model integration.
- The harness must use representative DS4 shapes, report max/mean abs, warm speed, compile flags, CPU feature path, and must reject any variant that is numerically non-identical or below the configured speedup threshold.
- Only after such a harness finds a real speedup should a default-off runtime source candidate be designed. Current accepted SOTA remains unchanged.


MXFP4 dot harness artifact design:

- Goal: make the microbench gate repeatable instead of relying on ad hoc `/tmp` snippets. This harness is not a runtime optimization and cannot promote SOTA by itself.
- Artifacts:
  - `.Agent/run-tools/mxfp4_dot_harness.cpp`: compares current row-wise `ggml_vec_dot_mxfp4_q8_0()` against the existing tested `ggml_gemv_mxfp4_8x8_q8_0()` repack kernel on DS4-like shapes.
  - `.Agent/run-tools/run_mxfp4_dot_harness.sh`: compiles the harness against `build-ds4-moe-stream/bin` and runs it, printing CPU flags and the exact compile command.
- Required output: per shape `repack_ms`, row-wise runtime, repack GEMV runtime, speedup, `max_abs`, `mean_abs`, and a checksum/sink to prevent dead-code elimination.
- Interpretation rule: a future runtime candidate is allowed only if a variant is numerically exact or within a documented tolerance and has enough warm speedup to justify a strict cold model run under the 16GB gates.


MXFP4 dot harness verification result:

- Harness command: `.Agent/run-tools/run_mxfp4_dot_harness.sh 200`.
- CPU feature path evidence: `/proc/cpuinfo` includes `avx2`, `fma`, `bmi1`, `bmi2`, `vaes`, and `vpclmulqdq`; no AVX512/VNNI flag is present. Compiler was `g++ 12.3.0` with `-O3 -march=native`.
- Compile command was printed by the script and links against `build-ds4-moe-stream/bin/libggml-cpu` and `libggml-base` with rpath set to the same build directory.
- Results:
  - up-like `k=4096 rows=2048 iters=200`: `repack_ms=3.951`, `row_ms=88.242`, `repack_gemv_ms=62.221`, warm speedup `1.418x`, `max_abs=0`, `mean_abs=0`, `sink=4.098e-07`.
  - down-like `k=2048 rows=4096 iters=200`: `repack_ms=2.920`, `row_ms=91.825`, `repack_gemv_ms=61.846`, warm speedup `1.485x`, `max_abs=0`, `mean_abs=0`, `sink=8.194e-07`.
- Interpretation: the existing tested 8x8 repack GEMV path is numerically identical to row-wise `ggml_vec_dot_mxfp4_q8_0()` for these DS4-like shapes and is materially faster in a warm microbench. The earlier ad hoc native multi-row prototype is therefore not the right implementation direction; reuse of the existing tested repack kernel is the only credible MXFP4 kernel path seen so far.
- Cold-start caveat: full-model hotset repack was historically rejected because persistent repack memory and cold conversion/read cost do not fit the current 16GB/page-cache/VRAM tradeoff. The harness result does not promote SOTA by itself. It only reopens a narrower design space: tiny, per-call or per-layer transient repack of the exact fallback rows, bounded by measured repack cost and without persistent host RAM growth.

Latest next-step plan after harness verification:

1. Design first, then execute: quantify a transient MXFP4 repack candidate before touching runtime source. Use the fine trace to estimate the maximum recoverable CPU fallback time and combine it with harness repack cost. A candidate is worth implementing only if the theoretical bound exceeds `1s` wall-clock and does not require additional persistent RAM inside the 16GB cgroup.
2. Candidate shape: default-off transient repack for CPU fallback up/down rows selected by `mul_mat_id`, reusing existing `ggml_gemv_mxfp4_8x8_q8_0()` only where rows are naturally grouped in multiples of 8 or can be safely batched without changing selected experts or arithmetic semantics.
3. Correctness gate before model run: add a short unit/microbench diagnostic that feeds the exact row groups through row-wise and transient-repack paths, requires `max_abs=0` or a documented bit-level explanation if not exact, and reports repack overhead separately from GEMV time.
4. Memory gate before model run: prove from allocation sizes that transient buffers are bounded and freed inside the op/layer. No persistent expert repack cache is allowed unless its size is explicitly budgeted under cgroup `memory.current <= 16000000000` including page cache.
5. Full strict-cold run only after gates pass: run France strict cold with the accepted SOTA config, `drop_caches`, `MemoryMax=16000000000`, O_DIRECT gate pack, same profile/pack hashes, and record output, TTFT, token rates, cgroup memory, page faults/refaults, gate counters, and pack counters.
6. Acceptance remains strict: only `eval_tok_s > 4.2`, correctness pass, RAM pass including page cache, TTFT within `+20%`, and no O_DIRECT fallback can replace current SOTA. A TTFT-over-gate speedup may be committed only as `not accepted` and cannot become SOTA.
7. Push discipline: when a compliant new SOTA appears, immediately commit source, plan, run metadata, command/env, hashes, and reproduction notes, then push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`. After push, clean rebuild from the pushed source and rerun strict cold before declaring it reproducible. Use git identity `L-Ark <fliangae@connect.ust.hk>`.
8. If the transient repack bound is weak or correctness/memory gates fail, reject without full-model trial and return to bottleneck profiling. Current accepted SOTA remains the `4.2 tok/s` cold-start record.


Transient down-only MXFP4 repack candidate design:

- Time: 2026-07-03 after commit `6608472d2`.
- Source basis: `ggml_compute_forward_mul_mat_id_one_chunk()` computes CPU fallback with one `vec_dot()` per output row. The existing tested `ggml_gemv_mxfp4_8x8_q8_0()` can compute 8 consecutive MXFP4 rows for the same Q8 activation vector with exact output in the harness.
- Bound script added: `.Agent/run-tools/analyze_transient_repack_bound.py` reads the complete CPU chunk trace from `/root/lfz/runs/vendor-ds4-16gb/20260703T100011Z-20260703T100011Z-cpu-fallback-fine-trace-limit2m/france-cpu40-vram0gb/cpu_chunk_trace.csv` and charges repack cost only to full 8-row groups; tail rows remain row-wise.
- Bound result from `.Agent/run-tools/analyze_transient_repack_bound.py --json-out /tmp/transient_repack_bound.json`:
  - `down:cne1=1`: eligible row fraction `0.9765625`, estimated net wall saving `686.77 ms` at 20 threads.
  - `down:cne1>1`: eligible row fraction `1.0`, estimated net wall saving `748.58 ms` at 20 threads.
  - down-only total: estimated net wall saving `1435.35 ms`.
  - up-only total: estimated net wall saving `-103.43 ms`, so up is explicitly excluded.
  - all up+down total: estimated net wall saving `1331.92 ms`, lower than down-only because up cne1=1 repack cost outweighs compute saving.
- Memory bound: DS4 down chunks in the trace are mostly `205` or `16` rows. For down `k=2048`, `nb=64` and `sizeof(block_mxfp4x8)=136`; the largest common `205`-row chunk repacks `200` rows, requiring `(200/8)*64*136 = 217600` bytes plus `800` bytes output per thread. Even if all 20 threads hit this path simultaneously, transient scratch is about `4.37 MB`, freed at chunk return and not persistent page cache. This is inside the 16GB cgroup budget.
- Runtime source candidate: add a default-off env gate `GGML_MOE_TRANSIENT_REPACK_DOWN=1`. When enabled, only `GGML_TYPE_MXFP4` tensors whose name contains `ffn_down_exps.weight` use transient 8-row repack inside CPU fallback. Full 8-row groups call `ggml_gemv_mxfp4_8x8_q8_0()`; tail rows continue using the original `vec_dot()` path. Default behavior remains unchanged when the env var is absent.
- Correctness expectation: arithmetic should be exact for full 8-row groups based on the harness (`max_abs=0`, `mean_abs=0`); tail rows are unchanged. The full France run still must pass semantic/coherence correctness because any ordering/layout bug would be visible in output.
- Build result: `cmake --build build-ds4-moe-stream -j 8 --target llama-cli` completed successfully with only existing warnings.
- Full practice command: run strict cold France with accepted SOTA config plus `--env GGML_MOE_TRANSIENT_REPACK_DOWN=1`, preserving `MemoryMax=16000000000`, `MemorySwapMax=0`, drop_caches, gate O_DIRECT pack, admission profile, top-k policy, and `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Acceptance: promote only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass. If it ties/regresses or output is wrong, reject and revert this runtime source candidate before pushing accepted source.


Transient down-only MXFP4 repack candidate result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T105753Z-20260703T-transient-down-repack-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: runtime source candidate enabled with `GGML_MOE_TRANSIENT_REPACK_DOWN=1`; otherwise accepted SOTA env/CLI, O_DIRECT gate pack, admission profile, top-k policy, drop_caches, and strict 16GB cgroup.
- Metrics: `eval_tok_s=3.4`, `prompt_tok_s=1.5`, `TTFT=29319.203769 ms`, `elapsed_seconds=85.81`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15051288576`, `pgmajfault=333912`, `workingset_refault_file=4373107`, `ram_ok=true`, `ram_limit_killed=false`.
- Gate/O_DIRECT counters changed materially: one expert pack `hits=5131 misses=1657 reads=5131 bytes=22866034688 direct_failures=0`; gate VRAM cache `hits=41080 misses=6788 hit_rate=85.8%`. This differs from accepted SOTA (`pack hits=4623 misses=0`, VRAM `hits=30528 misses=4623`), likely because the source change altered timing/page pressure enough to affect stream/cache behavior.
- Correctness: rejected by manual review. The answer was semantically normal but incomplete, ending at `It is a popular tourist destination, attracting millions of`, so it fails the complete coherent France-output requirement even though the heuristic set `correctness_ok=true`.
- Gap analysis: the harness warm speedup did not transfer to strict cold. The runtime candidate repacked inside every CPU fallback chunk, increasing source memory scans, stack scratch traffic, major faults/refaults, and gate pack/cache pressure. The theoretical model charged repack CPU time but underweighted cold page-cache side effects and interaction with the gate stream path. The changed cache counters and much higher refault count (`4.37M` vs about `1.66M` in the no-warmup tie and `~1.68M` in SOTA-like runs) explain the token-rate collapse.
- Verdict: rejected. It fails `eval_tok_s > 4.2` and manual correctness. The runtime source candidate was reverted immediately with `git restore ggml/src/ggml-cpu/ggml-cpu.c`, and `cmake --build build-ds4-moe-stream -j 8 --target llama-cli` was rerun. The current binary is rebuilt from clean source commit `6608472d2`, not the rejected runtime path.
- Keep artifact: `.Agent/run-tools/analyze_transient_repack_bound.py` remains useful as a repeatable bound checker, but the bound is now known to be insufficient unless it models cold page/refault interaction. Do not retry transient per-chunk repack without a stronger cold-source model or a design that avoids extra source scans.

Next direction after transient repack rejection:

- Current accepted SOTA remains `4.2 tok/s`; no source improvement was accepted.
- Do not pursue per-chunk transient repack for up or down in the current form. It has a narrow theoretical bound and a measured strict-cold regression to `3.4 tok/s`.
- Return to bottleneck localization with cold-aware evidence. The next candidate must reduce actual cold source reads/refaults or CPU fallback scheduling imbalance without adding another pass over expert rows.
- Plausible next diagnostic: compare accepted SOTA vs rejected transient run by `pgmajfault`, `workingset_refault_file`, pack hits/misses, and CPU fallback profile to quantify how much of the regression came from refault pressure versus arithmetic overhead. Only after that should another source candidate be designed.


Cold-aware comparison after transient repack rejection:

- Compared runs:
  - accepted SOTA `4.2`: `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`.
  - pre-candidate guard `4.1`: `/root/lfz/runs/vendor-ds4-16gb/20260703T095237Z-20260703T095236Z-pre-next-candidate-guard/france-cpu40-vram0gb`.
  - transient rejected `3.4`: `/root/lfz/runs/vendor-ds4-16gb/20260703T105753Z-20260703T-transient-down-repack-probe/france-cpu40-vram0gb`.
- Accepted SOTA: `eval_tok_s=4.2`, `prompt_tok_s=1.6`, `TTFT=28014.740620 ms`, `elapsed_seconds=60.89`, `pgmajfault=280239`, `workingset_refault_file=1674043`, pack `hits=4623 misses=0`, VRAM `hits=30528 misses=4623 hit_rate=86.8%`.
- Pre-candidate guard: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=29406.374764 ms`, `elapsed_seconds=62.51`, `pgmajfault=266453`, `workingset_refault_file=1678871`, pack `hits=4623 misses=0`, VRAM `hits=30528 misses=4623 hit_rate=86.8%`.
- Transient rejected: `eval_tok_s=3.4`, `prompt_tok_s=1.5`, `TTFT=29319.203769 ms`, `elapsed_seconds=85.81`, `pgmajfault=333912`, `workingset_refault_file=4373107`, pack `hits=5131 misses=1657`, VRAM `hits=41080 misses=6788 hit_rate=85.8%`.
- Interpretation: the regression is not a TTFT regression; TTFT stayed close to the guard. The failure is decode/steady work under cold cgroup pressure: elapsed time increased by about `23.3s` versus the pre-candidate guard, while file refaults increased by about `2.69M` and gate pack/cache counters shifted substantially. The transient repack path added extra source-row scans and scratch traffic that disrupted the gate stream/cache behavior and created more page churn than the warm microbench model predicted.
- Consequence for planning: any future CPU-kernel candidate must be cold-source neutral or reduce source reads. A warm arithmetic speedup is not enough under the 16GB page-cache constraint. Do not accept candidates whose cache counters diverge materially from the accepted SOTA unless the divergence is explained and total strict-cold metrics improve.
- Next bottleneck focus: preserve gate cache/pack hit shape while attacking CPU fallback. Candidate classes that add an extra pass over expert rows are deprioritized. Prefer scheduling/imbalance reductions, fewer CPU fallback calls, or moving work to already-loaded GPU/cache paths without increasing file refault pressure.


CPU fallback cne1 overpartition candidate design:

- Time: 2026-07-03 after rejected transient repack comparison commit `987d3ac7e`.
- Bottleneck basis: fine trace shows cne1=1 CPU fallback dominates thread-sum: `up:cne1=1 213850.057 ms`, `down:cne1=1 210147.554 ms`, about `423997.611 ms` thread-sum or `~21199.9 ms` at 20 threads. The cold comparison shows candidates that add source scans increase refaults and fail; the next candidate must not add scans.
- Source finding: in `ggml_compute_forward_mul_mat_id()`, when `nr1 == 1`, `chunk_size` starts at `64`, giving natural chunk counts of about `32` for up (`nr0=2048`) and `64` for down (`nr0=4096`). The current heuristic then collapses any `nchunk0*nchunk1 < nth*4` to exactly `nth` chunks, so cne1=1 often becomes only 20 large chunks. With `nth >= total_chunks`, each thread takes at most one chunk and there is no work stealing for page-fault stragglers.
- Candidate: add default-off env `GGML_MOE_CPU_OVERPARTITION_CNE1=1`. When enabled and `nr1 == 1`, skip the collapse-to-`nth` heuristic and keep the natural 64-row chunks. This changes only scheduling granularity; selected experts, arithmetic, source reads, gate cache, O_DIRECT pack, and memory layout are unchanged.
- Theoretical upper bound: no arithmetic speedup is expected. The only recoverable time is scheduling/page-fault tail. If improved granularity recovers just `5%` of the `~21199.9 ms` cne1=1 wall estimate, the bound is about `1.06s`; `10%` would be about `2.12s`. This is enough to justify one strict cold candidate because it does not add source scans and should not increase page-cache pressure materially.
- Risk: more chunks increase atomic scheduling overhead. However the candidate changes from 20 to roughly 32/64 chunks per cne1=1 op, not thousands, so overhead should be small compared with cold page-fault variance. If token rate does not improve, reject and revert.
- Practice: implement default-off env gate, rebuild, run strict cold France with accepted SOTA config plus `GGML_MOE_CPU_OVERPARTITION_CNE1=1`. Accept only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and gate pack/cache counters remain close to accepted SOTA.


CPU fallback cne1 overpartition candidate result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T110856Z-20260703T-cne1-overpartition-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: runtime source candidate enabled with `GGML_MOE_CPU_OVERPARTITION_CNE1=1`; otherwise accepted SOTA env/CLI, O_DIRECT gate pack, admission profile, top-k policy, drop_caches, and strict 16GB cgroup.
- Metrics: `eval_tok_s=3.9`, `prompt_tok_s=1.5`, `TTFT=30606.713915 ms`, `elapsed_seconds=65.10`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15103713280`, `pgmajfault=440429`, `workingset_refault_file=1697965`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: passed by manual review. The France answer was complete, semantic, and coherent.
- Gate/O_DIRECT counters stayed aligned with accepted SOTA: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_failures=0`; gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Diagnosis: preserving gate/cache shape was successful, but finer CPU chunking did not improve strict-cold wall time. Major faults increased (`440429` vs accepted `280239` and guard `266453`), and the added scheduling/atomic overhead plus more fault interleaving outweighed any straggler reduction. This disproves the simple overpartition lever under the current 16GB cgroup.
- Verdict: rejected. It fails `eval_tok_s > 4.2`. The runtime source candidate was reverted with `git restore ggml/src/ggml-cpu/ggml-cpu.c`, and `cmake --build build-ds4-moe-stream -j 8 --target llama-cli` was rerun from clean source commit `987d3ac7e`.

Next direction after overpartition rejection:

- Current accepted SOTA remains `4.2 tok/s`; no source improvement was accepted.
- Avoid CPU fallback changes that increase page faults even if they improve scheduling theory. Under 16GB, page-fault behavior dominates small CPU scheduling wins.
- Remaining viable directions need a stronger mechanism than per-chunk scheduling or per-chunk repack: reduce misses before CPU fallback, move a bounded additional expert subset into VRAM without causing OOM, or change gate/admission so pack/cache counters remain aligned while CPU fallback calls decrease.


Gate-only VRAM cache +32 slots candidate design:

- Time: 2026-07-03 after overpartition rejection commit `f38603e26`.
- Bottleneck focus: do not add CPU source scans or up/down one-stream traffic. Current accepted gate path is stable but has `3313` gate admission entries and only `3192` cache slots at `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, with final counters `hits=30528 misses=4623 hit_rate=86.8%`.
- VRAM basis: accepted SOTA stderr reports `VRAM cache: 13.2 GiB, 3192 slots (4.25 MiB each)` and CUDA0 free `238 MiB` at shutdown. Adding `32` slots costs about `136 MiB`, leaving a small safety margin without changing host RAM/page-cache behavior.
- Candidate: no source change. Increase only `GGML_MOE_STREAM_ONE_CACHE_MIB` from `13568` to `13704`. Keep `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, current gate admission profile, O_DIRECT gate expert pack, top-k policy, `cpu_moe=40`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, drop_caches, and strict 16GB cgroup.
- Theoretical upper bound: this can only reduce gate cache evictions/misses. If the extra 32 slots eliminate a small fraction of the `4623` misses, the possible gain is limited to avoided O_DIRECT pack reads/H2D for those misses; it cannot reduce CPU up/down fallback. This is a low-risk no-source probe because it uses otherwise free VRAM and should preserve correctness.
- Risk: CUDA OOM or lower free VRAM may increase allocator pressure. If cudaMalloc fails, counters change unexpectedly, TTFT rises, or token rate does not exceed `4.2`, reject and keep `13568`.
- Acceptance: promote only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, and pack/cache counters show no direct failures. If accepted, record exact cache slot count/hash/run and push immediately; if rejected, record as env-only diagnostic.


Gate-only VRAM cache +32 slots candidate result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T111601Z-20260703T-gate-cache13704-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: `GGML_MOE_STREAM_ONE_CACHE_MIB=13704` instead of `13568`; otherwise accepted gate-only one-stream/O_DIRECT pack config, top-k policy, `cpu_moe=40`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, drop_caches, and strict 16GB cgroup.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=30285.215928 ms`, `elapsed_seconds=63.36`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15094198272`, `pgmajfault=270283`, `workingset_refault_file=1661147`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: passed manual review. France answer was complete, semantic, and coherent.
- VRAM/cache: cache initialized as `13.4 GiB, 3224 slots (4.25 MiB each)`, CUDA0 free dropped from accepted `238 MiB` to `102 MiB`. Gate pack counters stayed direct-failure-free: `hits=4615 misses=0 reads=4615 direct_failures=0`; VRAM cache `hits=30536 misses=4615 hit_rate=86.9%`.
- Diagnosis: adding 32 slots reduced misses by only `8` (`4623 -> 4615`) and did not improve token rate. The remaining 102MiB free VRAM leaves little safety margin, so larger cache sizes have low upside and higher OOM/allocator-pressure risk.
- Verdict: rejected/tie. Keep `GGML_MOE_STREAM_ONE_CACHE_MIB=13568` as the accepted configuration. Do not continue gate-cache-size expansion unless a future profile shows materially larger miss reduction per slot.

Next direction after gate-cache expansion rejection:

- Current accepted SOTA remains `4.2 tok/s`; no source/config improvement was accepted.
- The stable gate path is already near its useful VRAM limit. Remaining work should focus on reducing CPU fallback calls or finding a correctness-preserving routing/cache change that does not alter gate cache shape or increase page faults.


Late-layer top2 second-range candidate design:

- Time: 2026-07-03 after gate-cache-size rejection commit `b49652523`.
- Bottleneck basis: complete CPU chunk trace aggregates fallback thread-sum for layers `30-39` as `103879.400 ms`, or about `5193.970 ms` at 20 threads. Current accepted config already uses top3 for layers `10-39`; changing only layers `30-39` from top3 to top2 can at most remove about one third of that late-layer fallback, a rough wall upper bound of `~1.73s` before output-trajectory effects.
- Prior boundary: early-layer pruning failed correctness/completeness, so this probe must be late-layer-only. Existing env supports only one layer range, so it cannot express `10-29 top3 + 30-39 top2` without a default-off second range.
- Source candidate: add optional env `GGML_MOE_KEEP_TOPK_LAYER2_RANGE` and `GGML_MOE_KEEP_TOPK_LAYER2_VALUE` to `ggml_moe_keep_topk_for_tensor()`. Range2 is checked before the existing range. If env is unset, default behavior is identical.
- Practice config: keep accepted SOTA env plus existing `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, and add `GGML_MOE_KEEP_TOPK_LAYER2_RANGE=30-39`, `GGML_MOE_KEEP_TOPK_LAYER2_VALUE=2`.
- Risk: this changes selected experts and can change answer semantics/length. The France output must be manually reviewed; heuristic correctness alone is insufficient.
- Acceptance: promote only if `eval_tok_s > 4.2`, RAM/correctness/TTFT/O_DIRECT gates pass, answer is complete/coherent, and gate pack/cache counters remain direct-failure-free. If it ties, regresses, truncates, or drifts semantically, revert source and record rejection.


Late-layer top2 second-range candidate result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T112234Z-20260703T-late-layer30-39-top2-probe/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: temporary default-off second range enabled with `GGML_MOE_KEEP_TOPK_LAYER2_RANGE=30-39`, `GGML_MOE_KEEP_TOPK_LAYER2_VALUE=2`; otherwise accepted SOTA config with layers `10-39` top3, gate O_DIRECT pack/cache, strict 16GB cgroup, and same CLI/thread settings.
- Metrics: `eval_tok_s=3.5`, `prompt_tok_s=1.7`, `TTFT=28826.576851 ms`, `elapsed_seconds=82.71`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15063646208`, `pgmajfault=319186`, `workingset_refault_file=3963195`, `ram_ok=true`, `ram_limit_killed=false`.
- Gate/cache counters changed badly: one expert pack `hits=5444 misses=1838 reads=5444 bytes=24260902912 direct_failures=0`; gate VRAM cache `hits=40590 misses=7282 hit_rate=84.8%`.
- Correctness: rejected by manual review. The answer was mostly semantic but contained an error in the French motto (`Fratinité`) and ended without a complete closing sentence. This fails the complete coherent France-output requirement.
- Diagnosis: the estimated late-layer CPU fallback reduction did not translate into speed. Changing late-layer routing changed the generation/cache trajectory, increased gate misses/refaults, and produced worse output. This is the same failure pattern as other pruning attempts, so additional top-k pruning is deprioritized.
- Verdict: rejected. It fails speed and manual correctness. The runtime source candidate was reverted with `git restore ggml/src/ggml-cpu/ggml-cpu.c`, and `cmake --build build-ds4-moe-stream -j 8 --target llama-cli` was rerun from clean source commit `b49652523`.

Next direction after late-layer pruning rejection:

- Current accepted SOTA remains `4.2 tok/s`.
- Do not continue top-k pruning unless a future candidate includes a stronger correctness-preserving routing model and prompt-set validation before full SOTA claims.
- Remaining optimization space is now very narrow: accepted gate cache is near optimal, CPU scheduling/repack/page-policy/top-k/updown one-stream have all been rejected. Next work should be measurement-first, not another source change: identify any untested no-source configuration with a hard upper bound, or produce a new bottleneck trace under the exact current clean binary to see whether the repeated line has shifted.


Current clean SOTA profile guard design:

- Time: 2026-07-03 after late-layer pruning rejection commit `1cb55fc5f`.
- Goal: stop blind source/config changes and refresh bottleneck evidence under the current clean binary after several rejected candidates.
- Source state: clean runtime source; no candidate patch. Accepted SOTA config is used unchanged except enabling CPU MoE profiling envs.
- Config delta: add `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`, and `GGML_KIMI_CPU_MOE_ELIGIBILITY_PROFILE=1` to the accepted gate O_DIRECT SOTA config. Preserve cache `13568`, gate name filter, admission profile, expert pack, top-k policy, `cpu_moe=40`, strict 16GB cgroup, and full France prompt.
- Purpose: record whether remaining wall time is still dominated by CPU up/down fallback, rejected one-stream eligibility, or another bucket. This run is diagnostic and cannot replace SOTA unless it unexpectedly exceeds `4.2` while all gates pass.
- Acceptance/handling: if it ties/regresses, record as a profile guard only. If it unexpectedly improves above `4.2`, rerun without profiling overhead before any SOTA claim.


Current clean SOTA profile guard result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T112700Z-20260703T-current-clean-profile-guard/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: profiling envs `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`, `GGML_KIMI_CPU_MOE_ELIGIBILITY_PROFILE=1`; otherwise accepted SOTA config.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30186.611077 ms`, `elapsed_seconds=63.18`.
- RAM/cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15099514880`, `pgmajfault=272573`, `workingset_refault_file=1653573`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: passed manual review. France answer was complete, semantic, and coherent.
- Gate path stayed exactly aligned with accepted SOTA: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_failures=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- CPU MoE profile: `calls=16920 total=2.013 ms/call`, with `cuda_single=0.419 ms/call` and `fallback_t0=1.581 ms/call`. The top fallback names are still early up/down tensors: `blk.1.ffn_down_exps`, `blk.1.ffn_up_exps`, `blk.2.ffn_down_exps`, `blk.0.ffn_up_exps`, `blk.0.ffn_down_exps`, and `blk.2.ffn_up_exps`, each around `5.1-5.5 ms/call` total/fallback.
- Interpretation: current bottleneck remains CPU up/down fallback, especially early layers. This also explains why late-layer top2 and gate cache expansion did not help. Earlier attempts to prune early layers failed correctness/completeness, so any future early-layer reduction requires a stronger correctness model or broader prompt-set validation before it can be trusted.
- Status: diagnostic only. It ties the repeated strict-cold line and does not replace the accepted `4.2 tok/s` SOTA.

Next direction after profile guard:

- Do not start another source candidate without a new mechanism that can attack early-layer up/down fallback while preserving output quality and gate/cache counters.
- Plausible next work is offline analysis of routing/output sensitivity: compare accepted vs failed pruning traces and identify whether a safer per-layer/per-expert pruning rule exists. Without that, continue treating current `4.2 tok/s` as the effective cold-start SOTA.


Offline MoE name-profile sensitivity analysis:

- Artifact: `.Agent/run-tools/analyze_moe_name_profile.py` parses `kimi_cpu_moe_name_profile` lines from a run stderr and aggregates visible fallback by layer/kind/band.
- Input: `/root/lfz/runs/vendor-ds4-16gb/20260703T112700Z-20260703T-current-clean-profile-guard/france-cpu40-vram0gb/stderr.txt`.
- Output: `/root/lfz/runs/vendor-ds4-16gb/20260703T112700Z-20260703T-current-clean-profile-guard/france-cpu40-vram0gb/moe_name_profile_analysis.json`.
- Parsed rows: top40 name-profile entries.
- Visible top-row fallback total: `15940.896 ms`, split `down=9037.536 ms`, `up=6903.360 ms`.
- Visible fallback by layer band:
  - layers `0-2`: `4449.396 ms`.
  - layers `3-9`: `5626.464 ms`.
  - layers `10-19`: `2469.615 ms`.
  - layers `20-29`: `1129.128 ms`.
  - layers `30-39`: `2266.293 ms`.
- Interpretation: the expensive visible fallback is heavily concentrated in early layers `0-9` (`~10.08s` of the top40 visible fallback). This matches the profile guard and explains why late-only pruning/cache work had little upside.
- Constraint: early-layer top-k/routing changes have already failed output completeness/correctness in strict France runs. Therefore the biggest remaining theoretical savings are also the highest correctness risk.
- Decision: do not start another pruning/top-k source candidate without a routing-sensitivity trace that proves which early-layer expert removals are semantically safe. Current evidence is insufficient to justify another full strict-cold candidate.

Next required evidence before another source candidate:

- Add or reuse a default-off routing trace that records selected expert ids/ranks per layer/token for accepted and rejected pruning runs, or otherwise collect equivalent routing evidence.
- Compare accepted output path versus failed early/late pruning paths to identify whether any per-layer/per-expert rule can reduce CPU fallback while preserving France output and a small prompt set.
- Until that evidence exists, current `4.2 tok/s` remains the effective vendor DeepSeek cold-start SOTA under the 16GB/page-cache/TTFT/correctness gates.


Temporary routing trace diagnostic design:

- Goal: collect routing evidence before any further pruning/top-k source candidate. Current profile shows early up/down fallback is the bottleneck, but prior early pruning broke output correctness/completeness.
- Source handling: use a temporary default-off instrumentation patch only. Do not keep this runtime source change after the diagnostic; revert and rebuild clean before committing records.
- Trace env: `GGML_MOE_ROUTE_TRACE_OUT={case_dir}/route_trace.csv`, optional limit `GGML_MOE_ROUTE_TRACE_LIMIT`.
- Trace content: `seq,tensor,layer,token,rank,expert,pruned`, emitted in the `ith==0` routing loop before CPU fallback. It records selected expert ids and whether the current top-k policy prunes that rank for up/down.
- Practice: run accepted SOTA config with route trace enabled under strict cold 16GB cgroup. The run is diagnostic only; because file I/O can perturb timing, it cannot replace SOTA even if rounded token rate looks good.
- Analysis target: compare accepted route distribution and pruned ranks against failed pruning runs or future prompt-set traces. A future pruning candidate must be justified by this route evidence before execution.


Temporary routing trace diagnostic result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T113716Z-20260703T-route-trace-accepted-diagnostic/france-cpu40-vram0gb`.
- Config delta from accepted SOTA: temporary route-trace source patch plus `GGML_MOE_ROUTE_TRACE_OUT={case_dir}/route_trace.csv`, `GGML_MOE_ROUTE_TRACE_LIMIT=2000000`; otherwise accepted SOTA config.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30367.496493 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15093653504`, `pgmajfault=272872`, `workingset_refault_file=1652061`, `ram_ok=true`, `ram_limit_killed=false`.
- Gate counters stayed aligned with accepted SOTA: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_failures=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Trace artifact: `route_trace.csv`, `109441` lines including header; analysis artifact: `route_trace_analysis.json` in the same run directory.
- Route distribution: each of `gate`, `up`, and `down` has `36480` route records. Current top-k policy keeps `19760` up/down records and prunes `16720` up/down records. Gate records are all kept.
- Band-level up/down kept/pruned counts:
  - layers `0-2`: kept `1824` per kind, pruned `912` per kind.
  - layers `3-9`: kept `4256` per kind, pruned `2128` per kind.
  - layers `10-39`: kept `4560` per 10-layer band per kind, pruned `4560` per 10-layer band per kind.
- Top route-frequency experts are mostly later layers, e.g. layer `37` expert `162`, layer `27` expert `152`, layer `24` expert `12`, layer `32` expert `124`. This shows route frequency alone does not match the early-layer fallback cost profile; early layers are expensive because each call is costlier, not because they dominate route count.
- Interpretation: future pruning cannot rely only on route frequency. It must combine per-layer fallback cost, rank/pruned status, and output sensitivity. The accepted run already prunes many ranks, but further pruning in either early or late layers has repeatedly changed output/cache trajectory and failed speed or correctness.
- Source handling: the route-trace source patch was reverted with `git restore ggml/src/ggml-cpu/ggml-cpu.c`; `cmake --build build-ds4-moe-stream -j 8 --target llama-cli` was rerun from clean source commit `87f1a7a98`.

Next direction after routing trace:

- Current evidence does not justify another top-k/pruning candidate. The biggest cost is early-layer up/down fallback, but early-layer changes are correctness-sensitive and route frequency is not enough to identify safe removals.
- A defensible next candidate would need a prompt-set route/correctness study or a mechanism that reduces early-layer compute without changing selected experts. Without that new mechanism, current `4.2 tok/s` remains the effective accepted cold-start SOTA.


Prompt-set cold SOTA portability study:

- Time: 2026-07-03 after commit `8f9bc9a3d`.
- Goal: test whether the current accepted France SOTA path is stable across a small prompt set before using prompt-specific routing/cache evidence for another optimization.
- Tooling update: `.Agent/run-tools/strict_ds4_runner.py` now accepts `--prompt` and `--case-name`, preserving the same strict 16GB cgroup, drop-caches, memory summary, command/env capture, and answer extraction flow. This is runner/tooling only; runtime source was not changed.
- Prompt set:
  - France: `Please introduce France in a short paragraph.`
  - Quantum: `Explain quantum computing briefly.`
  - Fibonacci: `Write a short Python function for Fibonacci.`
  - Japan: `Introduce Japan in a short paragraph.`
  - Climate: `Summarize climate change in one paragraph.`
- Config: current accepted SOTA env unchanged: `cpu_moe=40`, `vram_cache=0`, gate one-stream only, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, France gate admission profile, France gate expert pack with O_DIRECT, accepted top-k policy, `-c 256 -b 16 -ub 16`, strict cold `drop_caches`, and `MemoryMax=16000000000`.
- Summary artifact: `.Agent/runs/20260703-vendor-ds4-coldstart/promptset-current-sota-cold-summary.json`. Full run summaries are under `/root/lfz/runs/vendor-ds4-16gb/20260703T114755Z-20260703_promptset_current_sota_cold_france`, `...114903Z...quantum`, `...115132Z...fibonacci`, `...115412Z...japan`, and `...115534Z...climate`.

Prompt-set cold SOTA portability result:

- France: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=28867.812469 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15090991104`, pack `hits=4623 misses=0`, VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`, correctness heuristic pass and manual semantic check pass.
- Quantum: `eval_tok_s=1.7`, `prompt_tok_s=1.3`, `TTFT=30261.42291 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15007244288`, pack `hits=7522 misses=18330`, VRAM cache `hits=21793 misses=25852 hit_rate=45.7%`, correctness heuristic pass.
- Fibonacci: `eval_tok_s=1.6`, `prompt_tok_s=1.3`, `TTFT=32572.315251 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14993739776`, pack `hits=7910 misses=23381`, VRAM cache `hits=16645 misses=31291 hit_rate=34.7%`, correctness heuristic pass. Output includes Python code, but manual review should be stricter before using this prompt for acceptance because the generated compact list variant is not ideal style.
- Japan: `eval_tok_s=2.9`, `prompt_tok_s=1.4`, `TTFT=30494.691309 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15038717952`, pack `hits=5026 misses=3861`, VRAM cache `hits=25556 misses=8887 hit_rate=74.2%`, correctness heuristic pass.
- Climate: `eval_tok_s=2.2`, `prompt_tok_s=1.4`, `TTFT=31588.658273 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15043727360`, pack `hits=5897 misses=9019`, VRAM cache `hits=24329 misses=14916 hit_rate=62.0%`, correctness heuristic pass.
- Interpretation: the accepted France SOTA is prompt-specialized. It is RAM-compliant across the prompt set, but non-France prompts trigger many France-pack misses and much lower VRAM cache hit rates, causing `1.6-2.9 tok/s` instead of the France `4.1-4.2 tok/s` line.
- Decision: do not claim prompt-invariant `4.x tok/s`. Current accepted SOTA remains a France cold-start SOTA under the specified acceptance prompt.

Prompt-set one-trace route/cache evidence:

- Diagnostic config: same prompt-set/SOTA config plus `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`. This uses existing default-off trace support in `ggml/src/ggml-cuda/moe_stream.cu`; no runtime source patch was added.
- Trace runs: `/root/lfz/runs/vendor-ds4-16gb/20260703T120439Z-20260703_promptset_current_sota_cold_trace_france`, `...120548Z...quantum`, `...120820Z...fibonacci`, `...121059Z...japan`, `...121223Z...climate`.
- Trace metrics remained RAM-compliant: France `4.1`, Quantum `1.6`, Fibonacci `1.6`, Japan `2.8`, Climate `2.2` tok/s. These are diagnostic only because trace writing perturbs the run.
- Union analysis artifact: `.Agent/runs/20260703-vendor-ds4-coldstart/promptset-one-trace-union-analysis.json`.
- Per-prompt unique cold gate miss pairs: France `4599`, Quantum `5549`, Fibonacci `5879`, Japan `4654`, Climate `5162`.
- Union cold gate miss pairs across all five prompts: `8621` entries, estimated payload `38419038208 bytes` (`35.7805 GiB`).
- Existing historical union pack already matches this scale: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-promptset-gate-union-firstorder-20260702.pack` is `38420348928 bytes`; historical union admission profiles are in `/root/lfz/runs/vendor-ds4-16gb/20260702T114356Z-20260702_promptset_union_trace_inputs/union-analysis/`.
- Historical union-pack cold tests were reviewed rather than rebuilt because root disk has only about `7.1 GiB` free. The existing pack is enough for evidence.

Union-pack historical result review:

- Top6000 union profile run: `/root/lfz/runs/vendor-ds4-16gb/20260702T120943Z-20260702_promptset_union_pack_top6000_cold`.
  - Quantum `2.2`, Fibonacci `2.0`, Japan `2.8`, Climate `2.5`, France `3.1` tok/s; all listed summaries were RAM/correctness heuristic pass.
- Top5000 union profile run: `/root/lfz/runs/vendor-ds4-16gb/20260702T122316Z-20260702_promptset_union_pack_top5000_cold`.
  - Quantum `2.1`, Fibonacci `2.1`, Japan `2.8`, Climate `2.5`, France `3.2` tok/s; all listed summaries were RAM/correctness heuristic pass.
- Interpretation: union pack/profile improves the worst non-France prompts compared with the France-only pack, but it materially regresses the France acceptance prompt from `4.1-4.2` to `3.1-3.2`. Therefore it is rejected as a replacement for the current accepted SOTA.
- Bottleneck conclusion after prompt-set study: gate pack/admission can explain prompt portability, but it does not create a France SOTA candidate. For the acceptance prompt, the remaining bottleneck is still CPU up/down fallback, especially early layers `0-9`.

Next direction after prompt-set study:

- Do not spend more time on broad union gate packs for the France SOTA path unless the objective changes to average prompt-set performance.
- A valid next France-SOTA candidate must reduce early-layer up/down CPU fallback without changing selected experts or materially increasing page refaults. Candidate classes: CPU fallback microkernel/scheduling improvements, reducing per-call overhead for early `ffn_up_exps`/`ffn_down_exps`, or a correctness-preserving GPU assist for up/down rows.
- Before implementing such a source candidate, design must include a hard upper bound from measured early-layer fallback time (`~10.08s` visible top40 fallback, `~2.013 ms/call` CPU MoE average with `~1.581 ms/call` fallback component) and a rollback rule if France output, RAM, TTFT, or counters regress.


CPU chunk trace design for next France-SOTA candidate:

- Time: 2026-07-03 after prompt-set portability commit `e6105dd7c`.
- Goal: quantify whether early up/down fallback bottleneck is due to scheduling imbalance, per-thread work distribution, or per-chunk compute/memory cost.
- Tooling: added `.Agent/run-tools/analyze_cpu_chunk_trace.py` to aggregate `GGML_MOE_CPU_CHUNK_TRACE_OUT` CSV by kind, layer band, tensor, and CPU thread.
- Config: accepted France SOTA config unchanged plus `GGML_MOE_CPU_CHUNK_TRACE_OUT={case_dir}/cpu_chunk_trace.csv` and `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=500000`. This is diagnostic only because trace writing perturbs timing and the 500k-row limit truncates the tail.
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T122329Z-20260703_cpu_chunk_trace_current_sota_france/france-cpu40-vram0gb`.
- Versioned artifacts: `.Agent/runs/20260703-vendor-ds4-coldstart/cpu-chunk-trace-current-sota-summary.json` and `.Agent/runs/20260703-vendor-ds4-coldstart/cpu-chunk-trace-current-sota-analysis.json`.

CPU chunk trace result:

- Diagnostic run metrics: `eval_tok_s=4.0`, `prompt_tok_s=1.6`, `TTFT=28549.934553 ms`, `elapsed_seconds=62.17`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15105380352`, `pgmajfault=258746`, `workingset_refault_file=1664574`, `ram_ok=true`, correctness heuristic/manual France output pass.
- Gate path stayed aligned: expert pack `hits=4623 misses=0 direct_failures=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Trace rows captured: `500000`, exactly the configured cap, so totals are lower-bound/partial-run diagnostics.
- Sum of chunk wall times: `331934.650 ms`, split `down=173564.403 ms` over `277048` rows and `up=158370.247 ms` over `222952` rows.
- Parallel critical-path approximation by CPU thread: max thread `17469.732 ms`, median thread `16561.1175 ms`. The imbalance between median and max is only about `0.91s`, so scheduler/split-pool improvements have a hard upper bound below one second for this captured portion.
- Band totals in captured rows: layers `0-2` `55516.103 ms`, `3-9` `71964.752 ms`, `10-19` `70037.269 ms`, `20-29` `67246.542 ms`, `30-39` `67169.984 ms`.
- Top tensors by captured chunk time remain early-heavy: `blk.0.ffn_up_exps.weight` `10401.987 ms`, `blk.0.ffn_down_exps.weight` `9975.936 ms`, `blk.2.ffn_down_exps.weight` `9624.730 ms`, `blk.1.ffn_down_exps.weight` `9064.312 ms`, `blk.2.ffn_up_exps.weight` `8456.973 ms`, `blk.1.ffn_up_exps.weight` `7992.165 ms`.

Next direction after CPU chunk trace:

- Do not prioritize split-pool or thread scheduling; the measured imbalance is too small to produce a new France SOTA by itself.
- A plausible source candidate must reduce per-chunk up/down cost. The hard bound from this trace is approximately:
  - `10%` faster CPU fallback critical path saves `~1.75s`.
  - `20%` faster saves `~3.49s`.
  - `30%` faster saves `~5.24s`.
  - `50%` faster saves `~8.73s`.
- Given a diagnostic elapsed time of `62.17s`, a 20-30% fallback improvement is the first range likely to move rounded token rate beyond the current `4.2` SOTA. Smaller scheduling-only changes are unlikely to matter.
- Next implementation candidate should inspect the existing CPU MoE up/down matvec path and look for a low-risk per-chunk speedup that does not alter selected experts, top-k policy, gate cache, or O_DIRECT pack behavior. Any candidate must be rebuilt and strict-cold rerun with the accepted France prompt before promotion.


MoE CPU chunk-size candidate design:

- Time: 2026-07-03 after CPU chunk bottleneck trace commit `8e5153409`.
- Hypothesis: the current MoE CPU fallback uses fixed chunk size `16` (`64` only when `nr0==1 || nr1==1`). The CPU chunk trace hit the `500000` row cap and showed a balanced work-stealing distribution, so scheduler imbalance is not the main issue. However, very small chunks can still amplify atomic fetch, function-call, row-mapping, and loop overhead. Increasing MoE fallback chunk size may reduce per-chunk overhead without changing selected experts or math.
- Source candidate: add default-off env `GGML_MOE_CPU_CHUNK_SIZE`. When unset, behavior is byte-for-byte intended to match the previous chunk-size policy. When set to `32` or `64`, only MoE fallback chunk partitioning changes; ordinary matmul chunking is untouched. Both `ggml_compute_forward_mul_mat_id` and the existing fused up/gate fallback path use the helper for consistency.
- Theoretical upper bound: this cannot exceed the CPU fallback critical path from chunk trace (`~17.47s` captured max-thread lower bound). Since measured thread imbalance is only `~0.91s`, the realistic target is lower than a full microkernel rewrite. If chunk overhead is 5-10% of fallback time, expected save is `~0.9-1.7s`, likely at most a tie/slight improvement. If larger chunks hurt parallelism/cache locality, token rate will regress.
- Experiment order: rebuild, run strict-cold France with `GGML_MOE_CPU_CHUNK_SIZE=32`; if it is not clearly worse and remains correct/RAM-safe, test `64`. Preserve all accepted SOTA envs and 16GB cgroup. Do not promote unless `eval_tok_s > 4.2`, France answer passes, TTFT gate passes, and pack/cache counters remain aligned.
- Rollback: if both chunk-size values tie/regress or change output/RAM/TTFT/counters, revert `ggml/src/ggml-cpu/ggml-cpu.c`, rebuild clean, record rejection, and push docs only.


MoE CPU chunk-size candidate result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T123416Z-20260703_moe_cpu_chunk32_probe/france-cpu40-vram0gb`.
- Source delta during run: temporary `GGML_MOE_CPU_CHUNK_SIZE` override in `ggml/src/ggml-cpu/ggml-cpu.c`, tested with `GGML_MOE_CPU_CHUNK_SIZE=32`. Runtime source was reverted immediately after rejection with `git restore ggml/src/ggml-cpu/ggml-cpu.c` and `cmake --build build-ds4-moe-stream -j 8 --target llama-cli` from clean commit `8e5153409`.
- Metrics: `eval_tok_s=3.5`, `prompt_tok_s=1.6`, `TTFT=27902.340278 ms`, `elapsed_seconds=66.17`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15096217600`, `pgmajfault=533760`, `workingset_refault_file=1674932`, `ram_ok=true`, correctness pass.
- Gate counters stayed aligned with accepted SOTA: pack `hits=4623 misses=0 direct_failures=0`, VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Diagnosis: larger chunk size did not change gate/cache behavior but doubled major faults and slowed decode from the repeated `4.1` line to `3.5`. This means the fixed chunk size `16` is not the bottleneck; larger chunks likely harm locality/page-fault overlap more than they reduce atomic/function overhead.
- Verdict: rejected. Do not test `64`; `32` is already clearly worse. Current accepted SOTA remains `4.2 tok/s`.
- Versioned artifact: `.Agent/runs/20260703-vendor-ds4-coldstart/rejected-moe-cpu-chunk32-summary.json`.

Next direction after chunk-size rejection:

- Stop pursuing coarse chunk-size/scheduler-only changes. The next candidate needs to reduce actual per-dot or per-row cost while preserving memory locality, or move a targeted early up/down subset to a GPU/cache path without increasing page refaults.


Existing DS4 hot-dispatch bound check:

- Time: 2026-07-03 after chunk-size rejection commit `8e3cd0afc`.
- Goal: evaluate the remaining plan direction "move a targeted early up/down subset to GPU/cache" using the existing `DS4_HOT_PROFILE_JSON` / `DS4_HOT_DISPATCH=1` implementation before running a risky full-model probe.
- Source inspection: `src/llama-deepseek4-hot.cpp` extracts hot `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` rows into GPU tensors. `src/models/deepseek4.cpp` then runs a hot GPU path and a cold CPU path. The cold CPU path remaps hot picks to a shared real cold sentinel and masks the output. Therefore the current implementation only reduces CPU unique rows when a token has multiple hot picks that collapse to one sentinel; a single hot pick still costs one CPU sentinel row while also adding the GPU hot path.
- Bound artifact: `.Agent/runs/20260703-vendor-ds4-coldstart/ds4-hot-route-vram-bound.json`, derived from accepted route trace `/root/lfz/runs/vendor-ds4-16gb/20260703T113716Z-20260703T-route-trace-accepted-diagnostic/france-cpu40-vram0gb/route_trace.csv` and model tensor sizes.
- Memory model: DS4 hot allocates `k + P + 1` expert slots per tensor, with `P=6` selected experts. For this model, one layer at `k=16` costs about `293.25 MiB` of GPU memory for gate+up+down hot tensors and dummy slots. Current accepted SOTA has only about `238 MiB` CUDA free after the `13568 MiB` gate cache, so any meaningful hot profile would require shrinking the gate cache and risking extra gate misses.
- Route/CPU-row bound: `k=1` saves `0` CPU rows for all early layers in the accepted France trace. For early layers `0-9`, `k=4` saves at most `10` CPU unique rows in one layer; `k=8` saves at most `17` rows in one layer. The best single-layer `k=16` cases across all layers save `41` CPU rows but require `~293 MiB` VRAM for that one layer.
- Interpretation: the existing DS4 hot dispatch is not a good SOTA candidate under the current 16GB/VRAM budget. Its CPU-row reduction is weak unless `k` is large, while large `k` consumes enough VRAM to reduce the accepted gate cache. It also adds graph work and GPU hot matmuls, so the hard upper bound is not favorable.
- Verdict: do not run a full strict-cold DS4 hot-dispatch probe with the current implementation. A future hot-path candidate would need a different cold-path mechanism that truly skips hot picks on CPU instead of replacing them with a real sentinel row, plus an explicit VRAM/cache budget.

Next direction after DS4 hot bound check:

- Existing broad levers are now rejected by measurement or bound: gate union packs hurt France, scheduler/chunk-size is not enough, transient repack causes refault pressure, and current DS4 hot dispatch has weak CPU-row savings per VRAM.
- Continue with measurement-first work. The next useful diagnostic should target either page/refault source attribution for early up/down fallback or a correctness-preserving CPU special case that skips provably zero/sentinel work without changing selected expert outputs.


CPU fallback page-fault attribution design:

- Time: 2026-07-03 after DS4 hot-dispatch bound commit `8f7f66b9a`.
- Goal: identify whether the remaining cold-start up/down fallback cost is dominated by page faults/refaults from specific tensors/layers rather than arithmetic. Existing strict summaries expose only whole-run `pgmajfault` and `workingset_refault_file`, which is too coarse to design another source candidate.
- Source candidate: temporary default-off instrumentation in `ggml/src/ggml-cpu/ggml-cpu.c`, activated by `GGML_MOE_FALLBACK_FAULT_TRACE_OUT={case_dir}/fallback_fault_trace.csv`. Around each CPU fallback expert loop in `ggml_compute_forward_mul_mat_id`, record `tensor, expert, cne1, elapsed_ms, minor_fault_delta, major_fault_delta` using `getrusage(RUSAGE_SELF)` before and after the per-expert fallback work. Keep it per expert, not per chunk, to limit trace overhead.
- Expected insight: if major faults concentrate in a small set of early up/down tensors, future work should target page placement/O_DIRECT/hot packing for those tensors. If faults are spread and elapsed is mostly compute with low fault deltas, page-policy changes are unlikely to beat SOTA.
- Execution: run accepted France SOTA config under strict cold 16GB cgroup with the trace enabled. This is diagnostic only; trace overhead means it cannot replace SOTA even if token rate rounds high.
- Rollback: after the diagnostic, revert `ggml/src/ggml-cpu/ggml-cpu.c`, rebuild clean, record the analysis, and push docs/artifacts only unless a later non-diagnostic SOTA candidate passes all acceptance gates.


CPU fallback page-fault attribution result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T124828Z-20260703_fallback_fault_trace_current_sota_france/france-cpu40-vram0gb`.
- Source handling: temporary `GGML_MOE_FALLBACK_FAULT_TRACE_OUT` instrumentation was reverted with `git restore ggml/src/ggml-cpu/ggml-cpu.c`; `cmake --build build-ds4-moe-stream -j 8 --target llama-cli` was rerun from clean commit `8f7f66b9a`.
- Diagnostic run metrics: `eval_tok_s=3.9`, `prompt_tok_s=1.5`, `TTFT=28854.585053 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15106760704`, `pgmajfault=271525`, `workingset_refault_file=1877185`, `ram_ok=true`, correctness pass. The run is diagnostic only because fault tracing adds overhead.
- Versioned artifacts: `.Agent/runs/20260703-vendor-ds4-coldstart/fallback-fault-trace-current-sota-summary.json` and `.Agent/runs/20260703-vendor-ds4-coldstart/fallback-fault-trace-current-sota-analysis.json`.
- Trace rows captured: `500000`, exactly the configured cap.
- Captured thread-local fallback fault totals: `majflt=215254`, `minflt=498903`, elapsed trace threadsum `388857.025 ms`.
- By kind: `down` has `123516` major faults and `207722.614 ms`; `up` has `91738` major faults and `181134.411 ms`.
- By layer band major faults: `0-2` `26476`, `3-9` `45578`, `10-19` `48357`, `20-29` `47652`, `30-39` `47191`. Early layers contain the largest individual tensors, but major faults are broadly distributed across the model, not isolated to one tiny hotset.
- Top major-fault tensors include early up/down (`blk.1.ffn_down_exps`, `blk.2.ffn_down_exps`, `blk.0.ffn_up_exps`, `blk.1.ffn_up_exps`, `blk.0.ffn_down_exps`, `blk.2.ffn_up_exps`) and some later down tensors (`blk.35.ffn_down_exps`, `blk.36.ffn_down_exps`).
- Interpretation: cold page faults are a real part of CPU fallback cost, but they are too broad to solve with a small per-layer hotset under current VRAM. This also explains why transient repack and larger chunking increased refault pressure. A viable page strategy would need to reduce source scans or change I/O layout globally without consuming host RAM/page cache, not just pin a few early experts.

Next direction after fault attribution:

- Do not pursue small hotset/page-hint candidates unless they include a hard model for broad fault reduction. Current evidence favors either a true compute kernel improvement with no extra source scan, or a deeper I/O layout change that reduces distributed up/down page faults while preserving the accepted gate pack/cache behavior.


MXFP4 AVX2 dot microbench design:

- Time: 2026-07-03 after fallback fault attribution commit `311ba4e22`.
- Goal: test whether the scalar `ggml_vec_dot_mxfp4_q8_0()` has enough compute headroom for a no-extra-RAM runtime speedup. The host CPU is AMD EPYC 7B13 with AVX2 but no AVX512/VNNI, so the plausible SIMD path is AVX2 `pshufb` nibble lookup plus signed int16 multiply-add, not VNNI dot-product.
- Candidate scope: standalone harness only, no runtime source change. Compare current scalar row-wise dot with an AVX2 implementation that decodes the 16 low and 16 high MXFP4 nibbles via `_mm_shuffle_epi8`, sign-extends MXFP4 and Q8 bytes to int16, multiplies/adds with `_mm256_madd_epi16`, and applies the same per-block scale.
- Correctness gate: exact or near-exact equality to scalar for DS4-like shapes (`k=4096` up, `k=2048` down). Any nonzero diff must be explained before runtime integration.
- Theoretical acceptance threshold: runtime integration is only worth considering if AVX2 row-wise dot is at least `1.2x` faster in warm compute. A smaller speedup cannot overcome distributed cold page faults and graph/stream overhead seen in the fault trace.


MXFP4 AVX2 dot microbench result:

- Source inspection result: the premise that `ggml_vec_dot_mxfp4_q8_0()` was scalar on this host was wrong. The active x86 implementation is `ggml/src/ggml-cpu/arch/x86/quants.c`, which already has an AVX2 path using `_mm_shuffle_epi8`, `mul_sum_i8_pairs_float`, and FMA accumulation. On newer hosts the same helper can lower to VNNI; this AMD EPYC host has AVX2 but no AVX512/VNNI.
- A standalone scratch AVX2 harness was compiled to test a naive row-wise implementation. It produced incorrect values (`max_abs` in the tens of thousands) and was slower than the existing runtime symbol (`speedup=0.455` for `k=4096 rows=2048`, `0.474` for `k=2048 rows=4096`). The scratch file was removed and not committed because it is not a valid tool.
- Interpretation: there is no low-risk row-wise AVX2 runtime patch to add; the repo already contains the optimized x86 path. Further compute-kernel gains would require improving the existing `mul_sum_i8_pairs_float`/repack kernels themselves, which is a deeper kernel project and not a small SOTA candidate under the current cold-start constraints.
- Verdict: reject naive AVX2 row-wise dot direction. Current accepted SOTA remains `4.2 tok/s`.

Next direction after AVX2 dot inspection:

- Remaining viable work is now either a deeper existing-kernel optimization with its own microbench and correctness proof, or an I/O/layout redesign that reduces broad up/down page faults without consuming host RAM. Do not add another model-run candidate without a stronger bound than the rejected chunk/repack/hot/page probes.


Post-diagnostics clean SOTA guard:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T130030Z-20260703_post_diagnostics_clean_sota_guard/france-cpu40-vram0gb`.
- Purpose: verify that all temporary instrumentation/candidate source edits were reverted and the accepted SOTA path still reproduces after clean rebuild.
- Config: accepted France SOTA config unchanged, no diagnostic trace envs, strict cold `drop_caches`, `MemoryMax=16000000000`.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=28759.105881 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15084658688`, `pgmajfault=271934`, `workingset_refault_file=1718362`, `ram_ok=true`, correctness pass.
- Gate counters stayed aligned with accepted SOTA: pack `hits=4623 misses=0 direct_failures=0`, VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Verdict: current clean source reproduces the repeated strict-cold line (`4.1 tok/s`) and remains below the historical accepted `4.2 tok/s` SOTA. No new SOTA was found in this cycle.
- Versioned artifact: `.Agent/runs/20260703-vendor-ds4-coldstart/post-diagnostics-clean-sota-guard-summary.json`.

## Acceptance Rules

A new result can be promoted only if all conditions pass:

- `eval_tok_s > 4.2`.
- `memory_peak_bytes <= 16000000000` with page cache included in the same cgroup.
- `oom=0`, `oom_kill=0`, `ram_limit_killed=false`.
- France output is semantic, coherent, and non-repetitive.
- TTFT is within the accepted `+20%` gate.
- O_DIRECT pack counters have no direct failures or unexpected fallback.
- Run directory contains stdout/stderr, summary, cgroup memory files, exact command/env, pack/profile hashes, answer text, and timing/counter evidence.
- Source, plan, and metadata are committed and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- The pushed commit is clean-rebuilt and strict-cold rerun before declaring SOTA.

Rejected/tie handling:

- `eval_tok_s <= 4.2` is not promoted even if all gates pass.
- Correctness failure, RAM violation, TTFT violation, CUDA OOM, cache insertion failure, or pack direct failure requires rollback or rejection.
- TTFT-over-gate speedups may be committed only as explicitly `not accepted` diagnostics and cannot replace current SOTA.

## Assumptions

- This new plan file is the active next-stage plan; the long `20260701` plan remains the historical audit log.
- `4.2 tok/s` remains the historical highest observed/current accepted record only when citing its original run.
- `4.1 tok/s` is the currently repeated strict-cold reproduction line.
- No runtime source edits are included in this plan commit; the added files are repeatable run-tool harness artifacts for future candidate gating.

### 2026-07-03 Latest Plan Update: Post-Diagnostics Cold-Start Path

This section supersedes any older "latest head" note above if the commit id differs. Current repository observation:

- Local branch: `feat/ds4-moe-stream-on-vendor`.
- Current head: `2a8b3bb5fab995e750bf19e6123b7b388db7cbb1` (`vendor-ds4: record post diagnostics clean guard`).
- Push target remains `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- Git identity for future commits/pushes must remain `L-Ark <fliangae@connect.ust.hk>`.
- Runtime source must stay clean before starting the next candidate. Any temporary source patch must be reverted and clean-rebuilt before recording a rejection.

Current accepted SOTA remains unchanged:

- Accepted cold-start SOTA: `eval_tok_s=4.2` from `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`.
- Repeated strict-cold guard after later diagnostics: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=28759.105881 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15084658688`, `ram_ok=true`, `correctness_ok=true`, gate pack `hits=4623 misses=0 direct_failures=0`, gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%` from `/root/lfz/runs/vendor-ds4-16gb/20260703T130030Z-20260703_post_diagnostics_clean_sota_guard/france-cpu40-vram0gb`.
- The accepted configuration is still the vendor DS4 gate-only one-stream path with `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate O_DIRECT pack, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, strict cold `drop_caches`, and 16GB cgroup including page cache.

Hard promotion discipline for every future step:

1. Before any new practice run, update this plan with the bottleneck being tested, theory, hard upper bound, expected resource cost, acceptance gates, and rollback plan.
2. For any result with `eval_tok_s > 4.2` and passing all gates, stop exploration immediately.
3. Record full reproduction metadata: source commit, branch, binary sha256, model path and size, pack/profile path and sha256, exact command, env, cgroup limit, cgroup memory peak/file/anon/refault counters, TTFT, prompt/eval token rates, answer text, correctness judgment, pack counters, VRAM cache counters, and run directory.
4. Immediately commit source plus plan/run metadata and push to `ssd/vendor/deepseek-token-rate-16gb` using the `L-Ark` identity.
5. Rebuild clean from the pushed source and rerun strict cold before declaring the new SOTA reproducible.
6. If RAM exceeds 16GB including page cache, output correctness fails, TTFT exceeds the allowed gate, direct pack failures appear, or token rate does not beat `4.2`, the candidate is rejected. Revert source, clean rebuild, and record only diagnostic/rejected artifacts.

Next optimization focus:

- Do not continue broad one-stream up/down, independent down-cache sizing, CUDA graph, CPU affinity pinning, context-size trimming, warmup/no-warmup toggles, willneed/page-advice sweeps, early-layer top-k pruning, compact mmap pack-size sweeps, or ad hoc MXFP4 rewrites unless a new diagnostic gives a hard upper bound above about `1-2s` and a clear reason the previous rejection no longer applies.
- The current bottleneck remains correctness-preserving CPU up/down fallback work under cold page-cache pressure. The fine trace showed decode-like `cne1=1` MXFP4 up/down dominates; previous scheduling/page-source/kernel microbenches did not produce an acceptable speed signal.
- The immediate next artifact should be a reproducible MXFP4 correctness/performance harness, not a model run. The harness must compare candidate kernels against the exact runtime `ggml_vec_dot_mxfp4_q8_0()` behavior on DS4-like shapes, report numerical error and warm speed, and reject variants before runtime integration if they are not correctness-identical or do not reach the configured speedup threshold.
- Only if the harness finds a defensible speedup should the next runtime source candidate be designed. That candidate must be default-off, bounded in memory, preserve the accepted gate O_DIRECT path, and first run a short diagnostic showing the targeted fallback counter actually decreases before a full strict-cold SOTA run.

Next concrete checklist:

- [x] Verify worktree cleanliness and `ssd/vendor/deepseek-token-rate-16gb` push state before the next experiment.
- [x] Run `.Agent/run-tools/mxfp4_dot_harness.cpp` as the canonical gate for MXFP4 CPU fallback kernel ideas; latest result recorded below.
- [ ] If a kernel candidate passes the harness threshold, write a new plan subsection with theory and upper bound before touching runtime source.
- [x] Record that no current mechanism has a defensible path beyond `4.2 tok/s`; stop full-model candidate runs until a new bottleneck mechanism is identified.

### 2026-07-03 MXFP4 Harness Execution Result

Purpose:

- Execute the committed MXFP4 harness after the post-diagnostics plan update, before considering any new runtime source candidate.
- This is a repeatable microbench gate only; it cannot promote SOTA by itself and does not change runtime source.

Artifacts:

- Harness: `.Agent/run-tools/mxfp4_dot_harness.cpp`, sha256 `b742633e5203454d1008b5aadf37d02d7e4879cccfe08ce05af2317374c24da1`.
- Runner: `.Agent/run-tools/run_mxfp4_dot_harness.sh`, sha256 `e8b0752dea4a62f34bb42571736df2295d57ae03c6db8da8e054fe01349f4943`.
- Raw log: `.Agent/runs/20260703-vendor-ds4-coldstart/mxfp4-dot-harness-20260703T-latest.log`, sha256 `5709577a57b4f15a7af181b01006ff333f75dd4bfac781d9f49fb5a43347c835`.
- Versioned summary: `.Agent/runs/20260703-vendor-ds4-coldstart/mxfp4-dot-harness-latest-summary.json`.
- Compile command: `g++ -O3 -march=native -std=c++17 -Iggml/include -Iggml/src -Iggml/src/ggml-cpu .Agent/run-tools/mxfp4_dot_harness.cpp -L/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin -lggml-cpu -lggml-base -Wl,-rpath,/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin -pthread -ldl -lm -o /tmp/mxfp4_dot_harness`.

Results with `iters=300`:

| Shape | k | rows | row_ms | repack_gemv_ms | speedup | max_abs | mean_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| up-like | 4096 | 2048 | 132.533 | 92.976 | 1.425x | 0 | 0 |
| down-like | 2048 | 4096 | 139.103 | 92.770 | 1.499x | 0 | 0 |

Decision:

- The harness is valid and repeatable: the existing tested `ggml_gemv_mxfp4_8x8_q8_0()` repack kernel is numerically exact against row-wise `ggml_vec_dot_mxfp4_q8_0()` on these DS4-like shapes.
- This does not justify a new strict-cold model candidate because it is the already-evaluated repack direction. The warm speedup is not enough to overcome the strict 16GB cold-start constraints once repack memory, hotset lookup, page-cache displacement, and integration overhead are included.
- Current accepted SOTA remains unchanged at historical `4.2 tok/s`; repeated strict-cold reproduction remains `4.1 tok/s`.
- Next runtime source work requires a new correctness-identical kernel/layout idea with a stronger hard upper bound than the existing repack path. Until then, do not run another full model SOTA candidate.

### 2026-07-03 Transient Per-Op MXFP4 Repack Microbench Design

Design:

- Goal: test a new CPU fallback compute mechanism that differs from the rejected persistent/hotset repack direction.
- Observation from `ggml_compute_forward_mul_mat_id_one_chunk()`: for decode-like `cne1=1`, CPU fallback repeatedly calls row-wise `ggml_vec_dot_mxfp4_q8_0()` over all output rows of one routed expert. The same Q8 activation row is reused for thousands of MXFP4 rows.
- Existing harness result: persistent 8x8 repack GEMV is numerically exact and warm-compute faster (`1.425x` up-like, `1.499x` down-like), but persistent/hotset integration was rejected because it consumes memory and does not fit the 16GB cold-start/page-cache constraints.
- New candidate idea: repack only the current expert matrix into a small per-op/per-thread temporary buffer, immediately run `ggml_gemv_mxfp4_8x8_q8_0()`, then discard/reuse the buffer. This avoids persistent anonymous RAM and avoids changing routing, top-k, gate cache, O_DIRECT pack, or model math.

Theory and hard upper bound:

- Temporary memory size is bounded by one expert matrix per worker: up-like and down-like DS4 shapes are about a few MiB each after MXFP4 8x8 repack, so even 20 workers should be tens to low hundreds of MiB, not GiB. This must still be validated under the 16GB cgroup if it reaches runtime testing.
- The transient path reads the mmap source once to repack and then reads the temporary hot buffer for GEMV. It may reduce repeated activation reload/loop overhead, but it may increase cold source traffic and anonymous memory pressure.
- Fine trace measured decode-like `cne1=1` up/down fallback at about `21199.9 ms` wall-equivalent. If transient repack including repack cost gives speedup `S`, the optimistic bound is `21199.9*(1-1/S) ms`. At `S=1.15`, the bound is about `2.8s`; at `S=1.30`, about `4.9s`. This is large enough to test only if a microbench includes repack cost every iteration.

Practice plan:

- Add a run-tool-only transient repack harness; do not edit runtime source yet.
- Benchmark row-wise current runtime dot against `repack every iteration + ggml_gemv_mxfp4_8x8_q8_0()` on DS4-like shapes.
- Required output: compile command, CPU flags, shape, row-wise ms, transient repack+GEMV ms, repack-only ms, speedup including repack, max/mean abs, and checksum/sink.
- Proceed to runtime source design only if speedup including repack is at least `1.15x` on both up-like and down-like shapes with `max_abs=0` and `mean_abs=0`.
- If the harness fails that threshold, reject transient repack at microbench stage and do not run a full model candidate.

Runtime acceptance if microbench passes:

- Before touching runtime source, write a new plan subsection with the exact default-off env, scratch-buffer ownership, memory bound, correctness comparison method, and rollback plan.
- First runtime run must be diagnostic/short or profile-only and prove CPU fallback timing decreases without RAM/correctness/gate counter regressions.
- A full strict-cold France SOTA run is allowed only after the diagnostic shows the targeted counter decreases. Promotion still requires `eval_tok_s > 4.2`, strict 16GB including page cache, TTFT gate, pack direct failures `0`, and complete coherent France output.

Transient per-op MXFP4 repack microbench result:

- Harness: `.Agent/run-tools/mxfp4_transient_repack_harness.cpp`, sha256 `5b509a6def16494634733f6547b63142b8873c7cb8f3fde8fd5e0bd97c9e722c`.
- Runner: `.Agent/run-tools/run_mxfp4_transient_repack_harness.sh`, sha256 `06c614f8176cc7fda84c6096c7cd2b3df3a6336b26b0e5e78c860c421918b611`.
- Raw log: `.Agent/runs/20260703-vendor-ds4-coldstart/mxfp4-transient-repack-harness-latest.log`, sha256 `0ff206f435aae6fac6f96b48ffb2c65c5460c9459aacd26822541c52488b9b82`.
- Versioned summary: `.Agent/runs/20260703-vendor-ds4-coldstart/mxfp4-transient-repack-harness-summary.json`.

Results with `iters=300`:

| Shape | k | rows | src MiB | tmp MiB | row_ms | transient_ms | repack_only_ms | GEMV inside ms | speedup incl repack | max_abs | mean_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| up-like | 4096 | 2048 | 4.250 | 4.250 | 131.333 | 145.574 | 52.681 | 92.893 | 0.902x | 0 | 0 |
| down-like | 2048 | 4096 | 4.250 | 4.250 | 137.045 | 149.120 | 56.615 | 92.505 | 0.919x | 0 | 0 |

Diagnosis:

- The transient path is numerically exact, but it fails the required `>=1.15x` speedup threshold once repack cost is paid every iteration.
- The existing 8x8 GEMV kernel remains faster than row-wise dot by itself, but per-op repack costs about `52-57 ms` over 300 iterations for these shapes, more than the GEMV saving.
- Under cold-start model execution this would likely be worse because the transient path adds source copy, temporary writes, and anonymous memory pressure. It does not provide a defensible SOTA path under the strict 16GB page-cache constraint.

Verdict:

- Rejected at microbench stage. Do not implement transient per-op MXFP4 repack in runtime source and do not run a full model candidate for this direction.
- Current accepted SOTA remains historical `4.2 tok/s`; repeated strict-cold reproduction remains `4.1 tok/s`.
- Remaining work requires a new mechanism beyond row-wise dot, persistent/hotset repack, and transient per-op repack.

### 2026-07-03 Up/Down I/O Layout Feasibility Audit

Goal:

- Evaluate the remaining non-kernel direction: a broader up/down I/O or layout redesign that reduces cold page faults without consuming host RAM or disturbing the accepted gate O_DIRECT path.
- This is an evidence audit only. It does not change runtime source and does not run a model candidate.

Artifact:

- `.Agent/runs/20260703-vendor-ds4-coldstart/updown-io-layout-feasibility-audit.json`.
- Source profile: `/root/lfz/runs/vendor-ds4-16gb/20260703T100011Z-20260703T100011Z-cpu-fallback-fine-trace-limit2m/france-cpu40-vram0gb/fallback-profile.csv`.
- Source fault analysis: `.Agent/runs/20260703-vendor-ds4-coldstart/fallback-fault-trace-current-sota-analysis.json`.

Key numbers:

| Scope | entries | calls | fallback_ms | unique payload | call-weighted reads | avg calls/entry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode up | 2627 | 17940 | 9725.822 | 10.903 GiB | 74.458 GiB | 6.829 |
| decode down | 2627 | 17940 | 9673.781 | 10.903 GiB | 74.458 GiB | 6.829 |
| decode up+down | 5254 | 35880 | 19399.603 | 21.806 GiB | 148.916 GiB | 6.829 |
| all up+down | 7030 | 38222 | 26972.399 | 29.177 GiB | 158.636 GiB | 5.437 |

Decode top-N fallback coverage:

| top N | fallback_ms | coverage | unique payload | call-weighted reads |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2709.990 | 13.97% | 0.266 GiB | 26.928 GiB |
| 128 | 4262.371 | 21.97% | 0.531 GiB | 42.143 GiB |
| 256 | 6131.527 | 31.61% | 1.062 GiB | 57.437 GiB |
| 512 | 8506.409 | 43.85% | 2.125 GiB | 73.362 GiB |
| 1024 | 11711.597 | 60.37% | 4.250 GiB | 89.478 GiB |
| 2048 | 15698.329 | 80.92% | 8.500 GiB | 114.696 GiB |
| 4096 | 18780.805 | 96.81% | 17.000 GiB | 143.081 GiB |

Interpretation:

- Faults are broad, not a tiny hotset: the fault trace distributes major faults across all layer bands (`0-2`, `3-9`, `10-19`, `20-29`, `30-39`) and both up/down kinds.
- A mmap compact pack large enough to cover most decode fallback would itself consume many GiB of cgroup file cache, competing with the accepted 16GB page-cache budget. This matches earlier compact mmap top128/top256 tie/regression results.
- O_DIRECT into temporary buffers would avoid file page cache, but it still has to read/copy tens to hundreds of GiB call-weighted data and does not remove MXFP4 dot compute. The transient repack harness already showed that adding source copy/temp writes can erase the GEMV benefit even in warm microbench.
- Small hotsets are not enough: top128 covers only `21.97%` of decode fallback; top512 covers `43.85%` but needs `2.125 GiB` unique payload and still leaves most fallback work. Prior top512 down-pack short diagnostic reduced staging but remained far below SOTA because gate/cache and staging costs dominated.

Verdict:

- Reject broad up/down I/O/layout redesign as the next runtime candidate unless a future design changes the data movement model more fundamentally than mmap packs, O_DIRECT temp copies, or small hotsets.
- Do not run another full strict-cold model candidate for I/O/layout alone. A future candidate needs a new microbench or short diagnostic proving it reduces either broad source scan cost or MXFP4 compute without adding cgroup page-cache pressure.
- Current accepted SOTA remains `4.2 tok/s`; repeated strict-cold reproduction remains `4.1 tok/s`.

Current completion state:

- Kernel directions closed in this plan cycle: existing repack harness only, transient per-op repack, naive AVX2/prefetch/multi-row attempts from prior records.
- I/O/page directions closed in this plan cycle: compact mmap packs, willneed/no-warmup/no-repack, down-pack staging, and now broad up/down I/O-layout feasibility.
- No current mechanism has a defensible path above the `4.2 tok/s` promotion gate without a new algorithmic idea. The plan should remain open only for discovery of a new mechanism; do not keep launching full-model probes from already rejected classes.

### 2026-07-03 Plan Completion Audit

Artifact:

- `.Agent/runs/20260703-vendor-ds4-coldstart/plan-completion-audit.json`.

Audit result:

- Current head: `60d4c7502553992801a51110f95caa45296fbc21`.
- Remote verified: `ssd/vendor/deepseek-token-rate-16gb` points to `60d4c7502553992801a51110f95caa45296fbc21`.
- Git identity verified: `L-Ark <fliangae@connect.ust.hk>`.
- Worktree was clean before this audit update; no runtime source files are modified in the final plan cycle.
- Accepted SOTA remains historical `4.2 tok/s`; latest clean guard remains `4.1 tok/s` with strict 16GB cgroup, page cache included, correctness pass, and gate pack/cache counters aligned.

Requirement status:

- Vendor-only scope: satisfied.
- 16GB host RAM including page cache: satisfied for accepted SOTA and latest clean guard.
- Correctness and TTFT gates: satisfied for accepted SOTA/guard; rejected diagnostics were not promoted.
- Commit/push discipline: satisfied; records are pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- Harness gate before runtime kernel source: satisfied.
- Transient per-op repack: tested and rejected at microbench stage (`0.902x/0.919x` including repack).
- Broad up/down I/O layout: audited and rejected as next runtime candidate due to `21.806 GiB` decode unique payload and `148.916 GiB` call-weighted reads.
- Conditional runtime-source design after a passing kernel harness: not triggered because no kernel candidate passed the threshold.

Completion decision:

- All actionable tasks in this active plan cycle are complete.
- No current mechanism has a defensible path above the `4.2 tok/s` promotion gate under the strict 16GB/page-cache/correctness/TTFT constraints.
- Do not continue launching full-model probes from already rejected classes. Future progress requires a genuinely new algorithmic/kernel/data-movement mechanism, which must first be added to this plan with theory, hard upper bound, and a microbench or short diagnostic gate before runtime source changes.

### 2026-07-03 N-Gram Speculative Decode Probe Design

Goal:

- Continue toward the `10 tok/s` target with a new algorithmic class instead of repeating closed kernel/I/O/page-cache directions.
- Test built-in target-validated n-gram speculative decoding without a draft model. This can improve token rate only by accepting multiple generated tokens per target decode step.

Why this is correctness-safe enough to test:

- The target model still validates the drafted tokens before they are committed. A draft token is accepted only when it matches the target sampler result, so model arithmetic, routing, top-k policy, gate cache, and O_DIRECT expert pack behavior are unchanged for committed output.
- Manual/heuristic France correctness remains mandatory. Any incoherent, incomplete, or degenerate output is rejected regardless of speed.

Theory and hard bound:

- Current accepted cold-start SOTA is `4.2 tok/s`; the new target is `10 tok/s`.
- Ignoring overhead, this requires about `10 / 4.2 = 2.38` accepted output tokens per expensive target decode pass. Equivalently, each pass must accept roughly `1.38` extra drafted tokens on average.
- N-gram speculation can only reach that if the generated France paragraph contains repeated n-grams that reliably predict following tokens. This is unlikely for a short non-repetitive answer, so the diagnostic must inspect `n_drafted`, `n_accept`, acceptance rate, and token rate before any further sweep.

Practice plan:

- Run one strict cold France diagnostic with current accepted SOTA env/config plus:
  - `--spec-type ngram-simple`
  - `--spec-ngram-simple-size-n 3`
  - `--spec-ngram-simple-size-m 8`
  - `--spec-ngram-simple-min-hits 1`
- Keep `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate O_DIRECT pack, top-k policy, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, `drop_caches`, and 16GB cgroup.
- Acceptance for further work: proceed only if correctness passes, RAM/TTFT/gate counters pass, and logs show a non-trivial accepted draft rate with `eval_tok_s` at least above the repeated `4.1` line. Promote only if `eval_tok_s > 4.2` and all normal gates pass.
- If accepted draft count is zero/negligible or token rate regresses/ties, reject this class for France cold-start SOTA and do not sweep n-gram parameters blindly.

N-gram speculative decode probe result:

- Invalid first invocation: `/root/lfz/runs/vendor-ds4-16gb/20260703T145037Z-20260703_ngram_simple_spec_probe/france-cpu40-vram0gb`, `exit_status=127`, caused by passing a relative binary path into the systemd case directory. This is not a model result.
- Valid run: `/root/lfz/runs/vendor-ds4-16gb/20260703T145145Z-20260703_ngram_simple_spec_probe_absbin/france-cpu40-vram0gb`.
- Artifact: `.Agent/runs/20260703-vendor-ds4-coldstart/ngram-simple-spec-probe-summary.json`.
- Config delta: `--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 8 --spec-ngram-simple-min-hits 1`; otherwise current accepted SOTA env/config.

Metrics:

- `eval_tok_s=3.7`, `prompt_tok_s=1.6`, `TTFT=31729.485794 ms`, `elapsed_seconds=66.3`.
- RAM: `memory_peak_bytes=16000000000`, `memory_file_bytes=15077982208`, `pgmajfault=273206`, `workingset_refault_file=1824871`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: `correctness_ok=true`; France answer was semantic, coherent, and complete.
- Gate pack/cache counters changed from accepted SOTA shape: pack `hits=4795 misses=162 direct_failures=0`; VRAM cache `hits=34514 misses=4957 hit_rate=87.4%`.
- The logs did not emit explicit `n_drafted`/`n_accept` counters for `llama-cli` ngram-simple mode, so acceptance cannot be proven from counters. The speed result itself is below both the `4.2 tok/s` accepted SOTA and the repeated `4.1 tok/s` guard line.

Diagnosis:

- Target-validated n-gram speculation preserved answer quality, but it did not reduce effective decode cost. It likely added target batch/cache trajectory overhead without enough accepted draft tokens on this short non-repetitive France prompt.
- The required bound for `10 tok/s` was about `2.38` accepted tokens per target pass. This run provides no evidence of a substantial accepted-draft rate and regresses to `3.7 tok/s`.

Verdict:

- Rejected. Do not promote and do not sweep n-gram parameters blindly.
- Future speculative work needs either a real compatible draft model, implemented MTP/NextN support, or instrumentation proving a high accepted-draft rate before another strict cold model candidate.
- Current accepted SOTA remains `4.2 tok/s`; repeated strict-cold reproduction remains `4.1 tok/s`.

### 2026-07-03 MTP/NextN And Lookahead Decode Audit Design

MTP/NextN availability audit:

- Motivation: reaching `10 tok/s` from `4.2 tok/s` likely requires reducing full target forward passes per committed token, not another small CPU fallback optimization.
- Code inspection: `src/llama-model.cpp` and `src/llama-arch.cpp` can preserve NextN/MTP tensors, but comments mark these tensors as reserved/unused. The model graph code does not implement DeepSeek MTP decoding.
- GGUF metadata audit: `llama-gguf ... r n` for `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf` shows `n_kv=50`, `n_tensors=1328`, and no `nextn`, `mtp`, or `nextn_predict_layers` keys/tensors. The model is `deepseek4.*` metadata, not a GGUF with exposed NextN heads.
- Draft model audit: `/root/lfz/models` has no compatible small DeepSeek draft GGUF; the other large GGUF set is GLM and is not a compatible draft for DeepSeek tokenization/architecture.
- Conclusion: real MTP/speculative-with-draft is unavailable in the current artifacts. It cannot be the next runtime candidate without a new model artifact or implementing and validating DeepSeek MTP heads.

Lookahead candidate:

- `llama-lookahead` is built and uses target-model sampling plus verification n-grams. Code inspection shows tokens are sampled from the target context and candidate n-grams are kept only when token IDs match, so this is more correctness-safe than unvalidated approximation.
- Risk: the example hardcodes `W=15`, `N=5`, `G=15` and sets `n_parallel = W + G + 1 = 31`. This may increase KV/compute memory and CPU fallback work enough to regress or OOM, especially because the accepted SOTA has only about `238 MiB` free VRAM after the gate cache.
- Theoretical bound: to reach `10 tok/s`, lookahead must accept about `2.38` committed tokens per expensive decode step with low overhead. With 31 parallel sequences, the overhead is likely high unless `n_accept` is substantial.

Practice plan:

- Run one strict cold France diagnostic with `llama-lookahead` as the binary and the current accepted SOTA env/config.
- Keep the 16GB cgroup, gate O_DIRECT pack, gate cache, top-k policy, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, and deterministic sampling.
- Because `llama-lookahead` prints its own `decoded ... speed` and `n_accept` counters, parse those manually after the run. Promote only if France answer is complete/coherent, RAM/TTFT/gates pass, and effective decoded speed exceeds `4.2 tok/s`; continue only if `n_accept` proves a path toward the `10 tok/s` target.
- If it OOMs, cannot allocate, fails correctness, or regresses/ties, reject lookahead under current 16GB/vendor SOTA constraints.

Lookahead invocation adjustment:

- First lookahead invocation `/root/lfz/runs/vendor-ds4-16gb/20260703T150241Z-20260703_lookahead_decode_probe/france-cpu40-vram0gb` exited before model execution because `llama-lookahead` does not accept `--no-display-prompt`, which the strict runner always passes.
- This is not a model result. Use wrapper `.Agent/run-tools/llama-lookahead-strip-no-display-prompt.sh` for the actual diagnostic; it removes only `--no-display-prompt` and execs the built `llama-lookahead` binary.

Lookahead first result and one allowed follow-up:

- Direct strict-runner invocation failed because `llama-lookahead` does not accept `--no-display-prompt`; wrapper `.Agent/run-tools/llama-lookahead-strip-no-display-prompt.sh` removes only that flag.
- Wrapper run `/root/lfz/runs/vendor-ds4-16gb/20260703T150424Z-20260703_lookahead_decode_probe_wrapper/france-cpu40-vram0gb` loaded the model under the 16GB cgroup but failed before a valid answer: `decode: failed to initialize batch`, `llama_decode failed - increase KV cache size`, output only `France`, correctness failed.
- Diagnosis: the hardcoded lookahead batch (`W=15`, `N=5`, `G=15`, `n_parallel=31`) is incompatible with the accepted SOTA `-b 16 -ub 16` diagnostic shape. One follow-up run with `-b 256 -ub 64` is allowed to determine whether lookahead can run at all under the 16GB/page-cache constraints.
- This follow-up is still diagnostic only. It may be considered for promotion only if it completes a correct France answer, reports useful `n_accept`, stays within RAM/TTFT gates, and beats `4.2 tok/s` effective decoded speed. If it fails allocation, correctness, RAM, or speed, reject lookahead for the current SOTA path.

MTP/NextN and lookahead audit result:

- Artifact: `.Agent/runs/20260703-vendor-ds4-coldstart/mtp-lookahead-audit-summary.json`.
- MTP/NextN: unavailable for the current DeepSeek GGUF. `llama-gguf ... r n` reports `n_kv=50`, `n_tensors=1328`, and no `nextn`, `mtp`, or `nextn_predict_layers` metadata/tensors. Code inspection shows NextN/MTP tensors are preserved/reserved but not implemented in the DeepSeek runtime graph.
- Draft model: unavailable. `/root/lfz/models` contains no compatible small DeepSeek draft GGUF; GLM GGUF files are not compatible DeepSeek draft models.

Lookahead runs:

| Run | Result | Key evidence |
| --- | --- | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T150241Z-20260703_lookahead_decode_probe/france-cpu40-vram0gb` | invalid invocation | `llama-lookahead` rejected strict runner flag `--no-display-prompt`; not a model result |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T150424Z-20260703_lookahead_decode_probe_wrapper/france-cpu40-vram0gb` | rejected | wrapper removed `--no-display-prompt`, but decode failed with `sequence 1 is coupled to 0 ... diverged`, `decode: failed to initialize batch`; output only `France`; correctness failed |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T150625Z-20260703_lookahead_decode_probe_b256/france-cpu40-vram0gb` | rejected | even with `n_batch=256`, `n_ubatch=64`, decode failed with the same coupled-sequence/diverged error; output only `France`; correctness failed |

Lookahead diagnosis:

- The failure is not just the accepted SOTA `-b 16 -ub 16` setting. The larger `-b 256 -ub 64` diagnostic still fails before a valid answer.
- Both wrapper runs stayed within the 16GB cgroup but failed correctness and produced no usable `decoded speed` or `n_accept` counters.
- The error appears to come from model/context sequence coupling under this DeepSeek architecture and lookahead's 31 parallel verification sequences. This makes the built lookahead example incompatible with the current vendor DeepSeek SOTA path without deeper source work.

Verdict:

- Reject MTP/NextN/lookahead as a current no-source path to `10 tok/s`.
- Future work in this class requires either a compatible draft model artifact, a GGUF with real MTP/NextN heads plus implemented graph support, or source-level changes to make lookahead compatible with DeepSeek's coupled sequence behavior. None is available as an immediate SOTA candidate.
- Current accepted SOTA remains `4.2 tok/s`; repeated strict-cold reproduction remains `4.1 tok/s`.

### 2026-07-03 Next Plan After MTP/Lookahead Rejection

Current state:

- Accepted vendor DeepSeek cold-start SOTA remains `4.2 tok/s` with strict 16GB cgroup including page cache, correct France output, TTFT within gate, O_DIRECT gate expert pack, and current gate VRAM cache config.
- Latest strict guard remains `4.1 tok/s`; this is within observed cold-start variance but does not replace the accepted `4.2 tok/s` record.
- No runtime source changes are active after the rejected n-gram, MTP/NextN, and lookahead no-source probes. The worktree may contain only plan/record artifacts until the next explicitly gated source experiment.
- Target `10 tok/s` requires an algorithmic improvement. From `4.2 tok/s`, the hard lower bound is about `2.38` accepted output tokens per expensive target forward pass, or an equivalent reduction in broad CPU up/down fallback compute. Previously tested I/O-only, compact mmap, transient repack, CUDA graph, n-gram, and no-source lookahead paths do not provide that.

Mandatory reproducibility and push rule for all future SOTA candidates:

- Before every practice step, update this plan with bottleneck, theory, hard upper bound, expected RAM/VRAM cost, expected TTFT risk, and explicit accept/reject gates.
- If a candidate produces a compliant new SOTA, immediately record exact run path, source commit, binary hash, model hash or model path, pack/profile hashes, env vars, CLI args, cgroup settings, prompt text, full output text, token rates, TTFT, elapsed time, cgroup `memory.peak`, `memory.stat file/anon`, page/refault counters, OOM status, expert-pack counters, VRAM-cache counters, and correctness judgment.
- Immediately commit and push the source, plan, and result metadata to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using git identity `L-Ark <fliangae@connect.ust.hk>`.
- After pushing a compliant SOTA, clean rebuild or otherwise verify the pushed source, then rerun the strict cold France reproduction from the pushed commit before declaring the result reproducible. If this rerun fails correctness, RAM, TTFT, or speed gates, revert the SOTA claim and record the failure as rejected.
- Results that regress speed, fail correctness, exceed RAM, OOM, or violate the TTFT gate may be committed as rejected diagnostics, but must be clearly marked `rejected` and must not replace the accepted SOTA.

Next bottleneck focus:

- The remaining high-impact path is reducing target forward passes per committed token, because CPU up/down fallback and gate/cache I/O micro-optimizations have not shown enough headroom toward `10 tok/s`.
- Real DeepSeek MTP/NextN is unavailable in the current GGUF and runtime graph, and there is no compatible draft model. Therefore the next viable speculative direction is source-level lookahead compatibility, not another no-source parameter sweep.
- The immediate failure to explain is in `src/llama-batch.cpp`: lookahead creates coupled sequences by assigning a token to multiple sequence IDs, then the memory module reports those coupled sequences have diverged. Increasing `n_batch`/`n_ubatch` did not fix this, so the next step must inspect and, if defensible, modify the lookahead sequence/KV handling.

Phase A: Source-level lookahead compatibility design, no model SOTA run yet:

- Inspect `examples/lookahead/lookahead.cpp`, `src/llama-batch.cpp`, and DeepSeek memory/KV code to determine why `llama_memory_seq_cp(mem, 0, s, -1, -1)` plus later lookahead batch construction leaves coupled sequence positions divergent.
- Make the lookahead dimensions configurable behind default-off CLI/env knobs rather than hardcoded `W=15`, `N=5`, `G=15`; the first compatibility probe should use a tiny shape such as `W=2`, `N=3`, `G=2` only to prove correctness and counters, not to claim SOTA.
- Before running the probe, calculate batch-token cost for the selected shape. For the current hardcoded shape, one decode step can contain roughly `1 + W*(W-1)/2 + (N-2)*W + G*(N-1)` target tokens, which is far too expensive unless `n_accept` is high. A tiny compatibility shape has lower overhead but cannot plausibly reach `10 tok/s`; it is only a gate to decide whether source-level lookahead is technically usable.
- Preserve target validation: committed tokens must come only from verified target-model samples. Any approximation that bypasses target validation is rejected before model testing.

Phase B: Tiny lookahead compatibility probe:

- Implement only default-off instrumentation/configurability needed to run tiny lookahead under the accepted SOTA env, 16GB cgroup, strict cold `drop_caches`, and France prompt.
- Acceptance for continuing: run completes, France output is coherent for the requested length, RAM stays within 16GB including page cache, TTFT is within the allowed diagnostic window or clearly marked over-gate, and logs expose `decoded speed` plus `n_accept`.
- Rejection: any coupled-sequence failure, incomplete output, bad answer, OOM, RAM limit kill, or no measurable `n_accept` signal. If rejected, revert runtime source and record/push only the rejected diagnostic.

Phase C: Only if Phase B passes, scale lookahead with hard bounds:

- Sweep small shapes in increasing cost order and compute required acceptance before each run. A shape may proceed only if its theoretical accepted-token/pass bound can beat `4.2 tok/s` without excessive extra target tokens, KV pressure, or CPU fallback calls.
- Promote only if the strict cold France run beats `4.2 tok/s`, passes correctness, keeps host RAM under 16GB including page cache, and TTFT does not exceed the accepted gate by more than 20%.
- If a shape beats token rate but TTFT is over gate, record and push it as `not accepted / TTFT over gate`, then plan a separate TTFT recovery step before promotion.

Fallback if source-level lookahead is rejected:

- Do not repeat closed full-model probes from compact mmap packs, transient repack, down batch staging, CUDA graph, n-gram speculative decoding, or no-source lookahead.
- The next plan must introduce a genuinely new mechanism with a microbench or short diagnostic gate first, such as a real compatible draft model, a GGUF/runtime path with actual DeepSeek MTP heads, or a GPU-resident up/down compute path whose staging cost is proven lower than current CPU fallback before a full strict-cold run.

### 2026-07-03 Source-Level Lookahead Compatibility Audit Result

Artifact:

- `.Agent/runs/20260703-vendor-ds4-coldstart/source-lookahead-compat-audit-summary.json`.

Audit findings:

- `examples/lookahead/lookahead.cpp` requires true parallel KV branches: it sets `n_parallel = W + G + 1`, copies seq `0` to branch sequences, assigns the current token to all sequence IDs, and then writes lookahead/verification tokens into different sequence IDs.
- `src/llama-memory-deepseek4.cpp` currently has no functional branch-copy semantics: `llama_memory_deepseek4::seq_cp()` and `seq_keep()` are no-ops, while `seq_rm()` only supports full removal/clear. This explains the observed `sequence 1 is coupled to 0 ... diverged` failure: branch sequence metadata was never activated by the prefix copy.
- Patching only metadata is insufficient. DeepSeek4 `init_batch()` explicitly rejects ubatches unless `n_seqs_unq == 1`, with the runtime error that DeepSeek4 currently supports a single sequence per ubatch.
- More importantly, DeepSeek4 state is stored in global position-indexed tensors such as `attn_kv` and `indexer_kv`; it does not have the normal KV cache cell table that can attach different sequence IDs to different cells. Lookahead with `W >= 2` creates divergent branch tokens at the same positions across different sequence IDs, which the current DeepSeek4 memory model cannot represent correctly.

Decision:

- Reject source-level lookahead as the next immediate tiny-probe path. A tiny `W/N/G` run would either fail the same coupled-sequence/runtime single-sequence checks, or require patches that pass checks without representing correct branch KV state.
- Do not implement a default-off tiny lookahead probe for DeepSeek4 unless the memory model is redesigned to support per-branch KV/state storage and correct `seq_cp`/`seq_rm` semantics. That redesign would be large, would multiply memory pressure, and has no defensible path under the current 16GB host RAM and tight VRAM budget.
- Current accepted SOTA remains `4.2 tok/s`; no new model run was performed for this rejected path because it failed the source-audit correctness gate before execution.

Next pivot:

- Do not repeat lookahead, n-gram, compact mmap, transient repack, down-batch staging, or CUDA graph probes from already rejected classes.
- The next candidate must be a different high-bound mechanism with a microbench or short diagnostic gate first. Acceptable classes are: a real compatible DeepSeek draft/MTP artifact, a GGUF/runtime path with actual DeepSeek MTP heads, or a GPU-resident up/down compute path whose staging and compute savings are proven before any full strict-cold France run.

### 2026-07-03 Gate Plus Decode Top32 Up/Down Admission Probe Design

Artifact:

- `.Agent/runs/20260703-vendor-ds4-coldstart/gate-plus-top32-updown-admit-design.json`.
- Candidate profile: `.Agent/profiles/vendor-ds4/current_sota_gate_plus_decode_top32_updown.tsv`, sha256 `928ef314442f25bcfd13bf97d0ba5847883b4aa12b4be1b3730d3fd4456b23d7`.

Design:

- This is an env-only diagnostic using the existing one-stream VRAM cache path. No runtime source change is planned.
- Preserve the current gate admission profile, then admit only the decode fallback top32 up/down expert tensors from the Phase 1 fallback profile.
- Set `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_` so the existing substring filter allows gate/up/down names, while the admission profile restricts actual cache insertion to current gate entries plus top32 up/down entries.
- Keep `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, the current O_DIRECT gate expert pack, top-k policy, `cpu_moe=40`, strict cold `drop_caches`, and 16GB cgroup.

Theory and hard bound:

- Accepted SOTA generation estimate: `gen_time_s=32.875`, `tokens~=138.08`, `eval_tok_s=4.2`.
- Decode top32 up/down coverage: `1383.3 ms`, payload `0.133 GiB`, equivalent to about `32` current gate cache slots.
- If every covered fallback millisecond disappeared with zero overhead and zero gate-cache loss, hard upper bound is only `4.38 tok/s`.
- Therefore this candidate cannot move toward `10 tok/s` by itself. It is allowed only as a one-run low-risk SOTA edge test because it might slightly exceed `4.2 tok/s` if gate cache behavior remains stable.

Accept/reject gates:

- Accept only if strict cold France exceeds `4.2 tok/s`, `correctness_ok=true`, host RAM remains <=16GB including page cache, TTFT is not over the accepted gate by more than 20%, OOM counters stay zero, and gate/pack counters do not show a material regression.
- Reject immediately if `eval_tok_s <= 4.2`, correctness fails, RAM/TTFT fails, or VRAM cache/gate pack behavior regresses enough to explain a gap.
- Do not sweep larger top-N values blindly: top64 upper bound is still only about `4.52 tok/s`, while top512 needs about `2.125 GiB` and top2048 still only reaches about `7.35 tok/s` before gate loss. Larger hotsets are not a path to `10 tok/s` under current VRAM constraints.

### 2026-07-03 Gate Plus Decode Top32 Up/Down Env-Only Result

Runs:

- Invalid config: `/root/lfz/runs/vendor-ds4-16gb/20260703T153908Z-20260703_gate_plus_top32_updown_admit/france-cpu40-vram0gb`.
- Valid strict-cold SOTA-env diagnostic: `/root/lfz/runs/vendor-ds4-16gb/20260703T154249Z-20260703_gate_plus_top32_updown_admit_sotaenv/france-cpu40-vram0gb`.

Result:

- Invalid config omitted the accepted SOTA one-stream DS4/pack/top-k env and did not use cold `drop_caches`; it is not a SOTA diagnostic.
- Valid config used full SOTA env plus `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_` and the gate+top32 up/down admission profile. It produced `eval_tok_s=1.9`, `prompt_tok_s=1.1`, `TTFT=34390.930213 ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, no OOM kill.
- Manual correctness is rejected despite the runner heuristic passing: the answer ended mid-phrase (`with a thriving economy and`), so it is incomplete for the required France correctness gate.
- Counters: one expert pack `hits=5334 misses=48878 direct_reads=5334`; VRAM cache `hits=45651 misses=54212 hit_rate=45.7%`; file inputs `204492184`; `workingset_refault_file=6174886`.

Diagnosis:

- Env-only admission is not selective compute. Setting `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_` makes all gate/up/down tensors enter the one-stream GPU path. The admission profile only controls whether an expert is inserted into the VRAM cache; when an up/down expert is not admitted, the code still copies it to the temporary GPU buffer and computes it on GPU.
- Therefore the valid run measured broad uncached up/down GPU streaming, not top32-only GPU caching. That explains the huge pack misses, low VRAM hit rate, increased file input, slower TTFT, and `1.9 tok/s` regression.
- This run is rejected and must not be promoted.

### 2026-07-03 Selective Up/Down Admission Gate Probe Design

Design:

- Add a narrow default-off source gate in `ggml/src/ggml-cuda/moe_stream.cu` controlled by `GGML_MOE_STREAM_DECLINE_UNADMITTED_UPDOWN=1`.
- When enabled, and only for `ffn_up_exps` / `ffn_down_exps`, call the existing admission profile check before entering one-stream execution. If the up/down tensor is not admitted, return `false` so the existing CPU fallback path handles it. Gate tensors keep the current SOTA behavior and are not declined.
- Keep `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_` and use `.Agent/profiles/vendor-ds4/current_sota_gate_plus_decode_top32_updown.tsv`, so gate entries behave as before while only top32 up/down entries can enter one-stream GPU/cache.
- This patch must be default-off. A default-off guard run is required after rebuild before any candidate run.

Theory and hard bound:

- The top32 decode up/down coverage remains only `1383.3 ms`, with a hard upper bound of about `4.38 tok/s` if all covered time is removed with zero overhead and zero gate loss.
- This cannot move toward `10 tok/s`; it is only a controlled check of whether tiny selective GPU residency can edge past the `4.2 tok/s` SOTA.
- Because the margin is small, any extra admission lookup, H2D insert, CUDA scheduling, source page fault, or numerical output drift is enough to reject.

Accept/reject gates:

- First run a default-off guard with current SOTA env. It must remain in the `4.1-4.2 tok/s` class, preserve correctness, and keep pack/cache counters aligned.
- Then run one strict cold selective-top32 candidate with `GGML_MOE_STREAM_DECLINE_UNADMITTED_UPDOWN=1`.
- Accept only if it exceeds `4.2 tok/s`, produces a complete coherent France answer, keeps host RAM <=16GB including page cache, TTFT is within the 20% gate, and gate pack/cache counters do not materially regress.
- Reject and revert the source patch if the candidate is `<=4.2 tok/s`, answer is incomplete/incoherent, RAM/TTFT fails, or counters show broad up/down streaming instead of selective top32 execution.

### 2026-07-03 Selective Top32 Up/Down Admission Result

Artifact:

- `.Agent/runs/20260703-vendor-ds4-coldstart/selective-top32-updown-admit-result.json`.

Runs and results:

| Run | eval_tok_s | prompt_tok_s | TTFT ms | RAM | Correctness | Verdict |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T153908Z-20260703_gate_plus_top32_updown_admit/france-cpu40-vram0gb` | 1.4 | 1.1 | 30792.712491 | 16GB cgroup, no kill | true by heuristic | invalid config: missing SOTA env and cold flag |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T154249Z-20260703_gate_plus_top32_updown_admit_sotaenv/france-cpu40-vram0gb` | 1.9 | 1.1 | 34390.930213 | 16GB cgroup, no kill | manual fail: incomplete answer | rejected |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T155219Z-20260703_selective_updown_default_off_guard/france-cpu40-vram0gb` | 4.0 | 1.6 | 29542.273375 | 16GB cgroup, no kill | true | default-off guard passed counters but not SOTA |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T155425Z-20260703_selective_top32_updown_admit/france-cpu40-vram0gb` | 3.7 | 1.5 | 28952.413959 | 16GB cgroup, no kill | true | rejected |

Counter evidence:

- Default-off guard preserved SOTA-shaped counters: pack `hits=4623 misses=0 direct_reads=4623`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Env-only run proved admission is not selective compute: pack `hits=5334 misses=48878`; VRAM cache `hits=45651 misses=54212 hit_rate=45.7%`; it broadly streamed up/down and regressed to `1.9 tok/s` with an incomplete answer.
- Selective source-gated run avoided broad streaming but still regressed: pack `hits=4889 misses=1040 direct_reads=4889`; VRAM cache `hits=41298 misses=5929 hit_rate=87.4%`; file inputs `167396552`; `workingset_refault_file=3072103`; `eval_tok_s=3.7`.

Gap analysis:

- The top32 hard bound was only `4.38 tok/s`, so the available upside was tiny.
- Even when unadmitted up/down returned to CPU fallback, the admitted top32 path added cache admission overhead, extra source reads/page pressure, and some gate-cache churn. The extra misses/refaults outweighed the small covered fallback time.
- Larger top-N values are not justified: top64/top128 still have low hard upper bounds, while top512+ consumes GiB of VRAM and would materially displace the accepted gate cache. Top2048 cannot reach `10 tok/s` even before gate loss.

Verdict:

- Reject this direction. Do not promote.
- Temporary source patch `GGML_MOE_STREAM_DECLINE_UNADMITTED_UPDOWN` is reverted after recording this result. Keep only the rejected profile/design/result artifacts.
- Current accepted SOTA remains `4.2 tok/s`; latest guard remains `4.0-4.1 tok/s` class depending on cold-start variance.

### 2026-07-04 DeepSeek4 All-Output Batch Verification Probe Design

Motivation:

- Reaching `10 tok/s` from the accepted `4.2 tok/s` likely requires accepting multiple output tokens per expensive target forward pass.
- Prior n-gram/speculative/lookahead attempts did not show this because DeepSeek4 currently batches only prefill-style single-sequence ubatches where `n_outputs != n_tokens`.
- Speculative and lookup verification need logits for every draft token, so their target batch has `n_outputs == n_tokens`. In the current DeepSeek4 memory code this disables batched prefill and forces single-token ubatches, removing the main benefit of speculative verification.

Source audit:

- `src/llama-memory-deepseek4.cpp`: `batch_prefill_active = deepseek4_batch_prefill_enabled() && balloc.get_n_outputs() != balloc.get_n_tokens()`; otherwise split size is `1`.
- `src/models/deepseek4.cpp`: `batch_prefill = deepseek4_batch_prefill_enabled() && n_outputs != n_tokens`; otherwise multi-token graph build uses `reserve_only` with `work_tokens=1`.
- `examples/speculative-simple` and `examples/lookup` add draft tokens with `logits=true` for each token, so they require the `n_outputs == n_tokens` path to be batched to verify multiple target logits in one forward pass.

Design:

- Add a default-off env gate `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1` in both the DeepSeek4 memory and graph build files.
- When enabled, single-sequence ubatches with `n_outputs == n_tokens` may use the existing batched prefill graph path (`work_tokens=n_tokens`) instead of the single-token decode path.
- Do not change normal llama-cli greedy generation when no speculative/lookup draft batch exists; for ordinary decode `n_tokens=1`, behavior is equivalent.
- Keep this as a temporary source probe until correctness and speed are proven. If it fails correctness, allocation, or performance, revert source and keep only rejected records.

Theory and hard bound:

- If all-output target verification works and draft acceptance is high, target forwards can verify `1 + n_draft` tokens per pass. To reach `10 tok/s` from `4.2 tok/s`, average accepted tokens per expensive pass must be at least `2.38` before overhead.
- This source change alone does not create good draft tokens; it only removes the target-side batching blocker. First diagnostic uses ngram-simple because it is target-validated and requires no external draft artifact.
- If ngram-simple still has low acceptance, this patch should be rejected for SOTA, but the audit still proves that future draft/MTP paths require this all-output batch capability.

Practice plan:

1. Implement `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1` behind default-off helpers in `src/llama-memory-deepseek4.cpp` and `src/models/deepseek4.cpp`.
2. Build `build-ds4-moe-stream`.
3. Run a default-off strict cold SOTA guard to ensure the patch does not perturb current behavior.
4. Run one strict cold ngram-simple diagnostic with full accepted SOTA env plus `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1` and `LLAMA_DEEPSEEK4_BATCH_LOG=1`.
5. Accept for further speculative work only if logs prove multi-token all-output ubatches (`n_tokens>1`, `n_outputs==n_tokens`, `work_tokens>1`) and France correctness/RAM/TTFT gates pass. Promote only if speed exceeds `4.2 tok/s`; otherwise reject or continue only as a blocker-removal diagnostic.

### 2026-07-04 DeepSeek4 All-Output Batch Verification Current Status

Implementation status:

- Temporary source probe is applied locally and remains uncommitted: `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1` is implemented in `src/llama-memory-deepseek4.cpp` and `src/models/deepseek4.cpp`.
- The gate is default-off. Ordinary SOTA greedy decode should behave as before when the env var is unset.
- This is not an accepted SOTA change yet. It must either produce a compliant speedup and then be committed/pushed immediately, or be reverted after recording the rejected result.

Default-off guard run:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T160818Z-20260704_batch_all_outputs_default_off_guard/france-cpu40-vram0gb`.
- Config: full accepted SOTA env, strict cold `drop_caches`, 16GB cgroup, `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS` unset.
- Metrics: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30591.20202 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15104200704`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic/manual-review-required runner gate.
- Verdict: default-off guard passes as a behavior-preservation check, but it is below the accepted `4.2 tok/s` SOTA and must not replace the current SOTA.

Immediate next experiment:

1. Run one strict cold France diagnostic using full accepted SOTA env plus `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1` and `LLAMA_DEEPSEEK4_BATCH_LOG=1`.
2. Use ngram-simple target-validated drafting: `--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 8 --spec-ngram-simple-min-hits 1`, with the normal SOTA llama-cli args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
3. Inspect logs before judging speed. The run only proves the blocker is removed if logs contain multi-token all-output batches with `n_tokens>1`, `n_outputs==n_tokens`, and `work_tokens>1`.
4. Manually review the France answer. If it is incomplete, incoherent, or semantically wrong, mark correctness failed even if the runner heuristic passes.
5. Record all metrics: `eval_tok_s`, `prompt_tok_s`, `TTFT`, full answer text, memory peak/file bytes, cgroup kill status, page-cache accounting, pack counters, VRAM cache counters, and DeepSeek4 batch-log evidence.
6. If the candidate exceeds `4.2 tok/s` and satisfies correctness, RAM, and TTFT gates, immediately update this plan with exact reproduction information, commit the source plus records, and push to `ssd/vendor/deepseek-token-rate-16gb` using the `L-Ark` git identity.
7. If the candidate is not a compliant SOTA, revert the temporary source probe, rebuild clean, record the rejection, and push only the documentation/artifact update.

Priority after this probe:

- If all-output batching works but ngram-simple acceptance is too low, keep the finding as a blocker-removal diagnostic and design the next draft source around measured acceptance, not more cache hotset sweeps.
- If all-output batching does not actually create `work_tokens>1`, stop speculative work and return to bottleneck decomposition of current SOTA decode time before designing another optimization.
- Do not pursue broader up/down hotset caching unless a fresh bottleneck profile proves a hard upper bound above the current SOTA with enough margin; the top32 probe already showed the available upside is too small.

### 2026-07-04 DeepSeek4 All-Output Batch Verification Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/batch-all-outputs-ngram-result.json`.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T161842Z-20260704_batch_all_outputs_ngram_simple/france-cpu40-vram0gb`.
- Config: full accepted SOTA env plus `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1`, `LLAMA_DEEPSEEK4_BATCH_LOG=1`, and `--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 8 --spec-ngram-simple-min-hits 1`.
- Strict cold `drop_caches`, 16GB cgroup, `cpu_moe=40`.

Observed result:

- The run was manually terminated after `6min 1.705s` because stdout stopped advancing and stderr kept repeating the same all-output verification position.
- No valid final `eval_tok_s` or memory peak is available after termination. It is therefore not a compliant candidate.
- TTFT before first answer was `30915.400473 ms`, but final speed and RAM gates cannot be accepted because the run did not complete.
- Runner heuristic marked correctness true, but manual correctness fails: the answer ended mid-sentence after `France offers a unique blend of history, culture, and`.

Batch evidence:

- Multi-token all-output batches did occur: `n_tokens=9 n_outputs=9` appeared 5 times, and `n_tokens=2 n_outputs=2` appeared 1964 times.
- Graph build logs confirmed `reserve_only=0 work_tokens=9` and `reserve_only=0 work_tokens=2`, so the source probe did remove the previous graph-side `work_tokens=1` blocker.
- Failure mode: `n_tokens=2 reserve_only=0 work_tokens=2 start_pos=181` repeated 1961 times while stdout size stayed fixed during a 15 second sample. This suggests the ngram-simple verification/progress loop is rechecking the same position instead of committing progress.

Verdict:

- Reject as SOTA candidate.
- Revert the temporary source probe after recording this result. Do not commit all-output source changes.
- Keep only the artifact and plan record.
- Current accepted SOTA remains `4.2 tok/s` from the established vendor DeepSeek cold-start path.

Next optimization direction:

- Do not continue performance sweeps on this exact ngram-simple setup until the repeated `start_pos=181` progress bug is understood.
- If speculative batching remains the priority, inspect the ngram-simple/target verification loop and its interaction with DeepSeek4 memory positions before another strict run. The next proof must show monotonically advancing positions, a complete France answer, and a completed timing summary.
- Otherwise return to current-SOTA bottleneck decomposition and select the next candidate only if the hard upper bound has enough margin beyond `4.2 tok/s`.

### 2026-07-04 Server Speculative Progress Trace Design

Audit finding:

- `llama-cli` uses `tools/server/server-context.cpp`, not `examples/speculative-simple`, for completion generation.
- In `server_slot::update_batch()`, speculative mode appends `sampled` and every `spec_draft` token to `prompt.tokens` before target verification.
- After target verification, partial acceptance with `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` stores `slot.spec_draft = accepted`, restores the checkpoint, keeps the prompt at `ckpt.n_tokens`, restores the sampler, and `continue`s. The next `update_batch()` reuses the partial draft.
- DeepSeek4 is classified as `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` because partial `llama_memory_seq_rm(mem, 0, 1, -1)` is unsupported. Therefore server speculative decode always uses the checkpoint/partial-replay path for this model.
- The rejected all-output run repeatedly built `n_tokens=2 n_outputs=2 work_tokens=2 start_pos=181`, which is consistent with a partial-replay loop that is not committing progress.

Diagnostic plan:

1. Add a temporary default-off server trace gate, e.g. `LLAMA_SERVER_SPEC_TRACE=1`, around the speculative sections of `tools/server/server-context.cpp`.
2. Trace only compact state, not full logits:
   - `update_batch` entry: `sampled`, `prompt.tokens.size()`, `prompt.tokens.pos_next()`, `spec_draft.size()`, whether the draft is newly generated or reused.
   - verification result: original `n_draft`, `accepted.size()`, whether partial acceptance happened, checkpoint `n_tokens/pos_min/pos_max`, and post-restore prompt length.
   - commit result: `ids.size()`, `n_draft_total`, `n_draft_accepted`, new prompt length, new sampled token.
   - memory trim result around `llama_memory_seq_rm(... prompt.tokens.pos_next(), -1)`.
3. Reapply the all-output batch source probe only behind `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1`, run one short strict diagnostic with both trace gates enabled, and stop as soon as the repeated position is explained.
4. Accept source changes only if they produce a compliant SOTA. Trace-only and failed speculative source probes must be reverted before committing; only the plan/artifact records should be pushed.

Theory:

- If the trace shows repeated partial acceptance with the same `sampled`, same `spec_draft`, and same checkpoint `n_tokens`, then the server checkpoint replay path is incompatible with the current DeepSeek4 memory semantics and needs a semantic fix before speed testing.
- If prompt length and `sampled` advance while DeepSeek4 graph start_pos remains fixed, then the bug is in DeepSeek4 memory position restoration or graph position mapping.
- If no partial acceptance loop occurs with trace enabled, the earlier hang may have been caused by logging overhead or signal timing; rerun with a smaller generation cap before any performance claim.

Gate:

- This diagnostic is not a SOTA candidate. The required output is a clear root-cause trace and a complete record.
- Do not leave trace/all-output source changes in the committed tree unless a later run exceeds `4.2 tok/s` and passes correctness, RAM, and TTFT gates.

### 2026-07-04 Server Speculative Trace Result And Partial Fallback Design

Trace run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T164911Z-20260704_server_spec_fprintf_trace_ngram_simple/france-cpu40-vram0gb`.
- Config: full accepted SOTA env plus `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1`, `LLAMA_SERVER_SPEC_TRACE=1`, and ngram-simple `N=3 M=8 min_hits=1`.
- The run was manually terminated after trace proved the loop. It is not a SOTA candidate and has no valid final `eval_tok_s` or RAM peak.

Trace evidence:

- Before the loop, ngram-simple often produced empty drafts and normal prompt positions advanced one token at a time.
- At the failure point, the trace repeated:
  - `update_batch_entry sampled=305 prompt_size=181 pos_next=181 spec_draft=1 n_draft_max=8`
  - `reuse_draft sampled=305 prompt_size=181 pos_next=181 spec_draft=1 ckpt_n_tokens=181 ckpt_pos=[0,180]`
  - `update_batch_exit sampled=305 prompt_size=183 pos_next=183 inserted_draft=1 batch_tokens=2`
  - `verify n_draft=1 accepted=1 prompt_size=183 pos_next=183 ckpt_n_tokens=181 ckpt_pos=[0,180]`
  - `partial_restore n_draft=1 replay_draft=1 trim_from=181 trim_ok=1 prompt_size=181 pos_next=181 sampled=305`
- This repeated without advancing `prompt_size`, `pos_next`, checkpoint range, or `sampled`.

Diagnosis:

- `accepted=1` with `n_draft=1` means the first draft token was rejected and the target produced exactly one replacement token.
- For `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`, server restores the checkpoint and reuses the replacement token as `spec_draft`. That replay should usually finish on the next pass, but with DeepSeek4 all-output batched verification it repeats at the same checkpoint and never commits progress.
- Therefore the blocker is not just low ngram acceptance; it is the full-checkpoint partial-replay path interacting badly with DeepSeek4 all-output verification.

Next candidate design:

1. Add a temporary default-off env gate, e.g. `LLAMA_SERVER_SPEC_PARTIAL_SERIAL_FALLBACK=1`.
2. Add a slot-local `spec_skip_once` flag.
3. On `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` partial acceptance, when the fallback gate is enabled:
   - restore the checkpoint,
   - trim memory after the checkpoint as before,
   - keep prompt tokens at `ckpt.n_tokens`,
   - restore the sampler state,
   - clear `spec_draft`,
   - set `spec_skip_once=true`,
   - `continue`.
4. In the next `update_batch()`, if `spec_skip_once` is true, force `n_draft_max=0` for exactly one step. This re-evaluates the last `sampled` token through the normal single-token path, recomputes the rejected replacement token without all-output replay, and should allow progress.
5. Keep the all-output batch gate default-off and enable it only for this diagnostic.

Theory:

- Correctness should be preserved because every rejected/partial speculative step falls back to a normal target-model decode from the restored checkpoint.
- Speed upper bound depends on ngram acceptance. Fully accepted drafts can still use all-output verification; rejected first-draft cases pay one extra serial decode and likely do not help throughput.
- This is mainly a progress/correctness unblocker. Promote only if it unexpectedly exceeds `4.2 tok/s` under all gates; otherwise record and revert.

Gate:

- First run a strict cold diagnostic with `LLAMA_DEEPSEEK4_BATCH_ALL_OUTPUTS=1`, `LLAMA_SERVER_SPEC_PARTIAL_SERIAL_FALLBACK=1`, and compact trace.
- It must complete with a coherent France answer before any speed discussion.
- If it completes but is `<=4.2 tok/s`, incomplete, over RAM, or over TTFT gate, reject and revert source.
- If it exceeds `4.2 tok/s` while satisfying correctness/RAM/TTFT, immediately record exact reproduction information, commit source plus records, push to `ssd/vendor/deepseek-token-rate-16gb`, and rerun from pushed source.

### 2026-07-04 Server Partial Serial Fallback Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/server-spec-partial-fallback-result.json`.

Runs:

| Run | eval_tok_s | prompt_tok_s | TTFT ms | RAM | Correctness | Verdict |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T165910Z-20260704_server_spec_partial_serial_fallback_trace/france-cpu40-vram0gb` | 3.7 | 1.5 | 30768.69734 | 16GB cgroup, no kill | manual pass | diagnostic only; trace on |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T170139Z-20260704_server_spec_partial_serial_fallback_notrace/france-cpu40-vram0gb` | 3.7 | 1.6 | 31082.537347 | 16GB cgroup, no kill | manual pass | rejected: below 4.2 SOTA |

Trace conclusion:

- `LLAMA_SERVER_SPEC_PARTIAL_SERIAL_FALLBACK=1` did fix the infinite replay: the trace run showed `serial_fallback=1` twice and `spec_skip=1` twice, then continued to a complete answer.
- The root-cause trace is therefore confirmed: the previous all-output ngram-simple failure was a server full-checkpoint partial-replay loop, not a RAM or pack-cache failure.

Manual correctness:

- Both fallback runs produced a complete and coherent France answer. The answer names France, Western Europe, Paris/Eiffel/Louvre/Versailles, culture, cuisine, wine, fashion, economy, and ends cleanly.

Performance verdict:

- The no-trace run remains only `3.7 tok/s`, below the accepted `4.2 tok/s` cold-start SOTA.
- TTFT is within the 20% gate and RAM is compliant, but speed is not.
- Reject this source direction as a SOTA candidate and revert temporary source changes.

Next direction:

- Speculative ngram-simple is not a high-value path unless a draft source can produce much higher acceptance. The fallback makes it correct/progressing but does not create enough accepted tokens per target forward.
- Return to bottleneck decomposition of the current `4.2 tok/s` SOTA and prioritize candidates with a hard upper bound materially above `4.2`, not small speculative plumbing changes.

### 2026-07-04 Latest Plan After Speculative Rejection

Current accepted state:

- Accepted SOTA remains `4.2 tok/s` from `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`.
- Latest pushed head before this plan update: `b568e210f941f76faf12c1211ac47c1cd4dca57c` on `ssd/vendor/deepseek-token-rate-16gb`.
- The all-output batch and server partial-serial fallback source probes were reverted before commit. They are diagnostic records only.
- The GGUF model file is `146G`; accepted cold-start behavior cannot depend on fully warming the model in host page cache. The 16GB cgroup must include page cache, file cache, anon memory, and process memory.

Immediate phase A: clean SOTA guard.

1. Confirm worktree clean and pushed branch identity:
   - local work branch: `feat/ds4-moe-stream-on-vendor`;
   - push remote/branch: `ssd/vendor/deepseek-token-rate-16gb`;
   - git identity: `L-Ark <fliangae@connect.ust.hk>`.
2. Re-run current SOTA strict cold with the accepted env and `drop_caches` under 16GB cgroup.
3. Required guard result:
   - `eval_tok_s` in the accepted `4.1-4.2 tok/s` class;
   - `ram_ok=true`, `memory_peak_bytes <= 16000000000`, `ram_limit_killed=false`;
   - O_DIRECT pack counters near accepted shape, especially `misses=0`, `direct_failures=0`, `direct_fallbacks=0`;
   - gate VRAM hit rate near accepted `86-87%`;
   - France answer complete, coherent, and semantically correct;
   - TTFT not worse than the accepted baseline by more than 20%.
4. If this guard falls below `4.0 tok/s`, changes counters materially, or fails correctness/RAM, stop optimization and debug reproducibility before any new experiment.

Immediate phase B: bottleneck decomposition from current SOTA.

Run one diagnostic-only strict cold profile using the accepted config plus existing tracing knobs. Record all outputs under `.Agent/runs/20260704-vendor-ds4-coldstart/` and preserve the full remote run directory.

Required measurements:

- Per-token wall time split: prompt vs decode, TTFT, eval time, tokens generated.
- MoE stream/gate split: source load, H2D/D2H if present, kernel, sync, scatter, dontneed, cache hit/miss/insert counts.
- CPU fallback split: up vs down, prompt vs decode, layer/expert/tensor hot list, thread time, major/minor faults, and total fallback ms.
- Cgroup memory: `memory.peak`, `memory.stat file/anon`, `pgmajfault`, `workingset_refault_file`, OOM events.
- Server/runtime overhead: sampling/speculative disabled path overhead if measurable; CUDA graph remains disabled in accepted SOTA unless explicitly tested behind a rejected/diagnostic gate.
- Full France answer text for manual correctness.

The diagnostic run does not replace SOTA even if its token rate differs, because tracing can change timing.

Immediate phase C: candidate design gate.

Do not implement a new optimization until the phase B profile produces a hard-bound table. For each proposed candidate, write the following before code:

| Candidate | Removable component | Measured component ms | Required extra VRAM/RAM | Expected gate-cache loss | Theoretical upper-bound tok/s | Accept to implement? |
| --- | --- | ---: | ---: | ---: | ---: | --- |

Rules:

- Reject before coding if the upper bound is not materially above `4.2 tok/s` after expected overhead.
- Reject before coding if it requires reducing the accepted gate cache enough to lower hit rate below the accepted shape.
- Reject before coding if it relies on buffered host cache beyond the 16GB cgroup.
- Accept for implementation only if it removes actual CPU fallback compute/stall time, not just pointer layout, and has a plausible path to beat `4.2 tok/s`.

Candidate priority order:

1. CPU fallback compute/stall split. If major faults/source stalls dominate, design O_DIRECT or bounded async prefetch; if CPU math dominates, layout-only and mmap-only changes are not worth another full run.
2. Protected VRAM residency for a very small high-coverage expert subset. It must not displace the accepted gate cache. Use explicit VRAM accounting before running.
3. CPU kernel/fusion/threading changes for up/down fallback only if the trace shows poor CPU utilization or avoidable synchronization.
4. Speculative/MTP only with a better draft source and measured high acceptance. The ngram-simple path is rejected at `3.7 tok/s` even after the progress fallback.

Promotion rule for any new SOTA:

When a candidate exceeds the current accepted SOTA and passes RAM, correctness, and TTFT gates, immediately do all of the following before continuing exploration:

1. Write exact reproduction details into this plan and a machine-readable artifact:
   - source commit hash;
   - branch and remote;
   - binary sha256;
   - model path and size;
   - profile/pack paths and sha256;
   - full env and CLI args;
   - run directory;
   - `eval_tok_s`, `prompt_tok_s`, TTFT, total time;
   - memory peak/file/anon/page-fault stats;
   - pack/cache counters;
   - full France output and manual correctness verdict.
2. Commit source, plan, profiles, and result artifacts immediately.
3. Push immediately to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`.
4. Rebuild or rerun from the pushed source and verify the SOTA remains reproducible. If it cannot be reproduced, demote it to diagnostic/rejected until the gap is understood.

Rollback rule:

- If a candidate is slower than `4.2 tok/s`, fails France correctness, exceeds the 16GB host RAM budget including page cache, triggers OOM/kill, or violates the accepted TTFT gate, revert the runtime source before committing.
- It is acceptable to commit and push rejected experiment records, but the committed runtime tree must return to the last accepted SOTA path unless the new candidate is promoted.

### 2026-07-04 Phase A/B Current-SOTA Guard And Bottleneck Refresh

Phase A clean SOTA guard:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T172115Z-20260704_phaseA_clean_sota_guard_head4be08352/france-cpu40-vram0gb`.
- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/phaseA-clean-sota-guard-summary.json`.
- Result: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=29484.204486 ms`.
- RAM: `memory_peak_bytes=16000000000`, `memory_file_bytes=15101456384`, `pgmajfault=268389`, `workingset_refault_file=1629739`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: manual pass. The France answer is complete, coherent, and semantically correct.
- Counters: one expert pack `hits=4623 misses=0 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; gate VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- Verdict: current pushed head reproduces the accepted SOTA shape. Proceed to diagnostic profiling.

Phase B current-SOTA profile:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T172436Z-20260704_phaseB_current_sota_profile_head4be08352/france-cpu40-vram0gb`.
- Artifacts:
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseB-current-sota-profile-summary.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseB-current-sota-bottleneck-summary.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseB-current-sota-cpu-chunk-analysis.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseB-current-sota-name-profile-analysis.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseB-current-sota-hard-bound-table.json`
- Diagnostic result: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=29663.442807 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102832640`, `ram_ok=true`, correctness pass.
- One-stream/gate trace: rows `35151`, hit rate `86.85%`, `src0_ms=5236.633`, `src1_ms=156.695`, `kernel_ms=332.856`, `sync_ms=1039.234`, `total_ms=6986.810`.
- CPU fallback profile: total `25743.677 ms`, decode `18089.280 ms`, prompt `7654.397 ms`, up `12387.490 ms`, down `13356.187 ms`.
- CPU chunk trace: `500000` rows capped; multi-thread chunk-sum `328232.929 ms`, max thread `17238.912 ms`, median thread `16389.568 ms`, up chunk-sum `153310.287 ms`, down chunk-sum `174922.642 ms`.
- Page/cache pressure: `pgmajfault=265548`, `workingset_refault_file=1671406`.

Hard-bound table from Phase B:

| Candidate | Covered decode ms | Required extra residency | Hard upper bound | Decision |
| --- | ---: | ---: | ---: | --- |
| top128 up/down residency | `3741.536` | `544.0 MiB` | `4.61 tok/s` | reject as primary direction |
| top256 up/down residency | `4899.473` | `1075.2 MiB` | `4.80 tok/s` | reject as primary direction |
| top512 up/down residency | `6315.441` | `2112.2 MiB` | `5.05 tok/s` | reject unless gate cache can be preserved |
| top1024 up/down residency | `9080.314` | `4190.5 MiB` | `5.61 tok/s` | reject under current VRAM budget |
| top2048 up/down residency | `12647.096` | `8198.2 MiB` | `6.57 tok/s` | reject under current VRAM budget |
| all CPU up/down decode fallback removed | `18089.280` | `~22.3 GiB unique decode payload if naively resident` | `8.86 tok/s` | insufficient alone for 10 tok/s |
| all CPU up/down decode fallback plus all one-stream source time removed | `23325.913` | stacked CPU fallback + gate source removal | `13.36 tok/s` | only indicates required stacked wins/speculation |

Conclusion:

- The next optimization cannot be another simple hotset residency sweep; the hard upper bound is too low unless it also preserves gate cache and removes more than the covered fallback time.
- Pure CPU fallback elimination is not enough for `10 tok/s` by itself. Reaching `10 tok/s` needs either stacked wins across CPU fallback and gate source stalls, or a speculative/MTP path with much higher acceptance than ngram-simple.
- Before changing source, the next diagnostic must split CPU fallback into page-fault/source-stall vs CPU math. A no-drop/warm comparison under the same 16GB cgroup is useful only as diagnostic evidence; it is not an acceptable cold-start result because external hot page cache may not be charged to the new cgroup.

Next diagnostic plan:

1. Run the same Phase B profile without `drop_caches` while preserving the 16GB cgroup and SOTA env.
2. Compare cold vs no-drop:
   - `eval_tok_s`, TTFT, `pgmajfault`, `workingset_refault_file`, file inputs;
   - CPU fallback total/decode/up/down ms;
   - one-stream `src0_ms` and cache hit rate;
   - full France correctness.
3. If no-drop greatly reduces fallback ms and major faults, prioritize O_DIRECT/bounded async prefetch for CPU fallback source loads.
4. If no-drop does not materially reduce fallback ms, treat CPU fallback math as dominant and deprioritize IO/layout work; then design either CPU kernel/fusion changes or a higher-acceptance speculative/MTP source.
5. Do not promote the no-drop result regardless of speed; it is diagnostic only.

### 2026-07-04 Phase C No-Drop Diagnostic Result And Next Candidate

Phase C no-drop diagnostic:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T173550Z-20260704_phaseC_nodrop_profile_io_compute_split_head6b11e5a/france-cpu40-vram0gb`.
- Artifacts:
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseC-nodrop-profile-summary.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseC-nodrop-bottleneck-summary.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseC-nodrop-cpu-chunk-analysis.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseC-nodrop-name-profile-analysis.json`
  - `.Agent/runs/20260704-vendor-ds4-coldstart/phaseC-cold-vs-nodrop-comparison.json`
- Result: `eval_tok_s=7.2`, `prompt_tok_s=2.0`, `TTFT=26654.802489 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15104180224`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: manual pass. The France answer is complete, coherent, and semantically correct.
- Not promotable: this run intentionally skipped `drop_caches`; it may use global/external hot page cache that is not a valid cold-start setup, even though the process itself stayed in the 16GB cgroup.

Cold vs no-drop comparison:

| Metric | Cold Phase B | No-drop Phase C | Change |
| --- | ---: | ---: | ---: |
| `eval_tok_s` | `4.1` | `7.2` | `+75.6%` |
| `TTFT ms` | `29663.443` | `26654.802` | `-10.1%` |
| `pgmajfault` | `265548` | `120101` | `-54.8%` |
| CPU fallback total | `25743.677 ms` | `12952.670 ms` | `-49.7%` |
| CPU fallback decode | `18089.280 ms` | `7601.418 ms` | `-58.0%` |
| CPU fallback up | `12387.490 ms` | `5873.231 ms` | `-52.6%` |
| CPU fallback down | `13356.187 ms` | `7079.439 ms` | `-47.0%` |
| CPU chunk max thread | `17238.912 ms` | `9093.164 ms` | `-47.3%` |
| one-stream `src0_ms` | `5236.633 ms` | `5360.400 ms` | `+2.4%` |

Diagnosis:

- The main cold-start limiter is CPU up/down fallback source/page-fault stall, not gate O_DIRECT or gate VRAM cache. No-drop roughly halves fallback time while preserving gate counters.
- Existing broad `GGML_MOE_CPU_WILLNEED=1` remains rejected: it reduced major faults but regressed to `4.0 tok/s` on the current SOTA path, likely because it prefetches every active expert just before compute and adds reclaim/synchronization pressure.
- The next source candidate must be bounded and profile-guided. It should warm a small fixed set once after model mmap is available, not repeatedly advise all active experts inside every fallback op.

Next candidate design: profile-guided one-time CPU up/down prewarm.

- `attempt_id`: `20260704-cpu-prewarm-top512-profile`
- `attempt_kind`: `implementation/default-off/cold-page-stall-reduction`
- `hypothesis`: A one-time top512 up/down expert prewarm, driven by the Phase B fallback profile, can move the most valuable cold pages into the 16GB cgroup page cache before decode without repeatedly prefetching every active expert. This targets the exact cold/no-drop gap while avoiding the rejected broad WILLNEED behavior.
- `profile`: `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top512.tsv`, created from Phase B fallback profile and sorted by decode fallback contribution. It contains `512` unique up/down `(tensor, expert)` entries, `7947.407 ms` decode fallback coverage, and `2176 MiB` payload.
- `implementation`: add a default-off env path in `ggml/src/ggml-cpu/ggml-cpu.c`:
  - `GGML_MOE_CPU_PREWARM_PROFILE=<tsv>`;
  - optional `GGML_MOE_CPU_PREWARM_LIMIT=512`;
  - on first CPU fallback encounter for each tensor, thread 0 reads matching `(tensor, expert)` rows from the profile and calls existing `ggml_moe_cpu_willneed_pages()` once per listed expert;
  - record counters at exit: enabled, loaded entries, advised entries, bytes, misses/skipped.
- Hard upper bound from Phase B decode-only top512 profile: covers `7947.407 ms` decode fallback with `2176 MiB` unique payload; ideal upper bound is about `5.37 tok/s`. This candidate cannot reach `10 tok/s` alone. It is a bounded proof of whether page-stall reduction can beat `4.2` without relying on external warm cache.
- TTFT risk: prewarming `~2.1 GiB` can add startup/prompt latency. Accepted only if TTFT remains within the 20% gate relative to the accepted SOTA.
- Acceptance gate: promote only if strict cold `drop_caches` run exceeds `4.2 tok/s`, RAM including page cache stays `<=16000000000`, France answer is manually correct/complete/coherent, gate pack remains `misses=0 direct_failures=0`, gate VRAM hit rate stays accepted-like, and TTFT is within gate.
- Rollback: if it ties/regresses, fails correctness/RAM/TTFT, or counters show gate-cache disruption, revert runtime source and keep only records/artifacts.

### 2026-07-04 Top512 MADV Prewarm Probe Result

Temporary source:

- Added default-off `GGML_MOE_CPU_PREWARM_PROFILE` and `GGML_MOE_CPU_PREWARM_LIMIT` in `ggml/src/ggml-cpu/ggml-cpu.c`.
- The implementation loaded the profile and called existing `ggml_moe_cpu_willneed_pages()` once for matching top512 up/down experts.
- Default-off guard preserved behavior.

Runs:

| Run | eval_tok_s | prompt_tok_s | TTFT ms | RAM | Correctness | Verdict |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T175047Z-20260704_cpu_prewarm_default_off_guard/france-cpu40-vram0gb` | `4.1` | `1.6` | `29327.755079` | 16GB cgroup pass | manual pass | default-off guard pass |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T175229Z-20260704_cpu_prewarm_top512_candidate/france-cpu40-vram0gb` | `4.1` | `1.6` | `28566.146015` | 16GB cgroup pass | manual pass | rejected tie |

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-prewarm-top512-madvise-result.json`

Counters:

- MADV prewarm candidate: `enabled=1 entries=512 advised=512 skipped=0 bytes=2281701376`.
- Gate pack remained accepted-like: `hits=4623 misses=0 direct_failures=0 direct_fallbacks=0`.
- Gate VRAM remained accepted-like: `hits=30528 misses=4623 hit_rate=86.8%`.
- Major faults did not materially improve: default-off `268793`; MADV prewarm `267546`.

Verdict and gap:

- Reject as SOTA. It did not exceed `4.2 tok/s`.
- The one-time `MADV_WILLNEED` calls were issued, but Linux did not fault enough target pages into memory to reduce cold major faults or generation time.
- This narrows the next test: if we want to reproduce part of the no-drop benefit inside the 16GB cgroup, the next mechanism must force actual page residency for the bounded top512 set, not just advise it.

Next candidate design: blocking page-touch top512 prewarm.

- Add a second default-off switch to the same temporary source path, e.g. `GGML_MOE_CPU_PREWARM_TOUCH=1`.
- For each matched profile expert, touch one byte per OS page after `MADV_WILLNEED`, accumulating a volatile checksum so the compiler cannot remove the reads.
- Theory: touching forces the exact top512 expert pages into the cgroup page cache before decode. The cold/no-drop comparison shows the target upper bound is real, while the MADV-only run proves non-blocking advice is insufficient.
- Hard bound: still the same top512 decode-only profile, `7947.407 ms` covered decode fallback and ideal `~5.37 tok/s`.
- Risk: touching `~2.18 GiB` can raise TTFT and cause reclaim pressure under 16GB. A TTFT-over-gate result may be recorded as rejected/not accepted, but cannot be promoted.
- Acceptance gate: same as above: strict cold, >`4.2 tok/s`, RAM including page cache <=16GB, manual France correctness, gate counters accepted-like, TTFT within accepted gate.
- Rollback: if slower/tie/regression or TTFT/RAM/correctness fails, revert runtime source and keep only documentation/artifacts.

### 2026-07-04 Top512 Blocking Touch Prewarm Candidate Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-prewarm-top512-touch-candidate-sota.json`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T175854Z-20260704_cpu_prewarm_top512_touch_candidate/france-cpu40-vram0gb`

Config delta from accepted SOTA:

- Temporary/default-off source path in `ggml/src/ggml-cpu/ggml-cpu.c`.
- `GGML_MOE_CPU_PREWARM_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_updown_decode_top512.tsv`
- `GGML_MOE_CPU_PREWARM_LIMIT=512`
- `GGML_MOE_CPU_PREWARM_TOUCH=1`
- All accepted gate O_DIRECT pack/cache/top-k envs unchanged.

Initial result:

- `eval_tok_s=4.3`
- `prompt_tok_s=1.3`
- `TTFT=32228.053147 ms`
- TTFT gate: accepted SOTA TTFT `28014.740620 ms`; 20% limit `33617.688744 ms`; candidate is `+15.04%`, so it passes.
- RAM: `memory_peak_bytes=16000000000`, `memory_file_bytes=15085252608`, `ram_ok=true`, `ram_limit_killed=false`.
- Correctness: manual pass. France answer is complete, coherent, and semantically correct.
- Prewarm counters: `enabled=1 entries=512 advised=512 skipped=0 bytes=2281701376 checksum=66057597`.
- Gate pack counters: `hits=4623 misses=0 direct_failures=0 direct_fallbacks=0`.
- Gate VRAM counters: `hits=30528 misses=4623 hit_rate=86.8%`.

Verdict:

- This is a new initial cold-start SOTA candidate over the accepted `4.2 tok/s`, and it satisfies RAM, correctness, and TTFT gates on the first run.
- It is not final until source is committed/pushed and the same path is rebuilt/rerun from the pushed source.

Immediate promotion steps:

1. Commit and push the source, profile, plan, and artifacts immediately to `ssd/vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`.
2. Rebuild from the pushed source so build metadata is clean.
3. Rerun strict cold with the same env and `drop_caches`.
4. Promote only if the pushed-source rerun remains `>4.2 tok/s` and passes RAM, correctness, TTFT, and counter gates.
5. If pushed-source rerun fails, demote this to candidate-only and decide whether to keep source as rejected/default-off or revert runtime source.

### 2026-07-04 Pushed-Source Repro Result For Touch Prewarm

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-prewarm-top512-touch-pushed-repro-result.json`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T180612Z-20260704_cpu_prewarm_top512_touch_pushed_repro/france-cpu40-vram0gb`
- Source commit: `b1fb6e7a5df7884c33a5525cedba1f5e78640f69`
- Build metadata: `b14632-b1fb6e7a5`

Result:

- `eval_tok_s=4.2`
- `prompt_tok_s=1.7`
- `TTFT=32212.962507 ms`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15105830912`
- `ram_ok=true`, `ram_limit_killed=false`
- Correctness: manual pass. France answer is complete, coherent, and semantically correct.
- Prewarm counters: `enabled=1 entries=512 advised=512 skipped=0 bytes=2281701376 checksum=66057597`.
- Gate pack counters: `hits=4623 misses=0 direct_failures=0 direct_fallbacks=0`.
- Gate VRAM counters: `hits=30528 misses=4623 hit_rate=86.8%`.

Verdict:

- Not promoted. The pushed-source rerun tied the accepted `4.2 tok/s` SOTA but did not exceed it.
- The initial `4.3 tok/s` observation is treated as run variance until reproduced.
- Runtime source must be reverted to the accepted path. Keep the profile and artifacts as diagnostic evidence only.
- Current accepted cold-start SOTA remains `4.2 tok/s`.

Rollback action:

- Restore `ggml/src/ggml-cpu/ggml-cpu.c` from commit `41c60565c` and rebuild.
- Commit/push this plan update, pushed-repro artifact, and runtime-source rollback to `ssd/vendor/deepseek-token-rate-16gb`.

### 2026-07-04 Latest Plan After CPU Prewarm Rollback

Current accepted state:

- Accepted cold-start SOTA remains `4.2 tok/s`.
- Accepted source path is restored and pushed as `5484a1806` on `ssd/vendor/deepseek-token-rate-16gb`.
- The `4.3 tok/s` CPU touch-prewarm observation is rejected as unreproduced variance because pushed-source repro returned `4.2 tok/s`.
- Strict constraints remain unchanged: host RAM including page cache must stay inside the 16GB cgroup, France output must be complete/coherent/semantic, TTFT cannot rise more than 20% for an accepted result, and every new SOTA must be committed/pushed plus reproduced from pushed source before promotion.

Reason for changing direction:

- The current hard-bound table shows that top-N up/down residency alone cannot reach the target region: top512 is only about `5.05 tok/s`.
- Even eliminating all CPU up/down fallback is bounded around `8.86 tok/s`, below the long-term `10 tok/s` goal.
- No-drop diagnostics reach `7.2 tok/s`, proving cold page/source stalls are large, but still not enough and not accepted under strict cold-start rules.
- Therefore the next high-leverage path is speculative/MTP or another multi-token verification path. Small page-touch, compact-mmap, and down-cache variants have already failed or tied under the accepted gates.

Immediate next steps:

1. Verify source/remote/binary state:
   - `git status --short` must be clean.
   - local head and `ssd/vendor/deepseek-token-rate-16gb` must include `5484a1806` and this plan update.
   - accepted SOTA binary hash must remain `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62` unless a new accepted source change is promoted.
2. Inspect DS4 GGUF for internal speculative support:
   - completed initial tensor-name check: `mtp=0`, `draft=0`, `eagle=0`, `spec=0`, `next=0`, `head=3` (`hc_head_base`, `hc_head_fn`, `hc_head_scale`), `output=88` normal `blk.*.attn_output_*` tensors;
   - conclusion: no obvious internal MTP/draft/EAGLE tensors are present in the current GGUF, so internal MTP is not the immediate path.
3. Search local model inventory for compatible external draft models:
   - verify tokenizer/vocab compatibility before running;
   - calculate model size, RAM/VRAM budget, expected draft cost, required acceptance rate, and acceptance needed to beat `4.2 tok/s` and approach `10 tok/s`.
4. If a compatible draft/MTP path exists, run only a short strict-cold France diagnostic first:
   - record acceptance rate, target/draft forward counts, TTFT, token rate, RAM/page cache, exact output, and correctness;
   - reject immediately on output divergence, RAM violation, TTFT violation, or throughput below SOTA class.
5. If no viable speculative path exists, return to fallback/source optimization only after a new hard-bound design proves a path above `4.2 tok/s`:
   - split remaining CPU up/down fallback into compute vs source/page-stall cost;
   - preserve the accepted gate cache unless a slot-level calculation proves the tradeoff is worthwhile;
   - avoid repeating rejected pure hotset, touch-prewarm, broad one-stream, down-batch, and compact-mmap sweeps.

Promotion rule for all next work:

- When a compliant new SOTA appears, immediately record full reproducibility metadata, commit and push source/docs/artifacts to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, rebuild from pushed source, and rerun strict cold. Only the pushed-source rerun can become the new accepted SOTA.

### 2026-07-04 Target-Only Lookahead Diagnostic Design

Speculative/MTP feasibility result before this practice:

- Internal DS4 GGUF support is absent by tensor-name check: no `mtp`, `draft`, `eagle`, `spec`, or `next` tensors.
- Local external draft inventory is not compatible:
  - DeepSeek V4 target tokenizer: `tokenizer.ggml.pre=joyai-llm`, `vocab=129280`, `bos=0`, `eos=1`.
  - GLM-5.2 GGUF tokenizer: `tokenizer.ggml.pre=glm4`, `vocab=154880`, `bos=154822`, `eos=154820`.
  - MiniMax and GLM FP8 directories only contain metadata/tokenizer files, not a DS4-compatible small GGUF draft.
- Therefore normal `llama-speculative` / `llama-speculative-simple` with an external draft model is not currently runnable without downloading or creating a compatible draft model.

Next diagnostic:

- Test `llama-lookahead`, which is target-only lookahead decoding. It does not use an external draft model, but verifies multiple candidate n-grams in one target decode using parallel sequences.
- This is distinct from the previously rejected ngram-simple/server speculative path. It may increase output tokens per target decode if n-gram verification acceptance is high enough.
- It is a diagnostic first, not a SOTA candidate. It must run under strict cold `drop_caches` and the same 16GB cgroup/page-cache accounting.

Theory and hard bound:

- Current accepted generation speed is `4.2 tok/s`, roughly `238 ms/output token`.
- `llama-lookahead` has fixed source constants `W=15`, `N=5`, `G=15`; at most `N=5` tokens can be accepted per verification group, so the ideal upper bound is about `5 * 4.2 = 21 tok/s` if target batch overhead were free.
- Real cost is much higher because each iteration evaluates a larger batch with up to roughly `1 + G*(N-1) + (W-1) + W*(N-2) = 120` target positions and `W+G+1 = 31` sequences. If the extra batch work increases per-iteration time by more than the accepted-token gain, throughput will regress.
- To beat SOTA, effective decoded speed from the lookahead log must exceed `4.2 tok/s` and the output must remain semantic/coherent. To be worth deeper work toward `10 tok/s`, the short diagnostic should show either high `n_accept/n_predict` or a decoded speed clearly above SOTA while preserving accepted gate counters.

Planned run:

- Binary: `build-ds4-moe-stream/bin/llama-lookahead`.
- Model/env: accepted SOTA gate O_DIRECT pack/cache/top-k envs unchanged.
- CLI overrides: `-c 256 -b 128 -ub 16 -t 20 -tb 20`, `--n-cpu-moe 40`, `--defer-experts`, `--fit on`, greedy sampling.
- Strict cgroup: `MemoryMax=16000000000`, `MemorySwapMax=0`, global `drop_caches` before run.
- Prompt: `Please introduce France in a short paragraph.`

Acceptance/rejection:

- Promote only if full strict-cold run exceeds `4.2 tok/s`, RAM/correctness/TTFT gates pass, and pushed-source reproduction also exceeds `4.2 tok/s`.
- Reject if `llama-lookahead` fails to fit, lowers gate cache materially, violates 16GB RAM, produces incoherent output, or decoded speed is `<=4.2 tok/s`.

First mechanical result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T183353Z-20260704_lookahead_target_only_strict_cold_wrapper/france-cpu40-vram0gb`
- Wrapper reason: `llama-lookahead` rejects runner's default `--no-display-prompt`; wrapper only removes that flag and forwards all other args.
- Config: `-c 256 -b 128 -ub 16`, accepted SOTA gate envs, strict cold 16GB cgroup.
- Result: model loaded but generation failed after the first token with `llama_decode failed - increase KV cache size`; output was only `France`, so correctness failed and no token-rate result is valid.
- RAM stayed inside cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15169888256`, `oom=0`, `oom_kill=0`.
- Gate counters were already bad before failure: one expert pack `hits=986 misses=178`, gate VRAM cache `hits=151 misses=1164 hit_rate=11.5%`. This suggests lookahead's candidate branches are not aligned with the accepted France gate profile and heavily disturb the SOTA gate cache.

Single retry design:

- Run one more mechanical diagnostic with larger KV/batch room: `-c 512 -b 128 -ub 128`.
- Rationale: the first failure explicitly requested more KV cache, and avoiding ubatch splitting may remove a coupled-sequence split issue.
- If this retry still fails, violates RAM/VRAM, or keeps gate hit rate far below the accepted `86-87%`, reject `llama-lookahead` as a viable SOTA path and do not continue sweeping its parameters.

Second mechanical result and verdict:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T183621Z-20260704_lookahead_target_only_c512_ub128_retry/france-cpu40-vram0gb`
- Config: `-c 512 -b 128 -ub 128`, accepted SOTA gate envs, strict cold 16GB cgroup.
- Result: failed at the same point with `init: sequence 1 is coupled to 0 in the input batch, but have divereged` and `llama_decode failed - increase KV cache size`. Output was only `France`; correctness failed and no token-rate result is valid.
- RAM stayed inside cgroup: `memory_peak_bytes=16000000000`, `memory_file_bytes=15141363712`, `oom=0`, `oom_kill=0`.
- Gate counters again collapsed before failure: one expert pack `hits=986 misses=178`, gate VRAM cache `hits=151 misses=1164 hit_rate=11.5%`.
- Verdict: reject `llama-lookahead` for this SOTA path. It is mechanically incompatible with the current DeepSeek4 memory/coupled-sequence path and would also heavily disturb the accepted gate cache.

### 2026-07-04 No-Draft Ngram-Mod Speculative Diagnostic Design

Rationale:

- `llama-cli` exposes speculative arguments but does not call `common_speculative` in its generation loop.
- `llama-speculative` and `llama-speculative-simple` currently require `--model-draft` before reaching the no-draft `ngram_mod` implementation, even though `common_speculative_init()` supports `COMMON_SPECULATIVE_TYPE_NGRAM_MOD` without a draft model.
- `ngram_mod` is still target-verified: drafted tokens are committed only if the target sampler accepts them from target logits. Therefore output correctness should match target greedy decoding, aside from bugs in context rollback or sampling state.

Implementation plan:

- Patch only `examples/speculative-simple/speculative-simple.cpp`.
- Preserve default behavior when neither `--model-draft` nor `--spec-type` is supplied: still error out.
- If `--spec-type ngram-mod` is supplied without a draft model:
  - skip draft model load;
  - keep `params.speculative.draft.model == nullptr`;
  - let `common_speculative_init()` create the ngram implementation only.
- Improve the final stats line to report `common_speculative_n_max(spec, params_spec)` rather than draft-model `n_max` when no draft model is present.

Theory and bound:

- Each target decode can verify `1 + drafted_tokens` positions. With `n_max=8`, the ideal no-overhead upper bound is about `9 * 4.2 = 37.8 tok/s`.
- Real speed depends on ngram hit rate and acceptance. For the short France prompt, the likely draft rate is low until enough generated text accumulates, so this may tie/regress. It is still worth one short diagnostic because it is a different no-draft algorithm from the already rejected ngram-simple path.
- Use `--spec-type ngram-mod --spec-ngram-mod-n-match 4 --spec-ngram-mod-n-min 1 --spec-ngram-mod-n-max 8` for the first diagnostic to force possible matches without requiring long context.

Practice plan:

- Build `llama-speculative-simple` after the default-off source patch.
- Run strict cold France under 16GB cgroup with accepted SOTA gate envs and `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- If the example binary rejects runner-only args such as `--no-display-prompt`, use a wrapper that only filters incompatible display flags and does not change model/speculative configuration.

Acceptance/rejection:

- Promote only if full strict-cold run exceeds `4.2 tok/s`, RAM/correctness/TTFT/gate counters pass, and pushed-source reproduction also exceeds `4.2 tok/s`.
- Reject and revert the example source patch if throughput is `<=4.2`, correctness fails, target context rollback fails, RAM/TTFT gates fail, or accepted draft rate is too low to justify more work.

### 2026-07-04 No-Draft Ngram-Mod Execution Plan (Historical)

Status note:

- This section records the plan that preceded the no-draft `ngram-mod` diagnostic. It is superseded by the `No-Draft Ngram-Mod Diagnostic Result`, `Gate Cache Headroom Audit Result`, and `Immediate Plan: Gate13312 + Down Cache256 Short Diagnostic` sections below.

Current state:

- Accepted cold-start SOTA remains `4.2 tok/s` from the strict 16GB cgroup vendor DeepSeek path.
- Current pushed branch for all source/docs/artifacts remains `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- `llama-lookahead` is rejected and must not be swept further without a new design, because it fails mechanically on the current DeepSeek4 coupled-sequence path and collapses the gate-cache behavior.
- At the time of this historical plan, the only active source probe was the no-draft `ngram-mod` enablement in `examples/speculative-simple/speculative-simple.cpp`. It has since been rejected and reverted.

Immediate execution order:

1. Close the no-draft `ngram-mod` diagnostic.
   - Verify whether `llama-speculative-simple` accepts the strict runner's `--no-display-prompt`; if not, use a wrapper that removes only this display flag.
   - Run strict cold France with the accepted SOTA env unchanged: gate one-stream name filter `ffn_gate_exps`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, accepted gate frequency profile, O_DIRECT gate expert pack, `cpu_moe=40`, `--vram-cache-gb 0`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
   - Add only the speculative args: `--spec-type ngram-mod --spec-ngram-mod-n-match 4 --spec-ngram-mod-n-min 1 --spec-ngram-mod-n-max 8`.
   - Record: decoded speed, prompt speed if available, TTFT, `n_predict`, `n_drafted`, `n_accept`, acceptance rate, exact France output, RAM/page-cache cgroup stats, pack counters, VRAM cache counters, and any context rollback errors.

2. Decide immediately after the first valid no-draft run.
   - If throughput is `<=4.2 tok/s`, correctness fails, rollback/context fails, RAM exceeds 16GB including page cache, TTFT exceeds the accepted gate, pack direct fallbacks appear, or gate cache hit rate materially drops, reject the probe.
   - On rejection, revert `examples/speculative-simple/speculative-simple.cpp`, rebuild the accepted binary path, record the rejected run and reason in this document, then push only docs/artifacts.
   - If it exceeds `4.2 tok/s` and passes all gates, stop exploration and start the SOTA promotion protocol immediately.

3. SOTA promotion protocol for a passing no-draft result.
   - Record complete reproducibility metadata: run path, exact command/env, source head, build metadata, binary hashes, model/profile/pack hashes, memory including file/page cache, counters, TTFT, token rates, and full France output.
   - Commit source, plan, profile/artifact updates with git identity `L-Ark <fliangae@connect.ust.hk>`.
   - Push to `ssd/vendor/deepseek-token-rate-16gb`.
   - Clean rebuild from the pushed source and rerun strict cold with the same command/env.
   - Promote only if the pushed-source rerun still exceeds `4.2 tok/s` and passes every gate. If it ties or regresses, revert the runtime source and keep it as rejected diagnostic evidence.

4. If no-draft `ngram-mod` is rejected, return to cold-start bottleneck design before coding.
   - Refresh the accepted SOTA profile only if needed, then split the remaining cost into CPU fallback compute, fallback source/page stalls, one-stream source load, synchronization, and scatter.
   - Compute a hard upper bound before implementation. A design whose bound cannot exceed `4.2 tok/s` with margin is rejected on paper.
   - Preserve the accepted gate cache and O_DIRECT pack unless a slot-level calculation proves the tradeoff; current accepted gate cache is the core of the `4.2 tok/s` path.
   - Do not repeat rejected families: pure compact mmap, broad up/down hotset, down-batch staging, no-filter one-stream, `llama-lookahead`, ngram-simple/server partial fallback, or CPU page-touch prewarm sweeps.

Next non-speculative candidate class after rejection:

- Focus on cold page/source stalls without relying on warm global page cache.
- Candidate designs must explicitly account for the 16GB cgroup including file cache and must avoid displacing the accepted gate pack/cache working set.
- The most plausible next design is a narrow, measured source-elimination path for high-impact CPU up/down fallback reads, using O_DIRECT or bounded direct-read staging rather than buffered page-cache prewarm. Before coding, calculate covered bytes, per-call latency, expected removable time, cgroup memory impact, and the resulting token-rate upper bound.

### 2026-07-04 No-Draft Ngram-Mod Diagnostic Result

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T190009Z-20260704_ngram_mod_no_draft_strict_cold/france-cpu40-vram0gb`

Source/probe state:

- Temporary source probe: `examples/speculative-simple/speculative-simple.cpp` allowed `--spec-type ngram-mod` without `--model-draft`.
- Wrapper: `/root/lfz/runs/vendor-ds4-16gb/speculative-simple-filter-wrapper.sh`, filtering only the runner-only `--no-display-prompt` flag because `llama-speculative-simple` rejects it.
- Config: accepted SOTA env unchanged (`cpu_moe=40`, `--vram-cache-gb 0`, gate one-stream `ffn_gate_exps`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, accepted gate profile, O_DIRECT gate expert pack, top-k up/down policy, strict cold `drop_caches`, 16GB cgroup).
- Speculative args: `--spec-type ngram-mod --spec-ngram-mod-n-match 4 --spec-ngram-mod-n-min 1 --spec-ngram-mod-n-max 8`.

Metrics:

- `decoded_speed=2.879 tok/s` (`decoded 193 tokens in 67.043 seconds`)
- `common_perf_eval=3.25 tok/s` (`eval time = 54084.02 ms / 176 runs`)
- `prompt_tok_s=3.10`
- `encoded=8 tokens in 6.587 s`
- `n_predict=193`
- `n_draft=8`
- `n_drafted=32`
- `n_accept=12`
- `accept=37.500%`
- `TTFT=28386.843543 ms`
- `load_ms=19727.96`
- `prompt_eval_ms=20004.95`
- `elapsed_seconds=96.13`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15036387328`
- `pgmajfault=340518`
- `workingset_refault_file=5578109`
- `oom_seen=false`, `ram_limit_killed=false`, `ram_ok=true`
- Correctness: `true`; France output is semantic, coherent, and complete.
- One expert pack counters: `hits=5342 misses=2945 reads=5342 bytes=23806345216 failures=0 entries=4599 direct_enabled=1 direct_reads=5342 direct_failures=0 direct_fallbacks=0`
- VRAM cache counters: `hits=47998 misses=8287 hit_rate=85.3%`

Artifact hashes:

- `summary.json`: `89d5f29f7e3e10c6d09840f68ad97967e5501bacd42c4a1cd7b6ad2c2d7b0b2e`
- `stderr.txt`: `403e43bf035e5bd492f3bc7a03a3e061dce9e271b1d40afb5442aafa4d1af291`
- `stdout.txt`: `427eb63070a31415fec532cdb70b1b6453b8bca7ef6956e0e8e97f9cc198ddae`

France output:

```text
France is a country in Western Europe known for its rich history, culture, and iconic landmarks. It is known for its world-renowned cuisine, fine wines, and fashion. The country is also known for its art, literature, and philosophy. France is a popular tourist destination, with attractions such as the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. The country is also known for its natural beauty, with the French Alps, the Pyrenees, and the beautiful coastline of the French Riviera. Additionally, France is famous for its wine regions, such as Bordeaux, Burgundy, and Champagne, and its cuisine, including dishes like coq au vin and bouillabaisse. The country is also known for its rich history and culture, including the French Revolution and the Enlightenment period. Overall, France is a diverse and fascinating country with a rich cultural heritage and many attractions to explore.

France is a country of unparalleled elegance, refined
```

Verdict:

- Rejected. The run passes RAM, TTFT, and correctness gates, but throughput is below the accepted `4.2 tok/s` SOTA.
- Gap analysis: `ngram-mod` did verify and accept tokens (`37.5%` of drafted tokens accepted), but it introduced extra target/checkpoint overhead and increased total expert activity. The pack reads rose from the accepted France path's `4623` direct reads to `5342`, and additional pack misses were observed for speculative branch traffic. The accepted-token gain was far too small to pay for the extra target work.
- Action: revert `examples/speculative-simple/speculative-simple.cpp`, rebuild `llama-cli`, and do not continue sweeping no-draft ngram parameters without a new theory that changes the overhead/acceptance bound.
- Post-revert accepted binary hashes:
  - `llama-cli`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
  - `libggml-cpu.so.0.10.0`: `a6a3ea2d52fd8001716b56bb2703b686438d485253779079eb7f728494541f2a`

### 2026-07-04 Next Plan After Ngram-Mod Rejection

Current accepted SOTA:

- Still `4.2 tok/s` strict cold vendor DeepSeek under the 16GB cgroup with page cache included.
- Source/runtime path is back to the accepted `llama-cli` state. No rejected speculative source remains staged or dirty.

Next optimization direction:

1. Return to cold-start bottleneck localization before coding.
   - Use accepted SOTA trace/profile artifacts as the starting point and refresh only if required.
   - Re-split decode time into CPU up/down fallback compute, CPU fallback source/page stalls, gate one-stream source load, sync, and scatter.
   - The next design must identify removable time large enough to beat `4.2 tok/s` with margin. Small hotset or speculative sweeps are rejected on paper unless their hard bound changes.

2. Focus on source/page-stall elimination that remains cold-start legal.
   - The no-drop diagnostic at `7.2 tok/s` shows page/source stalls are real, but relying on global warm page cache is not accepted.
   - The next viable class should use bounded direct-read/O_DIRECT staging or another cgroup-accounted cold-start method for high-impact CPU up/down fallback reads.
   - Before implementation, calculate: covered entries, covered bytes, measured latency per miss/read, expected removable time, additional VRAM/RAM use, impact on accepted gate cache slots, and token-rate upper bound.

3. Preserve accepted gate behavior.
   - Keep `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, accepted gate frequency profile, O_DIRECT gate expert pack, and top-k up/down policy unless a slot-level tradeoff calculation proves a larger gain.
   - Stop immediately if gate hit rate falls materially from the accepted `86-87%`, direct pack failures appear, RAM including file cache exceeds 16GB, TTFT exceeds the accepted gate, or France output quality fails.

4. Promotion protocol remains mandatory.
   - Any compliant new SOTA must be fully recorded, committed, and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`.
   - After push, rebuild from pushed source and rerun strict cold. Only the pushed-source rerun can replace the accepted SOTA.

### 2026-07-04 Gate Cache Headroom Audit Design

Reason:

- The previous down-cache experiments failed mainly because accepted gate cache consumes almost all VRAM. With `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, the accepted path reports about `238MiB` CUDA free and `3192` gate slots.
- A tiny `256MiB` down-cache while preserving the full gate cache could not allocate a useful buffer and produced `0%` down-cache hit rate.
- Reducing context from `-c 256` to `-c 128` did not free useful VRAM. Therefore the only obvious VRAM knob is the gate cache itself.

Diagnostic:

- Run one no-source-change strict cold France audit with `GGML_MOE_STREAM_ONE_CACHE_MIB=13312`.
- This reduces the gate pool by `256MiB`, about `60` slots at `4.25MiB/slot`, from `3192` slots to about `3132` slots.
- All other accepted SOTA envs remain unchanged: same O_DIRECT gate expert pack, same gate admission profile, same top-k up/down pruning, same `cpu_moe=40`, same `--vram-cache-gb 0`, same `-c 256 -b 16 -ub 16 -t 20 -tb 20`, strict cold `drop_caches`, and 16GB cgroup.

Theory and bound:

- This run is not expected to exceed SOTA by itself. Its purpose is to test whether freeing about `256MiB` of VRAM materially hurts the accepted gate path.
- If the gate hit rate stays near the accepted `86-87%` and eval remains in the `4.1-4.2 tok/s` class, the freed VRAM may justify a separate down-cache allocation probe.
- If gate hit rate falls materially or throughput drops below the accepted class, then down-cache composition is rejected on paper because the slot tradeoff destroys the SOTA base path before adding any down benefit.

Acceptance/rejection for this audit:

- Continue to a down-cache design only if:
  - RAM including page cache stays inside `16000000000` bytes;
  - France output remains semantic, coherent, and complete;
  - TTFT remains within the accepted 20% gate;
  - one expert pack has no direct failures/fallbacks;
  - gate VRAM hit rate stays close to the accepted `86.8%`;
  - token rate remains at least SOTA-class (`>=4.1 tok/s`), preferably tied at `4.2 tok/s`.
- Reject the gate-cache-shrink branch if this audit regresses throughput, correctness, RAM, TTFT, pack counters, or gate hit rate. Do not run a down-cache combination after a failed gate-only audit.

### 2026-07-04 Gate Cache Headroom Audit Result

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T191757Z-20260704_gate_cache_13312_headroom_audit/france-cpu40-vram0gb`

Config:

- Same accepted SOTA runtime path and runner constraints: vendor DeepSeek, strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, `cpu_moe=40`, `--vram-cache-gb 0`, O_DIRECT gate expert pack, current gate admission profile, and CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Only intentional change from accepted SOTA config: `GGML_MOE_STREAM_ONE_CACHE_MIB=13312` instead of `13568`.

Metrics:

- `eval_tok_s=4.1`
- `prompt_tok_s=1.5`
- `TTFT=29281.374145 ms`
- `elapsed_seconds=62.33`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15103705088`
- `pgmajfault=276959`
- `workingset_refault_file=1724867`
- `ram_ok=true`, `oom_seen=false`, `ram_limit_killed=false`
- Correctness: `true`; France output is semantic, coherent, and complete.

Counters and VRAM:

- Gate one-stream cache init: `13.0 GiB`, `3132` slots.
- CUDA free after model/cache allocation: about `492MiB`.
- One expert pack counters: `hits=4639 misses=0 reads=4639 bytes=20673462272 failures=0 entries=4599 direct_enabled=1 direct_reads=4639 direct_failures=0 direct_fallbacks=0`.
- Gate VRAM cache counters: `hits=30512 misses=4639 hit_rate=86.8%`.

Artifact hashes:

- `summary.json`: `4a89440340363f67623dd8b7d56232d8279d4367a616774d8689f57aca706e86`
- `stderr.txt`: `569547c6aaaa9a8475908f08537148c1e1fd5a6f4751f5fe77f154852951cb26`
- `stdout.txt`: `ebf9dae6b55d32c03fcc9ab6a806c95814cddc60fd515b075d610222f96ed638`

Verdict:

- Passed as a headroom audit, not a new SOTA. Throughput remains SOTA-class (`4.1 tok/s`) and all RAM, correctness, TTFT, pack, and gate-hit gates pass.
- The gate hit rate did not regress from the accepted `86.8%`, while CUDA free increased enough to justify exactly one small down-cache composition probe.
- Do not promote `13312MiB` gate cache by itself. It is only a prerequisite for the next short diagnostic.

### 2026-07-04 Immediate Plan: Gate13312 + Down Cache256 Short Diagnostic

Bottleneck being targeted:

- Current accepted SOTA still spends the largest removable time in CPU up/down fallback: about `25.7s` total, about `18.1s` during decode.
- Prior down-batch experiments showed the CUDA kernel itself is cheap (`~0.035 ms/call`), but host-to-device staging dominated (`~4.538 ms/call`) and the large down-cache displaced the gate cache, dropping gate hit rate to about `68.9%`.
- The gate headroom audit shows that reducing the gate cache by only `256MiB` does not materially hurt gate hit rate. The next test asks whether that newly freed VRAM can support a useful down cache without repeating the prior large-cache regression.

Theory and bound:

- This is not expected to reach `10 tok/s` by itself. It is a narrow diagnostic to test whether down-cache allocation and staging can be made compatible with the accepted gate path.
- A useful signal requires all of the following:
  - `GGML_MOE_VRAM_CACHE_MIB=256` allocates a real cache instead of falling back to a tiny unusable allocation;
  - down-cache hits are nonzero and meaningful on a short France run;
  - gate hit rate remains near `86.8%`;
  - down batch `stage ms/call` drops materially below the prior accepted-probe value `4.538 ms/call`, and preferably below the previous top512-down staging value `3.705 ms/call`;
  - short-run throughput does not collapse below the SOTA class.
- If the short diagnostic cannot meet these gates, the full run is rejected on paper because the hard bound remains below the current accepted `4.2 tok/s` once staging/gate-regression overhead is included.

Implementation constraints:

- Patch only `ggml/src/ggml-cuda/moe_stream_batch.cu`, default-off, for the short diagnostic.
- Required temporary probe support:
  - add `GGML_TYPE_MXFP4` to `moe_stream_type_supported()`;
  - add `GGML_TYPE_MXFP4` to the compact MMVQ launcher type switch;
  - if the MMQ slot launcher is reached, add `GGML_TYPE_MXFP4` there as well;
  - preserve or restore the compact dst workspace correctness fix with `dst_tmp_rows=max(dst_cols,n_active)` where applicable.
- Build only the batch-probe binary first: `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream-batch-probe/bin/llama-cli`.
- Do not commit this source patch unless it produces a compliant new SOTA and then passes pushed-source reproduction.

Short diagnostic command shape:

- Binary: `build-ds4-moe-stream-batch-probe/bin/llama-cli`
- Runner: `.Agent/run-tools/strict_ds4_runner.py`
- Strict constraints: cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, France prompt, `cpu_moe=40`, `--vram-cache-gb 0`, `-n 32`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Keep accepted SOTA gate envs, except use `GGML_MOE_STREAM_ONE_CACHE_MIB=13312`.
- Add:
  - `GGML_MOE_STREAM_DOWN_BATCH=1`
  - `GGML_MOE_VRAM_CACHE_MIB=256`
  - `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top512-down-20260703.pack`
  - `GGML_MOE_IO_BACKEND=iouring`
  - `GGML_MOE_STAGE_PINNED=1`
  - `GGML_MOE_STAGE_PINNED_SLOTS=4`

Required records:

- Exact run path, full command/env, source diff, source head, build hash, model/profile/pack hashes, token rates, TTFT, elapsed time, RAM and file/page cache usage, OOM counters, exact France output prefix/full output, correctness verdict, gate pack counters, gate VRAM cache counters, down-cache allocation line, down-cache hit/miss counters, down-batch accept/decline counters, and stage/quant/kernel/D2H/scatter timings.

Decision rules:

- If the short diagnostic fails allocation, has zero/near-zero down-cache hits, collapses gate hit rate, exceeds RAM, fails correctness, triggers direct pack failures, or remains far below the accepted SOTA class, immediately revert `moe_stream_batch.cu`, rebuild clean, record rejection, and push only docs/artifacts.
- If the short diagnostic passes all gates, update this plan before a full run, then run full strict cold France under the same constraints.
- A full run may be promoted only if it exceeds `4.2 tok/s`, keeps TTFT within the accepted 20% gate, passes correctness and RAM including page cache, and has no pack/direct failures.
- On any compliant new SOTA, stop exploration immediately, record complete reproducibility information, commit source/docs/profiles/artifacts, push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then clean rebuild from pushed source and rerun strict cold before declaring it accepted.

### 2026-07-04 Gate13312 + Down Cache256 Short Diagnostic Result

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T194034Z-20260704_gate13312_downcache256_short/france-cpu40-vram0gb`

Temporary source probe:

- `ggml/src/ggml-cuda/moe_stream_batch.cu` only, never committed.
- Added `GGML_TYPE_MXFP4` to `moe_stream_type_supported()`.
- Added `GGML_TYPE_MXFP4` to compact MMVQ launcher support and MMQ slot launcher switch.
- Restored the up/gate temporary workspace guard by sizing `d_up`, `d_gate`, and related debug copies with `tmp_dst_rows=max(dst_cols,n_active)`.
- Built only `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream-batch-probe/bin/llama-cli`.

Config:

- Strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`.
- `cpu_moe=40`, `--vram-cache-gb 0`.
- Accepted SOTA gate envs preserved except `GGML_MOE_STREAM_ONE_CACHE_MIB=13312`.
- Added `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_VRAM_CACHE_MIB=256`, `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top512-down-20260703.pack`, `GGML_MOE_IO_BACKEND=iouring`, `GGML_MOE_STAGE_PINNED=1`, `GGML_MOE_STAGE_PINNED_SLOTS=4`, `GGML_MOE_BATCH_PROFILE=1`, `GGML_MOE_STREAM_DECLINE_DEBUG=1`.
- Short run args: `-n 32 -c 256 -b 16 -ub 16 -t 20 -tb 20`.

Metrics:

- `eval_tok_s=1.6`
- `prompt_tok_s=1.5`
- `TTFT=32025.669104 ms`
- `elapsed_seconds=48.77`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15087927296`
- `memory_max_events=8456`
- `pgmajfault=149689`
- `workingset_refault_file=166949`
- `ram_ok=true`, `oom_seen=false`, `ram_limit_killed=false`
- `correctness_ok=false`
- `correctness_reason=too_short,missing_europe,missing_expected_context`

Output:

```text
and each paragraph should be approximately 30 words, and in English. Summarize with a conclusion that includes 3 key takeaways.

**France: A Brief
```

This is not a valid France answer. Even ignoring the `-n 32` truncation, the generated text is semantically wrong for the requested prompt.

Counters:

- Gate one-stream cache init: `13.0 GiB`, `3132` slots.
- Down batch cache allocation succeeded: requested `256 MiB`, actual `256 MiB`, `60` slots of `4.25 MiB`.
- CUDA memory at report time: free about `164 MiB`, self/model about `17362 MiB`, unaccounted about `14582 MiB`.
- Down batch expert pack: `hits=815 misses=3262 read_failures=0 direct_reads=815 direct_fallbacks=0 iouring_reads=0 iouring_bytes=0 iouring_fallbacks=0 entries=512`.
- Down batch VRAM cache: `hits=1 misses=4077 preloads=0 hit_rate=0.0%`.
- Down batch profile: `calls=1243 avg_active=3.28 stage=5.922 ms quant=0.000 ms kernel=0.032 ms d2h=0.005 ms scatter=0.008 ms total=5.968 ms/call wall=5.981 ms/call`.
- Gate one expert pack: `hits=2679 misses=1237 reads=2679 bytes=11938824192 failures=0 direct_reads=2679 direct_failures=0 direct_fallbacks=0`.
- Gate VRAM cache: `hits=5512 misses=3916 hit_rate=58.5%`.

Artifact hashes:

- `summary.json`: `bf3b78ff06c1e6ab59e3d229a569f7e2ecaf9fa0a7e9c6d3ac9d8ea4c3315bfd`
- `stdout.txt`: `ec222664cdcb04ee875702033800815f1a5830657d504b7d07094ee3e8eac1f6`
- `stderr.txt`: `6351e5c2e20c1f7092dc9a6afacdce2b12c75e635465937342c845b983cb821a`

Verdict:

- Rejected. Do not run full strict cold for this candidate.
- Although the `256MiB` down cache allocated, it provided effectively no reuse (`0.0%` hit rate), made staging slower than previous down-batch probes (`5.922 ms/call` vs prior `4.538 ms/call` and top512-down `3.705 ms/call`), and destroyed the accepted gate-cache behavior (`58.5%` vs accepted `86.8%`).
- Correctness failed. This candidate cannot be considered an optimization even as an unaccepted TTFT-over-gate result.
- The likely failure modes are: too-small down cache causing LRU churn, down pack coverage too sparse for the short generation route, and/or unresolved MXFP4 down numerical mismatch. The next step must isolate correctness before any performance run.

Rollback:

- `ggml/src/ggml-cuda/moe_stream_batch.cu` was reverted with `git restore`.
- `build-ds4-moe-stream-batch-probe` was rebuilt from clean source.
- Post-rollback hashes:
  - `build-ds4-moe-stream-batch-probe/bin/llama-cli`: `866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`
  - `build-ds4-moe-stream-batch-probe/bin/libggml-cuda.so`: `d451bf606ade4a6a868283d6796dfe9dad9f577151be593b095c400b80d423da`
  - accepted `build-ds4-moe-stream/bin/llama-cli`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
- Worktree source was clean after rollback.

### 2026-07-04 Next Plan After Down Cache256 Rejection

Current accepted SOTA:

- Still `4.2 tok/s` from the strict cold vendor DeepSeek path with accepted gate O_DIRECT pack, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `cpu_moe=40`, and 16GB cgroup including page cache.
- No rejected runtime source remains active. The latest down-cache/MXFP4 probe source was reverted.

Immediate direction:

1. Correctness-first down/MXFP4 diagnosis before any more down performance runs.
   - Use existing CPU compare trace support (`GGML_MOE_STREAM_COMPARE_CPU_OUT`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT`, and related block trace support) or a narrow deterministic harness to compare CPU fallback vs CUDA down-batch outputs on the same MXFP4 down experts and source rows.
   - Start with a very small limit under the 16GB cgroup and accepted SOTA envs. The goal is not token rate; the goal is to prove whether MXFP4 down batch is numerically correct route by route.
   - Record max absolute error, mean error, bad/non-finite counts, tensor name, expert id, row id, source row metadata, and whether the first wrong generation token aligns with a bad down result.

2. Reject performance experiments until correctness is proven.
   - Do not re-enable `GGML_MOE_STREAM_DOWN_BATCH=1` for generation as an optimization candidate unless the compare harness shows acceptable numerical agreement and the France output is semantic and coherent.
   - Do not repeat the `256MiB` down-cache composition without a new cache admission/pinning design; its measured hit rate was `0.0%`.

3. If down/MXFP4 correctness is proven later, redesign cache admission before a full run.
   - A useful down-cache design must pin a small route-aware hotset instead of relying on 60-slot LRU churn.
   - It must preserve gate hit rate near `86.8%`, show nonzero down-cache hit rate in short diagnostics, and keep stage time below prior rejected values before any full strict cold run.

4. If down/MXFP4 correctness fails, abandon down-batch compute for the current SOTA path.
   - Return to cold-legal source/page-stall elimination or a different target-verified speculative design with a hard upper bound above `4.2 tok/s`.
   - Any new design must update this plan before implementation and must preserve RAM, correctness, and TTFT gates.

### 2026-07-04 Corrected Down Batch Compare Design

Correction to the previous rejection:

- The `20260704_gate13312_downcache256_short` probe intentionally restored the up/gate temporary workspace guard, but it did not restore the known down compact dst guard.
- Historical diagnostic `20260702-mxfp4-down-batch-dstrows-diagnostic-results` already identified the same failure mode: compact down batch writes `n_active` temporary rows (`tmp + j*ne01`), while the broken allocation/copy path used only `dst_cols=max_dst_id+1`. With top-k pruning, `dst_cols` can be smaller than `n_active`, causing overflow/repeated rows and wrong generation.
- Therefore the latest wrong output is not enough to conclude MXFP4 down arithmetic itself is bad. The next experiment must restore the down `tmp_dst_rows=max(dst_cols,n_active)` fix and verify with CPU compare before any performance claim.

Temporary source plan:

- Patch `ggml/src/ggml-cuda/moe_stream_batch.cu`:
  - add `GGML_TYPE_MXFP4` to the batch-supported type list;
  - add `GGML_TYPE_MXFP4` to compact MMVQ launcher support;
  - optionally add `GGML_TYPE_MXFP4` to MMQ slot launcher only if the existing template instance compiles;
  - in `ggml_cuda_moe_stream_batch()`, allocate/copy/zero down temporary `d_dst` and `h_dst` using `tmp_dst_rows=max(dst_cols,n_active)` rather than `dst_cols`.
- Patch `ggml/src/ggml-cpu/ggml-cpu.c` default-off:
  - after a successful `ggml_cuda_moe_stream_batch()` call and before clearing `matrix_row_counts`, call `ggml_moe_stream_compare_cpu_result()` for active experts when `GGML_MOE_STREAM_COMPARE_CPU_OUT` is set;
  - add optional `GGML_MOE_STREAM_COMPARE_CPU_NAME_FILTER` so the compare hook can be scoped to `ffn_down_exps`.

Diagnostic run:

- Strict cold 16GB cgroup, short correctness/compare diagnostic, not a SOTA candidate.
- Use the corrected batch-probe binary with:
  - accepted SOTA gate envs;
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13312`;
  - `GGML_MOE_STREAM_DOWN_BATCH=1`;
  - `GGML_MOE_VRAM_CACHE_MIB=256`;
  - down top512 pack;
  - `GGML_MOE_BATCH_PROFILE=1`;
  - `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`;
  - `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=32`;
  - `GGML_MOE_STREAM_COMPARE_CPU_NAME_FILTER=ffn_down_exps`;
  - keep runner default `-n 192` so the France correctness gate can pass; override only `-c 256 -b 16 -ub 16 -t 20 -tb 20`.

Decision rules:

- If compare shows large numerical error, non-finite output, wrong text, RAM failure, or the same near-zero down-cache hit/stage regression pattern, reject and revert the source again.
- If compare passes but performance/cache remains bad, reject as a correctness-only success and return to cache admission/staging design.
- If compare passes and short performance/cache signals unexpectedly improve, update this plan before a full strict cold correctness run.
- No source from this diagnostic may be committed unless it becomes part of a compliant new SOTA and passes pushed-source reproduction.

### 2026-07-04 Corrected Down Batch Compare Result

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T195615Z-20260704_corrected_down_batch_compare_full/france-cpu40-vram0gb`

Temporary source probe:

- `ggml/src/ggml-cuda/moe_stream_batch.cu`: enabled MXFP4 for batch diagnostics and fixed down compact temporary dst allocation/copy/zero size to `tmp_dst_rows=max(dst_cols,n_active)`.
- `ggml/src/ggml-cpu/ggml-cpu.c`: added default-off batch CPU compare after successful `ggml_cuda_moe_stream_batch()` and optional `GGML_MOE_STREAM_COMPARE_CPU_NAME_FILTER`.
- Source was diagnostic-only and was reverted after the run.

Config:

- Strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`.
- Accepted SOTA gate envs preserved except `GGML_MOE_STREAM_ONE_CACHE_MIB=13312`.
- `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_VRAM_CACHE_MIB=256`, down top512 pack, `GGML_MOE_IO_BACKEND=iouring`, pinned staging with 4 slots, `GGML_MOE_BATCH_PROFILE=1`.
- `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=32`, `GGML_MOE_STREAM_COMPARE_CPU_NAME_FILTER=ffn_down_exps`.
- Kept runner default `-n 192`; CLI override only `-c 256 -b 16 -ub 16 -t 20 -tb 20`.

Metrics:

- `eval_tok_s=2.6`
- `prompt_tok_s=1.5`
- `TTFT=31759.606672 ms`
- `elapsed_seconds=103.47`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15061602304`
- `memory_max_events=20139`
- `pgmajfault=224308`
- `workingset_refault_file=2376126`
- `ram_ok=true`, `oom_seen=false`, `ram_limit_killed=false`
- `correctness_ok=true`, `correctness_reason=heuristic_pass_manual_review_required`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. Renowned for its art, fashion, cuisine, and landmarks like the Eiffel Tower, the Louvre, and the Palace of Versailles, France is a major center for culture and history. It is a democratic republic with a strong economy, being a leader in industries such as aerospace, automotive, and luxury goods. France is also famous for its wine regions, such as Bordeaux and Burgundy, and its iconic landmarks, including the Eiffel Tower and the Louvre Museum. The country is a popular tourist destination, attracting millions of visitors each year. France is also known for its cuisine, wine, and fashion, and is a leader in these industries. The country has a rich history and culture, and is a major player in the global economy. France is a founding member of
```

The answer is semantically about France, coherent, and passes the current correctness heuristic, though it is repetitive and ends mid-sentence due the fixed `-n 192` budget.

Compare result:

- `compare_cpu.csv` rows: `32` data rows.
- `max_abs_max=1.1920929e-07`
- `max_mean_abs=1.41827e-08`
- `mean_of_mean_abs=5.27908e-09`
- `max_rel=8.37862e-07`
- Verdict: MXFP4 down batch is numerically correct after the down `tmp_dst_rows` fix, matching the earlier 2026-07-02 diagnosis.

Counters:

- Gate one-stream cache init: `13.0 GiB`, `3132` slots.
- Down batch cache: requested `256 MiB`, actual `256 MiB`, `60` slots of `4.25 MiB`.
- CUDA memory at report time: free about `164 MiB`.
- Down batch expert pack: `hits=14360 misses=10529 read_failures=0 direct_reads=14360 direct_fallbacks=0 iouring_reads=0 iouring_bytes=0 iouring_fallbacks=0 entries=512`.
- Down batch VRAM cache: `hits=1 misses=24889 preloads=0 hit_rate=0.0%`.
- Down batch profile: `calls=7644 avg_active=3.26 stage=4.740 ms quant=0.000 ms kernel=0.031 ms d2h=0.005 ms scatter=0.008 ms total=4.786 ms/call wall=4.797 ms/call`.
- Pinned staging: `copies=24889 waits=24885 fallbacks=0 slots=4 slot=4.25 MiB host_stage=34500.221 ms h2d=4193.746 ms`.
- Gate one expert pack: `hits=5071 misses=1235 reads=5071 bytes=22598647808 failures=0 direct_reads=5071 direct_failures=0 direct_fallbacks=0`.
- Gate VRAM cache: `hits=41562 misses=6306 hit_rate=86.8%`.

Artifact hashes:

- `summary.json`: `3a964083cbd3d0582f3075ffb8a61a766a84c568bfdd5cac149568c2ca5f13de`
- `stdout.txt`: `6d6bdf42b0b1dc1d93c03a4f1f09a78103eba1aa0a4a3346ecc2fc7c4e682d4d`
- `stderr.txt`: `fbf2467da71505c51909301e06440f2d018c87c34d4d59125261bd6a21cd6046`
- `compare_cpu.csv`: `bee5bd2f15bd33eb3d21ed9ba2d6b1a1a8fa02818fb4f07a926fa9896a1a420d`

Verdict:

- Correctness success, performance rejected. This is not a SOTA and must not be promoted.
- The arithmetic/root-cause question is resolved: the previous wrong output came from missing the down `tmp_dst_rows=max(dst_cols,n_active)` fix, not from inherent MXFP4 down arithmetic error.
- The remaining bottleneck is staging/cache locality: the CUDA kernel is only `0.031 ms/call`, while stage is `4.740 ms/call`, and the 60-slot down cache still has effectively no reuse.
- Gate behavior stayed healthy (`86.8%` hit rate), so the `13312MiB` gate-cache headroom is not the problem. The down cache admission/LRU policy and pack/source movement are the problem.

Rollback:

- Reverted `ggml/src/ggml-cpu/ggml-cpu.c` and `ggml/src/ggml-cuda/moe_stream_batch.cu`.
- Rebuilt clean `build-ds4-moe-stream-batch-probe`.
- Post-rollback hashes:
  - `build-ds4-moe-stream-batch-probe/bin/llama-cli`: `866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`
  - `build-ds4-moe-stream-batch-probe/bin/libggml-cuda.so`: `93c83225fcdfd83aeddfe59238b1f8430cf285d0583c282b40dce1baf95c044b`
  - `build-ds4-moe-stream-batch-probe/bin/libggml-cpu.so`: `57c7bd0544ae998a69c5f00f35b5a29a6147a811b10ea37083b022d13f878183`
  - accepted `build-ds4-moe-stream/bin/llama-cli`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
- Worktree source was clean after rollback; only this plan document remained modified.

### 2026-07-04 Next Plan After Corrected Down Batch Compare

Current accepted SOTA:

- Still `4.2 tok/s`; this run is only `2.6 tok/s`.
- Down-batch correctness is now proven for the sampled MXFP4 routes, but down-batch performance is rejected.

Next direction:

1. Stop testing down-batch arithmetic. It is correct with the known `tmp_dst_rows` fix.
2. Focus only on staging/cache locality if continuing down-batch:
   - design a route-aware pinned/admission policy rather than 60-slot LRU;
   - compute expected hit coverage from the actual down route trace before coding;
   - require nonzero meaningful down-cache hit rate and stage time below `3.705 ms/call` in a short diagnostic before any full run.
3. If a cache admission design cannot show a hard bound above `4.2 tok/s`, abandon down-batch and return to CPU fallback source/page-stall elimination.
4. Before the next implementation, update this plan with the exact cache policy, expected covered calls/bytes, VRAM cost, gate hit impact, theoretical token-rate bound, and rejection criteria.

### 2026-07-04 Down Cache Admission Bound: No-Run Decision

Inputs:

- Corrected down batch compare run: `eval_tok_s=2.6` for `-n 192`, so approximate generation time is `192/2.6 = 73.8s`.
- Accepted SOTA target to beat: `4.2 tok/s`, approximate generation time `192/4.2 = 45.7s`.
- Required saving from the corrected down-batch path to merely tie/beat SOTA: about `28.1s`.
- Corrected down-batch measured staging:
  - `calls=7644`
  - `stage=4.740 ms/call`
  - `stage_total≈36.24s`
  - runtime route copies/misses: `24889`
  - implied average staging cost per route copy: about `1.46 ms`.
- Down-only hotset metadata:
  - `/root/lfz/runs/vendor-ds4-16gb/20260703T042727Z-next-down-only-hotsets/decode_top512_down_meta.tsv`
  - sha256 `4a0db23c4e2bf56bcc19cd4288435161d67508a54d1f2e263625818adc058131`

Coverage table:

| Hotset | fallback_ms | route calls | payload |
| --- | ---: | ---: | ---: |
| top34 down | `1195.564` | `3316` | `144.5 MiB` |
| top60 down | `1755.483` | `4771` | `255.0 MiB` |
| top128 down | `2673.535` | `6754` | `544.0 MiB` |
| top256 down | `3785.421` | `8483` | `1088.0 MiB` |
| top512 down | `5319.314` | `10623` | `2176.0 MiB` |

Hard-bound estimate:

- A perfect 60-slot pinned down cache can only cover about `4771` of `24889` route copies. At the measured `~1.46 ms/copy`, the upper-bound stage saving is about `6.9s`.
- Corrected down batch would improve from roughly `73.8s` to `66.9s`, or only about `2.9 tok/s`, still far below the accepted `4.2 tok/s`.
- Even a perfect top512 down hotset would cover about `10623` route copies, upper-bound stage saving about `15.5s`, giving roughly `3.3 tok/s`, still below accepted SOTA and requiring `2.125 GiB` of payload that cannot fit without materially displacing the gate cache.
- Removing all measured down stage time would reach about `5.1 tok/s`, but that requires eliminating nearly all `24889` route loads, not a small pinned hotset. This would need a much larger resident set or a fundamentally different read/compute pipeline.

Decision:

- Do not implement a 60-slot or top128 pinned/admission down cache. Its hard upper bound is below the accepted `4.2 tok/s` SOTA.
- Do not spend more runs on small down-cache admission policies unless the design changes the amount of removable stage time, not merely which `255-544 MiB` subset is cached.
- The next implementation must target a larger bottleneck class: either combined up+down fallback elimination with a new movement model, or cold-legal CPU fallback source/page-stall reduction that does not trade away gate cache.

### 2026-07-04 Latest Plan: Cold-Legal Page-Pressure Diagnostic, Then Larger Fallback Movement

Current accepted SOTA remains:

- `eval_tok_s=4.2`
- strict cold `drop_caches`
- 16GB cgroup including page cache
- `MemorySwapMax=0`
- `correctness_ok=true`
- `TTFT=28014.740620 ms`
- accepted run: `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`
- accepted binary hash: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
- source/record branch for all accepted and rejected work: `ssd/vendor/deepseek-token-rate-16gb`

Latest bottleneck refresh:

| Metric | Cold current-SOTA profile | No-drop diagnostic | Difference |
| --- | ---: | ---: | ---: |
| eval tok/s | `4.1` | `7.2` | diagnostic-only upside |
| prompt tok/s | `1.5` | `2.0` | faster prompt with warm external cache |
| TTFT ms | `29663.442807` | `26654.802489` | `-3008.640318` |
| fallback total ms | `25743.677` | `12952.670` | `-12791.007` |
| fallback up ms | `12387.490` | `5873.231` | `-6514.259` |
| fallback down ms | `13356.187` | `7079.439` | `-6276.748` |
| prompt fallback ms | `7654.397` | `5351.252` | `-2303.145` |
| decode fallback ms | `18089.280` | `7601.418` | `-10487.862` |

Diagnostic sources:

- Cold current-SOTA profile: `/root/lfz/runs/vendor-ds4-16gb/20260703T172436Z-20260704_phaseB_current_sota_profile_head4be08352/france-cpu40-vram0gb/fallback-profile.csv`
- No-drop diagnostic profile: `/root/lfz/runs/vendor-ds4-16gb/20260703T173550Z-20260704_phaseC_nodrop_profile_io_compute_split_head6b11e5a/france-cpu40-vram0gb/fallback-profile.csv`

Interpretation:

- CPU fallback is still the dominant accepted-run bottleneck. In the cold profile it accounts for about `25.7s`, split almost evenly between up and down.
- The no-drop diagnostic proves that cold source/page stalls are real: fallback time drops by about `12.8s`, and decode fallback drops by about `10.5s`.
- The no-drop result is not acceptable as SOTA because it relies on globally warm page cache prepared outside the strict cold run. It is only a ceiling/proof of where time is being lost.
- Page/source-stall reduction alone cannot get to `10 tok/s`. For `-n 192`, `4.2 tok/s` implies about `45.7s` generation time, while `10 tok/s` would require about `19.2s`; the no-drop gap explains only about half of the required saving.
- Small down-cache admission is rejected by hard bound. The next accepted improvement must avoid stealing meaningful gate-cache capacity and must reduce either broad fallback source stalls or the fallback movement model itself.

Immediate experiment design: cold-legal dense mmap/page-cache pressure control.

Hypothesis:

- The accepted cold run keeps about `15.1GB` of file-backed pages in the 16GB cgroup. Even with expert mmap pages dropped after model load, dense mmap pages may compete with later up/down fallback pages during decode.
- Dropping dense mmap pages after prompt processing is cold-legal because it happens inside the same strict run/cgroup and does not rely on external warm cache.
- If dense pages are causing reclaim/refault pressure, `LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1` should reduce decode fallback stalls without reducing gate VRAM cache size.

Run plan:

1. Run one strict full France prompt diagnostic from the accepted SOTA env, with only this added env:
   - `LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1`
2. If the log shows no dense-drop activity or no measurable change, run a second strict diagnostic with:
   - `LLAMA_DROP_DENSE_MMAP_CACHE=1`
   - `LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1`
3. Keep all accepted SOTA knobs unchanged:
   - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
   - current gate admit profile
   - current gate expert pack with direct IO
   - `GGML_MOE_KEEP_TOPK_UPDOWN=4`
   - `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`
   - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`
   - `-c 256 -b 16 -ub 16 -t 20 -tb 20`
4. Collect:
   - token rates
   - TTFT
   - elapsed seconds
   - full France answer
   - cgroup `memory_peak_bytes`, `memory_file_bytes`, OOM/kill flags
   - `pgmajfault`, `workingset_refault_file`
   - gate cache hit/miss/hit rate
   - pack direct-read counters
   - fallback profile totals if enabled
   - stderr evidence for dense/expert mmap drops

Hard upper bound:

- Absolute ceiling for this class is the no-drop diagnostic at about `7.2 tok/s`, because dense-drop only tries to recover part of the cold/page-stall gap.
- Expected practical upside is smaller than `12.8s`, since dense-drop cannot make all up/down source pages resident and cannot remove CPU fallback math.
- This experiment is worth one or two no-source diagnostics because it is low-risk, preserves VRAM distribution, and directly tests a current bottleneck without another down-cache implementation.

Acceptance criteria:

- `memory_peak_bytes <= 16000000000` and page cache is charged inside the same 16GB cgroup.
- No swap and no cgroup kill.
- France output is semantic, coherent, and not a correctness regression.
- TTFT must stay within `20%` of the current accepted SOTA TTFT unless explicitly recorded as rejected:
  - conservative limit: `28014.740620 * 1.20 = 33617.688744 ms`
- `eval_tok_s` must be strictly above the accepted `4.2 tok/s` before promotion.
- If the gain is only a tie, record it as useful diagnostic/headroom evidence but do not promote it as SOTA.

Rejection criteria:

- Any RAM, OOM, correctness, or TTFT violation rejects the run.
- Any token-rate regression rejects the run.
- If fallback/profile counters show no reduction in major faults/refaults/fallback time, stop this class; do not keep sweeping dense-drop combinations.

Fallback plan if dense-drop diagnostics reject:

1. Return to design mode before any new source change.
2. Add or use instrumentation that splits fallback time into:
   - source/page-fault wait;
   - tensor staging/copy;
   - CPU math;
   - pack/direct-IO read time;
   - per-layer/per-expert route frequency.
3. Use that split to compute a hard upper bound for a larger movement-model change. The next source implementation must show a bound above `4.2 tok/s` before coding.
4. Candidate larger movement classes:
   - route-ordered up+down expert pack that reduces random fallback source movement for both tensors, not only down;
   - bounded async/O_DIRECT staging that overlaps source reads with current compute while staying inside the 16GB cgroup;
   - a fused up/gate/down fallback path only if the math and dataflow show real parallelism or eliminated movement, not merely a CUDA kernel speedup with the same source stalls;
   - CPU fallback kernel/layout improvements only after page/source stalls are proven secondary.

Mandatory record/push rule:

- Before every implementation run, update this plan with the exact hypothesis, bound, run config, and rejection criteria.
- For every run, record full metrics and the exact output answer.
- If a compliant new SOTA appears, immediately record all reproduction inputs, artifact hashes, run directory, command/env, correctness output, memory counters, and push source plus docs to `ssd/vendor/deepseek-token-rate-16gb`.
- After pushing a new SOTA, clean rebuild/rerun from the pushed source before treating it as fully promoted.

### 2026-07-04 Dense Mmap/Page-Pressure Diagnostic Result

Result summary:

| Run | Added env | eval tok/s | prompt tok/s | TTFT ms | elapsed s | memory peak | memory file | pgmajfault | refault file | correctness | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T201857Z-20260704_dense_mmap_after_prompt_sota_diag/france-cpu40-vram0gb` | `LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1` | `4.1` | `1.6` | `29603.986984` | `63.05` | `16000000000` | `15107354624` | `271357` | `1757057` | pass | reject, below `4.2` |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T202101Z-20260704_dense_mmap_cache_after_prompt_sota_diag/france-cpu40-vram0gb` | `LLAMA_DROP_DENSE_MMAP_CACHE=1`, `LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1` | `4.1` | `1.6` | `29968.404228` | `63.14` | `16000000000` | `15107481600` | `277572` | `1646116` | pass | reject, below `4.2` |

Correctness output for both runs:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Counters:

- Both runs kept the accepted gate-cache shape:
  - `VRAM cache: 13.2 GiB, 3192 slots`
  - `VRAM cache: hits=30528 misses=4623 hit_rate=86.8%`
  - gate pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_reads=4623 direct_fallbacks=0`
- `stderr.txt` did not show useful dense mmap drop evidence for either run under the current grep set.
- Page-cache stayed charged inside the 16GB cgroup, and both runs hit exactly the configured `memory_peak_bytes=16000000000`.

Artifact hashes:

- `20260704_dense_mmap_after_prompt_sota_diag`:
  - `summary.json`: `90ff4ed576c39d7eeb4f53c8951f41f91ef7707639c7998facb1dbb63b5af970`
  - `stdout.txt`: `b97b5c2536dfa8ef9a956bc71f79282d58f826976a8406ab099650dd9413becb`
  - `stderr.txt`: `48b469ec8e5f95c0379dbf3088909459fa78261543478fddf1c82c07c393775f`
- `20260704_dense_mmap_cache_after_prompt_sota_diag`:
  - `summary.json`: `60e89b29cfc0d6428df41288f5333542477bbd53ae5e4811b714c7b0b7cb92c2`
  - `stdout.txt`: `7f1e7481fcd57cf77726a4ec59b6c76b1d0557284ca3d41429505fc5ec129d35`
  - `stderr.txt`: `c5a7434a553f2814fe6e8ba785546a987ed5a96b8f2d95b6f02608f00ef5c93a`

Decision:

- Dense mmap/page-pressure env-only diagnostics are rejected. They preserve correctness/RAM/TTFT, but do not beat the accepted `4.2 tok/s`.
- Stop sweeping dense-drop combinations.
- Current accepted SOTA remains the strict cold `4.2 tok/s` run from `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`.

### 2026-07-04 Next Plan: Fallback Split Trace Before Next Source Optimization

Reason for pivot:

- The no-drop diagnostic shows a large page/source-stall opportunity, but dense-drop does not recover it.
- Small down-cache admission is hard-bound below SOTA.
- The next source change must be based on a more precise split of the accepted run's remaining CPU fallback time.

Trace requirements:

1. Per fallback call, record at least:
   - tensor role: `up`, `gate`, `down`, or other;
   - layer and expert id when available;
   - route count / active rows;
   - total fallback wall time;
   - time spent preparing source pointers / touching mmap pages;
   - time spent copying or staging source data;
   - time spent CPU matmul/math;
   - pack/O_DIRECT read time if the path uses a pack;
   - whether the source was mmap, gate pack, VRAM cache hit, or CPU fallback.
2. Aggregate by:
   - prompt vs decode;
   - layer;
   - tensor role;
   - expert id;
   - top route-frequency buckets.
3. The trace must be low enough overhead for a full France strict run, or must support a short diagnostic whose overhead is separately measured.

Theory and bound before coding:

- If most accepted-run fallback time is source/page wait, the next candidate should be bounded async/O_DIRECT staging or route-ordered up+down pack/source layout.
- If most fallback time is CPU math, source/layout work cannot reach `10 tok/s`; next candidate must target CPU kernel/layout or true GPU offload for the missed experts.
- If pack/direct IO dominates only gate misses, it is already solved for the accepted run because gate pack misses are zero and gate hit rate is stable.
- The next implementation is only justified if the measured removable bucket can plausibly save more than the gap from the current `4.2 tok/s` to a new strict SOTA.

Immediate action:

1. Inspect current vendor CPU fallback/profile code to find existing timing hooks and avoid duplicating instrumentation.
2. Add a default-off fallback split trace guarded by env vars.
3. Run one strict cold France profile with the accepted SOTA knobs and the trace enabled.
4. Record the overhead, output correctness, RAM, TTFT, token rate, and split totals.
5. Use the split totals to design the next optimization. Do not promote a trace run unless it accidentally improves token rate while satisfying all constraints.

Implementation detail for the first split trace:

- Extend the existing `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT` CSV rather than creating a parallel profiler.
- Add a default-off source-touch measurement guarded by:
  - `GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE=1`
- When enabled, immediately before the CPU fallback loop, thread 0 touches one byte per page of every still-active expert source tensor, records per-expert `touch_us`, then synchronizes before normal CPU fallback math.
- The existing `fallback_us` will then measure fallback compute/scheduling after pages have been touched; `touch_us + fallback_us` gives the cold source-plus-compute estimate for that call group.
- Add source-kind counters to each fallback-profile row:
  - `pack_mmap_calls`
  - `gguf_calls`
- Also extend `GGML_MOE_CPU_CHUNK_TRACE_OUT` with a `role` column and include `up_gate` chunks if that fused CPU op is used in a future run. This remains default-off.

Trace run env:

- Use all accepted SOTA env/args unchanged.
- Add:
  - `GGML_KIMI_CPU_MOE_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback-split-profile.csv`
  - `GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE=1`
- Optional short diagnostic only if the full trace overhead is too high:
  - `GGML_MOE_CPU_CHUNK_TRACE_OUT={case_dir}/chunk-trace.csv`
  - `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=20000`

Acceptance for trace quality:

- The run must still pass RAM and correctness.
- It does not need to beat SOTA; it is a diagnostic run.
- The profile must contain nonzero `touch_us` for active up/down fallback rows and preserve nonzero `fallback_us`, otherwise the split is not informative.
- If trace overhead is too large to interpret token rate, still use the touch/fallback proportions for bottleneck design, but mark token rate as trace-overhead affected.

### 2026-07-04 Fallback Split Trace Implementation And Result

Source change:

- File: `ggml/src/ggml-cpu/ggml-cpu.c`
- Added default-off split instrumentation:
  - extends `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT` with `touch_us`, `pack_mmap_calls`, and `gguf_calls`;
  - adds `GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE=1` to touch one byte per page for each still-active CPU fallback expert before fallback math;
  - extends `GGML_MOE_CPU_CHUNK_TRACE_OUT` with a `role` column;
  - adds chunk trace coverage for the fused CPU `up_gate` path if that op is used later.
- Built diagnostic binary without overwriting accepted SOTA binary:
  - `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream-split-trace/bin/llama-cli`
  - sha256 `47ba151f4ed383595dfa77741acce758729b19ac9af6124f2f00f683017e9d72`
- Accepted SOTA binary remains unchanged:
  - `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin/llama-cli`
  - sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`

Touch split trace run:

- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260703T204105Z-20260704_fallback_split_touch_profile/france-cpu40-vram0gb`
- Added env:
  - `GGML_KIMI_CPU_MOE_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback-split-profile.csv`
  - `GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE=1`
- Metrics:
  - `eval_tok_s=2.5`
  - `prompt_tok_s=1.0`
  - `TTFT=35484.318865 ms`
  - `elapsed_seconds=89.84`
  - `memory_peak_bytes=16000000000`
  - `memory_file_bytes=15098998784`
  - `pgmajfault=21616`
  - `workingset_refault_file=1667513`
  - `ram_ok=true`
  - `correctness_ok=true`
- Verdict:
  - diagnostic only;
  - rejected as performance/SOTA because token rate is below `4.2` and TTFT is above the accepted `20%` limit (`33617.688744 ms`);
  - useful because split profile is complete and correctness/RAM passed.

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Split profile aggregate:

| Phase/role | count | calls | fallback_us after touch | touch_us | source |
| --- | ---: | ---: | ---: | ---: | --- |
| prompt/up | `1820` | `1171` | `135121` | `5559231` | `gguf` |
| prompt/down | `1820` | `1171` | `138248` | `5490099` | `gguf` |
| decode/up | `17940` | `17940` | `1384622` | `20833182` | `gguf` |
| decode/down | `17940` | `17940` | `1389115` | `20784944` | `gguf` |
| total | `39520` | `38222` | `3047106` | `52667456` | `gguf` |

Interpretation:

- The trace overstates end-to-end source time because page touching is serial and diagnostic-only, but it gives a useful lower bound on hot CPU fallback math.
- Hot fallback math after source touch is only about `3.0s` total for this France run.
- Therefore CPU dot/math is not the dominant bottleneck in the accepted run.
- The bottleneck is source/page movement for up/down fallback experts. This is consistent with:
  - cold current-SOTA profile fallback total around `25.7s`;
  - no-drop fallback total around `13.0s`;
  - touch trace lowering major faults to `21616` but increasing TTFT due serial page touching.
- All profile rows used `gguf` source (`pack_mmap_calls=0`), so accepted up/down CPU fallback is still reading from the model mapping, not a route-ordered up/down pack.

Default-off guard run:

- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260703T204417Z-20260704_split_trace_default_off_guard/france-cpu40-vram0gb`
- Same split-trace binary, no trace/touch env enabled.
- Metrics:
  - `eval_tok_s=4.2`
  - `prompt_tok_s=1.5`
  - `TTFT=30006.935752 ms`
  - `elapsed_seconds=62.41`
  - `memory_peak_bytes=16000000000`
  - `memory_file_bytes=15100276736`
  - `pgmajfault=271786`
  - `workingset_refault_file=1683134`
  - `ram_ok=true`
  - `correctness_ok=true`
- Gate counters:
  - `VRAM cache: 13.2 GiB, 3192 slots`
  - `VRAM cache: hits=30528 misses=4623 hit_rate=86.8%`
  - gate pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_reads=4623 direct_fallbacks=0`
- Verdict:
  - default-off instrumentation does not break SOTA-level behavior;
  - this is a tie, not a new promoted SOTA.

Artifact hashes:

- Touch trace:
  - `summary.json`: `d1c4fad0daa1a55bd689a5ffd65f5393f1aba525dd91d03eff87b32a99beef19`
  - `stdout.txt`: `29bafedd5708872e9f8c1b16bb7c76af4497ec981b0551b7aede55d33e041053`
  - `stderr.txt`: `a950280cf1e3f7c5b92a60dc574353024f9ffacb6b92478c0cca65f2a6108173`
  - `fallback-split-profile.csv`: `03e23df371eafefbc918e7c5b9759845656f713a6191353f2e482579a0bf42bd`
- Default-off guard:
  - `summary.json`: `8450d435ae119190d45f58d332987e474788c96ef6b4faed2931222726e11cb1`
  - `stdout.txt`: `eb6f62f8d6cc5f4fb221820a3b5b1d133a3491f70f973cc2e783114112b1d474`
  - `stderr.txt`: `38a0b13be351d0ff6da253c28cf4ed3222f029d563229194b9e794cb4fc724b3`

Next optimization plan:

1. Stop optimizing CPU fallback math until source movement is materially reduced.
2. Inspect existing expert-pack implementations:
   - one-stream gate pack used by accepted SOTA;
   - batch/down pack code;
   - any CPU fallback pack mmap/direct path.
3. Design a route-ordered up/down source movement path that attacks both up and down, not only down:
   - candidate A: CPU fallback reads up/down from a route-ordered pack with `O_DIRECT` or mmap source, using profile order;
   - candidate B: bounded async source staging that overlaps next expert read with current hot CPU math;
   - candidate C: small resident hotset only if hard-bound coverage exceeds the current `4.2 tok/s` SOTA.
4. Hard bound before coding:
   - current hot CPU fallback math lower bound is about `3.0s`;
   - accepted cold fallback total is about `25.7s`;
   - no-drop fallback total is about `13.0s`;
   - any next candidate must plausibly remove at least several seconds of up/down source movement without shrinking the gate cache enough to lose the `86.8%` gate hit rate.
5. Before implementation, update this plan with:
   - exact source path;
   - bytes moved per expert and per run;
   - expected saved milliseconds;
   - RAM/VRAM cost;
   - correctness and TTFT rejection criteria.

### 2026-07-04 Immediate Experiment Design: CPU Fallback Pack Mmap For Up/Down Top256

Observed implementation state:

- Accepted stream-only build has `GGML_CUDA_MOE_STREAM_BATCH=OFF`; in that build `ggml_cuda_moe_expert_pack_mmap_ptr()` is a stub, so CPU fallback cannot use pack mmap.
- Batch-enabled build `build-ds4-moe-stream-batch-probe/bin/llama-cli` has:
  - sha256 `866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`;
  - `GGML_CUDA_MOE_STREAM=ON`;
  - `GGML_CUDA_MOE_STREAM_BATCH=ON`;
  - real `GGML_MOE_EXPERT_PACK` plus `GGML_MOE_CPU_FALLBACK_PACK_MMAP` implementation.
- Do not enable `GGML_MOE_STREAM_DOWN_BATCH` for this experiment. The only intended behavior change is CPU fallback source selection from model GGUF mmap to pack mmap for matching up/down entries.

Pack candidates from `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top512.tsv`:

| Pack | Coverage ms | Calls | Payload | File |
| --- | ---: | ---: | ---: | --- |
| top128 up/down | `3897.807` | `10027` | `544.0 MiB` | `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top128-updown-20260703.pack` |
| top256 up/down | `5646.756` | `13727` | `1088.0 MiB` | `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top256-updown-20260703.pack` |
| top512 up/down | `7947.407` | `17248` | `2176.0 MiB` | not selected for first run |

Pack hashes:

- top128: `f57ff2426647514c0145bb4b367b750f7a1753837b2ead22df091e9645483894`
- top256: `a6eb88346b2a4719f85db804b4c6bf71ad58ec6ca3719c4292d8c9a11767161d`

Theory:

- The split trace shows hot CPU fallback math is much smaller than cold fallback source movement.
- Pack mmap should reduce random source movement if the top up/down experts are clustered in a smaller route-ordered file instead of being pulled from the large GGUF mapping.
- The pack pages are still file-backed and counted inside the 16GB cgroup, so this is cold-legal if run under strict `drop_caches`.
- This does not consume VRAM and should preserve the accepted gate VRAM cache (`13.2GiB`, `3192` slots, `86.8%` hit rate).

Hard bound:

- Accepted generation time estimate: `192 / 4.2 = 45.7s`.
- Ideal top256 coverage saving: `5.65s`.
- Best-case generation time: about `40.1s`, or about `4.8 tok/s`.
- If mmap locality is weaker than expected or pack pages displace useful model pages, the run may tie or regress.

Run config:

- Binary: `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream-batch-probe/bin/llama-cli`
- Accepted SOTA env unchanged:
  - gate one-stream cache `13568MiB`;
  - current gate admit profile;
  - current gate expert pack with direct IO;
  - top-k up/down pruning;
  - `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Added env:
  - `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top256-updown-20260703.pack`
  - `GGML_MOE_CPU_FALLBACK_PACK_MMAP=1`
  - `GGML_KIMI_CPU_MOE_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback-packmmap-profile.csv`

Acceptance/rejection:

- Accept only if RAM/correctness pass, TTFT stays within `33617.688744 ms`, and `eval_tok_s > 4.2`.
- If it ties or regresses, record mmap hit counters and profile data, then reject.
- If it improves, immediately rebuild the batch-enabled binary from current pushed source, rerun from that source, record all hashes and push.

### 2026-07-04 CPU Fallback Pack Mmap Top256 Result

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T205623Z-20260704_cpu_fallback_packmmap_top256_updown/france-cpu40-vram0gb`
- Binary: `build-ds4-moe-stream-batch-probe/bin/llama-cli`
- Pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top256-updown-20260703.pack`
- Env delta:
  - `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top256-updown-20260703.pack`
  - `GGML_MOE_CPU_FALLBACK_PACK_MMAP=1`
  - profile envs enabled

Metrics:

- `eval_tok_s=4.0`
- `prompt_tok_s=1.6`
- `TTFT=30381.522449 ms`
- `elapsed_seconds=64.52`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15101743104`
- `pgmajfault=273403`
- `workingset_refault_file=1738482`
- `ram_ok=true`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Pack and gate counters:

- Batch expert pack loaded `256` entries.
- Pack mmap enabled: `1088.04 MiB`.
- `expert pack: hits=13423 misses=22457`.
- `kimi_cpu_fallback_pack_mmap: enabled=1 hits=13423 misses=22457 bytes=59818901504 fallback_gguf=22457`.
- Gate one-stream remained healthy:
  - gate pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_reads=4623 direct_fallbacks=0`
  - `VRAM cache: hits=30528 misses=4623 hit_rate=86.8%`

Fallback profile aggregate:

| Phase/role | count | calls | fallback_us |
| --- | ---: | ---: | ---: |
| prompt/up | `1820` | `1171` | `3304729` |
| prompt/down | `1820` | `1171` | `4261299` |
| decode/up | `17940` | `17940` | `9280655` |
| decode/down | `17940` | `17940` | `9466350` |
| total | `39520` | `38222` | `26313033` |

Artifact hashes:

- `summary.json`: `d0979da52cdff27e96f50596eb249bdc98f98ca084bd601a7d7eff2909ffc6d3`
- `stdout.txt`: `20f88a45b7867e45a9fa7a83a32582bbe32ae784822a69ba01db0a4a3eea1dcd`
- `stderr.txt`: `d2a67a7868373768c28e395204bf421bf4fe76fbd159da675bf629c1886e9ed3`
- `fallback-packmmap-profile.csv`: `f4bceab1a04e5363033e4501f9d83ac616ba5cbec16164233dcffc524cc696e7`

Verdict:

- Rejected. Correctness/RAM/TTFT pass, but token rate regressed below `4.2`.
- The pack mmap path works and hits, but it does not reduce page faults or fallback time:
  - major faults remain about the same as accepted cold;
  - `workingset_refault_file` is worse than the default-off guard;
  - fallback total is about `26.3s`, not better than the previous cold profile.
- Likely cause: pack mmap changes source location but still faults file-backed pages under the same 16GB pressure; the `1.1GiB` pack also competes with useful GGUF/model pages.

Immediate follow-up: one smaller-footprint top128 diagnostic.

- Rationale: top256 proves pack mmap functionality but may be too large. Top128 uses `544MiB`, covers `3897.807 ms`, and has a best-case bound around `4.6 tok/s`.
- Run only one strict cold top128 pack mmap diagnostic.
- Accept only if it beats `4.2` with RAM/correctness/TTFT pass.
- If top128 also ties/regresses, stop pack-mmap mmap-only experiments and move to an overlapped/direct staging design or a different movement model.

### 2026-07-04 CPU Fallback Pack Mmap Top128 Result

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T210008Z-20260704_cpu_fallback_packmmap_top128_updown/france-cpu40-vram0gb`
- Binary: `build-ds4-moe-stream-batch-probe/bin/llama-cli`
- Pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top128-updown-20260703.pack`

Metrics:

- `eval_tok_s=4.1`
- `prompt_tok_s=1.6`
- `TTFT=29132.34718 ms`
- `elapsed_seconds=62.33`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15100948480`
- `pgmajfault=273460`
- `workingset_refault_file=1648407`
- `ram_ok=true`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Pack and gate counters:

- Batch expert pack loaded `128` entries.
- Pack mmap enabled: `544.02 MiB`.
- `expert pack: hits=10051 misses=25829`.
- `kimi_cpu_fallback_pack_mmap: enabled=1 hits=10051 misses=25829 bytes=44791758848 fallback_gguf=25829`.
- Gate one-stream remained healthy:
  - gate pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_reads=4623 direct_fallbacks=0`
  - `VRAM cache: hits=30528 misses=4623 hit_rate=86.8%`

Fallback profile aggregate:

| Phase/role | count | calls | fallback_us |
| --- | ---: | ---: | ---: |
| prompt/up | `1820` | `1171` | `3109396` |
| prompt/down | `1820` | `1171` | `4277033` |
| decode/up | `17940` | `17940` | `9512016` |
| decode/down | `17940` | `17940` | `9244851` |
| total | `39520` | `38222` | `26143296` |

Artifact hashes:

- `summary.json`: `768e29c9c3c00608d447b6ed1412487f66cd280b30a4cd835cbbab6b8d500e76`
- `stdout.txt`: `fd4afe007672a0210d160ad2e9d8643fef9870d8128148d495eba1a10dc29a5d`
- `stderr.txt`: `225f352d03de4a4002feb3191486bf554e18e5b153863e23e78edc7bc249c383`
- `fallback-packmmap-profile.csv`: `adb3854dae80c5854f31cba1a73c1f523232a12c225dc2f0160a099e37fc938c`

Verdict:

- Rejected. Correctness/RAM/TTFT pass, but token rate is below the accepted `4.2`.
- Top128 has smaller footprint than top256 but still does not reduce major faults or fallback total.
- Stop mmap-only CPU fallback pack experiments. The source file changed, but the kernel still services file-backed random pages inside the same 16GB pressure envelope.

Next direction:

- The next source movement design must avoid relying on page-cache locality alone.
- Candidate to design next:
  - O_DIRECT pack read into a bounded anonymous/aligned staging buffer for CPU fallback, so source bytes do not enter the file page cache;
  - overlap next expert direct read with current hot CPU math where possible;
  - keep staging memory bounded and charged as anonymous memory inside the same 16GB cgroup;
  - preserve gate VRAM cache and current gate O_DIRECT pack.
- Before coding, compute:
  - direct read bytes per token from top128/top256/top512 coverage;
  - maximum useful overlap from hot CPU math lower bound (`~3.0s` total after source touch);
  - anonymous staging memory required per thread/expert;
  - whether the expected saved time can exceed `4.2 tok/s` without TTFT violation.

## 2026-07-04 Next Plan: Cold CPU Fallback O_DIRECT Staging

Current accepted SOTA remains:

- `eval_tok_s=4.2`
- `prompt_tok_s=1.6`
- `TTFT=28014.740620 ms`
- strict cold `drop_caches`
- 16GB cgroup including page cache
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15096049664`
- France correctness pass
- pushed source/docs branch: `ssd/vendor/deepseek-token-rate-16gb`

This plan must continue from the accepted SOTA path. The accepted binary/source path must remain reproducible while any new diagnostic code is default-off.

### Bottleneck From Latest Diagnostics

The latest split trace and pack mmap probes show that the current bottleneck is not gate cache and not hot CPU math alone:

- Gate one-stream VRAM cache is healthy:
  - `VRAM cache hit_rate=86.8%`
  - gate pack misses are zero in the accepted SOTA run.
- CPU fallback still handles up/down misses.
- Touch split profile showed hot CPU fallback math lower bound is about `3.0s` total, while source/page movement under cold 16GB pressure dominates.
- Top128/top256 pack mmap worked functionally, but still faulted file-backed pages under the same cgroup page-cache pressure and regressed to `4.1`/`4.0 tok/s`.

Therefore the next optimization should attack CPU fallback source movement, not another blind cache-size or top-k sweep.

### Hard Bound Before Implementation

Accepted generation time estimate:

- `192 / 4.2 = 45.7s`

Measured decode fallback coverage from the current SOTA up/down profile:

| Candidate | Covered fallback time | Full-run direct read bytes observed or implied | Ideal token-rate bound |
| --- | ---: | ---: | ---: |
| top128 up/down | `3897.807 ms` | `44.79 GB` observed from mmap hits | about `4.6 tok/s` |
| top256 up/down | `5646.756 ms` | `59.82 GB` observed from mmap hits | about `4.8 tok/s` |
| top512 up/down | `7947.407 ms` | not yet safe to stage; footprint `2176 MiB` | about `5.1 tok/s` ideal only |

The first diagnostic should use top128, not top256/top512:

- top128 has the smallest footprint (`544 MiB`) and lowest direct-read pressure.
- top256/top512 have better ideal bounds but likely read too many bytes without asynchronous overlap.
- If top128 cannot beat `4.2`, larger synchronous staging is unlikely to be useful.

Theoretical limitation:

- A synchronous O_DIRECT top128 implementation may need to read about `44.79 GB` during the run.
- Without overlap, this requires unrealistically high sustained effective IO bandwidth to save the full `3.9s`.
- The expected value of the first implementation is diagnostic: prove whether avoiding page-cache pollution reduces refaults enough to offset direct-read cost.

### Implementation Plan

Implement a default-off CPU fallback direct staging path:

- Add an exported CUDA-side pack API in `ggml/src/ggml-cuda/moe_stream_batch.cu`:
  - `ggml_cuda_moe_expert_pack_read(...)`
  - use the existing batch expert pack lookup;
  - reuse `GGML_MOE_IO_BACKEND=direct`;
  - keep the existing no-batch stub returning false.
- Add a CPU fallback staging path in `ggml/src/ggml-cpu/ggml-cpu.c` behind:
  - `GGML_MOE_CPU_FALLBACK_PACK_DIRECT=1`
- Scope the first run to decode fallback only:
  - do not increase prompt/TTFT risk by staging prompt fallback first;
  - keep prompt path on the current accepted behavior.
- Use one bounded 4096-byte-aligned anonymous staging buffer:
  - size is one expert tensor payload, about `4.25 MiB`;
  - allocate it from existing op workspace, so it is charged inside the 16GB cgroup;
  - add required workspace padding explicitly.
- Add per-expert barriers around staging and compute:
  - thread 0 reads one packed expert payload into the shared staging buffer;
  - all CPU fallback threads consume that staged payload;
  - all threads synchronize before the buffer is reused.
- Add minimal counters to stderr:
  - direct staging enabled;
  - hits;
  - misses/read failures;
  - bytes staged;
  - fallback-to-GGUF count.

This is intentionally a simple synchronous diagnostic. Do not add async/prefetch until the synchronous result proves the page-cache model is worth pursuing.

### Run Config

Use the batch-enabled diagnostic binary so the expert-pack reader is available:

- build target: `build-ds4-moe-stream-batch-probe/bin/llama-cli`
- strict runner: `.Agent/run-tools/strict_ds4_runner.py`
- base SOTA env unchanged:
  - `GGML_CUDA_DISABLE_GRAPHS=1`
  - `GGML_MOE_STREAM=1`
  - `GGML_MOE_STREAM_DONTNEED=1`
  - `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`
  - `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
  - current gate admit profile
  - current gate O_DIRECT expert pack
  - current top-k up/down pruning
- added env:
  - `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top128-updown-20260703.pack`
  - `GGML_MOE_IO_BACKEND=direct`
  - `GGML_MOE_CPU_FALLBACK_PACK_DIRECT=1`
  - fallback/name profile outputs enabled
- do not enable:
  - `GGML_MOE_CPU_FALLBACK_PACK_MMAP`
  - `GGML_MOE_STREAM_DOWN_BATCH`

### Acceptance And Rejection

Accept only if all are true:

- strict cold run uses `drop_caches`;
- host RAM cgroup including page cache stays within 16GB;
- `MemorySwapMax=0`;
- France output is semantic, correct, and coherent;
- `TTFT <= 33617.688744 ms` (`28014.740620 * 1.20`);
- `eval_tok_s > 4.2`;
- source/docs are committed and pushed to `ssd/vendor/deepseek-token-rate-16gb`;
- the promoted result is rerun from the pushed source and records binary/source hashes.

Reject if any are true:

- token rate ties or falls below `4.2`;
- correctness fails;
- RAM exceeds 16GB including page cache;
- TTFT exceeds the accepted limit, unless explicitly recorded as a rejected high-TTFT diagnostic;
- direct staging counters show most hits still fall back to GGUF;
- major faults/refaults remain unchanged while token rate regresses.

If rejected:

- record full metrics, stdout correctness text, stderr counters, hashes, and verdict in this document;
- revert the direct-staging source changes unless they are clearly useful default-off instrumentation for the next diagnostic;
- keep the accepted 4.2 SOTA as the current rollback point.

### Required Record And Push Discipline

For every practice run before changing code:

- update this plan with hypothesis, hard bound, exact run config, expected acceptance/rejection rule, and start time.

For any new compliant SOTA:

- immediately record exact reproduction info:
  - commit hash;
  - branch;
  - binary path and sha256;
  - model path and size/hash when available;
  - all env vars;
  - all CLI args;
  - run directory;
  - stdout correctness answer;
  - token-rate metrics;
  - TTFT;
  - elapsed time;
  - cgroup memory peak;
  - file/page-cache bytes;
  - relevant profiling counters and artifact hashes.
- immediately commit and push source plus docs to:
  - remote: `ssd` (`https://github.com/wici-ai/ssd-llama.git`)
  - branch: `vendor/deepseek-token-rate-16gb`
  - git identity: `L-Ark <fliangae@connect.ust.hk>`
- rebuild and rerun from the pushed source before calling the result promoted/reproducible.

Do not promote any unpushed local run as SOTA.

## 2026-07-03T21:16Z Execution: Top128 O_DIRECT Staging Diagnostic

Start time:

- `2026-07-03T21:16:33Z` on the remote host.

Hypothesis:

- The current `4.2 tok/s` SOTA is limited by CPU fallback up/down source movement under cold 16GB page-cache pressure.
- Mmap-only up/down packs did not help because they still populate/fault file-backed pages in the same cgroup.
- Reading top128 up/down expert payloads with O_DIRECT into one bounded anonymous staging buffer may reduce file refault pressure enough to beat `4.2 tok/s`, even if the first implementation is synchronous.

Hard bound and risk:

- Accepted generation time estimate: `192 / 4.2 = 45.7s`.
- Top128 up/down fallback coverage: `3897.807 ms`.
- Ideal upper bound if all covered fallback time vanished: about `4.6 tok/s`.
- Observed top128 hit payload volume from mmap probe: about `44.79 GB`.
- Because the first implementation is synchronous, it may regress if O_DIRECT read time exceeds avoided page-cache/refault time. This run is diagnostic and must not replace the accepted SOTA unless it passes all gates.

Code change scope:

- Implement default-off source instrumentation only.
- Add CUDA-side batch pack read export:
  - `ggml_cuda_moe_expert_pack_read(...)`.
- Add CPU fallback direct staging behind:
  - `GGML_MOE_CPU_FALLBACK_PACK_DIRECT=1`.
- Decode fallback only for the first run.
- Use one 4096-byte-aligned expert-sized staging buffer from op workspace.
- Add per-expert barriers to keep the shared buffer safe.
- Add stderr counters for direct hits/misses/read failures/bytes/fallback-to-GGUF.

Run config:

- Binary to build/run: `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream-batch-probe/bin/llama-cli`
- Strict runner: `/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/run-tools/strict_ds4_runner.py`
- Model: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`
- Prompt: `Please introduce France in a short paragraph.`
- Args: `-c 256 -b 16 -ub 16 -t 20 -tb 20 -ngl all --fit on --n-cpu-moe 40 --defer-experts`
- Base env:
  - `CUDA_VISIBLE_DEVICES=0`
  - `GGML_CUDA_DISABLE_GRAPHS=1`
  - `GGML_MOE_STREAM=1`
  - `GGML_MOE_STREAM_DONTNEED=1`
  - `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`
  - `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
  - `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`
  - `GGML_MOE_KEEP_TOPK_UPDOWN=4`
  - `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`
- Added env:
  - `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top128-updown-20260703.pack`
  - `GGML_MOE_IO_BACKEND=direct`
  - `GGML_MOE_CPU_FALLBACK_PACK_DIRECT=1`
  - `GGML_KIMI_CPU_MOE_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback-packdirect-profile.csv`
- Explicitly not enabled:
  - `GGML_MOE_CPU_FALLBACK_PACK_MMAP`
  - `GGML_MOE_STREAM_DOWN_BATCH`

Artifact hashes before run:

- Gate admit profile: `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`
- Up/down decode profile: `45c35b2cb00faa4b37ad793e5e63e6008b68235064f5fdeb23a7e843c437084e`
- Top128 up/down pack: `f57ff2426647514c0145bb4b367b750f7a1753837b2ead22df091e9645483894`
- Gate pack historical hash: `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`

Acceptance:

- strict cold `drop_caches` run;
- `memory_peak_bytes <= 16000000000` including file/page cache;
- `MemorySwapMax=0`;
- France answer is semantically correct and coherent;
- `TTFT <= 33617.688744 ms`;
- `eval_tok_s > 4.2`;
- direct staging counters show meaningful pack hits;
- immediately commit/push source and docs to `ssd/vendor/deepseek-token-rate-16gb`;
- rebuild/rerun from pushed source before promoting as SOTA.

Rejection:

- Any correctness/RAM/TTFT violation rejects the run.
- `eval_tok_s <= 4.2` rejects the run.
- If rejected, record full metrics and counters, then revert direct-staging source changes unless they are needed for the next explicitly planned diagnostic.

### 2026-07-03T21:29Z Top128 O_DIRECT Staging Result

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T212905Z-20260704_cpu_fallback_packdirect_top128_updown/france-cpu40-vram0gb`
- Binary: `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream-batch-probe/bin/llama-cli`
- Source state at run time: `88de4bb09-dirty` with local default-off direct-staging patch.
- This dirty source is not accepted and must not be promoted.

Binary/shared-library hashes at run time:

- `llama-cli`: `866890c34606a1a91a28d7ef53904b506680036391f7f8edb7dbcff13568c8bc`
- `libggml-cpu.so`: `707e560e16f34d46d2ccbf6ad9c4897757f51a4e770084763a6ea6e7b3e08ef1`
- `libggml-cuda.so`: `877205d77f402b42200d3669ac294e7c52da4dcc36e2a1155387601a51f3cd70`
- `libggml.so`: `95dc04232d3c2d502b882c1feaffd2e805a1aa724673b63a8fee5ffc09ac3e57`

Metrics:

- `eval_tok_s=2.9`
- `prompt_tok_s=1.5`
- `TTFT=31516.236517 ms`
- `elapsed_seconds=77.74`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15075065856`
- `memory_max_events=16542`
- `pgmajfault=264469`
- `workingset_refault_file=1526493`
- `ram_ok=true`
- `ram_limit_killed=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Direct staging and gate counters:

- Batch up/down pack:
  - loaded `128` entries from top128 up/down pack;
  - `hits=10051`;
  - `misses=25829`;
  - `read_failures=0`;
  - `direct_reads=10051`;
  - `direct_fallbacks=0`.
- CPU direct staging:
  - `enabled=1`;
  - `available=1`;
  - `hits=10051`;
  - `misses=25829`;
  - `read_failures=25829` in the CPU-side counter, because the wrapper currently cannot distinguish pack miss from read failure;
  - `bytes=44791758848`;
  - `fallback_mmap=0`;
  - `fallback_gguf=25829`.
- Gate one-stream remained healthy:
  - gate pack `hits=4623 misses=0 reads=4623 bytes=20602159104 direct_reads=4623 direct_fallbacks=0`;
  - `VRAM cache: hits=30528 misses=4623 hit_rate=86.8%`.

Fallback profile aggregate:

| Phase/role | count | calls | fallback_us |
| --- | ---: | ---: | ---: |
| prompt/up | `1820` | `1171` | `3159051` |
| prompt/down | `1820` | `1171` | `4408500` |
| decode/up | `17940` | `17940` | `18731302` |
| decode/down | `17940` | `17940` | `18372494` |
| total | `39520` | `38222` | `44671347` |

Note: this CSV profile only records `pack_mmap_calls` versus `gguf_calls`; it is not direct-staging aware. Use the stderr direct counters above for direct hit/miss accounting.

Artifact hashes:

- `summary.json`: `adde60e49ba67e264294e7c9ab606196b1f087ed7ced3c2326173b878b0f702e`
- `stdout.txt`: `988f05743f37872f6fefdaccc3c9a2d7d0b8f0dc9baeedf14b7df9413d775c24`
- `stderr.txt`: `40c68cb8436b56b1797aebaaf80837b5a1ed7352705db138d7c91b927e241ef0`
- `fallback-packdirect-profile.csv`: `2886abae122ccc576265584eef8bac48af920322f75c2d1a37a4ffd5060362af`
- `environment.txt`: `2c43c7b96800d5da4ce0354a484068fb4dc2906b62675044ba0b48b4a64ed4bd`
- `exact_command.txt`: `e3d225597c6e8058978ce1ca2e28187b437fd8317d1e1b99f4fb7539ca3447de`

Verdict:

- Rejected.
- Correctness, RAM, and TTFT pass, but `eval_tok_s=2.9` is far below the accepted `4.2`.
- The direct path functionally hit the expected top128 entries and avoided pack read failures, but synchronous O_DIRECT staged about `44.79 GB` during decode. That IO cost dominated the run.
- Fallback profile total rose to about `44.7s`, much worse than the previous top128 mmap diagnostic (`~26.1s`), even though major faults/refaults were slightly lower.
- Do not continue synchronous direct staging.
- Revert the direct-staging source patch; keep only this rejected-result record.

Next direction after this rejection:

- Current accepted SOTA remains the pushed `4.2 tok/s` run.
- CPU fallback source movement is still the bottleneck, but synchronous O_DIRECT staging is the wrong movement model.
- Any future pack-based CPU fallback attempt must use real overlap or reduce bytes moved:
  - async prefetch of next expert while current expert computes;
  - coalesced multi-expert reads only if access order is predictable;
  - or a smaller correctness-preserving up/down cache/admission set with much lower byte volume.
- Before coding another movement path, first measure whether there is enough per-expert CPU compute time to hide the next expert read. If not, move to a different bottleneck such as fewer CPU fallback calls or GPU-side up/down execution.

## 2026-07-03T22:10Z Next Plan: Gate One-Stream VRAM Cache Prefill

Current accepted SOTA remains:

- `eval_tok_s=4.2`
- strict cold `drop_caches`
- 16GB cgroup including page cache
- `MemorySwapMax=0`
- France correctness pass
- accepted gate one-stream cache `13568MiB`, `3192` slots, hit rate `86.8%`
- accepted gate pack O_DIRECT path has `direct_failures=0`, `direct_fallbacks=0`

### Bottleneck

Phase B current-SOTA one-stream trace:

- gate one-stream rows: `35151`
- cache misses: `4623`
- cache inserts: `3337`
- one-stream `src0_ms_all=5236.633 ms`
- one-stream miss `src0_ms=5214.433 ms`
- inserted-cache `src0_ms=3792.307 ms`

The accepted path spends about `5.2s` of generation time loading gate experts from the O_DIRECT pack/source into the VRAM cache. This is not a correctness-sensitive math path; it is movement time for the already accepted gate implementation.

### Why This Is Different From Rejected Up/Down Hotsets

- Rejected up/down hotsets tried to add new compute/offload behavior and disturbed CPU fallback or gate cache.
- This candidate only preloads the same gate experts that the accepted path would later load into the same one-stream VRAM cache.
- It does not increase the VRAM cache budget.
- It does not use host buffered page cache; prefill uses the existing gate expert pack with O_DIRECT.
- It shifts some existing generation source-load time into pre-decode/TTFT time. The accepted TTFT gate has headroom:
  - accepted TTFT: `28014.740620 ms`
  - 20% limit: `33617.688744 ms`
  - headroom: about `5603 ms`

### Hard Bound

Using `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv` sorted by frequency against the Phase B one-stream trace:

| Prefill top-N | payload | covered miss rows | covered `src0_ms` | generation-rate upper bound |
| ---: | ---: | ---: | ---: | ---: |
| `1024` | `4.25 GiB` | `1024` | `1155.704 ms` | about `4.3 tok/s` |
| `2048` | `8.50 GiB` | `2061` | `2376.439 ms` | about `4.4 tok/s` |
| `3000` | `12.45 GiB` | `3024` | `3448.563 ms` | about `4.55 tok/s` |
| `3192` | `13.25 GiB` | `3216` | `3659.925 ms` | about `4.57 tok/s` |

Use `top3000` for the first diagnostic:

- It covers most removable gate source-load time without filling every slot.
- It leaves about `192` slots for dynamic cache inserts.
- Expected O_DIRECT prefill read volume is about `12.45GiB`.
- Expected TTFT increase should be below the `5.6s` gate if O_DIRECT throughput is close to the current gate pack read path.

This cannot reach `10 tok/s` by itself. It is a narrow SOTA-edge candidate that stacks with the accepted gate path and may establish a higher cold-start baseline before a larger fallback/speculative design.

### Implementation Plan

Add default-off support in `ggml/src/ggml-cuda/moe_stream.cu`:

- `GGML_MOE_STREAM_ONE_PREFILL_PROFILE=<tsv>`
- `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`
- When enabled, one-stream cache keys use a stable `(tensor, expert)` hash instead of host pointer keys. This allows prefilled pack entries to be found later by normal `ggml_cuda_moe_stream_one()` calls.
- On first one-stream call after VRAM cache initialization:
  - load the TSV profile;
  - sort by third column frequency descending;
  - read at most `limit` profile entries from `GGML_MOE_STREAM_ONE_EXPERT_PACK` using the existing O_DIRECT pack reader;
  - insert them into the existing VRAM cache without incrementing normal miss counters;
  - synchronize the prefill stream before returning to normal execution;
  - print prefill counters: loaded entries, attempted, inserted, pack misses, read failures, bytes, elapsed ms.
- Default behavior must remain identical when `GGML_MOE_STREAM_ONE_PREFILL_PROFILE` is unset.

Implementation risk:

- Named keys under the prefill env are a behavior change for cache lookup. This is allowed only for the diagnostic env and must be default-off.
- Prefill can evict low-priority entries before use if the profile order is wrong. Sorting by frequency is the first conservative policy.
- Prefill may increase TTFT too much. If TTFT exceeds `33617.688744 ms`, reject even if generation token rate improves.

### Run Config

Build:

- `build-ds4-moe-stream/bin/llama-cli`

Strict cold run:

- strict runner `.Agent/run-tools/strict_ds4_runner.py`
- `drop_caches`
- `MemoryMax=16000000000`
- `MemorySwapMax=0`
- prompt: `Please introduce France in a short paragraph.`
- accepted SOTA env unchanged:
  - `GGML_CUDA_DISABLE_GRAPHS=1`
  - `GGML_MOE_STREAM=1`
  - `GGML_MOE_STREAM_DONTNEED=1`
  - `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`
  - `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
  - `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`
  - `GGML_MOE_KEEP_TOPK_UPDOWN=4`
  - `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`
- added env:
  - `GGML_MOE_STREAM_ONE_PREFILL_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`
  - `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`
- CLI override:
  - `-c 256 -b 16 -ub 16 -t 20 -tb 20`

### Acceptance

Accept only if all are true:

- `eval_tok_s > 4.2`
- France output is complete, coherent, and semantically correct.
- `memory_peak_bytes <= 16000000000` including page cache.
- `ram_limit_killed=false`, `oom_seen=false`.
- `TTFT <= 33617.688744 ms`.
- gate pack direct path has no direct failures/fallbacks.
- normal generation cache hit rate does not collapse after interpreting prefill counters.
- source, docs, and artifacts are immediately committed and pushed to `ssd/vendor/deepseek-token-rate-16gb`.
- pushed-source rebuild/rerun also exceeds `4.2 tok/s` before promotion.

Reject if any are true:

- `eval_tok_s <= 4.2`
- correctness fails or the answer ends incoherently.
- TTFT exceeds the 20% gate for an accepted result.
- RAM exceeds 16GB or the cgroup kills the run.
- prefill counters show large pack read failures.
- cache behavior collapses or pack direct fallbacks appear.

If rejected:

- record metrics, counters, answer, and artifact hashes;
- revert the runtime source patch;
- push only documentation/artifact records.

### 2026-07-03T21:56Z Gate Prefill Top3000 Initial Candidate Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/gate-prefill-top3000-candidate-result.json`
- artifact sha256: `357a7c97c54d29981cedfeb26259003063f345742f322671de8a68971539ae72`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T215630Z-20260704_gate_prefill_top3000_candidate/france-cpu40-vram0gb`
- Source state at run time: `52f88093d-dirty`
- Binary build metadata: `b14648-52f88093d`
- This is an initial SOTA candidate only. It is not promoted until pushed-source rerun passes.

Dirty candidate hashes:

- `llama-cli`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
- `libggml-cuda.so`: `f73af32dcfc67f23dadef8c20217f784350b7df6a4e7a67c647dad3d2cf89c36`
- `libggml-cpu.so`: `b7d0a3a7a65adac0e8ff2a83ce7bcbbeef8a5ce38ddd173cf8bd6b75abe9ed39`
- `libggml.so`: `e49b194a510b150e2eee78271fc6bfae51f7ea8113909528c829a003756b217f`
- gate profile: `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`
- gate pack historical hash: `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`

Metrics:

- `eval_tok_s=4.3`
- `prompt_tok_s=1.8`
- `TTFT=32587.062204 ms`
- TTFT gate: pass (`32587.062204 < 33617.688744`)
- `elapsed_seconds=64.07`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15102803968`
- `memory_max_events=16945`
- `pgmajfault=264753`
- `workingset_refault_file=1696731`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Manual correctness verdict:

- Pass. The France answer is complete, coherent, and semantically correct.

Prefill counters:

- profile loaded entries: `3313`
- limit: `3000`
- attempted: `3000`
- inserted: `3000`
- pack misses: `0`
- read failures: `0`
- bytes: `13369344000`
- elapsed: `4708.250 ms`

Gate pack/cache counters:

- one expert pack: `hits=4886 misses=0 reads=4886 bytes=21774204928 failures=0 entries=4599 direct_reads=4886 direct_failures=0 direct_fallbacks=0`
- VRAM cache after excluding prefill insertions from miss accounting: `hits=33265 misses=1886 hit_rate=94.6%`

One-trace aggregate:

| Slice | rows | src0_ms | total_ms |
| --- | ---: | ---: | ---: |
| all gate | `35151` | `6927.392` | `8269.649` |
| gate hits | `33265` | `4742.504` | `5737.827` |
| gate misses | `1886` | `2184.888` | `2531.822` |
| gate inserts | `600` | `710.526` | `821.257` |

Trace interpretation:

- Runtime cache hit rate improved from accepted `86.8%` to `94.6%`.
- Pack direct path remained clean with no misses, direct failures, or direct fallbacks.
- Prefill moved about `4.7s` of O_DIRECT/H2D source movement into TTFT while keeping TTFT under the 20% gate.
- The one-trace `src0_ms` includes prefilled-hit lookup/other overhead and is not directly comparable to the no-prefill trace, but normal generation token rate improved to `4.3 tok/s`.

Artifact hashes:

- `summary.json`: `84aa2ab9bcd9310fcf5e7b56634fc28cc15722f997ca47f77e57063bbc99ec30`
- `stdout.txt`: `e835f6417380c169ff34ac13d80b4d437483fef1eb211990e71490aafafefe4a`
- `stderr.txt`: `14543ceea118c4d3755bce31bede165948657489998c954ea4101cb52a814f22`
- `environment.txt`: `b6f20cebd94eb6d3ef9f35d8018b713a1a262cafda85af5ffa91bec21500d271`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `ad9b730842efba367af253affcf23653c94418fbaab86643bba90677b6eb98f7`

Initial verdict:

- This is a compliant initial SOTA candidate over the previous accepted `4.2 tok/s`.
- It passes RAM, correctness, TTFT, pack, and cache gates in this first dirty-source run.
- Immediate action: commit source, plan, and artifact; push to `ssd/vendor/deepseek-token-rate-16gb`; rebuild from pushed source; rerun strict cold with the same env.
- Promote only if the pushed-source rerun remains `>4.2 tok/s` with the same gates passing.

### 2026-07-03T22:08Z Gate Prefill Top3000 Pushed-Source Promotion

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/gate-prefill-top3000-pushed-repro-result.json`
- artifact sha256: `9052b6f3ecc4cc748a76a314e31fe0508206c93849c26448a664fdb0b2c4a658`

Source/repro status:

- Source commit: `503d75cd7`
- Commit message: `vendor-ds4: add gate prefill sota candidate`
- Pushed remote branch: `ssd/vendor/deepseek-token-rate-16gb`
- Build metadata from pushed source: `b14649-503d75cd7`
- Git identity used for commit: `L-Ark <fliangae@connect.ust.hk>`
- This run was rebuilt from the pushed source before promotion.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`

Config:

- vendor DeepSeek framework only.
- strict cold `drop_caches` before the case.
- 16GB cgroup: `MemoryMax=16000000000`, `MemorySwapMax=0`.
- `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`.
- gate one-stream only: `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`.
- gate VRAM cache size: `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`.
- gate O_DIRECT pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`.
- gate prefill profile: `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`.
- prefill limit: `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`.
- top-k policy: `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`.
- CLI extra args: `-c 256 -b 16 -ub 16 -t 20 -tb 20`.

Metrics:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=32892.55329 ms`
- TTFT gate: pass (`32892.55329 < 33617.688744`)
- `elapsed_seconds=63.94`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15102607360`
- `memory_max_events=17341`
- `pgmajfault=271465`
- `workingset_refault_file=1638812`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Manual correctness verdict:

- Pass. The France answer is complete, coherent, and semantically correct.

Prefill counters:

- profile loaded entries: `3313`
- limit: `3000`
- attempted: `3000`
- inserted: `3000`
- pack misses: `0`
- read failures: `0`
- bytes: `13369344000`
- elapsed: `4485.962 ms`

Gate pack/cache counters:

- one expert pack: `hits=4886 misses=0 reads=4886 bytes=21774204928 failures=0 entries=4599 direct_reads=4886 direct_failures=0 direct_fallbacks=0`
- VRAM cache after excluding prefill insertions from miss accounting: `hits=33265 misses=1886 hit_rate=94.6%`

One-trace aggregate:

| Slice | rows | src0_ms | total_ms |
| --- | ---: | ---: | ---: |
| all gate | `35151` | `6707.771` | `8050.224` |
| gate hits | `33265` | `4520.134` | `5513.780` |
| gate misses | `1886` | `2187.637` | `2536.444` |

Artifact hashes:

- `summary.json`: `cca15940de41427cdba1d6457d702888195fbd2cf7ce3a434f9644ce723b1423`
- `stdout.txt`: `1ec9f76ae3ec1a74b0701a6cb756ee2822f1a554cb3b6f67f436250fad823f71`
- `stderr.txt`: `186bc7727f28c254ac6e5a9e216c1577b831530fde11ea2e6febbd1b1170af69`
- `environment.txt`: `1be760a41fa1dcc46d74c05347331a605c145e7b5ede9a3dde03f045746e549e`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `421e58e3393917bd785df7a9ef300e59fab8f7eb7ccabd499f2e16a85ae84c64`
- `llama-cli`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
- `libggml-cuda.so`: `f73af32dcfc67f23dadef8c20217f784350b7df6a4e7a67c647dad3d2cf89c36`
- `libggml-cpu.so`: `b7d0a3a7a65adac0e8ff2a83ce7bcbbeef8a5ce38ddd173cf8bd6b75abe9ed39`
- `libggml.so`: `e49b194a510b150e2eee78271fc6bfae51f7ea8113909528c829a003756b217f`
- gate profile: `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`
- gate pack: `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`

Promotion verdict:

- Promoted as the current strict cold-start SOTA.
- Accepted SOTA is now `4.4 tok/s`, replacing the previous `4.2 tok/s`.
- It satisfies all active gates: strict 16GB host RAM including page cache, no swap, no cgroup kill, correct output, TTFT within +20%, no gate pack direct failures/fallbacks, and source/docs/artifacts pushed.

Next plan:

- Treat `503d75cd7` and the above run as the new baseline.
- Before any next implementation, re-profile the promoted SOTA bottleneck from its trace and counters.
- Highest-priority next hypotheses are:
  - reduce prefill TTFT cost while preserving the `94.6%` gate hit rate, likely by ranking top entries by saved miss cost rather than raw frequency;
  - test a smaller prefill limit sweep (`2048`, `2560`, `2816`) to recover TTFT headroom and compare generation rate;
  - investigate whether remaining `1886` gate misses are concentrated in a small set that can be prefetched without exceeding the TTFT gate.
- Any new practice run must first append its design, hard-bound, acceptance/rejection gates, and exact config to this plan. Any compliant SOTA must again be fully recorded and immediately pushed to `ssd/vendor/deepseek-token-rate-16gb`.

### 2026-07-03T22:19Z Prefill Limit Sweep Design

Goal:

- Continue from the promoted `4.4 tok/s` cold-start SOTA and search for a better gate-prefill point.
- Reduce prefill TTFT cost while preserving enough gate cache hits to improve generation rate.
- This is a no-source-change experiment from pushed source `cc42924ab`.

Current bottleneck from promoted SOTA:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- `eval_tok_s=4.4`, `TTFT=32892.55329 ms`, TTFT headroom to +20% gate is only `725.135454 ms`.
- Gate cache: `hits=33265 misses=1886 hit_rate=94.6%`.
- Gate prefill: `3000` entries, `13369344000` bytes, `4485.962 ms`.
- Measured prefill read/H2D throughput: about `2.98 GB/s`.
- Remaining gate miss path still costs about `2187.637 ms` of `src0_ms`; reducing it would help generation, but adding more prefill would exceed the TTFT gate unless prefill is re-ranked or made faster.

Hard-bound estimate:

Use the measured `4456448` bytes per gate expert payload and observed prefill throughput from the promoted run. The profile is sorted by frequency, so these bounds estimate TTFT cost and profile frequency coverage for smaller top-N cuts.

| Prefill limit | Payload GiB | Profile frequency coverage | Estimated prefill cost |
| ---: | ---: | ---: | ---: |
| `2048` | `8.50` | `91.31%` | `3062.4 ms` |
| `2560` | `10.63` | `95.56%` | `3828.0 ms` |
| `2816` | `11.69` | `97.07%` | `4210.8 ms` |
| `3000` current | `12.45` | `98.16%` | `4486.0 ms` |

Hypothesis:

- `3000` may over-prefill marginal low-frequency entries and spend too much TTFT near the acceptance limit.
- `2560` or `2816` may keep most of the cache benefit while reducing cold-start pressure and page/cache churn enough to equal or exceed `4.4 tok/s`.
- `2048` is a lower-bound probe: if it drops generation rate significantly, the remaining misses are still costly and the next path should be better ranking rather than smaller limits.

Practice config:

- Use `strict_ds4_runner.py` with cold `drop_caches` before each case.
- Use `MemoryMax=16000000000`, `MemorySwapMax=0`, `ram_kill_threshold_bytes=16000000000`.
- Keep vendor DeepSeek only, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`.
- Keep gate-only one-stream: `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`.
- Keep gate cache size: `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`.
- Keep O_DIRECT gate pack: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`.
- Keep profile: `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`.
- Sweep only `GGML_MOE_STREAM_ONE_PREFILL_LIMIT` over `2048`, `2560`, and `2816`.
- Keep CLI extra args: `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Prompt remains: `Please introduce France in a short paragraph.`

Acceptance gates:

- Promote only if `eval_tok_s > 4.4`.
- France output must be complete, coherent, and semantically correct by manual review.
- `memory_peak_bytes <= 16000000000`, with page cache included in `memory_file_bytes`.
- `ram_limit_killed=false`, `oom_seen=false`.
- `TTFT <= 33617.688744 ms`.
- Pack direct path must have `direct_failures=0` and `direct_fallbacks=0`.
- Source/docs/artifacts must be committed and pushed immediately to `ssd/vendor/deepseek-token-rate-16gb` for any accepted SOTA.

Rejection rules:

- Reject/tie if `eval_tok_s <= 4.4`.
- Reject if correctness fails, TTFT exceeds the gate, cgroup kills the run, OOM occurs, or pack direct failures/fallbacks appear.
- Record every run's metrics, answer, counters, run path, artifact hashes, and verdict in this plan even when rejected.

### 2026-07-03T22:31Z Prefill Limit Sweep Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/prefill-limit-sweep-result.json`
- artifact sha256: `8d524edfb8814f8efe21da600fdfcd1a22178a296aeb2bc1ec6bf17d47871628`

Source/repro status:

- Source HEAD during sweep: `d08537fa9`
- Runtime source baseline: `503d75cd7`
- This was a no-source-change sweep after the promoted `4.4 tok/s` SOTA.
- Current SOTA remains the top3000 prefill run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`

Results:

| Prefill limit | eval tok/s | prompt tok/s | TTFT ms | memory peak | memory file | cache hit rate | prefill ms | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `2048` | `4.3` | `1.7` | `32477.655528` | `16000000000` | `15097073664` | `92.6%` | `3232.749` | rejected, below SOTA |
| `2560` | `4.4` | `1.8` | `32990.806385` | `16000000000` | `15104090112` | `93.9%` | `4058.883` | tie, not promoted |
| `2816` | `4.3` | `1.8` | `32176.267438` | `16000000000` | `15087149056` | `94.4%` | `4488.403` | rejected, below SOTA |

Run paths:

- `2048`: `/root/lfz/runs/vendor-ds4-16gb/20260703T222454Z-20260704_prefill_limit_2048_sweep/france-cpu40-vram0gb`
- `2560`: `/root/lfz/runs/vendor-ds4-16gb/20260703T222634Z-20260704_prefill_limit_2560_sweep/france-cpu40-vram0gb`
- `2816`: `/root/lfz/runs/vendor-ds4-16gb/20260703T222811Z-20260704_prefill_limit_2816_sweep/france-cpu40-vram0gb`

Correctness output for all three runs:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Manual correctness verdict:

- Pass for all three runs. The France answer is complete, coherent, and semantically correct.

Gate/pack checks:

- `2048`: pack `direct_failures=0`, `direct_fallbacks=0`, prefill `pack_misses=0`, `read_failures=0`, gate cache `hits=32553 misses=2598`.
- `2560`: pack `direct_failures=0`, `direct_fallbacks=0`, prefill `pack_misses=0`, `read_failures=0`, gate cache `hits=33024 misses=2127`.
- `2816`: pack `direct_failures=0`, `direct_fallbacks=0`, prefill `pack_misses=0`, `read_failures=0`, gate cache `hits=33179 misses=1972`.

Trace summary:

| Prefill limit | all gate src0 ms | all gate total ms | miss rows | miss src0 ms |
| ---: | ---: | ---: | ---: | ---: |
| `2048` | `6285.951` | `7745.597` | `2598` | `3021.203` |
| `2560` | `6589.900` | `7981.713` | `2127` | `2496.265` |
| `2816` | `6863.720` | `8227.456` | `1972` | `2339.834` |

Artifact hashes:

- `2048 summary.json`: `8086e6d8b812738cf6c418882477e1a69059586aedd099b69fba7dfe87578e5a`
- `2048 stdout.txt`: `04767f2f5708cb57b103c4df7f1d0ba3b07a5eec117bc8ed7c5d610ec7d76f3e`
- `2048 stderr.txt`: `e346043efbd7e0a71f059d814e40e0603214984909f560d33d7be4852e6fe22b`
- `2048 one_trace.csv`: `d4ed7d5c7ec633d6ae3816f5aa467a525531e772fbfe8e11470ea9ca3d3c37fc`
- `2560 summary.json`: `4deb79c535540af31f524ce4904f6218590c5a4d49c2450a446926d5daa20141`
- `2560 stdout.txt`: `03453cbb0a00975ef7ca4dee8d02d75b8170df18026af5f3c9bfb9d877112eb9`
- `2560 stderr.txt`: `64969996e2cea21e94f262ca13cef39dcc08e19e2c14596a0e15eda182ccb2c2`
- `2560 one_trace.csv`: `ff9f2438cc7f1792d49cabcd8e7b7b320403dcef34eac4888c7b998074da0588`
- `2816 summary.json`: `d07e27385544de3b354091f4213ac72a4b71877426785b5f8271d9cdf5be3899`
- `2816 stdout.txt`: `c0dca6b2f181cf6ccc2276c38ceea841d0c0a6be5cb8c8e51378631a0aba4c65`
- `2816 stderr.txt`: `01d46b6cd61321ccbb98d2bfb8a4e76dc4ede4707bd74a9f67530650d6763f3f`
- `2816 one_trace.csv`: `8756a4b0d95a26cba7ba67bc07ea7cd4faca1c734fbaa02ac8d8407ff94cc335`

Conclusion:

- No new SOTA. Current accepted SOTA remains `4.4 tok/s` with top3000 prefill.
- The limit-only sweep shows that simply reducing top-N does not unlock more token rate.
- `2560` is a useful fallback/tie point with lower prefill bytes, but it is not promotable because it does not exceed `4.4 tok/s`.
- Next optimization should re-rank prefill entries by measured/predicted saved miss cost or target the remaining misses directly, instead of sweeping raw frequency top-N.

### 2026-07-03T22:38Z Tail-Fill Prefill Sweep Design

Goal:

- Use remaining VRAM cache slot headroom above top3000 prefill.
- Test whether adding the profile tail into the remaining `192` empty slots improves decode token rate beyond `4.4 tok/s` while staying inside the TTFT gate.
- This is a no-source-change experiment from the current pushed branch.

Current bottleneck and opportunity:

- Current promoted SOTA has `3192` VRAM cache slots and prefilled `3000`, leaving `192` empty slots.
- Remaining misses at top3000: `1886` miss rows, `2187.637 ms` miss `src0_ms`.
- Prefill is capped by code to `g_vcache.n_slots`, so any limit above `3192` is equivalent to `3192`.
- The profile tail after top3000 has nonzero frequency coverage:
  - top3000 to top3072 adds `72` entries and about `144` profile-frequency hits.
  - top3000 to top3136 adds `136` entries and about `272` profile-frequency hits.
  - top3000 to top3192 adds `192` entries and about `384` profile-frequency hits.

Hard-bound estimate:

Using the promoted run's measured prefill throughput and `4456448` bytes per gate expert payload:

| Prefill limit | Extra entries vs 3000 | Extra payload GiB | Estimated extra prefill cost | Profile coverage |
| ---: | ---: | ---: | ---: | ---: |
| `3072` | `72` | `0.30` | `107.7 ms` | `98.58%` |
| `3136` | `136` | `0.56` | `203.4 ms` | `98.96%` |
| `3192` | `192` | `0.80` | `287.1 ms` | `99.29%` |

Potential upper bound:

- Average top3000 miss `src0_ms` is about `1.16 ms/miss`.
- If the added tail entries avoid `144/272/384` misses, the rough generation-side source savings are up to about `167/316/446 ms` before accounting for lookup, kernel, and eviction effects.
- Since `eval_tok_s` excludes TTFT, any real miss reduction can improve token rate, while TTFT should remain under the `33617.688744 ms` gate if prefill cost follows the observed bound.

Main risk:

- `3192` fills every VRAM cache slot. Runtime misses may trigger LRU eviction immediately and can evict prefilled entries that have not been hit yet.
- Therefore `3072` and `3136` are included to leave some empty slots for runtime misses while still using more VRAM than top3000.

Practice config:

- Use `strict_ds4_runner.py` with cold `drop_caches` before each case.
- Use `MemoryMax=16000000000`, `MemorySwapMax=0`, `ram_kill_threshold_bytes=16000000000`.
- Keep vendor DeepSeek only, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`.
- Keep gate-only one-stream: `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`.
- Keep gate cache size: `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`.
- Keep O_DIRECT gate pack and current profile.
- Sweep only `GGML_MOE_STREAM_ONE_PREFILL_LIMIT` over `3072`, `3136`, and `3192`.
- Keep CLI extra args: `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Prompt remains: `Please introduce France in a short paragraph.`

Acceptance gates:

- Promote only if `eval_tok_s > 4.4`.
- France output must be complete, coherent, and semantically correct by manual review.
- `memory_peak_bytes <= 16000000000`, with page cache included.
- `ram_limit_killed=false`, `oom_seen=false`.
- `TTFT <= 33617.688744 ms`.
- Pack direct path must have `direct_failures=0` and `direct_fallbacks=0`.
- Any accepted SOTA must be recorded with full reproduction info and immediately pushed to `ssd/vendor/deepseek-token-rate-16gb`.

Rejection rules:

- Reject/tie if `eval_tok_s <= 4.4`.
- Reject if correctness fails, TTFT exceeds the gate, cgroup kills the run, OOM occurs, or pack direct failures/fallbacks appear.
- If `3192` regresses despite higher hit potential, inspect whether full-cache LRU eviction is the cause before designing a source change.

### 2026-07-03T22:51Z Tail-Fill Prefill Sweep Initial Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/tailfill-prefill-sweep-candidate-result.json`
- artifact sha256: `54a93532ae82721ce9fff91171a18693fee44dd336eeef345794d67c754ac2bf`

Source/repro status:

- Source HEAD during sweep: `e25d3164a`
- Runtime source baseline: `503d75cd7`
- This was a no-source-change sweep after the promoted `4.4 tok/s` SOTA.
- `3192` is an initial SOTA candidate only. It must be rerun from the pushed state before promotion.

Results:

| Prefill limit | eval tok/s | prompt tok/s | TTFT ms | memory peak | memory file | cache hit rate | prefill ms | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3072` | `4.4` | `1.8` | `31962.598706` | `16000000000` | `15105937408` | `94.6%` | `4816.666` | tie, not promoted |
| `3136` | `4.4` | `1.9` | `31694.987578` | `16000000000` | `15073787904` | `94.5%` | `4895.631` | tie, not promoted |
| `3192` | `4.6` | `1.9` | `30430.329319` | `16000000000` | `15097688064` | `94.4%` | `4818.177` | initial SOTA candidate |

Run paths:

- `3072`: `/root/lfz/runs/vendor-ds4-16gb/20260703T224237Z-20260704_tailfill_prefill_3072/france-cpu40-vram0gb`
- `3136`: `/root/lfz/runs/vendor-ds4-16gb/20260703T224414Z-20260704_tailfill_prefill_3136/france-cpu40-vram0gb`
- `3192`: `/root/lfz/runs/vendor-ds4-16gb/20260703T224550Z-20260704_tailfill_prefill_3192/france-cpu40-vram0gb`

Correctness output for all three runs:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Manual correctness verdict:

- Pass for all three runs. The France answer is complete, coherent, and semantically correct.

Gate/pack checks:

- `3072`: pack `direct_failures=0`, `direct_fallbacks=0`, prefill `pack_misses=0`, `read_failures=0`, gate cache `hits=33249 misses=1902`.
- `3136`: pack `direct_failures=0`, `direct_fallbacks=0`, prefill `pack_misses=0`, `read_failures=0`, gate cache `hits=33219 misses=1932`.
- `3192`: pack `direct_failures=0`, `direct_fallbacks=0`, prefill `pack_misses=0`, `read_failures=0`, gate cache `hits=33171 misses=1980`.

Trace summary:

| Prefill limit | all gate src0 ms | all gate total ms | miss rows | miss src0 ms |
| ---: | ---: | ---: | ---: | ---: |
| `3072` | `7034.576` | `8384.945` | `1902` | `2177.123` |
| `3136` | `7158.054` | `8502.376` | `1932` | `2226.671` |
| `3192` | `7043.107` | `8391.671` | `1980` | `2191.142` |

Artifact hashes:

- `3072 summary.json`: `b6a45dfbd643a4030cc5ed8b9f98a3543336fe558117cbace559435bcaf36920`
- `3072 stdout.txt`: `51c265161b8ae19d6ca44c7affcdc75d28714c008f60deea3a3d951bde0b5b7c`
- `3072 stderr.txt`: `564be65671ceef5024542e9beccfeb046dea4e91e907fac2d7ac28933612d65a`
- `3072 one_trace.csv`: `21bac916e220d122983bfe7a90d0b85df1a91ced75b6c9abcb224e1e51d564d8`
- `3136 summary.json`: `f5898ed6f3eb250b15e9e52e3fc5528e7caa7fffe388245051cffb21d095fd4c`
- `3136 stdout.txt`: `51be9c7fcb25e706e3355ac19cad80161b8c2cfed90236c99d755bbe29eb7d3d`
- `3136 stderr.txt`: `e48d0d6538a34dad4290dd322f7704b0949a89915cefd1896f1f247476b7db08`
- `3136 one_trace.csv`: `3898384c05ebecd0a461edbaf891994ed9d47d2231be01498129e86f88be0e88`
- `3192 summary.json`: `4b0ab80b00c9648f2d6892b4bf363131090e297498018ab460b8b6177c1e26c2`
- `3192 stdout.txt`: `007de1579611f81264f4f625688fc5db5250eda259afb45012756db104f12cdf`
- `3192 stderr.txt`: `ac895d8c585af78ef4c0a6dd50f74e63ac1330f2209896105474c1cd1e8746c0`
- `3192 one_trace.csv`: `ee4d9ef08a78b76e47a603985a9cfaec2035e3a4380f6c5e53f89e5abb972ab5`

Initial verdict:

- `3192` produced `4.6 tok/s`, above the current accepted `4.4 tok/s`, and passed RAM, TTFT, correctness, and pack gates.
- It is not yet promoted because the cache-hit and trace metrics do not clearly explain the speedup; it may include run-to-run variance.
- Immediate action: commit and push this source/docs/artifact state, then rerun `3192` strict cold from the pushed state. Promote only if the rerun remains `>4.4 tok/s` and all gates pass.

### 2026-07-03T23:00Z Tail-Fill 3192 Pushed Repro Rejection

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/tailfill-prefill-3192-pushed-repro-rejected.json`
- artifact sha256: `d52ff43cfe991a1070dfdcc7122a2a1e6cf5191a62360ae7da5c38e9cf674981`

Source/repro status:

- Source HEAD: `c6a03329d`
- Build metadata: `b14654-c6a03329d`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T225335Z-20260704_tailfill_prefill_3192_pushed_repro/france-cpu40-vram0gb`

Result:

- `eval_tok_s=4.3`
- `prompt_tok_s=1.8`
- `TTFT=32241.707711 ms`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15096586240`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Manual correctness verdict:

- Pass. The answer is complete, coherent, and semantically correct.

Counters:

- prefill: `attempted=3192 inserted=3192 bytes=14224982016 elapsed_ms=4654.953 pack_misses=0 read_failures=0`
- pack: `hits=5172 misses=0 reads=5172 bytes=23048749056 direct_reads=5172 direct_failures=0 direct_fallbacks=0`
- VRAM cache: `hits=33171 misses=1980 hit_rate=94.4%`

Trace:

| Slice | rows | src0_ms | total_ms |
| --- | ---: | ---: | ---: |
| all gate | `35151` | `6973.252` | `8333.827` |
| gate hits | `33171` | `4688.515` | `5683.304` |
| gate misses | `1980` | `2284.737` | `2650.523` |

Artifact hashes:

- `summary.json`: `50b1194fbc48d3b83cbf657bab9ad947dcb15ac38e74699d2d7f87a82f8e44eb`
- `stdout.txt`: `c46b2115a93a6d86a0e9d28919176b8e45fcf01d6f4d0fa996ac9b0e167b9bf5`
- `stderr.txt`: `84600ec41c95db80a9f2b302ed2538354c17ecddccc6d2d5542126b074978557`
- `environment.txt`: `9144b7e47be345dbc4cd22af6b8ff013d1007ee0b4d65a2c85c003641de1c2f6`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `522c15ce318d7bb885b2a4436a1d3c99ec2b3fe2992b440f16fdf10d88370b80`

Verdict:

- Rejected. The `4.6 tok/s` initial candidate did not reproduce from the pushed state.
- Current accepted strict cold SOTA remains `4.4 tok/s` from top3000 prefill.
- Full-cache `3192` also has worse miss count than top3000 (`1980` vs `1886`), consistent with LRU eviction reducing the expected benefit.
- Next optimization should avoid filling all slots blindly. Prefer a source-level cache policy change, such as preserving prefilled high-frequency entries from early eviction or reserving a small runtime-insert pool.

### 2026-07-03T23:07Z Protected Prefill Cache Policy Design

Goal:

- Prevent runtime misses from evicting high-frequency prefilled gate experts when the prefill uses most or all VRAM cache slots.
- Test whether protected top3192 prefill can convert the theoretical hit-rate headroom into a reproducible token-rate improvement.

Bottleneck:

- Current accepted top3000 SOTA: `4.4 tok/s`, `hits=33265 misses=1886 hit_rate=94.6%`.
- Full-cache top3192 without protection was unstable and rejected:
  - initial: `4.6 tok/s`, but pushed-source rerun: `4.3 tok/s`.
  - rerun miss count: `1980`, worse than top3000.
- This points to LRU eviction: filling all slots lets early runtime misses evict prefilled entries that have not yet been used.

Hard-bound from trace/profile:

Using the promoted top3000 trace and the same profile order, if prefilled entries are protected and runtime miss insertions cannot evict them:

| Protected prefill set | Protected-hit upper bound | Misses without runtime insert | Hit rate upper bound | Repeated outside-profile occurrences |
| ---: | ---: | ---: | ---: | ---: |
| `3000` | `33239` | `1912` | `94.56%` | `626` |
| `3072` | `33383` | `1768` | `94.97%` | `482` |
| `3136` | `33511` | `1640` | `95.33%` | `354` |
| `3192` | `33623` | `1528` | `95.65%` | `242` |

Expected upper bound:

- Protected top3192 can theoretically remove about `358` misses vs accepted top3000 (`1886 - 1528`) and about `452` misses vs unprotected top3192 rerun (`1980 - 1528`).
- Average top3000 miss `src0_ms` is about `1.16 ms`, so the rough generation-side source saving is up to about `415 ms` vs top3000 before accounting for runtime insert losses and kernel overhead.
- TTFT cost is unchanged from top3192 prefill and has passed the gate in both 3192 runs.

Implementation plan:

- Add a default-off env: `GGML_MOE_STREAM_ONE_PREFILL_PROTECT=1`.
- Add per-slot protection state in the one-stream VRAM cache.
- Prefill insertions mark slots protected only when the env is enabled.
- Normal runtime insertions may use empty slots first and evict only unprotected slots. If all slots are protected, insertion fails and the call falls back to the existing per-slot staging buffer without caching that miss.
- Fix cache miss accounting so failed insertions with `count_miss=true` still increment the miss counter.
- Default behavior with the env unset must match the current accepted SOTA path.

Practice config:

- First build and run a default-off guard with accepted top3000 config to ensure no behavior regression.
- Then run protected top3192 under strict cold `drop_caches`, 16GB cgroup, `cpu_moe=40`, gate O_DIRECT pack, and France prompt.
- Candidate env delta:
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3192`
  - `GGML_MOE_STREAM_ONE_PREFILL_PROTECT=1`
- Keep CLI extra args: `-c 256 -b 16 -ub 16 -t 20 -tb 20`.

Acceptance gates:

- Default-off guard must preserve correctness and stay in the `4.4 tok/s` class without RAM/TTFT regressions.
- Protected top3192 promotes only if `eval_tok_s > 4.4` and a pushed-source rerun also exceeds `4.4`.
- France output must be complete, coherent, and semantically correct.
- `memory_peak_bytes <= 16000000000`, page cache included.
- `ram_limit_killed=false`, `oom_seen=false`.
- `TTFT <= 33617.688744 ms`.
- Pack direct path must have `direct_failures=0` and `direct_fallbacks=0`.
- Any accepted SOTA must be recorded with full reproduction info and immediately pushed to `ssd/vendor/deepseek-token-rate-16gb`.

Rejection rules:

- Reject if default-off guard regresses materially.
- Reject if protected top3192 does not exceed `4.4 tok/s`, fails correctness, exceeds TTFT gate, breaks RAM gate, or shows pack direct failures/fallbacks.
- If rejected, revert the source patch and push only documentation/artifact records.

### 2026-07-03T23:17Z Protected Prefill Cache Policy Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/protected-prefill-cache-result.json`
- artifact sha256: `5838dc88d1fb47d8b930b81ea88f5afd9bd52451c12f3998d1da139b949084cb`

Source status:

- Source patch was reverted after rejection.
- No protected-prefill source change is retained in the final worktree.

Results:

| Run | eval tok/s | prompt tok/s | TTFT ms | memory peak | memory file | cache hit rate | protected slots | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default-off top3000 guard | `4.3` | `1.8` | `32714.155432` | `16000000000` | `15084904448` | `94.6%` | `0` | guard shape ok |
| protected top3192 | `4.4` | `1.8` | `31898.253672` | `16000000000` | `15101079552` | `95.7%` | `3192` | rejected/tie |

Run paths:

- default-off guard: `/root/lfz/runs/vendor-ds4-16gb/20260703T230647Z-20260704_prefill_protect_default_off_guard/france-cpu40-vram0gb`
- protected top3192: `/root/lfz/runs/vendor-ds4-16gb/20260703T230900Z-20260704_prefill_protect_3192_candidate/france-cpu40-vram0gb`

Correctness:

- Both runs produced the same complete, coherent France answer and passed manual correctness review.

Key counters:

- default-off top3000: `hits=33265 misses=1886 hit_rate=94.6% protected=0`, pack `direct_failures=0 direct_fallbacks=0`.
- protected top3192: `hits=33623 misses=1528 hit_rate=95.7% protected=3192`, pack `direct_failures=0 direct_fallbacks=0`.
- protected top3192 had `cache_inserts=0`, as intended.

Trace:

| Run | all gate src0 ms | all gate total ms | miss rows | miss src0 ms |
| --- | ---: | ---: | ---: | ---: |
| default-off top3000 | `6721.699` | `8078.077` | `1886` | `2180.777` |
| protected top3192 | `6648.259` | `7930.953` | `1528` | `1796.433` |

Artifact hashes:

- default-off `summary.json`: `2f3fcad00d27b8b8fa551de0ea582f6a11fc009c8e913e9661214d900a8d524c`
- default-off `stdout.txt`: `55ec23f41df6a6e1d416e6a43535cfa6cb8f92e7a230c3ce497b0a30c02ee587`
- default-off `stderr.txt`: `7b3a96edb9cba09c4c898e19e3553492c757cb705a7856b16050236ba6f143fe`
- default-off `one_trace.csv`: `5f6216726a888754d90507117fe54bf2e5dacd32a757a1189dd25120c7707017`
- protected `summary.json`: `e220528e502e87042e132b6d1938d61bee80ca96ec57ff580f56cb6afd4390cd`
- protected `stdout.txt`: `d46478e3064f967037f23332bfc197dd06ddcd780757bd3b1d39f3385681ccc3`
- protected `stderr.txt`: `db403ec91aff34a457f91e5d8140d3af69d4597d8914460e8620c16d15d0e527`
- protected `one_trace.csv`: `bcac798a3efb37116d7406639933f0b5a0d53069c1e0786cc36a4a03d4f95aaf`

Conclusion:

- Rejected. Protected prefill achieved the predicted miss reduction (`1528` misses) but did not improve token rate beyond `4.4`.
- Current accepted SOTA remains `4.4 tok/s` from top3000 prefill.
- This suggests the next bottleneck is not only miss count; the remaining source movement, sync/scatter overhead, or loss of runtime insert reuse matters.
- Next source-level direction should preserve a small runtime insert pool or reduce miss-path H2D/sync cost, rather than fully disabling runtime cache insertions.

### 2026-07-03T23:22Z Protected Prefix Plus Runtime Pool Design

Goal:

- Combine the useful part of protected prefill with the useful part of runtime cache insertion.
- Protect the high-frequency prefill prefix, but leave a small unprotected runtime insert pool for repeated tail misses.

Bottleneck after protected-all:

- Protected top3192 achieved the predicted `33623 hits / 1528 misses`, but `eval_tok_s=4.4` only tied SOTA.
- It also had `cache_inserts=0`, so repeated out-of-profile/profile-tail misses could not be reused.
- Current unprotected top3000 has worse theoretical protection but allows runtime inserts; it remains the accepted `4.4 tok/s` SOTA.

Hard-bound sequential simulation:

Using the promoted top3000 trace order, current admission profile, `3192` total slots, protected top-N prefix, and an LRU runtime pool of `3192 - N` unprotected slots:

| Protected prefix | Runtime pool slots | Sim hits | Sim misses | Sim hit rate | Runtime-pool hits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `3000` | `192` | `33491` | `1660` | `95.28%` | `252` |
| `3072` | `120` | `33566` | `1585` | `95.49%` | `183` |
| `3136` | `56` | `33633` | `1518` | `95.68%` | `122` |
| `3192` | `0` | `33623` | `1528` | `95.65%` | `0` |

Hypothesis:

- `3136` protected prefix + `56` runtime slots is the best predicted point: it slightly beats protected top3192's miss count while avoiding full-cache eviction of protected entries.
- It has lower prefill cost than top3192 and preserves enough runtime insertion to catch repeated tail keys.

Implementation plan:

- Re-apply the default-off protected-slot cache patch already tested:
  - `GGML_MOE_STREAM_ONE_PREFILL_PROTECT=1` marks prefill slots protected.
  - normal runtime insertions use empty slots first and then evict only unprotected slots.
  - if no unprotected slots exist, insertion fails and falls back to the staging buffer.
- No behavior change when the env is unset.
- Run protected `PREFILL_LIMIT=3136` first. If it exceeds `4.4`, commit/push source/docs/artifacts and rerun from pushed source before promotion.
- If it only ties or regresses, revert source again and record rejection.

Practice config:

- strict cold `drop_caches`, 16GB cgroup, `MemorySwapMax=0`.
- vendor DeepSeek only, `cpu_moe=40`, gate O_DIRECT pack, gate one-stream.
- env delta:
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3136`
  - `GGML_MOE_STREAM_ONE_PREFILL_PROTECT=1`
- Keep CLI extra args: `-c 256 -b 16 -ub 16 -t 20 -tb 20`.

Acceptance gates:

- Promote only if `eval_tok_s > 4.4`, and a pushed-source rerun also exceeds `4.4`.
- France output must be complete, coherent, and semantically correct.
- `memory_peak_bytes <= 16000000000`, including page cache.
- `ram_limit_killed=false`, `oom_seen=false`.
- `TTFT <= 33617.688744 ms`.
- Pack direct path must have `direct_failures=0` and `direct_fallbacks=0`.

Rejection rules:

- Reject if `eval_tok_s <= 4.4`, correctness fails, TTFT exceeds the gate, RAM gate fails, or pack direct failures/fallbacks appear.
- If rejected, revert source and push only docs/artifacts.

### 2026-07-03T23:30Z Protected Prefix Plus Runtime Pool Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/protected-runtime-pool-result.json`
- artifact sha256: `6cdc2b25cffe0eadc677b58ef1b68cc94df093bbe51fa3699edda1bebdd3befd`

Source status:

- Source patch was reverted after rejection.
- No protected/runtime-pool source change is retained in the final worktree.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T232143Z-20260704_prefill_protect_3136_pool_candidate/france-cpu40-vram0gb`

Metrics:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=34502.12893 ms`
- TTFT gate: fail (`34502.12893 > 33617.688744`)
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15104081920`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Manual correctness verdict:

- Pass. The answer is complete, coherent, and semantically correct.

Counters:

- prefill: `attempted=3136 inserted=3136 bytes=13975420928 elapsed_ms=5128.371 pack_misses=0 read_failures=0`
- pack: `hits=4663 misses=0 reads=4663 bytes=20780417024 direct_reads=4663 direct_failures=0 direct_fallbacks=0`
- VRAM cache: `hits=33624 misses=1527 hit_rate=95.7% protected=3136`
- runtime pool activity: `cache_inserts=241`

Trace:

| Slice | rows | src0_ms | total_ms |
| --- | ---: | ---: | ---: |
| all gate | `35151` | `6939.244` | `8218.065` |
| gate hits | `33624` | `5162.197` | `6157.414` |
| gate misses | `1527` | `1777.047` | `2060.651` |
| cache inserts | `241` | `299.524` | `344.429` |

Artifact hashes:

- `summary.json`: `644fe8757aebaeb7999d2323a6e8c83a5db90016ed6c8c8afaeb06895bf1e94a`
- `stdout.txt`: `6449d83b8bb4720dc9762696716cae20fcafdba10de4481bd6513a5c350155c9`
- `stderr.txt`: `fceb68c46ca89118d8dea1ac5eaae65d15f98c8e2eb1839476928a1b16e99035`
- `environment.txt`: `9e2ae12d003f0005e538364a5afc85789cfa1266d2ffce9b59f913b5d95652c9`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `33205a16c7cc142dc65886359fdee46659d37fb2cf8c9715df4f6faba9941c3f`

Verdict:

- Rejected. The hit-rate target and runtime-pool behavior were achieved, but token rate only tied `4.4` and TTFT exceeded the acceptance gate.
- Current accepted SOTA remains `4.4 tok/s` from top3000 prefill.
- This path suggests cache-hit count alone is no longer the dominant bottleneck. Next work should reduce miss-path and hit-path overhead directly, especially H2D/sync/scatter cost and prefill TTFT cost.

### 2026-07-03T23:30Z No-Trace SOTA Guard Design

Goal:

- Test whether the currently accepted top3000 prefill SOTA is paying measurable decode overhead for writing `one_trace.csv`.
- Keep the exact accepted runtime path and remove only `GGML_MOE_STREAM_ONE_TRACE_OUT`.

Bottleneck:

- Current accepted top3000 SOTA recorded `35151` gate trace rows.
- Trace writing includes a per-call mutex, timestamp formatting, and line-buffered file output when `GGML_MOE_STREAM_ONE_TRACE_OUT` is set.
- Accepted `4.4 tok/s` run used trace output; if trace I/O is on the hot path, a no-trace strict cold run may increase `eval_tok_s` without changing model math.

Hard-bound:

- The maximum direct trace-writing benefit is bounded by the traced gate `total_ms` overhead, but the exact file I/O cost is not separately recorded.
- If the only cost is line-buffered CSV output, expected improvement is likely small but nearly free to test.
- Correctness should be unchanged because this is instrumentation-only.

Practice config:

- No source change.
- strict cold `drop_caches`, 16GB cgroup, `MemorySwapMax=0`.
- vendor DeepSeek only, `cpu_moe=40`, gate O_DIRECT pack, gate one-stream.
- Keep accepted top3000 prefill:
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`
  - `GGML_MOE_STREAM_ONE_PREFILL_PROFILE=.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`
- Keep accepted cache and top-k env.
- Omit only `GGML_MOE_STREAM_ONE_TRACE_OUT`.
- Keep CLI extra args: `-c 256 -b 16 -ub 16 -t 20 -tb 20`.

Acceptance gates:

- Promote only if `eval_tok_s > 4.4`, and a repeated no-trace strict cold run also exceeds `4.4`.
- France output must be complete, coherent, and semantically correct.
- `memory_peak_bytes <= 16000000000`, including page cache.
- `ram_limit_killed=false`, `oom_seen=false`.
- `TTFT <= 33617.688744 ms`.
- Pack direct path must have `direct_failures=0` and `direct_fallbacks=0`.
- Record stderr counters, exact command, environment, summary, hashes, and correctness output even though there will be no `one_trace.csv`.

Rejection rules:

- Reject/tie if `eval_tok_s <= 4.4`.
- Reject if correctness fails, RAM gate fails, TTFT exceeds the gate, or pack direct failures/fallbacks appear.

### 2026-07-03T23:39Z No-Trace SOTA Guard Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/no-trace-sota-guard-result.json`
- artifact sha256: `2192fd9a25145666051ec75c8558a887b5357e227d94db977784fc156920a3c5`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T233237Z-20260704_top3000_no_trace_guard/france-cpu40-vram0gb`

Metrics:

- `eval_tok_s=4.3`
- `prompt_tok_s=1.9`
- `TTFT=31989.966998 ms`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15106867200`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Manual correctness verdict:

- Pass. The answer is complete, coherent, and semantically correct.

Counters:

- prefill: `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4450.097 pack_misses=0 read_failures=0`
- pack: `hits=4886 misses=0 reads=4886 bytes=21774204928 direct_reads=4886 direct_failures=0 direct_fallbacks=0`
- VRAM cache: `hits=33265 misses=1886 hit_rate=94.6%`

Artifact hashes:

- `summary.json`: `ad30fa465d03e69ff953f4a0fa60665132cfeee809294feb81d3b1c58c919485`
- `stdout.txt`: `b5a1e0152d675b4a95dd33683665160b1804e189b7fa7eaa7475968d10d33fc5`
- `stderr.txt`: `23f72cf0de69442e37b584a4797c8ac547da15d49b682d1c4767e779abc5bd5d`
- `environment.txt`: `15bf65f081f30992c44a935019a57b63fe30c34fdf9bd3a458676bd2fbdb110f`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`

Verdict:

- Rejected. Removing `GGML_MOE_STREAM_ONE_TRACE_OUT` did not improve token rate.
- Current accepted SOTA remains `4.4 tok/s` from top3000 prefill.
- Trace file output is not the bottleneck at this point; next work should focus on compute/synchronization/H2D path costs rather than instrumentation output.

### 2026-07-03T23:43Z Trace-Disabled Fast Path Design

Goal:

- Remove trace instrumentation overhead when `GGML_MOE_STREAM_ONE_TRACE_OUT` is not set.
- Keep full trace behavior unchanged when trace output is requested.

Bottleneck:

- The no-trace guard still ran through the timing/trace scaffolding:
  - several `std::chrono::steady_clock::now()` calls per expert invocation;
  - `one_trace_write()` call on every invocation;
  - mutex acquisition inside `one_trace_write()` before returning because no file is open.
- That means the previous no-trace experiment removed file output but not the hot-path timing/lock overhead.

Hard-bound:

- There are `35151` gate expert invocations in the France run.
- If timing + mutex overhead is only `1-5 us` per invocation, the upper bound is about `35-175 ms`.
- If contention from many CPU worker threads on the trace mutex is higher, the benefit could be larger. The test is low-risk because it is instrumentation-only when trace is disabled.

Implementation plan:

- Add a cached `one_trace_requested()` helper that checks `GGML_MOE_STREAM_ONE_TRACE_OUT`.
- In `ggml_cuda_moe_stream_one`, only take `steady_clock::now()` timing checkpoints and call `one_trace_write()` when trace is requested.
- Preserve all existing compute, H2D, D2H, sync, scatter, cache, pack, and correctness behavior.
- Preserve current trace behavior when `GGML_MOE_STREAM_ONE_TRACE_OUT` is set.

Practice config:

- Build source candidate.
- Run strict cold no-trace top3000 SOTA config:
  - no `GGML_MOE_STREAM_ONE_TRACE_OUT`;
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`;
  - same gate O_DIRECT pack, cache size, top-k policy, `cpu_moe=40`, 16GB cgroup, and CLI args.
- If it exceeds `4.4`, commit/push source/docs/artifacts and rerun from pushed source before promotion.
- If it does not exceed `4.4`, revert source and record rejection.

Acceptance gates:

- Promote only if `eval_tok_s > 4.4`, and a pushed-source rerun also exceeds `4.4`.
- France output must be complete, coherent, and semantically correct.
- `memory_peak_bytes <= 16000000000`, including page cache.
- `ram_limit_killed=false`, `oom_seen=false`.
- `TTFT <= 33617.688744 ms`.
- Pack direct path must have `direct_failures=0` and `direct_fallbacks=0`.

Rejection rules:

- Reject if `eval_tok_s <= 4.4`, correctness fails, RAM gate fails, TTFT exceeds the gate, or pack direct failures/fallbacks appear.
- If rejected, revert the source patch and push only docs/artifacts.
