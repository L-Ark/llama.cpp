# Vendor DeepSeek 16GB Cold-Start 下一阶段优化计划

## Summary

本计划从当前已 push 的 vendor DeepSeek cold-start 复现状态继续推进。最终结果必须体现在 `vendor` 框架，`ik_llama` 只能作为参考。

当前事实：

- 历史最高观测：`4.2 tok/s`，run `/root/lfz/runs/vendor-ds4-16gb/20260702T133103Z-20260702_pushed_onepack_odirect_sota_rerun/france-cpu40-vram0gb`。
- 2026-07-03 post-rollback strict cold guard 已复现 `4.2 tok/s`，run `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb`。
- 2026-07-03 多次 strict cold rerun 达到 `4.1-4.2 tok/s`，均满足：16GB cgroup 含 page cache、France 正确率、TTFT gate、O_DIRECT expert pack、VRAM cache counters。
- 下一阶段目标：稳定超过 `4.2 tok/s`；未超过 `4.2 tok/s` 的结果只能作为 diagnostic/rejected/tie，不得 promote。

## Current Baseline

- `source_head`: `cb5c74f096e7f22b90d3788baf2a4ccf2afcf3b2`，已 push 到 `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`。
- `runtime_binary_build`: stdout reports `build : b14557-1e9b327d5`。
- `runtime_source_note`: `1e9b327d57e4f75be101f1b88cf064df7d39a658..cb5c74f096e7f22b90d3788baf2a4ccf2afcf3b2` 之间只有 plan/record 文档变更，runtime source path unchanged。
- `binary_sha256`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`。
- `pack_sha256`: `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076` for `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`。
- `profile_sha256`: `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b` for `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`。
- `accepted_config`: `cpu_moe=40`, `--vram-cache-gb 0`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`, strict cold `drop_caches`, 16GB `systemd-run` cgroup with `MemorySwapMax=0`, CLI extra args `-c 256 -b 16 -ub 16 -t 20 -tb 20`。

Fresh strict cold reproduction evidence:

| Run | eval_tok_s | prompt_tok_s | TTFT ms | memory_peak | memory_file | correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T031923Z-20260703T031710Z-current-sota-strict-repro/france-cpu40-vram0gb` | 4.1 | 1.5 | 30470.663075 | 16000000000 | 15102509056 | true |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T032154Z-20260703T032138Z-current-sota-strict-repro-retry/france-cpu40-vram0gb` | 4.1 | 1.6 | 28857.563383 | 16000000000 | 15102316544 | true |
| `/root/lfz/runs/vendor-ds4-16gb/20260703T040442Z-20260703T040442Z-post-local-mmap-revert-guard/france-cpu40-vram0gb` | 4.2 | 1.6 | 28014.740620 | 16000000000 | 15096049664 | true |

These runs had `oom=0`, `oom_kill=0`, `ram_limit_killed=false`, one expert pack `hits=4623 misses=0 direct_reads=4623 direct_failures=0`, and VRAM cache `hits=30528 misses=4623 hit_rate=86.8%` with `3192` slots (`13.2GiB`).

Post-rollback guard France answer:

> France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.

## Bottleneck And Optimization Direction

Current measured bottleneck after O_DIRECT:

- Gate O_DIRECT pack path is no longer dominant.
- Gate miss source load is roughly `5.3s`.
- CPU up/down fallback is the main target, about `26.8s` total, with decode up/down about `19.3s`.
- Large buffered down pack is rejected because it competes with model mmap/page cache and raises refault pressure under the 16GB cgroup.
- Broad gate cache, global thread sweeps, and large buffered down-pack directions are deprioritized.

Next optimization direction:

1. Re-run one bottleneck slice on current pushed head `cb5c74f09` with the current accepted O_DIRECT config to refresh gate timing, fallback timing, cache counters, and cgroup page/refault data.
2. Generate top fallback hotsets from the existing fallback profile and the refreshed profile.
3. Evaluate only small, memory-controlled up/down fallback reductions: top `128` first, then top `256` only if top `128` does not regress gate cache or page-cache behavior.
4. Prefer O_DIRECT or VRAM-backed small expert sets. Avoid buffered packs large enough to materially displace page cache.
5. Reject without running full SOTA if theoretical covered fallback time is below about `2s`, because it is unlikely to beat run variance and the `4.2 tok/s` promote threshold.
6. Stop a candidate immediately if gate cache hit rate drops materially, expert pack direct fallbacks appear, RAM exceeds 16GB, TTFT violates gate, or France output degrades.

## Execution Plan

### Phase 0: Current Baseline Guard

- Run one strict cold baseline from pushed head `cb5c74f09` before any source change.
- Required result: `4.1 tok/s` class, `ram_ok=true`, `correctness_ok=true`, O_DIRECT pack counters unchanged.
- If baseline falls below `4.0 tok/s` or counters differ, stop and debug reproducibility before optimization.

### Phase 1: Trace/Profile Refresh

- Run accepted config with one-stream trace and CPU fallback profile enabled.
- Record:
  - one-stream `src0_ms`, `kernel_ms`, `sync_ms`, total/span/gap;
  - CPU fallback by prompt/decode, up/down, layer/expert;
  - cgroup `memory.peak`, `memory.stat file/anon`, `pgmajfault`, `workingset_refault_file`;
  - pack counters and VRAM cache counters;
  - France answer text.
- This trace run is diagnostic only and cannot replace the accepted SOTA metric.

### Phase 2: Top-N Up/Down Candidate Design

- Build hotsets from fallback profile using normalized fallback contribution by `(tensor, layer, expert)`.
- For each candidate, calculate before running:
  - covered fallback ms;
  - payload size;
  - expected VRAM or page-cache cost;
  - expected gate cache slot loss;
  - theoretical upper bound on token rate.
- Start with top `128` up/down pairs. Proceed to top `256` only if top `128` preserves current O_DIRECT/gate counters and does not regress token rate meaningfully.

### Phase 3: Candidate Practice

- Use default-off env flags for any new source path.
- Preserve current gate O_DIRECT pack config unless the candidate explicitly measures its effect.
- Run strict cold France under 16GB cgroup.
- If a candidate exceeds `4.2 tok/s` and passes all gates, stop further exploration and complete SOTA promotion steps immediately.

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
