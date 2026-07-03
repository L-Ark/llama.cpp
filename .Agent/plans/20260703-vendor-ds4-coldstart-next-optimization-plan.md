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
- No source edits are included in this plan commit.
