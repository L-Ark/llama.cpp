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
