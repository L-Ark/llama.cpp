# Vendor DeepSeek 16GB Cold-Start 下一阶段优化计划

## Summary

本计划从当前已 push 的 vendor DeepSeek cold-start 复现状态继续推进。最终结果必须体现在 `vendor` 框架，`ik_llama` 只能作为参考。

当前事实：

- 历史最高观测：`4.2 tok/s`，run `/root/lfz/runs/vendor-ds4-16gb/20260702T133103Z-20260702_pushed_onepack_odirect_sota_rerun/france-cpu40-vram0gb`。
- 本次可重复 strict cold 复现线：`4.1 tok/s`，两次 2026-07-03 strict rerun 都达到 `4.1 tok/s`。
- 两次复现均满足：16GB cgroup 含 page cache、France 正确率、TTFT gate、O_DIRECT expert pack、VRAM cache counters。
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

Both runs had `oom=0`, `oom_kill=0`, `ram_limit_killed=false`, one expert pack `hits=4623 misses=0 direct_reads=4623 direct_failures=0`, and VRAM cache `hits=30528 misses=4623 hit_rate=86.8%` with `3192` slots (`13.2GiB`).

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
