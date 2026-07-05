# Vendor DeepSeek 16GB Cold-Start 下一阶段优化计划

## Summary

本计划从当前已 push 的 vendor DeepSeek cold-start 复现状态继续推进。最终结果必须体现在 `vendor` 框架，`ik_llama` 只能作为参考。

### 2026-07-05 Latest Plan: External Artifact Refresh After Thread-Count Closure

本节是当前最新生效计划，覆盖下面所有较早的 `Latest Plan` / `Latest Active Plan` / `Latest Active Plan Override` 段落；旧段落只作为历史实验记录保留。后续优化继续围绕 strict cold-start vendor DeepSeek，不能把 warm page-cache、steady-state、trace/top1-only、ik_llama 或非 vendor 结果提升为 SOTA。本次更新完成 thread-count closure 之后的 external artifact refresh：当前公开 artifact、现有 runtime route、CPU microprobe、线程数参数、4Expert/Q4K GGUF、SSD Flash-MoE sidecar、DFlash/EAGLE draft 和磁盘状态都没有给出可直接实施的 10 tok/s source patch。

Current accepted strict cold SOTA 仍然是 `4.4 tok/s`：

- Accepted SOTA run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Current-head no-trace guard: `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`
- Guard metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32087.738292 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`
- Current source/artifact head before this plan update: `c9db532bcffa9cfd5e5e009d5ccbc5b7590b8421` (`vendor-ds4: reject strict cold thread sweep`)
- Push target for all future source/artifact updates: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- Git identity for future commits/pushes: `L-Ark <fliangae@connect.ust.hk>`
- Promotion gate remains strict: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, strict cold `drop_caches`, 16GB cgroup including file page cache, `MemorySwapMax=0`, no swap/OOM, France answer semantically correct and coherent, source plus artifacts committed and pushed, then clean pushed-source reproduction.

Current closed-route summary:

- Forced-batch target-verifier diagnostic did not show useful sublinear target verification: sequential fixed-text run `107.623408161 s`, batch16 fixed-text run `107.597260262 s`, speedup only `1.000243x`.
- MTP/source-union amortization is insufficient for current DSpark/MTP work: `W=5` projects only `8.454 tok/s`, `W=6` only `9.069 tok/s` even under an unrealistically favorable bound where non-MoE decode batches perfectly. `W=8` reaches `10.512 tok/s` only in an extreme no-overhead lower bound and has no compatible verifier/runtime proof.
- Native MXFP4 exact compact representation is closed: top4096 is the first positive raw exact hotset, but it needs about `5.33x` exact reduction to fit; measured/estimated lossless compression and entropy evidence are only about `1.04x-1.09x`.
- Antirez exact N=2 MTP remains closed for the `10 tok/s` objective: even perfect two-token acceptance requires `C_verify + C_draft <= 1.32x` one current target-token pass, while current decode CPU up/down fallback alone is about `139.9 ms/token`.
- External DSpark/MTP artifacts are still not source-ready for vendor: official/community DSpark remains safetensors/custom inference or MLX, while the public GGUF MTP sidecar uses unsupported `deepseek4_mtp_support`/`mtp.0.*` tensors and exact N=2 is already hard-bound negative.
- External MTP sidecar follow-up found only safetensors/vLLM/MLX sidecars (`FoxlightAI`, `canada-quant`, `LordNeel`, `inferencerlabs`, `mlx-community`) and no standalone vendor-loadable GGUF draft. Generic `llama-speculative` remains unusable for these because it expects a standalone draft model, not a DeepSeek4 MTP sidecar.
- Public lower-bit GGUF target files were screened. The current best concrete header-verified candidate, `0xSero/DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf`, is still below the `10 tok/s` hard-bound on a 32GB RTX 5090 if enough CPU MoE layers remain off-GPU; optimistic placement tops out around `9.33 tok/s` with zero reserve and around `9.07 tok/s` with 3GiB reserve.
- Latest public artifact refresh adds `cloudyu/DeepSeek-V4-Flash-4Expert-GGUF`, `teto3` Q4KExperts, `servantofares` SSD Flash-MoE sidecar, `RedHatAI` DFlash, `ManiacLabs` EAGLE3.1, and recent Huihui/CyberNeurova/bullerwins low-bit GGUFs. None is source-ready for vendor 10 tok/s: 4Expert+perfect source elimination is only `9.581 tok/s` in an optimistic bound, SSD source-page-only is `8.993 tok/s`, DFlash/EAGLE require custom vLLM overlays, and recent low-bit GGUFs do not beat the prior strongest 0xSero hard-bound.
- Joint source/page overlap plus CPU fallback batching is closed for current evidence. The CPU batch microprobe passed top1 but measured only `74.719 GiB/s` best source-payload bandwidth, below the `90.643 GiB/s` zero-overhead requirement and far below `130.298 GiB/s` with only `500 ms` overhead.
- No-source CPU thread-count sweep is closed for the accepted route. Strict-cold `-t/-tb` values `24`, `32`, `40`, and `60` produced `4.4`, `2.6`, `2.1`, and `1.8 tok/s` respectively; all outputs were semantically correct, but none exceeded the accepted `4.4 tok/s` SOTA and `t=60` also exceeded the TTFT gate.
- Disk state blocks large alternate-artifact empirical work unless space is freed deliberately: `/root` has about `1.8 GiB` free. The only clearly disposable candidate found is a `2.965 GB` Hugging Face `.incomplete` cache fragment, which is not enough for 50GB+ model tests. The `20.5 GB` France gate-miss pack is used by accepted SOTA and must be preserved for reproduction.
- Previously closed routes remain closed unless a new hard-bound artifact changes the limiting math: source/page-only prefetch or io_uring, CPU batch rewrite, extra full GPU MoE layer, rectangular `DS4_HOT_DISPATCH`, direct top768 Q8_0, raw/transposed/row-tile exact hot-batch kernels, CUDA graph wrapping, standalone MMVQ skip/write, sparse retained top64 graph, transient MXFP4 repack, no-source lookahead/ngram speculation, and raw top4096 residency.

Latest high-acceptance verifier/draft audit:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/high-acceptance-verifier-draft-current-audit.json`.
- Result: `no_compatible_high_acceptance_verifier_or_draft_path_available_now`; runtime source remains frozen.
- Local inventory: no local official DSpark safetensors checkpoint and no local antirez `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`; only the accepted target DeepSeek GGUF is available in the strict-cold vendor path.
- Current external check: official `deepseek-ai/DeepSeek-V4-Flash-DSpark` is still safetensors/custom inference with `dspark_block_size=5`, `dspark_target_layer_ids=[40,41,42]`, `dspark_markov_rank=256`; antirez still exposes one supplementary MTP GGUF, `3807602400` bytes / `3.546106 GiB`.
- Current vendor source still has no `deepseek4_mtp_support` architecture handling, no `mtp.0.*` tensor runtime, and no official DSpark Markov/confidence verifier path. Generic `llama-speculative` exists but needs a standalone draft model with matching vocab; the antirez file is supplementary, not standalone.
- Hard bound remains negative: exact N=2 is closed; official block-size 5 is below `10 tok/s` under source-union math; current forced-batch fixed-text diagnostic showed only `1.000243x` speedup; full MTP preload would take about `1916.814 ms` at the measured `1.85 GiB/s`, exceeding the current `1529.950452 ms` TTFT slack.
- Decision: do not download, code, or benchmark a DSpark/MTP verifier now. Reopen verifier/draft only if a concrete compatible artifact proves `A>=7`, or `A>=5` with measured sublinear target verification, plus full RAM/page-cache, VRAM, TTFT, and correctness accounting.

Latest non-native representation hard-bound:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/non-native-representation-next-hard-bound.json`.
- Result: `no_non_native_representation_source_patch_allowed_now`; runtime source remains frozen.
- Exact/native compression is still far below the needed reduction: required effective top4096 reduction is about `5.33x`, while zstd/entropy evidence remains about `1.04x-1.09x`, with no duplicate or zero MXFP4 sampled blocks.
- Q8_0 CPU-compatible arithmetic is token-stable for fixed-text top1, but it does not reduce payload enough: top768 fits at `3.188 GiB` but has only `6.48 tok/s` zero-overhead ceiling; top4096 first crosses `10 tok/s` at zero overhead but needs `17.0 GiB`.
- Standalone MMVQ/codebook skip/write is closed because fixed-text top1 failed (`142/145`, first mismatch position `9`) and its direct pool displaced the accepted gate cache.
- Graph-level zero-transfer sparse MMVQ/top64 has only `18.947 ms` paper margin and requires retained CUDA gate/up/down dataflow. Current accepted graph has `graph_gate_output_input_available=0`, `hot_active=0`, `hot_ready=0`, and many observed full up/down tensors are not CUDA-resident, so this is not source-ready.
- More aggressive low-bit, low-rank, sparsified, or learned-codebook representations have no top1 proof. They may only reopen through an offline/compare-only artifact proving fixed-text top1 plus `>5.33x` effective payload reduction or equivalent fallback removal before source changes.

Latest exact graph/dataflow hard-bound:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/exact-graph-dataflow-current-hard-bound.json`.
- Result: `no_exact_graph_dataflow_source_patch_allowed_now`; runtime source remains frozen.
- Source/dataflow audit: `build_expert_mix` receives `cur_ffn`, `selected_experts`, and `weights`, but no reusable retained gate output. The sparse retained probe hardcodes/records `graph_gate_output_input_available=0`; current `DS4_HOT_DISPATCH` recomputes gate/up/down and uses a K+P+1 rectangular hot-expert shape.
- Compact sparse top64 graph branch is closed for the current graph: paper bound is only `10.014 tok/s` with `18.947 ms` margin; one combine-copy lower bound at `24 GiB/s` consumes `17.197 ms`, leaving only `1.75 ms` before scheduler/launch/sync/correctness overhead, and the pushed probe shows `hot_active=0`, `hot_ready=0`, `dispatch_dual=0`.
- Adapting current `DS4_HOT_DISPATCH` is closed: 64 real sparse pairs inflate to `295` rectangular entries, `2507.5 MiB` up/down or `3761.25 MiB` gate/up/down payload; gate recompute bound is only `9.861 tok/s`.
- Current exact kernels are too slow for the remaining bounds: top128 needs about `246 GiB/s` before integration overhead, while raw/transposed/row-tile exact probes measured about `74.59`, `72.434`, and `62.907 GiB/s`. The top768 exact route needs `<=446.414 ms` overhead but measured exact kernel projections are about `1.82-2.16 s`.
- Full selected-MoE exact CUDA replacement is closed for current payload/kernel shape: full exact raw/transposed projections need about `2.0 s` kernel time against only `1.729 s` total fallback-removal margin, and unique decode up/down payload is `21.806 GiB`, too large for VRAM and the strict 16GB host/page-cache budget.

Latest external artifact watch:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/external-artifact-watch-after-exact-graph.json`.
- Result: `no_external_artifact_runtime_patch_allowed_now`; runtime source remains frozen.
- Local inventory correction: `/root/lfz/models/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-00001-of-00001.gguf` is only a symlink to `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`; it is not a new quantization or alternate target.
- Official `deepseek-ai/DeepSeek-V4-Flash-DSpark`, `fraserprice/DeepSeek-V4-Flash-DSpark`, and `autotrust/DeepSeek-V4-Flash-DSpark-4E` are safetensors/custom inference artifacts. They are useful reference material for DSpark semantics but are not directly loadable by current vendor GGUF code.
- MTP search still does not provide a compatible vendor route. Antirez exposes `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (`3807602400` bytes), but current vendor source cannot load `deepseek4_mtp_support` or run `mtp.0.*`; exact N=2 MTP remains closed by the cost bound.
- GGUF target search found lower-bit candidate files that could reduce total model bytes, for example antirez IQ2XXS (`86720111200` bytes), tarruda Q2_K split total (`96962159168` bytes), teamblobfish IQ1_M split total (`64508046688` bytes), sleepyeldrazi REAP-K128 uniform (`50439361920` bytes), and 0xSero Spark-Mini Q2 REAP (`52593532000` bytes). These are not the same accepted native model artifact and may change quality; no token-rate or correctness claim is allowed until strict tests pass.
- Decision: the only newly visible candidate class is an alternate lower-bit GGUF target/representation run. Before any download or benchmark, write `.Agent/runs/20260705-vendor-ds4-coldstart/alternate-gguf-target-representation-hard-bound.json` with selected repo/file, exact size, expected I/O/page-cache behavior under the 16GB cgroup, TTFT estimate, VRAM plan, vendor loadability check, accuracy risk, France answer gate, five-prompt semantic gate, and rollback criteria.

Latest alternate GGUF target hard-bound:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/alternate-gguf-target-representation-hard-bound.json`.
- Result: `no_alternate_gguf_download_or_benchmark_allowed_now`; runtime source remains frozen and no full model download was performed.
- Current disk state is also a practical blocker for full-model tests: `/root` has only about `1.8GB` free, while the smallest screened full candidate is `50439361920` bytes. Do not delete SOTA run evidence. Freeing space must not remove accepted SOTA reproduction artifacts.
- Vendor CUDA build supports relevant low-bit tensor kernels (`Q2_K`, `IQ2_XXS`, `IQ1_M`) and was built with `GGML_CUDA=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_MOE_STREAM=ON`.
- Header probes:
  - tarruda Q2_K first split shard is valid `general.architecture=deepseek4`, `split.count=3`, `expert_count=256`, but its direct size-only optimistic bound is only `5.739 tok/s`.
  - 0xSero Spark-Mini Q2 REAP partial header is valid `general.architecture=deepseek4`, `block_count=43`, `expert_count=144`, `expert_used_count=6`, `n_tensors=1328`, with expert tensors using `IQ2_XXS` gate/up and `Q2_K` down.
- Size-only optimistic bounds are below target for all screened candidates: tarruda Q2_K `5.739 tok/s`, antirez IQ2XXS `6.059 tok/s`, teamblobfish IQ1_M `6.889 tok/s`, sleepyeldrazi REAP-K128 Q2 `7.545 tok/s`, 0xSero Spark-Mini Q2 REAP `7.436 tok/s`.
- The strongest concrete VRAM-placement bound is still below target. For 0xSero Spark-Mini Q2 REAP, expert payload is `1019215872` bytes/layer and non-expert payload is about `8767249504` bytes. With 32GB RTX 5090 VRAM, optimistic GPU MoE placement leaves too many CPU MoE layers: zero reserve projects `24` GPU MoE layers / `19` CPU MoE layers and `9.334 tok/s`; 3GiB reserve projects `21` GPU MoE layers / `22` CPU MoE layers and `9.069 tok/s`.
- To clear `10 tok/s` under the same linear fallback model, this candidate would need about `33` GPU MoE layers, which requires roughly `39.49GiB` VRAM at zero reserve and `42.49GiB` with 3GiB reserve. The current machine has `32109 MiB`.
- Decision: do not download or benchmark current public lower-bit GGUF candidates for SOTA. Reopen this route only if a candidate proves enough compression/placement to keep CPU MoE fallback below the `10 tok/s` bound on this 32GB GPU, or if a full strict-cold empirical test is explicitly prepared after freeing disk without deleting accepted SOTA evidence.

Latest external MTP sidecar follow-up:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/external-mtp-sidecar-followup-audit.json`.
- Result: `no_vendor_loadable_high_acceptance_sidecar_found`; runtime source remains frozen and no model download was performed.
- `FoxlightAI/deepseek-v4-flash-mtp` is a `mtp.safetensors` sidecar (`6777733547` bytes), tagged `mtp-sidecar`/`skulk`, not a GGUF artifact and not a standalone draft model for `llama-speculative`.
- `canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`, `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`, and `LordNeel/DeepSeek-V4-Flash-Acti-MTP-W4A16-FP8` are large safetensors/vLLM artifacts; they do not provide a current vendor GGUF loader path.
- `inferencerlabs/DeepSeek-V4-Flash-MTP-MLX` and `mlx-community/DeepSeek-V4-Flash-MTP-bf16` are MLX/safetensors MTP artifacts, not current vendor GGUF.
- Hugging Face searches for `DeepSeek V4 Flash MTP sidecar GGUF`, `DeepSeek V4 Flash draft GGUF`, and `DeepSeek-V4-Flash MTP skulk` returned no vendor-loadable GGUF draft/sidecar candidate.
- Decision: do not implement an MTP/sidecar loader now. Reopen only if a concrete artifact is vendor-loadable or has a conversion plan with hard-bound acceptance, target verification cost, RAM/page-cache, VRAM, TTFT, and fixed-text/France correctness evidence before source changes.

Latest current-route and disk audit:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/current-route-and-disk-audit-after-mtp-sidecar.json`.
- Result: `no_source_ready_10tok_route_currently_open`; runtime source remains frozen and no files were deleted.
- Current accepted route uses `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, so that pack is part of the accepted SOTA reproduction surface and must not be removed.
- Current SOTA memory is fully at the strict limit: `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `memory.events.max=16879`, `pgmajfault=272731`, and correctness passed for the France prompt.
- Current trace split still points to source/page exposure plus CPU up/down fallback as the only large bucket: accepted decode window `31047.44671 ms`, source/page exposed saving `15856.166 ms`, hot fallback after source/page touch `3173.291 ms`, non-fallback decode `12017.99 ms`.
- Source/page-only cannot reach 10 tok/s (`8.993 tok/s` zero-overhead). Source/page plus measured CPU microprobe still cannot reach 10 tok/s (`9.750 tok/s`, `350.127 ms` short before overhead). Therefore a runtime patch using the current CPU microkernel family or same-op source overlap is not allowed.
- Disk audit: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/.cache/...incomplete` is a `2965889623` byte incomplete HF cache fragment; deleting it would free only about `2.96 GB`. Large reclaim candidates such as `GLM-5.2-UD-IQ3_XXS` (`263 GB`) or non-SOTA expert packs require explicit preservation/deletion policy because they may be user assets or historical evidence.
- Decision: immediate code work should stay frozen. The next meaningful progress requires one of: a new external vendor-loadable high-acceptance artifact, an exact kernel/representation proof above the already measured thresholds, or explicit disk-reclamation approval that preserves accepted SOTA evidence before downloading/testing alternate artifacts.

Latest no-source thread-count sweep:

- Plan artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/thread-count-sweep-plan.json`.
- Result artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/thread-count-sweep-result.json`.
- Run root: `/root/lfz/runs/vendor-ds4-16gb/20260705T115301Z-thread-count-sweep-current-head`.
- Constant config: vendor strict-cold France prompt, `cpu_moe=40`, `vram_cache=0`, accepted France gate-miss O_DIRECT pack, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_CUDA_DISABLE_GRAPHS=1`, `drop_caches` before each case, 16GB cgroup with `MemorySwapMax=0`.
- Results: `t=24` reached `eval_tok_s=4.4`, `prompt_tok_s=1.9`, `TTFT=31667.216978 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15101898752`, correctness passed, but it only tied SOTA and is not promotable. `t=32` reached `2.6 tok/s`, `t=40` reached `2.1 tok/s`, and `t=60` reached `1.8 tok/s` with `TTFT=34401.120694 ms`, over the gate.
- Gate-pack and VRAM-cache counters stayed aligned across cases (`pack hits=4886 misses=0`, VRAM cache `hits=33265 misses=1886 hit_rate=94.6%`), so the regressions are not from cache-admission drift. They are consistent with CPU fallback scheduling/memory-bandwidth contention.
- Decision: thread-count tuning does not reopen the 10 tok/s route and should not be repeated on this same accepted route unless a source/kernel change changes CPU fallback scaling.

Latest external artifact refresh after thread-count closure:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/external-artifact-refresh-4expert-ssd-dflash.json`.
- Result: `no_external_artifact_source_patch_allowed_now`; no full model download, no runtime source change, no file deletion, and accepted SOTA remains `4.4 tok/s`.
- Search scope: Hugging Face `DeepSeek-V4-Flash` sorted by latest update plus targeted repo API/README checks for `cloudyu/DeepSeek-V4-Flash-4Expert-GGUF`, `teto3/DeepSeek-V4-Flash-Base-Q4KExperts`, `servantofares/DeepSeek-V4-Flash-FP4-FP8-SSD`, `RedHatAI/DeepSeek-V4-Flash-speculator.dflash`, `inference-optimization` DFlash, `ManiacLabs/DeepSeek-V4-Flash-EAGLE3.1`, `eadx/Huihui-...-ds4-GGUF`, `audreyt/CyberNeurova-...-GGUF`, `bullerwins/DeepSeek-V4-Flash-GGUF`, and `lvyufeng/DeepSeek-V4-Flash-IQ1_M`.
- `cloudyu` 4Expert partial GGUF header confirms `general.architecture=deepseek4`, `expert_used_count=4`, Q4_K routed experts, and `4.5 MiB` per expert-role versus current native `4.25 MiB`. Current vendor has two source-readiness gaps: routing tensor name is `blk.*.ffn_gate_tid2eid.weight` while vendor probes `blk.*.ffn_gate_tid2eid`, and `ggml_cuda_moe_stream_one` does not allow Q4_K in the accepted stream/cache path. Even granting those patches plus perfect source/page elimination, the hard-bound is only `9.581 tok/s`, still `597 ms` short before integration overhead.
- `servantofares` SSD Flash-MoE sidecar is dense GGUF plus 43 routed-expert layer banks and requires `--moe-mode slot-bank` / `--moe-sidecar`, which current vendor accepted path does not expose. A perfect native SSD sidecar that only eliminates source/page stalls is bounded by `8.993 tok/s`, so slot-bank alone cannot reach 10.
- DFlash/EAGLE artifacts are not vendor-loadable GGUF drafts. `RedHatAI` DFlash is safetensors/custom vLLM support with up to 7 speculative tokens; `ManiacLabs` EAGLE reports vLLM B200:4 speedup but explicitly requires a patched vLLM overlay and no llama.cpp support. These can reopen only with a vendor verifier design proving acceptance, sublinear target verification, RAM/page-cache, VRAM, TTFT, fixed-text top1, and France correctness before coding.
- Recent low-bit/abliterated GGUF candidates (`eadx`, `audreyt`, `bullerwins`, `lvyufeng`) are either same size/type class as already bounded artifacts or larger than the prior strongest 0xSero `52.6GB` REAP candidate, which itself remained below 10 tok/s on this 32GB GPU.

Next active plan:

1. Keep the accepted `4.4 tok/s` strict-cold SOTA as the comparison baseline. Do not make a runtime/source patch before a new hard-bound artifact proves the route can clear the promotion gate.
2. Verifier/draft work is closed for current public artifacts, including the latest MTP sidecar follow-up and the refreshed DFlash/EAGLE artifacts. Reopen it only with a concrete vendor-compatible DSpark/MTP/DFlash/EAGLE/draft artifact and a hard-bound proving `A>=7`, or `A>=5` with measured sublinear target verification, while preserving the 16GB page-cache budget, accepted gate-cache behavior, TTFT limit, fixed-text top1, and France correctness.
3. External lower-bit/4Expert/sidecar target work is closed for the currently screened public candidates. Reopen only with a new hard-bound that clears `10 tok/s` on the current 32GB GPU with integration margin, or with a deliberately prepared empirical test after freeing enough disk space without deleting accepted SOTA reproduction evidence. Any alternate model/quantization run must still pass fixed-text, France, five-prompt semantic correctness, strict 16GB cgroup, and TTFT gates before it can be mentioned as SOTA.
4. Source/page overlap plus CPU fallback batching is closed for current evidence. No-source thread-count tuning is also closed for the accepted route. Reopen only if a new exact microkernel/probe exceeds the `90.643 GiB/s` zero-overhead threshold with enough integration margin, or if source/page overlap is proven in a prompt-general way without same-op regression and without violating the 16GB page-cache limit.
5. Non-native representation work inside the current native GGUF is closed for current artifacts. Reopen it only through an offline or compare-only artifact proving fixed-text top1 plus `>5.33x` effective payload reduction, or equivalent source+CPU fallback reduction, with VRAM/RAM/TTFT and overhead accounting before source changes.
6. Exact graph/dataflow work is closed for current artifacts. Reopen it only with a new placement/kernel proof that shows retained CUDA gate/up/down, hidden scheduler-copy counts, fixed-text top1, and enough margin above the current top64/top128/top768/full-MoE hard bounds before source changes.
7. Disk cleanup is not an optimization result and cannot be used as SOTA evidence. If disk must be reclaimed for alternate-artifact tests, preserve at minimum the accepted SOTA run, current-head guard run, current source, profiles, and the accepted France gate-miss expert pack before deleting any large run/model artifact.
8. The remaining route classes are: a new external high-acceptance verifier/draft artifact, a stronger correctness-verified alternate GGUF/representation artifact than the candidates screened here, or a genuinely new exact graph/dataflow proof satisfying item 6. Any of these must start from a hard-bound artifact under `.Agent/runs/20260705-vendor-ds4-coldstart/`.
9. First gate for any default-off source probe or alternate target run is fixed-text `llama-results` top1 under strict 16GB/no-swap cgroup, plus confirmation that the default accepted path is unchanged. Strict cold SOTA benchmarking is allowed only after correctness, RAM, TTFT, and default-path preservation pass.
10. If a compliant new SOTA appears, immediately record full reproduction metadata and push source plus artifacts to `ssd/vendor/deepseek-token-rate-16gb`. Required metadata: source commit, pushed remote branch, full env/CLI, run path, build command, binary hash if available, model path and size, profile/manifest hashes, token rates, TTFT, elapsed time, full France answer, cgroup `memory.peak`, `memory.current`, `memory.stat`, `memory.events`, page-cache bytes, cache/pack counters, and comparison to the previous `4.4 tok/s` SOTA. After push, do a clean pushed-source reproduction before treating it as accepted.

### 2026-07-05 Historical Plan: Forced-Batch, MTP-Union, and Exact-Compact Closure

本节已被上方最新生效计划覆盖；内容只作为历史实验记录保留。后续执行必须先在最新计划或 `.Agent/runs/20260705-vendor-ds4-coldstart/` 下写清硬性上界、正确性门槛、16GB page-cache 约束、TTFT 约束和完整复现信息，再做 runtime 改动或长跑。

Current accepted strict cold SOTA 仍然是 `4.4 tok/s`，不是 forced-batch 诊断结果，也不是任何 trace/warm/steady-state/top1-only 结果：

- Accepted SOTA run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Current-head no-trace guard: `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`
- Guard metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32087.738292 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`
- Source/runtime head before this documentation update: `b5d98651485d775b80e6b3f69e140f294824d6e9` (`vendor-ds4: reject mtp n2 verifier bound`)
- Push target for all future source/artifact updates: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- Git identity for future commits/pushes: `L-Ark <fliangae@connect.ust.hk>`
- Promotion gate remains strict: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, strict cold `drop_caches`, 16GB cgroup including file page cache, `MemorySwapMax=0`, no swap/OOM, France answer semantically correct and coherent, source plus artifacts committed and pushed, then clean pushed-source reproduction.

Latest forced-batch target-verifier diagnostic:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/forced-batch-target-verifier-diagnostic.json`.
- Purpose: coarse check whether current vendor fixed-text target evaluation shows sublinear multi-token batch behavior that could support an `A>=5` DSpark/MTP verifier route.
- Valid run root: `/root/lfz/runs/vendor-ds4-16gb/20260705T092532Z-forced-batch-verifier-cost-v2`.
- Aborted v1 root: `/root/lfz/runs/vendor-ds4-16gb/20260705T091814Z-forced-batch-verifier-cost`; this is rejected because the first script mistakenly hashed the full `145.42 GiB` model file before running.
- Shared config: `llama-results`, fixed France text, `-c 256 -b 16 -ub 16 -t 20 -tb 20 -ngl all --fit on -fa auto --n-cpu-moe 40 --defer-experts`, accepted gate-only one-stream pack/cache env, strict `drop_caches`, `MemoryMax=16000000000`, `MemorySwapMax=0`.
- Sequential mode: `--sequential-logits`, `elapsed_seconds=107.623408161`, `n_tokens=145`, `top1_matches_next_token=119`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14842507264`, `oom=0`, `oom_kill=0`, `swaps=0`.
- Batch16 mode: default `llama-results` chunking with `-b 16 -ub 16`, `elapsed_seconds=107.597260262`, `n_tokens=145`, `top1_matches_next_token=119`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14834040832`, `oom=0`, `oom_kill=0`, `swaps=0`.
- Comparison: `seq_over_batch16_speedup=1.0002430164014988`; batch16 is only `0.026147899 s` faster over the whole fixed-text run, i.e. no measurable fixed-text batch speedup at this coarse level.
- Limitation: this is not an MTP verifier implementation. It includes model/context startup, cache prefill, expert pack IO, and result writing, so it is a coarse negative/proxy diagnostic rather than a per-verifier-pass measurement. It cannot be promoted as SOTA.

Decision after this diagnostic:

- Current vendor target fixed-text path does not demonstrate the sublinear multi-token verifier cost needed to justify DSpark/MTP source work.
- Antirez exact N=2 MTP remains closed for the `10 tok/s` objective: at most 3 emitted tokens per target verification pass require `C_verify + C_draft <= 1.32x` one current token pass (`<=300 ms`), while decode CPU up/down fallback alone is about `139.9 ms/token`.
- DSpark/MTP can only reopen if a new artifact first proves `A>=5` average acceptance with target verification that avoids per-token CPU up/down fallback and fully accounts for the extra model/module bytes, VRAM, 16GB host/page-cache, TTFT, and correctness.
- The next viable optimization direction is a new hard-bound for an exact compact representation or other exact algorithm that reduces both source bytes and CPU up/down compute by construction. It must show enough margin over the accepted `4.4 tok/s` baseline before any runtime patch.

Latest MTP/source-union amortization bound:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/mtp-union-source-amortization-bound.json`.
- Input trace: `/root/lfz/fastllm_runs/strict16g-sota-wrapper-fulltrace-repeat1-20260622/repeat1/gate/route.trace.repeat1.tsv`, `sha256=0a04030521140cf645f91ea4c209cf70add463d21e82264a7b263423c6dbf76e`.
- Estimator: `/root/lfz/fastllm/tools/scripts/deepseek_v4_flash_mtp_union_estimator.py`, `sha256=76e5a15cf1496815919aacc21b914b1a08750cb4a7c982aa70fb113d0faf078e`.
- This is a route-trace source-byte union estimate, not draft acceptance evidence and not a vendor target-verifier benchmark.
- Expert-source amortization is weak relative to ideal multi-token batching: `W=5` gives only `1.387908x`, `W=6` gives `1.461946x`, `W=8` gives `1.661718x`, `W=16` gives `2.439050x`.
- With current SOTA numbers (`accepted_decode_window_ms=31047.44671`, `decoded_tokens=136.608765524`, `decode_source_page_delta_ms=16233.137`, `decode_fallback_ms=19115.399`, `non_fallback_decode_ms=11932.04771`), `W=5` is still only `8.454 tok/s` even under an unrealistically favorable bound where all non-MoE decode cost batches perfectly. `W=6` is still `9.069 tok/s` under the same unrealistic assumption.
- `W=8` reaches `10.512 tok/s` only in that extreme no-overhead lower bound, with just `665.975 ms` margin. There is no current vendor DSpark/MTP verifier, no compatible local DSpark/MTP artifact, and the forced-batch diagnostic showed `1.00024x` speedup, so this does not justify a runtime patch.
- Decision: source-union estimates do not reopen DSpark/MTP or multi-token verifier implementation. DSpark/MTP can only reopen with a concrete `A>=7` or `A>=5` verifier artifact that proves accepted-token rate, sublinear target verification, RAM/VRAM/TTFT accounting, and fixed-text/France correctness before coding.

Latest exact compact representation closure:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/exact-compact-representation-closure-bound.json`.
- The first positive raw exact hotset bound is top4096: `payload_gib=17.0`, `zero_overhead_tok_s=10.797`, `slack_to_10_ms=1008.842`. Top3072 is not enough: `payload_gib=12.75`, `zero_overhead_tok_s=9.946`, `slack_to_10_ms=-73.955`.
- Current measured extra pool preserving the accepted gate path is only about `3.188 GiB`; therefore top3072 would need about `4.0x` exact reduction and top4096 about `5.33x` exact reduction.
- Prior full top4096 zstd level 1 ratio was only `1.040149`; top512 prefix zstd level 10 was `1.040140`.
- New deterministic entropy sample read `256` evenly spaced top4096 entries and `4096` native MXFP4 blocks per entry: `1,048,576` sampled blocks, `1,048,576` unique blocks, `0` duplicate blocks, `0` zero blocks.
- Entropy result: byte entropy `7.658319` bits gives ideal byte lossless ratio `1.044616x`; independent native-block entropy gives only `1.085041x`. This is far below the `4.0x-5.33x` ratio required to make the top3072/top4096 hotsets fit.
- Decision: close exact compact representation for the current native MXFP4 payload. Do not implement lossless compression, dedup, dequantized resident, or raw top4096 residency runtime paths. Lower-bit/codebook/low-rank/sparsified/MMVQ-like paths are not exact representations and can only reopen with fixed-text top1/France correctness proof before performance; the prior MMVQ write probe already failed top1.

Next execution plan:

1. Keep the accepted `4.4 tok/s` strict-cold SOTA as the only baseline for comparison.
2. Do not repeat closed routes without new math: source/page-only prefetch or io_uring, CPU batch rewrite, extra full GPU MoE layer, rectangular `DS4_HOT_DISPATCH`, direct top768 Q8_0, raw/transposed/row-tile exact hot-batch kernels, CUDA graph wrapping, standalone MMVQ skip/write, current sparse retained top64 graph, transient MXFP4 repack, no-source lookahead/ngram speculation, antirez exact N=2 MTP, native MXFP4 lossless compression/dedup, and raw top4096 residency.
3. Before any new source edit, write a hard-bound artifact under `.Agent/runs/20260705-vendor-ds4-coldstart/` that includes exact bytes, expected saved milliseconds, kernel/transfer/sync/scatter overhead, VRAM footprint, host RAM/page-cache footprint, TTFT impact, correctness verifier, rollback criteria, full env/CLI, and expected token-rate ceiling.
4. First acceptable implementation candidate must either:
   - prove a compatible `A>=7` or otherwise high-acceptance DSpark/MTP verifier with sublinear target verification and strict rollback/commit semantics; or
   - prove a new non-native representation or algorithm with measured `>5.33x` effective payload reduction, fixed-text top1 proof, France correctness, and decode overhead below the top4096 `1008.842 ms` slack.
5. First gate for any default-off source probe is fixed-text `llama-results` top1 under strict 16GB/no-swap cgroup, plus confirmation that the default accepted path is unchanged. Strict cold SOTA benchmarking is allowed only after correctness, RAM, TTFT, and default-path preservation pass.
6. If a compliant new SOTA appears, immediately record full reproduction metadata and push source plus artifacts to `ssd/vendor/deepseek-token-rate-16gb`. Required metadata: source commit, pushed remote branch, full env/CLI, run path, build command, binary hash if available, model path and size, profile/manifest hashes, token rates, TTFT, elapsed time, full France answer, cgroup `memory.peak`, `memory.current`, `memory.stat`, `memory.events`, page-cache bytes, cache/pack counters, and comparison to the previous `4.4 tok/s` SOTA. After push, do a clean pushed-source reproduction before treating it as accepted.

### 2026-07-05 Historical Plan: Post-MMVQ Non-Duplicate Gate

本节已被上方最新生效计划覆盖；内容只作为历史实验记录保留。后续不能从已关闭路线直接继续写 runtime 代码，必须先在最新计划或 `.Agent/runs/20260705-vendor-ds4-coldstart/` 下写清新的硬性上界、正确性门槛和复现信息。

Current accepted strict cold SOTA 仍然是 `4.4 tok/s`：

- Accepted SOTA run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Current-head no-trace guard: `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`
- Guard metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32087.738292 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`
- Latest pushed head before this plan update: `1c63249c0818a485c36e3aa307311c8579a2cca8` (`vendor-ds4: reject mmvq hot skip top1`) on local branch `feat/ds4-moe-stream-on-vendor`, pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- Git identity for future pushes: `L-Ark <fliangae@connect.ust.hk>`
- Promotion gate remains strict: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, strict cold `drop_caches`, 16GB cgroup including file page cache, `MemorySwapMax=0`, no swap/OOM, France answer semantically correct and coherent, source plus artifacts committed and pushed, then clean pushed-source reproduction.

Latest rejected/closed routes that must not be repeated without a new hard-bound:

- `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1` strict-cold recheck tied SOTA only: `eval_tok_s=4.4`, `TTFT=33269.766728 ms`, RAM/correctness OK, decision `rejected_tie_not_new_sota`. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-lightning-indexer-recheck-result.json`.
- Current-head transient MXFP4 repack is closed by bound: zero-overhead all up/down route only `~4.58 tok/s`, far below the `10 tok/s` target. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/transient-repack-current-head-bound.json`.
- Standalone MMVQ hot-resident skip/write is rejected before strict cold benchmark. Fixed-text top1 failed with `same_top1=142/145`, `first_mismatch_pos=9`, `max_abs=4.79565`, and the probe's 544 MiB direct pool broke the accepted gate VRAM cache allocation. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/mmvq-hot-batch-skip-top1-rejected.json`.
- The dirty MMVQ runtime source probe was reverted; `llama-cli` and `llama-results` were rebuilt from clean source. No logit-changing source from that probe is retained.
- Already closed categories remain closed: sparse retained top64 on the current graph, full decode up/down CUDA/streaming, source/page-only prefetch or io_uring, CPU batch rewrite, extra full GPU MoE layer, rectangular `DS4_HOT_DISPATCH`, direct top768 Q8_0, raw/transposed/row-tile exact hot-batch kernels, CUDA graph wrapping, and no-source lookahead/ngram speculation.

Current bottleneck and target math:

- Accepted decode window is about `31047.447 ms`; the `10 tok/s` target requires saving about `17386.570 ms` while keeping total added overhead below about `1.6-1.7 s`.
- Decode CPU up/down fallback plus source/page exposure is still the only measured component large enough to matter. Gate-only, scheduler-only, source-only, and hot-compute-only routes do not have enough zero-overhead ceiling.
- Any future source edit must reduce source bytes and exact compute by construction, or introduce a correctness-verified algorithmic path. A top1-only diagnostic, trace run, warm page-cache run, or steady-state run cannot be promoted as cold SOTA.

Next execution plan:

1. Write a current-head non-duplicate mechanism screening artifact before any runtime patch. Required path: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-nonduplicate-next-screening-after-mmvq.json`.
2. The screening must explicitly answer whether current vendor DeepSeek code already has any usable path for:
   - compatible draft/MTP/NextN verified speculation;
   - no-source verifier/speculation that preserves the greedy token stream;
   - compact resident representation that can keep enough exact up/down work on GPU without violating VRAM or 16GB host/page-cache limits;
   - predictive async source overlap that happens before the consuming layer/op rather than repeating same-op touch/prefetch.
3. If the screening finds no existing mechanism with a hard-bound above `10 tok/s`, freeze runtime source changes and update this plan with the blocker: reaching 10 tok/s then requires either a compatible external draft/MTP artifact plus verifier integration, or a new exact compact representation design with measured bandwidth/VRAM/RAM proof.
4. If a credible mechanism is found, write a separate hard-bound artifact before coding. That artifact must include exact bytes, expected saved milliseconds, kernel/transfer/sync/scatter overhead, VRAM footprint, host RAM/page-cache footprint, TTFT impact, correctness verifier, rollback criteria, full env/CLI, and the expected token-rate ceiling.
5. Only after the hard-bound shows margin should a default-off source probe be implemented. The first gate is fixed-text `llama-results` top1 under strict 16GB/no-swap cgroup. Strict cold SOTA benchmarking is allowed only after correctness, RAM, TTFT, and default-path preservation pass.
6. If a compliant new SOTA appears, immediately record full reproduction metadata and push source plus artifacts to `ssd/vendor/deepseek-token-rate-16gb`. Required metadata: source commit, pushed remote branch, full env/CLI, run path, build command, binary hash if available, model path and size, profile/manifest hashes, token rates, TTFT, elapsed time, full France answer, cgroup `memory.peak`, `memory.current`, `memory.stat`, `memory.events`, page-cache bytes, cache/pack counters, and comparison to the previous `4.4 tok/s` SOTA. After push, do a clean pushed-source reproduction before treating it as accepted.

Current-head non-duplicate screening result:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-nonduplicate-next-screening-after-mmvq.json`.
- Source head: `745c2f5894ea679863969147124fe00a3cd69b36`; runtime-relevant code head remains `1c63249c0818a485c36e3aa307311c8579a2cca8` because `745c2f589` only changed the plan.
- Runtime source modified: no. Model run performed: no. This is a source/artifact screening gate before any runtime patch.
- Result: no current vendor DeepSeek mechanism with a credible `10 tok/s` hard-bound was found after the MMVQ rejection.
- Compatible draft/MTP/NextN verified speculation: no local compatible DeepSeek draft/MTP GGUF found; current source preserves/loads some NextN/MTP metadata but does not implement a DeepSeek4 target-verification path.
- No-source speculation/verifier: lookahead/ngram/lookup code exists, but historical DeepSeek4 runs diverged, replayed/stalled, accepted too few tokens, or performed below SOTA. `llama-results` top1 is only a correctness verifier and gives no token-rate gain by itself.
- Compact resident exact GPU up/down: no existing mechanism. Current `DS4_HOT_DISPATCH` is rectangular/gate-recompute based; sparse retained probe showed no retained graph-level gate tensor available to `build_expert_mix`, and existing Q8_0/MMVQ/raw/transposed/row-tile probes failed correctness or bandwidth bounds.
- Predictive source overlap: no existing mechanism. Existing backend MoE cache/prefetch and same-op source staging do not remove enough decode CPU up/down fallback; source-only overlap remains below the `10 tok/s` ceiling under strict cold 16GB including page cache.
- Decision: runtime patches are frozen until a new hard-bound artifact is written. The next viable 10 tok/s route must be either a compatible DeepSeek draft/MTP artifact plus verifier integration, or a new exact compact representation design with measured source bytes, compute bandwidth, VRAM, host RAM/page-cache, TTFT, and fixed-text top1 proof.

External DSpark/MTP current audit:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/external-dspark-mtp-current-audit.json`.
- Official `deepseek-ai/DeepSeek-V4-Flash-DSpark` exists and was last modified `2026-07-04T03:15:12.000Z`, but it is a safetensors + custom Python/PyTorch inference path, not a current vendor GGUF cold-start path. Its config has `num_nextn_predict_layers=1`, `dspark_block_size=5`, `dspark_target_layer_ids=[40,41,42]`, and `dspark_markov_rank=256`.
- `antirez/deepseek-v4-gguf` contains `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`, size `3807602400` bytes. A 16 MiB range parse of the GGUF header showed `general.architecture=deepseek4_mtp_support`, `tensor_count=32`, `deepseek4.mtp_layer_count=1`, `deepseek4.nextn_predict_layers=1`, and `deepseek4.expert_count=256`.
- Current vendor source has no support for `general.architecture=deepseek4_mtp_support`, `deepseek4.mtp_layer_count`, or `mtp.0.*` tensor runtime handling. Therefore this MTP file is not directly loadable or benchmarkable as a compliant SOTA candidate.
- Decision: external MTP is now the most concrete future 10 tok/s candidate class, but no runtime patch/download/benchmark is allowed yet. Next required artifact is `.Agent/runs/20260705-vendor-ds4-coldstart/deepseek4-mtp-loader-verifier-hard-bound-plan.json`, covering loader scope, verifier semantics, expected accepted-token speedup, 3.807 GB file/page-cache impact under the 16GB cgroup, VRAM/gate-cache preservation, TTFT, and fixed-text top1/France correctness gates.

DeepSeek4 MTP loader/verifier hard-bound plan:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/deepseek4-mtp-loader-verifier-hard-bound-plan.json`.
- Theory: with current `4.4 tok/s`, speculative/MTP only reaches `10 tok/s` if `tok_s = 4.4 * (A + 1) / (C_verify + C_draft)` exceeds 10. Ideal minimum average accepted draft tokens is `A >= 1.272727`, but this assumes one target verification pass is not linear in the number of verified tokens.
- For `A=5` (official `dspark_block_size`) the total target-verification plus draft cost must be `<=2.64x` one current target-token pass. For `A=7` (official vLLM example `num_speculative_tokens`) it must be `<=3.52x`.
- If target verification still invokes the expensive CPU up/down fallback once per verified token, the route cannot improve token rate; proving sublinear verification cost is the first hard gate.
- Current blockers before source edit: no `deepseek4_mtp_support` loader, no `mtp.0.*` runtime mapping, no target hidden export from layers `[40,41,42]`, no KV/cache commit-rollback verifier, no RAM/VRAM/TTFT proof for the extra `3807602400` byte MTP GGUF.
- Full MTP preload is currently rejected on TTFT math: `3.546 GiB / 1.85 GiB/s ~= 1.917s`, which exceeds the current `1.530s` TTFT slack versus the `33617.688744 ms` gate.
- Decision: route is `open_but_not_source_ready`. No runtime patch, full download, or strict-cold benchmark is allowed yet. Next required artifact is `.Agent/runs/20260705-vendor-ds4-coldstart/deepseek4-mtp-tensor-mapping-and-verify-design.json`.

DeepSeek4 MTP tensor mapping and verifier design:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/deepseek4-mtp-tensor-mapping-and-verify-design.json`.
- The antirez MTP GGUF has 32 tensors and maps mostly onto a supplementary DeepSeek4 layer: its HC, attention, MoE, and shared-expert tensors are structurally close to existing vendor DeepSeek4 fields, while `e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`, and separate `hc_head_*` require new supplementary MTP fields.
- This GGUF is not identical to official DSpark safetensors: the GGUF does not contain `markov_w1`, `markov_w2`, or `confidence_head` tensors. Treat it as the antirez one-layer MTP support model, not as the official DSpark module.
- Reference source `antirez/ds4` binds the same `mtp.0.*` tensors, keeps a separate MTP raw cache, and has an exact N=2 verifier path. Its README says the current MTP/speculative path is experimental, correctness-gated, and provides at most slight speedup.
- Decision: still no runtime patch. The next required artifact is `.Agent/runs/20260705-vendor-ds4-coldstart/deepseek4-mtp-n2-verifier-cost-bound.json`. If exact N=2 cannot show a meaningful hard-bound under strict 16GB RAM, TTFT, and correctness gates, close MTP for the 10 tok/s objective rather than implementing loader code.

DeepSeek4 MTP N=2 verifier cost bound:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/deepseek4-mtp-n2-verifier-cost-bound.json`.
- Exact N=2 can emit at most 3 tokens per target verification pass. At current `4.4 tok/s`, even with perfect two-token acceptance, `C_verify + C_draft` must be `<=1.32x` one current target-token pass, i.e. `<=300 ms`; this leaves only `72.7 ms` over the current one-token pass.
- Current decode CPU up/down fallback alone averages about `139.9 ms/token`. Unless target verification removes or batches that fallback far beyond any currently proven vendor path, the second verified token consumes more than the entire N=2 extra budget.
- Because MTP draft has nonzero layer/output/cache/file cost, and antirez documents the reference MTP path as only a slight speedup, exact N=2 MTP has no credible `10 tok/s` hard-bound under the current 16GB/TTFT/correctness constraints.
- Decision: close antirez exact N=2 MTP as a 10 tok/s route. Do not implement MTP loader/verifier for the 10 tok/s objective unless a new MTP/DSpark mechanism proves `A>=5` with sublinear target verification and full RAM/VRAM/TTFT/correctness accounting.

### 2026-07-05 Historical Plan: Current Head After Full Up/Down Bound

本节已被上方最新生效计划覆盖；内容只作为历史实验记录保留。

Current accepted strict cold SOTA 仍然是 `4.4 tok/s`，不是 `4.2`，也不是任何 trace/diagnostic run：

- Accepted SOTA run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Accepted SOTA metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Current-head guard after the latest rejected-route work: `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`, with `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32087.738292 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`
- Promotion gate: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, strict 16GB cgroup including file page cache, `MemorySwapMax=0`, no swap/OOM, France answer semantically correct/coherent, source plus artifacts committed and pushed, then clean pushed-source reproduction
- Model file: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, size `156148189760` bytes (`145.42 GiB`)
- Pushed source at the start of the lightning strict-cold benchmark: `4e8a6fcde6f4d51c2b740886f008032deeb2eb31` on local branch `feat/ds4-moe-stream-on-vendor`, pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`

Closed routes and constraints from the latest hard bounds:

- Sparse retained top64 route is closed on the current graph. The probe showed no retained graph-level gate output available to `build_expert_mix`, no active hot manager path in the accepted route, and a final hot/cold combine bound that leaves too little margin.
- Full decode up/down CUDA/streaming route is closed before runtime source edit. Decode up/down unique payload is `21.806 GiB`, call-weighted payload is `148.916 GiB`, exact GPU raw/transposed kernel projections exceed the full fallback-removal margin, and full resident payload violates VRAM/16GB page-cache constraints.
- Source/page-only, io_uring/source-only prefetch, CPU batch rewrite, extra full GPU MoE layer, current `DS4_HOT_DISPATCH` rectangular shapes, direct top768 Q8_0, raw/transposed/row-tile exact hot-batch kernels, standalone MMVQ hot-resident skip/write, and CUDA graph wrapping remain rejected unless a new hard-bound changes the limiting math.
- Current-head transient MXFP4 repack bound is also closed. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/transient-repack-current-head-bound.json`. Updated zero-overhead bound gives only `4.58 tok/s` (`~1.2s` estimated saving) versus the `~17.39s` saving required for 10 tok/s, and the historical strict-cold transient-down candidate already regressed to `3.4 tok/s`.
- The measured bottleneck is still decode CPU up/down fallback plus source/page behavior. Accepted decode window is about `31047.447 ms`; reaching `10 tok/s` needs about `17386.570 ms` saving while keeping total added overhead below roughly `1.6-1.7 s`.

Immediate one-time candidate screen:

- Candidate: `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1`.
- Plan artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-lightning-indexer-recheck-plan.json`.
- This is not the primary 10 tok/s route and does not remove CPU up/down fallback. It is allowed once because it is an existing default-off env, previous repeat once reached `4.5 tok/s` but failed pushed-source promotion, and current-head guard has more TTFT slack.
- Correctness precheck already passed under strict 16GB/no-swap cgroup: `/root/lfz/runs/vendor-ds4-16gb/20260705T072454Z-current-head-lightning-recheck/top1-lightning`, `same_top1=145/145`, `first_mismatch_pos=-1`, `memory_peak_bytes=16000000000`, `oom=0`, `oom_kill=0`.
- Strict cold result: `/root/lfz/runs/vendor-ds4-16gb/20260705T073651Z-20260705_current_head_lightning_recheck/france-lightning-current-head-cpu40-vram0gb`, `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=33269.766728 ms`, `elapsed_seconds=64.12`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15104135168`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`.
- Decision: `rejected_tie_not_new_sota`. This candidate passed correctness/RAM/TTFT, but it tied the accepted `4.4 tok/s` SOTA and failed the strict `eval_tok_s > 4.4` promotion gate. Stop sampling `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1` for SOTA.
- Result artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-lightning-indexer-recheck-result.json`.

Next optimization direction after the lightning candidate:

1. If lightning fails or only ties, restart from the current-head `4.4 tok/s` guard and run a fresh bottleneck split only when it answers a new question. Do not treat trace-overhead runs as SOTA candidates.
2. Any new runtime source edit must first have a hard-bound artifact that shows how it can save enough of the `~19.1 s` decode CPU up/down fallback while respecting strict cold 16GB RAM including page cache and `TTFT <= 33617.688744 ms`.
3. The next plausible 10 tok/s route must reduce both source bytes and exact compute by construction. Valid directions are: a new compact graph route only after proving a retained CUDA gate/hidden tensor and bounded final combine cost with meaningful margin; or a new exact up/down algorithm/layout that materially exceeds the measured `72-75 GiB/s` exact kernels and accounts for H2D/D2H/scatter/sync overhead.
4. Do not promote steady-state, warm page-cache, diagnostic trace, or top1-only results as cold SOTA. Cold SOTA must start after `drop_caches` inside the strict runner and must keep total cgroup memory including page cache under `16000000000` bytes.
5. Every compliant new SOTA must be documented in detail and pushed immediately to `ssd/vendor/deepseek-token-rate-16gb`. Required reproduction metadata: source commit, pushed remote branch, full env/CLI, run path, build command, model path and size, profile/manifest hashes, token rates, TTFT, elapsed time, full France answer, cgroup `memory.peak`, `memory.current`, `memory.stat`, `memory.events`, page-cache bytes, pack/cache counters, and comparison to the previous `4.4 tok/s` SOTA.

Current allowed source probe:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/mmvq-standalone-small-sota-hard-bound.json`.
- Scope: this is not a 10 tok/s route. It is only a small accepted-SOTA candidate because top128 standalone MMVQ projects about `401.874 ms` net saving before scatter/scheduler overhead, i.e. about `4.458 tok/s` on paper.
- Result artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/mmvq-hot-batch-skip-top1-rejected.json`.
- Result: rejected before strict cold benchmark. The default-off dirty source probe built, but fixed-text top1 failed with `same_top1=142/145`, `first_mismatch_pos=9`, `max_abs=4.79565`, `mean_abs=0.146238`. It also allocated a 544 MiB direct pool and caused the accepted gate VRAM cache allocation (`13.2 GiB`) to fail, so the config did not preserve the accepted gate path.
- Source action: the uncommitted runtime source probe was reverted after recording, and `llama-cli`/`llama-results` were rebuilt from clean source. Do not run a strict cold SOTA benchmark for standalone MMVQ skip/write. Reopen only with a new correctness-preserving design and a VRAM allocation plan that preserves gate cache behavior.

### 2026-07-05 Historical Active Plan Override After Sparse-Retained Planner

This section has been superseded by `2026-07-05 Latest Plan: Forced-Batch and MTP-Union Bounds After MTP Closure` and is retained only as historical experiment record.

Current accepted strict cold SOTA is still `4.4 tok/s`:

- Accepted run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Required promotion gate: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, strict 16GB cgroup including file page cache, `MemorySwapMax=0`, no OOM/no swap, semantically correct and coherent France answer, source plus artifacts committed and pushed, then clean pushed-source reproduction.
- Model file for reproduction: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, size `156148189760` bytes (`145.42 GiB`).
- Latest pushed source before this plan update: `ad7207f77fec975d4c99b86adf62af68eb487762` on local branch `feat/ds4-moe-stream-on-vendor` and remote `ssd/vendor/deepseek-token-rate-16gb`.

Latest closed findings:

- The sparse top64 pair profile is generated and reproducible: `64` `(layer, expert)` up/down pairs, `128` tensor entries, `544.0 MiB` up/down payload, `33` active layers, max `4` pairs in one layer.
- The one-stream sparse graph/dataflow probe is rejected as an implementation route. It showed `34800` gate rows, `570163200` bytes F32 `src1` H2D, `285081600` bytes `dst` D2H, `34800` CPU scatter rows, and `0` retained GPU output / returned GPU handle / zero-transfer-ready rows. Current one-stream gate cache cannot provide a graph-level GPU tensor for sparse up/down.
- The retained-GPU planner rejects current `DS4_HOT_DISPATCH` graph shapes for this sparse profile. Ideal up/down sparse payload is `544.0 MiB`, but the current rectangular per-layer shape would need `295` retained entries for `64` real pairs, inflating up/down payload to `2507.5 MiB` and gate/up/down payload to `3761.25 MiB`.
- Gate recompute remains rejected: the hard bound is `9.861 tok/s`, below the 10 tok/s target. The only still-viable graph route must reuse an existing gate result or equivalent retained hidden tensor, not recompute gate.
- Hot/cold final combine is a tight lower-bound risk. One backend crossing of `[n_embd, P, T]` for `n_embd=4096`, `P=6`, `33` active sparse layers, and `136.609` decoded-token estimate is about `0.413 GiB`; at `24 GiB/s` this costs `17.197 ms`, leaving only `1.750 ms` of the `18.947 ms` graph margin.

Immediate next work before this probe:

1. Do not run a strict cold performance/SOTA benchmark from the sparse profile alone.
2. Implement only a default-off DS4 graph placement/payload/copy-count probe in `src/models/deepseek4.cpp::build_expert_mix`. Proposed envs: `DS4_SPARSE_RETAINED_GRAPH_PROBE_OUT=<csv>` and, if needed, `DS4_SPARSE_PAIR_PROFILE_JSON=.Agent/profiles/vendor-ds4/current_sota_sparse_pair_top64_updown.profile.json`.
3. The probe must not change logits, fallback counts, default accepted behavior, or output tensors. It should record actual tensor/backend placement, selected-expert dimensions, hot manager state, rectangular payload estimate, compact sparse candidate status, and final hot/cold combine copy estimate.
4. Verification for this probe: build `llama-cli` and `llama-results`; run fixed-text default-off top1 and probe-enabled top1 under strict 16GB/no-swap cgroup; require `same_top1 == n_tokens`, no OOM/no swap, and no default-path behavior change.
5. Reject the route immediately if the probe shows any of: gate recompute, CPU-backend gate/up/down H2D-D2H-scatter round trip, current rectangular dummy payload, unbounded scheduler copy, or combine-copy cost that consumes the `18.947 ms` graph margin.
6. If the probe proves compact placement is feasible, write a separate hard-bound artifact before any logit-changing implementation. That bound must include expected fallback saving, kernel time, launch/sync, final combine copies, VRAM footprint, host RAM/page-cache footprint, TTFT impact, correctness gate, rollback criteria, and exact reproduction commands.
7. If compact placement is not feasible, close this sparse retained route and start a fresh bottleneck search from the accepted `4.4 tok/s` SOTA. Do not return to source-only prefetch, direct top768 Q8_0, raw/transposed hot-batch kernels, CUDA graph wrapping, or current `DS4_HOT_DISPATCH` rectangular shapes without a new hard-bound that exceeds the current one.

Sparse retained graph placement probe result:

- Source change: added a default-off CSV probe in `src/models/deepseek4.cpp::build_expert_mix`, gated only by `DS4_SPARSE_RETAINED_GRAPH_PROBE_OUT=<csv>`.
- Validated pushed source commit: `f208fb105fdb7119a520523faf04088e650d8f1c`.
- Validation run root: `/root/lfz/runs/vendor-ds4-16gb/20260705T064901Z-sparse-retained-graph-probe-pushed-validation`.
- Summary artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/sparse-retained-graph-probe-validation.json`; analyzer: `.Agent/run-tools/analyze_sparse_retained_graph_probe.py`.
- Default-off fixed-text top1 passed under strict 16GB/no-swap cgroup: `same_top1=145/145`, `first_mismatch_pos=-1`, `memory_peak_bytes=16000000000`, `oom=0`, `oom_kill=0`.
- Probe-enabled fixed-text top1 also passed under strict 16GB/no-swap cgroup: `same_top1=145/145`, `first_mismatch_pos=-1`, `memory_peak_bytes=16000000000`, `oom=0`, `oom_kill=0`.
- Probe CSV rows: `6579`; all observed calls had `mix_tokens=1` and `selected_ne0=6`.
- Backend placement observed for full gate/up/down tensors: `CPU_Mapped=5960`, `CUDA0=459`, `CUDA_Host=160` for each of gate/up/down. Graph intermediates `cur_ffn`, `selected_experts`, and `weights` were `no_buffer` at graph-build time.
- Gate reuse verdict: `graph_gate_output_input_available=0` for all rows. `build_expert_mix` receives selected IDs and weights, not a retained gate-output tensor. Therefore the accepted graph path still cannot implement the top64 zero-transfer route by reusing gate output.
- Hot manager verdict: `hot_active=0`, `hot_ready=0`, `dispatch_dual=0` for all rows in the accepted path. There is no active compact or rectangular hot branch in this accepted-path probe.
- Rectangular payload verdict: active rectangular payload is `0` in this accepted-path probe because `DS4_HOT_DISPATCH` is not active. The previous planner result still applies if current `DS4_HOT_DISPATCH` is used: the sparse top64 profile would inflate from `64` real pairs to `295` rectangular entries.
- Combine-copy lower bound from observed graph-build calls: `combine_bytes_total=646742016` bytes for the fixed-text run shape. This reinforces that any future hot/cold branch must account for final combine/copy explicitly before a performance benchmark.
- Decision: keep this default-off probe as diagnostic infrastructure. It is not a token-rate result and does not change the accepted `4.4 tok/s` SOTA.

Updated next work after this probe:

1. Do not implement a logit-changing sparse retained path on the current accepted graph as-is; it lacks retained gate-output input and mostly uses non-CUDA full up/down tensors.
2. Do not adapt current `DS4_HOT_DISPATCH` by feeding it the sparse top64 profile; the rectangular dummy payload inflation remains rejected.
3. Before any runtime source edit that changes logits, write a hard-bound/design artifact for one concrete compact retained CUDA branch:
   - how gate output or an equivalent hidden tensor becomes available to `build_expert_mix` without gate recompute;
   - how only the global top64 up/down pairs are resident or staged, without per-layer dummy inflation;
   - how hot up/swiglu/down and final hot/cold combine stay on CUDA or cross back with a bounded copy cost below the remaining `18.947 ms` graph margin;
   - exact VRAM footprint, host RAM/page-cache footprint, TTFT impact, launch/sync count, correctness verifier sequence, and rollback rule.
4. The next implementation, if the hard-bound still has positive margin, must be default-off and should start with a graph/new-op skeleton proving placement and copy counts before writing logits.
5. If no such compact retained design can meet the hard bound, close the sparse retained route and rerun a fresh bottleneck search from the accepted `4.4 tok/s` SOTA rather than repeating source-only prefetch, direct top768 Q8_0, raw/transposed hot-batch kernels, CUDA graph wrapping, or rectangular `DS4_HOT_DISPATCH`.

Compact retained CUDA branch hard-bound result:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/compact-retained-cuda-branch-hard-bound.json`.
- Status: `reject_compact_retained_cuda_branch_before_logit_changing_source_edit`.
- The only positive sparse top64 graph bound remains extremely tight: `10.014 tok/s` with only `18.947 ms` graph margin.
- Gate recompute is still below target: `9.861 tok/s`, `-191.953 ms` margin, so any sparse route that recomputes gate is rejected before source edit.
- Reusing a gate tensor is not currently available: pushed-source probe shows `graph_gate_output_input_available=0`, `hot_active=0`, and `dispatch_dual=0` across `6579` observed `build_expert_mix` calls.
- The final hot/cold combine lower bound is too close to the margin: `0.413 GiB` per accepted decode estimate, `17.197 ms` at `24 GiB/s`, leaving only `1.750 ms` before launch/split/event overhead. Even at `50 GiB/s`, only `10.693 ms` remains.
- Current `DS4_HOT_DISPATCH` remains rejected for this route because the sparse top64 profile inflates from `64` real pairs to `295` rectangular entries, i.e. `2507.5 MiB` up/down payload or `3761.25 MiB` gate/up/down payload.
- Decision: close the current sparse retained top64 route for now. Do not add a logit-changing sparse retained/hot write path on this graph. Reopen only if a new placement probe first proves a retained CUDA `gate_all` tensor and a bounded combine-copy cost with meaningful margin.
- Next optimization step must be a fresh bottleneck search from the accepted `4.4 tok/s` SOTA or a separate full-MoE CUDA design with its own hard-bound; it must not repeat source-only prefetch, direct top768 Q8_0, raw/transposed hot-batch kernels, CUDA graph wrapping, or rectangular `DS4_HOT_DISPATCH`.

Current-head accepted-path baseline guard:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-sota44-no-trace-after-sparse-close.json`.
- Source head and remote head: `9775091756ee0c3a0e5f06e68bd27c609be0e671`.
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`.
- Config: accepted path, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`, gate-only one-stream O_DIRECT pack, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, strict cold `drop_caches`, 16GB cgroup including page cache, no swap.
- Result: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32087.738292 ms`, `elapsed_seconds=62.9`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`.
- France answer was semantically correct and coherent. This ties the accepted SOTA class and is a guardrail/baseline record, not a new SOTA promotion.

Full decode up/down CUDA/streaming hard-bound result:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/full-decode-updown-cuda-streaming-hard-bound.json`; helper: `.Agent/run-tools/analyze_full_decode_updown_streaming_bound.py`.
- Status: `full_decode_updown_cuda_streaming_rejected_before_runtime_source_edit`.
- Decode up/down call-weighted payload is `148.916 GiB`; decode unique up/down payload is `21.806 GiB` (`2627` unique up rows and `2627` unique down rows).
- Total margin after perfect full fallback removal is only `1728.829 ms`.
- Exact GPU paths are rejected by kernel time alone: raw exact probe at `74.590 GiB/s` projects `1996.461 ms`, transposed exact probe at `72.434 GiB/s` projects `2055.886 ms`, and the best CPU microprobe-equivalent bandwidth projects `1993.014 ms`. All exceed the full-route margin before source transfer or integration overhead.
- Non-exact MMVQ speed is not enough to justify source work: at `276.254 GiB/s`, kernel time would be `539.054 ms`, leaving `1189.775 ms` for source and integration. That still requires `18.328 GiB/s` if every unique expert is read only once, or `125.163 GiB/s` if call-weighted source is streamed. Measured direct O_DIRECT prefill is only about `1.850 GiB/s`; current page-source effective rate is about `9.174 GiB/s`.
- Full resident payload is rejected: `21.806 GiB` decode up/down unique payload exceeds available VRAM and the 16GB host/page-cache budget. Preloading just that payload at measured O_DIRECT speed would add about `11.787 s` TTFT.
- Extra GPU MoE layer tradeoff is rejected: one full gate/up/down layer costs about `3264 MiB`, needs about `3026 MiB` beyond the current free VRAM reference, and would steal roughly `712` gate-cache slots. Estimated gate penalty is `1458.250 ms`, larger than the most optimistic one-layer full-trace fallback saving (`1172.844 ms`). This matches prior `cpu_moe=39/38` rejected runs.
- Decision: do not implement a full decode up/down streaming source path or a full-resident payload path from this bound. The next route must reduce both source bytes and exact compute substantially by construction, or pursue smaller accepted-SOTA improvements with the same RAM/TTFT/correctness gates.

Mandatory record/push rule:

- Every practice step must first update this plan or an artifact under `.Agent/runs/20260705-vendor-ds4-coldstart/`.
- If a result is a compliant new SOTA, immediately record all reproduction metadata and commit/push source plus artifacts to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`.
- New SOTA reproduction metadata must include: source commit, remote branch, full env/CLI, run path, build command, binary hash if available, model path and size, profile/manifest hashes, token rates, TTFT, elapsed time, full France answer, cgroup `memory.peak`, `memory.current`, `memory.stat`, `memory.events`, page-cache bytes, cache/pack counters, and comparison to the previous `4.4 tok/s` SOTA.
- After push, do a clean rebuild from the pushed source and rerun strict cold. Promote only if the pushed-source run still beats `4.4 tok/s` and passes every gate. TTFT-regressed or otherwise invalid candidates may be pushed as rejected records, but must be clearly marked not accepted.

### 2026-07-05 Current Active Plan Update

本节已被上方最新生效计划覆盖；内容只作为历史实验记录保留。

当前 accepted strict cold SOTA 仍为 `4.4 tok/s`，没有被后续 direct hot pool、Q8_0、CPU batch、payload compression、CUDA graph、VRAM recovery 或 hot-batch compare 候选替代：

- Accepted run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Accepted metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Accepted constraints: vendor DeepSeek, strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate-only one-stream O_DIRECT pack, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`
- Promotion gate remains: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, 16GB host RAM including file page cache, no swap, coherent/semantically correct France output, source+artifacts committed and pushed, then clean pushed-source reproduction
- Model file: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, size `156148189760` bytes (`145.42 GiB`)
- Latest pushed head before this document update: `e6075cc9a` (`vendor-ds4: bound sparse fused pair manager overhead`) on `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, author `L-Ark <fliangae@connect.ust.hk>`

Current measured bottleneck:

- Accepted decode is still dominated by DeepSeek CPU up/down fallback plus source/page behavior, not by gate cache alone.
- Current decode CPU up/down fallback is about `19029.457 ms`. Reaching `10 tok/s` requires saving about `17386.570 ms` while adding less than about `1642.887 ms` total overhead.
- Perfect source/page overlap alone has a zero-overhead ceiling around `8.993 tok/s`; source-only streaming/io_uring/prefetch remains rejected.
- Best CPU microprobe bandwidth is `74.719 GiB/s`, below the `90.643 GiB/s` zero-overhead requirement and far below the `130.298 GiB/s` requirement with `500 ms` integration overhead; CPU batch rewrite remains rejected.
- Exact raw up/down hot payload under current VRAM does not fit a 10 route: top3072 needs `12.75 GiB` and is still `73.955 ms` short before overhead; top4096 reaches a positive bound but needs `17.0 GiB` raw payload before gate/workspace. zstd/dedup only gave about `1.04x`, far below the needed `4.0x-5.33x`.
- `cpu_moe=41 + top768 exact Q8_0 hot residual` is the only remaining tight paper candidate. Its best zero-overhead bound is about `10.338 tok/s` with about `446.414 ms` decode slack, but only if the extra CPU layer and top768 source reads are mostly hidden. If top768 source is not hidden, slack drops to about `120 ms`, which is not enough for kernel/D2H/scatter/sync overhead.
- The first resident-source hot-batch compare implementation proves the correctness substrate but not the performance substrate. The current exact one-output-thread CPU-compatible kernel projects about `19.9 s` for the fixed-text ready set, far above the `446.414 ms` total top768 overhead budget and the rough `23 us/call` per-call budget.
- Optimized resident-source compute has a tight but nonzero test window only as a default-off microprobe. The fixed-text top768 ready set projects `135.817 GiB` resident source, `101.809 MiB` Q8_0 H2D, and `383.688 MiB` D2H output over `10503` ready ops. Source alone needs at least `304.241 GiB/s` to fit the whole `446.414 ms` budget; after H2D/D2H/launch/sync, the practical kernel target is about `400-550 GiB/s`.
- The no-layout warp probe is correct but still too slow: sample source bandwidth is only about `74.590 GiB/s` overall (`57.811 GiB/s` up, `104.641 GiB/s` down), projecting raw kernel time to about `1.82 s`. This closes persistent/pinned staging on the current raw layout, because kernel time alone exceeds the total budget.
- The only remaining top768 compute variant worth designing is a coalesced/transposed resident source layout. It must preserve MXFP4 block bytes and CPU-compatible arithmetic, use the same `3264 MiB` final pool shape, and add only a small streaming transform workspace. CPU-side transform is rejected on paper unless measured otherwise, because the async direct prefill path has only about `78 ms` TTFT margin.

Current execution note before the next run:

- Plan artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/fresh-sota44-current-head-trace-plan.json`.
- Purpose: rerun the accepted `4.4 tok/s` path on current pushed head `7785efc75` with diagnostic-only component tracing, because older fallback/chunk artifacts were generated before the latest default-off probe source changes.
- Runtime path must stay the accepted default: `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`, gate-only one-stream O_DIRECT pack, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, cold `drop_caches`, strict 16GB cgroup including page cache, no swap.
- Added diagnostic env only: `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`, `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback_profile.csv`, `GGML_MOE_CPU_CHUNK_TRACE_OUT={case_dir}/cpu_chunk_trace.csv`, `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=2000000`, `GGML_MOE_TTFT_TRACE_OUT={case_dir}/ttft_trace.csv`, `GGML_MOE_TTFT_TRACE_MAX_EVENTS=200000`, and the existing `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`.
- This trace is diagnostic only. If trace overhead lowers token rate, that is not a regression and must not be promoted or rejected as a SOTA candidate. The run is valid only if exit status is 0, RAM remains within `MemoryMax=16000000000` including page cache, no swap/OOM occurs, and the France answer remains semantically correct and coherent.
- Analysis after the run must separate: gate prefill/lookup, decode CPU up/down fallback, source/page exposure (`touch_us`, page faults, cgroup file cache), non-MoE dense/attention residual, scheduling/sync overhead, and trace overhead. The next runtime source edit is allowed only after writing a hard-bound artifact from this measured trace.
- Follow-up touch split: the full component trace does not enable `GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE=1`; therefore run a second diagnostic under the same accepted path plus `GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE=1` and `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback_profile.csv` only. Plan artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/fresh-sota44-current-head-touch-split-plan.json`. This is also diagnostic-only and cannot be promoted as SOTA.

Current-head trace result:

- Analysis artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/fresh-sota44-current-head-bottleneck-analysis.json`; helper: `.Agent/run-tools/analyze_current_head_bottleneck.py`; chunk aggregate: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-sota44-cpu-chunk-analysis.json`.
- Full component trace run: `/root/lfz/runs/vendor-ds4-16gb/20260705T032634Z-20260705_current_head_sota44_component_trace_absbin/france-trace-cpu40-vram0gb`; valid diagnostic, `eval_tok_s=4.1`, `prompt_tok_s=1.8`, `TTFT=31682.698396 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15071268864`, no swap/OOM, correctness passed. The lower token rate is trace overhead and is not a SOTA candidate.
- Touch split run: `/root/lfz/runs/vendor-ds4-16gb/20260705T033155Z-20260705_current_head_sota44_touch_split/france-touch-cpu40-vram0gb`; valid diagnostic, `eval_tok_s=2.6`, `TTFT=35858.926477 ms`, `memory_peak_bytes=16000000000`, no swap/OOM, correctness passed. The low token rate is active touch-profiler overhead and is not a SOTA candidate.
- Current measured decode bounds from accepted `4.4 tok/s`: accepted decode window `31047.447 ms`, estimated decoded tokens `136.609`, 10 tok/s target decode time `13660.877 ms`, required saving `17386.570 ms`.
- Decode CPU up/down fallback remains the only component with enough removable time: current-head full trace decode fallback is `19115.399 ms` (`up=10188.413 ms`, `down=8926.986 ms`). Removing it all has zero-overhead bound `11.449 tok/s` with `1728.829 ms` overhead margin.
- Estimated decode source/page delta is `16233.137 ms` (`full_trace_decode_fallback_ms - touch_split_hot_fallback_ms_after_touch`). Removing source/page alone has zero-overhead bound `9.221 tok/s`, still `1153.433 ms` short of 10 tok/s before overhead; source-only prefetch/streaming/io_uring remains closed.
- Hot fallback compute after pages are touched is only `2882.262 ms`; removing hot compute alone has zero-overhead bound `4.850 tok/s`, too small. Chunk tail gap is only `2329.894 ms`, zero-overhead bound `4.757 tok/s`; scheduler/tail-only remains closed. Gate one-stream excluding seq0 is `3572.092 ms`; gate-only remains closed.
- Next runtime source edit must target exact decode up/down fallback removal or a combined source/page + hot-compute path with total overhead below about `1.64-1.73s`. Do not implement another synchronous page-touch, broad/dedup `madvise`, packmmap/packdirect, gate-only, scheduler-only, or top768 resident-source variant without a new hard-bound that exceeds the current one.

Current-head candidate screen:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/current-head-exact-decode-updown-candidate-screen.json`; helper: `.Agent/run-tools/analyze_current_head_candidate_screen.py`.
- Source/page-only remains rejected: zero-overhead `9.221 tok/s`.
- Source/page plus best existing CPU microprobe remains rejected: zero-overhead `9.810 tok/s` using `74.719 GiB/s`.
- Source/page plus persistent MXFP4 repack hot compute remains rejected: zero-overhead `9.824 tok/s`, before persistent-repack memory/page-cache costs.
- Full fallback removal remains a theoretical upper bound only: `11.449 tok/s` with `1728.829 ms` margin, but no current exact implementation fits the source/VRAM/RAM constraints.
- Closest mixed paper route is source/page elimination + remaining CPU at microprobe speed + top128 hot residual exact kernel. Even after estimated gate penalty, it allows only `179.257 ms` for the top128 kernel/integration over `44.152 GiB` call-weighted source, requiring about `246.305 GiB/s`. Existing raw/transposed exact GPU probes are only `72-75 GiB/s`, so this is not implementable by wrapping the current kernel.
- Therefore no immediate runtime patch is allowed. The only next source edit that may be planned is a default-off kernel microprobe with a new memory-access design and a hard-bound explaining how it can exceed the top128 `~246 GiB/s` threshold while preserving fixed-text top1. Otherwise continue looking for a new exact algorithmic source; do not repeat cache-size, source-only, prefetch, or existing hot-batch kernel variants.

Next microprobe plan:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/top128-rowtile-q80-hot-batch-microprobe-plan.json`.
- The only currently allowed source edit is a default-off compare probe gated by `GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_ROW_TILE=1`. It must run after normal CPU fallback, compare against CPU `dst`, keep logits unchanged, and keep CPU fallback counts unchanged.
- Hypothesis: the existing warp2 kernels compute one output element per half-warp, so every output column rereads the same Q8_0 activation row from global memory. A row-tiled kernel can keep the exact 16-lane CPU-compatible accumulation order per output while sharing each Q8_0 block pair through shared memory across 16 columns. This is the only local mechanism identified so far that could plausibly move beyond the current `72-75 GiB/s` effective source bandwidth toward the required top128 `246.305 GiB/s`.
- Hard gate before any strict cold run: build passes; fixed-text `llama-results` top1 remains `same_top1 == n_tokens`; op-level compare `max_abs=0`; 16GB cgroup including page cache/no swap/no OOM; projected top128 kernel+integration is `<=179.257 ms` with margin. If the row-tile probe misses that bound, reject it and do not promote or benchmark it as SOTA.

Row-tile microprobe result:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/top128-rowtile-q80-hot-batch-microprobe-validation.json`.
- Source change is default-off under `GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_ROW_TILE=1`; it runs only in compare probe mode after CPU fallback, does not write logits, and does not clear fallback counts.
- Correctness passed: smoke raw/transposed compare had `max_abs=0`, and fixed-text transposed compare512 passed `same_top1=145/145`, op-level `max_abs=0`, no OOM/no swap under 16GB cgroup including page cache.
- Performance rejected: transposed compare512 sample measured only `62.907 GiB/s` effective source bandwidth, projecting top128 kernel time to about `701.853 ms` before integration overhead, far above the allowed `179.257 ms`. It is slower than the prior raw/transposed warp probes (`74.590/72.434 GiB/s`). Do not run strict cold SOTA benchmarks for row-tile/shared-Q8 variants; the accepted SOTA remains `4.4 tok/s`.

Next candidate after row-tile rejection:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/mmvq-hot-resident-compare-probe-plan.json`.
- Rationale: exact CPU-compatible Q8_0 resident kernels are now closed by measured bandwidth. The next distinct question is whether the existing optimized vendor CUDA `mmvq_rows` path can compute resident hot up/down experts fast enough, even though it is not expected to be bit-exact with CPU Q8_0 fallback.
- Allowed source edit: default-off compare-only path gated by `GGML_MOE_STREAM_MMVQ_HOT_BATCH_PROBE=1` together with the existing hot-batch compare env. It may extend the CPU callsite to pass F32 `src1` rows and strides. It must run after CPU fallback, compare against CPU `dst`, not write logits, and not clear fallback counts.
- Hard gate: if projected top128 MMVQ kernel+integration is not `<=179.257 ms`, reject the route immediately. If it is fast enough but op-level error is nonzero, the next required gate is a separate write/skip fixed-text top1 test before any strict cold benchmark.

MMVQ hot-resident compare result:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/mmvq-hot-resident-compare-probe-validation.json`.
- Correctness/safety: default-off compare-only probe built and ran under 16GB cgroup/no swap. It does not write logits or clear fallback counts. Fixed-text check passed `same_top1=145/145` because CPU fallback remains authoritative. Op-level error is nonzero (`max_abs=0.000209331512`, mostly down), so any future logit-changing use requires a separate write/skip top1 gate.
- Performance: kernel-only speed is the first probe to cross the top128 threshold: compare512 measured `276.254 GiB/s`, projecting top128 kernel time to `159.823 ms` versus the `179.257 ms` budget. But projected F32 H2D plus output D2H adds about `146.544 ms`, so the minimal standalone H2D+kernel+D2H path is about `306.367 ms` before scatter/scheduling. Therefore standalone MMVQ hot-resident skip/write is rejected for 10 tok/s.
- Next allowed design must be a fused/no-intermediate-transfer path that keeps at least the up/gate intermediate on GPU or otherwise removes most of the H2D/D2H transfer cost. Do not run strict cold SOTA benchmarks for standalone MMVQ hot-resident until that transfer bound is solved.

MMVQ fused/no-transfer hard-bound:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/mmvq-fused-transfer-hard-bound.json`; helper: `.Agent/run-tools/analyze_mmvq_fused_transfer_bound.py`.
- The analysis pairs decode `ffn_up_exps` and `ffn_down_exps` rows by `(layer, expert)` before selecting hot residuals. This is stricter than the earlier individual top128 screen and is the right unit for any fused up/gate/down route.
- CPU-backend fused skip/write remains rejected: even after removing up-output D2H and down-input H2D, the best measured-transfer bound is only `9.980 tok/s` (`pair_count=48`, `decode_ms=13688.377`, `27.501 ms` short of 10). Do not run a CPU-backend fused strict cold benchmark.
- Graph-level zero-transfer is only barely viable on paper if it reuses existing gate GPU output: best bound is `10.014 tok/s` at `pair_count=64`, with only `18.947 ms` margin. This assumes up/down hot pair kernels at the measured MMVQ speed, no intermediate H2D/D2H, no extra gate recompute, and gate cache impact already counted through the 544 MiB up/down payload.
- Gate recompute is rejected: `pair_count=64` with gate recompute projects `9.861 tok/s`, and adding gate payload worsens gate-cache pressure. Any future graph probe must reuse the existing gate result or otherwise prove an equivalent zero-transfer gate source.
- Existing full DS4 hot dispatch remains a different, already rejected design because it allocates per-layer K hot experts. A new source edit, if attempted, must be a sparse global-pair graph/probe or equivalent proof that only the selected hot up/down pairs are resident and that the hot branch stays on GPU end-to-end.
- Existing DS4 hot manager cannot directly implement the top64 pair bound: the ideal sparse up/down payload is `544 MiB`, but current manager dummy slots across the 33 active layers inflate it to `2507.5 MiB` for up/down-only if that mode existed, or `3761.25 MiB` for current gate/up/down manager behavior. The estimated gate-cache penalty is about `1093.688-1997.926 ms`, far above the `18.947 ms` graph zero-transfer margin.

Sparse pair top64 profile and next graph-probe plan:

- Profile helper: `.Agent/run-tools/create_ds4_sparse_pair_profile.py`.
- Generated artifacts: `.Agent/profiles/vendor-ds4/current_sota_sparse_pair_top64_updown.tsv`, `.Agent/profiles/vendor-ds4/current_sota_sparse_pair_top64_updown.offset_manifest.csv`, `.Agent/profiles/vendor-ds4/current_sota_sparse_pair_top64_updown.profile.json`, and `.Agent/runs/20260705-vendor-ds4-coldstart/sparse-pair-top64-profile-summary.json`.
- The profile selects `64` complete `(layer, expert)` up/down pairs from the current accepted SOTA fallback profile, with `128` tensor entries, `544 MiB` exact up/down payload, `33` active layers, and at most `4` pairs in any one layer. It is an input profile only; it is not a runtime path and does not change the accepted `4.4 tok/s` SOTA.
- Profile hashes for reproduction: TSV `5e3e11252c0c6449a73f98c19f4e4737dec081444138e5adfd7419acae2e717b`, offset manifest `8dca529d273726838e4915ff552a881062b0a1abbc6158bb562c0f66eeaa8188`, profile JSON `9ff95c307c771b05cb90c19ef10814133270c622a8beac50ecee7dadbe280cc6`.
- The first selected pair is `blk.37` expert `162`, with up offset `136714572032`, down offset `134432870656`, and `20.594 ms` summed hot fallback time in the source profile. This confirms the profile uses exact GGUF offsets rather than inferred tensor names only.
- Important correction to the paper bound: the current accepted gate stream is a CPU-backend helper. It computes gate chunks on GPU, then copies the result back to CPU `dst` and scatters there; it does not currently expose a graph-level reusable gate GPU tensor. Therefore the `10.014 tok/s` graph zero-transfer result remains a hard upper bound, not an implementable route, until a default-off graph/dataflow probe proves that gate output or an equivalent hidden tensor is retained on GPU through sparse up/down.
- Next source edit scope is restricted to a default-off sparse graph/dataflow probe, proposed env shape `DS4_SPARSE_PAIR_GRAPH_PROBE=1` plus `DS4_SPARSE_PAIR_PROFILE_JSON=.Agent/profiles/vendor-ds4/current_sota_sparse_pair_top64_updown.profile.json`. It must not change default accepted behavior, logits, fallback counts, or SOTA benchmarks.
- The probe must first prove placement and transfer shape: exactly the `544 MiB` selected sparse up/down payload resident or staged as declared, no per-layer dummy slots, no gate recompute, no hidden CPU-backend H2D/D2H between gate/up/down, and a real GPU buffer for gate output or an equivalent retained hidden activation. If it needs gate recompute, per-layer dummy payloads, CPU-backend round trips, or cannot prove gate GPU reuse, reject it before any performance run.
- Correctness gate before any logit-writing path: fixed-text top1 `same_top1 == n_tokens`, coherent France answer, op-level compare/tolerance documented if MMVQ writes replace CPU fallback, strict 16GB cgroup including page cache, `MemorySwapMax=0`, no OOM/no swap, and TTFT within the accepted gate for any promotable SOTA.
- If and only if the graph/dataflow probe proves zero-transfer feasibility, write a separate hard-bound artifact before implementation of a logit-changing path. The bound must include expected decode saving, kernel time, launch/sync, remaining transfers, VRAM/RAM footprint including page cache, TTFT impact, and rollback criteria. A strict cold SOTA benchmark is not allowed from the profile alone.

Sparse graph/dataflow probe result:

- Source change is default-off and diagnostic only: `DS4_SPARSE_PAIR_GRAPH_PROBE=1` plus `DS4_SPARSE_PAIR_GRAPH_PROBE_OUT=<csv>` records one-stream dataflow facts; default behavior is unchanged.
- Validation artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/sparse-graph-probe-validation.json`; run root: `/root/lfz/runs/vendor-ds4-16gb/20260705T054737Z-sparse-graph-probe-validation`.
- Build passed for `llama-cli` and `llama-results`. Default-off fixed-text top1 passed with `same_top1=145/145`, `first_mismatch_pos=-1`, no OOM/no swap under `MemoryMax=16000000000` and `MemorySwapMax=0`.
- Probe-enabled fixed-text top1 also passed with `same_top1=145/145`, `first_mismatch_pos=-1`, confirming the probe does not change logits.
- Dataflow verdict: `rejected_current_dataflow`. The CSV had `34800` one-stream rows and all were `gate` rows. It observed `570163200` bytes of F32 `src1` H2D, `285081600` bytes of `dst` D2H, `34800` CPU scatter rows, `0` retained GPU-output rows, `0` returned GPU-handle rows, and `0` zero-transfer-ready rows.
- Therefore the current CPU-backend one-stream helper cannot implement the top64 sparse graph zero-transfer route. It can cache/compute gate experts, but it returns through CPU memory and has no reusable graph-level GPU gate tensor. It also does not stream the sparse up/down pair rows in the accepted path.
- Next source work must not layer a logit-changing sparse pair path on this helper. The only viable continuation is a true graph-level retained-tensor design: either change graph/backend scheduling so gate, GLU/up, and selected down hot branch stay on CUDA tensors, or create an equivalent retained GPU buffer with explicit lifetime and no H2D/D2H round trip. Gate recompute, existing per-layer dummy hot manager, and CPU-backend scatter routes remain rejected.

Next retained-GPU sparse path design:

- Design artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/sparse-retained-gpu-path-next-design.json`.
- Code audit result: DeepSeek4 uses the model-specific `src/models/deepseek4.cpp::build_expert_mix` graph path. The generic `GGML_OP_MOE_FUSED_UP_GATE` path in `src/llama-graph.cpp` can fuse up/gate for eligible generic MoE decode graphs, but it does not solve down or final hot/cold placement and is not the final DS4 route.
- Existing `DS4_HOT_DISPATCH` is the closest graph-level retained mechanism, but its manager allocates dense per-layer `K + P + 1` hot tensors. For the current top64 sparse pair profile this is the already rejected `2507.5-3761.25 MiB` payload class, not the ideal `544 MiB` sparse payload.
- The next allowed source work is a default-off DS4 graph-level sparse retained-hot branch design/probe. It must use the global top64 pair profile, avoid per-layer dummy inflation, keep hot gate/up/swiglu/down on CUDA tensors or equivalent retained CUDA buffers, and prove scheduler-copy count for the final hot/cold combine before any logit-changing benchmark.
- The first pass must be a placement/payload/copy-count probe, not a strict cold SOTA run. Reject immediately if it introduces gate recompute, CPU-backend D2H/H2D/scatter, current DS4_HOT dense per-layer payload, or any unbounded scheduler copy that consumes the `18.947 ms` graph-bound margin.

Sparse retained planner probe result:

- Planner helper: `.Agent/run-tools/analyze_sparse_retained_planner.py`; result artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/sparse-retained-planner-probe.json`.
- Dry-run status: `planner_rejects_current_graph_shapes`; this is a planning/proof artifact only and does not change runtime behavior or SOTA.
- Payload facts: ideal top64 up/down sparse payload is `544.0 MiB`; adding gate payload would be `816.0 MiB`. The current DS4_HOT-style rectangular `[P,T]` ID shape would require `295` retained entries for the same profile: `64` real pair entries plus `231` dummy/padding entries. That inflates up/down-only payload to `2507.5 MiB` and gate/up/down payload to `3761.25 MiB`.
- Gate recompute remains rejected by the hard-bound: `9.861 tok/s`, `-191.953 ms` margin to 10 tok/s. Therefore the only possible graph route still requires real gate-output reuse, not recomputation or extra gate weights.
- Hot/cold final combine is now a measured lower-bound risk: with `n_embd=4096`, `P=6`, `33` active sparse layers, and `136.609` decoded-token estimate, one backend crossing of `[n_embd, P, T]` output is about `0.413 GiB`. At `24 GiB/s` that costs `17.197 ms`, leaving only `1.750 ms` of the `18.947 ms` graph margin before scheduler split/event/launch overhead. Even at `50 GiB/s`, it costs `8.255 ms`, leaving `10.693 ms`.
- Next allowed source edit is narrowed further: implement only a default-off compact hot-pick placement/new-op skeleton that proves no rectangular dummy payload, no gate recompute, and exact hot/cold combine copy count. Do not adapt current `DS4_HOT_DISPATCH` by merely feeding it the top64 sparse profile.

Latest closed decisions:

- Current serial top768 direct prefill is not promotable: combined short diagnostic under `cpu_moe=41`, gate cache `13568 MiB`, direct pool `3264 MiB` succeeded under 16GB/no-swap, but ran direct prefill before gate prefill. Direct top768 prefill was `1745.225 ms`, exceeding accepted TTFT slack by about `1020.09 ms`; at least `58.45%` of that prefill cost must be hidden before top768 can remain viable.
- Async top768 direct prefill diagnostic is complete and default-off. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/async-direct-prefill-overlap-diagnostic-result.json`. Cold cgroup runs with `drop_caches` before each trial showed async direct prefill can meet the continuation gate but with narrow margin: async run `/root/lfz/runs/vendor-ds4-16gb/20260705T004846Z-async-direct-prefill-overlap/async-cold-cgrouppeak` vs gate-only run `/root/lfz/runs/vendor-ds4-16gb/20260705T004846Z-async-direct-prefill-overlap/gate-only-cold-cgrouppeak`; service runtime delta `647 ms`, accepted TTFT slack `725.135454 ms`, effective direct prefill hide fraction about `63.3%` vs required `58.45%`, no OOM, no swap, direct inserted `768/768`, direct read/copy failures `0`. `memory_peak_bytes=16000000000` with page cache included in cgroup (`memory_file_bytes` about `15.14-15.18 GB`). This is not SOTA and does not improve token rate by itself.
- Default-off correctness after the async diagnostic source change passed fixed-text `llama-results` top1. Run: `/root/lfz/runs/vendor-ds4-16gb/20260705T005605Z-async-direct-prefill-overlap/defaultoff-top1-check`; `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14877229056`, no OOM, no swap. Therefore the default accepted path remains token-stable.
- Top768 batched/fused exact compute design is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/top768-batched-fused-exact-compute-design.json`. It clarifies that raw top768 resident compute alone is not a 10 tok/s route (`6.48 tok/s` zero-overhead bound); the `10.338 tok/s` paper route depends on source/page overlap plus top768 hot residual compute. The next allowed runtime source edit is only a default-off compare/probe path (`GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE=1`) that does not change logits or clear CPU fallback counts. Full skip/performance testing is not allowed until the probe projects total top768 compute overhead within `446.414 ms` and fixed-text top1 passes.
- HOT_BATCH_PROBE coverage skeleton is implemented default-off and validated. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-batch-probe-skeleton-validation.json`. It adds `GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_OUT` after CPU fallback, scans real `mul_mat_id` ops, checks top768 direct hot-pool readiness, and writes CSV counters without launching compute kernels, writing `dst`, or clearing CPU fallback counts. Default-off top1 passed with `same_top1=145/145`, `max_abs=0`; same-config `cpu_moe=41 + direct async + probe` selfcheck also passed with `same_top1=145/145`, `max_abs=0`. The fixed-text selfcheck saw `32724` ready rows and `0` not-ready rows, but this includes prompt/prefill rows and is not a decode token-rate claim.
- HOT_BATCH_COMPARE diagnostic is implemented default-off and validated. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-batch-compare-kernel-validation.json`. It extends `GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_OUT` with `GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_COMPARE=1` and `GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_COMPARE_MAX_RECORDS`, runs after CPU fallback, uses CPU `dst` as authoritative, compares resident-source GPU output against CPU fallback output, and does not clear CPU fallback counts or change logits by default.
- HOT_BATCH_COMPARE correctness passed for the bounded probes. Smoke run `/root/lfz/runs/vendor-ds4-16gb/20260705T014658Z-hot-batch-compare/smoke-hi-compare32` had `compare_ran=4`, `compare_ok=4`, `max_abs=0`, `diff_count=0`. Fixed-text selfcheck run `/root/lfz/runs/vendor-ds4-16gb/20260705T014841Z-hot-batch-compare/probe-cpu41-compare512` had `same_top1=145/145`, `max_abs=0`, `compare_ran=262`, `compare_ok=262`, op-level `max_abs=0`, `diff_count=0`. Default-off top1 after the compare source change also passed: `/root/lfz/runs/vendor-ds4-16gb/20260705T015230Z-hot-batch-compare/defaultoff-top1-check`, `same_top1=145/145`, `max_abs=0`, `memory.max=16000000000`, `memory.swap.max=0`, `oom=0`, `oom_kill=0`, `swaps=0`.
- HOT_BATCH_COMPARE current compute path is rejected for performance. The fixed-text compare512 sample covered `378` ready rows / `1,163,264` output floats and spent `214715 us` in kernel time, with sample total compute timing `229634 us`. Projecting by ready rows or output count gives about `19.9 s` for the full fixed-text ready set, orders of magnitude above the `446.414 ms` top768 overhead budget. This rejects the current one-output-thread exact CPU-compatible kernel as a performance path; it does not reject a future optimized parallel/token-stable kernel.
- HOT_BATCH_WARP hard-bound design is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-batch-warp-kernel-hard-bound-design.json`. A bit-exact warp/half-warp kernel can preserve the CPU-compatible accumulation order by mapping the 16 AVX accumulator lanes for each dot to 16 CUDA lanes and doing the final hsum in the same order. This is allowed only as a default-off compare microprobe, because the route is still tight: without pinned staging and one sync per op, D2H/H2D/launch overhead can consume the entire `446.414 ms` slack.
- HOT_BATCH_WARP probe is implemented default-off and validated for correctness, but rejected for the current raw source layout. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-batch-warp-probe-validation.json`. Corrected smoke run `/root/lfz/runs/vendor-ds4-16gb/20260705T022517Z-hot-batch-warp-probe/smoke-hi-compare32` passed `compare_ok=4/4`, `max_abs=0`, `diff_count=0`. Fixed-text run `/root/lfz/runs/vendor-ds4-16gb/20260705T022654Z-hot-batch-warp-probe/probe-cpu41-compare512` passed `same_top1=145/145`, `max_abs=0`, `compare_ok=262/262`, no OOM/no swap. However, sample kernel time was `21033 us` for `1.568 GiB` source, projecting to `1.82 s`; current-layout warp is therefore rejected before persistent/pinned staging.
- HOT_BATCH_TRANSPOSED hard-bound design is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-batch-transposed-layout-hard-bound-design.json`. The next source edit, if attempted, may only be a default-off transposed-pool prefill plus compare diagnostic. It must first prove transform/prefill TTFT overhead fits the `78 ms` margin and then prove projected kernel time is `<=250 ms`; otherwise close `cpu_moe=41 + top768`.
- HOT_BATCH_TRANSPOSED prefill timing probe is implemented default-off and validated. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-batch-transposed-prefill-validation.json`. It adds `GGML_MOE_STREAM_ONE_DIRECT_TRANSPOSE=1`, keeps a single final `3264 MiB` direct pool, uses one slot-sized device staging buffer, and reorders existing MXFP4 bytes on GPU. Cold diagnostic inserted `768/768` with `transform_ms=33.928`, no OOM/no swap, and cgroup `memory.peak=16000000000`; default-off fixed-text top1 still passed with `same_top1=145/145`, `max_abs=0`.
- HOT_BATCH_TRANSPOSED compare probe is implemented default-off and rejected for performance. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-batch-transposed-compare-validation.json`. It passed smoke and fixed-text correctness (`same_top1=145/145`, `compare_ok=262/262`, op-level `max_abs=0`) and default-off top1 after source change. But fixed-text sample kernel time was `21659 us` for `1.568 GiB`, projecting to about `1.87 s`, worse than raw warp and far above the `250 ms` cutoff. Therefore `cpu_moe=41 + top768` is closed.
- Current per-call Q8_0 CUDA path is rejected for performance. It measured about `3140.36 us/call` total and `1088.488 us/call` kernel sync, while top768 has only about `23 us/call` budget if it is to fit the `10 tok/s` bound.
- Q8_0 CPU-compatible arithmetic is token-stable for combined all-up/all-down in fixed-text `llama-results`, but the strict cold all-up performance path regressed to `1.8 tok/s` and raised TTFT above gate. It is correctness evidence only, not a SOTA path.
- CUDA graph is not an accepted DeepSeek SOTA change. It has no current evidence of improving the accepted path under the strict 16GB/TTFT/correctness gates.

Immediate execution plan:

1. Keep the accepted `4.4 tok/s` runtime path unchanged by default. No candidate may change default behavior until it has passed the fixed-text verifier and a strict cold benchmark.
2. Do not run a strict cold performance benchmark for the current hot-batch compare or warp implementation. They are correctness/measurement scaffolding only, and current raw-layout projected compute cost is above the top768 budget.
3. Do not add persistent allocations, pinned staging, CUDA graph wrapping, or skip/write mode around the current raw-layout warp kernel. Kernel time alone projects to about `1.82 s`, so those integration optimizations cannot make this route fit `446.414 ms`.
4. The `cpu_moe=41 + top768` route is now closed: raw scalar, raw warp, transposed prefill+warp, CPU batch, source-only overlap, and raw topN residency all miss the hard bound or the VRAM/TTFT gates. Do not add pinned staging, CUDA graph wrapping, skip/write mode, or more cache-size variants around this route.
5. The next step is a fresh bottleneck search from the accepted `4.4 tok/s` SOTA path under current pushed source. First rerun a strict accepted-path component trace with cold `drop_caches`, 16GB cgroup including page cache, no swap, France correctness, and per-token/per-component timings.
6. The fresh trace must separate: gate prefill/lookup, decode CPU up/down fallback, source/page exposure, non-MoE dense/attention time, scheduling/sync overhead, and any cgroup/page-cache pressure. Use that measured bottleneck to write the next hard-bound artifact before any new runtime source edit.
7. Any future probe pass condition remains strict: fixed-text `llama-results` exits 0 with `same_top1 == n_tokens`, op-level compare has `max_abs=0` for probed tasks or a separately justified token-stable tolerance with top1 proof, RAM remains within `MemoryMax=16000000000` including page cache, no swap/OOM, and any projected overhead fits the relevant hard budget with margin.
8. Any logit-changing candidate must pass fixed-text `llama-results` top1 with `same_top1 == n_tokens` before any strict cold performance run. France output must remain semantically correct, coherent, and complete.
9. Any strict cold performance run must record: run path, full command/env, source head, branch, binary/library hashes, model/profile/pack hashes, memory stats including file page cache, `oom`/`oom_kill`, swap counters, TTFT, prompt/eval token rates, counters, exact output, and correctness decision.
10. If a compliant new SOTA appears, stop exploration immediately. Commit and push source plus artifacts to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then clean-rebuild and reproduce from pushed source before promotion.
11. If a candidate regresses throughput, violates 16GB including page cache, fails correctness, or exceeds the accepted TTFT gate for a promotable result, revert/guard off runtime source back to the accepted SOTA path and keep only rejected documentation/artifacts.

### 2026-07-05 Historical Active Plan Override

本节已被上方最新生效计划覆盖；内容只作为历史实验记录保留。

当前 accepted strict cold SOTA 仍为 `4.4 tok/s`，没有被 direct-reader、Q8_0、CUDA graph 或其他候选替代：

- Accepted run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Accepted metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Accepted constraints: vendor DeepSeek, strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate-only one-stream O_DIRECT pack, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`
- Promotion gate remains: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, 16GB RAM including page cache, no swap, coherent/semantically correct France output, committed+pushed source/artifacts, then clean pushed-source reproduction
- Model file: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, size `156148189760` bytes (`145.42 GiB`)
- Latest pushed code/audit head before this plan update: `738bde401ecc1d9c9d67d788aad980bd5c528edb` (`vendor-ds4: bound async source cpu batch`) on `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`

Recent completed work and decisions:

- Top768 direct-reader manifest is generated and pushed: `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top768_touch_residual.offset_manifest.csv`, 768 entries, no missing tensors, payload `3422552064` bytes.
- Direct manifest loader and direct hot pool skeleton are implemented default-off. They must not affect the accepted SOTA path unless the corresponding env vars are set.
- O_DIRECT raw GGUF reads required align-down bounce reads because expert offsets are `256 mod 4096`; this is fixed in the direct reader skeleton.
- `cpu_moe=41 + gate cache 13568 MiB + direct pool 3264 MiB` fits as a diagnostic: direct pool has 768 slots, gate cache has 3192 slots, GPU sample peak left about `245-251 MiB` free.
- Full synchronous top768 direct prefill read `3422552064` bytes in `1723.238 ms` at about `1.850 GiB/s`, with `direct_failures=0` and `direct_fallbacks=0`. This is useful as a source substrate but is not promotable synchronously because accepted TTFT slack is only about `725 ms`; it needs about `998 ms` hidden/overlapped to satisfy the TTFT gate.
- The direct hot pool currently stages payload only; it is not yet connected to exact up/down compute and cannot by itself improve token rate.
- `tools/results/results.cpp` top1 comparator is fixed and pushed. The self-check passed with baseline/check status `0`, `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`; this verifier is now mandatory before any logit-changing candidate benchmark.
- The old dirty Q8_0 up-only candidate is not reproducible from pushed source and previously failed top1 (`same_top1=136/145`, first mismatch at token position 4). It is rejected as evidence and must not be used as a SOTA or correctness claim.
- Q8_0 correctness scaffold is now implemented default-off and validated as a diagnostic only. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-correctness-scaffold-validation.json`. Build passed for `llama-cli` and `llama-results`; default-off `llama-results` top1 self-check passed with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`; bounded CUDA MXFP4 x Q8_0 probe wrote 16 sampled records with `max_abs=2.38418579e-7`, `mean_abs=3.0733645e-8`. This is not a SOTA and does not change model outputs.
- Q8_0 broadened probe is also validated. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-broaden-probe-validation.json`. The new `GGML_MOE_STREAM_Q80_PROBE_NAME_FILTER` is default-off and only affects probe sampling. Bounded up probe covered `blk.0-2.ffn_up_exps.weight` with 256 records, `max_abs=4.76837158e-7`, `mean_abs=4.45870683e-8`; bounded down probe covered `blk.0-2.ffn_down_exps.weight` with 256 records, `max_abs=1.78813934e-7`, `mean_abs=1.0135409e-8`. This is still diagnostic only.
- Q8_0 output-writing subset now has its first token-level result. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-output-subset-validation.json`. The default-off `GGML_MOE_STREAM_Q80_WRITE_*` path still runs CPU fallback first, then overwrites `dst`, so it is a correctness gate only. Guarded up subset `blk.0.ffn_up_exps.weight`, `max_calls=64`, passed fixed-text `llama-results` with `same_top1=145/145`, `first_mismatch_pos=-1`, `NMSE=0`, while q80 write report showed `max_abs=1.43051147e-6`. Down subset `blk.0.ffn_down_exps.weight`, `max_calls=64`, failed with `same_top1=143/145`, `first_mismatch_pos=109`, `NMSE=1.296e-3`; therefore output-writing Q8_0 is guarded to `ffn_up_exps` only and down must not be enabled before root-cause analysis.
- Q8_0 up skip subset now has its first token-level result. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-up-skip-subset-validation.json`. The default-off `GGML_MOE_STREAM_Q80_SKIP_*` path runs before CPU fallback and clears `matrix_row_counts[cur_a]` only when CUDA returns success. It is guarded to `ffn_up_exps` only. For `blk.0.ffn_up_exps.weight`, `max_calls=64`, it skipped 64 CPU fallback calls and passed fixed-text `llama-results` with `same_top1=145/145`, `first_mismatch_pos=-1`, `NMSE=0`. Validation run: `/root/lfz/runs/vendor-ds4-16gb/20260704T184124Z-q80-up-skip-top1/blk0-up-skip64`; skip report has 64 records for `blk.0.ffn_up_exps.weight`, `cne1=1`, `ne01=2048`; `memory.peak=1172144128` and `memory.events` reported no OOM. This is correctness progress toward a performance path, but not a strict cold SOTA benchmark and not promotable as SOTA.
- Q8_0 full `blk.0.ffn_up_exps.weight` pre-fallback skip is rejected. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-up-skip-blk0-full-validation.json`. Clean validation run: `/root/lfz/runs/vendor-ds4-16gb/20260704T190709Z-q80-up-skip-top1/blk0-up-full-clean`; `GGML_MOE_STREAM_Q80_SKIP_MAX_CALLS=0` produced 870 skip records and failed fixed-text top1 with `same_top1=142/145`, `first_mismatch_pos=71`, `NMSE=8.062e-04`, `max_abs=4.73417`, `mean_abs=0.107156`. The cgroup did not OOM: `memory.max=16000000000`, `memory.swap.max=0`, `memory.peak=11989463040`, `oom=0`, `oom_kill=0`. This proves the current Q8_0 up-skip arithmetic is not token-stable when applied to all layer-0 up calls, likely because small local differences accumulate through later layers and flip low-margin top1 positions.
- Q8_0 up-skip call-limit sweep is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-up-skip-calllimit-sweep-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T191555Z-q80-up-skip-calllimit-sweep/blk0-up`; `max_calls=96` passed with `same_top1=145/145`, but `128` failed with `same_top1=144/145`, `first_mismatch_pos=135`; `192` and `256` also failed at token 135, and `512` failed at token 71 with `same_top1=143/145`. The cgroup did not OOM: `memory.max=16000000000`, `memory.swap.max=0`, `memory.peak=1391484928`, `oom=0`, `oom_kill=0`. The next root-cause target is the 97-128 call window, not broader layer expansion.
- Q8_0 up-skip fine call-limit sweep is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-up-skip-calllimit-fine-sweep-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T192407Z-q80-up-skip-calllimit-fine-sweep/blk0-up`; `max_calls=100` passed with `same_top1=145/145`, while `104`, `108`, `112`, `116`, `120`, and `124` all failed identically with `same_top1=144/145`, `first_mismatch_pos=135`, `max_abs=4.67701`, `mean_abs=0.10113`. The added records between pass and fail are record 100 expert 114, record 101 expert 157, record 102 expert 29, and record 103 expert 62. The next exact target is `max_calls=101,102,103`.
- Q8_0 up-skip exact call-limit sweep is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-up-skip-calllimit-exact-sweep-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T193308Z-q80-up-skip-calllimit-exact-sweep/blk0-up`; `max_calls=101`, `102`, and `103` all passed with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. Since `104` fails, the first non-token-stable event is record 103, `blk.0.ffn_up_exps.weight`, expert 62. The next root-cause step must compare output-write/probe behavior for record 103 to decide whether the failure is numerical MXFP4 x Q8_0 drift or a skip-path side effect.
- Q8_0 up output-write boundary check is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-up-output-write-boundary-validation.json`. Clean run: `/root/lfz/runs/vendor-ds4-16gb/20260704T194419Z-q80-up-output-write-boundary/blk0-up-clean`; output-write `max_calls=103` passed, while output-write `max_calls=104` failed with the same top1 boundary as skip: `same_top1=144/145`, `first_mismatch_pos=135`, `max_abs=4.67701`, `mean_abs=0.10113`. The local record-103 output difference is tiny (`max_abs=7.15255737e-7`, `mean_abs=4.58167051e-8`), but it still propagates through later layers and flips a low-margin token. This classifies the failure as numerical/token-stability drift in the CUDA MXFP4 x Q8_0 arithmetic, not a pre-fallback skip side effect.
- Q8_0 CPU-compatible CUDA arithmetic first gate is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpu-compatible-arithmetic-validation.json`. The new `GGML_MOE_STREAM_Q80_CPU_COMPAT=1` path is default-off and mimics the AVX2/FMA reference two-block, 8-lane accumulation order. It fixed the known boundary: output-write `max_calls=104` passed with `same_top1=145/145` and write records 98-103 all had local `max_abs=0`; pre-fallback skip `max_calls=104` also passed with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. This is correctness progress only, not a performance benchmark or SOTA.
- Q8_0 CPU-compatible full `blk.0.ffn_up_exps.weight` pre-fallback skip is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-blk0-full-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T200955Z-q80-cpucompat-up-skip/blk0-up-full`; `GGML_MOE_STREAM_Q80_CPU_COMPAT=1`, `GGML_MOE_STREAM_Q80_SKIP_MAX_CALLS=0` skipped 870 calls and passed fixed-text top1 with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. The previous scalar full-blk0 run failed with `same_top1=142/145`; this source change fixes that correctness blocker but is still not a strict cold performance SOTA.
- Q8_0 CPU-compatible individual full `blk.1.ffn_up_exps.weight` and `blk.2.ffn_up_exps.weight` pre-fallback skip is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-blk1-blk2-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T201929Z-q80-cpucompat-up-skip-individual/blk1-blk2`; source head `df733da0cb6a1406de1e2db3f7a2b5c16a4fa5be`; both filters skipped 870 calls and passed fixed-text top1 with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. The validation cgroup used `memory.max=16000000000`, `memory.swap.max=0`, `memory.peak=1025572864`, and `oom=0`/`oom_kill=0`. This is correctness expansion only; accepted SOTA remains `4.4 tok/s`.
- Q8_0 CPU-compatible all-up `ffn_up_exps.weight` pre-fallback skip is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-all-up-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T203834Z-q80-cpucompat-all-up/top1`; source head `e798311a89880fb3d344796d7508e871d17b2c8f`; it skipped `34800` calls across `blk.0` through `blk.39` and passed fixed-text top1 with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. The validation cgroup used `memory.max=16000000000`, `memory.swap.max=0`, `memory.peak=1026064384`, and `oom=0`/`oom_kill=0`. This clears the correctness gate for an up-only strict cold performance benchmark; accepted SOTA remains `4.4 tok/s`.
- Q8_0 CPU-compatible all-up strict cold performance benchmark is rejected. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-up-skip-perf-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T204845Z-q80-cpucompat-upskip-strict-cold/france-cpu40`; source head `1292ae8620c35bf3182269e773caa65f6e31ca55`; result `eval_tok_s=1.8`, `prompt_tok_s=1.3`, `TTFT=38267.907446 ms`, `elapsed_seconds=114.46`, `q80_skip_records=18688`, `ram_ok=true`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15086510080`, `oom=0`, `swaps=0`, `correctness_ok=true`. It fails both throughput and TTFT gates, so accepted SOTA remains `4.4 tok/s`.
- Q8_0 up-skip overhead profiling is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-upskip-overhead-profile-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T211727Z-q80-upskip-overhead-profile/france-cpu40`; source head `8f7791d3f22da9bed8fbd865478a940b8e1de4cd`; result remains rejected (`eval_tok_s=1.8`, `TTFT=39227.12188 ms`) but gives the timing breakdown: `18688` attempted/ok skip calls, `total_ms=58687.042`, `host_src0_ms=28106.781`, `kernel_sync_ms=20341.664`, `h2d_ms=6062.462`, `cuda_alloc_ms=2181.436`, `cuda_free_ms=1615.032`, `d2h_ms=184.696`, `scatter_ms=17.111`. This proves the current per-call Q8_0 skip path is dominated by copying full up expert payload and small synchronized kernels; persistent cudaMalloc removal alone cannot make it viable.
- Source-resident/batched Q8_0 hard-bound screening is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-source-resident-batched-hard-bound.json`. The top768 up/down manifest contains `382` up entries covering `11455/18688` (`61.3%`) of the profiled q80 up calls, with about `1.62 GiB` up payload. This is enough to justify a hot up source-residency design only if paired with batched/fused kernels, but it is not enough for 10 tok/s by itself. Persistent buffers only and source-residency-only are rejected.
- Q8_0 CPU-compatible down output-write first gate is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-down-write64-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T214046Z-q80-cpucompat-down-write/blk0-down-write64`; source head `714d4d66f65f546bc35423bdfcded3d6cf49cfef`; with `GGML_MOE_STREAM_Q80_ALLOW_DOWN=1`, `GGML_MOE_STREAM_Q80_CPU_COMPAT=1`, and output-write `max_calls=64`, fixed-text top1 passed with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`; write report has 64 records for `blk.0.ffn_down_exps.weight`, all `max_abs=0`. This reopens down correctness work under the explicit down env gate only.
- Q8_0 CPU-compatible down pre-fallback skip first gate is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-down-skip64-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T214925Z-q80-cpucompat-down-skip/blk0-down-skip64`; source head `f80c612fc2480447c54ce42201d1345503083a9d`; with `GGML_MOE_STREAM_Q80_ALLOW_DOWN=1`, `GGML_MOE_STREAM_Q80_CPU_COMPAT=1`, and pre-fallback skip `max_calls=64`, fixed-text top1 passed with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`; skip report has 64 records for `blk.0.ffn_down_exps.weight`. This proves bounded down skip is token-stable under CPU-compatible arithmetic.
- Q8_0 CPU-compatible full `blk.0.ffn_down_exps.weight` pre-fallback skip is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-down-blk0-full-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T220539Z-q80-cpucompat-down-skip/blk0-down-full`; source head `2dbb05f94bae512bc0dbd56e99b3509c5c94cf96`; with `GGML_MOE_STREAM_Q80_ALLOW_DOWN=1`, `GGML_MOE_STREAM_Q80_CPU_COMPAT=1`, and pre-fallback skip `max_calls=0`, fixed-text top1 passed with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`; skip report has `870` records for `blk.0.ffn_down_exps.weight`. The validation cgroup used `memory.max=16000000000`, `memory.swap.max=0`, `memory.peak=4901580800`, `memory.stat file=3876634624`, and `oom=0`/`oom_kill=0`. This is correctness expansion only; accepted SOTA remains `4.4 tok/s`.
- Q8_0 CPU-compatible individual full `blk.1.ffn_down_exps.weight` and `blk.2.ffn_down_exps.weight` pre-fallback skip is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-down-blk1-blk2-validation.json`. Run root: `/root/lfz/runs/vendor-ds4-16gb/20260704T221306Z-q80-cpucompat-down-skip-individual`; source head `f3c51c5c4138d55696db559cdf9c6f49542474ce`; both filters skipped `870` calls and passed fixed-text top1 with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. The validation cgroups used `memory.max=16000000000`, `memory.swap.max=0`, peak about `1.025 GB`, and `oom=0`/`oom_kill=0`. This is correctness expansion only; accepted SOTA remains `4.4 tok/s`.
- Q8_0 CPU-compatible all-down `ffn_down_exps.weight` pre-fallback skip is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-all-down-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T222102Z-q80-cpucompat-all-down/top1`; source head `ed457e3eca805f636aa78ad174ad016eb2b49152`; it skipped `34800` calls across `blk.0` through `blk.39` and passed fixed-text top1 with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. The validation cgroup used `memory.max=16000000000`, `memory.swap.max=0`, `memory.peak=1026875392`, `memory.stat file=1613824`, and `oom=0`/`oom_kill=0`. This clears down-only correctness; accepted SOTA remains `4.4 tok/s`.
- Q8_0 CPU-compatible combined all-up plus all-down pre-fallback skip is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-cpucompat-all-updown-validation.json`. Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T222845Z-q80-cpucompat-all-updown/top1`; source head `5134fde7ed22bb4b2b0f5dac2e2440f6cdc20c0f`; it skipped `69600` calls (`34800` up and `34800` down) and passed fixed-text top1 with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. The validation cgroup used `memory.max=16000000000`, `memory.swap.max=0`, `memory.peak=1026617344`, `memory.stat file=3080192`, and `oom=0`/`oom_kill=0`. This proves CPU-compatible Q8_0 arithmetic is token-stable for combined up/down, but it is not a performance result; accepted SOTA remains `4.4 tok/s`.
- Source-resident plus batched/fused Q8_0 hard-bound v2 is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q80-source-resident-batched-hard-bound-v2.json`. Result: rejected as a 10 tok/s route under the current VRAM shape. Decode top768 saves `9966.891 ms` with `3.188 GiB` payload but has only a `6.480 tok/s` zero-overhead ceiling. Top3072 saves `17312.615 ms` with `12.750 GiB` payload but still misses 10 tok/s by `73.955 ms` before any overhead. Top4096 is the first positive 10 tok/s bound (`10.797 tok/s`, `1008.842 ms` slack) but needs `17.000 GiB` raw payload before gate cache/workspace, which does not fit while preserving current gate behavior. Therefore do not implement raw topN source residency as the next 10 tok/s path.
- Exact CPU fallback batching/microkernel hard-bound is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/cpu-fallback-microkernel-hard-bound.json`. Result: standalone rejected for 10 tok/s. Touch split shows source/page exposure can account for `15856.166 ms`, leaving only `3173.291 ms` hot fallback compute after pages are hot. Removing all hot compute alone gives only a `4.900 tok/s` zero-overhead ceiling. Combined with perfect source/page overlap, 10 tok/s still requires at least `1530.404 ms` additional hot-compute saving, or `48.225%` of the hot fallback compute at zero overhead.
- Joint async source/page overlap plus CPU batching hard-bound is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/async-source-plus-cpu-batch-hard-bound.json`. Result: no runtime code yet. With perfect source/page overlap, saving `0%/25%/50%/75%/100%` of hot compute gives `8.993/9.488/10.041/10.663/11.367 tok/s`; the `50%` case has only `56.241 ms` zero-overhead slack. If source overlap is only `90%`, it requires `98.2%` of hot compute to be removed. Existing exact same-op parallel touch already regressed to `3.0 tok/s`, so do not implement a same-op async source path.
- Hot CPU fallback lower-bound audit is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/hot-cpu-fallback-lower-bound.json`. Result: reject a large CPU batch runtime rewrite without a microprobe. The current hot path is already the x86 MXFP4 x Q8_0 kernel (`ggml_vec_dot_mxfp4_q8_0`) on a 20-thread chunked CPU fallback, not a generic scalar path. Measured hot fallback after source/page exposure is `3173.291 ms` over `35880` expert calls and about `148.916 GiB` expert-call bytes, or about `46.928 GiB/s` and `88.442 us/call`. To make the joint 10 tok/s path viable, hot fallback must reach at least `90.643 GiB/s` with zero overhead, `130.298 GiB/s` with `500 ms` integration overhead, or `231.636 GiB/s` with `1000 ms` overhead. Even `90 GiB/s` is slightly short; `100 GiB/s` has only about `153.727 ms` slack before other overhead.
- CPU batch microprobe is implemented default-off and validated. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/cpu-batch-microprobe-validation.json`. It is enabled only by `GGML_MOE_CPU_BATCH_MICROPROBE_OUT`; it replays a bounded hot-data subset after normal fallback, compares against `dst`, and never writes model outputs. Fixed-text top1 passed for up and down probes with `same_top1=145/145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`. The stable repeats=64 probe measured `src0_gib_s=73.805/73.662` for up and `72.476/74.719` for down. The best source-payload bandwidth (`74.719 GiB/s`) is below the `90.643 GiB/s` zero-overhead requirement and far below the `130.298 GiB/s` requirement with `500 ms` overhead, so the CPU batch runtime rewrite path is rejected for 10 tok/s.
- Fallback-byte reduction hard-bound after the CPU probe is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/fallback-byte-reduction-hard-bound-after-cpu-probe.json`. Result: no new runtime candidate above 10 tok/s under the current VRAM/source shape. Even with perfect source/page overlap and the best measured CPU microprobe bandwidth, decode would be about `14011.0 ms` or `9.750 tok/s`, still `350.127 ms` short before overhead. Raw exact topN residency is also closed under current VRAM: top3072 needs `12.75 GiB` and still has `-73.955 ms` zero-overhead slack; top4096 first crosses 10 (`10.797 tok/s`) but needs `17.0 GiB` raw payload before gate/workspace. Source streaming/io_uring-only remains rejected because measured direct read is about `1.85 GiB/s` versus `90.643+ GiB/s` needed in the 10 tok/s class.
- Accepted-path VRAM footprint snapshot and top4096 reconciliation are complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/vram-footprint-snapshot-reconciliation.json`. Fresh accepted path reports `CUDA0 32109 = 238 free + (17362 self = 17339 model + 18 context + 4 compute) + 14508 unaccounted`, gate cache `13.2 GiB / 3192 slots`, and prefill `3000` entries / `13369344000` bytes / `4810.585 ms`. VRAM recovery alone is rejected: even `238 MiB free + cpu_moe=41 prior 3264 MiB + entire 13566 MiB gate slot payload = 17068 MiB`, which is still `340 MiB` short of top4096's `17408 MiB` before workspace/allocator overhead. Top3072 would require stealing about `9554 MiB` of gate payload even with cpu_moe=41, and still misses 10 tok/s before overhead.
- Exact top4096 payload-reduction audit is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/exact-payload-reduction-audit.json`. Generated reproducible top4096 profile and offset manifest: `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top4096_fallback_us.tsv` and `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top4096_fallback_us.offset_manifest.csv`, with `4096` entries, `17408 MiB`, and `18395.412 ms` residual sum. Full top4096 zstd level 1 ratio is only `1.040149`; top512 zstd level 10 ratio is `1.040140`; sampled `1,048,576` MXFP4 blocks across 256 entries had `0` duplicate blocks and `0` zero blocks. Exact compression/dedup is therefore rejected as a route to the required `4.0x-5.33x` payload reduction.
- Top768 source-hidden hard-bound recheck is complete. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/top768-source-hidden-hard-bound-recheck.json`. Result: the route is conditional only. Current combined short diagnostic with `cpu_moe=41`, gate cache `13568 MiB`, and top768 direct pool `3264 MiB` succeeds under 16GB/no-swap, but logs direct prefill before gate prefill, so the current source path is serial: top768 direct prefill `1745.225 ms`, gate prefill `3573.370 ms`. The direct prefill exceeds accepted TTFT slack by about `1020.09 ms`; at least `58.45%` of it must be hidden. Decode also has only `446.414 ms` zero-overhead slack for all top768 compute overhead, or about `23 us` per covered call; the existing per-call Q8_0 CUDA path (`avg_kernel_sync_us=1088.488`, `avg_total_us=3140.36`) is rejected.
- Latest written next-step plan is `.Agent/runs/20260705-vendor-ds4-coldstart/async-direct-prefill-overlap-diagnostic-plan.json`. Next implementation, if any, must be a default-off source-only diagnostic (`GGML_MOE_STREAM_ONE_DIRECT_PREFILL_ASYNC=1`) to test whether top768 direct prefill can overlap enough with gate prefill/prompt cold work while preserving 16GB/no-swap behavior. It must not change logits or accepted default behavior. If it cannot keep added TTFT cost within the `725.135454 ms` slack, reject `cpu_moe=41 + top768` as a 10 tok/s route before writing batched/fused exact compute kernels.

Current bottleneck:

- The accepted path is still dominated by DeepSeek CPU up/down fallback plus source/page behavior, not by gate cache alone.
- Current decode CPU up/down fallback is about `19029.457 ms`. Reaching `10 tok/s` requires saving about `17386.570 ms` while adding less than about `1642.887 ms` total overhead.
- `cpu_moe=41 + top768 exact Q8_0 hot residual` remains only a tight paper candidate: best-case zero-overhead bound is about `10.338 tok/s` with about `446 ms` slack, and only if the extra CPU layer and top768 source reads are hidden. If they are not hidden, slack drops to about `120 ms`, which is not enough for kernel/D2H/scatter/sync overhead.

Immediate execution plan:

1. Keep the accepted `4.4 tok/s` runtime path unchanged by default. No candidate may change default behavior until it has passed the verifier and a strict cold benchmark.
2. Before the next runtime source edit, write or update the corresponding artifact with: bottleneck component, removable time, hard upper bound, expected RAM/page-cache pressure, expected TTFT impact, correctness gate, rollback rule, and exact env/command skeleton.
3. Q8_0 all-up correctness is solved, but the strict cold performance path is closed in its current form. Up-only CPU-compatible skip made the run much slower (`1.8 tok/s`) and raised TTFT above the acceptance gate. Do not rerun this same all-up skip benchmark or tune its cache/env knobs.
4. The bottleneck is now measured, not guessed: current Q8_0 skip spends `58.687s` total inside the replacement path for `18688` calls. The largest buckets are full host `src0` expert copy (`28.107s`), kernel+device sync (`20.342s`), H2D (`6.062s`), and cudaMalloc/free (`3.796s`). D2H and scatter are small by comparison.
5. The next design must be rejected on paper unless it removes both source movement and per-call synchronized kernel cost. Persistent CUDA buffers alone can save only about `3.796s`, far below the `50s+` regression. Source residency/H2D elimination alone would still leave about `24.5s` of Q8_0 path time, which is unlikely to beat the estimated CPU up fallback it replaces. Therefore the next candidate must combine source residency or cache reuse with batched/fused Q8_0 compute, or it must pivot back to down correctness/source analysis.
6. The CPU batch, source-only streaming, raw topN residency, VRAM-recovery-alone, exact payload-compression/dedup, current serial top768 source path, and current per-call Q8_0 compute path are rejected for 10 tok/s under the current shape. Before any compute runtime source edit, execute `.Agent/runs/20260705-vendor-ds4-coldstart/async-direct-prefill-overlap-diagnostic-plan.json`: implement or design only a default-off async direct prefill overlap diagnostic. If it cannot hide at least about `58.5%` of the top768 direct prefill cost and keep TTFT within gate, reject top768 as a 10 route. If it passes, then write a separate batched/fused exact compute design with fixed-text top1 gate before any strict cold performance run.
7. The legacy non-CPU-compatible down path remains rejected (`same_top1=143/145`, first mismatch token 109). Down may only be enabled behind `GGML_MOE_STREAM_Q80_ALLOW_DOWN=1` together with `GGML_MOE_STREAM_Q80_CPU_COMPAT=1`, and each expanded scope must have a new op-level plus token-level proof before performance testing.
8. Any new Q8_0 skip/write scope must run the fixed-text `llama-results` top1 verifier and pass `same_top1 == n_tokens` before any performance benchmark. If top1 fails, revert or guard off that runtime capability and keep only rejected docs/artifacts.
9. Only after expanded up-skip correctness remains stable may a strict cold performance run be attempted. That run must use `MemoryMax=16000000000`, `MemorySwapMax=0`, `drop_caches`, full cgroup memory/page-cache logging, TTFT, prompt/eval token rates, counters, exact France output, and must be recorded as rejected unless it exceeds current SOTA and satisfies all gates.
10. Synchronous top768 direct prefill can be recorded as TTFT-rejected if useful for diagnosis, but cannot be promoted unless TTFT stays `<=33617.688744 ms`. Any promotable design must hide/overlap enough of the measured `1723.238 ms` prefill cost.
11. If a compliant new SOTA appears, stop exploration immediately and record: run path, full command/env, source head, branch, binary/library hashes, model/profile/pack hashes, memory stats including file page cache, `oom`/`oom_kill`, TTFT, prompt/eval token rates, counters, exact output, correctness decision, and enough reproduction steps for future rollback/replay. Commit and push source plus artifacts immediately to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then clean-rebuild and reproduce from pushed source before promotion.
12. If a candidate regresses throughput, violates RAM/page-cache, fails correctness, or exceeds TTFT gate for an accepted result, revert runtime source to the accepted SOTA path and keep only the rejected documentation/artifacts.

### 2026-07-04 Current Status Override

本节覆盖下面较早阶段的 `4.2 tok/s` 叙述；历史记录保留不改，最新执行以本节和文档尾部最新计划为准。

当前接受的 strict cold SOTA：

- `eval_tok_s=4.4`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Source/record branch: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- Latest pushed execution/audit head before the 2026-07-04 document update: `37c4981a015c68f4e1e132aa290679b491980df2` (`vendor-ds4: design cpu41 q80 hot residual`)
- Config: vendor DeepSeek, strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate-only one-stream (`ffn_gate_exps`), O_DIRECT gate expert pack, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, accepted profile/top-k envs, CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`
- Metrics: `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- TTFT gate for any future accepted SOTA remains `<=33617.688744 ms`
- Correctness answer for the accepted SOTA is semantic, coherent, and complete for `Please introduce France in a short paragraph.`

### 2026-07-04 Historical Active Plan Override

本节已被上方最新生效计划覆盖；内容只作为历史实验记录保留。

当前 accepted SOTA 仍为 `4.4 tok/s`，不是 lightning `4.5`：

- Accepted run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Accepted metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Accepted constraints: strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, vendor DeepSeek, `cpu_moe=40`, gate-only one-stream O_DIRECT pack, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`
- Promotion gate remains: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, 16GB RAM including page cache, no swap, France output coherent and semantically correct, and pushed-source reproduction passes.

Recent closed diagnostics:

- Lightning indexer no-source path is closed for now. Repeat candidate reached `4.5 tok/s` with acceptable TTFT once, but pushed-source reproduction only reached `4.4 tok/s` and had `TTFT=33618.90603 ms`, which is `1.217286 ms` over the acceptance gate. Therefore it is recorded as rejected/tie, not promoted.
- External MTP/draft path is closed for now. The available external DS4 MTP artifact is for a different Q2 model family and the reported implementation status is not compatible with the current vendor FP4/FP8 SOTA path. The accepted local model metadata still has no usable `mtp`, `draft`, `eagle`, `spec`, or `next` tensors.
- Re-running low-ceiling variants is prohibited unless a new hard-bound is written first: same-op serial/parallel page touch, synchronous direct/mmap staging, top-N up/down residency, route-specific/dedup CPU down prefetch, `mul_mat_id` src1 conversion skip, scalar Q8_0 CUDA one-stream, Q8_1 CUDA up/down stream, CPU repack/transient repack/hotset repack, CUDA graph, no-source ngram/lookahead/speculative, and chunk/affinity scheduler-only tuning are closed.
- Existing build/backend facts matter: the active build has CUDA and CPU backends only; OpenCL/Vulkan/SYCL are not built. `GGML_CUDA_MOE_STREAM_BATCH=OFF`, so the io_uring implementation in `moe_stream_batch.cu` is not on the accepted runtime path. The current one-stream CUDA path quantizes F32 activations to Q8_1, while the accepted CPU fallback uses MXFP4 x Q8_0 semantics. Generic CUDA has a Blackwell native FP4 MMQ path, but it is approximate relative to CPU fallback, not wired into one-stream, and cannot be promoted without token-level top1/output verification.

Fresh hard-bound for the next step:

- Current decode CPU up/down fallback is about `19029.457 ms`; accepted decode window estimate is about `31.04744671 s`.
- To reach `10 tok/s`, decode time must fall to about `13.6608765524 s`, so the required saving is about `17386.570 ms`.
- Removing all decode CPU up/down fallback gives a theoretical no-overhead ceiling of about `11.367 tok/s`, leaving only about `1642.887 ms` overhead budget.
- A top-N resident hotset under the current gate-cache/VRAM budget is not enough: best bounded nonduplicate allocation is only about `5.612 tok/s`.
- Streaming all up/down weights on demand would require about `90.6 GiB/s` effective source bandwidth before kernel, D2H, scatter, and sync overhead; io_uring or prefetch alone is therefore not a 10 tok/s path.

Current execution plan:

1. Keep the accepted `4.4 tok/s` runtime path unchanged unless a candidate first has a written hard-bound above the promotion gate. The completed and pushed audits now close the cheap paths: exact backend switch, source-only overlap, layout-only exact residency, Q8_0 compute without a source/cache solution, and generic VRAM/compression recovery.
2. Do not code another low-ceiling source/prefetch/scheduler/cache-size variant. Before every new experiment, update this plan with the bottleneck component, removable time, hard upper bound, expected RAM/page-cache pressure, expected TTFT impact, and rollback criteria.
3. The model-footprint VRAM recovery audit is complete. Pure `cpu_moe` tuning, lower `-ngl` dense/attention CPU placement, `cpu_moe=39/38`, and `cpu_moe=41` as a pure larger-gate-cache tradeoff are closed. However, `cpu_moe=41` frees about `3264 MiB` of expert VRAM and reopens exactly one design candidate: use that space for a top768 exact MXFP4 x Q8_0 hot residual up/down residency path while preserving the accepted gate cache.
4. The `cpu41 + top768 exact Q8_0 hot residual` design artifact is complete. The candidate has a positive but tight zero-overhead ceiling (`10.338 tok/s`, about `446 ms` slack) only if the new CPU layer and top768 hot entries are source-hidden. The next runtime step, if attempted, must be a default-off split-pool skeleton plus Q8_0 op/top1 verifier, not a performance benchmark.
5. Every logit-changing or numerically different path must first pass the fixed-text token-level top1 verifier or an equivalent deterministic correctness gate before any long performance run. For France, the final answer must remain semantically correct, coherent, and complete.
6. If a compliant new SOTA appears, stop exploration immediately and record: run path, full command/env, source head, branch, binary/library hashes, model/profile/pack hashes, memory stats including file page cache, `oom`/`oom_kill`, TTFT, prompt/eval token rates, counters, exact output, and correctness decision. Commit and push source plus artifacts immediately to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then clean-rebuild and reproduce from pushed source before promoting.
7. If a candidate regresses throughput, violates RAM/page-cache, fails correctness, or exceeds TTFT gate for an accepted result, revert runtime source to the accepted SOTA path and keep only the rejected documentation/artifacts.

### 2026-07-04 Exact Up/Down Backend Feasibility Audit

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/exact-updown-backend-feasibility-audit.json`.

Result: no existing built backend path is an immediate exact 10 tok/s candidate. This is a planning/audit result only; runtime source is unchanged and accepted SOTA remains `4.4 tok/s`.

Key findings:

- Accepted build has CUDA and CPU only; `GGML_CUDA_MOE_STREAM_BATCH=OFF`, OpenCL/Vulkan/SYCL are off, and only `libggml-base`, `libggml-cpu`, `libggml-cuda`, `libggml`, and `llama-cli` are present in the accepted build output.
- Accepted CUDA one-stream requires F32 `src1` and stages/quantizes activation to Q8_1. The accepted CPU MXFP4 fallback uses `ggml_vec_dot_mxfp4_q8_0` with Q8_0 activation semantics, so the existing one-stream Q8_1 path is not an exact replacement for CPU fallback up/down.
- Generic CUDA MMQ has a Blackwell native MXFP4 activation path, but it quantizes activation to FP4/MMQ storage, is not wired into one-stream expert-pack/cache, and is approximate relative to the CPU Q8_0 fallback until a token-level top1 verifier proves stability.
- The batch path contains direct/io_uring read support, but it is disabled in the accepted build and is an IO/source-submission mechanism only. It does not solve the measured on-demand up/down source requirement of about `90.6 GiB/s`.
- Therefore, the closed set now explicitly includes: one-stream Q8_1-as-exact, io_uring-only, batch-build-only, Blackwell FP4 performance run without top1/source hard-bound, and backend switch experiments without full validation.

Next plan after this audit:

1. Keep the accepted `4.4 tok/s` runtime path unchanged.
2. Before writing runtime code, choose exactly one remaining candidate class and write a hard-bound design: exact compact resident representation, verified exact/token-stable GPU MXFP4 x Q8_0 with source/cache solution, or predictive cross-op overlap.
3. Reject that candidate on paper unless it can plausibly remove at least `17386.570 ms` decode time while adding less than `1642.887 ms` overhead, keeping 16GB cgroup page-cache accounting and TTFT `<=33617.688744 ms`.
4. Any numerically different path must pass fixed-text token-level top1 verification before a strict cold performance benchmark.

### 2026-07-04 Predictive Source-Overlap Hard Bound

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/predictive-source-overlap-hard-bound.json`.

Result: source/page-only predictive overlap is rejected before runtime implementation. It does not close all overlap ideas, but it closes any candidate that only changes page/source fault timing without reducing hot CPU up/down compute.

Hard-bound result:

- Baseline decode up/down CPU fallback is `19029.457 ms`.
- Serial touch diagnostic leaves `3173.291 ms` decode fallback after source/page exposure is removed, so perfect source-only overlap can save at most `15856.166 ms`.
- That gives a zero-overhead decode token-rate ceiling of only `8.993 tok/s`, still `1530.404 ms` short of the `10 tok/s` decode window.
- Parallel touch residual gives an even lower zero-overhead ceiling of `8.697 tok/s`.
- Source overlap plus the entire previously closed scheduling/tail gap reaches only `10.105 tok/s` in a zero-overhead ideal with about `142 ms` slack, so it is not credible under cgroup refault pressure, thread contention, and TTFT gate.

Decision:

- Do not implement source-only predictive overlap, io_uring-only overlap, same-op page touch, or scheduler-gap-only tuning.
- The next viable class must combine source/page hiding with exact compute/offload, or use an exact compact resident representation, and must show `>17386.570 ms` decode saving with `<1642.887 ms` overhead before any runtime code is changed.

### 2026-07-04 Exact Compact Residency Hard Bound

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/exact-compact-residency-hard-bound.json`.

Result: layout-only exact compact residency is rejected before runtime implementation. This closes top-N/hotset residency as a primary path unless a new exact compression format or verified token-stable approximation changes the footprint.

Hard-bound result:

- Reaching `10 tok/s` requires removing about `17386.570 ms`, or `91.37%`, of the current decode up/down fallback.
- All-resident up/down decode payload is about `22.3295 GiB`; top2048 residency is about `8.1982 GiB` and saves only `12647.096 ms`.
- Optimistic interpolation says `10 tok/s` would need about `18.692 GiB` of exact up/down payload resident before any gate cache, workspace, D2H/scatter/sync, or cgroup refault cost.
- Keeping the current `13.25 GiB` gate cache plus that up/down payload would require about `31.942 GiB`, which does not fit current VRAM.
- Dedicating even `16 GiB` optimistically to up/down residency reaches only about `9.183 tok/s` with zero overhead and no gate penalty, and violates the 16GB host/page-cache rule if done through file cache.
- The best existing nonduplicate cache allocation remains only `5.612 tok/s`.

Decision:

- Do not implement layout-only exact residency, another top-N up/down hotset, or a gate-cache sacrifice that still has a sub-10 hard bound.
- Remaining source-code candidates must be combined compute+source designs: verified exact/token-stable GPU MXFP4 x Q8_0 plus source/cache solution, or a genuinely new exact compression/resident representation with measured footprint below the RAM/VRAM budget.

### 2026-07-04 Remaining Compute+Source Screening

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/remaining-compute-source-candidate-screening.json`.

Result: all cheap existing-path candidates are closed before another benchmark. The remaining path to `10 tok/s` is not an env flag or a cache-size sweep; it requires a new compute+source design.

Closed before source edit:

- Existing Q8_1 one-stream up/down reuse: fails token-level top1/full-output correctness and worsens cache/source behavior.
- Naive/scalar MXFP4 x Q8_0 CUDA one-stream: historical probe had `same_top1=136/145`, first mismatch at position 4, and compare runtime was far beyond the `1642.887 ms` overhead budget.
- Source-only overlap: zero-overhead ceiling only `8.993 tok/s`.
- Layout-only exact residency: needs about `18.692 GiB` exact up/down payload before gate/workspace.
- Backend/io_uring switch: current accepted build has no immediate exact path; io_uring is IO-only and batch is disabled.

Next required artifact before any runtime code:

- Write a concrete design for optimized exact MXFP4 x Q8_0 CUDA plus source/cache solution, or reject it on paper.
- That design must include arithmetic mapping to CPU fallback semantics, source/cache plan, expected overhead under `1642.887 ms`, fixed-text top1 verifier command, and rejection thresholds.

### 2026-07-04 Optimized Q8_0 Compute+Source Design

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/optimized-q80-compute-source-design.json`.

Result: optimized exact MXFP4 x Q8_0 CUDA is arithmetically designable, but it is rejected before runtime source changes because the current source/cache and VRAM allocation do not have a 10 tok/s ceiling after gate-cache penalty.

Design facts:

- CPU reference semantics are MXFP4 weight x Q8_0 activation: `ggml_vec_dot_mxfp4_q8_0`, scale `q8_0.d * mxfp4.e`, 32-element blocks.
- CUDA has reusable pieces (`quantize_f32_q8_0_block`, Q8_0 row quantization, MXFP4 nibble lookup, `dp4a`), but no existing MXFP4 x Q8_0 MMVQ path.
- A correct implementation would need a default-off CUDA helper accepting Q8_0 activation rows, a new `vec_dot_mxfp4_q8_0` device function, and fixed-text top1 verification before performance.

Hard-bound result:

- Perfect source/page overlap alone gives only `8.993 tok/s`.
- Combining source overlap with a resident top-K hot residual exact GPU subset only crosses 10 if `1.6-2.2 GiB` additional CUDA memory can be found without reducing gate cache, and even then slack is only about `61-243 ms` before any kernel/D2H/scatter/sync overhead.
- Current accepted runs have only about `238 MiB` CUDA free. If the hot up/down subset steals gate cache, the best zero-overhead combo is top256: `9.588 tok/s`, below target.

Decision:

- Do not implement Q8_0 CUDA kernel under the current VRAM allocation.
- Do not reduce gate cache to fund up/down residency unless a new bound exceeds `10 tok/s` after gate penalty and implementation overhead.
- Next candidate class is VRAM budget recovery or a genuinely new exact compression/resident representation, because compute-kernel work alone is not currently the limiting proof.

### 2026-07-04 VRAM Recovery And Exact Compression Audit

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/vram-recovery-exact-compression-audit.json`.

Result: VRAM budget recovery and generic exact compression do not reopen the Q8_0 compute+source path.

Key findings:

- Accepted VRAM shape is about `32109 MiB total = 238 MiB free + 17362 MiB self/model/context/compute + 14508 MiB unaccounted`.
- The large unaccounted region is gate-cache dominated; context/compute buffers are tiny. Reducing context to `-c 128` still left only about `237 MiB` free.
- Tiny independent down-cache probes with gate preserved either failed allocation, had only tiny fallback allocations, or produced `0.0%` useful hit rate.
- To make source-overlap + hot exact compute cross `10 tok/s` without reducing gate cache, the smallest useful hot subset needs about `1.6-2.2 GiB` raw payload. Fitting that into the current `238 MiB` free VRAM requires about `6.9x-9.1x` exact compression.
- zstd samples on packed expert payloads are only about `1.04x`, far below the required ratio; even a hypothetical compression path would have only `61-243 ms` zero-overhead slack for decompression/kernel overhead.

Decision:

- Do not pursue context-size VRAM probes, tiny independent up/down cache probes, generic exact compression, or Q8_0 kernel implementation without a new VRAM/source proof.
- Remaining nonclosed classes are higher-level: compatible multi-token/speculative path, a fundamentally different shared gate+up/down residency structure with a new hard-bound, or a model/runtime-level change that reduces the fixed GPU model footprint without changing correctness.

### 2026-07-04 Active Next Plan After VRAM Audit

Current accepted SOTA remains `4.4 tok/s`; runtime source is unchanged by the latest audits. The current pushed execution/audit head before this plan update is `37c4981a015c68f4e1e132aa290679b491980df2` on `ssd/vendor/deepseek-token-rate-16gb`.

The immediate bottleneck decision is now VRAM/source constrained, not kernel-arithmetic constrained:

- To make exact GPU up/down fallback removal useful, the current design needs at least about `1.6-2.2 GiB` additional usable CUDA memory while preserving the gate cache. The accepted run has only about `238 MiB` free.
- Context/compute buffers are too small to recover meaningful memory; `-c 128` still leaves only about `237 MiB` free.
- Generic exact compression is not useful; representative expert-pack samples compress only about `1.04x`, while the smallest useful hot subset would need about `6.9x-9.1x` compression to fit.
- Reducing gate cache to fund up/down residency is currently rejected because the best zero-overhead bound after gate penalty is below `10 tok/s`.

Next required artifact:

- Write `.Agent/runs/20260704-vendor-ds4-coldstart/model-footprint-vram-recovery-audit.json`.
- It must record whether accepted `cpu_moe=40` is already the minimal GPU model footprint, using existing evidence that `cpu_moe=39/38/37/36` increase model/self VRAM instead of freeing it.
- It must inspect whether moving dense/attention layers from GPU to CPU can free enough VRAM for a hot exact up/down subset, and must include a hard-bound for the CPU dense/attention cost before any runtime benchmark.
- It must reject this class unless the resulting design can free at least `1.6-2.2 GiB` usable CUDA memory, preserve enough gate-cache hit rate, keep host RAM including page cache under `16GB`, and keep accepted TTFT `<=33617.688744 ms`.

Execution rule for the next implementation attempt:

1. First update this plan and write the model-footprint audit artifact.
2. If the audit shows no credible `>10 tok/s` ceiling, do not run a benchmark for that path; mark it rejected and push the documentation.
3. If it shows a credible ceiling, run only a bounded diagnostic first under strict cgroup (`MemoryMax=16000000000`, `MemorySwapMax=0`) and strict cold `drop_caches`.
4. Any numerically different path must pass the fixed-text token-level top1 verifier before a long France run.
5. A result can be promoted only if it exceeds `4.4 tok/s`, has TTFT within gate, has `ram_ok=true` with file page cache counted, has coherent France output, is committed and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, and is reproduced from pushed source.

### 2026-07-04 Model-Footprint VRAM Recovery Audit

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/model-footprint-vram-recovery-audit.json`.

Result: pure model-footprint tuning is rejected, but the audit reopens one combined compute+source candidate. Runtime source is unchanged and accepted SOTA remains `4.4 tok/s`.

Closed findings:

- `--n-cpu-moe` semantics are explicit in `common/arg.cpp`: it keeps MoE weights of the first `N` layers on CPU. `cpu_moe=39/38` moves one or more MoE layers back to GPU, increasing model VRAM by about `3264 MiB` per layer and shrinking or breaking the gate cache. Existing strict runs dropped to `0.8-1.6 tok/s`.
- `cpu_moe=41` frees about one expert layer (`3264 MiB`) and allows a larger gate cache, but as a pure gate-cache tradeoff it was already rejected: strict cold `cpu41/vram14` reached only `1.6 tok/s`, and `cpu41/cache14336/t24/no-warmup` reached only `1.5 tok/s`.
- Lowering `-ngl` to move dense/attention layers to CPU is rejected before benchmark. GGUF tensor scan shows only about `136-166 MiB` dense payload per layer, so freeing `2.1-3.2 GiB` would require moving roughly `14-23` dense/attention layers to CPU. That adds CPU dense/attention work under a budget that has only about `1642.887 ms` total overhead after full fallback removal and risks TTFT/page-cache pressure.
- Old `8 tok/s`-class `cpu37/38/39/vram2` logs are not accepted SOTA evidence: they lack cold `drop_caches` and file-cache accounting, and later same-family cold-profile reruns with `drop_caches` reached only `1.2-1.3 tok/s`.

Reopened candidate:

- `cpu_moe=41` frees about `3264 MiB` while preserving enough VRAM for the accepted gate cache. That is exactly the payload for a top768 hot residual up/down set at `4.25 MiB` per tensor/expert slot.
- Under the serial-touch residual source model, source/page overlap alone leaves a decode window of about `15191.281 ms`. A top768 exact hot residual path saves about `2048.860 ms`. Adding the estimated source-hidden extra CPU layer cost from `cpu41` (`~72.041 ms`, late-layer residual average) gives a zero-overhead decode window of about `13214.463 ms`, or `10.338 tok/s`, with about `446 ms` slack to the `10 tok/s` line.
- If the newly CPU layer is not source-hidden, top768 still has only about `120 ms` zero-overhead slack, which is not enough for kernel/D2H/scatter/sync overhead. Therefore the candidate only remains open if source/page handling covers the new CPU layer too.
- This is not a promotable result; it is only a paper ceiling that justifies one concrete design step. Prior scalar Q8_0 CUDA probing failed token-level stability (`same_top1=136/145`), so correctness verification is the first hard gate.

Next execution plan:

1. Write a concrete `cpu41 + top768 exact Q8_0 hot residual` design artifact before runtime code.
2. Define the resident hotset layout and prove it fits in the `3264 MiB` freed by `cpu41` without reducing the accepted gate cache or increasing host page-cache pressure above the 16GB cgroup.
3. Define exact MXFP4 x Q8_0 CUDA arithmetic against the CPU fallback scale/order and the source/page plan for the extra CPU layer.
4. Run the fixed-text token-level top1 verifier before any strict cold France benchmark. If top1 mismatches, reject and revert source.
5. Promote only after strict cold `drop_caches`, `MemoryMax=16000000000`, `MemorySwapMax=0`, coherent France output, TTFT within gate, full run metadata, commit/push to `ssd/vendor/deepseek-token-rate-16gb`, and clean pushed-source reproduction.

### 2026-07-04 CPU41 Top768 Q8_0 Hot Residual Design

Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/cpu41-top768-q80-hot-residual-design.json`.

Result: design complete; only a default-off top1-first prototype is allowed. Runtime source is unchanged and accepted SOTA remains `4.4 tok/s`.

Design summary:

- Use `cpu_moe=41` to free about `3264 MiB` of GPU expert footprint, but do not treat `cpu41` itself as an optimization. Historical `cpu41` pure gate-cache runs reached only `1.5-1.6 tok/s`.
- Keep the accepted gate cache unchanged at `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`; add a separate fixed q80 hot pool of `768` slots (`3264 MiB`) for top residual up/down entries. A single shared LRU is not acceptable because it can evict gate entries and invalidate the no-gate-penalty bound.
- Hard-bound best case: source-only overlap leaves about `15191.281 ms` decode window; top768 residual saves about `2048.860 ms`; adding the estimated source-hidden extra CPU layer from `cpu41` gives about `13214.463 ms`, or `10.338 tok/s`, with only about `446 ms` zero-overhead slack.
- If the newly CPU layer is not source-hidden, the top768 bound has only about `120 ms` slack, too little for kernel/D2H/scatter/sync overhead. Therefore source-hidden async prefill is required.
- TTFT is a major risk: the accepted gate prefill reads `13.37 GB` in about `4.49 s`; a serialized top768 hot prefill adds about `3.42 GB`, estimated around `1.15 s`, while accepted TTFT slack is only about `725 ms`. A synchronous prefill mode may be recorded as TTFT-rejected but cannot be promoted.

Implementation gate:

1. First source change must be default-off and preserve accepted behavior when unset.
2. Generate/document the top768 up/down residual profile and pack source. The current gate pack alone is insufficient.
3. Implement split-pool allocation and prefill counters before enabling compute.
4. Implement exact MXFP4 x Q8_0 CUDA arithmetic for both down and up fallback paths. Down-only cannot reach the bound.
5. Run op compare and then fixed-text token-level top1 verifier before any strict cold performance run. Prior scalar Q8_0 probing failed `same_top1=136/145`, so correctness is the first likely failure point.
6. Only after top1 passes, run a strict cold diagnostic with full RAM/page-cache/TTFT/counter recording. Promote only if it exceeds current SOTA, stays under TTFT gate, and reproduces from pushed source.

Current bottleneck conclusion:

- Gate prefill/top3000 raised the accepted cold-start line from `4.2` to `4.4 tok/s`; this is the only currently accepted SOTA.
- Full current-SOTA CPU chunk trace shows fallback wall estimate about `27.072s`, chunk thread-average about `25.400s`, and only about `1.672s` scheduling/tail gap. Chunk scheduling, affinity, and overpartitioning are therefore closed for now.
- Source movement/pack/direct/page prefetch variants are closed for now by the async-bound artifact: measured bytes and direct-read bandwidth make top64/top128/top256 staging negative after overlap.
- No compatible local draft/MTP/NextN path exists; no-source speculative/lookahead/ngram paths are closed unless a compatible draft/MTP artifact appears.
- GPU up/down drift audit and lightweight sequential top1 verifier are complete. The verifier self-check passed under the 16GB cgroup, so the next active work may use token-level top1 matching before any performance benchmark.
- The current default-off DS4 hot-expert dual dispatch implementation is closed by the exact-up/down screening artifact: preserving the accepted gate cache leaves only about `238 MiB` CUDA free and gives only a `4.409 tok/s` no-overhead ceiling.
- The route-specific CPU down prefetch family is now closed: repeated prefetch tied at `4.4 tok/s` only after pushed-source reproduction, and deduplicated prefetch reduced advice volume but regressed to `4.1 tok/s`. Do not retry another synchronous `madvise`/page-touch variant without a genuinely new lower-pressure async source model and a new hard-bound.
- The `mul_mat_id` src1 conversion skip candidate is also closed by the 2026-07-04T05:16Z diagnostic: `convert_t0=0.002 ms/call` over `16920` calls, only `33.84 ms` total ideal savings, giving a no-overhead ceiling of about `4.405 tok/s`.
- Split up/down one-stream, scalar Q8_0 CUDA up, and no-source `ngram-map-k4v` have all been closed by correctness/performance gates. The current SOTA remains `4.4 tok/s`.
- Fresh SOTA hard-bound artifact `fresh-sota44-bottleneck-hard-bound.json` shows the only remaining non-speculative class with a 10 tok/s ceiling is exact full decode CPU up/down fallback removal: ideal ceiling `11.367 tok/s`, required decode saving `17386.570 ms`, and only `1642.887 ms` overhead budget after full fallback removal.
- The no-source touch-profile split diagnostic shows source/page exposure dominates cold CPU up/down fallback: decode fallback drops from `19029.457 ms` to `3173.291 ms` after active expert pages are touched. Synchronous touch is rejected because it adds `41626.545 ms` decode touch time, regresses to `2.6 tok/s`, and raises TTFT to `36167.863 ms`.
- The next active plan is a default-off bounded async/overlapped source-page preparation design for near-full decode up/down fallback. It may only change when pages are faulted, not logits or routing; it must preserve top1/output correctness, remain within the 16GB cgroup including page cache, and keep accepted TTFT within gate.
- Any future compliant result with `eval_tok_s > 4.4` must immediately be recorded with full reproducibility metadata, committed, pushed to `ssd/vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then reproduced from pushed source before promotion.

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
- 2026-07-04 lightweight sequential top1 verifier 已完成并 push：固定 France 文本在 accepted SOTA path 下 `same_top1=145/145`，`first_mismatch_pos=-1`，16GB cgroup 无 OOM。后续 GPU/offload 候选必须先过这个 verifier 或等价 token-level correctness gate。
- 当前最新计划：route-specific/dedup CPU down prefetch 已拒绝，`mul_mat_id` conversion skip 已因上界太小拒绝。下一步先做 fresh bottleneck table，把 CPU up/down fallback 拆成 compute、page/source、GPU-stream-decline/dispatch、sync/tail 几类；再按硬上界筛选 exact 候选。优先方向是：真正降低 CPU up/down fallback 的 exact offload 或低压力异步 source，而不是再做同步 page prefetch。
- 下一阶段目标：稳定超过 `4.4 tok/s`；未超过 `4.4 tok/s` 的结果只能作为 diagnostic/rejected/tie，不得 promote。
- 所有符合要求的新 SOTA 必须立刻记录完整复现信息并 push 到 `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`。记录必须足以未来从 push 后源码、profile、pack、runner 参数和 run artifact 完整复现。

## Current Baseline

- `current_pushed_head_before_this_update`: `c455d99a1` (`vendor-ds4: record invalid split top1 filter`)，已 push 到 `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`。
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

1. Treat `c455d99a1` as the current pushed documentation/source baseline before this plan update, while treating the accepted runtime behavior as the pushed `4.4 tok/s` SOTA path. Before a new source change, verify the worktree state and whether any probe source is still unaccepted.
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
  - required acceptance rate to beat the current accepted `4.4 tok/s` SOTA and to approach `10 tok/s`;
  - whether the implementation verifies target logits without changing final output correctness.
- If no compatible draft/MTP path exists, record that explicitly and do not spend more time on ngram-simple unless a new acceptance mechanism is designed.

### Phase 2: Short Speculative Diagnostic

- Only run if Phase 1 finds a compatible draft model or internal MTP path.
- Start with a short France diagnostic under strict 16GB cgroup and cold `drop_caches`.
- Record acceptance rate, target forward count, draft forward count, TTFT, RAM/page cache, exact output, and whether target verification preserves correctness.
- Reject immediately if output diverges, TTFT exceeds the 20% gate for an accepted result, or effective throughput is below the current SOTA class.

### Phase 3: Combined Fallback/Source Design If Speculative Is Not Viable

- Use existing 2026-07-04 trace artifacts to separate CPU fallback compute from page/source stalls more precisely before coding.
- Any new CPU/GPU fallback candidate must calculate a hard upper bound from measured removable time. A candidate whose bound is near the current accepted `4.4 tok/s` SOTA is not worth implementation.
- Preserve the accepted gate cache (`13568MiB`, `3192` slots, `86-87%` hit rate) unless the plan explicitly proves that sacrificing slots is compensated by a larger measured saving.
- Do not repeat rejected page-touch/top-N sweeps; top512 touch prewarm has already tied after pushed-source repro.

### Phase 4: SOTA Promotion Protocol

- If a candidate exceeds `4.4 tok/s` and passes RAM, correctness, TTFT, pack, and VRAM-cache gates, stop exploration immediately.
- Record exact run path, command/env, source head, build metadata, binary hashes, model/profile/pack hashes, memory stats including page cache, counters, TTFT, token rates, and France output.
- Commit and push source, plan, profiles, and artifacts to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`.
- Clean rebuild from the pushed source and rerun strict cold. Promote only if the pushed-source rerun still exceeds `4.4 tok/s` and passes every gate.
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

### 2026-07-03T23:51Z Trace-Disabled Fast Path Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/trace-disabled-fastpath-result.json`
- artifact sha256: `700a1df87cdb71ba14cc1659b37cae47f812f0c79840b03088704f7ee5880401`

Source status:

- Source patch was reverted after rejection.
- No trace-disabled fast-path source change is retained in the final worktree.

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T234245Z-20260704_trace_disabled_fastpath_candidate/france-cpu40-vram0gb`

Metrics:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=33188.274852 ms`
- TTFT gate: pass (`33188.274852 < 33617.688744`)
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15101935616`
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

- prefill: `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4662.410 pack_misses=0 read_failures=0`
- pack: `hits=4886 misses=0 reads=4886 bytes=21774204928 direct_reads=4886 direct_failures=0 direct_fallbacks=0`
- VRAM cache: `hits=33265 misses=1886 hit_rate=94.6%`

Artifact hashes:

- `summary.json`: `5dd8367076d23288c6b07293d624edcdbf65043e6b474626b2d29a7fd5d3cb09`
- `stdout.txt`: `537702d730b028fb485e89959f031dac66c8dd2ddd86c3ef9f18552d6c204005`
- `stderr.txt`: `982c1c06274fdba60a1764cb60a92eca7c23a3c2eff959081b38fb76f82912d2`
- `environment.txt`: `5335b6f306614bcd55b0adcb1d438fb8c18911844f21519877beda49676a8e18`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`

Verdict:

- Rejected/tie. Skipping no-trace timing and trace-lock overhead did not exceed the current `4.4 tok/s` SOTA.
- Current accepted SOTA remains `4.4 tok/s` from top3000 prefill.
- The next bottleneck is likely actual per-expert GPU/D2H/sync/scatter work, not trace instrumentation.

### 2026-07-04T00:20Z Next Plan: Deferred Per-Expert Stream Sync

Current accepted SOTA baseline:

- `eval_tok_s=4.4`
- strict cold start with `drop_caches`
- 16GB cgroup limit, including page cache
- `cpu_moe=40`
- `GGML_MOE_VRAM_CACHE_GB=0`
- `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
- gate-only one-stream cache with `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`
- `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`
- gate O_DIRECT pack:
  `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`
- run:
  `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- accepted TTFT gate remains `<= 33617.688744 ms`
- accepted SOTA commit is already pushed to `ssd/vendor/deepseek-token-rate-16gb`

Mandatory source/documentation rule:

- Before every new experiment, update this plan with the bottleneck hypothesis, expected upper bound, exact config, acceptance gates, rejection rules, and rollback behavior.
- When a new compliant SOTA appears, immediately record complete reproduction information, commit the source and artifacts, and push to the `ssd-llama` remote branch `vendor/deepseek-token-rate-16gb`.
- A promoted SOTA must be reproducible from the pushed branch. Promotion requires a pushed-source rerun with the same gates.
- Rejected experiments must still be recorded in this document and in `.Agent/runs/20260704-vendor-ds4-coldstart/`, but any rejected source patch must be reverted before push.
- Git author/committer for this work must be `L-Ark <fliangae@connect.ust.hk>`.

Bottleneck hypothesis:

- The previous no-trace and trace-disabled fast-path experiments tied or regressed, so trace file writing, trace mutex entry, and timestamp scaffolding are not the dominant remaining bottleneck.
- Cache hit-rate experiments showed that increasing protected runtime hits can improve counters without improving token rate, which means the hot path is likely dominated by actual per-expert GPU launch/synchronization, D2H copy, scatter, and CPU miss-side work.
- Current `ggml_cuda_moe_stream_one` has an existing deferred mode controlled by `GGML_MOE_STREAM_DEFER`. In non-deferred mode, each expert invocation can synchronize and scatter immediately. Deferred mode records the scatter work and lets `ggml_cuda_moe_stream_sync()` synchronize/scatter later at the CPU MoE boundary.
- This may reduce per-expert synchronization fragmentation and allow more outstanding CUDA stream work before a global boundary.

Hard-bound / expected upper limit:

- The France SOTA path has about `35151` gate expert invocations.
- If immediate per-expert synchronization/scatter contributes only `5 us` per invocation, the maximum gain is roughly `176 ms`, likely too small to move token rate materially.
- If synchronization fragmentation costs `20-50 us` per invocation under 20 CPU threads, the upper bound is roughly `0.7-1.8 s`.
- Given current eval throughput around `4.4 tok/s`, a realistic successful result would be `4.5-4.8 tok/s`; anything near or above `5.0 tok/s` would indicate that per-expert synchronization was a major bottleneck.
- If `GGML_MOE_STREAM_DEFER=1` changes ordering, cache-slot lifetime, or scatter timing incorrectly, correctness may fail even if throughput improves. Correctness must therefore be checked before any promotion.

Experiment design:

- First run a no-source-change strict cold candidate with `GGML_MOE_STREAM_DEFER=1`.
- Keep the accepted top3000 SOTA environment unchanged except for adding:
  `GGML_MOE_STREAM_DEFER=1`.
- Keep trace output enabled for the first diagnostic run so that counters and artifact parity are available, unless the run itself shows trace interaction with deferred mode.
- Do not change model, prompt, CLI batch settings, cgroup limit, pack path, cache size, prefill profile, or top-k settings.

Exact candidate command shape:

```bash
python3 .Agent/run-tools/strict_ds4_runner.py \
  --out-root /root/lfz/runs/vendor-ds4-16gb \
  --run-name 20260704_stream_defer_top3000_candidate \
  --case-name france \
  --cpu-moe 40 \
  --vram-cache-gb 0 \
  --memory-max-bytes 16000000000 \
  --ram-kill-threshold-bytes 16000000000 \
  --drop-caches-before-case \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env GGML_CUDA_DISABLE_GRAPHS=1 \
  --env GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39 \
  --env GGML_MOE_KEEP_TOPK_LAYER_VALUE=3 \
  --env GGML_MOE_KEEP_TOPK_UPDOWN=4 \
  --env GGML_MOE_STREAM=1 \
  --env GGML_MOE_STREAM_DEFER=1 \
  --env GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv \
  --env GGML_MOE_STREAM_DONTNEED=1 \
  --env GGML_MOE_STREAM_ONE_CACHE_MIB=13568 \
  --env GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1 \
  --env GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack \
  --env GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct \
  --env GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps \
  --env GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000 \
  --env GGML_MOE_STREAM_ONE_PREFILL_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv \
  --env GGML_MOE_STREAM_ONE_TRACE_OUT='{case_dir}/one_trace.csv' \
  --extra-arg=-c --extra-arg=256 \
  --extra-arg=-b --extra-arg=16 \
  --extra-arg=-ub --extra-arg=16 \
  --extra-arg=-t --extra-arg=20 \
  --extra-arg=-tb --extra-arg=20
```

Acceptance gates:

- `eval_tok_s > 4.4` on the first candidate.
- If the first candidate exceeds `4.4`, immediately commit/push docs/artifacts and rerun from the pushed source/branch before promotion.
- Final promoted SOTA must also have `eval_tok_s > 4.4` on pushed-source rerun.
- France answer must be semantically correct, coherent, and complete.
- `memory_peak_bytes <= 16000000000`, including page cache.
- `ram_limit_killed=false`, `oom_seen=false`, and `ram_ok=true`.
- `TTFT <= 33617.688744 ms`.
- O_DIRECT pack must remain clean: `direct_failures=0` and `direct_fallbacks=0`.

Rejection rules:

- Reject if throughput is `<= 4.4`, correctness fails, TTFT exceeds the gate, cgroup RAM limit is violated, the process is killed by the RAM guard, CUDA/OOM appears, or direct pack failures/fallbacks appear.
- Because this first attempt is config-only, rejection does not require a source revert.
- Record full metrics, prompt output, counters, artifact hashes, and final verdict in this document.
- Push the updated plan and rejection artifact to `ssd/vendor/deepseek-token-rate-16gb`.

Next actions after this experiment:

- If deferred sync improves throughput and passes all gates, inspect its trace/counters to decide whether to tune stream count, slot reuse, or deferred scatter batching.
- If deferred sync ties or regresses, treat synchronization deferral as not sufficient and move to deeper bottleneck localization:
  - split CPU miss-side read/dequant/compute time from GPU hit-side launch/D2H/scatter time;
  - quantify per-layer and per-expert latency distribution;
  - prioritize a true batched gate path only if the measured H2D/D2H/scatter cost dominates;
  - otherwise focus on CPU fallback miss path and pack read scheduling.

### 2026-07-04T00:07Z Deferred Per-Expert Stream Sync Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/stream-defer-top3000-result.json`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260703T235615Z-20260704_stream_defer_top3000_candidate/france-cpu40-vram0gb`

Config delta from accepted SOTA:

- Added `GGML_MOE_STREAM_DEFER=1`.
- Otherwise kept the accepted top3000 SOTA config.

Metrics:

- `eval_tok_s=null`
- `prompt_tok_s=null`
- `TTFT=null`
- `memory_peak_bytes=null`
- `memory_file_bytes=null`
- `memory_max_events=0`
- `ram_ok=false`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=false`
- correctness reason: `missing_france,missing_europe,missing_expected_context`

Observed state:

- The process exceeded the TTFT gate and produced no France answer.
- It was stopped manually after the run showed no semantic output progress.
- `stdout.txt` contains only the loading spinner and no answer text.
- `stderr.txt` confirms deferred mode was active:
  `[moe_stream] enabled (8 streams, GPU expert compute, defer_sync=1)`.
- Prefill completed:
  `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4602.643`.
- Pack direct path remained clean for the partial run:
  `direct_reads=3007 direct_failures=0 direct_fallbacks=0`.
- Partial cache counters before termination:
  `hits=25 misses=7 hit_rate=78.1%`.
- `one_trace.csv` has only `33` lines; the last expert row was:
  `31,4892.366,blk.4.ffn_gate_exps.weight,216,...,slot=7,...`.
- Resource samples stayed under the 16GB cgroup limit, with sampled memory around `15.96GB`.
- During the hang, GPU memory was about `31859 MiB`, GPU util was `0%`, and `llama-cli` was burning about one CPU core.

Diagnosis:

- Raw `GGML_MOE_STREAM_DEFER=1` is not valid as a config-only optimization for the current path.
- `ggml_cuda_moe_stream_one()` pushes a `deferred_scatter` record and intentionally keeps the stream slot `in_use` so `h_scratch` is not overwritten.
- `ggml_cuda_moe_stream_sync()` is called only after the CPU-side per-expert loop for the current `mul_mat_id` op.
- A single layer can need more than the `8` stream slots before that sync point. Once slots `0..7` are occupied, `acquire_slot()` spins waiting for a slot that cannot be released until the loop reaches `ggml_cuda_moe_stream_sync()`.
- This explains the stop at early `blk.4` trace rows and no generated answer.

Artifact hashes:

- `summary.json`: `8d28238db2cda2fa61f210b8e62e939408a3ac583b67b67c80e1adce59f60a09`
- `stdout.txt`: `464a16c0f8d91cd7c2cc0e9ad065ef6ee0dccc2fbfa224272f764eec769a4f00`
- `stderr.txt`: `1aac741247ebbdb512369b8d01d8bf17c79619d1c21363e0041d7402d390b53c`
- `environment.txt`: `546bfbddce3af2f588c7652bcfad5ccc3b2985e5d607bc85a70343f479c9d878`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `a98374a15d42906837209fed0cfced6143334de161f488a2f814c81de56c2fcf`
- `resource_samples.tsv`: `e6396b6fb61162bcb2e34cd1c70caadd0f8002878735ba8a6fa5093798d87ab7`

Verdict:

- Rejected. No token-rate result, no valid output, and TTFT gate failed.
- Current accepted SOTA remains `4.4 tok/s` from top3000 prefill.
- This failure is still useful: it proves that deferred sync can only be tested after adding a bounded flush/release mechanism.

### 2026-07-04T00:16Z Next Plan: Bounded Deferred Stream Flush

Goal:

- Convert raw deferred sync from an unsafe unlimited defer into a bounded batching experiment.
- The target behavior is to submit up to `MOE_STREAM_NSLOTS` expert jobs, then synchronize/scatter/release the completed pending slots before submitting more work.
- This keeps `h_scratch` lifetime correct while testing whether grouping several expert jobs before synchronization improves token rate.

Bottleneck:

- Current non-deferred SOTA synchronizes and scatters after every accepted expert invocation.
- Raw deferred mode proved that there are more accepted expert invocations per layer than available stream slots.
- A bounded flush can test whether the real bottleneck is per-expert immediate synchronization rather than cache misses or trace overhead.

Hard-bound / expected upper limit:

- The gate path has about `35151` expert invocations.
- Batching 8 at a time can at best reduce synchronization frequency by roughly `8x`, but it still pays one sync per occupied slot during flush and still scatters every result.
- If per-expert immediate synchronization costs `20-50 us`, a bounded flush could save roughly `0.6-1.5 s`.
- Realistic expected token rate if this helps is `4.5-4.8 tok/s`; a larger gain would imply the GPU work was previously serialized much more than expected.
- If GPU kernels are already small and CPU fallback/pack read dominates, bounded flush may tie or regress due to delayed scatter and extra bookkeeping.

Implementation plan:

- Add a helper that drains the current thread's `tls_pending` vector:
  - synchronize each pending slot's stream;
  - scatter from that slot's `h_scratch` into destination rows;
  - release the slot;
  - clear `tls_pending`.
- In `ggml_cuda_moe_stream_one()`, before acquiring a slot in deferred mode, call the drain helper when `tls_pending.size() >= MOE_STREAM_NSLOTS`.
- Keep the existing final `ggml_cuda_moe_stream_sync()` behavior by reusing the same drain helper.
- Do not change cache, pack, model, prompt, top-k, prefill, cgroup, CLI batch settings, or non-deferred behavior.
- Guard the candidate behind a new env, e.g. `GGML_MOE_STREAM_DEFER_BOUNDED=1`, so plain `GGML_MOE_STREAM_DEFER=1` behavior remains unchanged unless the new experiment flag is set.

Practice config:

- Build source candidate.
- Run strict cold top3000 SOTA config with:
  - `GGML_MOE_STREAM_DEFER=1`
  - `GGML_MOE_STREAM_DEFER_BOUNDED=1`
  - trace output enabled for the first candidate.
- Keep all accepted SOTA settings unchanged otherwise.

Acceptance gates:

- First candidate must finish and produce a coherent France answer.
- `eval_tok_s > 4.4`.
- If it exceeds `4.4`, commit/push source/docs/artifacts immediately and rerun from pushed source before promotion.
- Pushed-source rerun must also have `eval_tok_s > 4.4`.
- `memory_peak_bytes <= 16000000000`, including page cache.
- `ram_limit_killed=false`, `oom_seen=false`, and `ram_ok=true`.
- `TTFT <= 33617.688744 ms`.
- O_DIRECT pack must remain clean: `direct_failures=0` and `direct_fallbacks=0`.

Rejection rules:

- Reject if throughput is `<= 4.4`, correctness fails, TTFT exceeds the gate, cgroup RAM limit is violated, OOM appears, the process hangs, or direct pack failures/fallbacks appear.
- If rejected, revert the source patch and push only the docs/artifacts.
- Record metrics, output, counters, hashes, source diff summary, and root-cause verdict in this document.

### 2026-07-04T00:22Z Bounded Deferred Stream Flush Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/bounded-defer-top3000-result.json`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T001016Z-20260704_bounded_defer_top3000_candidate/france-cpu40-vram0gb`

Source patch tested:

- Added `GGML_MOE_STREAM_DEFER_BOUNDED`.
- Added a `moe_stream_drain_pending()` helper that synchronizes pending slots, scatters from `h_scratch`, releases slots, and clears `tls_pending`.
- In deferred mode, drained when `tls_pending.size() >= MOE_STREAM_NSLOTS` before acquiring another slot.
- Reused the same drain helper for final `ggml_cuda_moe_stream_sync()`.

Config delta from accepted SOTA:

- Added `GGML_MOE_STREAM_DEFER=1`.
- Added `GGML_MOE_STREAM_DEFER_BOUNDED=1`.
- Otherwise kept the accepted top3000 SOTA config unchanged.

Metrics:

- `eval_tok_s=4.3`
- `prompt_tok_s=1.8`
- `TTFT=31965.574447 ms`
- `elapsed_seconds=63.55`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15083008000`
- `memory_max_events=15846`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Counters:

- prefill: `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4578.906 pack_misses=0 read_failures=0`
- pack: `hits=4886 misses=0 reads=4886 bytes=21774204928 direct_reads=4886 direct_failures=0 direct_fallbacks=0`
- VRAM cache: `hits=33265 misses=1886 hit_rate=94.6%`
- trace rows: `35152`

Artifact hashes:

- `summary.json`: `32c2d3fd2f75cef0d826fc82f5d8f082e52d09fe8845a67afdfb81efb9f8fb2d`
- `stdout.txt`: `c4c20ae6467fc4d338ebe790d8d97fa198fa7c742de96a36a637cc945c8ea7f7`
- `stderr.txt`: `d833a22500672228e6288765efb23324b584ce5d2631e774e62502d3d65f6092`
- `environment.txt`: `b3d8adb5b1fd49944088cdc24dbd2860adc8c4a7e65c8d02f03712663d17f273`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `7844bafe72175526381bbf1b2bb9304bc794beacc5ee0b8fd4b234f7a5f8f2a9`
- `resource_samples.tsv`: `f3228e20d7ba563ae524b36f6beae1aa3bbcbc35c41b9209c52cfa11e14bd460`

Verdict:

- Rejected. The patch fixed the raw-defer hang and preserved correctness, RAM, pack direct, and TTFT gates, but token rate was `4.3`, below the accepted `4.4`.
- The source patch must not be retained. Revert `moe_stream.cu` and rebuild the SOTA binary before pushing final rejection docs.
- Current accepted SOTA remains `4.4 tok/s` from top3000 prefill.

Gap analysis:

- Bounded flush proved the slot-deadlock diagnosis correct, because the run completed and produced a valid answer.
- It did not improve throughput, which means per-expert immediate synchronization alone is not a large enough bottleneck at the current granularity, or the delayed drain/scatter bookkeeping offsets the benefit.
- Deferred trace rows do not include final drain sync/scatter timing, so the next step needs an explicit SOTA trace decomposition rather than relying on row totals alone.

### 2026-07-04T00:28Z Next Plan: SOTA Trace Bottleneck Decomposition

Goal:

- Locate the current `4.4 tok/s` SOTA bottleneck precisely before attempting another source optimization.
- Use existing accepted SOTA artifacts instead of running another model pass first.

Inputs:

- Accepted SOTA trace:
  `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb/one_trace.csv`
- Accepted SOTA stderr/summary:
  same run directory.
- Bounded-defer trace for comparison:
  `/root/lfz/runs/vendor-ds4-16gb/20260704T001016Z-20260704_bounded_defer_top3000_candidate/france-cpu40-vram0gb/one_trace.csv`

Analysis plan:

- Parse `one_trace.csv` with column names from the header.
- Break down totals and percentiles by:
  - layer;
  - `cache_hit`;
  - `cache_inserted`;
  - slot id;
  - `src0`, `src1`, `kernel`, `d2h`, `sync`, `scatter`, `dontneed`, and total.
- Compute total time in accepted SOTA rows and estimate which part is theoretically compressible.
- Compare SOTA non-deferred trace against bounded-defer trace to estimate how much immediate sync/scatter was moved out of row timing and whether it correlates with the observed `4.3` regression.

Decision rules:

- If SOTA row timing is dominated by cache-miss `src0`/pack read/H2D, next source plan should target miss-side IO/read scheduling or larger effective cache admission, not sync batching.
- If SOTA row timing is dominated by `kernel` or `src1`, next source plan should target fused/batched gate compute or src1 reuse.
- If SOTA row timing is dominated by `sync`/`scatter`, bounded-defer needs a more precise implementation with explicit drain timing and possibly per-layer batching.
- If trace row timing accounts for too little wall time, add explicit timers around CPU fallback, CPU-side expert loop, and `ggml_cuda_moe_stream_sync()`.

Deliverable:

- Write `.Agent/runs/20260704-vendor-ds4-coldstart/sota-trace-breakdown.json`.
- Append the numeric bottleneck result and next source experiment recommendation to this plan before making the next code change.

### 2026-07-04T00:34Z SOTA Trace Bottleneck Decomposition Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/sota-trace-breakdown.json`
- artifact sha256: `f6678bb53e3c737639b6a09f85c6f6ab4c60659278918cf19edaf08fe0cba80e`

Inputs:

- Accepted SOTA:
  `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Bounded-defer candidate:
  `/root/lfz/runs/vendor-ds4-16gb/20260704T001016Z-20260704_bounded_defer_top3000_candidate/france-cpu40-vram0gb`

Accepted SOTA summary:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=32892.55329 ms`
- `elapsed_seconds=63.94`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15102607360`
- `ram_ok=true`
- `correctness_ok=true`

SOTA trace totals:

- row count: `35151`
- trace span: `44636.433 ms`
- sum of row `total_ms`: `8050.224 ms`
- component sums:
  - `src0_ms=6707.771 ms` (`83.3%` of row total)
  - `sync_ms=628.240 ms` (`7.8%`)
  - `kernel_ms=317.330 ms` (`3.9%`)
  - `src1_ms=162.351 ms` (`2.0%`)
  - `d2h_ms=105.292 ms` (`1.3%`)
  - `dontneed_ms=74.877 ms` (`0.9%`)
  - `scatter_ms=49.787 ms` (`0.6%`)

Important refinement:

- `seq=0` includes the top3000 prefill and contributes about `4502 ms` inside `src0_ms`.
- Excluding `seq=0`, SOTA visible row time is only `3547.910 ms`.
- Excluding `seq=0`, component sums are:
  - `src0_ms=2206.068 ms` (`62.2%`)
  - `sync_ms=628.234 ms` (`17.7%`)
  - `kernel_ms=316.797 ms` (`8.9%`)
  - `src1_ms=162.300 ms` (`4.6%`)
  - `d2h_ms=105.283 ms` (`3.0%`)
  - `dontneed_ms=74.867 ms` (`2.1%`)
  - `scatter_ms=49.785 ms` (`1.4%`)

Cache-hit/miss split:

- SOTA cache hits: `33265`
- SOTA cache misses: `1886`
- Miss-only rows:
  - total `2536.444 ms`
  - `src0_ms=2187.637 ms` (`86.2%`)
  - `sync_ms=297.174 ms` (`11.7%`)
- Hit rows excluding `seq=0`:
  - total `1011.466 ms`
  - `sync_ms=331.060 ms` (`32.7%`)
  - `kernel_ms=287.413 ms` (`28.4%`)
  - `src1_ms=154.883 ms` (`15.3%`)
  - `d2h_ms=99.049 ms` (`9.8%`)
  - `dontneed_ms=68.955 ms` (`6.8%`)
  - `scatter_ms=47.207 ms` (`4.7%`)

Bounded-defer comparison:

- Bounded-defer `eval_tok_s=4.3`, delta `-0.1 tok/s`.
- Bounded-defer TTFT was `31965.574447 ms`, about `927 ms` lower than SOTA.
- Bounded-defer row `total_ms` sum was `7413.899 ms`, about `636 ms` lower than SOTA because sync/scatter moved outside row trace.
- Despite lower row-visible time, wall throughput regressed, proving that row-local timing alone is not enough for the next batching change.

Conclusion:

- `src0_ms` dominates visible SOTA row timing, but a large part is `seq=0` prefill and therefore mostly TTFT-side.
- After excluding prefill, trace-visible time is only about `3.55s`, far below the `63.94s` run wall time and the `44.64s` trace span.
- The next bottleneck is likely outside the per-row CUDA trace:
  - CPU-side expert loop overhead;
  - CPU fallback path for non-accepted experts/tensors;
  - final stream sync/drain timing;
  - thread barrier/wait time;
  - file-backed page-cache/refault behavior not captured as individual row timing.
- Do not start another CUDA batching optimization until these wall-time gaps are instrumented.

### 2026-07-04T00:41Z Next Plan: CPU-Side MoE Wall-Time Instrumentation

Goal:

- Add low-overhead optional timers around the CPU-side MoE path to account for wall time missing from `one_trace.csv`.
- The purpose is diagnostic first: identify the next high-leverage token-rate optimization without changing default SOTA behavior.

Bottleneck hypothesis:

- Accepted SOTA row trace accounts for only about `8.05s` total row work, and only `3.55s` after removing the prefill row.
- The model run wall time is `63.94s`, and trace span is `44.64s`.
- Therefore the current bottleneck is probably CPU-side loop/fallback/barrier/wait behavior around the streamed experts rather than row-local CUDA sync/scatter alone.

Implementation plan:

- Add optional profiling gated by a new env, e.g. `GGML_MOE_STREAM_CPU_TRACE_OUT`.
- In `ggml/src/ggml-cpu/ggml-cpu.c`, around the `use_gpu_stream` branch:
  - time the whole CPU-side streamed MoE block for `ith == 0`;
  - time the per-expert loop;
  - count accepted vs declined `ggml_cuda_moe_stream_one()` calls;
  - time `ggml_cuda_moe_stream_sync()`;
  - time the post-CUDA threadpool barrier;
  - record `src0->name`, `n_as`, total rows, accepted count, declined count, and elapsed microseconds.
- Keep the profiler disabled by default and write only when env is set.
- Do not change compute behavior, cache policy, pack path, prefill, or default SOTA path.

Practice config:

- Build source candidate with instrumentation.
- Run strict cold accepted SOTA top3000 config with the new CPU trace env set.
- Keep `GGML_MOE_STREAM_ONE_TRACE_OUT` enabled so row trace and CPU trace can be aligned.

Acceptance / rejection:

- This is diagnostic, not a SOTA promotion unless it unexpectedly improves throughput.
- It must preserve correctness, RAM, and pack direct behavior.
- If token rate drops materially due to profiling overhead, mark the run diagnostic-only and do not treat it as SOTA.
- If the instrumentation changes behavior or breaks correctness, revert source immediately.

Deliverable:

- `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-moe-walltrace-result.json`
- Append the wall-time breakdown and next optimization recommendation to this plan before any subsequent source optimization.

### 2026-07-04T00:45Z CPU-Side MoE Wall-Time Instrumentation Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-moe-walltrace-result.json`
- artifact sha256: `8db82a6ef6f2976b8062884474cec41bf1c79411c988b7e45f4efa1017d7f403`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T002803Z-20260704_cpu_moe_walltrace_top3000_diagnostic/france-cpu40-vram0gb`

Source status:

- Temporary diagnostic source added `GGML_MOE_STREAM_CPU_TRACE_OUT`.
- The diagnostic source was reverted after the run.
- Build directory was rebuilt after revert; CMake reports clean commit `68e82af87`.

Metrics:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=33628.923137 ms`
- TTFT gate: fail by about `11.234 ms` versus `33617.688744 ms`
- `elapsed_seconds=64.70`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15082663936`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Artifact hashes:

- `summary.json`: `fbb662dbbfc59ae9c350696da6721c9301e7b293d637a58d2f2628a7cc8a8e72`
- `stdout.txt`: `ca34d58b964622213783d1036f91bafcfad230eef2f9b972e19bda7818650aec`
- `stderr.txt`: `c0e36f151a24ed94617c42116bba9b7625ecfa58828e2802d0f169b1b1778e32`
- `environment.txt`: `f0e689b6dd6aa28d5ddf1959f61f22c1cd5e6b0858d9389559ab040c5a123b23`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `39f3d55852899b0b5f7f960b3d282d5cf6be0405e4b7f71eaf6f8cd8810c6895`
- `cpu_moe_walltrace.csv`: `109415577787ea0f17c3f0ab03b5af94537d25c56141f311432bb074403475ec`
- `resource_samples.tsv`: `23d2e048dcce6351edf4fcde7bb08ba0cfeb1e393633ddd2958144f5eacc9aa7`

CPU walltrace aggregate:

- CPU walltrace rows: `16920`
- trace span: about `40.13s`
- stream branch total time sum: `8.590s`
- rows before stream branch: `76000`
- accepted rows: `36480` (`48%`)
- declined rows: `39520` (`52%`)
- remaining rows after stream branch: `39520`

By role:

- gate:
  - ops `5640`
  - rows before `36480`
  - accepted rows `36480`
  - declined rows `0`
  - stream branch total `8.570s`
- up:
  - ops `5640`
  - rows before `19760`
  - accepted rows `0`
  - declined rows `19760`
  - stream branch total only `0.009s`
- down:
  - ops `5640`
  - rows before `19760`
  - accepted rows `0`
  - declined rows `19760`
  - stream branch total only `0.011s`

Conclusion:

- Current SOTA streams only `ffn_gate_exps`.
- `ffn_up_exps` and `ffn_down_exps` are declined by the stream path and fully handled by CPU fallback.
- This explains why `one_trace.csv` accounts for only a small fraction of total wall time.
- The largest remaining token-rate opportunity is no longer gate cache sync/scatter; it is reducing or eliminating up/down CPU fallback under the 16GB RAM and TTFT gates.

Verdict:

- Diagnostic-only. It did not produce a new accepted SOTA because TTFT narrowly failed and token rate only tied at `4.4`.
- Source instrumentation was reverted and not retained.
- Current accepted SOTA remains `4.4 tok/s`.

### 2026-07-04T00:53Z Next Plan: Config-Only Up/Down GPU Stream Probe

Goal:

- Test whether moving `ffn_up_exps` and `ffn_down_exps` from CPU fallback to the existing GPU stream path improves token rate.
- Avoid increasing VRAM cache footprint by keeping cache admission restricted to the current gate profile.

Bottleneck:

- CPU walltrace shows `39520` rows (`52%`) are declined by the stream path.
- All declined rows are up/down experts:
  - up declined rows: `19760`
  - down declined rows: `19760`
- These rows are then handled by CPU fallback and dominate the wall time outside `one_trace.csv`.

Theory / hard-bound:

- Accepted SOTA gate stream branch costs about `8.57s` and handles `36480` rows.
- Up/down CPU fallback rows are `39520`, slightly more than gate streamed rows.
- If GPU streaming up/down is correct and its H2D/kernel/D2H cost is similar to or moderately above gate stream cost, token rate could materially improve.
- The theoretical upper bound is large because removing most CPU fallback could reduce a large portion of the `~40s` trace span; a practical first success would be `>4.4 tok/s`, with `5-7 tok/s` plausible if CPU fallback is the dominant wall-time component.
- Risk: up/down streaming without a pack/cache may trigger large mmap/page-cache reads and H2D copies, hurting cold-start TTFT or token rate.

Experiment design:

- No source change.
- Keep accepted top3000 SOTA settings except remove the name filter:
  - do not pass `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`.
- Keep `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE` and `GGML_MOE_STREAM_ONE_PREFILL_PROFILE` on the current gate profile.
- Expected behavior:
  - gate experts stay cached/prefilled as before;
  - up/down experts become stream-eligible;
  - up/down experts are not admitted into the VRAM cache because the admit profile contains gate entries only;
  - up/down will use direct model mmap pages rather than the gate pack when no pack entry exists.

Practice config:

- strict cold `drop_caches`;
- 16GB cgroup including page cache;
- same `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, top-k envs, prefill limit `3000`, gate pack, and CLI args;
- trace enabled to verify up/down stream acceptance and correctness.

Acceptance gates:

- `eval_tok_s > 4.4`.
- France output must be semantically correct, coherent, and complete.
- `memory_peak_bytes <= 16000000000`, including page cache.
- `ram_limit_killed=false`, `oom_seen=false`, and `ram_ok=true`.
- `TTFT <= 33617.688744 ms`.
- O_DIRECT direct failures/fallbacks must remain `0`.
- Pack misses for up/down are expected and are not by themselves rejection, because this probe intentionally uses the existing gate pack only.

Rejection rules:

- Reject if token rate is `<=4.4`, correctness fails, TTFT fails, RAM fails, OOM appears, or direct failures/fallbacks appear.
- If rejected, record metrics, output, counters, hashes, and reason in this document and push docs/artifacts only.
- If accepted, immediately commit/push all reproduction records and rerun from pushed branch before promoting.

### 2026-07-04T00:58Z Config-Only Up/Down GPU Stream Probe Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/updown-stream-nofilter-result.json`
- artifact sha256: `d6f604863966256f31a20bba8b6ae0241c6074022043ffde8061ee57c847fead`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T003804Z-20260704_updown_stream_nofilter_top3000_probe/france-cpu40-vram0gb`

Config delta from accepted SOTA:

- Removed `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`.
- Kept gate cache admission profile and gate prefill profile.
- Kept current gate O_DIRECT pack.
- No source change.

Metrics:

- `eval_tok_s=1.8`
- `prompt_tok_s=1.2`
- `TTFT=36713.048232 ms`
- `elapsed_seconds=140.07`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15064793088`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- runner `correctness_ok=true`, but manual correctness rejected because the output ends with an incomplete trailing sentence.

Output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. Renowned for its art, fashion, cuisine, and landmarks like the Eiffel Tower, the Louvre, and the Palace of Versailles, France is a major center for culture and politics. It is a unitary semi-presidential republic with a strong democratic tradition, and its capital, Paris, is often called the "City of Light." Beyond its metropolitan borders, France also includes overseas regions and territories, making it a truly global nation. The country is famous for its wine regions, such as Bordeaux and Burgundy, as well as its iconic landmarks and world-renowned cuisine. France is also known for its rich history, including the French Revolution and its role in both World Wars. Today, it remains a global leader in art, fashion, and culture, with a thriving economy and
```

Counters:

- prefill: `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4552.723`
- pack: `hits=5545 misses=53774 reads=5545 bytes=24711004160 direct_reads=5545 direct_failures=0 direct_fallbacks=0`
- VRAM cache: `hits=43544 misses=56319 hit_rate=43.6%`
- trace rows: `99863`

Trace role split:

- gate:
  - rows `47869`
  - cache hits `43544`
  - cache misses `4325`
  - `src0_ms=14224.820`
  - `total_ms=16384.508`
- up:
  - rows `25997`
  - cache hits `0`
  - cache misses `25997`
  - `src0_ms=43907.096`
  - `total_ms=49436.853`
- down:
  - rows `25997`
  - cache hits `0`
  - cache misses `25997`
  - `src0_ms=43057.754`
  - `total_ms=48564.644`

Artifact hashes:

- `summary.json`: `e0a9ef0ee418f6a479f3cc84a4ba9e03826748c5495c875c9e917644d1b6ba0e`
- `stdout.txt`: `cb8c30843a6140d6aea5ad69126639dc869c15febe50fc7a99b4c50dbd9b4019`
- `stderr.txt`: `12fe183f69defe975a2ebce39423cc6feaf2e2f68deea5d62a51309da4bfb144`
- `environment.txt`: `e02418d2946c4cb6e0c34261cb15ab66b0c52e89651daedf77dce78cc71dfd21`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `271c03c23146f3c0ce3bf4776792095dc221a7cb25e6a95772bf87cf2f87e360`
- `resource_samples.tsv`: `c03198be8cae5079377db756394cc0e411adaf611feaa2f661830482bc97ab5f`

Hot expert coverage from this trace:

- up:
  - total rows `25997`, unique expert keys `3396`
  - top64 covers `28.79%`
  - top128 covers `40.45%`
  - top192 covers `47.26%`
  - top384 covers `60.20%`
  - top512 covers `66.09%`
- down has the same coverage as up because routed up/down expert ids match.
- gate:
  - total rows `47869`, unique expert keys `5204`
  - top192 covers `38.33%`
  - top512 covers `57.07%`

Verdict:

- Rejected. Full up/down streaming without pack/cache is much slower than CPU fallback under the cold 16GB cgroup.
- The bottleneck moved to up/down `src0_ms` and page-backed H2D; up/down had zero cache hits.
- Current accepted SOTA remains `4.4 tok/s`.

Conclusion:

- Do not stream all up/down experts.
- A plausible next direction is selective profile-gated up/down streaming:
  - cache/stream only hot up/down expert keys;
  - leave cold up/down on CPU fallback;
  - trade a small number of gate cache slots for hot up/down slots only if hit coverage justifies it.

### 2026-07-04T01:04Z Next Plan: Profile-Gated Hot Up/Down Stream

Goal:

- Test a selective source change that streams only experts present in an admission profile.
- Use it to stream/cache hot up/down experts while leaving cold up/down on CPU fallback.

Bottleneck:

- Full up/down stream was rejected because up/down had `0` cache hits and about `87s` combined `src0_ms`.
- CPU walltrace showed up/down CPU fallback is the main trace-external bottleneck.
- Hot expert coverage is nontrivial: top192 up/down keys cover about `47%` of up/down stream rows each.

Theory / hard-bound:

- If `192` hot up and `192` hot down keys are cached, they could cover roughly `47%` of up/down rows in this prompt.
- That would reduce CPU fallback rows without paying the full no-cache H2D cost for cold up/down.
- Cost: about `384` VRAM cache slots, roughly `1.6GB`, likely taken from gate cache capacity.
- Benefit must exceed any gate-hit loss and cold prefill cost.
- Because current gate cache hit rate is high (`94.6%`), the first test should be conservative and keep most gate slots.

Implementation plan:

- Add a default-off env, e.g. `GGML_MOE_STREAM_ONE_REQUIRE_ADMIT=1`.
- When enabled, `ggml_cuda_moe_stream_one()` should return `false` before acquiring a slot if `moe_stream_cache_admit_allows(src0_name, expert_index)` is false.
- Keep default behavior unchanged when the env is unset.
- Build a combined profile from current accepted SOTA/no-filter traces:
  - include the hottest gate keys first;
  - include a small hot up/down budget, starting with top96 up + top96 down or top128 up + top128 down;
  - keep total prefill limit at `3000` initially.

Practice config:

- strict cold `drop_caches`;
- 16GB cgroup including page cache;
- `GGML_MOE_STREAM_ONE_NAME_FILTER` unset so up/down can be eligible;
- `GGML_MOE_STREAM_ONE_REQUIRE_ADMIT=1`;
- combined gate+hot-up+hot-down admit/prefill profile;
- current gate pack remains enabled; up/down pack misses are expected unless a combined pack is later built.

Acceptance gates:

- `eval_tok_s > 4.4`.
- France output must be complete, coherent, and semantically correct.
- `memory_peak_bytes <= 16000000000`, including page cache.
- `ram_limit_killed=false`, `oom_seen=false`, and `ram_ok=true`.
- `TTFT <= 33617.688744 ms`.
- Direct failures/fallbacks must be `0`.

Rejection rules:

- Reject if token rate is `<=4.4`, correctness fails, TTFT fails, RAM fails, OOM appears, or direct failures/fallbacks appear.
- If rejected, revert source and push docs/artifacts only.
- If accepted, immediately commit/push source/docs/artifacts and rerun from pushed branch before promotion.

### 2026-07-04T01:12Z Refined Hot Up/Down Profile Plan

Reason for refinement:

- `one_prefill_maybe()` can prefill only entries available in the expert pack.
- The current pack is gate-only, so putting hot up/down entries into `GGML_MOE_STREAM_ONE_PREFILL_PROFILE` would create pack misses and reduce useful gate prefill.
- The first hot up/down test should therefore keep the SOTA gate-only prefill profile and use hot up/down only for runtime stream/cache admission.

Exact first candidate:

- Add default-off stream eligibility profile support:
  - `GGML_MOE_STREAM_ONE_REQUIRE_ADMIT=1`
  - `GGML_MOE_STREAM_ONE_REQUIRE_PROFILE=<path>`
- If `GGML_MOE_STREAM_ONE_REQUIRE_ADMIT=1`, `ggml_cuda_moe_stream_one()` returns `false` before acquiring a stream slot unless the tensor/expert key exists in the require profile.
- If `GGML_MOE_STREAM_ONE_REQUIRE_PROFILE` is unset, fall back to the existing cache-admit profile for compatibility.
- Keep `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE` separate from stream eligibility.

Profiles:

- Build stream allow profile:
  `.Agent/profiles/vendor-ds4/hot-updown64-stream-allow.tsv`
  - all gate keys observed in the no-filter diagnostic trace;
  - top64 hot up keys from the no-filter diagnostic trace;
  - top64 hot down keys from the no-filter diagnostic trace.
- Build cache admit profile:
  `.Agent/profiles/vendor-ds4/hot-updown64-cache-admit.tsv`
  - existing current SOTA gate cache profile;
  - top64 hot up keys;
  - top64 hot down keys.
- Keep prefill profile unchanged:
  `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`
- Keep prefill limit unchanged:
  `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`.

Why top64 first:

- Top64 up covers `28.79%` of up rows; top64 down covers `28.79%` of down rows.
- This costs about `128` extra cache slots if all hot up/down keys are inserted, leaving some room beyond the `3000` gate prefill inside the `3192` slot cache.
- Top128 or top192 may be tested later if top64 shows a positive trend without hurting gate hit-rate or TTFT.

Expected behavior:

- Gate keeps SOTA prefill behavior.
- Gate keys outside the cache-admit profile can still stream, because the stream allow profile includes all observed gate keys.
- Hot up/down keys stream and are admitted to VRAM cache after first use.
- Cold up/down keys decline from stream and remain on CPU fallback.

Acceptance gates:

- Same as before: `eval_tok_s > 4.4`, correct complete France answer, 16GB cgroup including page cache, TTFT `<=33617.688744 ms`, no OOM, no direct failures/fallbacks.

Rejection rules:

- If this does not exceed `4.4 tok/s`, or if correctness/TTFT/RAM/direct gates fail, revert the source patch and push only docs/profiles/artifacts.

### 2026-07-04T01:18Z Profile-Gated Hot Up/Down64 Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/hot-updown64-require-profile-result.json`
- artifact sha256: `53b688964e6ba1654661285979696d0375576f44dc23b69177130239e91b54eb`

Profiles:

- stream allow:
  `.Agent/profiles/vendor-ds4/hot-updown64-stream-allow.tsv`
  - sha256: `6388dc6b814c2fa75c3b9df35c8de81f20f4912699fbe2f1ff39f702837aadb2`
- cache admit:
  `.Agent/profiles/vendor-ds4/hot-updown64-cache-admit.tsv`
  - sha256: `cb53cdc2a19d8dd5c56f9f3d34cfedaedd80e041320bc9dae182b5e65ecf94c6`
- profile summary:
  `.Agent/profiles/vendor-ds4/hot-updown64-profile-summary.json`
  - sha256: `c312e6113be8f83e9704d942cc18ce7add2e1ec0849133b838ec2a844d55e426`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T005617Z-20260704_hot_updown64_require_profile_candidate/france-cpu40-vram0gb`

Source patch tested:

- Added default-off `GGML_MOE_STREAM_ONE_REQUIRE_ADMIT`.
- Added optional `GGML_MOE_STREAM_ONE_REQUIRE_PROFILE`.
- When enabled, `ggml_cuda_moe_stream_one()` declines before stream-slot acquisition if the tensor/expert key is absent from the require profile.

Source status:

- Rejected source patch was reverted.
- Build directory was rebuilt after revert; CMake reports clean commit `6e1439db0`.

Metrics:

- `eval_tok_s=3.6`
- `prompt_tok_s=1.8`
- `TTFT=33432.591735 ms`
- `elapsed_seconds=86.09`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15056429056`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- runner `correctness_ok=true`, but manual correctness fails.

Manual correctness:

- Rejected. The answer is complete and coherent, but it contains a semantic error:
  it says France is often called the "City of Light"; that nickname applies to Paris, not France.

Output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is renowned for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is famous for its cuisine, wine, and fashion, and is often called the "City of Light" due to its historical and cultural significance. The country is a major economic and political power in Europe, with a strong industrial and agricultural sector. It is also known for its art, literature, and philosophy, and has been a center for European culture for centuries. France is a unitary state with a presidential system of government, and its capital is Paris. The official language is French, and the currency is the Euro. The country is known for its rich history, diverse landscapes, and vibrant culture, making it a popular destination for tourists worldwide.
```

Counters:

- prefill: `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4847.512 pack_misses=0 read_failures=0`
- pack: `hits=5645 misses=1131 reads=5645 bytes=25156648960 direct_reads=5645 direct_failures=0 direct_fallbacks=0`
- VRAM cache: `hits=57702 misses=3776 hit_rate=93.9%`
- trace rows: `61478`

Trace role split:

- gate:
  - rows `46912`
  - cache hits `43264`
  - cache misses `3648`
  - cache inserted `863`
  - `src0_ms=11591.145`
  - `total_ms=13618.311`
- up:
  - rows `7283`
  - cache hits `7219`
  - cache misses `64`
  - cache inserted `64`
  - `src0_ms=283.361`
  - `total_ms=547.662`
- down:
  - rows `7283`
  - cache hits `7219`
  - cache misses `64`
  - cache inserted `64`
  - `src0_ms=287.290`
  - `total_ms=583.957`

Artifact hashes:

- `summary.json`: `21b39f5f5a3fe050dd0c4e5bb02f2b45ae3e03ac53d1a653f4aa734692c01519`
- `stdout.txt`: `15f3681ac8fe32517f76f623eb26165bb0260dac21ca59f988dc99675c273b62`
- `stderr.txt`: `37d90b38f993c03a0e3f93ba42771c1e58e3575156fd54b71b71af981c7ba591`
- `environment.txt`: `9850e3bcb643b22512e9cfd29957a292123e638d9ce78768f2a2fc46181cc56c`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `897b09b1e6c4a988068b7e84f98c78f1c19e91855e59ccd91774a40d6e037a2a`
- `resource_samples.tsv`: `c49d1348e7a171a92692b68071f2c5e4816c141ad3087a7cddb5468a2951f9e3`

Verdict:

- Rejected. Token rate was below SOTA and manual correctness failed.
- Current accepted SOTA remains `4.4 tok/s`.

Gap analysis:

- The profile gating mechanism worked: hot up/down each had only `64` cold misses and `7219` cache hits.
- The performance was still lower than SOTA, and output quality changed.
- Therefore the next up/down GPU work must first prove numerical/output correctness before trying larger hot sets, up/down packs, or cache split policies.

### 2026-07-04T01:26Z Next Plan: Up/Down GPU Numerical Audit

Goal:

- Determine whether the existing GPU stream implementation for `ffn_up_exps` and `ffn_down_exps` is numerically close enough to CPU fallback.
- Do this before spending more work on up/down pack/cache optimizations.

Bottleneck:

- CPU fallback for up/down is the main remaining wall-time bottleneck.
- However, both full up/down stream and hot64 up/down stream changed output quality.
- That may be because up/down GPU math differs from CPU fallback, or because the changed execution path perturbs generation enough despite acceptable numeric error.

Experiment design:

- Use existing `GGML_MOE_STREAM_COMPARE_CPU_OUT` compare machinery.
- Run a short diagnostic, not a SOTA candidate:
  - no source change if possible;
  - no-filter stream so up/down experts are accepted by GPU;
  - `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`;
  - small compare limit such as `64`;
  - short generation budget to reduce run time.
- Inspect max/mean absolute and relative error separately for gate/up/down.

Decision rules:

- If up/down GPU error is materially larger than gate error, fix up/down GPU correctness before further performance work.
- If numeric error is similar to gate but output still changes, treat the optimization as quality-risky and prefer CPU fallback pruning/packing instead.
- Do not promote any up/down GPU path until France output is manually correct and token rate exceeds `4.4` under the full strict gates.

### 2026-07-04T01:31Z Up/Down GPU Numerical Audit Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/updown-gpu-compare-result.json`
- artifact sha256: `3fde8d52bced459efd0270e0a5b6d1acaa51be6a02089bf0f6fb12b385eab03a`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T010527Z-20260704_updown_gpu_compare_diagnostic/france-cpu40-vram0gb`

Config:

- Diagnostic only, not a SOTA candidate.
- no-filter stream so gate/up/down are accepted by GPU.
- `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`
- `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=256`
- short generation budget with `-n 32`.

Metrics:

- `eval_tok_s=1.8`
- `prompt_tok_s=1.2`
- `TTFT=39417.875482 ms`
- `elapsed_seconds=53.86`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15147261952`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`

Compare samples:

- total compare rows: `256`
- gate samples: `118`
  - `max_abs_max=1.90734863e-06`
  - `mean_abs_mean=2.4726709927118645e-08`
  - `max_rel_max=2.35010323e-06`
- up samples: `69`
  - `max_abs_max=1.90734863e-06`
  - `mean_abs_mean=2.4209747195652175e-08`
  - `max_rel_max=3.80231751e-06`
- down samples: `69`
  - `max_abs_max=7.62939453e-06`
  - `mean_abs_mean=8.09173102936232e-08`
  - `max_rel_max=2.12846114e-06`

Artifact hashes:

- `summary.json`: `e15b25ccedf1b2c1aa595d869b750fd3333e38d97fecc94c72bdf4ade9aac01e`
- `stdout.txt`: `9e65bcbe303b297ed6580d051bf3de1acd5c887c4b65032d2f4856fcf25b5351`
- `stderr.txt`: `1a7db6ebdbc925d12dd251efb40a160aed7f33e94c4e0eab7494b5f21b9a1fa0`
- `environment.txt`: `5166357e2f7199a318c08e2b68ded310aaf3fa96749b6aedae8a2157f9564672`
- `exact_command.txt`: `7b26647ca30b68074a5b1d3d0356b71c6282cf647618a4a70ed6d81418e68a10`
- `compare_cpu.csv`: `382356e7880b0cc7a8669976c4d02fe4778f03299d9d0c645f08954848776390`
- `resource_samples.tsv`: `f870b344759b4bb3f95aa817fb39f85365e9bf8e0a12b31bdcfc0f9a0ca65010`

Conclusion:

- Sampled gate/up/down GPU stream outputs are numerically close to CPU fallback.
- Up/down GPU output-quality regressions are not explained by a large single-op numerical bug in the sampled rows.
- They are more likely caused by accumulated small differences and altered generation/routing trajectory.
- Because the strict user requirement prioritizes correct France output, do not promote up/down GPU streaming unless the full prompt output is manually correct and `eval_tok_s > 4.4`.

### 2026-07-04T01:36Z Next Plan: Refresh Current SOTA CPU Fallback Profile

Goal:

- Re-measure the current accepted gate-only SOTA with CPU fallback profiling enabled.
- Identify the exact up/down CPU fallback distribution by tensor/layer/expert before making another CPU-side optimization.

Rationale:

- Gate-side CUDA/cache work has been repeatedly optimized and now accounts for less of wall time.
- Up/down GPU streaming is numerically close per op but output-quality risky and slower in full prompt tests.
- The remaining practical path is CPU fallback reduction or CPU fallback scheduling/packing, so we need current fallback distribution under the exact accepted SOTA config.

Experiment design:

- No source change.
- Use accepted SOTA config:
  - gate-only stream filter;
  - current gate cache profile;
  - gate O_DIRECT pack;
  - top-k config `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `10-39=>3`;
  - strict cold `drop_caches`;
  - 16GB cgroup.
- Enable existing profiling:
  - `GGML_KIMI_CPU_MOE_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`
  - if available, fallback/profile envs already in this codebase.
- Keep France prompt and correctness check.

Acceptance / rejection:

- Diagnostic only; not a SOTA candidate unless throughput unexpectedly exceeds `4.4` and all gates pass.
- Must not change source.
- Record stderr profile, summary metrics, and any generated profile artifacts.

Deliverable:

- `.Agent/runs/20260704-vendor-ds4-coldstart/current-sota-cpu-fallback-profile.json`
- Append next optimization recommendation based on measured fallback profile.

### 2026-07-04T01:43Z Current SOTA CPU Fallback Profile Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/current-sota-cpu-fallback-profile.json`
- artifact sha256: `c2a9e1a5c6d8653bb909a4b5b8ea33d31eb5e78cd0986877a06ba197277f63e5`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T011127Z-20260704_current_sota_cpu_fallback_profile/france-cpu40-vram0gb`

Config:

- Diagnostic only, no source change.
- Exact current accepted gate-only SOTA config:
  - vendor DeepSeek;
  - strict cold `drop_caches`;
  - 16GB cgroup including file/page cache;
  - `cpu_moe=40`;
  - `GGML_MOE_VRAM_CACHE_GB=0`;
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`;
  - `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`;
  - gate O_DIRECT expert pack;
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`;
  - current gate admit profile;
  - current top-k envs.
- Added profile env:
  - `GGML_KIMI_CPU_MOE_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_ELIGIBILITY_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback_profile.csv`

Metrics:

- `eval_tok_s=4.3`
- `prompt_tok_s=1.8`
- `TTFT=32239.387852 ms`
- `elapsed_seconds=64.09`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15097294848`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is also celebrated for its cuisine, wine, fashion, and art. With its capital in Paris, France is a major economic and political power, a founding member of the European Union, and a permanent member of the United Nations Security Council. The country has a rich cultural heritage and continues to play a significant role in global affairs.
```

Profile summary:

- fallback entries: `7030`
- total up/down CPU fallback: `26438.059 ms`
- up fallback: `12786.863 ms`
  - `19760` rows
  - `19111` calls
  - `669.084 us/call`
- down fallback: `13651.196 ms`
  - `19760` rows
  - `19111` calls
  - `714.311 us/call`
- prompt fallback: `7408.602 ms`
  - `2342` calls
  - `3163.365 us/call`
- decode fallback: `19029.457 ms`
  - `35880` calls
  - `530.364 us/call`
- `pack_mmap_calls=0` for all profile rows.
- all profiled CPU fallback rows read from GGUF/default source, not a CPU fallback pack.

Top fallback tensors by total fallback time in this run:

- `blk.1.ffn_up_exps.weight`: `817.940 ms`
- `blk.0.ffn_down_exps.weight`: `783.378 ms`
- `blk.1.ffn_down_exps.weight`: `729.180 ms`
- `blk.2.ffn_up_exps.weight`: `710.795 ms`
- `blk.2.ffn_down_exps.weight`: `707.991 ms`
- `blk.0.ffn_up_exps.weight`: `686.698 ms`
- `blk.3.ffn_down_exps.weight`: `531.704 ms`
- `blk.4.ffn_down_exps.weight`: `474.856 ms`

Additional stderr counters:

- gate prefill: `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4594.171`
- gate pack: `hits=4886 misses=0 reads=4886 bytes=21774204928 direct_reads=4886 direct_failures=0 direct_fallbacks=0`
- gate VRAM cache: `hits=33265 misses=1886 hit_rate=94.6%`
- `kimi_cpu_moe_profile down calls=16920 total=2.077 ms/call cuda_single=0.501 post_cuda_barrier=0.001 fallback_t0=1.564 single_accept=35151 single_decline=38222`

Artifact hashes:

- `summary.json`: `af6da0b39b04c171f6a3ad9ed6d21ca26a5de0b6cb7f2cadd951065b8fe8ca9b`
- `stdout.txt`: `bae23fa0545581626f15e9916e4c08e69c4ab7dad3adb080477bd9d3a5aab269`
- `stderr.txt`: `20a4ab68f675ed0f24689ac13fd1ff559fb93040301c6a7711d811f003614b22`
- `environment.txt`: `a5d68cb666296df05a9344cb49efd4831d165c48b54685c630cd9053acd38185`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `one_trace.csv`: `8dcde4975db511339c6efd9d022d84172d208d0dc03fb3056a810f8b8218238d`
- `fallback_profile.csv`: `f334f867df76a653be7281a0ba38bd172c444062aaf5b883497c80fd2575fe81`
- `resource_samples.tsv`: `370c0c237c63eeb1a205f287472f229d66e7ecc025328775e2f78755e2a705cc`

Verdict:

- Diagnostic accepted as evidence, but not promoted as a new SOTA because profiling overhead produced `4.3 tok/s`.
- Current accepted strict cold SOTA remains:
  - `eval_tok_s=4.4`;
  - run `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`;
  - branch `ssd/vendor/deepseek-token-rate-16gb`;
  - commit `f276d0b32` before this documentation commit.

Bottleneck conclusion:

- Gate-side work is not the first priority now:
  - gate pack has zero misses;
  - gate VRAM cache hit rate is high;
  - gate prefill cost is about `4.6s`, already inside TTFT budget.
- The dominant remaining measured work is up/down CPU fallback:
  - total fallback `26.4s`;
  - decode fallback `19.0s`;
  - prompt fallback `7.4s`.
- Earlier CPU fallback pack paths are closed for now:
  - compact mmap top128/top256 worked mechanically but tied/regressed;
  - synchronous O_DIRECT direct staging hit the pack but regressed to `2.9 tok/s`;
  - pure source-location changes do not solve the current 16GB cold page/cache pressure.
- Earlier up/down GPU-stream paths are also not promotable:
  - full no-filter stream regressed to `1.8 tok/s` and failed output completeness;
  - hot up/down64 stream regressed to `3.6 tok/s` and failed manual semantic correctness;
  - per-op numerical audit was close, but full-output trajectory remains quality-risky.

### 2026-07-04T01:55Z Next Plan: CPU Fallback Scheduling And Compute Decomposition

Goal:

- Continue from the accepted `4.4 tok/s` SOTA and target a new strict cold SOTA above `4.4 tok/s`.
- Do not repeat already rejected mmap/direct staging, broad up/down GPU stream, CUDA graph, affinity, or blind top-k/cache-size sweeps.
- First localize whether the remaining up/down CPU fallback time is dominated by:
  - CPU GEMV compute;
  - thread/barrier scheduling;
  - page faults/refaults;
  - source preparation and GGUF pointer lookup;
  - prompt-phase batching shape versus decode single-row shape.

Current hard bound:

- Accepted generation estimate: `192 / 4.4 = 43.64s`.
- Measured total up/down CPU fallback in profile run: `26.44s`.
- Measured decode fallback alone: `19.03s`.
- If a safe optimization removed:
  - `1.0s`, ideal token rate is about `4.50 tok/s`;
  - `2.0s`, ideal token rate is about `4.61 tok/s`;
  - `4.0s`, ideal token rate is about `4.85 tok/s`;
  - `8.0s`, ideal token rate is about `5.39 tok/s`.
- Therefore the next candidate must have a plausible hard bound above `1-2s`; otherwise it is not worth a strict cold run.

Stage A: no-source profiling before implementation:

- Run one strict cold diagnostic from the accepted SOTA config with existing profiling only.
- Add or use existing timers to split CPU fallback into:
  - source pointer lookup / source preparation;
  - page fault or first-touch proxy where available;
  - Q8 activation preparation;
  - MXFP4 dot/GEMV compute;
  - thread wait/barrier time;
  - combine/post-processing time.
- Keep the France prompt and full correctness check.
- Record cgroup memory including page cache, `pgmajfault`, `workingset_refault_file`, TTFT, full output, and all profile hashes.

Stage B: candidate selection after Stage A:

- If CPU GEMV compute dominates:
  - evaluate a narrow MXFP4 CPU-kernel scheduling/layout change only if the measured bound is at least `1-2s`;
  - preferred candidates are default-off, minimal, and bitwise/near-bitwise equivalent;
  - do not repeat the earlier simple MXFP4 prefetch that already regressed.
- If thread/barrier wait dominates:
  - test one bounded scheduling change, such as grouping small fallback tasks to reduce wake/barrier cost;
  - do not repeat already rejected coarse `GGML_MOE_CPU_CHUNK_SIZE`, `OVERPARTITION_CNE1`, or global affinity probes unless the new split shows a different bottleneck.
- If page fault/refault dominates:
  - do not repeat mmap top128/top256 or synchronous O_DIRECT staging;
  - only consider an actually overlapped design where reads for the next expert happen while the current expert computes, and only after deriving an IO/compute overlap bound from measured bytes and call order.
- If prompt fallback dominates TTFT risk:
  - avoid prompt-path experiments unless the theoretical TTFT impact is explicitly bounded below the `20%` limit.
- If no component has a clear `>1s` bound:
  - stop source-level CPU fallback tweaks and switch back to higher-level algorithmic paths, such as correctness-preserving speculative verification, because small local tweaks cannot reach the long-term `10 tok/s` target.

Candidate acceptance gates:

- Strict cold `drop_caches`.
- Host RAM cgroup including page cache must stay `<=16000000000` bytes.
- `MemorySwapMax=0`.
- France output must be semantically correct, coherent, and manually reviewed.
- `TTFT <= 33617.688744 ms` for accepted SOTA.
- `eval_tok_s > 4.4`.
- Any larger token-rate jump must include extra correctness scrutiny and the exact output text in the record.

Required rollback rules:

- Any source patch that regresses token rate, fails correctness, violates RAM, or exceeds accepted TTFT must be reverted before continuing, unless it is explicitly kept as default-off diagnostic instrumentation and documented as such.
- If TTFT rises above the accepted limit but the result is otherwise informative, record and commit it only as rejected/high-TTFT evidence; do not call it accepted SOTA.
- The rollback point for accepted performance remains the pushed `4.4 tok/s` SOTA.

Required record and push discipline:

- Before every experiment, append the hypothesis, hard bound, exact env/CLI config, acceptance/rejection rules, and start time to this plan.
- For every run, record:
  - run directory;
  - source commit and dirty/clean state;
  - binary path and sha256;
  - exact env and CLI;
  - stdout correctness answer;
  - `eval_tok_s`, `prompt_tok_s`, TTFT, elapsed time;
  - cgroup peak memory and file/page-cache bytes;
  - `pgmajfault`, `workingset_refault_file`, and relevant counters;
  - all artifact hashes.
- When a new compliant SOTA appears, immediately:
  - record the full reproduction block in this plan;
  - commit source, plan, and artifacts;
  - push to remote `ssd` branch `vendor/deepseek-token-rate-16gb`;
  - use git identity `L-Ark <fliangae@connect.ust.hk>`;
  - rebuild and rerun from the pushed source before marking it reproducible.
- Do not promote any local-only or dirty-tree result as SOTA.

### 2026-07-04T01:29Z Execution: Current SOTA CPU Chunk Trace Diagnostic

Start time:

- `2026-07-04T01:29Z`

Hypothesis:

- The current accepted `4.4 tok/s` run still spends about `26.44s` in up/down CPU fallback.
- Existing aggregate profiles cannot distinguish whether this is mostly uniform MXFP4 GEMV work or thread scheduling / long-tail chunk imbalance.
- Existing `GGML_MOE_CPU_CHUNK_TRACE_OUT` records true fallback chunk wall time by role, expert, thread, row range, and chunk shape without changing math.
- If one or a few threads/chunks dominate tail latency by more than `1-2s`, the next source candidate should be scheduling/chunk assignment.
- If chunk time is broadly balanced and totals match fallback time, the next source candidate should focus on CPU GEMV/kernel arithmetic or higher-level algorithmic changes rather than scheduling.

Hard bound:

- Accepted generation estimate: `192 / 4.4 = 43.64s`.
- A `1.0s` reduction gives an ideal `4.50 tok/s`; a `2.0s` reduction gives an ideal `4.61 tok/s`.
- Therefore the diagnostic is useful only if chunk imbalance, tail wait, or a concentrated role/layer bucket exposes at least `1-2s` of plausible removable time.

Code change:

- None for this run.
- Use existing default-off tracing only.

Run config:

- Branch/source: `feat/ds4-moe-stream-on-vendor`, starting commit `fc622213b`.
- Binary: `/root/lfz/vendor/llama.cpp-deepseek-v4/build-ds4-moe-stream/bin/llama-cli`.
- Strict runner: `/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/run-tools/strict_ds4_runner.py`.
- Model: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`.
- Prompt: `Please introduce France in a short paragraph.`
- Effective llama args:
  - `-n 192`
  - `-c 256 -b 16 -ub 16`
  - `-t 20 -tb 20`
  - `-ngl all --fit on -fa auto`
  - `--temp 0 --top-p 1 --top-k 1 --seed 1 --no-display-prompt`
  - `--n-cpu-moe 40 --defer-experts`
- Base env from accepted SOTA:
  - `CUDA_VISIBLE_DEVICES=0`
  - `GGML_CUDA_DISABLE_GRAPHS=1`
  - `GGML_MOE_STREAM=1`
  - `GGML_MOE_VRAM_CACHE_GB=0`
  - `GGML_MOE_STREAM_DONTNEED=1`
  - `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`
  - `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`
  - `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`
  - `GGML_MOE_STREAM_ONE_PREFILL_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`
  - `GGML_MOE_KEEP_TOPK_UPDOWN=4`
  - `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`
- Added diagnostic env:
  - `GGML_MOE_CPU_CHUNK_TRACE_OUT={case_dir}/cpu_chunk_trace.csv`
  - `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=250000`
  - `GGML_KIMI_CPU_MOE_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_ELIGIBILITY_PROFILE=1`
  - `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback_profile.csv`
  - `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`

Acceptance / rejection:

- Diagnostic only, not a SOTA candidate unless it unexpectedly exceeds `4.4 tok/s` with all gates passing despite trace overhead.
- Must pass RAM, cgroup, and correctness gates to be used as bottleneck evidence.
- Record and analyze:
  - chunk trace row count;
  - total chunk ms by role;
  - max/p95 chunk ms;
  - per-thread total chunk ms;
  - imbalance between busiest and least busy thread;
  - top experts/layers by chunk ms;
  - gap between aggregate `fallback_t0` and summed chunk ms.
- If the trace shows no `>1s` scheduling/tail bound, do not implement another chunk-size or affinity tweak.

### 2026-07-04T01:35Z CPU Chunk Trace Diagnostic Result: Truncated

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T013256Z-20260704_cpu_chunk_trace_sota_diagnostic/france-cpu40-vram0gb`

Metrics:

- `eval_tok_s=4.3`
- `prompt_tok_s=1.9`
- `TTFT=32439.477620 ms`
- `elapsed_seconds=63.87`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15077658624`
- `pgmajfault=248618`
- `workingset_refault_file=1957406`
- `ram_ok=true`
- `ram_limit_killed=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Key counters:

- gate VRAM cache: `hits=33265 misses=1886 hit_rate=94.6%`
- down profile: `calls=16920 total=2.023 ms/call fallback_t0=1.507 ms/call cuda_single=0.503 single_accept=35151 single_decline=38222`
- fallback profile rows: `7030`
- `cpu_chunk_trace.csv` lines: `250001`

Artifact hashes:

- `summary.json`: `3a9ce3ed8d4ef324195f32f2fd2db847bd6cb9b0d390068901dd7a1c81515139`
- `stdout.txt`: `f84715c5b51cf052276191e2c2b18ba3bb69da1378ebfbcbd6b8868f18e4d947`
- `stderr.txt`: `d02401fbc2e40c569cfa13a7743ad9b2751dfdcd7ecebc840208ddc5d6d5bbc5`
- `environment.txt`: `b12808c3199a75a59c255ba2cf8c78f2efd2056c413f254cecf6733a862b38c3`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `cpu_chunk_trace.csv`: `39584432431372a8dcb900faa89a235130bf7b73f8e4923716881719f77b16d7`
- `fallback_profile.csv`: `bba7869287d9eb25ea254b0ca6d3de73909994a1362a361b3d11b2427660389d`
- `one_trace.csv`: `d573f4b69b1d9636a7bd990782fa31f488c1c500dbfad7f2cd42b4bb5bcd18d2`
- `resource_samples.tsv`: `b79136ae9f7b20c7fd46f3235cd1e70fdba26de0cd9267d2295f11860f46722b`

Verdict:

- Valid for RAM/TTFT/correctness sanity, but insufficient for full chunk-distribution analysis.
- `cpu_chunk_trace.csv` hit the configured `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=250000`, so per-thread totals, role totals, and tail-gap analysis are truncated.
- Do not use this trace to make a scheduling decision.

### 2026-07-04T01:42Z Next Run: Full CPU Chunk Trace Diagnostic

Rationale:

- The previous run proved chunk trace overhead still preserves RAM/correctness, but the trace limit was too low.
- Rerun the same strict cold diagnostic with a higher trace cap before deciding whether scheduling/chunk work has a real `>1s` bound.

Run config changes from the truncated run:

- `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=1000000`
- run name: `20260704_cpu_chunk_trace_sota_diagnostic_full`

Acceptance / rejection:

- Diagnostic only.
- Must pass RAM and correctness gates.
- Trace is usable only if row count is below the configured cap.
- If the trace still hits the cap, stop using row-level full trace and add a lower-overhead aggregate instrumentation plan instead.

### 2026-07-04T01:49Z Full CPU Chunk Trace Diagnostic Result

Artifacts:

- `.Agent/runs/20260704-vendor-ds4-coldstart/current-sota-cpu-chunk-trace-analysis.json`
  - sha256: `95ea0ba1fb9dac338c6b2354638f6ed0e346b90568d12b5f895cf24337e76024`
- `.Agent/runs/20260704-vendor-ds4-coldstart/current-sota-cpu-chunk-trace-full-result.json`
  - sha256: `84e84e10d866286461cd03bd711e279beefe03c4402f7ca66f8a25f4eef5c8eb`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T013715Z-20260704_cpu_chunk_trace_sota_diagnostic_full/france-cpu40-vram0gb`

Metrics:

- `eval_tok_s=4.1`
- `prompt_tok_s=1.8`
- `TTFT=33425.093549 ms`
- `elapsed_seconds=66.55`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15096012800`
- `pgmajfault=256425`
- `workingset_refault_file=1670050`
- `ram_ok=true`
- `ram_limit_killed=false`
- `oom_seen=false`
- `correctness_ok=true`

Correctness output:

```text
Here is a short paragraph introducing France:

France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.
```

Trace completeness:

- `cpu_chunk_trace.csv` rows: `909952`
- configured limit: `1000000`
- complete: true

Key trace numbers:

- total chunk CPU-thread time: `507993.630 ms`
- divided by 20 threads: `25399.682 ms`
- stderr fallback wall estimate: `27072.000 ms`
- gap between fallback wall and thread-average compute: `1672.318 ms`
- chunk time:
  - p50 `0.058 ms`
  - p95 `1.797 ms`
  - p99 `11.14549 ms`
  - max `104.473 ms`
- role totals:
  - down: `260767.295 ms` thread-sum, `13038.365 ms / 20`
  - up: `247226.335 ms` thread-sum, `12361.317 ms / 20`
- thread totals:
  - min `24519.779 ms`
  - max `27460.653 ms`
  - spread `2940.874 ms`
  - mean `25399.681 ms`

Top tensors by chunk thread-sum:

- `blk.0.ffn_up_exps.weight`: `18058.429 ms`
- `blk.1.ffn_down_exps.weight`: `16109.596 ms`
- `blk.2.ffn_up_exps.weight`: `15617.654 ms`
- `blk.0.ffn_down_exps.weight`: `15008.175 ms`
- `blk.2.ffn_down_exps.weight`: `14985.687 ms`
- `blk.1.ffn_up_exps.weight`: `13144.894 ms`
- `blk.3.ffn_up_exps.weight`: `9387.486 ms`
- `blk.4.ffn_down_exps.weight`: `9087.165 ms`

Artifact hashes:

- `summary.json`: `e9693163d6598d5c55ababf27836a3da35030887c1bc3cb8d1082cfd585ec1ac`
- `stdout.txt`: `f4dc6b468a3c87d99d713f3cdbb9574de1b59b2d5ff72aa73aa2d20f67188cc3`
- `stderr.txt`: `b697c56b42f9f1a75b5b7d0f11bc50b6b1bf528ec4105849736b008b3a1e26d9`
- `environment.txt`: `fd686423f67c2982cb0f6b1d07508f5082aef50273088ee784dcf55a1e4e862d`
- `exact_command.txt`: `0da8f189e64738306dd578e654382dafe5beaee60277d74c052db72c9a1bbcb7`
- `cpu_chunk_trace.csv`: `00da07e04e501d618b94f9ba17403d757ed6c27d3e3e1c12903bdb919d1bf18a`
- `fallback_profile.csv`: `5665dafac561926aca569839faeb6de7c5dc4722b3ae6b50582a40656f64939e`
- `one_trace.csv`: `158f12157bbdbf597faa4dc145a2a23120cd74a673f274f931f3a7ec4948143a`
- `resource_samples.tsv`: `4d6a7d811d6c1b4135ae4fc5b2c75c4c13993fd91bc64f3e4483b7bdc151c098`

Verdict:

- Diagnostic accepted as evidence; not a SOTA candidate.
- Correctness, RAM, and TTFT gates pass, but trace overhead lowers token rate to `4.1`.
- Current accepted strict cold SOTA remains `4.4 tok/s`.

Bottleneck conclusion:

- CPU fallback scheduling/chunk imbalance is not the dominant bottleneck:
  - `sum_chunk_ms / 20 = 25399.682 ms`;
  - aggregate fallback wall estimate is `27072.000 ms`;
  - the remaining gap is only about `1672 ms`.
- Per-thread run-wide spread is about `2941 ms`, which is too small to explain the gap to `10 tok/s`.
- The dominant residual work is still up/down source/page movement plus MXFP4 dot over active fallback experts.
- This current-SOTA trace agrees with the earlier 4.2-era split trace:
  - hot CPU fallback math after serial source touch was only about `3.0s`;
  - cold source/page movement dominated;
  - mmap pack, synchronous O_DIRECT staging, `willneed`, and early-layer top-k pruning all failed or were rejected.

Decision:

- Do not run more `GGML_MOE_CPU_CHUNK_SIZE`, overpartition, affinity, or chunk scheduling probes unless a future low-overhead trace shows a new `>1-2s` scheduling bound.
- Do not repeat compact mmap, synchronous O_DIRECT staging, page-touch/willneed, early-layer top-k pruning, or broad up/down GPU stream under the current evidence.

### 2026-07-04T02:00Z Next Plan: Source Movement Bound Or Algorithmic Pivot

Goal:

- Continue toward `10 tok/s` without repeating rejected local optimizations.
- Before any new source experiment, compute a hard upper bound for the only still-plausible source-movement variant: true asynchronous overlap of up/down source reads with hot CPU fallback math.

Known hard numbers:

- Current accepted SOTA: `4.4 tok/s`, about `43.64s` for 192 generated tokens.
- Current full trace:
  - fallback wall estimate about `27.1s`;
  - chunk CPU-thread average about `25.4s`;
  - scheduling/tail gap about `1.7s`.
- Earlier touch split:
  - hot CPU fallback math lower bound about `3.0s`;
  - source/page movement dominates the cold fallback wall time.
- Earlier top128 O_DIRECT staging:
  - moved about `44.79GB`;
  - regressed to `2.9 tok/s`;
  - synchronous IO cost dominated.

Next design step:

1. Estimate async-overlap feasibility before coding:
   - derive bytes that would need to be staged for top64/top128/top256 up/down;
   - estimate minimum IO time under measured direct-read throughput from prior runs;
   - compare that IO time to the measured hot compute overlap window (`~3.0s`) and per-token ordering constraints.
2. If async overlap cannot plausibly save `>1-2s` under the 16GB cgroup and TTFT gate:
   - close the source-movement family for now;
   - pivot to algorithmic work only.
3. Algorithmic candidates must preserve correctness:
   - no lossy top-k pruning without full France correctness;
   - no MTP/NextN unless the model/runtime has real compatible heads;
   - any speculative/lookahead path must verify every accepted token with the exact model path before being accepted.
4. If a source or algorithmic candidate passes the hard-bound screen:
   - append exact implementation plan before code;
   - implement default-off;
   - run strict cold France;
   - promote only if `eval_tok_s > 4.4`, RAM/correctness/TTFT pass, and reproduction info is committed and pushed.

Immediate deliverable before coding:

- Write a small bound artifact that uses current trace plus historical direct-staging bytes/results to decide whether async source overlap is worth implementing.
- If the bound is negative, document the pivot and do not spend another run on pack/direct/page-source variants.

### 2026-07-04T02:04Z Source Movement Async Bound Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/source-movement-async-bound.json`
- sha256: `d97f52ca6f58eb945e12b4473158f07c16d0e412a690f895f9a5eb2e7f9dcf78`

Inputs:

- accepted SOTA: `4.4 tok/s`
- accepted generation estimate: `43.636s` for 192 generated tokens
- current fallback wall estimate: `27.072s`
- current chunk thread-average estimate: `25.400s`
- scheduling gap estimate: `1.672s`
- hot CPU fallback math lower bound from touch split: `3.047s`
- historical top128 O_DIRECT direct-staging bytes: `44,791,758,848` bytes (`41.716 GiB`)
- historical top128 covered fallback time: `3.898s`
- measured direct-staging extra time under 16GB cgroup: `18.571s`
- measured direct throughput from that rejected run: `2.246 GiB/s`
- required top128 bandwidth without overlap: `10.702 GiB/s`
- required top128 bandwidth with full `3.047s` hot-compute overlap: `6.007 GiB/s`

Scenarios:

| Scenario | Bytes | Covered fallback | Measured IO time | Overlap window | Best-case net saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| top64 estimate | `20.858 GiB` | `2.500s` | `9.286s` | `3.047s` | `-3.739s` |
| top128 recorded | `41.716 GiB` | `3.898s` | `18.571s` | `3.047s` | `-11.626s` |
| top256 recorded | `55.711 GiB` | `5.647s` | `24.802s` | `3.047s` | `-16.108s` |

Conclusion:

- Do not implement async pack/direct/page-source staging now.
- The hard-bound screen is negative: prior direct staging moved too many bytes, and measured effective direct throughput under the strict 16GB cgroup is far below the bandwidth needed to save even `1-2s` after accounting for the hot compute overlap window.
- Close these source-movement variants for now:
  - compact mmap pack;
  - synchronous O_DIRECT staging;
  - async O_DIRECT staging without a fundamentally smaller byte volume;
  - `willneed` / page-touch prewarm;
  - chunk-size / affinity scheduling;
  - early-layer top-k pruning.

### 2026-07-04T02:08Z Next Plan: Exact-Verification Algorithmic Pivot

Goal:

- Continue toward `10 tok/s` after closing local source/page/scheduling tweaks.
- Any algorithmic speedup must preserve exact-model correctness for every accepted token, not just produce a plausible-looking paragraph.

Why this pivot is required:

- Local CPU fallback work cannot reach `10 tok/s`:
  - scheduling bound is only about `1-2s`;
  - hot CPU math lower bound is about `3s`;
  - source movement variants are negative under measured IO bandwidth;
  - lossy top-k changes failed correctness/performance;
  - broad up/down GPU stream changed output quality or regressed throughput.
- To reach `10 tok/s` from `4.4 tok/s`, the run needs roughly a `2.27x` effective decode speedup, which requires either:
  - multiple tokens accepted per expensive target-model step; or
  - a fundamentally different exact offload path for missed experts.

Next inspection before any code:

1. Inspect current vendor support for:
   - `llama-speculative`;
   - `llama-lookahead`;
   - n-gram speculative decoding;
   - any DeepSeek MTP/NextN tensor metadata in the GGUF;
   - graph assumptions that previously made lookahead incompatible.
2. Decide whether an exact verifier can run with the current DeepSeek/vendor graph:
   - draft proposals may be approximate;
   - final accepted tokens must be verified by the exact target model under the accepted SOTA config.
3. If no compatible draft/MTP/lookahead path exists:
   - record the blocker;
   - do not fake speculative speedup with unverified tokens;
   - next viable work must be a source-level exact verifier or a fundamentally correct up/down GPU/offload path.

Acceptance gates for any algorithmic candidate:

- Strict cold `drop_caches`.
- 16GB cgroup including page cache.
- `MemorySwapMax=0`.
- France answer must be semantic, coherent, complete, and manually reviewed.
- For speculative paths, every emitted token must be target-verified.
- `TTFT <= 33617.688744 ms` for accepted SOTA.
- `eval_tok_s > 4.4`.
- New compliant SOTA must be recorded in full, committed, pushed to `ssd/vendor/deepseek-token-rate-16gb`, then rerun from pushed source.

### 2026-07-04T02:17Z Algorithmic Support Inspection Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/algorithmic-support-inspection.json`
- sha256: `b9f62ac35891d87d65e1b40ff94b1a2b626718e08a98e66e5ee7fa1083c59f6d`

Code inspection:

- Existing binaries are present:
  - `build-ds4-moe-stream/bin/llama-lookahead`
  - `build-ds4-moe-stream/bin/llama-lookup`
  - `build-ds4-moe-stream/bin/llama-speculative`
  - `build-ds4-moe-stream/bin/llama-speculative-simple`
- `llama-speculative` / `llama-speculative-simple` require a draft model and check draft/target vocab compatibility.
- Local model inventory has no tokenizer-compatible small DeepSeek draft GGUF:
  - DeepSeek target GGUF is present;
  - GLM GGUF shards are present but are not compatible DeepSeek draft models.
- DeepSeek GGUF metadata grep showed DeepSeek4 keys and `hc_head_*` tensors, but no `mtp`, `nextn`, `nextn_predict_layers`, `draft`, or `eagle` keys.
- `hc_head_*` is used in `src/models/deepseek4.cpp` as the normal output head path before `result_norm` / `result_output`; it is not an MTP head.
- `src/llama-arch.cpp` explicitly comments that NextN/MTP tensors are currently ignored/reserved for future support.
- `src/llama-model.cpp` has `TODO` comments around MTP implementation.

Historical rejected algorithmic paths still apply:

- target-only `llama-lookahead` failed with DeepSeek4 coupled-sequence divergence and collapsed gate-cache hit rate;
- ngram-simple all-output path stalled/replayed, and the fallback-correct version only reached about `3.7 tok/s`;
- no-draft ngram-mod accepted only `37.5%` drafted tokens, increased target/checkpoint overhead, and reached `2.879 tok/s`;
- external draft speculation is unavailable without a compatible draft model artifact.

Verdict:

- No no-source algorithmic candidate is available now.
- Do not rerun:
  - `llama-lookahead`;
  - ngram-simple;
  - ngram-mod;
  - speculative decoding without a compatible draft model.

Next viable classes:

1. Obtain or build a compatible DeepSeek draft model, then run exact target verification.
2. Implement real DeepSeek4 MTP/NextN only if a GGUF with compatible heads exists.
3. Redesign DeepSeek4 branch KV state for lookahead only if a hard memory/batch bound proves it can work under 16GB RAM and current VRAM constraints.
4. Revisit GPU up/down offload only if full-output correctness can be guaranteed and source movement is bounded.

Current next practical direction:

- Since no compatible draft/MTP artifact is present, the next local engineering path is not another no-source speculative run.
- If continuing without new model artifacts, the only remaining high-bound code path is a correctness-first GPU up/down offload investigation:
  - start with a deterministic short compare that verifies complete France output for a smaller/hotter up/down set;
  - require exact output correctness before any performance run;
  - do not promote if semantic correctness drifts, even if per-op numeric error is small.

### 2026-07-04T02:28Z Next Plan: GPU Up/Down Output Drift Audit

Goal:

- Determine why GPU up/down offload paths are not promotable despite small sampled per-op numerical differences.
- Use existing recorded runs first; do not run another strict cold model pass until the old evidence is exhausted.

Inputs:

- accepted SOTA output:
  `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- full no-filter up/down GPU stream rejected run:
  `/root/lfz/runs/vendor-ds4-16gb/20260704T003804Z-20260704_updown_stream_nofilter_top3000_probe/france-cpu40-vram0gb`
- profile-gated hot up/down64 rejected run:
  `/root/lfz/runs/vendor-ds4-16gb/20260704T005617Z-20260704_hot_updown64_require_profile_candidate/france-cpu40-vram0gb`
- sampled GPU-vs-CPU numerical compare:
  `.Agent/runs/20260704-vendor-ds4-coldstart/updown-gpu-compare-result.json`

Questions to answer:

1. Does output drift appear as a harmless wording change, semantic error, or incomplete generation?
2. Do rejected GPU up/down runs share a common counter signature:
   - more gate pack reads/misses;
   - lower gate cache hit rate;
   - extra file inputs/refaults;
   - TTFT pressure;
   - role-specific up/down stream misses?
3. Is there evidence that per-op numerical error is too small to explain drift alone, implying generation trajectory sensitivity or changed gate/cache/source behavior?
4. Is any already-tested GPU up/down class worth another run?

Acceptance for the audit:

- This is an offline diagnostic artifact only.
- It cannot promote SOTA.
- It must produce a concrete next action:
  - either a new narrow correctness experiment with a hard bound;
  - or close GPU up/down offload for now and require new model/artifact support before continuing.

Deliverable:

- `.Agent/runs/20260704-vendor-ds4-coldstart/gpu-updown-output-drift-audit.json`

### 2026-07-04T02:35Z Latest Execution Plan Update

This section is the active plan before any further source change or model run.

Immediate objective:

- Finish the offline GPU up/down output drift audit from existing artifacts.
- Do not run another strict cold model pass and do not edit runtime source until this audit has a written verdict.
- The audit must be committed and pushed as documentation/artifact work even if it closes the path.

Required audit inputs:

- Accepted SOTA run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Rejected no-filter GPU up/down stream run: `/root/lfz/runs/vendor-ds4-16gb/20260704T003804Z-20260704_updown_stream_nofilter_top3000_probe/france-cpu40-vram0gb`
- Rejected profile-gated hot up/down64 run: `/root/lfz/runs/vendor-ds4-16gb/20260704T005617Z-20260704_hot_updown64_require_profile_candidate/france-cpu40-vram0gb`
- Sampled CPU-vs-GPU op compare: `.Agent/runs/20260704-vendor-ds4-coldstart/updown-gpu-compare-result.json`, sha256 `3fde8d52bced459efd0270e0a5b6d1acaa51be6a02089bf0f6fb12b385eab03a`

Audit output requirements:

- Write `.Agent/runs/20260704-vendor-ds4-coldstart/gpu-updown-output-drift-audit.json`.
- Include exact source/output differences between accepted SOTA and rejected GPU up/down runs.
- Classify each drift as complete semantic pass, incomplete generation, or semantic error.
- Compare counters for gate pack reads/misses, VRAM cache hit/miss rate, source/refault pressure, TTFT, and up/down stream/cache behavior.
- Explain why tiny sampled per-op differences (`~1e-6` to `~1e-5`) are or are not enough to explain full-sequence drift.
- End with one of two explicit decisions:
  - close current GPU up/down offload classes; or
  - propose one new narrow correctness experiment with a hard upper bound and a default-off implementation plan.

Decision rules after the audit:

1. If the audit confirms the existing GPU up/down classes perturb output trajectory or gate/source behavior:
   - close no-filter one-stream up/down;
   - close profile-gated hot64/topN scaling on the current shared gate cache;
   - do not rerun larger hotsets, down-cache sizing, or broad one-stream variants without a new deterministic correctness mechanism.
2. If a new GPU up/down experiment is still justified:
   - first design an exact verifier or deterministic compare path;
   - prove it preserves full France output before any performance run;
   - calculate removable fallback time, VRAM cost, host-byte movement, and TTFT impact before coding;
   - implement default-off only.
3. If no GPU up/down path remains:
   - record that local no-source/source-tweak paths are exhausted under current artifacts;
   - next viable work requires either a compatible DeepSeek draft/MTP/NextN artifact, or a fundamentally new exact GPU up/down implementation whose correctness can be verified token-by-token.

Promotion discipline for every future candidate:

- Update this plan before implementation.
- Run under strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`.
- Accepted threshold is now `eval_tok_s > 4.4`, not `>4.2`.
- Accepted TTFT must be `<=33617.688744 ms`.
- France output must be manually reviewed as semantic, coherent, and complete.
- Pack counters must show no direct failures/fallbacks on the accepted gate pack path.
- Host RAM/page cache must stay inside the 16GB cgroup, with no OOM kill.
- On any compliant new SOTA, stop exploration immediately, record full reproduction metadata, commit source/plan/artifacts/profiles, push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then clean rebuild and reproduce from pushed source before declaring it accepted.
- If performance regresses, correctness fails, TTFT exceeds the accepted gate, or RAM exceeds the cgroup limit, revert runtime source immediately and keep only rejected records/docs.

### 2026-07-04T02:55Z GPU Up/Down Output Drift Audit Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/gpu-updown-output-drift-audit.json`
- sha256: `42b9dc08cc8edd93571fc7a4a964643af410738b86bce7be5957bcf20e6fa02b`
- source head at audit: `7990911a557077ad4f434676890e145b64366ddb`
- audit type: offline existing runs only; no model run performed; no runtime source modified.

Accepted reference:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`
- `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Gate pack counters: `hits=4886`, `misses=0`, `direct_failures=0`, `direct_fallbacks=0`
- VRAM cache counters: `hits=33265`, `misses=1886`, `hit_rate=94.6%`
- Manual correctness: pass; France answer is semantic, coherent, and complete.

Rejected no-filter GPU up/down:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T003804Z-20260704_updown_stream_nofilter_top3000_probe/france-cpu40-vram0gb`
- `eval_tok_s=1.8`, `prompt_tok_s=1.2`, `TTFT=36713.048232 ms`
- RAM gate passes, but TTFT exceeds the accepted gate and token rate is far below SOTA.
- Manual correctness: fail; answer is about France but ends incomplete after `with a thriving economy and`.
- Counter signature: pack `misses=53774`, VRAM cache `hits=43544`, `misses=56319`, `hit_rate=43.6%`.
- Diagnosis: broad up/down one-stream makes too many no-pack/no-cache source loads, collapses cache behavior, and is not a candidate for larger sweeps.

Rejected profile-gated hot up/down64:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T005617Z-20260704_hot_updown64_require_profile_candidate/france-cpu40-vram0gb`
- `eval_tok_s=3.6`, `prompt_tok_s=1.8`, `TTFT=33432.591735 ms`
- RAM and TTFT gates pass, but token rate is below `4.4` and correctness fails.
- Manual correctness: fail; answer says France is often called the `City of Light`, which is a nickname for Paris, not France.
- Counter signature: pack `misses=1131`, VRAM cache `hits=57702`, `misses=3776`, `hit_rate=93.9%`.
- Diagnosis: profile gating avoids the no-filter cache collapse, but still changes the generated answer and remains slower than the accepted gate-only SOTA.

CPU-vs-GPU sampled numerical compare:

- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/updown-gpu-compare-result.json`, sha256 `3fde8d52bced459efd0270e0a5b6d1acaa51be6a02089bf0f6fb12b385eab03a`
- Gate max_abs about `1.907e-6`, up max_abs about `1.907e-6`, down max_abs about `7.629e-6`; mean_abs is around `1e-8` to `1e-7`.
- Interpretation: there is no evidence of a gross single-op arithmetic bug, but tiny per-op differences are still enough to change greedy decode trajectories. Full-sequence correctness remains the acceptance gate.

Verdict:

- Close current GPU up/down offload classes.
- Do not rerun no-filter one-stream up/down.
- Do not scale profile-gated hot64/topN on the current shared gate cache.
- Do not continue independent down-cache sizing under the current VRAM budget.
- Do not run larger hotset experiments unless there is a new deterministic correctness mechanism and hard-bound design first.

### 2026-07-04T03:00Z Active Plan After GPU Up/Down Audit

Current state:

- Accepted strict cold SOTA remains `4.4 tok/s`.
- Local source/page/scheduling/offload tweaks tried so far do not provide a viable path to `10 tok/s`:
  - chunk scheduling/affinity bound is too small;
  - source movement/direct/page prefetch bound is negative under measured bandwidth;
  - no compatible local draft/MTP/NextN artifact is present;
  - current GPU up/down offload classes fail correctness or performance.

Next allowed design work:

1. Exact verifier feasibility inspection:
   - inspect current `llama-cli`, `llama-speculative`, `llama-lookahead`, and DeepSeek4 graph code for an existing way to force/evaluate a known token sequence or dump top logits per step;
   - determine whether a candidate GPU up/down path can be checked token-by-token against the accepted CPU-fallback path without relying on semantic eyeballing only;
   - this inspection is code/read-only first and does not require a model run.
2. If an exact verifier path exists:
   - write a default-off design for a full-output verifier;
   - calculate the extra memory, runtime, and TTFT cost;
   - run it only as diagnostic evidence, not as SOTA promotion.
3. If no exact verifier path exists:
   - record that local work is at an artifact/model-support boundary;
   - next practical progress requires either a compatible DeepSeek draft/MTP/NextN model artifact, or a fundamentally new exact GPU up/down implementation with a correctness proof stronger than sampled per-op closeness.

Hard rule for the next implementation candidate:

- No source patch is allowed until the plan records:
  - the exact bottleneck it targets;
  - the hard upper bound in seconds and expected token-rate ceiling;
  - the correctness mechanism;
  - RAM/page-cache accounting under the 16GB cgroup;
  - TTFT impact estimate;
  - the rollback criteria.

### 2026-07-04T03:10Z Exact Verifier Feasibility Inspection Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/exact-verifier-feasibility-inspection.json`
- sha256: `72bc9058cbabbd6ba50dc5a09abb6b39386ac6fb3980b938a4435dc5e8360333`
- source head: `59e2c514bfb16c1ad41f3b56ac19e3af2cca5084`
- inspection type: source read-only; no model run; no runtime source modified.

Findings:

- `llama-debug --save-logits` only saves final prompt-position logits via `llama_get_logits_ith(ctx, tokens.size() - 1)`. It cannot verify each generated step.
- `llama-perplexity --save-all-logits` / `--kl-divergence-base` can evaluate fixed text windows and report aggregate KL / `Same top p`, but it does not emit per-position top1 mismatch, accepted token id, candidate token id, or margin report. Default mode also requires enough tokens for the ppl window.
- Server `n_probs` can report top logprobs during online generation, but it is a single-path generation API, not a forced dual-path verifier.
- The sampler layer can access logits, but there is no current CLI flag for a full forced-sequence top1/top2/margin report.
- `llama-results` is the best existing base: it tokenizes a fixed prompt, evaluates all positions with `logits=true`, stores all logits, and `--check` compares against a previous result file. Its current check is NMSE only and is not strong enough for future GPU up/down correctness gates.

Verdict:

- Existing no-source verifier is insufficient.
- The next implementation may be a diagnostic-only extension of `tools/results/results.cpp` to report top1/top2/margin per position.
- This verifier extension has `0 tok/s` direct performance upside and cannot be promoted as SOTA. Its purpose is to prevent invalid GPU/offload candidates from being accepted on weak semantic checks or sampled per-op comparisons.

Implementation plan for verifier extension:

1. Scope:
   - Modify only `tools/results/results.cpp` if possible.
   - Keep `llama-cli` and accepted runtime behavior unchanged.
   - Add default-off reporting/check behavior only.
2. Output requirements:
   - number of token positions compared;
   - same-top1 count and ratio;
   - first top1 mismatch position;
   - token ids and pieces for base top1, candidate top1, and actual next token where applicable;
   - base/candidate top1 and top2 logits;
   - margin to top2;
   - max_abs and mean_abs logits difference summary.
3. RAM accounting:
   - Before any diagnostic run, estimate `n_tokens * n_vocab * 4` for one logits matrix.
   - For France verifier text with `n_tokens <= 256` and `n_vocab <= 200000`, one logits matrix is `<=204800000` bytes before overhead. If actual metadata exceeds this bound or cgroup memory approaches 16GB, abort.
   - The verifier is diagnostic; it must not be used to claim SOTA token rate.
4. TTFT:
   - Not applicable to promotion because verifier runs are not SOTA candidates.
   - Still record elapsed time and memory if a diagnostic run is later executed.
5. Rollback criteria:
   - build failure;
   - any change outside diagnostic tool behavior;
   - inability to compare top1/margins from result files;
   - diagnostic memory violation under 16GB cgroup when tested.

### 2026-07-04T03:25Z `llama-results` Top1 Verifier Implementation Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/llama-results-top1-verifier-implementation.json`
- sha256: `7d400e3c3501728b1f451f971beba4b39c74331351747d421d28ee978db9629a`

Implementation:

- Changed `tools/results/results.cpp` only for tool-local behavior:
  - added private parsing for `--top1-report <json>`;
  - added `--top1-fail-on-mismatch`;
  - in `--check` mode, compares stored base logits against current logits and writes a JSON top1 report.
- Updated `tools/results/README.md` with the diagnostic usage.
- `llama-cli` and accepted runtime path are unchanged.

Build and light verification:

- Build command: `cmake --build build-ds4-moe-stream --target llama-results -j 16`
- Result: pass.
- `build-ds4-moe-stream/bin/llama-results` sha256: `5f78342ddf9bfeaf2a21c773f43a4b5ce83ffaebbe6762b376288301a273f98b`
- `git diff --check`: pass.
- Help/arg strip check: `build-ds4-moe-stream/bin/llama-results --help --top1-report /tmp/unused-top1.json` exits `0` and does not create the report file.

Verifier output fields:

- `n_tokens`, `n_vocab`, `same_top1`, `same_top1_ratio`
- `first_mismatch_pos`
- `base_top1_matches_next_token`, `calc_top1_matches_next_token`
- `max_abs`, `mean_abs`
- first 16 mismatch previews with actual next token, base/candidate top1/top2 ids/pieces/logits, and margins.

Status:

- This is a correctness diagnostic tool, not a token-rate optimization and not a SOTA candidate.
- Before relying on it for future GPU/offload candidates, run one DS4 self-check on the accepted fixed France text under a 16GB cgroup. The self-check should:
  - create a baseline `llama-results` GGUF from accepted SOTA env/config;
  - rerun `--check --top1-report --top1-fail-on-mismatch` against the same accepted path;
  - confirm `same_top1 == n_tokens`, `first_mismatch_pos == -1`, no cgroup OOM, and report memory/elapsed time.

### 2026-07-04T03:45Z `llama-results` Verifier Self-Check Result

Initial validation attempts:

- Full-result `llama-results` with SOTA `-b 16` first hit the original tool assumption `n_tokens_all <= cparams.n_batch`.
- After chunking by `llama_n_batch(ctx)`, the full-logits GGUF path still segfaulted under the 16GB cgroup near the end of baseline evaluation.
- Root cause for this path is practical verifier design, not a SOTA runtime regression: full logits materialization / all-logits result writing is not stable enough for DS4 under the strict 16GB cgroup.

Final verifier design:

- Added `--sequential-logits` to evaluate fixed text one token at a time.
- Allowed `--top1-report <json>` without `--check` or `--output` to write a lightweight per-position top1/top2/margin report.
- DS4 verifier self-check compares two lightweight top1 reports instead of writing full logits GGUF.

Final source/tool state:

- Changed files: `tools/results/results.cpp`, `tools/results/README.md`.
- `llama-cli` and accepted SOTA runtime path remain unchanged.
- `build-ds4-moe-stream/bin/llama-results` sha256: `d981fb8f26fb16c338e23acfe490f070e90b7b70ecad2a828bc503b65fca07a9`.
- Updated implementation artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/llama-results-top1-verifier-implementation.json`, sha256 `69cbf324653cd92366cd5b6ab9bdc3d38f3e9a23b6593a2abf379d499d22ee2c`.

Passing DS4 self-check:

- Result artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/llama-results-top1-selfcheck-light-result.json`
- sha256: `26fc0f157f6f35a1a96f7d807ff85c3eb54a4afa1d714f43532ebf29b785419b`
- Case dir: `/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light`
- Config: accepted SOTA env/config, fixed text = accepted France prompt + accepted France answer, `--sequential-logits`, `--top1-report`, 16GB cgroup with `MemorySwapMax=0`.
- `n_tokens=145`
- `same_top1=145`
- `same_top1_ratio=1.0`
- `first_mismatch_pos=-1`
- `baseline_top1_report_sha256=e6cf40c9c8bfc29f3b4c60fe3db8928b86861795d914eadfdf293202ee4476ef`
- `check_top1_report_sha256=e6cf40c9c8bfc29f3b4c60fe3db8928b86861795d914eadfdf293202ee4476ef`
- `memory_peak_bytes=16000000000`
- `oom=0`, `oom_kill=0`, `oom_group_kill=0`

Verdict:

- The lightweight sequential top1 verifier is usable for future DS4 correctness diagnostics under the 16GB cgroup.
- This is not a token-rate improvement and cannot be promoted as SOTA.
- Before any future GPU/offload candidate can run a performance benchmark, it must first pass this verifier or an equivalent token-level correctness check on the accepted France fixed text.

### 2026-07-04T04:20Z Latest Plan: DS4 Hot-Dispatch Verifier-First

Current accepted SOTA remains unchanged:

- `eval_tok_s=4.4`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Required promotion gate: strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, France output semantic/coherent, `TTFT <= 33617.688744 ms`, `eval_tok_s > 4.4`, no pack direct fallback, and pushed-source reproduction.

Why this is the next candidate:

- Measured current bottleneck is still CPU up/down fallback, not gate pack I/O. The accepted gate-only SOTA already gets gate pack `hits=4886 misses=0` and VRAM cache `hit_rate=94.6%`.
- Previously rejected GPU up/down candidates used one-stream/cache admission and either collapsed source/cache behavior or changed the generated answer. They are closed.
- A separate default-off DS4 hot-expert dual dispatch path already exists in `src/llama-deepseek4-hot.*` and `src/models/deepseek4.cpp`. It routes frequent experts through GPU-resident hot subsets and keeps cold picks on the original CPU tensors, then combines hot/cold outputs. Activation is `DS4_HOT_PROFILE_JSON=<profile.json>` plus `DS4_HOT_DISPATCH=1`.
- This path targets actual fallback compute and source stalls instead of only repacking source pages. It is different enough from the rejected one-stream up/down path to justify a narrow verifier-first investigation.

Hard bounds and current evidence:

- Existing `hot-updown64` profile summary shows top64 up and top64 down each cover `7484 / 25997 = 28.79%` of recorded up/down rows.
- The rejected one-stream/profile-gated hot64 result had `eval_tok_s=3.6` and a semantic error, so it is not evidence that dual dispatch is accepted. It only gives coverage and risk data.
- `ds4-hot-route-vram-bound.json` shows the per-layer hot-dispatch allocation shape can reach about `293.25 MiB` for one layer at `k=16`; a full profile must be summed before running because preserving the accepted `13568 MiB` gate cache is preferred unless the plan proves a larger gain.
- Upper-bound check before any run: use current fallback timing (`~25.7s` total CPU fallback; up/down roughly `12.4s/13.4s`) and the selected profile coverage. A candidate must have a theoretical ceiling comfortably above `4.4 tok/s` after graph overhead, extra GPU memory pressure, and any gate-cache reduction. If the bound is only a tie, reject without a model run.

Phase H0: Profile and VRAM feasibility, read-only:

1. Find or generate the minimal DS4 hot profile JSON required by `DS4_HOT_PROFILE_JSON`. Prefer an existing profile artifact; if none exists, generate one from committed profile data and record the exact script/inputs/hash before use.
2. Compute total hot subset VRAM bytes across all layers for candidate `k` values. Include gate/up/down hot tensors, dummy rows, remap tables, and any workspace.
3. Confirm the candidate can fit without reducing accepted gate one-stream cache (`GGML_MOE_STREAM_ONE_CACHE_MIB=13568`). If it cannot, calculate the exact gate-slot loss and expected pack/source penalty before continuing.
4. Record profile hash, model hash, source head, expected coverage, expected removable milliseconds, expected token-rate ceiling, and VRAM budget in an artifact.

Phase H1: Token-level correctness gate before performance:

1. Use the accepted fixed France text and the lightweight sequential top1 verifier.
2. Baseline report is the accepted SOTA report from `/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light/top1-baseline.json`.
3. Run candidate verifier under the same 16GB cgroup and accepted SOTA env, adding only `DS4_HOT_PROFILE_JSON=<profile.json>` and `DS4_HOT_DISPATCH=1`.
4. Pass condition: `same_top1 == n_tokens`, `first_mismatch_pos == -1`, `oom=0`, `oom_kill=0`, and `memory_peak_bytes <= 16000000000`.
5. If any top1 mismatch appears, reject this hot-dispatch profile immediately. Do not run a throughput benchmark and do not try to accept by manual semantic review.

Phase H2: Strict cold performance only after H1 passes:

1. Run the standard France cold-start benchmark with strict `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, accepted CLI args, accepted gate pack/profile/env, plus the verified hot-dispatch env.
2. Record token rates, TTFT, exact output, memory stats including file/page cache, cgroup OOM counters, pack counters, VRAM cache counters, hot-dispatch logs, source head, build hashes, profile hashes, and exact command.
3. Accept only if `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, `ram_ok=true`, `correctness_ok=true`, and all counters are clean.
4. If TTFT exceeds the gate but throughput improves, record and commit as `not accepted` only if useful for future TTFT reduction; do not promote.

Rollback and promotion:

- Runtime source changes are not required for H0/H1 because the dual-dispatch code already exists and is default-off. If any source patch becomes necessary, update this plan again before implementing it.
- On verifier failure, performance regression, correctness failure, RAM violation, TTFT gate violation for an accepted candidate, or gate-cache collapse, revert any runtime source patch immediately and keep only rejected records/docs.
- On a compliant new SOTA, stop exploration immediately, write full reproduction metadata, commit and push source/plan/artifacts/profiles to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then clean rebuild and reproduce from pushed source before declaring it accepted.

### 2026-07-04T04:45Z DS4 Hot-Dispatch H0 Bound Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/ds4-hot-dispatch-h0-bound-analysis.json`
- sha256: `ffc79d774f17e239d846268f1c4f6b9928d0967610b09a0fff4776a7f15294ba`
- source head: `2f1adade6245586d25d86a33ce29b58dccdaa13b`

Result:

- H0 does not pass. Do not run H1 verifier or H2 performance for the current DS4 hot-dispatch implementation.
- No ready `DS4_HOT_PROFILE_JSON` profile was found. A profile could be generated from route/profile data, but H0 shows it is not worth generating for the current implementation.
- Accepted SOTA VRAM shape leaves only about `238 MiB` free with the `13568 MiB` gate cache.
- Best partial hot-dispatch placement within `238 MiB` is one `k=8` layer using `191.25 MiB`; it saves only `17 / 6763 = 0.25%` CPU rows, with an optimistic no-overhead ceiling of `4.409 tok/s`.
- If the gate cache is shrunk enough to free `492 MiB`, the best placement saves `58 / 6763 = 0.86%` CPU rows, with an optimistic no-overhead ceiling of `4.431 tok/s`. This is not enough to justify risking gate-cache misses.
- Full-layer profiles are also poor:
  - `k=1`: `4080 MiB`, `0` rows saved, optimistic ceiling `4.40 tok/s`
  - `k=2`: `4590 MiB`, `62` rows saved, optimistic ceiling `4.43 tok/s`
  - `k=4`: `5610 MiB`, `183` rows saved, optimistic ceiling `4.50 tok/s`
  - `k=8`: `7650 MiB`, `432` rows saved, optimistic ceiling `4.65 tok/s`
  - `k=16`: `11730 MiB`, `1061` rows saved, optimistic ceiling `5.06 tok/s`
- These ceilings assume zero graph overhead, zero extra GPU contention, no verifier drift, and no penalty from stealing VRAM from the accepted gate cache. Real performance would be lower.

Conclusion:

- The current default-off DS4 hot-dispatch path is closed as a token-rate candidate under the present 16GB/VRAM constraints.
- It does not plausibly move toward `10 tok/s`; even the unrealistic all-layer `k=16` upper bound is only about `5.06 tok/s` before overhead and requires about `11.7 GiB` VRAM.
- Do not spend a model run on this path unless a future source design changes the cold path so hot picks are truly skipped without requiring large GPU hot subsets or gate-cache sacrifice.

Next read-only direction:

1. Analyze whether a true CPU fallback skip/zero-row mechanism can remove work for already GPU-handled/pruned picks without allocating large hot expert subsets.
2. Compute the hard upper bound from route trace rows and current fallback timing before writing code.
3. If and only if the bound can plausibly exceed the current `4.4 tok/s` SOTA by a meaningful margin and preserve correctness by construction, update this plan with the exact source design and verifier sequence.
4. Otherwise close that path too and move to deeper CPU MXFP4/layout work.

### 2026-07-04T05:05Z True Skip/Zero-Row Bound Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/true-skip-zero-row-bound-analysis.json`
- sha256: `0af36aa32980612a7ea52917bf884fa9c6553a40de295678f967377323275228`
- source head: `4b2621719c40972d02b509e81b3e6371585e6d57`

Evidence:

- Route trace counts for accepted SOTA: `up pruned0=19760`, `up pruned1=16720`, `down pruned0=19760`, `down pruned1=16720`.
- Current fallback profile counts: `up count=19760`, `down count=19760`.
- Therefore the remaining CPU fallback rows already match `pruned0`; `pruned1` rows are already zeroed/skipped before the CPU fallback loop.
- Historical CPU phase trace measured grouping plus zero-fill at only `82.008 ms` while zeroing `33440` pruned entries / `410910720` bytes.

Hard bounds:

- Perfectly eliminating all existing grouping/zero-fill would save only `0.082008s`, with optimistic no-overhead ceiling `4.412 tok/s`.
- Removing all remaining up/down CPU fallback would have a large reference ceiling (`26.438s` removable, optimistic `29.64 tok/s`), but that requires exact compute/offload/layout work. It is not available through skip/zero pruning.

Verdict:

- Close true skip/zero-row as a token-rate optimization path.
- Any additional row removal is approximate top-k pruning, not exact correctness-preserving skipping. Relevant top2 variants already failed trajectory/correctness/performance (`late10-last10-top2`, `late10-last5-top2`, `late10-last10-up-only-top2`).
- Next active direction must target exact elimination of remaining up/down CPU fallback through compute/offload/layout changes. If it can change logits, it must use the lightweight top1 verifier before any performance benchmark.

### 2026-07-04T03:45Z Current Plan Update: Exact Up/Down Fallback Elimination

Current accepted SOTA remains unchanged:

- `eval_tok_s=4.4`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Reproduction source branch: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- Source head at this plan update: `1bc0ad58ad162598a26a2427c29344dc2afb5b53`
- Promotion gate remains strict: cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, France answer semantic/coherent, `TTFT <= 33617.688744 ms`, `eval_tok_s > 4.4`, clean pack/cache counters, and pushed-source reproduction.

Current bottleneck:

- The remaining useful bottleneck is exact removal or acceleration of up/down CPU fallback work.
- Current fallback profile artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/current-sota-cpu-fallback-profile.json`
- Measured remaining fallback time: `26438.059 ms`
  - up: `12786.863 ms`
  - down: `13651.196 ms`
  - decode fallback: `19029.457 ms`
  - prompt fallback: `7408.602 ms`
- Hard-bound reference:
  - removing all decode up/down CPU fallback alone reaches only about `8.86 tok/s`, below the `10 tok/s` goal;
  - reaching `10 tok/s` requires stacking exact up/down fallback removal with another exact gain, most likely source/gate stall reduction or algorithmic multi-token execution.

Closed or rejected paths that should not be repeated without new evidence:

- DS4 hot-dispatch as currently implemented: H0 bound fails; preserving the accepted gate cache leaves only about `238 MiB` VRAM free and the optimistic ceiling is only `4.409 tok/s`.
- True skip/zero-row elimination: remaining CPU fallback already matches `pruned0`; perfect zero/group elimination ceiling is only `4.412 tok/s`.
- One-stream GPU up/down cache variants: either regress token rate, collapse cache/source behavior, or change the France answer.
- Current CUDA graph attempt: strict cold run tied/regressed around `4.1 tok/s`; no promotion.
- Existing CPU repack / no-repack / transient down-repack variants: tied or regressed; transient down-repack also produced incomplete output.
- Page/source prewarm variants already tried (`WILLNEED`, blocking touch, dense mmap/page drop, compact mmap packs, synchronous O_DIRECT staging): tied, regressed, or had a negative bandwidth/overlap bound.
- CPU chunk/affinity variants: tied, regressed, or violated the 16GB cgroup.
- Approximate top-k pruning beyond the accepted path: relevant top2 variants failed correctness, trajectory, or performance.
- Algorithmic speculative/MTP/lookahead without a compatible DeepSeek draft or MTP/NextN tensors: closed for the current local model inventory.

Immediate design step before any new runtime patch:

1. Create an `exact-updown-candidate-screening.json` artifact under `.Agent/runs/20260704-vendor-ds4-coldstart/`.
2. The artifact must aggregate the accepted SOTA metrics, fallback profile, hard-bound table, DS4 hot-dispatch H0 rejection, skip/zero rejection, GPU up/down drift audit, algorithmic-support inspection, source-movement bound, CUDA graph result, and repack/page/chunk rejected evidence.
3. For each candidate class, record:
   - exact code path it would change;
   - whether it is exact or can change logits;
   - expected removable milliseconds from hard measurements;
   - VRAM cost and whether it steals from the accepted `GGML_MOE_STREAM_ONE_CACHE_MIB=13568` gate cache;
   - host RAM impact including page cache inside the 16GB cgroup;
   - theoretical token-rate ceiling;
   - verifier needed before performance;
   - accept/reject decision.
4. Do not start a model run for a candidate whose no-overhead ceiling is only a tie, whose VRAM plan reduces the gate cache without a larger modeled gain, or whose correctness cannot be verified.

Candidate classes still allowed for future work:

1. Exact up/down compute/offload/layout redesign:
   - Goal: remove a large fraction of the measured `26438.059 ms` fallback without changing logits.
   - This cannot be the current DS4 hot-dispatch implementation unless the design changes so hot/cold routing avoids large duplicated GPU hot subsets and does not sacrifice the gate cache.
   - A candidate must show a hard ceiling above the current SOTA before implementation, and should have a credible stackable path toward `10 tok/s`.
2. Source/gate stall reduction only with a positive bandwidth/overlap model:
   - Existing source movement attempts are closed; a new attempt must explain why it avoids the previous negative O_DIRECT staging bound and why it preserves the gate pack hit behavior.
3. Algorithmic multi-token execution:
   - Only reopen if a compatible DeepSeek draft model, MTP/NextN tensors, or another exact/token-verified mechanism is available.
   - Any such path must pass the lightweight top1 verifier or an equivalent token-level check before throughput is measured.

Execution sequence for any candidate that passes screening:

1. Update this plan with the candidate-specific theory and hard upper bound before code changes.
2. Implement the smallest default-off source change possible.
3. Run the lightweight sequential top1 verifier on the accepted fixed France text if logits can change.
4. Only after verifier pass, run the strict cold France benchmark under the 16GB cgroup.
5. Record all metrics: `eval_tok_s`, `prompt_tok_s`, TTFT, full output, correctness judgment, elapsed time, cgroup memory/file/page-cache stats, OOM counters, pack/cache counters, VRAM cache counters, build hashes, source head, exact env, exact command, and run directory.
6. If it is a compliant new SOTA, immediately commit and push source, plan, artifacts, scripts, and profiles to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then reproduce from pushed source before declaring it accepted.
7. If token rate regresses, correctness fails, TTFT exceeds the accepted gate for a promoted result, RAM exceeds 16GB including page cache, or counters show hidden fallback/cache collapse, revert runtime source changes and keep only rejected records/docs.

Current next action:

- Produce the screening artifact first. If no candidate clears the screening gate, do not write speculative runtime code; the next plan update must explicitly state the missing hard bound or missing exact mechanism needed to continue toward `10 tok/s`.

### 2026-07-04T03:55Z Exact Up/Down Candidate Screening Result

Artifacts:

- `.Agent/run-tools/analyze_exact_updown_candidates.py`
- `.Agent/runs/20260704-vendor-ds4-coldstart/exact-updown-candidate-screening.json`
- screening JSON sha256: `b80a28749cadddd56473608b00379481a18fff6508f041835565ec8f85062768`
- source head for artifact: `ff9e2e8acb0bdace45dfc6792f094b2b09152fdc`

Current accepted SOTA remains unchanged:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=32892.55329 ms`
- `TTFT_limit=33617.688744 ms`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15102607360`
- `ram_ok=true`
- `correctness_ok=true`
- run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`

Screening verdict:

- No already-defined concrete runtime candidate passes the plan's screening gate.
- Do not run another full-model benchmark until a new exact compute/offload/layout mechanism has:
  - hard measured/theoretical bound above tie;
  - explicit VRAM budget that does not silently steal the accepted gate cache;
  - 16GB cgroup RAM plan including page cache;
  - correctness verifier path before performance if logits can change.

Candidate decisions recorded:

- `current_ds4_hot_dispatch`: rejected before model run. Preserving the accepted gate cache gives only `64.60 ms` optimistic fallback saving and `4.409 tok/s` ceiling.
- `true_skip_zero_row`: rejected by bound. Perfect existing zero/group elimination saves only `82.008 ms` and reaches only `4.412 tok/s`.
- `existing_one_stream_gpu_updown_cache_classes`: rejected for correctness and performance. Hot64 produced a semantic France error at `3.6 tok/s`; no-filter produced incomplete output at `1.8 tok/s`.
- `cuda_graph_enabled_on_current_sota`: rejected/tie-regress. Strict cold run measured `4.1 tok/s`.
- `cpu_repack_and_transient_repack`: rejected. Per-op transient repack fails microbench threshold, and full-model transient down-repack measured `3.4 tok/s`.
- `page_source_prewarm_packmmap_packdirect`: rejected by negative I/O/page-source bound. Direct/mmap variants either tied or regressed.
- `chunk_affinity_scheduler_only`: rejected by bound or RAM. Thread spread is about `2.94s`, chunk32 regressed to `3.5 tok/s`, and single-row chunks40 violated the RAM gate.
- `algorithmic_multi_token_without_local_draft_or_mtp`: closed because no compatible local DeepSeek draft model or MTP/NextN GGUF exists.

Next active work:

1. Inspect the active CPU fallback code path, especially `ggml_compute_forward_mul_mat_id`, DS4 fused up/gate fallback, MXFP4 dot dispatch, source tensor access, and thread chunk construction.
2. Identify a new exact mechanism that reduces actual up/down source/dot work. It must be materially different from:
   - current DS4 hot-dispatch;
   - one-stream GPU up/down cache;
   - persistent or transient repack;
   - page-touch/madvise/packmmap/packdirect;
   - chunk-size or affinity scheduling.
3. Before implementation, write a candidate-specific plan section with:
   - exact code path;
   - mathematical/logical correctness argument;
   - expected removable milliseconds;
   - VRAM/RAM/page-cache budget;
   - theoretical token-rate ceiling;
   - top1 verifier requirement;
   - strict cold benchmark command.
4. If no source-level exact mechanism can be found, record that evidence explicitly and pivot only to a new external artifact class, such as a compatible DeepSeek draft/MTP model. Do not promote or claim progress from closed/tie candidates.

### 2026-07-04T04:15Z Current Plan Update: Route-Specific CPU Down Prefetch Bound

Current accepted SOTA remains unchanged:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=32892.55329 ms`
- `TTFT_limit=33617.688744 ms`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15102607360`
- `ram_ok=true`
- `correctness_ok=true`
- run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- reproduction branch: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- source head before this plan update: `d558a10bdb85d4d8bc83da7652cfde37a785aa09`

New bottleneck reading from source inspection:

- Active SOTA is gate-only one-stream cache. The gate tensor is streamed/cached; `ffn_up_exps` and `ffn_down_exps` are still CPU fallback.
- `cpu-moe-walltrace-result.json` shows gate rows accepted by the stream path, while up/down rows are declined and handled by CPU fallback.
- Current fallback profile remains the dominant measured cost:
  - total up/down CPU fallback: `26438.059 ms`
  - up fallback: `12786.863 ms`
  - down fallback: `13651.196 ms`
  - decode fallback: `19029.457 ms`
  - prompt fallback: `7408.602 ms`
- No-drop diagnostic shows page/source stalls are a real part of the cold penalty, but cannot be promoted because global warm page cache violates the strict cold-start rule.
- The most relevant cold-vs-no-drop deltas for this candidate are:
  - down total delta: `6276.748 ms`
  - down decode delta: `5023.621 ms`
  - up decode delta: `5464.241 ms`

Candidate under design:

- Name: `route_specific_cpu_down_prefetch_from_up`.
- Exact code path: `ggml_compute_forward_mul_mat_id` in `ggml/src/ggml-cpu/ggml-cpu.c`, only for MoE tensors whose names identify `ffn_up_exps` and `ffn_down_exps`.
- Mechanism: after routing is known for an `ffn_up_exps` CPU fallback, prefetch the same layer's routed `ffn_down_exps` expert pages before the down op runs. The expected overlap window is the current up fallback plus activation/gate scheduling before the down fallback consumes those pages.
- Math/correctness: this must not change any tensor value, routing decision, expert order, top-k value, quantized dot product, or accumulation order. It only asks the kernel to make future mapped pages resident sooner. If implementation stays page-timing-only, the France semantic correctness check is still mandatory and the top1 verifier is optional. If any compute/layout/logit path changes, the lightweight sequential top1 verifier is mandatory before performance.
- Why it is distinct from rejected paths:
  - not broad `GGML_MOE_CPU_WILLNEED=1`, because that prefetched the current tensor just before compute and already regressed;
  - not blocking touch, because it must not synchronously read pages before compute;
  - not packmmap/packdirect/O_DIRECT staging, because it does not copy expert payload into a new source buffer;
  - not one-stream up/down cache or DS4 hot-dispatch, because it does not consume VRAM or change CPU/GPU split of expert compute.

Hard-bound requirement before code:

1. Create `.Agent/run-tools/analyze_cpu_down_prefetch_bound.py`.
2. Generate `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-down-prefetch-overlap-bound.json`.
3. The artifact must include input hashes, source head, accepted SOTA metrics, cold/no-drop fallback deltas, prior broad `WILLNEED` rejection, and a clear verdict.
4. Minimum theoretical model:
   - if all down decode page/source delta (`5023.621 ms`) can be hidden with no overhead, optimistic eval ceiling is about `5.25 tok/s`;
   - if all down total delta (`6276.748 ms`) can be hidden with no overhead, optimistic ceiling is about `5.5 tok/s`;
   - this is not a path to `10 tok/s` by itself. It is only a possible incremental SOTA candidate that may stack with a later exact up/source/gate or algorithmic improvement.
5. Do not patch runtime source or run a full model benchmark if the bound artifact says the ceiling is a tie, if overlap is not plausible, or if page-cache pressure is likely to erase the gain under the 16GB cgroup.

Implementation plan if the bound passes:

1. Add a default-off env flag, for example `GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP=1`.
2. Add a small CPU-side registry for MoE up/down tensor metadata: tensor name, data pointer, expert count, expert stride/bytes, and layer/tensor identity.
3. Register `ffn_up_exps` and `ffn_down_exps` tensors at the beginning of `ggml_compute_forward_mul_mat_id`.
4. On `ith == 0`, after routing counts are available for an up tensor, find the matching down tensor and call the existing `ggml_moe_cpu_willneed_pages()` only for actually routed experts.
5. Add counters printed at exit or trace time: calls, advised experts, advised bytes, missing matching down tensors, and failures. These counters must be recorded in the run artifact.
6. Keep the default runtime path unchanged unless the env flag is set.

Strict benchmark gate if implemented:

- Use the same accepted SOTA strict runner config:
  - cold `drop_caches`
  - 16GB cgroup including page cache
  - `MemorySwapMax=0`
  - `cpu_moe=40`
  - `GGML_MOE_VRAM_CACHE_GB=0`
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
  - gate-only `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`
  - O_DIRECT gate expert pack
  - `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`
  - CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`
- Promotion requires all of:
  - `eval_tok_s > 4.4`
  - `TTFT <= 33617.688744 ms`
  - `memory_peak_bytes <= 16000000000`
  - page cache included in cgroup memory accounting
  - `oom=0`, `oom_kill=0`, `ram_limit_killed=false`
  - France output semantic, coherent, and complete
  - no gate pack direct failures or hidden cache collapse
- If it improves throughput but TTFT exceeds the accepted gate, record and commit it as `not_accepted`, then continue working to reduce TTFT. Do not promote.

Required record/push discipline:

1. Before implementation, commit and push the bound artifact plus this plan update to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`.
2. If implementation produces a compliant new SOTA, immediately record all reproduction metadata: source head, build hashes, env, command, run directory, full metrics, full France output, cgroup memory/page-cache stats, OOM counters, cache/pack counters, and artifact hashes.
3. Immediately commit and push source, plan, scripts, profiles, and run artifacts to the same `ssd/vendor/deepseek-token-rate-16gb` branch.
4. Rebuild/re-run from the pushed source and only then declare the SOTA accepted.
5. If the candidate regresses, fails correctness, exceeds RAM, or violates TTFT for promotion, revert runtime source changes and keep only rejected records/docs.

Next action:

- Produce `cpu-down-prefetch-overlap-bound.json` first. If it passes the hard-bound gate, implement the smallest default-off prefetch patch and run one strict cold France benchmark. If it fails, do not implement this path; return to exact up/down compute/offload redesign or a new compatible DeepSeek draft/MTP artifact.

### 2026-07-04T04:24Z CPU Down Prefetch Bound Result

Artifacts:

- `.Agent/run-tools/analyze_cpu_down_prefetch_bound.py`
- `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-down-prefetch-overlap-bound.json`
- script sha256: `30309b4ca7f12507b0c0a43ba927eb4547672b32fe8dc45288c5be8db55bfd8c`
- bound JSON sha256: `5c7a9a466b45317065f37975ff5d9983846ddb9e569239e8e734ffa256afba2b`
- source head for artifact: `6af293d40f65e5ef1a7523ec8ddad10e023e7190`

Bound result:

- Accepted SOTA remains `4.4 tok/s`, `TTFT=32892.55329 ms`, 16GB cgroup including page cache, correctness pass.
- Active path is confirmed as gate-only one-stream cache; up/down tensors remain CPU fallback.
- Cold-vs-no-drop fallback deltas from parsed `fallback-profile.csv`:
  - `down_decode_delta=5023.621 ms`
  - `down_prompt_delta=1253.127 ms`
  - `down_total_delta=6276.748 ms`
  - `up_decode_delta=5464.241 ms`
- No-overhead ceiling if all down decode delta is hidden: `5.249 tok/s`.
- No-overhead ceiling if half of down decode delta is hidden: `4.787 tok/s`.
- No-overhead ceiling if all down total delta landed in the generation window: `5.515 tok/s`.
- Savings needed from the accepted generation-window estimate:
  - `4.5 tok/s`: about `689.943 ms`
  - `5.0 tok/s`: about `3725.694 ms`
  - `10.0 tok/s`: about `17386.570 ms`

Scale and risk:

- Down decode profile has `2627` unique down expert entries and `17940` calls.
- Unique down decode payload is about `10.90 GiB`; call-weighted down decode advised range would be about `74.46 GiB` if implemented naively per call.
- Down total unique payload is about `14.59 GiB`; call-weighted down total advised range is about `79.32 GiB`.
- This means the probe must stay route-specific, nonblocking, and default-off. It must not add a new copied pack or persistent host buffer, and all page cache remains charged to the strict 16GB cgroup.
- Prior broad/current-tensor `WILLNEED` remains a negative control: the current-tensor CPU willneed probe measured only `4.0 tok/s`; stream opwide willneed measured `1.4 tok/s`; late10 CPU willneed measured `2.5 tok/s`.

Decision:

- `route_specific_cpu_down_prefetch_from_up` passes the hard-bound gate for exactly one default-off runtime probe.
- This is not a standalone path to `10 tok/s`; it is an incremental SOTA candidate that may stack with later exact up/source/gate or algorithmic work.
- Proceed to implementation only with a default-off env flag such as `GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP=1`.
- The implementation must be page-timing-only: no tensor value, routing decision, top-k value, dot product, accumulation order, or CPU/GPU compute split may change.
- If the patch remains page-timing-only, semantic France correctness is mandatory and top1 verification is optional. If any compute/layout/logit path changes, run the lightweight sequential top1 verifier before performance.

Implementation gate:

1. Register MoE up/down tensor metadata in CPU code without changing default behavior.
2. On `ffn_up_exps` CPU fallback, after routing counts are available, prefetch matching routed `ffn_down_exps` pages for the same layer.
3. Add counters for calls, advised experts, advised bytes, missing down tensor matches, and failures.
4. Build and run one strict cold France benchmark with the accepted SOTA config plus `GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP=1`.
5. Reject and revert runtime source if:
   - `eval_tok_s <= 4.4`;
   - TTFT exceeds `33617.688744 ms` for a promoted result;
   - cgroup memory exceeds 16GB including page cache;
   - any OOM/kill/ram-limit flag appears;
   - France output is incomplete, incoherent, or semantically wrong;
   - gate pack direct failures appear or cache counters collapse;
   - counters show missing down tensor registration for most decode calls.
6. If the probe produces a compliant new SOTA, immediately record full reproduction metadata, commit and push source/plan/scripts/artifacts to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then reproduce from pushed source before promotion.

### 2026-07-04T04:38Z CPU Down Prefetch First Strict Run

Implementation:

- Added a default-off page-timing probe in `ggml/src/ggml-cpu/ggml-cpu.c`.
- Env flag: `GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP=1`.
- The default path remains disabled unless the env flag is set.
- The patch registers `ffn_down_exps` tensor metadata by layer and, when `ffn_up_exps` CPU fallback is about to begin, asks the kernel to prefetch the matching routed down expert pages.
- The patch does not change routing, top-k, tensor values, dot products, accumulation order, or CPU/GPU compute split.
- Prefetch is issued by `ith==0` after the existing fallback-preparation barrier and without adding a new all-thread barrier, so the request can overlap with other CPU fallback workers.

First strict cold run:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T043504Z-20260704_cpu_down_prefetch_from_up_probe/france-cpu40-vram0gb`
- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-down-prefetch-from-up-candidate-result.json`
- artifact sha256: `f022f8d90c2792a889b9b2ec6a0c82333b0aeb48109ed186dadabe7844432294`
- Source head during run: `fad73383c89edb39e3b290f98b95f0fc5ab783b6` with `ggml/src/ggml-cpu/ggml-cpu.c` dirty.
- Build hashes:
  - `llama-cli`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
  - `libggml_cpu`: `b357ca59f1d73c2fa290a384dfecc4004b16aae5425b84725a690ebc37b7dc89`
  - `libggml_cuda`: `eaeb2b8d199c6542f7bba83959e3f21732b721620cf87dec10ac53a0c1facbbc`

Metrics:

- `eval_tok_s=4.6`
- `prompt_tok_s=1.9`
- `TTFT=32564.078089 ms`
- `TTFT_limit=33617.688744 ms`
- `elapsed_seconds=62.5`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15095713792`
- `ram_ok=true`
- `oom_seen=false`
- `ram_limit_killed=false`
- `correctness_ok=true`
- France answer is semantic, coherent, and complete.

Counters:

- Gate pack: `hits=4886`, `misses=0`, `direct_reads=4886`, `direct_failures=0`, `direct_fallbacks=0`.
- Gate prefill: `attempted=3000`, `inserted=3000`, `pack_misses=0`, `read_failures=0`, `elapsed_ms=4691.859`.
- VRAM cache: `hits=33265`, `misses=1886`, `hit_rate=94.6%`.
- CPU down prefetch: `enabled=1`, `entries=40`, `down_registered=40`, `down_updated=5600`, `calls=5640`, `matched=5600`, `missing=40`, `advised_experts=18974`, `advised_bytes=84556644352`, `madvise_failures=0`.

Pre-push decision:

- This is a compliant candidate SOTA versus the previous accepted `4.4 tok/s`: throughput improved, TTFT stayed below the gate, RAM stayed within the strict 16GB cgroup including page cache, pack/cache counters are clean, and correctness passed.
- It is not final accepted SOTA yet because the first run was from dirty source.
- Required immediate action: commit and push source, plan, and candidate artifact to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then rebuild/rerun the same strict cold command from the pushed source.
- Promote only if pushed-source reproduction remains `eval_tok_s > 4.4` with all gates passing.

### 2026-07-04T04:45Z CPU Down Prefetch Pushed Repro Rejection

Pushed-source reproduction:

- Source commit tested: `834b8c214347d7161d5e737c0f984cdfe354eaad` (`vendor-ds4: add cpu down prefetch probe`)
- Remote branch before rejection: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T044151Z-20260704_cpu_down_prefetch_from_up_pushed_repro/france-cpu40-vram0gb`
- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-down-prefetch-from-up-pushed-repro-rejected.json`
- artifact sha256: `d7fdbb5a1646ba04a68c3f380c85ee361202748a273e738f88036ff6951193ef`
- Build hashes:
  - `llama-cli`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
  - `libggml_cpu`: `b357ca59f1d73c2fa290a384dfecc4004b16aae5425b84725a690ebc37b7dc89`
  - `libggml_cuda`: `eaeb2b8d199c6542f7bba83959e3f21732b721620cf87dec10ac53a0c1facbbc`

Metrics:

- `eval_tok_s=4.4`
- `prompt_tok_s=2.0`
- `TTFT=31898.198672 ms`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15100809216`
- `ram_ok=true`
- `oom_seen=false`
- `ram_limit_killed=false`
- `correctness_ok=true`
- France answer remained semantic, coherent, and complete.

Counters:

- Gate pack: `hits=4886`, `misses=0`, `direct_reads=4886`, `direct_failures=0`, `direct_fallbacks=0`.
- Gate prefill: `attempted=3000`, `inserted=3000`, `pack_misses=0`, `read_failures=0`, `elapsed_ms=4424.543`.
- VRAM cache: `hits=33265`, `misses=1886`, `hit_rate=94.6%`.
- CPU down prefetch: `enabled=1`, `entries=40`, `down_registered=40`, `down_updated=5600`, `calls=5640`, `matched=5600`, `missing=40`, `advised_experts=18974`, `advised_bytes=84556644352`, `madvise_failures=0`.

Decision:

- Reject as a promoted SOTA. The dirty-source first run reached `4.6 tok/s`, but the required pushed-source reproduction tied the accepted SOTA at `4.4 tok/s` instead of exceeding it.
- All correctness/RAM/TTFT/cache gates passed, so the evidence is useful diagnostic data, but it is not an accepted token-rate improvement.
- Runtime source must be reverted to the previous accepted path. Keep only the candidate and rejected repro artifacts plus this plan record.
- Current accepted SOTA remains `eval_tok_s=4.4` from `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`.
- Do not retry this exact route-specific down prefetch probe unless a new design reduces the `84.56GB` call-weighted advice volume, avoids the pushed-source tie, and has a new hard-bound/update section before implementation.

### 2026-07-04T04:53Z Deduplicated Down Prefetch Bound

Artifacts:

- `.Agent/run-tools/analyze_cpu_down_prefetch_dedup_bound.py`
- `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-down-prefetch-dedup-bound.json`
- script sha256: `8d0ff32f960d39f8790eb18037f2833596bddf61e8b556b97f222f6b556e12c0`
- bound JSON sha256: `52a2eb6c9d99433cd53d6e9a83aff421d208a3812385e93676fab0e454d42570`
- source head for artifact: `0107390f57dc0d8fc3b3f73fe30fe599e945dd49`

Why this is a new candidate:

- The rejected route-specific prefetch probe was exact and had clean counters, but the pushed-source reproduction tied at `4.4 tok/s`.
- Its counters showed repeated advice volume: `advised_experts=18974`, `advised_bytes=84556644352` (`78.75 GiB`).
- The observed unique down payload from the bound artifact is much smaller:
  - down total unique payload: `14.5886 GiB`, `3515` entries;
  - down decode unique payload: `10.9031 GiB`, `2627` entries.
- A dedup bitmap can reduce advice volume by about `81.5%` versus the rejected pushed run if it limits advice to unique routed down experts.
- This is materially different from the rejected probe because it targets the measured overhead gap instead of repeating the same per-call advice pattern.

Candidate:

- Name: `route_specific_cpu_down_prefetch_from_up_dedup`.
- Env flag: `GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP_DEDUP=1`.
- Code path: `ggml/src/ggml-cpu/ggml-cpu.c:ggml_compute_forward_mul_mat_id`.
- Mechanism: register `ffn_down_exps` tensors by layer, then when `ffn_up_exps` CPU fallback is about to run, prefetch only matching routed down experts that have not been advised before in this process.
- Additional state: small per-layer/expert advised bitmap; no expert payload copy, no persistent host pack, no VRAM consumption.
- Correctness: page-timing only. No routing, top-k, tensor value, dot-product, accumulation, or CPU/GPU compute split changes.

Inherited theoretical bound:

- Same best-case compute/page ceiling as the previous route-specific prefetch analysis:
  - hide all down decode delta: `5.249 tok/s`;
  - hide half down decode delta: `4.787 tok/s`;
  - not a standalone path to `10 tok/s`.
- Savings needed from accepted generation-window estimate:
  - `4.5 tok/s`: about `689.943 ms`;
  - `5.0 tok/s`: about `3725.694 ms`;
  - `10.0 tok/s`: about `17386.570 ms`.

Decision:

- Passes hard-bound gate for one default-off dedup probe.
- Must first commit and push this bound artifact and plan update to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- Then implement the smallest default-off source change possible.
- Run one strict cold France benchmark with accepted SOTA config plus `GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP_DEDUP=1`.
- If first run is `>4.4 tok/s`, commit and push immediately, rebuild from pushed source, and reproduce before promotion.
- Reject and revert runtime source if pushed-source reproduction is `<=4.4 tok/s`, TTFT exceeds the promotion gate, RAM exceeds 16GB including page cache, correctness fails, cache/pack counters collapse, or dedup counters show advice volume remains near the rejected `78.75 GiB` scale.

### 2026-07-04T05:05Z Deduplicated Down Prefetch Rejection

Run:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T050219Z-20260704_cpu_down_prefetch_dedup_probe/france-cpu40-vram0gb`
- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-down-prefetch-dedup-rejected.json`
- artifact sha256: `3b3b658eb1bfec1400323ef77dada61d9bd9f0bcd32541a6accf7279bb8341c4`
- Source head during run: `813ab55f9c5f202221a39d89bd6d9a615139c632` with `ggml/src/ggml-cpu/ggml-cpu.c` dirty.

Metrics:

- `eval_tok_s=4.1`
- `prompt_tok_s=1.9`
- `TTFT=33348.567464 ms`
- `TTFT_limit=33617.688744 ms`
- `elapsed_seconds=66.5`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=12968255488`
- `ram_ok=true`
- `oom_seen=false`
- `ram_limit_killed=false`
- `correctness_ok=true`
- France answer remained semantic, coherent, and complete.

Counters:

- Gate pack: `hits=4886`, `misses=0`, `direct_reads=4886`, `direct_failures=0`, `direct_fallbacks=0`.
- Gate prefill: `attempted=3000`, `inserted=3000`, `pack_misses=0`, `read_failures=0`, `elapsed_ms=4727.889`.
- VRAM cache: `hits=33265`, `misses=1886`, `hit_rate=94.6%`.
- Dedup prefetch: `enabled=1`, `entries=40`, `down_registered=40`, `down_updated=5600`, `calls=5640`, `matched=5600`, `missing=40`, `advised_experts=2970`, `skipped_already=16004`, `out_of_range=0`, `advised_bytes=13235650560`, `madvise_failures=0`.

Decision:

- Reject before push as a SOTA candidate. The dedup design successfully reduced advice volume from about `78.75 GiB` to about `12.33 GiB`, but strict cold speed regressed to `4.1 tok/s`.
- This closes the current page-timing prefetch family: broad `WILLNEED`, route-specific repeated prefetch, and dedup route-specific prefetch all failed to produce a reproducible accepted SOTA.
- Likely gap: under the 16GB cgroup, readahead/page-cache competition and prefetch timing offset any reduction in major faults or repeated advice overhead.
- Runtime source must be reverted to the accepted path. Keep only bound/rejection artifacts and this plan record.
- Current accepted SOTA remains `eval_tok_s=4.4`, run `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`.
- Next work should not try another synchronous `madvise`/page-touch variant. It must either target exact compute/offload of up/down fallback or introduce a genuinely asynchronous/lower-pressure source mechanism with a new hard-bound section first.

### 2026-07-04T05:32Z MulMatId Conversion Skip Bound

Candidate:

- Name: `delay_or_skip_mul_mat_id_src1_q8_conversion_when_gpu_stream_consumes_all_rows`.
- Code area: `ggml_compute_forward_mul_mat_id` currently prepares `src1` into `vec_dot_type` before routing/GPU-stream decisions.
- Reason it was considered exact: if a GPU stream path consumes all routed rows for an op, preparing the CPU fallback input can be avoided without changing routing, tensor values, dot products, accumulation order for executed rows, logits, or token selection.

Diagnostic run:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T051642Z-20260704_mulmatid_convert_profile_diagnostic/france-cpu40-vram0gb`
- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/mulmatid-convert-skip-bound.json`
- artifact sha256: `5bc6105af0072c0f1cc451a101d30fe0fcb1b50f28f8de356661a1ba6010bb6e`
- Source head: `a63ad426173f4da10b5f16f3d009f7a7d288b29c`
- Source dirty: `false`
- `combined.txt` sha256: `dc42f0bb88d840681cc608337a0c293221b7b8bf6d7a515c6c83219c000a7f28`
- `summary.json` sha256: `0275ef995c25e57f13d8196227ab565991c9d17a6bd14527bb4966a2a08617c4`

Metrics:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=31414.364861 ms`
- `elapsed_seconds=62.22`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15103029248`
- `ram_ok=true`
- `oom_seen=false`
- `ram_limit_killed=false`
- `correctness_ok=true`
- France answer remained semantic, coherent, and complete.

Profile line:

- `[kimi_cpu_moe_profile] down calls=16920 total=2.038 ms/call convert_t0=0.002 route=0.004 route_barrier=0.001 cuda_batch=0.000 cuda_single=0.476 post_cuda_barrier=0.001 fallback_t0=1.549 batch_accept=0 batch_decline=0 single_accept=35151 single_decline=38222`

Hard bound:

- Calls: `16920`
- Measured conversion component: `0.002 ms/call`
- Total ideal removable conversion time: `33.84 ms`
- Accepted SOTA reference decode window: `63.94s - 32.89255329s = 31.04744671s`
- Accepted decoded-token estimate at `4.4 tok/s`: `136.608765524`
- No-overhead ceiling if all measured conversion time disappeared: `4.404800989 tok/s`
- Absolute ideal gain: about `0.004801 tok/s`

Decision:

- Reject with no source patch and no benchmark run. The candidate is exact in principle, but the measured upper bound is far below run-to-run noise and cannot become a meaningful SOTA improvement.
- This confirms the current bottleneck is not `mul_mat_id` src1 conversion. The remaining material target is CPU up/down fallback time plus source/page stalls around those fallbacks.
- Do not implement delayed conversion unless a future profile shows conversion time at least hundreds of milliseconds under the accepted strict cold configuration.

### 2026-07-04T05:40Z Latest Next Plan After Prefetch And Conversion Closure

Current accepted SOTA remains:

- `eval_tok_s=4.4`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Branch for records/source: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- Required git identity: `L-Ark <fliangae@connect.ust.hk>`
- TTFT acceptance gate: `<=33617.688744 ms`
- Host RAM gate: strict 16GB cgroup including page cache and process memory, `MemorySwapMax=0`

Closed families:

- Synchronous page prefetch/touch for CPU down fallback is closed: broad `WILLNEED`, route-specific repeated prefetch, and deduplicated route-specific prefetch all failed to produce a reproducible accepted SOTA.
- `mul_mat_id` conversion skip is closed by hard-bound: ideal ceiling only about `4.405 tok/s`.
- DS4 hot dispatch, one-stream up/down cache, corrected down batch, O_DIRECT top-k staging, compact mmap, no-draft ngram, target-only lookahead, and earlier approximate top-k pruning remain closed unless a new hard-bound and correctness gate justify reopening.

Next design step:

1. Build a fresh post-prefetch bottleneck table from accepted SOTA artifacts, with separate rows for:
   - CPU up fallback prompt/decode time;
   - CPU down fallback prompt/decode time;
   - source/page stall delta from cold vs no-drop;
   - GPU-stream accept/decline and dispatch overhead;
   - synchronization/tail gap;
   - any remaining pack/direct gate source time.
2. For each row, compute a no-overhead token-rate ceiling using the accepted SOTA decode window. Only rows with credible ceiling above `4.4 tok/s` should advance to candidate design.
3. Prioritize exact candidates in this order:
   - true CPU up/down fallback reduction or offload that preserves logits and token-level top1 under the existing verifier;
   - genuinely asynchronous/lower-pressure source movement for the exact routed up/down experts, with bounded memory and no repeated synchronous `madvise`;
   - micro-optimizations only if the hard-bound is at least several hundred milliseconds under strict cold.
4. Before any implementation, update this plan with the chosen candidate's theory, hard metrics, upper bound, correctness risk, RAM/VRAM budget, expected TTFT effect, rollback rule, and exact benchmark command/env.
5. After implementation, run the token-level correctness gate first when the change touches arithmetic/offload. Then run strict cold France benchmark under the accepted 16GB cgroup.

Promotion rule remains unchanged:

- If a run is `eval_tok_s > 4.4`, `ram_ok=true`, `correctness_ok=true`, and `TTFT <=33617.688744 ms`, immediately commit and push source plus artifacts to `ssd/vendor/deepseek-token-rate-16gb`.
- After pushing, rebuild from pushed source and reproduce the result before calling it accepted SOTA.
- If pushed-source reproduction fails to exceed `4.4 tok/s`, or RAM/TTFT/correctness/cache counters fail, revert runtime source, keep rejected artifacts/docs, and push the rejection record.
- Each accepted or rejected experiment must include run path, full env/config, binary/source hashes, output answer, token rates, TTFT, elapsed time, cgroup memory/file-cache stats, cache counters, and artifact sha256.

### 2026-07-04T05:55Z Post-Prefetch Bottleneck Hard-Bound

Artifacts:

- Tool: `.Agent/run-tools/analyze_post_prefetch_bottleneck.py`
- Tool sha256: `663109370fa8195411d89745d5408b8820d5a20d964e1ffbb9032dfc49d71417`
- Result: `.Agent/runs/20260704-vendor-ds4-coldstart/post-prefetch-bottleneck-hard-bound.json`
- Result sha256: `f0474f3d66e5fb6ec22d7b0e86f727f06bb17f377deab4174daab1837b333c7f`
- Source head used for analysis: `0cc181c24480a710899aab96adf67261e406d396`

Baseline used for all ceilings:

- Accepted SOTA run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- `eval_tok_s=4.4`
- `elapsed_seconds=63.94`
- `TTFT=32892.55329 ms`
- Decode-window estimate: `31.04744671 s`
- Decoded-token estimate: `136.608765524`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15102607360`
- `ram_ok=true`
- `correctness_ok=true`

Top hard bounds under the accepted SOTA window:

| Row | Removable time | No-overhead ceiling | Decision |
| --- | ---: | ---: | --- |
| all decode up/down fallback + gate one-stream total removed | `27079.681 ms` | `34.430 tok/s` | reference only; not one implementation |
| all prompt+decode up/down fallback removed | `26438.059 ms` | `29.637 tok/s` | reference only |
| all decode up/down fallback removed | `19029.457 ms` | `11.367 tok/s` | only class that can reach 10 without speculation |
| cold-vs-nodrop total fallback delta removed | `12791.007 ms` | `7.483 tok/s` | diagnostic only; no-drop not promotable |
| cold-vs-nodrop decode fallback delta removed | `10487.862 ms` | `6.645 tok/s` | diagnostic only |
| decode up fallback removed | `9697.125 ms` | `6.398 tok/s` | useful only as a step, not enough alone |
| decode down fallback removed | `9332.332 ms` | `6.291 tok/s` | useful only as a step, not enough alone |
| gate one-stream total time removed | `8050.224 ms` | `5.940 tok/s` | not enough alone |

Interpretation:

- The next material target is exact full or near-full decode up/down fallback reduction.
- Micro-optimizations below several hundred milliseconds, including the conversion skip, cannot move SOTA meaningfully.
- Single-side up-only or down-only offload cannot reach 10 by itself, but can be used as a correctness-screened stepping stone if it preserves top1.
- Existing fused up/gate, nofilter up/down stream, hot64 up/down stream, down batch, packmmap/packdirect, and synchronous prefetch attempts stay closed.
- The old phaseB hard-bound used a slower `4.1 tok/s` diagnostic window; the new table is the active bound because it uses the accepted pushed `4.4 tok/s` SOTA run.

### 2026-07-04T06:05Z Next Candidate: Split Up/Down Top1 Verifier

Why this is the next practical screen:

- Full nofilter up/down GPU stream already failed output correctness and performance, but it combined multiple effects: up, down, gate cache pressure, and much worse cache/source behavior.
- The new hard-bound shows `decode up fallback removed` and `decode down fallback removed` each have about `6.3-6.4 tok/s` no-overhead ceiling. Either side alone is not enough for the final target, but a side that preserves top1 is a viable subproblem for later stacking.
- The existing top1 verifier can reject arithmetic/offload candidates before a full strict cold benchmark.

Planned verifier-only probes:

1. `split-up-only-top1`:
   - Baseline: existing accepted SOTA top1 baseline `/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light/top1-baseline.json`.
   - Fixed text: `/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light/fixed-france-text.txt`.
   - Config: accepted SOTA env plus `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`.
   - Keep accepted gate prefill/profile/pack/cache settings.
   - Run only `llama-results --sequential-logits --top1-report`; compare JSON to baseline.
2. `split-down-only-top1`:
   - Same baseline/fixed text.
   - Config: accepted SOTA env plus `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_down_exps`.
   - Run only `llama-results --sequential-logits --top1-report`; compare JSON to baseline.

Acceptance for this screen:

- The screen passes only if `same_top1=145/145`, `first_mismatch_pos=-1`, no cgroup OOM/OOM kill, and RAM remains within the strict 16GB cgroup including page cache.
- This is not a SOTA benchmark and cannot be promoted by itself.
- If either side fails top1, reject that side without full benchmark.
- If either side passes top1, update this plan again with a performance hard-bound and then run one strict cold France benchmark for that side.

Rollback / source rule:

- No source changes are planned for the verifier probes; only env changes and artifacts.
- Any later source change must be default-off, pass this verifier first when it changes arithmetic/offload, then pass strict cold France correctness/RAM/TTFT/token-rate gates.

### 2026-07-04T06:18Z Split Up/Down Top1 Invalid Filter Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/split-updown-top1-invalid-filter-result.json`
- artifact sha256: `6f5bed6fd01370d082ae6b2b92f62826a86a028e1f996e9b5fb3bb00115d6b20`

Runs:

- Invalid up-filter run: `/root/lfz/runs/vendor-ds4-16gb/20260704T055054Z-20260704_split_up_only_top1/france-cpu40-vram0gb`
- Invalid down-filter run: `/root/lfz/runs/vendor-ds4-16gb/20260704T055356Z-20260704_split_down_only_top1/france-cpu40-vram0gb`

Observed metrics for both invalid runs:

- `same_top1=138/145`
- `first_mismatch_pos=4`
- `memory_peak_bytes=16000000000`
- `ram_ok=true`
- `oom_seen=false`
- Candidate top1 sha256: `0a650ac7f8b686a619eb99ff13c8948368b056e1a3fedcc288aca376086abf56`

Why these runs are invalid:

- `GGML_MOE_STREAM_ONE_NAME_FILTER` currently supports only a single substring:
  `return name && strstr(name, filter) != nullptr`.
- The planned comma values `ffn_gate_exps,ffn_up_exps` and `ffn_gate_exps,ffn_down_exps` match no tensor names.
- Stderr contained only `[moe_stream] enabled...` and no one-expert-pack / VRAM-cache lines, proving one-stream expert pack/cache did not run.
- Therefore these runs did not test up-only or down-only offload. They are diagnostic only and cannot be used to reject split up/down.

Next source candidate before rerun:

- Add comma-list support to `moe_stream_one_name_filter_allows()` behind the existing env behavior:
  - if the filter has no comma, preserve the current exact `strstr(name, filter)` behavior;
  - if the filter contains commas, split into tokens and allow if any non-empty token is a substring of the tensor name;
  - no env/default behavior change;
  - accepted SOTA `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps` must remain behaviorally unchanged.
- This is a small diagnostic source patch, not a SOTA optimization by itself.
- After the patch, rebuild `llama-results`, rerun the two verifier-only probes, and record whether top1 passes.
- If both split probes fail top1, revert the source patch and keep only artifacts/docs.
- If one split probe passes top1, update this plan with a strict cold performance-bound section before running any full token-rate benchmark.

### 2026-07-04 Latest Active Plan Update

This is the active plan after the invalid comma-filter verifier result. It supersedes the older `4.2 tok/s` promotion threshold in historical sections; the accepted cold-start SOTA is now `4.4 tok/s`.

Current accepted SOTA guard:

- Source/record branch: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- Required git identity for future pushes: `L-Ark <fliangae@connect.ust.hk>`.
- Accepted run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`.
- Metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`.
- TTFT acceptance gate remains `<=33617.688744 ms`.
- Any new accepted SOTA must beat `4.4 tok/s`; ties or lower results are diagnostics/rejections only.

Active bottleneck and hard-bound:

- The latest hard-bound artifact is `.Agent/runs/20260704-vendor-ds4-coldstart/post-prefetch-bottleneck-hard-bound.json` at sha256 `f0474f3d66e5fb6ec22d7b0e86f727f06bb17f377deab4174daab1837b333c7f`.
- Current bottleneck is CPU up/down fallback, especially decode fallback. The hard-bound says removing all decode up/down fallback gives an ideal no-overhead ceiling of `11.367 tok/s`; up-only or down-only decode removal alone only reaches about `6.3-6.4 tok/s`.
- Therefore the next useful step is not another page-prefetch sweep. It is a correctness-first split up/down offload screen, then only benchmark a side that preserves exact token-level top1.

Step 1: comma-list filter diagnostic patch

- Add comma-list support to `moe_stream_one_name_filter_allows()` only for `GGML_MOE_STREAM_ONE_NAME_FILTER` values containing commas.
- Preserve the old single-substring behavior exactly when the filter has no comma; this keeps the accepted `ffn_gate_exps` SOTA path unchanged.
- Treat this as a diagnostic source patch, not a promoted optimization. It exists only so `ffn_gate_exps,ffn_up_exps` and `ffn_gate_exps,ffn_down_exps` can actually exercise the intended paths.
- Run `git diff --check`, rebuild `llama-results`, and do not run a full token-rate benchmark before the top1 verifier passes.

Step 2: split up/down top1 verifier rerun

- Baseline: `/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light/top1-baseline.json`.
- Fixed text: `/root/lfz/runs/vendor-ds4-16gb/20260704T030025Z-results-top1-selfcheck-light/fixed-france-text.txt`.
- Up-only case: accepted SOTA env plus `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`.
- Down-only case: accepted SOTA env plus `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_down_exps`.
- Each case must run under strict 16GB cgroup with `MemorySwapMax=0`, cold `drop_caches`, and `llama-results --sequential-logits --top1-report`.
- Passing criteria: `same_top1=145/145`, `first_mismatch_pos=-1`, `memory_peak_bytes<=16000000000`, no cgroup OOM/OOM kill.
- Record run paths, env/config, candidate JSON hashes, top1 comparison, cgroup memory stats, stderr cache/pack counters, and source head in `.Agent/runs/20260704-vendor-ds4-coldstart/`.

Step 3: branch decisions after verifier

- If both up-only and down-only fail top1, revert the comma-list diagnostic source patch, rebuild the accepted runtime path, record the rejection, commit/push docs/artifacts only, and keep SOTA at `4.4 tok/s`.
- If exactly one side passes top1, update this plan with a side-specific performance hard-bound before running a full strict cold France benchmark. The benchmark is allowed only for the passing side.
- If both sides pass top1, benchmark them separately first; do not combine them until each side has an individual strict cold result and a measured cache/RAM/TTFT profile.

Step 4: strict cold benchmark gate for a passing side

- Benchmark command must use the accepted SOTA runner shape: vendor DeepSeek, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=0`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, accepted gate profile/top-k envs, O_DIRECT one expert pack, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, strict cold `drop_caches`, and 16GB cgroup including page cache.
- A result is accepted only if all are true: `eval_tok_s > 4.4`, France output is semantic/coherent/complete, `TTFT<=33617.688744 ms`, `memory_peak_bytes<=16000000000`, page cache is inside the cgroup, no OOM/OOM kill, no expert-pack direct failures/fallbacks, and cache counters are explained.
- A candidate with TTFT over the gate may still be committed as an explicitly rejected/diagnostic result, but it cannot be promoted as accepted SOTA until TTFT is brought back under the gate.

Step 5: mandatory reproducibility and push protocol

- When a compliant new SOTA appears, stop exploration immediately.
- Record full reproducibility metadata before leaving the run: exact run path, exact env/CLI, source head, build command, binary hashes, model/profile/pack hashes, token rates, TTFT, elapsed time, full output answer, cgroup `memory.peak`, `memory.current`, `memory.stat`, `memory.events`, page-cache bytes, pack/cache counters, and comparison to the previous `4.4 tok/s` SOTA.
- Commit source, plan, profiles, and artifacts, then push immediately to `ssd/vendor/deepseek-token-rate-16gb`.
- After push, clean rebuild from the pushed source and rerun strict cold. Promote only if the pushed-source rerun still passes every gate and beats `4.4 tok/s`.
- If pushed-source reproduction fails, revert runtime source to the accepted path, keep the failed artifact as rejected, update this plan, commit/push the rejection record, and keep the accepted SOTA unchanged.

### 2026-07-04 Split Up/Down Top1 List-Filter Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/split-updown-top1-listfilter-result.json`
- Artifact sha256: `71a1655c09011a7256c70658bfd3e655d4768c38d4e1d4b9e7d34c35debf60ad`
- Run root: `/root/lfz/runs/vendor-ds4-16gb/20260704T061834Z-20260704_split_updown_top1_listfilter`
- Source head during verifier: `51d8fff187079183832c4645bbb9a16b0a86ba3e` with only the diagnostic comma-list filter patch dirty.

Verifier patch:

- Added comma-list parsing to `moe_stream_one_name_filter_allows()` only for filters containing commas.
- Preserved the old single-substring behavior for normal `ffn_gate_exps`.
- This patch was diagnostic only. It was reverted immediately after the verifier failed, and `build-ds4-moe-stream` was rebuilt back on clean source.

Results:

| Case | Filter | same_top1 | first_mismatch_pos | memory_peak_bytes | OOM | Verdict |
| --- | --- | ---: | ---: | ---: | --- | --- |
| split-up-only | `ffn_gate_exps,ffn_up_exps` | `139/145` | `9` | `16000000000` | false | rejected |
| split-down-only | `ffn_gate_exps,ffn_down_exps` | `134/145` | `4` | `16000000000` | false | rejected |

Important counters:

- The comma-list filter worked this time. Both stderr logs include one expert pack and VRAM cache lines, unlike the earlier invalid-filter runs.
- Up-only: one expert pack `hits=4497 misses=19563 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=32590 misses=21060 hit_rate=60.7%`.
- Down-only: one expert pack `hits=4474 misses=19567 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=32609 misses=21041 hit_rate=60.8%`.

Decision:

- Do not run a token-rate benchmark for this split up/down one-stream path. It fails token-level top1 correctness before performance measurement.
- Current accepted SOTA remains `4.4 tok/s`.
- Keep the rejection artifact and plan record, but do not keep the diagnostic source patch in the runtime source.

Next active direction:

1. Keep the current bottleneck target as CPU up/down fallback; the hard-bound still shows full decode up/down fallback removal is the only non-speculative class with a `10 tok/s` ceiling.
2. Do not retry one-stream up/down benchmark variants unless a new arithmetic-equivalence check proves their logits/top1 match the CPU path.
3. Next optimization attempt must first isolate why up/down one-stream changes top1:
   - compare GPU one-stream up/down math against the CPU fallback on the same fixed rows and activations;
   - distinguish arithmetic drift from cache/admission trajectory changes;
   - only after exact or top1-equivalent arithmetic is proven, revisit any up/down offload benchmark.
4. If arithmetic equivalence cannot be made exact cheaply, return to an exact CPU-side fallback reduction path with a new hard-bound, such as asynchronous source movement that does not displace the accepted gate cache and does not rely on global warm page cache.
5. Every new candidate still must pass RAM <=16GB including page cache, France correctness, TTFT <=`33617.688744 ms` for promotion, and pushed-source reproduction before becoming SOTA.

### 2026-07-04 Q8_0 CUDA Up/Down Path Screening

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/q80-cuda-updown-path-screening.json`
- Artifact sha256: `5db42dce9f8302f55db00b56364132cfe4b23fd597d087d8c1c2d21cbe4aac91`

Why the split verifier failed:

- Existing historical compare artifact `.Agent/runs/20260704-vendor-ds4-coldstart/updown-gpu-compare-result.json` showed individual streamed ops are close to CPU fallback (`~1e-6` max-abs scale), so this is not a gross CUDA kernel bug.
- The new fixed-text top1 verifier is stricter and failed at low-margin positions. That means the small numerical differences are still enough to alter token-level logits after multiple layers.
- Source inspection explains the mismatch:
  - CPU MXFP4 fallback uses `ggml_vec_dot_mxfp4_q8_0` with activation type `GGML_TYPE_Q8_0`.
  - CUDA MMVQ MXFP4 uses `quantize_row_q8_1_cuda` and `vec_dot_mxfp4_q8_1`.
  - `ggml_cuda_moe_stream_one()` currently requires F32 `src1` and ignores the nominal `src1_q8_1` parameters.
  - CUDA has a block-level `quantize_f32_q8_0_block` helper, but there is no current MXFP4 x Q8_0 MMVQ path.

Hard bound for this candidate class:

- Current accepted SOTA decode window estimate: `31.04744671 s`.
- Decoded token estimate: `136.608765524`.
- Target `10 tok/s` decode window: `13.6608765524 s`.
- Required decode saving: `17.3865701576 s`.
- Ideal full decode up/down fallback removal: `19.029457 s`, ceiling `11.367 tok/s`.
- Therefore a full decode up/down offload path has only about `1.6428868424 s` total overhead budget to still reach `10 tok/s`.

Decision:

- Close the existing Q8_1 one-stream up/down path for SOTA work unless an exact verifier later proves top1 stability.
- Do not spend more time on source movement/direct/prefetch variants; `source-movement-async-bound.json` already shows top64/top128/top256 direct movement is net negative under the 16GB cgroup.
- The next exact GPU/offload candidate is a default-off MXFP4 x Q8_0 one-stream path for up/down:
  - pass CPU `params->wdata` Q8_0 rows and row size from `ggml-cpu.c` into the CUDA one-stream call under an explicit env gate;
  - add a specialized CUDA MXFP4 x Q8_0 rows helper instead of the current Q8_1 helper;
  - keep accepted gate-only SOTA behavior unchanged by default;
  - first run the fixed-text `llama-results` top1 verifier, not a token-rate benchmark.

Implementation rules for this candidate:

1. Before source edits, write a source-level design section that names the exact files/functions and workspace sizes.
2. The first source patch must be default-off and must not change `ffn_gate_exps` accepted SOTA behavior.
3. The first run is a short arithmetic/top1 verifier. Passing means `same_top1=145/145`, no OOM, and 16GB cgroup compliance.
4. Only after top1 passes may a strict cold France benchmark run.
5. If the Q8_0 CUDA path is too slow to plausibly keep overhead under the `1.64s` budget, reject it after diagnostic artifacts and do not promote.

Source-level design before implementation:

- Env gate: `GGML_MOE_STREAM_ONE_MXFP4_Q80=1`. With this env unset, `ggml_cuda_moe_stream_one()` and the CPU call site must preserve current behavior exactly.
- CPU call site: `ggml/src/ggml-cpu/ggml-cpu.c`, inside `ggml_compute_forward_mul_mat_id()` one-stream branch.
  - Current call passes `(const float *) src1->data` and `NULL, 0` for the quantized activation parameters.
  - New diagnostic path should pass `wdata_stream` and `row_size_stream` for Q8_0 activation rows when `src0->type==GGML_TYPE_MXFP4`, `src1->type==GGML_TYPE_F32`, and the env gate is enabled.
  - Matrix row mapping must still select the exact `(i11, i12)` activation row corresponding to each routed row.
- CUDA entry point: `ggml/src/ggml-cuda/moe_stream.cu`, `ggml_cuda_moe_stream_one()`.
  - Add a Q8_0 activation branch only for MXFP4 under the env gate.
  - Copy only the selected Q8_0 activation rows to a per-slot `d_src1_q80` workspace. Q8_0 row bytes should be `ggml_row_size(GGML_TYPE_Q8_0, ne00)` from the CPU side; do not infer it from Q8_1 padding.
  - Reuse existing src0 VRAM cache / expert pack behavior; do not change gate pack/prefill/cache semantics.
- CUDA kernel/helper: add a specialized MXFP4 x Q8_0 rows helper rather than modifying the generic Q8_1 MMVQ path first.
  - It may live in `ggml/src/ggml-cuda/moe_stream.cu` for the diagnostic patch or in `ggml/src/ggml-cuda/mmvq.cu` if reusing the existing MMVQ launch structure is cleaner.
  - It must produce F32 output layout identical to the current one-stream branch so scatter code remains unchanged.
- Workspace budget:
  - Persistent extra device workspace should be per slot and proportional to `cne1 * row_size_q80`, plus `d_dst`; it must not allocate another expert-sized cache.
  - Host RAM must remain inside the 16GB cgroup, including page cache. No new persistent host pack/cache is allowed for this verifier.
- Verification sequence:
  - Build only after `git diff --check`.
  - Run fixed-text `llama-results --sequential-logits --top1-report` with the Q8_0 env enabled and a narrow up/down filter.
  - Record source diff hash, binary/lib hashes, top1 result, cgroup stats, and cache counters before deciding whether any benchmark is allowed.

### 2026-07-04 Q8_0 Scalar CUDA Probe Result

Artifacts:

- Top1 verifier: `.Agent/runs/20260704-vendor-ds4-coldstart/q80-up-top1-probe-result.json`
- Top1 artifact sha256: `2c676279c4135254aff45616ea586285e154fdf63ddcf2cc16bf3b7abc56528d`
- Compare probe: `.Agent/runs/20260704-vendor-ds4-coldstart/q80-up-compare-probe-result.json`
- Compare artifact sha256: `2ec6449db733b6c51e7ad30bdaa009c66b274d523ae6b107af7733aa53fd7b18`

Temporary source patch:

- Added a default-off `GGML_MOE_STREAM_ONE_MXFP4_Q80=1` path and `_q80` CUDA entry.
- Gate tensors were kept on the original one-stream path; Q8_0 was restricted to `ffn_up_exps` / `ffn_down_exps`.
- Added comma-list name filter support again for the diagnostic.
- Implemented a simple scalar MXFP4 x Q8_0 CUDA rows kernel in `moe_stream.cu`.
- This source patch was reverted after rejection; `build-ds4-moe-stream` was rebuilt on clean source head `767533259`.

Top1 verifier result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T064953Z-20260704_q80_up_top1_probe/split-up-only-q80-top1`
- Case: `ffn_gate_exps,ffn_up_exps` with `GGML_MOE_STREAM_ONE_MXFP4_Q80=1`
- `same_top1=136/145`
- `first_mismatch_pos=4`
- `memory_peak_bytes=16000000000`
- `oom_seen=false`
- Verdict: rejected before any token-rate benchmark.

Compare probe result:

- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T065419Z-20260704_q80_up_compare_probe/france-q80-up-compare`
- Command: short `llama-cli -n 4` with `GGML_MOE_STREAM_COMPARE_CPU_OUT`.
- Systemd result: timeout at about `7min`; this is already far beyond the `1.64s` total overhead budget needed for a 10 tok/s class implementation.
- Compare rows before timeout: `96`.
- Up per-op error remained small: `max_abs_max=4.76837158e-06`, `mean_abs_mean=4.219115604166667e-08`.
- Gate per-op error remained small: `max_abs_max=9.53674316e-07`.

Decision:

- Reject the scalar Q8_0 CUDA implementation.
- Do not run down-only or full strict cold token-rate benchmark for this patch.
- The failure mode is not a gross per-op arithmetic bug; it is accumulated small numerical drift plus an unusably slow scalar kernel.
- Current accepted SOTA remains `4.4 tok/s`.

Updated next direction:

1. Do not retry naive/scalar GPU up/down exactness paths.
2. A future GPU up/down attempt must either:
   - be a genuinely optimized MXFP4 x Q8_0 kernel with a hard performance model showing overhead plausibly below the `1.64s` budget, and pass fixed-text top1 before benchmark; or
   - use a verification/correction mechanism that guarantees token-level output correctness while still saving enough wall time.
3. Since source movement and existing GPU up/down classes are closed, the next design step should re-open bottleneck analysis at the algorithm level: full or near-full decode fallback removal, not top-N hotsets or page-cache tricks.
4. Any new source probe must start from clean head `767533259` or newer pushed head and must keep accepted gate-only SOTA behavior unchanged by default.

### 2026-07-04 Ngram-Map-K4V No-Source Probe Design

Screening artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/ngram-map-k4v-screening.json`
- Artifact sha256: `beb4bd63a700e8b53c8fdf5d124a53376712a0d98c6739a39d71b571bd06354a`

Why this candidate is still worth one probe:

- `algorithmic-support-inspection.json` closed `ngram-simple`, `ngram-mod`, lookahead, and draft-model speculation without a compatible draft, but no artifact records a strict cold `ngram-map-k4v` run.
- `ngram-map-k4v` is not the same as `ngram-simple`: it tracks up to four m-gram values per key and updates `n_accepted` through `common_ngram_map_accept()`, so it may avoid some repeated bad drafts.
- It is a no-source probe and still uses target verification, so it cannot become SOTA unless output correctness and all system gates pass.

Probe config:

- Base accepted SOTA env remains unchanged: vendor DeepSeek, `cpu_moe=40`, gate-only O_DIRECT one-stream, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, current accepted top-k/profile settings, strict cold `drop_caches`, and 16GB cgroup including page cache.
- CLI delta:
  - `--spec-type ngram-map-k4v`
  - `--spec-ngram-map-k4v-size-n 4`
  - `--spec-ngram-map-k4v-size-m 8`
  - `--spec-ngram-map-k4v-min-hits 1`
  - `--temp 0 --top-k 1 --top-p 1 --seed 1`
- Reason for `n=4,m=8`: less collision-prone than the rejected `ngram-simple n=3,m=8` while still short enough to produce matches in the France answer; `m=8` limits replay/verification burst cost.

Acceptance / rejection:

- Accept only if `eval_tok_s > 4.4`, France output is complete and semantically correct, `TTFT<=33617.688744 ms`, RAM stays within the 16GB cgroup including page cache, no OOM/OOM kill, and cache/pack counters are explainable.
- If accepted, stop, record full reproducibility metadata, commit/push immediately to `ssd/vendor/deepseek-token-rate-16gb`, then rerun from pushed source before promotion.
- Reject if the run times out, loops/stalls, output is incomplete/incorrect, token rate is `<=4.4`, TTFT exceeds the accepted SOTA gate, or RAM/cgroup limits fail.
- Since this is no-source, rejection does not require runtime source rollback; keep artifacts and plan only.

### 2026-07-04 Ngram-Map-K4V No-Source Probe Result

Artifact:

- Compact artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/ngram-map-k4v-probe-result.json`
- Compact artifact sha256: `5108009fc03cb3daead5f3d7f7e1dd394eb0098fe0b282a7ea1adc539a178726`
- Raw run root: `/root/lfz/runs/vendor-ds4-16gb/20260704T071633Z-20260704_ngram_map_k4v_probe`
- Case dir: `/root/lfz/runs/vendor-ds4-16gb/20260704T071633Z-20260704_ngram_map_k4v_probe/france-cpu40-vram0gb`
- Source head: `34ff6e331252111987223fdbe47319d784ce3569`

Result:

- Systemd result: timeout at `10min`; `systemd_run_exit=1`.
- No valid `eval_tok_s`, `prompt_tok_s`, TTFT, or perf footer was produced.
- `stdout.txt` grew to `1755629825` bytes and ended with repeated interactive prompts, so the run did not complete normally.
- `one-trace.csv` has `35152` rows including header and spans about `44.888s` of one-stream gate activity before the process later timed out.
- France answer prefix was semantically correct, but correctness gate is still `false` for promotion because the process timed out and did not terminate with a complete valid measured run.
- Cgroup memory files were not captured because the unit was terminated by the runtime limit before the post-run collection block executed. This is acceptable only for a rejected diagnostic, not for SOTA promotion.
- Stderr counters remained mechanically healthy before timeout:
  - one expert pack `hits=4886 misses=0 direct_reads=4886 direct_failures=0 direct_fallbacks=0`
  - prefill `attempted=3000 inserted=3000 bytes=13369344000 elapsed_ms=4925.120`
  - VRAM cache `hits=33265 misses=1886 hit_rate=94.6%`

Decision:

- Reject `ngram-map-k4v` for the accepted cold-start path.
- Current accepted SOTA remains `4.4 tok/s`.
- This closes the no-source speculative set currently available in this repo: `ngram-simple`, `ngram-mod`, target-only lookahead, server partial-serial fallback, and `ngram-map-k4v`.
- Do not retry another no-source ngram/lookahead run unless a new target-verification mechanism or compatible draft/MTP artifact appears and the plan is updated first with a new hard-bound.

### 2026-07-04 Latest Plan Update

Objective:

- Continue optimizing vendor DeepSeek cold-start token rate from the accepted `4.4 tok/s` SOTA.
- Final results must remain in the `vendor` framework. `ik_llama` can only be used as a reference.
- Hard gates remain unchanged: host RAM `<=16GB` including page cache, `MemorySwapMax=0`, France output semantic/coherent/complete, accepted TTFT `<=33617.688744 ms`, and no hidden warm global page-cache dependency.

Immediate source/record state:

- Current pushed source/record head before this plan update: `becd957f458e3316cac6221cdca4b0c0aa97b964`.
- Current accepted SOTA run remains `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`.
- Current accepted SOTA metrics remain `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`.
- New records and any future source commits must be pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using git identity `L-Ark <fliangae@connect.ust.hk>`.

Closed candidate classes:

- Synchronous page-touch/prefetch and source movement: closed by pushed-source repro and async-bound artifacts.
- Broad/top-N CPU fallback pack mmap/direct staging: closed by measured ties/regressions and bandwidth/refault pressure.
- Existing up/down one-stream offload: closed by fixed-text top1 verifier failures.
- Scalar MXFP4 x Q8_0 CUDA diagnostic: closed by top1 failure and unusable runtime.
- No-source speculation: closed by `ngram-simple`, `ngram-mod`, lookahead, server partial fallback, and `ngram-map-k4v`.
- `mul_mat_id` conversion skip: closed by hard upper bound of only about `33.84 ms` ideal saving.

Next execution sequence:

1. Baseline guard before any new source change.
   - Rerun the accepted SOTA command under strict cold `drop_caches` and 16GB cgroup if the runtime source, binary, model, profile, or pack state has changed since the last accepted run.
   - Record token rates, TTFT, cgroup `memory.peak/current/stat/events`, exact France output, pack counters, VRAM cache counters, binary hashes, source head, model/profile/pack hashes, and run path.
   - If the guard cannot reproduce the `4.4 tok/s` class while passing all gates, fix reproducibility before optimization.

2. Fresh bottleneck table.
   - Use the accepted SOTA path to produce a compact table of per-token wall time split into: gate one-stream source/load/kernel/sync, CPU up fallback, CPU down fallback, page/refault/source stall, scheduler/tail gap, and non-MoE overhead.
   - For each row, record measured removable time and a strict upper-bound token rate after ideal removal.
   - Do not implement a candidate whose hard upper bound is close to `4.4 tok/s`; it must have enough theoretical headroom to justify source risk.

3. Candidate scoring before implementation.
   - For every candidate, write in this plan first: theory, exact files/functions to touch, memory/VRAM footprint, expected removable time, upper-bound token rate, correctness risk, TTFT risk, rollback path, and acceptance/rejection criteria.
   - Prioritize only candidates that reduce full or near-full CPU up/down fallback while preserving logits/top1, or algorithm-level decode work that still verifies target output.
   - Avoid top-N hotsets, synchronous prefetch, global warm-cache tricks, and naive/scalar GPU kernels unless new measurements overturn the existing hard-bound.

4. Correctness-first implementation gate.
   - Any source-level up/down or decode-path candidate must first pass the fixed-text sequential top1 verifier or an equivalent token-level correctness gate before a token-rate benchmark.
   - A France benchmark is allowed only after top1/correctness passes.
   - If output is incomplete, incoherent, or top1 diverges without a proven correction mechanism, revert runtime source and record the rejection.

5. SOTA promotion and push protocol.
   - When a compliant result beats `4.4 tok/s`, stop exploration immediately.
   - Record exact run path, env/CLI, source head, diff summary, build command, binary hashes, model/profile/pack hashes, full France output, token rates, TTFT, elapsed time, cgroup memory including page cache, pack/cache counters, and comparison with the previous SOTA.
   - Commit source, plan, profiles, and artifacts immediately, then push to `ssd/vendor/deepseek-token-rate-16gb`.
   - Clean rebuild from the pushed source and rerun strict cold. Promote only if the pushed-source run still beats `4.4 tok/s` and passes every gate.
   - If the pushed-source rerun fails, mark the candidate rejected, revert runtime source to the accepted SOTA path, keep the rejected artifact/docs, commit/push the rejection, and continue from the last accepted SOTA.

Fresh bottleneck result:

- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/fresh-sota44-bottleneck-hard-bound.json`
- Artifact sha256: `968449a8f672d7c044ba43b23e9ce922cabc1ae58a317d73c406e76e58725ed6`
- Source head: `52a20d0828a2336db3a49b8f6fe5b263f9fb943c`
- Accepted SOTA basis: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Accepted SOTA decode window estimate: `31047.446710 ms`
- Same-output decoded-token estimate: `136.608766`
- Target 10 tok/s decode window: `13660.876552 ms`
- Required decode saving: `17386.570158 ms`
- Ideal full decode up/down fallback removal: `19029.457 ms`
- Remaining overhead budget for a 10 tok/s-class full-fallback-removal implementation: `1642.886842 ms`

Hard-bound conclusion:

| Candidate class | Removable decode ms | Ideal tok/s | Decision |
| --- | ---: | ---: | --- |
| all decode CPU up/down fallback removed | `19029.457` | `11.367` | only remaining non-speculative 10 tok/s ceiling |
| cold-vs-nodrop fallback delta removed | `12791.007` | `7.483` | not promotable; depends on warm global page cache |
| decode CPU up fallback removed | `9697.125` | `6.398` | insufficient alone |
| decode CPU down fallback removed | `9332.332` | `6.291` | insufficient alone |
| gate one-stream total removed | `8050.224` | `5.940` | insufficient alone and mixed prompt/decode trace time |
| gate one-stream src0/source removed | `6707.771` | `5.613` | insufficient alone |
| chunk scheduling/tail gap removed | `1672.318` | `4.650` | closed by bound |
| `mul_mat_id` src1 conversion removed | `33.840` | `4.405` | closed by bound |

Plan decision:

- Do not run another full strict-cold model benchmark until there is a concrete exact design that targets near-full decode CPU up/down fallback removal.
- Do not resume no-source speculation, top-N hotsets, synchronous prefetch/source movement, scalar GPU kernels, or scheduler-only experiments.
- The next candidate family is `exact full decode CPU up/down fallback removal path`, but it is only allowed to proceed after source-path inspection and a hard design showing overhead below about `1.64s`.

Touch-profile source/page split result:

- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/cpu-fallback-touch-split-diagnostic.json`
- Artifact sha256: `9be90deef85afb244cfbd629f04d72fc269f3c84556c7ff4928556f79a153d37`
- Run root: `/root/lfz/runs/vendor-ds4-16gb/20260704T075537Z-20260704_touch_profile_split_sota44`
- Case dir: `/root/lfz/runs/vendor-ds4-16gb/20260704T075537Z-20260704_touch_profile_split_sota44/france-touch-profile-cpu40-vram0gb`
- Config: accepted SOTA env plus `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback_profile.csv`, and `GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE=1`.
- Result: diagnostic rejected for promotion, but useful for bottleneck split.
  - `eval_tok_s=2.6`
  - `prompt_tok_s=1.3`
  - `TTFT=36167.863458 ms`, above accepted TTFT gate
  - `memory_peak_bytes=16000000000`
  - `memory_file_bytes=15105347584`
  - `ram_ok=true`
  - `correctness_ok=true`, France output semantic/coherent/complete

Split conclusion:

| Metric | No-touch profile | Touch-profile diagnostic | Interpretation |
| --- | ---: | ---: | --- |
| decode fallback wall | `19029.457 ms` | `3173.291 ms` after touch | most cold fallback wall is source/page-fault exposed |
| decode source/page touch | n/a | `41626.545 ms` | synchronous touch is far too expensive |
| total fallback wall | `26438.059 ms` | `3445.735 ms` after touch | hot dot/compute part alone is small enough |
| total source/page touch | n/a | `51754.842 ms` | serial pre-touch destroys token rate |
| major faults | accepted SOTA `271465` | `21569` | touch moves faults out of the fallback dot loop |

Source-path inspection:

- DeepSeek4 graph builds cold MoE with `build_lora_mm_id()` in `src/models/deepseek4.cpp:664-675` and default gate/up/down with `build_lora_mm_id()` in `src/models/deepseek4.cpp:709-726`.
- Generic down/separate up/gate CPU fallback is `ggml_compute_forward_mul_mat_id()` in `ggml/src/ggml-cpu/ggml-cpu.c:2749-3189`.
- Fused up/gate CPU fallback is `ggml_compute_forward_moe_up_gate()` in `ggml/src/ggml-cpu/ggml-cpu.c:3275-3544`.
- MXFP4 CPU dot uses `ggml_vec_dot_mxfp4_q8_0` with Q8_0 activations via `ggml/src/ggml-cpu/ggml-cpu.c:1340-1344`.
- Existing useful hooks:
  - chunk wall trace: `ggml/src/ggml-cpu/ggml-cpu.c:834-942`
  - fallback touch profile: `ggml/src/ggml-cpu/ggml-cpu.c:944-986`
  - touch before fallback loop: `ggml/src/ggml-cpu/ggml-cpu.c:3041-3063`
  - remaining fallback loop timer: `ggml/src/ggml-cpu/ggml-cpu.c:3065-3150`

Plan decision after touch split:

- Synchronous touch/page prewarm remains rejected. It proves the bottleneck but is not an optimization.
- The only plausible next implementation direction is default-off bounded parallel or async source-page preparation for up/down fallback:
  - it may only change timing of page faults / source preparation, not selected experts, math, logits, routing, or output;
  - it must use a bounded queue or bounded per-layer window so host RAM including page cache stays inside the 16GB cgroup;
  - it must preserve accepted gate cache behavior and avoid adding more persistent host or VRAM caches;
  - it must target near-full decode up/down fallback exposure, because partial/scheduler/source-only classes do not have a 10 tok/s ceiling.

Source-level design for the next source patch:

- First implementation should be `GGML_MOE_CPU_FALLBACK_PARALLEL_TOUCH=1`, not a long-lived async thread pool yet.
  - Reason: the touch split shows single-thread touch is the immediate bottleneck (`41626.545 ms` decode touch for `148.916 GiB` expert-call bytes). Before adding a persistent queue/thread lifecycle, test whether the existing ggml worker threads can parallelize page faults enough to reduce fallback exposure.
  - Expected upper bound: if parallel touch compresses decode touch below about `1.64s` and leaves hot fallback around `3.17s`, this source/page class can materially exceed the current `4.4 tok/s` SOTA. If touch remains above several seconds, this family cannot reach 10 tok/s without a deeper predictive/overlap mechanism.
- Exact files/functions to edit:
  - `ggml/src/ggml-cpu/ggml-cpu.c`
  - add `ggml_moe_cpu_fallback_parallel_touch_enabled()` near `ggml_moe_cpu_fallback_touch_profile_enabled()`;
  - add `ggml_moe_cpu_fallback_parallel_touch_decode_only()` defaulting to decode-only unless `GGML_MOE_CPU_FALLBACK_PARALLEL_TOUCH_DECODE_ONLY=0`;
  - edit `ggml_compute_forward_mul_mat_id()` around `ggml_kimi_cpu_fallback_pack_mmap_prepare()` and the current serial touch block (`ggml/src/ggml-cpu/ggml-cpu.c:3041-3063`);
  - leave `ggml_compute_forward_mul_mat_id_one_chunk()` and `ggml_vec_dot_mxfp4_q8_0` math untouched.
- Execution model:
  - `ith==0` still prepares `fallback_pack_mmap_ptrs` and zeroes `fallback_touch_us`.
  - all threads hit a barrier after prepare;
  - if `GGML_MOE_CPU_FALLBACK_PARALLEL_TOUCH=1` and the op is decode phase (`ids->ne[1] <= 1`) by default, each worker touches a disjoint subset of active experts with `for (cur_a = ith; cur_a < n_as; cur_a += nth)`;
  - each touched expert uses the same pointer source the fallback loop will use: `fallback_pack_mmap_ptrs[cur_a]` if present, otherwise `src0->data + cur_a * nb02`;
  - touch size remains `(size_t) ne01 * nb01`, exactly matching current expert source size;
  - all threads hit a second barrier before the existing fallback compute loop.
- RAM/page-cache bound:
  - no new persistent host allocation except the existing `fallback_touch_us` array;
  - no new VRAM allocation;
  - page cache content is not expanded beyond pages the fallback loop would fault anyway;
  - strict 16GB cgroup including page cache and `MemorySwapMax=0` remain mandatory for every run.
- Correctness proof:
  - the patch only reads expert weight pages before `vec_dot`; it does not change `matrix_row_counts`, `matrix_rows`, selected experts, activations, weights, math kernels, output buffers, or top-k pruning;
  - it cannot change logits except through a memory/race bug, which should be ruled out by France output and fixed-text top1 if any suspicious output appears.
- Metrics:
  - run the first diagnostic with `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT` enabled so `touch_us` is recorded per expert;
  - record token rates, TTFT, RAM/page cache, major faults, `touch_us`, fallback after touch, pack/VRAM counters, output answer, source head, binary hash, and artifact hashes.
- Rejection rules:
  - reject if RAM exceeds 16GB including page cache, OOM appears, output correctness fails, or TTFT exceeds the gate for a promoted result;
  - reject this family if parallel touch still leaves decode touch + hot fallback too high to plausibly beat `4.4 tok/s` or if it regresses token rate like serial touch.
- Promotion rules:
  - only promote if a strict cold run without diagnostic overhead beats `4.4 tok/s`, has `TTFT<=33617.688744 ms`, passes correctness, stays within 16GB including page cache, and is reproduced after push.

Parallel decode-only touch result:

- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/parallel-touch-decode-profile-result.json`
- Artifact sha256: `a19ecf1747c4ff672958450f41439293f269f4fb6b55c2fc2a5209a93ee5f2ea`
- Run root: `/root/lfz/runs/vendor-ds4-16gb/20260704T123650Z-20260704_parallel_touch_decode_profile`
- Case dir: `/root/lfz/runs/vendor-ds4-16gb/20260704T123650Z-20260704_parallel_touch_decode_profile/france-parallel-touch-cpu40-vram0gb`
- Temporary source patch: default-off `GGML_MOE_CPU_FALLBACK_PARALLEL_TOUCH=1` in `ggml/src/ggml-cpu/ggml-cpu.c`; decode-only by default; no math/top-k/routing changes.
- Build: succeeded. After rejection, runtime source was reverted and `build-ds4-moe-stream` rebuilt on clean source.
- Diagnostic config: accepted SOTA env plus `GGML_MOE_CPU_FALLBACK_PARALLEL_TOUCH=1`, `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback_profile.csv`.

Result:

- `eval_tok_s=3.0`
- `prompt_tok_s=1.8`
- `TTFT=32157.262724 ms`, inside accepted TTFT gate but not enough for promotion because token rate regressed
- `elapsed_seconds=77.32`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15101444096`
- `ram_ok=true`
- `correctness_ok=true`, France answer semantic/coherent/complete
- `pgmajfault=158740`, lower than accepted SOTA but wall time still regressed

Split comparison:

| Metric | Baseline/profile | Parallel touch diagnostic |
| --- | ---: | ---: |
| accepted SOTA eval tok/s | `4.4` | `3.0` |
| accepted SOTA elapsed | `63.94s` | `77.32s` |
| decode fallback wall | `19029.457 ms` | `3690.044 ms` after parallel touch |
| decode touch worker-time sum | n/a | `48312.036 ms` |
| wall delta vs accepted | n/a | `+13.38s` |

Decision:

- Reject and close same-op pre-touch, including serial and parallel variants.
- The mechanism proved that page-fault placement matters, but moving the faults into the same op with a barrier creates too much system pressure and wall-time regression.
- Do not run a no-profile SOTA attempt for this patch; the profiled no-touch baseline was already near SOTA class, while this profiled parallel-touch diagnostic is far below SOTA.
- Runtime source was reverted; keep only artifact/docs.

Next concrete work item:

- Rebuild the candidate list again from the current hard-bound table after closing same-op pre-touch.
- If source/page work is revisited, it must be predictive overlap before the consuming `mul_mat_id` op/layer, not same-op pre-touch.
- Otherwise move to a different exact compute path that can remove near-full decode up/down fallback while passing fixed-text top1 before any token-rate benchmark.

## 2026-07-04T12:54:58Z Next Plan: MXFP4 x Q8_0 CUDA Exact-Offload Probe

Latest pushed head before this plan update: `24a7407a487effcba15c4c40458220e244191e78` (`vendor-ds4: reject parallel fallback touch`) on `ssd/vendor/deepseek-token-rate-16gb`.

Current accepted SOTA remains:

- `eval_tok_s=4.4`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Config: vendor DeepSeek strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate-only one-stream cache, O_DIRECT gate expert pack, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`
- Metrics: `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Promotion TTFT gate remains `<=33617.688744 ms`.

Bottleneck after closing same-op touch:

- Same-op serial and parallel page touch are closed. They move page-fault time but regress wall time.
- Source movement / direct staging variants remain closed by `.Agent/runs/20260704-vendor-ds4-coldstart/source-movement-async-bound.json`.
- Existing Q8_1 CUDA up/down stream remains closed by token-level top1 mismatch.
- The current hard-bound table shows the only still-relevant high-ceiling class is exact compute/offload of decode up/down fallback:
  - decode up fallback removable: `9697.125 ms`, no-overhead ceiling about `6.40 tok/s`
  - decode down fallback removable: `9332.332 ms`, no-overhead ceiling about `6.29 tok/s`
  - all decode up/down fallback removable: `19029.457 ms`, no-overhead ceiling about `11.37 tok/s`
  - required for `10 tok/s`: save about `17.3866s` from the current decode window, leaving only about `1.6429s` total overhead if all decode up/down fallback is removed.

Why this probe can improve correctness versus the rejected CUDA up/down path:

- CPU MXFP4 fallback uses `ggml_vec_dot_mxfp4_q8_0` and quantizes F32 activations into `params->wdata` as Q8_0.
- Existing CUDA one-stream MMVQ uses `quantize_row_q8_1_cuda` and `vec_dot_mxfp4_q8_1`.
- Prior up/down GPU output drift was numerically small, but fixed-text top1 still failed. The next candidate must therefore use the same Q8_0 activation semantics as CPU fallback before any performance run.

Theory and upper bound:

- If the new path removes all decode up/down CPU fallback with no extra overhead, the measured ceiling is about `11.37 tok/s`.
- To meet `10 tok/s`, the combined Q8_0 row copy, CUDA kernel, D2H, scatter, sync, and any cache/source overhead must stay under about `1.64s` across the decode window.
- A naive CUDA kernel may pass correctness but fail performance; that is still useful because it validates or rejects the exact-Q8_0 math path before optimization.

Implementation plan:

1. Add a default-off env gate: `GGML_MOE_STREAM_ONE_MXFP4_Q80=1`.
2. Add a separate CUDA entry point instead of changing the current accepted path:
   - `ggml_cuda_moe_stream_one_q80(...)`
   - only accepts `GGML_TYPE_MXFP4`, `src1` converted to CPU Q8_0 in `params->wdata`, `cne1 <= MMVQ_MAX_BATCH_SIZE`, and existing name filters.
3. In `ggml_compute_forward_mul_mat_id()`, pass `params->wdata`, Q8_0 row size, Q8_0 row strides, and row mappings to the Q8_0 entry point only when the env is set. With the env unset, current SOTA behavior must be bit-for-bit unchanged at source-routing level.
4. In `ggml/src/ggml-cuda/moe_stream.cu`, stage only the selected Q8_0 rows to a device Q8_0 buffer and run an MXFP4 x Q8_0 row kernel.
5. The first kernel may be simple and verifier-oriented, but it must use the CPU formula:
   - per MXFP4 block: `scale = GGML_E8M0_TO_FP32_HALF(x.e) * fp16_to_fp32(q8_0.d)`
   - sum low and high nibbles against `q8_0.qs[0..31]`
   - output `sum(scale * integer_dot)` as F32.
6. Reuse the current VRAM expert cache and one-pack source path. Do not add persistent host caches. Do not shrink the accepted gate cache unless a later bounded experiment proves a reason.

Validation sequence:

1. Build `llama-cli` and `llama-results`.
2. Run `git diff --check`.
3. Run the existing fixed-text top1 verifier under the 16GB cgroup with `GGML_MOE_STREAM_ONE_MXFP4_Q80=1` and an up/down name filter candidate.
4. If any top1 mismatch appears, reject immediately; do not run token-rate benchmark.
5. If top1 passes, run strict cold France benchmark with all normal gates:
   - 16GB cgroup including page cache
   - `MemorySwapMax=0`
   - `drop_caches` before case
   - `TTFT<=33617.688744 ms` for promotion
   - semantic, coherent, complete France output
   - all token rate, TTFT, RAM/page-cache, answer, command, env, source head, binary hash, and artifact hash recorded.
6. If a strict cold result beats `4.4 tok/s`, immediately commit and push source plus reproducibility records to `ssd/vendor/deepseek-token-rate-16gb`, then reproduce from the pushed source before accepting it as new SOTA.
7. If correctness fails, performance regresses, RAM exceeds 16GB, or TTFT violates promotion gate, revert runtime source, rebuild accepted path, and commit/push only the rejected artifact/docs.

Immediate next concrete work item:

- Implement the default-off `GGML_MOE_STREAM_ONE_MXFP4_Q80=1` verifier probe and run fixed-text top1 before any token-rate measurement.

## 2026-07-04T13:13:30Z Correction: Do Not Repeat Scalar Q8_0 CUDA Probe

This section supersedes the immediate work item in `2026-07-04T12:54:58Z Next Plan`.

Audit result:

- While preparing the Q8_0 probe, the existing historical section `2026-07-04 Q8_0 Scalar CUDA Probe Result` was found in this same plan.
- That historical probe already implemented a default-off scalar MXFP4 x Q8_0 CUDA path with comma-list filter support.
- It was rejected before any token-rate benchmark:
  - `same_top1=136/145`
  - `first_mismatch_pos=4`
  - `memory_peak_bytes=16000000000`
  - `oom_seen=false`
  - compare probe timed out at about `7min`, far beyond the `1.64s` total overhead budget needed for a 10 tok/s class implementation
  - per-op up error was small, so the failure was accumulated drift plus unusable scalar-kernel speed, not a gross arithmetic bug
- Repeating the same scalar Q8_0 implementation is therefore closed and should not be run again.

What happened in this turn:

- A duplicate default-off Q8_0 scalar source prototype was briefly implemented locally after the new plan update.
- Before running verifier, the historical rejection above was found.
- The source prototype was reverted immediately.
- `build-ds4-moe-stream` was rebuilt on clean source.
- No Q8_0 verifier or token-rate run was performed for the duplicate prototype.

Current clean state after revert:

- Source head: `46a23c0cd2d70ee8a030df711abbc26cd741865f`
- `git status --short`: clean
- `llama-cli` sha256: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
- `llama-results` sha256: `d981fb8f26fb16c338e23acfe490f070e90b7b70ecad2a828bc503b65fca07a9`
- `libggml-cpu.so.0.10.0` sha256: `b7d0a3a7a65adac0e8ff2a83ce7bcbbeef8a5ce38ddd173cf8bd6b75abe9ed39`
- `libggml-cuda.so.0.10.0` sha256: `fada47dc0b5565224f7914ab3b077740474ea9b54cd850edd7224041d9572691`

Updated next concrete work item:

- Do not implement or run another scalar MXFP4 x Q8_0 CUDA one-stream probe.
- Rebuild the candidate list around non-duplicate paths:
  - an optimized MXFP4 x Q8_0 kernel only if it first has a hard performance model showing total decode overhead can plausibly fit below `1.64s`, then must pass fixed-text top1 before any benchmark;
  - or a correctness-preserving algorithmic/offload path that removes near-full decode up/down fallback and also reduces gate/source stall enough to make `10 tok/s` plausible.
- The next action is a design/bound step, not a source patch: produce a hard-bound candidate table that explicitly excludes scalar Q8_0, same-op touch, direct staging, top-N residency/hotsets, and existing Q8_1 up/down stream.

## 2026-07-04T13:23:00Z Non-Duplicate Hard-Bound Result

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/nonduplicate-hard-bound-after-q80-duplicate-audit.json`
- sha256: `c6146eaa66eb4a55384fe669e6296b336606502ff58b22ad7b4fd363b73737d6`

Purpose:

- Rebuild the candidate table after excluding closed paths:
  - same-op serial/parallel touch
  - synchronous direct/mmap source staging
  - top-N up/down residency/hotsets as a primary path
  - existing Q8_1 up/down CUDA stream
  - scalar MXFP4 x Q8_0 CUDA one-stream

Model inputs:

- Accepted SOTA decode window estimate: `31.04744671s`
- Decoded tokens estimate: `136.608765524`
- Target decode window for `10 tok/s`: `13.6608765524s`
- Required decode saving: about `17.38657s`
- Current gate cache profile: `3000` prefilled entries, about `12.45 GiB`; accepted cache budget is about `13.25 GiB`
- Gate trace: `35151` gate rows, `33265` hits, `1886` misses
- Estimated extra gate src0 cost per lost gate hit: about `1.024 ms`

Best optimistic cache-allocation combination under the current `13.25 GiB` gate-cache-class budget:

- gate cache reduced to top `2048` entries, about `8.5 GiB`
- up/down exact residency top `1024`, about `4.1905 GiB`
- total still fits the current cache budget class
- estimated lost gate hits versus top3000: `2320`
- estimated gate penalty: `2.376s`
- ideal up/down fallback saving: `9.080s`
- estimated decode window: `24.343s`
- estimated token rate: `5.612 tok/s`
- result: far below `10 tok/s`, even before exact-kernel overhead or correctness risk

Streaming-copy bound:

- Decode up/down expert-call source bytes from the touch split are about `148.916 GiB`.
- If all decode up/down CPU fallback were removed, the total extra overhead budget for a `10 tok/s` result is only about `1.6429s`.
- A streaming-copy design would therefore need about `90.6 GiB/s` effective H2D/source bandwidth before kernel/D2H/scatter/sync overhead.
- Decision: streaming all up/down weights on demand is not a viable 10 tok/s path.

Decision:

- No non-duplicate cache-allocation or streaming-copy candidate currently has a 10 tok/s hard bound.
- Do not implement another top-N residency, hotset, direct-staging, same-op prefetch/touch, Q8_1 up/down, or scalar Q8_0 source patch.
- The next source-level proposal must first show a new hard bound above 10 tok/s and a correctness gate. Plausible classes are limited to a genuinely new algorithmic/offload mechanism, such as a compact resident representation with exact/corrected logits, or a verified speculation/draft path that does not exist in the current artifact set.

Updated next concrete work item:

- Inspect whether any existing vendor/DS4 code path can provide a compact resident representation or verified speculation path without changing model semantics. If no such path exists in source/artifacts, record that as the next blocker candidate and avoid further low-ceiling source probes.

## 2026-07-04T13:28:25Z Latest Plan: 10 tok/s Gate After Non-Duplicate Audit

Latest pushed source state:

- Local branch: `feat/ds4-moe-stream-on-vendor`
- Required push target for this project: `ssd/vendor/deepseek-token-rate-16gb`
- Latest pushed head before this plan update: `16f0a7105ad04b7bf0499ba60159c8781f0fad07`
- Runtime source status at that head: clean accepted-path source; rejected Q8/touch prototypes are not present.

Current accepted SOTA remains unchanged:

- `eval_tok_s=4.4`
- `prompt_tok_s=1.8`
- `TTFT=32892.55329 ms`
- `elapsed_seconds=63.94`
- `memory_peak_bytes=16000000000`
- `memory_file_bytes=15102607360`
- `ram_ok=true`, including page cache under the 16GB cgroup with `MemorySwapMax=0`
- `correctness_ok=true`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Accepted config: vendor DeepSeek strict cold `drop_caches`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate-only one-stream cache with `ffn_gate_exps`, O_DIRECT gate expert pack, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Promotion TTFT gate remains `<=33617.688744 ms`.

Hard-bound status after the latest audits:

- Target for `10 tok/s` requires decode window about `13.6608765524s`.
- Accepted SOTA decode window estimate is `31.04744671s`, so a candidate must save about `17.38657s`.
- If a candidate removes all remaining decode up/down CPU fallback, the entire allowed extra overhead budget is only about `1.6429s`.
- Closed source families:
  - same-op serial/parallel page touch;
  - synchronous direct/mmap source staging;
  - top-N up/down residency or hotsets as the primary path;
  - existing Q8_1 CUDA up/down stream;
  - scalar MXFP4 x Q8_0 CUDA one-stream;
  - skip/zero-row pruning beyond already-active exact pruning;
  - `mul_mat_id` src1 Q8 conversion removal;
  - ngram/lookahead/speculative variants that already failed correctness, progress, or token-rate gates.
- Best optimistic non-duplicate cache-allocation bound under the current cache-budget class is only about `5.612 tok/s`.
- Streaming all up/down weights on demand would need about `90.6 GiB/s` effective source/H2D bandwidth before kernel, D2H, scatter, and sync overhead, so it is not a viable 10 tok/s path.

Algorithmic/speculation audit status:

- Existing source has NextN/MTP metadata/tensor loading, but DeepSeek4 does not currently execute a DS4 MTP/NextN verification path. Source comments also mark NextN/MTP tensors as preserved/ignored or TODO for future MTP implementation.
- `llama-speculative` and `llama-speculative-simple` require a compatible draft model. No local tokenizer-compatible small DeepSeek draft GGUF has been found; local GLM GGUF shards are not compatible DeepSeek draft models.
- Historical ngram/lookahead attempts are rejected:
  - all-output ngram-simple reached repeated verification/progress loops and incomplete output;
  - partial serial fallback restored progress and correctness but only reached about `3.7 tok/s`;
  - no-source `ngram-map-k4v` timed out at 10 minutes and produced repeated prompts instead of a valid footer;
  - target-only lookahead previously failed under DeepSeek4 coupled-sequence behavior and damaged cache hit behavior.

Updated optimization plan:

1. Compact exact resident representation feasibility gate.
   - Before writing source, define the exact representation, memory footprint, transfer path, kernel work, and correction needed to preserve logits.
   - It must have a hard-bound decode window `<=13.66s` under the same 16GB host RAM/page-cache limit.
   - It must explain how it avoids the current `1.64s` overhead ceiling if it claims to remove full up/down fallback.
   - If it changes numeric behavior, it must pass fixed-text top1 before any token-rate benchmark.

2. Verified speculation or draft-model gate.
   - Do not rerun ngram/lookahead variants without a concrete fix for the recorded progress-loop or coupled-KV-state failure.
   - Do not run draft speculative decoding unless a compatible DeepSeek draft/MTP artifact exists.
   - If a compatible draft/MTP artifact appears, first record model provenance, tokenizer compatibility, RAM/VRAM budget, and exact target-verification semantics; then run correctness before performance.

3. Predictive cross-op source overlap gate.
   - Same-op touch is closed. Any future source/page-fault work must overlap page faults before the consuming `mul_mat_id` op or layer.
   - The design must show which future rows can be predicted, how much wall time can be hidden, and why it will not increase TTFT or cgroup page-cache pressure.
   - This path is only worth implementing if its hard bound can beat the accepted `4.4 tok/s` SOTA with room after synchronization overhead.

4. No low-ceiling patches.
   - Do not implement another source patch unless the design step first shows a hard upper bound above the current SOTA and, for 10 tok/s work, a credible path to the `13.66s` decode-window target.
   - If no current source/artifact path passes this gate, record the blocker explicitly instead of spending time on another low-ceiling probe.

Execution rules for the next implementation:

- Every practical step must start by updating this plan with the bottleneck, theory, hard-bound calculation, expected token-rate range, validation command, and rejection criteria.
- Any candidate that affects logits must pass fixed-text top1/correctness before a token-rate benchmark.
- Any benchmark must record token rate, prompt rate, TTFT, elapsed time, RAM/page-cache counters, correctness answer, command/env, source head, binary hashes, run path, and artifact hash.
- A new accepted SOTA must be immediately committed and pushed to `ssd/vendor/deepseek-token-rate-16gb`, then reproduced from the pushed source before promotion.
- A rejected source candidate must be reverted, the accepted path rebuilt, and only rejected artifacts/docs committed and pushed.
- Git identity for pushes should remain `L-Ark <fliangae@connect.ust.hk>`.

Immediate next concrete action:

- Produce a compact/speculation path audit artifact that ties the source evidence and historical rejected artifacts to this latest plan, then commit and push this plan update plus the artifact. After that, only proceed to source implementation if a new compact, verified-speculation, or predictive-overlap design passes the hard-bound gate above.

## 2026-07-04T13:47:00Z Bounded No-Source Probe: Lightning Indexer

Purpose:

- Check one existing DeepSeek4 vendor code path that has not been recorded as closed in this plan: `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1`.
- This is not a 10 tok/s primary path because it does not remove the current decode up/down CPU fallback bottleneck (`19029.457 ms` decode fallback).
- It is allowed as a one-shot no-source bounded probe because it uses an existing default-off env flag and may reduce attention indexer launch/intermediate cost without changing the accepted source tree.

Theory and hard bound:

- The fused `ggml_lightning_indexer` path replaces the explicit `mul_mat -> relu -> weighted-sum` indexer score pipeline in `src/models/deepseek4.cpp`.
- It only applies when `indexer_head_dim == 128`, `indexer_n_head == 64`, `work_tokens == 1` or collapsed-q prefill, and `indexer_kv_prefix` is `F32`.
- It does not touch MoE routing, gate expert cache, up/down CPU fallback, or the one-stream gate pack.
- Therefore it cannot by itself approach `10 tok/s`; even perfect removal of the recorded gate one-stream total (`8050.224 ms`) would only bound to about `5.94 tok/s`, and lightning indexer is narrower than that.
- Expected outcome range: tie/regress is most likely; a small accepted SOTA is possible only if the fused indexer removes enough per-token graph/kernel overhead without disturbing output.

Validation command:

- Run exactly one strict cold France benchmark with accepted SOTA config plus `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1`:
  - 16GB cgroup including page cache;
  - `MemorySwapMax=0`;
  - `drop_caches` before the case;
  - `cpu_moe=40`;
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`;
  - gate-only one-stream cache and O_DIRECT gate pack;
  - CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`.

Acceptance/rejection:

- Accept only if `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, `memory_peak_bytes <= 16000000000`, page cache is charged inside the cgroup, no OOM/kill appears, and France output is semantic/coherent/complete.
- If accepted, immediately record full reproduction details, commit/push docs/config to `ssd/vendor/deepseek-token-rate-16gb`, then reproduce from pushed state before promoting.
- Reject if it ties/regresses, fails correctness, exceeds RAM/page-cache limit, increases TTFT beyond the gate, or appears inactive/no-effect. Because this probe has no source patch, rejection only needs an artifact and plan record.

Result:

- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/lightning-indexer-probe-rejected-ttft.json`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T134008Z-20260704_lightning_indexer_probe/france-lightning-indexer-cpu40-vram0gb`
- Source head: `b944c7a834f042ed17f3552c9aaa5e12c30b0547`
- Source patch: none; existing env flag only.
- Metrics:
  - `eval_tok_s=4.5`
  - `prompt_tok_s=1.8`
  - `TTFT=33788.731578 ms`
  - `elapsed_seconds=64.51`
  - `memory_peak_bytes=16000000000`
  - `memory_file_bytes=15098261504`
  - `ram_ok=true`
  - `correctness_ok=true`, France output is semantic/coherent/complete
  - gate cache counters unchanged: `hits=33265`, `misses=1886`, `hit_rate=94.6%`
  - prefill: `attempted=3000`, `inserted=3000`, `bytes=13369344000`, `elapsed_ms=5132.426`
- Verdict: reject as accepted SOTA because TTFT is `171.042834 ms` above the promotion gate `33617.688744 ms`, despite the rounded generation rate improving to `4.5 tok/s`.

Gap analysis:

- The speed improvement is not enough to accept because the TTFT gate is strict.
- The TTFT miss appears dominated by cold prefill variance/pressure: this run's prefill took `5132.426 ms`, while the accepted SOTA record was about `4485.962 ms`.
- A small prefill-limit reduction was modeled but not run:
  - limit `2950` saves about `85.5 ms` prefill but is not enough for this observed TTFT gap;
  - limit `2900` saves about `171.1 ms` prefill, but loses about `505` profile-score hits versus top3000, with an estimated gate penalty of about `517.1 ms`;
  - lower limits have worse net estimates.
- Therefore do not sweep prefill limits for this result without a new measured reason; it is likely to trade a tiny TTFT fix for decode regression.

Decision:

- Keep current accepted SOTA at `4.4 tok/s`.
- Keep `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1` as a useful not-accepted diagnostic result, but do not promote it.
- If revisited, the next design must reduce TTFT independently of shrinking the gate prefill or show a repeatable TTFT pass under the same strict cold procedure before promotion.

## 2026-07-04T14:05:00Z External Draft/MTP Audit

Purpose:

- Check whether an external compatible DeepSeek draft/MTP artifact exists that could unlock verified speculation without repeating rejected ngram/lookahead paths.

Findings:

- External search found `antirez/deepseek-v4-gguf` with files named:
  - `DeepSeek-V4-Q2.gguf`, about `56.5 GB`;
  - `DeepSeek-V4-Q2-MTP.gguf`, about `3.81 GB`.
- This is not immediately usable for the current accepted SOTA path:
  - the current accepted model is `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, size `156,148,189,760 bytes` (`145.4 GiB`, `ls -lh` `146G`);
  - the current vendor source loads/preserves NextN/MTP tensors for some architectures but does not execute a DeepSeek4 MTP verification path;
  - previous local GGUF metadata audit for the accepted model found no `mtp`, `nextn`, `nextn_predict_layers`, `draft`, or `eagle` keys/tensors;
  - the external forum thread for the antirez DS4 runtime explicitly describes MTP as currently broken and CUDA support as not fully ported from MLX, so this cannot be treated as a ready verified draft path.

Decision:

- Do not download or benchmark the external MTP GGUF as part of the current vendor SOTA path.
- Treat external MTP as a separate future project requiring:
  - tokenizer/vocab compatibility proof against the accepted target;
  - DS4 MTP runtime implementation in vendor source;
  - exact target verification semantics;
  - 16GB host RAM/page-cache and VRAM budget accounting;
  - correctness first, then token-rate benchmark.
- Current accepted SOTA remains `4.4 tok/s`.

Artifact:

- `.Agent/runs/20260704-vendor-ds4-coldstart/external-draft-mtp-audit.json`

## 2026-07-04T14:16:00Z Lightning Indexer Same-Config Repeat Check

Purpose:

- The first `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1` strict cold run reached `4.5 tok/s` but missed the TTFT promotion gate by only `171.042834 ms`.
- This section allows exactly one same-config repeat to determine whether the TTFT failure was cold prefill variance or a repeatable regression.
- This is not a parameter sweep:
  - do not change prefill limit;
  - do not change gate cache size;
  - do not change CLI batch/context/thread settings;
  - do not change runtime source.

Validation command:

- Same accepted SOTA config as before plus `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1`.
- Strict cold `drop_caches`, 16GB cgroup including page cache, `MemorySwapMax=0`.

Acceptance/rejection:

- If this repeat has `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, `ram_ok=true`, and France correctness passes, record it as a candidate and immediately push records, then run one pushed-state reproduction before promoting as accepted SOTA.
- If it ties/regresses, fails correctness/RAM, or TTFT remains above the gate, close lightning indexer as not accepted for now and keep only rejected records.

Result:

- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/lightning-indexer-repeat-candidate.json`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T135703Z-20260704_lightning_indexer_repeat/france-lightning-indexer-repeat-cpu40-vram0gb`
- Source patch: none; existing `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1` env flag only.
- Metrics:
  - `eval_tok_s=4.5`
  - `prompt_tok_s=1.8`
  - `TTFT=33285.240047 ms`, inside promotion gate
  - `elapsed_seconds=64.19`
  - `memory_peak_bytes=16000000000`
  - `memory_file_bytes=15093157888`
  - `ram_ok=true`
  - `correctness_ok=true`, France output is semantic/coherent/complete
  - gate cache counters unchanged: `hits=33265`, `misses=1886`, `hit_rate=94.6%`
  - prefill: `attempted=3000`, `inserted=3000`, `bytes=13369344000`, `elapsed_ms=4935.206`

Decision:

- This is a candidate only, not yet promoted.
- Immediately commit/push the plan and candidate artifact, then run one strict cold pushed-state reproduction with the same config.
- Promote `LLAMA_DEEPSEEK4_LIGHTNING_INDEXER=1` to accepted SOTA only if the pushed-state reproduction also has `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, strict 16GB RAM/page-cache pass, and France correctness pass.

Pushed-state reproduction result:

- Artifact: `.Agent/runs/20260704-vendor-ds4-coldstart/lightning-indexer-pushed-repro-rejected.json`
- Pushed source: `ca1a80c57cdd6210bcbba36d7f6bed3924d38066`
- Run: `/root/lfz/runs/vendor-ds4-16gb/20260704T140232Z-20260704_lightning_indexer_pushed_repro/france-lightning-indexer-pushed-repro-cpu40-vram0gb`
- Metrics:
  - `eval_tok_s=4.4`
  - `prompt_tok_s=1.9`
  - `TTFT=33618.90603 ms`
  - `memory_peak_bytes=16000000000`
  - `memory_file_bytes=15091015680`
  - `ram_ok=true`
  - `correctness_ok=true`, France output is semantic/coherent/complete
- Verdict: rejected as accepted SOTA. It tied the accepted `4.4 tok/s` and TTFT was `1.217286 ms` above the promotion gate.

Decision:

- Do not promote lightning indexer.
- Current accepted SOTA remains `4.4 tok/s`.
- Close this no-source lightning path for now; only revisit if a separate TTFT reduction or stronger repeated evidence is designed and recorded first.

## 2026-07-04T16:01:29Z Top768 Residual Profile And Pack Plan Update

Purpose:

- Update the active plan after generating the top768 up/down residual profile for the `cpu41 + exact Q8_0 hot residual` candidate.
- This is a planning/artifact update only. Runtime source is unchanged and the accepted SOTA remains `4.4 tok/s`.

Current accepted SOTA guardrail:

- Accepted run: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Accepted metrics: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32892.55329 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102607360`, `ram_ok=true`, `correctness_ok=true`
- Promotion gate remains: `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, 16GB cgroup including file page cache, `MemorySwapMax=0`, coherent France answer, immediate commit/push, then pushed-source reproduction.

New profile artifact:

- Profile path: `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top768_touch_residual.tsv`
- Source fallback profile: `/root/lfz/runs/vendor-ds4-16gb/20260704T075537Z-20260704_touch_profile_split_sota44/france-touch-profile-cpu40-vram0gb/fallback_profile.csv`
- Profile SHA256: `c8226df2f35b3f26a90ed33cdc5e432bdeb813fb48d47a02b7e82bbc8ba4695e`
- Entries: `768`
- Residual decode fallback covered by this profile: `2048.860 ms`
- Payload: `3422552064 bytes` (`3264.000 MiB`)

Pack status:

- The top768 up/down pack has not been generated.
- Intended durable pack path, if disk is made available: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top768-updown-touchresid-20260704.pack`
- Current free space on `/dev/root` at this update: `3221327872 bytes`.
- Required raw payload alone is `3422552064 bytes`; index/alignment overhead makes the real requirement larger.
- Therefore generating the pack now would likely fill the root filesystem and is not allowed.
- Do not use `/dev/shm` for this accepted-SOTA path because it is tmpfs/RAM-backed, volatile, and would confuse the 16GB host/page-cache accounting.
- Do not delete or move old reproducibility packs without explicit approval. Existing large packs are historical artifacts needed for reproducing earlier accepted/rejected runs.

Updated active plan:

1. Commit and push this plan update plus the top768 profile artifact before any runtime experiment.
2. Do not run a strict cold performance benchmark for `cpu41 + top768` until the source path is resolved. A benchmark without a durable pack or direct O_DIRECT GGUF reader would not reproduce.
3. Choose exactly one source path before coding:
   - durable-pack path: free or move enough disk, generate the top768 pack from the profile trace, record pack hash/size/index metadata, then continue to split-pool implementation;
   - direct-reader path: implement a default-off GGUF-offset O_DIRECT reader for the q80 hot pool so the extra 3.26 GiB pack file is not needed.
4. The first runtime source change must remain default-off and preserve the accepted `4.4 tok/s` path when unset.
5. Implement split-pool allocation and counters before compute: keep the accepted gate cache at `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`; add a separate fixed q80 hot pool for the top768 up/down entries. Do not share an LRU that can evict gate entries.
6. Implement exact MXFP4 x Q8_0 CUDA semantics only after the source path is reproducible. Cover both up and down fallback paths; down-only cannot reach the hard-bound.
7. Run op compare and the fixed-text token-level top1 verifier before any long France benchmark. If top1 mismatches or France output becomes incoherent, revert runtime source and keep only rejected docs/artifacts.
8. Any new compliant SOTA must be recorded with full reproduction information and immediately pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then reproduced from pushed source before promotion.

Decision:

- The profile generation step is complete and useful.
- The pack/source step is currently disk-constrained.
- Current accepted SOTA remains `4.4 tok/s`; no new performance result is promoted by this update.

## 2026-07-05 Direct Reader Path Selection

Purpose:

- Continue the `cpu41 + top768 exact Q8_0 hot residual` path without relying on a new 3.26 GiB pack file.
- This section is a required pre-practice plan update before touching runtime source.

Decision:

- Choose the direct-reader path for the next implementation step.
- Do not generate `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-decode-top768-updown-touchresid-20260704.pack` until disk is explicitly freed or old packs are explicitly moved/deleted.
- Do not use `/dev/shm` as a substitute pack location for accepted-SOTA work because it is RAM-backed and would weaken 16GB host/page-cache accounting.

Implementation boundary for this step:

1. Add only default-off source infrastructure. Accepted SOTA behavior must be byte-for-byte disabled unless the new env/config is explicitly set.
2. Preserve the accepted gate one-stream cache exactly: `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate-only filter, current O_DIRECT gate pack path, and current `4.4 tok/s` run remain the rollback point.
3. Build a separate q80 hot residual source path. It must not share the gate cache LRU and must not evict gate entries.
4. First implement metadata/source discovery and counters, not a performance benchmark:
   - map profile entries to GGUF tensor data offsets;
   - verify all `768` profile entries resolve to existing up/down tensors;
   - verify resolved byte sizes match the profile payload (`3422552064 bytes`);
   - expose counters/logging for resolved, unresolved, bytes, and whether direct-reader mode is active.
5. If direct GGUF tensor offsets are not readily available inside `moe_stream.cu`, add a small helper/artifact to derive an offset manifest from existing GGUF metadata and consume that manifest default-off. The manifest must be committed with hash and generation command before it can be used for strict runs.
6. Do not implement or enable MXFP4 x Q8_0 CUDA compute until the source path is reproducible and metadata resolution passes.
7. Do not run a long strict cold France performance benchmark in this step. The allowed validation is build/static validation and, if practical, a short metadata-resolution smoke test with the new feature disabled/enabled without changing logits.

Hard gates remain:

- New accepted SOTA requires `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, `memory_peak_bytes <= 16000000000`, file page cache charged inside the cgroup, no swap/OOM, coherent France output, immediate commit/push to `ssd/vendor/deepseek-token-rate-16gb`, and pushed-source reproduction.
- Any logit-changing path must pass op compare and fixed-text token-level top1 verification before a France benchmark.

Rollback criteria:

- If the default-off build changes accepted behavior, fails to build, cannot resolve top768 metadata, or requires extra host RAM/page-cache outside the 16GB cgroup, revert runtime source and keep only rejected documentation.

### 2026-07-05 Direct Reader Manifest Result

Result:

- Added `.Agent/run-tools/create_ds4_expert_offset_manifest.py` to generate a small offset manifest from the top768 profile and GGUF metadata without copying expert payload bytes.
- Generated manifest: `.Agent/profiles/vendor-ds4/current_sota_updown_decode_top768_touch_residual.offset_manifest.csv`
- Summary artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/top768-direct-reader-manifest-summary.json`

Validation:

- Entries resolved: `768`
- Missing tensors: `0`
- Size mismatches: `0`
- Payload bytes: `3422552064`
- Payload MiB: `3264.000`
- Residual ms sum: `2048.860`
- Manifest rows including header: `769`

Hashes:

- Generator SHA256: `794d3159615668ebf4ada4b50b857f3061ebbd6a4464f0c3c25275bb425c076c`
- Manifest SHA256: `1e4a87613b2c2bc3b6463405bc65f9a9ef704a0ca65b91afc824808c5c2698a2`
- Summary SHA256: `8b71b5b0e268000a8969cc38386ee0f6bd897f5d502e1ba085d721faf38887fb`

Decision:

- Direct-reader source path is now reproducible at the metadata level and does not need a 3.26 GiB pack file.
- This is still not a performance result and does not change accepted SOTA.
- Next source step may consume this manifest in default-off mode to initialize a separate q80 hot residual source/pool and counters. It must first preserve accepted behavior when disabled, then pass metadata smoke validation before any logit-changing compute path is enabled.

### 2026-07-05 Direct Manifest Loader Skeleton Result

Source change:

- Added a default-off direct manifest loader in `ggml/src/ggml-cuda/moe_stream.cu`.
- New envs:
  - `GGML_MOE_STREAM_ONE_DIRECT_MANIFEST`
  - `GGML_MOE_STREAM_ONE_DIRECT_MODEL`
  - `GGML_MOE_STREAM_ONE_DIRECT_IO=direct|odirect`
- The loader parses the committed top768 offset manifest, opens the model file, optionally opens the model with `O_DIRECT`, records counters, and reports at exit.
- It does not connect to current H2D, VRAM cache insertion, or compute. Direct reads remain unused until the later q80 hot pool step.
- Accepted gate cache behavior remains the rollback point; the new path is inert unless the direct manifest env is set.

Build validation:

- Command: `cmake --build build-ds4-moe-stream --target llama-cli -j 8`
- Result: success.
- New warnings from the added code were removed. Remaining warnings are pre-existing `ggml_cuda_moe_stream_link_anchor` missing declaration and unused existing parameters.

Metadata smoke:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/direct-manifest-loader-smoke.json`
- Run path: `/root/lfz/runs/vendor-ds4-16gb/20260704T162343Z-direct-manifest-metadata-smoke/short-cpu40`
- Purpose: metadata smoke only, not token-rate measurement and not SOTA.
- Command shape: strict 16GB systemd cgroup, no swap, `llama-cli -p Hi -n 1`, small `GGML_MOE_STREAM_ONE_CACHE_MIB=64`, direct manifest env enabled.
- Exit status: `0`
- Memory peak: `16000000000`
- `memory.events`: `oom=0`, `oom_kill=0`, `oom_group_kill=0`
- Loader log:
  - `O_DIRECT model reads enabled`
  - `loaded 768 entries bytes=3422552064`
  - exit report `hits=0 misses=0 reads=0 bytes=0 failures=0 direct_enabled=1 direct_reads=0 direct_failures=0 direct_fallbacks=0`

Decision:

- Keep the loader skeleton. It proves the source path can resolve top768 offsets from the committed manifest without generating a 3.26 GiB pack.
- This is not a performance result and current accepted SOTA remains `4.4 tok/s`.
- Next implementation step must add a separate q80 hot pool/prefill path, still default-off and not sharing the gate LRU. Only after that should exact MXFP4 x Q8_0 compute be wired and verified with op compare/top1.

### 2026-07-05 Direct Hot Pool Skeleton Plan

Purpose:

- Add the next default-off infrastructure layer for `cpu41 + top768 exact Q8_0 hot residual`: a separate direct hot pool and optional prefill path fed by the direct manifest.
- This is still source/path infrastructure, not a token-rate optimization result.

Implementation boundary:

1. Keep accepted SOTA behavior inert unless new envs are explicitly set.
2. Do not share the existing gate `g_vcache`, hash table, LRU/FIFO eviction, or gate prefill path.
3. Add a separate direct hot pool state keyed by the direct manifest entries.
4. Add env-gated allocation and prefill controls:
   - `GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB`
   - `GGML_MOE_STREAM_ONE_DIRECT_PREFILL_LIMIT`
5. Prefill may read model payload bytes from the direct manifest and copy them into the separate pool, but the pool must not be used by compute yet.
6. Record counters: pool slots, slot size, allocation bytes, prefill attempted/inserted/read failures/bytes/elapsed, direct read counters.
7. Validation for this step is build plus a low-risk smoke:
   - `POOL_MIB=0` or very small prefill limit is allowed;
   - no long France benchmark;
   - no token-rate promotion;
   - direct reads may be observed only for a deliberately tiny smoke and must stay within 16GB/no-swap cgroup.

Hard stops:

- If build fails, default-off path changes accepted behavior, allocation affects gate cache when env unset, or smoke causes OOM/swap, revert runtime source and keep rejected docs only.
- Do not wire exact MXFP4 x Q8_0 compute until this pool source path passes its smoke and the next plan update defines op-compare/top1 verification.

### 2026-07-05 Direct Hot Pool Skeleton Result

Source change:

- Added a separate direct hot pool state in `ggml/src/ggml-cuda/moe_stream.cu`.
- New envs:
  - `GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB`
  - `GGML_MOE_STREAM_ONE_DIRECT_PREFILL_LIMIT`
- The pool is independent from the accepted gate `g_vcache`; it has separate allocation, slots, manifest entries, and counters.
- The pool can prefill from the direct manifest into VRAM, but it is still not used by compute or lookup. Therefore this remains logit-neutral infrastructure.

Build validation:

- Command: `cmake --build build-ds4-moe-stream --target llama-cli -j 8`
- Result: success.
- Source SHA256 after fix: `f7bd37005f0f312c0f25a537ffe9a8ab8a713df15b5515bfba98e504aab1d374`

O_DIRECT gap and fix:

- First prefill smoke exposed a direct-read gap: the first GGUF expert slice had `model_offset % 4096 = 256`, while O_DIRECT requires aligned offsets.
- Before the fix, the pool inserted 1 entry but `direct_reads=0`, `direct_failures=1`, `direct_fallbacks=1`.
- Fixed by aligning the model offset down to 4096 bytes, reading a rounded-up O_DIRECT bounce buffer, and copying the requested expert slice into the destination buffer.

Passing smoke:

- Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/direct-hot-pool-prefill1-smoke.json`
- Run path: `/root/lfz/runs/vendor-ds4-16gb/20260704T164341Z-direct-hot-pool-prefill1-odirect-smoke/short-cpu40`
- Purpose: direct hot pool prefill=1 smoke only; not token-rate measurement and not SOTA.
- Exit status: `0`
- Memory peak: `789651456`
- `memory.events`: `oom=0`, `oom_kill=0`, `oom_group_kill=0`
- Pool log: `allocated 4.25 MiB slots=1 slot_sz=4456448`
- Prefill log: `attempted=1 inserted=1 bytes=4456448 elapsed_ms=17.764`
- Direct manifest report: `reads=1 bytes=4456448 failures=0 direct_reads=1 direct_failures=0 direct_fallbacks=0`

Decision:

- Keep the separate direct hot pool skeleton. It proves the top768 source path can prefill real GGUF expert payload from the raw model via O_DIRECT without generating a 3.26 GiB pack and without sharing the gate cache.
- This is still not a performance result; current accepted SOTA remains `4.4 tok/s`.
- Next plan step must decide whether to run a bounded larger prefill diagnostic under `cpu_moe=41` VRAM budget or move directly to exact MXFP4 x Q8_0 op-compare/top1 scaffolding. No long France benchmark is allowed before the compute path passes correctness verification.

### 2026-07-05 CPU41 Dual-Pool Fit Diagnostic Plan

Purpose:

- Verify the core VRAM assumption behind the `cpu41 + top768 exact Q8_0 hot residual` design: `cpu_moe=41` must be able to hold both the accepted gate cache and the separate top768 direct pool.

Diagnostic command shape:

- Short smoke only: `llama-cli -p Hi -n 1`.
- Strict 16GB systemd cgroup with `MemoryMax=16000000000` and `MemorySwapMax=0`.
- `--n-cpu-moe 41`
- `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
- `GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB=3264`
- `GGML_MOE_STREAM_ONE_DIRECT_PREFILL_LIMIT=0`
- Direct manifest/model env enabled, but no direct prefill payload read.
- No gate prefill profile for this diagnostic, to isolate allocation fit from TTFT/prefill cost.

Acceptance for this diagnostic:

- Build already passed.
- Process exits `0`.
- Direct pool allocates `3264 MiB` / `768` slots.
- Gate VRAM cache also allocates `13568 MiB` / about `3192` slots.
- No CUDA OOM, no host OOM, no swap, `oom_kill=0`.

Rejection:

- If either allocation fails, this exact VRAM layout is not viable and the plan must be revised before compute work.
- Even if it passes, it is still not a token-rate result and does not promote SOTA.

### 2026-07-05 CPU41 Dual-Pool Fit Diagnostic Result

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/cpu41-dualpool-fit-diagnostic.json`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T165213Z-cpu41-directpool3264-gate13568-fit/short-cpu41`

Result:

- Exit status: `0`
- Host cgroup memory peak: `788729856`
- `memory.events`: `oom=0`, `oom_kill=0`, `oom_group_kill=0`
- Direct hot pool allocation: `3264.00 MiB`, `768` slots, `slot_sz=4456448`, `pool_sz=3422552064`
- Gate VRAM cache allocation: `13.2 GiB`, `3192` slots, `4.25 MiB` each
- Resource sample GPU peak: `used=31860 MiB`, `free=251 MiB`
- Direct prefill was disabled: `attempted=0`, `reads=0`, `direct_reads=0`

Decision:

- The core `cpu_moe=41` VRAM assumption is viable in a short allocation diagnostic: full top768 direct pool and accepted-size gate cache can coexist.
- The free VRAM margin is very tight (`~251 MiB` observed in this short run), so the next implementation must avoid extra persistent GPU buffers and should treat full prefill/compute workspace growth as a first-class risk.
- This is not a token-rate result and does not change accepted SOTA.
- Next step should be correctness-first exact MXFP4 x Q8_0 op/top1 scaffolding, or a separately planned full-prefill timing diagnostic. No long France benchmark is allowed before compute verification.

### 2026-07-05 Full Direct Prefill Timing Diagnostic Plan

Purpose:

- Measure the source/prefill cost of loading all top768 direct hot pool entries from the raw GGUF model via O_DIRECT.
- This determines whether the direct pool path is already TTFT-impossible before exact Q8_0 compute is wired.

Diagnostic command shape:

- Short smoke only: `llama-cli -p Hi -n 1`.
- Strict 16GB systemd cgroup with `MemoryMax=16000000000` and `MemorySwapMax=0`.
- `--n-cpu-moe 41`
- `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
- `GGML_MOE_STREAM_ONE_DIRECT_POOL_MIB=3264`
- `GGML_MOE_STREAM_ONE_DIRECT_PREFILL_LIMIT=768`
- Direct manifest/model env enabled with `GGML_MOE_STREAM_ONE_DIRECT_IO=direct`.
- Gate prefill remains disabled to isolate direct prefill timing.
- The pool is still not used by compute, so this is logit-neutral infrastructure timing, not a token-rate result.

Acceptance for this diagnostic:

- Process exits `0`.
- No host OOM/swap: `oom=0`, `oom_kill=0`.
- Direct hot pool reports `attempted=768`, `inserted=768`, `read_failures=0`, `copy_failures=0`.
- Direct manifest reports `reads=768`, `direct_reads=768`, `direct_failures=0`, `direct_fallbacks=0`.
- Record elapsed_ms and compare it to the accepted TTFT slack (`~725 ms`) and earlier serialized top768 estimate (`~1148 ms`).

Decision rule:

- If full direct prefill is much above TTFT slack, do not promote or long-benchmark this path until prefill is overlapped/async or reduced.
- If it is near or below slack, proceed to exact MXFP4 x Q8_0 op-compare/top1 scaffolding.

### 2026-07-05 Full Direct Prefill Timing Diagnostic Result

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/full-direct-prefill768-timing.json`

Run:

- `/root/lfz/runs/vendor-ds4-16gb/20260704T165954Z-cpu41-directpool3264-prefill768-timing/short-cpu41`

Result:

- Exit status: `0`
- Host cgroup memory peak: `792145920`
- `memory.events`: `oom=0`, `oom_kill=0`, `oom_group_kill=0`
- Direct hot pool: `3264.00 MiB`, `768` slots, `slot_sz=4456448`
- Full direct prefill: `attempted=768`, `inserted=768`, `read_failures=0`, `copy_failures=0`
- Direct manifest reads: `reads=768`, `bytes=3422552064`, `direct_reads=768`, `direct_failures=0`, `direct_fallbacks=0`
- Measured synchronous direct prefill elapsed: `1723.238 ms`
- Effective throughput: about `1.850 GiB/s`
- Gate cache still allocated after direct prefill: `13.2 GiB`, `3192` slots
- Resource sample GPU peak: `used=31866 MiB`, `free=245 MiB`

TTFT analysis:

- Accepted TTFT slack is only about `725.135454 ms`.
- The earlier serialized estimate for top768 direct payload was about `1148 ms`.
- Measured synchronous prefill is `1723.238 ms`, about `998.103 ms` over accepted TTFT slack.
- Therefore, synchronous full top768 prefill is not promotable under the TTFT gate even though allocation and O_DIRECT reads work.

Decision:

- Keep the source path and direct pool infrastructure.
- Do not run a long France benchmark with synchronous full top768 prefill as an accepted candidate.
- Next plan must either:
  - make direct prefill overlapped/async/hidden enough to recover about `1.0s` TTFT, or
  - proceed to exact MXFP4 x Q8_0 op/top1 scaffolding while treating synchronous prefill as TTFT-rejected until overlap exists.

### 2026-07-05 Exact Q8_0 Correctness Scaffold Plan

Purpose:

- Move from source/pool infrastructure to correctness-first exact MXFP4 x Q8_0 validation.
- The previous Q8_0 up-only token verifier failed (`same_top1=136/145`, first mismatch at position `4`), even though an op-level compare showed small per-op deltas. Therefore the next work must tighten the correctness gate before any performance path is wired.

Existing evidence to respect:

- `llama-results` top1 verifier exists and passed self-check: `.Agent/runs/20260704-vendor-ds4-coldstart/llama-results-top1-verifier-implementation.json`.
- Prior Q8_0 up-only top1 candidate failed: `.Agent/runs/20260704-vendor-ds4-coldstart/q80-up-top1-probe-result.json`.
- Prior op compare was not enough to promote correctness: `.Agent/runs/20260704-vendor-ds4-coldstart/q80-up-compare-probe-result.json`.
- CPU reference semantics remain MXFP4 x Q8_0 via `ggml_vec_dot_mxfp4_q8_0` and Q8_0 activation.

Implementation boundary for this step:

1. Do not connect the direct hot pool to runtime compute in `llama-cli`.
2. Do not run a long France performance benchmark.
3. Inspect current source and prior diffs to identify why the earlier Q8_0 up-only top1 verifier failed.
4. Build or refresh the diagnostic correctness tooling only:
   - `llama-results` top1 verifier build/self-check, or
   - a minimal op/top1 scaffold that is default-off and diagnostic-only.
5. Any future logit-changing path must pass `same_top1 == n_positions` before it can be benchmarked.

This step can be accepted as progress only if it produces one of:

- a reproducible verifier/self-check record from current pushed source, or
- a precise reject/replan artifact explaining why the current Q8_0 path cannot yet satisfy top1 and what source change is required next.

SOTA status:

- Current accepted SOTA remains `4.4 tok/s`.
- No result in this section can promote SOTA because it is diagnostic/correctness infrastructure only.

### 2026-07-05 Exact Q8_0 Correctness Scaffold Result

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/results-top1-comparator-fix-selfcheck.json`

Finding:

- The existing `llama-results` single-run top1 report worked, but comparator mode with `--output result.gguf --check --top1-fail-on-mismatch` was not actually usable in the current sequential-logits path.
- Failed run: `/root/lfz/runs/vendor-ds4-16gb/20260704T171004Z-results-top1-current-selfcheck/selfcheck`
- Failure: baseline and check both exited `139` after writing the single top1 report, before writing/reading `result.gguf`.
- This was not an OOM kill: `oom=0`, `oom_kill=0`.

Root cause:

- `tools/results/results.cpp` materialized logits in `logits_calc`, but when writing `result.gguf` it called `llama_get_logits_ith(lctx, i)` again for every historical token position.
- In `--sequential-logits` mode those historical logits are not all valid in the context after the loop, so output writing could dereference invalid pointers and segfault.

Fix:

- Write the GGUF logits tensor directly from the already materialized `logits_calc` vector.
- This does not touch `llama-cli` or model runtime behavior.

Validation:

- Build command: `cmake --build build-ds4-moe-stream --target llama-results -j 8`
- Build result: passed.
- Passing selfcheck run: `/root/lfz/runs/vendor-ds4-16gb/20260704T171436Z-results-top1-current-selfcheck-fixed/selfcheck`
- Baseline status: `0`
- Check status: `0`
- Host cgroup memory peak: `12487258112`
- `memory.events`: `oom=0`, `oom_kill=0`, `oom_group_kill=0`
- `result.gguf` size: `75497472 bytes`
- Comparator report: `n_tokens=145`, `n_vocab=129280`, `same_top1=145`, `first_mismatch_pos=-1`, `max_abs=0`, `mean_abs=0`

Decision:

- Keep this verifier fix. It is required before any future exact Q8_0/direct-pool compute candidate can be safely tested.
- Future logit-changing candidates must produce a baseline `result.gguf`, run `--check --top1-report --top1-fail-on-mismatch`, and pass `same_top1 == n_tokens` before any long France benchmark or SOTA promotion.
- Current accepted SOTA remains `4.4 tok/s`.

### 2026-07-05 Latest Plan Update: Q4_K Top4 Combo Closure Before More Source Work

Current accepted SOTA remains:

- Token rate: `eval_tok_s=4.4`
- Prompt rate: `prompt_tok_s=1.8`
- TTFT guard run: `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`
- Prior accepted pushed repro: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`
- Current-head guard metrics: `TTFT=32087.738292 ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`
- Promotion TTFT gate: `TTFT <= 33617.688744 ms`
- Accepted model: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`
- Accepted model size: `156148189760 bytes` (`145.42 GiB`)
- Accepted SOTA source/record state is pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`

Promotion protocol, still mandatory:

1. A candidate must beat `4.4 tok/s`, not merely tie it.
2. The run must be strict cold-start vendor DeepSeek under `drop_caches`, `MemoryMax=16000000000`, `MemorySwapMax=0`, and page cache charged inside the same cgroup.
3. France output for `Please introduce France in a short paragraph.` must be semantically correct and coherent. Any logit-changing path must first pass the `llama-results` top1 verifier with `same_top1 == n_tokens`.
4. TTFT must be `<=33617.688744 ms` for accepted promotion. TTFT-over-gate results may be recorded and pushed only as rejected diagnostics.
5. Record exact run path, source head, diff summary, env/CLI, build command, binary hashes, model/profile/pack hashes, stdout answer, token rates, TTFT, elapsed time, cgroup `memory.peak`, `memory.stat`, `memory.events`, pack/cache counters, and comparison to the accepted SOTA.
6. When a compliant new SOTA appears, immediately commit and push source plus records to `ssd/vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`, then reproduce once from pushed source before marking it accepted.
7. If the pushed-source reproduction fails, keep the failed result as rejected, revert runtime source to the last accepted SOTA behavior, update this plan, commit/push the rejection record, and keep accepted SOTA unchanged.

Recent closure status:

- Thread-count sweep is closed for the current accepted route. Results are in `.Agent/runs/20260705-vendor-ds4-coldstart/thread-count-sweep-result.json`. `-t/-tb 24` tied `4.4 tok/s`; `32`, `40`, and `60` regressed, with `60` also missing TTFT.
- External artifact refresh is closed for direct source work now. Results are in `.Agent/runs/20260705-vendor-ds4-coldstart/external-artifact-refresh-4expert-ssd-dflash.json`. No external artifact is source-ready for immediate vendor SOTA promotion.
- CUDA graph is not part of the accepted SOTA route; current accepted env keeps `GGML_CUDA_DISABLE_GRAPHS=1`. Any future graph attempt must be default-off, correctness-verified, and compared under the same cold-start gate.
- The main remaining bottleneck is still CPU fallback and source/page movement for routed expert up/down work after the gate-only cache. Gate cache hit behavior is not the limiting factor in the rejected thread sweep.

Current hard-bound focus:

- The only still-plausible combination from the external refresh is `4Expert/top_k=4 + Q4_K routed experts + source/page elimination + faster CPU fallback`.
- Current native active expert payload uses `expert_used_count=6`, routed expert role size `4.25 MiB` per layer/expert, and about `148.916016 GiB` hot fallback payload in the accepted decode model.
- The 4Expert Q4_K candidate uses `expert_used_count=4`, routed expert role size `4.5 MiB`, so the active payload ratio is `(4/6) * (4.5/4.25) = 0.7058823529411764`.
- Estimated top4 Q4_K hot payload is `105.11718776470589 GiB`.
- After perfect source/page elimination, the remaining hot fallback time allowed for `10 tok/s` is only `1642.8865524000012 ms`.
- Therefore Q4_K CPU dot bandwidth must exceed `63.983 GiB/s` with zero integration overhead, and exceed `91.978 GiB/s` with only `500 ms` of routing/stream/cache/full-model overhead.
- Rule: reopen this route only if both DS4-like Q4_K shapes exceed `91.978 GiB/s` with margin. If either shape is below `63.983 GiB/s`, reject the route. A value between `63.983` and `91.978 GiB/s` is not source-ready.

Immediate execution plan:

1. Do not run another long France benchmark until the design step produces a hard-bound pass or a logit-changing candidate passes top1 correctness.
2. Fix the Q4_K dot microprobe validity first. The first harness run produced zero-output results (`sink` equal to the epsilon-only value), so those throughput numbers are invalid and must not be used.
3. Add diagnostic checks to `.Agent/run-tools/q4k_dot_harness.cpp`:
   - print nonzero evidence for Q4_K and Q8_K block fields (`d`, `dmin`, scales, `qs`, `bsums`);
   - print a float reference dot for row 0;
   - print the corresponding Q4_K x Q8_K dot output;
   - fail fast if warmup absolute output sum is zero.
4. Rerun the microprobe only after nonzero correctness evidence exists. Required shapes are:
   - up/gate-like: `k=4096`, `rows=2048`
   - down-like: `k=2048`, `rows=4096`
   - thread counts: `1`, `8`, `20`, `24`, `32`
5. Compare valid `q4_src_gib_s` against the `63.983` and `91.978 GiB/s` gates above. Record the result in `.Agent/runs/20260705-vendor-ds4-coldstart/`.
6. If Q4_K fails the hard-bound gate, close the 4Expert/Q4_K combo route and do not download the full 4Expert GGUF or implement Q4_K stream/cache support.
7. If Q4_K passes with margin, write a new source implementation plan before editing runtime code. That plan must explicitly cover:
   - 4Expert GGUF disk-space/repro policy;
   - routing tensor alias compatibility for `ffn_gate_tid2eid.weight`;
   - Q4_K one-stream/cache/pack support;
   - strict top1 verifier before benchmark;
   - France and five-prompt correctness;
   - 16GB cgroup/page-cache accounting;
   - TTFT gate and pushed-source reproduction.

SOTA status for this update:

- This is a planning update only.
- Current accepted SOTA remains `4.4 tok/s`.
- The invalid zero-output Q4_K harness measurements are not accepted performance evidence.

### 2026-07-05 Q4_K Top4 Combo Microprobe Result

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/q4k-top4-combo-microprobe-result.json`

Validity fix:

- The first Q4_K harness results were invalid because the warmup output was exactly zero.
- Root cause: the standalone harness called CPU quant/dot functions without `ggml_cpu_init()`, so the CPU fp16 lookup table used by `GGML_CPU_FP16_TO_FP32` inside the dot path was not initialized.
- Fix: call `ggml_cpu_init()` before quantization and add block-field diagnostics plus fail-fast zero-output detection.
- Valid evidence after the fix:
  - Q4_K block fields are nonzero: `q4_d=0.0351257324`, `q4_dmin=0.256347656`, `q4_scales_nz=12`, `q4_qs_nz=126`
  - Q8_K fields are nonzero: `q8_d=-0.124245711`, `q8_qs_nz=255`, `q8_bsums_nz=16`
  - up/gate-like first row: `ref0=-1146.38565`, `q4dot0=-1509.15625`, `warm_abs_sum=8976186.09`
  - down-like first row: `ref0=1243.30404`, `q4dot0=1085.05469`, `warm_abs_sum=12776267.4`

Valid microprobe results:

| threads | up/gate-like `src_gib_s` | down-like `src_gib_s` | verdict |
| --- | ---: | ---: | --- |
| 1 | `16.356` | `14.914` | below zero-overhead gate |
| 8 | `33.511` | `49.066` | below zero-overhead gate |
| 20 | `73.542` | `111.682` | mixed; up shape below 500ms-overhead gate |
| 24 | `89.257` | `86.978` | both below 500ms-overhead gate |
| 32 | `116.115` | `115.044` | passes reopen gate |

Decision:

- The hard microprobe reopen rule is satisfied only at `32` threads: both DS4-like shapes exceed the `91.978 GiB/s` 500ms-overhead gate.
- This does not change SOTA. It is not a model run, does not prove TTFT, does not prove France correctness, and does not prove 16GB end-to-end behavior.
- Current accepted SOTA remains `4.4 tok/s`.
- The route is reopened only for a source-design phase: `4Expert/top_k=4 + Q4_K routed experts + source/page elimination + faster CPU fallback`.

### 2026-07-05 Next Source Plan: Default-Off 4Expert/Q4_K Integration Probe

Purpose:

- Test whether the microprobe upper-bound can be converted into a real vendor DeepSeek path without breaking correctness or the 16GB/TTFT gates.
- This must stay design-first and correctness-first. No long France benchmark is allowed until loading/routing/Q4_K correctness is verified.

Required source design steps before runtime edits:

1. 4Expert GGUF acquisition and reproducibility policy.
   - Candidate: `cloudyu/DeepSeek-V4-Flash-4Expert-GGUF/ds4flash-4expert.gguf`.
   - Do not delete accepted SOTA model, accepted expert pack, accepted run records, or pushed-source repro artifacts.
   - Because `/root` free space is tight, first write an explicit disk plan that lists what can be removed or where the full GGUF can be placed. No full download until the disk plan preserves SOTA reproducibility.
   - Record full URL, expected size `164465760544 bytes`, SHA256 if downloaded, and exact local path.

2. Routing tensor compatibility.
   - The 4Expert candidate uses routing tensor name `blk.*.ffn_gate_tid2eid.weight`.
   - Current vendor probes and accepted native path use `blk.*.ffn_gate_tid2eid`.
   - Implement a default-off alias loader/probe only if needed; it must not alter accepted native SOTA behavior when the env/config is unset.
   - First validation is metadata/load-only, not a token-rate benchmark.

3. Q4_K stream/cache/pack support.
   - Current one-stream cache type allowlist covers `IQ3_XXS` and the DS4 experimental native types `MXFP4/F8_E4M3_B128`; Q4_K is not yet a supported stream/cache path.
   - Add Q4_K support behind an explicit env/config gate.
   - Prove cache slot size, tensor offsets, direct read alignment, pack manifest, and copy size before wiring compute.
   - The accepted gate-only native SOTA path must remain unchanged by default.

4. Correctness gates before performance.
   - Any logit-changing path must first create a baseline `result.gguf` and pass `llama-results --check --top1-report --top1-fail-on-mismatch` with `same_top1 == n_tokens`.
   - Then run the France prompt and record the exact answer. The answer must be semantically correct and coherent.
   - Then run the five-prompt warm/cold correctness set if the France prompt passes.

5. Strict performance gate only after correctness.
   - Benchmark only under strict cold `drop_caches`, `MemoryMax=16000000000`, `MemorySwapMax=0`, and page cache charged inside cgroup.
   - Promotion requires `eval_tok_s > 4.4`, `TTFT <= 33617.688744 ms`, `ram_ok=true`, no OOM/swap, and correct France output.
   - If TTFT exceeds the gate but token rate improves, commit and push only as a rejected diagnostic, labeled not accepted.

Execution order:

1. Write a disk/repro plan for the 4Expert GGUF and preserve all current SOTA artifacts.
2. Implement or test metadata-only 4Expert GGUF loading and routing alias behavior.
3. Implement default-off Q4_K stream/cache/pack support and validate with small read/cache diagnostics.
4. Run top1 correctness comparison before any long model benchmark.
5. Run strict cold France benchmark only after top1 and output correctness pass.
6. For any compliant new SOTA, immediately record full reproduction information, commit and push to `ssd/vendor/deepseek-token-rate-16gb`, then reproduce from pushed source before promotion.

### 2026-07-05 4Expert/Q4_K Disk Audit And Default-Off Admission Plan

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-q4k-disk-and-admission-plan.json`

Disk result:

- Full 4Expert GGUF download is not safe right now.
- Only large filesystem is `/dev/root`, size `993G`, used `991G`, available about `1.8G`.
- Candidate file is `cloudyu/DeepSeek-V4-Flash-4Expert-GGUF/ds4flash-4expert.gguf`, expected size `164465760544 bytes`.
- Practical free-space requirement is at least `180G` so the file can download, hash, and leave room for records.
- No deletion was performed.

Protected SOTA assets:

- `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`
- `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`
- `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`
- `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`

Large cleanup candidates that require explicit approval before deletion:

- `/root/lfz/models/GLM-5.2-UD-IQ3_XXS`, about `263G`, unrelated to current vendor DS4 SOTA.
- `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.expert-pack`, `147174760448 bytes`, not used by the accepted SOTA env but adjacent to accepted DeepSeek model assets.
- `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-promptset-gate-union-firstorder-20260702.pack`, `38420348928 bytes`, not the accepted France SOTA pack but useful for prompt-set diagnostics.

Source audit:

- Routing alias gap remains: current name is `blk.%d.ffn_gate_tid2eid`; 4Expert candidate uses `blk.%d.ffn_gate_tid2eid.weight`.
- Q4_K kernel support exists in CUDA MMVQ (`ggml/src/ggml-cuda/mmvq.cu` includes Q4_K vec-dot and switch support).
- Q4_K stream-one admission is currently blocked by CPU bridge and CUDA stream-one type gates.
- The patch to try now must be default-off:
  - add `GGML_MOE_STREAM_ONE_Q4K=1` gate;
  - allow Q4_K only in stream-one eligibility;
  - do not broaden down-batch or up-gate batch support;
  - keep accepted MXFP4 SOTA path unchanged when the env is unset.

Validation required for this patch:

1. Build must pass.
2. Default-off accepted-path guard must pass or at minimum prove no default runtime branch changed.
3. No SOTA promotion is possible from this patch alone because no 4Expert model run, TTFT, 16GB cgroup, or correctness benchmark has happened.
4. Full 4Expert download remains blocked until disk space is explicitly made available without deleting protected SOTA assets.

### 2026-07-05 Default-Off Q4_K Admission Validation

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-q4k-defaultoff-admission-validation.json`

Patch:

- Added `GGML_MOE_STREAM_ONE_Q4K=1` as an explicit env gate.
- CPU bridge now uses `ggml_cuda_moe_stream_supports_one_type()` only for stream-one eligibility.
- CUDA stream-one type gate allows `GGML_TYPE_Q4_K` only when `GGML_MOE_STREAM_ONE_Q4K` is set and `GGML_MOE_STREAM_ONE_NAME_FILTER` allows the tensor.
- Generic batch/down-batch/up-gate support is not broadened.
- Accepted MXFP4 SOTA behavior is unchanged when `GGML_MOE_STREAM_ONE_Q4K` is unset.

Validation:

- Build command passed: `cmake --build build-ds4-moe-stream --target llama-cli -j 8`
- Default-off guard did not set `GGML_MOE_STREAM_ONE_Q4K`.
- Guard 1:
  - Run: `/root/lfz/runs/vendor-ds4-16gb/20260705T131626Z-20260705_q4k_admission_defaultoff_guard/france-q4k-admission-defaultoff-cpu40-vram0gb`
  - `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=33939.958018 ms`
  - `memory_peak_bytes=16000000000`, `memory_file_bytes=15096578048`
  - `ram_ok=true`, `correctness_ok=true`, no OOM
  - TTFT was slightly over the promotion gate, so this run is guard-only and not promotable.
- Guard 2 repeat:
  - Run: `/root/lfz/runs/vendor-ds4-16gb/20260705T131824Z-20260705_q4k_admission_defaultoff_guard_repeat/france-q4k-admission-defaultoff-repeat-cpu40-vram0gb`
  - `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=32626.042709 ms`
  - `memory_peak_bytes=16000000000`, `memory_file_bytes=15098937344`
  - `ram_ok=true`, `correctness_ok=true`, no OOM
  - This repeat satisfies the accepted SOTA guard constraints but only ties `4.4 tok/s`, so it is not a new SOTA.

Decision:

- Commit and push this default-off infrastructure because it is source progress toward Q4_K/4Expert validation and repeat guard confirms no accepted-path regression.
- Current accepted SOTA remains `4.4 tok/s`.
- Full 4Expert validation remains blocked by disk space until at least `180G` safe free space is available or the user explicitly approves cleanup of non-SOTA large assets.

### 2026-07-05 4Expert Header Alias Probe And Plan

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-header-alias-probe-and-plan.json`

Header-only probe:

- Downloaded only the first `64MiB` of the cloudyu 4Expert GGUF, not the full model.
- Local header path: `/root/lfz/models/_gguf_header_probe/cloudyu-4expert/ds4flash-4expert.header64m.gguf.part`
- Header SHA256: `dfa48de9479bd36c7e9b02e58635726a3a5b2215aa17c975a44975cf89704222`
- Parsed header:
  - `general.architecture=deepseek4`
  - `block_count=43`
  - `expert_count=256`
  - `expert_used_count=4`
  - tensor type counts: `Q4_K=129`, `Q8_0=366`, `F32=492`, `F16=338`, `I32=3`
  - tensor info ends at byte `5845209`, so the 64MiB range is sufficient for metadata.

Routing alias evidence:

- Current vendor tensor name is `blk.%d.ffn_gate_tid2eid`.
- 4Expert header contains:
  - `blk.0.ffn_gate_tid2eid.weight`, `I32`, shape `[4, 129280]`
  - `blk.1.ffn_gate_tid2eid.weight`, `I32`, shape `[4, 129280]`
  - `blk.2.ffn_gate_tid2eid.weight`, `I32`, shape `[4, 129280]`
- 4Expert routed expert tensors are Q4_K, for example:
  - `blk.0.ffn_gate_exps.weight`, `Q4_K`, shape `[4096, 2048, 256]`
  - `blk.0.ffn_up_exps.weight`, `Q4_K`, shape `[4096, 2048, 256]`
  - `blk.0.ffn_down_exps.weight`, `Q4_K`, shape `[2048, 4096, 256]`

Prepared alias patch:

- Add `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`.
- For DeepSeek4 `ffn_gate_tid2eid` only:
  - first try the current tensor name;
  - if absent and the env is set, try the `.weight` alias;
  - create the tensor using the discovered metadata shape and selected name.
- Env unset behavior must stay unchanged for accepted native SOTA.

Validation required:

1. Build must pass.
2. Default-off accepted-path guard must pass with `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS` unset.
3. Full alias-on model load cannot be validated until the full 4Expert GGUF is available.
4. No SOTA promotion is possible from header-only metadata or alias infrastructure alone.

### 2026-07-05 Default-Off Tid2Eid Alias Validation

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-tid2eid-alias-defaultoff-validation.json`

Patch:

- Added `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`.
- In DeepSeek4 `llama-model.cpp`, `ffn_gate_tid2eid` now:
  - tries current name `blk.%d.ffn_gate_tid2eid` first;
  - if absent and env is set, tries `blk.%d.ffn_gate_tid2eid.weight`;
  - creates the tensor with the selected name and discovered metadata shape.
- Env unset behavior is unchanged.

Validation:

- Build passed: `cmake --build build-ds4-moe-stream --target llama-cli -j 8`
- Default-off guard did not set `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS` or `GGML_MOE_STREAM_ONE_Q4K`.
- Guard run:
  - `/root/lfz/runs/vendor-ds4-16gb/20260705T133334Z-20260705_tid2eid_alias_defaultoff_guard/france-tid2eid-alias-defaultoff-cpu40-vram0gb`
  - `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=33430.771645 ms`
  - `memory_peak_bytes=16000000000`, `memory_file_bytes=15095029760`
  - `ram_ok=true`, `correctness_ok=true`, no OOM

Decision:

- Commit and push this default-off alias infrastructure.
- Current accepted SOTA remains `4.4 tok/s`; this guard only ties SOTA and is not a new result.
- Alias-on full load and any Q4_K/4Expert correctness benchmark remain blocked until the complete 4Expert GGUF is available.

### 2026-07-05 Header-Only 4Expert Q4_K Direct Manifest

Artifacts:

- Script: `.Agent/run-tools/create_gguf_header_expert_manifest.py`
- Summary: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-q4k-header-manifest-summary.json`
- Generated direct manifest, not committed: `/root/lfz/models/_gguf_header_probe/cloudyu-4expert/ds4flash-4expert-q4k-all-experts.direct-manifest.csv`

Result:

- The manifest was generated from the 64MiB header range only; no full 4Expert model was downloaded.
- Header SHA256: `dfa48de9479bd36c7e9b02e58635726a3a5b2215aa17c975a44975cf89704222`
- Manifest SHA256: `42f127cd1d7e147d49797d9e8a860386e0e9f10fbfac5be37e86c14bddeebaa6`
- Manifest rows: `33024`
- Manifest tensors: `129`
- Per expert slice size: `4718592 bytes`
- Total Q4_K routed expert payload represented by manifest: `155826782208 bytes` (`145.125 GiB`)
- Data start: `5845216`
- Tensor info end: `5845209`
- First row: `blk.0.ffn_gate_exps.weight,0,8638978336,4718592`

Decision:

- Keep this script and summary. They make future Q4_K direct manifest generation reproducible once the full 4Expert GGUF is available.
- This is not a SOTA result and does not validate correctness or TTFT.
- Next blocked item for real 4Expert validation remains disk space for the full GGUF.

### 2026-07-05 Sparse/Header-Only Alias Validation Rejection

Artifact:

- `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-sparse-alias-validation-rejection.json`

Evaluation:

- `llama-gguf` can read GGUF header/data but does not instantiate `llama_model::load_tensors()`, so it cannot exercise the DeepSeek4 `tid2eid` alias code path.
- A sparse logical 4Expert file made from the 64MiB header range would map missing tensor data as zero-filled holes.
- Running `llama-cli` on that sparse file might exercise some loading code, but it would not be the real model:
  - missing dense and expert weights would be zero;
  - output correctness would be invalid;
  - TTFT/token-rate and page-cache behavior would not represent the real 4Expert GGUF.

Decision:

- Reject sparse/header-only files as a substitute for alias-on correctness, TTFT, or token-rate validation.
- The only valid next step for real 4Expert/Q4_K verification is a complete GGUF file or another real mmap/readable backing store for the full file contents.
- Minimum safe disk target remains `>=180G` free.
- No deletion was performed.
