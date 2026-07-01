# Vendor DeepSeek V4 16GB Token Rate Progress

## 2026-07-01

- Initialized vendor-only optimization plan for `/root/lfz/vendor/llama.cpp-deepseek-v4`.
- Current branch before implementation: `feat/ds4-moe-stream-on-vendor`.
- Current HEAD before implementation: `ef19fbdbf cuda: optionally drop streamed expert mmap pages`.
- Known invalid high-rate reference: `/root/lfz/runs/vendor-sota/20260701T025805Z/france-cpu60`, `eval_tok_s=9.9`, but max RSS was about 67.2 GiB, so it violates the 16 GB host RAM gate.
- Known clean 16 GB reference: `/root/lfz/runs/vendor-sota-ram16/20260701T031526Z-16gb-clean-final/france-cpu40`, `eval_tok_s=1.1`, `prompt_tok_s=0.8`, `memory_peak_bytes=16000000000`, answer was coherent.
- Action required before optimization acceptance: rerun the vendor baseline under the new strict runner and use that as the official baseline.

## Run Log

### 2026-07-01T04:44:44Z - Accepted hard-16GB vendor baseline

- Accepted current highest compliant token rate: `eval_tok_s=7.9`, `prompt_tok_s=4.7`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T044444Z-hard16g-cpu40-vs-cpu37/france-cpu40-vram2gb`.
- Repo commit used for the run: `7b2eab7783efe2cb11472ad8e129b04556ed3220`.
- Memory gate: `MemoryMax=16000000000`, `MemorySwapMax=0`, `ram_kill_threshold_bytes=16000000000`, `ram_ok=true`, `ram_limit_killed=false`, `oom_seen=false`.
- Host memory evidence: `memory_peak_bytes=16000000000`, `memory_max_events=58`. This touched the cgroup max boundary but did not exceed the hard limit or trigger OOM/kill.
- TTFT gate: `ttft_estimate_ms=12823.978055`. This is the accepted baseline for the next TTFT +20% ceiling: `15388.773666 ms`.
- Correctness gate: passed. The France answer is semantically correct and coherent; it contains a spelling typo (`Rennowned`) but no semantic issue.
- VRAM evidence: the run used the strict vendor runner with GPU sampling archived in `resource_samples.tsv`.
- Exact command and environment are committed in `.Agent/runs/20260701-vendor-ds4-16gb-token-rate/accepted-hard16g-cpu40-vram2.json`.

Rejected higher-rate candidates:

- `cpu_moe=37`, run directory `/root/lfz/runs/vendor-ds4-16gb/20260701T044444Z-hard16g-cpu40-vs-cpu37/france-cpu37-vram2gb`, reached `eval_tok_s=8.2` and correct output, but `ttft_estimate_ms=20090.7637`, which exceeds the accepted baseline by more than 20%.
- `cpu_moe=39`, run directory `/root/lfz/runs/vendor-ds4-16gb/20260701T044615Z-hard16g-cpu39-cpu38/france-cpu39-vram2gb`, reached `eval_tok_s=8.4` and correct output, but `ttft_estimate_ms=18726.305226`, which exceeds the accepted baseline by more than 20%.
- `cpu_moe=38`, run directory `/root/lfz/runs/vendor-ds4-16gb/20260701T044615Z-hard16g-cpu39-cpu38/france-cpu38-vram2gb`, reached `eval_tok_s=8.0` and correct output, but `ttft_estimate_ms=20417.454608`, which exceeds the accepted baseline by more than 20%.

### 2026-07-01T04:52:53Z - Accepted VRAM cache increase

- Optimization: keep `cpu_moe=40` and increase `GGML_MOE_VRAM_CACHE_GB` from `2` to `4`.
- Rationale: the previous accepted run spent most cgroup memory on file/page cache and touched the cgroup hard limit. More VRAM cache should reduce streamed expert/page-cache pressure while preserving the same CPU-MoE split and TTFT profile.
- Accepted current highest compliant token rate: `eval_tok_s=8.2`, `prompt_tok_s=4.8`.
- Previous accepted token rate: `eval_tok_s=7.9`, so measured gain is about `3.8%`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T045253Z-hard16g-cpu40-vram4/france-cpu40-vram4gb`.
- Repo commit used for the run: `ed37b098b5ae19d8ca1a8ae04f3359756544e585`.
- Memory gate: `MemoryMax=16000000000`, `MemorySwapMax=0`, `ram_kill_threshold_bytes=16000000000`, `ram_ok=true`, `ram_limit_killed=false`, `oom_seen=false`.
- Host memory evidence: `memory_peak_bytes=15793479680`, `memory_max_events=0`. Page cache is included in the cgroup accounting: `file=14950096896`.
- TTFT gate: `ttft_estimate_ms=12146.360264`, lower than the previous accepted `12823.978055`.
- Correctness gate: passed. The France answer is semantically correct and coherent; it contains the same spelling typo (`Rennowned`) but no semantic issue.
- Exact command and environment are committed in `.Agent/runs/20260701-vendor-ds4-16gb-token-rate/accepted-hard16g-cpu40-vram4.json`.

### 2026-07-01T05:52:37Z - Prompt-set steady-state stability test failed

- Test purpose: check whether one France warmup in the same 16 GB cgroup and same long-lived `llama-cli` process makes different prompts run near the 8-10 tok/s single-prompt steady-state range.
- Tested config: `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=4`, `MemoryMax=16000000000`, `MemorySwapMax=0`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb-promptset/20260701T055237Z-vram4-cpu40`.
- Result: failed steady-state stability. Rates were `1.6`, `1.8`, `1.3`, `1.0`, `1.2`, `0.9` tok/s for warmup France, France again, quantum, Fibonacci, Japan, and climate respectively.
- Memory evidence: `memory_peak=16000000000`, `memory.events max=422741`, `oom=0`, `oom_kill=0`; page cache was included in cgroup accounting with `file=14729252864`.
- Interpretation: France-only warmup does not heat enough prompt-dependent MoE expert pages under the 16 GB cgroup. The run remains dominated by file/page-cache churn and major faults (`pgmajfault=4774376`, `workingset_refault_file=147025029`).
- Follow-up bottleneck: to make prompt-independent steady state stable, we need either a broader multi-prompt warmup that stays within 16 GB, more selective/high-frequency expert residency, or a streaming/prefetch change that reduces page-cache churn across changing experts.
- Full reproducibility record is committed in `.Agent/runs/20260701-vendor-ds4-16gb-token-rate/promptset-vram4-cpu40-failed.json`.

### 2026-07-01T06:21:11Z - Cold-start profile, VRAM cache sweep

- Purpose: execute the first cold-start plan step with explicit `sync; echo 3 > /proc/sys/vm/drop_caches` before each case, and record page-fault/reclaim counters directly from the runner.
- Runner/tooling commit: `63c79da05 vendor-ds4: add cold-start profiling to runner`.
- Common config: `cpu_moe=40`, `MemoryMax=16000000000`, `MemorySwapMax=0`, `GGML_MOE_STREAM=1`, `GGML_MOE_STREAM_DONTNEED=1`, `GGML_CUDA_DISABLE_GRAPHS=1`.

| vram cache | eval tok/s | prompt tok/s | TTFT ms | memory max events | pgmajfault | workingset refault file | run dir |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 GB | 1.3 | 0.8 | 44034.028848 | 66823 | 826309 | 17684614 | `/root/lfz/runs/vendor-ds4-16gb/20260701T062433Z-cold-profile-cpu40-vram2/france-cpu40-vram2gb` |
| 4 GB | 1.2 | 0.7 | 45177.675222 | 76608 | 906331 | 20521336 | `/root/lfz/runs/vendor-ds4-16gb/20260701T062111Z-cold-profile-cpu40-vram4/france-cpu40-vram4gb` |
| 5 GB | 1.2 | 0.8 | 43243.127206 | 52576 | 730394 | 12260048 | `/root/lfz/runs/vendor-ds4-16gb/20260701T062724Z-cold-profile-cpu40-vram5/france-cpu40-vram5gb` |

- Correctness: all three outputs passed the France semantic/coherence heuristic.
- RAM: all three stayed inside the model cgroup hard limit with no OOM and no runner RAM-limit kill, but all reached `memory_peak_bytes=16000000000`, so the cold-start path is still at the cgroup boundary.
- Interpretation: cold-start performance is not improved by the warm-optimal `vram_cache=4`; all variants remain around `1.2-1.3 tok/s`. `vram5` reduced major faults/refaults but did not improve decode rate, indicating remaining serial I/O/reclaim or stream scheduling stalls.
- Next step: stop broad VRAM cache sweeping for now. Focus on reducing cold expert page churn with trace-guided expert residency/pinning or asynchronous prefetch/overlap. Any candidate must cold-validate with this runner.
- Full reproducibility record is committed in `.Agent/runs/20260701-vendor-ds4-16gb-token-rate/cold-profile-vram-sweep.json`.

### 2026-07-01T06:43:49Z - Cold-start MoE path tracing

- Purpose: identify the real MoE execution path before attempting another cold-start optimization.
- Added diagnostic-only trace hooks:
  - `GGML_MOE_STREAM_ONE_TRACE_OUT` in `ggml/src/ggml-cuda/moe_stream.cu`, intended to log single-expert CUDA stream calls with tensor name, expert id, cache hit/miss, and stage timings.
  - `GGML_MOE_CPU_CHUNK_TRACE_OUT` in `ggml/src/ggml-cpu/ggml-cpu.c`, logging CPU fallback chunk timings per tensor/expert/thread.
- Cold CUDA one-trace run: `/root/lfz/runs/vendor-ds4-16gb/20260701T064349Z-cold-one-trace-cpu40-vram2/france-cpu40-vram2gb`.
  - Result: `eval_tok_s=1.3`, `prompt_tok_s=0.8`, `ttft_estimate_ms=45086.194029`, `memory_peak_bytes=16000000000`, `memory_max_events=56138`, `pgmajfault=687084`, correctness passed.
  - No `one_trace.csv` was produced. Stderr had `[moe_stream] enabled` but no `first call` or VRAM cache report.
  - Interpretation: DS4 does not enter the effective `ggml_cuda_moe_stream_one` path. CPU-side gating allows MXFP4/F8, but `moe_stream_one` only accepts `GGML_TYPE_IQ3_XXS`, so the vendor DS4 model falls back to CPU expert compute.
- Cold CPU chunk trace run: `/root/lfz/runs/vendor-ds4-16gb/20260701T064950Z-cold-cpu-chunk-trace-cpu40-vram2/france-cpu40-vram2gb`.
  - Result: `eval_tok_s=3.6`, `prompt_tok_s=1.4`, `ttft_estimate_ms=31509.004709`, `memory_peak_bytes=16000000000`, `memory_max_events=19096`, `pgmajfault=312677`, correctness passed.
  - The run is diagnostic only because the trace file hit the configured `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=200000` and may perturb timings. It still confirms the real path and hotspot shape.
  - Trace file: `cpu_chunk_trace.csv`, about 15 MiB and 199982 data rows.
  - First covered rows sum to about `100568 ms` of chunk time across threads. The slowest individual chunks are concentrated in `blk.0.ffn_up_exps.weight` for experts `47` and `75`, with single chunks up to about `1053 ms`; this is consistent with cold page faults/reclaim on first expert touches.
- Failed optimization attempt: briefly allowed `moe_stream_one` to accept `GGML_TYPE_MXFP4` and `GGML_TYPE_F8_E4M3_B128` because the lower `mmvq` layer has kernels for these types.
  - Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T065246Z-cold-mxfp4-stream-cpu40-vram2/france-cpu40-vram2gb`.
  - Stderr confirmed the path was entered: `[moe_stream] first call: ne01=2048 ne00=4096 nb01=2176 src0_bytes=4456448 cne1=2`.
  - Rejected immediately: output degenerated into multilingual/garbled text, `correctness_ok=false`, `ttft_estimate_ms=73760.104238`, and the unit was manually stopped.
  - The code change was reverted. Conclusion: DS4 MXFP4/F8 cannot be enabled by a simple type gate; the single-expert stream path has a layout or kernel calling convention mismatch for these formats.
- Post-revert verification: `/root/lfz/runs/vendor-ds4-16gb/20260701T065951Z-cold-post-revert-verify-cpu40-vram2/france-cpu40-vram2gb`.
  - Result: `eval_tok_s=1.1`, `prompt_tok_s=0.7`, `ttft_estimate_ms=45936.46463`, `memory_peak_bytes=16000000000`, `memory_max_events=82645`, `pgmajfault=980573`, correctness passed.
  - This restores the correct CPU fallback behavior and is not a new SOTA.
- Next bottleneck to attack: build a correct MXFP4/F8 stream path by comparing `moe_stream_one` against the existing CUDA `mmvq` call conventions, or avoid this path and instead reduce CPU fallback cold page faults with trace-guided prefetch/residency.

### 2026-07-01T07:48:00Z - Rejected DS4 gate stream isolation and CPU WILLNEED prefetch

- Purpose: continue the cold-start plan after the previous type-gate-only stream attempt failed. Two diagnostic paths were tested:
  - Add default-off DS4 stream isolation controls: `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1` plus `GGML_MOE_STREAM_ONE_NAME_FILTER=<substring>`.
  - Pass the real `nb01` row stride into `ggml_cuda_moe_stream_mmvq_dev` instead of recomputing stride from `ne00`, to remove a possible padding/layout mismatch.
  - Add default-off CPU fallback prefetch: `GGML_MOE_CPU_WILLNEED=1`, which calls `madvise(MADV_WILLNEED)` for active expert pages before CPU chunk compute.
- Invalid setup runs: `/root/lfz/runs/vendor-ds4-16gb/20260701T074105Z-cold-ds4-stream-filter-gate-cpu40-vram2`, `/root/lfz/runs/vendor-ds4-16gb/20260701T074551Z-cold-ds4-stream-filter-gate-stride-cpu40-vram2`, and `/root/lfz/runs/vendor-ds4-16gb/20260701T075241Z-cold-cpu-willneed-cpu40-vram2` failed at model load because concurrent GLM runs consumed VRAM. These are not counted as model metrics.
- Valid DS4 gate-only stream isolation run after freeing GPU:
  - Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T074800Z-cold-ds4-stream-filter-gate-stride-freegpu-cpu40-vram2/france-cpu40-vram2gb`.
  - Config: `cpu_moe=40`, `vram_cache=2`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`.
  - Result: rejected. `eval_tok_s=2.5`, `prompt_tok_s=0.6`, `ttft_estimate_ms=52405.937441`, `memory_peak_bytes=16000000000`, `memory_max_events=34004`, `pgmajfault=358937`, but `correctness_ok=false`.
  - Answer degenerated into repeated punctuation and did not mention France or Europe: `preferably also as a list of  . ...`.
  - Interpretation: MXFP4 stream is already wrong for `ffn_gate_exps.weight`; the failure is not only an up/down fusion issue. Passing true `nb01` did not fix it.
- Valid CPU `MADV_WILLNEED` prefetch run after freeing GPU:
  - Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T075329Z-cold-cpu-willneed-freegpu-cpu40-vram2/france-cpu40-vram2gb`.
  - Config: `cpu_moe=40`, `vram_cache=2`, `drop_caches_before_case=true`, `GGML_MOE_CPU_WILLNEED=1`.
  - Result: rejected as performance regression. Correctness passed, but `eval_tok_s=1.0`, `prompt_tok_s=0.8`, `ttft_estimate_ms=43646.226705`, `memory_peak_bytes=16000000000`, `memory_max_events=61442`, `pgmajfault=605870`, `workingset_refault_file=19860280`.
  - Compared with cold vram2 profile (`eval_tok_s=1.3`, `ttft_estimate_ms=44034.028848`, `pgmajfault=826309`), `WILLNEED` reduced major faults and slightly improved TTFT, but worsened generation throughput. The likely gap is that prefetch increases reclaim/read-ahead contention and does not overlap enough useful compute under the 16GB cgroup.
- Next step: do not pursue broad `MADV_WILLNEED` as-is. For MXFP4/F8 stream correctness, compare `moe_stream_one` output against CPU for a tiny captured expert/input or add a targeted numeric diff harness before attempting another full model run.

### 2026-07-01T08:03:13Z - Rejected DS4 gate stream with MoE ids kernel

- Purpose: test whether the MXFP4 gate stream failure was caused by using the non-ids single-column MMVQ path. Added a default-off compact-rows helper, `ggml_cuda_moe_stream_mmvq_rows_dev`, that quantizes compact `cne1` rows and calls the existing MoE ids kernel with all expert ids set to zero for the current streamed expert.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T080313Z-cold-ds4-stream-filter-gate-idskernel-cpu40-vram2/france-cpu40-vram2gb`.
- Config: `cpu_moe=40`, `vram_cache=2`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`.
- Result: rejected. `eval_tok_s=2.1`, `prompt_tok_s=0.6`, `ttft_estimate_ms=51806.596358`, `memory_peak_bytes=16000000000`, `memory_max_events=32813`, `pgmajfault=347048`, `workingset_refault_file=7647805`, but `correctness_ok=false`.
- Output again degenerated into the same punctuation/list pattern and did not mention France or Europe.
- Interpretation: the failure is not explained by using the non-ids MMVQ path. The next useful step is a small numeric CPU-vs-GPU diff for one MXFP4 expert/input to identify whether the mismatch is in weight layout, input quantization, output layout, or the CUDA vec-dot itself.

### 2026-07-01T08:14:02Z - Rejected DS4 gate stream CPU-vs-GPU numeric compare

- Purpose: add a default-off numeric compare hook that recomputes CPU `vec_dot` for streamed DS4 gate experts and compares it against the GPU result written by the experimental stream path.
- Code change: `GGML_MOE_STREAM_COMPARE_CPU_OUT=<path>` writes `compare.csv`; `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=<n>` limits the number of streamed experts compared. The hook only runs after `ggml_cuda_moe_stream_one` reports `done=true`, so normal runs are unaffected.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T081402Z-cold-ds4-gate-stream-cpu-compare/france-cpu40-vram2gb`.
- Config: `cpu_moe=40`, `vram_cache=2`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=8`.
- Result: rejected. `eval_tok_s=2.6`, `prompt_tok_s=0.6`, `ttft_estimate_ms=51954.067808`, `memory_peak_bytes=16000000000`, `memory_max_events=33849`, `pgmajfault=348107`, `workingset_refault_file=7746423`, but `correctness_ok=false`.
- Answer: `preferably also as a list of  . ...`; it does not mention France or Europe and is semantically invalid.
- Numeric evidence from `compare.csv`:
  - Matching examples: `blk.0.ffn_gate_exps.weight` expert `222` had `max_abs=9.53674316e-07`; `blk.1.ffn_gate_exps.weight` expert `8` had `max_abs=9.53674316e-07`.
  - Failing examples: expert `35` had CPU `1.64784455` versus GPU `0`; expert `53` had CPU `5.24980307` versus GPU `0`; expert `245` had CPU `9.97797775` versus GPU `-0.41251725`; maximum observed `max_abs=10.390495`.
- Interpretation: the MXFP4 stream bug is not a uniform scatter/indexing failure, because some experts match exactly while others fail badly. The next isolation should correlate mismatches with VRAM cache hits, cache slots, source pointers, and DONTNEED behavior; run with `GGML_MOE_STREAM_DONTNEED=0` and with `GGML_MOE_VRAM_CACHE_GB=0` before changing kernels again.

### 2026-07-01T08:24:21Z - Rejected DS4 gate stream cache/DONTNEED isolation

- Purpose: determine whether the CPU-vs-GPU mismatches came from `GGML_MOE_STREAM_DONTNEED=1` discarding source pages before CPU comparison, or from the VRAM cache copy/slot path.
- Invalid setup run: `/root/lfz/runs/vendor-ds4-16gb/20260701T082240Z-cold-ds4-gate-stream-cpu-compare-dontneed0/france-cpu40-vram2gb` failed to load because a concurrent non-vendor process consumed VRAM. It is not counted as a model metric.
- Valid `DONTNEED=0` run: `/root/lfz/runs/vendor-ds4-16gb/20260701T082421Z-cold-ds4-gate-stream-cpu-compare-dontneed0-freegpu/france-cpu40-vram2gb`.
  - Result: rejected. `eval_tok_s=2.5`, `prompt_tok_s=0.6`, `ttft_estimate_ms=50723.998661`, `memory_peak_bytes=16000000000`, `memory_max_events=39815`, `pgmajfault=363729`, `workingset_refault_file=6540359`, `correctness_ok=false`.
  - `compare.csv` first eight rows were identical to the prior default `DONTNEED=1` run, including expert `35` CPU `1.64784455` versus GPU `0`, expert `222` matching at `9.53674316e-07`, and expert `245` CPU `9.97797775` versus GPU `-0.41251725`.
- Valid `VRAM_CACHE_GB=0` run: `/root/lfz/runs/vendor-ds4-16gb/20260701T082759Z-cold-ds4-gate-stream-cpu-compare-vram0/france-cpu40-vram0gb`.
  - Result: rejected. `eval_tok_s=1.8`, `prompt_tok_s=0.6`, `ttft_estimate_ms=54440.563489`, `memory_peak_bytes=16000000000`, `memory_max_events=36650`, `pgmajfault=365648`, `workingset_refault_file=8509598`, `correctness_ok=false`.
  - `compare.csv` first eight rows again matched the prior runs exactly.
- Interpretation: the DS4 MXFP4 gate stream numeric mismatch is not caused by source page discard or by the VRAM cache hit/miss/slot path. The next useful diagnostic is a lower-level MXFP4 block/input trace for one matching expert (`222` or `blk.1` expert `8`) and one failing expert (`35`, `53`, or `245`) to compare CPU `vec_dot` inputs against what the CUDA MMVQ kernel reads.

### 2026-07-01T08:37:36Z - Rejected DS4 gate stream non-ids kernel isolation

- Purpose: determine whether the compact-rows dedicated MoE ids kernel was responsible for the DS4 MXFP4 stream mismatch.
- Code change: added default-off `GGML_MOE_STREAM_ONE_DS4_NONIDS=1`, which keeps experimental DS4 streaming enabled but routes DS4 rows through the existing per-row single-column MMVQ helper instead of `ggml_cuda_moe_stream_mmvq_rows_dev`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T083736Z-cold-ds4-gate-stream-cpu-compare-nonids/france-cpu40-vram2gb`.
- Config: `cpu_moe=40`, `vram_cache=2`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_DS4_NONIDS=1`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=8`.
- Result: rejected. `eval_tok_s=2.3`, `prompt_tok_s=0.6`, `ttft_estimate_ms=53691.425728`, `memory_peak_bytes=16000000000`, `memory_max_events=33434`, `pgmajfault=344445`, `workingset_refault_file=7776321`, `correctness_ok=false`.
- `compare.csv` first eight rows were again identical to the ids-kernel, `DONTNEED=0`, and `VRAM_CACHE_GB=0` runs.
- Interpretation: the dedicated MoE ids kernel is not the source of the numeric mismatch. The remaining likely layer is the MXFP4/q8_1 CUDA vec-dot or quantized activation layout itself. Next diagnostic should compare per-block partial sums for one matching and one failing expert.

### 2026-07-01T08:53:25Z - Rejected DS4 gate stream MXFP4 block compare

- Purpose: localize the remaining DS4 MXFP4 stream mismatch below the full expert dot product by tracing per-block partial sums for the same row/column where CPU-vs-GPU compare reports the maximum error.
- Code change: added default-off `GGML_MOE_STREAM_COMPARE_BLOCK_OUT=<path>`. It writes `block_compare.csv` when `GGML_MOE_STREAM_COMPARE_CPU_OUT` is also active. The trace records per-32-element MXFP4 block partial sums using the CPU fallback `wdata` q8_0 row and a post-src1 re-quantization check.
- First diagnostic run: `/root/lfz/runs/vendor-ds4-16gb/20260701T084754Z-cold-ds4-gate-stream-block-compare/france-cpu40-vram2gb`.
  - Result was rejected and correctness failed, as expected. The first implementation used post-scatter `src1->data` only, which made some rows appear to match the wrong GPU value and showed the trace needed to use CPU fallback `wdata`.
- Corrected diagnostic run: `/root/lfz/runs/vendor-ds4-16gb/20260701T085325Z-cold-ds4-gate-stream-block-compare-wdata/france-cpu40-vram2gb`.
  - Result: rejected. `eval_tok_s=2.5`, `prompt_tok_s=0.6`, `ttft_estimate_ms=54019.149062`, `memory_peak_bytes=16000000000`, `memory_max_events=32703`, `pgmajfault=344653`, `workingset_refault_file=7680943`, `correctness_ok=false`.
  - `compare.csv` remained identical to prior gate-stream failures: examples include expert `35` CPU `1.64784455` versus GPU `0`, expert `245` CPU `9.97797775` versus GPU `-0.41251725`, and expert `222` matching to about `1e-6`.
  - `block_compare.csv` showed `cpu_wdata_total` reproduces the CPU compare value for all eight rows: seq `0` total `1.64784458`, seq `1` total `5.24980289`, seq `4` total `9.97797799`, seq `7` total `0.415755354`.
  - `post_src1_total` matched `cpu_wdata_total` for all eight rows, so the host-side q8_0/q8_1-style activation quantization and row selection are not the source of the GPU mismatch.
- Interpretation: the remaining bug is in the CUDA MXFP4 MMVQ vec-dot/kernel path or its low-level launch assumptions. A concrete next test is to validate the CUDA-only `VDR_MXFP4_Q8_1_MMVQ=4` choice against SYCL's `VDR_MXFP4_Q8_1_MMVQ=2`; higher-level cache, DONTNEED, ids-kernel, and activation quantization have now been ruled out.

### 2026-07-01T09:05:22Z - Rejected CUDA MXFP4 MMVQ VDR=2 test

- Purpose: test the concrete low-level hypothesis from the block trace: CUDA defines `VDR_MXFP4_Q8_1_MMVQ=4` while SYCL defines `2`; if CUDA's one-thread-per-MXFP4-block schedule were wrong, changing VDR to `2` should improve CPU-vs-GPU agreement.
- Code change tested: changed only `ggml/src/ggml-cuda/vecdotq.cuh` from `#define VDR_MXFP4_Q8_1_MMVQ 4` to `2`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T090522Z-cold-ds4-gate-stream-vdr2-compare/france-cpu40-vram2gb`.
- Config: `cpu_moe=40`, `vram_cache=2`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=8`.
- Result: rejected. `eval_tok_s=2.7`, `prompt_tok_s=0.6`, `ttft_estimate_ms=52739.864209`, `memory_peak_bytes=16000000000`, `memory_max_events=32188`, `pgmajfault=341250`, `workingset_refault_file=6931721`, `correctness_ok=false`.
- Answer degenerated into punctuation/quote repetition and did not mention France or Europe.
- Numeric result: core compare rows did not improve. Expert `35` still had CPU `1.64784455` versus GPU `0`; expert `245` still had CPU `9.97797775` versus GPU about `-0.412517`; expert `97` still had CPU `0.415755332` versus GPU `-1.87617469`.
- Action: reverted the VDR code change; it is not accepted. Next step is a CUDA-side debug kernel that writes actual per-block partial sums from the GPU for one failing row and one matching row.

### 2026-07-01T09:36:40Z - Corrected DS4 gate stream src1 row mapping; rejected for token rate

- Purpose: fix the DS4 MXFP4 gate stream numeric mismatch after q8 debug showed failing `row_id > 0` activations were staged as zero or wrong data on GPU while `row_id=0` matched CPU.
- Root cause: CPU `mul_mat_id` uses `i11 = id % ne11` when selecting the source activation row, but `ggml_cuda_moe_stream_one` staged F32 `src1` rows with raw `rows[k].i1`. This made expert rows with `id >= ne11` read the wrong activation row before CUDA q8 quantization.
- Code change: pass `ne11` as `src1_ne1` into `ggml_cuda_moe_stream_one` and stage with `rows[k].i1 % src1_ne1`; destination scatter still uses the original row id.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T093640Z-cold-ds4-gate-stream-src1-rowmod-fix/france-cpu40-vram2gb`.
- Config: `cpu_moe=40`, `vram_cache=2`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=8`.
- Result: correctness passed and CPU-vs-GPU compare is fixed, but token rate regressed. Metrics: `eval_tok_s=0.9`, `prompt_tok_s=0.6`, `ttft_estimate_ms=52299.175473`, `memory_peak_bytes=16000000000`, `memory_max_events=67170`, `memory_file_bytes=14877372416`, `pgmajfault=600997`, `workingset_refault_file=25222398`, `ram_ok=true`.
- Answer: `France is a Western European country known for its rich history, vibrant culture, and significant global influence...`; the response is semantically correct and coherent.
- Numeric evidence: first eight `compare.csv` rows now match CPU to about `1e-7` to `1e-6`. Examples: expert `35` CPU `1.4652648` vs GPU `1.46526492`; expert `245` CPU `10.0252676` vs GPU `10.0252686`.
- Acceptance: rejected for token rate because cold baseline vram2 remains `eval_tok_s=1.3` with TTFT `44034.028848 ms`. This run's TTFT increase is about `18.8%`, within the 20% gate, but generation throughput is lower.
- Action: keep and push as a correctness/diagnostic foundation only. Next optimization must profile the corrected stream path and recover the lost time before it can become an accepted SOTA.

### 2026-07-01T09:50:31Z - Accepted cold SOTA with corrected DS4 gate stream and 8GB VRAM cache

- Purpose: after fixing DS4 gate stream correctness, test whether increasing VRAM cache reduces corrected stream expert reloads enough to beat the cold baseline under the 16GB cgroup.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T095031Z-cold-ds4-gate-stream-src1-rowmod-vram8-trace/france-cpu40-vram8gb`.
- Config: `cpu_moe=40`, `vram_cache=8`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`.
- Result: accepted as current cold SOTA. `eval_tok_s=1.4`, `prompt_tok_s=0.6`, `ttft_estimate_ms=47573.774683`, `memory_peak_bytes=16000000000`, `memory_max_events=55388`, `memory_file_bytes=14967328768`, `pgmajfault=574410`, `workingset_refault_file=15540325`, `ram_ok=true`, `correctness_ok=true`.
- TTFT gate: previous cold vram2 baseline TTFT was `44034.028848 ms`; this is `+8.0%`, below the 20% limit.
- Answer: `France is a Western European country known for its rich history, vibrant culture, and significant global influence...`; semantically correct and coherent.
- Trace interpretation: vram2 corrected stream had `16642` inserts and summed `src0_ms=73379 ms`; vram8 has `7110` inserts and `src0_ms=36392 ms`. CUDA activation quantization, MXFP4 MMVQ, D2H, sync, and scatter together remain around `1.1 s`, so the bottleneck remains cold expert weight staging/page faults rather than GPU compute.
- Action: commit and push immediately. Next step is to probe higher VRAM cache sizes under the same 16GB RAM gate to find the cache knee, then stop when VRAM OOM, TTFT regression, or no token-rate gain appears.

### 2026-07-01T09:55:26Z - Accepted cold SOTA with corrected DS4 gate stream and 12GB VRAM cache

- Purpose: continue the VRAM cache knee sweep after vram8 improved cold token rate by reducing corrected stream expert reloads.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T095526Z-cold-ds4-gate-stream-src1-rowmod-vram12-trace/france-cpu40-vram12gb`.
- Config: `cpu_moe=40`, `vram_cache=12`, `drop_caches_before_case=true`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`.
- Result: accepted as current cold SOTA. `eval_tok_s=2.6`, `prompt_tok_s=0.8`, `ttft_estimate_ms=40836.003233`, `memory_peak_bytes=16000000000`, `memory_max_events=23908`, `memory_file_bytes=14958911488`, `pgmajfault=309216`, `workingset_refault_file=3371495`, `ram_ok=true`, `correctness_ok=true`.
- TTFT gate: original cold vram2 baseline TTFT was `44034.028848 ms`; this run is `-7.3%`, so it passes the TTFT limit.
- Answer: same coherent France paragraph as the vram8 run, semantically correct.
- Trace interpretation: vram8 had `7110` inserts and `src0_ms=36392 ms`; vram12 has `5129` inserts and `src0_ms=23564 ms`. The gain is still from fewer expert staging misses/page faults. CUDA quantize/MMVQ/D2H/scatter remain small.
- Action: commit and push immediately. Next step is to probe one higher VRAM cache size to identify whether there is still useful cache capacity before OOM or diminishing returns.

### 2026-07-01T09:59:42Z - Rejected vram14 cache cliff

- Purpose: test the next VRAM cache size after the accepted vram12 SOTA.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T095942Z-cold-ds4-gate-stream-src1-rowmod-vram14-trace/france-cpu40-vram14gb`.
- Config: `cpu_moe=40`, `vram_cache=14`, `drop_caches_before_case=true`, corrected DS4 gate stream enabled with trace.
- Result: rejected. Correctness passed and RAM stayed inside the 16GB cgroup, but `eval_tok_s=0.8`, `prompt_tok_s=0.6`, `ttft_estimate_ms=52889.140515`, `memory_max_events=72712`, `pgmajfault=606508`, `workingset_refault_file=26778917`.
- Trace evidence: `cache_hits=0`, `cache_inserts=0`, `src0_ms=91071 ms`, `dontneed_ms=6640 ms`. The requested 14GB VRAM cache appears not to have inserted any experts, likely due practical VRAM fit after model allocation.
- Interpretation: vram14 crosses a cache allocation cliff and must not be used. Probe vram13 next to locate whether the usable upper bound is exactly vram12 or if one more GB fits.

### 2026-07-01T10:05:46Z - Rejected vram13 CUDA OOM

- Purpose: locate the practical VRAM cache upper bound between accepted vram12 and rejected vram14.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T100546Z-cold-ds4-gate-stream-src1-rowmod-vram13-trace/france-cpu40-vram13gb`.
- Config: `cpu_moe=40`, `vram_cache=13`, `drop_caches_before_case=true`, corrected DS4 gate stream enabled with trace.
- Result: rejected before generation. Host RAM stayed inside the 16GB cgroup (`memory_peak_bytes=16000000000`, `memory_max_events=604`), but process exited `134` after CUDA OOM.
- Key stderr lines: `[moe_stream] VRAM cache: 13.0 GiB, 3132 slots (4.25 MiB each)`, then `ggml_cuda_compute_forward: GET_ROWS failed`, `CUDA error: out of memory`.
- Interpretation: vram13 is beyond practical VRAM fit with current model/settings. Current usable upper bound is vram12; further work should optimize remaining misses or free other VRAM rather than increasing the cache directly.

### 2026-07-01T10:08:28Z - Rejected 12800MiB stream cache

- Purpose: test a finer-grained cache size between accepted 12GiB and vram13 OOM using `GGML_MOE_STREAM_ONE_CACHE_MIB=12800`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T100828Z-cold-ds4-gate-stream-src1-rowmod-cache12800-trace/france-cpu40-vram12gb`.
- Result: rejected. Correctness passed and RAM stayed inside the 16GB cgroup, but `eval_tok_s=1.6`, `ttft_estimate_ms=46832.258829`, `memory_max_events=50973`, `pgmajfault=547758`, `workingset_refault_file=12232851`.
- Trace: `cache_hits=29723`, `cache_inserts=5030`, `src0_ms=26107 ms`, `total_ms=28434 ms`. This is only slightly fewer inserts than vram12 (`5129`) but has worse staging time, TTFT, and token rate.
- Interpretation: vram12 remains the current best cache point. Extra cache near the VRAM limit introduces pressure/variability that outweighs the small miss reduction.

### 2026-07-01T16:31:34Z - Warm/no-drop page-cache upper-bound diagnostic

- Purpose: determine whether the current clean `cpu40/cache13568MiB` gap is dominated by cold page/refault state or CPU fallback compute.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T163134Z-20260702_warm-page-bound-diagnostic-cache13568/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, no `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16`.
- Result: diagnostic-only, not promotable. `eval_tok_s=2.8`, `prompt_tok_s=0.8`, `first_answer_ms=40871.512337`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15064985600`, `pgmajfault=320963`, `correctness_ok=true`.
- Trace: same cache shape as cold (`hits=29855`, `misses=4898`) but `trace_span_ms=67656.44`, `src0_ms_sum=22225.714`, `total_ms_sum=24412.385`.
- Interpretation: warm global page-cache state can recover the old high-rate shape, so the gap is primarily cold page/refault/reclaim state. Not accepted because no `drop_caches` was used.

### 2026-07-01T16:36:01Z - Same-cgroup warmup then test diagnostic

- Purpose: test whether the warm/no-drop benefit can be prepared by a recorded warmup inside the same strict 16GB cgroup after cold `drop_caches`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T163601Z-20260702-same-cgroup-warmup-then-test-cache13568/france-cpu40-cache13568-same-cgroup-warmup`.
- Config: one `systemd-run` service with `MemoryMax=16000000000`, host `sync; echo 3 > /proc/sys/vm/drop_caches` before service, then warmup France and measured France in the same cgroup.
- Result: diagnostic-only, rejected as an optimization path. The service completed with `memory_peak_bytes=16000000000`, `memory_file_bytes=14977691648`, `ram_ok=true`.
- Warmup phase: `eval_tok_s=1.6`, `first_answer_ms=46899.346218`, `pgmajfault≈618456`, `trace_span_ms=109704.8`, correctness passed.
- Test phase: `eval_tok_s=1.6`, `first_answer_ms=54576.825008`, phase `pgmajfault≈611371`, cumulative cgroup `pgmajfault=1229030`, `trace_span_ms=112260.763`, correctness passed.
- Interpretation: a full warmup inside the same 16GB cgroup does not reproduce the global warm `2.8 tok/s` state; cgroup file-cache pressure/refault churn keeps the second run cold-like. Next work should reduce the cold working set or change page/refault scheduling in the first invocation, not rely on a full warmup.

### 2026-07-01T16:48:38Z - Rejected stream WILLNEED lookahead1

- Purpose: test whether a narrower gate-stream `MADV_WILLNEED` lookahead can capture some warm-page benefit without the reclaim pressure seen by `lookahead4`.
- Code change tested: default-off `GGML_MOE_STREAM_WILLNEED_LOOKAHEAD=<N>` in `ggml_compute_forward_mul_mat_id`; for `ffn_gate_exps`, thread 0 advises the next active expert page before streaming the current expert.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T164838Z-20260702_stream-willneed-lookahead1-cache13568/france-cpu40-vram0gb`.
- Config: `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_WILLNEED_LOOKAHEAD=1`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- Result: rejected. `eval_tok_s=1.4`, `prompt_tok_s=0.6`, `first_answer_ms=48253.97287`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978121728`, `pgmajfault=616083`, `workingset_refault_file=16346498`, `ram_ok=true`, `correctness_ok=true`.
- Trace: cache shape unchanged (`hits=29855`, `misses=4898`), `src0_ms_sum=21259.698`, `total_ms_sum=23660.217`, but `trace_span_ms=122386.105`.
- Interpretation: lookahead1 reduced traced `src0_ms` but made the end-to-end run slower, confirming the cost moved into untraced reclaim/refault gaps. Stop synchronous stream `MADV_WILLNEED` lookahead.
- Action: source reverted and clean rebuild restored `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push.

### 2026-07-01T16:55:53Z - Rejected VRAM cache hash lookup

- Purpose: test whether the existing linear VRAM cache lookup is a meaningful fixed overhead at `3192` slots and about `34.7k` stream calls.
- Code change tested: maintained the existing `vram_cache.ht` open-addressed hash table on insert/evict and replaced linear lookup with hash lookup. Cache policy and slot selection were unchanged.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T165553Z-20260702_vram-cache-hash-lookup/france-cpu40-vram0gb`.
- Config: `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=46636.224639`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978424832`, `pgmajfault=608857`, `workingset_refault_file=12035462`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `hits=29855`, `misses=4898`, `src0_ms_sum=25292.321`, `total_ms_sum=27569.415`, `trace_span_ms=108718.113`.
- Interpretation: hash lookup preserves behavior but does not move end-to-end throughput. Linear cache lookup is not a material bottleneck relative to cold page/refault and CPU fallback gaps.
- Action: source reverted and clean rebuild completed. `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`; no commit/push.

### 2026-07-01T17:02:58Z - Rejected cache13568 no-trace cold check

- Purpose: verify whether `one_trace.csv` logging itself is depressing the `13568MiB` cold boundary result.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T170258Z-20260702_cache13568-no-trace-cold-check/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, cold `drop_caches`, 16GB cgroup, France prompt, `-c 256 -b 16 -ub 16`.
- Result: diagnostic rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=48779.7506`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964670464`, `pgmajfault=619370`, `workingset_refault_file=11898996`, `ram_ok=true`, `correctness_ok=true`.
- Cache stderr: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`.
- Interpretation: disabling trace does not improve throughput; trace logging is not the main cold bottleneck. Continue with page/refault working-set or CPU fallback structural work.

### 2026-07-01T17:09:59Z - Rejected cpu39 cache10496MiB boundary

- Purpose: test whether `cpu_moe=39` plus a fractional `10496 MiB` stream cache fits below the `vram11` allocation cliff and recovers enough cache misses over `vram10` to improve token rate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T170959Z-20260702_cpu39-cache10496mib-boundary/france-cpu39-vram0gb`.
- Config: clean source, `cpu_moe=39`, `GGML_MOE_STREAM_ONE_CACHE_MIB=10496`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=49989.869425`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14935244800`, `pgmajfault=589874`, `workingset_refault_file=12469552`, `ram_ok=true`, `correctness_ok=true`.
- Cache/trace: `10.2 GiB`, `2469 slots`, `hits=28344`, `misses=5531`, `hit_rate=83.7%`; trace `rows=33875`, `trace_span_ms=109715.06`, `src0_ms_sum=28655.016`, `total_ms_sum=31059.553`.
- Interpretation: the fractional cache improved the `cpu39/vram10` miss count but only tied `1.6 tok/s` and had worse TTFT than the `cpu40/cache13568` boundary. Stop the `cpu39` fractional branch unless a separate change frees enough VRAM to keep the larger stream cache class.
- Action: no source change, no commit/push. This is not an accepted SOTA.

### 2026-07-01T17:19:31Z - Rejected cache13824 c224 fit

- Purpose: test whether lowering context from `256` to `224` frees enough VRAM for the previously OOMing `13824 MiB` stream cache.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T171931Z-20260702_cache13824-c224-fit/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 224 -b 16 -ub 16`.
- Result: rejected before generation. Cache allocated as `13.5 GiB`, `3252 slots`, then CUDA aborted with `GET_ROWS failed`, `CUDA error: out of memory` at the fifth streamed gate expert. No `eval_tok_s` or TTFT.
- Memory evidence: host RAM gate held (`memory_peak_bytes=16000000000`, `oom=0`, `oom_kill=0`); GPU free dropped to about `14 MiB`, so the failure is VRAM headroom.
- Interpretation: context reduction to `224` is not enough to make `13824 MiB` viable. Do not continue larger-cache probes unless a source change frees substantial VRAM; a `13696 MiB` probe has too small an upper bound to prioritize.
- Follow-up simulation: for `cpu39/cache10496`, skipping frequency-1 gate experts is predicted to reduce misses `5531 -> 4819`, about `3.7s` direct traced upper bound. This is a better next source probe than another fractional cache step.

### 2026-07-01T17:25:01Z - Rejected cpu39 cache10496 profile admission

- Purpose: combine the `cpu39/cache10496` GPU-layer tradeoff with profile-guided gate cache admission to recover the misses caused by the smaller stream cache.
- Code change tested: default-off `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<tsv>` in `ggml/src/ggml-cuda/moe_stream.cu`. On cache miss, experts absent from the profile are streamed through the existing staging path but not inserted into the VRAM cache.
- Profile: `/root/lfz/runs/vendor-ds4-16gb/cpu39_gate_freq_ge2.tsv`, generated from the prior `cpu39/cache10496` trace with frequency `>=2`; `3040` entries; sha256 `9c2e8253215457ba6889cfb77eb14cd654b62d2fdffb7d8c8013adc967af1e83`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T172501Z-20260702_cpu39-cache10496-profile-admission/france-cpu39-vram0gb`.
- Config: `cpu_moe=39`, `GGML_MOE_STREAM_ONE_CACHE_MIB=10496`, profile admission enabled, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=50852.470994`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14977093632`, `pgmajfault=583564`, `workingset_refault_file=11409126`, `ram_ok=true`, `correctness_ok=true`.
- Cache/trace: profile loaded `3040` entries; `hits=29056`, `misses=4819`, `cache_inserts=3439`, `src0_ms_sum=24427.7`, `total_ms_sum=26693.099`.
- Interpretation: the mechanism worked and matched the offline miss prediction, but the end-to-end token rate still only tied `1.6 tok/s` and TTFT remained worse than the `cpu40/cache13568` boundary. Gate cache pollution is not enough to produce a new SOTA.
- Action: source reverted and clean rebuild completed. Clean hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so.0.10.0=baa7460b7bd1fef1aec3ad037b788d81397ed3eab230983466bfce72f22458ae`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push because this is not an accepted SOTA.

### 2026-07-01T17:31:46Z - Rejected CUDA graphs enabled cache13568

- Purpose: test whether enabling CUDA graphs can reduce non-stream CUDA launch/scheduler overhead outside the per-expert trace.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T173146Z-20260702_cuda-graphs-enabled-cache13568/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_CUDA_DISABLE_GRAPHS=0`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=46589.645517`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14958915584`, `pgmajfault=608838`, `workingset_refault_file=12188046`, `ram_ok=true`, `correctness_ok=true`.
- Cache/trace: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `trace_span_ms=108365.71`, `src0_ms_sum=25238.358`, `total_ms_sum=27545.916`.
- Interpretation: CUDA graphs did not move token rate, TTFT, cache behavior, or trace span. The remaining gap is still cold page/refault and CPU fallback behavior, not graph launch overhead.
- Action: no source change, no commit/push.

### 2026-07-01T17:37:41Z - Rejected no-defer experts cache13568

- Purpose: test whether disabling `--defer-experts` can avoid the repeated expert mmap refault pattern seen in cold runs.
- Runner note: temporarily added `--no-defer-experts` support to `.Agent/run-tools/strict_ds4_runner.py` to reuse the strict cgroup runner, then reverted the runner after the experiment.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T173741Z-20260702_no-defer-experts-cache13568/france-cpu40-vram0gb`.
- Config: clean source, no `--defer-experts` in `exact_command.txt`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16`.
- Result: failed/rejected timeout. The run was killed after about `304` resource samples with no complete answer, no `eval_tok_s`, and no TTFT summary.
- Partial evidence: cgroup stayed at the 16GB ceiling (`memory_peak_bytes_from_samples=16000000000`), GPU free was about `234 MiB`, partial trace had `rows=6656`, `cache_hits=0`, `cache_inserts=6656`, `trace_span_ms=118575.8`, `src0_ms_sum=36159.303`, `total_ms_sum=37664.522`.
- Interpretation: no-defer expert mmap residency is much worse under the strict 16GB host RAM gate. It fails TTFT/correctness by timeout and does not reach a useful cache-reuse phase. Keep `--defer-experts`.
- Action: no model source change; temporary runner patch reverted. No commit/push.

### 2026-07-01T17:48:15Z - Rejected block read-ahead 4096 cache13568

- Purpose: test whether increasing block device read-ahead helps cold expert mmap page-in for 4.25MiB contiguous expert weights.
- System evidence: model file is on `/dev/vda1` mounted at `/`; original `/sys/block/vda/queue/read_ahead_kb=128`. The run temporarily set it to `4096` and restored it to `128` after termination.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T174815Z-20260702_block-readahead-4096-cache13568/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16`, block read-ahead `4096KB`.
- Result: failed/rejected timeout/regression. The run was killed after about `317` resource samples with no complete answer, no `eval_tok_s`, and no TTFT summary.
- Partial evidence: cgroup stayed at the 16GB ceiling (`memory_peak_bytes_from_samples=16000000000`), GPU free was about `238 MiB`, partial trace had `rows=12673`, `cache_hits=9354`, `cache_inserts=3319`, `trace_span_ms=319194.37`, `src0_ms_sum=47524.562`, `total_ms_sum=48716.777`.
- Interpretation: larger block read-ahead pulls too much data into the 16GB file-cache budget and makes reclaim/refault behavior much worse. Keep `/sys/block/vda/queue/read_ahead_kb=128`.
- Action: restored read-ahead to `128`; no source change, no commit/push.

### 2026-07-01T17:58:20Z - Rejected block read-ahead 512 cache13568

- Purpose: test whether a smaller block read-ahead increase can reduce expert miss latency without the severe reclaim regression seen at `4096KB`.
- System evidence: model file is on `/dev/vda1`; original `/sys/block/vda/queue/read_ahead_kb=128`, temporarily changed to `512`, restored/final value `128`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T175820Z-20260702_block-readahead-512-cache13568/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16`, block read-ahead `512KB`.
- Result: rejected. `eval_tok_s=1.4`, `prompt_tok_s=0.5`, `first_answer_ms=52169.410561`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14980349952`, `pgmajfault=472154`, `workingset_refault_file=17012380`, `ram_ok=true`, `correctness_ok=true`.
- Cache/trace: same cache shape as baseline (`hits=29855`, `misses=4898`); `trace_span_ms=130299.146`, `src0_ms_sum=23774.492`, `total_ms_sum=26043.127`.
- Interpretation: `512KB` lowers traced `src0_ms` but worsens trace span, TTFT, and token rate. The saved read time is more than offset by cgroup reclaim/refault gaps. Keep block read-ahead at `128KB`.
- Action: restored read-ahead to `128`; no source change, no commit/push.

### 2026-07-01T18:05:41Z - Rejected stream MADV_COLD after use

- Purpose: test whether replacing post-use `MADV_DONTNEED` with `MADV_COLD` can reduce immediate refault churn without keeping all source pages hot.
- Code change tested: default-off `GGML_MOE_STREAM_DONTNEED_COLD=1` in `ggml/src/ggml-cuda/moe_stream.cu`; when enabled and `MADV_COLD` is defined, `moe_stream_dontneed_source_pages()` uses `MADV_COLD` instead of `MADV_DONTNEED`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T180541Z-20260702_stream-madv-cold-after-use/france-cpu40-vram0gb`.
- Config: `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_DONTNEED_COLD=1`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=47010.926377`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14957977600`, `pgmajfault=585966`, `workingset_refault_file=11021118`, `ram_ok=true`, `correctness_ok=true`.
- Cache/trace: `hits=29855`, `misses=4898`; `trace_span_ms=110634.138`, `src0_ms_sum=25664.264`, `dontneed_ms_sum=2146.005`, `total_ms_sum=28914.328`.
- Interpretation: `MADV_COLD` did not improve end-to-end throughput. It may reduce some refault counters, but traced stream cost and trace span remain cold-baseline shaped. Not a SOTA.
- Action: source reverted and clean rebuild completed. Clean hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so.0.10.0=dfa48b3ce1be716e7e82278a3671406096894d06c25068d0e12f6d5e36c4e75b`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push.

### 2026-07-01T18:16:17Z - Rejected up/down no-cache stream with gate cache

- Purpose: test whether all-FFN stream regressed mainly because up/down experts polluted the gate VRAM cache. Gate experts stayed cacheable; `ffn_up_exps` and `ffn_down_exps` used CUDA stream compute but skipped VRAM cache lookup/insert.
- Code change tested: default-off `GGML_MOE_STREAM_NOCACHE_NAME_FILTER=<csv substrings>` in `ggml/src/ggml-cuda/moe_stream.cu`, plus CSV parsing for `GGML_MOE_STREAM_ONE_NAME_FILTER`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T181617Z-20260702_stream-updown-nocache-gate-cache/france-cpu40-vram0gb`.
- Config: `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_`, `GGML_MOE_STREAM_NOCACHE_NAME_FILTER=ffn_up_exps,ffn_down_exps`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 stream trace, `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: rejected/regression. `eval_tok_s=0.9`, `prompt_tok_s=0.5`, `first_answer_ms=55901.687693`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970798080`, `pgmajfault=28372`, `workingset_refault_file=15515849`, `ram_ok=true`, `correctness_ok=true`.
- Correctness: France answer is coherent and semantically correct.
- Trace: `rows=110697`, `cache_hits=32089`, `cache_inserts=4810`, `trace_span_ms=196932.945`, `src0_ms_sum=170626.622`, `total_ms_sum=188661.491`. Tensor split: gate `rows=36899 hits=32089 inserts=4810 src0_ms=25869.982`; up `rows=36899 hits=0 inserts=0 src0_ms=72573.873`; down `rows=36899 hits=0 inserts=0 src0_ms=72182.767`.
- Interpretation: the no-cache mechanism worked and avoided gate cache pollution, but broad up/down stream still pays cold H2D/page-read cost on every up/down expert. That adds about `145s` of up/down `src0_ms`, so this path is worse than CPU fallback under the 16GB cold-start gate.
- Action: source reverted and clean rebuild completed. Clean hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so.0.10.0=dff7f183052999b636b57a637ef2e1bea7a57016f4b40d90601a77b578997cb3`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push because this is not an accepted SOTA.

### 2026-07-01T18:30:53Z - Rejected op-wide gate WILLNEED cache13568

- Purpose: test whether pre-advising the active gate experts for each MoE op can reduce the clean cold `1.6 tok/s` refault gap without relying on external warm page cache.
- Code change tested: default-off `GGML_MOE_STREAM_OP_WILLNEED=1` in `ggml/src/ggml-cpu/ggml-cpu.c`, with `GGML_MOE_STREAM_OP_WILLNEED_NAME_FILTER=ffn_gate_exps` so only gate tensors are advised before the CUDA stream loop.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T183053Z-20260702_stream-opwide-willneed-gate-cache13568/france-cpu40-vram0gb`.
- Config: `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, op-wide WILLNEED enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: rejected/regression. `eval_tok_s=1.4`, `prompt_tok_s=0.7`, `first_answer_ms=45919.682075`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978084864`, `pgmajfault=607397`, `workingset_refault_file=16767150`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=120927.803`, `src0_ms_sum=19962.937`, `dontneed_ms_sum=1153.678`, `total_ms_sum=22237.299`.
- Interpretation: op-wide WILLNEED reduced directly traced gate `src0_ms` by about `5.3s` versus the clean `cache13568` control, but it increased untraced reclaim/refault pressure and widened trace span. The end-to-end token rate regressed to `1.4 tok/s`, so prefetching more page cache inside the 16GB cgroup is not enough.
- Action: source reverted and clean rebuild completed. Clean hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=dff7f183052999b636b57a637ef2e1bea7a57016f4b40d90601a77b578997cb3`. No commit/push because this is not an accepted SOTA.

### 2026-07-01T18:43:29Z - Rejected CPU fallback RAM tier top64

- Purpose: test whether copying hot `ffn_up_exps` and `ffn_down_exps` into anonymous RAM inside the same 16GB cgroup can reduce CPU fallback mmap refaults without streaming up/down through GPU.
- Code change tested: default-off `GGML_MOE_CPU_RAM_TIER_PROFILE=<tsv>` and `GGML_MOE_CPU_RAM_TIER_MAX_MIB=<MiB>` in `ggml/src/ggml-cpu/ggml-cpu.c`. Thread 0 copies active profiled experts into anonymous memory before CPU fallback chunks; workers use the anonymous pointer for loaded experts.
- Profile: `/root/lfz/runs/vendor-ds4-16gb/20260701T140952Z-20260701T_selective-hot-updown64-stream/hot_updown64_allowlist.tsv`, `128` entries, sha256 `549c62e4d1547b42d84b97474353a145292aefa88a2642c757cdd51998d18e7c`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T184329Z-20260702_cpu-fallback-ramtier-updown64/france-cpu40-vram0gb`.
- Config: `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, RAM tier top64 with `GGML_MOE_CPU_RAM_TIER_MAX_MIB=640`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=48077.79944`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14384857088`, `pgmajfault=583710`, `workingset_refault_file=11987917`, `ram_ok=true`, `correctness_ok=true`.
- Trace/RAM tier: RAM tier loaded `544.0 MiB`; gate trace `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=108707.099`, `src0_ms_sum=24885.488`, `total_ms_sum=27204.576`.
- Interpretation: top64 RAM tier worked mechanically and stayed inside the 16GB cgroup, but it only tied the clean `1.6 tok/s` line. Coverage is too small or first-use memcpy offsets the fallback benefit.

### 2026-07-01T18:47:38Z - Rejected CPU fallback RAM tier top128

- Purpose: increase RAM-tier coverage to test the useful upper bound of the hot up/down anonymous-copy approach.
- Profile: `/root/lfz/runs/vendor-ds4-16gb/20260701T141433Z-20260701T_selective-hot-updown128-stream/hot_updown128_allowlist.tsv`, `256` entries, sha256 `ab900b0e12d998d28f550d376c2c7333a4de82efed0ca520c1fc12972107c167`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T184738Z-20260702_cpu-fallback-ramtier-updown128/france-cpu40-vram0gb`.
- Config: same source patch, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, RAM tier top128 with `GGML_MOE_CPU_RAM_TIER_MAX_MIB=1200`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=47184.560027`, `memory_peak_bytes=16000000000`, `memory_file_bytes=13819363328`, `pgmajfault=565805`, `workingset_refault_file=11763302`, `ram_ok=true`, `correctness_ok=true`.
- Trace/RAM tier: RAM tier loaded `1088.0 MiB`; gate trace `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=107003.632`, `src0_ms_sum=25253.486`, `total_ms_sum=27521.0`.
- Interpretation: top128 reduced file/refault pressure a little more than top64, but still did not exceed `1.6 tok/s`. A 1.1GB anonymous hot expert tier is not sufficient to produce a valid SOTA under the cold 16GB gate.
- Action: source reverted and clean rebuild completed. Clean hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=dff7f183052999b636b57a637ef2e1bea7a57016f4b40d90601a77b578997cb3`. No commit/push because neither RAM-tier run is an accepted SOTA.

### 2026-07-01T18:57:44Z - Rejected 13.5GiB gate cache with no-op-offload

- Purpose: test whether disabling host op offload can free enough transient GPU memory for a larger `GGML_MOE_STREAM_ONE_CACHE_MIB=13824` gate expert cache to survive generation.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T185744Z-20260702_cache13824_no_op_offload_boundary/france-cpu40-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 224 -b 16 -ub 16 -t 20 -tb 20 --no-op-offload`.
- Result: failed/rejected. `eval_tok_s=None`, `prompt_tok_s=None`, `first_answer_ms=None`, `exit_status=134`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15125733376`, `pgmajfault=3036`, `workingset_refault_file=2357`, `ram_ok=true`, `correctness_ok=false` because no answer was generated.
- Failure: stderr shows `VRAM cache: 13.5 GiB, 3252 slots`, then `ggml_cuda_compute_forward: GET_ROWS failed` and `CUDA error: out of memory`. Resource samples reached about `32096MiB` GPU used / `14MiB` free at failure.
- Interpretation: `--no-op-offload` does not free enough GPU headroom for 13.5GiB gate cache. Host RAM remained inside the 16GB cgroup including page cache, but this is not a valid run because it aborts before output.
- Action: no source change, no commit/push.

### 2026-07-01T19:01:21Z - Rejected 13.5GiB gate cache with no-kv-offload

- Purpose: test whether moving KV off GPU can free enough memory for `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T190121Z-20260702_cache13824_no_kv_offload_boundary/france-cpu40-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 224 -b 16 -ub 16 -t 20 -tb 20 --no-kv-offload`.
- Result: failed/rejected. `eval_tok_s=None`, `prompt_tok_s=None`, `first_answer_ms=None`, `exit_status=139`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15203663872`, `pgmajfault=79503`, `workingset_refault_file=5060`, `ram_ok=true`, `correctness_ok=false` because no answer was generated.
- Failure: stderr shows `VRAM cache: 13.5 GiB, 3252 slots`, then a failed `4.57 MiB` CUDA allocation: `cudaMalloc failed: out of memory`, `failed to allocate CUDA0 buffer of size 4792320`. GPU samples again reached about `32096MiB` used / `14MiB` free.
- Interpretation: disabling KV offload does not make 13.5GiB gate cache viable. The limiting factor is practical CUDA allocation headroom/fragmentation at this cache size, not host RAM.
- Action: no source change, no commit/push.

### 2026-07-01T19:03:37Z - Rejected/tie 13696MiB gate cache boundary

- Purpose: map the largest stable gate-cache size below the 13824MiB OOM cliff.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T190337Z-20260702_cache13696_boundary/france-cpu40-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13696`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 224 -b 16 -ub 16 -t 20 -tb 20`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47574.661731ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970724352`, `pgmajfault=607130`, `workingset_refault_file=11862478`, `ram_ok=true`, `correctness_ok=true`.
- Correctness: France answer is coherent and semantically correct.
- Trace: `VRAM cache: 13.4 GiB, 3222 slots`; `rows=34753`, `cache_hits=29870`, `cache_inserts=4883`, `trace_span_ms=108775.806`, `src0_ms_sum=24988.505`, `kernel_ms_sum=391.101`, `dontneed_ms_sum=1188.299`, `total_ms_sum=27270.236`.
- Interpretation: increasing cache from `13568MiB` to `13696MiB` added about `30` slots and reduced misses by only about `15` versus the clean `cache13568` control. This is too small to move token rate beyond `1.6 tok/s`; cache-size-only tuning below the 13.5GiB OOM cliff is exhausted.
- Action: no source change, no commit/push.

### 2026-07-01T19:08:52Z - Rejected/tie cache13568 no-warmup

- Purpose: test whether disabling llama.cpp warmup lowers cold-start TTFT/page-cache pressure without hurting the known-good `13568MiB` gate stream cache.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T190852Z-20260702_cache13568_no_warmup/france-cpu40-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20 --no-warmup`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46584.569105ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14965346304`, `pgmajfault=603613`, `workingset_refault_file=12307596`, `ram_ok=true`, `correctness_ok=true`.
- Correctness: France answer is coherent and semantically correct.
- Trace: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=110472.473`, `src0_ms_sum=24812.119`, `kernel_ms_sum=391.748`, `dontneed_ms_sum=1221.557`, `total_ms_sum=27117.236`.
- Interpretation: `--no-warmup` slightly lowers TTFT versus recent clean controls but does not move eval token rate beyond `1.6 tok/s`; the cold MoE stream/page-refault bottleneck remains. Not a SOTA.
- Action: no source change, no commit/push.

### 2026-07-01T19:16:38Z - Rejected/tie combo skip-hit-DONTNEED + cache13696 + t24 + no-warmup

- Purpose: combine several individually correct but insufficient small improvements to see whether they cross the rounded token-rate boundary.
- Code change tested: default-off `GGML_MOE_STREAM_DONTNEED_HITS=0|1` in `ggml/src/ggml-cuda/moe_stream.cu`; when set to `0`, skip `MADV_DONTNEED` after cache hits while preserving current default behavior.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T191638Z-20260702_combo_skip_hit_dontneed_cache13696_t24_nowarmup/france-cpu40-vram0gb`.
- Config: patched source, `GGML_MOE_STREAM_DONTNEED_HITS=0`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13696`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 24 -tb 24 --no-warmup`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=45990.480752ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14941663232`, `pgmajfault=674529`, `workingset_refault_file=12382046`, `ram_ok=true`, `correctness_ok=true`.
- Correctness: France answer is coherent and semantically correct.
- Trace: `VRAM cache: 13.4 GiB, 3222 slots`; `rows=34753`, `cache_hits=29870`, `cache_inserts=4883`, `trace_span_ms=109689.049`, `src0_ms_sum=25297.52`, `kernel_ms_sum=378.559`, `dontneed_ms_sum=1030.684`, `dontneed_hit_sum=0.046`, `dontneed_miss_sum=1030.638`, `total_ms_sum=27393.121`.
- Interpretation: the combination worked mechanically, but the token rate still tied `1.6 tok/s`. Eliminating hit-path `DONTNEED` and adding a small cache/thread/no-warmup boundary is not enough; the remaining wall time is still dominated by cold page/refault behavior and non-streamed CPU fallback.
- Action: source reverted and clean rebuild completed. No commit/push.

### 2026-07-01T19:23:44Z - Rejected cpu41 14GB cache with t24 no-warmup

- Purpose: test whether one extra CPU MoE layer can free enough VRAM for a 14GB gate cache, and whether `t24 + --no-warmup` can overcome the added CPU fallback cost.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T192344Z-20260702_cpu41_cache14336_t24_nowarmup/france-cpu41-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=41`, `GGML_MOE_STREAM_ONE_CACHE_MIB=14336`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 24 -tb 24 --no-warmup`.
- Result: completed and rejected/regression. `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=43012.05977ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14973276160`, `pgmajfault=687486`, `workingset_refault_file=12694654`, `ram_ok=true`, `correctness_ok=true`.
- Correctness: France answer is coherent and semantically correct.
- Trace: `VRAM cache: 14.0 GiB, 3373 slots`; `rows=35622`, `cache_hits=30670`, `cache_inserts=4952`, `trace_span_ms=113278.671`, `src0_ms_sum=25703.067`, `kernel_ms_sum=403.436`, `dontneed_ms_sum=1272.506`, `total_ms_sum=28095.126`.
- Interpretation: the larger cache fits, but the extra CPU fallback layer increases rows/refault pressure enough to regress token rate. This closes the “more CPU MoE to fit larger cache” path unless a separate source change materially reduces CPU fallback cost.
- Action: no source change, no commit/push.

### 2026-07-01T19:28:56Z - Rejected cache13568 no-host load failure

- Purpose: test whether `--no-host` can lower host-side residency/page-cache pressure for GPU-offloaded tensors under the 16GB cgroup.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T192856Z-20260702_cache13568_no_host/france-cpu40-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20 --no-host`.
- Result: failed/rejected before model load. No token-rate metrics; `exit_status=1`, `memory_peak_bytes=609501184`, `memory_file_bytes=204439552`, `ram_ok=true`, `correctness_ok=false`.
- Failure: stderr shows `ggml_aligned_malloc: insufficient memory (attempted to allocate 130560.00 MB)`, `failed to allocate CPU_REPACK buffer`, and model load failure.
- Interpretation: plain `--no-host` is incompatible with the default repack path for this model. Follow-up is `--no-host --no-repack` to test whether the huge CPU_REPACK allocation can be avoided.
- Action: no source change, no commit/push.

### 2026-07-01T19:31:24Z - Rejected/tie cache13568 no-host no-repack

- Purpose: test whether `--no-host --no-repack` avoids the huge CPU_REPACK allocation and reduces host-side page/cache pressure.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T193124Z-20260702_cache13568_no_host_no_repack/france-cpu40-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20 --no-host --no-repack`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47863.786741ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14973636608`, `pgmajfault=601079`, `workingset_refault_file=11857635`, `ram_ok=true`, `correctness_ok=true`.
- Correctness: France answer is coherent and semantically correct.
- Trace: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=108360.648`, `src0_ms_sum=25004.993`, `kernel_ms_sum=390.698`, `dontneed_ms_sum=1207.695`, `total_ms_sum=27294.078`.
- Interpretation: `--no-host --no-repack` runs, but cache shape and cold refault behavior stay baseline-shaped and token rate remains `1.6 tok/s`. No-host is not a SOTA path.
- Action: no source change, no commit/push.

### 2026-07-01T19:37:20Z - Diagnostic current CPU fallback bottleneck trace

- Purpose: refresh the bottleneck map after exhausting cache/thread/no-host/simple page-advice probes.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb`.
- Config: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, CPU chunk trace enabled with `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=2000000`, cold `drop_caches`, 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: diagnostic-only. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=47630.142084ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964211712`, `pgmajfault=575052`, `workingset_refault_file=12049946`, `ram_ok=true`, `correctness_ok=true`.
- Gate trace: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=112371.567`, `src0_ms_sum=27141.642`, `total_ms_sum=29441.733`.
- CPU chunk trace: `1667384` rows, `125806993` bytes. Thread-sum time is `ffn_up_exps=688777.888ms`, `ffn_down_exps=594024.621ms`. Parallel-normalized fallback estimate is `64140.125ms`, split `up=34438.894ms`, `down=29701.231ms`.
- Top layers by parallel-normalized fallback: layer `1=3358.883ms`, `2=3302.751ms`, `0=3238.435ms`, `19=2151.23ms`, `3=2104.288ms`, `10=2033.229ms`. The top three layers cover only about `15%` of fallback cost, so the remaining bottleneck is broad up/down fallback, not a single pathological layer.
- Generated profiles:
  - `cpu_fallback_topcost128_profile.tsv`, sha256 `d8c5a3d7acdd7edd68f79156297b73dc58548f3fdaf409bf9fce11a9908f91c1`, coverage `6.72%`.
  - `cpu_fallback_topcost256_profile.tsv`, sha256 `16d679891f833cc049c690d8dfee466cc9d101e4f09a428aac31ebebe0191410`, coverage `11.77%`.
  - `cpu_fallback_topcost512_profile.tsv`, sha256 `6a5fb5b0f991808b8bf8a60d9b877cb88c93f8a8f9f463bfcf2d07987817e38a`, coverage `20.29%`.
- Interpretation: future work should stop small gate/cache/thread/no-host probes. A useful SOTA now likely requires structural CPU fallback reduction or a larger, bounded RAM-tier/read-scheduling change; small top-N profiles are unlikely to move token rate because coverage is too low.
- Action: no source change, no commit/push.

### 2026-07-01T19:47:29Z - Rejected/tie CPU fallback RAM tier top512 current profile

- Purpose: test a larger CPU fallback RAM-tier profile derived from the current bottleneck trace, covering top512 up/down tensor/expert entries.
- Code change tested: reused default-off `GGML_MOE_CPU_RAM_TIER_PROFILE` / `GGML_MOE_CPU_RAM_TIER_MAX_MIB` patch in `ggml/src/ggml-cpu/ggml-cpu.c`.
- Profile: `/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb/cpu_fallback_topcost512_profile.tsv`, sha256 `6a5fb5b0f991808b8bf8a60d9b877cb88c93f8a8f9f463bfcf2d07987817e38a`, estimated fallback coverage `20.29%`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T194729Z-20260702_cpu_fallback_ramtier_top512_current_profile/france-cpu40-vram0gb`.
- Config: patched source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_CPU_RAM_TIER_PROFILE=<top512 profile>`, `GGML_MOE_CPU_RAM_TIER_MAX_MIB=2304`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=48557.903671ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=12684492800`, `pgmajfault=567206`, `workingset_refault_file=11612222`, `ram_ok=true`, `correctness_ok=true`.
- RAM tier: loaded `512` profile entries, final anonymous tier `2176.0MiB`; page cache dropped to `12.68GB`, inside the same 16GB cgroup budget.
- Trace: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=110472.283`, `src0_ms_sum=25128.771`, `kernel_ms_sum=394.271`, `dontneed_ms_sum=1204.844`, `total_ms_sum=27430.624`.
- Interpretation: top512 RAM tier works mechanically and slightly lowers fault/refault counters, but token rate still ties `1.6` and TTFT worsens. Medium top-N RAM-tier coverage is not sufficient; first-use copy and reduced page-cache budget offset the reuse benefit.
- Action: source reverted and clean rebuild completed. No commit/push.

### 2026-07-01T19:54:53Z - Rejected/regression CPU fallback RAM tier top1024 current profile

- Purpose: test the practical upper bound of the current RAM-tier approach by loading the top1024 up/down tensor/expert entries from the current CPU fallback bottleneck profile.
- Code change tested: reused default-off `GGML_MOE_CPU_RAM_TIER_PROFILE` / `GGML_MOE_CPU_RAM_TIER_MAX_MIB` patch in `ggml/src/ggml-cpu/ggml-cpu.c`.
- Profile: `/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb/cpu_fallback_topcost1024_profile.tsv`, sha256 `99d857356b8e7403df31127e0d0b629014cbd6db21dad7e5b4b94d0f834bfdf3`, estimated fallback coverage `34.19%`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T195453Z-20260702_cpu_fallback_ramtier_top1024_current_profile/france-cpu40-vram0gb`.
- Config: patched source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_CPU_RAM_TIER_PROFILE=<top1024 profile>`, `GGML_MOE_CPU_RAM_TIER_MAX_MIB=4608`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: completed and rejected/regression. `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=50564.398072ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=10403127296`, `pgmajfault=531322`, `workingset_refault_file=11334691`, `ram_ok=true`, `correctness_ok=true`.
- RAM tier: loaded `1024` entries, final anonymous tier `4352.0MiB`; page cache dropped to `10.40GB`, still inside the same 16GB cgroup.
- Trace: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=113952.31`, `src0_ms_sum=25816.382`, `kernel_ms_sum=398.545`, `dontneed_ms_sum=1218.506`, `total_ms_sum=28130.935`.
- Interpretation: broadening RAM-tier coverage lowers fault counters but regresses token rate and TTFT. The anonymous tier displaces too much useful page cache and first-use copies are too expensive. Current top-N RAM-tier is not a viable cold SOTA path.
- Action: source reverted and clean rebuild completed. No commit/push.

### 2026-07-01T20:03:36Z - Rejected/tie lazy-after-use CPU fallback RAM tier top512

- Purpose: test whether caching profiled up/down experts only after they are naturally read by CPU fallback avoids the first-use penalty of eager RAM-tier loading.
- Code change tested: default-off `GGML_MOE_CPU_LAZY_RAM_TIER_PROFILE` / `GGML_MOE_CPU_LAZY_RAM_TIER_MAX_MIB` in `ggml/src/ggml-cpu/ggml-cpu.c`; before compute it uses an already-loaded anonymous copy, and after thread 0 finishes an active profiled expert it copies the mmap expert for future reuse.
- Profile: `/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb/cpu_fallback_topcost512_profile.tsv`, sha256 `6a5fb5b0f991808b8bf8a60d9b877cb88c93f8a8f9f463bfcf2d07987817e38a`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T200336Z-20260702_cpu_fallback_lazy_ramtier_top512/france-cpu40-vram0gb`.
- Config: patched source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, lazy RAM tier top512 with `GGML_MOE_CPU_LAZY_RAM_TIER_MAX_MIB=2304`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=47417.390374ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=12675297280`, `pgmajfault=592960`, `workingset_refault_file=11491094`, `ram_ok=true`, `correctness_ok=true`.
- Lazy tier: loaded `512` entries, final anonymous tier `2176.0MiB`; page cache dropped to `12.68GB`.
- Trace: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=107966.851`, `src0_ms_sum=24907.113`, `kernel_ms_sum=386.255`, `dontneed_ms_sum=1192.567`, `total_ms_sum=27181.674`.
- Interpretation: lazy loading improves over eager top512 in TTFT and trace span, but still does not move token rate beyond `1.6`. Top-N RAM-tier caching is not enough; the remaining CPU fallback bottleneck needs a more direct reduction in page touches or compute work.
- Action: source reverted and clean rebuild completed. No commit/push.

### 2026-07-01T20:13:16Z - Completed page-fault/read-amplification clean repro diagnostic

- Purpose: continue the 2.6 tok/s forensic path by checking whether current clean cold runs still show the high `pgmajfault` / file-input shape that differs from the historical correct `2.6 tok/s` run.
- Plan entry: `20260702-page-fault-readamp-clean-repro`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T201316Z-20260702_page_fault_readamp_clean_repro/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: diagnostic only, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47453.251439ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978854912`, `pgmajfault=605634`, `workingset_refault_file=11871712`, `ram_ok=true`, `correctness_ok=true`.
- Correctness: pass. France answer is semantically correct and coherent.
- `/usr/bin/time -v`: current wall `2:08.51`, major faults `606032`, minor faults `2626823`, file system inputs `256344128`, max RSS `15595768 KB`.
- Historical 2.6 comparison: old correct run wall `1:29.13`, major faults `309783`, minor faults `2016378`, file system inputs `154862552`, cgroup `pgmajfault=309216`, `workingset_refault_file=3371495`.
- Trace comparison: current `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=108889.963`, `src0_ms_sum=25288.845`, `total_ms_sum=27598.21`; old 2.6 `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `trace_span_ms=71223.061`, `src0_ms_sum=23564.133`, `total_ms_sum=25786.372`.
- Interpretation: the direct gate stream copy timing differs only slightly, but current cold runs have roughly 2x major faults and much higher file inputs/refaults. The missing 2.6 behavior is not a gate cache policy issue. Next work should inspect the `ffn_up_exps` / `ffn_down_exps` CPU fallback path and design a correctness-safe structural reduction instead of continuing small gate-cache/thread/no-host probes.
- Action: no source change, no commit/push.

### 2026-07-01T20:21:54Z - Completed up-only stream correctness probe

- Purpose: validate whether `ffn_up_exps` can use the existing corrected DS4 stream path before implementing a gate+up source path.
- Finding before run: GGUF header shows `ffn_gate_exps` and `ffn_up_exps` have the same shape/type (`[4096,2048,256]`, `MXFP4`, `1140850688` bytes per layer tensor); `ffn_down_exps` is `[2048,4096,256]`, also `MXFP4`.
- Plan entry: `20260702-up-only-stream-compare`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T202154Z-20260702_up_only_stream_compare/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_up_exps`, `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=16`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: diagnostic only, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47151.246844ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14923268096`, `pgmajfault=770071`, `workingset_refault_file=19699902`, `ram_ok=true`, `correctness_ok=true`.
- Compare: `compare_cpu.csv` has 16 rows, all `ffn_up_exps`; `max_abs_max=9.53674316e-07`, `mean_abs_max=2.63753464e-08`, `max_rel_max=8.22526591e-07`. This validates up stream numeric correctness for sampled experts.
- Trace: `rows=47458`, `n_tensors=40`, `cache_hits=41418`, `cache_inserts=6040`, `trace_span_ms=144981.391`, `src0_ms_sum=31250.244`, `kernel_ms_sum=564.656`, `total_ms_sum=34326.729`.
- Interpretation: up stream is correct but not SOTA alone because gate/down CPU fallback remain and page/refault pressure worsens. Next correctness check is `ffn_down_exps` before attempting no-filter all-MoE stream.
- Action: no source change, no commit/push.

### 2026-07-01T20:34:00Z - Started down-only stream correctness probe

- Purpose: validate whether `ffn_down_exps` also agrees with CPU fallback under the corrected DS4 rows stream path.
- Plan entry: `20260702-down-only-stream-compare`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T202655Z-20260702_down_only_stream_compare/france-cpu40-vram0gb`.
- Config: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_down_exps`, `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=16`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Result: diagnostic only, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=44542.535707ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14961033216`, `pgmajfault=547684`, `workingset_refault_file=11372958`, `ram_ok=true`, `correctness_ok=true`.
- Compare: `compare_cpu.csv` has 16 rows, all `ffn_down_exps`; `max_abs_max=3.81469727e-06`, `mean_abs_max=5.7837542e-07`, `max_rel_max=1.25252972e-06`. This validates down stream numeric correctness for sampled experts.
- Trace: first call `ne01=4096`, `ne00=2048`, `nb01=1088`; `rows=34258`, `n_tensors=40`, `cache_hits=29196`, `cache_inserts=5062`, `trace_span_ms=107487.845`, `src0_ms_sum=25732.602`, `kernel_ms_sum=427.647`, `total_ms_sum=28150.005`.
- Interpretation: down stream is correct. Since gate, up, and down each pass sampled compare individually, the next performance probe is no-filter all-MoE stream with compare disabled.
- Action: no source change, no commit/push.

### 2026-07-01T20:40:00Z - Started all-MoE stream no-filter performance probe

- Purpose: measure whether streaming gate/up/down together removes enough CPU fallback to exceed the clean `1.6 tok/s` line.
- Plan entry: `20260702-all-moe-stream-no-filter`.
- Config to run: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, no `GGML_MOE_STREAM_ONE_NAME_FILTER`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, compare disabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: if this exceeds the current valid token-rate line with RAM/correctness/TTFT passing, immediately stop and complete full SOTA reproduction record plus source push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, followed by clean rebuild/rerun before promotion.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T203112Z-20260702_all_moe_stream_no_filter/france-cpu40-vram0gb`.
- Result: rejected/regression. `eval_tok_s=1.0`, `prompt_tok_s=0.5`, `TTFT=52832.556299ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970703872`, `pgmajfault=30851`, `workingset_refault_file=20696007`, `ram_ok=true`, `correctness_ok=true`.
- Trace: streamed gate/up/down equally (`36899` rows each, `110697` total), `cache_hits=76412`, `cache_inserts=34285`, hit rate `69.0%`, `trace_span_ms=177220.334`, `src0_ms_sum=158749.492`, `kernel_ms_sum=1374.723`, `total_ms_sum=169332.924`.
- Interpretation: all-MoE stream is numerically correct enough for the France prompt, but much slower because VRAM cache competition and H2D/page-in dominate. The right follow-up is gate+up only, leaving down on CPU.
- Action: no source change, no commit/push.

### 2026-07-01T20:50:00Z - Started gate+up multifilter stream source probe

- Purpose: stream accepted gate plus numerically validated up experts while avoiding down-stream cache pressure.
- Plan entry: `20260702-gate-up-multifilter-stream`.
- Source change planned: default-compatible comma-separated `GGML_MOE_STREAM_ONE_NAME_FILTER` handling in `ggml/src/ggml-cuda/moe_stream.cu`; existing empty/single-filter behavior must remain unchanged.
- Config to run after rebuild: `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 stream enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, compare disabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: reject and revert source unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T203735Z-20260702_gate_up_multifilter_stream/france-cpu40-vram0gb`.
- Result: rejected/regression. `eval_tok_s=1.3`, `prompt_tok_s=0.6`, `TTFT=49185.272747ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964355072`, `pgmajfault=286601`, `workingset_refault_file=12049338`, `ram_ok=true`, `correctness_ok=true`.
- Trace: only gate/up streamed; `rows=63700`, `cache_hits=48394`, `cache_inserts=15306`, hit rate `76.0%`, `trace_span_ms=124768.843`, `src0_ms_sum=73807.596`, `kernel_ms_sum=757.638`, `total_ms_sum=79082.874`. Gate and up each had `31850` rows and `7653` inserts.
- Interpretation: gate+all-up is still too broad. Extra stream misses and `src0` time outweigh CPU fallback savings. Future stream work must be selective hot up layers/experts, or a different cache partition/scheduling design.
- Action: source patch reverted and clean rebuild completed; build returned to `ggml commit: aa8d6f916`. No commit/push.

### 2026-07-01T20:55:00Z - Hot up layer stream bound analysis

- Purpose: decide whether to implement a narrower gate+hot-up source probe after gate+all-up regressed.
- CPU fallback estimate: up fallback parallel-normalized total is `34438.894ms`. Top up layers are `1=1867.479ms`, `0=1830.977ms`, `2=1816.512ms`, `19=1256.338ms`, `3=1112.161ms`, `5=1088.347ms`.
- Coverage: top3 up layers cover `5514.968ms` (`16.01%` of up fallback); top6 cover `8971.814ms` (`26.05%`).
- Stream cost estimate from rejected gate+up trace: top3 up layers would add about `2454` streamed rows, `1370` inserts, `6745.3ms src0`, and `7093.7ms total`.
- Decision: do not run hot-up-layer stream now. The estimated stream cost already exceeds top3 CPU fallback saving, and top6 would likely add more cache pressure. More name-filter combinations are low priority without a better cache/read scheduling change.
- Action: no source change, no commit/push.

### 2026-07-01T21:05:00Z - Started down-only stream no-compare sanity run

- Purpose: verify whether the previous down-only stream tie was affected by CPU/GPU compare overhead.
- Plan entry: `20260702-down-only-stream-no-compare`.
- Config to run: clean source, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_down_exps`, compare disabled, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: no source change and no commit/push unless it unexpectedly becomes a fully gated higher token-rate candidate and the SOTA reproduction/push gate is completed.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T204807Z-20260702_down_only_stream_no_compare/france-cpu40-vram0gb`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=43920.660705ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970474496`, `pgmajfault=535980`, `workingset_refault_file=11154862`, `ram_ok=true`, `correctness_ok=true`.
- Trace: only down tensors streamed; `rows=34258`, `cache_hits=29196`, `cache_inserts=5062`, `trace_span_ms=105209.863`, `src0_ms_sum=25997.467`, `kernel_ms_sum=422.328`, `total_ms_sum=28393.829`.
- Interpretation: compare overhead was not hiding a SOTA; down-only remains a tie. Combined with up-only/all-MoE/gate+up results, ordinary name-filter stream combinations should be closed until a cache/read scheduling change alters the cost model.
- Action: no source change, no commit/push.

### 2026-07-01T21:10:00Z - Closed ordinary name-filter stream combinations

- Evidence: up-only and down-only stream are numerically correct but tie `1.6`; all-MoE regresses to `1.0`; gate+up regresses to `1.3`; hot-up-layer bound is negative.
- Decision: do not spend more runs on plain gate+down or top-layer name-filter variants. The bottleneck has shifted from arithmetic correctness to stream miss/cache/page-in cost.
- Next priority: design a default-off mechanism that directly reduces miss cost or page-cache churn, such as cache admission/partition, bounded async prefetch, or a reuse-aware source page lifetime policy.

### 2026-07-01T21:15:00Z - Started VRAM cache hash lookup probe

- Purpose: reduce fixed CPU-side VRAM cache lookup overhead without changing arithmetic or stream scope.
- Plan entry: `20260702-vram-cache-hash-lookup-gate`.
- Source change planned: default-off `GGML_MOE_STREAM_HASH_LOOKUP=1` path in `ggml/src/ggml-cuda/moe_stream.cu`; default path remains current linear slot scan.
- Config to run after rebuild: `GGML_MOE_STREAM_HASH_LOOKUP=1`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: reject and revert source unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T205549Z-20260702_vram_cache_hash_lookup_gate/france-cpu40-vram0gb`.
- Result: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46540.808484ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14980935680`, `pgmajfault=613608`, `workingset_refault_file=12009511`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=109928.624`, `src0_ms_sum=25296.821`, `kernel_ms_sum=389.464`, `total_ms_sum=27592.874`.
- Interpretation: O(1) lookup preserves behavior but does not move token rate. Linear cache slot scan is not a major bottleneck in current gate-only cold runs.
- Action: source patch reverted and clean rebuild completed; build returned to `ggml commit: aa8d6f916`. No commit/push.

### 2026-07-01T21:25:00Z - Started CPU fallback MADV_RANDOM read-amplification probe

- Purpose: test whether disabling kernel readahead for CPU fallback expert pages reduces cold file inputs/refault pressure.
- Plan entry: `20260702-cpu-fallback-madv-random`.
- Source change planned: default-off `GGML_MOE_CPU_MADV_RANDOM=1` in `ggml/src/ggml-cpu/ggml-cpu.c`, applied only to active CPU fallback expert pages after gate-streamed experts have been zeroed out.
- Config to run after rebuild: `GGML_MOE_CPU_MADV_RANDOM=1`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: reject and revert source unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T210425Z-20260702_cpu_fallback_madv_random/france-cpu40-vram0gb`.
- Result: rejected/timeout. The run was killed after `first_answer_ms=131415.288893`; `eval_tok_s=null`, `prompt_tok_s=null`, `ttft_estimate_ms=131415.288893`, and full gated metrics were not produced.
- Correctness/RAM status: not promotable. `summary.json` reports a heuristic `correctness_ok=true`, but the answer was partial and mixed with termination/log noise; `ram_ok=false` because memory files were incomplete after the kill.
- Partial trace: `rows=17191`, `cache_hits=13490`, `cache_inserts=3701`, `trace_span_ms=288329.438`, `src0_ms_sum=19144.928`, `kernel_ms_sum=232.46`, `dontneed_ms_sum=917.552`, `total_ms_sum=20673.021`.
- Interpretation: `MADV_RANDOM` is a severe regression for CPU fallback expert pages, likely because it disables useful readahead for contiguous expert reads. This matches the earlier stream source `MADV_RANDOM` timeout and closes random/no-readahead advice as a near-term direction.
- Action: source patch reverted and clean rebuild required before the next probe. No commit/push because there is no accepted SOTA.

### 2026-07-02T00:00:00Z - Started gate cache admission min2 profile probe

- Purpose: test a bounded cache-admission policy instead of increasing cache size or changing page advice.
- Plan entry: `20260702-gate-cache-admit-min2-profile`.
- Offline simulation source: clean gate trace `/root/lfz/runs/vendor-ds4-16gb/20260701T201316Z-20260702_page_fault_readamp_clean_repro/france-cpu40-vram0gb/one_trace.csv`.
- Simulation result: current `3192`-slot LRU has `4898` inserted loads; caching only `(tensor, expert)` keys with count `>=2` gives `3116` eligible keys, `1418` one-time staging loads, and `4534` total loads. Expected direct traced `src0` upper bound is about `364 * 5.16ms ~= 1.9s`.
- Source change planned: default-off `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<csv>` and `GGML_MOE_STREAM_CACHE_ADMIT_MIN_USES=2` in `ggml/src/ggml-cuda/moe_stream.cu`; non-admitted misses use the existing staging buffer and do not occupy VRAM cache.
- Gate: reject and revert unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Profile: `/root/lfz/runs/vendor-ds4-16gb/cache_admit_profiles/gate_min2_from_20260701T201316Z.csv`, sha256 `6e79d51b0b595859d3ffd589f413ae8a56e3604d21943b701fc2dc31222ad81d`.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T212038Z-20260702_gate_cache_admit_min2_profile/france-cpu40-vram0gb`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46501.495934ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14968098816`, `pgmajfault=595128`, `workingset_refault_file=11433671`, `ram_ok=true`, `correctness_ok=true`.
- Trace: admission matched simulation: `rows=34753`, `cache_hits=30219`, `cache_inserts=3116`, `staging_non_admitted=1418`, `trace_span_ms=107486.742`, `src0_ms_sum=22979.714`, `kernel_ms_sum=382.29`, `dontneed_ms_sum=1115.399`, `total_ms_sum=25166.967`.
- Interpretation: cache admission reduced direct gate copy/page-in time by about `2.3s`, but did not improve rounded token rate beyond `1.6`. The long trace span and end-to-end wall remain dominated by untraced reclaim/CPU fallback behavior, so simple gate-cache admission is not enough.
- Action: source patch reverted and clean rebuild required. No commit/push because there is no accepted SOTA.

### 2026-07-02T00:00:00Z - Started CPU fallback chunk-size 32 probe

- Purpose: test the plan's `up/down compute speed` direction with a narrowly scoped CPU fallback scheduling parameter.
- Plan entry: `20260702-cpu-fallback-chunk32`.
- Trace basis: `/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb/cpu_chunk_trace.csv` contains `1,667,384` up/down chunks; summed traced chunk ms is `up=688777.888`, `down=594024.621`.
- Source change planned: default-off `GGML_MOE_CPU_CHUNK_SIZE=32` in `ggml/src/ggml-cpu/ggml-cpu.c`, preserving existing `chunk_size=64` for `nr0 == 1 || nr1 == 1`.
- Gate: reject and revert unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T212807Z-20260702_cpu_fallback_chunk32/france-cpu40-vram0gb`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.8`, `TTFT=41706.09388ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14979452928`, `pgmajfault=491558`, `workingset_refault_file=11934422`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=104695.023`, `src0_ms_sum=24559.658`, `kernel_ms_sum=390.653`, `dontneed_ms_sum=1217.548`, `total_ms_sum=26861.594`.
- Interpretation: chunk32 improves TTFT and trace span but does not move rounded token rate beyond `1.6`. It is not a SOTA alone.
- Action: continue only into one combined boundary probe with cache admission; if the combination is not promoted, revert all source changes and clean rebuild. No commit/push for this tie.

### 2026-07-02T00:00:00Z - Started chunk32 plus cache-admit min2 combined probe

- Purpose: combine two independent small wins: lower TTFT/trace span from `GGML_MOE_CPU_CHUNK_SIZE=32` and lower direct gate `src0_ms` from min2 cache admission.
- Plan entry: `20260702-chunk32-cache-admit-min2-combined`.
- Config to run after rebuild: `GGML_MOE_CPU_CHUNK_SIZE=32`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/runs/vendor-ds4-16gb/cache_admit_profiles/gate_min2_from_20260701T201316Z.csv`, `GGML_MOE_STREAM_CACHE_ADMIT_MIN_USES=2`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: reject and revert both source changes unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T213246Z-20260702_chunk32_cache_admit_min2_combined/france-cpu40-vram0gb`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=44178.221165ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970384384`, `pgmajfault=494525`, `workingset_refault_file=11333408`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `rows=34753`, `cache_hits=30219`, `cache_inserts=3116`, `staging_non_admitted=1418`, `trace_span_ms=102982.959`, `src0_ms_sum=23256.709`, `kernel_ms_sum=380.856`, `dontneed_ms_sum=1139.654`, `total_ms_sum=25464.565`.
- Interpretation: combined small wins produce the best gate trace/span of this group, but end-to-end eval remains `1.6`. The remaining bottleneck is not simple gate cache pollution or CPU fallback chunk scheduling; next work needs a deeper CPU fallback/page-reclaim breakdown or a larger structural change.
- Action: both source changes reverted and clean rebuild required. No commit/push because there is no accepted SOTA.

### 2026-07-02T00:00:00Z - Started CPU fallback chunk-size 64 boundary probe

- Purpose: extend the `chunk32` signal by testing whether a larger multi-row CPU fallback chunk further reduces scheduling overhead.
- Plan entry: `20260702-cpu-fallback-chunk64`.
- Prior signal: `chunk32` improved TTFT to `41706.09388ms` and gate trace span to `104695.023ms`, but remained `eval_tok_s=1.6`.
- Source change planned: default-off `GGML_MOE_CPU_CHUNK_SIZE=64` in `ggml/src/ggml-cpu/ggml-cpu.c`, preserving current default and preserving the existing single-row `chunk_size=64` path.
- Gate: reject and revert unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T214016Z-20260702_cpu_fallback_chunk64/france-cpu40-vram0gb`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.8`, `TTFT=42533.796173ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14974631936`, `pgmajfault=439220`, `workingset_refault_file=12248606`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=105003.531`, `src0_ms_sum=26322.059`, `kernel_ms_sum=390.073`, `dontneed_ms_sum=1211.469`, `total_ms_sum=28624.892`.
- Interpretation: chunk64 preserves lower TTFT but worsens traced gate `src0_ms`/total relative to chunk32 and still ties `1.6`. Larger multi-row chunks are not enough and may be losing load balance.
- Action: source patch reverted and clean rebuild required. No commit/push because there is no accepted SOTA.

### 2026-07-02T00:00:00Z - Started thread16 reclaim-contention probe

- Purpose: test whether the remaining cold-start wall is sensitive to CPU fallback thread count and cgroup reclaim contention.
- Plan entry: `20260702-thread16-reclaim-contention-probe`.
- Tooling change: fixed `.Agent/run-tools/strict_ds4_runner.py` elapsed-time parsing for `/usr/bin/time -v` values like `2:08.51`; this is run-record tooling only and does not affect model behavior.
- Config to run: clean source, no model source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 16 -tb 16`.
- Gate: reject unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T214659Z-20260702_thread16_reclaim_contention/france-cpu40-vram0gb`.
- Result: completed and rejected/regression. `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=48076.625549ms`, corrected `elapsed_seconds=135.86`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14983475200`, `pgmajfault=848549`, `workingset_refault_file=12118944`, `ram_ok=true`, `correctness_ok=true`.
- `/usr/bin/time -v`: user `494.27s`, system `818.69s`, wall `2:15.86`, major faults `849582`, file inputs `258392264`.
- Trace: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=116750.166`, `src0_ms_sum=25501.426`, `kernel_ms_sum=400.558`, `dontneed_ms_sum=1228.624`, `total_ms_sum=27846.671`.
- Interpretation: fewer threads reduce CPU parallelism and do not reduce reclaim enough to help; t16 is slower and has more major faults. Do not pursue lower thread count.
- Action: no model source change; no commit/push.

### 2026-07-02T00:00:00Z - Started thread24 CPU-parallelism probe

- Purpose: test the opposite side of the thread-count tradeoff after t16 regressed.
- Plan entry: `20260702-thread24-cpu-parallelism-probe`.
- Config to run: clean source, no model source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 24 -tb 24`.
- Gate: reject unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T215044Z-20260702_thread24_cpu_parallelism/france-cpu40-vram0gb`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47206.004663ms`, corrected `elapsed_seconds=129.2`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14976692224`, `pgmajfault=668293`, `workingset_refault_file=12143176`, `ram_ok=true`, `correctness_ok=true`.
- `/usr/bin/time -v`: user `800.85s`, system `1069.78s`, wall `2:09.20`, major faults `668703`, file inputs `259072000`.
- Trace: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=109311.238`, `src0_ms_sum=25528.211`, `kernel_ms_sum=394.271`, `dontneed_ms_sum=1236.551`, `total_ms_sum=27851.837`.
- Interpretation: t24 ties rounded token rate and does not improve the wall/trace shape; system time is higher than t20/t16. With t16 regression, close simple thread-count tuning.
- Action: no model source change; no commit/push.

### 2026-07-02T00:00:00Z - Closed simple thread-count tuning

- Evidence: t16 regressed to `1.5 tok/s`; t24 tied `1.6 tok/s` and did not improve trace span or wall. Existing t20 remains the best tested clean setting.
- Next priority: add/collect op-level CPU fallback wall tracing or design a structural reduction in up/down fallback work. Nearby thread counts should not be run unless a new mechanism changes the cost model.

### 2026-07-02T00:00:00Z - Started CPU fallback op-level wall trace diagnostic

- Purpose: replace chunk-sum inference with direct per-op CPU fallback wall timing after gate stream has removed gate rows.
- Plan entry: `20260702-cpu-fallback-op-wall-trace`.
- Source change planned: default-off `GGML_MOE_CPU_OP_TRACE_OUT=<csv>` in `ggml/src/ggml-cpu/ggml-cpu.c`; when enabled, thread 0 writes one row per `mul_mat_id` op with tensor name/type, remaining active experts/rows, dimensions, and fallback wall time.
- Config to run after rebuild: `GGML_MOE_CPU_OP_TRACE_OUT={case_dir}/cpu_op_trace.csv`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled for `ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: diagnostic only. RAM/correctness must pass for comparability. Revert source after collecting trace unless a later no-trace reproduction proves a real SOTA.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T215815Z-20260702_cpu_fallback_op_wall_trace/france-cpu40-vram0gb`.
- Result: completed diagnostic, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=48357.974517ms`, corrected `elapsed_seconds=130.87`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14977716224`, `pgmajfault=606034`, `workingset_refault_file=11947939`, `ram_ok=true`, `correctness_ok=true`.
- CPU op trace: `op_rows=16680`; gate stream removed gate fallback (`gate active_rows=0`, wall `71.917ms`). Remaining CPU fallback wall is `ffn_up_exps=41098.899ms` and `ffn_down_exps=36159.187ms`.
- Top wall tensors: `blk.0.up=2015.699ms`, `blk.1.up=1971.926ms`, `blk.2.up=1943.363ms`, `blk.1.down=1886.923ms`, `blk.0.down=1745.392ms`, `blk.2.down=1735.727ms`.
- Interpretation: the bottleneck is confirmed as broad up/down CPU fallback wall. Gate fallback is already effectively eliminated. Future work should target structural up/down fallback reduction; more gate cache tuning, nearby thread counts, or plain name-filter streaming is low priority.
- Action: trace source patch reverted and clean rebuild required. No commit/push because this is diagnostic and not an accepted SOTA.

### 2026-07-02T00:00:00Z - Started selective hot-up stream probe

- Purpose: test a narrow structural reduction in up CPU fallback using only up layers with positive net estimate from the gate+all-up trace.
- Plan entry: `20260702-selective-hot-up-stream`.
- Offline bound: selected up layers `3,5,6,7,10,11,12,15,18,29,31,35` have about `+1.9s` estimated net (`CPU wall - stream total`) under the previous gate+all-up trace, while all-up and early `0/1/2` up layers are negative.
- Source change planned: default-compatible comma-separated `GGML_MOE_STREAM_ONE_NAME_FILTER` in `ggml/src/ggml-cuda/moe_stream.cu`.
- Config to run after rebuild: gate plus selected up layers, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 stream enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: reject and revert unless token rate exceeds `1.6`, RAM/correctness pass, and TTFT stays within gate. If accepted, immediately complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T220650Z-20260702_selective_hot_up_stream/france-cpu40-vram0gb`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=48055.622383ms`, corrected `elapsed_seconds=123.83`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14974443520`, `pgmajfault=485931`, `workingset_refault_file=9550051`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `rows=43270`, `cache_hits=36204`, `cache_inserts=7066`, `trace_span_ms=103802.707`; gate `rows=33307/inserts=5480/src0_ms=26878.954/total_ms=29212.292`, selected up `rows=9963/inserts=1586/src0_ms=8404.0/total_ms=9049.1`.
- Interpretation: 12-layer selective up lowers wall versus some controls but worsens TTFT and increases gate misses. Not a SOTA.

### 2026-07-02T00:00:00Z - Started selective hot-up top6 stream probe

- Purpose: reduce gate cache competition from the 12-layer selective run while keeping the strongest positive-net up layers.
- Plan entry: `20260702-selective-hot-up-top6-stream`.
- Config: gate plus up layers `5,6,7,15,31,35`, same vram12/cold/16GB France setup.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T221045Z-20260702_selective_hot_up_top6_stream/france-cpu40-vram0gb`.
- Result: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46496.582582ms`, corrected `elapsed_seconds=127.64`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14949691392`, `pgmajfault=557174`, `workingset_refault_file=11427759`, `ram_ok=true`, `correctness_ok=true`.
- Trace: `rows=39658`, `cache_hits=33592`, `cache_inserts=6066`, `trace_span_ms=109462.517`; gate `rows=34502/inserts=5294/src0_ms=27347.2/total_ms=29707.1`, top6 up `rows=5156/inserts=772/src0_ms=4374.7/total_ms=4699.6`.
- Interpretation: top6 reduces up stream cost but still increases gate inserts and does not improve token rate. Selective up-streaming without explicit cache partition/admission is closed.
- Action: comma-filter source reverted and clean rebuild required. No commit/push because there is no accepted SOTA.

### 2026-07-02T00:00:00Z - Started expert keep-top5 up/down pruning probe

- Purpose: test a larger structural reduction in the confirmed up/down CPU fallback bottleneck.
- Plan entry: `20260702-expert-keep-top5-updown`.
- Theory: op trace shows `up≈41.1s` and `down≈36.2s` fallback wall. Keeping 5 of 6 routed experts for up/down has a rough upper bound near `12.9s` wall reduction, but may degrade quality.
- Source change planned: default-off `GGML_MOE_KEEP_TOPK_UPDOWN=5`; skip routed ranks `id >= 5` for `ffn_up_exps` and `ffn_down_exps`, explicitly zeroing their output rows. Gate tensors remain unpruned.
- Config to run: `cpu_moe=40`, gate stream only, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- Gate: reject and revert unless token rate exceeds `1.6`, RAM/TTFT pass, and France output is manually correct/coherent. If accepted, complete full reproduction/push/rerun SOTA gate.
- Run directory: `/root/lfz/runs/vendor-ds4-16gb/20260701T221922Z-20260702_expert_keep_top5_updown/france-cpu40-vram0gb`.
- Result: accepted candidate pending pushed-commit rerun. `eval_tok_s=1.9`, `prompt_tok_s=0.7`, `TTFT=43426.525893ms`, corrected `elapsed_seconds=127.94`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970519552`, `pgmajfault=551675`, `workingset_refault_file=11094865`, `ram_ok=true`, `correctness_ok=true`.
- Manual correctness: pass. France answer is coherent and semantically correct, covering France/French Republic, Western Europe, history/culture/global influence, landmarks, cuisine/wine/fashion, EU membership, Paris, diverse regions, French Revolution, and economic/political role. Minor Eiffel Tower repetition is acceptable.
- Trace: `rows=43123`, `cache_hits=37697`, `cache_inserts=5426`, `trace_span_ms=110181.209`, `src0_ms_sum=29038.595`, `total_ms_sum=31722.375`.
- SOTA gate: passes RAM including page cache, correctness, and TTFT gate; token rate improves from clean reproducible `1.6` to `1.9`.
- Reproduction record: run directory now contains `source_state_precommit.txt`, `source_diff_precommit.patch`, `source_diff_precommit.stat`, `binary_sha256_precommit.txt`, `binary_stat_precommit.txt`, `binary_version_precommit.txt`, `model_stat.txt`, `runner_sha256_precommit.txt`, `plan_snapshot_precommit.md`, `progress_snapshot_precommit.md`, plus exact command/env/stdout/stderr/summary/trace/cgroup files.
- Action: immediately commit and push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before final promotion.
