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
