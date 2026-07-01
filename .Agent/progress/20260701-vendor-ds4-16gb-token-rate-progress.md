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
