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
