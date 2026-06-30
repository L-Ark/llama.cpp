# Vendor DeepSeek V4 Optimization Progress

## 2026-06-30 10:20 CST - Constraint Restatement

User restated the hard gates for all vendor optimization work:

- Host RAM strictly below 16 GB, including page cache.
- Use VRAM as fully as possible.
- `n64` is allowed only for fast speed screening.
- Final acceptance requires at least `n96`, with no garbled tail.
- TTFT must not increase by more than 20%.

Conclusion: treat these as hard acceptance gates. Any optimization that misses one gate is rejected or reverted.

## 2026-06-30 10:22 CST - Strict Baseline Design

Repository: `/root/lfz/vendor/llama.cpp-deepseek-v4`

Baseline command shape:

- Binary: `build-a101-nvcc/bin/llama-completion`
- Model: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`
- Prompt: `Question: Please introduce France in a short paragraph.\n\nAnswer:`
- Flags: `-ngl 8 -c 512 -n 96 --ignore-eos --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-display-prompt -no-cnv -ub 1 -t 24 -tb 24`
- Environment: `CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 GGML_MOE_VRAM_CACHE_MIB=16384 IK_LLAMA_DROP_DEFERRED_EXPERT_PAGES=1`
- Service limits: `MemoryMax=14G`, `MemorySwapMax=0`
- Required monitor: continuous cgroup `memory.current`, `memory.peak`, `memory.stat`, plus `nvidia-smi` samples.

Expected reference from previous non-final vendor runs: coherent output, decode approximately `0.57-0.59 tok/s`, VRAM approximately `25.9 GiB`, with host mapped model much larger than cgroup resident pages.

## 2026-06-30 10:39 CST - S1/S2 Preflight And Harness Receipt

Preflight passed on `/root/lfz/vendor/llama.cpp-deepseek-v4`:

- Branch: `wip/deepseek-v4-support`.
- Push remote: `lark git@github.com:L-Ark/llama.cpp.git`.
- Required binaries found: `build-a101-nvcc/bin/llama-completion`, `build-a101-nvcc/bin/llama-ds4-expert-profile`.
- Model found: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`.
- Required tools found: `nvidia-smi`, `systemd-run`, `python3`.
- GPU free at preflight: RTX 5090, 32607 MiB total, 31807 MiB free.

Added `.Agent/run-tools/strict-ds4-run.sh` and Python helpers. The harness:

- Runs the benchmark through `systemd-run --wait --collect` with `MemoryMax=14G` and `MemorySwapMax=0`.
- Samples live cgroup `memory.current`, `memory.peak`, and `memory.stat`.
- Samples `nvidia-smi` during the run.
- Records command/env/git status/stdout/stderr, estimates TTFT as first observed stdout byte with the same method for baseline/candidates, and writes `summary.json` plus `summary.md`.
- Defaults to `CUDA_VISIBLE_DEVICES=0`, `GGML_CUDA_NO_PINNED=1`, `GGML_MOE_VRAM_CACHE_MIB=16384`, and `LLAMA_MMAP_LOW_RAM=1`.

Dry-run validation passed:

```bash
bash -n .Agent/run-tools/strict-ds4-run.sh
python3 -m py_compile .Agent/run-tools/strict_ds4_runner.py .Agent/run-tools/check-ds4-summary.py .Agent/run-tools/compare-ds4-runs.py
.Agent/run-tools/strict-ds4-run.sh --dry-run
grep -E "MemoryMax=14G|MemorySwapMax=0|memory.peak|nvidia-smi|llama-completion|LLAMA_MMAP_LOW_RAM" /tmp/ds4-strict-dry-run.log
```

Decision packet:

- Problem: older reference runs are not sufficient because final gates require live host RAM including page cache and same-method TTFT.
- Inferred invariants: the vendor checkout and `.Agent` plan own this work; strict acceptance is cgroup-limited service execution; output text, memory, VRAM, and timing must be preserved per run.
- Options considered: ad hoc shell wrappers, systemd service plus external monitor, or modifying benchmark code for instrumentation.
- Chosen approach: external systemd harness first, because it records the acceptance evidence without touching model execution code.
- Risks: first-byte TTFT is measured from redirected stdout visibility, so it is an operational TTFT proxy rather than internal token timing; it remains valid for baseline/candidate comparison because the same harness is used.
- Validation: dry-run and syntax/py_compile checks passed; next step is strict `n64` path check followed by strict `n96` baseline.

## 2026-06-30 11:03 CST - S3 First Baseline Attempt Rejected

Run: `.Agent/runs/20260630-ds4-token-rate/baseline-n64`

Result: rejected as a baseline record.

Evidence:

- Command used strict `MemoryMax=14G`, `MemorySwapMax=0`, `LLAMA_MMAP_LOW_RAM=1`, `GGML_MOE_VRAM_CACHE_MIB=16384`, `-ngl 8`, and default warmup.
- Live cgroup sampling worked; peak was `15032385536` bytes, below the final `<16000000000` byte gate.
- VRAM peak was about `25850` MiB.
- The process reached model load, scheduler reserve, warmup, and started generation. Output text was coherent France text.
- The run was stopped after about 19 minutes because it remained dominated by strict low-RAM load/warmup/page reclaim and had not produced a final llama perf block. Therefore `decode_tokens_per_second` was missing and the record cannot be used as S3 baseline.
- Source inspection found `LLAMA_MMAP_LOW_RAM` implemented in `src/llama-mmap.cpp`; no implementation of `IK_LLAMA_DROP_DEFERRED_EXPERT_PAGES` exists in this checkout, so that old progress variable is not a valid knob here.
- `llama-completion --help` confirms `--warmup, --no-warmup` is available and warmup defaults to enabled.

Decision packet:

- Problem: the planned strict baseline command spends excessive time in default warmup/load under the 14G cgroup cap and does not yield timely perf evidence.
- Inferred invariants: final acceptance compares baseline and candidate with the same harness and command policy; the user requires TTFT not to regress versus the strict baseline, not default warmup specifically.
- Options considered: repeat the same default-warmup run, raise MemoryMax, add unsupported page-drop env, or disable warmup for strict baseline/candidate measurements.
- Chosen approach: add `--no-warmup` to the harness command for all strict measurements. This avoids measuring an empty-run warmup page-reclaim path, keeps the cgroup cap unchanged, and applies equally to baseline and candidate.
- Risks: TTFT baseline will be lower than default-warmup TTFT, making the later 120% TTFT gate stricter. This is acceptable because it does not mask a regression.
- Validation: rerun `baseline-n64` using a fresh label, then promote to `baseline-n96` only if the no-warmup run produces memory, TTFT, output, and decode timing.


## 2026-06-30 11:22 CST - S3 Strict Baseline Accepted

Run directory: `.Agent/runs/20260630-ds4-token-rate/baseline-n96`

Strict `n96` baseline passed the current acceptance checks using the same harness and command policy that candidates must use.

Evidence:

- Git SHA: `5d360bc3dfefb3f6f727ceb6c412bf7a1bf73aee`.
- Command policy: `MemoryMax=14G`, `MemorySwapMax=0`, `LLAMA_MMAP_LOW_RAM=1`, `GGML_CUDA_NO_PINNED=1`, `GGML_MOE_VRAM_CACHE_MIB=16384`, `-ngl 8`, `--no-warmup`, deterministic greedy sampling, `n_predict=96`.
- Decode token rate: `0.24 tok/s`.
- TTFT: `753.0239360332489 s` by the strict harness first-stdout-byte method.
- Host memory peak: `15032385536` bytes, below the required `<16000000000` byte gate, with `1761` memory samples.
- GPU memory peak: `25854 MiB`.
- Wall time: `941.1688921451569 s`.
- Output quality: pass; coherent France paragraph and no malformed tail.
- Exact output tail: `...attracting millions of visitors each year to its cities, countryside, and coastline. The official language is French, and the currency is the`.

Decision packet:

- Problem: establish a strict current baseline before accepting any optimization.
- Inferred invariants: strict acceptance is tied to the vendor checkout, branch `wip/deepseek-v4-support`, live cgroup memory including file/page-cache accounting, deterministic France prompt output, and same-method TTFT for baseline/candidate comparison.
- Options considered: rerun the default-warmup baseline, accept old reference runs, or accept the completed no-warmup strict baseline.
- Chosen approach: accept the completed no-warmup strict baseline because it uses the validated cgroup/GPU harness, preserves memory and output evidence, and imposes a stricter TTFT comparison for later candidates.
- Risks: decode rate is lower than older references, likely because strict low-RAM/page-cache behavior dominates; future tuning must compare against this exact strict method and cannot use `n64` as final evidence.
- Validation: `python3 .Agent/run-tools/check-ds4-summary.py .Agent/runs/20260630-ds4-token-rate/baseline-n96/summary.json --max-memory-bytes 16000000000 --require-n-predict 96 --require-output-quality pass` returned PASS.
