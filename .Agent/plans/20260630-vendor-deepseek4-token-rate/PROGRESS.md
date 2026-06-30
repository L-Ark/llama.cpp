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
