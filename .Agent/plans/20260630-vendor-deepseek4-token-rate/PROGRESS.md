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
