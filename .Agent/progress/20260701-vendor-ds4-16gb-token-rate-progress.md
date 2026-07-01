# Vendor DeepSeek V4 16GB Token Rate Progress

## 2026-07-01

- Initialized vendor-only optimization plan for `/root/lfz/vendor/llama.cpp-deepseek-v4`.
- Current branch before implementation: `feat/ds4-moe-stream-on-vendor`.
- Current HEAD before implementation: `ef19fbdbf cuda: optionally drop streamed expert mmap pages`.
- Known invalid high-rate reference: `/root/lfz/runs/vendor-sota/20260701T025805Z/france-cpu60`, `eval_tok_s=9.9`, but max RSS was about 67.2 GiB, so it violates the 16 GB host RAM gate.
- Known clean 16 GB reference: `/root/lfz/runs/vendor-sota-ram16/20260701T031526Z-16gb-clean-final/france-cpu40`, `eval_tok_s=1.1`, `prompt_tok_s=0.8`, `memory_peak_bytes=16000000000`, answer was coherent.
- Action required before optimization acceptance: rerun the vendor baseline under the new strict runner and use that as the official baseline.

## Run Log

No accepted optimization runs yet.

