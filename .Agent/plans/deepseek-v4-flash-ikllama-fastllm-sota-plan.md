# DeepSeek V4 Flash ik_llama FastLLM SOTA Plan

## Goal

Integrate DeepSeek V4 Flash into `ik_llama` using the quantization format closest to the fastLLM production baseline, then measure and improve strict-16GB host-RAM token rate until it reaches or exceeds the fastLLM SOTA.

Primary comparison target:

- fastLLM reference run: `/root/lfz/fastllm_runs/strict16g-sota-wrapper-exactenv-repeat1-20260622/repeat1/gate/summary.json`
- fastLLM strict16g SOTA: `decode_tok_s_p50 = 2.81 tok/s`
- Reconfirm reference: `/root/lfz/fastllm_runs/strict16g-sota-wrapper-exactenv-repeat3-20260622/repeat3/gate/summary.json`, around `2.78 tok/s`

Hard runtime constraint for reported results:

- Host RAM must be limited to `<= 16GB` with cgroup/systemd for all main comparisons.
- Unlimited-RAM runs may only be diagnostic and must not be reported as SOTA.

## Model Format

Use the native FP4/FP8 GGUF that is closest to the original fastLLM safetensors path:

- Hugging Face repo: `nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF`
- File: `DeepSeek-V4-Flash-FP4-FP8-native.gguf`
- Expected size: about `146GB`
- Expected quantization: dense FP8 (`F8_E4M3_B128`) plus routed experts in MXFP4 / FP4-equivalent layout

Do not use low-bit q2/q4 community GGUF as the main comparison, because that is not equivalent to fastLLM original FP4/FP8 DeepSeek V4 Flash path.

Storage policy:

- Delete the old `/root/lfz/models/DeepSeek-V4-Flash` HF/safetensors directory after recording its size and checksum metadata.
- Download the native GGUF to `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/`.
- Record download and inventory artifacts under `/root/lfz/fastllm_runs/ikllama-deepseek-v4-flash-fp4fp8-20260622/`.

## Implementation Plan

1. **Inventory and download**
   - Stop stale local GPU jobs from previous tasks.
   - Record old model directory contents, `du -sh`, and `config.json` hash before deletion.
   - Delete old safetensors to free disk space.
   - Download the native FP4/FP8 GGUF with resumable tooling.
   - Verify GGUF header, tensor count, `general.architecture`, and quant types with `gguf_dump.py`.

2. **DeepSeek4 loader support**
   - Add a distinct `LLM_ARCH_DEEPSEEK4`; do not silently alias it to `DEEPSEEK2`.
   - Add GGUF metadata mapping for the DeepSeek V4 fields: 43 layers, hidden 4096, head 64, KV head 1, top-6, 256 routed experts, one shared expert, YaRN, sliding window, `sqrtsoftplus`, and `swiglu_limit=10`.
   - Add tensor name mapping for DSv4 tensors: `wq_a/wq_b/wkv/wo_a/wo_b`, `attn_sinks`, compressor/indexer tensors, shared expert tensors, routed expert tensors, and MTP tensors if present.

3. **DeepSeek4 graph and kernels**
   - Build a DSv4-specific graph rather than reusing DeepSeek2 blindly.
   - Cover MLA/NSA attention, routing with `sqrtsoftplus`, top-6 expert selection, shared expert path, and clamped SwiGLU.
   - Add or wire native GGUF quant support required by the file: `F8_E4M3_B128`, MXFP4, and any file-type enum needed by the loader.
   - First pass may disable MTP; MTP becomes a later optimization after the base graph is correct.

4. **Strict16 direct baseline**
   - Run first-forward, 8-token, 32-token, then 128-in/256-out benchmark under `MemoryMax=16G`.
   - Use `temperature=0`, `top-k=1`, `top-p=1.0`, batch 1, 3 repeats.
   - Record p50/worst decode tok/s, TTFT, prefill tok/s, output hash, RSS peak, VRAM peak, SSD read bytes if available, and failure logs.

5. **FastLLM-derived optimization ladder**
   - Apply one optimization at a time; only keep changes that improve strict16g reproducibly.
   - Ladder order:
     1. `--defer-experts` / lazy expert residency.
     2. Direct I/O or io_uring expert reads with fixed staging buffers.
     3. 1GB VRAM hot expert cache.
     4. RAM profile cache that still keeps total host RSS within 16GB.
     5. Route-trace hotset generation and profile reuse.
     6. Expert grouped reads or pack/manifest only if GGUF layout creates excessive small reads.
     7. MTP/speculative decode only after direct baseline is stable.

## Gates and Artifacts

Run directory:

`/root/lfz/fastllm_runs/ikllama-deepseek-v4-flash-fp4fp8-20260622/`

Required artifacts:

- `download/summary.json`: old model cleanup, download source, file size, checksums.
- `inventory/summary.json`: GGUF architecture, tensor count, quant types, key metadata.
- `first-forward/summary.json`: one-token survival under strict16g.
- `direct-baseline/repeat{1,2,3}/summary.json`: strict16g token-rate baseline.
- `fastllm-sota-reference.json`: copied/extracted fastLLM SOTA metrics and source paths.
- `optimization-*/summary.json`: before/after metrics for each optimization.
- `FINAL_REPORT.md`: final comparison against fastLLM SOTA and remaining bottlenecks.

Success criteria:

- Minimum integration success: DeepSeek V4 Flash native FP4/FP8 GGUF generates 256 tokens in ik_llama under 16GB host RAM.
- Performance success: strict16g p50 decode tok/s `>= 2.81` with no worse-than-baseline smoke quality.

Commit rules:

- Use author `L-Ark <fliangae@connect.ust.hk>`.
- Commit immediately after any reproducible strict16g performance improvement.
- Push committed improvements to the current remote branch.

## Implementation Status - 2026-06-22

Completed:

- Deleted the old fastLLM-format `/root/lfz/models/DeepSeek-V4-Flash` directory after recording cleanup artifacts.
- Started native FP4/FP8 GGUF download from `nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF`.
- Added first-pass `LLM_ARCH_DEEPSEEK4` loader support and GGUF tensor/metadata mapping.
- Added minimal `F8_E4M3_B128` type support and FP8 model file-type display/mapping.
- Fixed DeepSeek4 loader path so expert tensors can remain deferred instead of forcing a merged up/gate expert allocation.
- Added `sqrt(softplus)` MoE gating support.
- Added a minimal short-context `build_deepseek4()` graph and dispatch entry.
- Added conservative DeepSeek4 fallbacks that disable unsupported fused up/gate FFN and fused MoE up/gate paths for this architecture only.
- Added a CUDA Flash Attention dispatch for 512-dim Q/K/V with GQA ratios divisible by 16.
- Applied DeepSeek4 `swiglu_clamp_exp` / shared clamp limits to the active FFN and MoE elementwise paths.
- Added `inventory/summary.json` from the sparse/header GGUF probe and a watcher that launches strict16g direct baseline automatically after the full GGUF finishes downloading.
- Verified sparse/header GGUF probe can load dense tensors, defer experts, initialize KV cache, build the graph, and run one token with default Flash Attention and default fused flags using:
  - `--defer-experts`
  - `GGML_CUDA_NO_PINNED=1`
- Verified an 8-token sparse/header probe reaches `1.45 tok/s` eval speed. This is only a control-flow smoke result because the GGUF is sparse/incomplete.

Current blockers:

- The full 156,148,189,760-byte GGUF is still downloading; real token rate cannot be reported from the sparse probe.
- The current baseline uses conservative non-fused up/gate execution for DeepSeek4; restoring fused expert kernels is a performance task.
- The minimal graph intentionally skips the full DeepSeek4 HC mixer, compressed attention state, indexer path, and MTP. It is a bring-up path, not the final quality/performance implementation.

Next steps:

1. Finish the native GGUF download with resumable `aria2c`.
2. Let `watch/run_baseline_when_ready.sh` run direct baseline automatically after the full GGUF reaches `156148189760` bytes and the `.aria2` control file disappears.
3. Restore performance one subsystem at a time:
   - first Flash Attention compatibility for 512-dim latent KV,
   - then FP4/FP8 fused expert up/gate,
   - then full DeepSeek4 HC/indexer/compressed attention semantics,
   - finally fastLLM-derived expert caching and read-path optimizations.
