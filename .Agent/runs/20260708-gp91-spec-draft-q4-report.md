# GP91 Kimi model-based speculative draft smoke

This is a dev-only compatibility and feasibility smoke. It does not change runtime behavior and does not claim SOTA.

## Asset

- Draft model: `Kimi-K2-Instruct-DRAFT-0.6B-32k-Q4_0.gguf`
- Source URL: `https://huggingface.co/jukofyork/Kimi-K2-Instruct-DRAFT-0.6B-v3.0-GGUF/resolve/main/Kimi-K2-Instruct-DRAFT-0.6B-32k-Q4_0.gguf`
- Remote path: `/root/lfz/models/kimi-draft/Kimi-K2-Instruct-DRAFT-0.6B-32k-Q4_0.gguf`
- Size: `427M` on disk, GGUF-reported `419.73 MiB`
- SHA256: `d2d602be55b40bdac44ed17a6d24f69237a0ae7c3da69b24d13fc54c686c93d8`
- Draft architecture: `qwen2`, `651.50M` params, `163840` vocab, tokenizer pre `kimi-k2`.

## Build

- Built diagnostic binary on remote:

```bash
cd /root/lfz/tmp/kimi-stage2m-align
cmake --build build-cuda-batch --target llama-speculative-simple -j 16
```

## Runs

| run | prompt | draft placement | result | decision |
|---|---|---|---|---|
| `.Agent/runs/20260708-gp91-spec-draft-q4-france-n32-v3` | raw Kimi ChatML France, N32 | GPU draft, default 32k ctx | draft context OOM: failed to allocate `321.75 MiB` after draft load | reject |
| `.Agent/runs/20260708-gp91-spec-draft-q4-france-n32-cpudraft` | raw Kimi ChatML France, N32 | CPU draft, draft ctx 512 | target/draft vocabs reported incompatible; assert `failed to detokenize id_last` | reject |
| `.Agent/runs/20260708-gp91-spec-draft-q4-plaintext-n16-cpudraft` | plain France, N16 | CPU draft, draft ctx 512 | began output, but remained too slow and was terminated; vocab incompatible warning present | reject |
| `.Agent/runs/20260708-gp91-spec-draft-q4-plaintext-n16-gpudraft512` | plain France, N16 | GPU draft, draft ctx 512 | began output, but remained too slow and was terminated after >3 min; vocab incompatible warning present | reject |

## Key Evidence

- The target model and draft model do not pass clean vocab compatibility:

```text
the target and draft vocabs are not compatible - tokens will be translated between the two
```

- Raw ChatML prompt fails in the speculative translation path:

```text
GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last") failed
```

- Default GPU draft fails because current SOTA VRAM cache leaves too little headroom for the draft model at its default 32k context:

```text
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 321.75 MiB on device 0: cudaMalloc failed: out of memory
failed to create draft context
```

- Plain-text CPU/GPU draft can begin generation, but N16 did not finish within a useful time window and is slower than the accepted non-speculative baseline. This is not a viable token-rate path as-is.

## Decision

Reject `Kimi-K2-Instruct-DRAFT-0.6B-32k-Q4_0` with the current `llama-speculative-simple` path as a primary Kimi token-rate optimization. It has compatibility issues with the current Kimi-K2.7-Code target and does not show a plausible performance path under the 16 GB RAM / 32 GB VRAM target.

Model-based speculation should only be revisited with either:

- a draft model proven tokenizer/special-token compatible with Kimi-K2.7-Code, or
- a patched speculative path that handles Kimi special tokens safely and then demonstrates high acceptance plus a real speedup on dev prompts.
