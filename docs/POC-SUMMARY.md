# Schema-Driven Graph Construction — Proof of Concept

## Motivation

llama.cpp supports over 100 model architectures, each with per-arch handling
in `load_hparams()`, `load_tensors()`, and `build_graph()`. The existing
approach is proven and thorough — it's what got the project to 100+ models.

As the number of architectures grows, though, most new models are standard
transformer variants that share the same graph topology, differing only in
configuration (norm type, activation, QKV layout, etc.). This POC explores
whether those common cases can be handled by a single generic path, reducing
the per-model surface area while preserving the existing custom builders for
architectures that genuinely need them.

## Approach

A Python script (`derive_schema.py`, 679 lines) reads a HuggingFace model's
`config.json` and `model.safetensors.index.json` (the tensor name index). From
those two files it determines the graph configuration: norm type, QKV layout,
attention style, FFN gating, activation function, RoPE settings, tied
embeddings, SSM parameters, MoE routing, hybrid layer patterns, and more.

The deriver covers all **125 architectures** currently supported by llama.cpp,
across 8 categories: standard transformers, MoE, SSM/Mamba, hybrid
SSM+attention, encoder-only (BERT), encoder-decoder (T5), RWKV, and
multi-head latent attention (DeepSeek MLA). Architecture-specific features
(AltUp, Laurel, DSA, short convolution, chunked attention, dual-path MoE)
are detected and recorded in a dedicated `features` section of the schema.

During GGUF conversion, the deriver runs automatically and writes two metadata
keys into the file:

- **`layer_operations`** — the sequence of operations per transformer layer
- **`ffn_activation`** — the activation function used in the feed-forward network

On the C++ side, a data-driven builder reads those keys and assembles the
computation graph operation by operation — it follows the blueprint that was
derived from the model's own files. There is no hardcoded default behavior;
the GGUF keys *are* the configuration. Without them, the builder has nothing
to build.

Architectures with non-transformer topologies or unusual structural
requirements would still need custom builders, but standard transformer
variants — which make up the majority of supported models — can be fully
described by these two keys.

## What Changed

**2,415 lines of per-model C++ removed.** These were the repetitive per-arch
switch cases where each new model required its own block of hparam loading,
tensor loading, and graph building code — often duplicating logic already
present in other architectures. The 137 cases across three functions were
consolidated into data-driven defaults:

| Function | Upstream Cases | Lines Removed | This Branch |
|----------|---------------|---------------|-------------|
| `build_graph()` | 56 per-arch configs | ~570 | Data-driven builder, configured by GGUF keys |
| `load_tensors()` | 51 per-arch tensor loaders | ~1,540 | Generic loader, gated by tensor set |
| `load_hparams()` | 30 per-arch hparam loaders | ~305 | Generic default |

### Added

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/derive_schema.py` | 679 | Derives graph schema from HF model files |
| `src/llama-graph-builder.cpp` | 831 | Data-driven builder + auto-config from GGUF keys |
| `src/llama-graph-builder.h` | 123 | Builder config struct |

Net delta: **+104 lines** (2,520 added, 2,416 removed).

## Results

Three models were downloaded from HuggingFace, converted, and served using
only the generic path — no per-model code was involved:

| Model | Architecture | Activation | Speed |
|-------|-------------|------------|-------|
| Qwen2.5-0.5B | qwen2 | silu | 230 t/s |
| Qwen3-0.6B | qwen3 | silu | 229 t/s |
| Gemma2-2B | gemma2 | gelu | 147 t/s |

All **327 architecture unit tests** pass unchanged.

## What This Could Enable

Currently, adding support for a new standard transformer model requires
writing C++ across three functions — hparam loading, tensor loading, and graph
construction — even when the new model's topology is identical to an existing
one. With this approach, the workflow for a new model becomes:

```bash
python convert_hf_to_gguf.py <model_dir> --outfile model.gguf
./llama-cli -m model.gguf
```

No C++ changes, no PR, no review cycle. The schema deriver reads the model's
own files and the builder constructs the graph from those derived instructions.
A new standard transformer architecture could go from HuggingFace release to
running inference in minutes rather than days.

## Scope and Next Steps

This is a proof of concept. The schema deriver can identify and classify all
**125 architectures** currently in llama.cpp — standard transformers, MoE,
SSM/Mamba, hybrid SSM+attention, encoder-only, encoder-decoder, RWKV, MLA,
and architecture-specific features like AltUp (Gemma3N), DSA (GLM), short
convolution (LFM2), chunked attention (LLAMA4), and dual-path MoE (Gemma4).

The end-to-end serving pipeline (derive → convert → serve) has been validated
on standard transformer variants. Extending the C++ builder to act on the
full vocabulary is the next phase. The plan is to build this out further to
cover all known architecture categories end-to-end.
