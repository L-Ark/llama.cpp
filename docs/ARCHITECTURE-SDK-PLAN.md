# llama.cpp Fork — Architecture SDK & Dynamic Model Discovery

## Repository
- **Fork**: `nisparks/llama.cpp`
- **Upstream**: `ggml-org/llama.cpp`
- **Status**: Fork reset to upstream HEAD, tracking upstream on `master`; work on `feature/arch-sdk`

## Vision
Eliminate the manual toil of adding new model architectures to llama.cpp by:
1. **Phase 1**: Document the current process with a comprehensive SDK/guide
2. **Phase 2**: Refactor into composable building blocks
3. **Phase 3**: Dynamic architecture discovery from model config

## The Problem
Every new model architecture requires ~500-2000 lines of hand-coded C++ in
`src/llama-model.cpp` and related files. This involves:
- Reading the model's config.json and modeling_*.py from HuggingFace
- Manually mapping tensor names to GGML tensor names
- Coding up the attention pattern (MHA/GQA/MQA/sliding window)
- Coding up the FFN variant (SwiGLU, GeGLU, etc.)
- Coding up normalization (RMSNorm, LayerNorm, pre/post)
- Coding up position encoding (RoPE, RoPE-scaled, ALiBi, YaRN)
- Handling special features (MoE routing, cross-attention, etc.)
- PR review cycle: ~1-3 weeks per architecture

This is unsustainable with the pace of new model releases.

---

## Phase 1: Architecture SDK & Documentation

**Goal**: Make it trivially easy for anyone to add a new architecture by following
a clear guide, templates, and tooling. Don't change the architecture yet — just
document and scaffold what exists.

### Deliverables

#### 1.1 Architecture Contributor Guide
- Step-by-step walkthrough: "How to add a new model architecture to llama.cpp"
- Covers the full pipeline: GGUF conversion → tensor loading → graph construction → testing
- Real worked example with a simple model (e.g., adding a hypothetical "MiniLM" from scratch)

#### 1.2 Architecture Anatomy Reference
- Document every existing architecture in a standardized format:
  ```
  Model: Qwen3
  Config fields: num_hidden_layers, num_attention_heads, num_key_value_heads, ...
  Attention: GQA (heads=40, kv_heads=8)
  FFN: SwiGLU (hidden=5120, intermediate=27648)
  Norm: RMSNorm (pre-norm, eps=1e-6)
  Position: RoPE (theta=1e6)
  Special: none
  Tensor map: {attn.q_proj → blk.N.attn_q.weight, ...}
  Files touched: llama-model.cpp, llama-model-loader.cpp, llama-arch.h
  ```
- Generate this for all 50+ supported architectures

#### 1.3 Architecture Scaffold CLI Tool
- `./scripts/new-arch.py --name mymodel --config path/to/config.json`
- Reads config.json, infers architecture components
- Generates skeleton C++ code with TODOs for manual review
- Generates tensor name mapping boilerplate
- Generates GGUF conversion script additions

#### 1.4 Architecture Test Harness
- Standardized test for any architecture: load model, run inference, compare output
- Golden output tests for each supported architecture
- CI integration: new arch PRs must pass the test harness

#### 1.5 Architecture Component Catalog
- Catalog of all building blocks used across architectures:
  ```
  Attention types:   MHA, GQA, MQA, SlidingWindow, CrossAttention
  FFN types:         SwiGLU, GeGLU, GELU, ReLU, MoE
  Norm types:        RMSNorm, LayerNorm (pre/post)
  Position types:    RoPE, RoPE-NTK, RoPE-YaRN, ALiBi, NoPE
  Embeddings:        Standard, TiedWeights
  Special:           MoE routing, interleaved attention, vision projector
  ```
- For each component: which models use it, what config fields control it,
  where the C++ implementation lives

### Files Involved (current codebase)
```
src/llama-arch.h              # Architecture enum + tensor name mappings
src/llama-arch.cpp            # Tensor name string tables
src/llama-model.cpp           # Model graph construction (THE big file)
src/llama-model-loader.cpp    # GGUF tensor loading
src/llama-hparams.h           # Hyperparameter structs
src/llama-context.cpp         # Context/KV cache management
convert_hf_to_gguf.py         # HuggingFace → GGUF conversion
gguf-py/gguf/constants.py     # GGUF tensor name constants
```

---

## Phase 2: Composable Block Refactor

**Goal**: Refactor the monolithic `llama-model.cpp` into composable, reusable blocks
that can be assembled declaratively. Maintain backwards compatibility.

### Deliverables

#### 2.1 Block Interface Definition
```cpp
struct llm_block {
    virtual ggml_tensor * build(ggml_context * ctx, ...) = 0;
};

struct llm_attention_gqa : llm_block { ... };
struct llm_ffn_swiglu : llm_block { ... };
struct llm_norm_rms : llm_block { ... };
struct llm_rope : llm_block { ... };
```

#### 2.2 Architecture as Block Composition
```cpp
// Instead of 500 lines of hand-coded graph construction:
auto arch = llm_architecture()
    .embedding(standard_embedding)
    .repeat(n_layers, [](auto layer) {
        layer.pre_norm(rms_norm(eps))
             .attention(gqa_attention(n_heads, n_kv_heads))
             .post_norm(rms_norm(eps))
             .ffn(swiglu_ffn(hidden_size, intermediate_size));
    })
    .final_norm(rms_norm(eps))
    .lm_head(tied_weights ? embed : separate);
```

#### 2.3 Migrate Existing Architectures
- Convert top 10 architectures to block composition
- Verify identical output via test harness
- Keep legacy code path as fallback

#### 2.4 Universal Tensor Name Convention
- Define a standard naming scheme that works for all architectures
- Update GGUF converter to use universal names
- Backward compatibility: support both old and new names

---

## Phase 3: Dynamic Architecture Discovery

**Goal**: New models work automatically if they use standard transformer blocks.
Config.json → compute graph with zero C++ changes.

### Deliverables

#### 3.1 Config Parser
- Read model config from GGUF metadata (already stored during conversion)
- Map config fields to block types automatically
- Handle aliases (different models use different field names for the same thing)

#### 3.2 Architecture Registry
- Registry of known block types with config → block mapping rules
- Fallback: if config doesn't match known patterns, error with guidance

#### 3.3 Graph Builder
- Assemble compute graph from config + block registry
- No per-model C++ code needed for standard transformer variants
- Automatic tensor name derivation from config

#### 3.4 Escape Hatch: Custom Blocks
- Plugin system for novel components
- Register custom blocks that can be composed into architectures
- Enables researchers to experiment without forking

---

---

## Testing Strategy

### Existing Coverage (inherited from upstream)
- `test-llama-archs.cpp` — validates all 126 architecture compute graphs build correctly
- `test-backend-ops.cpp` — validates all GGML operations across backends
- `tools/server/tests/` — 23 pytest modules for server API
- Sanitizers (ASan, TSan, UBSan) in CI

### New Tests Required

#### T1: Golden Output Regression Suite (P0 — before any refactor)
- **Purpose**: Capture reference outputs for key architectures, compare after changes
- **Method**: For each architecture family, run inference on a tiny synthetic model
  with a fixed prompt/seed, store output logits as golden reference
- **Architectures to cover first**: LLaMA, Qwen3, Gemma4, DeepSeek2, Mamba2, T5
- **File**: `tests/test-golden-outputs.cpp`
- **CI**: Run on every PR to `feature/arch-sdk`

#### T2: Block Equivalence Tests (Phase 2)
- **Purpose**: Prove each composable block produces identical GGML graph to original
- **Method**: Build graph with original monolithic code, build with new block composition,
  compare tensor-by-tensor
- **File**: `tests/test-block-equivalence.cpp`

#### T3: Config-to-Graph Roundtrip Tests (Phase 3)
- **Purpose**: Prove dynamic discovery produces identical graph to hand-coded version
- **Method**: For each architecture, build graph from config.json dynamically, compare
  to existing hand-coded graph
- **File**: `tests/test-config-roundtrip.cpp`

#### T4: Conversion Pipeline Tests (new)
- **Purpose**: Validate HF→GGUF conversion produces correct output
- **Method**: Convert small HF models, load GGUF, compare tensor values to originals
- **File**: `tests/test-convert-hf.py`

#### T5: Performance Regression Tracking (P2)
- **Purpose**: Detect inference speed and memory regressions
- **Method**: Benchmark key operations, compare to baseline, alert on >5% regression
- **File**: `tests/test-perf-regression.cpp`
- **CI**: Weekly scheduled run, results stored in artifact

---

## Additional Contributions (from other plans)

### TurboQuant KV Cache Port
- Cherry-pick turbo2/turbo3/turbo4 KV cache types from TurboQuant fork
- Add CUDA kernels for TurboQuant quantization/dequantization
- Benchmark and document quality vs compression tradeoffs
- Submit as PR to upstream

### Management Dashboard
- See: llm-dashboard-plan.md
- SvelteKit app on :3000 for model management, HF downloads, MCP/Copilot

---

## Getting Started

The fork has been reset and the feature branch is active:

```bash
git clone https://github.com/nisparks/llama.cpp.git
cd llama.cpp
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git checkout feature/arch-sdk
```

To stay current with upstream:
```bash
git fetch upstream
git rebase upstream/master
git push origin feature/arch-sdk --force-with-lease
```

### Phase 1 Deliverables (Complete)
- **1.1** Architecture Contributor Guide — `docs/ARCHITECTURE-CONTRIBUTOR-GUIDE.md`
- **1.2** Architecture Anatomy Reference — `docs/ARCHITECTURE-CATALOG.md`
- **1.3** Architecture Scaffold CLI Tool — `scripts/new-arch.py`
- **1.4** Architecture Test Harness — `tests/test-golden-outputs.cpp`
- **1.5** Architecture Component Catalog — included in ARCHITECTURE-CATALOG.md

### Phase 2 (Complete)
- **2.1** Block Interface — `llm_build_std_transformer` with 22 config flags
- **2.2** End-to-End Scaffold — `scripts/new-arch.py --apply --std-transformer`

### Phase 3 (Complete)
- **3.1** `llm_transformer_config_from_hparams()` — auto-detects all config
  parameters from model tensors and hparams at runtime
- **3.2** Auto-fallback in `build_graph()` — any unknown architecture that
  passes auto-config uses the generic builder automatically
- **3.3** FFN activation type stored in GGUF metadata
  (`{arch}.feed_forward_activation`) — upstream discussion started
- **3.4** 77 of 112 architectures migrated to generic builder (65%)
  across 13 migration waves. 96% reduction in graph builder boilerplate.

### Remaining custom builders (35 architectures)
These require genuinely different compute graph structures:
- 16 SSM/hybrid (Mamba, RWKV, Delta-Net) — not transformers
- 6 encoder/decoder (BERT, T5) — different graph topology
- 13 unique features (MLA, altup, binary weights, grouped experts, etc.)

---

## Success Criteria — Results
- **Phase 1**: A contributor can add a new standard transformer architecture in
  <5 minutes using `new-arch.py --apply` (vs. days before) ✅
- **Phase 2**: Adding a new architecture requires 2-8 lines of config
  (vs. 100-200 lines of C++ before) ✅
- **Phase 3**: Standard transformer models work with zero graph builder code
  via the auto-fallback `default:` path ✅
