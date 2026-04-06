# Schema-Driven Graph Construction — Proof of Concept

This branch demonstrates that llama.cpp can build computation graphs
automatically from HuggingFace model files, **without any per-model
C++ code**.

## What This Proves

1. A new model can be converted and served with **zero C++ changes**
2. The graph schema is **derived automatically** from `config.json` + tensor names
3. **2,415 lines of per-model C++** have been eliminated
4. All **327 architecture tests** still pass

## Quick Start (5 minutes)

### Prerequisites

```bash
# CUDA toolkit (for GPU inference)
# Python 3.10+ with pip
pip install torch transformers sentencepiece safetensors huggingface_hub
```

### Step 1: Clone and Build

```bash
git clone https://github.com/nisparks/llama.cpp.git
cd llama.cpp
git checkout feat/graph-schema

cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-cli
```

### Step 2: Download a Model

Pick any standard transformer from HuggingFace:

```bash
# Qwen3 (newest, 0.6B — fast to download)
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/qwen3-0.6b

# Or Qwen2.5 (0.5B)
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir models/qwen2.5-0.5b

# Or Gemma2 (2B — needs HF auth + license acceptance)
huggingface-cli download google/gemma-2-2b --local-dir models/gemma2-2b
```

### Step 3: Verify the Schema Deriver

See what the deriver extracts from the model files — **zero hardcoded knowledge**:

```bash
python scripts/derive_schema.py models/qwen3-0.6b/
```

Output (derived automatically from config.json + safetensors):
```json
{
  "version": 1,
  "model_type": "causal_lm",
  "tensors": {
    "global": {"token_embd": {"shape": ["n_embd", "n_vocab"]}, "output_norm": {"shape": ["n_embd"]}},
    "per_layer": {
      "attn_norm": {"shape": ["n_embd"]},
      "attn_q": {"shape": ["n_embd", "n_embd_head_k * n_head"]},
      "attn_k": {"shape": ["n_embd", "n_embd_k_gqa"]},
      "attn_v": {"shape": ["n_embd", "n_embd_v_gqa"]},
      "attn_out": {"shape": ["n_embd_head_k * n_head", "n_embd"]},
      "attn_q_norm": {"shape": ["n_embd_head_k"]},
      "attn_k_norm": {"shape": ["n_embd_head_k"]},
      "ffn_norm": {"shape": ["n_embd"]},
      "ffn_gate": {"shape": ["n_embd", "n_ff"]},
      "ffn_up": {"shape": ["n_embd", "n_ff"]},
      "ffn_down": {"shape": ["n_ff", "n_embd"]}
    }
  },
  "embedding": {"token_embd": {}, "position_embd": "rope"},
  "layers": {
    "default": {
      "pre_norm": {"type": "rms_norm"},
      "attention": {"type": "standard", "qkv": "separate", "qk_norm": true, "rope": {"type": "neox"}},
      "ffn_norm": {"type": "rms_norm"},
      "ffn": {"type": "gated", "activation": "silu"},
      "residual": {"type": "sequential"}
    }
  },
  "output": {"norm": {"type": "rms_norm"}, "projection": {"tied_to": "token_embd"}}
}
```

Every feature above was detected from the actual model files:
- `"qk_norm": true` — found `q_norm` tensors in safetensors
- `"activation": "silu"` — from `config.json` `hidden_act` field
- `"type": "gated"` — found `gate_proj` tensors in safetensors
- `"tied_to": "token_embd"` — from `config.json` `tie_word_embeddings` field

### Step 4: Convert to GGUF

The converter automatically calls the schema deriver and writes
the graph metadata into the GGUF:

```bash
python convert_hf_to_gguf.py models/qwen3-0.6b/ \
    --outfile models/qwen3-0.6b.gguf \
    --outtype f16
```

### Step 5: Run Inference

```bash
./build/bin/llama-cli \
    -m models/qwen3-0.6b.gguf \
    -ngl 99 \
    -p "The meaning of life is" \
    -n 50
```

Expected output: coherent text generation at 200+ tokens/sec on GPU.

## What's Different From Upstream

### Removed: 2,415 lines of per-model C++

The upstream `llama-model.cpp` has ~137 per-architecture switch cases
across three functions. This branch collapses them into generic defaults:

| Function | Upstream Cases | This Branch |
|----------|---------------|-------------|
| `build_graph()` | 56 per-arch configs | Generic default (auto-configured) |
| `load_tensors()` | 51 per-arch tensor loaders | Generic default (tensor-set-gated) |
| `load_hparams()` | 30 per-arch hparam loaders | Generic default |

### Added: Schema Deriver (679 lines of Python)

`scripts/derive_schema.py` reads two files from any HF model:
- `config.json` — activation, norm type, MoE, RoPE, tied embeddings, SSM params, etc.
- `model.safetensors.index.json` — tensor names → QKV style, FFN type, norms, SSM, MoE, etc.

Covers all **125 architectures** in llama.cpp across 8 categories:
- Standard transformers, MoE, SSM/Mamba, hybrid SSM+attention,
  encoder-only (BERT), encoder-decoder (T5), RWKV, and DeepSeek MLA
- Plus architecture-specific features: AltUp, Laurel, DSA, short
  convolution, chunked attention, interleaved MoE, dual-path MoE

No per-model knowledge. No lookup tables. Every feature derived from data.

### Added: Generic Builder (831 lines of C++)

`src/llama-graph-builder.cpp` — a single composable builder that handles
all standard transformer variants through configuration flags, replacing
the 56 per-model builder classes.

## How It Works

```
HuggingFace Model Files
    │
    ├── config.json ──────────────┐
    │                             │
    └── model.safetensors ────────┤
                                  ▼
                        derive_schema.py
                        (zero tribal knowledge)
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  layer_operations:       │
                    │  "norm,rope,attn,filter, │
                    │   residual,ffn_norm,ffn, │
                    │   residual,cvec"         │
                    │                          │
                    │  ffn_activation: "silu"   │
                    └─────────────────────────┘
                                  │
                    convert_hf_to_gguf.py
                    (writes keys to GGUF)
                                  │
                                  ▼
                            model.gguf
                    (weights + graph metadata)
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  C++ auto-config reads   │
                    │  GGUF keys + detects     │
                    │  features from tensors   │
                    │                          │
                    │  Generic builder         │
                    │  constructs ggml graph   │
                    └─────────────────────────┘
                                  │
                                  ▼
                            Inference
```

## Validated Models

All models below were converted with the fixed pipeline and verified to have
both `ffn_activation` and `layer_operations` GGUF keys written correctly.

| Model | Features Derived | GGUF Keys | Inference |
|-------|-----------------|-----------|-----------|
| Qwen2.5-0.5B | RMS norm, separate QKV, gated SiLU, RoPE, tied embeddings | `silu` / `norm,rope,attn,filter,residual,ffn_norm,ffn,residual,cvec` | 230 t/s |
| Qwen3-0.6B | + QK-norm (auto-detected from tensors) | `silu` / `norm,qkv,qk_norm,rope,attn,filter,residual,ffn_norm,ffn,residual,cvec` | 229 t/s |
| Gemma2-2B | Post-norms, sliding window, GELU, gated FFN | `gelu_pytorch_tanh` / `norm,rope,attn,filter,post_norm,residual,ffn_norm,ffn,residual,cvec` | 147 t/s |

### GGUF Key Verification

You can verify that the schema metadata was written correctly:

```bash
python3 -c "
import sys; sys.path.insert(0, 'gguf-py')
import gguf
r = gguf.GGUFReader('models/your-model.gguf')
for f in r.fields.values():
    if 'layer_op' in f.name or 'activation' in f.name:
        print(f'{f.name}: {bytes(f.parts[f.data[0]].tolist()).decode()}')
"
```

## Running the Tests

```bash
cmake --build build -j$(nproc) --target test-llama-archs
./build/bin/test-llama-archs
# Expected: 327 OKs, 0 failures
```

## Design Decisions

### Schema metadata writes via `_write_schema_metadata()`

Many model converter classes (Gemma2, Phi3, etc.) override
`set_gguf_parameters()` without calling `super()`. To ensure that
`ffn_activation` and `layer_operations` are always written regardless
of per-model overrides, we use a separate `_write_schema_metadata()`
method that runs after `set_gguf_parameters()` in the
`prepare_metadata_with_tokens()` lifecycle.

### Derive from data, never from model names

`derive_schema.py` never checks `model_type` or architecture name to
decide behavior. Every feature (QK-norm, gated FFN, post-attention norms,
MoE, RoPE) is detected from explicit config keys or tensor name patterns.
If a feature can't be determined from the data, the schema says "unknown"
— it never guesses.
