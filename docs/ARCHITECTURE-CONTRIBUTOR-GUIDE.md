# Architecture Contributor Guide

> **Goal:** Add a new model architecture to llama.cpp in under 2 hours.

This guide walks you through every file you need to touch, in order, with copy-paste-ready
templates. A worked example ("MyModel") is threaded throughout so you can follow along.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Quick Reference: Files to Touch](#3-quick-reference-files-to-touch)
4. [Step 1 — Register the Architecture Enum](#step-1--register-the-architecture-enum)
5. [Step 2 — Map the Architecture Name](#step-2--map-the-architecture-name)
6. [Step 3 — Declare Tensor Names](#step-3--declare-tensor-names)
7. [Step 4 — Register GGUF Constants (Python)](#step-4--register-gguf-constants-python)
8. [Step 5 — Load Hyperparameters](#step-5--load-hyperparameters)
9. [Step 6 — Load Tensors](#step-6--load-tensors)
10. [Step 7 — Build the Compute Graph](#step-7--build-the-compute-graph)
11. [Step 8 — Register the Graph Builder](#step-8--register-the-graph-builder)
12. [Step 9 — Add to the Build System](#step-9--add-to-the-build-system)
13. [Step 10 — Write the HuggingFace Converter](#step-10--write-the-huggingface-converter)
14. [Step 11 — Test Your Architecture](#step-11--test-your-architecture)
15. [Common Patterns](#common-patterns)
16. [Troubleshooting](#troubleshooting)

---

## 1. Overview

Every model in llama.cpp follows the same lifecycle:

```
HuggingFace weights          GGUF file             Compute graph
  (PyTorch .safetensors)  →  (convert_hf_to_gguf)  →  (llama_decode)
       ↓                         ↓                        ↓
  config.json              Metadata + tensors       GGML operations
  tokenizer.json           in one file              on CPU/GPU
```

To add a new architecture you register it in the C++ side (enum, tensors, hparams,
graph builder) and the Python side (GGUF constants, converter class).

**What you'll produce:**

| Artifact | File(s) |
|----------|---------|
| Architecture enum | `src/llama-arch.h` |
| Name + tensor mapping | `src/llama-arch.cpp` |
| Hyperparameter loading | `src/llama-model.cpp` (`load_hparams`) |
| Tensor loading | `src/llama-model.cpp` (`load_tensors`) |
| Graph builder class | `src/models/mymodel.cpp` + declaration in `src/models/models.h` |
| Graph builder registration | `src/llama-model.cpp` (`build_graph`) |
| Build system entry | `src/CMakeLists.txt` |
| GGUF constants | `gguf-py/gguf/constants.py` |
| HuggingFace converter | `convert_hf_to_gguf.py` |

---

## 2. Prerequisites

- C++17 compiler, CMake ≥ 3.14
- Python 3.10+ with `torch`, `safetensors`, `sentencepiece` (for conversion)
- The model's `config.json` from HuggingFace (to understand its architecture)
- A small quantized checkpoint for testing (Q4_0 or Q8_0, ≤ 1 GB ideal)

**Read the model's config.json first.** It tells you everything:

```jsonc
{
  "architectures": ["MyModelForCausalLM"],
  "hidden_size": 2048,               // → n_embd
  "num_hidden_layers": 24,           // → n_layer
  "num_attention_heads": 16,         // → n_head
  "num_key_value_heads": 4,          // → n_head_kv  (GQA if < n_head)
  "intermediate_size": 5504,         // → n_ff
  "rms_norm_eps": 1e-5,              // → f_norm_rms_eps
  "rope_theta": 10000.0,             // → rope_freq_base
  "max_position_embeddings": 4096,   // → n_ctx_train
  "vocab_size": 32000                // → n_vocab
}
```

---

## 3. Quick Reference: Files to Touch

```
src/llama-arch.h              ← 1 line   (enum entry)
src/llama-arch.cpp            ← ~20 lines (name map + tensor set)
src/llama-model.cpp           ← ~40 lines (hparams + tensors + graph switch)
src/models/models.h           ← ~4 lines  (class declaration)
src/models/mymodel.cpp        ← ~100 lines (graph builder — the main work)
src/CMakeLists.txt            ← 1 line    (build entry)
gguf-py/gguf/constants.py     ← ~20 lines (arch enum + tensor list)
convert_hf_to_gguf.py         ← ~30 lines (converter class)
```

> **Tip:** Use `scripts/new-arch.py` to generate most of this automatically.
> See [Architecture Scaffold CLI Tool](#scaffold-tool) below.

---

## Step 1 — Register the Architecture Enum

**File:** `src/llama-arch.h`

Add your architecture to `enum llm_arch`, **before `LLM_ARCH_UNKNOWN`**:

```cpp
// src/llama-arch.h
enum llm_arch {
    LLM_ARCH_LLAMA       = 0,
    // ... existing architectures ...
    LLM_ARCH_KIMI_LINEAR,
    LLM_ARCH_MYMODEL,      // ← ADD THIS
    LLM_ARCH_UNKNOWN,
};
```

**Convention:** `LLM_ARCH_` + uppercase model name. Use underscores for multi-word
names (e.g., `LLM_ARCH_DEEP_SEEK`).

---

## Step 2 — Map the Architecture Name

**File:** `src/llama-arch.cpp`

### 2a. Add to `LLM_ARCH_NAMES`

This maps the enum to the string stored in GGUF metadata:

```cpp
// Near the top of llama-arch.cpp, in LLM_ARCH_NAMES
{ LLM_ARCH_MYMODEL, "mymodel" },
```

The string must match what your GGUF converter writes as `general.architecture`.

### 2b. Add tensor name set

In the `llm_arch_tensor_names()` function, add a case listing every tensor your model uses:

```cpp
case LLM_ARCH_MYMODEL:
    return {
        // Embeddings
        LLM_TENSOR_TOKEN_EMBD,
        LLM_TENSOR_OUTPUT_NORM,
        LLM_TENSOR_OUTPUT,
        // Attention (per-layer)
        LLM_TENSOR_ATTN_NORM,
        LLM_TENSOR_ATTN_Q,
        LLM_TENSOR_ATTN_K,
        LLM_TENSOR_ATTN_V,
        LLM_TENSOR_ATTN_OUT,
        // FFN (per-layer)
        LLM_TENSOR_FFN_NORM,
        LLM_TENSOR_FFN_GATE,
        LLM_TENSOR_FFN_DOWN,
        LLM_TENSOR_FFN_UP,
    };
```

**How to choose tensors:**
- Standard GQA transformer → use the set above (same as Qwen2, InternLM2, etc.)
- MoE → add `LLM_TENSOR_FFN_GATE_INP`, `LLM_TENSOR_FFN_{GATE,DOWN,UP}_EXPS`
- QK-norm → add `LLM_TENSOR_ATTN_Q_NORM`, `LLM_TENSOR_ATTN_K_NORM`
- Bias terms → add `LLM_TENSOR_ATTN_Q_B`, etc. (check `LLM_TENSOR_*` in `llama-arch.h`)

---

## Step 3 — Declare Tensor Names

> This step is part of 2b above. The tensor names (e.g., `"blk.%d.attn_q"`) are
> already defined globally in `LLM_TENSOR_NAMES`. You just need to list which ones
> your architecture uses.

No new tensor name strings are needed unless your model has a novel tensor type
not already in `enum llm_tensor`.

---

## Step 4 — Register GGUF Constants (Python)

**File:** `gguf-py/gguf/constants.py`

### 4a. Add to `MODEL_ARCH` enum

```python
class MODEL_ARCH(IntEnum):
    # ... existing ...
    MYMODEL = auto()
```

### 4b. Add to `MODEL_ARCH_NAMES`

```python
MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    # ... existing ...
    MODEL_ARCH.MYMODEL: "mymodel",   # Must match C++ LLM_ARCH_NAMES
}
```

### 4c. Add tensor list

```python
MODEL_ARCH.MYMODEL: [
    MODEL_TENSOR.TOKEN_EMBD,
    MODEL_TENSOR.OUTPUT_NORM,
    MODEL_TENSOR.OUTPUT,
    MODEL_TENSOR.ROPE_FREQS,
    MODEL_TENSOR.ATTN_NORM,
    MODEL_TENSOR.ATTN_Q,
    MODEL_TENSOR.ATTN_K,
    MODEL_TENSOR.ATTN_V,
    MODEL_TENSOR.ATTN_OUT,
    MODEL_TENSOR.FFN_NORM,
    MODEL_TENSOR.FFN_GATE,
    MODEL_TENSOR.FFN_DOWN,
    MODEL_TENSOR.FFN_UP,
],
```

This list must match the C++ tensor set from Step 2b.

---

## Step 5 — Load Hyperparameters

**File:** `src/llama-model.cpp`, function `load_hparams()`

Add a case to the architecture switch. This reads GGUF metadata into `hparams`:

```cpp
case LLM_ARCH_MYMODEL:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
            case 24: type = LLM_TYPE_7B;  break;
            case 32: type = LLM_TYPE_13B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } break;
```

**Common keys to read:**

| Key | hparams field | When needed |
|-----|---------------|-------------|
| `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS` | `f_norm_rms_eps` | RMSNorm models |
| `LLM_KV_ATTENTION_LAYERNORM_EPS` | `f_norm_eps` | LayerNorm models |
| `LLM_KV_EXPERT_COUNT` | `n_expert` | MoE models |
| `LLM_KV_EXPERT_USED_COUNT` | `n_expert_used` | MoE models |
| `LLM_KV_EXPERT_FEED_FORWARD_LENGTH` | `n_ff_exp` | MoE models |
| `LLM_KV_FFN_ACTIVATION` | `ffn_activation` | Auto-loaded for all models (silu/gelu/relu) |
| `LLM_KV_SSM_CONV_KERNEL` | `ssm_d_conv` | SSM/Mamba models |
| `LLM_KV_SSM_INNER_SIZE` | `ssm_d_inner` | SSM/Mamba models |
| `LLM_KV_SSM_STATE_SIZE` | `ssm_d_state` | SSM/Mamba models |

> **Note:** Basic params like `n_embd`, `n_layer`, `n_head`, `n_head_kv`, `n_ff`,
> `n_vocab`, `n_ctx_train`, `rope_freq_base`, and `ffn_activation` are loaded
> automatically for all architectures before the switch statement. You only need
> to read architecture-specific keys here.

---

## Step 6 — Load Tensors

**File:** `src/llama-model.cpp`, function `load_tensors()`

Add a case that creates tensor objects for your architecture. The tensor helper
`tn(LLM_TENSOR_*, "weight", layer_index)` generates the correct GGUF tensor name.

```cpp
case LLM_ARCH_MYMODEL:
    {
        // Global tensors
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

        // Tied embeddings fallback
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }

        // Per-layer tensors
        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff,   n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        }
    } break;
```

**Tensor dimension patterns:**

| Tensor | Shape | Notes |
|--------|-------|-------|
| `tok_embd` | `{n_embd, n_vocab}` | |
| `wq` | `{n_embd, n_embd_head_k * n_head}` | Full Q projection |
| `wk` | `{n_embd, n_embd_k_gqa}` | GQA-compressed K |
| `wv` | `{n_embd, n_embd_v_gqa}` | GQA-compressed V |
| `wo` | `{n_embd_head_k * n_head, n_embd}` | Output projection |
| `ffn_gate` | `{n_embd, n_ff}` | Gated FFN only |
| `ffn_up` | `{n_embd, n_ff}` | |
| `ffn_down` | `{n_ff, n_embd}` | |

---

## Step 7 — Build the Compute Graph

This is the core of your architecture — where you define how tensors flow through
the model to produce logits.

### 7a. Declare the class

**File:** `src/models/models.h`

```cpp
struct llm_build_mymodel : public llm_graph_context {
    llm_build_mymodel(const llama_model & model, const llm_graph_params & params);
};
```

### 7b. Implement the graph builder

**File:** `src/models/mymodel.cpp`

```cpp
#include "models.h"

llm_build_mymodel::llm_build_mymodel(
        const llama_model & model,
        const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // ==============================
    // 1. Input embeddings
    // ==============================
    inpL = build_inp_embd(model.tok_embd);

    // Position tensor (for RoPE)
    ggml_tensor * inp_pos = build_inp_pos();

    // KV-cache attention input
    auto * inp_attn = build_attn_inp_kv();

    // Attention scale: 1/sqrt(d_head) unless model overrides
    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f/sqrtf(float(n_embd_head))
        : hparams.f_attention_scale;

    // Output token filter (for efficient partial decoding)
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // ==============================
    // 2. Transformer layers
    // ==============================
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // --- Pre-attention norm ---
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // --- Self-attention ---
        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // Q, K, V projections
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Reshape for multi-head attention
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // Apply RoPE to Q and K
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            // Attention + output projection
            // Note: build_attn is overloaded for different cache modes:
            //   build_attn_inp_kv()       → standard KV-cache (most models)
            //   build_attn_inp_no_cache() → embedding/no-cache mode
            //   build_attn_inp_kv_iswa()  → interleaved sliding-window
            //   build_attn_inp_cross()    → encoder-decoder cross-attention
            cur = build_attn(inp_attn, model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, NULL, NULL, kq_scale, il);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // --- Pre-FFN norm ---
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // --- Feed-forward network (gated / SwiGLU) ---
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        cur = build_cvec(cur, il);
        inpL = cur;
    }

    // ==============================
    // 3. Output head
    // ==============================
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
```

### Key building blocks

| Helper | Purpose | Common arguments |
|--------|---------|------------------|
| `build_inp_embd()` | Token → embedding lookup | `model.tok_embd` |
| `build_inp_pos()` | Position indices for RoPE | — |
| `build_attn_inp_kv()` | KV-cache setup | — |
| `build_norm()` | Normalization layer | weight, bias, `LLM_NORM_RMS` or `LLM_NORM` |
| `build_lora_mm()` | Matrix multiply (LoRA-aware) | weight, input (optional: scale) |
| `build_attn()` | Full attention computation | Overloaded for kv/no-cache/cross/iswa modes |
| `build_ffn()` | Feed-forward network | up/gate/down weights, activation, layout |
| `build_moe_ffn()` | Mixture-of-Experts FFN | router + expert weights |
| `build_cvec()` | Control vector injection | — |
| `cb()` | Debug callback (tensor naming) | name string, layer index |

### Variations

**LayerNorm instead of RMSNorm:**
```cpp
cur = build_norm(inpL,
        model.layers[il].attn_norm,
        model.layers[il].attn_norm_b,   // bias tensor (non-NULL for LayerNorm)
        LLM_NORM, il);                  // LLM_NORM instead of LLM_NORM_RMS
```

**GELU activation instead of SiLU:**
```cpp
cur = build_ffn(cur, ..., LLM_FFN_GELU, LLM_FFN_PAR, il);
```

**Sequential FFN (no gate, classic MLP):**
```cpp
cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, NULL,
        NULL, NULL, NULL,                   // no gate
        model.layers[il].ffn_down, NULL, NULL,
        NULL,
        LLM_FFN_GELU, LLM_FFN_SEQ, il);   // SEQ instead of PAR
```

**Mixture-of-Experts:**
```cpp
cur = build_moe_ffn(cur,
        model.layers[il].ffn_gate_inp,      // router
        model.layers[il].ffn_up_exps,
        model.layers[il].ffn_gate_exps,
        model.layers[il].ffn_down_exps,
        nullptr,
        n_expert, n_expert_used,
        LLM_FFN_SILU, false,
        hparams.expert_weights_scale,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
        il);
```

**QK-Normalization:**
```cpp
Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
```

---

## Step 8 — Register the Graph Builder

**File:** `src/llama-model.cpp`, function `build_graph()`

```cpp
case LLM_ARCH_MYMODEL:
    {
        llm = std::make_unique<llm_build_mymodel>(*this, params);
    } break;
```

---

## Step 9 — Add to the Build System

**File:** `src/CMakeLists.txt`

Add your model source file in alphabetical order within the `models/` section:

```cmake
             models/mpt.cpp
             models/mymodel.cpp          # ← ADD THIS
             models/nemotron-h.cpp
```

---

## Step 10 — Write the HuggingFace Converter

**File:** `convert_hf_to_gguf.py`

```python
@ModelBase.register("MyModelForCausalLM")
class MyModelModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MYMODEL

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # Add any architecture-specific GGUF metadata here.
        # Most standard params (n_embd, n_layer, n_head, n_ff, etc.)
        # are written by super().set_gguf_parameters().
        #
        # Example for MoE:
        # self.gguf_writer.add_expert_count(self.hparams["num_experts"])
        # self.gguf_writer.add_expert_used_count(self.hparams["num_experts_per_tok"])
```

**HuggingFace → GGUF tensor name mapping** is handled automatically via the
`MODEL_TENSOR_NAMES` you registered in Step 4c, combined with the
`TensorNameMap` class. If your model's HuggingFace weight names follow standard
conventions (e.g., `model.layers.0.self_attn.q_proj.weight`), it should Just Work™.

> **Important:** The `@ModelBase.register()` decorator is **required**. It takes
> the HuggingFace architecture class name(s) from `config.json`'s `"architectures"`
> field (e.g., `"MyModelForCausalLM"`) and registers the converter so it's
> automatically selected during conversion. You can register multiple names if the
> model has aliases.

If your model uses non-standard weight names, override `modify_tensors()`:

```python
def modify_tensors(self, data_torch, name, bid):
    # Remap non-standard HF names
    name = name.replace("my_custom_prefix.", "model.")
    yield from super().modify_tensors(data_torch, name, bid)
```

---

## Step 11 — Test Your Architecture

### Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Convert from HuggingFace
python convert_hf_to_gguf.py /path/to/mymodel-hf --outtype q8_0

# Run inference
./build/bin/llama-cli -m mymodel.gguf -p "Hello, world!" -n 50
```

### Golden output tests

Add your architecture to `tests/test-golden-outputs.cpp` to enable regression testing:

```cpp
// In the architecture list at the top of the file
{ "mymodel", LLM_ARCH_MYMODEL },
```

Then run:

```bash
# Capture golden outputs
./build/bin/test-golden-outputs --capture --arch mymodel

# Verify (in CI or after changes)
./build/bin/test-golden-outputs --compare --arch mymodel
```

### Checklist

- [ ] Model loads without errors
- [ ] `llama-cli` produces coherent text
- [ ] Perplexity is within expected range (`llama-perplexity`)
- [ ] Quantized variants (Q4_0, Q8_0) work correctly
- [ ] Golden output test passes
- [ ] No memory leaks (run with AddressSanitizer)

---

## Common Patterns

### Pattern: Standard GQA Transformer (63% of models)

> GQA + RoPE + SiLU/SwiGLU + RMSNorm + Parallel FFN

This is the dominant pattern. If your model matches this, you can reuse the
LLaMA graph builder directly or copy it with minimal changes.

**Models using this pattern:** LLaMA, Qwen2, Mistral, Baichuan, DeepSeek,
InternLM2, Command-R, OLMo, and ~60 more.

### Pattern: Adding Bias Terms

Some models (GPT-2, Falcon, Jais) use bias in attention/FFN projections.
Add bias tensors in `load_tensors()`:

```cpp
layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
```

And use them in the graph builder:

```cpp
Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
```

### Pattern: Tied Embeddings

Many models share `tok_embd` and `output` weights. Handle this in `load_tensors()`:

```cpp
output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
if (output == NULL) {
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
}
```

### Pattern: Sliding Window Attention

Use `build_attn_inp_kv_unified()` or check the Gemma2/Cohere2 implementations
for interleaved sliding-window + global attention (ISWA).

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `unsupported architecture: mymodel` | Missing `build_graph` case | Add case in Step 8 |
| `tensor not found: blk.0.attn_q.weight` | Tensor name mismatch | Check GGUF tensor names match `LLM_TENSOR_NAMES` |
| Garbage output | Wrong RoPE config or attention scale | Compare with reference model's config.json |
| `GGML_ASSERT` failure | Tensor shape mismatch | Verify dimensions in `load_tensors()` |
| Conversion error | Missing converter class | Ensure class name matches HF `architectures` field |
| NaN/Inf in output | Missing normalization or wrong eps | Check `f_norm_rms_eps` is being read |

---

## Scaffold Tool

Use `scripts/new-arch.py` to generate boilerplate:

```bash
# Basic usage — generates a standard GQA transformer
python scripts/new-arch.py --name mymodel

# Specify components
python scripts/new-arch.py --name mymodel \
    --attention gqa \
    --ffn parallel \
    --norm rmsnorm \
    --activation silu \
    --position rope

# MoE variant
python scripts/new-arch.py --name mymodel --moe

# Read from HuggingFace config.json
python scripts/new-arch.py --name mymodel --config /path/to/config.json
```

The tool generates `src/models/mymodel.cpp` and prints exact instructions for
the remaining file modifications.

---

*This guide is part of the [Architecture SDK](ARCHITECTURE-SDK-PLAN.md) initiative.*
