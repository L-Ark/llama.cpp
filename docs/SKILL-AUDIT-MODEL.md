# Skill: Auditing a New Model Implementation

This guide explains how to systematically audit a model's implementation in
llama.cpp by cross-referencing HuggingFace source files against the C++/Python
code. Use this when a new model is released, when bugs are suspected, or before
contributing a new architecture.

## When This Skill Applies

- A new SOTA model is released and you need to verify llama.cpp supports it
- A model produces incorrect output and you need to find the bug
- You want to proactively identify issues before they're reported

## Step 1: Gather HuggingFace Ground Truth

These are the source of truth. Everything in llama.cpp must match these.

### 1a. Fetch config.json (no model download needed)

```python
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download("<org>/<model>", "config.json", token="<token>")
config = json.load(open(path))
text_cfg = config.get("text_config", config)
```

Key fields to extract and cross-reference:
- **Architecture**: `architectures[0]` — what builder class is used?
- **Attention**: `num_attention_heads`, `num_key_value_heads`, `head_dim`,
  `attention_bias`, any custom keys like `attention_k_eq_v`
- **FFN**: `intermediate_size`, `hidden_act`, MoE keys (`num_experts`,
  `top_k_experts`, `moe_intermediate_size`)
- **Norms**: `rms_norm_eps`, whether Gemma-style `(1+w)*x` or standard `w*x`
- **Position**: `rope_theta`, `rope_parameters`, `partial_rotary_factor`,
  `sliding_window`, `layer_types`
- **Vocab**: `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id`,
  `tie_word_embeddings`
- **Special features**: shared KV layers, per-layer embeddings, vision config

### 1b. Fetch tensor names (no weights needed)

```python
from huggingface_hub import get_safetensors_metadata
meta = get_safetensors_metadata("<org>/<model>", token="<token>")
names = list(meta.weight_map.keys())
```

### 1c. Fetch tokenizer config

```python
path = hf_hub_download("<org>/<model>", "tokenizer_config.json", token="<token>")
tok_config = json.load(open(path))
# Check: added_tokens, bos/eos/pad tokens, tokenizer class, pre_tokenizer
```

### 1d. Check multiple size variants

Many models ship in multiple sizes (E2B, 31B, 26B-A4B, etc.) with different
config values. **Always check at least the smallest AND largest variant.**
Config keys that are `null` in one variant may be active in another.

## Step 2: Cross-Reference load_hparams()

File: `src/llama-model.cpp`, search for `case LLM_ARCH_<MODEL>:` in the
`load_hparams` switch.

For each config key from Step 1, verify:

| Check | How |
|-------|-----|
| Key is read | `ml.get_key(LLM_KV_*, hparams.field, ...)` exists |
| Default is correct | Optional keys use `false` flag; required keys fail clearly |
| Value is stored correctly | hparams field type matches (uint32, float, array) |
| All variants covered | Large/MoE/shared-KV variants don't have unread keys |

**Common bugs:**
- Config key exists in HF but is never read (feature silently ignored)
- Key read for one variant but not another (works for small, breaks for large)
- Default value is wrong (e.g., `n_layer_kv_from_start` hardcoded instead of read)

## Step 3: Cross-Reference load_tensors()

File: `src/llama-model.cpp`, search for the model's case in the `load_tensors`
switch.

For each tensor name from Step 1b, verify:

| Check | How |
|-------|-----|
| Tensor is created | `create_tensor(tn(LLM_TENSOR_*, ...), {dims}, flags)` |
| Dimensions are correct | Match HF weight shapes (transpose if needed) |
| Expert dims use n_ff_exp | Not n_ff (these differ for MoE models) |
| Optional tensors gated | `TENSOR_NOT_REQUIRED` or `has(LLM_TENSOR_*)` |

**Common bugs:**
- Expert FFN dimensions use `n_ff` instead of `n_ff_exp` (shape mismatch)
- Missing tensors for model variants (shared expert gate, per-layer embeddings)
- Tensors created that don't exist in GGUF (ghost tensors → count mismatch)

## Step 4: Cross-Reference the Builder

File: `src/models/<model>.cpp` or `src/llama-graph-builder.cpp` (generic)

### 4a. Attention audit

| HF Config | C++ Must Do |
|-----------|-------------|
| `num_key_value_heads` | Correct GQA ratio in Q/K/V split |
| `head_dim` | Correct reshape dimensions after projection |
| `attention_k_eq_v: true` | V reuses K (no separate V projection) |
| `rope_parameters` per layer type | Different theta/factor for SWA vs full |
| `sliding_window` | SWA cache enabled, correct window size |
| `num_kv_shared_layers` | KV cache reuse for tail layers |

### 4b. FFN audit

| HF Config | C++ Must Do |
|-----------|-------------|
| `hidden_act` | Correct activation function |
| `intermediate_size` | Dense FFN dimension |
| `moe_intermediate_size` | Expert FFN dimension (different from dense!) |
| `num_experts` / `top_k_experts` | MoE routing parameters |
| `use_double_wide_mlp` | 2x FFN width for shared KV layers |

### 4c. Normalization audit

| HF Config | C++ Must Do |
|-----------|-------------|
| `rms_norm_eps` | Correct epsilon value |
| Gemma-style `(1+w)*x` | Converter adds 1.0 to norm weights (`norm_shift`) |
| Standard `w*x` | Converter does NOT add 1.0 (`norm_shift = 0`) |
| Post-norms present | Tensors and builder both handle them |

### 4d. Embedding/output audit

| HF Config | C++ Must Do |
|-----------|-------------|
| `tie_word_embeddings` | Output uses token embedding (no separate output weight) |
| `vocab_size` | Token embedding dims match |
| `bos_token_id` / `eos_token_id` | Correct special token IDs |
| `add_bos_token` | BOS added/not added during tokenization |
| `final_logit_softcapping` | Logit capping applied after output projection |

## Step 5: Cross-Reference the Converter

File: `convert_hf_to_gguf.py`, search for the model class.

### 5a. Tensor mapping audit

For each HF tensor name, verify:
- `modify_tensors()` maps it to the correct GGUF tensor name
- Dimensions are preserved or correctly transposed
- Expert indices are extracted and mapped
- Per-layer variation (shared KV, MoE vs dense) is handled

### 5b. Vocabulary audit

For `set_vocab()`, verify:
- Correct tokenizer class used (`LlamaHfVocab`, `SentencePieceVocab`, etc.)
- Token types classified correctly (`CONTROL` vs `USER_DEFINED`)
- Special tokens that need rendering are NOT marked as `CONTROL`
- BOS/EOS/PAD token IDs match config.json
- `add_bos_token` matches the model's expectations

### 5c. GGUF parameters audit

For `set_gguf_parameters()`, verify every value matches config.json:
- Head dimensions (global vs SWA if applicable)
- KV head counts (may be per-layer array for multi-head-dim models)
- Sliding window pattern (must match `layer_types` array length)
- RoPE parameters (theta, dimension count, frequency factors)

## Step 6: Test

### 6a. Synthetic (no model needed)

Build the synthetic config + tensor list and run through derive_schema.py:
```python
from derive_schema import derive_schema
schema = derive_schema(config, tensor_names)
```

### 6b. Conversion test

```bash
python convert_hf_to_gguf.py <model_dir>/ --outfile test.gguf --outtype f16
# Check: no errors, correct tensor count, GGUF keys present
```

### 6c. Inference test

```bash
./build/bin/llama-cli -m test.gguf -ngl 99 -p "Hello" -n 20
# Check: coherent output, no crashes, reasonable speed
```

### 6d. Perplexity test (if reference available)

Compare perplexity against HuggingFace Transformers on the same prompt to
verify numerical correctness.

## Common Bug Patterns by Architecture Category

### Standard Transformers
- Wrong activation function (gelu vs silu vs relu)
- Missing attention bias
- Incorrect RoPE theta or scaling

### MoE Models
- Expert FFN dim (n_ff_exp) vs dense FFN dim (n_ff) confusion
- Missing shared expert tensors or gate
- Incorrect expert count or top-k routing

### Multi-Head-Dim Models (Gemma4, DeepSeek MLA)
- Different head dims for different layer types not propagated
- Per-layer KV head count arrays not written
- RoPE dim counts wrong for partial rotary factor

### Vision-Language Models
- Padding token ID hardcoded instead of read from config
- Vision path missing scale factors that text path has
- Multimodal token type handling incorrect

### Shared KV Cache Models
- All shared layers reusing same source layer (should be structured)
- Double-wide MLP not applied to shared layers
- Per-layer embedding dimensions mismatched

## Checklist Template

Copy this for each new model audit:

```
Model: _______________
Variants checked: [ ] small  [ ] large  [ ] MoE

Config Keys:
[ ] All HF config keys read in load_hparams
[ ] Default values correct for optional keys
[ ] All variants covered (null vs non-null keys)

Tensors:
[ ] All HF tensors mapped in converter
[ ] All GGUF tensors created in load_tensors
[ ] Dimensions correct (especially MoE expert dims)
[ ] Optional tensors properly gated

Builder:
[ ] Attention: head dims, GQA ratio, K=V handling, RoPE
[ ] FFN: activation, gate/no-gate, MoE routing
[ ] Norms: epsilon, pre/post, Gemma-style shift
[ ] Embeddings: tied, scaling, per-layer

Tokenizer:
[ ] Correct vocab type (SPM, BPE, WPM)
[ ] Token types classified correctly
[ ] BOS/EOS behavior matches model expectations
[ ] Special tokens render when needed

Tests:
[ ] derive_schema.py produces correct schema
[ ] Conversion succeeds with correct tensor count
[ ] Inference produces coherent output
[ ] 327 architecture tests still pass
```
