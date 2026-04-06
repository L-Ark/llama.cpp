# Skill: Adding a New Model Architecture to derive_schema.py

This guide explains how to extend `scripts/derive_schema.py` to support a
new model architecture. Follow these steps when a new model is released on
HuggingFace that the deriver doesn't fully classify.

## When This Skill Applies

- A new model's `config.json` has config keys the deriver doesn't read
- A new model's safetensors have tensor name patterns the deriver doesn't match
- The deriver produces an incomplete or incorrect schema for a model

## Step 1: Gather Data (No Model Download Needed)

Fetch the model's metadata from HuggingFace without downloading weights:

```python
from huggingface_hub import get_safetensors_metadata
import json, re

# Fetch config.json via web
# https://huggingface.co/<org>/<model>/raw/main/config.json

# Fetch tensor names (no weights downloaded)
meta = get_safetensors_metadata("<org>/<model>")
names = list(meta.weight_map.keys())

# Extract unique per-layer suffixes
layer_suffixes = set()
for n in names:
    m = re.match(r'.*(?:layers?|blocks?|h)\.\d+\.(.*)', n)
    if m:
        layer_suffixes.add(m.group(1))
print(sorted(layer_suffixes))
```

## Step 2: Identify New Patterns

Compare the model's data against the deriver's existing vocabulary:

### Config Keys (in `derive_schema()`, "FROM CONFIG" section)
- Check if `config.json` has keys not listed in the `_get_cfg()` calls
- Common new keys: custom norm epsilons, attention variants, MoE parameters,
  position encoding config, layer pattern overrides

### Tensor Patterns (in `derive_schema()`, "FROM TENSORS" section)
- Check if layer suffixes match any existing regex patterns
- Look for new tensor types: custom attention projections, new norm positions,
  specialized FFN variants, state-space parameters

## Step 3: Add Config Key Reading

In the "FROM CONFIG" section of `derive_schema()`, add new `_get_cfg()` calls:

```python
# Example: adding a new attention variant config
new_attention_param = _get_cfg(text_cfg, config, "new_param_name",
                                "alternate_param_name")
```

Rules:
- Always use `_get_cfg(text_cfg, config, ...)` to check text_config first
- Use `false`/`None` as implicit default (don't guess when missing)
- Add multiple key names if different models use different names for the same thing

## Step 4: Add Tensor Pattern Detection

In the "FROM TENSORS" section, add new regex patterns:

```python
has_new_feature = any(re.search(r'new_tensor_pattern', s)
                      for s in layer_suffixes)
```

Rules:
- Use `\b` word boundaries to prevent false matches
- Match the most specific pattern possible
- Add the variable to the `else` branch (no tensors) as `False`

## Step 5: Add to COMBINE Section

Derive boolean flags from config + tensors:

```python
is_new_arch = new_config_param is not None or has_new_feature
```

## Step 6: Add to Schema Output

For architecture-specific features, add to the `features` dict:

```python
if is_new_arch:
    features["new_feature"] = {}
    if new_config_param:
        features["new_feature"]["param"] = new_config_param
```

For new layer types or attention variants, add to the layer spec or
attention spec as appropriate.

For new tensor types, add to the tensor manifest (`lt` dict):

```python
if has_new_feature:
    lt["new_tensor"] = {"shape": ["dim1", "dim2"]}
```

## Step 7: Add Type Annotations

All new dict literals must use `dict[str, Any]` annotations to pass the
`ty` type checker:

```python
new_spec: dict[str, Any] = {"type": "new_type"}
```

## Step 8: Test

Run the deriver against the model's config + tensor names:

```python
from derive_schema import derive_schema
schema = derive_schema(config, tensor_names)
print(json.dumps(schema, indent=2))
```

Verify:
1. `ty check scripts/derive_schema.py` passes
2. Existing models still produce correct schemas (regression test)
3. The 327 architecture unit tests still pass

## File Reference

| File | Purpose |
|------|---------|
| `scripts/derive_schema.py` | The deriver — this is where you add patterns |
| `convert_hf_to_gguf.py` | Calls `_derive_schema()` during conversion |
| `src/llama-graph-builder.cpp` | C++ side — reads GGUF keys (see C++ skill) |
| `ty.toml` | Type checker config — `./scripts` is in extra-paths |

## Architecture Category Reference

The deriver currently handles these categories:
- Standard transformers (RoPE/learned pos, RMS/LayerNorm, gated/sequential FFN)
- MoE (routed experts, shared experts, expert gates)
- SSM/Mamba (conv1d, selective scan, state space)
- Hybrid SSM+Attention (per-layer type switching)
- Encoder-only (BERT post-norm, pooler, token type embeddings)
- Encoder-decoder (T5 cross-attention, relative position bias)
- RWKV (time mix, channel mix)
- Multi-head Latent Attention (DeepSeek MLA, KV LoRA compression)
- AltUp + Laurel (Gemma3N)
- Dynamic Sparse Attention (GLM-DSA)
- Short Convolution (LFM2)
- Chunked Attention (LLAMA4)
- Dual-path MoE (Gemma4)
