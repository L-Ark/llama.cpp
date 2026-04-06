# Skill: Adding New C++ Graph Building Blocks

This guide explains how to extend the C++ graph construction system when a
new model introduces tensor types, operations, or graph structures that the
generic builder doesn't handle.

## When This Skill Applies

- A new model uses tensors the generic loader doesn't create
- A new model needs graph operations the generic builder doesn't construct
- A new model's GGUF metadata keys aren't read by the auto-config
- The generic builder produces incorrect graphs for a model that was previously
  handled by the deriver correctly

## Architecture Overview

```
GGUF file
  │
  ├── load_hparams() ──── reads config values into hparams struct
  │                        File: src/llama-model.cpp (switch on arch)
  │
  ├── load_tensors() ──── creates ggml_tensor objects from GGUF weights
  │                        File: src/llama-model.cpp (switch on arch)
  │
  └── build_graph()  ──── constructs the ggml computation graph
                           File: src/llama-model.cpp (switch on arch)
                           Generic: src/llama-graph-builder.cpp
```

## Step 1: Identify What's Missing

### New Tensor Type
If the model has tensors the generic loader doesn't create:

1. Check if the tensor name is in `src/llama-arch.cpp` (`LLM_TENSOR_*` enum)
2. Check if it's in the `llama_layer` struct (`src/llama-model.h`)
3. Check if `load_tensors()` generic default creates it

### New GGUF Metadata Key
If the model writes config values the C++ side doesn't read:

1. Check `src/llama-arch.h` for `LLM_KV_*` enum
2. Check `src/llama-arch.cpp` for key name string mapping
3. Check `load_hparams()` for where it's read

### New Graph Operation
If the model needs computation steps the builder doesn't have:

1. Check `src/llama-graph-builder.h` for `llm_transformer_config` flags
2. Check `src/llama-graph-builder.cpp` for the builder constructor

## Step 2: Add a New Tensor Type

### 2a. Register the tensor name

In `src/llama-arch.h`, add to the `LLM_TENSOR` enum:
```cpp
LLM_TENSOR_NEW_TENSOR,
```

In `src/llama-arch.cpp`, add the name mapping in the appropriate architecture
sections and in the tensor name table:
```cpp
{ LLM_TENSOR_NEW_TENSOR, "blk.%d.new_tensor" },
```

### 2b. Add to the layer struct

In `src/llama-model.h`, add to `struct llama_layer`:
```cpp
ggml_tensor * new_tensor = nullptr;
```

### 2c. Create in load_tensors()

In the generic default case of `load_tensors()` in `src/llama-model.cpp`:
```cpp
if (has(LLM_TENSOR_NEW_TENSOR))
    layer.new_tensor = create_tensor(
        tn(LLM_TENSOR_NEW_TENSOR, "weight", i),
        {dim1, dim2}, TENSOR_NOT_REQUIRED);
```

Rules:
- Use `has(LLM_TENSOR_X)` to gate creation (prevents ghost tensors)
- Use `TENSOR_NOT_REQUIRED` for optional tensors
- Use `0` for required tensors (will error if missing)

## Step 3: Add a New GGUF Metadata Key

### 3a. Register the key

In `src/llama-arch.h`, add to `LLM_KV` enum:
```cpp
LLM_KV_NEW_PARAM,
```

In `src/llama-arch.cpp`, add the name mapping:
```cpp
{ LLM_KV_NEW_PARAM, "%s.new_param" },
```

### 3b. Add to hparams struct

In `src/llama-hparams.h`:
```cpp
uint32_t new_param = 0; // default value
```

### 3c. Read in load_hparams()

In the generic default case:
```cpp
ml.get_key(LLM_KV_NEW_PARAM, hparams.new_param, false); // false = optional
```

### 3d. Write from Python converter

In `gguf-py/gguf/constants.py`:
```python
NEW_PARAM = "{arch}.new_param"
```

In `gguf-py/gguf/gguf_writer.py`:
```python
def add_new_param(self, value: int) -> None:
    self.add_uint32(Keys.LLM.NEW_PARAM.format(arch=self.arch), value)
```

## Step 4: Add a New Builder Config Flag

### 4a. Add to config struct

In `src/llama-graph-builder.h`:
```cpp
struct llm_transformer_config {
    // ... existing flags ...
    bool new_feature = false;
};
```

### 4b. Set from auto-config

In `llm_transformer_config_from_hparams()` in `src/llama-graph-builder.cpp`:

From tensors:
```cpp
config.new_feature = (model.layers[0].new_tensor != nullptr);
```

From layer_operations:
```cpp
if (has_op("new_token")) config.new_feature = true;
```

From hparams:
```cpp
config.new_feature = (hparams.new_param > 0);
```

### 4c. Use in the builder

In `llm_build_std_transformer::llm_build_std_transformer()`:
```cpp
if (config.new_feature) {
    // Build the new graph operation
    cur = ggml_new_operation(ctx0, cur, model.layers[il].new_tensor);
}
```

## Step 5: Add to Architecture Metadata Table

In `get_arch_meta()` in `src/llama-graph-builder.cpp`:

```cpp
case LLM_ARCH_NEW_ARCH:
    return {OPS_STD, ACT_SILU, false};
```

Or define a new ops string if the architecture has a unique operation sequence:
```cpp
static constexpr const char * OPS_NEW = "norm,rope,attn,new_op,filter,...";
```

## Step 6: Update layer_operations Vocabulary

If adding a new token to the layer_operations string:

### Python side (derive_schema.py)
In `_derive_layer_operations()` in `convert_hf_to_gguf.py`:
```python
if schema_has_new_feature:
    ops.append("new_op")
```

### C++ side (parser)
In `llm_transformer_config_from_hparams()`:
```cpp
if (has_op("new_op")) config.new_feature = true;
```

## Step 7: Test

1. `cmake --build build -j$(nproc) --target test-llama-archs`
2. `./build/bin/test-llama-archs` → 327+ OKs, 0 failures
3. `ty check scripts/derive_schema.py` → passes
4. Convert and serve a real model if available

## File Reference

| File | What to Change |
|------|---------------|
| `src/llama-arch.h` | `LLM_TENSOR_*` and `LLM_KV_*` enums |
| `src/llama-arch.cpp` | Tensor name strings, KV key strings |
| `src/llama-hparams.h` | New hparam fields |
| `src/llama-model.h` | New fields in `llama_layer` struct |
| `src/llama-model.cpp` | `load_hparams()`, `load_tensors()`, `build_graph()` |
| `src/llama-graph-builder.h` | `llm_transformer_config` flags |
| `src/llama-graph-builder.cpp` | `get_arch_meta()`, auto-config, builder constructor |
| `gguf-py/gguf/constants.py` | GGUF key constants |
| `gguf-py/gguf/gguf_writer.py` | Writer methods for new keys |
| `scripts/derive_schema.py` | Schema deriver patterns |
| `convert_hf_to_gguf.py` | `_derive_layer_operations()`, `_write_schema_metadata()` |

## Common Patterns

### MoE Expert Tensors
Expert tensors use 3D shapes: `{n_embd, n_ff_exp, n_expert}`.
Use `hparams.n_ff_exp` (not `n_ff`) for expert FFN dimensions.

### Per-Layer Variable Config
Some models have different configs per layer (e.g., hybrid SSM+attention).
Use `hparams.n_head_kv(il) == 0` to detect recurrent layers,
or `hparams.recurrent_layer_arr[il]` for explicit marking.

### Sliding Window Attention
SWA uses ISWA (Interleaved SWA) cache. Enable via `config.iswa = true`.
Check `hparams.swa_type` and `hparams.n_swa`.

### Shared Experts
Shared experts have separate FFN dimensions (`n_ff_shexp`).
Tensors: `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`,
optionally `ffn_gate_inp_shexp` (shared expert gate).
