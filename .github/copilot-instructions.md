# Copilot Instructions — Schema-Driven Architecture SDK

This branch (`feat/graph-schema`) implements a schema-driven graph
construction system. Read `docs/POC-SUMMARY.md` for the overview.

## Skills

When working on this codebase, refer to these guides:

### Adding support for a new model
Read `docs/SKILL-NEW-MODEL-SCHEMA.md` — covers extending `derive_schema.py`
with new config keys and tensor patterns. No C++ changes needed for standard
transformer variants.

### Adding new C++ graph building blocks
Read `docs/SKILL-NEW-CPP-BUILDING-BLOCKS.md` — covers adding tensor types,
GGUF metadata keys, builder config flags, and graph operations. Needed when
a model introduces genuinely new computation patterns.

### Auditing a model implementation
Read `docs/SKILL-AUDIT-MODEL.md` — systematic methodology for verifying a
model's implementation by cross-referencing HuggingFace config/tensors against
C++ and Python code. Use when a new model ships, bugs are suspected, or before
contributing.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/derive_schema.py` | Derives graph schema from HF model files |
| `convert_hf_to_gguf.py` | Converter — calls deriver, writes GGUF keys |
| `src/llama-graph-builder.cpp` | Generic builder + auto-config from GGUF |
| `src/llama-graph-builder.h` | Builder config struct (30 flags) |
| `src/llama-model.cpp` | `load_hparams()`, `load_tensors()`, `build_graph()` |

## Core Principles

1. **Zero tribal knowledge** — `derive_schema.py` reads only `config.json` +
   `model.safetensors.index.json`. No model-type heuristics.
2. **Data-driven builder** — the GGUF keys ARE the configuration. The builder
   follows the blueprint derived from the model's own files.
3. **Tensor-gated loading** — `has(LLM_TENSOR_X)` gates all tensor creation
   in the generic default to prevent ghost tensors.
4. **Expert dimensions** — MoE expert tensors use `n_ff_exp` (not `n_ff`).

## Testing

```bash
# Type-check Python
ty check scripts/derive_schema.py

# Build and run architecture tests
cmake --build build -j$(nproc) --target test-llama-archs
./build/bin/test-llama-archs
# Expected: 327+ OKs, 0 failures

# End-to-end: derive → convert → serve
python scripts/derive_schema.py <model_dir>/
python convert_hf_to_gguf.py <model_dir>/ --outfile model.gguf --outtype f16
./build/bin/llama-cli -m model.gguf -ngl 99 -p "Hello" -n 20
```
