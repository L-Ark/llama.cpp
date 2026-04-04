# Architecture Component Catalog

> Auto-generated analysis of all 113 model implementations in `src/models/`.
> This catalog documents which building blocks each architecture uses.

## Summary Statistics

| Component | Most Common | Count | Percentage |
|-----------|-------------|-------|------------|
| **Attention** | GQA | 101/113 | 89% |
| **Position Encoding** | RoPE | 86/113 | 76% |
| **Normalization** | RMSNorm | 86/113 | 76% |
| **Activation** | SILU | 71/113 | 63% |
| **FFN Layout** | Parallel | 70/113 | 62% |

## Dominant Pattern (63% of models)

```
GQA Attention + RoPE + SILU + RMSNorm + Parallel FFN
```

Models using this exact pattern: Llama, Qwen2, Qwen3, Mistral, Baichuan,
DeepSeek, Arctic, Internlm2, Orion, Xverse, Olmo, Command-R, and ~60 more.

**Implication for Phase 3**: A single dynamic graph builder covering this
pattern would handle the majority of architectures with zero per-model code.

## Building Block Inventory

### Attention Types
| Type | Count | Models |
|------|-------|--------|
| GQA (Grouped Query) | 101 | Most modern LLMs |
| MHA (Multi-Head) | 12 | GPT2, BERT variants, older models |
| State-Space (no attention) | 5+ | Mamba, RWKV |
| Cross-Attention | 1 | T5 |

### Activation Functions
| Activation | Count | Models |
|-----------|-------|--------|
| SILU | 71 | Llama, Qwen, DeepSeek, Mistral, etc. |
| GELU | 16 | Gemma, Falcon, GPT2, MPT, Phi2, Starcoder |
| SWIGLU | 5 | ChatGLM, GLM4, Neo-BERT, Plamo2 |
| RELU | 1 | Arcee, Nemotron, PLM |
| None (SSM) | 16 | Mamba, RWKV, Jamba |

### Normalization
| Type | Count | Models |
|------|-------|--------|
| RMSNorm | 86 | Modern LLMs (more stable, efficient) |
| LayerNorm | 23 | BERT, GPT2, Falcon, older models |
| None | 3 | Pure state-space (Mamba-base, RWKV-base) |
| Mixed | 1 | Chameleon (RMSNorm + LayerNorm) |

### Position Encoding
| Type | Count | Models |
|------|-------|--------|
| RoPE | 86 | Standard for modern LLMs |
| None/Learned | 22 | BERT, GPT2, state-space models |
| ALiBi | 1 | T5 |
| RoPE-YaRN/NTK | varies | Long-context variants |

### FFN Layout
| Layout | Count | Description |
|--------|-------|-------------|
| Parallel | 70 | gate(x) ⊙ up(x), then down — standard gated FFN |
| Sequential | 24 | up → activation → down — classical MLP |

### Special Features
| Feature | Count | Models |
|---------|-------|--------|
| QK Normalization | ~35 | Qwen3, Exaone, GLM4-MoE, Hunyuan, etc. |
| MoE Routing | 15 | All *-moe variants |
| State-Space Hybrid | 4+ | Jamba, Falcon-H1, Granite-Hybrid |
| Sliding Window | 1 | OLMo2 |
| Vision/Multimodal | 6-8 | Gemma3/4, GLM4, Chameleon, CogVLM |

## Complexity Distribution

| Complexity | Line Range | Count | Examples |
|-----------|-----------|-------|----------|
| Simple | <100 lines | ~30 | Mamba (55), RWKV7 (91), basic transformers |
| Medium | 100-200 lines | ~60 | Most standard architectures |
| Complex | 200-400 lines | ~20 | Qwen35 (385), Kimi-Linear (383) |
| Very Complex | >400 lines | ~3 | Qwen3Next (529) |

## Architecture Family Groups

### Group 1: Standard Transformer (~75 models)
Pattern: Embed → (Norm → Attn → Norm → FFN) × N → Norm → Head

### Group 2: MoE Transformer (15 models)
Pattern: Embed → (Norm → Attn → Norm → MoE_FFN) × N → Norm → Head
Addition: Router network selects top-K experts per token

### Group 3: State-Space (5+ models)
Pattern: Embed → (SSM_Block) × N → Norm → Head
No attention mechanism — O(1) memory, linear scaling

### Group 4: Hybrid (4+ models)
Pattern: Embed → (Attn_Block | SSM_Block alternating) × N → Norm → Head
Mix of attention and state-space layers

### Group 5: Encoder-Decoder (2 models)
Pattern: Encoder(Embed → SelfAttn × N) → Decoder(CrossAttn + SelfAttn × N) → Head
T5-style with bidirectional encoder
