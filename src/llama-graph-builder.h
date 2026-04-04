#pragma once

// llama-graph-builder.h -- Composable block configuration for standard transformers
//
// Phase 2 of the Architecture SDK: instead of 100+ lines of hand-coded graph
// construction per model, configure a generic builder with a few structs.
//
// Usage:
//   // Default config (63% of models: GQA + RoPE + SiLU + RMSNorm):
//   case LLM_ARCH_QWEN2:
//       llm = std::make_unique<llm_build_std_transformer>(*this, params);
//       break;
//
//   // MoE model:
//   case LLM_ARCH_OLMOE:
//       llm_transformer_config cfg;
//       cfg.moe = true;
//       llm = std::make_unique<llm_build_std_transformer>(*this, params, cfg);
//       break;
//
//   // Gemma-style with ISWA + post-norms + softcap:
//   case LLM_ARCH_GEMMA3:
//       llm_transformer_config cfg;
//       cfg.iswa = true;
//       cfg.qk_norm = true;
//       cfg.attn_post_norm = true;
//       cfg.ffn_post_norm = true;
//       cfg.logit_softcap = true;
//       cfg.token_embd_scale = true;
//       cfg.act = LLM_FFN_GELU;
//       llm = std::make_unique<llm_build_std_transformer>(*this, params, cfg);
//       break;

#include "llama-model.h"
#include "llama-graph.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Transformer block configuration
// ---------------------------------------------------------------------------

struct llm_transformer_config {
    // Normalization
    llm_norm_type norm = LLM_NORM_RMS;     // LLM_NORM_RMS (76%) or LLM_NORM

    // Activation function
    llm_ffn_op_type act = LLM_FFN_SILU;    // LLM_FFN_SILU (63%), LLM_FFN_GELU, LLM_FFN_RELU

    // FFN layout
    llm_ffn_gate_type ffn_type = LLM_FFN_PAR;  // LLM_FFN_PAR (gated/62%) or LLM_FFN_SEQ (classic MLP)

    // Attention features
    bool attn_bias  = false;    // Attention Q/K/V/O bias (GPT-2, Falcon, Jais)
    bool qk_norm    = false;    // QK normalization (Qwen3, Exaone, Gemma3, ~35 models)
    bool use_rope   = true;     // RoPE position encoding (76% of models)
    bool iswa       = false;    // Interleaved Sliding Window Attention (Gemma3/4, Phi-3)

    // FFN features
    bool ffn_bias   = false;    // FFN bias terms
    bool moe        = false;    // Mixture-of-Experts FFN (auto-detected per layer)
    bool moe_shared = false;    // MoE with shared expert (DeepSeek, Qwen MoE)

    // Output
    bool output_bias = false;   // Output projection bias
    bool tied_embeddings = true; // Fall back to tok_embd if output is NULL

    // Post-norm (Gemma 3/4, CogVLM)
    bool attn_post_norm = false;
    bool ffn_post_norm  = false;

    // Gemma-specific features (also used by others)
    bool logit_softcap     = false;  // tanh(logits/cap)*cap (Gemma 3/4)
    bool token_embd_scale  = false;  // Scale token embeddings by sqrt(n_embd) (Gemma)

    // Input features
    bool combined_qkv  = false;  // Use wqkv instead of separate wq/wk/wv (GPT-2, Bloom, Falcon, Jais)
    bool use_pos_embd  = false;  // Learned position embeddings (GPT-2, Jais)
    bool global_tok_norm = false; // Pre-layer token normalization (Bloom)

    // Residual connection
    bool parallel_ffn  = false;  // FFN uses pre-attn norm instead of post-attn (Falcon)

    // Attention cache mode
    bool no_attn_cache = false;  // No KV cache (embedding models, diffusion)
};

// ---------------------------------------------------------------------------
// Auto-configure from model hparams
//
// Phase 3 foundation: reads architecture metadata from hparams and constructs
// the right config automatically. Returns true if the architecture can be
// handled by the standard transformer builder.
// ---------------------------------------------------------------------------

bool llm_transformer_config_from_hparams(
        const llama_hparams & hparams,
        const llama_model   & model,
        llm_transformer_config & config);

// ---------------------------------------------------------------------------
// Standard transformer graph builder
//
// Covers ~80%+ of architectures with configuration. The standard pattern is:
//
//   Embed -> [scale] -> (Norm -> Attn -> [PostNorm] -> Norm -> FFN/MoE -> [PostNorm]) x N
//   -> Norm -> Head -> [softcap]
//
// For models that don't fit (MLA, SSM/hybrid, encoder-decoder), use a
// model-specific builder as before.
// ---------------------------------------------------------------------------

struct llm_build_std_transformer : public llm_graph_context {
    llm_build_std_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_transformer_config & config = {});
};
