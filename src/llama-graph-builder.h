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
#include "models/models.h"

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
    bool v_norm     = false;    // V normalization with raw ggml_rms_norm (Gemma 4)
    bool use_rope   = true;     // RoPE position encoding (76% of models)
    bool iswa       = false;    // Interleaved Sliding Window Attention (Gemma3/4, Phi-3)
    bool attn_gate  = false;    // Sigmoid gating on attention output before o_proj (AFMoE)
    bool attn_q_gate = false;   // Query projection packs query + attention gate halves (Qwen3.5)
    llama_rope_type rope_override = LLAMA_ROPE_TYPE_NONE; // Override the model's RoPE type when needed
    bool raw_rope   = false;    // Apply RoPE with raw theta only (no context scaling inputs)

    // FFN features
    bool ffn_bias   = false;    // FFN bias terms
    bool moe        = false;    // Mixture-of-Experts FFN (auto-detected per layer)
    bool moe_shared = false;    // MoE with shared expert (DeepSeek, Qwen MoE)
    bool moe_shared_gate = false; // Shared expert output gated by a sigmoid router (Qwen3.5-MoE)
    bool moe_norm_weights = false; // Force normalized expert weights even if hparams omit it
    llama_expert_gating_func_type moe_gating = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE; // Optional router override

    // Output
    bool output_bias = false;   // Output projection bias
    bool tied_embeddings = true; // Fall back to tok_embd if output is NULL

    // Post-norm (Gemma 3/4, CogVLM)
    bool attn_post_norm = false;
    bool ffn_post_norm  = false;
    bool post_norm_after_residual = false; // Post-norm applied to (attn+residual) not just attn (GLM4-MOE)

    // Gemma-specific features (also used by others)
    bool logit_softcap     = false;  // tanh(logits/cap)*cap (Gemma 3/4)
    bool token_embd_scale  = false;  // Scale token embeddings by sqrt(n_embd) (Gemma)

    // Input features
    bool combined_qkv  = false;  // Use wqkv instead of separate wq/wk/wv (GPT-2, Bloom, Falcon, Jais)
    bool use_pos_embd  = false;  // Learned position embeddings (GPT-2, Jais)
    bool global_tok_norm = false; // Pre-layer token normalization (Bloom)

    // Residual connection
    bool parallel_ffn  = false;  // FFN uses pre-attn norm instead of post-attn (Falcon)

    // Attention scaling
    bool q_pre_scale   = false;  // Scale Q by 1/sqrt(head_dim) before attention (Phi-2)
    bool yarn_kq_scale = false;  // DeepSeek-style YaRN kq pre-scaling without MLA

    // Attention cache mode
    bool no_attn_cache = false;  // No KV cache (embedding models, diffusion)

    // Hybrid attention/SSM
    bool hybrid = false;         // Layers alternate between attention and SSM (Jamba, Granite-Hybrid)
    bool hybrid_parallel = false; // All layers run BOTH attention AND SSM in parallel (Falcon-H1)
    bool hybrid_delta = false;   // Recurrent layers use delta-net instead of Mamba/Mamba2
    bool hybrid_delta_interleave_repeat = false; // Repeat Q/K heads via grouped expansion when V heads > K heads (Qwen3Next)

    // Encoder (BERT-style post-norm)
    bool encoder_post_norm = false; // Post-norm encoder: norm AFTER attn/FFN, not before (BERT family)

    // Pure SSM (Mamba)
    bool pure_ssm = false;          // ALL layers use SSM instead of attention (Mamba, Mamba2)

    // Vision expert routing (CogVLM)
    bool vision_expert = false;     // Select between text/vision weight tensors based on input type

    // Deepstack embedding injection (Qwen3VL)
    bool deepstack = false;         // Add slices of input embeddings to early layer outputs

    // Per-layer token embeddings / residual injection (Gemma4)
    bool per_layer_embd = false;

    // Per-layer attention geometry (Gemma4)
    bool per_layer_attn_dims = false;
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

// Architecture metadata (layer operations, default activation, cache mode)
struct llm_arch_meta {
    const char * layer_ops;
    llm_ffn_op_type default_act;
    bool no_attn_cache;
};

const llm_arch_meta * get_arch_meta(llm_arch arch);

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

struct llm_build_std_transformer : public llm_build_delta_net_base {
    llm_build_std_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_transformer_config & config = {});

private:
    ggml_tensor * build_norm_gated(
            ggml_tensor * input,
            ggml_tensor * weights,
            ggml_tensor * gate,
            int           layer);

    ggml_tensor * build_layer_hybrid_delta_net(
            llm_graph_input_rs * inp,
            ggml_tensor *        cur,
            int                  il);

    const llama_model & model;
    const llm_transformer_config cfg_;
};

struct llm_altup_transformer_config {
    bool token_embd_scale = true;
    bool v_norm = true;
    bool logit_softcap = true;
    int n_layer_sparsity = 10;
    float sparsity_std_mul = 1.6448533535003662f;
};

struct llm_build_altup_transformer : public llm_graph_context {
    llm_build_altup_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_altup_transformer_config & config = {});

private:
    ggml_tensor * calc_magnitude(ggml_tensor * x);
    ggml_tensor * laurel(ggml_tensor * cur, int il);
    ggml_tensor * gaussian_topk(ggml_tensor * x);
    ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
    ggml_tensor * altup_predict(ggml_tensor * cur, int il);
    ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);

    const llama_model & model;
    const llm_altup_transformer_config cfg_;
    const int64_t n_embd_head;
    const int64_t n_embd_altup;
    const int64_t n_altup;
    const int i_altup_act;
};

struct llm_t5_transformer_config {
    bool decoder = false;
};

struct llm_build_t5_transformer : public llm_graph_context {
    llm_build_t5_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_t5_transformer_config & config = {});

private:
    ggml_tensor * build_layer_self_attn(
            llm_graph_input_attn_no_cache * inp_attn_no_cache,
            llm_graph_input_attn_kv *       inp_attn_kv,
            ggml_tensor *                   pos_bucket,
            ggml_tensor *                   cur,
            int                             il);

    ggml_tensor * build_layer_cross_attn(
            llm_graph_input_attn_cross * inp_attn_cross,
            ggml_tensor *                embd_enc,
            ggml_tensor *                cur,
            int                          il);

    ggml_tensor * build_layer_ffn(
            ggml_tensor * cur,
            int           il);

    const llama_model & model;
    const llm_t5_transformer_config cfg_;
};

struct llm_hybrid_mamba2_transformer_config {
    llm_norm_type norm = LLM_NORM_RMS;
    llm_ffn_op_type ffn_act = LLM_FFN_SWIGLU;
    llm_ffn_gate_type ffn_type = LLM_FFN_SEQ;
    bool combined_qkv = true;
    bool qk_norm = true;
    bool use_rope = true;
    bool attn_bias = false;
    bool attn_post_norm = true;
    bool ffn_post_norm = true;
};

struct llm_build_hybrid_mamba2_transformer : public llm_build_mamba_base {
    llm_build_hybrid_mamba2_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_hybrid_mamba2_transformer_config & config = {});

private:
    ggml_tensor * build_layer_attn(
            llm_graph_input_attn_kv * inp_attn,
            ggml_tensor *             cur,
            int                       il);

    ggml_tensor * build_layer_mamba2(
            llm_graph_input_rs * inp_rs,
            ggml_tensor *        cur,
            int                  il);

    ggml_tensor * build_layer_ffn(
            ggml_tensor * cur,
            int           il);

    const llama_model & model;
    const llm_hybrid_mamba2_transformer_config cfg_;
    ggml_tensor * inp_pos_ = nullptr;
};

struct llm_hybrid_shortconv_transformer_config {
    llm_norm_type norm = LLM_NORM_RMS;
    llm_ffn_op_type dense_ffn_act = LLM_FFN_SILU;
    llm_ffn_gate_type dense_ffn_type = LLM_FFN_PAR;
    bool qk_norm = true;
    bool moe_norm_weights = true;
};

template <bool iswa>
struct llm_build_hybrid_shortconv_transformer : public llm_graph_context {
    llm_build_hybrid_shortconv_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_hybrid_shortconv_transformer_config & config = {});

private:
    ggml_tensor * build_layer_attn(
            ggml_tensor * cur,
            int           il);

    ggml_tensor * build_layer_shortconv(
            llm_graph_input_rs * inp_recr,
            ggml_tensor *        cur,
            int                  il);

    ggml_tensor * build_layer_ffn(
            ggml_tensor * cur,
            int           il);

    const llama_model & model;
    const llm_hybrid_shortconv_transformer_config cfg_;
    ggml_tensor * inp_pos_ = nullptr;
    llm_graph_input_rs * inp_recr_ = nullptr;
    llm_graph_input_attn_kv * inp_attn_kv_ = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa_ = nullptr;
};

struct llm_hybrid_mamba2_single_op_transformer_config {
    llm_norm_type norm = LLM_NORM_RMS;
    llm_ffn_op_type dense_ffn_act = LLM_FFN_RELU_SQR;
    llm_ffn_gate_type dense_ffn_type = LLM_FFN_PAR;
    bool attn_bias = false;
    bool use_hparams_attn_scale = false;
    bool moe_norm_weights = false;
    llama_expert_gating_func_type moe_gating = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    bool shared_expert = false;
    bool latent_moe = false;
    bool apply_cvec_after_residual = true;
};

struct llm_build_hybrid_mamba2_single_op_transformer : public llm_build_mamba_base {
    llm_build_hybrid_mamba2_single_op_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_hybrid_mamba2_single_op_transformer_config & config = {});

private:
    ggml_tensor * build_layer_attn(
            llm_graph_input_attn_kv * inp_attn,
            ggml_tensor *             cur,
            int                       il);

    ggml_tensor * build_layer_mamba2(
            llm_graph_input_rs * inp_rs,
            ggml_tensor *        cur,
            int                  il) const;

    ggml_tensor * build_layer_ffn(
            ggml_tensor * cur,
            int           il);

    const llama_model & model;
    const llm_hybrid_mamba2_single_op_transformer_config cfg_;
};

struct llm_build_mla_kda_hybrid : public llm_build_delta_net_base {
    llm_build_mla_kda_hybrid(
            const llama_model & model,
            const llm_graph_params & params);

private:
    ggml_tensor * build_norm_sigmoid_gated(
            ggml_tensor * input,
            ggml_tensor * weights,
            ggml_tensor * gate,
            int           layer);

    ggml_tensor * build_kda_conv1d(
            ggml_tensor * conv_states_all,
            ggml_tensor * conv_state_all,
            ggml_tensor * input,
            ggml_tensor * proj_w,
            ggml_tensor * conv_w,
            int64_t       qkv,
            int64_t       kv_head);

    ggml_tensor * build_layer_kda(
            llm_graph_input_rs * inp,
            ggml_tensor *        cur,
            int                  il);

    ggml_tensor * build_layer_mla(
            llm_graph_input_attn_kv * inp_attn_kv,
            llm_graph_input_attn_k   * inp_attn_k,
            ggml_tensor *             cur,
            int                       il);

    ggml_tensor * build_layer_ffn(
            ggml_tensor * cur,
            int           il);

    const llama_model & model;
};

struct llm_mla_transformer_config {
    float embd_scale = 1.0f;
    float residual_scale = 1.0f;
    float lmhead_scale = 1.0f;
    bool rope_factors = false;
    bool flatten_v = false;
    bool absorb_kv = false;
    bool yarn_kq_scale = false;
    bool attn_temp = false;
    uint32_t nextn_layers = 0;
    llm_ffn_op_type ffn_act = LLM_FFN_SILU;
    llm_ffn_gate_type ffn_type = LLM_FFN_PAR;
};

struct llm_build_mla_transformer : public llm_graph_context {
    llm_build_mla_transformer(
            const llama_model & model,
            const llm_graph_params & params,
            const llm_mla_transformer_config & config = {});

private:
    ggml_tensor * build_layer_mla(
            llm_graph_input_attn_kv * inp_attn,
            llm_graph_input_attn_k   * inp_attn_k,
            ggml_tensor *             cur,
            int                       il);

    ggml_tensor * build_layer_ffn(
            ggml_tensor * cur,
            int           il);

    const llama_model & model;
    const llm_mla_transformer_config cfg_;
    ggml_tensor * inp_pos_ = nullptr;
    ggml_tensor * inp_attn_scale_ = nullptr;
};
