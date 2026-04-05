// llama-graph-builder.cpp -- Generic transformer graph builder
//
// Implements the standard transformer pattern used by ~80% of architectures:
//   Embed -> [scale] -> (Norm -> Attn -> [PostNorm] -> Norm -> FFN/MoE -> [PostNorm]) x N
//   -> Norm -> Head -> [softcap]
//
// Supports: dense FFN, MoE FFN, ISWA, QK-norm, post-norms, logit softcap,
// token embedding scaling, all norm/activation/FFN-layout combinations.
//
// See llama-graph-builder.h for configuration options.

#include "llama-graph-builder.h"
#include "models/models.h"

#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// Per-architecture metadata table
//
// Maps known architectures to their layer operations, default activation,
// and special flags. This eliminates per-architecture switch cases in
// build_graph() — the builder self-configures from this table + auto-detect.
//
// Priority: GGUF metadata > this table > tensor auto-detection.
// ---------------------------------------------------------------------------

// Common layer operation patterns
#define OPS_STD       "norm,rope,attn,filter,residual,ffn_norm,ffn,residual,cvec"
#define OPS_NO_ROPE   "norm,attn,filter,residual,ffn_norm,ffn,residual,cvec"
#define OPS_QK        "norm,qkv,qk_norm,rope,attn,filter,residual,ffn_norm,ffn,residual,cvec"
#define OPS_PN        "norm,rope,attn,filter,post_norm,residual,ffn_norm,ffn,ffn_post_norm,residual,cvec"
#define OPS_PN_QK     "norm,qkv,qk_norm,rope,attn,filter,post_norm,residual,ffn_norm,ffn,ffn_post_norm,residual,cvec"
#define OPS_PN_ATT    "norm,rope,attn,filter,post_norm,residual,ffn_norm,ffn,residual,cvec"

// Activation shorthands (LLM_FFN_SILU is the default for unspecified)
#define ACT_DEFAULT LLM_FFN_SILU
#define ACT_GELU    LLM_FFN_GELU
#define ACT_RELU2   LLM_FFN_RELU_SQR
#define ACT_SWIGLU  LLM_FFN_SWIGLU

struct llm_arch_meta {
    const char *    layer_ops;
    llm_ffn_op_type default_act;
    bool            no_attn_cache;
};

static const llm_arch_meta * get_arch_meta(llm_arch arch) {
    // ── Standard RoPE ────────────────────────────────────────────────
    static const llm_arch_meta STD           = { OPS_STD,     ACT_DEFAULT, false };
    static const llm_arch_meta STD_GELU      = { OPS_STD,     ACT_GELU,    false };
    static const llm_arch_meta STD_RELU2     = { OPS_STD,     ACT_RELU2,   false };
    static const llm_arch_meta STD_SWIGLU    = { OPS_STD,     ACT_SWIGLU,  false };
    static const llm_arch_meta STD_NC        = { OPS_STD,     ACT_DEFAULT, true  };
    static const llm_arch_meta STD_GELU_NC   = { OPS_STD,     ACT_GELU,    true  };

    // ── No RoPE ──────────────────────────────────────────────────────
    static const llm_arch_meta NOROPE        = { OPS_NO_ROPE, ACT_DEFAULT, false };
    static const llm_arch_meta NOROPE_GELU   = { OPS_NO_ROPE, ACT_GELU,    false };

    // ── QK-norm ──────────────────────────────────────────────────────
    static const llm_arch_meta QK            = { OPS_QK,      ACT_DEFAULT, false };
    static const llm_arch_meta QK_NC         = { OPS_QK,      ACT_DEFAULT, true  };

    // ── Post-norms ───────────────────────────────────────────────────
    static const llm_arch_meta PN_QK_GELU    = { OPS_PN_QK,   ACT_GELU,    false };
    static const llm_arch_meta PN_QK         = { OPS_PN_QK,   ACT_DEFAULT, false };
    static const llm_arch_meta PN            = { OPS_PN,       ACT_DEFAULT, false };
    static const llm_arch_meta PN_ATT        = { OPS_PN_ATT,  ACT_DEFAULT, false };

    switch (arch) {
        // ── Standard with RoPE + SiLU (majority) ──
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_FALCON:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2VL:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
            return &STD;

        // ── Standard + GELU ──
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_GEMMA:
            return &STD_GELU;

        // ── Standard + RELU_SQR ──
        case LLM_ARCH_JAIS2:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_ARCEE:
            return &STD_RELU2;

        // ── Standard + SWIGLU ──
        case LLM_ARCH_CHATGLM:
            return &STD_SWIGLU;

        // ── Standard + no KV cache (embedding models) ──
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_DREAM:
            return &STD_NC;

        case LLM_ARCH_LLADA_MOE:
            return &STD_NC;

        case LLM_ARCH_GEMMA_EMBEDDING:
            return &STD_GELU_NC;

        // ── No RoPE + SiLU ──
        case LLM_ARCH_JAIS:
            return &NOROPE;

        // ── No RoPE + GELU ──
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MPT:
        case LLM_ARCH_GPT2:
            return &NOROPE_GELU;

        // ── QK-norm ──
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAINCODER:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_MINIMAX_M2:
            return &QK;

        case LLM_ARCH_RND1:
            return &QK_NC;

        // ── Post-norms + QK-norm + GELU ──
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
            return &PN_QK_GELU;

        // ── Post-norms + QK-norm + SiLU ──
        case LLM_ARCH_OLMO2:
            return &PN_QK;

        // ── Post-norms (both) ──
        case LLM_ARCH_GROK:
            return &PN;

        // ── Attention post-norm only ──
        case LLM_ARCH_SEED_OSS:
            return &PN_ATT;

        default:
            return nullptr;
    }
}

#undef OPS_STD
#undef OPS_NO_ROPE
#undef OPS_QK
#undef OPS_PN
#undef OPS_PN_QK
#undef OPS_PN_ATT
#undef ACT_DEFAULT
#undef ACT_GELU
#undef ACT_RELU2
#undef ACT_SWIGLU

llm_build_std_transformer::llm_build_std_transformer(
        const llama_model & model,
        const llm_graph_params & params,
        const llm_transformer_config & config) : llm_build_mamba_base(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // ==============================
    // 1. Input embeddings
    // ==============================
    inpL = build_inp_embd(model.tok_embd);

    // Gemma-style: scale token embeddings by sqrt(n_embd), skip for raw embeddings (vision)
    if (config.token_embd_scale) {
        inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(float(n_embd)) : 1.0f);
        cb(inpL, "inp_scaled", -1);
    }

    // Learned position embeddings (GPT-2, Jais)
    if (config.use_pos_embd && model.pos_embd) {
        ggml_tensor * inp_pos_embd = build_inp_pos();
        ggml_tensor * pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos_embd);
        cb(pos, "pos_embd", -1);
        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL_pos", -1);
    }

    // Global token normalization before layers (Bloom)
    if (config.global_tok_norm && model.tok_norm) {
        inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, config.norm, 0);
        cb(inpL, "inp_norm", -1);
    }

    // Position tensor (for RoPE — skip if model doesn't actually use RoPE)
    ggml_tensor * inp_pos = (config.use_rope && rope_type != LLAMA_ROPE_TYPE_NONE)
        ? build_inp_pos() : nullptr;

    const int effective_n_layer = n_layer - (int) hparams.nextn_predict_layers;

    // Per-layer embed injection (Gemma 4) — testing disabled
    ggml_tensor * inp_per_layer = nullptr;
    if (false && model.tok_embd_per_layer) {
        const int64_t n_embd_per_layer = hparams.n_embd_per_layer;

        ggml_tensor * per_layer_raw;
        if (ubatch.token && res->t_inp_tokens) {
            per_layer_raw = ggml_get_rows(ctx0, model.tok_embd_per_layer, res->t_inp_tokens);
            per_layer_raw = ggml_reshape_3d(ctx0, per_layer_raw,
                    n_embd_per_layer, effective_n_layer, n_tokens);
            per_layer_raw = ggml_scale(ctx0, per_layer_raw, sqrtf(float(n_embd_per_layer)));
        } else {
            const int64_t embd_size = model.tok_embd_per_layer->ne[0];
            ggml_tensor * padding = ggml_view_1d(ctx0, model.tok_embd_per_layer, embd_size, 0);
            per_layer_raw = ggml_cast(ctx0, padding, GGML_TYPE_F32);
            per_layer_raw = ggml_reshape_3d(ctx0, per_layer_raw,
                    n_embd_per_layer, effective_n_layer, 1);
        }
        cb(per_layer_raw, "per_layer_raw", -1);

        ggml_tensor * per_layer_proj = ggml_mul_mat(ctx0, model.per_layer_model_proj, inpL);
        per_layer_proj = ggml_scale(ctx0, per_layer_proj, 1.0f / sqrtf(float(n_embd)));
        per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj,
                n_embd_per_layer, effective_n_layer, n_tokens);
        per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm,
                nullptr, LLM_NORM_RMS, -1);

        inp_per_layer = ggml_add(ctx0, per_layer_proj, per_layer_raw);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, 1.0f / sqrtf(2.0f));
        inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
        cb(inp_per_layer, "inp_per_layer", -1);
    }

    // Attention input -- hybrid, no-cache, ISWA, or standard KV cache
    llm_graph_input_mem_hybrid    * inp_hybrid    = nullptr;
    llm_graph_input_attn_no_cache * inp_attn_nc   = nullptr;
    llm_graph_input_attn_kv_iswa  * inp_attn_iswa = nullptr;
    llm_graph_input_attn_kv       * inp_attn_kv   = nullptr;

    if (config.hybrid) {
        inp_hybrid = build_inp_mem_hybrid();
    } else if (config.no_attn_cache) {
        inp_attn_nc = build_attn_inp_no_cache();
    } else if (config.iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn_kv = build_attn_inp_kv();
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f/sqrtf(float(n_embd_head))
        : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // ==============================
    // 2. Transformer layers
    // ==============================

    for (int il = 0; il < effective_n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // --- Pre-attention norm ---
        ggml_tensor * norm_b = (config.norm == LLM_NORM) ? model.layers[il].attn_norm_b : nullptr;
        cur = build_norm(inpL, model.layers[il].attn_norm, norm_b, config.norm, il);
        cb(cur, "attn_norm", il);

        // Save for parallel residual (Falcon-style: FFN uses attn_norm output)
        ggml_tensor * attn_norm_out = cur;

        // --- Self-attention or SSM ---
        if (config.hybrid && hparams.n_head_kv(il) == 0) {
            // SSM (Mamba) layer for hybrid models
            cur = build_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
        } else {
            // Per-layer RoPE frequency (ISWA models have different freq per layer)
            float rope_freq_base  = freq_base;
            float rope_freq_scale = freq_scale;
            ggml_tensor * rope_factors = nullptr;

            if (config.use_rope) {
                if (config.iswa) {
                    rope_freq_base  = model.get_rope_freq_base(cparams, il);
                    rope_freq_scale = model.get_rope_freq_scale(cparams, il);
                }
                rope_factors = model.get_rope_factors(cparams, il);
            }

            // Q, K, V projections
            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;

            // Auto-detect combined QKV from tensor presence
            bool use_combined_qkv = config.combined_qkv ||
                (model.layers[il].wqkv != nullptr && model.layers[il].wq == nullptr);

            if (use_combined_qkv && model.layers[il].wqkv) {
                // Combined QKV projection (GPT-2, Bloom, Falcon, Jais)
                ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);
                if (model.layers[il].bqkv) {
                    qkv = ggml_add(ctx0, qkv, model.layers[il].bqkv);
                }
                cb(qkv, "qkv", il);

                // Split via 3D views
                Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
                        n_embd_head*qkv->nb[0], qkv->nb[1], 0);
                Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
                        n_embd_head*qkv->nb[0], qkv->nb[1], n_embd*qkv->nb[0]);
                Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
                        n_embd_head*qkv->nb[0], qkv->nb[1], (n_embd + n_embd_k_gqa)*qkv->nb[0]);
            } else {
                // Separate Q, K, V projections (standard)
                Qcur = build_lora_mm(model.layers[il].wq, cur);
                Kcur = build_lora_mm(model.layers[il].wk, cur);
                Vcur = build_lora_mm(model.layers[il].wv, cur);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Attention bias
            if (config.attn_bias) {
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                }
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                }
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                }
            }

            // Reshape for multi-head attention (only for separate Q/K/V)
            if (!config.combined_qkv) {
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            }

            // QK normalization (before RoPE)
            // Detect pre-reshape vs post-reshape norm from weight shape
            if (config.qk_norm) {
                if (model.layers[il].attn_q_norm) {
                    const bool pre_reshape = (model.layers[il].attn_q_norm->ne[0] != n_embd_head);
                    if (pre_reshape && !use_combined_qkv) {
                        // Pre-reshape norm: undo reshape, norm, re-reshape (MiniMax-M2)
                        Qcur = ggml_reshape_2d(ctx0, Qcur, n_embd_head * n_head, n_tokens);
                        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, config.norm, il);
                        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                    } else {
                        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, config.norm, il);
                    }
                    cb(Qcur, "Qcur_norm", il);
                }
                if (model.layers[il].attn_k_norm) {
                    const bool pre_reshape = (model.layers[il].attn_k_norm->ne[0] != n_embd_head);
                    if (pre_reshape && !use_combined_qkv) {
                        Kcur = ggml_reshape_2d(ctx0, Kcur, n_embd_head * n_head_kv, n_tokens);
                        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, config.norm, il);
                        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                    } else {
                        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, config.norm, il);
                    }
                    cb(Kcur, "Kcur_norm", il);
                }
            }

            // V normalization (Gemma 4 — raw RMS norm without learned weights)
            if (config.v_norm) {
                Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
                cb(Vcur, "Vcur_norm", il);
            }

            // RoPE position encoding
            if (config.use_rope && rope_type != LLAMA_ROPE_TYPE_NONE) {
                if (hparams.use_mrope()) {
                    // Multi-section RoPE (Qwen2VL, GLM4, PaddleOCR)
                    int sections[4];
                    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

                    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, sections, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors,
                            n_rot, sections, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                } else {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                }

                cb(Qcur, "Qcur_rope", il);
                cb(Kcur, "Kcur_rope", il);
            }

            // Attention computation + output projection
            ggml_tensor * wo_b = config.attn_bias ? model.layers[il].bo : nullptr;

            if (config.hybrid) {
                cur = build_attn(inp_hybrid->get_attn(), model.layers[il].wo, wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            } else if (config.no_attn_cache) {
                cur = build_attn(inp_attn_nc, model.layers[il].wo, wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            } else if (config.iswa) {
                cur = build_attn(inp_attn_iswa, model.layers[il].wo, wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            } else {
                cur = build_attn(inp_attn_kv, model.layers[il].wo, wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            }
        }

        // Output token filtering at last layer (before post-norm and residual)
        if (il == effective_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Attention post-norm (after filtering)
        if (config.attn_post_norm && model.layers[il].attn_post_norm) {
            cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, config.norm, il);
            cb(cur, "attn_post_norm", il);
        }

        // Residual connection after attention
        ggml_tensor * attn_out = cur;
        ggml_tensor * ffn_inp;

        if (config.parallel_ffn) {
            // Parallel residual (Falcon): FFN uses attn_norm, not attn output
            ffn_inp = inpSA;  // Will add both attn + FFN to residual at end
        } else {
            // Residual scaling (Granite)
            if (hparams.f_residual_scale != 0.0f) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }

        // --- Pre-FFN norm ---
        ggml_tensor * ffn_norm_input = config.parallel_ffn ? attn_norm_out : ffn_inp;
        ggml_tensor * ffn_norm_b = (config.norm == LLM_NORM) ? model.layers[il].ffn_norm_b : nullptr;
        if (model.layers[il].ffn_norm) {
            cur = build_norm(ffn_norm_input, model.layers[il].ffn_norm, ffn_norm_b, config.norm, il);
        } else {
            // No FFN norm: use the input directly (Falcon with parallel residual)
            cur = ffn_norm_input;
        }
        cb(cur, "ffn_norm", il);

        // --- Feed-forward network (dense, MoE, dual MoE+MLP, or MoE+shared) ---
        if (config.moe && model.layers[il].ffn_gate_inp) {
            // Check for Gemma4-style dual FFN (shared MLP + MoE with separate post-norms)
            if (model.layers[il].ffn_post_norm_1) {
                // Dual FFN: shared MLP path
                ggml_tensor * cur_mlp = build_norm(attn_out,
                        model.layers[il].ffn_norm, nullptr, config.norm, il);
                cb(cur_mlp, "ffn_norm_1", il);

                cur_mlp = build_ffn(cur_mlp,
                        model.layers[il].ffn_up,   nullptr, nullptr,
                        model.layers[il].ffn_gate, nullptr, nullptr,
                        model.layers[il].ffn_down, nullptr, nullptr,
                        nullptr, config.act, LLM_FFN_PAR, il);
                cur_mlp = build_norm(cur_mlp,
                        model.layers[il].ffn_post_norm_1, nullptr, config.norm, il);
                cb(cur_mlp, "ffn_mlp", il);

                // Dual FFN: MoE expert path
                ggml_tensor * cur_moe = build_norm(attn_out,
                        model.layers[il].ffn_pre_norm_2, nullptr, config.norm, il);
                cb(cur_moe, "ffn_norm_2", il);

                // Custom MoE router scaling (Gemma4)
                ggml_tensor * logits = nullptr;
                if (model.layers[il].ffn_gate_inp_s) {
                    ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
                    tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf(float(n_embd)));
                    tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
                    logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp);
                    cb(logits, "ffn_moe_logits", il);
                }

                cur_moe = build_moe_ffn(cur_moe,
                        logits ? nullptr : model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        config.act, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_NONE ? (llama_expert_gating_func_type) hparams.expert_gating_func : LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX),
                        il, logits,
                        model.layers[il].ffn_gate_up_exps,
                        nullptr, nullptr,
                        model.layers[il].ffn_down_exps_s);
                cur_moe = build_norm(cur_moe,
                        model.layers[il].ffn_post_norm_2, nullptr, config.norm, il);
                cb(cur_moe, "ffn_moe", il);

                cur = ggml_add(ctx0, cur_mlp, cur_moe);
                cb(cur, "ffn_out", il);
            } else {
                // Standard MoE
                ggml_tensor * moe_out = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        config.act, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_NONE ? (llama_expert_gating_func_type) hparams.expert_gating_func : LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX),
                        il);
                cb(moe_out, "ffn_moe_out", il);

                if (config.moe_shared && model.layers[il].ffn_up_shexp) {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                            model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                            model.layers[il].ffn_down_shexp, nullptr, nullptr,
                            nullptr, config.act, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);
                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                } else {
                    cur = moe_out;
                }
            }
        } else {
            // Dense FFN
            ggml_tensor * up_b   = config.ffn_bias ? model.layers[il].ffn_up_b   : nullptr;
            ggml_tensor * gate_b = config.ffn_bias ? model.layers[il].ffn_gate_b : nullptr;
            ggml_tensor * down_b = config.ffn_bias ? model.layers[il].ffn_down_b : nullptr;

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   up_b,   nullptr,
                    model.layers[il].ffn_gate, gate_b, nullptr,
                    model.layers[il].ffn_down, down_b, nullptr,
                    nullptr,
                    config.act, config.ffn_type, il);
            cb(cur, "ffn_out", il);
        }

        // FFN post-norm
        if (config.ffn_post_norm && model.layers[il].ffn_post_norm) {
            cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, config.norm, il);
            cb(cur, "ffn_post_norm", il);
        }

        // Final residual connection
        if (config.parallel_ffn) {
            cur = ggml_add(ctx0, cur, attn_out);
            cur = ggml_add(ctx0, cur, inpSA);
        } else {
            // Residual scaling on FFN output (Granite)
            if (hparams.f_residual_scale != 0.0f) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }
            cur = ggml_add(ctx0, cur, ffn_inp);
        }
        cb(cur, "l_out", il);

        // Per-layer embed injection (Gemma 4)
        if (inp_per_layer && model.layers[il].per_layer_inp_gate) {
            ggml_tensor * pe_in = cur;

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur);
            cur = ggml_gelu(ctx0, cur);

            // Extract 2D slice for this layer from [n_embd_per_layer, n_tokens, n_layer]
            ggml_tensor * inp_this_layer = ggml_view_2d(ctx0, inp_per_layer,
                    inp_per_layer->ne[0], inp_per_layer->ne[1],
                    ggml_row_size(inp_per_layer->type, inp_per_layer->ne[0]),
                    il * inp_per_layer->ne[0] * inp_per_layer->ne[1] * ggml_element_size(inp_per_layer));

            if (il == effective_n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur);
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            cur = ggml_add(ctx0, pe_in, cur);
        }

        cur = build_cvec(cur, il);
        inpL = cur;
    }

    // ==============================
    // 3. Output head
    // ==============================
    cur = inpL;

    ggml_tensor * out_norm_b = (config.norm == LLM_NORM) ? model.output_norm_b : nullptr;
    cur = build_norm(cur, model.output_norm, out_norm_b, config.norm, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    if (config.output_bias && model.output_b) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }

    // Final logit softcapping: cap * tanh(logits / cap)
    if (config.logit_softcap && hparams.f_final_logit_softcapping > 0.0f) {
        float cap = hparams.f_final_logit_softcapping;
        cur = ggml_scale(ctx0, cur, 1.0f / cap);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, cap);
    }

    // Logit scaling (Granite, Command-R)
    if (hparams.f_logit_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }

    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// ---------------------------------------------------------------------------
// Auto-configure from hparams
// ---------------------------------------------------------------------------

bool llm_transformer_config_from_hparams(
        const llama_hparams & hparams,
        const llama_model   & model,
        llm_transformer_config & config) {

    // Reject architectures we can't handle
    if (hparams.ssm_d_inner > 0 || hparams.is_recurrent(0)) {
        return false;  // SSM/hybrid models need custom builders
    }
    if (hparams.is_mla()) {
        return false;  // MLA (DeepSeek v2) needs custom builder
    }

    // Normalization: detect by which epsilon is set
    if (hparams.f_norm_rms_eps > 0.0f && hparams.f_norm_eps == 0.0f) {
        config.norm = LLM_NORM_RMS;
    } else {
        config.norm = LLM_NORM;
    }

    // RoPE
    config.use_rope = (hparams.rope_type != LLAMA_ROPE_TYPE_NONE);

    // Position embeddings (if no RoPE and model has pos_embd)
    config.use_pos_embd = (!config.use_rope && model.pos_embd != nullptr);

    // ISWA (sliding window attention)
    config.iswa = (hparams.swa_type != LLAMA_SWA_TYPE_NONE && hparams.n_swa > 0);

    // MoE
    if (hparams.n_expert > 0) {
        config.moe = true;
        config.moe_shared = (hparams.n_expert_shared > 0);
    }

    // QK normalization: detect by checking if first layer has q_norm tensor
    if (hparams.use_kq_norm || (model.layers.size() > 0 && model.layers[0].attn_q_norm != nullptr)) {
        config.qk_norm = true;
    }

    // Combined QKV: detect by checking if first layer has wqkv tensor
    if (model.layers.size() > 0 && model.layers[0].wqkv != nullptr) {
        config.combined_qkv = true;
    }

    // Attention bias: detect from layer tensors
    if (model.layers.size() > 0 && (model.layers[0].bq != nullptr || model.layers[0].bqkv != nullptr)) {
        config.attn_bias = true;
    }

    // FFN bias
    if (model.layers.size() > 0 && model.layers[0].ffn_up_b != nullptr) {
        config.ffn_bias = true;
    }

    // Output bias
    if (model.output_b != nullptr) {
        config.output_bias = true;
    }

    // Parallel residual (Falcon-style)
    config.parallel_ffn = hparams.use_par_res;

    // Global token normalization (Bloom)
    config.global_tok_norm = (model.tok_norm != nullptr);

    // Post-norms: detect from layer tensors
    if (model.layers.size() > 0) {
        config.attn_post_norm = (model.layers[0].attn_post_norm != nullptr);
        config.ffn_post_norm  = (model.layers[0].ffn_post_norm  != nullptr);
    }

    // Logit softcapping
    if (hparams.f_final_logit_softcapping > 0.0f && hparams.f_final_logit_softcapping < 1000.0f) {
        config.logit_softcap = true;
    }

    // Token embedding scaling (Gemma-style): detect by f_embedding_scale or architecture
    if (hparams.f_embedding_scale > 0.0f) {
        config.token_embd_scale = true;
    }

    // Activation: auto-detect from GGUF metadata if available
    if (strcmp(hparams.ffn_activation, "gelu") == 0 ||
        strcmp(hparams.ffn_activation, "gelu_new") == 0 ||
        strcmp(hparams.ffn_activation, "gelu_fast") == 0 ||
        strcmp(hparams.ffn_activation, "gelu_pytorch_tanh") == 0) {
        config.act = LLM_FFN_GELU;
    } else if (strcmp(hparams.ffn_activation, "relu") == 0) {
        config.act = LLM_FFN_RELU;
    } else if (strcmp(hparams.ffn_activation, "relu_sqr") == 0 ||
               strcmp(hparams.ffn_activation, "relu2") == 0) {
        config.act = LLM_FFN_RELU_SQR;
    } else if (strcmp(hparams.ffn_activation, "swiglu") == 0 ||
               strcmp(hparams.ffn_activation, "silu") == 0) {
        config.act = LLM_FFN_SILU;
    }
    // else: leave at default (will be set by arch table below)

    // FFN type: if no gate tensor in first layer, it's sequential
    if (model.layers.size() > 0 && model.layers[0].ffn_gate == nullptr && !config.moe) {
        config.ffn_type = LLM_FFN_SEQ;
    }

    // ---- Apply per-architecture metadata ----
    // Priority: GGUF metadata > arch table > tensor auto-detection
    const llm_arch_meta * meta = get_arch_meta(model.arch);

    // Layer operations: GGUF overrides arch table
    const char * ops_str = nullptr;
    if (hparams.layer_operations[0] != '\0') {
        ops_str = hparams.layer_operations;
    } else if (meta) {
        ops_str = meta->layer_ops;
    }

    if (ops_str) {
        const std::string ops(ops_str);

        auto has_op = [&ops](const char * token) -> bool {
            return ops.find(token) != std::string::npos;
        };

        if (has_op("post_norm"))     config.attn_post_norm = true;
        if (has_op("ffn_post_norm")) config.ffn_post_norm  = true;
        if (has_op("qk_norm"))       config.qk_norm = true;
        if (!has_op("rope"))         config.use_rope = false;
    }

    // Activation: arch table provides fallback for old GGUF files without ffn_activation
    if (meta && hparams.ffn_activation[0] == '\0') {
        config.act = meta->default_act;
    }

    // No-KV-cache mode (embedding/diffusion models)
    if (meta && meta->no_attn_cache) {
        config.no_attn_cache = true;
    }

    return true;
}
