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
#include "llama-memory-recurrent.h"

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

const llm_arch_meta * get_arch_meta(llm_arch arch) {
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
    static const llm_arch_meta PN_QK_SWIGLU  = { OPS_PN_QK,   ACT_SWIGLU,  false };
    static const llm_arch_meta PN            = { OPS_PN,       ACT_DEFAULT, false };
    static const llm_arch_meta PN_SWIGLU     = { OPS_PN,       ACT_SWIGLU,  false };
    static const llm_arch_meta PN_ATT        = { OPS_PN_ATT,  ACT_DEFAULT, false };
    static const llm_arch_meta PN_ATT_SWIGLU = { OPS_PN_ATT,  ACT_SWIGLU,  false };

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
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_CHAMELEON:
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
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
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

        // ── Encoder (BERT family) ──
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_EUROBERT:
            return &NOROPE_GELU;

        // ── No RoPE + SiLU ──
        case LLM_ARCH_JAIS:
        case LLM_ARCH_JAMBA:
            return &NOROPE;

        // ── Pure SSM (Mamba) ──
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
            return &STD;  // SSM auto-detected from tensors

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
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
            return &QK;

        case LLM_ARCH_RND1:
            return &QK_NC;

        // ── Post-norms + QK-norm + GELU ──
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
            return &PN_QK_GELU;

        // ── Post-norms + QK-norm + SiLU ──
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_AFMOE:
            return &PN_QK;

        // ── Post-norms + QK-norm + SWIGLU ──
        case LLM_ARCH_PLAMO3:
            return &PN_QK_SWIGLU;

        // ── Post-norms (both) ──
        case LLM_ARCH_GROK:
            return &PN;

        // ── Attention post-norm only ──
        case LLM_ARCH_SEED_OSS:
            return &PN_ATT;

        // ── Post-norms + SWIGLU ──
        case LLM_ARCH_GLM4:
            return &PN_SWIGLU;

        // ── Attention post-norm + SWIGLU (post-norm after residual) ──
        case LLM_ARCH_GLM4_MOE:
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
        const llm_transformer_config & config) : llm_build_delta_net_base(params), model(model), cfg_(config) {

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

    // Per-layer embed injection (Gemma 4)
    ggml_tensor * inp_per_layer = nullptr;
    if (config.per_layer_embd && model.per_layer_tok_embd) {
        const int64_t n_embd_per_layer = hparams.n_embd_per_layer;
        inp_per_layer = build_per_layer_inputs_raw(
                model.per_layer_tok_embd,
                n_embd_per_layer,
                effective_n_layer);
        cb(inp_per_layer, "per_layer_raw", -1);
        inp_per_layer = project_per_layer_inputs_common(
                inpL,
                inp_per_layer,
                model.per_layer_model_proj,
                model.per_layer_proj_norm,
                n_embd_per_layer,
                effective_n_layer);
    }

    // Attention input -- pure SSM, hybrid, no-cache, ISWA, or standard KV cache
    llm_graph_input_rs            * rs_inp        = nullptr;
    llm_graph_input_mem_hybrid    * inp_hybrid    = nullptr;
    llm_graph_input_attn_no_cache * inp_attn_nc   = nullptr;
    llm_graph_input_attn_kv_iswa  * inp_attn_iswa = nullptr;
    llm_graph_input_attn_kv       * inp_attn_kv   = nullptr;

    if (config.pure_ssm) {
        rs_inp = build_rs_inp();
    } else if (config.hybrid || config.hybrid_parallel) {
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

        if (config.pure_ssm) {
            // ── PURE SSM PATH (Mamba, Mamba2) ──
            cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            if (hparams.ssm_n_group > 0) {
                cur = build_mamba2_layer(rs_inp, cur, model, ubatch, il);
            } else {
                cur = build_mamba_layer(rs_inp, cur, model, ubatch, il);
            }

            if (il == effective_n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            cur = ggml_add(ctx0, cur, inpL);
            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);
            inpL = cur;
            continue;
        }

        if (config.encoder_post_norm) {
            // ── ENCODER POST-NORM PATH (BERT family) ──
            cur = inpL;

            // Self-attention
            {
                ggml_tensor * Qcur = nullptr;
                ggml_tensor * Kcur = nullptr;
                ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, cur);
                    if (model.layers[il].bqkv) {
                        cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    }
                    Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                    Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                    Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd + hparams.n_embd_v_gqa()));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
                }

                // QK norm (Nomic-BERT, Jina-BERT-V3)
                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_reshape_2d(ctx0, Qcur, n_embd_head * n_head, n_tokens);
                    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, model.layers[il].attn_q_norm_b, LLM_NORM, il);
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                }
                if (model.layers[il].attn_k_norm) {
                    Kcur = ggml_reshape_2d(ctx0, Kcur, n_embd_head * n_head_kv, n_tokens);
                    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, model.layers[il].attn_k_norm_b, LLM_NORM, il);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                }

                // RoPE (Nomic-BERT, Jina-BERT-V3)
                if (config.use_rope && rope_type != LLAMA_ROPE_TYPE_NONE) {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                }

                cur = build_attn(inp_attn_nc, model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            }

            if (il == effective_n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Residual + attention output norm
            cur = ggml_add(ctx0, cur, inpL);
            cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

            // Optional second norm (Jina-BERT-V2)
            if (model.layers[il].attn_norm_2) {
                cur = ggml_add(ctx0, cur, inpL);
                cur = build_norm(cur, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, il);
            }

            ggml_tensor * ffn_inp = cur;

            // FFN (sequential GELU by default)
            if (model.layers[il].ffn_gate) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                        config.act, LLM_FFN_PAR, il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                        NULL, NULL, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                        config.act, LLM_FFN_SEQ, il);
            }

            // Residual + layer output norm
            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);
            inpL = cur;
            continue;
        }

        // ── DECODER PRE-NORM PATH (standard transformers) ──

        // Vision expert routing: select text or vision weight tensors
        ggml_tensor * layer_wo       = model.layers[il].wo;
        ggml_tensor * layer_ffn_gate = model.layers[il].ffn_gate;
        ggml_tensor * layer_ffn_down = model.layers[il].ffn_down;
        ggml_tensor * layer_ffn_up   = model.layers[il].ffn_up;

        if (config.vision_expert && !ubatch.token) {
            layer_wo       = model.layers[il].visexp_attn_wo;
            layer_ffn_gate = model.layers[il].visexp_ffn_gate;
            layer_ffn_down = model.layers[il].visexp_ffn_down;
            layer_ffn_up   = model.layers[il].visexp_ffn_up;
        }

        // --- Pre-attention norm ---
        ggml_tensor * norm_b = (config.norm == LLM_NORM) ? model.layers[il].attn_norm_b : nullptr;
        cur = build_norm(inpL, model.layers[il].attn_norm, norm_b, config.norm, il);
        cb(cur, "attn_norm", il);

        // Save for parallel residual (Falcon-style: FFN uses attn_norm output)
        ggml_tensor * attn_norm_out = cur;

        // --- Self-attention or SSM ---
        if (config.hybrid_parallel) {
            // Parallel attention+SSM: both run on the same layer, outputs summed (Falcon-H1)
            ggml_tensor * attn_cur = cur;
            ggml_tensor * ssm_cur  = cur;

            // Attention path
            {
                float rope_freq_base  = freq_base;
                float rope_freq_scale = freq_scale;
                ggml_tensor * rope_factors = nullptr;

                if (config.use_rope) {
                    rope_factors = model.get_rope_factors(cparams, il);
                }

                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, attn_cur);
                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, attn_cur);
                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, attn_cur);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                if (config.use_rope && rope_type != LLAMA_ROPE_TYPE_NONE) {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                attn_cur = build_attn(inp_hybrid->get_attn(),
                        model.layers[il].wo, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
                cb(attn_cur, "attn_out", il);
            }

            // SSM path (reuses same norm output)
            if (hparams.ssm_n_group > 0) {
                ssm_cur = build_mamba2_layer(inp_hybrid->get_recr(), ssm_cur, model, ubatch, il);
            } else {
                ssm_cur = build_mamba_layer(inp_hybrid->get_recr(), ssm_cur, model, ubatch, il);
            }
            cb(ssm_cur, "ssm_out", il);

            // Sum attention + SSM + residual  
            cur = ggml_add(ctx0, attn_cur, ssm_cur);
            inpSA = ggml_add(ctx0, cur, inpSA);
            cur = inpSA;
        } else if (config.hybrid && hparams.is_recurrent(il)) {
            if (config.hybrid_delta) {
                cur = build_layer_hybrid_delta_net(inp_hybrid->get_recr(), cur, il);
            } else if (hparams.ssm_n_group > 0) {
                // SSM layer for hybrid models (auto-detect Mamba vs Mamba2)
                cur = build_mamba2_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
            } else {
                cur = build_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
            }
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
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            ggml_tensor * gate = nullptr;
            const bool layer_attn_q_gate = config.attn_q_gate && !hparams.is_recurrent(il);
            const bool layer_has_kv = !config.iswa || hparams.has_kv(il);

            // Vision expert: use layer_wqkv for combined QKV routing
            ggml_tensor * layer_wqkv = (config.vision_expert && !ubatch.token)
                ? model.layers[il].visexp_attn_wqkv : model.layers[il].wqkv;

            // Auto-detect combined QKV from tensor presence
            bool use_combined_qkv = config.combined_qkv ||
                (layer_wqkv != nullptr && model.layers[il].wq == nullptr);

            if (layer_attn_q_gate) {
                ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
                cb(Qcur_full, "Qcur_full", il);

                Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
                        ggml_element_size(Qcur_full) * n_embd_head * 2,
                        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
                        0);
                gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
                        ggml_element_size(Qcur_full) * n_embd_head * 2,
                        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
                        ggml_element_size(Qcur_full) * n_embd_head);
                gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

                if (layer_has_kv) {
                    Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
                    Vcur = model.layers[il].wv
                        ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                        : Kcur;
                }
            } else if (use_combined_qkv && layer_wqkv) {
                // Combined QKV projection (GPT-2, Bloom, Falcon, Jais, CogVLM)
                ggml_tensor * qkv = build_lora_mm(layer_wqkv, cur);
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
                if (layer_has_kv) {
                    Kcur = build_lora_mm(model.layers[il].wk, cur);
                    Vcur = model.layers[il].wv
                        ? build_lora_mm(model.layers[il].wv, cur)
                        : Kcur;
                }
            }

            cb(Qcur, "Qcur", il);
            if (Kcur) {
                cb(Kcur, "Kcur", il);
            }
            if (Vcur) {
                cb(Vcur, "Vcur", il);
            }

            // Attention bias
            if (config.attn_bias) {
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                }
                if (Kcur && model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                }
                if (Vcur && model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                }
            }

            // Reshape for multi-head attention (only for separate Q/K/V)
            if (!use_combined_qkv) {
                if (!layer_attn_q_gate) {
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                }
                if (Kcur) {
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                }
                if (Vcur) {
                    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
                }
            }

            // QK normalization (before RoPE)
            // Detect pre-reshape vs post-reshape norm from weight shape
            // Auto-detect norm type: if bias exists, use LayerNorm; else use model's norm type
            if (config.qk_norm) {
                if (model.layers[il].attn_q_norm) {
                    ggml_tensor * q_norm_b = model.layers[il].attn_q_norm_b;
                    llm_norm_type qk_norm_type = q_norm_b ? LLM_NORM : config.norm;
                    const bool pre_reshape = (model.layers[il].attn_q_norm->ne[0] != n_embd_head);
                    if (pre_reshape && !use_combined_qkv) {
                        // Pre-reshape norm: undo reshape, norm, re-reshape (MiniMax-M2)
                        Qcur = ggml_reshape_2d(ctx0, Qcur, n_embd_head * n_head, n_tokens);
                        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, q_norm_b, qk_norm_type, il);
                        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                    } else {
                        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, q_norm_b, qk_norm_type, il);
                    }
                    cb(Qcur, "Qcur_norm", il);
                }
                if (model.layers[il].attn_k_norm) {
                    ggml_tensor * k_norm_b = model.layers[il].attn_k_norm_b;
                    llm_norm_type qk_norm_type = k_norm_b ? LLM_NORM : config.norm;
                    const bool pre_reshape = (model.layers[il].attn_k_norm->ne[0] != n_embd_head);
                    if (Kcur && pre_reshape && !use_combined_qkv) {
                        Kcur = ggml_reshape_2d(ctx0, Kcur, n_embd_head * n_head_kv, n_tokens);
                        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, k_norm_b, qk_norm_type, il);
                        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                    } else if (Kcur) {
                        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, k_norm_b, qk_norm_type, il);
                    }
                    if (Kcur) {
                        cb(Kcur, "Kcur_norm", il);
                    }
                }
            }

            // V normalization (Gemma 4 — raw RMS norm without learned weights)
            if (config.v_norm && Vcur) {
                Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
                cb(Vcur, "Vcur_norm", il);
            }

            // RoPE position encoding
            // Conditional RoPE: skip every Nth layer (AFMoE)
            bool layer_use_rope = config.use_rope && rope_type != LLAMA_ROPE_TYPE_NONE;
            if (layer_use_rope && hparams.n_no_rope_layer_step > 0) {
                layer_use_rope = (il + 1) % hparams.n_no_rope_layer_step != 0;
            }
            if (layer_use_rope) {
                if (hparams.use_mrope()) {
                    // Multi-section RoPE (Qwen2VL, GLM4, PaddleOCR)
                    int sections[4];
                    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

                    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, sections, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    if (Kcur) {
                        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors,
                                n_rot, sections, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                    }
                } else {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    if (Kcur) {
                        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                                n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                    }
                }

                cb(Qcur, "Qcur_rope", il);
                if (Kcur) {
                    cb(Kcur, "Kcur_rope", il);
                }
            }

            // Attention computation + output projection
            ggml_tensor * wo_b = config.attn_bias ? model.layers[il].bo : nullptr;

            // When attention gating or sub-norm is active, defer wo projection
            const bool layer_attn_gate = config.attn_gate && model.layers[il].wqkv_gate != nullptr;
            bool defer_wo = layer_attn_q_gate || layer_attn_gate || (model.layers[il].attn_sub_norm != nullptr);
            ggml_tensor * attn_wo = defer_wo ? nullptr : layer_wo;
            ggml_tensor * attn_wo_b = defer_wo ? nullptr : wo_b;

            if (config.hybrid) {
                cur = build_attn(inp_hybrid->get_attn(), attn_wo, attn_wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            } else if (config.no_attn_cache) {
                cur = build_attn(inp_attn_nc, attn_wo, attn_wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            } else if (config.iswa) {
                cur = build_attn(inp_attn_iswa, attn_wo, attn_wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            } else {
                cur = build_attn(inp_attn_kv, attn_wo, attn_wo_b,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            }

            // Attention gating: sigmoid(gate) * attn_out → wo projection
            if (layer_attn_q_gate && gate) {
                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate", il);
                cur = ggml_mul(ctx0, cur, gate);
                cur = build_lora_mm(layer_wo, cur, model.layers[il].wo_s);
                if (wo_b) { cur = ggml_add(ctx0, cur, wo_b); }
                cb(cur, "attn_o_proj", il);
            } else if (layer_attn_gate) {
                ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_norm_out);
                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate", il);
                cur = ggml_mul(ctx0, cur, gate);
                cur = build_lora_mm(layer_wo, cur, model.layers[il].wo_s);
                cb(cur, "attn_o_proj", il);
            }
            // Attention sub-norm: norm after attention, before wo (BitNet)
            else if (model.layers[il].attn_sub_norm) {
                cur = build_norm(cur, model.layers[il].attn_sub_norm, nullptr, LLM_NORM_RMS, il);
                cb(cur, "attn_sub_norm", il);
                cur = build_lora_mm(layer_wo, cur, model.layers[il].wo_s);
                if (wo_b) { cur = ggml_add(ctx0, cur, wo_b); }
                cb(cur, "attn_out", il);
            }
        }

        // Output token filtering at last layer (before post-norm and residual)
        if (il == effective_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Attention post-norm (after filtering)
        if (config.attn_post_norm && model.layers[il].attn_post_norm && !config.post_norm_after_residual) {
            cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, config.norm, il);
            cb(cur, "attn_post_norm", il);
        }

        // Residual connection after attention
        ggml_tensor * attn_out = cur;
        ggml_tensor * ffn_inp;

        if (config.parallel_ffn) {
            // Parallel residual (Falcon): FFN uses attn_norm, not attn output
            ffn_inp = inpSA;  // Will add both attn + FFN to residual at end
        } else if (config.hybrid_parallel) {
            // Hybrid parallel: cur already includes residual (attn+ssm+residual)
            ffn_inp = cur;
            cb(ffn_inp, "ffn_inp", il);
        } else {
            // Residual scaling (Granite)
            if (hparams.f_residual_scale != 0.0f) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }

        // Post-norm after residual (GLM4-MOE variant): norm applied to residual sum
        if (config.attn_post_norm && model.layers[il].attn_post_norm && config.post_norm_after_residual) {
            cur = build_norm(ffn_inp, model.layers[il].attn_post_norm, nullptr, config.norm, il);
            cb(cur, "attn_post_norm", il);
        }

        // --- Pre-FFN norm ---
        ggml_tensor * ffn_norm_input = config.parallel_ffn ? attn_norm_out : ffn_inp;
        if (!config.parallel_ffn &&
                config.attn_post_norm &&
                config.post_norm_after_residual &&
                model.layers[il].attn_post_norm &&
                model.layers[il].ffn_norm == nullptr) {
            ffn_norm_input = cur;
        }
        ggml_tensor * ffn_norm_b = (config.norm == LLM_NORM) ? model.layers[il].ffn_norm_b : nullptr;
        if (model.layers[il].ffn_norm) {
            cur = build_norm(ffn_norm_input, model.layers[il].ffn_norm, ffn_norm_b, config.norm, il);
        } else if (config.parallel_ffn && model.layers[il].attn_norm_2) {
            // Falcon-40B: separate norm for FFN path in parallel residual mode
            ggml_tensor * norm_2_b = (config.norm == LLM_NORM) ? model.layers[il].attn_norm_2_b : nullptr;
            cur = build_norm(ffn_norm_input, model.layers[il].attn_norm_2, norm_2_b, config.norm, il);
        } else {
            // No FFN norm: use the input directly (Falcon-7B with parallel residual)
            cur = ffn_norm_input;
        }
        cb(cur, "ffn_norm", il);

        // --- Feed-forward network (dense, MoE, dual MoE+MLP, or MoE+shared) ---
        // MoE experts use separate gate tensors, so SWIGLU (fused gate) → SILU (split gate)
        const llm_ffn_op_type moe_act = (config.act == LLM_FFN_SWIGLU) ? LLM_FFN_SILU
                                      : (config.act == LLM_FFN_GEGLU)  ? LLM_FFN_GELU
                                      : (config.act == LLM_FFN_REGLU)  ? LLM_FFN_RELU
                                      : config.act;
        const bool moe_norm_w = config.moe_norm_weights || hparams.expert_weights_norm;
        const llama_expert_gating_func_type moe_gating =
                config.moe_gating != LLAMA_EXPERT_GATING_FUNC_TYPE_NONE
                ? config.moe_gating
                : (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_NONE
                    ? (llama_expert_gating_func_type) hparams.expert_gating_func
                    : LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX);
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
                        moe_act, moe_norm_w,
                        hparams.expert_weights_scale,
                        moe_gating,
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
                // Pre-compute router logits if chunk experts need them
                ggml_tensor * router_logits = nullptr;
                if (model.layers[il].ffn_up_chexps) {
                    router_logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur);
                    cb(router_logits, "ffn_moe_logits", il);
                }

                ggml_tensor * moe_out = build_moe_ffn(cur,
                        router_logits ? nullptr : model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        moe_act, moe_norm_w,
                        hparams.expert_weights_scale,
                        moe_gating,
                        il, router_logits,
                        model.layers[il].ffn_gate_up_exps,
                        model.layers[il].ffn_up_exps_s,
                        model.layers[il].ffn_gate_exps_s,
                        model.layers[il].ffn_down_exps_s);
                cb(moe_out, "ffn_moe_out", il);

                if (config.moe_shared && model.layers[il].ffn_up_shexp) {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                            model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                            model.layers[il].ffn_down_shexp, nullptr, nullptr,
                            nullptr, moe_act, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);
                    if (config.moe_shared_gate && model.layers[il].ffn_gate_inp_shexp) {
                        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
                        cb(shared_gate, "shared_expert_gate", il);
                        shared_gate = ggml_sigmoid(ctx0, shared_gate);
                        cb(shared_gate, "shared_expert_gate_sigmoid", il);
                        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
                        cb(ffn_shexp, "ffn_shexp_gated", il);
                    }
                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                } else if (model.layers[il].ffn_up_chexps) {
                    // Chunk MoE (GroveMoE): second MoE applied to output of first
                    cur = moe_out;
                    const int64_t n_chunk_expert = n_expert / hparams.n_group_experts;
                    ggml_tensor * chunk_moe_out = build_moe_ffn(cur,
                            nullptr,
                            model.layers[il].ffn_up_chexps,
                            model.layers[il].ffn_gate_chexps,
                            model.layers[il].ffn_down_chexps,
                            nullptr,
                            n_chunk_expert, n_expert_used > (uint32_t) n_chunk_expert ? (uint32_t) n_chunk_expert : n_expert_used,
                            moe_act, hparams.expert_weights_norm,
                            hparams.expert_weights_scale,
                            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                            il,
                            router_logits);  // reuse same router logits
                    cb(chunk_moe_out, "ffn_adj_moe_out", il);
                    cur = ggml_add(ctx0, cur, ggml_scale(ctx0, chunk_moe_out, hparams.expert_group_scale));
                    cb(cur, "ffn_out", il);
                } else {
                    cur = moe_out;
                }
            }
        } else if (hparams.xielu_beta[il] != 0.0f) {
            // xIELU activation (Apertus): up → xIELU → down
            cur = build_lora_mm(layer_ffn_up, cur);
            cb(cur, "ffn_up", il);

            cur = ggml_xielu(ctx0, cur,
                    hparams.xielu_alpha_n[il], hparams.xielu_alpha_p[il],
                    hparams.xielu_beta[il], hparams.xielu_eps[il]);
            cb(cur, "ffn_xielu", il);

            cur = build_lora_mm(layer_ffn_down, cur);
            cb(cur, "ffn_out", il);
        } else if (model.layers[il].ffn_sub_norm) {
            // FFN with sub-norm (BitNet): up+gate → silu → sub_norm → down
            cur = build_ffn(cur,
                    layer_ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    layer_ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    nullptr,        nullptr, nullptr,
                    nullptr,
                    config.act, LLM_FFN_PAR, il);
            cb(cur, "ffn_sub_out", il);

            cur = build_norm(cur, model.layers[il].ffn_sub_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_sub_norm", il);

            cur = build_lora_mm(layer_ffn_down, cur, model.layers[il].ffn_down_s);
            cb(cur, "ffn_out", il);
        } else {
            // Dense FFN
            ggml_tensor * up_b   = config.ffn_bias ? model.layers[il].ffn_up_b   : nullptr;
            ggml_tensor * gate_b = config.ffn_bias ? model.layers[il].ffn_gate_b : nullptr;
            ggml_tensor * down_b = config.ffn_bias ? model.layers[il].ffn_down_b : nullptr;

            cur = build_ffn(cur,
                    layer_ffn_up,   up_b,   nullptr,
                    layer_ffn_gate, gate_b, nullptr,
                    layer_ffn_down, down_b, nullptr,
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

            ggml_tensor * inp_this_layer = view_2d_slice_3d(inp_per_layer, il);

            if (il == effective_n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur);
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            cur = ggml_add(ctx0, pe_in, cur);
        }

        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);

        // Deepstack embedding injection (Qwen3VL): add slices of input embeddings
        if (config.deepstack && il < (int) hparams.n_deepstack_layers) {
            ggml_tensor * ds = ggml_view_2d(ctx0, res->t_inp_embd,
                    n_embd, n_tokens, res->t_inp_embd->nb[1],
                    (il + 1) * n_embd * ggml_element_size(res->t_inp_embd));
            cur = ggml_add(ctx0, cur, ds);
            cb(cur, "deepstack_out", il);
        }

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

ggml_tensor * llm_build_std_transformer::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_std_transformer::build_layer_hybrid_delta_net(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = nullptr;
    ggml_tensor * z = nullptr;

    if (model.layers[il].wqkv) {
        qkv_mixed = build_lora_mm(model.layers[il].wqkv, cur, model.layers[il].wqkv_s);
        qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        z = build_lora_mm(model.layers[il].wqkv_gate, cur, model.layers[il].wqkv_gate_s);
        cb(z, "z", il);
    } else {
        GGML_ASSERT(model.layers[il].ssm_in != nullptr);
        GGML_ASSERT(num_v_heads % num_k_heads == 0);

        const int64_t head_ratio = num_v_heads / num_k_heads;

        ggml_tensor * mixed_qkvz = build_lora_mm(model.layers[il].ssm_in, cur, model.layers[il].ssm_in_s);
        cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

        const int64_t qkvz_new_dim = 2 * head_k_dim + 2 * head_v_dim * head_ratio;
        ggml_tensor * mixed_qkvz_reshaped = ggml_reshape_4d(ctx0, mixed_qkvz, qkvz_new_dim, num_k_heads, n_seq_tokens, n_seqs);

        ggml_tensor * query =
                ggml_view_4d(ctx0, mixed_qkvz_reshaped, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3], 0);
        ggml_tensor * key =
                ggml_view_4d(ctx0, mixed_qkvz_reshaped, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        head_k_dim * ggml_element_size(mixed_qkvz_reshaped));
        ggml_tensor * value =
                ggml_view_4d(ctx0, mixed_qkvz_reshaped, head_v_dim * head_ratio, num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        2 * head_k_dim * ggml_element_size(mixed_qkvz_reshaped));
        z = ggml_view_4d(ctx0, mixed_qkvz_reshaped, head_v_dim * head_ratio, num_k_heads, n_seq_tokens, n_seqs,
                mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                (2 * head_k_dim + head_v_dim * head_ratio) * ggml_element_size(mixed_qkvz_reshaped));

        cb(query, "q", il);
        cb(key, "k", il);
        cb(value, "v", il);

        z = ggml_cont(ctx0, z);
        cb(z, "z", il);

        ggml_tensor * query_flat = ggml_cont_3d(ctx0, query, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
        ggml_tensor * key_flat   = ggml_cont_3d(ctx0, key,   head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
        ggml_tensor * value_flat = ggml_cont_3d(ctx0, value, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

        cb(query_flat, "query_flat", il);
        cb(key_flat, "key_flat", il);
        cb(value_flat, "value_flat", il);

        qkv_mixed = ggml_concat(ctx0, query_flat, key_flat, 0);
        qkv_mixed = ggml_concat(ctx0, qkv_mixed, value_flat, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);
    }

    GGML_ASSERT(z != nullptr);

    ggml_tensor * beta = nullptr;
    ggml_tensor * alpha = nullptr;
    if (model.layers[il].ssm_beta_alpha) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);

        const int64_t head_ratio = num_v_heads / num_k_heads;

        ggml_tensor * mixed_ba = build_lora_mm(model.layers[il].ssm_beta_alpha, cur);
        cb(mixed_ba, "linear_attn_mixed_ba", il);

        ggml_tensor * mixed_ba_reshaped = ggml_reshape_4d(ctx0, mixed_ba, 2 * head_ratio, num_k_heads, n_seq_tokens, n_seqs);
        ggml_tensor * beta_view =
                ggml_view_4d(ctx0, mixed_ba_reshaped, head_ratio, num_k_heads, n_seq_tokens, n_seqs,
                        mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 0);
        ggml_tensor * alpha_view =
                ggml_view_4d(ctx0, mixed_ba_reshaped, head_ratio, num_k_heads, n_seq_tokens, n_seqs,
                        mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3],
                        head_ratio * ggml_element_size(mixed_ba_reshaped));

        cb(beta_view, "beta", il);

        beta = ggml_cont(ctx0, beta_view);
        beta = ggml_sigmoid(ctx0, beta);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        cb(beta, "beta_sigmoid", il);

        alpha = ggml_cont_3d(ctx0, alpha_view, num_v_heads, n_seq_tokens, n_seqs);
        cb(alpha, "alpha", il);
    } else {
        beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        cb(beta, "beta", il);

        beta = ggml_sigmoid(ctx0, beta);
        cb(beta, "beta_sigmoid", il);

        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
        cb(alpha, "alpha", il);
    }

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    ggml_tensor * last_conv_states =
            ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                    conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
            ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels, n_seqs, conv_states_all->nb[1],
                    kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output, "conv_output_raw", il);

    conv_output = ggml_silu(ctx0, conv_output);
    cb(conv_output, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_output->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_output, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_output, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_output));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_output, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_output->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    q_conv = ggml_l2_norm(ctx0, q_conv, hparams.f_norm_rms_eps);
    k_conv = ggml_l2_norm(ctx0, k_conv, hparams.f_norm_rms_eps);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        if (cfg_.hybrid_delta_interleave_repeat) {
            const int64_t repeat_factor = num_v_heads / num_k_heads;

            ggml_tensor * q_reshaped = ggml_reshape_4d(ctx0, q_conv, head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);
            ggml_tensor * k_reshaped = ggml_reshape_4d(ctx0, k_conv, head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);

            ggml_tensor * q_repeated = ggml_repeat_4d(ctx0, q_reshaped, head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);
            ggml_tensor * k_repeated = ggml_repeat_4d(ctx0, k_reshaped, head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);

            q_conv = ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
            k_conv = ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
        } else {
            q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
            k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        }
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    auto attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
}

// ---------------------------------------------------------------------------
// Reusable hybrid Mamba2 transformer builder (PLaMo2-style family)
// ---------------------------------------------------------------------------

llm_build_hybrid_mamba2_transformer::llm_build_hybrid_mamba2_transformer(
        const llama_model & model,
        const llm_graph_params & params,
        const llm_hybrid_mamba2_transformer_config & config) :
        llm_build_mamba_base(params),
        model(model),
        cfg_(config) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "embedding_output", -1);

    inp_pos_ = build_inp_pos();
    auto * inp_hybrid = build_inp_mem_hybrid();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, cfg_.norm, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_mamba2(inp_hybrid->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp_hybrid->get_attn(), cur, il);
        }

        if (cfg_.attn_post_norm && model.layers[il].attn_post_norm) {
            cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, cfg_.norm, il);
            cb(cur, "attn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "attn_residual", il);
        residual = cur;

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, cfg_.norm, il);
        cb(cur, "ffn_pre_norm", il);

        cur = build_layer_ffn(cur, il);

        if (cfg_.ffn_post_norm && model.layers[il].ffn_post_norm) {
            cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, cfg_.norm, il);
            cb(cur, "ffn_post_norm", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "ffn_residual", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, cfg_.norm, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    ggml_set_output(cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_hybrid_mamba2_transformer::build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             cur,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_q = hparams.n_embd_head_k();
    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    const int64_t n_embd_head_v = hparams.n_embd_head_v();
    const int32_t layer_n_head = hparams.n_head(il);
    const int32_t layer_n_head_kv = hparams.n_head_kv(il);

    ggml_tensor * Qcur = nullptr;
    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;

    if (cfg_.combined_qkv && layer.wqkv) {
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur);
        cb(qkv, "qkv", il);

        const int64_t q_offset = 0;
        const int64_t k_offset = n_embd_head_q * layer_n_head;
        const int64_t v_offset = k_offset + n_embd_head_k * layer_n_head_kv;

        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, layer_n_head, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1],
                q_offset * ggml_element_size(qkv));
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, layer_n_head_kv, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                k_offset * ggml_element_size(qkv));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, layer_n_head_kv, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                v_offset * ggml_element_size(qkv));
    } else {
        Qcur = build_lora_mm(layer.wq, cur);
        Kcur = build_lora_mm(layer.wk, cur);
        Vcur = build_lora_mm(layer.wv, cur);

        if (cfg_.attn_bias) {
            if (layer.bq) {
                Qcur = ggml_add(ctx0, Qcur, layer.bq);
            }
            if (layer.bk) {
                Kcur = ggml_add(ctx0, Kcur, layer.bk);
            }
            if (layer.bv) {
                Vcur = ggml_add(ctx0, Vcur, layer.bv);
            }
        }

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, layer_n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, layer_n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, layer_n_head_kv, n_tokens);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    if (cfg_.qk_norm) {
        if (layer.attn_q_norm) {
            Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, cfg_.norm, il);
            cb(Qcur, "Qcur_normed", il);
        }
        if (layer.attn_k_norm) {
            Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, cfg_.norm, il);
            cb(Kcur, "Kcur_normed", il);
        }
    }

    if (cfg_.use_rope && rope_type != LLAMA_ROPE_TYPE_NONE) {
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos_, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos_, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }

    ggml_tensor * out = build_attn(inp_attn,
            layer.wo, cfg_.attn_bias ? layer.bo : nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
            1.0f / sqrtf(float(n_embd_head_v)), il);
    cb(out, "attn_out", il);
    return out;
}

ggml_tensor * llm_build_hybrid_mamba2_transformer::build_layer_mamba2(
        llm_graph_input_rs * inp_rs,
        ggml_tensor *        cur,
        int                  il) {
    if (hparams.ssm_n_group > 0) {
        return build_mamba2_layer(inp_rs, cur, model, ubatch, il);
    }
    return build_mamba2_layer_zero_group(inp_rs, cur, model, ubatch, il);
}

ggml_tensor * llm_build_hybrid_mamba2_transformer::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, nullptr,
            nullptr,        nullptr, nullptr,
            layer.ffn_down, nullptr, nullptr,
            nullptr,
            cfg_.ffn_act, cfg_.ffn_type, il);
    cb(cur, "ffn_out", il);
    return cur;
}

// ---------------------------------------------------------------------------
// Reusable hybrid shortconv transformer builder (LFM2 family)
// ---------------------------------------------------------------------------

template <bool iswa>
llm_build_hybrid_shortconv_transformer<iswa>::llm_build_hybrid_shortconv_transformer(
        const llama_model & model,
        const llm_graph_params & params,
        const llm_hybrid_shortconv_transformer_config & config) :
        llm_graph_context(params),
        model(model),
        cfg_(config) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);
    ggml_build_forward_expand(gf, cur);

    if constexpr (iswa) {
        auto * inp_hybrid = build_inp_mem_hybrid_iswa();
        inp_recr_ = inp_hybrid->get_recr();
        inp_attn_iswa_ = inp_hybrid->get_attn();
    } else {
        auto * inp_hybrid = build_inp_mem_hybrid();
        inp_recr_ = inp_hybrid->get_recr();
        inp_attn_kv_ = inp_hybrid->get_attn();
    }

    inp_pos_ = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * prev_cur = cur;

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, cfg_.norm, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        cur = hparams.is_recurrent(il)
            ? build_layer_shortconv(inp_recr_, cur, il)
            : build_layer_attn(cur, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
        }

        cur = ggml_add(ctx0, prev_cur, cur);

        ggml_tensor * ffn_norm_out = build_norm(cur, model.layers[il].ffn_norm, nullptr, cfg_.norm, il);
        cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        ggml_tensor * ffn_out = build_layer_ffn(ffn_norm_out, il);
        cur = ggml_add(ctx0, cur, ffn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, nullptr, cfg_.norm, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

template <bool iswa>
ggml_tensor * llm_build_hybrid_shortconv_transformer<iswa>::build_layer_attn(
        ggml_tensor * cur,
        int           il) {
    GGML_ASSERT(hparams.n_embd_v_gqa(il) == hparams.n_embd_k_gqa(il));

    const auto n_embd_head = hparams.n_embd_head_v();
    const auto layer_n_head = hparams.n_head(il);
    const auto layer_n_head_kv = hparams.n_head_kv(il);

    ggml_tensor * q = build_lora_mm(model.layers[il].wq, cur);
    cb(q, "model.layers.{}.self_attn.q_proj", il);
    ggml_tensor * k = build_lora_mm(model.layers[il].wk, cur);
    cb(k, "model.layers.{}.self_attn.k_proj", il);
    ggml_tensor * v = build_lora_mm(model.layers[il].wv, cur);
    cb(v, "model.layers.{}.self_attn.v_proj", il);

    q = ggml_reshape_3d(ctx0, q, n_embd_head, layer_n_head, n_tokens);
    k = ggml_reshape_3d(ctx0, k, n_embd_head, layer_n_head_kv, n_tokens);
    v = ggml_reshape_3d(ctx0, v, n_embd_head, layer_n_head_kv, n_tokens);

    if (cfg_.qk_norm) {
        q = build_norm(q, model.layers[il].attn_q_norm, nullptr, cfg_.norm, il);
        cb(q, "model.layers.{}.self_attn.q_layernorm", il);
        k = build_norm(k, model.layers[il].attn_k_norm, nullptr, cfg_.norm, il);
        cb(k, "model.layers.{}.self_attn.k_layernorm", il);
    }

    q = ggml_rope_ext(ctx0, q, inp_pos_, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    k = ggml_rope_ext(ctx0, k, inp_pos_, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    ggml_tensor * out = nullptr;
    if constexpr (iswa) {
        out = build_attn(inp_attn_iswa_,
                model.layers[il].wo, nullptr,
                q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
    } else {
        out = build_attn(inp_attn_kv_,
                model.layers[il].wo, nullptr,
                q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
    }

    cb(out, "model.layers.{}.self_attn.out_proj", il);
    return out;
}

template <bool iswa>
ggml_tensor * llm_build_hybrid_shortconv_transformer<iswa>::build_layer_shortconv(
        llm_graph_input_rs * inp_recr,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp_recr->mctx;
    const uint32_t kv_head = mctx_cur->get_head();
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs = ubatch.n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);

    const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;

    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    ggml_tensor * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
    cb(bcx, "model.layers.{}.conv.in_proj", il);

    constexpr int64_t n_chunks = 3;
    GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
    const int64_t chunk_size = bcx->ne[0] / n_chunks;

    ggml_tensor * b = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2],
            bcx->nb[1], bcx->nb[2], 0 * chunk_size * ggml_element_size(bcx));
    ggml_tensor * c = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2],
            bcx->nb[1], bcx->nb[2], 1 * chunk_size * ggml_element_size(bcx));
    ggml_tensor * x = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2],
            bcx->nb[1], bcx->nb[2], 2 * chunk_size * ggml_element_size(bcx));

    ggml_tensor * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

    ggml_tensor * conv_state = mctx_cur->get_r_l(il);
    ggml_tensor * conv_rs = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
    ggml_tensor * conv = ggml_reshape_3d(ctx0, conv_rs, d_conv, n_embd, n_seqs);

    bx = ggml_concat(ctx0, conv, bx, 0);
    GGML_ASSERT(bx->ne[0] > conv->ne[0]);

    ggml_tensor * new_conv = ggml_view_3d(ctx0, bx, conv->ne[0], bx->ne[1], bx->ne[2],
            bx->nb[1], bx->nb[2], (bx->ne[0] - conv->ne[0]) * ggml_element_size(bx));
    GGML_ASSERT(ggml_are_same_shape(conv, new_conv));

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_conv,
            ggml_view_1d(ctx0, conv_state, ggml_nelements(new_conv),
                    kv_head * d_conv * n_embd * ggml_element_size(new_conv))));

    ggml_tensor * conv_out = ggml_ssm_conv(ctx0, bx, model.layers[il].shortconv.conv);
    cb(conv_out, "model.layers.{}.conv.conv", il);

    ggml_tensor * y = ggml_mul(ctx0, c, conv_out);
    y = build_lora_mm(model.layers[il].shortconv.out_proj, y);
    cb(y, "model.layers.{}.conv.out_proj", il);

    y = ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);
    return y;
}

template <bool iswa>
ggml_tensor * llm_build_hybrid_shortconv_transformer<iswa>::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];

    if (layer.ffn_gate_inp) {
        ggml_tensor * out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                cfg_.dense_ffn_act, cfg_.moe_norm_weights,
                hparams.expert_weights_scale,
                static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
                il);
        cb(out, "ffn_out", il);
        return out;
    }

    ggml_tensor * out = build_ffn(cur,
            layer.ffn_up,   layer.ffn_up_b,   nullptr,
            layer.ffn_gate, layer.ffn_gate_b, nullptr,
            layer.ffn_down, layer.ffn_down_b, nullptr,
            nullptr,
            cfg_.dense_ffn_act, cfg_.dense_ffn_type, il);
    cb(out, "ffn_out", il);
    return out;
}

template struct llm_build_hybrid_shortconv_transformer<true>;
template struct llm_build_hybrid_shortconv_transformer<false>;

// ---------------------------------------------------------------------------
// Reusable hybrid Mamba2 single-operator builder (Nemotron-H-style family)
// ---------------------------------------------------------------------------

llm_build_hybrid_mamba2_single_op_transformer::llm_build_hybrid_mamba2_single_op_transformer(
        const llama_model & model,
        const llm_graph_params & params,
        const llm_hybrid_mamba2_single_op_transformer_config & config) :
        llm_build_mamba_base(params),
        model(model),
        cfg_(config) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);
    ggml_build_forward_expand(gf, cur);

    auto * inp_hybrid = build_inp_mem_hybrid();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = cur;

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, cfg_.norm, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_mamba2(inp_hybrid->get_recr(), cur, il);
        } else if (hparams.n_ff(il) == 0) {
            cur = build_layer_attn(inp_hybrid->get_attn(), cur, il);
        } else {
            cur = build_layer_ffn(cur, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "operator_residual", il);

        if (cfg_.apply_cvec_after_residual) {
            cur = build_cvec(cur, il);
        }
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, nullptr, cfg_.norm, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_hybrid_mamba2_single_op_transformer::build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             cur,
        int                       il) {
    const auto & layer = model.layers[il];
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int32_t layer_n_head = hparams.n_head(il);
    const int32_t layer_n_head_kv = hparams.n_head_kv(il);

    ggml_tensor * q = build_lora_mm(layer.wq, cur);
    cb(q, "Qcur", il);
    if (cfg_.attn_bias && layer.bq) {
        q = ggml_add(ctx0, q, layer.bq);
        cb(q, "Qcur", il);
    }

    ggml_tensor * k = build_lora_mm(layer.wk, cur);
    cb(k, "Kcur", il);
    if (cfg_.attn_bias && layer.bk) {
        k = ggml_add(ctx0, k, layer.bk);
        cb(k, "Kcur", il);
    }

    ggml_tensor * v = build_lora_mm(layer.wv, cur);
    cb(v, "Vcur", il);
    if (cfg_.attn_bias && layer.bv) {
        v = ggml_add(ctx0, v, layer.bv);
        cb(v, "Vcur", il);
    }

    q = ggml_reshape_3d(ctx0, q, n_embd_head, layer_n_head, n_tokens);
    k = ggml_reshape_3d(ctx0, k, n_embd_head, layer_n_head_kv, n_tokens);
    v = ggml_reshape_3d(ctx0, v, n_embd_head, layer_n_head_kv, n_tokens);

    cb(q, "Qcur", il);
    cb(k, "Kcur", il);
    cb(v, "Vcur", il);

    const float kq_scale = cfg_.use_hparams_attn_scale && hparams.f_attention_scale != 0.0f
        ? hparams.f_attention_scale
        : 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * out = build_attn(inp_attn,
            layer.wo, cfg_.attn_bias ? layer.bo : nullptr,
            q, k, v, nullptr, nullptr, nullptr, kq_scale, il);
    cb(out, "attn_out", il);
    return out;
}

ggml_tensor * llm_build_hybrid_mamba2_single_op_transformer::build_layer_mamba2(
        llm_graph_input_rs * inp_rs,
        ggml_tensor *        cur,
        int                  il) const {
    if (hparams.ssm_n_group > 0) {
        return build_mamba2_layer(inp_rs, cur, model, ubatch, il);
    }
    return build_mamba2_layer_zero_group(inp_rs, cur, model, ubatch, il);
}

ggml_tensor * llm_build_hybrid_mamba2_single_op_transformer::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];

    if (!layer.ffn_gate_inp) {
        ggml_tensor * out = build_ffn(cur,
                layer.ffn_up,   layer.ffn_up_b,   layer.ffn_up_s,
                nullptr,        nullptr,          nullptr,
                layer.ffn_down, layer.ffn_down_b, layer.ffn_down_s,
                nullptr,
                cfg_.dense_ffn_act, cfg_.dense_ffn_type, il);
        cb(out, "ffn_out", il);
        return out;
    }

    GGML_ASSERT(n_expert > 0 && n_expert_used > 0);

    ggml_tensor * inp_emb = cur;
    ggml_tensor * inp_latent = cur;

    if (cfg_.latent_moe && layer.ffn_latent_down) {
        inp_latent = ggml_mul_mat(ctx0, layer.ffn_latent_down, cur);
    }

    ggml_tensor * router_logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(router_logits, "ffn_moe_logits", il);

    const auto moe_gating = cfg_.moe_gating == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE
        ? static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func)
        : cfg_.moe_gating;

    ggml_tensor * moe_out = build_moe_ffn(inp_latent,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            cfg_.dense_ffn_act,
            cfg_.moe_norm_weights || hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            moe_gating,
            il,
            router_logits,
            nullptr,
            layer.ffn_up_exps_s,
            nullptr,
            layer.ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    if (cfg_.latent_moe && layer.ffn_latent_up) {
        moe_out = ggml_mul_mat(ctx0, layer.ffn_latent_up, moe_out);
    }

    if (cfg_.shared_expert && layer.ffn_up_shexp && layer.ffn_down_shexp) {
        ggml_tensor * ffn_shexp = build_ffn(inp_emb,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                nullptr,              nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr,
                cfg_.dense_ffn_act, cfg_.dense_ffn_type, il);
        cb(ffn_shexp, "ffn_shexp", il);
        moe_out = ggml_add(ctx0, moe_out, ffn_shexp);
    }

    cb(moe_out, "ffn_out", il);
    return moe_out;
}

// ---------------------------------------------------------------------------
// Reusable MLA + KDA hybrid builder (Kimi Linear family)
// ---------------------------------------------------------------------------

llm_build_mla_kda_hybrid::llm_build_mla_kda_hybrid(
        const llama_model & model,
        const llm_graph_params & params) :
        llm_build_delta_net_base(params),
        model(model) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    bool has_mla_absorb = false;
    bool has_mla_fallback = false;
    for (uint32_t il = 0; il < model.layers.size(); ++il) {
        if (hparams.is_recurrent(il)) {
            continue;
        }
        has_mla_absorb = has_mla_absorb || (model.layers[il].wk_b != nullptr && model.layers[il].wv_b != nullptr);
        has_mla_fallback = has_mla_fallback || (model.layers[il].wkv_b != nullptr);
    }

    GGML_ASSERT(has_mla_absorb || has_mla_fallback);
    GGML_ASSERT(!(has_mla_absorb && has_mla_fallback));

    auto * inp_hybrid_kv = has_mla_fallback ? build_inp_mem_hybrid() : nullptr;
    auto * inp_hybrid_k  = has_mla_absorb   ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs        = has_mla_absorb ? inp_hybrid_k->get_recr() : inp_hybrid_kv->get_recr();
    auto * inp_attn_kv   = has_mla_fallback ? inp_hybrid_kv->get_attn() : nullptr;
    auto * inp_attn_k    = has_mla_absorb   ? inp_hybrid_k->get_attn() : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    GGML_ASSERT(ubatch.n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_kda(inp_rs, cur, il);
        } else {
            cur = build_layer_mla(inp_attn_kv, inp_attn_k, cur, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_mla_kda_hybrid::build_norm_sigmoid_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

ggml_tensor * llm_build_mla_kda_hybrid::build_kda_conv1d(
        ggml_tensor * conv_states_all,
        ggml_tensor * conv_state_all,
        ggml_tensor * input,
        ggml_tensor * proj_w,
        ggml_tensor * conv_w,
        int64_t       qkv,
        int64_t       kv_head) {
    const int64_t n_head    = hparams.n_head();
    const int64_t head_dim  = hparams.n_embd_head_kda;
    const int64_t d_conv    = hparams.ssm_d_conv;
    const int64_t d_inner   = head_dim * n_head;
    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
            (d_conv - 1) * ggml_element_size(conv_state_all),
            n_embd_r_total * ggml_element_size(conv_state_all),
            qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, input);
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, ubatch.n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
            conv_x->nb[1], conv_x->nb[2], ubatch.n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, last_conv_x,
                ggml_view_3d(ctx0, conv_states_all,
                    d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1) * ggml_element_size(conv_states_all),
                    n_embd_r_total * ggml_element_size(conv_states_all),
                    (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * out = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    out = ggml_reshape_2d(ctx0, out, d_inner, n_tokens);
    out = ggml_silu(ctx0, out);

    return ggml_reshape_4d(ctx0, out, head_dim, n_head, ubatch.n_seq_tokens, n_seqs);
}

ggml_tensor * llm_build_mla_kda_hybrid::build_layer_kda(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto & layer = model.layers[il];
    const auto * mctx_cur = inp->mctx;
    const auto kv_head = mctx_cur->get_head();

    const int64_t n_head   = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner  = n_head * head_dim;
    const int64_t n_seqs   = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    cb(conv_states_all, "conv_states_all", il);
    ggml_tensor * conv_state_all = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * Qcur = build_kda_conv1d(conv_states_all, conv_state_all, cur, layer.wq, layer.ssm_q_conv, 0, kv_head);
    ggml_tensor * Kcur = build_kda_conv1d(conv_states_all, conv_state_all, cur, layer.wk, layer.ssm_k_conv, 1, kv_head);
    ggml_tensor * Vcur = build_kda_conv1d(conv_states_all, conv_state_all, cur, layer.wv, layer.ssm_v_conv, 2, kv_head);
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    cb(g1, "g1", il);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
    g1 = ggml_softplus(ctx0, g1);
    g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
    g1 = ggml_mul(ctx0, g1, A);
    cb(g1, "kda_g1", il);
    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
    cb(beta, "kda_beta", il);
    beta = ggml_sigmoid(ctx0, beta);

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

    Qcur = ggml_l2_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
    Kcur = ggml_l2_norm(ctx0, Kcur, hparams.f_norm_rms_eps);

    auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

    ggml_tensor * output = ggml_cont(ctx0, attn_out.first);
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur_3d, cur_3d->ne[0], n_seq_tokens * n_seqs);
    ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
    ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
    cb(g2, "g2", il);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_tokens);

    ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_tokens);
    ggml_tensor * gated = build_norm_sigmoid_gated(attn_out_final, layer.ssm_o_norm, g2, il);
    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);

    ggml_tensor * out = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(out, "kda_out", il);
    return out;
}

ggml_tensor * llm_build_mla_kda_hybrid::build_layer_mla(
        llm_graph_input_attn_kv * inp_attn_kv,
        llm_graph_input_attn_k   * inp_attn_k,
        ggml_tensor *             cur,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t n_head = hparams.n_head();
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const float kq_scale_mla = 1.0f / sqrtf((float) n_embd_head_k_mla);

    ggml_tensor * q = nullptr;
    if (layer.wq_a && layer.wq_b && layer.attn_q_a_norm) {
        q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q, "q", il);
        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q, "q_a_norm", il);
        q = ggml_mul_mat(ctx0, layer.wq_b, q);
    } else {
        q = ggml_mul_mat(ctx0, layer.wq, cur);
    }
    cb(q, "mla_q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_mla),
            ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_mla),
            ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));
    cb(q_nope, "q_nope", il);
    cb(q_pe, "q_pe", il);

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
    cb(kv_cmpr, "kv_cmpr", il);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr_norm", il);

    if (layer.wk_b && layer.wv_b) {
        GGML_ASSERT(inp_attn_k != nullptr);

        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        ggml_tensor * Vcur = kv_cmpr;

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        ggml_tensor * out = build_attn(inp_attn_k, layer.wo, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
        cb(out, "mla_out", il);
        return out;
    }

    GGML_ASSERT(inp_attn_kv != nullptr);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
    const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

    ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(kv->type, kv_per_head),
            ggml_row_size(kv->type, kv_per_head * n_head),
            0);
    ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
            ggml_row_size(kv->type, kv_per_head),
            ggml_row_size(kv->type, kv_per_head * n_head),
            ggml_row_size(kv->type, n_embd_head_qk_nope));
    Vcur = ggml_cont(ctx0, Vcur);

    ggml_tensor * k_pe_target = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
    ggml_tensor * k_pe_repeated = ggml_repeat(ctx0, k_pe, k_pe_target);
    ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, k_pe_repeated, 0);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * out = build_attn(inp_attn_kv, layer.wo, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_mla, il);
    cb(out, "mla_out", il);
    return out;
}

ggml_tensor * llm_build_mla_kda_hybrid::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];

    if (layer.ffn_gate_inp) {
        const llama_expert_gating_func_type moe_gating =
                hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_NONE
                ? (llama_expert_gating_func_type) hparams.expert_gating_func
                : LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                moe_gating,
                il);
        cb(moe_out, "ffn_moe_out", il);

        if (layer.ffn_up_shexp) {
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
        } else {
            cur = moe_out;
        }
    } else {
        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    cb(cur, "ffn_out", il);
    return cur;
}

// ---------------------------------------------------------------------------
// Reusable pure MLA transformer builder (MiniCPM3, PLM family)
// ---------------------------------------------------------------------------

llm_build_mla_transformer::llm_build_mla_transformer(
        const llama_model & model,
        const llm_graph_params & params,
        const llm_mla_transformer_config & config) :
        llm_graph_context(params),
        model(model),
        cfg_(config) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    if (cfg_.embd_scale != 1.0f) {
        inpL = ggml_scale(ctx0, inpL, cfg_.embd_scale);
        cb(inpL, "inp_scaled", -1);
    }

    inp_pos_ = build_inp_pos();
    auto * inp_attn = cfg_.absorb_kv ? nullptr : build_attn_inp_kv();
    auto * inp_attn_k = cfg_.absorb_kv ? build_attn_inp_k() : nullptr;
    if (cfg_.attn_temp) {
        inp_attn_scale_ = build_inp_attn_scale();
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    const int effective_n_layer = n_layer - (int) cfg_.nextn_layers;

    for (int il = 0; il < effective_n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_layer_mla(inp_attn, inp_attn_k, cur, il);

        if (il == effective_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        if (cfg_.residual_scale != 1.0f) {
            cur = ggml_scale(ctx0, cur, cfg_.residual_scale);
            cb(cur, "hidden_scaled", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);

        if (cfg_.residual_scale != 1.0f) {
            cur = ggml_scale(ctx0, cur, cfg_.residual_scale);
            cb(cur, "hidden_scaled_ffn", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (cfg_.lmhead_scale != 1.0f) {
        cur = ggml_scale(ctx0, cur, cfg_.lmhead_scale);
        cb(cur, "lmhead_scaling", -1);
    }

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_mla_transformer::build_layer_mla(
        llm_graph_input_attn_kv * inp_attn,
        llm_graph_input_attn_k   * inp_attn_k,
        ggml_tensor *             cur,
        int                       il) {
    const auto & layer = model.layers[il];

    const uint32_t n_embd_head_qk_rope = hparams.n_rot();
    const uint32_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const uint32_t n_embd_head_v = hparams.n_embd_head_v_mla();
    const uint32_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const uint32_t kv_lora_rank = hparams.n_lora_kv;
    float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
    if (cfg_.yarn_kq_scale) {
        GGML_ASSERT(ext_factor >= 0.0f);
        const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
        const float mscale = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
        kq_scale = mscale * mscale / sqrtf(float(n_embd_head_k));
    }

    ggml_tensor * q = nullptr;
    if (layer.wq_a && layer.wq_b && layer.attn_q_a_norm) {
        q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q, "q", il);
        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q, "q_a_norm", il);
        q = ggml_mul_mat(ctx0, layer.wq_b, q);
    } else {
        q = ggml_mul_mat(ctx0, layer.wq, cur);
    }
    cb(q, "q_proj", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k * n_head),
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k * n_head),
            ggml_row_size(q->type, n_embd_head_qk_nope));
    cb(q_nope, "q_nope", il);
    cb(q_pe, "q_pe", il);

    ggml_tensor * kv_pe_compressed = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_pe_compressed, "kv_pe_compressed", il);

    ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compressed, kv_lora_rank, n_tokens,
            kv_pe_compressed->nb[1], 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compressed, n_embd_head_qk_rope, 1, n_tokens,
            kv_pe_compressed->nb[1],
            kv_pe_compressed->nb[1],
            ggml_row_size(kv_pe_compressed->type, kv_lora_rank));
    cb(kv_compressed, "kv_compressed", il);
    cb(k_pe, "k_pe", il);

    kv_compressed = build_norm(kv_compressed, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_compressed, "kv_compressed_norm", il);

    ggml_tensor * rope_factors = cfg_.rope_factors ? model.get_rope_factors(cparams, il) : nullptr;
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos_, rope_factors,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos_, rope_factors,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe_rope", il);
    cb(k_pe, "k_pe_rope", il);

    ggml_tensor * out = nullptr;
    if (cfg_.absorb_kv) {
        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        cb(q_nope, "q_nope_perm", il);

        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        cb(q_nope_absorbed, "q_nope_absorbed", il);

        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
        cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

        ggml_tensor * q_states = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        cb(q_states, "q_states", il);

        kv_compressed = ggml_reshape_3d(ctx0, kv_compressed, kv_lora_rank, 1, n_tokens);
        cb(kv_compressed, "kv_compressed_reshape", il);

        ggml_tensor * k_states = ggml_concat(ctx0, kv_compressed, k_pe, 0);
        cb(k_states, "k_states", il);

        ggml_tensor * v_states = kv_compressed;
        cb(v_states, "v_states", il);

        if (inp_attn_scale_) {
            q_states = ggml_mul(ctx0, q_states, inp_attn_scale_);
            cb(q_states, "q_states_attn_temp_scaled", il);
        }

        out = build_attn(inp_attn_k,
                layer.wo, nullptr,
                q_states, k_states, v_states, nullptr, nullptr, layer.wv_b, kq_scale, il);
    } else {
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_compressed);
        cb(kv, "kv", il);

        ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + n_embd_head_v)),
                0);
        ggml_tensor * v_states = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + n_embd_head_v)),
                ggml_row_size(kv->type, n_embd_head_qk_nope));
        cb(k_nope, "k_nope", il);
        cb(v_states, "v_states_view", il);

        v_states = ggml_cont(ctx0, v_states);
        cb(v_states, "v_states", il);
        if (cfg_.flatten_v) {
            v_states = ggml_view_2d(ctx0, v_states, n_embd_head_v * n_head, n_tokens,
                    ggml_row_size(kv->type, n_embd_head_v * n_head),
                    0);
            cb(v_states, "v_states_flat", il);
        }

        ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
        ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
        cb(q_states, "q_states", il);
        cb(k_states, "k_states", il);

        if (inp_attn_scale_) {
            q_states = ggml_mul(ctx0, q_states, inp_attn_scale_);
            cb(q_states, "q_states_attn_temp_scaled", il);
        }

        out = build_attn(inp_attn,
                layer.wo, nullptr,
                q_states, k_states, v_states, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(out, "attn_out", il);
    return out;
}

ggml_tensor * llm_build_mla_transformer::build_layer_ffn(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];
    if (!layer.ffn_gate_inp) {
        const bool gated = cfg_.ffn_type != LLM_FFN_SEQ;

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                gated ? layer.ffn_gate : nullptr, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr,
                cfg_.ffn_act, cfg_.ffn_type, il);
        cb(cur, "ffn_out", il);
        return cur;
    }

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            cfg_.ffn_act, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr,
            layer.ffn_gate_up_exps);
    cb(moe_out, "ffn_moe_out", il);

    if (layer.ffn_up_shexp && layer.ffn_gate_shexp && layer.ffn_down_shexp) {
        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                cfg_.ffn_act, cfg_.ffn_type, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }

    cb(cur, "ffn_out", il);
    return cur;
}

// ---------------------------------------------------------------------------
// Auto-configure from hparams
// ---------------------------------------------------------------------------

bool llm_transformer_config_from_hparams(
        const llama_hparams & hparams,
        const llama_model   & model,
        llm_transformer_config & config) {

    // Reject architectures we can't handle
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

    // Hybrid SSM+Attention: detect from recurrent layer markers or SSM tensors + attention tensors
    if (model.layers.size() > 0) {
        bool has_ssm = (model.layers[0].ssm_in != nullptr);
        bool has_attn = (model.layers[0].wq != nullptr || model.layers[0].wqkv != nullptr);
        // Check if some layers have SSM and some have attention
        if (has_ssm && has_attn) {
            // ALL layers have both SSM and attention → parallel (Falcon-H1)
            // vs alternating layers → regular hybrid
            bool all_parallel = true;
            for (uint32_t il = 1; il < model.layers.size(); ++il) {
                bool layer_ssm  = (model.layers[il].ssm_in != nullptr);
                bool layer_attn = (model.layers[il].wq != nullptr || model.layers[il].wqkv != nullptr);
                if (!(layer_ssm && layer_attn)) {
                    all_parallel = false;
                    break;
                }
            }
            if (all_parallel) {
                config.hybrid_parallel = true;
            } else {
                config.hybrid = true;
            }
        } else if (has_ssm) {
            // Check other layers for attention to detect hybrid
            for (uint32_t il = 1; il < model.layers.size(); ++il) {
                if (model.layers[il].wq != nullptr || model.layers[il].wqkv != nullptr) {
                    config.hybrid = true;
                    break;
                }
            }
            if (!config.hybrid) {
                config.pure_ssm = true;
            }
        } else if (has_attn && hparams.has_recurrent_layers()) {
            // Some layers are recurrent (flagged in hparams) but layer 0 is attention
            config.hybrid = true;
        }
    }

    bool has_combined_qkv = false;
    bool has_attn_bias = false;
    bool has_attn_gate = false;
    bool has_attn_post_norm = false;
    bool has_ffn_post_norm = false;
    bool has_vision_expert = false;

    for (uint32_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        const bool recurrent = hparams.is_recurrent(il);

        has_combined_qkv  = has_combined_qkv  || (!recurrent && layer.wqkv != nullptr && layer.wq == nullptr);
        has_attn_bias     = has_attn_bias     || layer.bq != nullptr || layer.bqkv != nullptr || layer.bo != nullptr;
        has_attn_gate     = has_attn_gate     || (!recurrent && layer.wqkv_gate != nullptr);
        has_attn_post_norm = has_attn_post_norm || layer.attn_post_norm != nullptr;
        has_ffn_post_norm  = has_ffn_post_norm  || layer.ffn_post_norm  != nullptr;
        has_vision_expert = has_vision_expert || layer.visexp_attn_wqkv != nullptr;
    }

    if (hparams.use_kq_norm || (model.layers.size() > 0 && model.layers[0].attn_q_norm != nullptr)) {
        config.qk_norm = true;
    }

    config.combined_qkv = has_combined_qkv;
    config.attn_bias    = has_attn_bias;
    config.attn_gate    = has_attn_gate;

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
        config.attn_post_norm = has_attn_post_norm;
        config.ffn_post_norm  = has_ffn_post_norm;

        // Encoder post-norm (BERT): detect from attn_out_norm tensor
        if (model.layers[0].attn_out_norm != nullptr && model.layers[0].layer_out_norm != nullptr) {
            config.encoder_post_norm = true;
            config.no_attn_cache = true;
        }

        // Pure SSM (Mamba): detect from SSM tensors + no attention tensors
        if (model.layers[0].ssm_in != nullptr && model.layers[0].wq == nullptr && model.layers[0].wqkv == nullptr) {
            config.pure_ssm = true;
        }

        // Vision expert routing (CogVLM): detect from visexp tensors
        if (has_vision_expert) {
            config.vision_expert = true;
        }
    }

    // Deepstack embedding injection (Qwen3VL)
    if (hparams.n_deepstack_layers > 0) {
        config.deepstack = true;
    }

    // Logit softcapping
    if (hparams.f_final_logit_softcapping > 0.0f && hparams.f_final_logit_softcapping < 1000.0f) {
        config.logit_softcap = true;
    }

    // Token embedding scaling (Gemma-style): detect by f_embedding_scale or architecture
    if (hparams.f_embedding_scale > 0.0f) {
        config.token_embd_scale = true;
    }
    // AFMoE uses MuP scaling (sqrt(n_embd)) for embeddings
    if (model.arch == LLM_ARCH_AFMOE) {
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

        // Parse comma-separated tokens for exact matching
        auto has_op = [&ops](const char * token) -> bool {
            const std::string tok(token);
            size_t pos = 0;
            while ((pos = ops.find(tok, pos)) != std::string::npos) {
                // Check this is a complete token (bounded by start/end/comma)
                bool at_start = (pos == 0 || ops[pos - 1] == ',');
                bool at_end   = (pos + tok.size() >= ops.size() || ops[pos + tok.size()] == ',');
                if (at_start && at_end) return true;
                pos += tok.size();
            }
            return false;
        };

        // Attention features
        if (has_op("qk_norm"))       config.qk_norm = true;
        if (has_op("post_norm"))     config.attn_post_norm = true;
        if (!has_op("rope"))         config.use_rope = false;

        // FFN features
        if (has_op("ffn_post_norm")) config.ffn_post_norm  = true;

        // Pre-norm absence: if ops_str exists but doesn't start with "norm",
        // the model may not use pre-attention normalization (unusual but possible)
    }

    // Activation: arch table provides fallback for old GGUF files without ffn_activation
    if (meta && hparams.ffn_activation[0] == '\0') {
        config.act = meta->default_act;
    }

    // FFN type: if no gate tensor in first layer, it's sequential
    // Must come AFTER activation is resolved (SWIGLU implies SEQ even in MoE models)
    if (model.layers.size() > 0 && model.layers[0].ffn_gate == nullptr) {
        if (!config.moe || config.act == LLM_FFN_SWIGLU) {
            config.ffn_type = LLM_FFN_SEQ;
        }
    }

    // No-KV-cache mode (embedding/diffusion models)
    if (meta && meta->no_attn_cache) {
        config.no_attn_cache = true;
    }

    // Post-norm after residual (GLM4-MOE): attn_post_norm applied to (attn+residual) sum
    if (model.arch == LLM_ARCH_GLM4_MOE) {
        config.post_norm_after_residual = true;
    }

    switch (model.arch) {
        case LLM_ARCH_QWEN3NEXT:
            config.qk_norm = true;
            config.attn_q_gate = true;
            config.hybrid_delta = true;
            config.hybrid_delta_interleave_repeat = true;
            config.post_norm_after_residual = true;
            config.moe_shared_gate = true;
            config.moe_norm_weights = true;
            config.moe_gating = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
            break;
        case LLM_ARCH_QWEN35:
            config.qk_norm = true;
            config.attn_q_gate = true;
            config.hybrid_delta = true;
            config.post_norm_after_residual = true;
            break;
        case LLM_ARCH_QWEN35MOE:
            config.qk_norm = true;
            config.attn_q_gate = true;
            config.hybrid_delta = true;
            config.post_norm_after_residual = true;
            config.moe_shared_gate = true;
            config.moe_norm_weights = true;
            config.moe_gating = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
            break;
        default:
            break;
    }

    return true;
}
