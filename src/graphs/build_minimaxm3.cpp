#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_minimaxm3() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                nullptr, nullptr,
                nullptr, nullptr,
                model.layers[il].wq, nullptr,
                model.layers[il].wk, nullptr,
                model.layers[il].wv, nullptr,
                model.layers[il].attn_q_norm,
                model.layers[il].attn_k_norm,
                0.0f, il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                model.layers[il].wo, nullptr,
                Kcur, Vcur, Qcur, KQ_mask,
                n_tokens, kv_head, n_kv, kq_scale, cb, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,    nullptr,
                    model.layers[il].ffn_up_exps,     nullptr,
                    model.layers[il].ffn_gate_exps,   nullptr,
                    model.layers[il].ffn_down_exps,   nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr,
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SWIGLU_OAI_MOE, cb, il, gf, true,
                    model.layers[il].ffn_up_gate_exps);
            cb(cur, "ffn_out", il);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
