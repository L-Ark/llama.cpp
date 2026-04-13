#include "models.h"

llm_build_deepseek2::llm_build_deepseek2(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(model.arch == LLM_ARCH_DEEPSEEK2OCR);

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int effective_n_layers = hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < effective_n_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        const int n_embd_head = hparams.n_embd / hparams.n_head();
        const int rope_type_ocr = GGML_ROPE_TYPE_NEOX;
        GGML_ASSERT(n_embd_head == n_embd_head_k && n_embd_head == n_embd_head_v);

        ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
        ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
        ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
        cb(Qcur, "q", il);
        cb(Kcur, "k", il);
        cb(Vcur, "v", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head, n_tokens);

        GGML_ASSERT(fabs(freq_base - 10000.0) < 1e-4);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_embd_head, rope_type_ocr, 0, freq_base, 1, 0, 1, 0, 0);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_embd_head, rope_type_ocr, 0, freq_base, 1, 0, 1, 0, 0);
        cb(Qcur, "q_pe", il);
        cb(Kcur, "k_pe", il);

        cur = build_attn(inp_attn,
                model.layers[il].wo, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        if (il == effective_n_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    model.layers[il].ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp, nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
