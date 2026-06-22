#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <cmath>

ggml_cgraph * llm_build_context::build_deepseek4() {
    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH ||
        model.split_mode == LLAMA_SPLIT_MODE_ATTN) {
        GGML_ABORT("deepseek4: split graph mode is not wired yet");
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    int32_t n_tokens = this->n_tokens;
    const int32_t n_active_layers = hparams.n_layer - hparams.nextn_predict_layers;
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_k(0);
    const int64_t rope_dim = hparams.n_rot;
    const int64_t nope_dim = head_dim - rope_dim;
    const int64_t total_q_dim = head_dim * n_head;
    const float kq_scale = 1.0f / std::sqrt(float(head_dim));
    const auto rope_mode = rope_type == LLAMA_ROPE_TYPE_NONE ? LLAMA_ROPE_TYPE_NEOX : rope_type;

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    auto build_grouped_out = [&](ggml_tensor * attn_out, const llama_layer & layer, int il) -> ggml_tensor * {
        const int64_t group_dim = layer.attn_out_a->ne[0];
        const int64_t n_groups = total_q_dim / group_dim;
        const int64_t out_rank = layer.attn_out_b->ne[0] / n_groups;

        GGML_ASSERT(group_dim > 0);
        GGML_ASSERT(n_groups > 0);
        GGML_ASSERT(layer.attn_out_b->ne[0] == n_groups * out_rank);

        ggml_tensor * grouped = nullptr;
        for (int64_t g = 0; g < n_groups; ++g) {
            ggml_tensor * xg = ggml_view_2d(ctx0, attn_out,
                    group_dim, n_tokens,
                    attn_out->nb[1],
                    g * group_dim * attn_out->nb[0]);
            ggml_tensor * wg = ggml_view_2d(ctx0, layer.attn_out_a,
                    group_dim, out_rank,
                    layer.attn_out_a->nb[1],
                    g * out_rank * layer.attn_out_a->nb[1]);
            ggml_tensor * og = ggml_mul_mat(ctx0, wg, xg);
            cb(og, "attn_group_out", il);
            grouped = grouped ? ggml_concat(ctx0, grouped, og, 0) : og;
        }

        ggml_tensor * out = ggml_mul_mat(ctx0, layer.attn_out_b, grouped);
        cb(out, "attn_out_proj", il);
        return out;
    };

    auto build_attention = [&](ggml_tensor * cur, const llama_layer & layer, int il) -> ggml_tensor * {
        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q, "q_a", il);
        q = llm_build_norm(ctx0, q, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
        q = ggml_mul_mat(ctx0, layer.wq_b, q);
        q = ggml_reshape_3d(ctx0, q, head_dim, n_head, n_tokens);
        q = ggml_rms_norm(ctx0, q, hparams.f_norm_rms_eps);
        cb(q, "q", il);

        ggml_tensor * q_nope = ggml_view_3d(ctx0, q,
                nope_dim, n_head, n_tokens,
                q->nb[1], q->nb[2], 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, q,
                rope_dim, n_head, n_tokens,
                q->nb[1], q->nb[2], nope_dim * q->nb[0]);
        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, rope_dim, rope_mode,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(q_states, "q_states", il);

        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.attn_kv_latent, cur);
        kv = llm_build_norm(ctx0, kv, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
        kv = ggml_reshape_3d(ctx0, kv, head_dim, 1, n_tokens);
        cb(kv, "kv_latent", il);

        ggml_tensor * k_nope = ggml_view_3d(ctx0, kv,
                nope_dim, 1, n_tokens,
                kv->nb[1], kv->nb[2], 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv,
                rope_dim, 1, n_tokens,
                kv->nb[1], kv->nb[2], nope_dim * kv->nb[0]);
        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, rope_dim, rope_mode,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        ggml_tensor * k_states = ggml_concat(ctx0, k_nope, k_pe, 0);
        cb(k_states, "k_states", il);

        ggml_tensor * v_states = ggml_reshape_2d(ctx0, kv, head_dim, n_tokens);
        cb(v_states, "v_states", il);

        ggml_tensor * out = llm_build_kv(ctx0, lctx, kv_self, gf,
                nullptr, nullptr,
                k_states, v_states, q_states, KQ_mask,
                n_tokens, kv_head, n_kv, kq_scale, cb, il, layer.attn_sinks, hparams.n_swa);
        out = ggml_reshape_3d(ctx0, out, head_dim, n_head, n_tokens);
        out = ggml_reshape_2d(ctx0, out, total_q_dim, n_tokens);
        cb(out, "attn_out", il);

        return build_grouped_out(out, layer, il);
    };

    for (int il = 0; il < n_active_layers; ++il) {
        const llama_layer & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);
        cur = build_attention(cur, layer, il);

        if (il == n_active_layers - 1) {
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            cb(cur, "last_attn", il);
            cb(inpSA, "last_ffn_inp", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_norm(ctx0, ffn_inp, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    true,
                    hparams.expert_weights_scale,
                    (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                    cb, il, gf, false, nullptr);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
