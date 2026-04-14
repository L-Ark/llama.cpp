#pragma once

#include "llama-model.h"
#include "llama-graph.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;
    ggml_tensor * build_mamba2_layer_zero_group(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;
};

struct llm_build_delta_net_base : public llm_build_mamba_base {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);
};

struct llm_build_rwkv6_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       bool                 is_qrwkv,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

struct llm_build_rwkv7_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    ggml_tensor * build_rwkv7_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

struct llm_rwkv6_config {
    bool use_tok_norm = true;
    bool is_qrwkv = false;
    bool assert_n_embd_r = false;
    bool use_channel_mix = true;
    bool use_rescale = false;
    int expected_token_shift_count = 0;
    llm_norm_type attn_norm = LLM_NORM;
    llm_norm_type ffn_norm = LLM_NORM;
    llm_norm_type output_norm = LLM_NORM;
};

struct llm_build_rwkv6_family : public llm_build_rwkv6_base {
    llm_build_rwkv6_family(const llama_model & model, const llm_graph_params & params, const llm_rwkv6_config & config);
};

struct llm_rwkv7_config {
    bool use_tok_norm = true;
    bool assert_n_embd_r = false;
    bool use_channel_mix = true;
    int expected_token_shift_count = 0;
    llm_norm_type attn_norm = LLM_NORM;
    llm_norm_type ffn_norm = LLM_NORM;
    llm_norm_type output_norm = LLM_NORM;
};

struct llm_build_rwkv7_family : public llm_build_rwkv7_base {
    llm_build_rwkv7_family(const llama_model & model, const llm_graph_params & params, const llm_rwkv7_config & config);
};
