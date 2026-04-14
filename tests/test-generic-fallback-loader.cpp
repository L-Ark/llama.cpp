#include "ggml.h"
#include "gguf.h"
#include "llama.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        std::abort(); \
    } \
} while (0)

namespace {

using gguf_context_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using file_ptr         = std::unique_ptr<FILE, int (*)(FILE *)>;
using llama_model_ptr  = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

constexpr uint32_t n_vocab   = 32;
constexpr uint32_t n_ctx     = 32;
constexpr uint32_t n_embd    = 16;
constexpr uint32_t n_head    = 2;
constexpr uint32_t n_ff      = 24;
constexpr uint32_t n_layer   = 2;
constexpr uint32_t n_experts = 2;

struct gguf_fixture_builder {
    gguf_fixture_builder(struct gguf_context * gguf_ctx, struct ggml_context * ggml_ctx) :
            gguf_ctx(gguf_ctx), ggml_ctx(ggml_ctx) {}

    void add_tensor(const std::string & name, std::initializer_list<int64_t> dims) {
        const std::vector<int64_t> shape(dims);
        struct ggml_tensor * tensor = nullptr;

        switch (shape.size()) {
            case 1:
                tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, shape[0]);
                break;
            case 2:
                tensor = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, shape[0], shape[1]);
                break;
            case 3:
                tensor = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2]);
                break;
            default:
                GGML_ABORT("unsupported tensor rank");
        }

        GGML_ASSERT(tensor != nullptr);
        ggml_format_name(tensor, "%s", name.c_str());
        gguf_add_tensor(gguf_ctx, tensor);

        auto & values = tensor_data.emplace_back((size_t) ggml_nelements(tensor));
        const float base = float((std::hash<std::string>{}(name) % 251) + 1) * 1.0e-4f;
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = base + float(i % 13) * 1.0e-5f;
        }
        gguf_set_tensor_data(gguf_ctx, name.c_str(), values.data());
    }

    struct gguf_context * gguf_ctx;
    struct ggml_context * ggml_ctx;
    std::vector<std::vector<float>> tensor_data;
};

static void add_common_kv(llama_model_saver & ms) {
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_GPTJ));
    ms.add_kv(LLM_KV_VOCAB_SIZE,           n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,       n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,     n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,      n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,          n_layer);
    ms.add_kv(LLM_KV_ATTENTION_CAUSAL,     true);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,  n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL, false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,          1.0f);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,     1.0e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1.0e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,     1.0e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,  uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({
        n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4,
    }));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
    ms.add_kv(LLM_KV_EXPERT_COUNT,        n_experts);
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,   uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT, uint32_t(0));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,  uint32_t(0));
}

static void write_gptj_fixture(FILE * file) {
    gguf_context_ptr gguf_ctx(gguf_init_empty(), gguf_free);

    struct ggml_init_params params = {
        /* mem_size   = */ 2u * 1024 * 1024,
        /* mem_buffer = */ nullptr,
        /* no_alloc   = */ true,
    };
    ggml_context_ptr ggml_ctx(ggml_init(params), ggml_free);

    CHECK(gguf_ctx != nullptr);
    CHECK(ggml_ctx != nullptr);

    llama_model_saver ms(LLM_ARCH_GPTJ, gguf_ctx.get());
    add_common_kv(ms);

    gguf_fixture_builder builder(gguf_ctx.get(), ggml_ctx.get());
    const auto tn = LLM_TN(LLM_ARCH_GPTJ);
    const uint32_t n_embd_head = n_embd / n_head;
    const uint32_t n_embd_gqa  = n_embd_head * n_head;

    builder.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
    builder.add_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx});
    builder.add_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab});

    builder.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", 0), {n_embd});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_QKV,  "weight", 0), {n_embd, n_embd + 2*n_embd_gqa});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", 0), {n_embd, n_embd});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", 0), {n_embd_head});
    builder.add_tensor(tn(LLM_TENSOR_FFN_NORM,    "weight", 0), {n_embd});
    builder.add_tensor(tn(LLM_TENSOR_FFN_GATE,    "weight", 0), {n_embd, n_ff});
    builder.add_tensor(tn(LLM_TENSOR_FFN_UP,      "weight", 0), {n_embd, n_ff});
    builder.add_tensor(tn(LLM_TENSOR_FFN_DOWN,    "weight", 0), {n_ff, n_embd});
    builder.add_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", 0), {n_embd});

    builder.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", 1), {n_embd});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", 1), {n_embd, n_embd});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", 1), {n_embd, n_embd_gqa});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", 1), {n_embd, n_embd_gqa});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", 1), {n_embd, n_embd});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", 1), {n_embd_head});
    builder.add_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", 1), {n_embd});
    builder.add_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", 1), {n_embd});
    builder.add_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", 1), {n_embd, n_experts});
    builder.add_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", 1), {n_embd, n_ff, n_experts});
    builder.add_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", 1), {n_ff, n_embd, n_experts});

    CHECK(gguf_write_to_file_ptr(gguf_ctx.get(), file, false));
}

static llama_model_ptr load_model(FILE * file) {
    struct llama_model_params params = llama_model_default_params();
    params.use_mmap = false;

    return llama_model_ptr(llama_model_load_from_file_ptr(file, params), llama_model_free);
}

} // namespace

int main() {
    file_ptr file(tmpfile(), fclose);

#ifdef _WIN32
    if (!file) {
        fprintf(stderr, "tmpfile() unavailable on Windows, skipping generic fallback loader test\n");
        return EXIT_SUCCESS;
    }
#else
    CHECK(file != nullptr);
#endif

    write_gptj_fixture(file.get());
    CHECK(std::fflush(file.get()) == 0);
    rewind(file.get());

    llama_backend_init();
    llama_model_ptr model = load_model(file.get());

    CHECK(model != nullptr);
    CHECK(model->layers.size() == n_layer);

    const auto & layer0 = model->layers[0];
    CHECK(layer0.wqkv != nullptr);
    CHECK(layer0.wq == nullptr);
    CHECK(layer0.wk == nullptr);
    CHECK(layer0.wv == nullptr);
    CHECK(layer0.attn_q_norm != nullptr);
    CHECK(layer0.attn_k_norm == nullptr);
    CHECK(layer0.attn_post_norm == nullptr);
    CHECK(layer0.ffn_gate_inp == nullptr);
    CHECK(layer0.ffn_gate != nullptr);
    CHECK(layer0.ffn_up != nullptr);
    CHECK(layer0.ffn_down != nullptr);
    CHECK(layer0.ffn_up_exps == nullptr);
    CHECK(layer0.ffn_down_exps == nullptr);
    CHECK(layer0.ffn_post_norm != nullptr);

    const auto & layer1 = model->layers[1];
    CHECK(layer1.wqkv == nullptr);
    CHECK(layer1.wq != nullptr);
    CHECK(layer1.wk != nullptr);
    CHECK(layer1.wv != nullptr);
    CHECK(layer1.attn_q_norm == nullptr);
    CHECK(layer1.attn_k_norm != nullptr);
    CHECK(layer1.attn_post_norm != nullptr);
    CHECK(layer1.ffn_gate_inp != nullptr);
    CHECK(layer1.ffn_gate == nullptr);
    CHECK(layer1.ffn_up == nullptr);
    CHECK(layer1.ffn_down == nullptr);
    CHECK(layer1.ffn_up_exps != nullptr);
    CHECK(layer1.ffn_down_exps != nullptr);
    CHECK(layer1.ffn_post_norm == nullptr);

    model.reset();
    llama_backend_free();

    return EXIT_SUCCESS;
}
