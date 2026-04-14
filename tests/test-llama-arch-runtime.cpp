#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"

#include <cstdio>

#define REQUIRE(expr)                                                     \
    do {                                                                  \
        if (!(expr)) {                                                    \
            std::fprintf(stderr, "requirement failed: %s\n", #expr);      \
            return 1;                                                     \
        }                                                                 \
    } while (false)

int main() {
    for (const auto arch : llm_arch_all()) {
        REQUIRE(llm_arch_has_runtime_traits_coverage(arch));
    }

    REQUIRE(llm_arch_is_recurrent(LLM_ARCH_MAMBA));
    REQUIRE(!llm_arch_is_recurrent(LLM_ARCH_LLAMA));

    REQUIRE(llm_arch_is_hybrid(LLM_ARCH_JAMBA));
    REQUIRE(!llm_arch_is_hybrid(LLM_ARCH_GEMMA2));

    REQUIRE(llm_arch_is_diffusion(LLM_ARCH_DREAM));
    REQUIRE(!llm_arch_is_diffusion(LLM_ARCH_LLAMA));

    REQUIRE(!llm_arch_supports_sm_tensor(LLM_ARCH_DEEPSEEK2));
    REQUIRE(llm_arch_supports_sm_tensor(LLM_ARCH_LLAMA));

    REQUIRE(llm_arch_uses_encoder_pass(LLM_ARCH_T5ENCODER));
    REQUIRE(!llm_arch_uses_encoder_pass(LLM_ARCH_QWEN35MOE));

    REQUIRE(llm_arch_prefers_embedding_outputs(LLM_ARCH_GEMMA_EMBEDDING));
    REQUIRE(!llm_arch_prefers_embedding_outputs(LLM_ARCH_GLM4));

    REQUIRE(llm_arch_default_full_attention_interval(LLM_ARCH_QWEN35MOE) == 4);
    REQUIRE(llm_arch_default_full_attention_interval(LLM_ARCH_LLAMA) == 0);

    REQUIRE(llm_arch_default_sliding_window_pattern(LLM_ARCH_GEMMA3) == 6);
    REQUIRE(llm_arch_default_sliding_window_pattern(LLM_ARCH_LLAMA) == 0);

    REQUIRE(llm_arch_uses_sliding_window_metadata(LLM_ARCH_PHI3));
    REQUIRE(!llm_arch_uses_sliding_window_metadata(LLM_ARCH_QWEN35MOE));

    REQUIRE(llm_arch_uses_explicit_swa_pattern(LLM_ARCH_STEP35));
    REQUIRE(!llm_arch_uses_explicit_swa_pattern(LLM_ARCH_GEMMA3));

    uint32_t expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    REQUIRE(llm_arch_default_expert_gating_func(LLM_ARCH_GLM_DSA, expert_gating_func));
    REQUIRE(expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID);
    REQUIRE(!llm_arch_default_expert_gating_func(LLM_ARCH_GEMMA2, expert_gating_func));

    bool expert_weights_norm = false;
    REQUIRE(llm_arch_default_expert_weights_norm(LLM_ARCH_QWEN35MOE, expert_weights_norm));
    REQUIRE(expert_weights_norm);
    REQUIRE(!llm_arch_default_expert_weights_norm(LLM_ARCH_LLAMA, expert_weights_norm));

    return 0;
}
