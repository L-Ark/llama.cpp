#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"

#include <cassert>

int main() {
    assert(llm_arch_is_recurrent(LLM_ARCH_MAMBA));
    assert(!llm_arch_is_recurrent(LLM_ARCH_LLAMA));

    assert(llm_arch_is_hybrid(LLM_ARCH_JAMBA));
    assert(!llm_arch_is_hybrid(LLM_ARCH_GEMMA2));

    assert(llm_arch_is_diffusion(LLM_ARCH_DREAM));
    assert(!llm_arch_is_diffusion(LLM_ARCH_LLAMA));

    assert(!llm_arch_supports_sm_tensor(LLM_ARCH_DEEPSEEK2));
    assert(llm_arch_supports_sm_tensor(LLM_ARCH_LLAMA));

    assert(llm_arch_uses_encoder_pass(LLM_ARCH_T5ENCODER));
    assert(!llm_arch_uses_encoder_pass(LLM_ARCH_QWEN35MOE));

    assert(llm_arch_prefers_embedding_outputs(LLM_ARCH_GEMMA_EMBEDDING));
    assert(!llm_arch_prefers_embedding_outputs(LLM_ARCH_GLM4));

    assert(llm_arch_default_full_attention_interval(LLM_ARCH_QWEN35MOE) == 4);
    assert(llm_arch_default_full_attention_interval(LLM_ARCH_LLAMA) == 0);

    assert(llm_arch_default_sliding_window_pattern(LLM_ARCH_GEMMA3) == 6);
    assert(llm_arch_default_sliding_window_pattern(LLM_ARCH_LLAMA) == 0);

    assert(llm_arch_uses_sliding_window_metadata(LLM_ARCH_PHI3));
    assert(!llm_arch_uses_sliding_window_metadata(LLM_ARCH_QWEN35MOE));

    assert(llm_arch_uses_explicit_swa_pattern(LLM_ARCH_STEP35));
    assert(!llm_arch_uses_explicit_swa_pattern(LLM_ARCH_GEMMA3));

    uint32_t expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    assert(llm_arch_default_expert_gating_func(LLM_ARCH_GLM_DSA, expert_gating_func));
    assert(expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID);
    assert(!llm_arch_default_expert_gating_func(LLM_ARCH_GEMMA2, expert_gating_func));

    bool expert_weights_norm = false;
    assert(llm_arch_default_expert_weights_norm(LLM_ARCH_QWEN35MOE, expert_weights_norm));
    assert(expert_weights_norm);
    assert(!llm_arch_default_expert_weights_norm(LLM_ARCH_LLAMA, expert_weights_norm));

    return 0;
}
