// Golden output regression test for llama.cpp architectures.
//
// Captures deterministic logit outputs from synthetic models and compares
// them against stored reference files to detect numerical regressions.
//
// Usage:
//   ./test-golden-outputs --capture              # generate golden files
//   ./test-golden-outputs --compare              # compare against golden files
//   ./test-golden-outputs --capture --arch llama  # capture single arch
//   ./test-golden-outputs                        # auto: compare if files exist, else capture

#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>

// ── constants ────────────────────────────────────────────────────────────────

static const uint64_t GOLDEN_SEED       = 42;
static const uint32_t GOLDEN_N_TOKENS   = 128;
static const uint32_t GOLDEN_N_VOCAB    = 128;
static const char     GOLDEN_MAGIC[4]   = {'G', 'O', 'L', 'D'};
static const double   GOLDEN_TOLERANCE  = 1e-6;   // max absolute difference

// ── architecture list ────────────────────────────────────────────────────────

struct golden_arch_entry {
    llm_arch    arch;
    const char *name;
    bool        moe;     // whether to build as MoE
    bool        encode;  // whether to call llama_encode before decode (T5)
};

static const golden_arch_entry golden_archs[] = {
    { LLM_ARCH_LLAMA,     "llama",     false, false },
    { LLM_ARCH_QWEN3,     "qwen3",     false, false },
    // LLM_ARCH_GEMMA4 is currently broken upstream (FIXME @ngxson), skip it
    { LLM_ARCH_DEEPSEEK2, "deepseek2", true,  false },  // MOE mandatory
    { LLM_ARCH_MAMBA2,    "mamba2",    false, false },
    { LLM_ARCH_T5,        "t5",        false, true  },
};
static const size_t N_GOLDEN_ARCHS = sizeof(golden_archs) / sizeof(golden_archs[0]);

// ── helpers (same logic as test-llama-archs.cpp) ─────────────────────────────

static void golden_set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<llama_token> golden_get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr golden_get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 128;

    uint32_t n_vocab = GOLDEN_N_VOCAB;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4;
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    }

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 64 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(32));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool golden_silent_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> golden_get_model_and_ctx(
        struct gguf_context * gguf_ctx, const size_t seed) {
    // CPU-only: empty device list (null-terminated)
    std::vector<ggml_backend_dev_t> devs = {nullptr};

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = golden_silent_progress;
    model_params.devices = devs.data();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx, golden_set_tensor_data, &tmp, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> golden_get_logits(
        llama_model * model, llama_context * lctx,
        const std::vector<llama_token> & tokens, bool encode) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens * n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

// ── golden file I/O ──────────────────────────────────────────────────────────

static std::string golden_file_path(const std::string & dir, const char * arch_name) {
    return dir + "/" + arch_name + ".bin";
}

static bool golden_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

static bool golden_write(const std::string & path, const char * arch_name,
                          uint64_t seed, const std::vector<float> & logits) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for writing\n", path.c_str());
        return false;
    }

    // magic
    fwrite(GOLDEN_MAGIC, 1, 4, f);

    // arch name (length-prefixed)
    uint32_t name_len = (uint32_t) strlen(arch_name);
    fwrite(&name_len, sizeof(name_len), 1, f);
    fwrite(arch_name, 1, name_len, f);

    // seed
    fwrite(&seed, sizeof(seed), 1, f);

    // logits
    uint64_t count = logits.size();
    fwrite(&count, sizeof(count), 1, f);
    fwrite(logits.data(), sizeof(float), logits.size(), f);

    fclose(f);
    return true;
}

static bool golden_read(const std::string & path, std::string & out_name,
                         uint64_t & out_seed, std::vector<float> & out_logits) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    // magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, GOLDEN_MAGIC, 4) != 0) {
        fprintf(stderr, "ERROR: bad magic in %s\n", path.c_str());
        fclose(f);
        return false;
    }

    // arch name
    uint32_t name_len;
    if (fread(&name_len, sizeof(name_len), 1, f) != 1 || name_len > 256) {
        fclose(f);
        return false;
    }
    out_name.resize(name_len);
    if (fread(&out_name[0], 1, name_len, f) != name_len) {
        fclose(f);
        return false;
    }

    // seed
    if (fread(&out_seed, sizeof(out_seed), 1, f) != 1) {
        fclose(f);
        return false;
    }

    // logits
    uint64_t count;
    if (fread(&count, sizeof(count), 1, f) != 1) {
        fclose(f);
        return false;
    }
    out_logits.resize(count);
    if (fread(out_logits.data(), sizeof(float), count, f) != count) {
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// ── run inference for a single architecture ──────────────────────────────────

static std::vector<float> run_inference(const golden_arch_entry & entry, uint64_t seed) {
    if (!llama_model_saver_supports_arch(entry.arch)) {
        throw std::runtime_error(std::string("arch not supported by model saver: ") + entry.name);
    }

    gguf_context_ptr gguf_ctx = golden_get_gguf_ctx(entry.arch, entry.moe);
    auto model_and_ctx = golden_get_model_and_ctx(gguf_ctx.get(), seed);

    const std::vector<llama_token> tokens = golden_get_tokens(GOLDEN_N_TOKENS, GOLDEN_N_VOCAB, seed);
    return golden_get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens, entry.encode);
}

// ── capture mode ─────────────────────────────────────────────────────────────

static int capture_golden(const std::string & golden_dir, const char * target_arch) {
    int captured = 0;
    int skipped  = 0;

    for (size_t i = 0; i < N_GOLDEN_ARCHS; i++) {
        const golden_arch_entry & entry = golden_archs[i];

        if (target_arch && strcmp(target_arch, entry.name) != 0) {
            continue;
        }

        printf("Capturing %-12s ... ", entry.name);
        fflush(stdout);

        try {
            std::vector<float> logits = run_inference(entry, GOLDEN_SEED);
            std::string path = golden_file_path(golden_dir, entry.name);
            if (!golden_write(path, entry.name, GOLDEN_SEED, logits)) {
                printf("FAIL (write error)\n");
                return 1;
            }
            printf("OK  (%zu floats -> %s)\n", logits.size(), path.c_str());
            captured++;
        } catch (const std::exception & e) {
            printf("SKIP (%s)\n", e.what());
            skipped++;
        }
    }

    printf("\nCapture complete: %d captured, %d skipped\n", captured, skipped);
    return 0;
}

// ── compare mode ─────────────────────────────────────────────────────────────

static int compare_golden(const std::string & golden_dir, const char * target_arch) {
    int passed  = 0;
    int failed  = 0;
    int skipped = 0;

    for (size_t i = 0; i < N_GOLDEN_ARCHS; i++) {
        const golden_arch_entry & entry = golden_archs[i];

        if (target_arch && strcmp(target_arch, entry.name) != 0) {
            continue;
        }

        std::string path = golden_file_path(golden_dir, entry.name);
        if (!golden_file_exists(path)) {
            printf("%-12s : SKIP (no golden file: %s)\n", entry.name, path.c_str());
            skipped++;
            continue;
        }

        // load golden reference
        std::string ref_name;
        uint64_t    ref_seed;
        std::vector<float> ref_logits;
        if (!golden_read(path, ref_name, ref_seed, ref_logits)) {
            printf("%-12s : FAIL (cannot read golden file)\n", entry.name);
            failed++;
            continue;
        }

        if (ref_seed != GOLDEN_SEED) {
            printf("%-12s : FAIL (seed mismatch: golden=%" PRIu64 " expected=%" PRIu64 ")\n",
                   entry.name, ref_seed, GOLDEN_SEED);
            failed++;
            continue;
        }

        printf("Comparing    %-12s ... ", entry.name);
        fflush(stdout);

        try {
            std::vector<float> logits = run_inference(entry, GOLDEN_SEED);

            if (logits.size() != ref_logits.size()) {
                printf("FAIL (size mismatch: got %zu, expected %zu)\n",
                       logits.size(), ref_logits.size());
                failed++;
                continue;
            }

            // find max absolute difference
            double max_diff = 0.0;
            size_t max_idx  = 0;
            for (size_t j = 0; j < logits.size(); j++) {
                double diff = fabs((double)logits[j] - (double)ref_logits[j]);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_idx  = j;
                }
            }

            if (max_diff > GOLDEN_TOLERANCE) {
                printf("FAIL (max_diff=%.3e at idx=%zu, got=%.6f ref=%.6f)\n",
                       max_diff, max_idx, logits[max_idx], ref_logits[max_idx]);
                failed++;
            } else {
                printf("OK   (max_diff=%.3e)\n", max_diff);
                passed++;
            }
        } catch (const std::exception & e) {
            printf("SKIP (%s)\n", e.what());
            skipped++;
        }
    }

    printf("\nCompare complete: %d passed, %d failed, %d skipped\n", passed, failed, skipped);
    return failed > 0 ? 1 : 0;
}

// ── main ─────────────────────────────────────────────────────────────────────

static void usage(char ** argv) {
    printf("Usage: %s [--capture|--compare] [--arch name] [--golden-dir dir] [-v|--verbose]\n", argv[0]);
    printf("\n");
    printf("  --capture       Generate golden reference files\n");
    printf("  --compare       Compare against golden reference files\n");
    printf("  --arch name     Process only the named architecture\n");
    printf("  --golden-dir d  Directory for golden files (default: tests/golden)\n");
    printf("  -v, --verbose   Verbose logging\n");
    printf("\nDefault: compare if golden files exist, capture if they don't.\n");
}

int main(int argc, char ** argv) {
    common_init();

    enum { MODE_AUTO, MODE_CAPTURE, MODE_COMPARE } mode = MODE_AUTO;
    const char * target_arch = nullptr;
    std::string golden_dir = "tests/golden";
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--capture") == 0) {
            mode = MODE_CAPTURE;
        } else if (strcmp(argv[i], "--compare") == 0) {
            mode = MODE_COMPARE;
        } else if (strcmp(argv[i], "--arch") == 0 || strcmp(argv[i], "-a") == 0) {
            if (i + 1 < argc) {
                target_arch = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--golden-dir") == 0) {
            if (i + 1 < argc) {
                golden_dir = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv);
            return 1;
        }
    }

    // suppress noisy model-loading logs
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level;
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    // validate --arch if specified
    if (target_arch) {
        bool found = false;
        for (size_t i = 0; i < N_GOLDEN_ARCHS; i++) {
            if (strcmp(target_arch, golden_archs[i].name) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "ERROR: unknown architecture '%s'. Supported:", target_arch);
            for (size_t i = 0; i < N_GOLDEN_ARCHS; i++) {
                fprintf(stderr, " %s", golden_archs[i].name);
            }
            fprintf(stderr, "\n");
            return 1;
        }
    }

    // auto-detect mode: if any golden file exists, compare; otherwise capture
    if (mode == MODE_AUTO) {
        bool any_exist = false;
        for (size_t i = 0; i < N_GOLDEN_ARCHS; i++) {
            if (golden_file_exists(golden_file_path(golden_dir, golden_archs[i].name))) {
                any_exist = true;
                break;
            }
        }
        mode = any_exist ? MODE_COMPARE : MODE_CAPTURE;
        printf("Auto-detected mode: %s\n", mode == MODE_CAPTURE ? "capture" : "compare");
    }

    try {
        if (mode == MODE_CAPTURE) {
            return capture_golden(golden_dir, target_arch);
        } else {
            return compare_golden(golden_dir, target_arch);
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "Fatal error: %s\n", err.what());
        return 1;
    }
}
