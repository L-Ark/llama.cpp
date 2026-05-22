// Decode-only batched streaming MoE path.
//
// This file intentionally includes the MMQ-id kernel family without the MMVQ
// headers used by moe_stream.cu; several CUDA helper headers define unguarded
// device functions and cannot be mixed in one translation unit.

#include "common.cuh"
#include "mmq_id_common.cuh"
#include "quantize.cuh"
#include "quantize_id.cuh"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

bool ggml_cuda_moe_stream_batch(
    int  src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_stride);

bool ggml_cuda_moe_stream_preload_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes);

bool ggml_cuda_moe_stream_up_gate_batch(
    int  src0_type_int,
    const char *src0_up_name,
    const void *src0_up_data,
    const char *src0_gate_name,
    const void *src0_gate_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    int unary_op,
    float limit,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_stride);

}

struct batch_ctx {
    cudaStream_t stream = nullptr;
    void * d_src0 = nullptr;     size_t d_src0_sz = 0;
    void * d_src1_f32 = nullptr; size_t d_src1_f32_sz = 0;
    void * d_src1_q8 = nullptr;  size_t d_src1_q8_sz = 0;
    void * d_src1_q8_one = nullptr; size_t d_src1_q8_one_sz = 0;
    void * d_dst = nullptr;      size_t d_dst_sz = 0;
    void * d_up = nullptr;       size_t d_up_sz = 0;
    void * d_gate = nullptr;     size_t d_gate_sz = 0;
    void * d_handoff = nullptr;  size_t d_handoff_sz = 0;
    int32_t * d_ids_src1 = nullptr; size_t d_ids_src1_sz = 0;
    int32_t * d_ids_dst = nullptr; size_t d_ids_dst_sz = 0;
    int32_t * d_x_ids = nullptr; size_t d_x_ids_sz = 0;
    int32_t * d_bounds = nullptr;  size_t d_bounds_sz = 0;
    void * h_src1 = nullptr;     size_t h_src1_sz = 0;
    void * h_dst = nullptr;      size_t h_dst_sz = 0;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stage = nullptr;
    cudaEvent_t ev_quant = nullptr;
    cudaEvent_t ev_kernel = nullptr;
    cudaEvent_t ev_d2h = nullptr;
    int32_t h_ids_dst[128] = {};
    int32_t h_up_gate_ids_dst[128] = {};
    int32_t h_ids_src1[128] = {};
    int32_t h_x_ids[128] = {};
    int32_t h_bounds[129] = {};
};

static batch_ctx g_batch;
static std::mutex g_batch_mu;
static std::atomic<bool> g_batch_inited{false};

struct moe_gpu_handoff {
    const float * host_ptr = nullptr;
    const float * d_data = nullptr;
    size_t bytes = 0;
    int64_t ne01 = 0;
    int64_t dst_cols = 0;
    uint64_t serial = 0;
};

static moe_gpu_handoff g_handoff;

struct batch_profile {
    bool enabled = false;
    uint64_t calls = 0;
    uint64_t active_experts = 0;
    double stage_ms = 0.0;
    double quant_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double scatter_ms = 0.0;
};

static batch_profile g_bprof;

static void batch_profile_report_atexit() {
    if (!g_bprof.enabled || g_bprof.calls == 0) return;
    const double calls = (double)g_bprof.calls;
    std::fprintf(stderr,
        "[moe_stream_batch] profile: calls=%lu avg_active=%.2f "
        "stage=%.3f ms quant=%.3f ms kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms total=%.3f ms/call\n",
        g_bprof.calls,
        (double)g_bprof.active_experts / calls,
        g_bprof.stage_ms / calls,
        g_bprof.quant_ms / calls,
        g_bprof.kernel_ms / calls,
        g_bprof.d2h_ms / calls,
        g_bprof.scatter_ms / calls,
        (g_bprof.stage_ms + g_bprof.quant_ms + g_bprof.kernel_ms + g_bprof.d2h_ms + g_bprof.scatter_ms) / calls);
}

struct batch_vram_cache {
    void * pool = nullptr;
    size_t slot_sz = 0;
    int n_slots = 0;
    uintptr_t slot_key[16384] = {};
    uint64_t slot_used[16384] = {};
    uint64_t clock = 1;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t preloads = 0;
};

static batch_vram_cache g_bcaches[2];
static bool g_bcache_inited[2] = {};

struct profile_entry {
    int expert_idx = -1;
    char tensor[96] = {};
};

static std::vector<profile_entry> g_profile;
static bool g_profile_loaded = false;
static bool g_profile_enabled = false;
static std::mutex g_profile_mu;
static char g_preloaded_tensors[256][96] = {};
static int g_n_preloaded_tensors = 0;

static void batch_cache_report_atexit() {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t preloads = 0;
    for (batch_vram_cache &c : g_bcaches) {
        hits += c.hits;
        misses += c.misses;
        preloads += c.preloads;
    }
    const uint64_t total = hits + misses;
    if (total == 0) return;
    std::fprintf(stderr, "[moe_stream_batch] VRAM cache: hits=%lu misses=%lu preloads=%lu hit_rate=%.1f%%\n",
                 hits, misses, preloads, 100.0 * hits / total);
}

static uint64_t batch_key_hash(const char *name, int expert_idx) {
    uint64_t h = 1469598103934665603ULL;
    if (name) {
        for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
            h ^= (uint64_t)*p;
            h *= 1099511628211ULL;
        }
    }
    h ^= (uint64_t)(uint32_t)expert_idx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h ? h : 1;
}

static void profile_lookup_name(const char *name, char *out, size_t out_sz) {
    if (!name || out_sz == 0) return;
    std::snprintf(out, out_sz, "%s", name);
    char *p = std::strstr(out, ".ffn_up_exps.");
    if (!p) p = std::strstr(out, ".ffn_gate_exps.");
    if (p) {
        char tail[96];
        std::snprintf(tail, sizeof(tail), "%s", p + std::strlen(".ffn_up_exps."));
        if (std::strstr(p, ".ffn_gate_exps.")) {
            std::snprintf(tail, sizeof(tail), "%s", p + std::strlen(".ffn_gate_exps."));
        }
        *p = '\0';
        std::snprintf(out + std::strlen(out), out_sz - std::strlen(out), ".ffn_down_exps.%s", tail);
    }
}

static bool tensor_already_preloaded(const char *name) {
    if (!name || !name[0]) return true;
    for (int i = 0; i < g_n_preloaded_tensors; ++i) {
        if (std::strcmp(g_preloaded_tensors[i], name) == 0) return true;
    }
    if (g_n_preloaded_tensors < (int)(sizeof(g_preloaded_tensors) / sizeof(g_preloaded_tensors[0]))) {
        std::snprintf(g_preloaded_tensors[g_n_preloaded_tensors++], sizeof(g_preloaded_tensors[0]), "%s", name);
    }
    return false;
}

static void load_profile_once() {
    if (g_profile_loaded) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    if (g_profile_loaded) return;
    const char *path = std::getenv("GGML_MOE_VRAM_PROFILE");
    if (!path || !path[0]) {
        g_profile_loaded = true;
        return;
    }
    FILE *f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "[moe_stream_batch] profile preload: open failed: %s\n", path);
        g_profile_loaded = true;
        return;
    }
    char line[512];
    if (!std::fgets(line, sizeof(line), f)) {
        std::fclose(f);
        g_profile_loaded = true;
        return;
    }
    while (std::fgets(line, sizeof(line), f)) {
        profile_entry e;
        unsigned long long rank = 0, count = 0, cumulative = 0, tensor_base = 0;
        size_t expert_bytes = 0;
        if (std::sscanf(line, "%llu,%llu,%zu,%llu,0x%llx,%d,%95[^\n]",
                    &rank, &count, &expert_bytes, &cumulative, &tensor_base, &e.expert_idx, e.tensor) == 7 &&
                e.expert_idx >= 0 && e.tensor[0]) {
            g_profile.push_back(e);
        }
    }
    std::fclose(f);
    g_profile_enabled = !g_profile.empty();
    if (g_profile_enabled) {
        std::fprintf(stderr, "[moe_stream_batch] profile preload: loaded %zu entries from %s\n",
                     g_profile.size(), path);
    }
    g_profile_loaded = true;
}

static int batch_cache_id_for_size(size_t expert_sz) {
    const char *fused_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE");
    if (fused_env && fused_env[0] && fused_env[0] != '0' && expert_sz <= 2ULL*1024ULL*1024ULL) {
        return 1;
    }
    return 0;
}

static batch_vram_cache * batch_cache_get(size_t expert_sz) {
    const int cid = batch_cache_id_for_size(expert_sz);
    if (g_bcache_inited[cid]) {
        batch_vram_cache *c = &g_bcaches[cid];
        return (c->pool && expert_sz <= c->slot_sz) ? c : nullptr;
    }
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_GB");
    const size_t budget_gb = env ? (size_t)std::atoi(env) : 16;
    if (budget_gb == 0) {
        g_bcache_inited[cid] = true;
        return nullptr;
    }
    size_t this_budget_gb = budget_gb;
    const char *fused_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE");
    if (fused_env && fused_env[0] && fused_env[0] != '0' && budget_gb > 8 && cid == 1) {
        this_budget_gb = 8;
    }
    const size_t budget = this_budget_gb * 1024ULL * 1024ULL * 1024ULL;
    batch_vram_cache *c = &g_bcaches[cid];
    c->slot_sz = expert_sz;
    c->n_slots = (int)(budget / expert_sz);
    if (c->n_slots > 16384) c->n_slots = 16384;
    if (c->n_slots < 1) {
        g_bcache_inited[cid] = true;
        return nullptr;
    }
    const size_t alloc = (size_t)c->n_slots * expert_sz;
    if (cudaMalloc(&c->pool, alloc) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: cudaMalloc %.1f GiB FAILED\n",
                     alloc / (1024.0*1024.0*1024.0));
        c->n_slots = 0;
        g_bcache_inited[cid] = true;
        return nullptr;
    }
    std::fprintf(stderr, "[moe_stream_batch] VRAM cache: %.1f GiB, %d slots (%.2f MiB each)\n",
                 alloc / (1024.0*1024.0*1024.0), c->n_slots,
                 expert_sz / (1024.0*1024.0));
    std::atexit(batch_cache_report_atexit);
    g_bcache_inited[cid] = true;
    return c;
}

static int batch_cache_lookup_slot(batch_vram_cache *c, uintptr_t key) {
    if (!c || !c->pool || c->n_slots == 0) return -1;
    for (int slot = 0; slot < c->n_slots; ++slot) {
        if (c->slot_key[slot] == key) {
            c->slot_used[slot] = c->clock++;
            ++c->hits;
            return slot;
        }
    }
    return -1;
}

static int batch_cache_insert_slot(batch_vram_cache *c, uintptr_t key, const void *host_data, size_t sz, cudaStream_t st, bool allow_evict, bool preload) {
    if (!c || !c->pool || c->n_slots == 0 || sz > c->slot_sz) return -1;
    int slot = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < c->n_slots; ++i) {
        if (c->slot_key[i] == 0) {
            slot = i;
            break;
        }
        if (!allow_evict) continue;
        if (c->slot_used[i] < oldest) {
            oldest = c->slot_used[i];
            slot = i;
        }
    }
    if (slot < 0) return -1;
    c->slot_key[slot] = key;
    c->slot_used[slot] = c->clock++;
    void *dst = (char *)c->pool + (size_t)slot * c->slot_sz;
    cudaMemcpyAsync(dst, host_data, sz, cudaMemcpyHostToDevice, st);
    if (preload) {
        ++c->preloads;
    } else {
        ++c->misses;
    }
    return slot;
}

static void preload_profile_for_tensor(
        const char *tensor_name, const void *src0_data, int64_t n_as, size_t nb02, size_t src0_bytes, cudaStream_t st) {
    load_profile_once();
    if (!g_profile_enabled || !tensor_name || !tensor_name[0]) return;
    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    if (tensor_already_preloaded(tensor_name)) return;
    char lookup_name[96] = {};
    profile_lookup_name(tensor_name, lookup_name, sizeof(lookup_name));
    int loaded = 0;
    size_t slot_share = (size_t)cache->n_slots;
    const char *share_env = std::getenv("GGML_MOE_VRAM_PROFILE_SHARE");
    if (share_env && share_env[0] && share_env[0] != '0' &&
        (std::strstr(tensor_name, ".ffn_down_exps.") ||
         std::strstr(tensor_name, ".ffn_up_exps.") ||
         std::strstr(tensor_name, ".ffn_gate_exps."))) {
        slot_share = slot_share / 3;
        if (slot_share < 1) slot_share = 1;
    }
    const size_t profile_limit = slot_share < g_profile.size() ? slot_share : g_profile.size();
    for (size_t ip = 0; ip < profile_limit; ++ip) {
        const profile_entry &e = g_profile[ip];
        if (std::strcmp(e.tensor, tensor_name) != 0 && std::strcmp(e.tensor, lookup_name) != 0) continue;
        if (e.expert_idx < 0 || e.expert_idx >= n_as) continue;
        const uintptr_t key = batch_key_hash(tensor_name, e.expert_idx);
        if (batch_cache_lookup_slot(cache, key) >= 0) continue;
        const char *expert_host = (const char *)src0_data + (size_t)e.expert_idx * nb02;
        if (batch_cache_insert_slot(cache, key, expert_host, src0_bytes, st, false, true) < 0) break;
        ++loaded;
    }
    if (loaded > 0) {
        std::fprintf(stderr, "[moe_stream_batch] profile preload: %s loaded=%d\n", tensor_name, loaded);
    }
}

static bool ensure_dev(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFree(p);
    p = nullptr;
    cur = 0;
    if (cudaMalloc(&p, need) != cudaSuccess) return false;
    cur = need;
    return true;
}

static bool ensure_host_pinned(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFreeHost(p);
    p = nullptr;
    cur = 0;
    if (cudaHostAlloc(&p, need, cudaHostAllocDefault) != cudaSuccess) return false;
    cur = need;
    return true;
}

static bool gpu_handoff_enabled() {
    const char *env = std::getenv("GGML_MOE_GPU_HANDOFF");
    return env && env[0] && env[0] != '0';
}

static bool init_batch_once() {
    if (g_batch_inited.load(std::memory_order_acquire)) return g_batch.stream != nullptr;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    if (g_batch_inited.load(std::memory_order_acquire)) return g_batch.stream != nullptr;
    const char *env = std::getenv("GGML_MOE_STREAM");
    if (env && env[0] && env[0] != '0' && cudaSetDevice(0) == cudaSuccess) {
        if (cudaStreamCreate(&g_batch.stream) != cudaSuccess) {
            g_batch.stream = nullptr;
        }
        const char *prof_env = std::getenv("GGML_MOE_BATCH_PROFILE");
        g_bprof.enabled = prof_env && prof_env[0] && prof_env[0] != '0';
        if (g_batch.stream && g_bprof.enabled) {
            cudaEventCreate(&g_batch.ev_start);
            cudaEventCreate(&g_batch.ev_stage);
            cudaEventCreate(&g_batch.ev_quant);
            cudaEventCreate(&g_batch.ev_kernel);
            cudaEventCreate(&g_batch.ev_d2h);
            std::atexit(batch_profile_report_atexit);
        }
    }
    g_batch_inited.store(true, std::memory_order_release);
    return g_batch.stream != nullptr;
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    if (!init_batch_once()) return false;
    if ((ggml_type)src0_type_int != GGML_TYPE_IQ3_XXS || !src0_data || !src0_name) return false;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    if (!batch_cache_get(expert_bytes)) return false;
    preload_profile_for_tensor(src0_name, src0_data, n_as, nb02, expert_bytes, g_batch.stream);
    cudaStreamSynchronize(g_batch.stream);
    return true;
}

static bool launch_iq3_xxs_mmq_id_batch(
        const char * d_src0, const int * d_src1_q8, const int32_t * d_ids_dst,
        const int32_t * d_bounds, const int32_t * d_x_ids, float * d_dst,
        int64_t ne00, int64_t ne01, int64_t src0_stride, int64_t src0_channel_stride,
        int64_t n_active, int64_t dst_cols, cudaStream_t st) {
    const mmq_args_id args = {
        d_src0, GGML_TYPE_IQ3_XXS, d_src1_q8, d_ids_dst, d_bounds, d_x_ids, d_dst,
        ne00, ne01, dst_cols, src0_stride, n_active, ne01,
        n_active, n_active, src0_channel_stride, 0, 0,
        1, 1, 0, 0, 0,
        false, n_active};
    ggml_backend_cuda_context * null_ctx = nullptr;
    launch_mul_mat_q_id<GGML_TYPE_IQ3_XXS, 8>(*null_ctx, args, st);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_iq3_xxs_mmq_id_one(
        const char * d_src0, const int * d_src1_q8, const int32_t * d_ids_dst,
        const int32_t * d_bounds, const int32_t * d_x_ids, float * d_dst,
        int64_t ne00, int64_t ne01, int64_t src0_stride, int64_t src0_channel_stride,
        int64_t dst_cols, cudaStream_t st) {
    const mmq_args_id args = {
        d_src0, GGML_TYPE_IQ3_XXS, d_src1_q8, d_ids_dst, d_bounds, d_x_ids, d_dst,
        ne00, ne01, dst_cols, src0_stride, 1, ne01,
        1, 1, src0_channel_stride, 0, 0,
        1, 1, 0, 0, 0,
        false, 1};
    ggml_backend_cuda_context * null_ctx = nullptr;
    launch_mul_mat_q_id<GGML_TYPE_IQ3_XXS, 8>(*null_ctx, args, st);
    return cudaGetLastError() == cudaSuccess;
}

static __device__ __forceinline__ float moe_stream_silu(float x) {
    return x / (1.0f + expf(-x));
}

static __global__ void moe_stream_up_gate_fuse_kernel(
        const float *up, const float *gate, float *dst,
        const int32_t *dst_ids, int n_active, int64_t ne01, int unary_op, float limit, bool read_compact) {
    const int j = (int)blockIdx.y;
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_active || col >= ne01) return;
    const int row = dst_ids[j];
    const int read_row = read_compact ? j : row;
    const int64_t read_off = (int64_t)read_row * ne01 + col;
    const int64_t write_off = (int64_t)row * ne01 + col;
    float u = up[read_off];
    float g = gate[read_off];
    float r = 0.0f;
    switch ((ggml_unary_op)unary_op) {
        case GGML_UNARY_OP_SILU:
            if (limit < 1e-6f) {
                r = u * moe_stream_silu(g);
            } else {
                float gate_v = fminf(moe_stream_silu(g), limit);
                float up_v = fmaxf(-limit, fminf(limit, u));
                r = up_v * gate_v;
            }
            break;
        case GGML_UNARY_OP_RELU:
            r = u * fmaxf(g, 0.0f);
            break;
        case GGML_UNARY_OP_GELU: {
            constexpr float GELU_COEF_A    = 0.044715f;
            constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
            r = 0.5f * g * u * (1.0f + tanhf(SQRT_2_OVER_PI * g * (1.0f + GELU_COEF_A * g * g)));
            break;
        }
        default:
            r = 0.0f;
            break;
    }
    dst[write_off] = r;
}

static void moe_stream_dump_prefuse_rows(const char *tag, const float *buf, int n_active, int64_t ne01) {
    std::fprintf(stderr, "[moe_stream] up/gate prefuse %s:", tag);
    for (int r = 0; r < n_active; ++r) {
        double l1 = 0.0;
        double max_abs = 0.0;
        for (int64_t c = 0; c < ne01; ++c) {
            const double v = buf[(size_t)r * (size_t)ne01 + (size_t)c];
            const double av = std::fabs(v);
            l1 += av;
            if (av > max_abs) max_abs = av;
        }
        std::fprintf(stderr, " row%d_l1=%g row%d_max=%g", r, l1, r, max_abs);
    }
    std::fprintf(stderr, "\n");
}

static void moe_stream_dump_f32_rows(const char *tag, const float *buf, int n_rows, int64_t ne00) {
    std::fprintf(stderr, "[moe_stream] up/gate %s:", tag);
    for (int r = 0; r < n_rows; ++r) {
        double l1 = 0.0;
        double max_abs = 0.0;
        for (int64_t c = 0; c < ne00; ++c) {
            const double v = buf[(size_t)r * (size_t)ne00 + (size_t)c];
            const double av = std::fabs(v);
            l1 += av;
            if (av > max_abs) max_abs = av;
        }
        std::fprintf(stderr, " row%d_l1=%g row%d_max=%g", r, l1, r, max_abs);
    }
    std::fprintf(stderr, "\n");
}

extern "C" bool ggml_cuda_moe_stream_up_gate_batch(
    int  src0_type_int,
    const char *src0_up_name,
    const void *src0_up_data,
    const char *src0_gate_name,
    const void *src0_gate_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    int unary_op,
    float limit,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_stride) {
    if (!init_batch_once()) return false;
    if ((ggml_type)src0_type_int != GGML_TYPE_IQ3_XXS || !src1_f32 || !src0_up_data || !src0_gate_data) return false;
    if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_RELU && unary_op != GGML_UNARY_OP_GELU) return false;
    GGML_UNUSED(src1_nb1);

    int active_experts[128];
    int32_t dst_ids[128];
    int32_t token_ids[128];
    int n_active = 0;
    int max_dst_id = -1;
    for (int64_t e = 0; e < n_as; ++e) {
        if (matrix_row_counts[e] != 1) {
            if (matrix_row_counts[e] > 1) return false;
            continue;
        }
        if (n_active >= 128) return false;
        const ggml_moe_row_mapping * r = matrix_rows + e*rows_stride;
        active_experts[n_active] = (int)e;
        dst_ids[n_active] = r[0].i1;
        token_ids[n_active] = r[0].i2;
        if (dst_ids[n_active] > max_dst_id) max_dst_id = dst_ids[n_active];
        ++n_active;
    }
    if (n_active <= 0 || max_dst_id < 0) return false;

    static std::atomic<int> first_up_gate{0};
    if (first_up_gate.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] batched up/gate decode path active: experts=%d ne01=%ld ne00=%ld\n",
                     n_active, (long)ne01, (long)ne00);
        std::fprintf(stderr, "[moe_stream] up/gate routes:");
        for (int j = 0; j < n_active; ++j) {
            std::fprintf(stderr, " #%d:e%d->dst%d/tok%d", j, active_experts[j], dst_ids[j], token_ids[j]);
        }
        std::fprintf(stderr, "\n");
    }

    std::lock_guard<std::mutex> lk(g_batch_mu);
    batch_ctx &bc = g_batch;
    cudaStream_t st = bc.stream;

    const size_t src0_bytes = (size_t)ne01 * nb01;
    const int64_t ne00_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t src1_f32_bytes = (size_t)n_active * ne00 * sizeof(float);
    const size_t src1_q8_bytes = (size_t)n_active * ne00_padded * sizeof(block_q8_1) / QK8_1
        + (size_t)get_mmq_x_max_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) * sizeof(block_q8_1_mmq);
    const size_t src1_q8_one_bytes = (size_t)ne00_padded * sizeof(block_q8_1) / QK8_1
        + (size_t)get_mmq_x_max_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) * sizeof(block_q8_1_mmq);
    const int64_t dst_cols = max_dst_id + 1;
    const size_t dst_bytes = (size_t)dst_cols * ne01 * sizeof(float);
    const size_t ids_bytes = (size_t)n_active * sizeof(int32_t);
    const size_t bounds_bytes = (size_t)(n_active + 1) * sizeof(int32_t);
    const bool use_handoff = gpu_handoff_enabled();

    bool ok = ensure_dev(bc.d_src1_f32, bc.d_src1_f32_sz, src1_f32_bytes)
        && ensure_dev(bc.d_src1_q8, bc.d_src1_q8_sz, src1_q8_bytes)
        && ensure_dev(bc.d_src1_q8_one, bc.d_src1_q8_one_sz, src1_q8_one_bytes)
        && ensure_dev(bc.d_dst, bc.d_dst_sz, dst_bytes)
        && ensure_dev(bc.d_up, bc.d_up_sz, dst_bytes)
        && ensure_dev(bc.d_gate, bc.d_gate_sz, dst_bytes)
        && (!use_handoff || ensure_dev(bc.d_handoff, bc.d_handoff_sz, dst_bytes))
        && ensure_dev((void *&)bc.d_ids_src1, bc.d_ids_src1_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_ids_dst, bc.d_ids_dst_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_x_ids, bc.d_x_ids_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_bounds, bc.d_bounds_sz, bounds_bytes)
        && ensure_host_pinned(bc.h_src1, bc.h_src1_sz, src1_f32_bytes)
        && ensure_host_pinned(bc.h_dst, bc.h_dst_sz, dst_bytes);
    if (!ok) return false;

    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return false;
    preload_profile_for_tensor(src0_up_name, src0_up_data, n_as, nb02, src0_bytes, st);
    preload_profile_for_tensor(src0_gate_name, src0_gate_data, n_as, nb02, src0_bytes, st);

    char up_key_name[128] = {};
    char gate_key_name[128] = {};
    const bool shared_tensor_name = src0_up_name && src0_gate_name && std::strcmp(src0_up_name, src0_gate_name) == 0;
    const bool disambiguate_halves = shared_tensor_name || src0_up_data == src0_gate_data;
    std::snprintf(up_key_name, sizeof(up_key_name), "%s%s", src0_up_name ? src0_up_name : "up", disambiguate_halves ? ":up" : "");
    std::snprintf(gate_key_name, sizeof(gate_key_name), "%s%s", src0_gate_name ? src0_gate_name : "gate", disambiguate_halves ? ":gate" : "");

    const char *serial_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_SERIAL");
    const bool serial_up_gate = serial_env && serial_env[0] && serial_env[0] != '0';
    static std::atomic<int> first_serial_up_gate{0};
    if (serial_up_gate && first_serial_up_gate.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] up/gate serial MMQ probe active\n");
    }

    auto stage_tensor = [&](const char *key_name, const void *host_base, void *d_out) -> bool {
        for (int j = 0; j < n_active; ++j) {
            const char *expert_host = (const char *)host_base + (size_t)active_experts[j] * nb02;
            const uintptr_t cache_key = batch_key_hash(key_name, active_experts[j]);
            int cache_slot = batch_cache_lookup_slot(cache, cache_key);
            if (cache_slot < 0) {
                cache_slot = batch_cache_insert_slot(cache, cache_key, expert_host, src0_bytes, st, true, false);
            }
            if (cache_slot < 0) return false;
            bc.h_x_ids[j] = cache_slot;
        }
        if (cudaMemsetAsync(d_out, 0, dst_bytes, st) != cudaSuccess) return false;
        if (serial_up_gate) {
            bc.h_bounds[0] = 0;
            bc.h_bounds[1] = 1;
            if (cudaMemcpyAsync(bc.d_bounds, bc.h_bounds, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
            for (int j = 0; j < n_active; ++j) {
                bc.h_up_gate_ids_dst[0] = j;
                bc.h_up_gate_ids_dst[1] = bc.h_x_ids[j];
                if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_up_gate_ids_dst, sizeof(int32_t), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
                if (cudaMemcpyAsync(bc.d_x_ids, bc.h_up_gate_ids_dst + 1, sizeof(int32_t), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
                quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32 + (size_t)j * ne00, bc.d_ids_src1, bc.d_src1_q8_one,
                    GGML_TYPE_IQ3_XXS, ne00, ne00, ne00, ne00,
                    ne00_padded, 1, 1, 1, st);
                if (cudaGetLastError() != cudaSuccess) return false;
                if (!launch_iq3_xxs_mmq_id_one(
                        (const char *)cache->pool, (const int *)bc.d_src1_q8_one, bc.d_ids_dst, bc.d_bounds,
                        bc.d_x_ids, (float *)d_out, ne00, ne01, nb01, src0_bytes, dst_cols, st)) {
                    return false;
                }
            }
            return true;
        }
        if (cudaMemcpyAsync(bc.d_x_ids, bc.h_x_ids, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
        return launch_iq3_xxs_mmq_id_batch(
            (const char *)cache->pool, (const int *)bc.d_src1_q8, bc.d_ids_dst, bc.d_bounds,
            bc.d_x_ids, (float *)d_out, ne00, ne01, nb01, src0_bytes, n_active, dst_cols, st);
    };

    for (int j = 0; j < n_active; ++j) {
        const char *src1_base = (const char *)src1_f32;
        const char *src_row = src1_base + (size_t)token_ids[j] * src1_nb2;
        std::memcpy((char *)bc.h_src1 + (size_t)j * ne00 * sizeof(float), src_row, (size_t)ne00 * sizeof(float));
        bc.h_ids_src1[j] = j;
        bc.h_ids_dst[j] = dst_ids[j];
        bc.h_up_gate_ids_dst[j] = j;
        bc.h_bounds[j] = j;
    }
    bc.h_bounds[n_active] = n_active;

    const char *src1_dump_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_SRC1_DUMP");
    const bool src1_dump = src1_dump_env && src1_dump_env[0] && src1_dump_env[0] != '0';
    static std::atomic<int> src1_dump_count{0};
    if (src1_dump && src1_dump_count.fetch_add(1) == 0) {
        moe_stream_dump_f32_rows("src1", (const float *)bc.h_src1, n_active, ne00);
    }

    if (cudaMemcpyAsync(bc.d_src1_f32, bc.h_src1, src1_f32_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_ids_src1, bc.h_ids_src1, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_up_gate_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_bounds, bc.h_bounds, bounds_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;

    if (!serial_up_gate) {
        quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32, bc.d_ids_src1, bc.d_src1_q8,
            GGML_TYPE_IQ3_XXS, ne00, ne00, n_active * ne00, n_active * ne00,
            ne00_padded, n_active, 1, 1, st);
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    float * fused_d = use_handoff ? (float *)bc.d_handoff : (float *)bc.d_dst;
    if (!stage_tensor(up_key_name, src0_up_data, bc.d_up)) return false;
    if (!stage_tensor(gate_key_name, src0_gate_data, bc.d_gate)) return false;

    const char *prefuse_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_PREFUSE_DUMP");
    const bool prefuse_dump = prefuse_env && prefuse_env[0] && prefuse_env[0] != '0';
    static std::atomic<int> prefuse_dump_count{0};
    if (prefuse_dump && prefuse_dump_count.fetch_add(1) == 0) {
        if (cudaMemcpyAsync(bc.h_dst, bc.d_up, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        moe_stream_dump_prefuse_rows("up", (const float *)bc.h_dst, n_active, ne01);
        if (cudaMemcpyAsync(bc.h_dst, bc.d_gate, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        moe_stream_dump_prefuse_rows("gate", (const float *)bc.h_dst, n_active, ne01);
    }

    for (int j = 0; j < n_active; ++j) {
        bc.h_ids_dst[j] = dst_ids[j];
    }
    if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;

    dim3 block(256);
    dim3 grid((unsigned int)((ne01 + block.x - 1) / block.x), (unsigned int)n_active);
    moe_stream_up_gate_fuse_kernel<<<grid, block, 0, st>>>(
        (const float *)bc.d_up, (const float *)bc.d_gate, fused_d,
        bc.d_ids_dst, n_active, ne01, unary_op, limit, true);
    if (cudaGetLastError() != cudaSuccess) return false;

    if (use_handoff) {
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        g_handoff.host_ptr = dst;
        g_handoff.d_data = fused_d;
        g_handoff.bytes = dst_bytes;
        g_handoff.ne01 = ne01;
        g_handoff.dst_cols = dst_cols;
        ++g_handoff.serial;
        return true;
    }

    if (cudaMemcpyAsync(bc.h_dst, bc.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;

    const float *tmp = (const float *)bc.h_dst;
    for (int j = 0; j < n_active; ++j) {
        float *dst_row = (float *)((char *)dst + (size_t)dst_ids[j] * dst_nb1 + (size_t)token_ids[j] * dst_nb2);
        std::memcpy(dst_row, tmp + (size_t)dst_ids[j] * ne01, (size_t)ne01 * sizeof(float));
    }
    return true;
}

extern "C" bool ggml_cuda_moe_stream_batch(
    int  src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const int64_t *matrix_row_counts,
    const ggml_moe_row_mapping *matrix_rows,
    int64_t rows_stride) {
    if (!init_batch_once()) return false;
    if ((ggml_type)src0_type_int != GGML_TYPE_IQ3_XXS || !src1_f32) return false;

    int active_experts[128];
    int32_t dst_ids[128];
    int32_t token_ids[128];
    int n_active = 0;
    int max_dst_id = -1;
    for (int64_t e = 0; e < n_as; ++e) {
        if (matrix_row_counts[e] != 1) {
            if (matrix_row_counts[e] > 1) return false;
            continue;
        }
        if (n_active >= 128) return false;
        const ggml_moe_row_mapping * r = matrix_rows + e*rows_stride;
        active_experts[n_active] = (int)e;
        dst_ids[n_active] = r[0].i1;
        token_ids[n_active] = r[0].i2;
        if (dst_ids[n_active] > max_dst_id) max_dst_id = dst_ids[n_active];
        ++n_active;
    }
    if (n_active <= 0 || max_dst_id < 0) return false;

    static std::atomic<int> first_batch{0};
    const int batch_call = first_batch.fetch_add(1);
    const char *trace_env = std::getenv("GGML_MOE_BATCH_TRACE");
    if (batch_call == 0) {
        std::fprintf(stderr, "[moe_stream] batched decode path active: experts=%d ne01=%ld ne00=%ld\n",
                     n_active, (long)ne01, (long)ne00);
    }
    if (trace_env && trace_env[0] && trace_env[0] != '0' && batch_call < 64) {
        std::fprintf(stderr, "[moe_stream_batch] trace call=%d tensor=%s experts=%d ne01=%ld ne00=%ld\n",
                     batch_call, src0_name ? src0_name : "(null)", n_active, (long)ne01, (long)ne00);
    }

    std::lock_guard<std::mutex> lk(g_batch_mu);
    batch_ctx &bc = g_batch;
    cudaStream_t st = bc.stream;
    const bool profile = g_bprof.enabled && bc.ev_start && bc.ev_stage && bc.ev_quant && bc.ev_kernel && bc.ev_d2h;

    const size_t src0_bytes = (size_t)ne01 * nb01;
    const size_t src0_all_bytes = (size_t)n_active * src0_bytes;
    const int64_t ne00_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t src1_f32_bytes = (size_t)n_active * ne00 * sizeof(float);
    const size_t src1_q8_bytes = (size_t)n_active * ne00_padded * sizeof(block_q8_1) / QK8_1
        + (size_t)get_mmq_x_max_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) * sizeof(block_q8_1_mmq);
    const int64_t dst_cols = max_dst_id + 1;
    const bool use_handoff =
        gpu_handoff_enabled() &&
        g_handoff.host_ptr == src1_f32 &&
        g_handoff.d_data &&
        g_handoff.ne01 == ne00 &&
        g_handoff.dst_cols >= dst_cols;
    const size_t dst_bytes = (size_t)dst_cols * ne01 * sizeof(float);
    const size_t ids_bytes = (size_t)n_active * sizeof(int32_t);
    const size_t bounds_bytes = (size_t)(n_active + 1) * sizeof(int32_t);

    bool ok = ensure_dev(bc.d_src0, bc.d_src0_sz, src0_all_bytes)
        && (use_handoff || ensure_dev(bc.d_src1_f32, bc.d_src1_f32_sz, src1_f32_bytes))
        && ensure_dev(bc.d_src1_q8, bc.d_src1_q8_sz, src1_q8_bytes)
        && ensure_dev(bc.d_dst, bc.d_dst_sz, dst_bytes)
        && ensure_dev((void *&)bc.d_ids_src1, bc.d_ids_src1_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_ids_dst, bc.d_ids_dst_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_x_ids, bc.d_x_ids_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_bounds, bc.d_bounds_sz, bounds_bytes)
        && (use_handoff || ensure_host_pinned(bc.h_src1, bc.h_src1_sz, src1_f32_bytes))
        && ensure_host_pinned(bc.h_dst, bc.h_dst_sz, dst_bytes);
    if (!ok) return false;

    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return false;
    preload_profile_for_tensor(src0_name, src0_data, n_as, nb02, src0_bytes, st);

    if (profile) cudaEventRecord(bc.ev_start, st);

    for (int j = 0; j < n_active; ++j) {
        const char *expert_host = (const char *)src0_data + (size_t)active_experts[j] * nb02;
        const uintptr_t cache_key = batch_key_hash(src0_name, active_experts[j]);
        int cache_slot = batch_cache_lookup_slot(cache, cache_key);
        if (cache_slot < 0) {
            cache_slot = batch_cache_insert_slot(cache, cache_key, expert_host, src0_bytes, st, true, false);
        }
        if (cache_slot < 0) return false;
        bc.h_x_ids[j] = cache_slot;

        if (!use_handoff) {
            const char *src1_base = (const char *)src1_f32;
            const char *src_row = src1_base + (size_t)dst_ids[j] * src1_nb1 + (size_t)token_ids[j] * src1_nb2;
            std::memcpy((char *)bc.h_src1 + (size_t)j * ne00 * sizeof(float), src_row, (size_t)ne00 * sizeof(float));
        }
        bc.h_ids_src1[j] = j;
        bc.h_ids_dst[j] = dst_ids[j];
        bc.h_bounds[j] = j;
    }
    bc.h_bounds[n_active] = n_active;

    if (!use_handoff && cudaMemcpyAsync(bc.d_src1_f32, bc.h_src1, src1_f32_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_ids_src1, bc.h_ids_src1, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_x_ids, bc.h_x_ids, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_bounds, bc.h_bounds, bounds_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemsetAsync(bc.d_dst, 0, dst_bytes, st) != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_stage, st);

    if (use_handoff) {
        static std::atomic<int> first_handoff_consume{0};
        if (first_handoff_consume.fetch_add(1) == 0) {
            std::fprintf(stderr, "[moe_stream_batch] GPU handoff consumed: ne00=%ld dst_cols=%ld\n",
                         (long)ne00, (long)dst_cols);
        }
        quantize_mmq_q8_1_cuda_id(g_handoff.d_data, bc.d_ids_dst, bc.d_src1_q8,
            GGML_TYPE_IQ3_XXS, ne00, g_handoff.ne01, g_handoff.ne01 * g_handoff.dst_cols, g_handoff.ne01 * g_handoff.dst_cols,
            ne00_padded, n_active, 1, 1, st);
        g_handoff.host_ptr = nullptr;
        g_handoff.d_data = nullptr;
    } else {
        quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32, bc.d_ids_src1, bc.d_src1_q8,
            GGML_TYPE_IQ3_XXS, ne00, ne00, n_active * ne00, n_active * ne00,
            ne00_padded, n_active, 1, 1, st);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_quant, st);

    if (!launch_iq3_xxs_mmq_id_batch(
            (const char *)cache->pool, (const int *)bc.d_src1_q8, bc.d_ids_dst, bc.d_bounds,
            bc.d_x_ids, (float *)bc.d_dst, ne00, ne01, nb01, src0_bytes, n_active, dst_cols, st)) {
        return false;
    }
    if (profile) cudaEventRecord(bc.ev_kernel, st);
    if (cudaMemcpyAsync(bc.h_dst, bc.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_d2h, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;

    float stage_ms = 0.0f;
    float quant_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    if (profile) {
        cudaEventElapsedTime(&stage_ms, bc.ev_start, bc.ev_stage);
        cudaEventElapsedTime(&quant_ms, bc.ev_stage, bc.ev_quant);
        cudaEventElapsedTime(&kernel_ms, bc.ev_quant, bc.ev_kernel);
        cudaEventElapsedTime(&d2h_ms, bc.ev_kernel, bc.ev_d2h);
    }

    const float *tmp = (const float *)bc.h_dst;
    const auto scatter_start = std::chrono::steady_clock::now();
    for (int j = 0; j < n_active; ++j) {
        float *dst_row = (float *)((char *)dst + (size_t)dst_ids[j] * dst_nb1 + (size_t)token_ids[j] * dst_nb2);
        std::memcpy(dst_row, tmp + (size_t)dst_ids[j] * ne01, (size_t)ne01 * sizeof(float));
    }
    if (profile) {
        const auto scatter_end = std::chrono::steady_clock::now();
        const double scatter_ms = std::chrono::duration<double, std::milli>(scatter_end - scatter_start).count();
        ++g_bprof.calls;
        g_bprof.active_experts += (uint64_t)n_active;
        g_bprof.stage_ms += stage_ms;
        g_bprof.quant_ms += quant_ms;
        g_bprof.kernel_ms += kernel_ms;
        g_bprof.d2h_ms += d2h_ms;
        g_bprof.scatter_ms += scatter_ms;
    }
    return true;
}
