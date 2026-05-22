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
#include <mutex>

#include <cuda_runtime.h>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

bool ggml_cuda_moe_stream_batch(
    int  src0_type_int,
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

}

struct batch_ctx {
    cudaStream_t stream = nullptr;
    void * d_src0 = nullptr;     size_t d_src0_sz = 0;
    void * d_src1_f32 = nullptr; size_t d_src1_f32_sz = 0;
    void * d_src1_q8 = nullptr;  size_t d_src1_q8_sz = 0;
    void * d_dst = nullptr;      size_t d_dst_sz = 0;
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
    int32_t h_ids_src1[128] = {};
    int32_t h_x_ids[128] = {};
    int32_t h_bounds[129] = {};
};

static batch_ctx g_batch;
static std::mutex g_batch_mu;
static std::atomic<bool> g_batch_inited{false};

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
};

static batch_vram_cache g_bcache;
static bool g_bcache_inited = false;

static void batch_cache_report_atexit() {
    const uint64_t total = g_bcache.hits + g_bcache.misses;
    if (total == 0) return;
    std::fprintf(stderr, "[moe_stream_batch] VRAM cache: hits=%lu misses=%lu hit_rate=%.1f%%\n",
                 g_bcache.hits, g_bcache.misses, 100.0 * g_bcache.hits / total);
}

static void batch_cache_init(size_t expert_sz) {
    if (g_bcache_inited) return;
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_GB");
    const size_t budget_gb = env ? (size_t)std::atoi(env) : 16;
    if (budget_gb == 0) {
        g_bcache_inited = true;
        return;
    }
    const size_t budget = budget_gb * 1024ULL * 1024ULL * 1024ULL;
    g_bcache.slot_sz = expert_sz;
    g_bcache.n_slots = (int)(budget / expert_sz);
    if (g_bcache.n_slots > 16384) g_bcache.n_slots = 16384;
    if (g_bcache.n_slots < 1) {
        g_bcache_inited = true;
        return;
    }
    const size_t alloc = (size_t)g_bcache.n_slots * expert_sz;
    if (cudaMalloc(&g_bcache.pool, alloc) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: cudaMalloc %.1f GiB FAILED\n",
                     alloc / (1024.0*1024.0*1024.0));
        g_bcache.n_slots = 0;
        g_bcache_inited = true;
        return;
    }
    std::fprintf(stderr, "[moe_stream_batch] VRAM cache: %.1f GiB, %d slots (%.2f MiB each)\n",
                 alloc / (1024.0*1024.0*1024.0), g_bcache.n_slots,
                 expert_sz / (1024.0*1024.0));
    std::atexit(batch_cache_report_atexit);
    g_bcache_inited = true;
}

static int batch_cache_lookup_slot(uintptr_t key) {
    if (!g_bcache.pool || g_bcache.n_slots == 0) return -1;
    for (int slot = 0; slot < g_bcache.n_slots; ++slot) {
        if (g_bcache.slot_key[slot] == key) {
            g_bcache.slot_used[slot] = g_bcache.clock++;
            ++g_bcache.hits;
            return slot;
        }
    }
    return -1;
}

static int batch_cache_insert_slot(uintptr_t key, const void *host_data, size_t sz, cudaStream_t st) {
    if (!g_bcache.pool || g_bcache.n_slots == 0 || sz > g_bcache.slot_sz) return -1;
    int slot = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < g_bcache.n_slots; ++i) {
        if (g_bcache.slot_key[i] == 0) {
            slot = i;
            break;
        }
        if (g_bcache.slot_used[i] < oldest) {
            oldest = g_bcache.slot_used[i];
            slot = i;
        }
    }
    if (slot < 0) return -1;
    g_bcache.slot_key[slot] = key;
    g_bcache.slot_used[slot] = g_bcache.clock++;
    void *dst = (char *)g_bcache.pool + (size_t)slot * g_bcache.slot_sz;
    cudaMemcpyAsync(dst, host_data, sz, cudaMemcpyHostToDevice, st);
    ++g_bcache.misses;
    return slot;
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
        false, 1};
    ggml_backend_cuda_context * null_ctx = nullptr;
    launch_mul_mat_q_id<GGML_TYPE_IQ3_XXS, 8>(*null_ctx, args, st);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool ggml_cuda_moe_stream_batch(
    int  src0_type_int,
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
    if (first_batch.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] batched decode path active: experts=%d ne01=%ld ne00=%ld\n",
                     n_active, (long)ne01, (long)ne00);
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
    const size_t dst_bytes = (size_t)dst_cols * ne01 * sizeof(float);
    const size_t ids_bytes = (size_t)n_active * sizeof(int32_t);
    const size_t bounds_bytes = (size_t)(n_active + 1) * sizeof(int32_t);

    bool ok = ensure_dev(bc.d_src0, bc.d_src0_sz, src0_all_bytes)
        && ensure_dev(bc.d_src1_f32, bc.d_src1_f32_sz, src1_f32_bytes)
        && ensure_dev(bc.d_src1_q8, bc.d_src1_q8_sz, src1_q8_bytes)
        && ensure_dev(bc.d_dst, bc.d_dst_sz, dst_bytes)
        && ensure_dev((void *&)bc.d_ids_src1, bc.d_ids_src1_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_ids_dst, bc.d_ids_dst_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_x_ids, bc.d_x_ids_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_bounds, bc.d_bounds_sz, bounds_bytes)
        && ensure_host_pinned(bc.h_src1, bc.h_src1_sz, src1_f32_bytes)
        && ensure_host_pinned(bc.h_dst, bc.h_dst_sz, dst_bytes);
    if (!ok) return false;

    if (!g_bcache_inited) {
        batch_cache_init(src0_bytes);
    }
    if (!g_bcache.pool || g_bcache.n_slots <= 0) return false;

    if (profile) cudaEventRecord(bc.ev_start, st);

    for (int j = 0; j < n_active; ++j) {
        const char *expert_host = (const char *)src0_data + (size_t)active_experts[j] * nb02;
        const uintptr_t cache_key = (uintptr_t)expert_host;
        int cache_slot = batch_cache_lookup_slot(cache_key);
        if (cache_slot < 0) {
            cache_slot = batch_cache_insert_slot(cache_key, expert_host, src0_bytes, st);
        }
        if (cache_slot < 0) return false;
        bc.h_x_ids[j] = cache_slot;

        const char *src1_base = (const char *)src1_f32;
        const char *src_row = src1_base + (size_t)dst_ids[j] * src1_nb1 + (size_t)token_ids[j] * src1_nb2;
        std::memcpy((char *)bc.h_src1 + (size_t)j * ne00 * sizeof(float), src_row, (size_t)ne00 * sizeof(float));
        bc.h_ids_src1[j] = j;
        bc.h_ids_dst[j] = dst_ids[j];
        bc.h_bounds[j] = j;
    }
    bc.h_bounds[n_active] = n_active;

    if (cudaMemcpyAsync(bc.d_src1_f32, bc.h_src1, src1_f32_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_ids_src1, bc.h_ids_src1, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_x_ids, bc.h_x_ids, ids_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemcpyAsync(bc.d_bounds, bc.h_bounds, bounds_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    if (cudaMemsetAsync(bc.d_dst, 0, dst_bytes, st) != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_stage, st);

    quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32, bc.d_ids_src1, bc.d_src1_q8,
        GGML_TYPE_IQ3_XXS, ne00, ne00, n_active * ne00, n_active * ne00,
        ne00_padded, n_active, 1, 1, st);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_quant, st);

    if (!launch_iq3_xxs_mmq_id_batch(
            (const char *)g_bcache.pool, (const int *)bc.d_src1_q8, bc.d_ids_dst, bc.d_bounds,
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
