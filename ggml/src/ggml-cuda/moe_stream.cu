// ik_llama_fork: streaming MoE compute — invoked from ggml.c's CPU mul_mat_id
// path when GGML_MOE_STREAM=1.  Copies one expert's quantized weights H2D into
// a small VRAM staging buffer, runs the existing MMVQ kernel on GPU, copies
// the float32 result D2H, scatters into dst per the row-mapping.
//
// Uses a pool of N (default 8) CUDA streams with per-slot VRAM staging so
// multiple CPU threads can submit GPU work concurrently without a global lock.
//
// Author: Zili Meng <zilim@ieee.org>

#include "common.cuh"
#include "mmvq-args.h"
#include "mmvq-templates.cuh"
#include "quantize.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

// src1_f32: F32 activations (will be Q8_1 quantized on GPU)
//           Layout: cne1 rows × ne00 cols, with stride row_size_f32 bytes per row.
//           If src1_f32 is NULL, falls back to caller-provided src1_q8_1.
bool ggml_cuda_moe_stream_one(
    int  src0_type_int,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    int64_t cne1,
    const void *src1_q8_1,             // optional pre-quantized (unused if src1_f32 set)
    size_t src1_padded_num_cols,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows);

bool ggml_cuda_moe_stream_available(void);
int  ggml_cuda_host_register(void *p, size_t n);
// Sync all outstanding GPU work submitted by ggml_cuda_moe_stream_one and
// perform deferred dst scatter.  Call once at end of a mul_mat_id op (the
// sequential path's thread 0 invokes this after the per-expert loop).
void ggml_cuda_moe_stream_sync(void);

} // extern "C"

// === VRAM Expert Cache (PowerInfer pattern) =================================
//
// A fixed-size VRAM pool that caches recently-used expert weight tensors.
// On cache hit: kernel runs directly from VRAM (zero H2D latency).
// On cache miss: H2D into a cache slot, then kernel.  FIFO eviction.
//
// Budget: GGML_MOE_VRAM_CACHE_GB env var (default 16 GiB).
// Each expert is ~4.6 MiB → ~3500 slots in 16 GiB.

// Hash table for O(1) lookup.  Power-of-2 sized, linear probing.
// Mapping: key (host pointer) → slot index in VRAM pool.
#define VRAM_CACHE_HT_SIZE 16384   // power of 2, ~2× max slots
#define VRAM_CACHE_HT_MASK (VRAM_CACHE_HT_SIZE - 1)

struct vram_ht_entry {
    uintptr_t key;     // 0 = empty
    int       slot;
};

struct vram_cache {
    void *   pool       = nullptr;
    size_t   pool_sz    = 0;
    size_t   slot_sz    = 0;
    int      n_slots    = 0;
    int      next_evict = 0;         // FIFO pointer
    uintptr_t slot_key[16384] = {};  // reverse: slot → key (for eviction)
    uint64_t  slot_used[16384] = {};
    vram_ht_entry ht[VRAM_CACHE_HT_SIZE] = {};
    std::atomic<uint64_t> clock{1};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
};

static vram_cache g_vcache;
static bool       g_vcache_inited = false;

static void vram_cache_report_atexit() {
    uint64_t h = g_vcache.hits.load();
    uint64_t m = g_vcache.misses.load();
    if (h + m == 0) return;
    std::fprintf(stderr, "[moe_stream] VRAM cache: hits=%lu misses=%lu hit_rate=%.1f%%\n",
                 h, m, 100.0 * h / (h + m));
}

static void vram_cache_init(size_t expert_sz) {
    if (g_vcache_inited) return;
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_GB");
    size_t budget_gb = env ? (size_t)std::atoi(env) : 16;
    if (budget_gb == 0) { g_vcache_inited = true; return; }
    size_t budget = budget_gb * 1024ULL * 1024ULL * 1024ULL;
    g_vcache.slot_sz = expert_sz;
    g_vcache.n_slots = (int)(budget / expert_sz);
    if (g_vcache.n_slots > 16384) g_vcache.n_slots = 16384;
    if (g_vcache.n_slots < 1) { g_vcache_inited = true; return; }
    size_t alloc = (size_t)g_vcache.n_slots * expert_sz;
    if (cudaMalloc(&g_vcache.pool, alloc) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] VRAM cache: cudaMalloc %.1f GiB FAILED\n",
                     alloc / (1024.0*1024.0*1024.0));
        g_vcache.n_slots = 0;
        g_vcache_inited = true;
        return;
    }
    g_vcache.pool_sz = alloc;
    std::fprintf(stderr, "[moe_stream] VRAM cache: %.1f GiB, %d slots (%.2f MiB each)\n",
                 alloc / (1024.0*1024.0*1024.0), g_vcache.n_slots,
                 expert_sz / (1024.0*1024.0));
    std::atexit(vram_cache_report_atexit);
    g_vcache_inited = true;
}

// O(1) hash table lookup. Returns VRAM pointer if cached, else nullptr.
static void *vram_cache_lookup(uintptr_t key) {
    if (!g_vcache.pool || g_vcache.n_slots == 0) return nullptr;
    for (int slot = 0; slot < g_vcache.n_slots; ++slot) {
        if (g_vcache.slot_key[slot] == key) {
            g_vcache.slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);
            g_vcache.hits.fetch_add(1, std::memory_order_relaxed);
            return (char *)g_vcache.pool + (size_t)slot * g_vcache.slot_sz;
        }
    }
    return nullptr;
}

// Insert expert into cache. LRU eviction of slot's previous owner.
static void *vram_cache_insert(uintptr_t key, const void *host_data, size_t sz, cudaStream_t st) {
    if (!g_vcache.pool || g_vcache.n_slots == 0 || sz > g_vcache.slot_sz) return nullptr;
    int slot = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < g_vcache.n_slots; ++i) {
        if (g_vcache.slot_key[i] == 0) {
            slot = i;
            break;
        }
        if (g_vcache.slot_used[i] < oldest) {
            oldest = g_vcache.slot_used[i];
            slot = i;
        }
    }
    if (slot < 0) {
        return nullptr;
    }
    g_vcache.slot_key[slot] = key;
    g_vcache.slot_used[slot] = g_vcache.clock.fetch_add(1, std::memory_order_relaxed);

    void *dst = (char *)g_vcache.pool + (size_t)slot * g_vcache.slot_sz;
    cudaMemcpyAsync(dst, host_data, sz, cudaMemcpyHostToDevice, st);
    g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
    return dst;
}

// === Pool of GPU stream slots ==============================================
//
// Each slot has its own stream + VRAM staging buffers.  A CPU thread atomically
// claims a slot via round-robin counter, does its H2D + kernel + D2H + sync on
// that slot, then releases.  Slots are independent — no shared resources.

#define MOE_STREAM_NSLOTS 8

struct deferred_scatter {
    int     slot;
    float * dst_base;
    size_t  dst_nb1;
    size_t  dst_nb2;
    int64_t ne01;
    int64_t cne1;
    int32_t i1[8];
    int32_t i2[8];
};

struct slot_ctx {
    cudaStream_t stream  = nullptr;
    void *  d_src0       = nullptr;
    size_t  d_src0_sz    = 0;
    void *  d_src1       = nullptr;       // Q8_1 quantized src1
    size_t  d_src1_sz    = 0;
    void *  d_src1_f32   = nullptr;       // F32 src1 staging (persistent)
    size_t  d_src1_f32_sz = 0;
    void *  d_dst        = nullptr;
    size_t  d_dst_sz     = 0;
    void *  h_scratch    = nullptr;
    size_t  h_scratch_sz = 0;
    void *  h_bounce     = nullptr;
    size_t  h_bounce_sz  = 0;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

// One queue of deferred scatters per CPU thread (thread_local).
static thread_local std::vector<deferred_scatter> tls_pending;

static slot_ctx          g_slots[MOE_STREAM_NSLOTS];
static std::atomic<int>  g_slot_rr{0};
static std::atomic<bool> g_avail{false};
static std::atomic<bool> g_inited{false};
static bool              g_defer_sync = false;
static std::mutex        g_init_mu;
// Slot resize is rare (only when needed > current); guard with one mutex
// rather than per-slot.  Compute path doesn't need it.
static std::mutex        g_resize_mu;

static bool ensure_dev(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFree(p);
    p = nullptr; cur = 0;
    if (cudaMalloc(&p, need) != cudaSuccess) return false;
    cur = need;
    return true;
}

static bool ensure_host_pinned(void *&p, size_t &cur, size_t need) {
    if (cur >= need) return true;
    if (p) cudaFreeHost(p);
    p = nullptr; cur = 0;
    if (cudaHostAlloc(&p, need, cudaHostAllocDefault) != cudaSuccess) return false;
    cur = need;
    return true;
}

static void init_once() {
    if (g_inited.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (g_inited.load(std::memory_order_acquire)) return;

    const char *env = std::getenv("GGML_MOE_STREAM");
    bool enabled = env && env[0] && env[0] != '0';
    const char *defer_env = std::getenv("GGML_MOE_STREAM_DEFER");
    g_defer_sync = defer_env && defer_env[0] && defer_env[0] != '0';
    if (enabled && cudaSetDevice(0) == cudaSuccess) {
        bool ok = true;
        for (int i = 0; i < MOE_STREAM_NSLOTS; ++i) {
            if (cudaStreamCreate(&g_slots[i].stream) != cudaSuccess) { ok = false; break; }
        }
        if (ok) {
            g_avail.store(true, std::memory_order_release);
            std::fprintf(stderr, "[moe_stream] enabled (%d streams, GPU expert compute, defer_sync=%d)\n",
                         MOE_STREAM_NSLOTS, g_defer_sync ? 1 : 0);
        }
    }
    g_inited.store(true, std::memory_order_release);
}

extern "C" bool ggml_cuda_moe_stream_available(void) {
    init_once();
    return g_avail.load(std::memory_order_acquire);
}

extern "C" int ggml_cuda_host_register(void *p, size_t n) {
    if (!p || n == 0) return 0;
    cudaError_t err = cudaHostRegister(p, n, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] cudaHostRegister %.1f GiB OK\n",
                     n / (1024.0*1024.0*1024.0));
        return 1;
    }
    std::fprintf(stderr, "[moe_stream] cudaHostRegister %.1f GiB FAILED: %s\n",
                 n / (1024.0*1024.0*1024.0), cudaGetErrorString(err));
    return 0;
}

// Acquire a slot via test-and-set on per-slot flag; spin briefly if all busy.
static int acquire_slot() {
    for (int attempt = 0; attempt < 1024; ++attempt) {
        int s = g_slot_rr.fetch_add(1, std::memory_order_relaxed) & (MOE_STREAM_NSLOTS - 1);
        if (!g_slots[s].in_use.test_and_set(std::memory_order_acquire)) {
            return s;
        }
    }
    // Fallback: hard-spin on slot 0
    while (g_slots[0].in_use.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    return 0;
}

static void release_slot(int s) {
    g_slots[s].in_use.clear(std::memory_order_release);
}

extern "C" bool ggml_cuda_moe_stream_one(
    int  src0_type_int,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    int64_t cne1,
    const void *src1_q8_1,
    size_t src1_padded_num_cols,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows)
{
    init_once();
    if (!g_avail.load(std::memory_order_acquire)) return false;

    const ggml_type t0 = (ggml_type)src0_type_int;
    if (t0 != GGML_TYPE_IQ3_XXS) return false;
    if (cne1 < 1 || cne1 > MMVQ_MAX_BATCH_SIZE) return false;
    if (!src1_f32) return false;   // require F32 src1 for on-GPU quantization

    static std::atomic<int> first_call{0};
    if (first_call.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] first call: ne01=%ld ne00=%ld nb01=%zu src0_bytes=%zu cne1=%ld\n",
                     (long)ne01, (long)ne00, nb01, (size_t)ne01 * nb01, (long)cne1);
        std::fflush(stderr);
    }

    const size_t src0_bytes      = (size_t)ne01 * nb01;
    const size_t src1_f32_bytes  = (size_t)cne1 * ne00 * sizeof(float);
    const int64_t src1_padded    = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_bytes = (size_t)cne1 * (src1_padded / QK8_1) * sizeof(block_q8_1);
    const size_t dst_bytes       = (size_t)ne01 * cne1 * sizeof(float);

    int s = acquire_slot();
    slot_ctx &ctx = g_slots[s];

    // d_src1_f32 = staging for F32 src1 (sized once); d_src1 = Q8_1 quantized.
    bool ok = true;
    {
        std::lock_guard<std::mutex> rk(g_resize_mu);
        ok = ensure_dev(ctx.d_src0, ctx.d_src0_sz, src0_bytes)
          && ensure_dev(ctx.d_src1, ctx.d_src1_sz, src1_q8_1_bytes)
          && ensure_dev(ctx.d_dst,  ctx.d_dst_sz,  dst_bytes)
          && ensure_host_pinned(ctx.h_scratch, ctx.h_scratch_sz, dst_bytes);
    }
    if (!ok) { release_slot(s); return false; }

    cudaStream_t st = ctx.stream;

    // Initialize VRAM cache on first call (needs to know expert size)
    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lk(g_init_mu);
        if (!g_vcache_inited) vram_cache_init(src0_bytes);
    }

    // VRAM cache lookup: if this expert is already in VRAM, skip H2D entirely.
    uintptr_t cache_key = (uintptr_t)src0_data;
    void *cached_vram = vram_cache_lookup(cache_key);
    const void *kernel_src0 = nullptr;

    if (cached_vram) {
        kernel_src0 = cached_vram;
    } else {
        void *inserted = vram_cache_insert(cache_key, src0_data, src0_bytes, st);
        if (inserted) {
            kernel_src0 = inserted;
        } else {
            if (cudaMemcpyAsync(ctx.d_src0, src0_data, src0_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) { release_slot(s); return false; }
            kernel_src0 = ctx.d_src0;
        }
    }

    // src1 (F32) → persistent VRAM staging → quantize to Q8_1 on GPU
    {
        std::lock_guard<std::mutex> rk(g_resize_mu);
        if (!ensure_dev(ctx.d_src1_f32, ctx.d_src1_f32_sz, src1_f32_bytes)) {
            release_slot(s); return false;
        }
        if (!ctx.h_bounce || ctx.h_bounce_sz < src1_f32_bytes) {
            ensure_host_pinned(ctx.h_bounce, ctx.h_bounce_sz, src1_f32_bytes);
        }
    }
    if (ctx.h_bounce && ctx.h_bounce_sz >= src1_f32_bytes) {
        char *bounce = (char *)ctx.h_bounce;
        const char *src1_base = (const char *)src1_f32;
        for (int64_t k = 0; k < cne1; ++k) {
            const int32_t i1 = rows[k].i1;
            const int32_t i2 = rows[k].i2;
            const char *src_row = src1_base + (size_t)i1 * src1_nb1 + (size_t)i2 * src1_nb2;
            std::memcpy(bounce + (size_t)k * ne00 * sizeof(float), src_row, (size_t)ne00 * sizeof(float));
        }
        cudaMemcpyAsync(ctx.d_src1_f32, ctx.h_bounce, src1_f32_bytes, cudaMemcpyHostToDevice, st);
    } else {
        release_slot(s);
        return false;
    }
    quantize_row_q8_1_cuda((const float *)ctx.d_src1_f32, ctx.d_src1, ne00, cne1, 1, src1_padded, t0, st);

    mmvq_args args{
        /* vx_u     */ kernel_src0,
        /* vx_g     */ nullptr,
        /* bias_u   */ nullptr,
        /* bias_g   */ nullptr,
        /* vy       */ ctx.d_src1,
        /* dst      */ (float *)ctx.d_dst,
        /* ids_data */ nullptr,
        /* ncols_x  */ (int)ne00,
        /* nrows_x  */ (int)ne01,
        /* nrows_y  */ (int)src1_padded,
        /* ncols_y  */ (int)cne1,
        /* nrows_dst*/ (int)ne01,
        /* ne2      */ 1,
        /* nb02     */ (uint64_t)0,
        /* nb12     */ (uint64_t)0,
        /* nb2      */ (uint64_t)0,
        /* ids_nb0  */ (uint64_t)0,
        /* bias_nb1 */ (uint64_t)0,
        /* unary_op */ GGML_UNARY_OP_COUNT,
        /* limit    */ INFINITY,
    };

    extern void mul_mat_vec_iq3_xxs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
    mul_mat_vec_iq3_xxs_q8_1_cuda(args, st);

    if (cudaMemcpyAsync(ctx.h_scratch, ctx.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) { release_slot(s); return false; }

    if (!g_defer_sync) {
        if (cudaStreamSynchronize(st) != cudaSuccess) { release_slot(s); return false; }
        const float *src_buf = (const float *)ctx.h_scratch;
        for (int64_t k = 0; k < cne1; ++k) {
            const int32_t i1 = rows[k].i1;
            const int32_t i2 = rows[k].i2;
            float *dst_row = (float *)((char *)dst + i1 * dst_nb1 + i2 * dst_nb2);
            const float *src_row = src_buf + k * ne01;
            std::memcpy(dst_row, src_row, (size_t)ne01 * sizeof(float));
        }
        release_slot(s);
        return true;
    }

    // Defer scatter until ggml_cuda_moe_stream_sync().  Slot remains "in_use"
    // until then so its h_scratch isn't trampled.
    deferred_scatter ds;
    ds.slot     = s;
    ds.dst_base = dst;
    ds.dst_nb1  = dst_nb1;
    ds.dst_nb2  = dst_nb2;
    ds.ne01     = ne01;
    ds.cne1     = cne1;
    for (int64_t k = 0; k < cne1 && k < 8; ++k) {
        ds.i1[k] = rows[k].i1;
        ds.i2[k] = rows[k].i2;
    }
    tls_pending.push_back(ds);
    return true;
}

extern "C" void ggml_cuda_moe_stream_sync(void) {
    if (tls_pending.empty()) return;
    // Sync each used slot and scatter
    for (auto &ds : tls_pending) {
        slot_ctx &ctx = g_slots[ds.slot];
        cudaStreamSynchronize(ctx.stream);
        const float *src_buf = (const float *)ctx.h_scratch;
        for (int64_t k = 0; k < ds.cne1; ++k) {
            const int32_t i1 = ds.i1[k];
            const int32_t i2 = ds.i2[k];
            float *dst_row = (float *)((char *)ds.dst_base + i1 * ds.dst_nb1 + i2 * ds.dst_nb2);
            const float *src_row = src_buf + k * ds.ne01;
            std::memcpy(dst_row, src_row, ds.ne01 * sizeof(float));
        }
        release_slot(ds.slot);
    }
    tls_pending.clear();
}
