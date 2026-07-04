
#ifndef GGML_CUDA_MOE_STREAM
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
extern "C" {
typedef struct { int32_t i1; int32_t i2; } ggml_moe_row_mapping;
void ggml_cuda_moe_stream_link_anchor(void) {}
bool ggml_cuda_moe_stream_available(void) { return false; }
int ggml_cuda_host_register(void *, size_t) { return 0; }
bool ggml_cuda_moe_stream_one(int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const float *, size_t, size_t, int64_t, int64_t, const void *, size_t, float *, size_t, size_t, const ggml_moe_row_mapping *) { return false; }
bool ggml_cuda_moe_stream_mmvq_dev(int, const void *, int64_t, int64_t, size_t, const float *, void *, float *, cudaStream_t) { return false; }
bool ggml_cuda_moe_stream_mmvq_rows_dev(int, const void *, int64_t, int64_t, size_t, const float *, void *, const int32_t *, int64_t, float *, cudaStream_t) { return false; }
bool ggml_cuda_moe_stream_mmvq_batch_dev(int, const void *, int64_t, int64_t, const float *, void *, float *, const int32_t *, int64_t, int64_t, cudaStream_t) { return false; }
void ggml_cuda_moe_stream_sync(void) {}
}
#else
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
#include "mmvq.cuh"
#include "quantize.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

void ggml_cuda_moe_stream_link_anchor(void) {}

// src1_f32: F32 activations (will be Q8_1 quantized on GPU)
//           Layout: cne1 rows × ne00 cols, with stride row_size_f32 bytes per row.
//           If src1_f32 is NULL, falls back to caller-provided src1_q8_1.
bool ggml_cuda_moe_stream_one(
    int  src0_type_int,
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    int64_t src1_ne1,
    int64_t cne1,
    const void *src1_q8_1,             // optional pre-quantized (unused if src1_f32 set)
    size_t src1_padded_num_cols,
    float *dst,
    size_t dst_nb1, size_t dst_nb2,
    const ggml_moe_row_mapping *rows);

bool ggml_cuda_moe_stream_available(void);
int  ggml_cuda_host_register(void *p, size_t n);
bool ggml_cuda_moe_stream_mmvq_dev(
    int src0_type_int,
    const void *d_src0,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *d_src1_f32,
    void *d_src1_q8,
    float *d_dst,
    cudaStream_t stream);
bool ggml_cuda_moe_stream_mmvq_rows_dev(
    int src0_type_int,
    const void *d_src0,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *d_src1_f32,
    void *d_src1_q8,
    const int32_t *d_x_ids,
    int64_t cne1,
    float *d_dst,
    cudaStream_t stream);
bool ggml_cuda_moe_stream_mmvq_batch_dev(
    int src0_type_int,
    const void *d_src0,
    int64_t ne01,
    int64_t ne00,
    const float *d_src1_f32,
    void *d_src1_q8,
    float *d_dst,
    const int32_t *d_x_ids,
    int64_t n_active,
    int64_t src0_stride,
    cudaStream_t stream);
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
// Budget: GGML_MOE_STREAM_ONE_CACHE_MIB / GB, or the shared GGML_MOE_VRAM_CACHE_GB.
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

static bool moe_stream_dontneed_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_DONTNEED");
        return env && env[0] && env[0] != '0';
    }();
    return enabled != 0;
}

static void moe_stream_dontneed_source_pages(const void * ptr, size_t size) {
#if defined(__linux__)
    if (!moe_stream_dontneed_enabled() || ptr == nullptr || size == 0) {
        return;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }

    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = begin + size;
    const uintptr_t aligned_begin = begin & ~static_cast<uintptr_t>(page_size - 1);
    const uintptr_t aligned_end = (end + static_cast<uintptr_t>(page_size - 1)) & ~static_cast<uintptr_t>(page_size - 1);
    if (aligned_end <= aligned_begin) {
        return;
    }

    (void) madvise(reinterpret_cast<void *>(aligned_begin), aligned_end - aligned_begin, MADV_DONTNEED);
#else
    (void) ptr;
    (void) size;
#endif
}

static bool moe_stream_cache_admit_allows(const char * tensor, int64_t expert) {
    static std::mutex mu;
    static bool initialized = false;
    static bool enabled = false;
    static std::unordered_set<std::string> allow;

    if (!initialized) {
        std::lock_guard<std::mutex> lk(mu);
        if (!initialized) {
            const char * path = std::getenv("GGML_MOE_STREAM_CACHE_ADMIT_PROFILE");
            if (path && path[0]) {
                FILE * fp = std::fopen(path, "r");
                if (!fp) {
                    std::fprintf(stderr, "[moe_stream] cache admission: failed to open %s\n", path);
                } else {
                    char line[4096];
                    while (std::fgets(line, sizeof(line), fp)) {
                        char name[3072];
                        long long expert_id = -1;
                        if (std::sscanf(line, "%3071[^\t]\t%lld", name, &expert_id) == 2 && expert_id >= 0) {
                            allow.insert(std::string(name) + "\t" + std::to_string(expert_id));
                        }
                    }
                    std::fclose(fp);
                    enabled = true;
                    std::fprintf(stderr, "[moe_stream] cache admission: loaded %zu entries from %s\n",
                            allow.size(), path);
                }
            }
            initialized = true;
        }
    }

    if (!enabled) {
        return true;
    }
    if (!tensor) {
        return false;
    }
    const std::string key = std::string(tensor) + "\t" + std::to_string((long long) expert);
    return allow.find(key) != allow.end();
}

static void vram_cache_init(size_t expert_sz) {
    if (g_vcache_inited) return;
    const char *env_one_mib = std::getenv("GGML_MOE_STREAM_ONE_CACHE_MIB");
    const char *env_one_gb = std::getenv("GGML_MOE_STREAM_ONE_CACHE_GB");
    const char *env_shared_mib = std::getenv("GGML_MOE_VRAM_CACHE_MIB");
    const char *env_shared_gb = std::getenv("GGML_MOE_VRAM_CACHE_GB");
    size_t budget_mib = 16ULL * 1024ULL;
    if (env_one_mib && env_one_mib[0]) {
        budget_mib = (size_t)std::strtoull(env_one_mib, nullptr, 10);
    } else if (env_one_gb && env_one_gb[0]) {
        budget_mib = (size_t)std::strtoull(env_one_gb, nullptr, 10) * 1024ULL;
    } else if (env_shared_mib && env_shared_mib[0]) {
        budget_mib = 0;
    } else if (env_shared_gb && env_shared_gb[0]) {
        budget_mib = (size_t)std::strtoull(env_shared_gb, nullptr, 10) * 1024ULL;
    }
    if (budget_mib == 0) { g_vcache_inited = true; return; }
    size_t budget = budget_mib * 1024ULL * 1024ULL;
    g_vcache.slot_sz = expert_sz;
    g_vcache.n_slots = (int)(budget / expert_sz);
    if (g_vcache.n_slots > 16384) g_vcache.n_slots = 16384;
    if (g_vcache.n_slots < 1) { g_vcache_inited = true; return; }
    size_t alloc = (size_t)g_vcache.n_slots * expert_sz;
    if (cudaMalloc(&g_vcache.pool, alloc) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream] VRAM cache: cudaMalloc %.1f GiB FAILED\n",
                     alloc / (1024.0*1024.0*1024.0));
        cudaGetLastError();
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
static void *vram_cache_insert_impl(uintptr_t key, const void *host_data, size_t sz, cudaStream_t st, bool count_miss) {
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
    if (count_miss) {
        g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
    }
    return dst;
}

static void *vram_cache_insert(uintptr_t key, const void *host_data, size_t sz, cudaStream_t st) {
    return vram_cache_insert_impl(key, host_data, sz, st, true);
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
    void *  d_ids        = nullptr;
    size_t  d_ids_sz     = 0;
    void *  h_scratch    = nullptr;
    size_t  h_scratch_sz = 0;
    void *  h_bounce     = nullptr;
    size_t  h_bounce_sz  = 0;
    void *  h_src0_pack  = nullptr;
    size_t  h_src0_pack_sz = 0;
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


struct one_expert_pack_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct one_expert_pack_state {
    int fd = -1;
    int fd_direct = -1;
    bool direct_enabled = false;
    std::vector<one_expert_pack_entry> entries;
    bool inited = false;
    bool enabled = false;
    std::mutex mu;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_failures{0};
    std::atomic<uint64_t> direct_fallbacks{0};
};

static one_expert_pack_state g_one_pack;

static bool one_pack_read_exact_fd(int fd, void * dst, size_t sz, uint64_t off) {
    char * out = (char *) dst;
    size_t done = 0;
    while (done < sz) {
        const ssize_t got = ::pread(fd, out + done, sz - done, (off_t)(off + done));
        if (got <= 0) {
            return false;
        }
        done += (size_t) got;
    }
    return true;
}

static void one_pack_report_atexit() {
    if (!g_one_pack.enabled) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream] one expert pack: hits=%lu misses=%lu reads=%lu bytes=%lu failures=%lu entries=%zu"
        " direct_enabled=%d direct_reads=%lu direct_failures=%lu direct_fallbacks=%lu\n",
        g_one_pack.hits.load(), g_one_pack.misses.load(), g_one_pack.reads.load(),
        g_one_pack.bytes.load(), g_one_pack.failures.load(), g_one_pack.entries.size(),
        g_one_pack.direct_enabled ? 1 : 0,
        g_one_pack.direct_reads.load(), g_one_pack.direct_failures.load(),
        g_one_pack.direct_fallbacks.load());
}

static void one_pack_init_once() {
    std::lock_guard<std::mutex> lk(g_one_pack.mu);
    if (g_one_pack.inited) {
        return;
    }
    const char * path = std::getenv("GGML_MOE_STREAM_ONE_EXPERT_PACK");
    if (!path || !path[0]) {
        g_one_pack.inited = true;
        return;
    }
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[moe_stream] one expert pack: open failed: %s\n", path);
        g_one_pack.inited = true;
        return;
    }

    char magic[16] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t n_entries = 0;
    uint64_t data_start = 0;
    uint64_t off = 0;
    if (!one_pack_read_exact_fd(fd, magic, sizeof(magic), off)) goto invalid_header;
    off += sizeof(magic);
    if (!one_pack_read_exact_fd(fd, &version, sizeof(version), off)) goto invalid_header;
    off += sizeof(version);
    if (!one_pack_read_exact_fd(fd, &header_size, sizeof(header_size), off)) goto invalid_header;
    off += sizeof(header_size);
    if (!one_pack_read_exact_fd(fd, &n_entries, sizeof(n_entries), off)) goto invalid_header;
    off += sizeof(n_entries);
    if (!one_pack_read_exact_fd(fd, &data_start, sizeof(data_start), off)) goto invalid_header;
    if (std::memcmp(magic, "GGMLMOEPACKv1", 13) != 0 ||
            version != 1 || header_size < 40 || data_start < header_size || n_entries > 10000000ULL) {
        goto invalid_header;
    }

    {
        std::vector<one_expert_pack_entry> entries;
        entries.resize((size_t)n_entries);
        uint64_t index_off = header_size;
        for (uint64_t i = 0; i < n_entries; ++i) {
            uint32_t reserved = 0;
            one_expert_pack_entry & e = entries[(size_t)i];
            if (!one_pack_read_exact_fd(fd, e.tensor, sizeof(e.tensor), index_off)) goto invalid_index;
            index_off += sizeof(e.tensor);
            if (!one_pack_read_exact_fd(fd, &e.expert_idx, sizeof(e.expert_idx), index_off)) goto invalid_index;
            index_off += sizeof(e.expert_idx);
            if (!one_pack_read_exact_fd(fd, &reserved, sizeof(reserved), index_off)) goto invalid_index;
            index_off += sizeof(reserved);
            if (!one_pack_read_exact_fd(fd, &e.offset, sizeof(e.offset), index_off)) goto invalid_index;
            index_off += sizeof(e.offset);
            if (!one_pack_read_exact_fd(fd, &e.nbytes, sizeof(e.nbytes), index_off)) goto invalid_index;
            index_off += sizeof(e.nbytes);
            e.tensor[sizeof(e.tensor) - 1] = '\0';
        }
        std::sort(entries.begin(), entries.end(), [](const one_expert_pack_entry & a, const one_expert_pack_entry & b) {
            const int name_cmp = std::strcmp(a.tensor, b.tensor);
            if (name_cmp != 0) {
                return name_cmp < 0;
            }
            if (a.expert_idx != b.expert_idx) {
                return a.expert_idx < b.expert_idx;
            }
            return a.nbytes < b.nbytes;
        });
        g_one_pack.fd = fd;
        const char * io_env = std::getenv("GGML_MOE_STREAM_ONE_EXPERT_PACK_IO");
        if (io_env && (std::strcmp(io_env, "direct") == 0 || std::strcmp(io_env, "odirect") == 0)) {
#if defined(__linux__) && defined(O_DIRECT)
            g_one_pack.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
            if (g_one_pack.fd_direct >= 0) {
                g_one_pack.direct_enabled = true;
                std::fprintf(stderr, "[moe_stream] one expert pack: O_DIRECT payload reads enabled: %s\n", path);
            } else {
                std::fprintf(stderr, "[moe_stream] one expert pack: O_DIRECT open failed; using buffered payload reads: %s\n", path);
            }
#else
            std::fprintf(stderr, "[moe_stream] one expert pack: O_DIRECT requested but unavailable; using buffered payload reads\n");
#endif
        }
        g_one_pack.entries = std::move(entries);
        g_one_pack.enabled = true;
        g_one_pack.inited = true;
        std::atexit(one_pack_report_atexit);
        std::fprintf(stderr, "[moe_stream] one expert pack: loaded %zu entries from %s\n", g_one_pack.entries.size(), path);
        return;
    }

invalid_index:
    std::fprintf(stderr, "[moe_stream] one expert pack: short index: %s\n", path);
    ::close(fd);
    g_one_pack.inited = true;
    return;
invalid_header:
    std::fprintf(stderr, "[moe_stream] one expert pack: invalid header: %s\n", path);
    ::close(fd);
    g_one_pack.inited = true;
}

static const one_expert_pack_entry * one_pack_lookup(const char * tensor_name, int64_t expert_idx, size_t nbytes) {
    one_pack_init_once();
    if (!g_one_pack.enabled || !tensor_name || !tensor_name[0]) {
        return nullptr;
    }
    size_t lo = 0;
    size_t hi = g_one_pack.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const one_expert_pack_entry & e = g_one_pack.entries[mid];
        int cmp = std::strcmp(e.tensor, tensor_name);
        if (cmp == 0) {
            if (e.expert_idx < expert_idx) cmp = -1;
            else if (e.expert_idx > expert_idx) cmp = 1;
            else if (e.nbytes < nbytes) cmp = -1;
            else if (e.nbytes > nbytes) cmp = 1;
        }
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_one_pack.entries.size()) {
        const one_expert_pack_entry & e = g_one_pack.entries[lo];
        if (std::strcmp(e.tensor, tensor_name) == 0 && e.expert_idx == expert_idx && e.nbytes == nbytes) {
            ++g_one_pack.hits;
            return &e;
        }
    }
    ++g_one_pack.misses;
    return nullptr;
}

static bool one_pack_read_entry(const one_expert_pack_entry * entry, void * dst, size_t sz) {
    if (!entry || g_one_pack.fd < 0 || entry->nbytes != sz) {
        return false;
    }
    if (g_one_pack.fd_direct >= 0) {
        if (one_pack_read_exact_fd(g_one_pack.fd_direct, dst, sz, entry->offset)) {
            ++g_one_pack.reads;
            ++g_one_pack.direct_reads;
            g_one_pack.bytes.fetch_add(sz);
            return true;
        }
        ++g_one_pack.direct_failures;
        ++g_one_pack.direct_fallbacks;
    }
    if (!one_pack_read_exact_fd(g_one_pack.fd, dst, sz, entry->offset)) {
        ++g_one_pack.failures;
        return false;
    }
    ++g_one_pack.reads;
    g_one_pack.bytes.fetch_add(sz);
    return true;
}

struct one_direct_manifest_state {
    int fd = -1;
    int fd_direct = -1;
    bool direct_enabled = false;
    bool inited = false;
    bool enabled = false;
    std::vector<one_expert_pack_entry> entries;
    std::mutex mu;
    uint64_t manifest_bytes = 0;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_failures{0};
    std::atomic<uint64_t> direct_fallbacks{0};
};

static one_direct_manifest_state g_one_direct_manifest;

static void one_direct_manifest_report_atexit() {
    if (!g_one_direct_manifest.enabled) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream] one direct manifest: entries=%zu manifest_bytes=%lu hits=%lu misses=%lu"
        " reads=%lu bytes=%lu failures=%lu direct_enabled=%d direct_reads=%lu"
        " direct_failures=%lu direct_fallbacks=%lu\n",
        g_one_direct_manifest.entries.size(),
        g_one_direct_manifest.manifest_bytes,
        g_one_direct_manifest.hits.load(),
        g_one_direct_manifest.misses.load(),
        g_one_direct_manifest.reads.load(),
        g_one_direct_manifest.bytes.load(),
        g_one_direct_manifest.failures.load(),
        g_one_direct_manifest.direct_enabled ? 1 : 0,
        g_one_direct_manifest.direct_reads.load(),
        g_one_direct_manifest.direct_failures.load(),
        g_one_direct_manifest.direct_fallbacks.load());
}

static bool one_direct_manifest_parse_row(char * line, one_expert_pack_entry & entry) {
    char tensor[128] = {};
    long long expert = -1;
    unsigned long long offset = 0;
    unsigned long long nbytes = 0;
    const int n = std::sscanf(line, "%127[^,],%lld,%llu,%llu", tensor, &expert, &offset, &nbytes);
    if (n != 4 || expert < 0 || nbytes == 0) {
        return false;
    }
    entry = one_expert_pack_entry{};
    std::snprintf(entry.tensor, sizeof(entry.tensor), "%s", tensor);
    entry.expert_idx = (int32_t) expert;
    entry.offset = (uint64_t) offset;
    entry.nbytes = (uint64_t) nbytes;
    return true;
}

static void one_direct_manifest_init_once() {
    std::lock_guard<std::mutex> lk(g_one_direct_manifest.mu);
    if (g_one_direct_manifest.inited) {
        return;
    }
    g_one_direct_manifest.inited = true;

    const char * manifest_path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_MANIFEST");
    if (!manifest_path || !manifest_path[0]) {
        return;
    }
    const char * model_path = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_MODEL");
    if (!model_path || !model_path[0]) {
        std::fprintf(stderr,
            "[moe_stream] one direct manifest: GGML_MOE_STREAM_ONE_DIRECT_MODEL is required when manifest is set\n");
        return;
    }

    FILE * fp = std::fopen(manifest_path, "r");
    if (!fp) {
        std::fprintf(stderr, "[moe_stream] one direct manifest: failed to open manifest %s\n", manifest_path);
        return;
    }

    std::vector<one_expert_pack_entry> entries;
    uint64_t total_bytes = 0;
    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        one_expert_pack_entry entry;
        if (!one_direct_manifest_parse_row(line, entry)) {
            continue;
        }
        total_bytes += entry.nbytes;
        entries.push_back(entry);
    }
    std::fclose(fp);

    if (entries.empty()) {
        std::fprintf(stderr, "[moe_stream] one direct manifest: no entries loaded from %s\n", manifest_path);
        return;
    }

    std::sort(entries.begin(), entries.end(), [](const one_expert_pack_entry & a, const one_expert_pack_entry & b) {
        const int name_cmp = std::strcmp(a.tensor, b.tensor);
        if (name_cmp != 0) {
            return name_cmp < 0;
        }
        if (a.expert_idx != b.expert_idx) {
            return a.expert_idx < b.expert_idx;
        }
        return a.nbytes < b.nbytes;
    });

    const int fd = ::open(model_path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[moe_stream] one direct manifest: model open failed: %s\n", model_path);
        return;
    }

    g_one_direct_manifest.fd = fd;
    const char * io_env = std::getenv("GGML_MOE_STREAM_ONE_DIRECT_IO");
    if (io_env && (std::strcmp(io_env, "direct") == 0 || std::strcmp(io_env, "odirect") == 0)) {
#if defined(__linux__) && defined(O_DIRECT)
        g_one_direct_manifest.fd_direct = ::open(model_path, O_RDONLY | O_DIRECT);
        if (g_one_direct_manifest.fd_direct >= 0) {
            g_one_direct_manifest.direct_enabled = true;
            std::fprintf(stderr, "[moe_stream] one direct manifest: O_DIRECT model reads enabled: %s\n", model_path);
        } else {
            std::fprintf(stderr, "[moe_stream] one direct manifest: O_DIRECT open failed; using buffered model reads: %s\n", model_path);
        }
#else
        std::fprintf(stderr, "[moe_stream] one direct manifest: O_DIRECT requested but unavailable; using buffered model reads\n");
#endif
    }

    g_one_direct_manifest.entries = std::move(entries);
    g_one_direct_manifest.manifest_bytes = total_bytes;
    g_one_direct_manifest.enabled = true;
    std::atexit(one_direct_manifest_report_atexit);
    std::fprintf(stderr, "[moe_stream] one direct manifest: loaded %zu entries bytes=%lu from %s\n",
            g_one_direct_manifest.entries.size(), g_one_direct_manifest.manifest_bytes, manifest_path);
}

[[maybe_unused]] static const one_expert_pack_entry * one_direct_manifest_lookup(const char * tensor_name, int64_t expert_idx, size_t nbytes) {
    one_direct_manifest_init_once();
    if (!g_one_direct_manifest.enabled || !tensor_name || !tensor_name[0]) {
        return nullptr;
    }
    size_t lo = 0;
    size_t hi = g_one_direct_manifest.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const one_expert_pack_entry & e = g_one_direct_manifest.entries[mid];
        int cmp = std::strcmp(e.tensor, tensor_name);
        if (cmp == 0) {
            if (e.expert_idx < expert_idx) cmp = -1;
            else if (e.expert_idx > expert_idx) cmp = 1;
            else if (e.nbytes < nbytes) cmp = -1;
            else if (e.nbytes > nbytes) cmp = 1;
        }
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_one_direct_manifest.entries.size()) {
        const one_expert_pack_entry & e = g_one_direct_manifest.entries[lo];
        if (std::strcmp(e.tensor, tensor_name) == 0 && e.expert_idx == expert_idx && e.nbytes == nbytes) {
            ++g_one_direct_manifest.hits;
            return &e;
        }
    }
    ++g_one_direct_manifest.misses;
    return nullptr;
}

[[maybe_unused]] static bool one_direct_manifest_read_entry(const one_expert_pack_entry * entry, void * dst, size_t sz) {
    if (!entry || g_one_direct_manifest.fd < 0 || entry->nbytes != sz) {
        return false;
    }
    if (g_one_direct_manifest.fd_direct >= 0) {
        if (one_pack_read_exact_fd(g_one_direct_manifest.fd_direct, dst, sz, entry->offset)) {
            ++g_one_direct_manifest.reads;
            ++g_one_direct_manifest.direct_reads;
            g_one_direct_manifest.bytes.fetch_add(sz);
            return true;
        }
        ++g_one_direct_manifest.direct_failures;
        ++g_one_direct_manifest.direct_fallbacks;
    }
    if (!one_pack_read_exact_fd(g_one_direct_manifest.fd, dst, sz, entry->offset)) {
        ++g_one_direct_manifest.failures;
        return false;
    }
    ++g_one_direct_manifest.reads;
    g_one_direct_manifest.bytes.fetch_add(sz);
    return true;
}

static bool ensure_host_pinned(void *&p, size_t &cur, size_t need);

struct one_prefill_entry {
    std::string tensor;
    int64_t expert = -1;
    int64_t score = 0;
};

struct one_prefill_state {
    bool initialized = false;
    bool enabled = false;
    bool registered = false;
    bool done = false;
    size_t limit = 0;
    std::vector<one_prefill_entry> entries;
    std::mutex mu;
    uint64_t attempted = 0;
    uint64_t inserted = 0;
    uint64_t pack_misses = 0;
    uint64_t read_failures = 0;
    uint64_t bytes = 0;
    double elapsed_ms = 0.0;
};

static one_prefill_state g_one_prefill;

static void one_prefill_report_atexit() {
    if (!g_one_prefill.enabled && g_one_prefill.attempted == 0) {
        return;
    }
    std::fprintf(stderr,
        "[moe_stream] one prefill: enabled=%d loaded=%zu limit=%zu attempted=%lu inserted=%lu"
        " pack_misses=%lu read_failures=%lu bytes=%lu elapsed_ms=%.3f\n",
        g_one_prefill.enabled ? 1 : 0,
        g_one_prefill.entries.size(),
        g_one_prefill.limit,
        g_one_prefill.attempted,
        g_one_prefill.inserted,
        g_one_prefill.pack_misses,
        g_one_prefill.read_failures,
        g_one_prefill.bytes,
        g_one_prefill.elapsed_ms);
}

static void one_prefill_init_once_locked() {
    if (g_one_prefill.initialized) {
        return;
    }
    g_one_prefill.initialized = true;

    const char * path = std::getenv("GGML_MOE_STREAM_ONE_PREFILL_PROFILE");
    if (!path || !path[0]) {
        return;
    }

    const char * limit_env = std::getenv("GGML_MOE_STREAM_ONE_PREFILL_LIMIT");
    g_one_prefill.limit = (limit_env && limit_env[0]) ? (size_t) std::strtoull(limit_env, nullptr, 10) : 0;

    FILE * fp = std::fopen(path, "r");
    if (!fp) {
        std::fprintf(stderr, "[moe_stream] one prefill: failed to open profile %s\n", path);
        return;
    }

    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        char tensor[3072];
        long long expert = -1;
        long long score = 0;
        const int n = std::sscanf(line, "%3071[^\t]\t%lld\t%lld", tensor, &expert, &score);
        if (n >= 2 && expert >= 0) {
            one_prefill_entry e;
            e.tensor = tensor;
            e.expert = expert;
            e.score = (n >= 3) ? score : 0;
            g_one_prefill.entries.push_back(std::move(e));
        }
    }
    std::fclose(fp);

    std::sort(g_one_prefill.entries.begin(), g_one_prefill.entries.end(),
        [](const one_prefill_entry & a, const one_prefill_entry & b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            const int name_cmp = std::strcmp(a.tensor.c_str(), b.tensor.c_str());
            if (name_cmp != 0) {
                return name_cmp < 0;
            }
            return a.expert < b.expert;
        });

    if (g_one_prefill.limit == 0 || g_one_prefill.limit > g_one_prefill.entries.size()) {
        g_one_prefill.limit = g_one_prefill.entries.size();
    }
    g_one_prefill.enabled = !g_one_prefill.entries.empty();
    if (g_one_prefill.enabled && !g_one_prefill.registered) {
        g_one_prefill.registered = true;
        std::atexit(one_prefill_report_atexit);
    }
    std::fprintf(stderr, "[moe_stream] one prefill: loaded %zu entries from %s limit=%zu\n",
            g_one_prefill.entries.size(), path, g_one_prefill.limit);
}

static bool one_prefill_enabled() {
    std::lock_guard<std::mutex> lk(g_one_prefill.mu);
    one_prefill_init_once_locked();
    return g_one_prefill.enabled;
}

static uintptr_t one_named_cache_key(const char * tensor, int64_t expert) {
    uint64_t h = 1469598103934665603ULL;
    if (tensor) {
        for (const unsigned char * p = (const unsigned char *) tensor; *p; ++p) {
            h ^= (uint64_t) *p;
            h *= 1099511628211ULL;
        }
    }
    h ^= 0xff;
    h *= 1099511628211ULL;
    uint64_t x = (uint64_t) expert;
    for (int i = 0; i < 8; ++i) {
        h ^= (x >> (i * 8)) & 0xffU;
        h *= 1099511628211ULL;
    }
    if (h == 0) {
        h = 1;
    }
    return (uintptr_t) h;
}

static bool one_prefill_key_mode_enabled() {
    static int enabled = [] {
        const char * path = std::getenv("GGML_MOE_STREAM_ONE_PREFILL_PROFILE");
        return path && path[0] ? 1 : 0;
    }();
    return enabled != 0;
}

static uintptr_t one_cache_key_for(const char * tensor, int64_t expert, const void * src0_data) {
    return one_prefill_key_mode_enabled() ? one_named_cache_key(tensor, expert) : (uintptr_t) src0_data;
}

static void one_prefill_maybe(slot_ctx & ctx, cudaStream_t st, size_t src0_bytes) {
    if (!one_prefill_key_mode_enabled() || !one_prefill_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lk(g_one_prefill.mu);
    if (g_one_prefill.done || !g_one_prefill.enabled || !g_vcache.pool || g_vcache.n_slots <= 0) {
        return;
    }
    g_one_prefill.done = true;

    size_t n = g_one_prefill.limit;
    if (n > (size_t) g_vcache.n_slots) {
        n = (size_t) g_vcache.n_slots;
    }
    if (n > g_one_prefill.entries.size()) {
        n = g_one_prefill.entries.size();
    }

    {
        std::lock_guard<std::mutex> rk(g_resize_mu);
        if (!ensure_host_pinned(ctx.h_src0_pack, ctx.h_src0_pack_sz, src0_bytes)) {
            g_one_prefill.read_failures += n;
            return;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) {
        const one_prefill_entry & e = g_one_prefill.entries[i];
        g_one_prefill.attempted++;
        const one_expert_pack_entry * pack_entry = one_pack_lookup(e.tensor.c_str(), e.expert, src0_bytes);
        if (!pack_entry) {
            g_one_prefill.pack_misses++;
            continue;
        }
        if (!one_pack_read_entry(pack_entry, ctx.h_src0_pack, src0_bytes)) {
            g_one_prefill.read_failures++;
            continue;
        }
        const uintptr_t key = one_named_cache_key(e.tensor.c_str(), e.expert);
        if (vram_cache_insert_impl(key, ctx.h_src0_pack, src0_bytes, st, false)) {
            g_one_prefill.inserted++;
            g_one_prefill.bytes += src0_bytes;
        } else {
            g_one_prefill.read_failures++;
        }
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        g_one_prefill.read_failures++;
    }
    const auto t1 = std::chrono::steady_clock::now();
    g_one_prefill.elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "[moe_stream] one prefill: completed attempted=%lu inserted=%lu bytes=%lu elapsed_ms=%.3f\n",
        g_one_prefill.attempted,
        g_one_prefill.inserted,
        g_one_prefill.bytes,
        g_one_prefill.elapsed_ms);
}

struct one_trace_state {
    std::mutex mu;
    FILE * fp = nullptr;
    bool initialized = false;
    std::chrono::steady_clock::time_point start;
    std::atomic<uint64_t> seq{0};
};

static one_trace_state g_one_trace;

static double trace_elapsed_ms(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(t - g_one_trace.start).count();
}

static FILE * one_trace_fp_locked(std::chrono::steady_clock::time_point start_time) {
    if (!g_one_trace.initialized) {
        g_one_trace.initialized = true;
        g_one_trace.start = start_time;
        const char * path = std::getenv("GGML_MOE_STREAM_ONE_TRACE_OUT");
        if (path && path[0]) {
            g_one_trace.fp = std::fopen(path, "w");
            if (g_one_trace.fp) {
                std::setvbuf(g_one_trace.fp, nullptr, _IOLBF, 0);
                std::fprintf(g_one_trace.fp,
                    "seq,t_ms,tensor,expert,src0_ptr,src0_bytes,cne1,cache_hit,cache_inserted,slot,src0_ms,src1_ms,kernel_ms,d2h_ms,sync_ms,scatter_ms,dontneed_ms,total_ms\n");
            } else {
                std::fprintf(stderr, "[moe_stream] failed to open one trace: %s\n", path);
            }
        }
    }
    return g_one_trace.fp;
}

static void one_trace_write(
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    size_t src0_bytes,
    int64_t cne1,
    bool cache_hit,
    bool cache_inserted,
    int slot,
    std::chrono::steady_clock::time_point t0,
    std::chrono::steady_clock::time_point t_src0,
    std::chrono::steady_clock::time_point t_src1,
    std::chrono::steady_clock::time_point t_kernel,
    std::chrono::steady_clock::time_point t_d2h,
    std::chrono::steady_clock::time_point t_sync,
    std::chrono::steady_clock::time_point t_scatter,
    std::chrono::steady_clock::time_point t_dontneed) {
    std::lock_guard<std::mutex> lk(g_one_trace.mu);
    FILE * fp = one_trace_fp_locked(t0);
    if (!fp) {
        return;
    }
    const uint64_t seq = g_one_trace.seq.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(fp,
        "%lu,%.3f,%s,%ld,%p,%zu,%ld,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
        (unsigned long) seq,
        trace_elapsed_ms(t0),
        src0_name ? src0_name : "",
        (long) expert_index,
        src0_data,
        src0_bytes,
        (long) cne1,
        cache_hit ? 1 : 0,
        cache_inserted ? 1 : 0,
        slot,
        std::chrono::duration<double, std::milli>(t_src0 - t0).count(),
        std::chrono::duration<double, std::milli>(t_src1 - t_src0).count(),
        std::chrono::duration<double, std::milli>(t_kernel - t_src1).count(),
        std::chrono::duration<double, std::milli>(t_d2h - t_kernel).count(),
        std::chrono::duration<double, std::milli>(t_sync - t_d2h).count(),
        std::chrono::duration<double, std::milli>(t_scatter - t_sync).count(),
        std::chrono::duration<double, std::milli>(t_dontneed - t_scatter).count(),
        std::chrono::duration<double, std::milli>(t_dontneed - t0).count());
}

static bool moe_stream_one_experimental_ds4_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4");
        return env && env[0] && env[0] != '0';
    }();
    return enabled != 0;
}

static bool moe_stream_one_ds4_nonids_enabled() {
    static int enabled = [] {
        const char * env = std::getenv("GGML_MOE_STREAM_ONE_DS4_NONIDS");
        return env && env[0] && env[0] != '0';
    }();
    return enabled != 0;
}

static bool moe_stream_one_name_filter_allows(const char * name) {
    static const char * filter = std::getenv("GGML_MOE_STREAM_ONE_NAME_FILTER");
    if (!filter || !filter[0]) {
        return true;
    }
    return name && std::strstr(name, filter) != nullptr;
}

static const int8_t moe_stream_mxfp4_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static int moe_stream_env_int(const char * name, int fallback) {
    const char * env = std::getenv(name);
    return env && env[0] ? std::atoi(env) : fallback;
}

static FILE * moe_stream_q8_debug_fp() {
    static FILE * fp = nullptr;
    static int initialized = 0;
    static std::mutex mu;

    std::lock_guard<std::mutex> lk(mu);
    if (!initialized) {
        initialized = 1;
        const char * path = std::getenv("GGML_MOE_STREAM_Q8_DEBUG_OUT");
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOLBF, 0);
                std::fprintf(fp,
                    "tensor,expert,row_id,i12,out_col,block,gpu_q81_part,gpu_q81_total,gpu_value,d_q,e,sumi_q81\n");
            } else {
                std::fprintf(stderr, "[moe_stream_q8_debug] failed to open trace: %s\n", path);
            }
        }
    }
    return fp;
}

static void moe_stream_q8_debug_maybe(
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    int64_t ne00,
    size_t nb01,
    const void * d_src1_q8,
    const void * d_dst,
    int64_t src1_padded,
    int64_t ne01,
    int64_t cne1,
    const ggml_moe_row_mapping * rows,
    cudaStream_t st) {
    FILE * fp = moe_stream_q8_debug_fp();
    if (!fp || !src0_data || !d_src1_q8 || !d_dst || !rows) {
        return;
    }

    const int target_expert = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_EXPERT", -1);
    const int target_row    = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_ROW", -1);
    const int target_i12    = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_I12", -1);
    const int target_col    = moe_stream_env_int("GGML_MOE_STREAM_Q8_DEBUG_COL", -1);
    if (target_expert >= 0 && expert_index != target_expert) {
        return;
    }
    if (target_col < 0 || target_col >= ne01) {
        return;
    }

    int64_t target_k = -1;
    for (int64_t k = 0; k < cne1; ++k) {
        if ((target_row < 0 || rows[k].i1 == target_row) && (target_i12 < 0 || rows[k].i2 == target_i12)) {
            target_k = k;
            break;
        }
    }
    if (target_k < 0) {
        return;
    }

    const size_t src1_q8_row_bytes = (size_t)(src1_padded / QK8_1) * sizeof(block_q8_1);
    std::vector<uint8_t> q8_row(src1_q8_row_bytes);
    float gpu_value = 0.0f;
    const char * d_q8_row = (const char *) d_src1_q8 + (size_t) target_k * src1_q8_row_bytes;
    const float * d_dst_value = (const float *) d_dst + (size_t) target_k * ne01 + (size_t) target_col;

    if (cudaMemcpyAsync(q8_row.data(), d_q8_row, src1_q8_row_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        return;
    }
    if (cudaMemcpyAsync(&gpu_value, d_dst_value, sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        return;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        return;
    }

    const block_q8_1 * y = (const block_q8_1 *) q8_row.data();
    const block_mxfp4 * x = (const block_mxfp4 *) ((const char *) src0_data + (size_t) target_col * nb01);
    const int64_t nb = ne00 / QK_MXFP4;
    double total = 0.0;

    flockfile(fp);
    for (int64_t ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const int v0 = moe_stream_mxfp4_values[x[ib].qs[j] & 0x0F];
            const int v1 = moe_stream_mxfp4_values[x[ib].qs[j] >> 4];
            sumi += y[ib].qs[j] * v0;
            sumi += y[ib].qs[j + QK_MXFP4/2] * v1;
        }
        const float d_q = __low2float(y[ib].ds);
        const double part = (double) GGML_E8M0_TO_FP32_HALF(x[ib].e) * (double) d_q * (double) sumi;
        total += part;
        std::fprintf(fp,
            "%s,%" PRId64 ",%d,%d,%d,%" PRId64 ",%.9g,%.9g,%.9g,%.9g,%d,%d\n",
            src0_name ? src0_name : "",
            expert_index,
            rows[target_k].i1,
            rows[target_k].i2,
            target_col,
            ib,
            part,
            total,
            (double) gpu_value,
            (double) d_q,
            (int) x[ib].e,
            sumi);
    }
    funlockfile(fp);
}

static bool moe_stream_one_type_allowed(ggml_type type, const char * name) {
    if (type == GGML_TYPE_IQ3_XXS) {
        return moe_stream_one_name_filter_allows(name);
    }
    if (!moe_stream_one_experimental_ds4_enabled()) {
        return false;
    }
    if (type != GGML_TYPE_MXFP4 && type != GGML_TYPE_F8_E4M3_B128) {
        return false;
    }
    return moe_stream_one_name_filter_allows(name);
}

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
    const char *src0_name,
    int64_t expert_index,
    const void *src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float *src1_f32,
    size_t src1_nb1, size_t src1_nb2,
    int64_t src1_ne1,
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
    if (!moe_stream_one_type_allowed(t0, src0_name)) return false;
    if (cne1 < 1 || cne1 > MMVQ_MAX_BATCH_SIZE) return false;
    if (src1_ne1 <= 0) return false;
    if (!src1_f32) return false;   // require F32 src1 for on-GPU quantization

    static std::atomic<int> first_call{0};
    if (first_call.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] first call: ne01=%ld ne00=%ld nb01=%zu src0_bytes=%zu cne1=%ld\n",
                     (long)ne01, (long)ne00, nb01, (size_t)ne01 * nb01, (long)cne1);
        std::fflush(stderr);
    }
    one_direct_manifest_init_once();

    const size_t src0_bytes      = (size_t)ne01 * nb01;
    const size_t src1_f32_bytes  = (size_t)cne1 * ne00 * sizeof(float);
    const int64_t src1_padded    = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_bytes = (size_t)cne1 * (src1_padded / QK8_1) * sizeof(block_q8_1);
    const size_t dst_bytes       = (size_t)ne01 * cne1 * sizeof(float);
    const auto ts0 = std::chrono::steady_clock::now();

    int s = acquire_slot();
    slot_ctx &ctx = g_slots[s];

    // d_src1_f32 = staging for F32 src1 (sized once); d_src1 = Q8_1 quantized.
    bool ok = true;
    {
        std::lock_guard<std::mutex> rk(g_resize_mu);
        ok = ensure_dev(ctx.d_src0, ctx.d_src0_sz, src0_bytes)
          && ensure_dev(ctx.d_src1, ctx.d_src1_sz, src1_q8_1_bytes)
          && ensure_dev(ctx.d_dst,  ctx.d_dst_sz,  dst_bytes)
          && ensure_dev(ctx.d_ids,  ctx.d_ids_sz,  (size_t)cne1 * sizeof(int32_t))
          && ensure_host_pinned(ctx.h_scratch, ctx.h_scratch_sz, dst_bytes);
    }
    if (!ok) { release_slot(s); return false; }

    cudaStream_t st = ctx.stream;

    // Initialize VRAM cache on first call (needs to know expert size)
    if (!g_vcache_inited) {
        std::lock_guard<std::mutex> lk(g_init_mu);
        if (!g_vcache_inited) vram_cache_init(src0_bytes);
    }
    one_prefill_maybe(ctx, st, src0_bytes);

    // VRAM cache lookup: if this expert is already in VRAM, skip H2D entirely.
    uintptr_t cache_key = one_cache_key_for(src0_name, expert_index, src0_data);
    void *cached_vram = vram_cache_lookup(cache_key);
    const bool cache_hit = cached_vram != nullptr;
    bool cache_inserted = false;
    const void *kernel_src0 = nullptr;

    if (cached_vram) {
        kernel_src0 = cached_vram;
    } else {
        const void * copy_src = src0_data;
        if (const one_expert_pack_entry * pack_entry = one_pack_lookup(src0_name, expert_index, src0_bytes)) {
            std::lock_guard<std::mutex> rk(g_resize_mu);
            if (!ensure_host_pinned(ctx.h_src0_pack, ctx.h_src0_pack_sz, src0_bytes)) {
                release_slot(s);
                return false;
            }
            if (one_pack_read_entry(pack_entry, ctx.h_src0_pack, src0_bytes)) {
                copy_src = ctx.h_src0_pack;
            }
        }
        void *inserted = nullptr;
        if (moe_stream_cache_admit_allows(src0_name, expert_index)) {
            inserted = vram_cache_insert(cache_key, copy_src, src0_bytes, st);
        } else {
            g_vcache.misses.fetch_add(1, std::memory_order_relaxed);
        }
        if (inserted) {
            cache_inserted = true;
            kernel_src0 = inserted;
        } else {
            if (cudaMemcpyAsync(ctx.d_src0, copy_src, src0_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) { release_slot(s); return false; }
            kernel_src0 = ctx.d_src0;
        }
    }
    const auto t_src0 = std::chrono::steady_clock::now();

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
        const size_t src1_ne1 = (src1_nb1 > 0 && src1_nb2 >= src1_nb1) ? src1_nb2 / src1_nb1 : 1;
        for (int64_t k = 0; k < cne1; ++k) {
            const int32_t i1 = src1_ne1 > 0 ? rows[k].i1 % (int32_t) src1_ne1 : 0;
            const int32_t i2 = rows[k].i2;
            const char *src_row = src1_base + (size_t)i1 * src1_nb1 + (size_t)i2 * src1_nb2;
            std::memcpy(bounce + (size_t)k * ne00 * sizeof(float), src_row, (size_t)ne00 * sizeof(float));
        }
        cudaMemcpyAsync(ctx.d_src1_f32, ctx.h_bounce, src1_f32_bytes, cudaMemcpyHostToDevice, st);
    } else {
        release_slot(s);
        return false;
    }
    const auto t_src1 = std::chrono::steady_clock::now();
    if ((t0 == GGML_TYPE_MXFP4 || t0 == GGML_TYPE_F8_E4M3_B128) &&
            moe_stream_one_experimental_ds4_enabled() &&
            !moe_stream_one_ds4_nonids_enabled()) {
        if (cudaMemsetAsync(ctx.d_ids, 0, (size_t)cne1 * sizeof(int32_t), st) != cudaSuccess) {
            release_slot(s);
            return false;
        }
        if (!ggml_cuda_moe_stream_mmvq_rows_dev(src0_type_int, kernel_src0, ne01, ne00, nb01,
                    (const float *)ctx.d_src1_f32, ctx.d_src1, (const int32_t *)ctx.d_ids, cne1,
                    (float *)ctx.d_dst, st)) {
            release_slot(s);
            return false;
        }
    } else {
        const int64_t src1_q8_row_bytes = src1_padded * (int64_t)sizeof(block_q8_1) / QK8_1;
        for (int64_t k = 0; k < cne1; ++k) {
            const float *d_src1_row = (const float *)((const char *)ctx.d_src1_f32 + (size_t)k * ne00 * sizeof(float));
            void *d_src1_q8_row = (char *)ctx.d_src1 + (size_t)k * src1_q8_row_bytes;
            float *d_dst_row = (float *)ctx.d_dst + k * ne01;
            if (!ggml_cuda_moe_stream_mmvq_dev(src0_type_int, kernel_src0, ne01, ne00, nb01, d_src1_row, d_src1_q8_row, d_dst_row, st)) {
                release_slot(s);
                return false;
            }
        }
    }
    const auto t_kernel = std::chrono::steady_clock::now();

    if (t0 == GGML_TYPE_MXFP4) {
        moe_stream_q8_debug_maybe(src0_name, expert_index, src0_data, ne00, nb01, ctx.d_src1, ctx.d_dst,
                src1_padded, ne01, cne1, rows, st);
    }

    if (cudaMemcpyAsync(ctx.h_scratch, ctx.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) { release_slot(s); return false; }
    const auto t_d2h = std::chrono::steady_clock::now();

    if (!g_defer_sync) {
        if (cudaStreamSynchronize(st) != cudaSuccess) { release_slot(s); return false; }
        const auto t_sync = std::chrono::steady_clock::now();
        const float *src_buf = (const float *)ctx.h_scratch;
        for (int64_t k = 0; k < cne1; ++k) {
            const int32_t i1 = rows[k].i1;
            const int32_t i2 = rows[k].i2;
            float *dst_row = (float *)((char *)dst + i1 * dst_nb1 + i2 * dst_nb2);
            const float *src_row = src_buf + k * ne01;
            std::memcpy(dst_row, src_row, (size_t)ne01 * sizeof(float));
        }
        const auto t_scatter = std::chrono::steady_clock::now();
        moe_stream_dontneed_source_pages(src0_data, src0_bytes);
        const auto t_dontneed = std::chrono::steady_clock::now();
        one_trace_write(src0_name, expert_index, src0_data, src0_bytes, cne1, cache_hit, cache_inserted, s,
                        ts0, t_src0, t_src1, t_kernel, t_d2h, t_sync, t_scatter, t_dontneed);
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
    one_trace_write(src0_name, expert_index, src0_data, src0_bytes, cne1, cache_hit, cache_inserted, s,
                    ts0, t_src0, t_src1, t_kernel, t_d2h, t_d2h, t_d2h, t_d2h);
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

#endif // GGML_CUDA_MOE_STREAM
