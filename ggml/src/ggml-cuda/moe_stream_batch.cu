// Decode-only batched streaming MoE path.
//
// This file intentionally includes the MMQ-id kernel family without the MMVQ
// headers used by moe_stream.cu; several CUDA helper headers define unguarded
// device functions and cannot be mixed in one translation unit.

#include "common.cuh"
#include "mmq_id_common.cuh"
#include "quantize.cuh"
#include "quantize_id.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

#include <cuda_runtime.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

void ggml_cuda_moe_stream_batch_link_anchor(void) {}

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
bool ggml_cuda_moe_stream_register_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes);

bool ggml_cuda_moe_stream_cache_contains(
    const char *src0_name,
    size_t expert_bytes,
    int expert_idx);

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

bool ggml_cuda_moe_stream_mmvq_dev(
    int src0_type_int,
    const void *d_src0,
    int64_t ne01,
    int64_t ne00,
    const float *d_src1_f32,
    void *d_src1_q8,
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

}

struct pinned_stage_slot {
    void *host = nullptr;
    cudaEvent_t done = nullptr;
    bool pending = false;
};

struct pinned_stage_ring {
    std::vector<pinned_stage_slot> slots;
    size_t slot_sz = 0;
    size_t next = 0;
    bool failed = false;
    bool report_registered = false;
    uint64_t copies = 0;
    uint64_t waits = 0;
    uint64_t fallbacks = 0;
};

struct batch_ctx {
    cudaStream_t stream = nullptr;
    cudaStream_t prefetch_stream = nullptr;
    cudaStream_t up_stream = nullptr;
    cudaStream_t gate_stream = nullptr;
    cudaStream_t up_copy_stream = nullptr;
    cudaStream_t gate_copy_stream = nullptr;
    void * d_src0 = nullptr;     size_t d_src0_sz = 0;
    void * d_src1_f32 = nullptr; size_t d_src1_f32_sz = 0;
    void * d_src1_q8 = nullptr;  size_t d_src1_q8_sz = 0;
    void * d_src1_q8_up = nullptr;   size_t d_src1_q8_up_sz = 0;
    void * d_src1_q8_gate = nullptr; size_t d_src1_q8_gate_sz = 0;
    void * d_src1_q8_one = nullptr; size_t d_src1_q8_one_sz = 0;
    void * d_dst = nullptr;      size_t d_dst_sz = 0;
    void * d_up = nullptr;       size_t d_up_sz = 0;
    void * d_gate = nullptr;     size_t d_gate_sz = 0;
    void * d_handoff = nullptr;  size_t d_handoff_sz = 0;
    int32_t * d_ids_src1 = nullptr; size_t d_ids_src1_sz = 0;
    int32_t * d_ids_dst = nullptr; size_t d_ids_dst_sz = 0;
    int32_t * d_x_ids = nullptr; size_t d_x_ids_sz = 0;
    int32_t * d_x_ids_up = nullptr;   size_t d_x_ids_up_sz = 0;
    int32_t * d_x_ids_gate = nullptr; size_t d_x_ids_gate_sz = 0;
    int32_t * d_bounds = nullptr;  size_t d_bounds_sz = 0;
    void * h_src1 = nullptr;     size_t h_src1_sz = 0;
    void * h_dst = nullptr;      size_t h_dst_sz = 0;
    pinned_stage_ring stage_ring;
    pinned_stage_ring stage_ring_gate;
    pinned_stage_ring stage_ring_up_aux;
    pinned_stage_ring stage_ring_gate_aux;
    cudaEvent_t ev_stage_ready = nullptr;
    cudaEvent_t ev_up_done = nullptr;
    cudaEvent_t ev_gate_done = nullptr;
    cudaEvent_t ev_up_copy_aux_done = nullptr;
    cudaEvent_t ev_gate_copy_aux_done = nullptr;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stage = nullptr;
    cudaEvent_t ev_quant = nullptr;
    cudaEvent_t ev_kernel = nullptr;
    cudaEvent_t ev_d2h = nullptr;
    cudaEvent_t ev_up_start = nullptr;
    cudaEvent_t ev_gate_start = nullptr;
    cudaEvent_t ev_up = nullptr;
    cudaEvent_t ev_gate = nullptr;
    int32_t h_ids_dst[128] = {};
    int32_t h_up_gate_ids_dst[128] = {};
    int32_t h_ids_src1[128] = {};
    int32_t h_x_ids[128] = {};
    int32_t h_x_ids_up[128] = {};
    int32_t h_x_ids_gate[128] = {};
    int32_t h_bounds[129] = {};
};

static batch_ctx g_batch;
static std::mutex g_batch_mu;
static std::atomic<bool> g_batch_inited{false};
static std::atomic<bool> g_pinned_stage_report_registered{false};

struct registered_tensor {
    int type = 0;
    char name[128] = {};
    const void *data = nullptr;
    int64_t n_as = 0;
    size_t nb02 = 0;
    size_t expert_bytes = 0;
};

static std::mutex g_registered_mu;
static std::vector<registered_tensor> g_registered_tensors;

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
    double up_ms = 0.0;
    double gate_ms = 0.0;
    double fuse_ms = 0.0;
};

static batch_profile g_bprof;
static batch_profile g_uprof;

struct expert_pack_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct expert_pack_state {
    FILE *file = nullptr;
#if !defined(_WIN32)
    int fd_direct = -1;
#endif
    std::vector<expert_pack_entry> entries;
    bool inited = false;
    bool enabled = false;
    bool reported_io_backend = false;
    int io_backend = 0; // 0=buffered, 1=direct
    std::mutex mu;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> read_failures{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> direct_fallbacks{0};
};

static expert_pack_state g_expert_pack;

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

static void up_gate_profile_report_atexit() {
    if (!g_uprof.enabled || g_uprof.calls == 0) return;
    const double calls = (double)g_uprof.calls;
    std::fprintf(stderr,
        "[moe_stream_batch] up/gate profile: calls=%lu avg_active=%.2f "
        "stage=%.3f ms quant=%.3f ms up=%.3f ms gate=%.3f ms fuse=%.3f ms "
        "kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms total=%.3f ms/call\n",
        g_uprof.calls,
        (double)g_uprof.active_experts / calls,
        g_uprof.stage_ms / calls,
        g_uprof.quant_ms / calls,
        g_uprof.up_ms / calls,
        g_uprof.gate_ms / calls,
        g_uprof.fuse_ms / calls,
        g_uprof.kernel_ms / calls,
        g_uprof.d2h_ms / calls,
        g_uprof.scatter_ms / calls,
        (g_uprof.stage_ms + g_uprof.quant_ms + g_uprof.kernel_ms + g_uprof.d2h_ms + g_uprof.scatter_ms) / calls);
}

struct batch_vram_cache {
    void * pool = nullptr;
    size_t slot_sz = 0;
    int n_slots = 0;
    uintptr_t slot_key[16384] = {};
    uint64_t slot_used[16384] = {};
    uint32_t slot_hits[16384] = {};
    bool slot_pinned[16384] = {};
    bool slot_prefetch_down[16384] = {};
    uint64_t clock = 1;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t preloads = 0;
    uint64_t pinned = 0;
    uint64_t down_prefetch_loads = 0;
    uint64_t down_prefetch_hits = 0;
    uint64_t down_prefetch_evicted = 0;
};

static batch_vram_cache g_bcaches[2];
static bool g_bcache_inited[2] = {};

struct profile_entry {
    int expert_idx = -1;
    size_t expert_bytes = 0;
    char tensor[96] = {};
};

static std::vector<profile_entry> g_profile;
static bool g_profile_loaded = false;
static bool g_profile_enabled = false;
static std::mutex g_profile_mu;
static char g_preloaded_tensors[256][96] = {};
static int g_n_preloaded_tensors = 0;

struct batch_route_profile_entry {
    int expert_idx = -1;
    uint64_t count = 0;
    size_t expert_bytes = 0;
    char tensor[96] = {};
};

struct batch_route_trace_entry {
    uint64_t seq = 0;
    int expert_idx = -1;
    size_t expert_bytes = 0;
    char tensor[96] = {};
};

static std::vector<batch_route_profile_entry> g_route_profile;
static std::vector<batch_route_trace_entry> g_route_trace;
static std::mutex g_route_profile_mu;
static const char * g_route_profile_out = nullptr;
static const char * g_route_trace_out = nullptr;
static bool g_route_profile_inited = false;
static uint64_t g_route_trace_seq = 0;

static bool batch_route_profile_enabled() {
    const char *route_env = std::getenv("GGML_MOE_ROUTE_PROFILE");
    if (route_env && route_env[0]) return route_env[0] != '0';
    const char *prof_env = std::getenv("GGML_MOE_BATCH_PROFILE");
    return prof_env && prof_env[0] && prof_env[0] != '0';
}

static void batch_route_profile_report_atexit() {
    std::vector<batch_route_profile_entry> rows;
    std::vector<batch_route_trace_entry> trace;
    {
        std::lock_guard<std::mutex> lk(g_route_profile_mu);
        rows = g_route_profile;
        trace = g_route_trace;
    }

    if (g_route_profile_out && g_route_profile_out[0] && !rows.empty()) {
        std::sort(rows.begin(), rows.end(),
            [](const batch_route_profile_entry &a, const batch_route_profile_entry &b) {
                if (a.count != b.count) return a.count > b.count;
                const int name_cmp = std::strcmp(a.tensor, b.tensor);
                if (name_cmp != 0) return name_cmp < 0;
                return a.expert_idx < b.expert_idx;
            });

        FILE *f = std::fopen(g_route_profile_out, "w");
        if (!f) {
            std::fprintf(stderr, "[moe_stream_batch] route profile: open failed: %s\n", g_route_profile_out);
        } else {
            std::fprintf(f, "rank,count,expert_bytes,cumulative_bytes,tensor_base,expert_idx,tensor\n");
            size_t cumulative = 0;
            for (size_t i = 0; i < rows.size(); ++i) {
                const batch_route_profile_entry &e = rows[i];
                cumulative += e.expert_bytes;
                std::fprintf(f, "%zu,%lu,%zu,%zu,0x0,%d,%s\n",
                             i + 1, e.count, e.expert_bytes, cumulative, e.expert_idx, e.tensor);
            }
            std::fclose(f);
            std::fprintf(stderr, "[moe_stream_batch] route profile written: %s (%zu entries)\n",
                         g_route_profile_out, rows.size());
        }
    }

    if (g_route_trace_out && g_route_trace_out[0] && !trace.empty()) {
        FILE *f = std::fopen(g_route_trace_out, "w");
        if (!f) {
            std::fprintf(stderr, "[moe_stream_batch] route trace: open failed: %s\n", g_route_trace_out);
        } else {
            std::fprintf(f, "seq,expert_bytes,tensor_base,expert_idx,tensor\n");
            for (const batch_route_trace_entry &e : trace) {
                std::fprintf(f, "%lu,%zu,0x0,%d,%s\n",
                             e.seq, e.expert_bytes, e.expert_idx, e.tensor);
            }
            std::fclose(f);
            std::fprintf(stderr, "[moe_stream_batch] route trace written: %s (%zu events)\n",
                         g_route_trace_out, trace.size());
        }
    }
}

static void batch_route_profile_init_once() {
    if (g_route_profile_inited) return;
    std::lock_guard<std::mutex> lk(g_route_profile_mu);
    if (g_route_profile_inited) return;
    g_route_profile_out = batch_route_profile_enabled() ? std::getenv("GGML_MOE_BATCH_PROFILE_OUT") : nullptr;
    g_route_trace_out = std::getenv("GGML_MOE_ROUTE_TRACE_OUT");
    if ((g_route_profile_out && g_route_profile_out[0]) || (g_route_trace_out && g_route_trace_out[0])) {
        std::atexit(batch_route_profile_report_atexit);
    }
    g_route_profile_inited = true;
}

static void batch_route_profile_hit(const char *tensor_name, int expert_idx, size_t expert_bytes) {
    batch_route_profile_init_once();
    if (!tensor_name || !tensor_name[0]) return;
    const bool profile_enabled = g_route_profile_out && g_route_profile_out[0];
    const bool trace_enabled = g_route_trace_out && g_route_trace_out[0];
    if (!profile_enabled && !trace_enabled) return;

    std::lock_guard<std::mutex> lk(g_route_profile_mu);
    if (profile_enabled) {
        bool found = false;
        for (batch_route_profile_entry &e : g_route_profile) {
            if (e.expert_idx == expert_idx && std::strcmp(e.tensor, tensor_name) == 0) {
                ++e.count;
                found = true;
                break;
            }
        }
        if (!found) {
            batch_route_profile_entry e;
            e.expert_idx = expert_idx;
            e.count = 1;
            e.expert_bytes = expert_bytes;
            std::snprintf(e.tensor, sizeof(e.tensor), "%s", tensor_name);
            g_route_profile.push_back(e);
        }
    }
    if (trace_enabled) {
        batch_route_trace_entry e;
        e.seq = ++g_route_trace_seq;
        e.expert_idx = expert_idx;
        e.expert_bytes = expert_bytes;
        std::snprintf(e.tensor, sizeof(e.tensor), "%s", tensor_name);
        g_route_trace.push_back(e);
    }
}

static void batch_cache_report_atexit() {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t preloads = 0;
    uint64_t pinned = 0;
    uint64_t down_prefetch_loads = 0;
    uint64_t down_prefetch_hits = 0;
    uint64_t down_prefetch_evicted = 0;
    for (batch_vram_cache &c : g_bcaches) {
        hits += c.hits;
        misses += c.misses;
        preloads += c.preloads;
        pinned += c.pinned;
        down_prefetch_loads += c.down_prefetch_loads;
        down_prefetch_hits += c.down_prefetch_hits;
        down_prefetch_evicted += c.down_prefetch_evicted;
    }
    const uint64_t total = hits + misses;
    if (total == 0) return;
    if (pinned > 0) {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: hits=%lu misses=%lu preloads=%lu pinned=%lu hit_rate=%.1f%%\n",
                     hits, misses, preloads, pinned, 100.0 * hits / total);
    } else {
        std::fprintf(stderr, "[moe_stream_batch] VRAM cache: hits=%lu misses=%lu preloads=%lu hit_rate=%.1f%%\n",
                     hits, misses, preloads, 100.0 * hits / total);
    }
    for (int ic = 0; ic < 2; ++ic) {
        const batch_vram_cache &c = g_bcaches[ic];
        const uint64_t c_total = c.hits + c.misses;
        if (c_total == 0) continue;
        const char *label = ic == 1 ? "upgate" : "down";
        std::fprintf(stderr,
            "[moe_stream_batch] VRAM cache %s: slots=%d slot=%.2f MiB "
            "hits=%lu misses=%lu preloads=%lu pinned=%lu hit_rate=%.1f%%\n",
            label, c.n_slots, c.slot_sz / (1024.0 * 1024.0),
            c.hits, c.misses, c.preloads, c.pinned, 100.0 * c.hits / c_total);
    }
    if (down_prefetch_loads > 0) {
        std::fprintf(stderr, "[moe_stream_batch] down prefetch: loads=%lu hits=%lu evicted_unused=%lu useful_rate=%.1f%%\n",
                     down_prefetch_loads, down_prefetch_hits, down_prefetch_evicted,
                     100.0 * down_prefetch_hits / down_prefetch_loads);
    }
}

static bool profile_protect_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PROTECT");
    return env && env[0] && env[0] != '0';
}

static size_t profile_reserve_slots(const batch_vram_cache *c) {
    if (!c || c->n_slots <= 1) return 0;

    const char *slots_env = std::getenv("GGML_MOE_VRAM_PROFILE_RESERVE_SLOTS");
    if (slots_env && slots_env[0]) {
        long reserve = std::atol(slots_env);
        if (reserve < 0) reserve = 0;
        if (reserve >= c->n_slots) reserve = c->n_slots - 1;
        return (size_t)reserve;
    }

    const char *pct_env = std::getenv("GGML_MOE_VRAM_PROFILE_RESERVE_PCT");
    long pct = (pct_env && pct_env[0]) ? std::atol(pct_env) : 20;
    if (pct < 0) pct = 0;
    if (pct > 95) pct = 95;

    size_t reserve = ((size_t)c->n_slots * (size_t)pct + 99) / 100;
    if (reserve >= (size_t)c->n_slots) reserve = (size_t)c->n_slots - 1;
    return reserve;
}

static size_t profile_preload_slot_budget(const batch_vram_cache *c) {
    if (!c || c->n_slots <= 0) return 0;
    if (!profile_protect_enabled()) return (size_t)c->n_slots;
    const size_t reserve = profile_reserve_slots(c);
    return (size_t)c->n_slots > reserve ? (size_t)c->n_slots - reserve : 1;
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

static bool profile_has_tensor_locked(const char *name) {
    if (!name || !name[0]) return false;
    for (const profile_entry &e : g_profile) {
        if (std::strcmp(e.tensor, name) == 0) return true;
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
            e.expert_bytes = expert_bytes;
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
    const char *split_env = std::getenv("GGML_MOE_VRAM_CACHE_SPLIT");
    if (fused_env && fused_env[0] && fused_env[0] != '0' &&
            split_env && split_env[0] && split_env[0] != '0' &&
            expert_sz <= 4ULL*1024ULL*1024ULL) {
        return 1;
    }
    if (fused_env && fused_env[0] && fused_env[0] != '0' && expert_sz <= 2ULL*1024ULL*1024ULL) {
        return 1;
    }
    return 0;
}

static size_t batch_cache_budget_mib_for_id(size_t budget_mib, int cid) {
    const char *fused_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE");
    const bool fused = fused_env && fused_env[0] && fused_env[0] != '0';
    const char *split_env = std::getenv("GGML_MOE_VRAM_CACHE_SPLIT");
    const bool split = fused && split_env && split_env[0] && split_env[0] != '0';

    if (split) {
        const char *pct_env = std::getenv("GGML_MOE_VRAM_CACHE_UPGATE_PCT");
        long upgate_pct = (pct_env && pct_env[0]) ? std::atol(pct_env) : 63;
        if (upgate_pct < 1) upgate_pct = 1;
        if (upgate_pct > 99) upgate_pct = 99;
        const size_t upgate_mib = (budget_mib * (size_t)upgate_pct) / 100ULL;
        if (cid == 1) return upgate_mib > 0 ? upgate_mib : 1;
        return budget_mib > upgate_mib ? budget_mib - upgate_mib : 1;
    }

    if (fused && budget_mib > 8192 && cid == 1) {
        return 8192;
    }
    return budget_mib;
}

static batch_vram_cache * batch_cache_get(size_t expert_sz) {
    const int cid = batch_cache_id_for_size(expert_sz);
    if (g_bcache_inited[cid]) {
        batch_vram_cache *c = &g_bcaches[cid];
        if (c->pool && expert_sz <= c->slot_sz) return c;
        if (!c->pool || expert_sz <= c->slot_sz) return nullptr;

        cudaFree(c->pool);
        c->pool = nullptr;
        c->slot_sz = 0;
        c->n_slots = 0;
        std::fill_n(c->slot_key, 16384, (uintptr_t)0);
        std::fill_n(c->slot_used, 16384, (uint64_t)0);
        std::fill_n(c->slot_hits, 16384, (uint32_t)0);
        std::fill_n(c->slot_pinned, 16384, false);
        std::fill_n(c->slot_prefetch_down, 16384, false);
        c->clock = 1;
        c->hits = 0;
        c->misses = 0;
        c->preloads = 0;
        c->pinned = 0;
        c->down_prefetch_loads = 0;
        c->down_prefetch_hits = 0;
        c->down_prefetch_evicted = 0;
        g_bcache_inited[cid] = false;
    }
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_GB");
    const size_t budget_gb = env ? (size_t)std::atoi(env) : 16;
    const char *env_mib = std::getenv("GGML_MOE_VRAM_CACHE_MIB");
    size_t budget_mib = env_mib && env_mib[0] ? (size_t)std::strtoull(env_mib, nullptr, 10) : budget_gb * 1024ULL;
    if (budget_mib == 0) {
        g_bcache_inited[cid] = true;
        return nullptr;
    }
    size_t this_budget_mib = batch_cache_budget_mib_for_id(budget_mib, cid);
    const size_t budget = this_budget_mib * 1024ULL * 1024ULL;
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
        cudaGetLastError();
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
            if (c->slot_hits[slot] != UINT32_MAX) {
                ++c->slot_hits[slot];
            }
            ++c->hits;
            if (c->slot_prefetch_down[slot]) {
                ++c->down_prefetch_hits;
                c->slot_prefetch_down[slot] = false;
            }
            return slot;
        }
    }
    return -1;
}

static bool batch_cache_contains_slot(const batch_vram_cache *c, uintptr_t key) {
    if (!c || !c->pool || c->n_slots == 0) return false;
    for (int slot = 0; slot < c->n_slots; ++slot) {
        if (c->slot_key[slot] == key) return true;
    }
    return false;
}

static void batch_cache_clear_slot(batch_vram_cache *c, int slot) {
    if (!c || slot < 0 || slot >= c->n_slots) return;
    c->slot_key[slot] = 0;
    c->slot_used[slot] = 0;
    c->slot_hits[slot] = 0;
    if (c->slot_pinned[slot] && c->pinned > 0) {
        --c->pinned;
    }
    c->slot_pinned[slot] = false;
    c->slot_prefetch_down[slot] = false;
}

static int batch_cache_find_slot(batch_vram_cache *c, uintptr_t key) {
    if (!c || !c->pool || c->n_slots == 0) return -1;
    for (int slot = 0; slot < c->n_slots; ++slot) {
        if (c->slot_key[slot] == key) return slot;
    }
    return -1;
}

static bool batch_slot_is_avoided(int slot, const int *avoid_slots, int n_avoid_slots) {
    if (!avoid_slots || n_avoid_slots <= 0) return false;
    for (int i = 0; i < n_avoid_slots; ++i) {
        if (avoid_slots[i] == slot) return true;
    }
    return false;
}

static bool cache_policy_lfu_lru_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_CACHE_POLICY");
    return env && std::strcmp(env, "lfu_lru") == 0;
}

static int expert_pack_entry_cmp(const expert_pack_entry &e, const char *tensor_name, int expert_idx, size_t nbytes) {
    const int name_cmp = std::strcmp(e.tensor, tensor_name);
    if (name_cmp != 0) return name_cmp;
    if (e.expert_idx != expert_idx) return e.expert_idx < expert_idx ? -1 : 1;
    if (e.nbytes != nbytes) return e.nbytes < nbytes ? -1 : 1;
    return 0;
}

static size_t expert_pack_direct_alignment() {
    return 4096;
}

static uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static void expert_pack_report_atexit() {
    if (!g_expert_pack.enabled) return;
    std::fprintf(stderr,
                 "[moe_stream_batch] expert pack: hits=%lu misses=%lu read_failures=%lu direct_reads=%lu direct_fallbacks=%lu entries=%zu\n",
                 g_expert_pack.hits.load(), g_expert_pack.misses.load(),
                 g_expert_pack.read_failures.load(), g_expert_pack.direct_reads.load(),
                 g_expert_pack.direct_fallbacks.load(), g_expert_pack.entries.size());
}

static bool expert_pack_read_exact(FILE *file, void *dst, size_t sz) {
    char *out = (char *)dst;
    size_t done = 0;
    while (done < sz) {
        const size_t chunk = sz - done;
        const size_t got = std::fread(out + done, 1, chunk, file);
        if (got == 0) return false;
        done += got;
    }
    return true;
}

static void expert_pack_init_once() {
    std::lock_guard<std::mutex> lk(g_expert_pack.mu);
    if (g_expert_pack.inited) return;

    const char *io_backend_env = std::getenv("GGML_MOE_IO_BACKEND");
    if (io_backend_env && io_backend_env[0]) {
        if (std::strcmp(io_backend_env, "direct") == 0) {
            g_expert_pack.io_backend = 1;
        } else if (std::strcmp(io_backend_env, "mmap") == 0) {
            g_expert_pack.io_backend = 0;
        } else if (!g_expert_pack.reported_io_backend) {
            std::fprintf(stderr,
                "[moe_stream_batch] GGML_MOE_IO_BACKEND=%s requested; v1 runtime supports mmap/buffered and direct expert-pack reads\n",
                io_backend_env);
            g_expert_pack.reported_io_backend = true;
        }
    }

    const char *path = std::getenv("GGML_MOE_EXPERT_PACK");
    if (!path || !path[0]) {
        g_expert_pack.inited = true;
        return;
    }

    FILE *file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: open failed: %s\n", path);
        g_expert_pack.inited = true;
        return;
    }

    char magic[16] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t n_entries = 0;
    uint64_t data_start = 0;
    if (!expert_pack_read_exact(file, magic, sizeof(magic)) ||
            !expert_pack_read_exact(file, &version, sizeof(version)) ||
            !expert_pack_read_exact(file, &header_size, sizeof(header_size)) ||
            !expert_pack_read_exact(file, &n_entries, sizeof(n_entries)) ||
            !expert_pack_read_exact(file, &data_start, sizeof(data_start)) ||
            std::memcmp(magic, "GGMLMOEPACKv1", 13) != 0 ||
            version != 1 || header_size < 40 || data_start < header_size || n_entries > 10000000ULL) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: invalid header: %s\n", path);
        std::fclose(file);
        g_expert_pack.inited = true;
        return;
    }

    std::vector<expert_pack_entry> entries;
    entries.resize((size_t)n_entries);
    for (uint64_t i = 0; i < n_entries; ++i) {
        expert_pack_entry &e = entries[(size_t)i];
        uint32_t reserved = 0;
        if (!expert_pack_read_exact(file, e.tensor, sizeof(e.tensor)) ||
                !expert_pack_read_exact(file, &e.expert_idx, sizeof(e.expert_idx)) ||
                !expert_pack_read_exact(file, &reserved, sizeof(reserved)) ||
                !expert_pack_read_exact(file, &e.offset, sizeof(e.offset)) ||
                !expert_pack_read_exact(file, &e.nbytes, sizeof(e.nbytes))) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: short index: %s\n", path);
            std::fclose(file);
            g_expert_pack.inited = true;
            return;
        }
        e.tensor[sizeof(e.tensor) - 1] = '\0';
    }

    std::sort(entries.begin(), entries.end(),
        [](const expert_pack_entry &a, const expert_pack_entry &b) {
            const int name_cmp = std::strcmp(a.tensor, b.tensor);
            if (name_cmp != 0) return name_cmp < 0;
            if (a.expert_idx != b.expert_idx) return a.expert_idx < b.expert_idx;
            return a.nbytes < b.nbytes;
        });

    g_expert_pack.file = file;
#if !defined(_WIN32)
    if (g_expert_pack.io_backend == 1) {
#if defined(O_DIRECT)
        g_expert_pack.fd_direct = ::open(path, O_RDONLY | O_DIRECT);
#else
        g_expert_pack.fd_direct = -1;
#endif
        if (g_expert_pack.fd_direct < 0) {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: direct open failed; using buffered reads: %s\n", path);
            g_expert_pack.io_backend = 0;
        } else {
            std::fprintf(stderr, "[moe_stream_batch] expert pack: direct reads enabled: %s\n", path);
        }
    }
#else
    if (g_expert_pack.io_backend == 1) {
        std::fprintf(stderr, "[moe_stream_batch] expert pack: direct reads are not supported on this platform; using buffered reads\n");
        g_expert_pack.io_backend = 0;
    }
#endif
    g_expert_pack.entries = std::move(entries);
    g_expert_pack.enabled = true;
    g_expert_pack.inited = true;
    std::atexit(expert_pack_report_atexit);
    std::fprintf(stderr, "[moe_stream_batch] expert pack: loaded %zu entries from %s\n",
                 g_expert_pack.entries.size(), path);
}

static const expert_pack_entry * expert_pack_lookup(const char *tensor_name, int expert_idx, size_t nbytes) {
    expert_pack_init_once();
    if (!g_expert_pack.enabled || !tensor_name || !tensor_name[0]) return nullptr;

    size_t lo = 0;
    size_t hi = g_expert_pack.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int cmp = expert_pack_entry_cmp(g_expert_pack.entries[mid], tensor_name, expert_idx, nbytes);
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_expert_pack.entries.size() &&
            expert_pack_entry_cmp(g_expert_pack.entries[lo], tensor_name, expert_idx, nbytes) == 0) {
        ++g_expert_pack.hits;
        return &g_expert_pack.entries[lo];
    }
    ++g_expert_pack.misses;
    return nullptr;
}

static bool expert_pack_read_entry(const expert_pack_entry *entry, void *dst, size_t sz) {
    if (!entry || !g_expert_pack.file || entry->nbytes != sz) return false;

#if !defined(_WIN32)
    if (g_expert_pack.io_backend == 1 && g_expert_pack.fd_direct >= 0) {
        const uint64_t alignment = expert_pack_direct_alignment();
        const size_t read_sz = (size_t)align_up_u64((uint64_t)sz, alignment);
        if ((entry->offset % alignment) == 0 &&
                ((uintptr_t)dst % alignment) == 0 &&
                read_sz >= sz) {
            char *out = (char *)dst;
            size_t done = 0;
            while (done < read_sz) {
                const size_t chunk = std::min(read_sz - done, (size_t)64 * 1024 * 1024);
                const ssize_t got = ::pread(g_expert_pack.fd_direct, out + done, chunk, (off_t)(entry->offset + done));
                if (got <= 0) {
                    ++g_expert_pack.direct_fallbacks;
                    break;
                }
                done += (size_t)got;
            }
            if (done == read_sz) {
                ++g_expert_pack.direct_reads;
                return true;
            }
        } else {
            ++g_expert_pack.direct_fallbacks;
        }
    }
#endif

    std::lock_guard<std::mutex> lk(g_expert_pack.mu);
#if defined(_WIN32)
    if (_fseeki64(g_expert_pack.file, (int64_t)entry->offset, SEEK_SET) != 0) {
#else
    if (::fseeko(g_expert_pack.file, (off_t)entry->offset, SEEK_SET) != 0) {
#endif
        ++g_expert_pack.read_failures;
        return false;
    }
    if (!expert_pack_read_exact(g_expert_pack.file, dst, sz)) {
        ++g_expert_pack.read_failures;
        return false;
    }
    return true;
}

static bool stage_pinned_enabled() {
    const char *env = std::getenv("GGML_MOE_STAGE_PINNED");
    return env && env[0] && env[0] != '0';
}

static int stage_pinned_slot_count() {
    const char *env = std::getenv("GGML_MOE_STAGE_PINNED_SLOTS");
    long n = (env && env[0]) ? std::atol(env) : 16;
    if (n < 1) n = 1;
    if (n > 256) n = 256;
    return (int)n;
}

static void pinned_stage_release(pinned_stage_ring &ring) {
    for (pinned_stage_slot &slot : ring.slots) {
        if (slot.pending && slot.done) {
            cudaEventSynchronize(slot.done);
            slot.pending = false;
        }
        if (slot.host) {
            cudaFreeHost(slot.host);
            slot.host = nullptr;
        }
        if (slot.done) {
            cudaEventDestroy(slot.done);
            slot.done = nullptr;
        }
    }
    ring.slots.clear();
    ring.slot_sz = 0;
    ring.next = 0;
}

static void pinned_stage_report_atexit() {
    auto report_ring = [](const char *name, const pinned_stage_ring &ring) {
        if (ring.copies == 0 && ring.fallbacks == 0) return;
        std::fprintf(stderr,
            "[moe_stream_batch] pinned staging%s: copies=%lu waits=%lu fallbacks=%lu slots=%zu slot=%.2f MiB\n",
            name, ring.copies, ring.waits, ring.fallbacks, ring.slots.size(), ring.slot_sz / (1024.0 * 1024.0));
    };
    report_ring("", g_batch.stage_ring);
    report_ring(" gate", g_batch.stage_ring_gate);
    report_ring(" up_aux", g_batch.stage_ring_up_aux);
    report_ring(" gate_aux", g_batch.stage_ring_gate_aux);
}

static bool pinned_stage_ensure(pinned_stage_ring &ring, size_t need, bool force) {
    if (!force && !stage_pinned_enabled()) return false;
    if (ring.failed) return false;
    const size_t alloc_need = (size_t)align_up_u64((uint64_t)need, (uint64_t)expert_pack_direct_alignment());

    const int n_slots = stage_pinned_slot_count();
    if (!ring.slots.empty() && (int)ring.slots.size() == n_slots && ring.slot_sz >= alloc_need) {
        return true;
    }

    pinned_stage_release(ring);
    ring.slot_sz = alloc_need;
    ring.slots.resize((size_t)n_slots);
    for (pinned_stage_slot &slot : ring.slots) {
        if (cudaHostAlloc(&slot.host, ring.slot_sz, cudaHostAllocDefault) != cudaSuccess ||
                cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming) != cudaSuccess) {
            pinned_stage_release(ring);
            ring.failed = true;
            std::fprintf(stderr,
                "[moe_stream_batch] pinned staging: init failed for %d slots of %.2f MiB; falling back to direct H2D\n",
                n_slots, need / (1024.0 * 1024.0));
            return false;
        }
    }

    if (!g_pinned_stage_report_registered.exchange(true)) {
        std::atexit(pinned_stage_report_atexit);
    }
    ring.report_registered = true;
    std::fprintf(stderr, "[moe_stream_batch] pinned staging: enabled, %d slots of %.2f MiB\n",
                 n_slots, need / (1024.0 * 1024.0));
    return true;
}

static bool batch_cache_copy_h2d(
        pinned_stage_ring &ring,
        void *dst, const void *host_data, size_t sz, cudaStream_t st,
        const expert_pack_entry *pack_entry) {
    const bool use_pinned_stage = stage_pinned_enabled() || pack_entry;
    if (use_pinned_stage) {
        if (pinned_stage_ensure(ring, sz, pack_entry != nullptr)) {
            pinned_stage_slot &slot = ring.slots[ring.next++ % ring.slots.size()];
            if (slot.pending) {
                if (cudaEventSynchronize(slot.done) != cudaSuccess) return false;
                slot.pending = false;
                ++ring.waits;
            }

            if (pack_entry) {
                if (!expert_pack_read_entry(pack_entry, slot.host, sz)) {
                    return false;
                }
            } else {
                std::memcpy(slot.host, host_data, sz);
            }
            if (cudaMemcpyAsync(dst, slot.host, sz, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
            if (cudaEventRecord(slot.done, st) != cudaSuccess) {
                cudaStreamSynchronize(st);
                return false;
            }
            slot.pending = true;
            ++ring.copies;
            return true;
        }
        ++ring.fallbacks;
    }

    return cudaMemcpyAsync(dst, host_data, sz, cudaMemcpyHostToDevice, st) == cudaSuccess;
}

static bool batch_cache_copy_h2d(
        void *dst, const void *host_data, size_t sz, cudaStream_t st,
        const expert_pack_entry *pack_entry) {
    return batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, pack_entry);
}

static int batch_cache_insert_slot(
        batch_vram_cache *c, uintptr_t key, const void *host_data, size_t sz, cudaStream_t st,
        bool allow_evict, bool preload, const int *avoid_slots = nullptr, int n_avoid_slots = 0,
        bool do_copy = true, const char *tensor_name = nullptr, int expert_idx = -1,
        bool prefetch_down = false) {
    if (!c || !c->pool || c->n_slots == 0 || sz > c->slot_sz) return -1;
    const bool pin_slot = preload && profile_protect_enabled();
    if (pin_slot && c->pinned >= profile_preload_slot_budget(c)) return -1;

    int slot = -1;
    uint64_t oldest = UINT64_MAX;
    uint32_t lowest_hits = UINT32_MAX;
    const bool lfu_lru = cache_policy_lfu_lru_enabled();
    for (int i = 0; i < c->n_slots; ++i) {
        if (batch_slot_is_avoided(i, avoid_slots, n_avoid_slots)) continue;
        if (c->slot_key[i] == 0) {
            slot = i;
            break;
        }
        if (!allow_evict) continue;
        if (c->slot_pinned[i]) continue;
        if (lfu_lru) {
            if (c->slot_hits[i] < lowest_hits ||
                    (c->slot_hits[i] == lowest_hits && c->slot_used[i] < oldest)) {
                lowest_hits = c->slot_hits[i];
                oldest = c->slot_used[i];
                slot = i;
            }
        } else {
            if (c->slot_used[i] < oldest) {
                oldest = c->slot_used[i];
                slot = i;
            }
        }
    }
    if (slot < 0) return -1;
    if (c->slot_pinned[slot] && !pin_slot && c->pinned > 0) {
        --c->pinned;
    }
    if (c->slot_key[slot] != 0 && c->slot_prefetch_down[slot]) {
        ++c->down_prefetch_evicted;
    }
    c->slot_key[slot] = key;
    c->slot_used[slot] = c->clock++;
    c->slot_hits[slot] = 0;
    if (pin_slot && !c->slot_pinned[slot]) {
        ++c->pinned;
    }
    c->slot_pinned[slot] = pin_slot;
    c->slot_prefetch_down[slot] = prefetch_down;
    void *dst = (char *)c->pool + (size_t)slot * c->slot_sz;
    auto clear_slot = [&]() {
        c->slot_key[slot] = 0;
        c->slot_used[slot] = 0;
        c->slot_hits[slot] = 0;
        if (c->slot_pinned[slot] && c->pinned > 0) {
            --c->pinned;
        }
        c->slot_pinned[slot] = false;
        c->slot_prefetch_down[slot] = false;
    };
    if (do_copy) {
        const expert_pack_entry *pack_entry = expert_pack_lookup(tensor_name, expert_idx, sz);
        if (!batch_cache_copy_h2d(dst, host_data, sz, st, pack_entry)) {
            if (pack_entry && !batch_cache_copy_h2d(dst, host_data, sz, st, nullptr)) {
                clear_slot();
                return -1;
            }
            if (!pack_entry) {
                clear_slot();
                return -1;
            }
        }
        if (cudaGetLastError() != cudaSuccess) {
            clear_slot();
            return -1;
        }
    }
    if (preload) {
        ++c->preloads;
    } else {
        ++c->misses;
    }
    if (prefetch_down) {
        ++c->down_prefetch_loads;
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
    const size_t preload_budget = profile_preload_slot_budget(cache);
    const int cache_id = batch_cache_id_for_size(src0_bytes);
    const bool use_lookup = !profile_has_tensor_locked(tensor_name);
    size_t seen_for_cache = 0;
    for (size_t ip = 0; ip < g_profile.size(); ++ip) {
        const profile_entry &e = g_profile[ip];
        if (e.expert_bytes != 0 && batch_cache_id_for_size(e.expert_bytes) != cache_id) continue;
        if (seen_for_cache++ >= preload_budget) break;
        if (std::strcmp(e.tensor, tensor_name) != 0 && std::strcmp(e.tensor, lookup_name) != 0) continue;
        if (!use_lookup && std::strcmp(e.tensor, tensor_name) != 0) continue;
        if (e.expert_idx < 0 || e.expert_idx >= n_as) continue;
        const uintptr_t key = batch_key_hash(tensor_name, e.expert_idx);
        if (batch_cache_find_slot(cache, key) >= 0) continue;
        const char *expert_host = (const char *)src0_data + (size_t)e.expert_idx * nb02;
        if (batch_cache_insert_slot(cache, key, expert_host, src0_bytes, st, false, true,
                nullptr, 0, true, tensor_name, e.expert_idx) < 0) break;
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

static bool moe_stream_type_supported(ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ2_S;
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
        if (g_batch.stream && cudaStreamCreate(&g_batch.prefetch_stream) != cudaSuccess) {
            g_batch.prefetch_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.up_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.up_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.gate_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.gate_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.up_copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.up_copy_stream = nullptr;
        }
        if (g_batch.stream && cudaStreamCreateWithFlags(&g_batch.gate_copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_batch.gate_copy_stream = nullptr;
        }
        if (g_batch.stream) {
            cudaEventCreateWithFlags(&g_batch.ev_stage_ready, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_up_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_gate_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_up_copy_aux_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_batch.ev_gate_copy_aux_done, cudaEventDisableTiming);
        }
        const char *prof_env = std::getenv("GGML_MOE_BATCH_PROFILE");
        g_bprof.enabled = prof_env && prof_env[0] && prof_env[0] != '0';
        g_uprof.enabled = g_bprof.enabled;
        if (g_batch.stream && g_bprof.enabled) {
            cudaEventCreate(&g_batch.ev_start);
            cudaEventCreate(&g_batch.ev_stage);
            cudaEventCreate(&g_batch.ev_quant);
            cudaEventCreate(&g_batch.ev_kernel);
            cudaEventCreate(&g_batch.ev_d2h);
            cudaEventCreate(&g_batch.ev_up_start);
            cudaEventCreate(&g_batch.ev_gate_start);
            cudaEventCreate(&g_batch.ev_up);
            cudaEventCreate(&g_batch.ev_gate);
            std::atexit(batch_profile_report_atexit);
            std::atexit(up_gate_profile_report_atexit);
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
    if (!moe_stream_type_supported((ggml_type)src0_type_int) || !src0_data || !src0_name) return false;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    if (!batch_cache_get(expert_bytes)) return false;
    preload_profile_for_tensor(src0_name, src0_data, n_as, nb02, expert_bytes, g_batch.stream);
    cudaStreamSynchronize(g_batch.stream);
    return true;
}

extern "C" bool ggml_cuda_moe_stream_cache_contains(
    const char *src0_name,
    size_t expert_bytes,
    int expert_idx) {
    if (!src0_name || !src0_name[0] || expert_bytes == 0 || expert_idx < 0) return false;
    const int cid = batch_cache_id_for_size(expert_bytes);
    if (!g_bcache_inited[cid]) return false;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    const batch_vram_cache *cache = &g_bcaches[cid];
    return batch_cache_contains_slot(cache, batch_key_hash(src0_name, expert_idx));
}

extern "C" bool ggml_cuda_moe_stream_register_tensor(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes) {
    if (!moe_stream_type_supported((ggml_type)src0_type_int) || !src0_name || !src0_name[0] || !src0_data) return false;
    if (!std::strstr(src0_name, ".ffn_down_exps.")) return true;

    std::lock_guard<std::mutex> lk(g_registered_mu);
    for (registered_tensor &t : g_registered_tensors) {
        if (std::strcmp(t.name, src0_name) == 0) {
            t.type = src0_type_int;
            t.data = src0_data;
            t.n_as = n_as;
            t.nb02 = nb02;
            t.expert_bytes = expert_bytes;
            return true;
        }
    }

    registered_tensor t;
    t.type = src0_type_int;
    std::snprintf(t.name, sizeof(t.name), "%s", src0_name);
    t.data = src0_data;
    t.n_as = n_as;
    t.nb02 = nb02;
    t.expert_bytes = expert_bytes;
    g_registered_tensors.push_back(t);
    return true;
}

static bool down_prefetch_enabled() {
    const char *env = std::getenv("GGML_MOE_PREFETCH_DOWN");
    return env && env[0] && env[0] != '0';
}

static int down_prefetch_depth() {
    const char *env = std::getenv("GGML_MOE_PREFETCH_DOWN_DEPTH");
    long depth = (env && env[0]) ? std::atol(env) : 8;
    if (depth < 1) depth = 1;
    if (depth > 128) depth = 128;
    return (int)depth;
}

static bool down_name_for_up_gate(const char *src_name, char *out, size_t out_sz) {
    if (!src_name || !src_name[0] || !out || out_sz == 0) return false;
    const char *needle = std::strstr(src_name, ".ffn_up_exps.");
    if (!needle) needle = std::strstr(src_name, ".ffn_gate_exps.");
    if (!needle) return false;

    const size_t prefix = (size_t)(needle - src_name);
    const char *suffix = needle + std::strlen(".ffn_up_exps.");
    if (std::strstr(needle, ".ffn_gate_exps.") == needle) {
        suffix = needle + std::strlen(".ffn_gate_exps.");
    }
    std::snprintf(out, out_sz, "%.*s.ffn_down_exps.%s", (int)prefix, src_name, suffix);
    return true;
}

static void preload_registered_down_for_active(const char *src_name, const int *active_experts, int n_active) {
    if (!down_prefetch_enabled() || !g_batch.prefetch_stream || !src_name || !active_experts || n_active <= 0) return;

    char down_name[128] = {};
    if (!down_name_for_up_gate(src_name, down_name, sizeof(down_name))) return;

    registered_tensor rt;
    bool found = false;
    {
        std::lock_guard<std::mutex> lk(g_registered_mu);
        for (const registered_tensor &t : g_registered_tensors) {
            if (std::strcmp(t.name, down_name) == 0) {
                rt = t;
                found = true;
                break;
            }
        }
    }
    if (!found || !rt.data || rt.expert_bytes == 0) return;

    batch_vram_cache *cache = batch_cache_get(rt.expert_bytes);
    if (!cache) return;

    int loaded = 0;
    const int max_loads = down_prefetch_depth();
    for (int j = 0; j < n_active; ++j) {
        if (loaded >= max_loads) break;
        const int expert = active_experts[j];
        if (expert < 0 || expert >= rt.n_as) continue;
        const uintptr_t key = batch_key_hash(rt.name, expert);
        if (batch_cache_find_slot(cache, key) >= 0) continue;
        const char *expert_host = (const char *)rt.data + (size_t)expert * rt.nb02;
        if (batch_cache_insert_slot(cache, key, expert_host, rt.expert_bytes, g_batch.prefetch_stream, true, true,
                nullptr, 0, true, rt.name, expert, true) >= 0) {
            ++loaded;
        }
    }

    static std::atomic<int> first_down_prefetch{0};
    if (loaded > 0 && first_down_prefetch.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream_batch] down prefetch active: %s loaded=%d\n", down_name, loaded);
    }
}

static bool launch_moe_mmq_id_batch(
        ggml_type src0_type,
        const char * d_src0, const int * d_src1_q8, const int32_t * d_ids_dst,
        const int32_t * d_bounds, const int32_t * d_x_ids, float * d_dst,
        int64_t ne00, int64_t ne01, int64_t src0_stride, int64_t src0_channel_stride,
        int64_t n_active, int64_t dst_cols, cudaStream_t st) {
    const mmq_args_id args = {
        d_src0, src0_type, d_src1_q8, d_ids_dst, d_bounds, d_x_ids, d_dst,
        ne00, ne01, dst_cols, src0_stride, n_active, ne01,
        n_active, n_active, src0_channel_stride, 0, 0,
        1, 1, 0, 0, 0,
        false, n_active};
    ggml_backend_cuda_context * null_ctx = nullptr;
    switch (src0_type) {
        case GGML_TYPE_IQ3_XXS: launch_mul_mat_q_id<GGML_TYPE_IQ3_XXS, 8>(*null_ctx, args, st); break;
        case GGML_TYPE_IQ2_S:   launch_mul_mat_q_id<GGML_TYPE_IQ2_S,   8>(*null_ctx, args, st); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_moe_mmq_id_one(
        ggml_type src0_type,
        const char * d_src0, const int * d_src1_q8, const int32_t * d_ids_dst,
        const int32_t * d_bounds, const int32_t * d_x_ids, float * d_dst,
        int64_t ne00, int64_t ne01, int64_t src0_stride, int64_t src0_channel_stride,
        int64_t dst_cols, cudaStream_t st) {
    const mmq_args_id args = {
        d_src0, src0_type, d_src1_q8, d_ids_dst, d_bounds, d_x_ids, d_dst,
        ne00, ne01, dst_cols, src0_stride, 1, ne01,
        1, 1, src0_channel_stride, 0, 0,
        1, 1, 0, 0, 0,
        false, 1};
    ggml_backend_cuda_context * null_ctx = nullptr;
    switch (src0_type) {
        case GGML_TYPE_IQ3_XXS: launch_mul_mat_q_id<GGML_TYPE_IQ3_XXS, 8>(*null_ctx, args, st); break;
        case GGML_TYPE_IQ2_S:   launch_mul_mat_q_id<GGML_TYPE_IQ2_S,   8>(*null_ctx, args, st); break;
        default: return false;
    }
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
    ggml_type src0_type = (ggml_type)src0_type_int;
    if (!moe_stream_type_supported(src0_type) || !src1_f32 || !src0_up_data || !src0_gate_data) return false;
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
    const bool profile = g_uprof.enabled && bc.ev_start && bc.ev_stage && bc.ev_quant && bc.ev_kernel && bc.ev_d2h;

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
        && ensure_dev(bc.d_src1_q8_up, bc.d_src1_q8_up_sz, src1_q8_bytes)
        && ensure_dev(bc.d_src1_q8_gate, bc.d_src1_q8_gate_sz, src1_q8_bytes)
        && ensure_dev(bc.d_src1_q8_one, bc.d_src1_q8_one_sz, src1_q8_one_bytes)
        && ensure_dev(bc.d_dst, bc.d_dst_sz, dst_bytes)
        && ensure_dev(bc.d_up, bc.d_up_sz, dst_bytes)
        && ensure_dev(bc.d_gate, bc.d_gate_sz, dst_bytes)
        && (!use_handoff || ensure_dev(bc.d_handoff, bc.d_handoff_sz, dst_bytes))
        && ensure_dev((void *&)bc.d_ids_src1, bc.d_ids_src1_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_ids_dst, bc.d_ids_dst_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_x_ids, bc.d_x_ids_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_x_ids_up, bc.d_x_ids_up_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_x_ids_gate, bc.d_x_ids_gate_sz, ids_bytes)
        && ensure_dev((void *&)bc.d_bounds, bc.d_bounds_sz, bounds_bytes)
        && ensure_host_pinned(bc.h_src1, bc.h_src1_sz, src1_f32_bytes)
        && ensure_host_pinned(bc.h_dst, bc.h_dst_sz, dst_bytes);
    if (!ok) return false;

    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return false;
    preload_registered_down_for_active(src0_up_name, active_experts, n_active);

    char up_key_name[128] = {};
    char gate_key_name[128] = {};
    const bool shared_tensor_name = src0_up_name && src0_gate_name && std::strcmp(src0_up_name, src0_gate_name) == 0;
    const bool disambiguate_halves = shared_tensor_name || src0_up_data == src0_gate_data;
    std::snprintf(up_key_name, sizeof(up_key_name), "%s%s", src0_up_name ? src0_up_name : "up", disambiguate_halves ? ":up" : "");
    std::snprintf(gate_key_name, sizeof(gate_key_name), "%s%s", src0_gate_name ? src0_gate_name : "gate", disambiguate_halves ? ":gate" : "");

    const char *profile_upgate_env = std::getenv("GGML_MOE_VRAM_PROFILE_UPGATE");
    const bool profile_upgate = !profile_upgate_env || !profile_upgate_env[0] || profile_upgate_env[0] != '0';
    if (profile_upgate) {
        preload_profile_for_tensor(up_key_name, src0_up_data, n_as, nb02, src0_bytes, st);
        preload_profile_for_tensor(gate_key_name, src0_gate_data, n_as, nb02, src0_bytes, st);
    }
    if (profile) cudaEventRecord(bc.ev_start, st);

    const char *serial_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_SERIAL");
    const bool serial_up_gate = serial_env && serial_env[0] && serial_env[0] != '0';
    static std::atomic<int> first_serial_up_gate{0};
    if (serial_up_gate && first_serial_up_gate.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] up/gate serial MMQ probe active\n");
    }
    const char *iq2s_batch_env = std::getenv("GGML_MOE_STREAM_IQ2S_BATCH_MMVQ");
    const bool iq2s_batch_mmvq = !iq2s_batch_env || !iq2s_batch_env[0] || iq2s_batch_env[0] != '0';
    static std::atomic<int> first_iq2s_batch_mmvq{0};
    if (src0_type == GGML_TYPE_IQ2_S && iq2s_batch_mmvq && first_iq2s_batch_mmvq.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] IQ2_S batched MMVQ up/gate path active\n");
    }
    const char *parallel_env = std::getenv("GGML_MOE_STREAM_UP_GATE_PARALLEL");
    const bool parallel_up_gate =
        parallel_env && parallel_env[0] && parallel_env[0] != '0' &&
        src0_type == GGML_TYPE_IQ2_S && iq2s_batch_mmvq && !serial_up_gate &&
        bc.up_stream && bc.gate_stream && bc.ev_stage_ready && bc.ev_up_done && bc.ev_gate_done;
    static std::atomic<int> first_parallel_up_gate{0};
    if (parallel_up_gate && first_parallel_up_gate.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] IQ2_S parallel up/gate streams active\n");
    }
    const char *parallel_stage_env = std::getenv("GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE");
    const bool parallel_stage =
        parallel_up_gate && parallel_stage_env && parallel_stage_env[0] && parallel_stage_env[0] != '0';
    static std::atomic<int> first_parallel_stage{0};
    if (parallel_stage && first_parallel_stage.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] up/gate parallel CPU staging active\n");
    }

    auto stage_tensor = [&](
            const char *key_name, const void *host_base, void *d_out,
            cudaStream_t run_stream, int32_t *d_x_ids, void *d_src1_q8,
            int32_t *h_x_ids, int *slots_out,
            const int *avoid_slots, int n_avoid_slots) -> bool {
        for (int j = 0; j < n_active; ++j) {
            const char *expert_host = (const char *)host_base + (size_t)active_experts[j] * nb02;
            const uintptr_t cache_key = batch_key_hash(key_name, active_experts[j]);
            int cache_slot = batch_cache_lookup_slot(cache, cache_key);
            if (cache_slot < 0) {
                cache_slot = batch_cache_insert_slot(
                    cache, cache_key, expert_host, src0_bytes, run_stream, true, false,
                    avoid_slots, n_avoid_slots, true, key_name, active_experts[j]);
            }
            if (cache_slot < 0) return false;
            h_x_ids[j] = cache_slot;
            if (slots_out) slots_out[j] = cache_slot;
            batch_route_profile_hit(key_name, active_experts[j], src0_bytes);
        }
        if (cudaMemsetAsync(d_out, 0, dst_bytes, run_stream) != cudaSuccess) return false;
        if (cudaMemcpyAsync(d_x_ids, h_x_ids, ids_bytes, cudaMemcpyHostToDevice, run_stream) != cudaSuccess) return false;
        if (src0_type == GGML_TYPE_IQ2_S) {
            if (iq2s_batch_mmvq) {
                return ggml_cuda_moe_stream_mmvq_batch_dev(
                    src0_type, cache->pool, ne01, ne00, (const float *)bc.d_src1_f32, d_src1_q8,
                    (float *)d_out, d_x_ids, n_active, cache->slot_sz, run_stream);
            }
            for (int j = 0; j < n_active; ++j) {
                const char *d_expert = (const char *)cache->pool + (size_t)h_x_ids[j] * cache->slot_sz;
                float *d_row = (float *)d_out + (size_t)j * ne01;
                const float *d_src1_row = (const float *)bc.d_src1_f32 + (size_t)j * ne00;
                if (!ggml_cuda_moe_stream_mmvq_dev(
                        src0_type, d_expert, ne01, ne00, d_src1_row, bc.d_src1_q8_one, d_row, run_stream)) {
                    return false;
                }
            }
            return true;
        }
        if (serial_up_gate) {
            bc.h_bounds[0] = 0;
            bc.h_bounds[1] = 1;
            if (cudaMemcpyAsync(bc.d_bounds, bc.h_bounds, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, run_stream) != cudaSuccess) return false;
            for (int j = 0; j < n_active; ++j) {
                bc.h_up_gate_ids_dst[0] = j;
                bc.h_up_gate_ids_dst[1] = h_x_ids[j];
                if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_up_gate_ids_dst, sizeof(int32_t), cudaMemcpyHostToDevice, run_stream) != cudaSuccess) return false;
                if (cudaMemcpyAsync(d_x_ids, bc.h_up_gate_ids_dst + 1, sizeof(int32_t), cudaMemcpyHostToDevice, run_stream) != cudaSuccess) return false;
                quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32 + (size_t)j * ne00, bc.d_ids_src1, bc.d_src1_q8_one,
                    src0_type, ne00, ne00, ne00, ne00,
                    ne00_padded, 1, 1, 1, run_stream);
                if (cudaGetLastError() != cudaSuccess) return false;
                if (!launch_moe_mmq_id_one(
                        src0_type,
                        (const char *)cache->pool, (const int *)bc.d_src1_q8_one, bc.d_ids_dst, bc.d_bounds,
                        d_x_ids, (float *)d_out, ne00, ne01, nb01, src0_bytes, dst_cols, run_stream)) {
                    return false;
                }
            }
            return true;
        }
        return launch_moe_mmq_id_batch(
            src0_type,
            (const char *)cache->pool, (const int *)d_src1_q8, bc.d_ids_dst, bc.d_bounds,
            d_x_ids, (float *)d_out, ne00, ne01, nb01, src0_bytes, n_active, dst_cols, run_stream);
    };

    struct stage_copy_job {
        int slot = -1;
        void *dst = nullptr;
        const void *host_data = nullptr;
        const expert_pack_entry *pack_entry = nullptr;
    };

    auto clear_stage_jobs = [&](const std::vector<stage_copy_job> &jobs) {
        for (const stage_copy_job &job : jobs) {
            batch_cache_clear_slot(cache, job.slot);
        }
    };

    auto plan_tensor = [&](
            const char *key_name, const void *host_base,
            int32_t *h_x_ids, int *slots_out,
            const int *avoid_slots, int n_avoid_slots,
            std::vector<stage_copy_job> &jobs) -> bool {
        jobs.clear();
        for (int j = 0; j < n_active; ++j) {
            const char *expert_host = (const char *)host_base + (size_t)active_experts[j] * nb02;
            const uintptr_t cache_key = batch_key_hash(key_name, active_experts[j]);
            int cache_slot = batch_cache_lookup_slot(cache, cache_key);
            if (cache_slot < 0) {
                cache_slot = batch_cache_insert_slot(
                    cache, cache_key, expert_host, src0_bytes, st, true, false,
                    avoid_slots, n_avoid_slots, false, key_name, active_experts[j]);
                if (cache_slot < 0) return false;
                const expert_pack_entry *pack_entry = expert_pack_lookup(key_name, active_experts[j], src0_bytes);
                void *dst_slot = (char *)cache->pool + (size_t)cache_slot * cache->slot_sz;
                jobs.push_back({cache_slot, dst_slot, expert_host, pack_entry});
            }
            h_x_ids[j] = cache_slot;
            if (slots_out) slots_out[j] = cache_slot;
            batch_route_profile_hit(key_name, active_experts[j], src0_bytes);
        }
        return true;
    };

    auto copy_stage_jobs = [&](const std::vector<stage_copy_job> &jobs, cudaStream_t run_stream, pinned_stage_ring &ring) -> bool {
        if (cudaSetDevice(0) != cudaSuccess) return false;
        for (const stage_copy_job &job : jobs) {
            if (!batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, job.pack_entry)) {
                if (!job.pack_entry ||
                        !batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, nullptr)) {
                    return false;
                }
            }
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        }
        return true;
    };

    auto split_stage_jobs = [&](const std::vector<stage_copy_job> &jobs,
            std::vector<stage_copy_job> &a, std::vector<stage_copy_job> &b) {
        a.clear();
        b.clear();
        a.reserve((jobs.size() + 1) / 2);
        b.reserve(jobs.size() / 2);
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (i & 1) {
                b.push_back(jobs[i]);
            } else {
                a.push_back(jobs[i]);
            }
        }
    };

    auto launch_tensor = [&](
            void *d_out, cudaStream_t run_stream, int32_t *d_x_ids, void *d_src1_q8,
            int32_t *h_x_ids) -> bool {
        if (cudaMemsetAsync(d_out, 0, dst_bytes, run_stream) != cudaSuccess) return false;
        if (cudaMemcpyAsync(d_x_ids, h_x_ids, ids_bytes, cudaMemcpyHostToDevice, run_stream) != cudaSuccess) return false;
        return ggml_cuda_moe_stream_mmvq_batch_dev(
            src0_type, cache->pool, ne01, ne00, (const float *)bc.d_src1_f32, d_src1_q8,
            (float *)d_out, d_x_ids, n_active, cache->slot_sz, run_stream);
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
    if (profile) cudaEventRecord(bc.ev_stage, st);

    const bool iq2s_batched_stage = src0_type == GGML_TYPE_IQ2_S && iq2s_batch_mmvq;
    if (!serial_up_gate && !iq2s_batched_stage) {
        quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32, bc.d_ids_src1, bc.d_src1_q8,
            src0_type, ne00, ne00, n_active * ne00, n_active * ne00,
            ne00_padded, n_active, 1, 1, st);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    if (profile) cudaEventRecord(bc.ev_quant, st);

    float * fused_d = use_handoff ? (float *)bc.d_handoff : (float *)bc.d_dst;
    if (parallel_up_gate) {
        if (cudaEventRecord(bc.ev_stage_ready, st) != cudaSuccess) return false;
        if (cudaStreamWaitEvent(bc.up_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (cudaStreamWaitEvent(bc.gate_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (bc.up_copy_stream && cudaStreamWaitEvent(bc.up_copy_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (bc.gate_copy_stream && cudaStreamWaitEvent(bc.gate_copy_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (profile && bc.ev_up_start) cudaEventRecord(bc.ev_up_start, bc.up_stream);
        auto parallel_fail = [&]() -> bool {
            cudaStreamSynchronize(bc.up_stream);
            cudaStreamSynchronize(bc.gate_stream);
            if (bc.up_copy_stream) cudaStreamSynchronize(bc.up_copy_stream);
            if (bc.gate_copy_stream) cudaStreamSynchronize(bc.gate_copy_stream);
            return false;
        };

        int up_slots[128] = {};
        if (parallel_stage) {
            std::vector<stage_copy_job> up_jobs;
            std::vector<stage_copy_job> gate_jobs;
            if (!plan_tensor(up_key_name, src0_up_data, bc.h_x_ids_up, up_slots, nullptr, 0, up_jobs)) {
                clear_stage_jobs(up_jobs);
                return parallel_fail();
            }
            if (!plan_tensor(gate_key_name, src0_gate_data, bc.h_x_ids_gate, nullptr, up_slots, n_active, gate_jobs)) {
                clear_stage_jobs(up_jobs);
                clear_stage_jobs(gate_jobs);
                return parallel_fail();
            }

            const char *stage_split_env = std::getenv("GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT");
            const bool stage_split = stage_split_env && stage_split_env[0] && stage_split_env[0] != '0' &&
                bc.up_copy_stream && bc.gate_copy_stream && bc.ev_up_copy_aux_done && bc.ev_gate_copy_aux_done;
            static std::atomic<int> first_stage_split{0};
            if (stage_split && first_stage_split.fetch_add(1) == 0) {
                std::fprintf(stderr, "[moe_stream] up/gate split CPU staging active\n");
            }

            if (stage_split) {
                std::vector<stage_copy_job> up_jobs_a;
                std::vector<stage_copy_job> up_jobs_b;
                std::vector<stage_copy_job> gate_jobs_a;
                std::vector<stage_copy_job> gate_jobs_b;
                split_stage_jobs(up_jobs, up_jobs_a, up_jobs_b);
                split_stage_jobs(gate_jobs, gate_jobs_a, gate_jobs_b);

                bool up_a_ok = true;
                bool up_b_ok = true;
                bool gate_a_ok = true;
                bool gate_b_ok = true;
                std::thread up_a_thread([&]() {
                    up_a_ok = copy_stage_jobs(up_jobs_a, bc.up_stream, bc.stage_ring);
                });
                std::thread up_b_thread([&]() {
                    up_b_ok = copy_stage_jobs(up_jobs_b, bc.up_copy_stream, bc.stage_ring_up_aux);
                });
                std::thread gate_a_thread([&]() {
                    gate_a_ok = copy_stage_jobs(gate_jobs_a, bc.gate_stream, bc.stage_ring_gate);
                });
                std::thread gate_b_thread([&]() {
                    gate_b_ok = copy_stage_jobs(gate_jobs_b, bc.gate_copy_stream, bc.stage_ring_gate_aux);
                });

                up_a_thread.join();
                up_b_thread.join();
                if (!up_a_ok || !up_b_ok) {
                    gate_a_thread.join();
                    gate_b_thread.join();
                    clear_stage_jobs(up_jobs);
                    clear_stage_jobs(gate_jobs);
                    return parallel_fail();
                }
                if (cudaEventRecord(bc.ev_up_copy_aux_done, bc.up_copy_stream) != cudaSuccess ||
                        cudaStreamWaitEvent(bc.up_stream, bc.ev_up_copy_aux_done, 0) != cudaSuccess) {
                    gate_a_thread.join();
                    gate_b_thread.join();
                    return parallel_fail();
                }
                if (!launch_tensor(bc.d_up, bc.up_stream, bc.d_x_ids_up, bc.d_src1_q8_up, bc.h_x_ids_up)) {
                    gate_a_thread.join();
                    gate_b_thread.join();
                    return parallel_fail();
                }
                if (profile) cudaEventRecord(bc.ev_up, bc.up_stream);

                if (profile && bc.ev_gate_start) cudaEventRecord(bc.ev_gate_start, bc.gate_stream);
                gate_a_thread.join();
                gate_b_thread.join();
                if (!gate_a_ok || !gate_b_ok) {
                    clear_stage_jobs(up_jobs);
                    clear_stage_jobs(gate_jobs);
                    return parallel_fail();
                }
                if (cudaEventRecord(bc.ev_gate_copy_aux_done, bc.gate_copy_stream) != cudaSuccess ||
                        cudaStreamWaitEvent(bc.gate_stream, bc.ev_gate_copy_aux_done, 0) != cudaSuccess) {
                    return parallel_fail();
                }
                if (!launch_tensor(bc.d_gate, bc.gate_stream, bc.d_x_ids_gate, bc.d_src1_q8_gate, bc.h_x_ids_gate)) {
                    return parallel_fail();
                }
            } else {
                bool up_copy_ok = true;
                bool gate_copy_ok = true;
                std::thread up_thread([&]() {
                    up_copy_ok = copy_stage_jobs(up_jobs, bc.up_stream, bc.stage_ring);
                });
                std::thread gate_thread([&]() {
                    gate_copy_ok = copy_stage_jobs(gate_jobs, bc.gate_stream, bc.stage_ring_gate);
                });

                up_thread.join();
                if (!up_copy_ok) {
                    gate_thread.join();
                    clear_stage_jobs(up_jobs);
                    clear_stage_jobs(gate_jobs);
                    return parallel_fail();
                }
                if (!launch_tensor(bc.d_up, bc.up_stream, bc.d_x_ids_up, bc.d_src1_q8_up, bc.h_x_ids_up)) {
                    gate_thread.join();
                    return parallel_fail();
                }
                if (profile) cudaEventRecord(bc.ev_up, bc.up_stream);

                if (profile && bc.ev_gate_start) cudaEventRecord(bc.ev_gate_start, bc.gate_stream);
                gate_thread.join();
                if (!gate_copy_ok) {
                    clear_stage_jobs(up_jobs);
                    clear_stage_jobs(gate_jobs);
                    return parallel_fail();
                }
                if (!launch_tensor(bc.d_gate, bc.gate_stream, bc.d_x_ids_gate, bc.d_src1_q8_gate, bc.h_x_ids_gate)) {
                    return parallel_fail();
                }
            }
        } else {
            if (!stage_tensor(
                    up_key_name, src0_up_data, bc.d_up, bc.up_stream,
                    bc.d_x_ids_up, bc.d_src1_q8_up, bc.h_x_ids_up, up_slots, nullptr, 0)) {
                return parallel_fail();
            }
            if (profile) cudaEventRecord(bc.ev_up, bc.up_stream);
            if (profile && bc.ev_gate_start) cudaEventRecord(bc.ev_gate_start, bc.gate_stream);
            if (!stage_tensor(
                    gate_key_name, src0_gate_data, bc.d_gate, bc.gate_stream,
                    bc.d_x_ids_gate, bc.d_src1_q8_gate, bc.h_x_ids_gate, nullptr, up_slots, n_active)) {
                return parallel_fail();
            }
        }
        if (profile) cudaEventRecord(bc.ev_gate, bc.gate_stream);
        if (cudaEventRecord(bc.ev_up_done, bc.up_stream) != cudaSuccess) return parallel_fail();
        if (cudaEventRecord(bc.ev_gate_done, bc.gate_stream) != cudaSuccess) return parallel_fail();
        if (cudaStreamWaitEvent(st, bc.ev_up_done, 0) != cudaSuccess) return parallel_fail();
        if (cudaStreamWaitEvent(st, bc.ev_gate_done, 0) != cudaSuccess) return parallel_fail();
    } else {
        if (!stage_tensor(
                up_key_name, src0_up_data, bc.d_up, st,
                bc.d_x_ids, bc.d_src1_q8, bc.h_x_ids, nullptr, nullptr, 0)) {
            return false;
        }
        if (profile) cudaEventRecord(bc.ev_up, st);
        if (!stage_tensor(
                gate_key_name, src0_gate_data, bc.d_gate, st,
                bc.d_x_ids, bc.d_src1_q8, bc.h_x_ids, nullptr, nullptr, 0)) {
            return false;
        }
        if (profile) cudaEventRecord(bc.ev_gate, st);
    }

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
    if (profile) cudaEventRecord(bc.ev_kernel, st);

    if (use_handoff) {
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        g_handoff.host_ptr = dst;
        g_handoff.d_data = fused_d;
        g_handoff.bytes = dst_bytes;
        g_handoff.ne01 = ne01;
        g_handoff.dst_cols = dst_cols;
        ++g_handoff.serial;
        if (profile) {
            float stage_ms = 0.0f;
            float quant_ms = 0.0f;
            float up_ms = 0.0f;
            float gate_ms = 0.0f;
            float fuse_ms = 0.0f;
            float kernel_ms = 0.0f;
            cudaEventElapsedTime(&stage_ms, bc.ev_start, bc.ev_stage);
            cudaEventElapsedTime(&quant_ms, bc.ev_stage, bc.ev_quant);
            if (parallel_up_gate && bc.ev_up_start && bc.ev_gate_start) {
                cudaEventElapsedTime(&up_ms, bc.ev_up_start, bc.ev_up);
                cudaEventElapsedTime(&gate_ms, bc.ev_gate_start, bc.ev_gate);
            } else {
                cudaEventElapsedTime(&up_ms, bc.ev_quant, bc.ev_up);
                cudaEventElapsedTime(&gate_ms, bc.ev_up, bc.ev_gate);
            }
            cudaEventElapsedTime(&fuse_ms, bc.ev_gate, bc.ev_kernel);
            cudaEventElapsedTime(&kernel_ms, bc.ev_quant, bc.ev_kernel);
            ++g_uprof.calls;
            g_uprof.active_experts += (uint64_t)n_active;
            g_uprof.stage_ms += stage_ms;
            g_uprof.quant_ms += quant_ms;
            g_uprof.up_ms += up_ms;
            g_uprof.gate_ms += gate_ms;
            g_uprof.fuse_ms += fuse_ms;
            g_uprof.kernel_ms += kernel_ms;
        }
        return true;
    }

    if (cudaMemcpyAsync(bc.h_dst, bc.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_d2h, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;

    float stage_ms = 0.0f;
    float quant_ms = 0.0f;
    float up_ms = 0.0f;
    float gate_ms = 0.0f;
    float fuse_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    if (profile) {
        cudaEventElapsedTime(&stage_ms, bc.ev_start, bc.ev_stage);
        cudaEventElapsedTime(&quant_ms, bc.ev_stage, bc.ev_quant);
        if (parallel_up_gate && bc.ev_up_start && bc.ev_gate_start) {
            cudaEventElapsedTime(&up_ms, bc.ev_up_start, bc.ev_up);
            cudaEventElapsedTime(&gate_ms, bc.ev_gate_start, bc.ev_gate);
        } else {
            cudaEventElapsedTime(&up_ms, bc.ev_quant, bc.ev_up);
            cudaEventElapsedTime(&gate_ms, bc.ev_up, bc.ev_gate);
        }
        cudaEventElapsedTime(&fuse_ms, bc.ev_gate, bc.ev_kernel);
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
        ++g_uprof.calls;
        g_uprof.active_experts += (uint64_t)n_active;
        g_uprof.stage_ms += stage_ms;
        g_uprof.quant_ms += quant_ms;
        g_uprof.up_ms += up_ms;
        g_uprof.gate_ms += gate_ms;
        g_uprof.fuse_ms += fuse_ms;
        g_uprof.kernel_ms += kernel_ms;
        g_uprof.d2h_ms += d2h_ms;
        g_uprof.scatter_ms += scatter_ms;
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
    ggml_type src0_type = (ggml_type)src0_type_int;
    if (!moe_stream_type_supported(src0_type) || !src1_f32) return false;
    ggml_cuda_moe_stream_register_tensor(src0_type_int, src0_name, src0_data, n_as, nb02, (size_t)ne01 * nb01);

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
    if (down_prefetch_enabled() && bc.prefetch_stream) {
        cudaStreamSynchronize(bc.prefetch_stream);
    }
    const bool profile = g_bprof.enabled && bc.ev_start && bc.ev_stage && bc.ev_quant && bc.ev_kernel && bc.ev_d2h;
    const char *down_parallel_stage_env = std::getenv("GGML_MOE_DOWN_PARALLEL_STAGE");
    const bool down_parallel_stage =
        down_parallel_stage_env && down_parallel_stage_env[0] && down_parallel_stage_env[0] != '0' &&
        bc.up_stream && bc.gate_stream && bc.ev_stage_ready && bc.ev_up_done && bc.ev_gate_done;
    static std::atomic<int> first_down_parallel_stage{0};
    if (down_parallel_stage && first_down_parallel_stage.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream_batch] down parallel CPU staging active\n");
    }

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

    struct down_stage_copy_job {
        int slot = -1;
        void *dst = nullptr;
        const void *host_data = nullptr;
        const expert_pack_entry *pack_entry = nullptr;
    };

    auto clear_down_stage_jobs = [&](const std::vector<down_stage_copy_job> &jobs) {
        for (const down_stage_copy_job &job : jobs) {
            batch_cache_clear_slot(cache, job.slot);
        }
    };

    auto copy_down_stage_jobs = [&](const std::vector<down_stage_copy_job> &jobs, cudaStream_t run_stream, pinned_stage_ring &ring) -> bool {
        if (cudaSetDevice(0) != cudaSuccess) return false;
        for (const down_stage_copy_job &job : jobs) {
            if (!batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, job.pack_entry)) {
                if (!job.pack_entry ||
                        !batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, nullptr)) {
                    return false;
                }
            }
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        }
        return true;
    };

    std::vector<down_stage_copy_job> down_jobs_a;
    std::vector<down_stage_copy_job> down_jobs_b;

    for (int j = 0; j < n_active; ++j) {
        const char *expert_host = (const char *)src0_data + (size_t)active_experts[j] * nb02;
        const uintptr_t cache_key = batch_key_hash(src0_name, active_experts[j]);
        batch_route_profile_hit(src0_name, active_experts[j], src0_bytes);
        int cache_slot = batch_cache_lookup_slot(cache, cache_key);
        if (cache_slot < 0) {
            cache_slot = batch_cache_insert_slot(cache, cache_key, expert_host, src0_bytes, st, true, false,
                    nullptr, 0, !down_parallel_stage, src0_name, active_experts[j]);
            if (down_parallel_stage && cache_slot >= 0) {
                const expert_pack_entry *pack_entry = expert_pack_lookup(src0_name, active_experts[j], src0_bytes);
                void *dst_slot = (char *)cache->pool + (size_t)cache_slot * cache->slot_sz;
                down_stage_copy_job job{cache_slot, dst_slot, expert_host, pack_entry};
                if ((int)(down_jobs_a.size() + down_jobs_b.size()) & 1) {
                    down_jobs_b.push_back(job);
                } else {
                    down_jobs_a.push_back(job);
                }
            }
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

    if (down_parallel_stage && (!down_jobs_a.empty() || !down_jobs_b.empty())) {
        bool copy_a_ok = true;
        bool copy_b_ok = true;
        std::thread copy_a([&]() {
            copy_a_ok = copy_down_stage_jobs(down_jobs_a, bc.up_stream, bc.stage_ring);
        });
        std::thread copy_b([&]() {
            copy_b_ok = copy_down_stage_jobs(down_jobs_b, bc.gate_stream, bc.stage_ring_gate);
        });
        copy_a.join();
        copy_b.join();
        if (!copy_a_ok || !copy_b_ok) {
            clear_down_stage_jobs(down_jobs_a);
            clear_down_stage_jobs(down_jobs_b);
            return false;
        }
        if (cudaEventRecord(bc.ev_up_done, bc.up_stream) != cudaSuccess) return false;
        if (cudaEventRecord(bc.ev_gate_done, bc.gate_stream) != cudaSuccess) return false;
        if (cudaStreamWaitEvent(st, bc.ev_up_done, 0) != cudaSuccess) return false;
        if (cudaStreamWaitEvent(st, bc.ev_gate_done, 0) != cudaSuccess) return false;
    }

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
            src0_type, ne00, g_handoff.ne01, g_handoff.ne01 * g_handoff.dst_cols, g_handoff.ne01 * g_handoff.dst_cols,
            ne00_padded, n_active, 1, 1, st);
        g_handoff.host_ptr = nullptr;
        g_handoff.d_data = nullptr;
    } else {
        quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32, bc.d_ids_src1, bc.d_src1_q8,
            src0_type, ne00, ne00, n_active * ne00, n_active * ne00,
            ne00_padded, n_active, 1, 1, st);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_quant, st);

    if (!launch_moe_mmq_id_batch(
            src0_type,
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
