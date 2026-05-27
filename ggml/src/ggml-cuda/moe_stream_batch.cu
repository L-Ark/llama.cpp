// Decode-only batched streaming MoE path.
//
// This file intentionally includes the MMQ-id kernel family without the MMVQ
// headers used by moe_stream.cu; several CUDA helper headers define unguarded
// device functions and cannot be mixed in one translation unit.

#include "common.cuh"
#include "ggml-quants.h"
#include "iqk/iqk_mul_mat.h"
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

#define MOE_STREAM_MAX_ACTIVE 512

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

extern "C" {

typedef struct {
    int32_t i1;
    int32_t i2;
} ggml_moe_row_mapping;

typedef struct {
    float direct_up;
    float direct_gate;
    float direct_fused;
    float repack_up;
    float repack_gate;
    float repack_fused;
} ggml_moe_iq2_replay_result;

bool ggml_cuda_moe_iq2_prompt_replay(
        int src0_type_int,
        const void *up_expert,
        const void *gate_expert,
        int64_t ne01,
        int64_t ne00,
        size_t nb01,
        const float *src1_row_f32,
        int64_t col,
        int unary_op,
        float limit,
        ggml_moe_iq2_replay_result *out);

void ggml_cuda_moe_stream_batch_link_anchor(void) {
    (void)&ggml_cuda_moe_iq2_prompt_replay;
}

void ggml_cuda_moe_ttft_trace_mark(const char *label);

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
bool ggml_cuda_moe_stream_preload_tensor_prompt(
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
    cudaEvent_t copy_start = nullptr;
    cudaEvent_t copy_done = nullptr;
    bool pending = false;
    bool timing_pending = false;
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
    uint64_t h2d_timed = 0;
    double slot_wait_ms = 0.0;
    double host_stage_ms = 0.0;
    double enqueue_ms = 0.0;
    double h2d_ms = 0.0;
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
    void * d_src1_q8k = nullptr; size_t d_src1_q8k_sz = 0;
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
    cudaEvent_t ev_gate_work_start = nullptr;
    cudaEvent_t ev_up_compute_start = nullptr;
    cudaEvent_t ev_gate_compute_start = nullptr;
    cudaEvent_t ev_gate_start = nullptr;
    cudaEvent_t ev_up = nullptr;
    cudaEvent_t ev_gate = nullptr;
    int32_t h_ids_dst[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_up_gate_ids_dst[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_ids_src1[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_x_ids[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_x_ids_up[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_x_ids_gate[MOE_STREAM_MAX_ACTIVE] = {};
    int32_t h_bounds[MOE_STREAM_MAX_ACTIVE + 1] = {};
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
    uint64_t up_stage_jobs = 0;
    uint64_t gate_stage_jobs = 0;
    double stage_ms = 0.0;
    double quant_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double scatter_ms = 0.0;
    double up_ms = 0.0;
    double gate_ms = 0.0;
    double up_wait_ms = 0.0;
    double gate_wait_ms = 0.0;
    double up_compute_ms = 0.0;
    double gate_compute_ms = 0.0;
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
    // RAM hot tier
    void *ram_tier_base = nullptr;
    size_t ram_tier_bytes = 0;
    struct ram_tier_entry { size_t offset; size_t nbytes; };
    std::vector<ram_tier_entry> ram_tier_index; // parallel to entries[], non-zero offset = resident
    std::atomic<uint64_t> ram_tier_hits{0};
    std::atomic<uint64_t> ram_tier_total{0};
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
        "stage=%.3f ms quant=%.3f ms up=%.3f ms gate=%.3f ms "
        "up_stage_jobs=%.2f gate_stage_jobs=%.2f "
        "up_wait=%.3f ms gate_wait=%.3f ms up_compute=%.3f ms gate_compute=%.3f ms fuse=%.3f ms "
        "kernel=%.3f ms d2h=%.3f ms scatter=%.3f ms total=%.3f ms/call\n",
        g_uprof.calls,
        (double)g_uprof.active_experts / calls,
        g_uprof.stage_ms / calls,
        g_uprof.quant_ms / calls,
        g_uprof.up_ms / calls,
        g_uprof.gate_ms / calls,
        (double)g_uprof.up_stage_jobs / calls,
        (double)g_uprof.gate_stage_jobs / calls,
        g_uprof.up_wait_ms / calls,
        g_uprof.gate_wait_ms / calls,
        g_uprof.up_compute_ms / calls,
        g_uprof.gate_compute_ms / calls,
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
static std::vector<profile_entry> g_prompt_profile;
static bool g_prompt_profile_loaded = false;
static bool g_prompt_profile_enabled = false;
static std::mutex g_profile_mu;
static char g_preloaded_tensors[256][96] = {};
static int g_n_preloaded_tensors = 0;
static std::atomic<int> g_profile_preload_calls{0};

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

struct batch_ttft_trace_entry {
    uint64_t seq = 0;
    double t_ms = 0.0;
    double copy_ms = 0.0;
    int expert_idx = -1;
    int cache_hit = 0;
    int pack_hit = 0;
    int ram_hit = 0;
    size_t expert_bytes = 0;
    char op[24] = {};
    char tensor[96] = {};
};

static std::vector<batch_route_profile_entry> g_route_profile;
static std::vector<batch_route_trace_entry> g_route_trace;
static std::mutex g_route_profile_mu;
static const char * g_route_profile_out = nullptr;
static const char * g_route_trace_out = nullptr;
static bool g_route_profile_inited = false;
static uint64_t g_route_trace_seq = 0;
static std::vector<batch_ttft_trace_entry> g_ttft_trace;
static std::mutex g_ttft_trace_mu;
static const char * g_ttft_trace_out = nullptr;
static std::atomic<bool> g_ttft_trace_inited{false};
static size_t g_ttft_trace_cap = 100000;
static uint64_t g_ttft_trace_seq = 0;
static std::chrono::steady_clock::time_point g_ttft_trace_start;

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

static void batch_ttft_trace_report_atexit() {
    std::vector<batch_ttft_trace_entry> rows;
    {
        std::lock_guard<std::mutex> lk(g_ttft_trace_mu);
        rows = g_ttft_trace;
    }
    if (!g_ttft_trace_out || !g_ttft_trace_out[0] || rows.empty()) return;

    FILE *f = std::fopen(g_ttft_trace_out, "w");
    if (!f) {
        std::fprintf(stderr, "[moe_stream_batch] TTFT trace: open failed: %s\n", g_ttft_trace_out);
        return;
    }
    std::fprintf(f, "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n");
    for (const batch_ttft_trace_entry &e : rows) {
        std::fprintf(f, "%lu,%.3f,%s,%s,%d,%zu,%d,%d,%d,%.3f\n",
                     e.seq, e.t_ms, e.op, e.tensor, e.expert_idx, e.expert_bytes,
                     e.cache_hit, e.pack_hit, e.ram_hit, e.copy_ms);
    }
    std::fclose(f);
    std::fprintf(stderr, "[moe_stream_batch] TTFT trace written: %s (%zu events)\n",
                 g_ttft_trace_out, rows.size());
}

static bool batch_ttft_trace_enabled() {
    if (g_ttft_trace_inited.load(std::memory_order_acquire)) return g_ttft_trace_out && g_ttft_trace_out[0];
    std::lock_guard<std::mutex> lk(g_ttft_trace_mu);
    if (g_ttft_trace_inited.load(std::memory_order_acquire)) return g_ttft_trace_out && g_ttft_trace_out[0];
    g_ttft_trace_out = std::getenv("GGML_MOE_TTFT_TRACE_OUT");
    const char *cap_env = std::getenv("GGML_MOE_TTFT_TRACE_MAX_EVENTS");
    if (cap_env && cap_env[0]) {
        const long cap = std::atol(cap_env);
        if (cap > 0) g_ttft_trace_cap = (size_t)cap;
    }
    if (g_ttft_trace_out && g_ttft_trace_out[0]) {
        g_ttft_trace_start = std::chrono::steady_clock::now();
        std::atexit(batch_ttft_trace_report_atexit);
    }
    g_ttft_trace_inited.store(true, std::memory_order_release);
    return g_ttft_trace_out && g_ttft_trace_out[0];
}

static void batch_ttft_trace_record(
        const char *op,
        const char *tensor_name,
        int expert_idx,
        size_t expert_bytes,
        bool cache_hit,
        bool pack_hit,
        bool ram_hit,
        double copy_ms) {
    if (!batch_ttft_trace_enabled()) return;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_ttft_trace_mu);
    if (g_ttft_trace.size() >= g_ttft_trace_cap) return;

    batch_ttft_trace_entry e;
    e.seq = ++g_ttft_trace_seq;
    e.t_ms = std::chrono::duration<double, std::milli>(now - g_ttft_trace_start).count();
    e.copy_ms = copy_ms;
    e.expert_idx = expert_idx;
    e.cache_hit = cache_hit ? 1 : 0;
    e.pack_hit = pack_hit ? 1 : 0;
    e.ram_hit = ram_hit ? 1 : 0;
    e.expert_bytes = expert_bytes;
    std::snprintf(e.op, sizeof(e.op), "%s", op ? op : "");
    std::snprintf(e.tensor, sizeof(e.tensor), "%s", tensor_name ? tensor_name : "");
    g_ttft_trace.push_back(e);
}

extern "C" void ggml_cuda_moe_ttft_trace_mark(const char *label) {
    char op[24] = {};
    std::snprintf(op, sizeof(op), "mark_%s", label ? label : "");
    batch_ttft_trace_record(op, "", -1, 0, false, false, false, 0.0);
}

extern "C" void ggml_cuda_moe_ttft_trace_event(
        const char *op,
        const char *tensor_name,
        int marker_value,
        size_t expert_bytes,
        double elapsed_ms) {
    batch_ttft_trace_record(op, tensor_name, marker_value, expert_bytes, false, false, false, elapsed_ms);
}

struct batch_ttft_call_scope {
    const char *op = nullptr;
    const char *tensor = nullptr;
    int n_active = 0;
    size_t expert_bytes = 0;
    bool enabled = false;
    std::chrono::steady_clock::time_point start;

    batch_ttft_call_scope(const char *op_, const char *tensor_, int n_active_, size_t expert_bytes_)
        : op(op_), tensor(tensor_), n_active(n_active_), expert_bytes(expert_bytes_),
          enabled(batch_ttft_trace_enabled()),
          start(enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {
    }

    ~batch_ttft_call_scope() {
        if (!enabled) return;
        const auto end = std::chrono::steady_clock::now();
        batch_ttft_trace_record(
            op,
            tensor,
            n_active,
            expert_bytes,
            false,
            false,
            false,
            std::chrono::duration<double, std::milli>(end - start).count());
    }
};

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

static size_t prompt_profile_preload_scan_budget(const batch_vram_cache *c) {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PROMPT_MAX_ENTRIES");
    if (env && env[0]) {
        long max_entries = std::atol(env);
        if (max_entries > 0) return (size_t)max_entries;
    }
    return profile_preload_slot_budget(c);
}

static bool profile_preload_evict_enabled() {
    const char *env = std::getenv("GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT");
    return env && env[0] && env[0] != '0';
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
    static bool max_inited = false;
    static long max_tensors = 0;
    if (!max_inited) {
        const char *max_env = std::getenv("GGML_MOE_VRAM_PROFILE_PRELOAD_MAX_TENSORS");
        max_tensors = (max_env && max_env[0]) ? std::atol(max_env) : 0;
        max_inited = true;
    }
    if (max_tensors > 0 && g_n_preloaded_tensors >= max_tensors) return true;
    if (g_n_preloaded_tensors < (int)(sizeof(g_preloaded_tensors) / sizeof(g_preloaded_tensors[0]))) {
        std::snprintf(g_preloaded_tensors[g_n_preloaded_tensors++], sizeof(g_preloaded_tensors[0]), "%s", name);
    }
    return false;
}

static bool profile_has_tensor_locked(const std::vector<profile_entry> &entries, const char *name) {
    if (!name || !name[0]) return false;
    for (const profile_entry &e : entries) {
        if (std::strcmp(e.tensor, name) == 0) return true;
    }
    return false;
}

static bool load_profile_file(const char *path, std::vector<profile_entry> &entries, const char *label) {
    if (!path || !path[0]) {
        return false;
    }
    FILE *f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "[moe_stream_batch] %s preload: open failed: %s\n", label, path);
        return false;
    }
    char line[512];
    if (!std::fgets(line, sizeof(line), f)) {
        std::fclose(f);
        return false;
    }
    while (std::fgets(line, sizeof(line), f)) {
        profile_entry e;
        unsigned long long rank = 0, count = 0, cumulative = 0, tensor_base = 0;
        size_t expert_bytes = 0;
        if (std::sscanf(line, "%llu,%llu,%zu,%llu,0x%llx,%d,%95[^\n]",
                    &rank, &count, &expert_bytes, &cumulative, &tensor_base, &e.expert_idx, e.tensor) == 7 &&
                e.expert_idx >= 0 && e.tensor[0]) {
            e.expert_bytes = expert_bytes;
            entries.push_back(e);
        }
    }
    std::fclose(f);
    if (!entries.empty()) {
        std::fprintf(stderr, "[moe_stream_batch] %s preload: loaded %zu entries from %s\n",
                     label, entries.size(), path);
    }
    return !entries.empty();
}

static void load_profile_once() {
    if (g_profile_loaded) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    if (g_profile_loaded) return;
    const char *path = std::getenv("GGML_MOE_VRAM_PROFILE");
    g_profile_enabled = load_profile_file(path, g_profile, "profile");
    g_profile_loaded = true;
}

static void load_prompt_profile_once() {
    if (g_prompt_profile_loaded) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    if (g_prompt_profile_loaded) return;
    const char *path = std::getenv("GGML_MOE_VRAM_PROFILE_PROMPT");
    g_prompt_profile_enabled = load_profile_file(path, g_prompt_profile, "prompt profile");
    g_prompt_profile_loaded = true;
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

static size_t env_mib_or_default(const char *name, size_t fallback) {
    const char *env = std::getenv(name);
    if (!env || !env[0]) return fallback;
    return (size_t)std::strtoull(env, nullptr, 10);
}

static size_t batch_cache_effective_budget_mib(size_t requested_mib) {
    static bool inited = false;
    static size_t effective_mib = 0;
    if (inited) return effective_mib;

    effective_mib = requested_mib;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t info_err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (info_err != cudaSuccess) {
        std::fprintf(stderr,
            "[moe_stream_batch] VRAM cache budget: requested=%zu MiB actual=%zu MiB cudaMemGetInfo failed: %s\n",
            requested_mib, effective_mib, cudaGetErrorString(info_err));
        cudaGetLastError();
        inited = true;
        return effective_mib;
    }

    const size_t free_mib = free_bytes / (1024ULL * 1024ULL);
    const size_t total_mib = total_bytes / (1024ULL * 1024ULL);
    const size_t graph_reserve_mib = env_mib_or_default("GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB", 0);
    const size_t safety_mib = env_mib_or_default("GGML_MOE_VRAM_CACHE_SAFETY_MIB", graph_reserve_mib > 0 ? 128 : 0);
    const char *clamp_env = std::getenv("GGML_MOE_VRAM_CACHE_AUTO_CLAMP");
    const bool clamp = (clamp_env && clamp_env[0] && clamp_env[0] != '0') || graph_reserve_mib > 0;

    if (clamp) {
        const size_t reserved_mib = graph_reserve_mib + safety_mib;
        const size_t max_cache_mib = free_mib > reserved_mib ? free_mib - reserved_mib : 0;
        if (effective_mib > max_cache_mib) {
            effective_mib = max_cache_mib;
        }
    }

    std::fprintf(stderr,
        "[moe_stream_batch] VRAM cache budget: requested=%zu MiB actual=%zu MiB free=%zu MiB total=%zu MiB graph_reserve=%zu MiB safety=%zu MiB clamp=%d\n",
        requested_mib, effective_mib, free_mib, total_mib, graph_reserve_mib, safety_mib, clamp ? 1 : 0);
    inited = true;
    return effective_mib;
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
    budget_mib = batch_cache_effective_budget_mib(budget_mib);
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
    if (g_expert_pack.ram_tier_base) {
        std::fprintf(stderr,
                     "[moe_stream_batch] RAM tier: hits=%lu total=%lu hit_rate=%.1f%% resident=%.2f MiB\n",
                     g_expert_pack.ram_tier_hits.load(), g_expert_pack.ram_tier_total.load(),
                     g_expert_pack.ram_tier_total.load() > 0 ?
                         100.0 * g_expert_pack.ram_tier_hits.load() / g_expert_pack.ram_tier_total.load() : 0.0,
                     g_expert_pack.ram_tier_bytes / (1024.0 * 1024.0));
    }
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
    const char *disable_env = std::getenv("GGML_MOE_EXPERT_PACK_RUNTIME_DISABLE");
    if (disable_env && disable_env[0] && disable_env[0] != '0') return nullptr;
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

static std::once_flag g_ram_tier_once;

static void ram_tier_init() {
    if (!g_expert_pack.enabled) return;
    const char *ram_tier_env = std::getenv("GGML_MOE_RAM_TIER_MIB");
    if (!ram_tier_env || !ram_tier_env[0] || ram_tier_env[0] == '0') return;
    const size_t budget = (size_t)std::atol(ram_tier_env) * 1024ULL * 1024ULL;
    if (budget == 0) return;

    load_profile_once();
    if (g_profile.empty()) {
        std::fprintf(stderr, "[moe_stream_batch] RAM tier: no profile loaded, skipping\n");
        return;
    }

    // Skip the top entries that are already in the VRAM cache.
    // The VRAM cache preloads ~n_slots entries per tensor from the profile.
    // Default skip = 500 (approximate VRAM slot count); override with GGML_MOE_RAM_TIER_SKIP.
    size_t skip = 500;
    const char *skip_env = std::getenv("GGML_MOE_RAM_TIER_SKIP");
    if (skip_env && skip_env[0]) skip = (size_t)std::atol(skip_env);

    g_expert_pack.ram_tier_index.resize(g_expert_pack.entries.size());
    void *region = ::mmap(nullptr, budget, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) {
        std::fprintf(stderr, "[moe_stream_batch] RAM tier: mmap %zu MiB failed\n", budget / (1024*1024));
        return;
    }
    g_expert_pack.ram_tier_base = region;
    g_expert_pack.ram_tier_bytes = budget;
    size_t loaded = 0, n_loaded = 0, n_skipped = 0;
    for (const profile_entry &pe : g_profile) {
        if (loaded >= budget) break;
        if (n_skipped < skip) { ++n_skipped; continue; }
        size_t lo2 = 0, hi2 = g_expert_pack.entries.size();
        while (lo2 < hi2) {
            size_t mid2 = lo2 + (hi2 - lo2) / 2;
            int cmp2 = expert_pack_entry_cmp(g_expert_pack.entries[mid2], pe.tensor, pe.expert_idx, pe.expert_bytes);
            if (cmp2 < 0) lo2 = mid2 + 1; else hi2 = mid2;
        }
        if (lo2 >= g_expert_pack.entries.size() ||
            expert_pack_entry_cmp(g_expert_pack.entries[lo2], pe.tensor, pe.expert_idx, pe.expert_bytes) != 0)
            continue;
        const expert_pack_entry &ent = g_expert_pack.entries[lo2];
        if (loaded + ent.nbytes > budget) break;
        char *dst_ptr = (char *)region + loaded;
        {
            std::lock_guard<std::mutex> lk2(g_expert_pack.mu);
            if (::fseeko(g_expert_pack.file, (off_t)ent.offset, SEEK_SET) != 0) continue;
            if (!expert_pack_read_exact(g_expert_pack.file, dst_ptr, (size_t)ent.nbytes)) continue;
        }
        g_expert_pack.ram_tier_index[lo2] = {loaded, (size_t)ent.nbytes};
        loaded += (size_t)ent.nbytes;
        ++n_loaded;
    }
    std::fprintf(stderr, "[moe_stream_batch] RAM tier: loaded %zu entries (skipped %zu), %.2f MiB into anonymous mmap\n",
                 n_loaded, skip, loaded / (1024.0 * 1024.0));
    if (loaded > 0) {
        cudaError_t err = cudaHostRegister(region, loaded, cudaHostRegisterDefault);
        if (err == cudaSuccess) {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister succeeded (%.2f MiB pinned)\n",
                         loaded / (1024.0 * 1024.0));
        } else {
            std::fprintf(stderr, "[moe_stream_batch] RAM tier: cudaHostRegister failed (%s), using unpinned path\n",
                         cudaGetErrorString(err));
        }
    }
}

static bool expert_pack_read_entry(const expert_pack_entry *entry, void *dst, size_t sz) {
    if (!entry || !g_expert_pack.file || entry->nbytes != sz) return false;

    std::call_once(g_ram_tier_once, ram_tier_init);

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

static bool pinned_stage_profile_enabled() {
    return g_bprof.enabled || g_uprof.enabled;
}

struct batch_copy_trace {
    bool pack_hit = false;
    bool ram_hit = false;
};

static void pinned_stage_collect_timing(pinned_stage_ring &ring, pinned_stage_slot &slot) {
    if (!slot.timing_pending || !slot.copy_start || !slot.copy_done) return;
    if (cudaEventSynchronize(slot.copy_done) == cudaSuccess) {
        float h2d_ms = 0.0f;
        if (cudaEventElapsedTime(&h2d_ms, slot.copy_start, slot.copy_done) == cudaSuccess) {
            ring.h2d_ms += h2d_ms;
            ++ring.h2d_timed;
        }
    }
    slot.timing_pending = false;
}

static void pinned_stage_release(pinned_stage_ring &ring) {
    for (pinned_stage_slot &slot : ring.slots) {
        if (slot.pending && slot.done) {
            cudaEventSynchronize(slot.done);
            slot.pending = false;
        }
        pinned_stage_collect_timing(ring, slot);
        if (slot.host) {
            cudaFreeHost(slot.host);
            slot.host = nullptr;
        }
        if (slot.done) {
            cudaEventDestroy(slot.done);
            slot.done = nullptr;
        }
        if (slot.copy_start) {
            cudaEventDestroy(slot.copy_start);
            slot.copy_start = nullptr;
        }
        if (slot.copy_done) {
            cudaEventDestroy(slot.copy_done);
            slot.copy_done = nullptr;
        }
    }
    ring.slots.clear();
    ring.slot_sz = 0;
    ring.next = 0;
}

static void pinned_stage_report_atexit() {
    auto report_ring = [](const char *name, pinned_stage_ring &ring) {
        for (pinned_stage_slot &slot : ring.slots) {
            if (slot.pending && slot.done) {
                cudaEventSynchronize(slot.done);
                slot.pending = false;
            }
            pinned_stage_collect_timing(ring, slot);
        }
        if (ring.copies == 0 && ring.fallbacks == 0) return;
        if (ring.h2d_timed > 0 || ring.host_stage_ms > 0.0 || ring.enqueue_ms > 0.0 || ring.slot_wait_ms > 0.0) {
            std::fprintf(stderr,
                "[moe_stream_batch] pinned staging%s: copies=%lu waits=%lu fallbacks=%lu slots=%zu slot=%.2f MiB "
                "slot_wait=%.3f ms host_stage=%.3f ms enqueue=%.3f ms h2d=%.3f ms h2d_timed=%lu\n",
                name, ring.copies, ring.waits, ring.fallbacks, ring.slots.size(), ring.slot_sz / (1024.0 * 1024.0),
                ring.slot_wait_ms, ring.host_stage_ms, ring.enqueue_ms, ring.h2d_ms, ring.h2d_timed);
        } else {
            std::fprintf(stderr,
                "[moe_stream_batch] pinned staging%s: copies=%lu waits=%lu fallbacks=%lu slots=%zu slot=%.2f MiB\n",
                name, ring.copies, ring.waits, ring.fallbacks, ring.slots.size(), ring.slot_sz / (1024.0 * 1024.0));
        }
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
    const bool profile_stage = pinned_stage_profile_enabled();
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
        if (profile_stage) {
            if (cudaEventCreate(&slot.copy_start) != cudaSuccess ||
                    cudaEventCreate(&slot.copy_done) != cudaSuccess) {
                if (slot.copy_start) {
                    cudaEventDestroy(slot.copy_start);
                    slot.copy_start = nullptr;
                }
                if (slot.copy_done) {
                    cudaEventDestroy(slot.copy_done);
                    slot.copy_done = nullptr;
                }
            }
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
        const expert_pack_entry *pack_entry,
        batch_copy_trace *copy_trace = nullptr) {
    if (copy_trace) {
        copy_trace->pack_hit = pack_entry != nullptr;
        copy_trace->ram_hit = false;
    }

    // RAM tier fast path: if the entry is resident in the registered mmap,
    // do a direct H2D from the pinned region, bypassing the staging slot.
    if (pack_entry && g_expert_pack.ram_tier_base != nullptr) {
        std::call_once(g_ram_tier_once, ram_tier_init);
        const size_t idx = (size_t)(pack_entry - g_expert_pack.entries.data());
        if (idx < g_expert_pack.ram_tier_index.size()) {
            const auto &rt = g_expert_pack.ram_tier_index[idx];
            if (rt.nbytes == sz) {
                const void *src = (const char *)g_expert_pack.ram_tier_base + rt.offset;
                if (cudaMemcpyAsync(dst, src, sz, cudaMemcpyHostToDevice, st) == cudaSuccess) {
                    ++g_expert_pack.ram_tier_hits;
                    ++g_expert_pack.ram_tier_total;
                    if (copy_trace) copy_trace->ram_hit = true;
                    return true;
                }
            }
        }
        ++g_expert_pack.ram_tier_total;
    }

    const bool use_pinned_stage = stage_pinned_enabled() || pack_entry;
    if (use_pinned_stage) {
        if (pinned_stage_ensure(ring, sz, pack_entry != nullptr)) {
            const bool profile_stage = pinned_stage_profile_enabled();
            pinned_stage_slot &slot = ring.slots[ring.next++ % ring.slots.size()];
            if (slot.pending) {
                const auto wait_start = profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (cudaEventSynchronize(slot.done) != cudaSuccess) return false;
                if (profile_stage) {
                    const auto wait_end = std::chrono::steady_clock::now();
                    ring.slot_wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
                }
                slot.pending = false;
                ++ring.waits;
            }
            pinned_stage_collect_timing(ring, slot);

            const auto host_start = profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (pack_entry) {
                if (!expert_pack_read_entry(pack_entry, slot.host, sz)) {
                    return false;
                }
            } else {
                std::memcpy(slot.host, host_data, sz);
            }
            if (profile_stage) {
                const auto host_end = std::chrono::steady_clock::now();
                ring.host_stage_ms += std::chrono::duration<double, std::milli>(host_end - host_start).count();
            }
            const auto enqueue_start = profile_stage ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (profile_stage && slot.copy_start) {
                if (cudaEventRecord(slot.copy_start, st) != cudaSuccess) return false;
            }
            if (cudaMemcpyAsync(dst, slot.host, sz, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
            if (profile_stage && slot.copy_done) {
                if (cudaEventRecord(slot.copy_done, st) != cudaSuccess) return false;
                slot.timing_pending = true;
            }
            if (cudaEventRecord(slot.done, st) != cudaSuccess) {
                cudaStreamSynchronize(st);
                return false;
            }
            if (profile_stage) {
                const auto enqueue_end = std::chrono::steady_clock::now();
                ring.enqueue_ms += std::chrono::duration<double, std::milli>(enqueue_end - enqueue_start).count();
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
    return batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, pack_entry, nullptr);
}

static int batch_cache_insert_slot(
        batch_vram_cache *c, uintptr_t key, const void *host_data, size_t sz, cudaStream_t st,
        bool allow_evict, bool preload, const int *avoid_slots = nullptr, int n_avoid_slots = 0,
        bool do_copy = true, const char *tensor_name = nullptr, int expert_idx = -1,
        bool prefetch_down = false, bool pin_preload = true) {
    if (!c || !c->pool || c->n_slots == 0 || sz > c->slot_sz) return -1;
    const bool pin_slot = preload && pin_preload && profile_protect_enabled();
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
        batch_copy_trace copy_trace;
        const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, pack_entry, &copy_trace)) {
            if (pack_entry && !batch_cache_copy_h2d(g_batch.stage_ring, dst, host_data, sz, st, nullptr, &copy_trace)) {
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
        if (copy_start != std::chrono::steady_clock::time_point{}) {
            const auto copy_end = std::chrono::steady_clock::now();
            batch_ttft_trace_record(
                preload ? "preload_load" : "runtime_load",
                tensor_name,
                expert_idx,
                sz,
                false,
                copy_trace.pack_hit,
                copy_trace.ram_hit,
                std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
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

static void preload_profile_entries_for_tensor(
        const std::vector<profile_entry> &entries,
        const char *label,
        const char *tensor_name,
        const void *src0_data,
        int64_t n_as,
        size_t nb02,
        size_t src0_bytes,
        cudaStream_t st,
        bool pin_preload,
        bool allow_evict,
        bool apply_skip,
        bool track_tensor,
        size_t scan_budget) {
    if (entries.empty() || !tensor_name || !tensor_name[0]) return;
    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return;
    static bool skip_inited = false;
    static long skip_calls = 0;
    if (apply_skip && !skip_inited) {
        const char *skip_env = std::getenv("GGML_MOE_VRAM_PROFILE_SKIP_FIRST_PRELOADS");
        skip_calls = (skip_env && skip_env[0]) ? std::atol(skip_env) : 0;
        if (skip_calls < 0) skip_calls = 0;
        skip_inited = true;
    }
    const int call_idx = g_profile_preload_calls.fetch_add(1);
    if (apply_skip && skip_calls > 0 && call_idx < skip_calls) return;
    if (track_tensor && tensor_already_preloaded(tensor_name)) return;
    char lookup_name[96] = {};
    profile_lookup_name(tensor_name, lookup_name, sizeof(lookup_name));
    int loaded = 0;
    const size_t preload_budget = scan_budget > 0 ? scan_budget : profile_preload_slot_budget(cache);
    const int cache_id = batch_cache_id_for_size(src0_bytes);
    const bool use_lookup = !profile_has_tensor_locked(entries, tensor_name);
    size_t seen_for_cache = 0;
    for (size_t ip = 0; ip < entries.size(); ++ip) {
        const profile_entry &e = entries[ip];
        if (e.expert_bytes != 0 && batch_cache_id_for_size(e.expert_bytes) != cache_id) continue;
        if (seen_for_cache++ >= preload_budget) break;
        if (std::strcmp(e.tensor, tensor_name) != 0 && std::strcmp(e.tensor, lookup_name) != 0) continue;
        if (!use_lookup && std::strcmp(e.tensor, tensor_name) != 0) continue;
        if (e.expert_idx < 0 || e.expert_idx >= n_as) continue;
        const uintptr_t key = batch_key_hash(tensor_name, e.expert_idx);
        if (batch_cache_find_slot(cache, key) >= 0) continue;
        const char *expert_host = (const char *)src0_data + (size_t)e.expert_idx * nb02;
        if (batch_cache_insert_slot(cache, key, expert_host, src0_bytes, st, allow_evict, true,
                nullptr, 0, true, tensor_name, e.expert_idx, false, pin_preload) < 0) break;
        ++loaded;
    }
    if (loaded > 0) {
        std::fprintf(stderr, "[moe_stream_batch] %s preload: %s loaded=%d\n", label, tensor_name, loaded);
    }
}

static void preload_profile_for_tensor(
        const char *tensor_name, const void *src0_data, int64_t n_as, size_t nb02, size_t src0_bytes, cudaStream_t st) {
    load_profile_once();
    if (!g_profile_enabled) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    preload_profile_entries_for_tensor(
        g_profile, "profile", tensor_name, src0_data, n_as, nb02, src0_bytes, st,
        true, profile_preload_evict_enabled(), true, true, profile_preload_slot_budget(batch_cache_get(src0_bytes)));
}

static void preload_prompt_profile_for_tensor(
        const char *tensor_name, const void *src0_data, int64_t n_as, size_t nb02, size_t src0_bytes, cudaStream_t st) {
    load_prompt_profile_once();
    if (!g_prompt_profile_enabled) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    preload_profile_entries_for_tensor(
        g_prompt_profile, "prompt profile", tensor_name, src0_data, n_as, nb02, src0_bytes, st,
        false, false, false, false, prompt_profile_preload_scan_budget(batch_cache_get(src0_bytes)));
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

static __device__ __forceinline__ int moe_q8k_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static __global__ void moe_quantize_row_q8_k_kernel(const float * x, block_q8_K * y, int nblocks) {
    const int ib = blockIdx.x;
    if (ib >= nblocks || threadIdx.x != 0) {
        return;
    }

    const float * xb = x + (size_t)ib * QK_K;
    block_q8_K * yb = y + ib;

    float max = 0.0f;
    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }

    if (amax == 0.0f) {
        yb->d = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            yb->qs[j] = 0;
        }
        for (int j = 0; j < QK_K/16; ++j) {
            yb->bsums[j] = 0;
        }
        return;
    }

    const float iscale = -127.0f / max;
    for (int j = 0; j < QK_K; ++j) {
        int v = moe_q8k_nearest_int(iscale * xb[j]);
        if (v > 127) {
            v = 127;
        }
        yb->qs[j] = (int8_t)v;
    }
    for (int j = 0; j < QK_K/16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) {
            sum += yb->qs[j*16 + ii];
        }
        yb->bsums[j] = (int16_t)sum;
    }
    yb->d = 1.0f / iscale;
}

static __device__ __forceinline__ int moe_iq3_xxs_q8k_block_sum(
        const block_iq3_xxs * x,
        const block_q8_K * y) {
    const uint8_t * q3 = x->qs;
    const uint8_t * gas = x->qs + QK_K/4;
    const int8_t  * q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint32_t aux32 =
            ((uint32_t)gas[0]) |
            ((uint32_t)gas[1] << 8) |
            ((uint32_t)gas[2] << 16) |
            ((uint32_t)gas[3] << 24);
        gas += sizeof(uint32_t);
        const int32_t ls = (int32_t)(2 * (aux32 >> 28) + 1);
        int32_t sumi = 0;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l + 0]);
            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l + 1]);
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
            for (int j = 0; j < 4; ++j) {
                const int s1 = (signs & kmask_iq2xs[j + 0]) ? -1 : 1;
                const int s2 = (signs & kmask_iq2xs[j + 4]) ? -1 : 1;
                sumi += (int32_t)grid1[j] * (int32_t)q8[j + 0] * s1;
                sumi += (int32_t)grid2[j] * (int32_t)q8[j + 4] * s2;
            }
            q8 += 8;
        }
        q3 += 8;
        bsum += sumi * ls;
    }
    return bsum;
}

static __device__ __forceinline__ int moe_iq2_s_q8k_block_sum(
        const block_iq2_s * x,
        const block_q8_K * y) {
    const int8_t  * q8 = y->qs;
    const uint8_t * qs = x->qs;
    const uint8_t * qh = x->qh;
    const uint8_t * signs = qs + QK_K/8;
    int bsum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const int ls1 = 1 + 2*(x->scales[ib32] & 0x0f);
        const int ls2 = 1 + 2*(x->scales[ib32] >> 4);
        int sumi1 = 0;
        int sumi2 = 0;
        for (int l = 0; l < 2; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300)));
            for (int j = 0; j < 8; ++j) {
                sumi1 += (int)q8[j] * (int)grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1 : 1);
            }
            q8 += 8;
        }
        for (int l = 2; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300)));
            for (int j = 0; j < 8; ++j) {
                sumi2 += (int)q8[j] * (int)grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1 : 1);
            }
            q8 += 8;
        }
        bsum += ls1 * sumi1 + ls2 * sumi2;
        qs += 4;
        signs += 4;
    }
    return bsum;
}

static __device__ __forceinline__ int moe_iq2_s_value(
        const block_iq2_s * x,
        int ib32,
        int lane) {
    const int l = lane / 8;
    const int j = lane & 7;
    const uint8_t * qs = x->qs + 4*ib32;
    const uint8_t * signs = x->qs + QK_K/8 + 4*ib32;
    const uint8_t qh = x->qh[ib32];
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | ((qh << (8 - 2*l)) & 0x300)));
    const int ls = l < 2 ? 1 + 2*(x->scales[ib32] & 0x0f) : 1 + 2*(x->scales[ib32] >> 4);
    const int sign = (signs[l] & kmask_iq2xs[j]) ? -1 : 1;
    return ls * (int)grid[j] * sign;
}

static __device__ __forceinline__ float moe_iq2_s_q8k_r8_block_dot(
        const block_iq2_s * x,
        const block_q8_K * y) {
    int max_abs = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        for (int lane = 0; lane < 32; ++lane) {
            const int v = moe_iq2_s_value(x, ib32, lane);
            const int av = v < 0 ? -v : v;
            max_abs = max(max_abs, av);
        }
    }

    float dnew = (float)max_abs * (1.0f / 124.0f);
    const bool needs_scaling = dnew >= 1.0f;
    if (!needs_scaling) {
        dnew = 1.0f;
    }

    int bsum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        for (int lane = 0; lane < 32; ++lane) {
            const int v = moe_iq2_s_value(x, ib32, lane);
            const int q = needs_scaling ? moe_q8k_nearest_int((float)v / dnew) : v;
            bsum += q * (int)y->qs[32*ib32 + lane];
        }
    }
    const float scale = __half2float(__float2half_rn(0.125f * __half2float(x->d) * dnew));
    return scale * y->d * (float)bsum;
}

static __global__ void moe_iq3_xxs_q8k_mat_kernel(
        int src0_type,
        const char * __restrict__ x_pool,
        const block_q8_K * __restrict__ y,
        const int32_t * __restrict__ x_ids,
        const int32_t * __restrict__ dst_ids,
        float * __restrict__ dst,
        int64_t ne00,
        int64_t ne01,
        size_t nb01,
        size_t slot_sz,
        int64_t dst_cols,
        bool iq2_direct) {
    const int row = (int)blockIdx.x;
    const int active = (int)blockIdx.y;
    if (row >= ne01) {
        return;
    }
    const int dst_id = dst_ids ? dst_ids[active] : active;
    if (dst_id < 0 || dst_id >= dst_cols) {
        return;
    }
    const int slot = x_ids[active];
    const char * x_row_base = x_pool + (size_t)slot * slot_sz + (size_t)row * nb01;
    const block_q8_K * y_row = y + (size_t)active * (ne00 / QK_K);

    float sum = 0.0f;
    for (int ib = threadIdx.x; ib < ne00 / QK_K; ib += blockDim.x) {
        if (src0_type == GGML_TYPE_IQ3_XXS) {
            const block_iq3_xxs * x_iq3 = (const block_iq3_xxs *)x_row_base;
            const float d = __half2float(x_iq3[ib].d) * y_row[ib].d;
            sum += 0.25f * d * (float)moe_iq3_xxs_q8k_block_sum(x_iq3 + ib, y_row + ib);
        } else if (src0_type == GGML_TYPE_IQ2_S) {
            const block_iq2_s * x_iq2 = (const block_iq2_s *)x_row_base;
            if (iq2_direct) {
                const float d = __half2float(x_iq2[ib].d) * y_row[ib].d;
                sum += 0.125f * d * (float)moe_iq2_s_q8k_block_sum(x_iq2 + ib, y_row + ib);
            } else {
                sum += moe_iq2_s_q8k_r8_block_dot(x_iq2 + ib, y_row + ib);
            }
        }
    }

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dst[(size_t)dst_id * ne01 + row] = sdata[0];
    }
}

static bool exact_prompt_q8k_enabled() {
    const char *env = std::getenv("GGML_MOE_STREAM_PROMPT_UP_GATE");
    return env && (std::strcmp(env, "exact-q8-k") == 0 ||
        std::strcmp(env, "exact-q8-k-iq2-probe") == 0 ||
        std::strcmp(env, "exact-q8-k-iq2-direct-probe") == 0);
}

static bool exact_prompt_q8k_iq2_probe_enabled() {
    const char *env = std::getenv("GGML_MOE_STREAM_PROMPT_UP_GATE");
    return env && std::strcmp(env, "exact-q8-k-iq2-probe") == 0;
}

static bool exact_prompt_q8k_iq2_direct_probe_enabled() {
    const char *env = std::getenv("GGML_MOE_STREAM_PROMPT_UP_GATE");
    return env && std::strcmp(env, "exact-q8-k-iq2-direct-probe") == 0;
}

static bool launch_moe_iq3_xxs_q8k_batch(
        ggml_type src0_type,
        const char *x_pool,
        const block_q8_K *y,
        const int32_t *dst_ids,
        const int32_t *x_ids,
        float *dst,
        int64_t ne00,
        int64_t ne01,
        size_t nb01,
        size_t slot_sz,
        int n_active,
        int64_t dst_cols,
        bool iq2_direct,
        cudaStream_t stream) {
    if (ne00 % QK_K != 0 || ne01 <= 0 || n_active <= 0 || dst_cols <= 0) {
        return false;
    }
    dim3 block(64);
    dim3 grid((unsigned int)ne01, (unsigned int)n_active);
    moe_iq3_xxs_q8k_mat_kernel<<<grid, block, block.x * sizeof(float), stream>>>(
        src0_type, x_pool, y, x_ids, dst_ids, dst, ne00, ne01, nb01, slot_sz, dst_cols, iq2_direct);
    return cudaGetLastError() == cudaSuccess;
}

static float moe_iq2_selftest_value_for_prompt(int col) {
    const int a = col * 19 + (col / 7) * 3;
    const float wave = std::cos(0.017f * (float)(col + 5));
    return 0.11f * (float)((a % 29) - 14) + wave;
}

static void moe_iq2_selftest_fill_row(uint8_t * dst, int row, int k) {
    const int n_blocks = k / QK_K;
    block_iq2_s * blocks = (block_iq2_s *)dst;
    for (int ib = 0; ib < n_blocks; ++ib) {
        block_iq2_s & b = blocks[ib];
        b.d = ggml_fp32_to_fp16(0.0625f * (float)(row + 2 + ib));
        for (int i = 0; i < QK_K/8; ++i) {
            b.qs[i] = (uint8_t)((row * 37 + ib * 17 + i * 11) & 0xff);
            b.qs[QK_K/8 + i] = (uint8_t)((row * 19 + ib * 13 + i * 7) & 0xff);
        }
        for (int i = 0; i < QK_K/32; ++i) {
            b.qh[i] = (uint8_t)((row * 29 + ib * 5 + i * 3) & 0xff);
            const int lo = (row + ib + i) & 0x0f;
            const int hi = (row * 3 + ib + i * 2) & 0x0f;
            b.scales[i] = (uint8_t)(lo | (hi << 4));
        }
    }
}

extern "C" bool ggml_cuda_moe_iq2_q8k_r8_selftest(void) {
    constexpr int k = 256 * 2;
    constexpr int n_rows = 8;
    constexpr int n_active = 4;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "[moe_stream_batch] IQ2_S CUDA comparator: cudaSetDevice failed\n");
        return false;
    }

    ggml_quantize_init(GGML_TYPE_IQ2_S);
    ggml_quantize_init(GGML_TYPE_Q8_K);

    const ggml_type_traits_t q8_traits = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    if (!q8_traits.from_float) {
        std::fprintf(stderr, "[moe_stream_batch] IQ2_S CUDA comparator: missing quantization callbacks\n");
        return false;
    }

    const size_t iq2_stride = ggml_row_size(GGML_TYPE_IQ2_S, k);
    const size_t q8_stride = ggml_row_size(GGML_TYPE_Q8_K, k);
    const size_t r8_stride = ggml_row_size(GGML_TYPE_Q8_K_R8, k);
    const size_t slot_sz = (size_t)n_rows * iq2_stride;

    std::vector<float> prompt(k);
    std::vector<uint8_t> iq2(slot_sz);
    std::vector<uint8_t> q8(q8_stride);
    std::vector<uint8_t> r8((size_t)n_rows * r8_stride);
    std::vector<float> expect(n_rows);
    std::vector<float> got(n_rows);
    std::vector<uint8_t> iq2_multi((size_t)n_active * slot_sz);
    std::vector<uint8_t> q8_multi((size_t)n_active * q8_stride);
    std::vector<float> expect_multi((size_t)n_active * n_rows);
    std::vector<float> got_multi((size_t)n_active * n_rows);

    for (int r = 0; r < n_rows; ++r) {
        moe_iq2_selftest_fill_row(iq2.data() + (size_t)r * iq2_stride, r, k);
    }
    for (int c = 0; c < k; ++c) {
        prompt[c] = moe_iq2_selftest_value_for_prompt(c);
    }
    q8_traits.from_float(prompt.data(), q8.data(), k);

    if (!iqk_convert_repack_q8_r8(GGML_TYPE_IQ2_S, k, iq2.data(), iq2_stride, r8.data(), k, n_rows) ||
            !iqk_mul_mat(n_rows, 1, k, GGML_TYPE_Q8_K_R8, r8.data(), r8_stride,
                GGML_TYPE_Q8_K, q8.data(), q8_stride, expect.data(), 0, 0, 1)) {
        std::fprintf(stderr, "[moe_stream_batch] IQ2_S CUDA comparator: CPU repack oracle failed\n");
        return false;
    }

    for (int a = 0; a < n_active; ++a) {
        uint8_t *iq2_base = iq2_multi.data() + (size_t)a * slot_sz;
        for (int r = 0; r < n_rows; ++r) {
            moe_iq2_selftest_fill_row(iq2_base + (size_t)r * iq2_stride, a * n_rows + r, k);
        }
        std::vector<float> prompt_a(k);
        for (int c = 0; c < k; ++c) {
            prompt_a[c] = moe_iq2_selftest_value_for_prompt(c + a * 13);
        }
        q8_traits.from_float(prompt_a.data(), q8_multi.data() + (size_t)a * q8_stride, k);
        if (!iqk_convert_repack_q8_r8(GGML_TYPE_IQ2_S, k, iq2_base, iq2_stride, r8.data(), k, n_rows) ||
                !iqk_mul_mat(n_rows, 1, k, GGML_TYPE_Q8_K_R8, r8.data(), r8_stride,
                    GGML_TYPE_Q8_K, q8_multi.data() + (size_t)a * q8_stride, q8_stride,
                    expect_multi.data() + (size_t)a * n_rows, 0, 0, 1)) {
            std::fprintf(stderr, "[moe_stream_batch] IQ2_S CUDA comparator: multi-active CPU oracle failed active=%d\n", a);
            return false;
        }
    }

    cudaStream_t stream = nullptr;
    uint8_t *d_iq2 = nullptr;
    uint8_t *d_iq2_multi = nullptr;
    block_q8_K *d_q8 = nullptr;
    block_q8_K *d_q8_multi = nullptr;
    int32_t *d_x_ids = nullptr;
    int32_t *d_dst_ids = nullptr;
    float *d_dst = nullptr;
    int32_t *d_x_ids_multi = nullptr;
    int32_t *d_dst_ids_multi = nullptr;
    float *d_dst_multi = nullptr;

    bool ok = cudaStreamCreate(&stream) == cudaSuccess;
    ok = ok && cudaMalloc(&d_iq2, iq2.size()) == cudaSuccess;
    ok = ok && cudaMalloc(&d_iq2_multi, iq2_multi.size()) == cudaSuccess;
    ok = ok && cudaMalloc(&d_q8, q8.size()) == cudaSuccess;
    ok = ok && cudaMalloc(&d_q8_multi, q8_multi.size()) == cudaSuccess;
    ok = ok && cudaMalloc(&d_x_ids, sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&d_dst_ids, sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&d_dst, got.size() * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&d_x_ids_multi, n_active * sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&d_dst_ids_multi, n_active * sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&d_dst_multi, got_multi.size() * sizeof(float)) == cudaSuccess;

    const int32_t zero = 0;
    if (ok) {
        ok = ok && cudaMemcpyAsync(d_iq2, iq2.data(), iq2.size(), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_iq2_multi, iq2_multi.data(), iq2_multi.size(), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_q8, q8.data(), q8.size(), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_q8_multi, q8_multi.data(), q8_multi.size(), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_x_ids, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_dst_ids, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemsetAsync(d_dst, 0, got.size() * sizeof(float), stream) == cudaSuccess;
        int32_t ids_multi[n_active];
        for (int a = 0; a < n_active; ++a) {
            ids_multi[a] = a;
        }
        ok = ok && cudaMemcpyAsync(d_x_ids_multi, ids_multi, sizeof(ids_multi), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_dst_ids_multi, ids_multi, sizeof(ids_multi), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemsetAsync(d_dst_multi, 0, got_multi.size() * sizeof(float), stream) == cudaSuccess;
    }
    if (ok) {
        ok = launch_moe_iq3_xxs_q8k_batch(GGML_TYPE_IQ2_S, (const char *)d_iq2, d_q8,
                d_dst_ids, d_x_ids, d_dst, k, n_rows, iq2_stride, slot_sz, 1, 1, false, stream);
    }
    ok = ok && cudaMemcpyAsync(got.data(), d_dst, got.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    if (ok) {
        ok = launch_moe_iq3_xxs_q8k_batch(GGML_TYPE_IQ2_S, (const char *)d_iq2_multi, d_q8_multi,
                d_dst_ids_multi, d_x_ids_multi, d_dst_multi, k, n_rows, iq2_stride, slot_sz,
                n_active, n_active, false, stream);
    }
    ok = ok && cudaMemcpyAsync(got_multi.data(), d_dst_multi, got_multi.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;

    cudaFree(d_dst_multi);
    cudaFree(d_dst_ids_multi);
    cudaFree(d_x_ids_multi);
    cudaFree(d_dst);
    cudaFree(d_dst_ids);
    cudaFree(d_x_ids);
    cudaFree(d_q8_multi);
    cudaFree(d_q8);
    cudaFree(d_iq2_multi);
    cudaFree(d_iq2);
    if (stream) {
        cudaStreamDestroy(stream);
    }

    float max_abs = 0.0f;
    float max_expect_abs = 0.0f;
    int max_row = -1;
    for (int r = 0; ok && r < n_rows; ++r) {
        const float diff = std::fabs(expect[r] - got[r]);
        max_expect_abs = std::max(max_expect_abs, std::fabs(expect[r]));
        if (diff > max_abs) {
            max_abs = diff;
            max_row = r;
        }
    }
    const float tol = std::max(2.0f, max_expect_abs * 1.0e-5f);
    ok = ok && max_abs <= tol;
    float multi_max_abs = 0.0f;
    float multi_max_expect_abs = 0.0f;
    int multi_max_active = -1;
    int multi_max_row = -1;
    for (int a = 0; ok && a < n_active; ++a) {
        for (int r = 0; r < n_rows; ++r) {
            const size_t idx = (size_t)a * n_rows + r;
            const float diff = std::fabs(expect_multi[idx] - got_multi[idx]);
            multi_max_expect_abs = std::max(multi_max_expect_abs, std::fabs(expect_multi[idx]));
            if (diff > multi_max_abs) {
                multi_max_abs = diff;
                multi_max_active = a;
                multi_max_row = r;
            }
        }
    }
    const float multi_tol = std::max(2.0f, multi_max_expect_abs * 1.0e-5f);
    ok = ok && multi_max_abs <= multi_tol;
    std::fprintf(stderr, "[moe_stream_batch] IQ2_S CUDA comparator: %s rows=%d k=%d max_abs=%g tol=%g row=%d\n",
            ok ? "ok" : "failed", n_rows, k, max_abs, tol, max_row);
    if (!ok && max_row >= 0) {
        std::fprintf(stderr, "[moe_stream_batch] IQ2_S CUDA comparator row %d expect=%g got=%g\n",
                max_row, expect[max_row], got[max_row]);
    }
    std::fprintf(stderr,
            "[moe_stream_batch] IQ2_S CUDA multi-active comparator: %s active=%d rows=%d max_abs=%g tol=%g active_row=%d/%d\n",
            multi_max_abs <= multi_tol ? "ok" : "failed", n_active, n_rows,
            multi_max_abs, multi_tol, multi_max_active, multi_max_row);
    if (multi_max_abs > multi_tol && multi_max_active >= 0 && multi_max_row >= 0) {
        const size_t idx = (size_t)multi_max_active * n_rows + multi_max_row;
        std::fprintf(stderr,
                "[moe_stream_batch] IQ2_S CUDA multi-active mismatch active=%d row=%d expect=%g got=%g\n",
                multi_max_active, multi_max_row, expect_multi[idx], got_multi[idx]);
    }
    return ok;
}

static float moe_host_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static float moe_host_up_gate_fuse(float u, float g, int unary_op, float limit) {
    switch ((ggml_unary_op) unary_op) {
        case GGML_UNARY_OP_SILU:
            if (limit < 1.0e-6f) {
                return u * moe_host_silu(g);
            } else {
                const float gate_v = std::min(moe_host_silu(g), limit);
                const float up_v = std::max(-limit, std::min(limit, u));
                return up_v * gate_v;
            }
        case GGML_UNARY_OP_RELU:
            return u * std::max(g, 0.0f);
        case GGML_UNARY_OP_GELU: {
            constexpr float GELU_COEF_A    = 0.044715f;
            constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
            return 0.5f * g * u * (1.0f + std::tanh(SQRT_2_OVER_PI * g * (1.0f + GELU_COEF_A * g * g)));
        }
        default:
            return 0.0f;
    }
}

extern "C" bool ggml_cuda_moe_iq2_prompt_replay(
        int src0_type_int,
        const void *up_expert,
        const void *gate_expert,
        int64_t ne01,
        int64_t ne00,
        size_t nb01,
        const float *src1_row_f32,
        int64_t col,
        int unary_op,
        float limit,
        ggml_moe_iq2_replay_result *out) {
    if (!out || src0_type_int != GGML_TYPE_IQ2_S || !up_expert || !gate_expert || !src1_row_f32 ||
            ne01 <= 0 || ne00 <= 0 || ne00 % QK_K != 0 || col < 0 || col >= ne01) {
        return false;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        return false;
    }

    const size_t expert_bytes = (size_t)ne01 * nb01;
    const size_t q8k_bytes = (size_t)(ne00 / QK_K) * sizeof(block_q8_K);
    const size_t src1_bytes = (size_t)ne00 * sizeof(float);
    const size_t dst_bytes = (size_t)ne01 * sizeof(float);

    cudaStream_t stream = nullptr;
    uint8_t *d_up = nullptr;
    uint8_t *d_gate = nullptr;
    float *d_src1 = nullptr;
    block_q8_K *d_q8k = nullptr;
    int32_t *d_x_ids = nullptr;
    int32_t *d_dst_ids = nullptr;
    float *d_dst = nullptr;

    bool ok = cudaStreamCreate(&stream) == cudaSuccess;
    ok = ok && cudaMalloc(&d_up, expert_bytes) == cudaSuccess;
    ok = ok && cudaMalloc(&d_gate, expert_bytes) == cudaSuccess;
    ok = ok && cudaMalloc(&d_src1, src1_bytes) == cudaSuccess;
    ok = ok && cudaMalloc(&d_q8k, q8k_bytes) == cudaSuccess;
    ok = ok && cudaMalloc(&d_x_ids, sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&d_dst_ids, sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&d_dst, dst_bytes) == cudaSuccess;

    const int32_t zero = 0;
    if (ok) {
        ok = ok && cudaMemcpyAsync(d_up, up_expert, expert_bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_gate, gate_expert, expert_bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_src1, src1_row_f32, src1_bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_x_ids, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_dst_ids, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    if (ok) {
        const int nblocks = (int)(ne00 / QK_K);
        moe_quantize_row_q8_k_kernel<<<nblocks, 1, 0, stream>>>(d_src1, d_q8k, nblocks);
        ok = cudaGetLastError() == cudaSuccess;
    }

    auto run_one = [&](const uint8_t *d_expert, bool direct, float *value) -> bool {
        if (cudaMemsetAsync(d_dst, 0, dst_bytes, stream) != cudaSuccess) {
            return false;
        }
        if (!launch_moe_iq3_xxs_q8k_batch(GGML_TYPE_IQ2_S, (const char *)d_expert, d_q8k,
                    d_dst_ids, d_x_ids, d_dst, ne00, ne01, nb01, expert_bytes, 1, 1, direct, stream)) {
            return false;
        }
        return cudaMemcpyAsync(value, d_dst + col, sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    };

    if (ok) ok = run_one(d_up, true, &out->direct_up);
    if (ok) ok = run_one(d_gate, true, &out->direct_gate);
    if (ok) ok = run_one(d_up, false, &out->repack_up);
    if (ok) ok = run_one(d_gate, false, &out->repack_gate);
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;

    if (ok) {
        out->direct_fused = moe_host_up_gate_fuse(out->direct_up, out->direct_gate, unary_op, limit);
        out->repack_fused = moe_host_up_gate_fuse(out->repack_up, out->repack_gate, unary_op, limit);
    }

    cudaFree(d_dst);
    cudaFree(d_dst_ids);
    cudaFree(d_x_ids);
    cudaFree(d_q8k);
    cudaFree(d_src1);
    cudaFree(d_gate);
    cudaFree(d_up);
    if (stream) {
        cudaStreamDestroy(stream);
    }
    return ok;
}

static bool moe_q8k_selftest(cudaStream_t stream) {
    constexpr int nblocks = 3;
    std::vector<float> host_x((size_t)nblocks * QK_K);
    for (size_t i = 0; i < host_x.size(); ++i) {
        const int v = (int)((i * 37 + 11) % 257) - 128;
        host_x[i] = (float)v / 17.0f + ((i & 7) == 0 ? 0.03125f : -0.015625f);
    }

    std::vector<block_q8_K> expect(nblocks);
    std::vector<block_q8_K> got(nblocks);
    std::memset(expect.data(), 0, expect.size() * sizeof(block_q8_K));
    std::memset(got.data(), 0, got.size() * sizeof(block_q8_K));
    quantize_row_q8_K_ref(host_x.data(), expect.data(), (int64_t)host_x.size());

    float * d_x = nullptr;
    block_q8_K * d_y = nullptr;
    const size_t x_bytes = host_x.size() * sizeof(float);
    const size_t y_bytes = got.size() * sizeof(block_q8_K);
    if (cudaMalloc(&d_x, x_bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_y, y_bytes) != cudaSuccess) {
        cudaFree(d_x);
        return false;
    }
    bool ok = true;
    ok = ok && cudaMemcpyAsync(d_x, host_x.data(), x_bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    ok = ok && cudaMemsetAsync(d_y, 0, y_bytes, stream) == cudaSuccess;
    if (ok) {
        moe_quantize_row_q8_k_kernel<<<nblocks, 1, 0, stream>>>(d_x, d_y, nblocks);
        ok = cudaGetLastError() == cudaSuccess;
    }
    ok = ok && cudaMemcpyAsync(got.data(), d_y, y_bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;
    cudaFree(d_y);
    cudaFree(d_x);

    if (ok && std::memcmp(expect.data(), got.data(), y_bytes) != 0) {
        ok = false;
    }
    std::fprintf(stderr, "[moe_stream_batch] Q8_K CUDA quant selftest: %s\n", ok ? "ok" : "failed");
    return ok;
}

static bool prompt_up_gate_stream_enabled() {
    const char *env = std::getenv("GGML_MOE_STREAM_PROMPT_UP_GATE");
    return env && (std::strcmp(env, "unsafe-q8-1") == 0 ||
        std::strcmp(env, "exact-q8-k") == 0 ||
        std::strcmp(env, "exact-q8-k-iq2-probe") == 0 ||
        std::strcmp(env, "exact-q8-k-iq2-direct-probe") == 0);
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
        const char *q8k_selftest_env = std::getenv("GGML_MOE_Q8K_SELFTEST");
        if (g_batch.stream && q8k_selftest_env && q8k_selftest_env[0] && q8k_selftest_env[0] != '0') {
            if (!moe_q8k_selftest(g_batch.stream)) {
                std::fprintf(stderr, "[moe_stream_batch] Q8_K CUDA quant selftest failed; exact prompt Q8_K path disabled\n");
            }
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
            cudaEventCreate(&g_batch.ev_gate_work_start);
            cudaEventCreate(&g_batch.ev_up_compute_start);
            cudaEventCreate(&g_batch.ev_gate_compute_start);
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

extern "C" bool ggml_cuda_moe_stream_preload_tensor_prompt(
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
    preload_prompt_profile_for_tensor(src0_name, src0_data, n_as, nb02, expert_bytes, g_batch.stream);
    cudaStreamSynchronize(g_batch.stream);
    return true;
}

extern "C" bool ggml_cuda_moe_stream_preload_tensor_async(
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
    return true;
}

extern "C" bool ggml_cuda_moe_stream_preload_expert_async(
    int src0_type_int,
    const char *src0_name,
    const void *src0_data,
    int64_t n_as,
    size_t nb02,
    size_t expert_bytes,
    int expert_idx) {
    if (!init_batch_once()) return false;
    if (!moe_stream_type_supported((ggml_type)src0_type_int) || !src0_data || !src0_name) return false;
    if (expert_idx < 0 || expert_idx >= n_as) return false;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    batch_vram_cache *cache = batch_cache_get(expert_bytes);
    if (!cache) return false;
    const uintptr_t key = batch_key_hash(src0_name, expert_idx);
    if (batch_cache_find_slot(cache, key) >= 0) return true;
    const char *expert_host = (const char *)src0_data + (size_t)expert_idx * nb02;
    return batch_cache_insert_slot(cache, key, expert_host, expert_bytes, g_batch.stream, false, true,
            nullptr, 0, true, src0_name, expert_idx) >= 0;
}

extern "C" bool ggml_cuda_moe_stream_preload_synchronize(void) {
    if (!init_batch_once()) return false;
    std::lock_guard<std::mutex> lk(g_batch_mu);
    if (!g_batch.stream) return false;
    return cudaStreamSynchronize(g_batch.stream) == cudaSuccess;
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
        int64_t n_active, int64_t dst_cols, cudaStream_t st, bool dynamic_mmq_x = false) {
    const mmq_args_id args = {
        d_src0, src0_type, d_src1_q8, d_ids_dst, d_bounds, d_x_ids, d_dst,
        ne00, ne01, dst_cols, src0_stride, n_active, ne01,
        n_active, n_active, src0_channel_stride, 0, 0,
        1, 1, 0, 0, 0,
        false, n_active};
    ggml_backend_cuda_context * null_ctx = nullptr;
    switch (src0_type) {
        case GGML_TYPE_IQ3_XXS:
            if (dynamic_mmq_x) {
                mul_mat_q_case_id<GGML_TYPE_IQ3_XXS>(*null_ctx, args, st);
            } else {
                launch_mul_mat_q_id<GGML_TYPE_IQ3_XXS, 8>(*null_ctx, args, st);
            }
            break;
        case GGML_TYPE_IQ2_S:
            if (dynamic_mmq_x) {
                mul_mat_q_case_id<GGML_TYPE_IQ2_S>(*null_ctx, args, st);
            } else {
                launch_mul_mat_q_id<GGML_TYPE_IQ2_S, 8>(*null_ctx, args, st);
            }
            break;
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
    const bool decline_debug = []() {
        const char *env = std::getenv("GGML_MOE_STREAM_DECLINE_DEBUG");
        return env && env[0] && env[0] != '0';
    }();
    auto decline = [&](const char *reason) -> bool {
        if (decline_debug) {
            std::fprintf(stderr,
                "[moe_stream] up/gate batch declined: reason=%s tensor=%s rows_stride=%ld type=%d\n",
                reason, src0_up_name ? src0_up_name : "", (long)rows_stride, src0_type_int);
        }
        return false;
    };
    if (!init_batch_once()) return decline("init_batch_once");
    ggml_type src0_type = (ggml_type)src0_type_int;
    if (!moe_stream_type_supported(src0_type)) return decline("unsupported_type");
    if (!src1_f32 || !src0_up_data || !src0_gate_data) return decline("missing_input");
    if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_RELU && unary_op != GGML_UNARY_OP_GELU) return decline("unsupported_unary");
    GGML_UNUSED(src1_nb1);
    const bool prompt_exact_requested = exact_prompt_q8k_enabled();
    const bool exact_prompt_type = src0_type == GGML_TYPE_IQ3_XXS ||
        (src0_type == GGML_TYPE_IQ2_S &&
            (exact_prompt_q8k_iq2_probe_enabled() || exact_prompt_q8k_iq2_direct_probe_enabled()));
    const bool prompt_mode = rows_stride > 1 &&
        (prompt_exact_requested ? exact_prompt_type : prompt_up_gate_stream_enabled());
    const bool exact_prompt_q8k = prompt_mode && prompt_exact_requested && exact_prompt_type;
    if (decline_debug && rows_stride > 1 && !prompt_mode) {
        const char *env = std::getenv("GGML_MOE_STREAM_PROMPT_UP_GATE");
        std::fprintf(stderr,
            "[moe_stream] up/gate prompt mode disabled: env=%s exact_requested=%d exact_type=%d stream_enabled=%d tensor=%s\n",
            env ? env : "", prompt_exact_requested ? 1 : 0, exact_prompt_type ? 1 : 0,
            prompt_up_gate_stream_enabled() ? 1 : 0, src0_up_name ? src0_up_name : "");
    }
    int active_experts[MOE_STREAM_MAX_ACTIVE];
    int32_t dst_ids[MOE_STREAM_MAX_ACTIVE];
    int32_t flat_dst_ids[MOE_STREAM_MAX_ACTIVE];
    int32_t token_ids[MOE_STREAM_MAX_ACTIVE];
    int n_active = 0;
    int max_dst_id = -1;
    const int64_t dst_row_stride = dst_nb1 > 0 ? (int64_t)(dst_nb2 / dst_nb1) : 0;
    if (prompt_mode && dst_row_stride <= 0) return decline("bad_prompt_dst_stride");
    for (int64_t e = 0; e < n_as; ++e) {
        if (matrix_row_counts[e] <= 0) {
            continue;
        }
        if (matrix_row_counts[e] > 1 && !prompt_mode) return decline("multirow_requires_prompt_mode");
        const ggml_moe_row_mapping * r = matrix_rows + e*rows_stride;
        for (int64_t ir = 0; ir < matrix_row_counts[e]; ++ir) {
            if (n_active >= MOE_STREAM_MAX_ACTIVE) return decline("too_many_active_routes");
            active_experts[n_active] = (int)e;
            const int64_t dst_row = r[ir].i1;
            const int64_t flat_dst_row = prompt_mode ? (int64_t)r[ir].i2 * dst_row_stride + r[ir].i1 : r[ir].i1;
            if (dst_row < 0 || dst_row > INT32_MAX || flat_dst_row < 0 || flat_dst_row > INT32_MAX) return decline("bad_route_row");
            dst_ids[n_active] = (int32_t)dst_row;
            flat_dst_ids[n_active] = (int32_t)flat_dst_row;
            token_ids[n_active] = r[ir].i2;
            if (flat_dst_ids[n_active] > max_dst_id) max_dst_id = flat_dst_ids[n_active];
            ++n_active;
        }
    }
    if (n_active <= 0 || max_dst_id < 0) return decline("no_active_routes");

    static std::atomic<int> first_up_gate{0};
    if (first_up_gate.fetch_add(1) == 0) {
        std::fprintf(stderr, "[moe_stream] batched up/gate %s path active: rows=%d ne01=%ld ne00=%ld\n",
                     prompt_mode ? "prompt" : "decode", n_active, (long)ne01, (long)ne00);
        std::fprintf(stderr, "[moe_stream] up/gate routes:");
        for (int j = 0; j < n_active; ++j) {
            std::fprintf(stderr, " #%d:e%d->dst%d/flat%d/tok%d",
                         j, active_experts[j], dst_ids[j], flat_dst_ids[j], token_ids[j]);
        }
        std::fprintf(stderr, "\n");
    }

    std::lock_guard<std::mutex> lk(g_batch_mu);
    batch_ctx &bc = g_batch;
    cudaStream_t st = bc.stream;
    const bool profile = g_uprof.enabled && bc.ev_start && bc.ev_stage && bc.ev_quant && bc.ev_kernel && bc.ev_d2h;

    const size_t src0_bytes = (size_t)ne01 * nb01;
    batch_ttft_call_scope ttft_scope("call_upgate", src0_up_name, n_active, src0_bytes);
    const int64_t ne00_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t src1_f32_bytes = (size_t)n_active * ne00 * sizeof(float);
    const size_t src1_q8_bytes = (size_t)n_active * ne00_padded * sizeof(block_q8_1) / QK8_1
        + (size_t)get_mmq_x_max_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) * sizeof(block_q8_1_mmq);
    const size_t src1_q8_one_bytes = (size_t)ne00_padded * sizeof(block_q8_1) / QK8_1
        + (size_t)get_mmq_x_max_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) * sizeof(block_q8_1_mmq);
    const size_t src1_q8k_bytes = exact_prompt_q8k ? (size_t)n_active * (ne00 / QK_K) * sizeof(block_q8_K) : 0;
    const int64_t dst_cols = max_dst_id + 1;
    const size_t dst_bytes = (size_t)dst_cols * ne01 * sizeof(float);
    const size_t ids_bytes = (size_t)n_active * sizeof(int32_t);
    const size_t bounds_bytes = (size_t)(n_active + 1) * sizeof(int32_t);
    const bool use_handoff = !prompt_mode && gpu_handoff_enabled();

    bool ok = ensure_dev(bc.d_src1_f32, bc.d_src1_f32_sz, src1_f32_bytes)
        && ensure_dev(bc.d_src1_q8, bc.d_src1_q8_sz, src1_q8_bytes)
        && (!exact_prompt_q8k || ensure_dev(bc.d_src1_q8k, bc.d_src1_q8k_sz, src1_q8k_bytes))
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
    if (!ok) return decline("alloc_workspace");

    batch_vram_cache *cache = batch_cache_get(src0_bytes);
    if (!cache) return decline("cache_unavailable");
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
        !exact_prompt_q8k &&
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
        const char *fresh_env = std::getenv("GGML_MOE_STREAM_EXACT_PROMPT_FRESH_COPY");
        const bool exact_fresh_copy = exact_prompt_q8k && (!fresh_env || !fresh_env[0] || fresh_env[0] != '0');
        for (int j = 0; j < n_active; ++j) {
            const char *expert_host = (const char *)host_base + (size_t)active_experts[j] * nb02;
            const uintptr_t cache_key = batch_key_hash(key_name, active_experts[j]);
            int cache_slot = exact_fresh_copy ? -1 : batch_cache_lookup_slot(cache, cache_key);
            if (cache_slot < 0) {
                cache_slot = batch_cache_insert_slot(
                    cache, cache_key, expert_host, src0_bytes, run_stream, true, false,
                    avoid_slots, n_avoid_slots, true, key_name, active_experts[j]);
            } else {
                batch_ttft_trace_record("cache_hit", key_name, active_experts[j], src0_bytes, true, false, false, 0.0);
            }
            if (cache_slot < 0) return false;
            h_x_ids[j] = cache_slot;
            if (slots_out) slots_out[j] = cache_slot;
            batch_route_profile_hit(key_name, active_experts[j], src0_bytes);
        }
        if (cudaMemsetAsync(d_out, 0, dst_bytes, run_stream) != cudaSuccess) return false;
        if (cudaMemcpyAsync(d_x_ids, h_x_ids, ids_bytes, cudaMemcpyHostToDevice, run_stream) != cudaSuccess) return false;
        if (exact_prompt_q8k) {
            static std::atomic<int> first_exact_q8k{0};
            if (first_exact_q8k.fetch_add(1) == 0) {
                std::fprintf(stderr, "[moe_stream] exact %s x Q8_K prompt up/gate path active\n",
                    src0_type == GGML_TYPE_IQ2_S ? "IQ2_S" : "IQ3_XXS");
            }
            return launch_moe_iq3_xxs_q8k_batch(
                src0_type, (const char *)cache->pool, (const block_q8_K *)bc.d_src1_q8k,
                bc.d_ids_dst, d_x_ids, (float *)d_out,
                ne00, ne01, nb01, cache->slot_sz, n_active, dst_cols,
                src0_type == GGML_TYPE_IQ2_S && exact_prompt_q8k_iq2_direct_probe_enabled(),
                run_stream);
        }
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
            d_x_ids, (float *)d_out, ne00, ne01, nb01, src0_bytes, n_active, dst_cols, run_stream, prompt_mode);
    };

    struct stage_copy_job {
        int slot = -1;
        void *dst = nullptr;
        const void *host_data = nullptr;
        const expert_pack_entry *pack_entry = nullptr;
        int expert_idx = -1;
        char tensor[96] = {};
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
                stage_copy_job job;
                job.slot = cache_slot;
                job.dst = dst_slot;
                job.host_data = expert_host;
                job.pack_entry = pack_entry;
                job.expert_idx = active_experts[j];
                std::snprintf(job.tensor, sizeof(job.tensor), "%s", key_name);
                jobs.push_back(job);
            } else {
                batch_ttft_trace_record("cache_hit", key_name, active_experts[j], src0_bytes, true, false, false, 0.0);
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
            batch_copy_trace copy_trace;
            const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (!batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, job.pack_entry, &copy_trace)) {
                if (!job.pack_entry ||
                        !batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, nullptr, &copy_trace)) {
                    return false;
                }
            }
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
            if (copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                batch_ttft_trace_record(
                    "runtime_load",
                    job.tensor,
                    job.expert_idx,
                    src0_bytes,
                    false,
                    copy_trace.pack_hit,
                    copy_trace.ram_hit,
                    std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
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
        bc.h_ids_dst[j] = flat_dst_ids[j];
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
    if (exact_prompt_q8k) {
        if (ne00 % QK_K != 0) return false;
        const int nblocks = (int)(n_active * (ne00 / QK_K));
        moe_quantize_row_q8_k_kernel<<<nblocks, 1, 0, st>>>((const float *)bc.d_src1_f32, (block_q8_K *)bc.d_src1_q8k, nblocks);
        if (cudaGetLastError() != cudaSuccess) return false;
    } else if (!serial_up_gate && !iq2s_batched_stage) {
        quantize_mmq_q8_1_cuda_id((const float *)bc.d_src1_f32, bc.d_ids_src1, bc.d_src1_q8,
            src0_type, ne00, ne00, n_active * ne00, n_active * ne00,
            ne00_padded, n_active, 1, 1, st);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    if (profile) cudaEventRecord(bc.ev_quant, st);

    float * fused_d = use_handoff ? (float *)bc.d_handoff : (float *)bc.d_dst;
    int up_stage_jobs_count = 0;
    int gate_stage_jobs_count = 0;
    const char *point_tensor_env = std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_POINT_TENSOR");
    const bool point_dump = point_tensor_env && point_tensor_env[0] &&
        src0_up_name && std::strstr(src0_up_name, point_tensor_env);
    const int point_active = point_dump && std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_POINT_ACTIVE") ?
        std::atoi(std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_POINT_ACTIVE")) : -1;
    const int point_flat = point_dump && std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_POINT_FLAT") ?
        std::atoi(std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_POINT_FLAT")) : -1;
    const int point_col = point_dump && std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_POINT_COL") ?
        std::atoi(std::getenv("GGML_MOE_STREAM_FUSED_UP_GATE_POINT_COL")) : -1;
    if (exact_prompt_q8k && n_active * 2 > cache->n_slots) {
        const bool exact_debug = []() {
            const char *env = std::getenv("GGML_MOE_STREAM_EXACT_PROMPT_DEBUG");
            return env && env[0] && env[0] != '0';
        }();
        auto exact_fail = [&](const char *where) -> bool {
            if (exact_debug) {
                std::fprintf(stderr,
                    "[moe_stream] exact prompt chunked staging failed: %s tensor=%s routes=%d cache_slots=%d\n",
                    where, src0_up_name ? src0_up_name : "", n_active, cache->n_slots);
            }
            return false;
        };
        const int evictable_slots = cache->n_slots > (int)cache->pinned ? cache->n_slots - (int)cache->pinned : 0;
        if (evictable_slots < 2) return exact_fail("evictable_slots_lt_2");
        const int chunk_routes = std::max(1, std::min(MOE_STREAM_MAX_ACTIVE, evictable_slots / 2));
        static std::atomic<int> first_chunked_exact{0};
        if (first_chunked_exact.fetch_add(1) == 0) {
            std::fprintf(stderr,
                "[moe_stream] exact prompt Q8_K chunked staging active: routes=%d cache_slots=%d chunk=%d\n",
                n_active, cache->n_slots, chunk_routes);
        }
        if (cudaMemsetAsync(bc.d_up, 0, dst_bytes, st) != cudaSuccess) return exact_fail("memset_up");
        if (cudaMemsetAsync(bc.d_gate, 0, dst_bytes, st) != cudaSuccess) return exact_fail("memset_gate");
        if (cudaMemsetAsync(fused_d, 0, dst_bytes, st) != cudaSuccess) return exact_fail("memset_fused");

        auto stage_exact_chunk = [&](
                const char *key_name,
                const void *host_base,
                int first,
                int count,
                int32_t *d_x_ids,
                int32_t *h_x_ids,
                int *slots_out,
                const int *avoid_slots,
                int n_avoid_slots,
                float *d_out) -> bool {
            for (int jj = 0; jj < count; ++jj) {
                const int j = first + jj;
                const char *expert_host = (const char *)host_base + (size_t)active_experts[j] * nb02;
                const uintptr_t cache_key = batch_key_hash(key_name, active_experts[j]);
                int effective_avoid[MOE_STREAM_MAX_ACTIVE * 2] = {};
                int n_effective_avoid = 0;
                for (int ia = 0; ia < n_avoid_slots && n_effective_avoid < (int)(sizeof(effective_avoid) / sizeof(effective_avoid[0])); ++ia) {
                    effective_avoid[n_effective_avoid++] = avoid_slots[ia];
                }
                if (slots_out) {
                    for (int ia = 0; ia < jj && n_effective_avoid < (int)(sizeof(effective_avoid) / sizeof(effective_avoid[0])); ++ia) {
                        effective_avoid[n_effective_avoid++] = slots_out[ia];
                    }
                }
                const int cache_slot = batch_cache_insert_slot(
                    cache, cache_key, expert_host, src0_bytes, st, true, false,
                    effective_avoid, n_effective_avoid, true, key_name, active_experts[j],
                    false, false);
                if (cache_slot < 0) return exact_fail("cache_insert");
                h_x_ids[jj] = cache_slot;
                if (slots_out) slots_out[jj] = cache_slot;
                batch_route_profile_hit(key_name, active_experts[j], src0_bytes);
            }
            if (cudaMemcpyAsync(d_x_ids, h_x_ids, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice, st) != cudaSuccess) return exact_fail("copy_x_ids");
            if (!launch_moe_iq3_xxs_q8k_batch(
                src0_type, (const char *)cache->pool,
                (const block_q8_K *)bc.d_src1_q8k + (size_t)first * (ne00 / QK_K),
                bc.d_ids_dst, d_x_ids, d_out,
                ne00, ne01, nb01, cache->slot_sz, count, dst_cols,
                src0_type == GGML_TYPE_IQ2_S && exact_prompt_q8k_iq2_direct_probe_enabled(),
                st)) {
                return exact_fail("launch_q8k_batch");
            }
            return true;
        };

        dim3 block(256);
        dim3 grid_x((unsigned int)((ne01 + block.x - 1) / block.x));
        for (int first = 0; first < n_active; first += chunk_routes) {
            const int count = std::min(chunk_routes, n_active - first);
            int up_slots[MOE_STREAM_MAX_ACTIVE] = {};
            int gate_slots[MOE_STREAM_MAX_ACTIVE] = {};
            for (int jj = 0; jj < count; ++jj) {
                bc.h_ids_dst[jj] = jj;
            }
            if (cudaMemcpyAsync(bc.d_ids_dst, bc.h_ids_dst, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice, st) != cudaSuccess) {
                return exact_fail("copy_chunk_dst_ids");
            }
            if (!stage_exact_chunk(up_key_name, src0_up_data, first, count,
                    bc.d_x_ids_up, bc.h_x_ids_up, up_slots, nullptr, 0, (float *)bc.d_up)) {
                return exact_fail("stage_up_chunk");
            }
            if (!stage_exact_chunk(gate_key_name, src0_gate_data, first, count,
                    bc.d_x_ids_gate, bc.h_x_ids_gate, gate_slots, up_slots, count, (float *)bc.d_gate)) {
                return exact_fail("stage_gate_chunk");
            }
            if (point_dump && point_active >= first && point_active < first + count &&
                    point_col >= 0 && point_col < ne01) {
                const int local = point_active - first;
                float up_point = 0.0f;
                float gate_point = 0.0f;
                if (cudaMemcpyAsync(&up_point, (const float *)bc.d_up + (size_t)local * ne01 + point_col,
                            sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) return exact_fail("copy_point_up");
                if (cudaMemcpyAsync(&gate_point, (const float *)bc.d_gate + (size_t)local * ne01 + point_col,
                            sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) return exact_fail("copy_point_gate");
                if (cudaStreamSynchronize(st) != cudaSuccess) return exact_fail("sync_point");
                std::fprintf(stderr,
                    "[moe_stream] exact chunk point prefuse: tensor=%s active=%d local=%d expert=%d token=%d dst=%d flat=%d col=%d up=%g gate=%g chunk_first=%d chunk_count=%d\n",
                    src0_up_name, point_active, local, active_experts[point_active],
                    token_ids[point_active], dst_ids[point_active], flat_dst_ids[point_active],
                    point_col, (double)up_point, (double)gate_point, first, count);
            }
            dim3 grid(grid_x.x, (unsigned int)count);
            moe_stream_up_gate_fuse_kernel<<<grid, block, 0, st>>>(
                (const float *)bc.d_up, (const float *)bc.d_gate, fused_d,
                bc.d_ids_dst, count, ne01, unary_op, limit, true);
            if (cudaGetLastError() != cudaSuccess) return exact_fail("launch_fuse");
            const size_t chunk_bytes = (size_t)count * (size_t)ne01 * sizeof(float);
            if (cudaMemcpyAsync(bc.h_dst, fused_d, chunk_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return exact_fail("copy_chunk_d2h");
            if (cudaStreamSynchronize(st) != cudaSuccess) return exact_fail("sync_chunk");
            const float *tmp = (const float *)bc.h_dst;
            if (point_dump && point_active >= first && point_active < first + count &&
                    point_col >= 0 && point_col < ne01) {
                const int local = point_active - first;
                std::fprintf(stderr,
                    "[moe_stream] exact chunk point fused: tensor=%s active=%d local=%d flat=%d col=%d fused=%g\n",
                    src0_up_name, point_active, local, point_flat, point_col,
                    (double)tmp[(size_t)local * ne01 + point_col]);
            }
            for (int jj = 0; jj < count; ++jj) {
                const int j = first + jj;
                float *dst_row = (float *)((char *)dst + (size_t)dst_ids[j] * dst_nb1 + (size_t)token_ids[j] * dst_nb2);
                std::memcpy(dst_row, tmp + (size_t)jj * ne01, (size_t)ne01 * sizeof(float));
            }
        }
        return true;
    }

    if (parallel_up_gate) {
        if (cudaEventRecord(bc.ev_stage_ready, st) != cudaSuccess) return false;
        if (cudaStreamWaitEvent(bc.up_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (cudaStreamWaitEvent(bc.gate_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (bc.up_copy_stream && cudaStreamWaitEvent(bc.up_copy_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (bc.gate_copy_stream && cudaStreamWaitEvent(bc.gate_copy_stream, bc.ev_stage_ready, 0) != cudaSuccess) return false;
        if (profile && bc.ev_up_start) cudaEventRecord(bc.ev_up_start, bc.up_stream);
        if (profile && bc.ev_gate_work_start) cudaEventRecord(bc.ev_gate_work_start, bc.gate_stream);
        auto parallel_fail = [&]() -> bool {
            cudaStreamSynchronize(bc.up_stream);
            cudaStreamSynchronize(bc.gate_stream);
            if (bc.up_copy_stream) cudaStreamSynchronize(bc.up_copy_stream);
            if (bc.gate_copy_stream) cudaStreamSynchronize(bc.gate_copy_stream);
            return false;
        };

        int up_slots[MOE_STREAM_MAX_ACTIVE] = {};
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
            up_stage_jobs_count = (int)up_jobs.size();
            gate_stage_jobs_count = (int)gate_jobs.size();

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
                if (profile && bc.ev_up_compute_start) cudaEventRecord(bc.ev_up_compute_start, bc.up_stream);
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
                if (profile && bc.ev_gate_compute_start) cudaEventRecord(bc.ev_gate_compute_start, bc.gate_stream);
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
                if (profile && bc.ev_up_compute_start) cudaEventRecord(bc.ev_up_compute_start, bc.up_stream);
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
                if (profile && bc.ev_gate_compute_start) cudaEventRecord(bc.ev_gate_compute_start, bc.gate_stream);
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

    if (point_dump && point_active >= 0 && point_active < n_active && point_col >= 0 && point_col < ne01) {
        if (cudaMemcpyAsync(bc.h_dst, bc.d_up, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        const float *point_tmp = (const float *)bc.h_dst;
        const float up_active = point_tmp[(size_t)point_active * (size_t)ne01 + (size_t)point_col];
        const float up_flat = point_flat >= 0 && point_flat < dst_cols ?
            point_tmp[(size_t)point_flat * (size_t)ne01 + (size_t)point_col] : 0.0f;
        if (cudaMemcpyAsync(bc.h_dst, bc.d_gate, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        point_tmp = (const float *)bc.h_dst;
        const float gate_active = point_tmp[(size_t)point_active * (size_t)ne01 + (size_t)point_col];
        const float gate_flat = point_flat >= 0 && point_flat < dst_cols ?
            point_tmp[(size_t)point_flat * (size_t)ne01 + (size_t)point_col] : 0.0f;
        std::fprintf(stderr,
            "[moe_stream] up/gate point prefuse: tensor=%s active=%d active_expert=%d active_token=%d active_dst=%d active_flat=%d flat=%d col=%d up_active=%g gate_active=%g up_flat=%g gate_flat=%g prompt_mode=%d\n",
            src0_up_name, point_active,
            active_experts[point_active], token_ids[point_active], dst_ids[point_active], flat_dst_ids[point_active],
            point_flat, point_col, up_active, gate_active, up_flat, gate_flat, prompt_mode ? 1 : 0);
    }

    for (int j = 0; j < n_active; ++j) {
        bc.h_ids_dst[j] = prompt_mode ? j : flat_dst_ids[j];
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
        const char *handoff_sync_env = std::getenv("GGML_MOE_GPU_HANDOFF_SYNC");
        const bool sync_handoff = profile ||
            (handoff_sync_env && handoff_sync_env[0] && handoff_sync_env[0] != '0');
        if (sync_handoff && cudaStreamSynchronize(st) != cudaSuccess) return false;
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
            float up_wait_ms = 0.0f;
            float gate_wait_ms = 0.0f;
            float up_compute_ms = 0.0f;
            float gate_compute_ms = 0.0f;
            float fuse_ms = 0.0f;
            float kernel_ms = 0.0f;
            cudaEventElapsedTime(&stage_ms, bc.ev_start, bc.ev_stage);
            cudaEventElapsedTime(&quant_ms, bc.ev_stage, bc.ev_quant);
            if (parallel_up_gate && bc.ev_up_start && bc.ev_gate_start) {
                cudaEventElapsedTime(&up_ms, bc.ev_up_start, bc.ev_up);
                cudaEventElapsedTime(&gate_ms, bc.ev_gate_start, bc.ev_gate);
                if (parallel_stage && bc.ev_gate_work_start && bc.ev_up_compute_start && bc.ev_gate_compute_start) {
                    cudaEventElapsedTime(&up_wait_ms, bc.ev_up_start, bc.ev_up_compute_start);
                    cudaEventElapsedTime(&gate_wait_ms, bc.ev_gate_work_start, bc.ev_gate_compute_start);
                    cudaEventElapsedTime(&up_compute_ms, bc.ev_up_compute_start, bc.ev_up);
                    cudaEventElapsedTime(&gate_compute_ms, bc.ev_gate_compute_start, bc.ev_gate);
                }
            } else {
                cudaEventElapsedTime(&up_ms, bc.ev_quant, bc.ev_up);
                cudaEventElapsedTime(&gate_ms, bc.ev_up, bc.ev_gate);
            }
            cudaEventElapsedTime(&fuse_ms, bc.ev_gate, bc.ev_kernel);
            cudaEventElapsedTime(&kernel_ms, bc.ev_quant, bc.ev_kernel);
            ++g_uprof.calls;
            g_uprof.active_experts += (uint64_t)n_active;
            g_uprof.up_stage_jobs += (uint64_t)up_stage_jobs_count;
            g_uprof.gate_stage_jobs += (uint64_t)gate_stage_jobs_count;
            g_uprof.stage_ms += stage_ms;
            g_uprof.quant_ms += quant_ms;
            g_uprof.up_ms += up_ms;
            g_uprof.gate_ms += gate_ms;
            g_uprof.up_wait_ms += up_wait_ms;
            g_uprof.gate_wait_ms += gate_wait_ms;
            g_uprof.up_compute_ms += up_compute_ms;
            g_uprof.gate_compute_ms += gate_compute_ms;
            g_uprof.fuse_ms += fuse_ms;
            g_uprof.kernel_ms += kernel_ms;
        }
        return true;
    }

    if (cudaMemcpyAsync(bc.h_dst, bc.d_dst, dst_bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    if (profile) cudaEventRecord(bc.ev_d2h, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;

    if (point_dump && point_active >= 0 && point_active < n_active && point_col >= 0 && point_col < ne01) {
        const float *point_tmp = (const float *)bc.h_dst;
        const float fused_active = point_tmp[(size_t)point_active * (size_t)ne01 + (size_t)point_col];
        const float fused_flat = point_flat >= 0 && point_flat < dst_cols ?
            point_tmp[(size_t)point_flat * (size_t)ne01 + (size_t)point_col] : 0.0f;
        std::fprintf(stderr,
            "[moe_stream] up/gate point fused: tensor=%s active=%d active_expert=%d active_token=%d active_dst=%d active_flat=%d flat=%d col=%d fused_active=%g fused_flat=%g prompt_mode=%d\n",
            src0_up_name, point_active,
            active_experts[point_active], token_ids[point_active], dst_ids[point_active], flat_dst_ids[point_active],
            point_flat, point_col, fused_active, fused_flat, prompt_mode ? 1 : 0);
    }

    float stage_ms = 0.0f;
    float quant_ms = 0.0f;
    float up_ms = 0.0f;
    float gate_ms = 0.0f;
    float up_wait_ms = 0.0f;
    float gate_wait_ms = 0.0f;
    float up_compute_ms = 0.0f;
    float gate_compute_ms = 0.0f;
    float fuse_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    if (profile) {
        cudaEventElapsedTime(&stage_ms, bc.ev_start, bc.ev_stage);
        cudaEventElapsedTime(&quant_ms, bc.ev_stage, bc.ev_quant);
        if (parallel_up_gate && bc.ev_up_start && bc.ev_gate_start) {
            cudaEventElapsedTime(&up_ms, bc.ev_up_start, bc.ev_up);
            cudaEventElapsedTime(&gate_ms, bc.ev_gate_start, bc.ev_gate);
            if (parallel_stage && bc.ev_gate_work_start && bc.ev_up_compute_start && bc.ev_gate_compute_start) {
                cudaEventElapsedTime(&up_wait_ms, bc.ev_up_start, bc.ev_up_compute_start);
                cudaEventElapsedTime(&gate_wait_ms, bc.ev_gate_work_start, bc.ev_gate_compute_start);
                cudaEventElapsedTime(&up_compute_ms, bc.ev_up_compute_start, bc.ev_up);
                cudaEventElapsedTime(&gate_compute_ms, bc.ev_gate_compute_start, bc.ev_gate);
            }
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
        const int32_t src_row = prompt_mode ? j : flat_dst_ids[j];
        std::memcpy(dst_row, tmp + (size_t)src_row * ne01, (size_t)ne01 * sizeof(float));
    }
    if (profile) {
        const auto scatter_end = std::chrono::steady_clock::now();
        const double scatter_ms = std::chrono::duration<double, std::milli>(scatter_end - scatter_start).count();
        ++g_uprof.calls;
        g_uprof.active_experts += (uint64_t)n_active;
        g_uprof.up_stage_jobs += (uint64_t)up_stage_jobs_count;
        g_uprof.gate_stage_jobs += (uint64_t)gate_stage_jobs_count;
        g_uprof.stage_ms += stage_ms;
        g_uprof.quant_ms += quant_ms;
        g_uprof.up_ms += up_ms;
        g_uprof.gate_ms += gate_ms;
        g_uprof.up_wait_ms += up_wait_ms;
        g_uprof.gate_wait_ms += gate_wait_ms;
        g_uprof.up_compute_ms += up_compute_ms;
        g_uprof.gate_compute_ms += gate_compute_ms;
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
    batch_ttft_call_scope ttft_scope("call_down", src0_name, n_active, src0_bytes);
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
        int expert_idx = -1;
        char tensor[96] = {};
    };

    auto clear_down_stage_jobs = [&](const std::vector<down_stage_copy_job> &jobs) {
        for (const down_stage_copy_job &job : jobs) {
            batch_cache_clear_slot(cache, job.slot);
        }
    };

    auto copy_down_stage_jobs = [&](const std::vector<down_stage_copy_job> &jobs, cudaStream_t run_stream, pinned_stage_ring &ring) -> bool {
        if (cudaSetDevice(0) != cudaSuccess) return false;
        for (const down_stage_copy_job &job : jobs) {
            batch_copy_trace copy_trace;
            const auto copy_start = batch_ttft_trace_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (!batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, job.pack_entry, &copy_trace)) {
                if (!job.pack_entry ||
                        !batch_cache_copy_h2d(ring, job.dst, job.host_data, src0_bytes, run_stream, nullptr, &copy_trace)) {
                    return false;
                }
            }
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
            if (copy_start != std::chrono::steady_clock::time_point{}) {
                const auto copy_end = std::chrono::steady_clock::now();
                batch_ttft_trace_record(
                    "runtime_load",
                    job.tensor,
                    job.expert_idx,
                    src0_bytes,
                    false,
                    copy_trace.pack_hit,
                    copy_trace.ram_hit,
                    std::chrono::duration<double, std::milli>(copy_end - copy_start).count());
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
                down_stage_copy_job job;
                job.slot = cache_slot;
                job.dst = dst_slot;
                job.host_data = expert_host;
                job.pack_entry = pack_entry;
                job.expert_idx = active_experts[j];
                std::snprintf(job.tensor, sizeof(job.tensor), "%s", src0_name ? src0_name : "");
                if ((int)(down_jobs_a.size() + down_jobs_b.size()) & 1) {
                    down_jobs_b.push_back(job);
                } else {
                    down_jobs_a.push_back(job);
                }
            }
        } else {
            batch_ttft_trace_record("cache_hit", src0_name, active_experts[j], src0_bytes, true, false, false, 0.0);
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
            bc.d_x_ids, (float *)bc.d_dst, ne00, ne01, nb01, src0_bytes, n_active, dst_cols, st, false)) {
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
